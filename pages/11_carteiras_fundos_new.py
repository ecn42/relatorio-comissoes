# app.py
# Streamlit app to:
# 1) Sync newest CVM CDA FI monthly ZIPs (with a "force re-download" option).
# 2) For input CNPJs, find the latest month (among synced months) where the
#    CNPJ is NOT present in cda_fi_CONFID_YYYYMM.csv.
# 3) For a selected CNPJ, open that month’s ZIP, join cda_fi_BLC_1..8 files,
#    filter by the CNPJ, show the rows, and store/update them in SQLite.
# 4) Compute PCT_CARTEIRA (0–1 fraction) for each row from VL_MERC_POS_FINAL
#    and include it in the CSV stored in the DB.
# 5) Portfolio summary section: Load from DB, dropdown select fund, show allocation summary,
#    top 10 positions, and donut chart.
# Updated: Store BLC data as rows in a proper SQLite table (blc_data) instead of CSV BLOB
# for better querying and reliability. Added DENOM_SOCIAL to fundos_meta.
# Donut chart with Ceres Wealth colors and transparent background.
# Legend integrated into the plot, positioned to the right.

from __future__ import annotations

import csv
import io
import json
import os
import re
import sqlite3
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go

# ------------------------ Config and regexes ------------------------
tab1, tab2, tab3 = st.tabs(["Carteiras dos Fundos", "Baixar CDI", "Rentabilidade dos Fundos"])

with tab1:
    BASE_URL = "https://dados.cvm.gov.br/dados/FI/DOC/CDA/DADOS/"
    USER_AGENT = "Mozilla/5.0 (CVM-CDA-FI-Sync/1.5)"

    # Match names like cda_fi_202509.zip and capture 202509
    ZIP_RE = re.compile(r"cda_fi_(\d{6})\.zip", re.IGNORECASE)

    # Inside ZIPs
    CONFID_RE = re.compile(r"(^|/)(cda_fi_CONFID_\d{6}\.csv)$", re.IGNORECASE)


    def blc_pattern(yyyymm: str) -> re.Pattern:
        # cda_fi_BLC_[1..8]_YYYYMM.csv (possibly in a subfolder)
        return re.compile(rf"(^|/)cda_fi_BLC_[1-8]_{yyyymm}\.csv$", re.IGNORECASE)


    # ------------------------ CSV limits (fix large-field error) ------------------


    def set_csv_field_size_limit() -> int:
        """
        Increase Python csv.field_size_limit to handle very large fields in CVM
        files. Works across platforms (Windows can raise OverflowError).
        """
        max_int = sys.maxsize
        while True:
            try:
                csv.field_size_limit(max_int)
                return max_int
            except OverflowError:
                max_int = int(max_int / 10)


    CSV_LIMIT = set_csv_field_size_limit()


    # ------------------------ Helpers ------------------------


    def ensure_dir(path: str) -> str:
        os.makedirs(path, exist_ok=True)
        return path


    def fetch_remote_zip_names() -> List[str]:
        """
        Scrape the CVM directory index and return all cda_fi_YYYYMM.zip names.
        Robust to extra HTML around the links.
        """
        headers = {"User-Agent": USER_AGENT}
        resp = requests.get(BASE_URL, headers=headers, timeout=30)
        resp.raise_for_status()
        html = resp.text

        # Prefer grabbing from anchor hrefs; keep just the filename
        href_pat = r'href="(?:[^"]*/)?(cda_fi_\d{6}\.zip)"'
        hrefs = re.findall(href_pat, html, flags=re.IGNORECASE)
        candidates = set(hrefs)

        # Fallback: any plain match in HTML
        if not candidates:
            candidates = set(m.group(0) for m in ZIP_RE.finditer(html))

        if not candidates:
            raise RuntimeError(
                "No cda_fi_YYYYMM.zip files found at the CVM index page."
            )

        def ym_key(name: str) -> int:
            m = re.search(r"(\d{6})", name)
            return int(m.group(1)) if m else -1

        names = sorted(candidates, key=ym_key)
        return names


    def newest_k(names: List[str], k: int = 6) -> List[str]:
        return sorted(names, key=lambda n: int(ZIP_RE.search(n).group(1)))[-k:]


    def file_exists(path: str) -> bool:
        try:
            return os.path.isfile(path) and os.path.getsize(path) > 0
        except Exception:
            return False


    def download_one(url: str, dest: str) -> Tuple[str, bool, Optional[str]]:
        """
        Download a file to dest; returns (dest, success, error_message).
        Overwrites existing file atomically.
        """
        headers = {"User-Agent": USER_AGENT}
        tmp = dest + ".part"
        try:
            with requests.get(url, stream=True, headers=headers, timeout=120) as r:
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        if chunk:
                            f.write(chunk)
            os.replace(tmp, dest)  # atomic overwrite
            return dest, True, None
        except Exception as e:
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except Exception:
                pass
            return dest, False, str(e)


    def sync_latest(
        dest_dir: str, k: int = 6, force: bool = False
    ) -> Tuple[List[str], List[str], List[str], List[str]]:
        """
        Ensure the newest k zip files exist locally.
        force=False: download only missing files.
        force=True: re-download and overwrite the newest k files.
        Returns (target_names, already_present, downloaded_now, failed)
        """
        all_names = fetch_remote_zip_names()
        target = newest_k(all_names, k=k)
        ensure_dir(dest_dir)

        already, to_get = [], []
        if force:
            to_get = target[:]  # overwrite all target files
        else:
            for name in target:
                path = os.path.join(dest_dir, name)
                if file_exists(path):
                    already.append(name)
                else:
                    to_get.append(name)

        downloaded, failed = [], []
        if to_get:
            with ThreadPoolExecutor(max_workers=min(4, len(to_get))) as ex:
                futures = []
                for name in to_get:
                    url = BASE_URL + name
                    dest = os.path.join(dest_dir, name)
                    futures.append(ex.submit(download_one, url, dest))
                for fut in as_completed(futures):
                    dest, ok, err = fut.result()
                    nm = os.path.basename(dest)
                    if ok:
                        downloaded.append(nm)
                    else:
                        failed.append(f"{nm}: {err}")

        return target, already, downloaded, failed


    def guess_delimiter(sample: str) -> str:
        # CVM CSVs are usually ';', but detect robustly.
        candidates = [";", ",", "|", "\t"]
        counts = {d: sample.count(d) for d in candidates}
        return max(counts, key=counts.get) if counts else ";"


    def decode_bytes(b: bytes) -> str:
        for enc in ("utf-8-sig", "latin1", "cp1252"):
            try:
                return b.decode(enc)
            except UnicodeDecodeError:
                continue
        return b.decode("utf-8", errors="replace")


    def normalize_cnpj(s: str) -> Optional[str]:
        digits = re.sub(r"\D", "", s or "")
        return digits if len(digits) == 14 else None


    # ------------- CONFID reader (to build set of CNPJs per month) -------------


    def read_confid_cnpjs_from_zip(zip_path: str) -> Tuple[str, Set[str]]:
        """
        Extract the set of CNPJ_FUNDO_CLASSE from cda_fi_CONFID_YYYYMM.csv inside
        the given zip. Returns (yyyymm, set_of_cnpjs).
        """
        m = ZIP_RE.search(os.path.basename(zip_path))
        yyyymm = m.group(1) if m else "000000"
        cnpjs: Set[str] = set()

        with zipfile.ZipFile(zip_path) as zf:
            member = None
            for name in zf.namelist():
                if CONFID_RE.search(name):
                    member = name
                    break
            if not member:
                return yyyymm, cnpjs

            data = zf.read(member)
            text = decode_bytes(data)
            lines = text.splitlines()
            if not lines:
                return yyyymm, cnpjs

            delim = guess_delimiter(lines[0])
            reader = csv.reader(io.StringIO(text), delimiter=delim)
            try:
                header = next(reader)
            except StopIteration:
                return yyyymm, cnpjs

            header_norm = [h.strip().upper() for h in header]
            try:
                idx = header_norm.index("CNPJ_FUNDO_CLASSE")
            except ValueError:
                return yyyymm, cnpjs

            for row in reader:
                if idx < len(row):
                    c = normalize_cnpj(row[idx])
                    if c:
                        cnpjs.add(c)

        return yyyymm, cnpjs


    # ------------------------ Parse CNPJ input (robust) ------------------------


    def parse_cnpj_input(text: str) -> List[str]:
        """
        Accept CNPJs with any punctuation and common separators.
        We split on whitespace/commas/semicolons/pipes, normalize each token.
        """
        tokens = re.split(r"[\s,;|]+", text or "")
        out: List[str] = []
        seen: Set[str] = set()
        for t in tokens:
            c = normalize_cnpj(t)
            if c and c not in seen:
                out.append(c)
                seen.add(c)
        return out


    # ------------------------ Absent month logic ------------------------


    def latest_absent_month_for_cnpj(
        cnpj: str, months_desc: List[str], cnpjs_by_month: Dict[str, Set[str]]
    ) -> Optional[str]:
        for ym in months_desc:
            if cnpj not in cnpjs_by_month.get(ym, set()):
                return ym
        return None


    def ym_label(ym: Optional[str]) -> str:
        if not ym:
            return "(present in all synced months)"
        return f"{ym[:4]}-{ym[4:]}"


    # ------------------------ BLC joining/filtering ------------------------


    def filter_df_by_cnpj(df: pd.DataFrame, cnpj: str) -> pd.DataFrame:
        """
        Try common columns; if none, return empty.
        """
        candidates = []
        for col in df.columns:
            up = str(col).strip().upper()
            if up in ("CNPJ_FUNDO", "CNPJ_FUNDO_CLASSE", "CNPJ"):
                candidates.append(col)

        if not candidates:
            return df.iloc[0:0].copy()

        def norm_series(s: pd.Series) -> pd.Series:
            return s.astype(str).str.replace(r"\D", "", regex=True)

        mask = False
        for col in candidates:
            mask = mask | (norm_series(df[col]) == cnpj)
        return df.loc[mask].copy()


    def manual_filter_rows_by_cnpj(
        text: str, sep: str, filename: str, cnpj: str
    ) -> pd.DataFrame:
        """
        Ultra-robust fallback: split lines on the separator even if quotes are
        broken; fix width by joining extras; keep only rows with the CNPJ.
        """
        lines = text.splitlines()
        if not lines:
            return pd.DataFrame()

        header = [h.strip() for h in lines[0].split(sep)]
        n = len(header)
        cidx = [i for i, h in enumerate(header) if "CNPJ" in h.upper()]
        if not cidx:
            return pd.DataFrame()

        out_rows: List[Dict[str, str]] = []
        for ln in lines[1:]:
            if not ln:
                continue
            parts = ln.split(sep)
            if len(parts) < n:
                parts = parts + [""] * (n - len(parts))
            elif len(parts) > n:
                parts = parts[: n - 1] + [sep.join(parts[n - 1 :])]

            match = False
            for i in cidx:
                if i < len(parts):
                    if re.sub(r"\D", "", parts[i]) == cnpj:
                        match = True
                        break
            if match:
                row = {header[i]: (parts[i] if i < len(parts) else "") for i in range(n)}
                row["_ARQUIVO"] = os.path.basename(filename)
                out_rows.append(row)

        if not out_rows:
            return pd.DataFrame()
        return pd.DataFrame(out_rows)


    def read_blc_joined_for_cnpj_from_zip(zip_path: str, cnpj: str) -> pd.DataFrame:
        """
        Open cda_fi_BLC_1..8_YYYYMM.csv inside the ZIP, read robustly (tolerate
        malformed quoting and huge fields), filter by CNPJ while streaming,
        and return the union.
        """
        m = ZIP_RE.search(os.path.basename(zip_path))
        yyyymm = m.group(1) if m else "000000"

        pieces: List[pd.DataFrame] = []
        with zipfile.ZipFile(zip_path) as zf:
            pat = blc_pattern(yyyymm)
            members = sorted(
                [n for n in zf.namelist() if pat.search(n)], key=str.lower
            )

            for name in members:
                data = zf.read(name)
                text = decode_bytes(data)
                lines = text.splitlines()
                if not lines:
                    continue
                sep = guess_delimiter(lines[0]) or ";"

                # Strategy A: normal tolerant CSV parsing
                try:
                    reader = pd.read_csv(
                        io.StringIO(text),
                        sep=sep,
                        dtype=str,
                        engine="python",
                        on_bad_lines="skip",
                        quoting=csv.QUOTE_MINIMAL,
                        chunksize=150_000,
                    )
                    for chunk in reader:
                        chunk["_ARQUIVO"] = os.path.basename(name)
                        filt = filter_df_by_cnpj(chunk, cnpj)
                        if not filt.empty:
                            pieces.append(filt)
                    continue
                except Exception:
                    pass

                # Strategy B: treat quotes as normal characters
                try:
                    reader = pd.read_csv(
                        io.StringIO(text),
                        sep=sep,
                        dtype=str,
                        engine="python",
                        on_bad_lines="skip",
                        quoting=csv.QUOTE_NONE,
                        escapechar="\\",
                        chunksize=150_000,
                    )
                    for chunk in reader:
                        chunk["_ARQUIVO"] = os.path.basename(name)
                        filt = filter_df_by_cnpj(chunk, cnpj)
                        if not filt.empty:
                            pieces.append(filt)
                    continue
                except Exception:
                    pass

                # Strategy C: manual splitter fallback
                df_manual = manual_filter_rows_by_cnpj(text, sep, name, cnpj)
                if not df_manual.empty:
                    pieces.append(df_manual)

        if not pieces:
            return pd.DataFrame()
        return pd.concat(pieces, ignore_index=True, sort=False)


    # ------------------------ Portfolio share (0–1 fraction) ---------------------


    def add_pct_carteira_by_vl_merc(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add PCT_CARTEIRA (0–1 fraction) based on VL_MERC_POS_FINAL over the total
        of the filtered DataFrame (i.e., the fund's positions for the month).
        Handles Brazilian decimal formats (e.g., 1.234.567,89) and parentheses
        for negatives.
        """
        if df.empty:
            return df

        # Find the VL_MERC_POS_FINAL column (case-insensitive; substring fallback)
        cols_norm = {c: str(c).strip().upper() for c in df.columns}
        col = next((c for c, n in cols_norm.items() if n == "VL_MERC_POS_FINAL"), None)
        if col is None:
            col = next((c for c, n in cols_norm.items() if "VL_MERC_POS_FINAL" in n), None)
        if col is None:
            # Column not found; nothing to compute
            df["PCT_CARTEIRA"] = 0.0
            return df

        s = df[col].astype(str).str.strip()
        # Normalize: remove non-breaking spaces
        s = s.str.replace("\u00a0", "", regex=False)
        # Keep digits, separators, signs, parentheses
        s = s.str.replace(r"[^\d,.\-\(\)]", "", regex=True)
        # Parentheses as negatives: (123,45) -> -123,45
        s = s.str.replace(r"^\((.*)\)$", r"-\1", regex=True)

        # Detect separator patterns and normalize to dot-decimal
        both = s.str.contains(",", na=False) & s.str.contains(r"\.", na=False)
        # If both present, assume '.' thousands and ',' decimal
        s = s.where(
            ~both, s.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
        )
        only_comma = s.str.contains(",", na=False) & ~s.str.contains(r"\.", na=False)
        # If only comma present, treat comma as decimal
        s = s.where(~only_comma, s.str.replace(",", ".", regex=False))
        # Else: either '.' decimal or digits only

        num = pd.to_numeric(s, errors="coerce")
        total = num.sum(min_count=1)

        if pd.isna(total) or total == 0:
            df["PCT_CARTEIRA"] = 0.0
            return df

        # Fraction of total (0..1)
        df["PCT_CARTEIRA"] = (num / total).round(10)
        return df


    # ------------------------ SQLite storage ------------------------


    def ensure_db_schema(conn: sqlite3.Connection) -> None:
        cur = conn.cursor()
        # fundos_meta with denom_social
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS fundos_meta (
                cnpj TEXT PRIMARY KEY,
                denom_social TEXT,
                last_month_yyyymm TEXT NOT NULL,
                last_updated_utc TEXT NOT NULL,
                n_rows INTEGER NOT NULL,
                columns_json TEXT NOT NULL
            )
            """
        )
        # blc_data: all columns as TEXT for robustness (parse nums as needed)
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS blc_data (
                cnpj TEXT NOT NULL,
                yyyymm TEXT NOT NULL,
                P_FUNDO_CLASSE TEXT,
                CNPJ_FUNDO_CLASSE TEXT,
                DENOM_SOCIAL TEXT,
                DT_COMPTC TEXT,
                TP_APLIC TEXT,
                TP_ATIVO TEXT,
                EMISSOR_LIGADO TEXT,
                TP_NEGOC TEXT,
                QT_VENDA_NEGOC TEXT,
                VL_VENDA_NEGOC TEXT,
                QT_AQUIS_NEGOC TEXT,
                VL_AQUIS_NEGOC TEXT,
                QT_POS_FINAL TEXT,
                VL_MERC_POS_FINAL TEXT,
                VL_CUSTO_POS_FINAL TEXT,
                DT_CONFID_APLIC TEXT,
                CNPJ_FUNDO_CLASSE_COTA TEXT,
                ID_SUBCLASSE TEXT,
                NM_FUNDO_CLASSE_SUBCLASSE_COTA TEXT,
                _ARQUIVO TEXT,
                CD_ATIVO TEXT,
                DS_ATIVO TEXT,
                CD_ISIN TEXT,
                DT_INI_VIGENCIA TEXT,
                DT_FIM_VIGENCIA TEXT,
                PF_PJ_EMISSOR TEXT,
                CPF_CNPJ_EMISSOR TEXT,
                EMISSOR TEXT,
                PCT_CARTEIRA TEXT
            )
            """
        )
        # Index for faster queries
        cur.execute("CREATE INDEX IF NOT EXISTS idx_blc_cnpj ON blc_data(cnpj)")
        conn.commit()


    def upsert_blc_in_db(
        db_path: str, cnpj: str, ym: str, df: pd.DataFrame
    ) -> Tuple[str, Optional[str]]:
        """
        Insert or update the stored BLC for a CNPJ based on month (YYYYMM).
        Stores rows in blc_data table and updates meta.
        Returns (action, previous_month) where action in {"inserted","updated","kept"}.
        """
        # Define known columns here to avoid scope issues
        known_cols = [
            "P_FUNDO_CLASSE", "CNPJ_FUNDO_CLASSE", "DENOM_SOCIAL", "DT_COMPTC",
            "TP_APLIC", "TP_ATIVO", "EMISSOR_LIGADO", "TP_NEGOC", "QT_VENDA_NEGOC",
            "VL_VENDA_NEGOC", "QT_AQUIS_NEGOC", "VL_AQUIS_NEGOC", "QT_POS_FINAL",
            "VL_MERC_POS_FINAL", "VL_CUSTO_POS_FINAL", "DT_CONFID_APLIC",
            "CNPJ_FUNDO_CLASSE_COTA", "ID_SUBCLASSE", "NM_FUNDO_CLASSE_SUBCLASSE_COTA",
            "_ARQUIVO", "CD_ATIVO", "DS_ATIVO", "CD_ISIN", "DT_INI_VIGENCIA",
            "DT_FIM_VIGENCIA", "PF_PJ_EMISSOR", "CPF_CNPJ_EMISSOR", "EMISSOR", "PCT_CARTEIRA"
        ]

        conn = sqlite3.connect(db_path)
        ensure_db_schema(conn)
        cur = conn.cursor()

        cur.execute("SELECT last_month_yyyymm FROM fundos_meta WHERE cnpj=?", (cnpj,))
        row = cur.fetchone()
        now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        cols_list = list(df.columns)
        cols_json = json.dumps(cols_list, ensure_ascii=False)
        n_rows = int(len(df))

        denom_social = ""
        if not df.empty and "DENOM_SOCIAL" in df.columns:
            denom_social = str(df["DENOM_SOCIAL"].iloc[0]).strip()

        if row is None:
            # Insert new
            cur.execute(
                "DELETE FROM blc_data WHERE cnpj=?",
                (cnpj,),
            )
            cur.execute(
                "INSERT INTO fundos_meta (cnpj, denom_social, last_month_yyyymm, last_updated_utc, "
                "n_rows, columns_json) VALUES (?, ?, ?, ?, ?, ?)",
                (cnpj, denom_social, ym, now_iso, n_rows, cols_json),
            )
            action = "inserted"
            prev_ym = None
        else:
            prev_ym = row[0]
            if int(ym) > int(prev_ym):
                # Update (newer month)
                cur.execute(
                    "DELETE FROM blc_data WHERE cnpj=?",
                    (cnpj,),
                )
                cur.execute(
                    "UPDATE fundos_meta SET denom_social=?, last_month_yyyymm=?, last_updated_utc=?, "
                    "n_rows=?, columns_json=? WHERE cnpj=?",
                    (denom_social, ym, now_iso, n_rows, cols_json, cnpj),
                )
                action = "updated"
            else:
                action = "kept"
                conn.close()
                return action, prev_ym

        # Insert data rows (append after delete)
        if not df.empty:
            try:
                # Select only known columns + cnpj, yyyymm
                df_insert = df[[col for col in known_cols if col in df.columns]].copy()
                df_insert["cnpj"] = cnpj
                df_insert["yyyymm"] = ym
                # Ensure PCT_CARTEIRA is str
                if "PCT_CARTEIRA" in df_insert.columns:
                    df_insert["PCT_CARTEIRA"] = df_insert["PCT_CARTEIRA"].astype(str)
                # Fill NaN with empty str for TEXT columns
                df_insert = df_insert.fillna("")
                df_insert.to_sql(
                    "blc_data",
                    conn,
                    if_exists="append",
                    index=False,
                    method="multi",
                    chunksize=1000,
                )
            except Exception as insert_err:
                st.warning(f"to_sql failed: {insert_err}. Falling back to manual insert.")
                # Fallback: manual insert
                cur.execute("DELETE FROM blc_data WHERE cnpj=?", (cnpj,))
                for _, row in df.iterrows():
                    values = [
                        cnpj, ym,
                        *[row.get(col, "") for col in known_cols]
                    ]
                    placeholders = ", ".join(["?" for _ in values])
                    columns_str = "(cnpj, yyyymm, " + ", ".join(known_cols) + ")"
                    sql = f"INSERT INTO blc_data {columns_str} VALUES ({placeholders})"
                    cur.execute(sql, values)

        conn.commit()
        conn.close()
        return action, prev_ym


    # ------------------------ Portfolio Summary Functions ------------------------


    def load_fund_data_from_db(db_path: str, cnpj: str) -> Optional[Tuple[pd.DataFrame, str, str]]:
        """
        Load BLC rows from blc_data table and meta for CNPJ.
        Returns (df, yyyymm, denom_social).
        """
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute(
            "SELECT denom_social, last_month_yyyymm FROM fundos_meta WHERE cnpj=?", (cnpj,)
        )
        row = cur.fetchone()
        if not row:
            conn.close()
            return None

        denom_social, yyyymm = row

        df = pd.read_sql_query(
            "SELECT * FROM blc_data WHERE cnpj=? ORDER BY DS_ATIVO, EMISSOR",
            params=(cnpj,),
            con=conn,
        )
        # Convert PCT_CARTEIRA back to numeric
        if "PCT_CARTEIRA" in df.columns:
            df["PCT_CARTEIRA"] = pd.to_numeric(df["PCT_CARTEIRA"], errors="coerce").fillna(0.0)
        else:
            df["PCT_CARTEIRA"] = 0.0

        conn.close()

        return df, yyyymm, denom_social


    def categorize_tp_ativo(tp_ativo: str) -> str:
        """
        Map TP_ATIVO to broader allocation categories (custom mapping based on common CVM types).
        Adjust as needed for accuracy.
        """
        if pd.isna(tp_ativo) or not str(tp_ativo).strip():
            return "Outros"
        
        tp_upper = str(tp_ativo).strip().upper()
        
        # Depósitos a prazo e outros títulos de IF (CDB, RDB, LF, etc.)
        if any(kw in tp_upper for kw in ["CDB", "RDB", "LETRA FINANCEIRA", "LF", "LCR"]):
            return "Depósitos a prazo e outros títulos de IF"
        
        # Títulos Públicos (NTN, LTN, etc.)
        if any(kw in tp_upper for kw in ["TÍTULO PÚBLICO", "NTN", "LTN", "LFT", "TESOURO"]):
            return "TÍTULOS PÚBLICOS"
        
        # Operações Compromissadas
        if "COMPROMISSADA" in tp_upper:
            return "OPERAÇÕES COMPROMISSADAS"
        
        # Ações (Stocks)
        if "AÇÃO" in tp_upper:
            return "AÇÕES"
        
        # Fundo de Investimento
        if "FUNDO" in tp_upper:
            return "FUNDO DE INVESTIMENTO"
        
        # Imóveis
        if "IMOBILIÁRIO" in tp_upper or "FII" in tp_upper:
            return "IMÓVEIS"
        
        # Outros (default)
        return "Outros"


    def prepare_allocation_summary(df: pd.DataFrame) -> pd.DataFrame:
        """
        Group by categorized TP_ATIVO, sum PCT_CARTEIRA.
        """
        if "TP_ATIVO" not in df.columns or "PCT_CARTEIRA" not in df.columns or df["PCT_CARTEIRA"].sum() == 0:
            return pd.DataFrame()
        
        df["CATEGORIA"] = df["TP_ATIVO"].apply(categorize_tp_ativo)
        alloc = df.groupby("CATEGORIA")["PCT_CARTEIRA"].sum().reset_index()
        alloc = alloc[alloc["PCT_CARTEIRA"] > 0].sort_values("PCT_CARTEIRA", ascending=False).head(5)
        if alloc.empty:
            return pd.DataFrame()
        alloc["PCT_STR"] = (alloc["PCT_CARTEIRA"] * 100).round(2).astype(str) + "%"
        return alloc


    def prepare_top_positions(df: pd.DataFrame, ym: str) -> pd.DataFrame:
        """
        Aggregate positions by unique asset (TP_ATIVO + EMISSOR + Maturity), sum PCT_CARTEIRA.
        Format maturity from DT_FIM_VIGENCIA (MM/YYYY).
        """
        required_cols = ["TP_ATIVO", "DT_FIM_VIGENCIA", "PCT_CARTEIRA"]
        if not all(col in df.columns for col in required_cols):
            return pd.DataFrame()
        
        # Fallback for EMISSOR
        if "EMISSOR" not in df.columns:
            if "EMISSOR_LIGADO" in df.columns:
                df["EMISSOR"] = df["EMISSOR_LIGADO"]
            else:
                df["EMISSOR"] = "N/A"
        
        # Filter non-zero positions
        df_pos = df[df["PCT_CARTEIRA"] > 0].copy()
        if df_pos.empty:
            return pd.DataFrame()
        
        # Parse maturity
        def format_maturity(date_str):
            if pd.isna(date_str) or not str(date_str).strip():
                return ""
            date_str = str(date_str).strip()
            try:
                dt = pd.to_datetime(date_str, errors="coerce")
                if pd.isna(dt):
                    return ""
                return dt.strftime("%m/%Y")
            except:
                m = re.search(r"(\d{4})([/-]?)(\d{2})", date_str)
                if m:
                    return f"{m.group(3)}/{m.group(1)}"
                return ""
        
        df_pos["MATURITY"] = df_pos["DT_FIM_VIGENCIA"].apply(format_maturity)
        df_pos["POSICAO_KEY"] = (
            df_pos["TP_ATIVO"].astype(str).str.strip() + " - " +
            df_pos["EMISSOR"].astype(str).str.strip() + " - " +
            df_pos["MATURITY"]
        ).str.rstrip(" -")
        df_pos["PCT_NUM"] = df_pos["PCT_CARTEIRA"]
        
        top = df_pos.groupby("POSICAO_KEY")["PCT_NUM"].sum().reset_index()
        top = top.sort_values("PCT_NUM", ascending=False).head(10)
        if top.empty:
            return pd.DataFrame()
        top["PCT_STR"] = (top["PCT_NUM"] * 100).round(2).astype(str) + "%"
        top["POSICAO_FULL"] = top["POSICAO_KEY"] + " - " + top["PCT_STR"]
        return top


    # ------------------------ Streamlit UI ------------------------

    st.set_page_config(
        page_title="CVM CDA FI - CONFID + BLC Checker", layout="centered"
    )
    st.title("CVM CDA FI - CONFID + BLC Checker")

    st.caption(
        "Sync newest CDA FI ZIPs from CVM (with overwrite option). For input "
        "CNPJs, find the most recent month they are NOT present in "
        "cda_fi_CONFID_YYYYMM.csv. Then, for that month, join cda_fi_BLC_1..8 and "
        "show only the rows for the chosen CNPJ. Results are saved in a local "
        "SQLite database. Adds PCT_CARTEIRA (0–1 fraction) from VL_MERC_POS_FINAL."
    )

    default_dir = os.path.join(os.getcwd(), "data_cda_fi")
    dest_dir = st.text_input("Local data folder", value=default_dir)

    db_default = os.path.join(os.getcwd(), "carteira_fundos.db")
    db_path = st.text_input("SQLite database path", value=db_default)

    k = st.number_input(
        "How many newest months to work with",
        min_value=1,
        max_value=24,
        value=6,
        step=1,
    )

    c1, c2 = st.columns(2)
    with c1:
        sync_btn = st.button("Sync newest files")
    with c2:
        force_sync_btn = st.button("Force re-download newest files (overwrite)")

    if sync_btn or force_sync_btn:
        with st.spinner("Syncing from CVM..."):
            t0 = time.time()
            try:
                target, already, downloaded, failed = sync_latest(
                    dest_dir, k=int(k), force=bool(force_sync_btn)
                )
                elapsed = time.time() - t0
                st.success(
                    f"Done in {elapsed:.1f}s. Target: {len(target)} | "
                    f"Already: {len(already)} | Downloaded: {len(downloaded)}"
                )
                st.write("Target months:", ", ".join(target))
                if downloaded:
                    st.write("Downloaded:", ", ".join(downloaded))
                if failed:
                    st.error("Failed downloads:")
                    for msg in failed:
                        st.write("- " + msg)
            except Exception as e:
                st.error(f"Sync failed: {e}")

    st.subheader("Check CNPJ presence in CONFID and view BLC rows")

    cnpj_text = st.text_area(
        "Paste one or more CNPJs (punctuation ok: 00.000.000/0000-00, etc.)",
        height=120,
    )

    check_btn = st.button(
        "Find latest month each CNPJ is NOT present (among synced months)"
    )

    if "absent_map" not in st.session_state:
        st.session_state.absent_map = {}
    if "checked_cnpjs" not in st.session_state:
        st.session_state.checked_cnpjs = []

    if check_btn:
        cnpjs = parse_cnpj_input(cnpj_text)
        st.session_state.checked_cnpjs = cnpjs
        if not cnpjs:
            st.warning("Please enter at least one valid 14-digit CNPJ.")
        else:
            try:
                # Use newest K local files (sorted by YYYYMM)
                local = [
                    n
                    for n in os.listdir(dest_dir)
                    if ZIP_RE.search(n) and os.path.isfile(os.path.join(dest_dir, n))
                ]
                if not local:
                    st.error(
                        "No local ZIPs found. Click one of the 'Sync' buttons first."
                    )
                else:
                    local_sorted = sorted(
                        local, key=lambda n: int(ZIP_RE.search(n).group(1))
                    )[-int(k) :]
                    local_paths = [os.path.join(dest_dir, n) for n in local_sorted]

                    cnpjs_by_month: Dict[str, Set[str]] = {}
                    prog = st.progress(0.0)
                    for i, zp in enumerate(local_paths, start=1):
                        ym, s = read_confid_cnpjs_from_zip(zp)
                        cnpjs_by_month[ym] = s
                        prog.progress(i / len(local_paths))

                    months_desc = sorted(
                        list(cnpjs_by_month.keys()), key=int, reverse=True
                    )

                    absent_map = {}
                    rows = []
                    for cnpj in cnpjs:
                        ym = latest_absent_month_for_cnpj(
                            cnpj, months_desc, cnpjs_by_month
                        )
                        absent_map[cnpj] = ym
                        rows.append(
                            {
                                "CNPJ": cnpj,
                                "Latest month absent": ym_label(ym),
                            }
                        )

                    st.session_state.absent_map = absent_map
                    st.dataframe(pd.DataFrame(rows), use_container_width=True)
            except Exception as e:
                st.error(f"Error while checking CNPJs: {e}")

    # Section to load BLC rows for the selected CNPJ's absent month
    if st.session_state.checked_cnpjs:
        st.markdown("Select a CNPJ to view and store its BLC rows:")
        selected_cnpj = st.selectbox(
            "CNPJ (normalized digits)",
            st.session_state.checked_cnpjs,
            index=0,
        )
        show_blc_btn = st.button("Show BLC rows for that absent month and save")

        if show_blc_btn:
            ym = st.session_state.absent_map.get(selected_cnpj)
            if not ym:
                st.info(
                    "That CNPJ is present in all synced months, so there is no "
                    "absent month to inspect."
                )
            else:
                zip_name = f"cda_fi_{ym}.zip"
                zip_path = os.path.join(dest_dir, zip_name)
                if not os.path.isfile(zip_path):
                    st.error(
                        f"ZIP {zip_name} not found locally. Increase K or re-sync."
                    )
                else:
                    with st.spinner(
                        f"Reading BLC files for {selected_cnpj} in {ym_label(ym)}..."
                    ):
                        try:
                            df_blc = read_blc_joined_for_cnpj_from_zip(
                                zip_path, selected_cnpj
                            )
                            # Add share of portfolio (0–1 fraction) based on VL_MERC_POS_FINAL
                            df_blc = add_pct_carteira_by_vl_merc(df_blc)

                            if df_blc.empty:
                                st.warning(
                                    "No matching rows found in cda_fi_BLC_1..8 for "
                                    f"{selected_cnpj} in {ym_label(ym)}."
                                )
                            else:
                                st.dataframe(
                                    df_blc, use_container_width=True, height=420
                                )
                                # Save/update in SQLite (now as table rows)
                                action, prev = upsert_blc_in_db(
                                    db_path, selected_cnpj, ym, df_blc
                                )
                                if action == "inserted":
                                    st.success(
                                        f"Saved {len(df_blc)} rows for CNPJ "
                                        f"{selected_cnpj} at {ym_label(ym)} to DB."
                                    )
                                elif action == "updated":
                                    st.success(
                                        f"Existing CNPJ updated from {ym_label(prev)} "
                                        f"to {ym_label(ym)} with {len(df_blc)} rows."
                                    )
                                else:
                                    st.info(
                                        f"Database already has {ym_label(prev)} for "
                                        f"{selected_cnpj}. New month {ym_label(ym)} is "
                                        "not newer; keeping existing snapshot."
                                    )

                                csv_bytes = df_blc.to_csv(index=False).encode("utf-8")
                                st.download_button(
                                    "Download filtered BLC as CSV",
                                    data=csv_bytes,
                                    file_name=f"blc_{selected_cnpj}_{ym}.csv",
                                    mime="text/csv",
                                )
                        except Exception as e:
                            st.error(f"Failed to load BLC data: {e}")

    # ------------------------ New Section: Portfolio Summary from DB ------------------------

    st.markdown("---")
    st.subheader("Resumo de Carteira de Fundos (do Banco de Dados)")

    # Load list of funds from DB
    try:
        conn = sqlite3.connect(db_path)
        funds_df = pd.read_sql_query(
            "SELECT cnpj, last_month_yyyymm, n_rows FROM fundos_meta ORDER BY cnpj", conn
        )
        conn.close()
    except Exception:
        funds_df = pd.DataFrame()

    if funds_df.empty:
        st.info("Nenhum fundo encontrado no banco de dados. Adicione fundos na seção anterior primeiro.")
    else:
        # Dropdown to select fund
        def format_cnpj(cnpj):
            mask = funds_df['cnpj'] == cnpj
            if mask.any():
                ym = funds_df.loc[mask, 'last_month_yyyymm'].iloc[0]
                return f"{cnpj} ({ym[:4]}-{ym[4:]})"
            return cnpj
        
        cnpj_options = funds_df["cnpj"].tolist()
        selected_cnpj_summary = st.selectbox(
            "Selecione um fundo (CNPJ)",
            options=cnpj_options,
            format_func=format_cnpj
        )
        
        if selected_cnpj_summary:
            with st.spinner("Carregando dados do fundo..."):
                try:
                    fund_data = load_fund_data_from_db(db_path, selected_cnpj_summary)
                    if fund_data is None:
                        st.error("Dados não encontrados para este CNPJ.")
                    else:
                        df, ym, denom_social = fund_data
                        if df.empty:
                            st.warning("Nenhuma posição encontrada para este fundo.")
                        else:
                            total_positions = len(df[df['PCT_CARTEIRA'] > 0])
                            st.markdown(f"**Fundo:** {denom_social} ({selected_cnpj_summary})")
                            st.markdown(f"**Mês:** {ym[:4]}-{ym[4:]} | **Total de Posições:** {total_positions}")
                            
                            # Allocation Summary
                            alloc_df = prepare_allocation_summary(df)
                            if not alloc_df.empty:
                                st.markdown("### PORTFÓLIO, ALOCAÇÃO E PRINCIPAIS TESES")
                                for _, row in alloc_df.iterrows():
                                    st.markdown(f"{row['PCT_STR']} {row['CATEGORIA']}")
                                
                                # Donut Chart with Ceres Wealth colors and transparent background
    # Donut Chart with Ceres Wealth colors and transparent background
    # Donut Chart with Ceres Wealth colors and transparent background
                                if len(alloc_df) > 1:
                                    # Ceres Wealth colors
                                    ceres_colors = [
                                        "#8c6239",
                                        "#dedede",
                                        "#88888B",
                                        "#b08568",
                                        "#1f6c9c",
                                        "#973E11",
                                        "#997a00"
                                    ]
                                    # Use first N colors where N = len(alloc_df)
                                    num_cats = len(alloc_df)
                                    colors_to_use = ceres_colors[:num_cats]
                                    
                                    # Create custom labels with percentages
                                    alloc_df["LABEL_PCT"] = (
                                        alloc_df["CATEGORIA"] + " - " +
                                        (alloc_df["PCT_CARTEIRA"] * 100).round(1).astype(str) + "%"
                                    )
                                    
                                    fig = go.Figure(data=[go.Pie(
                                        labels=alloc_df["CATEGORIA"].tolist(),
                                        values=(alloc_df["PCT_CARTEIRA"] * 100).tolist(),
                                        text=alloc_df["LABEL_PCT"].tolist(),
                                        textposition='inside',
                                        textinfo='percent',
                                        hole=0.3,  # Donut style
                                        marker=dict(colors=colors_to_use)
                                    )])
                                    
                                    # Transparent background with minimal margins
                                    fig.update_layout(
                                        title=" ",
                                        paper_bgcolor='rgba(0,0,0,0)',
                                        plot_bgcolor='rgba(0,0,0,0)',
                                        showlegend=True,  # Enable the legend
                                        legend=dict(
                                            orientation="h",  # Horizontal orientation
                                            yanchor="top",
                                            y=-0.15,  # Adjust this value to position lower/higher
                                            xanchor="center",
                                            x=0.5
                                        ),
                                            margin=dict(l=20, r=20, t=20, b=80),  # Tight margins
                                            height=450,
                                    )
                                    
                                    fig.update_traces(
                                        textfont=dict(size=16),
                                        hovertemplate='<b>%{label}</b><br>%{value:.1f}%<extra></extra>'
                                    )
                                    
                                    st.plotly_chart(fig, use_container_width=True)
                                else:
                                    st.info("Não há categorias suficientes para o gráfico de alocação.")
                            else:
                                st.warning("Não foi possível preparar o resumo de alocação (verifique TP_ATIVO e PCT_CARTEIRA).")
                            
                            # Top Positions
                            top_df = prepare_top_positions(df, ym)
                            if not top_df.empty:
                                st.markdown("### MAIORES POSIÇÕES NA CARTEIRA")
                                for _, row in top_df.iterrows():
                                    st.markdown(f"{row['POSICAO_FULL']}")
                            else:
                                st.warning("Não foi possível preparar as top posições (verifique colunas TP_ATIVO, EMISSOR/EMISSOR_LIGADO, DT_FIM_VIGENCIA).")
                            
                            # Option to download full data as CSV
                            csv_bytes = df.to_csv(index=False).encode("utf-8")
                            st.download_button(
                                "Download CSV completo do fundo",
                                data=csv_bytes,
                                file_name=f"carteira_{selected_cnpj_summary}_{ym}.csv",
                                mime="text/csv",
                            )
                except Exception as e:
                    st.error(f"Erro ao processar resumo: {e}")

    st.caption(
        "Notes: Only the newest K months available locally are checked for "
        "absence. The app looks for cda_fi_CONFID_YYYYMM.csv and "
        "cda_fi_BLC_[1..8]_YYYYMM.csv inside each ZIP. Malformed lines are "
        "skipped; quotes issues are handled with fallback parsing. Database "
        "stores one snapshot per CNPJ (latest month only, as table rows). PCT_CARTEIRA is a "
        "0–1 fraction computed from VL_MERC_POS_FINAL over the total of the rows. "
        "Resumo: Usa TP_ATIVO para categorizar alocações e agrupa posições por ativo + emissor + vencimento."
    )


with tab2:
    st.markdown("Página para baixar dados do CDI será implementada em breve.")

with tab3:
    st.markdown("Página para visualizar rentabilidade dos fundos será implementada em breve.")
        # streamlit_app.py
    # CVM FI (INF_DIARIO) + comparação com CDI/IBOV a partir de SQLite
    # - Tabelas: great_tables (GT) renderizadas inline com
    #   streamlit_extras.great_tables.great_tables
    # - Export: SVG (vetor, transparente) e PNG (via conversão client-side)
    # - Gráficos: Plotly (barras/linhas) mantidos
    # - Novidades:
    #   1) Controles para largura de TODAS as tabelas (sidebar). Altura automática.
    #   2) Cabeçalho com cor sólida = 1ª cor da paleta CERES (sem “white on white”).
    #   3) Somente linhas horizontais (sem barras verticais).
    #   4) Visual mais “clean” (hover, leve zebra, sem cara de Excel).
    #   5) Downloads PNG/SVG em alta definição com ALTURA EXATA da tabela renderizada.

    import os
    import re
    import sqlite3
    import zipfile
    from typing import Dict, Iterable, List, Optional, Tuple

    import base64
    import math

    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    import requests
    import streamlit as st
    import streamlit.components.v1 as components
    from plotly.subplots import make_subplots

    # great_tables
    from great_tables import GT, style, loc

    # streamlit-extras for Great Tables inline rendering
    # Official API: great_tables(table: GT, width='stretch'|'content'|int)
    try:
        from streamlit_extras.great_tables import great_tables as stx_great_tables
    except Exception:
        stx_great_tables = None  # fallback HTML

    BASE_URL = "https://dados.cvm.gov.br/dados/FI/DOC/INF_DIARIO/DADOS"
    FILE_PREFIX = "inf_diario_fi"
    START_YEAR = 2021
    DEFAULT_DB_PATH = "./data/data_fundos.db"
    DEFAULT_BENCH_DB_PATH = "./data/data_cdi_ibov.db"

    # ---------------------- Color Theme (CERES) -------------------------

    CERES_COLORS = [
        "#8c6239",  # bronze (1ª cor da paleta)
        "#dedede",
        "#dedede",
        "#8c6239",
        "#8c6239",
        "#7a6200",
        "#013220",
    ]


    def get_palette() -> List[str]:
        return st.session_state.get("ceres_colors", CERES_COLORS)


    def name_color_map(names: List[str]) -> Dict[str, str]:
        pal = get_palette()
        return {name: pal[i % len(pal)] for i, name in enumerate(names)}


    # --------------- Utilities (paths, dates, strings) -----------------


    def ensure_dir(path: str) -> None:
        os.makedirs(path, exist_ok=True)


    def yyyymm_list(start_year: int = START_YEAR) -> List[str]:
        today = pd.Timestamp.today().normalize()
        start = pd.Period(f"{start_year:04d}-01", freq="M")
        end = today.to_period("M")
        months = []
        p = start
        while p <= end:
            months.append(f"{p.year:04d}{p.month:02d}")
            p = p + 1
        return months


    def last_n_months(n: int) -> List[str]:
        end = pd.Timestamp.today().normalize().to_period("M")
        months = []
        for k in range(n):
            p = end - k
            months.append(f"{p.year:04d}{p.month:02d}")
        return months


    def month_to_url(yyyymm: str) -> str:
        return f"{BASE_URL}/{FILE_PREFIX}_{yyyymm}.zip"


    def month_to_local_path(data_dir: str, yyyymm: str) -> str:
        return os.path.join(data_dir, f"{FILE_PREFIX}_{yyyymm}.zip")


    def sanitize_cnpj(s: str) -> str:
        if s is None:
            return ""
        return re.sub(r"\D", "", s)


    # ---------------------- Download management ------------------------


    def download_file(url: str, dest_path: str, timeout: int = 60) -> None:
        tmp_path = dest_path + ".part"
        with requests.get(url, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            chunk = 1 << 14
            with open(tmp_path, "wb") as f:
                for b in r.iter_content(chunk_size=chunk):
                    if b:
                        f.write(b)
        os.replace(tmp_path, dest_path)


    def refresh_local_archive(
        data_dir: str, force_last_n: int = 2, start_year: int = START_YEAR
    ) -> Tuple[List[str], List[str], List[str]]:
        ensure_dir(data_dir)
        all_months = yyyymm_list(start_year)
        forced = set(last_n_months(force_last_n))
        downloaded, skipped = [], []

        progress = st.progress(0.0, text="Checando e baixando arquivos...")
        for i, ym in enumerate(all_months, start=1):
            url = month_to_url(ym)
            local_path = month_to_local_path(data_dir, ym)
            must_fetch = (ym in forced) or (not os.path.exists(local_path))
            try:
                if must_fetch:
                    download_file(url, local_path)
                    downloaded.append(ym)
                else:
                    skipped.append(ym)
            except Exception as e:
                st.error(f"Falha ao baixar {ym}: {e}")
            progress.progress(i / len(all_months))
        progress.empty()
        return all_months, downloaded, skipped


    def inner_csv_name(zf: zipfile.ZipFile) -> str:
        csvs = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not csvs:
            raise FileNotFoundError("Sem CSV dentro do ZIP")
        if len(csvs) == 1:
            return csvs[0]
        return sorted(csvs)[0]


    def _normalize(col: str) -> str:
        return col.replace("\ufeff", "").strip().upper()


    def _detect_columns_from_header(
        zf: zipfile.ZipFile, csv_name: str, encoding: str
    ) -> Optional[Tuple[str, str, str]]:
        try:
            with zf.open(csv_name) as f:
                header_df = pd.read_csv(
                    f, sep=";", nrows=0, encoding=encoding, engine="python"
                )
        except Exception:
            return None

        orig_cols = list(header_df.columns)
        norm_map = {_normalize(c): c for c in orig_cols}

        cnpj_norm = (
            "CNPJ_FUNDO_CLASSE"
            if "CNPJ_FUNDO_CLASSE" in norm_map
            else ("CNPJ_FUNDO" if "CNPJ_FUNDO" in norm_map else None)
        )
        dt_norm = "DT_COMPTC" if "DT_COMPTC" in norm_map else None
        quota_norm = "VL_QUOTA" if "VL_QUOTA" in norm_map else None

        if not (cnpj_norm and dt_norm and quota_norm):
            return None

        return norm_map[cnpj_norm], norm_map[dt_norm], norm_map[quota_norm]


    def _parse_decimal_series_flexible(s: pd.Series) -> pd.Series:
        if s is None or len(s) == 0:
            return pd.to_numeric(s, errors="coerce")
        x = s.astype(str).str.strip()
        dec_comma = x.str.contains(r",\d{1,12}$", regex=True, na=False)
        dec_dot = x.str.contains(r"\.\d{1,12}$", regex=True, na=False)

        y = x.copy()
        y.loc[dec_comma] = (
            y.loc[dec_comma]
            .str.replace(".", "", regex=False)
            .str.replace(",", ".", regex=False)
        )
        y.loc[dec_dot] = y.loc[dec_dot].str.replace(",", "", regex=False)
        rest = ~(dec_comma | dec_dot)
        if rest.any():
            y.loc[rest] = y.loc[rest].str.replace(",", "", regex=False)
        return pd.to_numeric(y, errors="coerce")


    # ---------------------------- Database -----------------------------


    def get_conn(db_path: str) -> sqlite3.Connection:
        db_dir = os.path.dirname(db_path) or "."
        ensure_dir(db_dir)
        conn = sqlite3.connect(
            db_path, detect_types=sqlite3.PARSE_DECLTYPES, check_same_thread=False
        )
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA temp_store=MEMORY;")
        conn.execute("PRAGMA foreign_keys=ON;")
        init_db(conn)
        return conn


    def get_conn_readonly(db_path: str) -> sqlite3.Connection:
        if not os.path.exists(db_path):
            raise FileNotFoundError(
                f"Arquivo de benchmarks não encontrado: {db_path}"
            )
        uri = f"file:{os.path.abspath(db_path)}?mode=ro"
        conn = sqlite3.connect(
            uri, uri=True, detect_types=sqlite3.PARSE_DECLTYPES, check_same_thread=False
        )
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn


    def init_db(conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS nav_daily (
                cnpj TEXT NOT NULL,
                dt   TEXT NOT NULL,  -- YYYY-MM-DD
                vl_quota REAL NOT NULL,
                PRIMARY KEY (cnpj, dt)
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cnpjs (
                cnpj TEXT PRIMARY KEY,
                display_name TEXT,
                ccy TEXT
            );
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_nav_cnpj_dt ON nav_daily(cnpj, dt);"
        )


    def db_mtime(db_path: str) -> float:
        return os.path.getmtime(db_path) if os.path.exists(db_path) else 0.0


    def upsert_cnpj_meta(
        conn: sqlite3.Connection, cnpj: str, display_name: str, ccy: str
    ) -> None:
        conn.execute(
            """
            INSERT INTO cnpjs (cnpj, display_name, ccy)
            VALUES (?, ?, ?)
            ON CONFLICT(cnpj) DO UPDATE SET
                display_name = excluded.display_name,
                ccy = excluded.ccy;
            """,
            (cnpj, display_name, ccy),
        )


    def cnpjs_in_db(conn: sqlite3.Connection) -> List[str]:
        cur = conn.execute(
            """
            SELECT DISTINCT cnpj FROM nav_daily
            ORDER BY cnpj;
            """
        )
        return [r[0] for r in cur.fetchall()]


    def cnpj_meta_map_from_db(conn: sqlite3.Connection) -> Dict[str, Dict[str, str]]:
        cur = conn.execute(
            "SELECT cnpj, COALESCE(display_name, ''), COALESCE(ccy, '') FROM cnpjs;"
        )
        out: Dict[str, Dict[str, str]] = {}
        for cnpj, name, ccy in cur.fetchall():
            out[cnpj] = {"display_name": name, "ccy": ccy}
        return out


    def insert_nav_daily_batch(
        conn: sqlite3.Connection, rows: List[Tuple[str, str, float]]
    ) -> int:
        if not rows:
            return 0
        sql = "INSERT OR IGNORE INTO nav_daily (cnpj, dt, vl_quota) VALUES (?, ?, ?)"
        before = conn.total_changes
        conn.executemany(sql, rows)
        conn.commit()
        return conn.total_changes - before


    # ---------------- Ingestion from monthly ZIPs into DB ---------------


    def ingest_zip_for_cnpjs_to_db(
        zip_path: str, target_cnpjs: Optional[Iterable[str]], conn: sqlite3.Connection
    ) -> int:
        if not os.path.exists(zip_path):
            return 0

        with zipfile.ZipFile(zip_path) as zf:
            csv_name = inner_csv_name(zf)
            encodings = ("utf-8", "utf-8-sig", "latin1", "cp1252")
            last_err = None
            total_inserted = 0
            target_set = set(target_cnpjs) if target_cnpjs else None

            for enc in encodings:
                try:
                    cols = _detect_columns_from_header(zf, csv_name, enc)
                    if cols is None:
                        continue
                    cnpj_col, dt_col, quota_col = cols

                    with zf.open(csv_name) as f:
                        chunks = pd.read_csv(
                            f,
                            sep=";",
                            usecols=[cnpj_col, dt_col, quota_col],
                            dtype={cnpj_col: str, dt_col: str, quota_col: str},
                            encoding=enc,
                            engine="python",
                            chunksize=200_000,
                        )
                        for ch in chunks:
                            ch = ch.rename(
                                columns={
                                    cnpj_col: "CNPJ_ID",
                                    dt_col: "DT_COMPTC",
                                    quota_col: "VL_QUOTA",
                                }
                            )
                            ch["CNPJ"] = ch["CNPJ_ID"].astype(str).map(sanitize_cnpj)

                            if target_set is not None:
                                ch = ch[ch["CNPJ"].isin(target_set)]
                                if ch.empty:
                                    continue

                            ch["DT_COMPTC"] = pd.to_datetime(
                                ch["DT_COMPTC"], errors="coerce", format="%Y-%m-%d"
                            )
                            ch["VL_QUOTA"] = _parse_decimal_series_flexible(
                                ch["VL_QUOTA"]
                            )
                            ch = ch.dropna(subset=["DT_COMPTC", "VL_QUOTA"])

                            if ch.empty:
                                continue

                            ch["DT_STR"] = ch["DT_COMPTC"].dt.strftime("%Y-%m-%d")
                            rows = list(
                                zip(
                                    ch["CNPJ"].astype(str).tolist(),
                                    ch["DT_STR"].tolist(),
                                    ch["VL_QUOTA"].astype(float).tolist(),
                                )
                            )

                            total_inserted += insert_nav_daily_batch(conn, rows)
                    return total_inserted
                except Exception as e:
                    last_err = e
                    continue

            if last_err:
                raise last_err
            return 0


    def preload_cnpjs_to_db(
        cnpj_map: Dict[str, Dict[str, str]],
        months: Tuple[str, ...],
        data_dir: str,
        conn: sqlite3.Connection,
    ) -> int:
        if not cnpj_map:
            return 0

        for cnpj, meta in cnpj_map.items():
            upsert_cnpj_meta(
                conn,
                cnpj=cnpj,
                display_name=meta.get("display_name", ""),
                ccy=meta.get("ccy", ""),
            )

        total_inserted = 0
        target_set = list(cnpj_map.keys())
        n_months = len(months)
        progress = st.progress(0.0, text="Pré-carregando em DB...")

        for i, ym in enumerate(months, start=1):
            zip_path = month_to_local_path(data_dir, ym)
            try:
                inserted = ingest_zip_for_cnpjs_to_db(zip_path, target_set, conn)
                total_inserted += inserted
            except Exception as e:
                st.error(f"Falha ao ingerir {ym}: {e}")
            progress.progress(i / n_months, text=f"Pré-carregando {i}/{n_months}...")
        progress.empty()
        return total_inserted


    def update_db_for_existing_cnpjs(
        months: Tuple[str, ...], data_dir: str, conn: sqlite3.Connection
    ) -> int:
        known = cnpjs_in_db(conn)
        if not known:
            return 0

        total_inserted = 0
        n_months = len(months)
        progress = st.progress(0.0, text="Atualizando DB...")

        for i, ym in enumerate(months, start=1):
            zip_path = month_to_local_path(data_dir, ym)
            try:
                inserted = ingest_zip_for_cnpjs_to_db(zip_path, known, conn)
                total_inserted += inserted
            except Exception as e:
                st.error(f"Falha ao atualizar {ym}: {e}")
            progress.progress(i / n_months, text=f"Atualizando {i}/{n_months}...")
        progress.empty()
        return total_inserted


    # ------------------------ Query from DB ----------------------------


    @st.cache_data(show_spinner=False)
    def load_history_from_db_cached(
        db_path: str, cnpj_digits: str, cache_buster: float
    ) -> pd.DataFrame:
        conn = get_conn(db_path)
        try:
            df = pd.read_sql_query(
                """
                SELECT dt as DT_COMPTC, vl_quota as VL_QUOTA
                FROM nav_daily
                WHERE cnpj = ?
                ORDER BY dt ASC
                """,
                conn,
                params=(cnpj_digits,),
                parse_dates=["DT_COMPTC"],
            )
        finally:
            conn.close()
        if df.empty:
            return pd.DataFrame(
                columns=["DT_COMPTC", "VL_QUOTA"]
            ).astype({"DT_COMPTC": "datetime64[ns]", "VL_QUOTA": "float64"})
        return df


    # -------------------- Benchmarks: CDI / IBOV (DB) -------------------


    def _list_tables(conn: sqlite3.Connection) -> List[str]:
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table','view')"
        )
        return [r[0] for r in cur.fetchall()]


    def _resolve_table_name(conn: sqlite3.Connection, table_hint: str) -> str:
        names = _list_tables(conn)
        for n in names:
            if n.lower() == table_hint.lower():
                return n
        raise ValueError(
            f"Tabela '{table_hint}' não encontrada. Disponíveis: {names}"
        )


    def _table_columns(conn: sqlite3.Connection, table: str) -> List[str]:
        cur = conn.execute(f"PRAGMA table_info({table});")
        return [r[1] for r in cur.fetchall()]


    @st.cache_data(show_spinner=False)
    def load_benchmark_daily_returns(
        db_path: str, table_hint: str, cache_buster: float
    ) -> pd.DataFrame:
        conn = get_conn_readonly(db_path)
        try:
            table = _resolve_table_name(conn, table_hint)
            cols = _table_columns(conn, table)
            if "daily_return" in cols:
                q = f"SELECT date, daily_return FROM {table} ORDER BY date"
                df = pd.read_sql_query(q, conn, parse_dates=["date"])
                df["daily_return"] = pd.to_numeric(
                    df["daily_return"], errors="coerce"
                )
            elif table.lower() == "cdi" and "daily_rate_pct" in cols:
                q = f"SELECT date, daily_rate_pct FROM {table} ORDER BY date"
                df = pd.read_sql_query(q, conn, parse_dates=["date"])
                df["daily_return"] = pd.to_numeric(
                    df["daily_rate_pct"], errors="coerce"
                ) / 100.0
                df = df[["date", "daily_return"]]
            elif table.lower() == "ibov" and "close" in cols:
                q = f"SELECT date, close FROM {table} ORDER BY date"
                df = pd.read_sql_query(q, conn, parse_dates=["date"])
                df["close"] = pd.to_numeric(df["close"], errors="coerce")
                df["daily_return"] = df["close"].pct_change()
                df = df.dropna(subset=["daily_return"])[["date", "daily_return"]]
            else:
                raise ValueError(
                    f"Tabela {table} não possui colunas esperadas: {cols}"
                )
        finally:
            conn.close()
        df = df.dropna(subset=["date", "daily_return"]).reset_index(drop=True)
        return df


    # ------------------- Analytics and charts --------------------------


    def daily_series_from_nav(df_daily: pd.DataFrame) -> pd.Series:
        dfd = df_daily.sort_values("DT_COMPTC")
        s = dfd.set_index("DT_COMPTC")["VL_QUOTA"].pct_change()
        return s.dropna()


    def series_daily_to_monthly(s: pd.Series) -> pd.Series:
        if s is None or s.empty:
            return pd.Series(dtype=float)
        s = s.copy()
        s.index = pd.to_datetime(s.index).tz_localize(None)
        grp = s.groupby(s.index.to_period("M"))
        ret_m = grp.apply(lambda g: float(np.prod(1.0 + g.values) - 1.0))
        ret_m.index = ret_m.index.to_timestamp("M")
        ret_m.name = "ret_m"
        return ret_m


    def series_daily_to_cumret(s: pd.Series) -> pd.Series:
        if s is None or s.empty:
            return pd.Series(dtype=float)
        s = s.copy()
        s.index = pd.to_datetime(s.index).tz_localize(None)
        return (1.0 + s).cumprod() - 1.0


    def compound(series: pd.Series) -> float:
        if series is None or series.empty:
            return float("nan")
        return float(np.prod(1.0 + series.values) - 1.0)


    def _pct_str(x: float, places: int = 2) -> str:
        if pd.isna(x):
            return "-"
        return f"{x*100:.{places}f}%"


    def _pp_str(x: float, places: int = 2) -> str:
        if pd.isna(x):
            return "-"
        return f"{x*100:.{places}f} p.p."


    def annual_returns_from_monthly(ret_m: pd.Series) -> pd.Series:
        if ret_m.empty:
            return pd.Series(dtype=float)
        s = ret_m.copy()
        s.index = pd.to_datetime(s.index)
        grp = s.groupby(s.index.year)

        def _year_ret(g: pd.Series) -> float:
            months = g.index.month.nunique()
            if months == 12:
                return float(np.prod(1.0 + g.values) - 1.0)
            return np.nan

        yr = grp.apply(_year_ret)
        yr.index.name = "year"
        return yr.dropna()


    def make_monthly_plotly_chart_multi(
        series_map: Dict[str, pd.Series],
        title: str,
        color_map: Optional[Dict[str, str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
    ) -> go.Figure:
        default_height = 400
        if not series_map or all(s.empty for s in series_map.values()):
            return go.Figure(
                layout=dict(
                    title=title,
                    height=height or default_height,
                    width=width,
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                )
            )

        all_months = sorted(
            set().union(
                *[set(pd.to_datetime(s.index)) for s in series_map.values()]
            )
        )

        fig = make_subplots(specs=[[{"secondary_y": False}]])
        pal = get_palette()
        for idx, (name, s) in enumerate(series_map.items()):
            s = s.sort_index().reindex(all_months)
            col = (color_map or {}).get(name, pal[idx % len(pal)])
            fig.add_trace(
                go.Bar(
                    x=[pd.Timestamp(x).strftime("%Y-%m") for x in s.index],
                    y=s.values,
                    name=name,
                    marker_color=col,
                    hovertemplate=(
                        "Série: "
                        + name
                        + "<br>Mês: %{x}<br>Retorno: %{y:.2%}<extra></extra>"
                    ),
                )
            )

        fig.update_layout(
            title=title,
            barmode="group",
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0
            ),
            height=height or default_height,
            width=width,
            margin=dict(l=40, r=30, t=60, b=40),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        fig.update_yaxes(
            title_text="Retorno mensal",
            tickformat=".0%",
            zeroline=True,
            zerolinecolor="#888",
            gridcolor="#eee",
        )
        fig.update_xaxes(title_text="Mês", showgrid=False)
        return fig


    def make_nav_line_chart_multi(
        cumret_map: Dict[str, pd.Series],
        title: str,
        color_map: Optional[Dict[str, str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
    ) -> go.Figure:
        default_height = 400
        if not cumret_map:
            return go.Figure(
                layout=dict(
                    title=title,
                    height=height or default_height,
                    width=width,
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                )
            )

        pal = get_palette()
        fig = go.Figure()
        for idx, (name, s) in enumerate(cumret_map.items()):
            s = s.sort_index()
            col = (color_map or {}).get(name, pal[idx % len(pal)])
            fig.add_trace(
                go.Scatter(
                    x=s.index,
                    y=s.values,
                    mode="lines",
                    name=name,
                    line=dict(width=2.2, color=col),
                    hovertemplate=(
                        "Série: "
                        + name
                        + "<br>Data: %{x|%d-%m-%Y}"
                        + "<br>Ret. acumulado: %{y:.2%}<extra></extra>"
                    ),
                )
            )
        fig.update_layout(
            title=title,
            height=height or default_height,
            width=width,
            margin=dict(l=40, r=30, t=50, b=40),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        fig.update_yaxes(
            title_text="Retorno acumulado", tickformat=".0%", gridcolor="#eee"
        )
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)
        fig.update_layout(
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.1,
                xanchor="center",
                x=0.5,
                title=None,
                bgcolor="rgba(0,0,0,0)",
            ),
            margin=dict(b=80),
        )
        return fig


    # ---------- Great Tables: theme, render inline, export helpers ----------


    def render_gt_inline(
        gt_obj: GT,
        *,
        width_px: Optional[int] = None,
        width_mode: str = "stretch",  # 'stretch' | 'content'
        fallback_height: int = 400,
    ) -> None:
        """
        Renderiza uma GT inline (via streamlit-extras) ou via HTML de fallback.
        """
        if stx_great_tables is not None:
            try:
                if isinstance(width_px, int) and width_px > 0:
                    stx_great_tables(gt_obj, width=width_px)
                else:
                    stx_great_tables(gt_obj, width=width_mode)  # type: ignore
                return
            except Exception as e:
                st.warning(
                    f"Falha ao usar streamlit-extras.great_tables: {e}. "
                    "Tentando fallback por HTML bruto."
                )

        # Fallback: tenta obter HTML do objeto GT
        html_str = None
        if hasattr(gt_obj, "as_raw_html"):
            try:
                html_str = gt_obj.as_raw_html()
            except Exception:
                html_str = None
        if not html_str and hasattr(gt_obj, "render"):
            try:
                out = gt_obj.render()
                if isinstance(out, dict):
                    html_str = out.get("html", None)
            except Exception:
                html_str = None
        if not html_str:
            for meth in ("as_html", "to_html"):
                if hasattr(gt_obj, meth):
                    try:
                        html_str = getattr(gt_obj, meth)()
                        break
                    except Exception:
                        pass
        if not html_str and hasattr(gt_obj, "show"):
            try:
                res = gt_obj.show()
                if isinstance(res, str) and res.strip():
                    html_str = res
                elif hasattr(res, "_repr_html_"):
                    h = res._repr_html_()
                    if isinstance(h, str) and h.strip():
                        html_str = h
            except Exception:
                pass

        if not html_str:
            st.error(
                "Não foi possível renderizar a tabela GT inline. "
                "Instale/atualize: pip install -U streamlit-extras great-tables"
            )
            return

        components.html(html_str, height=fallback_height, scrolling=True)


    def ceres_gt_css(
        pal: List[str],
        font_px: int,
    ) -> str:
        header_bg = pal[0] if pal else "#8c6239"
        body_text = "#f1f1f1"
        border_col = "rgba(222,222,222,0.18)"
        zebra = "rgba(255,255,255,0.02)"
        hover_bg = "rgba(255,255,255,0.05)"
        line_height = max(1.15, min(1.6, font_px / 11.5))
        font_stack = (
            "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Ubuntu, "
            "Cantarell, 'Helvetica Neue', Arial, 'Noto Sans', sans-serif"
        )
        return f"""
        :root {{
        --gt-font-size: {font_px}px;
        --gt-border: {border_col};
        --gt-header: {header_bg};
        }}
        * {{ box-sizing: border-box; }}
        html, body {{
        margin: 0; padding: 0;
        background: transparent !important;
        font-family: {font_stack};
        color: {body_text};
        font-size: var(--gt-font-size);
        }}
        .ceres-wrap {{
        width: 100% !important;
        height: 100% !important;
        background: transparent !important;
        display: block;
        color: {body_text};
        font-family: {font_stack} !important;
        font-size: var(--gt-font-size) !important;
        line-height: {line_height} !important;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
        }}
        .ceres-wrap table.gt_table,
        .ceres-wrap table.gt_table * {{
        font-family: inherit !important;
        font-size: inherit !important;
        }}
        table.gt_table {{
        width: 100% !important;
        max-width: 100% !important;
        margin: 0 !important;
        border-collapse: separate !important;
        border-spacing: 0 !important;
        background: transparent !important;
        border-top: 1px solid var(--gt-border) !important;
        border-bottom: 1px solid var(--gt-border) !important;
        border-left: none !important;
        border-right: none !important;
        border-radius: 0 !important;
        overflow: hidden !important;
        }}
        table.gt_table th,
        table.gt_table td {{
        font-variant-numeric: tabular-nums lining-nums;
        }}
        table.gt_table .gt_left {{ text-align: left !important; }}
        table.gt_table .gt_center {{ text-align: center !important; }}
        table.gt_table .gt_right {{ text-align: right !important; }}
        table.gt_table .gt_top {{ vertical-align: top !important; }}
        table.gt_table .gt_middle {{ vertical-align: middle !important; }}
        table.gt_table .gt_bottom {{ vertical-align: bottom !important; }}
        table.gt_table thead,
        table.gt_table thead tr,
        table.gt_table thead tr th,
        table.gt_table thead tr td,
        thead.gt_col_headings,
        thead.gt_col_headings tr,
        thead.gt_col_headings tr th.gt_col_heading,
        .gt_column_spanner_outer,
        .gt_column_spanner {{
        background: var(--gt-header) !important;
        color: #ffffff !important;
        text-transform: uppercase !important;
        letter-spacing: .03em !important;
        font-weight: 800 !important;
        border-right: none !important;
        border-bottom: 1px solid var(--gt-border) !important;
        padding: 12px 14px !important;
        text-align: center !important;
        }}
        tbody.gt_table_body tr td {{
        background: transparent !important;
        color: {body_text} !important;
        padding: 11px 14px !important;
        border-top: 1px solid var(--gt-border) !important;
        border-right: none !important;
        font-weight: 600 !important;
        vertical-align: middle !important;
        }}
        tbody.gt_table_body tr:nth-child(even) td {{
        background: {zebra} !important;
        }}
        tbody.gt_table_body tr:hover td {{
        background: {hover_bg} !important;
        }}
        tbody.gt_table_body tr {{
        line-height: {line_height} !important;
        }}
        .gt_table .gt_table_body, .gt_table .gt_col_headings {{
        background: transparent !important;
        }}
        """

    def inject_gt_runtime_css(font_px: int, pal: List[str]) -> None:
        """
        CSS global para as tabelas inline (fundo transparente, cabeçalho bronze,
        sem barras verticais).
        """
        css = ceres_gt_css(pal=pal, font_px=font_px)
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


    def estimate_gt_render_height(
        *, font_px: int, n_rows: int, n_header_rows: int = 1, title_rows: int = 0
    ) -> int:
        """
        Estima altura necessária para caber a tabela (quase sem sobra).
        """
        header_h = math.ceil(font_px * 2.2)
        row_h = math.ceil(font_px * 2.0)
        title_h = math.ceil(font_px * 2.4)
        pad = math.ceil(font_px * 0.6)
        height_px = (
            title_rows * title_h + n_header_rows * header_h + n_rows * row_h + pad
        )
        return int(height_px)


    def gt_to_html_str(gt_obj: GT) -> str:
        """
        Obtém HTML do GT em diferentes versões.
        """
        if hasattr(gt_obj, "as_raw_html"):
            try:
                html_str = gt_obj.as_raw_html()
                if isinstance(html_str, str) and html_str.strip():
                    return html_str
            except Exception:
                pass

        if hasattr(gt_obj, "render"):
            try:
                out = gt_obj.render()
                if isinstance(out, dict) and isinstance(out.get("html"), str):
                    html_str = out["html"]
                    if html_str.strip():
                        return html_str
            except Exception:
                pass

        for meth in ("as_html", "to_html"):
            if hasattr(gt_obj, meth):
                try:
                    html_str = getattr(gt_obj, meth)()
                    if isinstance(html_str, str) and html_str.strip():
                        return html_str
                except Exception:
                    pass

        if hasattr(gt_obj, "show"):
            try:
                res = gt_obj.show()
                if isinstance(res, str) and res.strip():
                    return res
                if hasattr(res, "_repr_html_"):
                    h = res._repr_html_()
                    if isinstance(h, str) and h.strip():
                        return h
            except Exception:
                pass

        raise RuntimeError("great_tables: não consegui obter HTML para exportar.")


    def gt_to_svg_bytes(
        *,
        gt_obj: GT,
        pal: List[str],
        width_px: int,
        height_px: int,
        font_px: int,
    ) -> bytes:
        """
        Envolve o HTML do GT em SVG com foreignObject.
        Força largura/altura para preencher o canvas por completo.
        """
        html_inner = gt_to_html_str(gt_obj)
        css = ceres_gt_css(pal=pal, font_px=font_px)
        svg = f"""<?xml version="1.0" encoding="UTF-8" standalone="no"?>
    <svg xmlns="http://www.w3.org/2000/svg"
        width="{width_px}" height="{height_px}"
        viewBox="0 0 {width_px} {height_px}" preserveAspectRatio="none">
    <foreignObject x="0" y="0" width="{width_px}" height="{height_px}">
        <div xmlns="http://www.w3.org/1999/xhtml"
            class="ceres-wrap" style="width:{width_px}px;height:{height_px}px;">
        <style>{css}</style>
        {html_inner}
        </div>
    </foreignObject>
    </svg>
    """
        return svg.encode("utf-8")


    def render_gt_inline_with_runtime_export(
        *,
        gt_obj: GT,
        filename_base: str,
        width_px: int,
        n_rows: int,
        font_px: int,
        key_suffix: str,
        n_header_rows: int = 1,
        title_rows: int = 0,
    ) -> None:
        """
        Renderiza a GT inline e adiciona botões de export (SVG/PNG) usando
        a altura EXATA medida do elemento da tabela no browser.
        - Largura: controlada pelo parâmetro width_px (slider mantém).
        - Altura: medida via getBoundingClientRect().height no momento do clique.
        """
        pal = get_palette()
        css = ceres_gt_css(pal=pal, font_px=font_px)
        css_js = css.replace("\\", "\\\\").replace("`", "\\`")
        table_html = gt_to_html_str(gt_obj)
        # Altura do iframe para exibir sem barra de rolagem (estimativa só para UI)
        est_h = estimate_gt_render_height(
            font_px=font_px,
            n_rows=n_rows,
            n_header_rows=n_header_rows,
            title_rows=title_rows,
        )
        # Espaço extra para a barra de botões
        frame_height = int(est_h + 60)

        html = f"""
        <div id="wrap_{key_suffix}" style="width:{width_px}px;margin:0;padding:0;background:transparent;">
        <style>{css}</style>
        {table_html}
        <div style="margin-top:8px;display:flex;gap:8px;">
            <button id="btn_svg_{key_suffix}" style="padding:6px 10px;cursor:pointer;">
            Download SVG
            </button>
            <button id="btn_png_{key_suffix}" style="padding:6px 10px;cursor:pointer;">
            Download PNG (HD)
            </button>
        </div>
        </div>
        <script>
        (function() {{
            const wrap = document.getElementById("wrap_{key_suffix}");
            const table = wrap.querySelector("table.gt_table");
            if (table) {{
            table.style.width = "{width_px}px";
            table.style.maxWidth = "{width_px}px";
            }}
            function makeSVG() {{
            const w = {width_px};
            const h = Math.ceil(table.getBoundingClientRect().height);
            const css = `{css_js}`;
            const svg = `<?xml version="1.0" encoding="UTF-8" standalone="no"?>` +
                `<svg xmlns="http://www.w3.org/2000/svg" width="${{w}}" height="${{h}}" viewBox="0 0 ${{w}} ${{h}}" preserveAspectRatio="none">` +
                `<foreignObject x="0" y="0" width="${{w}}" height="${{h}}">` +
                `<div xmlns="http://www.w3.org/1999/xhtml" class="ceres-wrap" style="width:${{w}}px;height:${{h}}px;">` +
                `<style>` + css + `</style>` +
                table.outerHTML +
                `</div></foreignObject></svg>`;
            return {{ svg, w, h }};
            }}
            function dlBlob(blob, name) {{
            const a = document.createElement("a");
            a.href = URL.createObjectURL(blob);
            a.download = name;
            document.body.appendChild(a);
            a.click();
            URL.revokeObjectURL(a.href);
            a.remove();
            }}
            document.getElementById("btn_svg_{key_suffix}").onclick = function() {{
            const {{ svg }} = makeSVG();
            const blob = new Blob([svg], {{ type: "image/svg+xml" }});
            dlBlob(blob, "{filename_base}.svg");
            }};
            document.getElementById("btn_png_{key_suffix}").onclick = function() {{
            const {{ svg, w, h }} = makeSVG();
            const img = new Image();
            img.onload = function() {{
                const scale = 3;
                const canvas = document.createElement("canvas");
                canvas.width = Math.round(w * scale);
                canvas.height = Math.round(h * scale);
                const ctx = canvas.getContext("2d");
                ctx.imageSmoothingEnabled = true;
                ctx.imageSmoothingQuality = "high";
                ctx.scale(scale, scale);
                ctx.clearRect(0, 0, w, h);
                ctx.drawImage(img, 0, 0, w, h);
                canvas.toBlob(function(blob) {{
                dlBlob(blob, "{filename_base}.png");
                }}, "image/png");
            }};
            img.src = "data:image/svg+xml;charset=utf-8," + encodeURIComponent(svg);
            }};
        }})();
        </script>
        """
        components.html(html, height=frame_height, scrolling=True)


    # ---------- great_tables: styling helpers ----------


    def _gt_apply_header_theme(
        tbl: GT, header_colored: bool, pal: List[str], table_font_px: int
    ) -> GT:
        # Texto do cabeçalho (cor vem do CSS global)
        tbl = tbl.tab_style(
            style=[
                style.text(
                    color="#ffffff",
                    weight="bold",
                    size=f"{table_font_px}px",
                    transform="uppercase",
                )
            ],
            locations=loc.column_labels(),
        )
        return tbl


    def _gt_apply_body_theme(tbl: GT, pal: List[str], table_font_px: int) -> GT:
        tbl = tbl.tab_style(
            style=[style.text(color="#f1f1f1", size=f"{table_font_px}px")],
            locations=loc.body(),
        )
        if hasattr(tbl, "opt_vertical_padding"):
            tbl = tbl.opt_vertical_padding(scale=0.8)
        return tbl


    # -------- great_tables builders ---------


    def build_performance_comparison_table_gt(
        rows: List[Dict[str, object]],
        years_cols: Optional[List[int]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        table_font_size: Optional[int] = None,
        header_colored: bool = False,
    ) -> Tuple[GT, int, Optional[int]]:
        if not rows:
            df = pd.DataFrame(columns=["Série", "Ccy", "As of", "YTD", "3Y"])
            gt_tbl = GT(df)
            return gt_tbl, int(height or 400), width

        all_years = set()
        for r in rows:
            yrs = getattr(r["yr_returns"], "index", [])
            all_years |= set(int(y) for y in yrs)

        curr_years = [pd.Timestamp(r["as_of"]).year for r in rows]
        max_curr = max(curr_years) if curr_years else None
        if max_curr is not None:
            all_years = {y for y in all_years if y != max_curr}
        years_cols = (
            sorted(all_years, reverse=True)[:5] if years_cols is None else years_cols
        )

        def fmt_val(fmt: str, v: float) -> str:
            if pd.isna(v):
                return "-"
            if fmt == "pp":
                return f"{v*100:.2f} p.p."
            return f"{v*100:.2f}%"

        recs: List[Dict[str, str]] = []
        for r in rows:
            fmt = str(r.get("fmt", "pct"))
            as_of_str = pd.Timestamp(r["as_of"]).strftime("%d.%m.%Y")
            row_dict: Dict[str, str] = {
                "Série": str(r["name"]),
                "Ccy": str(r.get("ccy", "")),
                "As of": as_of_str,
                "YTD": fmt_val(fmt, r.get("ytd", np.nan)),
                "3Y": fmt_val(fmt, r.get("r3y", np.nan)),
            }
            yr: pd.Series = r["yr_returns"]
            for y in years_cols:
                v = float(yr.get(y, np.nan)) if isinstance(yr, pd.Series) else np.nan
                row_dict[str(y)] = fmt_val(fmt, v)
            recs.append(row_dict)

        df = pd.DataFrame.from_records(recs)

        pal = get_palette()
        fsize = int(table_font_size or 14)

        gt_tbl = GT(df).cols_align(columns=list(df.columns), align="center")
        gt_tbl = _gt_apply_header_theme(gt_tbl, header_colored, pal, fsize)
        gt_tbl = _gt_apply_body_theme(gt_tbl, pal, fsize)

        return gt_tbl, int(height or 400), width


    def build_last_12m_returns_table_gt(
        series_map: Dict[str, pd.Series],
        row_formats: Optional[Dict[str, str]] = None,
        row_color_modes: Optional[Dict[str, str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        table_font_size: Optional[int] = None,
        header_colored: bool = False,
    ) -> Tuple[GT, int, Optional[int]]:
        row_formats = row_formats or {}

        if not series_map or all(s.empty for s in series_map.values()):
            df = pd.DataFrame(columns=["Série"])
            gt_tbl = GT(df)
            return gt_tbl, int(height or 400), width

        all_months = sorted(
            set().union(
                *[set(pd.to_datetime(s.index)) for s in series_map.values()]
            )
        )
        last12 = all_months[-12:]
        month_labels = [pd.Timestamp(d).strftime("%b %Y") for d in last12]

        def fmt_val(name: str, v: float) -> str:
            fmt = row_formats.get(name, "pct")
            if pd.isna(v):
                return "-"
            return f"{v*100:.2f}%" if fmt == "pct" else f"{v*100:.2f} p.p."

        records: List[Dict[str, str]] = []
        for name, s in series_map.items():
            row: Dict[str, str] = {"Série": name}
            for date, lbl in zip(last12, month_labels):
                val = float(s.get(date, np.nan)) if not s.empty else np.nan
                row[lbl] = fmt_val(name, val)
            records.append(row)

        df = pd.DataFrame.from_records(records)

        pal = get_palette()
        fsize = int(table_font_size or 12)

        gt_tbl = GT(df)
        gt_tbl = gt_tbl.cols_align(columns=list(df.columns), align="center")
        gt_tbl = _gt_apply_header_theme(gt_tbl, header_colored, pal, fsize)
        gt_tbl = _gt_apply_body_theme(gt_tbl, pal, fsize)

        return gt_tbl, int(height or 400), width


    def build_window_perf_table_gt(
        series_m_map: Dict[str, pd.Series],
        series_d_map: Dict[str, pd.Series],
        order: Optional[List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        table_font_size: Optional[int] = None,
        header_colored: bool = False,
    ) -> Tuple[GT, int, Optional[int]]:
        def compound_last_n(ret_m: pd.Series, n: int) -> float:
            if ret_m is None or ret_m.empty:
                return float("nan")
            s = ret_m.dropna()
            if len(s) < n:
                return float("nan")
            return float(np.prod(1.0 + s.tail(n).values) - 1.0)

        def last_1m(ret_m: pd.Series) -> float:
            if ret_m is None or ret_m.empty:
                return float("nan")
            s = ret_m.dropna()
            if s.empty:
                return float("nan")
            return float(s.iloc[-1])

        def ann_vol_12m(daily: pd.Series) -> float:
            if daily is None or daily.empty:
                return float("nan")
            s = daily.dropna()
            if s.empty:
                return float("nan")
            end = pd.to_datetime(s.index.max())
            start = end - pd.DateOffset(months=12)
            win = s[(s.index > start) & (s.index <= end)]
            if len(win) < 60:
                return float("nan")
            return float(win.std(ddof=1) * np.sqrt(252.0))

        # FIX: removido parêntese extra no final da linha
        names = order or list(series_m_map.keys())

        records: List[Dict[str, float]] = []
        for name in names:
            ret_m = series_m_map.get(name, pd.Series(dtype=float))
            ret_d = series_d_map.get(name, pd.Series(dtype=float))

            p12 = compound_last_n(ret_m, 12)
            p6 = compound_last_n(ret_m, 6)
            p1 = last_1m(ret_m)
            v12 = ann_vol_12m(ret_d)

            records.append(
                {
                    "Série": name,
                    "Perf 12m": p12,
                    "Perf 6m": p6,
                    "Perf 1m": p1,
                    "Vol 12m": v12,
                }
            )

        df = pd.DataFrame.from_records(records)

        pal = get_palette()
        fsize = int(table_font_size or 12)

        gt_tbl = GT(df).cols_align(columns=list(df.columns), align="center")
        if hasattr(gt_tbl, "fmt_percent"):
            gt_tbl = (
                gt_tbl.fmt_percent(
                    columns=["Perf 12m", "Perf 6m", "Perf 1m"], decimals=2
                ).fmt_percent(columns=["Vol 12m"], decimals=2)
            )
        gt_tbl = _gt_apply_header_theme(gt_tbl, header_colored, pal, fsize)
        gt_tbl = _gt_apply_body_theme(gt_tbl, pal, fsize)

        return gt_tbl, int(height or 400), width

    def build_performance_table_gt(
        display_name: str,
        ccy: str,
        as_of: pd.Timestamp,
        ret_m: pd.Series,
        df_daily: pd.DataFrame,
        height: Optional[int] = None,
        width: Optional[int] = None,
        table_font_size: Optional[int] = None,
        header_colored: bool = False,
    ) -> Tuple[GT, int, Optional[int]]:
        def compound_local(series: pd.Series) -> float:
            if series is None or series.empty:
                return float("nan")
            return float(np.prod(1.0 + series.values) - 1.0)

        df = df_daily.sort_values("DT_COMPTC")
        last_dt = df["DT_COMPTC"].max()
        y0 = pd.Timestamp(year=last_dt.year, month=1, day=1, tz=last_dt.tz)
        first_ytd = df.loc[df["DT_COMPTC"] >= y0]
        if not first_ytd.empty:
            v0 = first_ytd.iloc[0]["VL_QUOTA"]
            v1 = df.iloc[-1]["VL_QUOTA"]
            ytd = float(v1 / v0 - 1.0) if v0 and v0 > 0 else np.nan
        else:
            ytd = np.nan

        r3y = compound_local(ret_m.tail(36)) if len(ret_m) >= 36 else np.nan

        yr = annual_returns_from_monthly(ret_m)
        years_available = sorted(yr.index.tolist(), reverse=True)
        curr_year = last_dt.year
        years_available = [y for y in years_available if y != curr_year]
        years_cols = years_available[:5]

        rec: Dict[str, object] = {
            "Solutions": display_name,
            "Ccy": ccy,
            "Performance as of": as_of.strftime("%d.%m.%Y"),
            "YTD (%)": ytd,
            "3Y (%)": r3y,
        }
        for y in years_cols:
            rec[str(y)] = float(yr.get(y, np.nan))

        df_row = pd.DataFrame([rec])

        pal = get_palette()
        fsize = int(table_font_size or 14)

        gt_tbl = GT(df_row).cols_align(columns=list(df_row.columns), align="left")
        pct_cols = ["YTD (%)", "3Y (%)"] + [str(y) for y in years_cols]
        if hasattr(gt_tbl, "fmt_percent"):
            gt_tbl = gt_tbl.fmt_percent(columns=pct_cols, decimals=2)
        gt_tbl = _gt_apply_header_theme(gt_tbl, header_colored, pal, fsize)
        gt_tbl = _gt_apply_body_theme(gt_tbl, pal, fsize)

        return gt_tbl, int(height or 400), width


    # -------------------------- UI State --------------------------------


    def init_session_state() -> None:
        if "cnpj_list" not in st.session_state:
            st.session_state.cnpj_list = {}
        if "cnpj_display_names" not in st.session_state:
            st.session_state.cnpj_display_names = {}
        if "preload_requested" not in st.session_state:
            st.session_state.preload_requested = False
        if "update_requested" not in st.session_state:
            st.session_state.update_requested = False
        if "ceres_colors" not in st.session_state:
            st.session_state.ceres_colors = CERES_COLORS.copy()
        if "table_font_size" not in st.session_state:
            st.session_state.table_font_size = 12
        if "table_header_colored" not in st.session_state:
            st.session_state.table_header_colored = False
        if "table_width_px" not in st.session_state:
            st.session_state.table_width_px = 1100
        # Alturas mantidas apenas para compat (não exibidas na UI)
        if "height_perf_table_px" not in st.session_state:
            st.session_state.height_perf_table_px = 360
        if "height_window_table_px" not in st.session_state:
            st.session_state.height_window_table_px = 320
        if "height_12m_table_px" not in st.session_state:
            st.session_state.height_12m_table_px = 340


    def add_cnpj(cnpj_input: str, display_name: str, ccy: str) -> bool:
        cnpj_digits = sanitize_cnpj(cnpj_input)
        if len(cnpj_digits) != 14:
            st.error("CNPJ deve ter 14 dígitos (apenas números).")
            return False
        if cnpj_digits in st.session_state.cnpj_list:
            st.warning(f"CNPJ {cnpj_digits} já foi adicionado.")
            return False
        st.session_state.cnpj_list[cnpj_digits] = ccy
        st.session_state.cnpj_display_names[cnpj_digits] = (
            display_name.strip() if display_name.strip() else f"CNPJ {cnpj_digits}"
        )
        return True


    def remove_cnpj(cnpj_digits: str) -> None:
        if cnpj_digits in st.session_state.cnpj_list:
            del st.session_state.cnpj_list[cnpj_digits]
            del st.session_state.cnpj_display_names[cnpj_digits]


    # -------------------------- Sidebar UI ------------------------------


    def render_cnpj_manager_sidebar(
        data_dir: str, start_year: int, db_path: str
    ) -> Tuple[
        str,
        str,
        int,
        str,
        int,
        bool,
        bool,
        str,
        bool,
        bool,
        int,
        int,
        int,
        int,
    ]:
        st.header("Configuração")

        data_dir = st.text_input(
            "Pasta local para os arquivos ZIP",
            value=data_dir,
            help="Arquivos ZIP serão armazenados/atualizados aqui.",
        )

        db_path = st.text_input(
            "Caminho do banco (SQLite)",
            value=db_path,
            help="Ex.: ./data/data_fundos.db",
        )

        start_year = st.number_input(
            "Ano inicial",
            min_value=2000,
            max_value=2100,
            value=start_year,
            step=1,
            help="Coletar a partir deste ano (recomendado: 2021+).",
        )

        force_last_n = st.slider(
            "Sempre rebaixar últimos N meses (ZIP)",
            min_value=1,
            max_value=12,
            value=2,
            help="Reapresentações são comuns no mês corrente/anterior.",
        )

        if st.button("Baixar/Atualizar ZIPs", key="update_files"):
            with st.spinner("Baixando/verificando arquivos..."):
                all_months, downloaded, skipped = refresh_local_archive(
                    data_dir=data_dir,
                    force_last_n=force_last_n,
                    start_year=start_year,
                )
            st.success(
                f"Pronto. Baixados: {len(downloaded)} | Mantidos: "
                f"{len(skipped)} | Total desde {start_year}: {len(all_months)}"
            )

        st.divider()
        st.subheader("Gerenciar CNPJs")

        cnpj_input = st.text_input(
            "CNPJ do fundo/classe",
            value="",
            placeholder="00.000.000/0000-00",
            help="Use o CNPJ da classe conforme INF_DIARIO.",
            key="cnpj_input",
        )
        display_name = st.text_input(
            "Nome para exibição",
            value="",
            help="Nome amigável para identificar o fundo",
            key="display_name",
        )
        ccy = st.text_input("Moeda", value="BRL", key="ccy")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Adicionar CNPJ", key="add_cnpj"):
                if add_cnpj(cnpj_input, display_name, ccy):
                    st.success("CNPJ adicionado!")
                    st.rerun()
        with col2:
            pass

        if st.session_state.cnpj_list:
            st.divider()
            st.subheader("CNPJs adicionados")
            for cnpj_digits in list(st.session_state.cnpj_list.keys()):
                c1_row, c2_row = st.columns([3, 1])
                with c1_row:
                    st.write(
                        f"**{st.session_state.cnpj_display_names[cnpj_digits]}** "
                        f"({cnpj_digits}) — "
                        f"{st.session_state.cnpj_list[cnpj_digits]}"
                    )
                with c2_row:
                    if st.button("❌", key=f"remove_{cnpj_digits}", help="Remover"):
                        remove_cnpj(cnpj_digits)
                        st.rerun()

            st.divider()
            preload_clicked = st.button(
                "Pré-carregar em DB (todos listados)", key="preload_db"
            )
        else:
            preload_clicked = False

        update_last_n = st.slider(
            "Atualizar DB: últimos N meses",
            min_value=1,
            max_value=24,
            value=2,
            help="Ingerir apenas os últimos N meses para CNPJs já no DB.",
        )
        update_clicked = st.button("Atualizar DB (últimos N meses)", key="update_db")

        st.divider()

        st.subheader("Benchmarks (CDI/IBOV)")
        bench_db_path = st.text_input(
            "Caminho do banco (benchmarks)",
            value=DEFAULT_BENCH_DB_PATH,
            help="Banco com tabelas 'cdi' e/ou 'ibov'.",
            key="bench_db_path",
        )
        compare_cdi = st.checkbox("Comparar com CDI", value=False, key="cmp_cdi")
        compare_ibov = st.checkbox("Comparar com IBOV", value=False, key="cmp_ibov")

        st.divider()
        st.subheader("Tabelas — Tamanho")
        table_width_px = st.slider(
            "Largura (px) — TODAS as tabelas",
            min_value=600,
            max_value=2400,
            value=int(st.session_state.table_width_px),
            step=20,
            help="Todas as tabelas manterão a mesma largura.",
            key="table_width_px",
        )
        # Alturas passam a ser automáticas (iguais à tabela na exportação).
        height_perf_table_px = int(st.session_state.height_perf_table_px)
        height_window_table_px = int(st.session_state.height_window_table_px)
        height_12m_table_px = int(st.session_state.height_12m_table_px)

        st.divider()
        st.subheader("Estilo / Paleta")
        st.number_input(
            "Tamanho da fonte (px) nas tabelas",
            min_value=8,
            max_value=24,
            value=int(st.session_state.table_font_size),
            step=1,
            key="table_font_size",
        )
        cols = st.columns(2)
        for i, default_col in enumerate(CERES_COLORS):
            with cols[i % 2]:
                picked = st.color_picker(
                    f"Cor {i+1}",
                    value=st.session_state.ceres_colors[i],
                    key=f"ceres_color_{i}",
                )
                st.session_state.ceres_colors[i] = picked
        if st.button("Resetar paleta para padrão", key="reset_palette"):
            st.session_state.ceres_colors = CERES_COLORS.copy()
            st.rerun()

        st.divider()
        st.subheader("Visualizar")

        conn = get_conn(db_path)
        try:
            cnpjs_db = cnpjs_in_db(conn)
            meta_map = cnpj_meta_map_from_db(conn)
        finally:
            conn.close()

        options = [""] + cnpjs_db
        selected_cnpj = st.selectbox(
            "Selecione um CNPJ (carregado no DB)",
            options=options,
            index=0,
            format_func=lambda x: (
                "— selecione —"
                if x == ""
                else f"{meta_map.get(x, {}).get('display_name', f'CNPJ {x}')} ({x})"
            ),
            key="selected_cnpj_view",
        )

        return (
            selected_cnpj,
            data_dir,
            int(start_year),
            db_path,
            int(update_last_n),
            preload_clicked,
            update_clicked,
            bench_db_path,
            compare_cdi,
            compare_ibov,
            int(table_width_px),
            int(height_perf_table_px),
            int(height_window_table_px),
            int(height_12m_table_px),
        )


    # ------------------------------ Main --------------------------------


    def main() -> None:
        st.set_page_config(
            page_title="CVM FI - DB Loader e Visualização", layout="wide"
        )
        st.title("CVM FI - Informe Diário (INF_DIARIO) — DB e Visualização")

        init_session_state()

        # CSS global para as tabelas inline
        inject_gt_runtime_css(
            font_px=int(st.session_state.get("table_font_size", 12)),
            pal=get_palette(),
        )

        with st.sidebar:
            (
                selected_cnpj,
                data_dir,
                start_year,
                db_path,
                update_last_n,
                preload_clicked,
                update_clicked,
                bench_db_path,
                compare_cdi,
                compare_ibov,
                table_width_px,
                height_perf_table_px,
                height_window_table_px,
                height_12m_table_px,
            ) = render_cnpj_manager_sidebar(
                data_dir="./data/inf_diario",
                start_year=START_YEAR,
                db_path=DEFAULT_DB_PATH,
            )

        conn = get_conn(db_path)

        if preload_clicked:
            months_all = tuple(yyyymm_list(start_year))
            cnpj_map = {
                cnpj: {
                    "display_name": st.session_state.cnpj_display_names.get(
                        cnpj, f"CNPJ {cnpj}"
                    ),
                    "ccy": st.session_state.cnpj_list.get(cnpj, "BRL"),
                }
                for cnpj in st.session_state.cnpj_list
            }
            st.info(
                f"Pré-carregando {len(cnpj_map)} CNPJ(s) em DB para "
                f"{len(months_all)} mês(es)."
            )
            with st.spinner("Ingestão em andamento..."):
                inserted = preload_cnpjs_to_db(
                    cnpj_map=cnpj_map, months=months_all, data_dir=data_dir, conn=conn
                )
            st.success(f"Pré-carregamento concluído. Linhas novas: {inserted}")

        if update_clicked:
            months_update = tuple(last_n_months(update_last_n))
            st.info(
                f"Atualizando DB para CNPJs existentes — últimos {update_last_n} meses."
            )
            with st.spinner("Atualizando DB..."):
                inserted = update_db_for_existing_cnpjs(
                    months=months_update, data_dir=data_dir, conn=conn
                )
            st.success(f"Atualização concluída. Linhas novas: {inserted}")

        conn.close()

        if not selected_cnpj:
            st.info(
                "👈 Para visualizar: garanta que o CNPJ está carregado no DB "
                "(pré-carregado/atualizado), depois selecione um CNPJ."
            )
            return

        mtime_key = db_mtime(db_path)
        with st.spinner("Carregando do DB..."):
            df_daily = load_history_from_db_cached(
                db_path=db_path, cnpj_digits=selected_cnpj, cache_buster=mtime_key
            )

        if df_daily.empty:
            st.warning(
                "Nenhum dado no DB para este CNPJ. "
                "Pré-carregue/Atualize antes de visualizar."
            )
            return

        # Meta
        conn = get_conn(db_path)
        try:
            meta_map = cnpj_meta_map_from_db(conn)
        finally:
            conn.close()
        display_name = meta_map.get(selected_cnpj, {}).get(
            "display_name",
            st.session_state.cnpj_display_names.get(selected_cnpj, selected_cnpj),
        )
        ccy = meta_map.get(selected_cnpj, {}).get(
            "ccy", st.session_state.cnpj_list.get(selected_cnpj, "BRL")
        )
        fund_name = display_name
        st.header(f"Análise: {display_name}")

        # ---------- Comparativos ----------
        daily_map: Dict[str, pd.Series] = {}
        fund_daily = daily_series_from_nav(df_daily)
        if fund_daily.empty:
            st.warning("Série diária do fundo vazia após pct_change().")
            return
        daily_map[fund_name] = fund_daily

        bench_mtime_key = db_mtime(bench_db_path)

        if compare_cdi:
            try:
                cdi_daily_df = load_benchmark_daily_returns(
                    bench_db_path, "cdi", bench_mtime_key
                )
                daily_map["CDI"] = (
                    cdi_daily_df.set_index("date")["daily_return"].dropna()
                )
            except Exception as e:
                st.warning(f"Falha ao carregar CDI: {e}")

        if compare_ibov:
            try:
                ibov_daily_df = load_benchmark_daily_returns(
                    bench_db_path, "ibov", bench_mtime_key
                )
                daily_map["IBOV"] = (
                    ibov_daily_df.set_index("date")["daily_return"].dropna()
                )
            except Exception as e:
                st.warning(f"Falha ao carregar IBOV: {e}")

        if len(daily_map) >= 1:
            starts = [s.index.min() for s in daily_map.values()]
            ends = [s.index.max() for s in daily_map.values()]
            start_common = max(starts)
            end_common = min(ends)
            daily_map = {
                name: s[(s.index >= start_common) & (s.index <= end_common)]
                for name, s in daily_map.items()
            }

        ret_m_map: Dict[str, pd.Series] = {
            name: series_daily_to_monthly(s) for name, s in daily_map.items()
        }
        cumret_map: Dict[str, pd.Series] = {
            name: series_daily_to_cumret(s) for name, s in daily_map.items()
        }

        # ------------- Derivados ----------
        row_formats: Dict[str, str] = {}
        row_color_modes: Dict[str, str] = {}

        fund_m = ret_m_map[fund_name]

        if "CDI" in ret_m_map:
            cdi_m = ret_m_map["CDI"]
            idx = fund_m.index.intersection(cdi_m.index)
            ratio_m = (fund_m.reindex(idx) / cdi_m.reindex(idx)).replace(
                [np.inf, -np.inf], np.nan
            )
            ratio_name = f"% CDI"
            ret_m_map[ratio_name] = ratio_m
            row_formats[ratio_name] = "pct"
            row_color_modes[ratio_name] = "ratio"

        if "IBOV" in ret_m_map:
            ibov_m = ret_m_map["IBOV"]
            idx = fund_m.index.intersection(ibov_m.index)
            diff_m = fund_m.reindex(idx) - ibov_m.reindex(idx)
            diff_name = f"Diferença (p.p.)"
            ret_m_map[diff_name] = diff_m
            row_formats[diff_name] = "pp"
            row_color_modes[diff_name] = "raw"

        # ---------------------- Gráficos e Tabelas -----------------------
        tbl_font_size = int(st.session_state.get("table_font_size", 12))

        # 1) Retornos mensais — Barras (Plotly)
        st.subheader("Retorno mensal — barras (comparativo)")
        base_series_for_bars = {
            k: v
            for k, v in ret_m_map.items()
            if (k == fund_name) or (k in ("CDI", "IBOV"))
        }
        names_for_bars = list(base_series_for_bars.keys())
        bar_color_map = name_color_map(names_for_bars)
        default_bar_height = 380
        default_bar_width = 1100
        c1, c2 = st.columns(2)
        with c1:
            bar_height = st.number_input(
                "Altura (px) — Retornos mensais (gráfico)",
                min_value=200,
                max_value=2000,
                value=default_bar_height,
                step=20,
                key="height_ml",
            )
        with c2:
            bar_width = st.number_input(
                "Largura (px) — Retornos mensais (gráfico)",
                min_value=500,
                max_value=3000,
                value=default_bar_width,
                step=50,
                key="width_ml",
            )
        fig_ml = make_monthly_plotly_chart_multi(
            base_series_for_bars,
            "Retornos mensais",
            color_map=bar_color_map,
            height=bar_height,
            width=bar_width,
        )
        st.plotly_chart(fig_ml, use_container_width=False)

        # 2) Tabela de performance (comparativo)
        st.subheader("Tabela de performance (comparativo)")
        last_dt = max([s.index.max() for s in daily_map.values()])

        rows = []
        for name in [fund_name] + [n for n in ("CDI", "IBOV") if n in ret_m_map]:
            r_m = ret_m_map[name]
            r_d = daily_map[name]
            rows.append(
                dict(
                    name=name,
                    ccy="BRL" if name in ("CDI", "IBOV") else ccy,
                    as_of=last_dt,
                    ytd=_ytd_from_daily_returns(r_d),
                    r3y=compound(r_m.tail(36)) if len(r_m) >= 36 else np.nan,
                    yr_returns=annual_returns_from_monthly(r_m),
                )
            )

        if "CDI" in ret_m_map:
            ytd_fund = _ytd_from_daily_returns(daily_map[fund_name])
            ytd_cdi = _ytd_from_daily_returns(daily_map["CDI"])
            ytd_ratio = (
                ytd_fund / ytd_cdi if pd.notna(ytd_fund) and ytd_cdi else np.nan
            )

            r3y_fund = compound(ret_m_map[fund_name].tail(36))
            r3y_cdi = compound(ret_m_map["CDI"].tail(36))
            r3y_ratio = (
                r3y_fund / r3y_cdi if pd.notna(r3y_fund) and r3y_cdi else np.nan
            )

            yrs_f = annual_returns_from_monthly(ret_m_map[fund_name])
            yrs_c = annual_returns_from_monthly(ret_m_map["CDI"])
            yrs_ratio = (yrs_f / yrs_c).replace([np.inf, -np.inf], np.nan).dropna()

            rows.append(
                dict(
                    name=f"% CDI",
                    ccy="",
                    as_of=last_dt,
                    ytd=ytd_ratio,
                    r3y=r3y_ratio,
                    yr_returns=yrs_ratio,
                    fmt="pct",
                    color_mode="ratio",
                )
            )

        if "IBOV" in ret_m_map:
            ytd_fund = _ytd_from_daily_returns(daily_map[fund_name])
            ytd_ibov = _ytd_from_daily_returns(daily_map["IBOV"])
            ytd_diff = (
                ytd_fund - ytd_ibov
                if pd.notna(ytd_fund) and pd.notna(ytd_ibov)
                else np.nan
            )

            r3y_fund = compound(ret_m_map[fund_name].tail(36))
            r3y_ibov = compound(ret_m_map["IBOV"].tail(36))
            r3y_diff = (
                r3y_fund - r3y_ibov
                if pd.notna(r3y_fund) and pd.notna(r3y_ibov)
                else np.nan
            )

            yrs_f = annual_returns_from_monthly(ret_m_map[fund_name])
            yrs_i = annual_returns_from_monthly(ret_m_map["IBOV"])
            yrs_diff = (yrs_f - yrs_i).dropna()

            rows.append(
                dict(
                    name=f"Diferença (p.p.)",
                    ccy="",
                    as_of=last_dt,
                    ytd=ytd_diff,
                    r3y=r3y_diff,
                    yr_returns=yrs_diff,
                    fmt="pp",
                    color_mode="raw",
                )
            )

        gt_tbl, _, _ = build_performance_comparison_table_gt(
            rows,
            height=height_perf_table_px,
            width=table_width_px,
            table_font_size=tbl_font_size,
            header_colored=True,
        )
        render_gt_inline_with_runtime_export(
            gt_obj=gt_tbl,
            filename_base=(
                f"tabela_performance_comparativo_{sanitize_cnpj(selected_cnpj)}"
            ),
            width_px=int(table_width_px),
            n_rows=len(rows),
            font_px=int(tbl_font_size),
            key_suffix="perf_cmp",
            n_header_rows=1,
            title_rows=0,
        )

        # 2.5) Janelas 1m/6m/12m e Vol 12m
        st.subheader("Janelas: 1m, 6m, 12m e Vol 12m (horizontal)")
        base_metric_names = [fund_name] + [n for n in ("CDI", "IBOV") if n in daily_map]
        series_m_sel = {n: ret_m_map[n] for n in base_metric_names}
        series_d_sel = {n: daily_map[n] for n in base_metric_names}

        gt_tbl_win, _, _ = build_window_perf_table_gt(
            series_m_map=series_m_sel,
            series_d_map=series_d_sel,
            order=base_metric_names,
            height=height_window_table_px,
            width=table_width_px,
            table_font_size=tbl_font_size,
            header_colored=True,
        )
        render_gt_inline_with_runtime_export(
            gt_obj=gt_tbl_win,
            filename_base=f"tabela_janelas_{sanitize_cnpj(selected_cnpj)}",
            width_px=int(table_width_px),
            n_rows=len(base_metric_names),
            font_px=int(tbl_font_size),
            key_suffix="win_tbl",
            n_header_rows=1,
            title_rows=0,
        )

        # 3) Últimos 12 meses — Retornos mensais (comparativo)
        st.subheader("Últimos 12 meses — Retornos mensais (comparativo)")
        gt_tbl_12m, _, _ = build_last_12m_returns_table_gt(
            ret_m_map,
            row_formats=row_formats,
            row_color_modes=row_color_modes,
            height=height_12m_table_px,
            width=table_width_px,
            table_font_size=tbl_font_size,
            header_colored=True,
        )
        n_series = len(ret_m_map)
        render_gt_inline_with_runtime_export(
            gt_obj=gt_tbl_12m,
            filename_base=f"tabela_ultimos_12m_{sanitize_cnpj(selected_cnpj)}",
            width_px=int(table_width_px),
            n_rows=n_series,
            font_px=int(tbl_font_size),
            key_suffix="last12",
            n_header_rows=1,
            title_rows=0,
        )

        # 4) Série (linha) — Retorno acumulado (Plotly)
        st.subheader("Série (linha) — Retorno acumulado (composto)")
        base_line_series = {
            k: v
            for k, v in cumret_map.items()
            if (k == fund_name) or (k in ("CDI", "IBOV"))
        }
        names_for_lines = list(base_line_series.keys())
        line_color_map = name_color_map(names_for_lines)
        default_line_height = 620
        default_line_width = 800
        l1, l2 = st.columns(2)
        with l1:
            line_height = st.number_input(
                "Altura (px) — Linha acumulada (gráfico)",
                min_value=200,
                max_value=2000,
                value=default_line_height,
                step=20,
                key="height_line",
            )
        with l2:
            line_width = st.number_input(
                "Largura (px) — Linha acumulada (gráfico)",
                min_value=500,
                max_value=3000,
                value=default_line_width,
                step=50,
                key="width_line",
            )

        nav_line = make_nav_line_chart_multi(
            base_line_series,
            "",
            color_map=line_color_map,
            height=line_height,
            width=line_width,
        )
        st.plotly_chart(nav_line, use_container_width=False)

        with st.expander("Detalhes e download"):
            st.write(
                "- Fonte: CVM Dados Abertos - FI/DOC/INF_DIARIO\n"
                "- Armazenamento: SQLite (data_fundos.db)\n"
                "- Retorno mensal: composto por mês-calendário a partir de diários\n"
                "- 12m/3y: composto de retornos mensais\n"
                "- YTD: composto de retornos diários desde 1º dia útil do ano\n"
                "- Linha: retorno acumulado composto (base 0 = início comum)\n"
                "- Benchmarks: CDI/IBOV de data_cdi_ibov.db ('cdi'/'ibov')\n"
                "- Tabelas: %CDI e Diferença p.p. quando comparadores ligados\n"
                "- Tema: paleta CERES; fundo transparente; sem barras verticais\n"
                "- Exports: SVG/PNG em alta definição (escala 3x), com altura exata"
            )

            last_dt = df_daily["DT_COMPTC"].max()
            first_dt = df_daily["DT_COMPTC"].min()
            st.write(
                f"Série diária do fundo de {first_dt.date()} até {last_dt.date()} "
                f"({len(df_daily)} observações)."
            )
            st.download_button(
                "Baixar série diária (CSV)",
                df_daily.to_csv(index=False).encode("utf-8"),
                file_name=(f"inf_diario_{sanitize_cnpj(selected_cnpj)}_diario.csv"),
                mime="text/csv",
            )

            fund_monthly_df = (
                series_daily_to_monthly(fund_daily)
                .rename("ret_m")
                .reset_index()
                .rename(columns={"index": "month_end"})
            )
            st.download_button(
                "Baixar retornos mensais (CSV)",
                fund_monthly_df.to_csv(index=False).encode("utf-8"),
                file_name=(f"inf_diario_{sanitize_cnpj(selected_cnpj)}_mensal.csv"),
                mime="text/csv",
            )


    def _ytd_from_daily_returns(daily_ret: pd.Series) -> float:
        if daily_ret is None or daily_ret.empty:
            return float("nan")
        daily_ret = daily_ret.dropna()
        if daily_ret.empty:
            return float("nan")
        last_dt = daily_ret.index.max()
        y0 = pd.Timestamp(year=last_dt.year, month=1, day=1, tz=last_dt.tz)
        ytd_slice = daily_ret[daily_ret.index >= y0]
        if ytd_slice.empty:
            return float("nan")
        return float(np.prod(1.0 + ytd_slice.values) - 1.0)


    def render() -> None:
        main()


    if __name__ == "__main__":
        main()