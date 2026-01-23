# app.py
# Streamlit app to:
# 1) Sync newest CVM CDA FI monthly ZIPs (with a "force re-download" option).
# 2) For input CNPJs, find the latest month (among synced months) where the
#    CNPJ is NOT present in cda_fi_CONFID_YYYYMM.csv.
# 3) For a selected CNPJ, open that month’s ZIP, join cda_fi_BLC_1..8 files,
#    filter by the CNPJ, show the rows, and store/update them in SQLite.
# 4) Compute PCT_CARTEIRA (0–1 fraction) for each row from VL_MERC_POS_FINAL
#    and include it in the CSV stored in the DB.
# 5) Bulk mode: check and save many CNPJs at once to the DB.
# Updated: Store BLC data as rows in a proper SQLite table (blc_data) instead of
# CSV BLOB for better querying and reliability. Added DENOM_SOCIAL to
# fundos_meta. Added bulk processing helpers and UI. Added index on (cnpj, yyyymm).
# New: Button to fill EMISSOR in carteira_fundos.db for Debêntures using a
# mapping from df_debentures.db ("Codigo do Ativo" -> "Empresa").
# New (Dec/2025):
# - When showing "latest absent" month per CNPJ, also check the DB to report if
#   it's up-to-date, needs update, or is missing.
# - Bulk mode now logs per-fund feedback with progress, and skips heavy parsing
#   if DB is already up-to-date for the target month (configurable).

from __future__ import annotations

import csv
import io
import json
import os
import re
import sqlite3
import sys
import time
import unicodedata
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go

### Simple Authentication
if not st.session_state.get("authenticated", False):
    st.warning("Please enter the password on the Home page first.")
    st.write("Status: Não Autenticado")
    st.stop()

# prevent the rest of the page from running
st.write("Autenticado")

# ------------------------ Config and regexes ------------------------
tab1, tab2, tab3 = st.tabs(
    ["Carteiras dos Fundos", "Baixar CDI", "Rentabilidade dos Fundos"]
)

with tab1:
    BASE_URL = "https://dados.cvm.gov.br/dados/FI/DOC/CDA/DADOS/"
    USER_AGENT = "Mozilla/5.0 (CVM-CDA-FI-Sync/1.5)"

    # Match names like cda_fi_202509.zip and capture 202509
    ZIP_RE = re.compile(r"cda_fi_(\d{6})\.zip", re.IGNORECASE)

    # Inside ZIPs
    CONFID_RE = re.compile(r"(^|/)(cda_fi_CONFID_\d{6}\.csv)$", re.IGNORECASE)

    def blc_pattern(yyyymm: str) -> re.Pattern:
        # cda_fi_BLC_[1..8]_YYYYMM.csv (possibly in a subfolder)
        return re.compile(
            rf"(^|/)cda_fi_BLC_[1-8]_{yyyymm}\.csv$", re.IGNORECASE
        )

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
            with requests.get(
                url, stream=True, headers=headers, timeout=120
            ) as r:
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

    def ym_cmp(a: Optional[str], b: Optional[str]) -> int:
        """
        Compare YYYYMM strings; treat None as -infinity.
        Returns -1, 0, 1 for a<b, a==b, a>b respectively.
        """
        if a is None and b is None:
            return 0
        if a is None:
            return -1
        if b is None:
            return 1
        ai, bi = int(a), int(b)
        return -1 if ai < bi else (1 if ai > bi else 0)

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
                row = {
                    header[i]: (parts[i] if i < len(parts) else "")
                    for i in range(n)
                }
                row["_ARQUIVO"] = os.path.basename(filename)
                out_rows.append(row)

        if not out_rows:
            return pd.DataFrame()
        return pd.DataFrame(out_rows)

    def read_blc_joined_for_cnpj_from_zip(
        zip_path: str, cnpj: str
    ) -> pd.DataFrame:
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
            col = next(
                (c for c, n in cols_norm.items() if "VL_MERC_POS_FINAL" in n), None
            )
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
        # Indexes for faster queries
        cur.execute("CREATE INDEX IF NOT EXISTS idx_blc_cnpj ON blc_data(cnpj)")
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_blc_cnpj_ym ON blc_data(cnpj, yyyymm)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_blc_cd_ativo ON blc_data(CD_ATIVO)"
        )
        conn.commit()

    def get_meta_for_cnpj(
        conn: sqlite3.Connection, cnpj: str
    ) -> Optional[Dict[str, str]]:
        """
        Fetch fundos_meta row for a CNPJ. Returns dict or None.
        Keys: last_month_yyyymm, last_updated_utc, denom_social, n_rows
        """
        cur = conn.cursor()
        row = cur.execute(
            "SELECT last_month_yyyymm, last_updated_utc, denom_social, n_rows "
            "FROM fundos_meta WHERE cnpj=?",
            (cnpj,),
        ).fetchone()
        if not row:
            return None
        return {
            "last_month_yyyymm": row[0],
            "last_updated_utc": row[1],
            "denom_social": row[2],
            "n_rows": str(row[3]),
        }

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
            "P_FUNDO_CLASSE",
            "CNPJ_FUNDO_CLASSE",
            "DENOM_SOCIAL",
            "DT_COMPTC",
            "TP_APLIC",
            "TP_ATIVO",
            "EMISSOR_LIGADO",
            "TP_NEGOC",
            "QT_VENDA_NEGOC",
            "VL_VENDA_NEGOC",
            "QT_AQUIS_NEGOC",
            "VL_AQUIS_NEGOC",
            "QT_POS_FINAL",
            "VL_MERC_POS_FINAL",
            "VL_CUSTO_POS_FINAL",
            "DT_CONFID_APLIC",
            "CNPJ_FUNDO_CLASSE_COTA",
            "ID_SUBCLASSE",
            "NM_FUNDO_CLASSE_SUBCLASSE_COTA",
            "_ARQUIVO",
            "CD_ATIVO",
            "DS_ATIVO",
            "CD_ISIN",
            "DT_INI_VIGENCIA",
            "DT_FIM_VIGENCIA",
            "PF_PJ_EMISSOR",
            "CPF_CNPJ_EMISSOR",
            "EMISSOR",
            "PCT_CARTEIRA",
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
                "INSERT INTO fundos_meta (cnpj, denom_social, last_month_yyyymm, "
                "last_updated_utc, n_rows, columns_json) VALUES (?, ?, ?, ?, ?, ?)",
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
                    "UPDATE fundos_meta SET denom_social=?, last_month_yyyymm=?, "
                    "last_updated_utc=?, n_rows=?, columns_json=? WHERE cnpj=?",
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
                    df_insert["PCT_CARTEIRA"] = df_insert["PCT_CARTEIRA"].astype(
                        str
                    )
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
                st.warning(
                    f"to_sql failed: {insert_err}. Falling back to manual insert."
                )
                # Fallback: manual insert
                cur.execute("DELETE FROM blc_data WHERE cnpj=?", (cnpj,))
                for _, row in df.iterrows():
                    values = [
                        cnpj,
                        ym,
                        *[row.get(col, "") for col in known_cols],
                    ]
                    placeholders = ", ".join(["?" for _ in values])
                    columns_str = "(cnpj, yyyymm, " + ", ".join(known_cols) + ")"
                    sql = f"INSERT INTO blc_data {columns_str} VALUES ({placeholders})"
                    cur.execute(sql, values)

        conn.commit()
        conn.close()
        return action, prev_ym

    # ------------------------ Saved CNPJs management ------------------------

    def get_saved_cnpjs(db_path: str) -> List[str]:
        """Fetch all CNPJs from the saved_cnpjs table in cnpj-fundos.db."""
        if not os.path.isfile(db_path):
            return []
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        try:
            # Ensure the table exists even if file was empty/new
            cur.execute(
                "CREATE TABLE IF NOT EXISTS saved_cnpjs "
                "(cnpj TEXT PRIMARY KEY, ccy TEXT DEFAULT 'BRL', added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
            )
            rows = cur.execute("SELECT cnpj FROM saved_cnpjs ORDER BY added_at ASC").fetchall()
            return [r[0] for r in rows]
        except Exception:
            return []
        finally:
            conn.close()

    def save_cnpjs(db_path: str, cnpjs: List[str]):
        """Save a list of CNPJs to the saved_cnpjs table, avoiding duplicates."""
        if not cnpjs:
            return
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        try:
            cur.execute(
                "CREATE TABLE IF NOT EXISTS saved_cnpjs "
                "(cnpj TEXT PRIMARY KEY, ccy TEXT DEFAULT 'BRL', added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
            )
            now_iso = time.strftime("%Y-%m-%d %H:%M:%S")
            data = [(c, "BRL", now_iso) for c in cnpjs]
            cur.executemany(
                "INSERT OR IGNORE INTO saved_cnpjs (cnpj, ccy, added_at) VALUES (?, ?, ?)",
                data
            )
            conn.commit()
        except Exception as e:
            st.error(f"Error saving CNPJs to DB: {e}")
        finally:
            conn.close()

    # ------------------------ Bulk helpers ------------------------

    def build_cnpjs_by_month_from_local(
        dest_dir: str, k: int
    ) -> Tuple[List[str], Dict[str, Set[str]], Dict[str, str]]:
        """
        Build month -> set(CNPJs) from local ZIPs (newest k) and map yyyymm -> zip path.
        Returns (months_desc, cnpjs_by_month, zip_paths_by_ym).
        """
        local = [
            n
            for n in os.listdir(dest_dir)
            if ZIP_RE.search(n) and os.path.isfile(os.path.join(dest_dir, n))
        ]
        if not local:
            raise FileNotFoundError(
                "No local ZIPs found. Click one of the 'Sync' buttons first."
            )
        local_sorted = sorted(local, key=lambda n: int(ZIP_RE.search(n).group(1)))[
            -int(k) :
        ]
        local_paths = [os.path.join(dest_dir, n) for n in local_sorted]

        cnpjs_by_month: Dict[str, Set[str]] = {}
        zip_paths_by_ym: Dict[str, str] = {}
        for zp in local_paths:
            ym, s = read_confid_cnpjs_from_zip(zp)
            cnpjs_by_month[ym] = s
            zip_paths_by_ym[ym] = zp

        months_desc = sorted(list(cnpjs_by_month.keys()), key=int, reverse=True)
        return months_desc, cnpjs_by_month, zip_paths_by_ym

    def process_one_cnpj_bulk(
        cnpj: str,
        months_desc: List[str],
        cnpjs_by_month: Dict[str, Set[str]],
        zip_paths_by_ym: Dict[str, str],
        db_path: str,
        *,
        only_absent: bool = True,
        fallback_ym: Optional[str] = None,
    ) -> Dict[str, object]:
        """
        For a CNPJ, decide which month to process (latest absent or fallback),
        read/join BLC, compute PCT_CARTEIRA, and upsert into DB.
        Returns a result dict with status.
        Early DB check prevents heavy parsing when already up-to-date.
        """
        result: Dict[str, object] = {
            "CNPJ": cnpj,
            "YM": None,
            "Action": None,
            "PrevYM": None,
            "Rows": 0,
            "Error": None,
        }
        try:
            ym = latest_absent_month_for_cnpj(cnpj, months_desc, cnpjs_by_month)
            if ym is None:
                if only_absent:
                    result["Action"] = "skipped_no_absent"
                    return result
                if fallback_ym:
                    ym = fallback_ym

            if ym is None:
                result["Action"] = "skipped_no_month"
                return result

            # Early DB check to skip heavy parsing if up-to-date
            conn = sqlite3.connect(db_path)
            ensure_db_schema(conn)
            meta = get_meta_for_cnpj(conn, cnpj)
            conn.close()
            prev_ym = meta["last_month_yyyymm"] if meta else None
            if prev_ym is not None and int(prev_ym) >= int(ym):
                result["Action"] = "up_to_date"
                result["PrevYM"] = prev_ym
                result["YM"] = ym
                return result

            if ym not in zip_paths_by_ym:
                result["Error"] = f"No local ZIP for {ym}. Increase K or re-sync."
                result["Action"] = "error"
                return result

            zip_path = zip_paths_by_ym[ym]
            df_blc = read_blc_joined_for_cnpj_from_zip(zip_path, cnpj)
            df_blc = add_pct_carteira_by_vl_merc(df_blc)

            if df_blc.empty:
                result["Action"] = "no_rows"
                result["YM"] = ym
                return result

            action, prev = upsert_blc_in_db(db_path, cnpj, ym, df_blc)
            result["Action"] = action
            result["PrevYM"] = prev
            result["Rows"] = int(len(df_blc))
            result["YM"] = ym
            return result
        except Exception as e:
            result["Error"] = str(e)
            result["Action"] = "error"
            return result

    def bulk_process_cnpjs(
        cnpjs: List[str],
        db_path: str,
        dest_dir: str,
        k: int,
        *,
        only_absent: bool = True,
        fallback_ym: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Bulk: build months/CNPJ presence once, then process each CNPJ.
        Returns a DataFrame with results.
        """
        months_desc, cnpjs_by_month, zip_paths_by_ym = build_cnpjs_by_month_from_local(
            dest_dir, k
        )

        results: List[Dict[str, object]] = []
        for cnpj in cnpjs:
            res = process_one_cnpj_bulk(
                cnpj,
                months_desc,
                cnpjs_by_month,
                zip_paths_by_ym,
                db_path,
                only_absent=only_absent,
                fallback_ym=fallback_ym,
            )
            results.append(res)
        return pd.DataFrame(results)

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

    db_default = os.path.join(os.getcwd(), "databases", "carteira_fundos.db")
    db_path = st.text_input("SQLite database path", value=db_default, key="db_path_input")

    cnpj_db_default = os.path.join(os.getcwd(), "databases", "cnpj-fundos.db")
    cnpj_db_path = st.text_input("CNPJ database path", value=cnpj_db_default, key="cnpj_db_path_input")

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

    # Load saved CNPJs for pre-filling
    saved_list = get_saved_cnpjs(cnpj_db_path)
    default_cnpj_text = "\n".join(saved_list)

    cnpj_text = st.text_area(
        "Paste one or more CNPJs (punctuation ok: 00.000.000/0000-00, etc.)",
        value=default_cnpj_text,
        height=120,
    )

    check_btn = st.button(
        "Find latest month each CNPJ is NOT present (among synced months)"
    )

    if "absent_map" not in st.session_state:
        st.session_state.absent_map = {}
    if "checked_cnpjs" not in st.session_state:
        st.session_state.checked_cnpjs = []

    # Helper for DB status label
    def db_status_for(absent_ym: Optional[str], db_ym: Optional[str]) -> str:
        if absent_ym is None:
            return "No absent month (local)"
        if db_ym is None:
            return "Missing in DB (needs insert)"
        cmpv = ym_cmp(db_ym, absent_ym)
        if cmpv == 0:
            return "Up-to-date (matches DB)"
        if cmpv < 0:
            return "Needs update (DB older)"
        return "DB newer than target"

    if check_btn:
        cnpjs = parse_cnpj_input(cnpj_text)
        st.session_state.checked_cnpjs = cnpjs
        if not cnpjs:
            st.warning("Please enter at least one valid 14-digit CNPJ.")
        else:
            # Save new ones to the persistent DB
            save_cnpjs(cnpj_db_path, cnpjs)
            try:
                # Use newest K local files (sorted by YYYYMM)
                local = [
                    n
                    for n in os.listdir(dest_dir)
                    if ZIP_RE.search(n)
                    and os.path.isfile(os.path.join(dest_dir, n))
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

                    # DB lookup
                    conn = sqlite3.connect(db_path)
                    ensure_db_schema(conn)

                    absent_map = {}
                    rows = []
                    for cnpj in cnpjs:
                        ym = latest_absent_month_for_cnpj(
                            cnpj, months_desc, cnpjs_by_month
                        )
                        absent_map[cnpj] = ym
                        meta = get_meta_for_cnpj(conn, cnpj)
                        db_ym = meta["last_month_yyyymm"] if meta else None
                        status = db_status_for(ym, db_ym)
                        rows.append(
                            {
                                "CNPJ": cnpj,
                                "Latest absent (local)": ym_label(ym),
                                "DB month (fundos_meta)": ym_label(db_ym)
                                if db_ym
                                else "-",
                                "DB status": status,
                            }
                        )

                    conn.close()
                    st.session_state.absent_map = absent_map
                    st.dataframe(pd.DataFrame(rows), width='stretch')
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
                # Pre-check DB status for user feedback
                conn = sqlite3.connect(db_path)
                ensure_db_schema(conn)
                meta = get_meta_for_cnpj(conn, selected_cnpj)
                conn.close()
                db_ym = meta["last_month_yyyymm"] if meta else None
                if db_ym is not None and int(db_ym) >= int(ym):
                    st.info(
                        f"DB already has {ym_label(db_ym)} for {selected_cnpj} "
                        f"(target absent: {ym_label(ym)}). Keeping DB snapshot."
                    )

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
                                    df_blc, width='stretch', height=420
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

    # ------------------------ Bulk: process and save many CNPJs ------------------------

    st.subheader("Processar em lote e salvar no DB")

    st.caption(
        "Processa todos os CNPJs acima de uma vez. Para cada CNPJ, seleciona o "
        "mês mais recente em que está ausente no CONFID (entre os meses locais). "
        "Caso não haja mês ausente, você pode optar por pular ou usar um YYYYMM "
        "específico (fallback). Agora mostra feedback por fundo e evita ler ZIP "
        "quando o DB já está atualizado."
    )

    col_bulk1, col_bulk2, col_bulk3 = st.columns(3)
    with col_bulk1:
        only_absent = st.checkbox(
            "Processar somente quando houver mês ausente", value=True
        )
    with col_bulk2:
        fallback_ym = st.text_input(
            "Fallback YYYYMM (opcional)", value="", placeholder="202501"
        ).strip()
    with col_bulk3:
        skip_uptodate = st.checkbox(
            "Pular CNPJs já atualizados no DB", value=True
        )

    if fallback_ym and not re.fullmatch(r"\d{6}", fallback_ym):
        st.warning("Fallback YYYYMM deve ser 6 dígitos. Ex.: 202501")
        fallback_ym = ""

    bulk_btn = st.button("Processar e salvar TODOS os CNPJs acima")

    if bulk_btn:
        cnpjs_bulk = parse_cnpj_input(cnpj_text)
        if not cnpjs_bulk:
            st.warning("Insira ao menos um CNPJ válido.")
        else:
            with st.spinner("Processando em lote..."):
                try:
                    # Build month maps once
                    (
                        months_desc,
                        cnpjs_by_month,
                        zip_paths_by_ym,
                    ) = build_cnpjs_by_month_from_local(dest_dir, int(k))

                    # Prepare logging/progress
                    progress = st.progress(0.0)
                    log_container = st.container()

                    results: List[Dict[str, object]] = []
                    n = len(cnpjs_bulk)
                    for i, cnpj in enumerate(cnpjs_bulk, start=1):
                        # Determine target month
                        ym = latest_absent_month_for_cnpj(
                            cnpj, months_desc, cnpjs_by_month
                        )
                        if ym is None and not only_absent and fallback_ym:
                            ym = fallback_ym

                        # DB meta
                        conn = sqlite3.connect(db_path)
                        ensure_db_schema(conn)
                        meta = get_meta_for_cnpj(conn, cnpj)
                        conn.close()
                        db_ym = meta["last_month_yyyymm"] if meta else None

                        # Skip conditions
                        if ym is None:
                            res = {
                                "CNPJ": cnpj,
                                "YM": None,
                                "Action": "skipped_no_absent"
                                if only_absent
                                else "skipped_no_month",
                                "PrevYM": db_ym,
                                "Rows": 0,
                                "Error": None,
                            }
                            log_container.write(
                                f"• {cnpj}: sem mês ausente local "
                                f"(DB: {ym_label(db_ym)})."
                            )
                            results.append(res)
                            progress.progress(i / n)
                            continue

                        if skip_uptodate and db_ym is not None and int(db_ym) >= int(
                            ym
                        ):
                            res = {
                                "CNPJ": cnpj,
                                "YM": ym,
                                "Action": "up_to_date",
                                "PrevYM": db_ym,
                                "Rows": 0,
                                "Error": None,
                            }
                            log_container.write(
                                f"• {cnpj}: já atualizado no DB "
                                f"(DB {ym_label(db_ym)} >= alvo {ym_label(ym)}). "
                                "Pulando."
                            )
                            results.append(res)
                            progress.progress(i / n)
                            continue

                        # Process
                        if ym not in zip_paths_by_ym:
                            res = {
                                "CNPJ": cnpj,
                                "YM": ym,
                                "Action": "error",
                                "PrevYM": db_ym,
                                "Rows": 0,
                                "Error": f"ZIP ausente para {ym}.",
                            }
                            log_container.write(
                                f"• {cnpj}: erro - ZIP ausente para {ym_label(ym)}."
                            )
                            results.append(res)
                            progress.progress(i / n)
                            continue

                        try:
                            zip_path = zip_paths_by_ym[ym]
                            df_blc = read_blc_joined_for_cnpj_from_zip(
                                zip_path, cnpj
                            )
                            df_blc = add_pct_carteira_by_vl_merc(df_blc)

                            if df_blc.empty:
                                res = {
                                    "CNPJ": cnpj,
                                    "YM": ym,
                                    "Action": "no_rows",
                                    "PrevYM": db_ym,
                                    "Rows": 0,
                                    "Error": None,
                                }
                                log_container.write(
                                    f"• {cnpj}: sem linhas em BLC para "
                                    f"{ym_label(ym)}."
                                )
                            else:
                                action, prev = upsert_blc_in_db(
                                    db_path, cnpj, ym, df_blc
                                )
                                res = {
                                    "CNPJ": cnpj,
                                    "YM": ym,
                                    "Action": action,
                                    "PrevYM": prev,
                                    "Rows": int(len(df_blc)),
                                    "Error": None,
                                }
                                if action == "inserted":
                                    log_container.write(
                                        f"• {cnpj}: inserido {ym_label(ym)} "
                                        f"({len(df_blc)} linhas)."
                                    )
                                elif action == "updated":
                                    log_container.write(
                                        f"• {cnpj}: atualizado de "
                                        f"{ym_label(prev)} para {ym_label(ym)} "
                                        f"({len(df_blc)} linhas)."
                                    )
                                else:
                                    log_container.write(
                                        f"• {cnpj}: mantido (DB "
                                        f"{ym_label(prev)} >= {ym_label(ym)})."
                                    )
                            results.append(res)
                        except Exception as e:
                            res = {
                                "CNPJ": cnpj,
                                "YM": ym,
                                "Action": "error",
                                "PrevYM": db_ym,
                                "Rows": 0,
                                "Error": str(e),
                            }
                            log_container.write(
                                f"• {cnpj}: erro ao processar {ym_label(ym)} - {e}"
                            )
                            results.append(res)

                        progress.progress(i / n)

                    st.success("Lote concluído.")
                    df_res = pd.DataFrame(results)
                    if not df_res.empty:
                        st.dataframe(df_res, width='stretch', height=380)
                        st.download_button(
                            "Baixar relatório (CSV)",
                            data=df_res.to_csv(index=False).encode("utf-8"),
                            file_name="resultado_lote_cnpjs.csv",
                            mime="text/csv",
                        )
                        # Simple summary
                        summary = (
                            df_res["Action"].value_counts().to_dict()
                            if "Action" in df_res.columns
                            else {}
                        )
                        st.write("Resumo por ação:", summary)
                    else:
                        st.info("Nenhum resultado.")
                except Exception as e:
                    st.error(f"Falha no processamento em lote: {e}")

    # ------------------------ Debêntures -> EMISSOR mapping ------------------------

    def strip_accents(s: str) -> str:
        if s is None:
            return ""
        return "".join(
            ch
            for ch in unicodedata.normalize("NFD", str(s))
            if unicodedata.category(ch) != "Mn"
        )

    def norm_label(s: str) -> str:
        s = strip_accents(s)
        s = re.sub(r"[\s_]+", " ", s).strip().upper()
        return s

    def quote_ident(ident: str) -> str:
        return '"' + str(ident).replace('"', '""') + '"'

    def find_debentures_table_and_cols(
        conn: sqlite3.Connection,
    ) -> Tuple[str, str, str]:
        """
        Try to find a table in df_debentures.db that has columns corresponding to
        'Codigo do Ativo' and 'Empresa' (robust to accents/spacing).
        Returns (table_name, codigo_col_name, empresa_col_name).
        """
        cur = conn.cursor()
        tables = [
            r[0]
            for r in cur.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            ).fetchall()
        ]
        for t in tables:
            cols = [
                r[1]
                for r in cur.execute(
                    f"PRAGMA table_info({quote_ident(t)})"
                ).fetchall()
            ]
            norm_map = {norm_label(c): c for c in cols}
            code_col = None
            empresa_col = None

            # Prefer exact normalized keys
            if "CODIGO DO ATIVO" in norm_map:
                code_col = norm_map["CODIGO DO ATIVO"]
            if "EMPRESA" in norm_map:
                empresa_col = norm_map["EMPRESA"]

            # Fallbacks
            if not code_col:
                for k, orig in norm_map.items():
                    if k in (
                        "CODIGO ATIVO",
                        "CODIGO DO PAPEL",
                        "COD ATIVO",
                        "CODIGO",
                        "COD DO ATIVO",
                    ):
                        code_col = orig
                        break
            if not empresa_col:
                for k, orig in norm_map.items():
                    if k in ("EMISSOR", "RAZAO SOCIAL", "RAZAO_SOCIAL"):
                        empresa_col = orig
                        break

            if code_col and empresa_col:
                return t, code_col, empresa_col

        # Last resort: pick first table with >= 2 columns in order (as user stated)
        for t in tables:
            rows = cur.execute(f"PRAGMA table_info({quote_ident(t)})").fetchall()
            if len(rows) >= 2:
                return t, rows[0][1], rows[1][1]

        raise RuntimeError(
            "Não foi possível localizar uma tabela com colunas "
            "'Codigo do Ativo' e 'Empresa' em df_debentures.db."
        )

    def add_debentures_names_to_db(
        db_path: str, deb_db_path: str
    ) -> Tuple[int, int]:
        """
        Load mapping (Codigo do Ativo -> Empresa) from df_debentures.db and
        update blc_data.EMISSOR for rows where TP_APLIC indicates Debêntures.
        Returns (n_mappings_loaded, n_rows_updated).
        """
        if not os.path.isfile(deb_db_path):
            raise FileNotFoundError(f"Arquivo não encontrado: {deb_db_path}")

        # Read mapping from df_debentures.db
        deb_conn = sqlite3.connect(deb_db_path)
        try:
            table, code_col, emp_col = find_debentures_table_and_cols(deb_conn)
            q_t = quote_ident(table)
            q_code = quote_ident(code_col)
            q_emp = quote_ident(emp_col)
            rows = deb_conn.cursor().execute(
                f"SELECT {q_code}, {q_emp} FROM {q_t}"
            ).fetchall()
        finally:
            deb_conn.close()

        mapping: Dict[str, str] = {}
        for code, emp in rows:
            c = (str(code) if code is not None else "").strip()
            e = (str(emp) if emp is not None else "").strip()
            if c and e:
                mapping[c] = e
        n_map = len(mapping)
        if n_map == 0:
            return 0, 0

        # Update carteira_fundos.db (blc_data)
        conn = sqlite3.connect(db_path)
        ensure_db_schema(conn)
        cur = conn.cursor()

        # Temp mapping table for efficient join-like update
        cur.execute(
            "CREATE TEMP TABLE IF NOT EXISTS deb_map ("
            "code TEXT PRIMARY KEY, empresa TEXT)"
        )
        cur.execute("DELETE FROM deb_map")
        items = list(mapping.items())
        chunk = 2000
        for i in range(0, len(items), chunk):
            cur.executemany(
                "INSERT INTO deb_map (code, empresa) VALUES (?, ?)",
                items[i : i + chunk],
            )
        conn.commit()

        # Perform update for Debêntures rows where EMISSOR is empty
        cur.execute("BEGIN")
        cur.execute(
            """
            UPDATE blc_data
            SET EMISSOR = (
                SELECT empresa FROM deb_map m WHERE m.code = blc_data.CD_ATIVO
            )
            WHERE (TP_APLIC IN ('Debêntures','Debentures','DEBÊNTURES','DEBENTURES'))
              AND CD_ATIVO IS NOT NULL AND TRIM(CD_ATIVO) <> ''
              AND (EMISSOR IS NULL OR TRIM(EMISSOR) = '')
              AND EXISTS (
                SELECT 1 FROM deb_map m WHERE m.code = blc_data.CD_ATIVO
              )
            """
        )
        updated = cur.rowcount if cur.rowcount is not None else 0
        conn.commit()
        conn.close()
        return n_map, int(updated)

    st.subheader("Preencher EMISSOR para Debêntures (df_debentures.db)")
    deb_db_default = os.path.join(os.getcwd(), "databases", "df_debentures.db")
    deb_db_path = st.text_input(
        "Caminho do df_debentures.db (possui colunas 'Codigo do Ativo' e 'Empresa')",
        value=deb_db_default,
    )
    deb_btn = st.button("Adicionar nomes de Debêntures (preencher EMISSOR)")

    if deb_btn:
        with st.spinner("Atualizando EMISSOR com nomes de Debêntures..."):
            try:
                n_map, n_upd = add_debentures_names_to_db(db_path, deb_db_path)
                if n_upd > 0:
                    st.success(
                        f"Atualizado EMISSOR em {n_upd} linha(s). "
                        f"Mapeamentos carregados: {n_map}."
                    )
                else:
                    st.info(
                        f"Nenhuma linha precisou de atualização. "
                        f"Mapeamentos carregados: {n_map}."
                    )
            except Exception as e:
                st.error(f"Falha ao atualizar nomes de Debêntures: {e}")

    st.caption(
        "Notes: Only the newest K months available locally are checked for "
        "absence. The app looks for cda_fi_CONFID_YYYYMM.csv and "
        "cda_fi_BLC_[1..8]_YYYYMM.csv inside each ZIP. Malformed lines are "
        "skipped; quotes issues are handled with fallback parsing. Database "
        "stores one snapshot per CNPJ (latest month only, as table rows). "
        "PCT_CARTEIRA is a 0–1 fraction computed from VL_MERC_POS_FINAL over the "
        "total of the rows. The 'Adicionar nomes de Debêntures' button fills "
        "EMISSOR for Debêntures using df_debentures.db mapping."
    )