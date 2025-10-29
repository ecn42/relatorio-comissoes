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
                                    textinfo='text',
                                    hole=0.3,  # Donut style
                                    marker=dict(colors=colors_to_use)
                                )])
                                
                                # Transparent background with minimal margins
                                fig.update_layout(
                                    title=" ",
                                    paper_bgcolor='rgba(0,0,0,0)',
                                    plot_bgcolor='rgba(0,0,0,0)',
                                    showlegend=False,
                                    margin=dict(l=0, r=0, t=30, b=0),
                                    height=500
                                )
                                
                                fig.update_traces(
                                    textfont=dict(size=12),
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