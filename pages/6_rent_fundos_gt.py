# streamlit_app.py
# CVM FI (INF_DIARIO) - Ferramenta de Carga e Gerenciamento de Dados
# - Ingestão de dados da CVM para SQLite
# - Gerenciamento de lista de CNPJs (lote e unitário)
# - Atualização de histórico

import os
import re
import sqlite3
import zipfile
from typing import Dict, Iterable, List, Optional, Tuple

import base64
import math

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path


### Simple Authentication
if not st.session_state.get("authenticated", False):
    st.warning("Please enter the password on the Home page first.")
    st.write("Status: Não Autenticado")
    st.stop()

  # prevent the rest of the page from running
st.write("Autenticado")

BASE_URL = "https://dados.cvm.gov.br/dados/FI/DOC/INF_DIARIO/DADOS"
FILE_PREFIX = "inf_diario_fi"
START_YEAR = 2021


LOCAL_DB_PATH = Path("./databases/data_fundos.db")
CLOUD_DB_PATH = Path("/mnt/databases/data/data_fundos.db")

if CLOUD_DB_PATH.exists():
    DEFAULT_DB_PATH = CLOUD_DB_PATH
else:
    DEFAULT_DB_PATH = LOCAL_DB_PATH

LOCAL_BENCH_DB_PATH = Path("./databases/data_cdi_ibov.db")
CLOUD_BENCH_DB_PATH = Path("/mnt/databases/data/data_cdi_ibov.db")

if CLOUD_BENCH_DB_PATH.exists():
    DEFAULT_BENCH_DB_PATH = CLOUD_BENCH_DB_PATH
else:
    DEFAULT_BENCH_DB_PATH = LOCAL_BENCH_DB_PATH

CNPJ_DB_PATH = Path("./databases/cnpj-fundos.db")


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
) -> Optional[Tuple[str, str, str, Optional[str]]]:
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
    subclasse_norm = "ID_SUBCLASSE" if "ID_SUBCLASSE" in norm_map else None

    if not (cnpj_norm and dt_norm and quota_norm):
        return None

    return (
        norm_map[cnpj_norm],
        norm_map[dt_norm],
        norm_map[quota_norm],
        norm_map[subclasse_norm] if subclasse_norm else None,
    )


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
    # Check if id_subclasse exists
    cur = conn.execute("PRAGMA table_info(nav_daily);")
    cols = [r[1] for r in cur.fetchall()]
    
    if cols and "id_subclasse" not in cols:
        # Migration: Rename old table, create new one, copy data
        conn.execute("ALTER TABLE nav_daily RENAME TO nav_daily_old;")
        conn.execute(
            """
            CREATE TABLE nav_daily (
                cnpj TEXT NOT NULL,
                dt   TEXT NOT NULL,
                id_subclasse TEXT NOT NULL DEFAULT '',
                vl_quota REAL NOT NULL,
                PRIMARY KEY (cnpj, dt, id_subclasse)
            );
            """
        )
        conn.execute(
            """
            INSERT INTO nav_daily (cnpj, dt, vl_quota)
            SELECT cnpj, dt, vl_quota FROM nav_daily_old;
            """
        )
        conn.execute("DROP TABLE nav_daily_old;")
    else:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS nav_daily (
                cnpj TEXT NOT NULL,
                dt   TEXT NOT NULL,  -- YYYY-MM-DD
                id_subclasse TEXT NOT NULL DEFAULT '',
                vl_quota REAL NOT NULL,
                PRIMARY KEY (cnpj, dt, id_subclasse)
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


def init_cnpj_db() -> sqlite3.Connection:
    ensure_dir(os.path.dirname(CNPJ_DB_PATH) or ".")
    conn = sqlite3.connect(CNPJ_DB_PATH, check_same_thread=False)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS saved_cnpjs (
            cnpj TEXT PRIMARY KEY,
            ccy TEXT DEFAULT 'BRL',
            added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
    )
    conn.commit()
    return conn


def load_saved_cnpjs(conn: sqlite3.Connection) -> Dict[str, str]:
    cur = conn.execute("SELECT cnpj, ccy FROM saved_cnpjs")
    return {r[0]: r[1] for r in cur.fetchall()}


def save_cnpj_persistent(conn: sqlite3.Connection, cnpj: str, ccy: str) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO saved_cnpjs (cnpj, ccy) VALUES (?, ?)",
        (cnpj, ccy),
    )
    conn.commit()


def delete_cnpj_persistent(conn: sqlite3.Connection, cnpj: str) -> None:
    conn.execute("DELETE FROM saved_cnpjs WHERE cnpj = ?", (cnpj,))
    conn.commit()


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
    conn: sqlite3.Connection, rows: List[Tuple[str, str, str, float]]
) -> int:
    if not rows:
        return 0
    sql = "INSERT OR IGNORE INTO nav_daily (cnpj, dt, id_subclasse, vl_quota) VALUES (?, ?, ?, ?)"
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
                cols_detected = _detect_columns_from_header(zf, csv_name, enc)
                if cols_detected is None:
                    continue
                cnpj_col, dt_col, quota_col, sub_col = cols_detected

                use_cols = [cnpj_col, dt_col, quota_col]
                if sub_col:
                    use_cols.append(sub_col)

                with zf.open(csv_name) as f:
                    chunks = pd.read_csv(
                        f,
                        sep=";",
                        usecols=use_cols,
                        dtype={c: str for c in use_cols},
                        encoding=enc,
                        engine="python",
                        chunksize=200_000,
                    )
                    for ch in chunks:
                        rename_map = {
                            cnpj_col: "CNPJ_ID",
                            dt_col: "DT_COMPTC",
                            quota_col: "VL_QUOTA",
                        }
                        if sub_col:
                            rename_map[sub_col] = "ID_SUBCLASSE"
                        
                        ch = ch.rename(columns=rename_map)
                        
                        if "ID_SUBCLASSE" not in ch.columns:
                            ch["ID_SUBCLASSE"] = ""
                        else:
                            ch["ID_SUBCLASSE"] = ch["ID_SUBCLASSE"].fillna("").astype(str)

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
                                ch["ID_SUBCLASSE"].tolist(),
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
            SELECT dt as DT_COMPTC, vl_quota as VL_QUOTA, id_subclasse as ID_SUBCLASSE
            FROM nav_daily
            WHERE cnpj = ?
            ORDER BY dt ASC, id_subclasse ASC
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


# -------------------------- UI State --------------------------------


def init_session_state() -> None:
    if "preload_requested" not in st.session_state:
        st.session_state.preload_requested = False
    if "update_requested" not in st.session_state:
        st.session_state.update_requested = False
    
    # Initialize from DB if first run
    if "cnpj_list" not in st.session_state or "cnpj_display_names" not in st.session_state:
        conn = init_cnpj_db()
        st.session_state.cnpj_list = load_saved_cnpjs(conn)
        st.session_state.cnpj_display_names = {
            c: f"CNPJ {c}" for c in st.session_state.cnpj_list
        }
        conn.close()


def add_cnpj(cnpj_input: str, ccy: str) -> bool:
    cnpj_digits = sanitize_cnpj(cnpj_input)
    if len(cnpj_digits) != 14:
        st.error("CNPJ deve ter 14 dígitos (apenas números).")
        return False
    if cnpj_digits in st.session_state.cnpj_list:
        st.warning(f"CNPJ {cnpj_digits} já foi adicionado.")
        return False
    
    # Save to session and DB
    st.session_state.cnpj_list[cnpj_digits] = ccy
    st.session_state.cnpj_display_names[cnpj_digits] = f"CNPJ {cnpj_digits}"
    
    conn = init_cnpj_db()
    save_cnpj_persistent(conn, cnpj_digits, ccy)
    conn.close()
    return True


def remove_cnpj(cnpj_digits: str) -> None:
    if cnpj_digits in st.session_state.cnpj_list:
        del st.session_state.cnpj_list[cnpj_digits]
        del st.session_state.cnpj_display_names[cnpj_digits]
        conn = init_cnpj_db()
        delete_cnpj_persistent(conn, cnpj_digits)
        conn.close()


# -------------------------- Bulk CNPJ helpers -----------------------


def parse_cnpj_bulk_text(text: str) -> Tuple[List[str], List[str]]:
    """
    Parseia um bloco de texto e extrai CNPJs válidos (14 dígitos).
    Aceita separadores como quebra de linha, vírgula, ponto-e-vírgula, pipe e espaço.
    Retorna (validos, invalidos).
    """
    if not text:
        return [], []
    tokens = re.split(r"[\s,;|]+", text.strip())
    valid: List[str] = []
    invalid: List[str] = []
    seen: set = set()
    for tok in tokens:
        if not tok:
            continue
        digits = sanitize_cnpj(tok)
        if len(digits) == 14:
            if digits not in seen:
                valid.append(digits)
                seen.add(digits)
        else:
            invalid.append(tok)
    return valid, invalid


# -------------------------- UI Logic ---------------------------------

def render_bulk_ingestion() -> None:
    st.subheader("Gerenciar CNPJs")
    
    bulk_text = st.text_area(
        "Cole aqui os CNPJs para monitoramento (um por linha, vírgula ou espaço)",
        value="",
        placeholder="00.000.000/0000-00\n11.111.111/1111-11...",
        key="bulk_cnpj_text",
        height=180,
    )
    
    col1, col2 = st.columns([1, 1])
    with col1:
        default_ccy = st.text_input("Moeda padrão", value="BRL", key="bulk_default_ccy")
    with col2:
        if st.button("Processar e Salvar CNPJs", key="btn_bulk_add", use_container_width=True):
            valid, invalid = parse_cnpj_bulk_text(bulk_text)
            if not valid:
                st.warning("Nenhum CNPJ novo e válido detectado.")
            else:
                conn = init_cnpj_db()
                added = 0
                for v in valid:
                    if v not in st.session_state.cnpj_list:
                        save_cnpj_persistent(conn, v, default_ccy)
                        st.session_state.cnpj_list[v] = default_ccy
                        st.session_state.cnpj_display_names[v] = f"CNPJ {v}"
                        added += 1
                conn.close()
                if added > 0:
                    st.success(f"Adicionados {added} novos CNPJs!")
                else:
                    st.info("Todos os CNPJs já estavam na lista.")
                if invalid:
                    st.warning(f"Tokens ignorados (inválidos): {', '.join(invalid[:5])}...")
                st.rerun()

    if st.session_state.cnpj_list:
        st.divider()
        st.subheader("CNPJs Monitorados")
        # Multi-column display for the list to save space
        cnpjs = sorted(list(st.session_state.cnpj_list.keys()))
        cols = st.columns(3)
        for i, cnpj in enumerate(cnpjs):
            with cols[i % 3]:
                inner_col1, inner_col2 = st.columns([4, 1])
                inner_col1.write(f"`{cnpj}` ({st.session_state.cnpj_list[cnpj]})")
                if inner_col2.button("❌", key=f"del_{cnpj}", help=f"Remover {cnpj}"):
                    remove_cnpj(cnpj)
                    st.rerun()
        
        if st.button("Limpar Lista Completa", key="clear_all"):
            conn = init_cnpj_db()
            for c in list(st.session_state.cnpj_list.keys()):
                delete_cnpj_persistent(conn, c)
            conn.close()
            st.session_state.cnpj_list = {}
            st.session_state.cnpj_display_names = {}
            st.rerun()

def render_data_management_expander(data_dir: str, start_year: int, db_path: str):
    with st.expander("🛠️ Configurações de Dados e Sistema", expanded=False):
        st.info("Configurações para carga de dados da CVM (INF_DIARIO)")
        
        c1, c2 = st.columns(2)
        new_data_dir = c1.text_input("Pasta ZIPs", value=data_dir)
        new_db_path = c2.text_input("Banco SQLite", value=db_path)
        
        c3, c4 = st.columns(2)
        new_start_year = c3.number_input("Ano inicial", 2000, 2100, start_year)
        force_n = c4.slider("Refazer últimos N meses", 1, 12, 2)
        
        if st.button("Baixar/Atualizar ZIPs da CVM", use_container_width=True):
            with st.spinner("Atualizando arquivos..."):
                refresh_local_archive(new_data_dir, force_n, int(new_start_year))
            st.success("Arquivos atualizados.")

        st.divider()
        st.write("**Carga no Banco de Dados**")
        
        update_n = st.slider("Atualizar histórico (meses)", 1, 24, 2)
        col_a, col_b = st.columns(2)
        
        preload = col_a.button("Carga Total (Todos Listados)", use_container_width=True)
        update = col_b.button("Carga Parcial (Últimos N)", use_container_width=True)
        
        return {
            "data_dir": new_data_dir,
            "db_path": new_db_path,
            "start_year": int(new_start_year),
            "preload_clicked": preload,
            "update_clicked": update,
            "update_last_n": update_n
        }




# ------------------------------ Main --------------------------------


def render_analytics_section(db_path: str, bench_db_path: str):
    if not st.session_state.cnpj_list:
        st.info("Adicione CNPJs acima para começar a análise.")
        return

    st.divider()
    st.subheader("📈 Análise de Performance")
    
    selected_cnpjs = st.multiselect(
        "Selecione os fundos para comparar",
        options=sorted(list(st.session_state.cnpj_list.keys())),
        format_func=lambda x: f"{x} - {st.session_state.cnpj_list[x]}",
        key="selected_cnpjs_analytics"
    )
    
    if not selected_cnpjs:
        st.write("Selecione um ou mais fundos no campo acima.")
        return

    # Load Benchmarks
    try:
        df_cdi = load_benchmark_daily_returns(bench_db_path, "CDI", cache_buster=0)
        df_ibov = load_benchmark_daily_returns(bench_db_path, "IBOV", cache_buster=0)
    except Exception as e:
        st.error(f"Erro ao carregar benchmarks: {e}")
        df_cdi = pd.DataFrame(columns=["date", "daily_return"])
        df_ibov = pd.DataFrame(columns=["date", "daily_return"])

    plot_data = pd.DataFrame()
    
    with st.spinner("Calculando rentabilidades..."):
        for cnpj in selected_cnpjs:
            df_hist = load_history_from_db_cached(db_path, cnpj, cache_buster=0)
            if df_hist.empty:
                st.warning(f"Sem dados no banco para o CNPJ {cnpj}. Faça a carga no painel abaixo.")
                continue
            
            # Daily returns
            s_ret = daily_series_from_nav(df_hist)
            if s_ret.empty:
                continue
                
            # Cumulative returns
            s_cum = (1.0 + s_ret).cumprod() - 1.0
            
            name = f"Fundo {cnpj}"
            if plot_data.empty:
                plot_data = s_cum.to_frame(name=name)
            else:
                plot_data = plot_data.join(s_cum.rename(name), how='outer')

    if plot_data.empty:
        return

    # Align benchmarks to plot data range
    start_date = plot_data.index.min()
    end_date = plot_data.index.max()
    
    if not df_cdi.empty:
        df_cdi_filt = df_cdi[(df_cdi["date"] >= start_date) & (df_cdi["date"] <= end_date)].copy()
        if not df_cdi_filt.empty:
            df_cdi_filt = df_cdi_filt.set_index("date")["daily_return"]
            s_cdi_cum = (1.0 + df_cdi_filt).cumprod() - 1.0
            plot_data = plot_data.join(s_cdi_cum.rename("CDI"), how='left')

    if not df_ibov.empty:
        df_ibov_filt = df_ibov[(df_ibov["date"] >= start_date) & (df_ibov["date"] <= end_date)].copy()
        if not df_ibov_filt.empty:
            df_ibov_filt = df_ibov_filt.set_index("date")["daily_return"]
            s_ibov_cum = (1.0 + df_ibov_filt).cumprod() - 1.0
            plot_data = plot_data.join(s_ibov_cum.rename("IBOV"), how='left')

    plot_data = plot_data.fillna(method='ffill').fillna(0)

    # Visualization
    fig = go.Figure()
    for col in plot_data.columns:
        dash = 'dash' if col in ["CDI", "IBOV"] else None
        width = 2 if col in ["CDI", "IBOV"] else 3
        fig.add_trace(go.Scatter(
            x=plot_data.index, 
            y=plot_data[col] * 100, 
            mode='lines', 
            name=col,
            line=dict(width=width, dash=dash)
        ))
    
    fig.update_layout(
        title="Rentabilidade Acumulada (%)",
        xaxis_title="Data",
        yaxis_title="Retorno (%)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)

    # Simplified Metrics Table
    st.write("**Resumo de Retornos**")
    metrics = []
    for col in plot_data.columns:
        total_ret = plot_data[col].iloc[-1]
        metrics.append({
            "Nome": col,
            "Retorno Total (%)": f"{total_ret * 100:.2f}%",
            "Início": plot_data.index[0].strftime("%d/%m/%Y"),
            "Fim": plot_data.index[-1].strftime("%d/%m/%Y")
        })
    st.table(pd.DataFrame(metrics))


def main() -> None:
    st.set_page_config(
        page_title="CVM FI - Rentabilidade", layout="wide"
    )
    st.title("📊 Monitoramento de Rentabilidade de Fundos (CVM)")

    init_session_state()

    # 1. Advanced Settings & CVM Data (Now at top)
    cfg = render_data_management_expander(
        data_dir="./data/inf_diario",
        start_year=START_YEAR,
        db_path=DEFAULT_DB_PATH,
    )

    # 2. CNPJ Management (Persistent)
    st.divider()
    render_bulk_ingestion()

    # 3. Analytics
    render_analytics_section(db_path=DEFAULT_DB_PATH, bench_db_path=DEFAULT_BENCH_DB_PATH)

    # 4. Background Processing
    conn = get_conn(cfg["db_path"])

    if cfg["preload_clicked"]:
        months_all = tuple(yyyymm_list(cfg["start_year"]))
        cnpj_map = {
            cnpj: {
                "display_name": f"CNPJ {cnpj}",
                "ccy": st.session_state.cnpj_list.get(cnpj, "BRL"),
            }
            for cnpj in st.session_state.cnpj_list
        }
        st.info(f"Pré-carregando {len(cnpj_map)} CNPJ(s) para {len(months_all)} mês(es).")
        with st.spinner("Ingestão em andamento..."):
            inserted = preload_cnpjs_to_db(
                cnpj_map=cnpj_map, months=months_all, data_dir=cfg["data_dir"], conn=conn
            )
        st.success(f"Carga concluída. Linhas novas: {inserted}")

    if cfg["update_clicked"]:
        months_update = tuple(last_n_months(cfg["update_last_n"]))
        st.info(f"Atualizando últimos {cfg['update_last_n']} meses.")
        with st.spinner("Atualizando DB..."):
            inserted = update_db_for_existing_cnpjs(
                months=months_update, data_dir=cfg["data_dir"], conn=conn
            )
        st.success(f"Atualização concluída. Linhas novas: {inserted}")

    conn.close()


def render() -> None:
    main()


if __name__ == "__main__":
    main()
        