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
#   6) NOVO: Adição em lote de CNPJs e menu de fundos movido para uma aba.

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
from pathlib import Path

# great_tables
from great_tables import GT, style, loc

# streamlit-extras for Great Tables inline rendering
# Official API: great_tables(table: GT, width='stretch'|'content'|int)
try:
    from streamlit_extras.great_tables import great_tables as stx_great_tables
except Exception:
    stx_great_tables = None  # fallback HTML


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


def bulk_add_cnpjs(
    cnpjs: List[str], default_ccy: str = "BRL", replace: bool = False
) -> Dict[str, int]:
    """
    Adiciona em lote à sessão. Se replace=True, limpa lista atual antes.
    Não usa nomes customizados; display_name = 'CNPJ {digits}'.
    Retorna contadores: added, duplicates.
    """
    if replace:
        st.session_state.cnpj_list = {}
        st.session_state.cnpj_display_names = {}
    added = 0
    duplicates = 0
    for c in cnpjs:
        if c in st.session_state.cnpj_list:
            duplicates += 1
            continue
        st.session_state.cnpj_list[c] = default_ccy
        st.session_state.cnpj_display_names[c] = f"CNPJ {c}"
        added += 1
    return {"added": added, "duplicates": duplicates}


# -------------------------- Sidebar UI ------------------------------


def render_config_sidebar(
    data_dir: str, start_year: int, db_path: str
) -> Tuple[str, int, str, int, int, int, int]:
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

    return (
        data_dir,
        int(start_year),
        db_path,
        int(table_width_px),
        int(height_perf_table_px),
        int(height_window_table_px),
        int(height_12m_table_px),
    )


def render_fundos_tab(db_path: str) -> Tuple[str, int, bool, bool, str, bool, bool]:
    st.header("Rentabilidade de Fundos")

    st.subheader("Gerenciar CNPJs")
    with st.expander("Adicionar em lote", expanded=True):
        colb1, colb2 = st.columns([2, 1])
        with colb1:
            bulk_text = st.text_area(
                "Cole CNPJs (um por linha, vírgula ou espaço)",
                value="",
                placeholder="00.000.000/0000-00\n11.111.111/1111-11, 22222222000122 ...",
                key="bulk_cnpj_text",
                height=140,
            )
        with colb2:
            bulk_file = st.file_uploader(
                "Ou faça upload (CSV/TXT)",
                type=["csv", "txt"],
                key="bulk_cnpj_file",
                help="O conteúdo será analisado para extrair CNPJs.",
            )
        colb3, colb4, colb5 = st.columns([1, 1, 2])
        with colb3:
            default_ccy = st.text_input(
                "Moeda padrão", value="BRL", key="bulk_default_ccy"
            )
        with colb4:
            replace_flag = st.checkbox(
                "Substituir lista atual", value=False, key="bulk_replace"
            )
        with colb5:
            if st.button("Adicionar em lote", key="btn_bulk_add"):
                text_sources = [bulk_text or ""]
                if bulk_file is not None:
                    try:
                        content = bulk_file.read()
                        try:
                            text_sources.append(content.decode("utf-8"))
                        except Exception:
                            text_sources.append(content.decode("latin-1"))
                    except Exception:
                        pass
                all_text = "\n".join([t for t in text_sources if t])
                valid, invalid = parse_cnpj_bulk_text(all_text)
                if not valid and not invalid:
                    st.warning("Nenhum CNPJ detectado.")
                else:
                    res = bulk_add_cnpjs(valid, default_ccy, replace_flag)
                    st.success(
                        f"Adicionados: {res['added']} | Duplicados ignorados: {res['duplicates']} | "
                        f"Inválidos: {len(invalid)}"
                    )
                    st.rerun()

    # Adição unitária (continua disponível)
    ccol1, ccol2, ccol3 = st.columns([2, 2, 1])
    with ccol1:
        cnpj_input = st.text_input(
            "CNPJ do fundo/classe",
            value="",
            placeholder="00.000.000/0000-00",
            help="Use o CNPJ da classe conforme INF_DIARIO.",
            key="cnpj_input_tab",
        )
    with ccol2:
        ccy = st.text_input("Moeda", value="BRL", key="ccy_tab")
    with ccol3:
        if st.button("Adicionar CNPJ", key="add_cnpj_tab"):
            # Sem nome customizado: usa padrão
            if add_cnpj(cnpj_input, display_name="", ccy=ccy):
                st.success("CNPJ adicionado!")
                st.rerun()

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
                if st.button("❌", key=f"remove_tab_{cnpj_digits}", help="Remover"):
                    remove_cnpj(cnpj_digits)
                    st.rerun()
        if st.button("Remover todos", key="remove_all_cnpjs"):
            st.session_state.cnpj_list = {}
            st.session_state.cnpj_display_names = {}
            st.rerun()

    st.divider()
    st.subheader("Carga no DB (para os CNPJs listados)")
    preload_clicked = (
        st.button("Pré-carregar em DB (todos listados)", key="preload_db_tab")
        if st.session_state.cnpj_list
        else False
    )
    update_last_n = st.slider(
        "Atualizar DB: últimos N meses",
        min_value=1,
        max_value=24,
        value=2,
        help="Ingerir apenas os últimos N meses para CNPJs já no DB.",
        key="update_last_n_tab",
    )
    update_clicked = st.button("Atualizar DB (últimos N meses)", key="update_db_tab")

    st.divider()
    st.subheader("Benchmarks (CDI/IBOV)")
    bench_db_path = st.text_input(
        "Caminho do banco (benchmarks)",
        value=DEFAULT_BENCH_DB_PATH,
        help="Banco com tabelas 'cdi' e/ou 'ibov'.",
        key="bench_db_path_tab",
    )
    cols_bm = st.columns(2)
    with cols_bm[0]:
        compare_cdi = st.checkbox("Comparar com CDI", value=False, key="cmp_cdi_tab")
    with cols_bm[1]:
        compare_ibov = st.checkbox("Comparar com IBOV", value=False, key="cmp_ibov_tab")

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
        key="selected_cnpj_view_tab",
    )

    st.divider()
    st.subheader("Manutenção e Exportação")
    m_col1, m_col2 = st.columns(2)
    with m_col1:
        # Download button for the whole DB
        if st.button("Preparar CSV para Download (Todo o DB)", key="prep_download_db"):
            conn = get_conn(db_path)
            try:
                df_all = pd.read_sql_query("SELECT * FROM nav_daily", conn)
                csv_data = df_all.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Clique aqui para Baixar CSV",
                    data=csv_data,
                    file_name="data_fundos_full.csv",
                    mime="text/csv",
                )
            except Exception as e:
                st.error(f"Erro ao preparar download: {e}")
            finally:
                conn.close()

    with m_col2:
        reingest_clicked = st.button(
            "RE-INGERIR TODO O HISTÓRICO",
            key="btn_reingest_all",
            help="Limpa a tabela nav_daily e re-processa todos os ZIPs para os CNPJs já conhecidos no banco.",
        )
        if reingest_clicked:
            st.warning("Isso irá apagar os dados atuais da tabela nav_daily e re-ler todos os arquivos ZIP. Deseja prosseguir?")
            if st.button("Sim, limpar e re-ingerir", key="confirm_reingest"):
                st.session_state.full_reingest_confirmed = True
                st.rerun()

    return (
        selected_cnpj,
        int(update_last_n),
        bool(preload_clicked),
        bool(update_clicked),
        bench_db_path,
        bool(compare_cdi),
        bool(compare_ibov),
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
            data_dir,
            start_year,
            db_path,
            table_width_px,
            height_perf_table_px,
            height_window_table_px,
            height_12m_table_px,
        ) = render_config_sidebar(
            data_dir="./data/inf_diario",
            start_year=START_YEAR,
            db_path=DEFAULT_DB_PATH,
        )

    # Painel principal: Rentabilidade de Fundos (inclui menu movido e bulk add)
    (tab_rf,) = st.tabs(["Rentabilidade de Fundos"])
    with tab_rf:
        (
            selected_cnpj,
            update_last_n,
            preload_clicked,
            update_clicked,
            bench_db_path,
            compare_cdi,
            compare_ibov,
        ) = render_fundos_tab(db_path=db_path)

    # Executa cargas no DB conforme ações no painel da aba
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

    if st.session_state.get("full_reingest_confirmed", False):
        st.session_state.full_reingest_confirmed = False
        known_cnpjs = cnpjs_in_db(conn)
        meta_map = cnpj_meta_map_from_db(conn)
        
        if not known_cnpjs:
            st.warning("Nenhum CNPJ no banco para re-ingerir.")
        else:
            st.info(f"Iniciando re-ingestão total para {len(known_cnpjs)} CNPJs.")
            with st.spinner("Limpando e re-ingerindo..."):
                conn.execute("DELETE FROM nav_daily;")
                conn.commit()
                
                months_all = tuple(yyyymm_list(start_year))
                cnpj_map_to_reingest = {
                    c: meta_map.get(c, {"display_name": f"CNPJ {c}", "ccy": "BRL"})
                    for c in known_cnpjs
                }
                
                inserted = preload_cnpjs_to_db(
                    cnpj_map=cnpj_map_to_reingest,
                    months=months_all,
                    data_dir=data_dir,
                    conn=conn
                )
            st.success(f"Re-ingestão concluída. Total de linhas: {inserted}")

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
    st.plotly_chart(fig_ml, width='content')

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
    st.plotly_chart(nav_line, width='content')

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
        