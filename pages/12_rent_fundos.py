# streamlit_app.py
# CVM FI (INF_DIARIO) + comparação com CDI/IBOV a partir de SQLite
# - Carrega/atualiza DB de fundos (data_fundos.db)
# - Compara com benchmarks (data_cdi_ibov.db: tabelas cdi/ibov)
# - Visualização consistente (meses alinhados e acumulado composto)
# - Tabelas incluem: Fund % de CDI e Fund - IBOV (p.p.), quando ativados
# - Cores: aplica a paleta CERES_COLORS em gráficos e tabelas
# - Fundos: cada gráfico/tabela em sua própria linha com controles de
#   altura/largura
# - Fundos: todos os gráficos com use_container_width=False
# - Fundos: removed cell background coloring (green/red), keep transparent
# - Novo: Tabela horizontal com Performance 12m, Performance 6m, Performance
#   1m e Volatilidade 12m

import os
import re
import sqlite3
import zipfile
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots

BASE_URL = "https://dados.cvm.gov.br/dados/FI/DOC/INF_DIARIO/DADOS"
FILE_PREFIX = "inf_diario_fi"
START_YEAR = 2021
DEFAULT_DB_PATH = "./data/data_fundos.db"
DEFAULT_BENCH_DB_PATH = "./data/data_cdi_ibov.db"

# ---------------------- Color Theme (CERES) -------------------------

CERES_COLORS = [
    "#8c6239",
    "#dedede",
    "#7F7F85",
    "#8c6239",
    "#1e5f88",
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
    """
    Abre DB em modo somente leitura (não cria arquivo).
    Lança erro claro se o arquivo não existir.
    """
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
    conn.execute("CREATE INDEX IF NOT EXISTS idx_nav_cnpj_dt ON nav_daily(cnpj, dt);")


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
    """
    Retorna DataFrame: ['date','daily_return'] (decimal).
    Aceita:
      - 'cdi': daily_return ou daily_rate_pct (%)
      - 'ibov': daily_return ou close (pct_change)
    """
    conn = get_conn_readonly(db_path)
    try:
        table = _resolve_table_name(conn, table_hint)
        cols = _table_columns(conn, table)
        if "daily_return" in cols:
            q = f"SELECT date, daily_return FROM {table} ORDER BY date"
            df = pd.read_sql_query(q, conn, parse_dates=["date"])
            df["daily_return"] = pd.to_numeric(df["daily_return"], errors="coerce")
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
    default_height = 380
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
        set().union(*[set(pd.to_datetime(s.index)) for s in series_map.values()])
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
    default_height = 300
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
    return fig


def build_performance_table_fig(
    display_name: str,
    ccy: str,
    as_of: pd.Timestamp,
    ret_m: pd.Series,
    df_daily: pd.DataFrame,
    height: Optional[int] = None,
    width: Optional[int] = None,
) -> go.Figure:
    r3y = compound(ret_m.tail(36)) if len(ret_m) >= 36 else np.nan

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

    yr = annual_returns_from_monthly(ret_m)
    years_available = sorted(yr.index.tolist(), reverse=True)
    curr_year = last_dt.year
    years_available = [y for y in years_available if y != curr_year]
    years_cols = years_available[:5]

    columns = [
        "Solutions",
        "Ccy",
        "Performance as of",
        "YTD (%)",
        "3Y (%)",
    ] + [str(y) for y in years_cols]

    as_of_str = as_of.strftime("%d.%m.%Y")
    row = (
        [display_name, ccy, as_of_str]
        + [_pct_str(ytd), _pct_str(r3y)]
        + [_pct_str(yr.get(y, np.nan)) for y in years_cols]
    )

    pal = get_palette()
    header_font_color = pal[3 % len(pal)]
    cell_font_color = pal[1 % len(pal)]

    table_fig = go.Figure(
        data=[
            go.Table(
                columnwidth=[280, 60, 140, 70, 70] + [60 for _ in years_cols],
                header=dict(
                    values=columns,
                    fill_color="rgba(0,0,0,0)",
                    align="left",
                    font=dict(color=header_font_color, size=12),
                    height=28,
                ),
                cells=dict(
                    values=[[v] for v in row],
                    align="left",
                    height=26,
                    fill_color="rgba(0,0,0,0)",
                    font=dict(color=cell_font_color, size=12),
                ),
            )
        ]
    )
    table_fig.update_layout(
        title="",
        height=height or 180,
        width=width,
        margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return table_fig


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


def build_performance_comparison_table_fig(
    rows: List[Dict[str, object]],
    years_cols: Optional[List[int]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
) -> go.Figure:
    """
    rows: list de dicts com chaves obrigatórias:
      name, ccy, as_of (Timestamp), ytd (float), r3y (float),
      yr_returns (pd.Series index=ano -> float)
    Opcional por linha:
      fmt: 'pct' (default) ou 'pp'
      color_mode: ignorado para cores (sem cores de fundo)
    """
    default_height = 180
    if not rows:
        return go.Figure(
            layout=dict(
                title="Performance Snapshot - Comparativo",
                height=height or default_height,
                width=width,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
        )

    all_years = set()
    for r in rows:
        yrs = getattr(r["yr_returns"], "index", [])
        all_years |= set(int(y) for y in yrs)
    curr_years = [pd.Timestamp(r["as_of"]).year for r in rows]
    max_curr = max(curr_years)
    all_years = {y for y in all_years if y != max_curr}
    years_cols = (
        sorted(all_years, reverse=True)[:5] if years_cols is None else years_cols
    )

    columns = ["Série", "Ccy", "As of", "YTD", "3Y"] + [str(y) for y in years_cols]

    col_vals: List[List[str]] = [[] for _ in columns]

    def _fmt_val(fmt: str, v: float) -> str:
        return _pct_str(v) if fmt == "pct" else _pp_str(v)

    for r in rows:
        fmt = str(r.get("fmt", "pct"))

        as_of_str = pd.Timestamp(r["as_of"]).strftime("%d.%m.%Y")
        vals = [str(r["name"]), str(r.get("ccy", "")), as_of_str]

        base_vals = [r.get("ytd", np.nan), r.get("r3y", np.nan)]
        for v in base_vals:
            vals.append(_fmt_val(fmt, v))

        yr: pd.Series = r["yr_returns"]
        for y in years_cols:
            v = float(yr.get(y, np.nan)) if isinstance(yr, pd.Series) else np.nan
            vals.append(_fmt_val(fmt, v))

        for i, v in enumerate(vals):
            col_vals[i].append(v)

    pal = get_palette()
    header_font_color = pal[3 % len(pal)]
    cell_font_color = pal[1 % len(pal)]

    table_fig = go.Figure(
        data=[
            go.Table(
                columnwidth=[180, 60, 100, 80, 80] + [60 for _ in years_cols],
                header=dict(
                    values=columns,
                    fill_color="rgba(0,0,0,0)",
                    align="left",
                    font=dict(color=header_font_color, size=12),
                    height=28,
                ),
                cells=dict(
                    values=col_vals,
                    align="left",
                    height=26,
                    fill_color="rgba(0,0,0,0)",
                    font=dict(color=cell_font_color, size=12),
                ),
            )
        ]
    )
    auto_height = default_height + 26 * max(0, len(rows) - 1)
    table_fig.update_layout(
        title="Performance Snapshot - Comparativo",
        height=height or auto_height,
        width=width,
        margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return table_fig


def build_last_12m_returns_table_multi(
    series_map: Dict[str, pd.Series],
    row_formats: Optional[Dict[str, str]] = None,  # name -> 'pct'|'pp'
    row_color_modes: Optional[Dict[str, str]] = None,  # kept for API compat
    height: Optional[int] = None,
    width: Optional[int] = None,
) -> go.Figure:
    default_height = 120
    if not series_map or all(s.empty for s in series_map.values()):
        return go.Figure(
            layout=dict(
                title="",
                height=height or default_height,
                width=width,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
        )
    row_formats = row_formats or {}
    row_color_modes = row_color_modes or {}

    # União de meses e corte para últimos 12
    all_months = sorted(
        set().union(*[set(pd.to_datetime(s.index)) for s in series_map.values()])
    )
    last12 = all_months[-12:]
    months_labels = [pd.Timestamp(d).strftime("%b %Y") for d in last12]

    # colunas
    values: List[List[str]] = []

    names = list(series_map.keys())
    values.append(names)

    def _fmt_val(name: str, v: float) -> str:
        fmt = row_formats.get(name, "pct")
        return _pct_str(v) if fmt == "pct" else _pp_str(v)

    for m_date in last12:
        col_vals = []
        for name, s in series_map.items():
            val = float(s.get(m_date, np.nan)) if not s.empty else np.nan
            col_vals.append(_fmt_val(name, val))
        values.append(col_vals)

    pal = get_palette()
    header_font_color = pal[3 % len(pal)]
    cell_font_color = pal[1 % len(pal)]

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=["Série"] + months_labels,
                    fill_color="rgba(0,0,0,0)",
                    align="center",
                    font=dict(color=header_font_color, size=11, family="Arial"),
                    height=28,
                ),
                cells=dict(
                    values=values,
                    align="center",
                    height=26,
                    fill_color="rgba(0,0,0,0)",
                    font=dict(color=cell_font_color, size=11, family="Arial"),
                ),
            )
        ]
    )

    auto_height = 100 + 24 * max(0, len(series_map) - 1)
    fig.update_layout(
        title="",
        height=height or auto_height,
        width=width,
        margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    return fig


# -------- New: Horizontal table with 12m/6m/1m performance + 12m vol ---------


def _compound_last_n_months(ret_m: pd.Series, n: int) -> float:
    if ret_m is None or ret_m.empty:
        return float("nan")
    s = ret_m.dropna()
    if len(s) < n:
        return float("nan")
    return float(np.prod(1.0 + s.tail(n).values) - 1.0)


def _last_1m(ret_m: pd.Series) -> float:
    if ret_m is None or ret_m.empty:
        return float("nan")
    s = ret_m.dropna()
    if s.empty:
        return float("nan")
    return float(s.iloc[-1])


def _ann_vol_12m_from_daily(daily: pd.Series) -> float:
    if daily is None or daily.empty:
        return float("nan")
    s = daily.dropna()
    if s.empty:
        return float("nan")
    end = pd.to_datetime(s.index.max())
    start = end - pd.DateOffset(months=12)
    win = s[(s.index > start) & (s.index <= end)]
    # Require a minimum number of observations for stability
    if len(win) < 60:
        return float("nan")
    vol = float(win.std(ddof=1) * np.sqrt(252.0))
    return vol


def build_window_perf_table_multi(
    series_m_map: Dict[str, pd.Series],
    series_d_map: Dict[str, pd.Series],
    order: Optional[List[str]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
) -> go.Figure:
    """
    Tabela horizontal com colunas:
    ['Série', 'Perf 12m', 'Perf 6m', 'Perf 1m', 'Vol 12m'].
    - Perf 12m/6m/1m: composto a partir de retornos mensais (ret_m).
    - Vol 12m: volatilidade anualizada a partir de retornos diários dos
      últimos 12 meses (std * sqrt(252)).
    """
    names = order or list(series_m_map.keys())
    if not names:
        return go.Figure(
            layout=dict(
                title="",
                height=height or 140,
                width=width,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
        )

    col_labels = ["Série", "Perf 12m", "Perf 6m", "Perf 1m", "Vol 12m"]
    col_values: List[List[str]] = [[] for _ in col_labels]

    for name in names:
        ret_m = series_m_map.get(name, pd.Series(dtype=float))
        ret_d = series_d_map.get(name, pd.Series(dtype=float))

        p12 = _compound_last_n_months(ret_m, 12)
        p6 = _compound_last_n_months(ret_m, 6)
        p1 = _last_1m(ret_m)
        v12 = _ann_vol_12m_from_daily(ret_d)

        row_vals = [name, _pct_str(p12), _pct_str(p6), _pct_str(p1), _pct_str(v12)]
        for i, v in enumerate(row_vals):
            col_values[i].append(v)

    pal = get_palette()
    header_font_color = pal[3 % len(pal)]
    cell_font_color = pal[1 % len(pal)]

    table_fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=col_labels,
                    fill_color="rgba(0,0,0,0)",
                    align="center",
                    font=dict(color=header_font_color, size=12),
                    height=28,
                ),
                cells=dict(
                    values=col_values,
                    align="center",
                    height=26,
                    fill_color="rgba(0,0,0,0)",
                    font=dict(color=cell_font_color, size=12),
                ),
            )
        ]
    )

    auto_height = 140 + 26 * max(0, len(names) - 1)
    table_fig.update_layout(
        title="",
        height=height or auto_height,
        width=width,
        margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return table_fig


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
    # Initialize color palette state
    if "ceres_colors" not in st.session_state:
        st.session_state.ceres_colors = CERES_COLORS.copy()


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
) -> Tuple[str, str, int, str, int, bool, bool, str, bool, bool]:
    """
    Retorna:
      selected_cnpj, data_dir, start_year, db_path,
      update_last_n, preload_clicked, update_clicked,
      bench_db_path, compare_cdi, compare_ibov
    """
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
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(
                    f"**{st.session_state.cnpj_display_names[cnpj_digits]}** "
                    f"({cnpj_digits}) — "
                    f"{st.session_state.cnpj_list[cnpj_digits]}"
                )
            with col2:
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
    st.subheader("Tema de cores (CERES)")
    st.caption("Altere rapidamente as cores do tema abaixo.")
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
    )


# ------------------------------ Main --------------------------------


def main() -> None:
    st.set_page_config(
        page_title="CVM FI - DB Loader e Visualização", layout="wide"
    )
    st.title("CVM FI - Informe Diário (INF_DIARIO) — DB e Visualização")

    init_session_state()

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

    # ---------- Comparativos: construir a partir de retornos diários ----------
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

    # Interseção de datas (comparação justa)
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

    # ------------- Derivados: % CDI e Diferença p.p. vs IBOV ----------
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
        row_color_modes[ratio_name] = "ratio"  # mantido para formatação apenas

    if "IBOV" in ret_m_map:
        ibov_m = ret_m_map["IBOV"]
        idx = fund_m.index.intersection(ibov_m.index)
        diff_m = fund_m.reindex(idx) - ibov_m.reindex(idx)
        diff_name = f"Diferença (p.p.)"
        ret_m_map[diff_name] = diff_m
        row_formats[diff_name] = "pp"
        row_color_modes[diff_name] = "raw"

    # ---------------------- Gráficos e Tabelas -----------------------
    # Cada um em sua própria linha, com controles de altura e largura
    # 1) Retornos mensais — Barras
    st.subheader("Retorno mensal — barras (comparativo)")
    base_series_for_bars = {
        k: v for k, v in ret_m_map.items() if (k == fund_name) or (k in ("CDI", "IBOV"))
    }
    names_for_bars = list(base_series_for_bars.keys())
    bar_color_map = name_color_map(names_for_bars)
    default_bar_height = 380
    default_bar_width = 1100
    c1, c2 = st.columns(2)
    with c1:
        bar_height = st.number_input(
            "Altura (px) — Retornos mensais",
            min_value=200,
            max_value=2000,
            value=default_bar_height,
            step=20,
            key="height_ml",
        )
    with c2:
        bar_width = st.number_input(
            "Largura (px) — Retornos mensais",
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

    # 2) Tabela de Performance (comparativo)
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
        # YTD e 3Y
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

    default_perf_table_height = 180 + 26 * max(0, len(rows) - 1)
    default_perf_table_width = 1100
    t1, t2 = st.columns(2)
    with t1:
        perf_table_height = st.number_input(
            "Altura (px) — Tabela de performance",
            min_value=140,
            max_value=2000,
            value=int(default_perf_table_height),
            step=20,
            key="height_perf_table",
        )
    with t2:
        perf_table_width = st.number_input(
            "Largura (px) — Tabela de performance",
            min_value=500,
            max_value=3000,
            value=default_perf_table_width,
            step=50,
            key="width_perf_table",
        )
    table_fig = build_performance_comparison_table_fig(
        rows, height=perf_table_height, width=perf_table_width
    )
    st.plotly_chart(table_fig, use_container_width=False)

    # 2.5) Nova Tabela — Performance 12m, 6m, 1m e Vol 12m (horizontal)
    st.subheader("Janelas: 1m, 6m, 12m e Vol 12m (horizontal)")
    base_metric_names = [fund_name] + [n for n in ("CDI", "IBOV") if n in daily_map]
    series_m_sel = {n: ret_m_map[n] for n in base_metric_names}
    series_d_sel = {n: daily_map[n] for n in base_metric_names}

    default_window_height = 140 + 26 * max(0, len(base_metric_names) - 1)
    default_window_width = 900
    w1, w2 = st.columns(2)
    with w1:
        window_table_height = st.number_input(
            "Altura (px) — Tabela janelas",
            min_value=120,
            max_value=2000,
            value=int(default_window_height),
            step=20,
            key="height_window_table",
        )
    with w2:
        window_table_width = st.number_input(
            "Largura (px) — Tabela janelas",
            min_value=500,
            max_value=3000,
            value=default_window_width,
            step=50,
            key="width_window_table",
        )

    fig_window_table = build_window_perf_table_multi(
        series_m_map=series_m_sel,
        series_d_map=series_d_sel,
        order=base_metric_names,
        height=window_table_height,
        width=window_table_width,
    )
    st.plotly_chart(fig_window_table, use_container_width=False)

    # 3) Últimos 12 meses — Tabela
    st.subheader("Últimos 12 meses — Retornos mensais (comparativo)")
    default_12m_height = max(120, 100 + 24 * max(0, len(ret_m_map) - 1))
    default_12m_width = 1100
    m1, m2 = st.columns(2)
    with m1:
        last12_height = st.number_input(
            "Altura (px) — Tabela 12 meses",
            min_value=120,
            max_value=2000,
            value=int(default_12m_height),
            step=20,
            key="height_12m_table",
        )
    with m2:
        last12_width = st.number_input(
            "Largura (px) — Tabela 12 meses",
            min_value=500,
            max_value=3000,
            value=default_12m_width,
            step=50,
            key="width_12m_table",
        )
    fig_12m_table = build_last_12m_returns_table_multi(
        ret_m_map,
        row_formats=row_formats,
        row_color_modes=row_color_modes,
        height=last12_height,
        width=last12_width,
    )
    st.plotly_chart(fig_12m_table, use_container_width=False)

    # 4) Série (linha) — Retorno acumulado
    st.subheader("Série (linha) — Retorno acumulado (composto)")
    base_line_series = {
        k: v
        for k, v in cumret_map.items()
        if (k == fund_name) or (k in ("CDI", "IBOV"))
    }
    names_for_lines = list(base_line_series.keys())
    line_color_map = name_color_map(names_for_lines)
    default_line_height = 300
    default_line_width = 1100
    l1, l2 = st.columns(2)
    with l1:
        line_height = st.number_input(
            "Altura (px) — Linha acumulada",
            min_value=200,
            max_value=2000,
            value=default_line_height,
            step=20,
            key="height_line",
        )
    with l2:
        line_width = st.number_input(
            "Largura (px) — Linha acumulada",
            min_value=500,
            max_value=3000,
            value=default_line_width,
            step=50,
            key="width_line",
        )

    nav_line = make_nav_line_chart_multi(
        base_line_series,
        "Retorno acumulado (composto) — comparativo",
        color_map=line_color_map,
        height=line_height,
        width=line_width,
    )
    st.plotly_chart(nav_line, use_container_width=False)

    with st.expander("Detalhes e download"):
        st.write(
            "- Fonte: CVM Dados Abertos - FI/DOC/INF_DIARIO\n"
            "- Armazenamento: SQLite (data_fundos.db)\n"
            "- Retorno mensal: composto a partir de retornos diários por "
            "mês-calendário\n"
            "- 12m/3y: composto de retornos mensais\n"
            "- YTD: composto de retornos diários desde 1º dia útil do ano\n"
            "- Linha: retorno acumulado composto (base 0 = início comum)\n"
            "- Benchmarks: CDI/IBOV lidos de data_cdi_ibov.db "
            "(tabelas 'cdi'/'ibov')\n"
            "- Tabelas incluem: Fund % de CDI (quando ligado) e "
            "Fund - IBOV em p.p.\n"
            "- Tema: paleta CERES aplicada em gráficos e tipografia; fundos e "
            "gráficos com fundo transparente."
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
            file_name=f"inf_diario_{selected_cnpj}_diario.csv",
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
            file_name=f"inf_diario_{selected_cnpj}_mensal.csv",
            mime="text/csv",
        )


def render() -> None:
    main()


if __name__ == "__main__":
    main()