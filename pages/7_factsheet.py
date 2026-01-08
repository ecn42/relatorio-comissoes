# pages/05_Export_Factsheet_Data.py
# RAW factsheet data -> XLSX (clean zebra style)
# - Headers centered
# - Font: Century Gothic
# - CNPJ in meta formatted as 00.000.000/0000-00
# - White sheets, bold headers (white text on blank background where requested)
# - Odd data rows shaded light gray (where applicable)
# - No gridlines, no freeze panes
# - All "PCT" columns renamed to "%"
# - Layout changes per request:
#   - meta: B1 white bold size 14, B2 white size 8, B4 white size 10
#     (blank background)
#   - line_index: daily returns with columns [date, Fundo, CDI, IBOV]
#     (always include both) + cumulative returns [fundo_cum, cdi_cum, ibov_cum]
#   - perf_table layout (per fund benchmark selection):
#       | SÉRIE | YTD | 3M | 6M | 12M | 24M | 36M |
#       | FUNDO | ... |
#       | BENCH | ... |  (BENCH = CDI or IBOV depending on selection)
#       | % CDI | ... |  (if BENCH = CDI, ratio FUNDO/BENCH)
#       | DIFERENÇA (P.P.) | ... | (if BENCH = IBOV, FUNDO - BENCH)
#       | VOL   | ... |   (fund vol, annualized; rf = CDI only for Sharpe)
#       | SHARPE| ... |   (fund Sharpe, annualized, rf = CDI)
#     - Century Gothic size 10, centered, header white bold (blank background),
#       all cols ~100 px, rows ~23 px
#   - top_positions: Century Gothic size 10; widths A=187 px, B=320 px, C=100 px;
#       rows ~22 px; header white bold (blank background); uppercase
# - New features:
#   - Choose which funds will use IBOV as benchmark (default is CDI)
#   - Bulk download: generate one XLSX per fund and provide a ZIP with all files
#   - Alternative input: paste CNPJs with benchmarks (cnpj;CDI or cnpj;IBOV)
#     and generate a ZIP for that list

from __future__ import annotations

import io
import os
import re
import sqlite3
import zipfile
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st


### Simple Authentication
if not st.session_state.get("authenticated", False):
    st.warning("Please enter the password on the Home page first.")
    st.write("Status: Não Autenticado")
    st.stop()
    
  # prevent the rest of the page from running
st.write("Autenticado")

# ---------------------- Data helpers ----------------------


def sanitize_cnpj(s: str) -> str:
    return re.sub(r"\D", "", s or "")


def format_cnpj(digits: str) -> str:
    d = sanitize_cnpj(digits)
    if len(d) != 14:
        return digits
    return f"{d[0:2]}.{d[2:5]}.{d[5:8]}/{d[8:12]}-{d[12:14]}"


def ensure_conn(db_path: str) -> sqlite3.Connection:
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"DB not found: {db_path}")
    return sqlite3.connect(db_path)


def daily_returns_from_nav(df_daily: pd.DataFrame) -> pd.Series:
    dfd = df_daily.sort_values("DT_COMPTC")
    return dfd.set_index("DT_COMPTC")["VL_QUOTA"].pct_change().dropna()


def monthly_from_daily(s: pd.Series) -> pd.Series:
    if s.empty:
        return pd.Series(dtype=float)
    s = s.copy()
    s.index = pd.to_datetime(s.index).tz_localize(None)
    grp = s.groupby(s.index.to_period("M"))
    ret_m = grp.apply(lambda g: float(np.prod(1.0 + g.values) - 1.0))
    ret_m.index = ret_m.index.to_timestamp("M")
    return ret_m


def ytd_from_daily(s: pd.Series) -> float:
    if s.empty:
        return float("nan")
    last_dt = s.index.max()
    y0 = pd.Timestamp(year=last_dt.year, month=1, day=1)
    sub = s[s.index >= y0]
    if sub.empty:
        return float("nan")
    return float(np.prod(1.0 + sub.values) - 1.0)


def annual_from_monthly(ret_m: pd.Series) -> pd.Series:
    if ret_m.empty:
        return pd.Series(dtype=float)
    s = ret_m.copy()
    s.index = pd.to_datetime(s.index)
    grp = s.groupby(s.index.year)

    def _year(g: pd.Series) -> float:
        return float(np.prod(1.0 + g.values) - 1.0) if len(g) >= 12 else np.nan

    vr = grp.apply(_year)
    vr.index.name = "year"
    return vr.dropna()


def _list_tables(conn: sqlite3.Connection) -> List[str]:
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type IN ('table','view')")
    return [r[0] for r in cur.fetchall()]


def _resolve_table(conn: sqlite3.Connection, candidates: List[str]) -> Optional[str]:
    names = _list_tables(conn)
    low = {n.lower(): n for n in names}
    for c in candidates:
        if c.lower() in low:
            return low[c.lower()]
    for n in names:
        if any(c.lower() in n.lower() for c in candidates):
            return n
    return None


def _table_columns(conn: sqlite3.Connection, table: str) -> List[str]:
    cur = conn.execute(f"PRAGMA table_info({table})")
    return [r[1] for r in cur.fetchall()]


def _resolve_date_col(cols: List[str]) -> Optional[str]:
    cand = ["date", "dt", "data", "refdate", "ref_date"]
    for c in cand:
        for col in cols:
            if col.lower() == c:
                return col
    for c in cand:
        for col in cols:
            if c in col.lower():
                return col
    return None


def load_benchmark_daily(db_paths: List[str], kind: str) -> pd.Series:
    kind = kind.lower()
    if kind not in ("cdi", "ibov"):
        return pd.Series(dtype=float)
    table_cands = {
        "cdi": ["cdi", "bench_cdi", "selic", "cdi_diario"],
        "ibov": ["ibov", "ibovespa", "bovespa", "ibov_daily"],
    }[kind]
    for db_path in db_paths:
        if not db_path or not os.path.exists(db_path):
            continue
        conn = sqlite3.connect(db_path)
        try:
            table = _resolve_table(conn, table_cands)
            if not table:
                continue
            cols = _table_columns(conn, table)
            date_col = _resolve_date_col(cols)
            if not date_col:
                continue
            has_dr = any(c.lower() == "daily_return" for c in cols)
            has_dr_pct = any(c.lower() == "daily_rate_pct" for c in cols)
            price_col = None
            for c in ["close", "price", "fechamento", "valor", "vl_close"]:
                for col in cols:
                    if col.lower() == c:
                        price_col = col
                        break
                if price_col:
                    break
            if kind == "cdi":
                if has_dr:
                    q = (
                        f"SELECT {date_col} as dt, daily_return as ret "
                        f"FROM {table} ORDER BY {date_col}"
                    )
                    df = pd.read_sql_query(q, conn, parse_dates=["dt"])
                    s = pd.to_numeric(df["ret"], errors="coerce").dropna()
                    s.index = df["dt"]
                    if not s.empty:
                        return s
                if has_dr_pct:
                    q = (
                        f"SELECT {date_col} as dt, daily_rate_pct as ret "
                        f"FROM {table} ORDER BY {date_col}"
                    )
                    df = pd.read_sql_query(q, conn, parse_dates=["dt"])
                    s = pd.to_numeric(df["ret"], errors="coerce").dropna() / 100.0
                    s.index = df["dt"]
                    if not s.empty:
                        return s
            else:
                if has_dr:
                    q = (
                        f"SELECT {date_col} as dt, daily_return as ret "
                        f"FROM {table} ORDER BY {date_col}"
                    )
                    df = pd.read_sql_query(q, conn, parse_dates=["dt"])
                    s = pd.to_numeric(df["ret"], errors="coerce").dropna()
                    s.index = df["dt"]
                    if not s.empty:
                        return s
                if price_col:
                    q = (
                        f"SELECT {date_col} as dt, {price_col} as px "
                        f"FROM {table} ORDER BY {date_col}"
                    )
                    df = pd.read_sql_query(q, conn, parse_dates=["dt"])
                    px = pd.to_numeric(df["px"], errors="coerce")
                    s = px.pct_change().dropna()
                    if not s.empty:
                        s.index = df["dt"].iloc[1:]
                        return s
        finally:
            conn.close()
    return pd.Series(dtype=float)


def load_nav_series(db_path: str, cnpj: str) -> pd.DataFrame:
    conn = ensure_conn(db_path)
    try:
        df = pd.read_sql_query(
            """
            SELECT dt as DT_COMPTC, vl_quota as VL_QUOTA
            FROM nav_daily
            WHERE cnpj = ?
            ORDER BY dt ASC
            """,
            conn,
            params=(cnpj,),
            parse_dates=["DT_COMPTC"],
        )
        return df
    finally:
        conn.close()


def categorize_tp_ativo(tp_ativo: str) -> str:
    # if pd.isna(tp_ativo) or not str(tp_ativo).strip():
    return tp_ativo
    # s = str(tp_ativo).upper()
    # if any(kw in s for kw in ["CDB", "RDB", "LETRA FINANCEIRA", "LF", "LCR"]):
    #     return "Depósitos a prazo e outros títulos de IF"
    # if any(kw in s for kw in ["TÍTULO PÚBLICO", "NTN", "LTN", "LFT", "TESOURO"]):
    #     return "TÍTULOS PÚBLICOS"
    # if "COMPROMISSADA" in s:
    #     return "OPERAÇÕES COMPROMISSADAS"
    # if "AÇÃO" in s:
    #     return "AÇÕES"
    # if "FUNDO" in s:
    #     return "FUNDO DE INVESTIMENTO"
    # if "IMOBILIÁRIO" in s or "FII" in s:
    #     return "IMÓVEIS"
    # return "Outros"


def get_alloc_donut(df_blc: pd.DataFrame) -> pd.DataFrame:
    if df_blc.empty:
        return pd.DataFrame(columns=["TP_ATIVO", "PCT"])
    if "PCT_CARTEIRA" not in df_blc.columns or "TP_ATIVO" not in df_blc.columns:
        return pd.DataFrame(columns=["TP_ATIVO", "PCT"])
    tmp = df_blc.copy()
    tmp["TP_APLIC"] = tmp["TP_ATIVO"].apply(categorize_tp_ativo)
    agg = (
        tmp.groupby("TP_ATIVO")["PCT_CARTEIRA"]
        .sum()
        .reset_index()
        .rename(columns={"PCT_CARTEIRA": "PCT"})
        .sort_values("PCT", ascending=False)
    )
    # if len(agg) > 6:
    #     top = agg.head(5)
    #     rest = pd.DataFrame(
    #         [["Outros", float(agg["PCT"].iloc[5:].sum())]],
    #         columns=["TP_ATIVO", "PCT"],
    #     )
    #     agg = pd.concat([top, rest], ignore_index=True)
    return agg.reset_index(drop=True)


def get_top_positions(df_blc: pd.DataFrame) -> pd.DataFrame:
    """
    TOP 10 posições:
    - First column uses TP_APLIC (if missing, falls back to TP_ATIVO).
    - Second column (EMISSOR) is chosen by TP_* rules:
        DS_ATIVO for:
          AÇÃO ORDINÁRIA, AÇÃO PREFERENCIAL, BDR NÃO PATROCINADO, BDR NÍVEL I,
          BDR NÍVEL II, BDR NÍVEL III, CERTIFICADO DE DEPÓSITO DE AÇÕES,
          CONTRATO FUTURO, OPÇÃO DE COMPRA, OPÇÃO DE VENDA, OUTROS
        NM_FUNDO_CLASSE_SUBCLASSE_COTA for:
          FI IMOBILIÁRIO, FIDC, FUNDOS DE INVESTIMENTO E DE COTAS
        FUNDOS DE ÍNDICE: prefer DS_ATIVO; else NM_FUNDO_CLASSE_SUBCLASSE_COTA.
      - For TP_* not on the list: use EMISSOR; if missing, fallback to DS_ATIVO.
    - Single-letter names like "N" or "S" are blanked out.
    Returns columns ["TP_APLIC", "EMISSOR", "PCT"].
    """
    if df_blc.empty or "PCT_CARTEIRA" not in df_blc.columns:
        return pd.DataFrame(columns=["TP_APLIC", "EMISSOR", "PCT"])

    work = df_blc.copy()

    # Ensure grouping and rules columns exist
    if "TP_APLIC" not in work.columns and "TP_ATIVO" not in work.columns:
        work["TP_APLIC"] = "N/A"
        work["TP_ATIVO"] = "N/A"
    elif "TP_APLIC" not in work.columns:
        work["TP_APLIC"] = work.get("TP_ATIVO", "N/A")
    elif "TP_ATIVO" not in work.columns:
        work["TP_ATIVO"] = work.get("TP_APLIC", "N/A")

    work["PCT"] = pd.to_numeric(work["PCT_CARTEIRA"], errors="coerce").fillna(0.0)

    # Sets to drive the source of the "name" (2nd column)
    ds_types = {
        "AÇÃO ORDINÁRIA",
        "AÇÃO PREFERENCIAL",
        "BDR NÃO PATROCINADO",
        "BDR NÍVEL I",
        "BDR NÍVEL II",
        "BDR NÍVEL III",
        "CERTIFICADO DE DEPÓSITO DE AÇÕES",
        "CONTRATO FUTURO",
        "OPÇÃO DE COMPRA",
        "OPÇÃO DE VENDA",
        "OUTROS",
    }
    nm_types = {
        "FI IMOBILIÁRIO",
        "FIDC",
        "FUNDO DE INVESTIMENTO E DE COTAS",
    }
    special_fundos_indice = "FUNDOS DE ÍNDICE"

    def _has_value(v: object) -> bool:
        return (v is not None) and (not pd.isna(v)) and (str(v).strip() != "")

    def _blank_single_letter(v: object) -> str:
        s = "" if v is None or pd.isna(v) else str(v).strip()
        # If the final name is a single alphabetic character (e.g., "N" or "S"),
        # keep it blank instead.
        if len(s) == 1 and s.isalpha():
            return ""
        return s

    # Use TP_ATIVO for deciding the source column; if absent, fallback to TP_APLIC
    rules_col = "TP_ATIVO" if "TP_ATIVO" in work.columns else "TP_APLIC"

    def _pick_name(row: pd.Series) -> str:
        tp_raw = row.get(rules_col, "")
        tp = ("" if pd.isna(tp_raw) else str(tp_raw)).upper().strip()

        ds = row.get("DS_ATIVO", None)
        nm = row.get("NM_FUNDO_CLASSE_SUBCLASSE_COTA", None)

        em = row.get("EMISSOR", None)
        if not _has_value(em):
            em = row.get("EMISSOR_LIGADO", em)

        if tp in ds_types:
            return str(ds).strip() if _has_value(ds) else (
                str(em).strip() if _has_value(em) else "N/A"
            )
        if tp in nm_types:
            return str(nm).strip() if _has_value(nm) else (
                str(em).strip() if _has_value(em) else "N/A"
            )
        if tp == special_fundos_indice:
            if _has_value(ds):
                return str(ds).strip()
            if _has_value(nm):
                return str(nm).strip()
            return str(em).strip() if _has_value(em) else "N/A"

        # Default: use EMISSOR; if missing, fallback to DS_ATIVO
        if _has_value(em):
            return str(em).strip()
        return str(ds).strip() if _has_value(ds) else "N/A"

    # Construct display "EMISSOR" per rules above
    work["EMISSOR"] = work.apply(_pick_name, axis=1)
    # Blank out single-letter names like "N" or "S"
    work["EMISSOR"] = work["EMISSOR"].map(_blank_single_letter)

    group_col = "TP_APLIC" if "TP_APLIC" in work.columns else "TP_ATIVO"

    grp = (
        work.groupby([group_col, "EMISSOR"], as_index=False)["PCT"]
        .sum()
        .sort_values("PCT", ascending=False)
        .head(10)
    )

    # Ensure first column is named TP_APLIC in the output
    grp = grp.rename(columns={group_col: "TP_APLIC"})
    return grp.reset_index(drop=True)

def load_blc_snapshot(conn_path: str, cnpj: str) -> Tuple[pd.DataFrame, str, str]:
    conn = ensure_conn(conn_path)
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT denom_social, last_month_yyyymm FROM fundos_meta WHERE cnpj=?",
            (cnpj,),
        )
        meta = cur.fetchone()
        if not meta:
            return pd.DataFrame(), "", ""
        denom_social, yyyymm = meta
        df = pd.read_sql_query(
            "SELECT * FROM blc_data WHERE cnpj=?",
            con=conn,
            params=(cnpj,),
        )
        if "PCT_CARTEIRA" in df.columns:
            df["PCT_CARTEIRA"] = pd.to_numeric(
                df["PCT_CARTEIRA"], errors="coerce"
            ).fillna(0.0)
        return df, yyyymm, denom_social
    finally:
        conn.close()


# ---------------------- Line/Perf helpers ----------------------


def build_daily_returns_table(
    fund_daily: pd.Series, cdi_daily: pd.Series, ibov_daily: pd.Series
) -> pd.DataFrame:
    """
    Build table with daily and cumulative returns:

    Columns:
      - date
      - Fundo, CDI, IBOV        (daily returns)
      - fundo_cum, cdi_cum, ibov_cum (cumulative returns)

    Cumulative = (1 + daily).cumprod() - 1, by column.
    """
    cols = [
        "date",
        "Fundo",
        "CDI",
        "IBOV",
        "fundo_cum",
        "cdi_cum",
        "ibov_cum",
    ]
    if fund_daily is None or fund_daily.empty:
        return pd.DataFrame(columns=cols)

    # Determine common range
    min_dt = fund_daily.index.min()
    max_dt = fund_daily.index.max()
    if cdi_daily is not None and not cdi_daily.empty:
        min_dt = max(min_dt, cdi_daily.index.min())
        max_dt = min(max_dt, cdi_daily.index.max())
    if ibov_daily is not None and not ibov_daily.empty:
        min_dt = max(min_dt, ibov_daily.index.min())
        max_dt = min(max_dt, ibov_daily.index.max())

    def finalize(df: pd.DataFrame) -> pd.DataFrame:
        df = df.reset_index()
        # Robustly rename the first column (index) to 'date'
        df = df.rename(columns={df.columns[0]: "date"})
        # Ensure all expected columns exist
        for c in ["Fundo", "CDI", "IBOV", "fundo_cum", "cdi_cum", "ibov_cum"]:
            if c not in df.columns:
                df[c] = np.nan
        # Enforce column order
        df = df[cols]
        # Ensure dtype of date is datetime
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        return df

    # Fallback: just align on available index if date range is invalid
    if pd.isna(min_dt) or pd.isna(max_dt) or min_dt > max_dt:
        df = pd.DataFrame({"Fundo": fund_daily})
        if cdi_daily is not None and not cdi_daily.empty:
            df["CDI"] = cdi_daily
        if ibov_daily is not None and not ibov_daily.empty:
            df["IBOV"] = ibov_daily
        df = df.sort_index()
        # Cumulative returns
        if "Fundo" in df.columns:
            df["fundo_cum"] = (1.0 + df["Fundo"].fillna(0.0)).cumprod() - 1.0
        if "CDI" in df.columns:
            df["cdi_cum"] = (1.0 + df["CDI"].fillna(0.0)).cumprod() - 1.0
        if "IBOV" in df.columns:
            df["ibov_cum"] = (1.0 + df["IBOV"].fillna(0.0)).cumprod() - 1.0
        return finalize(df)

    # Align on common date range
    f = fund_daily[(fund_daily.index >= min_dt) & (fund_daily.index <= max_dt)]
    c = (
        cdi_daily[(cdi_daily.index >= min_dt) & (cdi_daily.index <= max_dt)]
        if cdi_daily is not None
        else None
    )
    b = (
        ibov_daily[(ibov_daily.index >= min_dt) & (ibov_daily.index <= max_dt)]
        if ibov_daily is not None
        else None
    )

    df = pd.DataFrame({"Fundo": f})
    if c is not None:
        df["CDI"] = c.reindex(df.index)
    if b is not None:
        df["IBOV"] = b.reindex(df.index)

    df = df.sort_index()

    # Cumulative returns
    if "Fundo" in df.columns:
        df["fundo_cum"] = (1.0 + df["Fundo"].fillna(0.0)).cumprod() - 1.0
    if "CDI" in df.columns:
        df["cdi_cum"] = (1.0 + df["CDI"].fillna(0.0)).cumprod() - 1.0
    if "IBOV" in df.columns:
        df["ibov_cum"] = (1.0 + df["IBOV"].fillna(0.0)).cumprod() - 1.0

    return finalize(df)


def _window_subset(
    s: pd.Series, last_dt: pd.Timestamp, *, months: Optional[int] = None, ytd: bool = False
) -> pd.Series:
    if s is None or s.empty:
        return pd.Series(dtype=float)
    if ytd:
        start = pd.Timestamp(year=last_dt.year, month=1, day=1)
    else:
        if months is None:
            return pd.Series(dtype=float)
        start = last_dt - pd.DateOffset(months=months)
    sub = s[(s.index >= start) & (s.index <= last_dt)].dropna()
    return sub


def _cum_return_window(
    s: pd.Series, last_dt: pd.Timestamp, *, months: Optional[int] = None, ytd: bool = False
) -> float:
    sub = _window_subset(s, last_dt, months=months, ytd=ytd)
    if sub.empty:
        return float("nan")
    return float(np.prod(1.0 + sub.values) - 1.0)


def _vol_window(
    s: pd.Series, last_dt: pd.Timestamp, *, months: Optional[int] = None, ytd: bool = False
) -> float:
    sub = _window_subset(s, last_dt, months=months, ytd=ytd)
    if len(sub) < 2 or sub.std(ddof=0) == 0:
        return float("nan")
    return float(sub.std(ddof=0) * np.sqrt(252.0))


def _sharpe_window(
    fund: pd.Series,
    rf: Optional[pd.Series],
    last_dt: pd.Timestamp,
    *,
    months: Optional[int] = None,
    ytd: bool = False,
) -> float:
    sub_f = _window_subset(fund, last_dt, months=months, ytd=ytd)
    if sub_f.empty:
        return float("nan")
    if rf is not None and not rf.empty:
        sub_rf = _window_subset(rf, last_dt, months=months, ytd=ytd)
        exc = sub_f.reindex(sub_rf.index).dropna() - sub_rf.dropna()
        exc = exc.dropna()
    else:
        exc = sub_f
    if len(exc) < 2 or exc.std(ddof=0) == 0:
        return float("nan")
    return float((exc.mean() / exc.std(ddof=0)) * np.sqrt(252.0))


def build_perf_table_stats(
    fund_daily: pd.Series,
    cdi_daily: pd.Series,
    ibov_daily: pd.Series,
    benchmark: str = "CDI",
) -> pd.DataFrame:
    """
    Build perf_table with layout:
    | SÉRIE | YTD | 3M | 6M | 12M | 24M | 36M |
    Rows: FUNDO, BENCH (CDI or IBOV), RELATIVE (% CDI or DIFERENÇA P.P.),
          VOL (fund), SHARPE (fund, rf=CDI)
    Returns are cumulative; VOL is annualized std; SHARPE is annualized with
    rf=CDI.
    """
    if fund_daily is None or fund_daily.empty:
        return pd.DataFrame(
            columns=["Série", "YTD", "3M", "6M", "12M", "24M", "36M"]
        )

    last_dt = fund_daily.index.max()
    for s in [cdi_daily, ibov_daily]:
        if s is not None and not s.empty:
            last_dt = min(last_dt, s.index.max())

    cols = ["Série", "YTD", "3M", "6M", "12M", "24M", "36M"]

    def row_vals(series_name: str, values: Dict[str, float]) -> Dict[str, float]:
        r = {"Série": series_name}
        r.update(values)
        return r

    def bench_series() -> Tuple[str, pd.Series]:
        bmk = (benchmark or "CDI").upper()
        if bmk == "IBOV":
            return "IBOV", ibov_daily
        return "CDI", cdi_daily

    fund_vals = {
        "YTD": _cum_return_window(fund_daily, last_dt, ytd=True),
        "3M": _cum_return_window(fund_daily, last_dt, months=3),
        "6M": _cum_return_window(fund_daily, last_dt, months=6),
        "12M": _cum_return_window(fund_daily, last_dt, months=12),
        "24M": _cum_return_window(fund_daily, last_dt, months=24),
        "36M": _cum_return_window(fund_daily, last_dt, months=36),
    }

    bench_name, bench_daily = bench_series()
    if bench_daily is not None and not bench_daily.empty:
        bench_vals = {
            "YTD": _cum_return_window(bench_daily, last_dt, ytd=True),
            "3M": _cum_return_window(bench_daily, last_dt, months=3),
            "6M": _cum_return_window(bench_daily, last_dt, months=6),
            "12M": _cum_return_window(bench_daily, last_dt, months=12),
            "24M": _cum_return_window(bench_daily, last_dt, months=24),
            "36M": _cum_return_window(bench_daily, last_dt, months=36),
        }
    else:
        bench_vals = {k: float("nan") for k in ["YTD", "3M", "6M", "12M", "24M", "36M"]}

    # Relative row
    rel_name = "% CDI" if bench_name == "CDI" else "DIFERENÇA (P.P.)"
    rel_vals: Dict[str, float] = {}
    for k in ["YTD", "3M", "6M", "12M", "24M", "36M"]:
        fr = fund_vals[k]
        br = bench_vals[k]
        if pd.isna(fr) or pd.isna(br):
            rel = float("nan")
        else:
            if bench_name == "CDI":
                rel = fr / br if br != 0 else float("nan")
            else:
                rel = fr - br
        rel_vals[k] = float(rel)

    # Vol and Sharpe (rf=CDI)
    vol_vals = {
        "YTD": _vol_window(fund_daily, last_dt, ytd=True),
        "3M": _vol_window(fund_daily, last_dt, months=3),
        "6M": _vol_window(fund_daily, last_dt, months=6),
        "12M": _vol_window(fund_daily, last_dt, months=12),
        "24M": _vol_window(fund_daily, last_dt, months=24),
        "36M": _vol_window(fund_daily, last_dt, months=36),
    }
    rf_for_sharpe = cdi_daily
    sharpe_vals = {
        "YTD": _sharpe_window(fund_daily, rf_for_sharpe, last_dt, ytd=True),
        "3M": _sharpe_window(fund_daily, rf_for_sharpe, last_dt, months=3),
        "6M": _sharpe_window(fund_daily, rf_for_sharpe, last_dt, months=6),
        "12M": _sharpe_window(fund_daily, rf_for_sharpe, last_dt, months=12),
        "24M": _sharpe_window(fund_daily, rf_for_sharpe, last_dt, months=24),
        "36M": _sharpe_window(fund_daily, rf_for_sharpe, last_dt, months=36),
    }

    rows = [
        row_vals("FUNDO", fund_vals),
        row_vals(bench_name, bench_vals),
        row_vals(rel_name, rel_vals),
        row_vals("VOL", vol_vals),
        row_vals("SHARPE", sharpe_vals),
    ]
    df = pd.DataFrame(rows, columns=cols)
    return df


# ---------------------- Styling writer ----------------------

FONT_NAME = "Century Gothic"
LIGHT_GRAY = "#efefef"


def _to_upper(x):
    return x.upper() if isinstance(x, str) else x


def _px_to_char(px: int) -> int:
    # Rough Excel mapping: ~7 px per character (varies by font/DPI)
    return max(1, round(px / 7))


def _px_to_points(px: int) -> float:
    # Excel row height is in points; at ~96 DPI, 1 px ≈ 0.75 pt
    return round(px * 0.75, 2)


def write_table_zebra(
    writer: pd.ExcelWriter,
    df: pd.DataFrame,
    sheet: str,
    *,
    percent_cols: Optional[List[str]] = None,
    date_cols: Optional[List[str]] = None,
) -> None:
    """
    White sheet; centered bold header (no fill).
    Odd data rows (after header) shaded light gray via conditional formatting.
    No gridlines. No freeze panes.
    """
    df = df.copy()
    df.to_excel(writer, sheet_name=sheet, index=False, header=False, startrow=1)
    wb = writer.book
    ws = writer.sheets[sheet]

    ws.hide_gridlines(2)

    header_fmt = wb.add_format(
        {
            "bold": True,
            "font_name": FONT_NAME,
            "font_size": 11,
            "align": "center",
            "valign": "vcenter",
        }
    )
    base_fmt = wb.add_format({"font_name": FONT_NAME, "font_size": 11})
    pct_fmt = wb.add_format({"num_format": "0.00%", "font_name": FONT_NAME})
    date_fmt = wb.add_format({"num_format": "yyyy-mm-dd", "font_name": FONT_NAME})
    zebra_fill = wb.add_format({"bg_color": LIGHT_GRAY})

    n_rows, n_cols = df.shape

    # Header
    for c, name in enumerate(df.columns):
        ws.write(0, c, name, header_fmt)

    # Column widths and default formats
    for c, name in enumerate(df.columns):
        col_fmt = base_fmt
        if percent_cols and name in percent_cols:
            col_fmt = pct_fmt
        elif date_cols and name in date_cols:
            col_fmt = date_fmt
        ws.set_column(c, c, 16, col_fmt)

    # Zebra on odd data rows (after header) using LIGHT_GRAY
    if n_rows > 0 and n_cols > 0:
        ws.conditional_format(
            1,
            0,
            n_rows,
            n_cols - 1,
            {
                "type": "formula",
                "criteria": "=MOD(ROW()-1,2)=1",
                "format": zebra_fill,
            },
        )

    ws.set_landscape()
    ws.fit_to_pages(1, 0)
    ws.set_margins(0.3, 0.3, 0.5, 0.5)


# ---------------------- Workbook builder ----------------------


def create_xlsx_bytes_for_fund(
    *,
    denom: str,
    cnpj: str,
    yyyymm: str,
    blc_df: pd.DataFrame,
    nav_df: pd.DataFrame,
    cdi_daily: pd.Series,
    ibov_daily: pd.Series,
    benchmark: str,
    include_raw_blc: bool,
) -> bytes:
    # Compute fund daily returns
    fund_daily = daily_returns_from_nav(nav_df)

    # Build tables (rename PCT -> %)
    alloc_df = get_alloc_donut(blc_df).rename(columns={"PCT": "%"})
    top_df = get_top_positions(blc_df).rename(columns={"PCT": "%"})
    line_df = build_daily_returns_table(fund_daily, cdi_daily, ibov_daily)
    perf_df = build_perf_table_stats(fund_daily, cdi_daily, ibov_daily, benchmark)

    # Meta (uppercase values; special cells white font with blank background)
    meta_df = pd.DataFrame(
        [
            ["Fundo", denom],
            ["CNPJ", format_cnpj(cnpj)],
            ["Snapshot BLC (yyyymm)", yyyymm],
            [
                "Série INF_DIARIO min/max",
                f"{nav_df['DT_COMPTC'].min().date()} / "
                f"{nav_df['DT_COMPTC'].max().date()}",
            ],
            ["Comparador", benchmark],
            ["Export UTC", pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S")],
        ],
        columns=["Campo", "Valor"],
    )

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as xw:
        # -------- meta --------
        meta_up = meta_df.copy()
        meta_up["Campo"] = meta_up["Campo"].apply(_to_upper)
        meta_up["Valor"] = meta_up["Valor"].apply(_to_upper)
        meta_up.to_excel(xw, sheet_name="meta", index=False, header=False)
        ws = xw.sheets["meta"]
        wb = xw.book
        ws.hide_gridlines(2)

        cg11_black = wb.add_format({"font_name": FONT_NAME, "font_size": 11})
        white_bold_14 = wb.add_format(
            {
                "font_name": FONT_NAME,
                "font_size": 14,
                "bold": True,
                "font_color": "white",
            }
        )
        white_8 = wb.add_format(
            {"font_name": FONT_NAME, "font_size": 8, "font_color": "white"}
        )
        white_10 = wb.add_format(
            {"font_name": FONT_NAME, "font_size": 10, "font_color": "white"}
        )

        for i in range(len(meta_up)):
            ws.write(i, 0, meta_up.iloc[i, 0], cg11_black)
            ws.write(i, 1, meta_up.iloc[i, 1], cg11_black)
        # Specific cells: B1, B2, B4 (white text, blank background)
        if len(meta_up) >= 1:
            ws.write(0, 1, meta_up.iloc[0, 1], white_bold_14)
        if len(meta_up) >= 2:
            ws.write(1, 1, meta_up.iloc[1, 1], white_8)
        if len(meta_up) >= 4:
            ws.write(3, 1, meta_up.iloc[3, 1], white_10)

        ws.set_column(0, 0, 28, cg11_black)
        ws.set_column(1, 1, 64, cg11_black)
        ws.set_landscape()
        ws.fit_to_pages(1, 0)
        ws.set_margins(0.3, 0.3, 0.5, 0.5)

        # -------- line_index (daily + cumulative returns) --------
        write_table_zebra(
            xw,
            line_df,
            "line_index",
            percent_cols=[
                "Fundo",
                "CDI",
                "IBOV",
                "fundo_cum",
                "cdi_cum",
                "ibov_cum",
            ],
            date_cols=["date"] if "date" in line_df.columns else [],
        )

        # -------- perf_table (custom layout, uppercase, CG size 10, centered)
        headers = ["SÉRIE", "YTD", "3M", "6M", "12M", "24M", "36M"]
        perf_up = perf_df.copy()
        perf_up.columns = headers  # enforce order/labels
        perf_up["SÉRIE"] = perf_up["SÉRIE"].apply(_to_upper)

        ws = wb.add_worksheet("perf_table")
        xw.sheets["perf_table"] = ws
        ws.hide_gridlines(2)

        cg10_center = wb.add_format(
            {
                "font_name": FONT_NAME,
                "font_size": 10,
                "align": "center",
                "valign": "vcenter",
            }
        )
        cg10_center_pct = wb.add_format(
            {
                "font_name": FONT_NAME,
                "font_size": 10,
                "align": "center",
                "valign": "vcenter",
                "num_format": "0.00%",
            }
        )
        cg10_center_num = wb.add_format(
            {
                "font_name": FONT_NAME,
                "font_size": 10,
                "align": "center",
                "valign": "vcenter",
                "num_format": "0.00",
            }
        )
        header_white = wb.add_format(
            {
                "font_name": FONT_NAME,
                "font_size": 10,
                "bold": True,
                "font_color": "white",
                "align": "center",
                "valign": "vcenter",
            }
        )

        # Header row (white text, blank background)
        for c, name in enumerate(headers):
            ws.write(0, c, name, header_white)

        # Column widths ~100 px => ~14 char
        col_w = _px_to_char(100)
        for c in range(len(headers)):
            ws.set_column(c, c, col_w, cg10_center)

        # Data rows (VOL and DIFERENÇA rows as %, SHARPE as number)
        for r in range(len(perf_up)):
            label = str(perf_up.iloc[r, 0]).upper()
            ws.write(r + 1, 0, label, cg10_center)
            is_sharpe = label == "SHARPE"
            for c, col in enumerate(headers[1:], start=1):
                val = perf_up.iloc[r][col]
                fmt = cg10_center_num if is_sharpe else cg10_center_pct
                ws.write(r + 1, c, val, fmt)

        # Zebra striping: first data row blank, second shaded (#F2F2F2), etc.
        zebra_fill_f2 = wb.add_format({"bg_color": "#F2F2F2"})
        if len(perf_up) > 0:
            ws.conditional_format(
                1,  # first data row
                0,
                len(perf_up),  # last data row
                len(headers) - 1,
                {
                    "type": "formula",
                    "criteria": "=MOD(ROW()-2,2)=1",
                    "format": zebra_fill_f2,
                },
            )

        ws.set_default_row(_px_to_points(23))
        ws.set_landscape()
        ws.fit_to_pages(1, 0)
        ws.set_margins(0.3, 0.3, 0.5, 0.5)

        # -------- alloc_donut (zebra style) --------
        write_table_zebra(
            xw, alloc_df, "alloc_donut", percent_cols=["%"], date_cols=[]
        )

        # -------- top_positions (uppercase, widths/heights) --------
        top_up = top_df.copy()
        top_up.columns = [
            _to_upper(c) if isinstance(c, str) else c for c in top_up.columns
        ]
        for col in top_up.columns:
            if top_up[col].dtype == "object":
                top_up[col] = top_up[col].apply(_to_upper)
        top_up.to_excel(
            xw, sheet_name="top_positions", index=False, header=False, startrow=1
        )
        ws = xw.sheets["top_positions"]
        wb = xw.book
        ws.hide_gridlines(2)

        cg10_left = wb.add_format(
            {
                "font_name": FONT_NAME,
                "font_size": 10,
                "align": "left",
                "valign": "vcenter",
            }
        )
        cg10_center_pct2 = wb.add_format(
            {
                "font_name": FONT_NAME,
                "font_size": 10,
                "align": "center",
                "valign": "vcenter",
                "num_format": "0.00%",
            }
        )
        header_white2 = wb.add_format(
            {
                "font_name": FONT_NAME,
                "font_size": 10,
                "bold": True,
                "font_color": "white",
                "align": "center",
                "valign": "vcenter",
            }
        )

        # Header (white bold, blank background)
        for c, name in enumerate(top_up.columns):
            ws.write(0, c, name, header_white2)

        widths_px = [187, 320, 100]
        for c in range(len(top_up.columns)):
            width_char = _px_to_char(widths_px[c] if c < len(widths_px) else 100)
            if top_up.columns[c] == "%":
                ws.set_column(c, c, width_char, cg10_center_pct2)
            else:
                ws.set_column(c, c, width_char, cg10_left)

        # Zebra striping for top_positions data rows (first blank, second shaded)
        n_rows = len(top_up)
        n_cols = len(top_up.columns)
        if n_rows > 0 and n_cols > 0:
            ws.conditional_format(
                1,
                0,
                n_rows,
                n_cols - 1,
                {
                    "type": "formula",
                    "criteria": "=MOD(ROW()-2,2)=1",
                    "format": zebra_fill_f2,
                },
            )

        ws.set_default_row(_px_to_points(22))
        ws.set_landscape()
        ws.fit_to_pages(1, 0)
        ws.set_margins(0.3, 0.3, 0.5, 0.5)

        # -------- raw blc (optional; zebra style) --------
        if include_raw_blc:
            write_table_zebra(xw, blc_df, "blc_snapshot")

    return buf.getvalue()


# ---------------------- Streamlit page ----------------------


def parse_pasted_benchmarks(text: str) -> Tuple[Dict[str, str], List[str]]:
    """
    Parse lines like:
      00.000.000/0000-00;CDI
      11222333000155;IBOV
    Returns (mapping, messages). `mapping` is {cnpj14: 'CDI'|'IBOV'}.
    `messages` holds warnings/errors to surface to the user.
    """
    mapping: Dict[str, str] = {}
    msgs: List[str] = []
    if not text:
        return mapping, msgs
    for i, raw in enumerate(text.splitlines(), start=1):
        line = (raw or "").strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split(";") if p.strip()]
        if len(parts) != 2:
            msgs.append(f"Linha {i}: formato inválido -> '{raw}'")
            continue
        cnpj_raw, bench_raw = parts
        cnpj = sanitize_cnpj(cnpj_raw)
        if len(cnpj) != 14:
            msgs.append(f"Linha {i}: CNPJ inválido '{cnpj_raw}' -> '{cnpj}'")
            continue
        bench = bench_raw.upper()
        if bench not in {"CDI", "IBOV"}:
            msgs.append(
                f"Linha {i}: benchmark inválido '{bench_raw}' (use CDI ou IBOV)"
            )
            continue
        if cnpj in mapping:
            msgs.append(
                f"Linha {i}: CNPJ duplicado {format_cnpj(cnpj)} — "
                "última ocorrência prevalece."
            )
        mapping[cnpj] = bench
    return mapping, msgs


def main() -> None:
    st.set_page_config(
        page_title="Export factsheet data — Century Gothic", layout="wide"
    )
    st.title("Exportar dados do factsheet — XLSX (Century Gothic, cabeçalho central)")

    c1, c2, c3 = st.columns([2, 2, 2])

    with c1:
        carteira_db = st.text_input(
            "SQLite carteira_fundos.db",
            value=os.path.abspath("./databases/carteira_fundos.db"),
        )
        perf_db = st.text_input(
            "SQLite data_fundos.db (INF_DIARIO)",
            value=os.path.abspath("./databases/data_fundos.db"),
        )
        bench_db = st.text_input(
            "Benchmarks DB (CDI/IBOV)",
            value=os.path.abspath("./databases/data_cdi_ibov.db"),
        )

    with c2:
        try:
            conn = ensure_conn(carteira_db)
            fundos_df = pd.read_sql_query(
                "SELECT cnpj, denom_social FROM fundos_meta ORDER BY denom_social",
                conn,
            )
            conn.close()
        except Exception:
            fundos_df = pd.DataFrame()

        if fundos_df.empty:
            st.error("Nenhum fundo encontrado em fundos_meta.")
            return

        # Fund selection for single export
        cnpj_sel = st.selectbox(
            "Selecione o CNPJ para exportação individual",
            options=fundos_df["cnpj"].tolist(),
            format_func=lambda c: (
                f"{fundos_df.loc[fundos_df['cnpj']==c, 'denom_social'].iloc[0]} ({c})"
            ),
        )

        # Choose which funds will use IBOV as benchmark (default CDI)
        options_labels = [
            f"{row['denom_social']} ({row['cnpj']})" for _, row in fundos_df.iterrows()
        ]
        label_to_cnpj = {
            f"{row['denom_social']} ({row['cnpj']})": row["cnpj"]
            for _, row in fundos_df.iterrows()
        }
        ibov_labels = st.multiselect(
            "Fundos que usarão IBOV como benchmark na perf_table "
            "(os demais usarão CDI):",
            options=options_labels,
            default=[],
        )
        ibov_cnpjs = {label_to_cnpj[lbl] for lbl in ibov_labels}

    with c3:
        include_raw_blc = st.checkbox("Incluir aba 'blc_snapshot'", value=False)
        st.caption(
            "Nota: VOL e SHARPE usam CDI como taxa livre de risco (independente "
            "do benchmark selecionado para o retorno relativo)."
        )

    st.divider()

    # Preload benchmarks once
    cdi_daily = load_benchmark_daily([perf_db, bench_db], "cdi")
    ibov_daily = load_benchmark_daily([perf_db, bench_db], "ibov")

    colA, colB = st.columns([1, 1])
    with colA:
        do_single = st.button("Gerar XLSX (fundo selecionado)")
    with colB:
        do_zip = st.button("Gerar ZIP (todos os fundos)")

    st.markdown("")
    with st.expander("Gerar ZIP a partir de lista colada de CNPJs e benchmarks"):
        st.caption(
            "Cole um CNPJ por linha, no formato CNPJ;CDI ou CNPJ;IBOV. "
            "Linhas em branco e linhas iniciadas com '#' serão ignoradas."
        )
        paste_text = st.text_area(
            "Lista (um por linha)",
            height=160,
            placeholder=(
                "00.000.000/0000-00;CDI\n"
                "11.222.333/0001-55;IBOV\n"
                "# Comentários são permitidos\n"
                "11222333000155;CDI"
            ),
        )
        do_zip_pasted = st.button("Gerar ZIP (lista colada)")

    if do_single:
        # Load snapshots for selected fund
        blc_df, yyyymm, denom = load_blc_snapshot(carteira_db, cnpj_sel)
        if blc_df.empty:
            st.error("BLC do fundo não encontrada no DB de carteiras.")
            return
        nav_df = load_nav_series(perf_db, cnpj_sel)
        if nav_df.empty:
            st.error("Série INF_DIARIO (VL_QUOTA) não encontrada no DB.")
            return

        benchmark = "IBOV" if cnpj_sel in ibov_cnpjs else "CDI"
        xlsx_bytes = create_xlsx_bytes_for_fund(
            denom=denom,
            cnpj=cnpj_sel,
            yyyymm=yyyymm,
            blc_df=blc_df,
            nav_df=nav_df,
            cdi_daily=cdi_daily,
            ibov_daily=ibov_daily,
            benchmark=benchmark,
            include_raw_blc=include_raw_blc,
        )

        st.success(
            f"Planilha gerada para {denom} "
            f"({format_cnpj(cnpj_sel)}) — Benchmark: {benchmark}."
        )
        st.download_button(
            "Baixar XLSX",
            data=xlsx_bytes,
            file_name=f"{sanitize_cnpj(cnpj_sel)}.xlsx",
            mime=("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
        )
        return

    if do_zip:
        zip_buf = io.BytesIO()
        created = 0
        skipped = []

        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for _, row in fundos_df.iterrows():
                cnpj = row["cnpj"]
                denom = row["denom_social"]
                try:
                    blc_df, yyyymm, _denom2 = load_blc_snapshot(carteira_db, cnpj)
                    if blc_df.empty:
                        skipped.append((cnpj, "sem BLC"))
                        continue
                    nav_df = load_nav_series(perf_db, cnpj)
                    if nav_df.empty:
                        skipped.append((cnpj, "sem INF_DIARIO"))
                        continue
                    benchmark = "IBOV" if cnpj in ibov_cnpjs else "CDI"
                    xls_bytes = create_xlsx_bytes_for_fund(
                        denom=denom,
                        cnpj=cnpj,
                        yyyymm=yyyymm,
                        blc_df=blc_df,
                        nav_df=nav_df,
                        cdi_daily=cdi_daily,
                        ibov_daily=ibov_daily,
                        benchmark=benchmark,
                        include_raw_blc=include_raw_blc,
                    )
                    fname = f"{sanitize_cnpj(cnpj)}.xlsx"
                    zf.writestr(fname, xls_bytes)
                    created += 1
                except Exception as e:
                    skipped.append((cnpj, f"erro: {e}"))

        if created == 0:
            st.error("Nenhum arquivo gerado para o ZIP.")
            if skipped:
                st.write("Ignorados:", skipped)
            return

        st.success(f"ZIP gerado com {created} arquivos.")
        if skipped:
            st.caption(
                "Alguns fundos foram ignorados (sem dados ou erro). "
                "Veja a lista abaixo."
            )
            st.write(skipped)

        st.download_button(
            "Baixar ZIP (todos os fundos)",
            data=zip_buf.getvalue(),
            file_name="factsheets.zip",
            mime="application/zip",
        )
        return

    if do_zip_pasted:
        mapping, msgs = parse_pasted_benchmarks(paste_text)
        if msgs:
            st.warning("Avisos/erros de parsing:")
            st.write(msgs)
        if not mapping:
            st.error("Nenhuma linha válida foi encontrada.")
            return

        # Check existence in fundos_meta
        known_cnpjs = set(fundos_df["cnpj"].astype(str))
        denom_by_cnpj = {
            str(row["cnpj"]): row["denom_social"] for _, row in fundos_df.iterrows()
        }
        unknown = [c for c in mapping.keys() if c not in known_cnpjs]
        if unknown:
            st.info("Alguns CNPJs não existem em fundos_meta e serão ignorados.")
            st.write([format_cnpj(c) for c in unknown])

        zip_buf = io.BytesIO()
        created = 0
        skipped: List[Tuple[str, str]] = []

        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for cnpj, benchmark in mapping.items():
                if cnpj not in known_cnpjs:
                    skipped.append((cnpj, "CNPJ não encontrado em fundos_meta"))
                    continue
                try:
                    blc_df, yyyymm, denom2 = load_blc_snapshot(carteira_db, cnpj)
                    if blc_df.empty:
                        skipped.append((cnpj, "sem BLC"))
                        continue
                    nav_df = load_nav_series(perf_db, cnpj)
                    if nav_df.empty:
                        skipped.append((cnpj, "sem INF_DIARIO"))
                        continue
                    denom = denom2 or denom_by_cnpj.get(cnpj, cnpj)
                    xls_bytes = create_xlsx_bytes_for_fund(
                        denom=denom,
                        cnpj=cnpj,
                        yyyymm=yyyymm,
                        blc_df=blc_df,
                        nav_df=nav_df,
                        cdi_daily=cdi_daily,
                        ibov_daily=ibov_daily,
                        benchmark=benchmark,
                        include_raw_blc=include_raw_blc,
                    )
                    fname = f"{sanitize_cnpj(cnpj)}.xlsx"
                    zf.writestr(fname, xls_bytes)
                    created += 1
                except Exception as e:
                    skipped.append((cnpj, f"erro: {e}"))

        if created == 0:
            st.error("Nenhum arquivo gerado para o ZIP da lista colada.")
            if skipped:
                st.write("Ignorados:", [(format_cnpj(c), r) for c, r in skipped])
            return

        st.success(f"ZIP (lista colada) gerado com {created} arquivos.")
        if skipped:
            st.caption(
                "Alguns fundos foram ignorados (sem dados, fora do cadastro, "
                "ou erro). Veja a lista abaixo."
            )
            st.write([(format_cnpj(c), r) for c, r in skipped])

        st.download_button(
            "Baixar ZIP (lista colada)",
            data=zip_buf.getvalue(),
            file_name="factsheets_lista_colada.zip",
            mime="application/zip",
        )
        return


if __name__ == "__main__":
    main()