# pages/05_Export_Factsheet_Data.py
# RAW factsheet data -> XLSX (clean zebra style)
# - Headers centered
# - Font: Montserrat
# - CNPJ in meta formatted as 00.000.000/0000-00
# - White sheets, bold header (no fill), odd data rows shaded light gray
# - No gridlines, no freeze panes
# - All "PCT" columns renamed to "%"

from __future__ import annotations

import io
import os
import re
import sqlite3
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

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
                    q = f"SELECT {date_col} as dt, daily_return as ret FROM {table} ORDER BY {date_col}"
                    df = pd.read_sql_query(q, conn, parse_dates=["dt"])
                    s = pd.to_numeric(df["ret"], errors="coerce").dropna()
                    s.index = df["dt"]
                    if not s.empty:
                        return s
                if has_dr_pct:
                    q = f"SELECT {date_col} as dt, daily_rate_pct as ret FROM {table} ORDER BY {date_col}"
                    df = pd.read_sql_query(q, conn, parse_dates=["dt"])
                    s = pd.to_numeric(df["ret"], errors="coerce").dropna() / 100.0
                    s.index = df["dt"]
                    if not s.empty:
                        return s
            else:
                if has_dr:
                    q = f"SELECT {date_col} as dt, daily_return as ret FROM {table} ORDER BY {date_col}"
                    df = pd.read_sql_query(q, conn, parse_dates=["dt"])
                    s = pd.to_numeric(df["ret"], errors="coerce").dropna()
                    s.index = df["dt"]
                    if not s.empty:
                        return s
                if price_col:
                    q = f"SELECT {date_col} as dt, {price_col} as px FROM {table} ORDER BY {date_col}"
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
    if pd.isna(tp_ativo) or not str(tp_ativo).strip():
        return "Outros"
    s = str(tp_ativo).upper()
    if any(kw in s for kw in ["CDB", "RDB", "LETRA FINANCEIRA", "LF", "LCR"]):
        return "Depósitos a prazo e outros títulos de IF"
    if any(kw in s for kw in ["TÍTULO PÚBLICO", "NTN", "LTN", "LFT", "TESOURO"]):
        return "TÍTULOS PÚBLICOS"
    if "COMPROMISSADA" in s:
        return "OPERAÇÕES COMPROMISSADAS"
    if "AÇÃO" in s:
        return "AÇÕES"
    if "FUNDO" in s:
        return "FUNDO DE INVESTIMENTO"
    if "IMOBILIÁRIO" in s or "FII" in s:
        return "IMÓVEIS"
    return "Outros"


def get_alloc_donut(df_blc: pd.DataFrame) -> pd.DataFrame:
    if df_blc.empty:
        return pd.DataFrame(columns=["CATEGORIA", "PCT"])
    if "PCT_CARTEIRA" not in df_blc.columns or "TP_ATIVO" not in df_blc.columns:
        return pd.DataFrame(columns=["CATEGORIA", "PCT"])
    tmp = df_blc.copy()
    tmp["CATEGORIA"] = tmp["TP_ATIVO"].apply(categorize_tp_ativo)
    agg = (
        tmp.groupby("CATEGORIA")["PCT_CARTEIRA"]
        .sum()
        .reset_index()
        .rename(columns={"PCT_CARTEIRA": "PCT"})
        .sort_values("PCT", ascending=False)
    )
    if len(agg) > 6:
        top = agg.head(5)
        rest = pd.DataFrame(
            [["Outros", float(agg["PCT"].iloc[5:].sum())]],
            columns=["CATEGORIA", "PCT"],
        )
        agg = pd.concat([top, rest], ignore_index=True)
    return agg.reset_index(drop=True)


def get_top_positions(df_blc: pd.DataFrame) -> pd.DataFrame:
    if df_blc.empty or "PCT_CARTEIRA" not in df_blc.columns:
        return pd.DataFrame(columns=["ATIVO", "EMISSOR", "PCT"])
    work = df_blc.copy()
    if "EMISSOR" not in work.columns:
        work["EMISSOR"] = work.get("EMISSOR_LIGADO", "N/A")
    if "TP_ATIVO" not in work.columns:
        work["TP_ATIVO"] = "N/A"
    work["PCT"] = work["PCT_CARTEIRA"]
    grp = (
        work.groupby(["TP_ATIVO", "EMISSOR"])["PCT"]
        .sum()
        .reset_index()
        .sort_values("PCT", ascending=False)
        .head(10)
    )
    grp = grp.rename(columns={"TP_ATIVO": "ATIVO"})
    return grp.reset_index(drop=True)


def build_line_index(fund_daily: pd.Series, bench_daily: Optional[pd.Series]) -> pd.DataFrame:
    series_map: Dict[str, pd.Series] = {"Fundo": fund_daily}
    if bench_daily is not None and not bench_daily.empty:
        series_map["Benchmark"] = bench_daily

    starts = [s.index.min() for s in series_map.values()]
    ends = [s.index.max() for s in series_map.values()]
    start_common = max(starts)
    end_common = min(ends)

    out = pd.DataFrame(index=pd.date_range(start_common, end_common, freq="D"))
    for k, v in series_map.items():
        s = v[(v.index >= start_common) & (v.index <= end_common)]
        out[k] = (1.0 + s).cumprod().reindex(out.index)

    out = out.dropna(how="all").reset_index().rename(columns={"index": "date"})
    return out


def build_perf_table(
    fund_daily: pd.Series, bench_daily: Optional[pd.Series], bench_name: str
) -> pd.DataFrame:
    fund_m = monthly_from_daily(fund_daily)
    ytd_fund = ytd_from_daily(fund_daily)
    r3y_fund = float(np.prod(1.0 + fund_m.tail(36).values) - 1.0) if len(fund_m) >= 36 else np.nan
    years_f = annual_from_monthly(fund_m)
    all_years = sorted(years_f.index.tolist(), reverse=True)
    years_cols = all_years[:5]

    def row_fmt(name: str, ytd: float, r3: float, yrs: pd.Series) -> Dict[str, float]:
        rec = {"Série": name, "YTD": ytd, "3Y": r3}
        for y in years_cols:
            rec[str(y)] = float(yrs.get(y, np.nan))
        return rec

    rows = [row_fmt("Fundo", ytd_fund, r3y_fund, years_f)]

    if bench_daily is not None and not bench_daily.empty and bench_name:
        b_m = monthly_from_daily(bench_daily)
        ytd_b = ytd_from_daily(bench_daily)
        r3y_b = float(np.prod(1.0 + b_m.tail(36).values) - 1.0) if len(b_m) >= 36 else np.nan
        years_b = annual_from_monthly(b_m)
        rows.append(row_fmt(bench_name, ytd_b, r3y_b, years_b))

        # One derived row only
        if bench_name.upper() == "CDI":
            ytd_ratio = ytd_fund / ytd_b if (pd.notna(ytd_fund) and ytd_b) else np.nan
            r3y_ratio = r3y_fund / r3y_b if (pd.notna(r3y_fund) and r3y_b) else np.nan
            yrs_ratio = (years_f / years_b).replace([np.inf, -np.inf], np.nan)
            rows.append(row_fmt("% CDI", ytd_ratio, r3y_ratio, yrs_ratio))
        elif bench_name.upper() == "IBOV":
            ytd_diff = ytd_fund - ytd_b if (pd.notna(ytd_fund) and pd.notna(ytd_b)) else np.nan
            r3y_diff = r3y_fund - r3y_b if (pd.notna(r3y_fund) and pd.notna(r3y_b)) else np.nan
            yrs_diff = (years_f - years_b).dropna()
            rows.append(row_fmt("Diferença (p.p.)", ytd_diff, r3y_diff, yrs_diff))

    df = pd.DataFrame(rows)
    if "Série" in df.columns:
        df = df.drop_duplicates(subset=["Série"], keep="first")
    return df


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


# ---------------------- Styling writer ----------------------

FONT_NAME = "Montserrat"  # requires the font to be installed in the system
CERES_BRONZE = "#8c6239"
LIGHT_GRAY = "#efefef"


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

    # Zebra on odd data rows (after header)
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


# ---------------------- Streamlit page ----------------------

def main() -> None:
    st.set_page_config(page_title="Export factsheet data — Montserrat", layout="wide")
    st.title("Exportar dados do factsheet — XLSX (Montserrat, cabeçalho central)")

    c1, c2, c3 = st.columns([2, 2, 2])

    with c1:
        carteira_db = st.text_input(
            "SQLite carteira_fundos.db", value=os.path.abspath("./carteira_fundos.db")
        )
        perf_db = st.text_input(
            "SQLite data_fundos.db (INF_DIARIO)", value=os.path.abspath("./data/data_fundos.db")
        )
        bench_db = st.text_input(
            "Benchmarks DB (CDI/IBOV)", value=os.path.abspath("./data/data_cdi_ibov.db")
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

        cnpj_sel = st.selectbox(
            "Selecione o CNPJ",
            options=fundos_df["cnpj"].tolist(),
            format_func=lambda c: (
                f"{fundos_df.loc[fundos_df['cnpj']==c, 'denom_social'].iloc[0]} ({c})"
            ),
        )

        comparator = st.selectbox("Comparador (opcional)", ["Nenhum", "CDI", "IBOV"], index=1)

    with c3:
        include_raw_blc = st.checkbox("Incluir aba 'blc_snapshot'", value=False)

    st.divider()
    if not st.button("Gerar XLSX"):
        return

    # Load snapshots
    blc_df, yyyymm, denom = load_blc_snapshot(carteira_db, cnpj_sel)
    if blc_df.empty:
        st.error("BLC do fundo não encontrada no DB de carteiras.")
        return
    nav_df = load_nav_series(perf_db, cnpj_sel)
    if nav_df.empty:
        st.error("Série INF_DIARIO (VL_QUOTA) não encontrada no DB.")
        return

    fund_daily = daily_returns_from_nav(nav_df)

    bench_name = ""
    bench_daily = None
    if comparator in ("CDI", "IBOV"):
        s = load_benchmark_daily([perf_db, bench_db], comparator.lower())
        if not s.empty:
            bench_daily = s
            bench_name = comparator

    # Build tables (rename PCT -> %)
    alloc_df = get_alloc_donut(blc_df).rename(columns={"PCT": "%"})
    top_df = get_top_positions(blc_df).rename(columns={"PCT": "%"})
    line_df = build_line_index(fund_daily, bench_daily)
    if bench_name:
        line_df = line_df.rename(columns={"Benchmark": bench_name})
    perf_df = build_perf_table(fund_daily, bench_daily, bench_name)

    # Meta
    meta_df = pd.DataFrame(
        [
            ["Fundo", denom],
            ["CNPJ", format_cnpj(cnpj_sel)],
            ["Snapshot BLC (yyyymm)", yyyymm],
            [
                "Série INF_DIARIO min/max",
                f"{nav_df['DT_COMPTC'].min().date()} / "
                f"{nav_df['DT_COMPTC'].max().date()}",
            ],
            ["Comparador", bench_name or "(nenhum)"],
            ["Export UTC", pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S")],
        ],
        columns=["Campo", "Valor"],
    )

    # Write XLSX
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as xw:
        # meta: only fund name bronze
        meta_df.to_excel(xw, sheet_name="meta", index=False, header=False)
        ws = xw.sheets["meta"]
        wb = xw.book
        ws.hide_gridlines(2)

        base_fmt = wb.add_format({"font_name": FONT_NAME, "font_size": 11})
        bronze_fmt = wb.add_format(
            {"bold": True, "font_color": CERES_BRONZE, "font_name": FONT_NAME, "font_size": 14}
        )

        ws.write(0, 0, "Fundo", base_fmt)
        ws.write(0, 1, meta_df.iloc[0, 1], bronze_fmt)
        for i in range(1, len(meta_df)):
            ws.write(i, 0, meta_df.iloc[i, 0], base_fmt)
            ws.write(i, 1, meta_df.iloc[i, 1], base_fmt)

        ws.set_column(0, 0, 28, base_fmt)
        ws.set_column(1, 1, 64, base_fmt)
        ws.set_landscape()
        ws.fit_to_pages(1, 0)
        ws.set_margins(0.3, 0.3, 0.5, 0.5)

        # line_index (headers centered)
        write_table_zebra(
            xw,
            line_df,
            "line_index",
            percent_cols=[],
            date_cols=["date"] if "date" in line_df.columns else [],
        )

        # perf_table (headers centered; all but 'Série' as %)
        write_table_zebra(
            xw,
            perf_df,
            "perf_table",
            percent_cols=[c for c in perf_df.columns if c != "Série"],
            date_cols=[],
        )

        # alloc_donut (%)
        write_table_zebra(xw, alloc_df, "alloc_donut", percent_cols=["%"], date_cols=[])

        # top_positions (%)
        write_table_zebra(xw, top_df, "top_positions", percent_cols=["%"], date_cols=[])

        # raw blc (optional)
        if include_raw_blc:
            write_table_zebra(xw, blc_df, "blc_snapshot")

    st.success("Planilha gerada.")
    st.download_button(
        "Baixar XLSX",
        data=buf.getvalue(),
        file_name=f"{sanitize_cnpj(cnpj_sel)}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


if __name__ == "__main__":
    main()