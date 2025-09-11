# pages/1_📥_Data_Downloader.py
from __future__ import annotations

import io
import re
from typing import List, Optional

import pandas as pd
import requests
import streamlit as st


st.set_page_config(page_title="Data Downloader", page_icon="📥", layout="wide")
st.title("📥 Data Downloader")
st.caption(
    "Fetch from Yahoo Finance, Banco Central (SGS), and CVM (INF_DIARIO_FI). "
    "Download as CSV and upload into Graph Studio."
)

# ----------------------- Minimal state init (per tab) ---------------------- #


def _init_state():
    for k in ["yf_df", "bcb_df", "cvm_df"]:
        if k not in st.session_state:
            st.session_state[k] = pd.DataFrame()


_init_state()

# ----------------------- Utils -------------------------------------------- #


def ensure_datetime(df: pd.DataFrame, col: str = "Date") -> pd.DataFrame:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce").dt.tz_localize(None)
    return df


def df_to_csv_bytes(
    df: pd.DataFrame,
    sep: str = ",",
    decimal: str = ".",
    with_bom: bool = True,
) -> bytes:
    csv_str = df.to_csv(index=False, sep=sep, decimal=decimal)
    enc = "utf-8-sig" if with_bom else "utf-8"
    return csv_str.encode(enc)


def render_csv_downloads(df: pd.DataFrame, base_name: str) -> None:
    if df.empty:
        return
    st.write("Downloads:")
    c1, c2 = st.columns(2)
    with c1:
        data_std = df_to_csv_bytes(df, sep=",", decimal=".", with_bom=True)
        st.download_button(
            "Download CSV (UTF-8 BOM, comma, decimal dot)",
            data=data_std,
            file_name=f"{base_name}.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with c2:
        data_excel = df_to_csv_bytes(df, sep=";", decimal=",", with_bom=True)
        st.download_button(
            "Download CSV (Excel-ptBR/EU: UTF-8 BOM, semicolon, decimal comma)",
            data=data_excel,
            file_name=f"{base_name}_excel.csv",
            mime="text/csv",
            use_container_width=True,
        )


def clean_df(
    df: pd.DataFrame,
    date_col: str = "Date",
    row_policy: str = "all_na",  # "keep" | "all_na" | "any_na"
    drop_date_na: bool = True,
    ffill: bool = False,
    bfill: bool = False,
    drop_dupes: bool = True,
    drop_empty_cols: bool = False,
    sort_by_date: bool = True,
) -> pd.DataFrame:
    d = df.copy()
    if date_col in d.columns:
        d[date_col] = pd.to_datetime(
            d[date_col], errors="coerce"
        ).dt.tz_localize(None)

    # Identify value columns (everything except the date)
    value_cols = [c for c in d.columns if c != date_col]

    if drop_empty_cols and value_cols:
        keep_vals = [c for c in value_cols if not d[c].isna().all()]
        d = d[[date_col] + keep_vals] if date_col in d else d[keep_vals]
        value_cols = keep_vals

    if drop_date_na and date_col in d.columns:
        d = d.dropna(subset=[date_col])

    # Optional fills before dropping rows
    if value_cols:
        if ffill:
            d[value_cols] = d[value_cols].ffill()
        if bfill:
            d[value_cols] = d[value_cols].bfill()

        if row_policy == "all_na":
            d = d.dropna(how="all", subset=value_cols)
        elif row_policy == "any_na":
            d = d.dropna(how="any", subset=value_cols)
        # "keep" -> do nothing

    if drop_dupes and date_col in d.columns:
        d = d.sort_values(date_col).drop_duplicates(date_col, keep="last")

    if sort_by_date and date_col in d.columns:
        d = d.sort_values(date_col)

    return d.reset_index(drop=True)


def cleaning_ui(
    df: pd.DataFrame, date_col: str = "Date", key: str = ""
) -> pd.DataFrame:
    if df.empty:
        return df

    def k(name: str) -> str:
        return f"{key}_{name}" if key else name

    st.markdown("### Cleaning")
    c1, c2, c3 = st.columns(3)
    with c1:
        row_choice = st.selectbox(
            "Row NaN handling",
            [
                "Keep all rows",
                "Drop rows where all values are NaN",
                "Drop rows where any value is NaN",
            ],
            index=1,
            key=k("row_choice"),
            help="Applied to all columns except the Date column.",
        )
        drop_empty_cols = st.checkbox(
            "Drop columns that are entirely NaN",
            value=False,
            key=k("drop_empty_cols"),
        )
    with c2:
        ffill = st.checkbox(
            "Forward-fill values", value=False, key=k("ffill")
        )
        bfill = st.checkbox(
            "Back-fill values", value=False, key=k("bfill")
        )
    with c3:
        drop_date_na = st.checkbox(
            "Drop invalid Date rows", value=True, key=k("drop_date_na")
        )
        drop_dupes = st.checkbox(
            "Drop duplicate dates (keep last)",
            value=True,
            key=k("drop_dupes"),
        )

    policy = (
        "keep"
        if row_choice.startswith("Keep")
        else "all_na"
        if "all values" in row_choice
        else "any_na"
    )

    cleaned = clean_df(
        df,
        date_col=date_col,
        row_policy=policy,
        drop_date_na=drop_date_na,
        ffill=ffill,
        bfill=bfill,
        drop_dupes=drop_dupes,
        drop_empty_cols=drop_empty_cols,
    )

    st.caption(
        f"Cleaned → rows {df.shape[0]} → {cleaned.shape[0]}, "
        f"cols {df.shape[1]} → {cleaned.shape[1]}"
    )
    return cleaned


# ----------------------- Sources ------------------------------------------ #


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_yfinance(
    tickers: List[str],
    start: str,
    end: str,
    interval: str = "1d",
    fields: Optional[List[str]] = None,
) -> pd.DataFrame:
    import yfinance as yf

    fields = fields or ["Adj Close"]

    df = yf.download(
        tickers=" ".join(tickers),
        start=start,
        end=end,
        interval=interval,
        auto_adjust=False,
        group_by="ticker",
        progress=False,
        threads=True,
    )
    if df.empty:
        return pd.DataFrame()

    df = df.reset_index().rename(columns={"Date": "Date"})
    out = pd.DataFrame({"Date": pd.to_datetime(df["Date"], errors="coerce")})

    # Multi-ticker yields MultiIndex cols; single-ticker yields flat cols
    if isinstance(df.columns, pd.MultiIndex):
        for t in tickers:
            for f in fields:
                if (t, f) in df.columns:
                    out[f"{t}_{f}"] = df[(t, f)]
    else:
        # Single ticker; columns are raw like 'Open', 'Close', ...
        t = tickers[0]
        for f in fields:
            if f in df.columns:
                out[f"{t}_{f}"] = df[f]

    out = out.sort_values("Date")
    return ensure_datetime(out, "Date")


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_bcb_sgs(
    series_ids: List[int | str],
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    def fmt_date(d: Optional[str]) -> Optional[str]:
        if not d:
            return None
        return pd.to_datetime(d).strftime("%d/%m/%Y")

    params_base = {"formato": "json"}
    if start:
        params_base["dataInicial"] = fmt_date(start)
    if end:
        params_base["dataFinal"] = fmt_date(end)

    dfs = []
    for sid in series_ids:
        url = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{sid}/dados"
        r = requests.get(url, params=params_base, timeout=30)
        r.raise_for_status()
        j = r.json()
        d = pd.DataFrame(j)
        if d.empty:
            continue
        d["Date"] = pd.to_datetime(d["data"], dayfirst=True, errors="coerce")
        d[str(sid)] = pd.to_numeric(
            d["valor"].astype(str).str.replace(",", ".", regex=False),
            errors="coerce",
        )
        dfs.append(d[["Date", str(sid)]])

    if not dfs:
        return pd.DataFrame()

    out = dfs[0]
    for i in range(1, len(dfs)):
        out = out.merge(dfs[i], on="Date", how="outer")

    out = out.sort_values("Date").reset_index(drop=True)
    return ensure_datetime(out, "Date")


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_cvm_inf_diario_fi(
    cnpjs: Optional[List[str]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    years: Optional[List[int]] = None,
) -> pd.DataFrame:
    def clean_cnpj(s: str) -> str:
        return re.sub(r"[^0-9]", "", s or "")

    if not years and (start or end):
        s = pd.to_datetime(start) if start else pd.Timestamp("2015-01-01")
        e = pd.to_datetime(end) if end else pd.Timestamp.utcnow()
        years = list(range(s.year, e.year + 1))
    years = years or [pd.Timestamp.utcnow().year]

    frames = []
    for y in years:
        url = (
            "https://dados.cvm.gov.br/dados/FI/DOC/INF_DIARIO/DADOS/"
            f"inf_diario_fi_{y}.csv"
        )
        r = requests.get(url, timeout=60)
        if r.status_code != 200:
            continue
        buf = io.StringIO(r.content.decode("utf-8", errors="ignore"))
        df = pd.read_csv(
            buf,
            sep=";",
            usecols=["CNPJ_FUNDO", "DT_COMPTC", "VL_QUOTA"],
            dtype={"CNPJ_FUNDO": str, "VL_QUOTA": str},
        )
        df["CNPJ_FUNDO"] = df["CNPJ_FUNDO"].map(clean_cnpj)
        df["Date"] = pd.to_datetime(
            df["DT_COMPTC"], format="%Y-%m-%d", errors="coerce"
        )
        df["VL_QUOTA"] = pd.to_numeric(
            df["VL_QUOTA"].astype(str).str.replace(",", ".", regex=False),
            errors="coerce",
        )
        frames.append(df[["CNPJ_FUNDO", "Date", "VL_QUOTA"]])

    if not frames:
        return pd.DataFrame()

    d = pd.concat(frames, ignore_index=True)

    if start:
        d = d[d["Date"] >= pd.to_datetime(start)]
    if end:
        d = d[d["Date"] <= pd.to_datetime(end)]
    if cnpjs:
        cnpjs_clean = [clean_cnpj(x) for x in cnpjs]
        d = d[d["CNPJ_FUNDO"].isin(cnpjs_clean)]

    if d.empty:
        return pd.DataFrame()

    wide = (
        d.pivot_table(
            index="Date",
            columns="CNPJ_FUNDO",
            values="VL_QUOTA",
            aggfunc="last",
        )
        .reset_index()
        .sort_values("Date")
    )
    wide = wide.rename(
        columns=lambda c: c if c == "Date" else f"{c}_VL_QUOTA"
    )
    return ensure_datetime(wide, "Date")


# ----------------------- UI ------------------------------------------------ #


tab_yf, tab_bcb, tab_cvm = st.tabs(
    ["Yahoo Finance", "BCB (SGS)", "CVM (INF_DIARIO_FI)"]
)

# ----------------------- Yahoo Finance Tab -------------------------------- #


with tab_yf:
    st.subheader("Yahoo Finance")
    with st.form("yf_form"):
        tks = st.text_input("Tickers (comma/space)", "AAPL, MSFT, GOOGL")
        start = st.date_input("Start date")
        end = st.date_input("End date")
        interval = st.selectbox("Interval", ["1d", "1wk", "1mo"], index=0)
        fields = st.multiselect(
            "Fields",
            ["Open", "High", "Low", "Close", "Adj Close", "Volume"],
            default=["Adj Close"],
        )
        btn_yf = st.form_submit_button("Fetch")

    if btn_yf:
        tickers = [
            t.strip().upper() for t in re.split(r"[,\s]+", tks) if t.strip()
        ]
        df = fetch_yfinance(
            tickers=tickers,
            start=str(start),
            end=str(end),
            interval=interval,
            fields=fields,
        )
        if df.empty:
            st.error("No data returned.")
            st.session_state["yf_df"] = pd.DataFrame()
        else:
            st.success(f"Fetched shape: {df.shape}.")
            st.session_state["yf_df"] = df

    df_yf = st.session_state["yf_df"]
    if not df_yf.empty:
        st.markdown("##### Raw preview")
        st.dataframe(df_yf.head(20))
        cleaned_yf = cleaning_ui(df_yf, date_col="Date", key="yf")
        st.markdown("##### Cleaned preview")
        st.dataframe(cleaned_yf.head(20))
        base = "yf_download"
        render_csv_downloads(cleaned_yf, base)
        if st.button("Clear YF data", key="clear_yf"):
            st.session_state["yf_df"] = pd.DataFrame()
            st.info("Cleared Yahoo Finance data.")

# ----------------------- BCB (SGS) Tab ------------------------------------ #


with tab_bcb:
    st.subheader("BCB (SGS)")
    with st.form("bcb_form"):
        sids = st.text_input("Series IDs (comma/space)", "11, 433, 444")
        start = st.text_input("Start date (YYYY-MM-DD)", "")
        end = st.text_input("End date (YYYY-MM-DD)", "")
        btn_bcb = st.form_submit_button("Fetch")

    if btn_bcb:
        ids = [s.strip() for s in re.split(r"[,\s]+", sids) if s.strip()]
        df = fetch_bcb_sgs(ids, start or None, end or None)
        if df.empty:
            st.error("No data returned.")
            st.session_state["bcb_df"] = pd.DataFrame()
        else:
            st.success(f"Fetched shape: {df.shape}.")
            st.session_state["bcb_df"] = df

    df_bcb = st.session_state["bcb_df"]
    if not df_bcb.empty:
        st.markdown("##### Raw preview")
        st.dataframe(df_bcb.head(20))
        cleaned_bcb = cleaning_ui(df_bcb, date_col="Date", key="bcb")
        st.markdown("##### Cleaned preview")
        st.dataframe(cleaned_bcb.head(20))
        base = "bcb_download"
        if start or end:
            base += f"_{start or ''}_{end or ''}"
        render_csv_downloads(cleaned_bcb, base)
        if st.button("Clear BCB data", key="clear_bcb"):
            st.session_state["bcb_df"] = pd.DataFrame()
            st.info("Cleared BCB data.")

# ----------------------- CVM (INF_DIARIO_FI) Tab -------------------------- #


with tab_cvm:
    st.subheader("CVM (INF_DIARIO_FI)")
    with st.form("cvm_form"):
        cnpjs = st.text_area(
            "CNPJs (one per line or comma-separated)",
            "",
            help="Tip: filtering by a few funds keeps downloads small.",
        )
        years = st.text_input(
            "Years (comma/space; leave empty to use date range)", ""
        )
        start = st.text_input("Start date (YYYY-MM-DD; optional)", "")
        end = st.text_input("End date (YYYY-MM-DD; optional)", "")
        btn_cvm = st.form_submit_button("Fetch")

    if btn_cvm:
        cnpj_list = [
            re.sub(r"[^0-9]", "", x)
            for x in re.split(r"[,\n\s]+", cnpjs.strip())
            if x.strip()
        ] or None
        year_list = [
            int(x) for x in re.split(r"[,\s]+", years) if x.strip().isdigit()
        ] or None
        df = fetch_cvm_inf_diario_fi(
            cnpjs=cnpj_list,
            start=start or None,
            end=end or None,
            years=year_list,
        )
        if df.empty:
            st.error("No data returned.")
            st.session_state["cvm_df"] = pd.DataFrame()
        else:
            st.success(f"Fetched shape: {df.shape}.")
            st.session_state["cvm_df"] = df

    df_cvm = st.session_state["cvm_df"]
    if not df_cvm.empty:
        st.markdown("##### Raw preview")
        st.dataframe(df_cvm.head(20))
        cleaned_cvm = cleaning_ui(df_cvm, date_col="Date", key="cvm")
        st.markdown("##### Cleaned preview")
        st.dataframe(cleaned_cvm.head(20))
        base = "cvm_inf_diario"
        render_csv_downloads(cleaned_cvm, base)
        if st.button("Clear CVM data", key="clear_cvm"):
            st.session_state["cvm_df"] = pd.DataFrame()
            st.info("Cleared CVM data.")