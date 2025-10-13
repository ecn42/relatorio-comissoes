import io
import re
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from typing import List, Tuple

import pandas as pd
import requests
import streamlit as st

BASE_URL = (
    "https://dados.cvm.gov.br/dados/FI/DOC/INF_DIARIO/DADOS/"
)  # inf_diario_fi_YYYYMM.zip
USER_AGENT = "streamlit-fi-fast/2.0"


# ---------------- Helpers ---------------- #
def sanitize_cnpj(raw: str) -> str:
    d = re.sub(r"\D", "", raw or "")
    if len(d) != 14:
        raise ValueError("CNPJ deve ter 14 dígitos.")
    return d


def months_between(start: date, end: date) -> List[Tuple[int, int]]:
    y, m = start.year, start.month
    out = []
    while (y < end.year) or (y == end.year and m <= end.month):
        if y >= 2021:
            out.append((y, m))
        if m == 12:
            y, m = y + 1, 1
        else:
            m += 1
    return out


def to_url(y: int, m: int) -> str:
    return f"{BASE_URL}inf_diario_fi_{y}{m:02d}.zip"


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def fetch_zip_bytes(url: str) -> bytes:
    r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=90)
    if r.status_code == 404:
        raise FileNotFoundError(url)
    r.raise_for_status()
    return r.content


def empty_df() -> pd.DataFrame:
    return pd.DataFrame(columns=["data", "vl_quota"])


def parse_month_bytes(buf: bytes, cnpj: str) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(buf)) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not names:
            return empty_df()
        name = names[0]

        # Detecta dinamicamente qual coluna de CNPJ existe
        with zf.open(name) as f_head:
            head = pd.read_csv(f_head, sep=";", encoding="latin-1", nrows=0)
            cols = set(map(str, head.columns))
        if "CNPJ_FUNDO" in cols:
            cnpj_col = "CNPJ_FUNDO"
        elif "CNPJ_FUNDO_CLASSE" in cols:
            cnpj_col = "CNPJ_FUNDO_CLASSE"
        else:
            # Layout inesperado
            return empty_df()

        usecols = [cnpj_col, "DT_COMPTC", "VL_QUOTA"]

        # Leitura em chunks filtrando pelo CNPJ
        parts = []
        with zf.open(name) as f:
            reader = pd.read_csv(
                f,
                sep=";",
                encoding="latin-1",
                dtype=str,
                usecols=usecols,
                chunksize=400_000,
            )
            for chunk in reader:
                # normaliza CNPJ e filtra
                chunk[cnpj_col] = chunk[cnpj_col].astype(str).str.replace(
                    r"\D", "", regex=True
                )
                sel = chunk[cnpj_col] == cnpj
                if not sel.any():
                    continue

                part = chunk.loc[sel, ["DT_COMPTC", "VL_QUOTA"]].copy()
                part["data"] = pd.to_datetime(
                    part["DT_COMPTC"], errors="coerce", dayfirst=True
                )
                part["vl_quota"] = pd.to_numeric(
                    part["VL_QUOTA"].astype(str).str.replace(",", ".", regex=False),
                    errors="coerce",
                )
                part = part.dropna(subset=["data", "vl_quota"])
                parts.append(part[["data", "vl_quota"]])

        if not parts:
            return empty_df()

        df = pd.concat(parts, ignore_index=True)
        df = df.sort_values("data").drop_duplicates(subset=["data"], keep="last")
        return df


def load_period_fast(cnpj: str, start: date, end: date, workers: int = 8) -> pd.DataFrame:
    months = months_between(start, end)
    if not months:
        return empty_df()

    urls = [to_url(y, m) for (y, m) in months]
    results: List[pd.DataFrame] = []
    prog = st.progress(0.0)

    def _one(u: str) -> pd.DataFrame:
        try:
            buf = fetch_zip_bytes(u)
        except FileNotFoundError:
            return empty_df()
        return parse_month_bytes(buf, cnpj)

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_one, u) for u in urls]
        for fut in as_completed(futures):
            df = fut.result()
            if not df.empty:
                results.append(df)
            done += 1
            prog.progress(done / len(urls))

    if not results:
        return empty_df()

    all_df = pd.concat(results, ignore_index=True).sort_values("data")
    mask = (all_df["data"].dt.date >= start) & (all_df["data"].dt.date <= end)
    return all_df.loc[mask].drop_duplicates(subset=["data"], keep="last")


def compute_returns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.sort_values("data").copy()
    df["ret_diario"] = df["vl_quota"].pct_change()
    df["ret_acum"] = (1 + df["ret_diario"].fillna(0.0)).cumprod() - 1
    return df


# ---------------- UI ---------------- #
st.set_page_config(page_title="Rentabilidade Fundos (CVM) – rápido", layout="wide")
st.title("Rentabilidade diária de Fundos (CVM) – rápido, sem banco")

with st.sidebar:
    cnpj_input = st.text_input("CNPJ (classe/fundo)", placeholder="00.000.000/0000-00")
    today = date.today()
    default_start = date(max(2021, today.year), max(1, today.month - 3), 1)
    period = st.date_input(
        "Período (2021+)",
        value=(default_start, today),
        min_value=date(2021, 1, 1),
        max_value=today,
        format="DD/MM/YYYY",
    )
    workers = st.slider("Paralelismo (downloads)", 2, 16, 8)
    run = st.button("Buscar e plotar")

if run:
    try:
        cnpj = sanitize_cnpj(cnpj_input)
    except ValueError as e:
        st.error(str(e))
        st.stop()

    if isinstance(period, (list, tuple)):
        start_date, end_date = period
    else:
        st.error("Selecione um intervalo de datas.")
        st.stop()

    if start_date > end_date:
        st.error("Data inicial deve ser anterior à final.")
        st.stop()

    with st.spinner("Baixando ZIPs e unindo CSVs do período..."):
        df = load_period_fast(cnpj, start_date, end_date, workers=workers)

    if df.empty:
        st.warning(
            "Nenhum dado encontrado. Observação: em muitos meses recentes a "
            "CVM publica CNPJ_FUNDO_CLASSE (CNPJ da classe). Verifique se o "
            "CNPJ informado é o da classe correspondente."
        )
        st.stop()

    df = compute_returns(df)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Retorno acumulado no período", f"{df['ret_acum'].iloc[-1]:.2%}")
    with col2:
        st.metric("Primeiro dia útil", df["data"].iloc[0].strftime("%d/%m/%Y"))
    with col3:
        st.metric("Último dia útil", df["data"].iloc[-1].strftime("%d/%m/%Y"))

    st.subheader("Rentabilidade diária")
    st.line_chart(df.set_index("data")["ret_diario"], height=320)

    st.subheader("Retorno acumulado")
    st.line_chart(df.set_index("data")["ret_acum"], height=320)

    st.download_button(
        "Baixar CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=f"rentabilidade_{cnpj}_{start_date}_{end_date}.csv",
        mime="text/csv",
    )

st.caption(
    "O app detecta automaticamente CNPJ_FUNDO vs CNPJ_FUNDO_CLASSE, "
    "faz downloads em paralelo e cacheia os ZIPs por 24h."
)