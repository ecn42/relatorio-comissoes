# app.py
# -*- coding: utf-8 -*-

import datetime as dt
import io
import sqlite3
from pathlib import Path
from typing import Tuple

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st


### Simple Authentication
if not st.session_state.get("authenticated", False):
    st.warning("Please enter the password on the Home page first.")
    st.write("Status: Não Autenticado")
    st.stop()
    
  # prevent the rest of the page from running
st.write("Autenticado")

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="CDI e Ibovespa - Rentabilidade Acumulada",
    layout="centered",
)

DB_PATH = Path("./data/data_cdi_ibov.db")
CDI_SERIES_ID = 12  # SGS: CDI Over - taxa diária (% a.d.)
IBOV_CSV_URL = "https://stooq.com/q/d/l/?s=%5Ebvp&i=d"
DEFAULT_START_DATE = dt.date(2021, 1, 1)


# -----------------------------------------------------------------------------
# DB helpers
# -----------------------------------------------------------------------------

def ensure_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS cdi (
                date TEXT PRIMARY KEY,
                daily_rate_pct REAL,   -- taxa % a.d. (SGS 'valor')
                daily_return REAL      -- fração ao dia (ex.: 0.0003)
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS ibov (
                date TEXT PRIMARY KEY,
                close REAL,
                daily_return REAL      -- fração ao dia (ex.: 0.01)
            )
            """
        )
        conn.commit()


def upsert_cdi(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    with sqlite3.connect(DB_PATH) as conn:
        rows = [
            (d.strftime("%Y-%m-%d"), r, ret)
            for d, r, ret in zip(
                df["data"].dt.date, df["taxa_ad_pct"], df["daily_return"]
            )
        ]
        conn.executemany(
            """
            INSERT INTO cdi (date, daily_rate_pct, daily_return)
            VALUES (?, ?, ?)
            ON CONFLICT(date) DO UPDATE SET
              daily_rate_pct=excluded.daily_rate_pct,
              daily_return=excluded.daily_return
            """,
            rows,
        )
        conn.commit()
        return len(rows)


def upsert_ibov(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    with sqlite3.connect(DB_PATH) as conn:
        rows = [
            (d.strftime("%Y-%m-%d"), c, ret)
            for d, c, ret in zip(
                df["data"].dt.date, df["close"], df["daily_return"]
            )
        ]
        conn.executemany(
            """
            INSERT INTO ibov (date, close, daily_return)
            VALUES (?, ?, ?)
            ON CONFLICT(date) DO UPDATE SET
              close=excluded.close,
              daily_return=excluded.daily_return
            """,
            rows,
        )
        conn.commit()
        return len(rows)


def read_cdi(start_date: dt.date, end_date: dt.date) -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(
            """
            SELECT date, daily_rate_pct, daily_return
            FROM cdi
            WHERE date BETWEEN ? AND ?
            ORDER BY date
            """,
            conn,
            params=(
                start_date.strftime("%Y-%m-%d"),
                end_date.strftime("%Y-%m-%d"),
            ),
        )
    if df.empty:
        return df
    df["data"] = pd.to_datetime(df["date"])
    return df.drop(columns=["date"])


def read_ibov(start_date: dt.date, end_date: dt.date) -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(
            """
            SELECT date, close, daily_return
            FROM ibov
            WHERE date BETWEEN ? AND ?
            ORDER BY date
            """,
            conn,
            params=(
                start_date.strftime("%Y-%m-%d"),
                end_date.strftime("%Y-%m-%d"),
            ),
        )
    if df.empty:
        return df
    df["data"] = pd.to_datetime(df["date"])
    return df.drop(columns=["date"])


def get_last_dates() -> Tuple[dt.date | None, dt.date | None]:
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute("SELECT MAX(date) FROM cdi")
        cdi_max = cur.fetchone()[0]
        cur.execute("SELECT MAX(date) FROM ibov")
        ibov_max = cur.fetchone()[0]
    cdi_date = dt.date.fromisoformat(cdi_max) if cdi_max else None
    ibov_date = dt.date.fromisoformat(ibov_max) if ibov_max else None
    return cdi_date, ibov_date


# -----------------------------------------------------------------------------
# Fetchers
# -----------------------------------------------------------------------------

def fetch_cdi_sgs_12(
    start_date: dt.date, end_date: dt.date
) -> pd.DataFrame:
    """
    Busca CDI (série 12) no SGS/BCB.
    Retorna DataFrame com colunas: data (datetime), taxa_ad_pct (float),
    daily_return (float).
    """
    base_url = (
        f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{CDI_SERIES_ID}/dados"
    )
    params = {
        "formato": "json",
        "dataInicial": start_date.strftime("%d/%m/%Y"),
        "dataFinal": end_date.strftime("%d/%m/%Y"),
    }
    r = requests.get(base_url, params=params, timeout=30)
    r.raise_for_status()
    js = r.json()
    df = pd.DataFrame(js)
    if df.empty:
        return pd.DataFrame(columns=["data", "taxa_ad_pct", "daily_return"])

    # Normaliza
    df["data"] = pd.to_datetime(df["data"], dayfirst=True, errors="coerce")
    df["valor"] = (
        df["valor"].astype(str).str.replace(",", ".", regex=False).astype(float)
    )
    df = df.dropna(subset=["data", "valor"]).sort_values("data").reset_index(
        drop=True
    )

    out = pd.DataFrame()
    out["data"] = df["data"]
    out["taxa_ad_pct"] = df["valor"]
    out["daily_return"] = out["taxa_ad_pct"] / 100.0  # % a.d. -> fração
    return out


def fetch_ibov_stooq() -> pd.DataFrame:
    """
    Baixa IBOV do Stooq (^BVP) em CSV diário.
    Retorna colunas: data (datetime), close (float).
    """
    r = requests.get(IBOV_CSV_URL, timeout=30)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    if df.empty:
        return pd.DataFrame(columns=["data", "close"])
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    df.rename(columns={"Date": "data", "Close": "close"}, inplace=True)
    return df[["data", "close"]]


# -----------------------------------------------------------------------------
# Transformations
# -----------------------------------------------------------------------------

def compute_returns_ibov(
    df_prices: pd.DataFrame, start_date: dt.date, end_date: dt.date
) -> pd.DataFrame:
    m = (df_prices["data"].dt.date >= start_date) & (
        df_prices["data"].dt.date <= end_date
    )
    sub = df_prices.loc[m].copy()
    if sub.empty:
        return pd.DataFrame(columns=["data", "close", "daily_return"])
    sub["daily_return"] = sub["close"].pct_change().fillna(0.0)
    return sub[["data", "close", "daily_return"]]


def add_cumulative(df: pd.DataFrame, return_col: str) -> pd.DataFrame:
    if df.empty:
        return df
    idx = (1.0 + df[return_col]).cumprod()
    out = df.copy()
    out["indice"] = idx
    out["rentab_acum_%"] = (idx - 1.0) * 100.0
    return out


# -----------------------------------------------------------------------------
# Update routine
# -----------------------------------------------------------------------------

def update_data(
    start_date: dt.date, end_date: dt.date
) -> Tuple[int, int]:
    """
    Rebusca toda a janela [start, end] e faz upsert nas tabelas.
    Retorna: (linhas_cdi, linhas_ibov)
    """
    # CDI
    cdi_df = fetch_cdi_sgs_12(start_date, end_date)
    cdi_rows = upsert_cdi(cdi_df)

    # IBOV
    ibov_prices = fetch_ibov_stooq()
    ibov_df = compute_returns_ibov(ibov_prices, start_date, end_date)
    ibov_rows = upsert_ibov(ibov_df)

    return cdi_rows, ibov_rows


# -----------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------

def plot_combined(
    cdi_cum: pd.DataFrame,
    ibov_cum: pd.DataFrame,
    start_date: dt.date,
) -> None:
    title = (
        f"Rentabilidade Acumulada - CDI vs IBOV "
        f"(desde {start_date.strftime('%d/%m/%Y')})"
    )
    fig = go.Figure()
    if not cdi_cum.empty:
        fig.add_trace(
            go.Scatter(
                x=cdi_cum["data"],
                y=cdi_cum["rentab_acum_%"],
                mode="lines",
                name="CDI",
                line=dict(color="#2ca02c", width=2),
            )
        )
    if not ibov_cum.empty:
        fig.add_trace(
            go.Scatter(
                x=ibov_cum["data"],
                y=ibov_cum["rentab_acum_%"],
                mode="lines",
                name="IBOV",
                line=dict(color="#d62728", width=2),
            )
        )
    fig.update_layout(
        title=title,
        template="plotly_white",
        hovermode="x unified",
        yaxis_title="Rentab. acumulada (%)",
        xaxis_title="Data",
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_single(
    df_cum: pd.DataFrame, label: str, color: str, start_date: dt.date
) -> None:
    title = f"{label} - Rentabilidade acumulada (desde {start_date.strftime('%d/%m/%Y')})"
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df_cum["data"],
            y=df_cum["rentab_acum_%"],
            mode="lines",
            name=label,
            line=dict(color=color, width=2),
        )
    )
    fig.update_layout(
        title=title,
        template="plotly_white",
        hovermode="x unified",
        yaxis_title="Rentab. acumulada (%)",
        xaxis_title="Data",
    )
    st.plotly_chart(fig, use_container_width=True)


# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------

def main() -> None:
    ensure_db(DB_PATH)

    st.title("CDI e Ibovespa - Rentabilidade Acumulada")
    st.caption(
        "Fontes: CDI (SGS/BCB, série 12, % a.d.) • IBOV (Stooq ^BVP). "
        "IBOV é preço (sem dividendos)."
    )

    c1, c2 = st.columns(2)
    with c1:
        start_date = st.date_input(
            "Data inicial",
            value=DEFAULT_START_DATE,
            min_value=dt.date(2000, 1, 1),
        )
    with c2:
        end_date = st.date_input(
            "Data final",
            value=dt.date.today(),
            max_value=dt.date.today(),
        )

    upd = st.button("Atualizar dados", type="primary")

    if upd:
        with st.spinner("Atualizando dados (CDI e IBOV)..."):
            try:
                cdi_rows, ibov_rows = update_data(start_date, end_date)
                st.success(
                    f"Atualizado: CDI {cdi_rows} linhas • IBOV {ibov_rows} linhas."
                )
            except Exception as e:
                st.error(f"Erro ao atualizar: {e}")

    # Leitura do DB para o intervalo selecionado
    cdi = read_cdi(start_date, end_date)
    ibov = read_ibov(start_date, end_date)

    # Se DB estiver vazio, faça uma carga inicial
    if cdi.empty and ibov.empty:
        with st.spinner("Carregando dados iniciais..."):
            try:
                update_data(start_date, end_date)
                cdi = read_cdi(start_date, end_date)
                ibov = read_ibov(start_date, end_date)
                st.success("Carga inicial concluída.")
            except Exception as e:
                st.error(f"Erro na carga inicial: {e}")
                st.stop()

    cdi_cum = add_cumulative(cdi.rename(columns={"daily_return": "ret"}), "ret")
    if not cdi_cum.empty:
        cdi_cum.rename(
            columns={"ret": "daily_return"}, inplace=True
        )  # volta o nome

    ibov_cum = add_cumulative(
        ibov.rename(columns={"daily_return": "ret"}), "ret"
    )
    if not ibov_cum.empty:
        ibov_cum.rename(columns={"ret": "daily_return"}, inplace=True)

    top1, top2, top3, top4 = st.columns(4)
    with top1:
        last_cdi = cdi["data"].max() if not cdi.empty else None
        st.metric("Última data CDI", f"{last_cdi.date() if last_cdi else '-'}")
    with top2:
        last_ibov = ibov["data"].max() if not ibov.empty else None
        st.metric("Última data IBOV", f"{last_ibov.date() if last_ibov else '-'}")
    with top3:
        cdi_total = (
            cdi_cum["rentab_acum_%"].iloc[-1] if not cdi_cum.empty else 0.0
        )
        st.metric("CDI no período", f"{cdi_total:.2f}%")
    with top4:
        ibov_total = (
            ibov_cum["rentab_acum_%"].iloc[-1] if not ibov_cum.empty else 0.0
        )
        st.metric("IBOV no período", f"{ibov_total:.2f}%")

    plot_combined(cdi_cum, ibov_cum, start_date)

    with st.expander("Ver individualmente"):
        if not cdi_cum.empty:
            plot_single(cdi_cum, "CDI", "#2ca02c", start_date)
        if not ibov_cum.empty:
            plot_single(ibov_cum, "IBOV", "#d62728", start_date)

    st.subheader("Amostra de dados (últimas linhas)")
    colA, colB = st.columns(2)
    with colA:
        st.markdown("CDI (retornos diários)")
        st.dataframe(
            cdi.tail(10),
            use_container_width=True,
            hide_index=True,
        )
    with colB:
        st.markdown("IBOV (preços e retornos diários)")
        st.dataframe(
            ibov.tail(10),
            use_container_width=True,
            hide_index=True,
        )

    # Downloads
    c1, c2 = st.columns(2)
    with c1:
        if not cdi.empty:
            csv = cdi.assign(
                date=cdi["data"].dt.date
            ).drop(columns=["data"]).to_csv(index=False)
            st.download_button(
                "Baixar CSV - CDI (diário)",
                data=csv.encode("utf-8"),
                file_name="cdi_diario.csv",
                mime="text/csv",
            )
    with c2:
        if not ibov.empty:
            csv = ibov.assign(
                date=ibov["data"].dt.date
            ).drop(columns=["data"]).to_csv(index=False)
            st.download_button(
                "Baixar CSV - IBOV (diário)",
                data=csv.encode("utf-8"),
                file_name="ibov_diario.csv",
                mime="text/csv",
            )

    st.caption(
        "Notas: "
        "1) CDI: série 12 do SGS (taxa diária % a.d.); retorno diário = taxa/100. "
        "2) IBOV do Stooq (^BVP) é índice de preço (sem dividendos). "
        "3) Finais de semana/feriados não aparecem por não haver pregão/taxa."
    )


if __name__ == "__main__":
    main()