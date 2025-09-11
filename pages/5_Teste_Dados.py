import io
import zipfile
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd
import plotly.express as px
import requests
import streamlit as st


# --------------- Configs --------------- #
st.set_page_config(
    page_title="Risk & Macro Dashboard (BCB + CVM)",
    layout="wide",
    initial_sidebar_state="expanded",
)

TODAY = date.today()

# Séries SGS pré-mapeadas (pode expandir à vontade)
SGS_SERIES_PRESET: Dict[str, int] = {
    "Selic efetiva (% a.d.) [11]": 11,
    "Selic meta (% a.a.) [432]": 432,
    "CDI / DI [12]": 12,
    "IPCA (var mensal, %) [433]": 433,
    "IBOV [7]": 7,
    "PIB (preços correntes, R$) [1207]": 1207,
}

BCB_BASE = "https://api.bcb.gov.br/dados/serie/bcdata.sgs"
CVM_CAD_FI = (
    "https://dados.cvm.gov.br/dados/FI/CAD/DADOS/cad_fi.csv"
)
CVM_INF_DIARIO_BASE = (
    "https://dados.cvm.gov.br/dados/FI/DOC/INF_DIARIO/DADOS"
)


# --------------- Utils --------------- #
def _br_to_date(s: str) -> date:
    return datetime.strptime(s, "%d/%m/%Y").date()


def _to_br_date(d: date) -> str:
    return d.strftime("%d/%m/%Y")


def _safe_get(url: str, timeout: int = 30, max_retries: int = 3) -> requests.Response:
    last_exc = None
    for _ in range(max_retries):
        try:
            r = requests.get(url, timeout=timeout)
            if r.status_code == 200:
                return r
        except requests.RequestException as e:
            last_exc = e
    if last_exc:
        raise last_exc
    raise RuntimeError(f"HTTP {r.status_code} for {url}")


def _daterange_chunks_10y(
    start: date, end: date
) -> List[Tuple[date, date]]:
    # BCB limita 10 anos por requisição (JSON/CSV).
    chunks = []
    cur_start = start
    while cur_start <= end:
        cur_end = date(cur_start.year + 10, cur_start.month, cur_start.day)
        if cur_end > end:
            cur_end = end
        # Ajuste para meses com fim diferente (ex: 29/02)
        try:
            _ = _to_br_date(cur_end)
        except ValueError:
            cur_end = date(cur_end.year, cur_end.month, 28)
        chunks.append((cur_start, cur_end))
        cur_start = date(cur_end.year, cur_end.month, cur_end.day)
        # avança 1 dia
        cur_start = cur_start.replace(day=min(cur_start.day + 1, 28))
    # Corrigir possível overlap/avanço bizarro (garante monotonicidade)
    fixed = []
    last_end = None
    for s, e in chunks:
        if last_end and s <= last_end:
            s = date(last_end.year, last_end.month, last_end.day + 1)
        if s <= e:
            fixed.append((s, e))
        last_end = e
    return fixed


# --------------- BCB / SGS --------------- #
@st.cache_data(show_spinner=False)
def sgs_fetch_one(
    series_id: int, start: date, end: date
) -> pd.DataFrame:
    # Faz chunking automático por limite de 10 anos
    chunks = _daterange_chunks_10y(start, end)
    dfs = []
    for s, e in chunks:
        url = (
            f"{BCB_BASE}.{series_id}/dados?"
            f"formato=json&dataInicial={_to_br_date(s)}&dataFinal={_to_br_date(e)}"
        )
        r = _safe_get(url)
        data = r.json()
        df = pd.DataFrame(data)
        if df.empty:
            continue
        # Campos: data, valor
        df["data"] = pd.to_datetime(df["data"], format="%d/%m/%Y")
        df["valor"] = pd.to_numeric(
            df["valor"].str.replace(",", "."), errors="coerce"
        )
        dfs.append(df)
    if not dfs:
        return pd.DataFrame(columns=["data", "valor"])
    out = pd.concat(dfs, ignore_index=True)
    out = out.dropna(subset=["valor"]).sort_values("data")
    out = out.reset_index(drop=True)
    return out


@st.cache_data(show_spinner=False)
def sgs_fetch_many(
    series_map: Dict[str, int], start: date, end: date
) -> pd.DataFrame:
    frames = []
    for label, sid in series_map.items():
        df = sgs_fetch_one(sid, start, end)
        if not df.empty:
            df = df.rename(columns={"valor": label})
            frames.append(df[["data", label]])
    if not frames:
        return pd.DataFrame()
    out = frames[0]
    for f in frames[1:]:
        out = pd.merge(out, f, on="data", how="outer")
    return out.sort_values("data").reset_index(drop=True)


# --------------- CVM: Cadastro FI --------------- #
@st.cache_data(show_spinner=False)
def cvm_fetch_cad_fi() -> pd.DataFrame:
    r = _safe_get(CVM_CAD_FI, timeout=60)
    df = pd.read_csv(
        io.BytesIO(r.content), sep=";", decimal=",", encoding="latin1"
    )
    # Normaliza colunas que sempre usamos
    # (pode variar conforme versão; ajuste se necessário)
    for col in ["CNPJ_FUNDO", "DENOM_SOCIAL", "SIT", "DT_REG", "DT_CANCEL"]:
        if col not in df.columns:
            df[col] = None
    return df


# --------------- CVM: Informe Diário FI --------------- #
def _try_read_csv_bytes(
    raw: bytes, expected_name: Optional[str] = None
) -> pd.DataFrame:
    try:
        # Tenta ZIP
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            name = expected_name
            if name and name in z.namelist():
                with z.open(name) as f:
                    return pd.read_csv(
                        f,
                        sep=";",
                        decimal=",",
                        encoding="latin1",
                        low_memory=False,
                    )
            # senão, tenta o primeiro CSV do zip
            for nm in z.namelist():
                if nm.lower().endswith(".csv"):
                    with z.open(nm) as f:
                        return pd.read_csv(
                            f,
                            sep=";",
                            decimal=",",
                            encoding="latin1",
                            low_memory=False,
                        )
    except zipfile.BadZipFile:
        pass
    # Tenta CSV puro
    return pd.read_csv(
        io.BytesIO(raw),
        sep=";",
        decimal=",",
        encoding="latin1",
        low_memory=False,
    )


@st.cache_data(show_spinner=False)
def cvm_fetch_inf_diario_for_year(year: int) -> pd.DataFrame:
    # Tenta CSV puro e ZIP
    csv_url = f"{CVM_INF_DIARIO_BASE}/inf_diario_fi_{year}.csv"
    zip_url = f"{CVM_INF_DIARIO_BASE}/inf_diario_fi_{year}.csv.zip"
    urls = [csv_url, zip_url]
    last_exc = None
    for url in urls:
        try:
            r = _safe_get(url, timeout=120)
            df = _try_read_csv_bytes(
                r.content, expected_name=f"inf_diario_fi_{year}.csv"
            )
            # Normaliza nomes esperados
            needed = [
                "DT_COMPTC",
                "CNPJ_FUNDO",
                "VL_QUOTA",
                "VL_PATRIM_LIQ",
                "CAPTC_DIA",
                "RESG_DIA",
                "NR_COTST",
            ]
            for c in needed:
                if c not in df.columns:
                    df[c] = None
            # Tipos
            df["DT_COMPTC"] = pd.to_datetime(
                df["DT_COMPTC"], format="%Y-%m-%d", errors="coerce"
            ).fillna(
                pd.to_datetime(df["DT_COMPTC"], format="%d/%m/%Y", errors="coerce")
            )
            for num in [
                "VL_QUOTA",
                "VL_PATRIM_LIQ",
                "CAPTC_DIA",
                "RESG_DIA",
                "NR_COTST",
            ]:
                df[num] = pd.to_numeric(df[num], errors="coerce")
            return df[needed]
        except Exception as e:
            last_exc = e
    if last_exc:
        raise last_exc
    return pd.DataFrame(
        columns=[
            "DT_COMPTC",
            "CNPJ_FUNDO",
            "VL_QUOTA",
            "VL_PATRIM_LIQ",
            "CAPTC_DIA",
            "RESG_DIA",
            "NR_COTST",
        ]
    )


@st.cache_data(show_spinner=False)
def cvm_fetch_inf_diario(years: List[int]) -> pd.DataFrame:
    dfs = []
    for y in years:
        df = cvm_fetch_inf_diario_for_year(y)
        if not df.empty:
            dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    out = pd.concat(dfs, ignore_index=True)
    out = out.dropna(subset=["DT_COMPTC"]).sort_values("DT_COMPTC")
    return out.reset_index(drop=True)


# --------------- Analytics: Fund Performance --------------- #
def compute_fund_returns(
    df_inf: pd.DataFrame, cnpjs: List[str]
) -> pd.DataFrame:
    if df_inf.empty:
        return pd.DataFrame()
    sel = df_inf[df_inf["CNPJ_FUNDO"].isin(cnpjs)].copy()
    if sel.empty:
        return pd.DataFrame()
    # Para retorno 1/3/12M, resample por mês: último valor do mês
    # (evita gaps irregulares e "asof" manual)
    sel["DT_COMPTC_M"] = sel["DT_COMPTC"].dt.to_period("M").dt.to_timestamp(
        "M"
    )
    last_by_m = (
        sel.sort_values(["CNPJ_FUNDO", "DT_COMPTC"])
        .groupby(["CNPJ_FUNDO", "DT_COMPTC_M"], as_index=False)
        .last()
    )
    # Última data por fundo
    latest = (
        last_by_m.groupby("CNPJ_FUNDO", as_index=False)["DT_COMPTC_M"].max()
    ).rename(columns={"DT_COMPTC_M": "LAST_M"})
    merged = pd.merge(last_by_m, latest, on="CNPJ_FUNDO", how="inner")

    # Monta datas-alvo 1/3/12M atrás
    merged["M_1"] = merged["LAST_M"] - pd.offsets.MonthEnd(1)
    merged["M_3"] = merged["LAST_M"] - pd.offsets.MonthEnd(3)
    merged["M_12"] = merged["LAST_M"] - pd.offsets.MonthEnd(12)

    # Pega quotas nessas referências via merge
    ref = merged[
        ["CNPJ_FUNDO", "DT_COMPTC_M", "VL_QUOTA", "VL_PATRIM_LIQ", "LAST_M"]
    ].copy()
    ref = ref.rename(columns={"DT_COMPTC_M": "REF_M", "VL_QUOTA": "VLQ_REF"})
    # Join para 1M
    df1 = pd.merge(
        merged,
        ref.rename(columns={"REF_M": "M_1", "VLQ_REF": "VLQ_1"}),
        on=["CNPJ_FUNDO", "M_1"],
        how="left",
    )
    # 3M
    df1 = pd.merge(
        df1,
        ref.rename(columns={"REF_M": "M_3", "VLQ_REF": "VLQ_3"}),
        on=["CNPJ_FUNDO", "M_3"],
        how="left",
    )
    # 12M
    df1 = pd.merge(
        df1,
        ref.rename(columns={"REF_M": "M_12", "VLQ_REF": "VLQ_12"}),
        on=["CNPJ_FUNDO", "M_12"],
        how="left",
    )

    # Última cota (VLQ_LAST) e PL na última data
    last_rows = df1[df1["DT_COMPTC_M"] == df1["LAST_M"]].copy()
    last_rows = last_rows.rename(columns={"VL_QUOTA": "VLQ_LAST"})

    def ret(a: float, b: float) -> Optional[float]:
        if pd.notna(a) and pd.notna(b) and b != 0:
            return (a / b) - 1.0
        return None

    last_rows["RET_1M"] = last_rows.apply(
        lambda r: ret(r["VLQ_LAST"], r["VLQ_1"]), axis=1
    )
    last_rows["RET_3M"] = last_rows.apply(
        lambda r: ret(r["VLQ_LAST"], r["VLQ_3"]), axis=1
    )
    last_rows["RET_12M"] = last_rows.apply(
        lambda r: ret(r["VLQ_LAST"], r["VLQ_12"]), axis=1
    )

    out = last_rows[
        [
            "CNPJ_FUNDO",
            "LAST_M",
            "VLQ_LAST",
            "VL_PATRIM_LIQ",
            "RET_1M",
            "RET_3M",
            "RET_12M",
        ]
    ].drop_duplicates(subset=["CNPJ_FUNDO"])

    return out.reset_index(drop=True)


# --------------- UI Components --------------- #
def page_macro_bcb():
    st.subheader("Macro (BCB/SGS)")
    with st.sidebar:
        st.markdown("BCB/SGS - Parâmetros")
        start = st.date_input(
            "Data inicial",
            value=date(2010, 1, 1),
            min_value=date(1980, 1, 1),
            max_value=TODAY,
        )
        end = st.date_input(
            "Data final", value=TODAY, min_value=start, max_value=TODAY
        )
        series_labels = st.multiselect(
            "Séries SGS",
            list(SGS_SERIES_PRESET.keys()),
            default=[
                "Selic efetiva (% a.d.) [11]",
                "Selic meta (% a.a.) [432]",
                "CDI / DI [12]",
                "IPCA (var mensal, %) [433]",
            ],
        )
        label_to_id = {k: SGS_SERIES_PRESET[k] for k in series_labels}
        btn = st.button("Baixar séries do BCB", type="primary")

    if btn:
        with st.spinner("Consultando BCB/SGS..."):
            df = sgs_fetch_many(label_to_id, start, end)
        if df.empty:
            st.warning("Nenhum dado retornado para o período/seleção.")
            return
        st.success(
            f"{df.shape[0]} linhas, {df.shape[1]-1} séries | "
            "Obs: o BCB limita 10 anos por requisição; o app já "
            "segmenta automaticamente."
        )
        st.dataframe(df.tail(20), use_container_width=True)

        # Gráficos: uma figura por série
        for col in [c for c in df.columns if c != "data"]:
            fig = px.line(df, x="data", y=col, title=col)
            st.plotly_chart(fig, use_container_width=True)

        # Download
        st.download_button(
            "Exportar CSV (todas as séries)",
            df.to_csv(index=False).encode("utf-8"),
            file_name="bcb_series.csv",
            mime="text/csv",
        )

        st.caption(
            "Fonte: BCB/SGS - https://api.bcb.gov.br/dados/serie/bcdata.sgs"
        )


def page_fundos_cvm():
    st.subheader("Fundos (CVM) - Cadastro e Informe Diário")
    with st.sidebar:
        st.markdown("CVM - Parâmetros")
        st.markdown("Selecione os anos para baixar o Informe Diário:")
        years = st.multiselect(
            "Anos",
            list(range(2015, TODAY.year + 1)),
            default=[TODAY.year, TODAY.year - 1],
        )
        search = st.text_input(
            "Buscar por nome (contém) no Cadastro de FI (opcional):"
        )
        cnpjs_input = st.text_area(
            "Lista de CNPJs (um por linha, opcional):",
            help="Se preenchido, tem prioridade sobre a busca por nome.",
        )
        btn_cad = st.button("Carregar Cadastro de FI")
        btn_inf = st.button("Baixar Informe Diário (anos acima)", type="primary")

    cad = None
    if btn_cad:
        with st.spinner("Baixando Cadastro de FI..."):
            cad = cvm_fetch_cad_fi()
        if cad is None or cad.empty:
            st.warning("Cadastro vazio.")
        else:
            st.success(f"Cadastro de FI: {cad.shape[0]} registros.")
            if search:
                filt = cad[
                    cad["DENOM_SOCIAL"]
                    .astype(str)
                    .str.contains(search, case=False, na=False)
                ].copy()
                st.info(f"Encontrados {filt.shape[0]} fundos por nome.")
                st.dataframe(
                    filt[["CNPJ_FUNDO", "DENOM_SOCIAL", "SIT"]].head(200),
                    use_container_width=True,
                )
            else:
                st.dataframe(
                    cad[["CNPJ_FUNDO", "DENOM_SOCIAL", "SIT"]].head(200),
                    use_container_width=True,
                )
            st.download_button(
                "Exportar Cadastro (CSV)",
                cad.to_csv(index=False).encode("utf-8"),
                file_name="cvm_cad_fi.csv",
                mime="text/csv",
            )

    if btn_inf:
        if not years:
            st.warning("Selecione ao menos um ano.")
            return
        with st.spinner("Baixando Informe Diário da CVM..."):
            inf = cvm_fetch_inf_diario(years)
        if inf.empty:
            st.warning("Nenhum dado de Informe Diário retornado.")
            return
        st.success(
            f"Informe Diário: {inf['DT_COMPTC'].min().date()} "
            f"até {inf['DT_COMPTC'].max().date()} "
            f"| {inf.shape[0]:,} linhas".replace(",", ".")
        )
        st.dataframe(inf.tail(50), use_container_width=True)
        st.download_button(
            "Exportar Informe Diário (CSV)",
            inf.to_csv(index=False).encode("utf-8"),
            file_name="cvm_inf_diario.csv",
            mime="text/csv",
        )

        # Seleção de fundos por CNPJ (prioritário) ou por busca no cadastro
        cnpjs = []
        if cnpjs_input.strip():
            cnpjs = [
                c.strip()
                for c in cnpjs_input.replace(" ", "").split("\n")
                if c.strip()
            ]
        elif search and cad is not None and not cad.empty:
            sel = cad[
                cad["DENOM_SOCIAL"]
                .astype(str)
                .str.contains(search, case=False, na=False)
            ]["CNPJ_FUNDO"].dropna()
            cnpjs = sel.astype(str).tolist()[:20]
            if cnpjs:
                st.info(
                    f"Selecionando primeiros {len(cnpjs)} CNPJs pela busca."
                )

        if cnpjs:
            perf = compute_fund_returns(inf, cnpjs)
            if perf.empty:
                st.warning(
                    "Não foi possível calcular performance para os CNPJs "
                    "selecionados (dados insuficientes?)."
                )
            else:
                # Junta nomes do cadastro (se disponível)
                if cad is not None and not cad.empty:
                    perf = perf.merge(
                        cad[["CNPJ_FUNDO", "DENOM_SOCIAL"]],
                        on="CNPJ_FUNDO",
                        how="left",
                    )
                # Ordena por PL (desc)
                perf = perf.sort_values(
                    "VL_PATRIM_LIQ", ascending=False
                ).reset_index(drop=True)

                # Formata percentuais
                show = perf.copy()
                for c in ["RET_1M", "RET_3M", "RET_12M"]:
                    show[c] = (show[c] * 100).round(2)
                st.markdown("Performance (1M / 3M / 12M) e PL (último mês):")
                st.dataframe(
                    show[
                        [
                            "DENOM_SOCIAL",
                            "CNPJ_FUNDO",
                            "LAST_M",
                            "VLQ_LAST",
                            "VL_PATRIM_LIQ",
                            "RET_1M",
                            "RET_3M",
                            "RET_12M",
                        ]
                    ],
                    use_container_width=True,
                )

                st.download_button(
                    "Exportar Performance (CSV)",
                    perf.to_csv(index=False).encode("utf-8"),
                    file_name="funds_performance.csv",
                    mime="text/csv",
                )


def main():
    st.title("Monitoramento — Risco de Mercado & Comitê (BCB + CVM)")
    tab1, tab2 = st.tabs(["Macro (BCB/SGS)", "Fundos (CVM)"])
    with tab1:
        page_macro_bcb()
    with tab2:
        page_fundos_cvm()

    st.divider()
    st.markdown(
        "- MVP: coleta/visualização de dados BCB/SGS e CVM.\n"
        "- Próximos passos (sugeridos):\n"
        "  - Exposição por estratégia/issuer/rating/sector: upload de posições "
        "(CSV/Parquet) e mapeamentos; cruzar com CVM/ANBIMA.\n"
        "  - VaR e Stress: histórico (BCB/SGS e mercado) + cenários fixos "
        "e parametrizáveis.\n"
        "  - Curvas DI (nominal) e IPCA+ (real): integrar com ANBIMA (ETTJ) "
        "ou B3 (DI futuro) para curva por vértices.\n"
        "  - EUA 10Y e diferencial Brasil x EUA: integrar FRED.\n"
        "  - Prêmios de debêntures: integrar ANBIMA SND.\n"
        "  - Aderência/Enquadramento: motor de regras (JSON) baseado em "
        "política/limites.\n"
    )


if __name__ == "__main__":
    main()