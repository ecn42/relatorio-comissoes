import re
import unicodedata
from io import BytesIO
from typing import Literal, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# ----------------------------- Config e Estilo ------------------------------ #
st.set_page_config(
    page_title="Dashboard de Carteira - NET",
    page_icon="📊",
    layout="wide",
)


# -------------------------- Utilidades de Texto ----------------------------- #
def _strip_accents(s: str) -> str:
    nfkd = unicodedata.normalize("NFD", s)
    return "".join(c for c in nfkd if unicodedata.category(c) != "Mn")


def _clean_invisibles(s: str) -> str:
    if not isinstance(s, str):
        s = str(s)
    # Remove ASCII/Unicode control chars
    s = re.sub(r"[\u0000-\u001F\u007F]", "", s)
    # Remove zero-width chars e BOM
    s = re.sub(r"[\u200B-\u200D\uFEFF]", "", s)
    # Espaço não-quebrante -> espaço normal
    s = s.replace("\u00A0", " ")
    return s


def _normalize_col(col: str) -> str:
    s = _clean_invisibles(str(col))
    s = s.strip().lower().replace("\n", " ")
    s = _strip_accents(s)
    for ch in [",", ";", ":", "|"]:
        s = s.replace(ch, " ")
    s = " ".join(s.split())
    s = s.replace(" ", "_")
    return s


def _safe_filename(name: str) -> str:
    s = _clean_invisibles(name)
    s = _strip_accents(s)
    s = re.sub(r"[^\w\-.()]+", "_", s)
    s = s.strip("._")
    return s or "arquivo"


def _safe_sheet_name(name: str) -> str:
    s = _safe_filename(name).replace("-", "_")
    if not s:
        s = "Sheet"
    if len(s) > 31:
        s = s[:31]
    return s


# ---------------------------- Mapeamento de Colunas ------------------------- #
def _synonyms_dict() -> dict[str, str]:
    # keys NORMALIZADOS; values = nomes padronizados
    return {
        "produto": "produto",
        "sub_produto": "sub_produto",
        "subproduto": "sub_produto",
        "fator_de_risco": "fator_risco",
        "fator_risco": "fator_risco",
        "fator_de_risco_1": "fator_risco",
        "fator_de_risco_2": "fator_risco",
        "fator_de_risco_3": "fator_risco",
        "fator": "fator_risco",
        "fator_de": "fator_risco",
        "fator de risco": "fator_risco",
        "fator_riscos": "fator_risco",
        "fatores_de_risco": "fator_risco",
        "ativo": "ativo",
        "codigo": "codigo",
        "cod": "codigo",
        "emissor": "emissor",
        "data_vencimento": "data_vencimento",
        "data_de_vencimento": "data_vencimento",
        "vencimento": "data_vencimento",
        "quantidade": "quantidade",
        "qtd": "quantidade",
        "net": "net",
        "valor_liquido": "net",
        "valor_net": "net",
        "posicao_liquida": "net",
        "data_posicao": "data_posicao",
        "data_de_posicao": "data_posicao",
        "data_atualizacao": "data_atualizacao",
        "data_de_atualizacao": "data_atualizacao",
        "cetip": "cetip",
        "subordinacao": "subordinacao",
        "subordinação": "subordinacao",
        "subordinacao_geral": "subordinacao",
        "rating": "rating",
        "classificacao": "rating",
        "classificacao_de_risco": "rating",
        "rating_emissor": "rating",
    }


def _rename_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    syn = _synonyms_dict()
    new_cols = {}
    for c in df.columns:
        key = _normalize_col(c)
        mapped = syn.get(key, key)
        new_cols[c] = mapped
    df = df.rename(columns=new_cols)
    return df, new_cols


# ----------------------------- Conversões básicas --------------------------- #
def _to_number(series: pd.Series) -> pd.Series:
    if series.dtype == "O":
        s = series.astype(str).str.strip()
        mask = s.str.contains(",") & s.str.contains(".")
        s_loc = s.copy()
        s_loc[mask] = (
            s_loc[mask]
            .str.replace(".", "", regex=False)
            .str.replace(",", ".", regex=False)
        )
        s_loc[~mask] = s_loc[~mask].str.replace(",", ".", regex=False)
        return pd.to_numeric(s_loc, errors="coerce")
    return pd.to_numeric(series, errors="coerce")


def _parse_dates(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", dayfirst=True)


# ------------------------------ Leitura XLSX -------------------------------- #
def _read_excel_autoheader(
    data_bytes: bytes, sheet_name: str, scan_rows: int = 15
) -> pd.DataFrame:
    # Detecção automática da linha de cabeçalho
    df_raw = pd.read_excel(
        BytesIO(data_bytes), sheet_name=sheet_name, header=None, engine="openpyxl"
    )

    syn = _synonyms_dict()
    alias_keys = set(syn.keys())

    best_row = 0
    best_score = -1
    max_r = min(scan_rows, len(df_raw) - 1)

    for r in range(0, max_r + 1):
        row_vals = df_raw.iloc[r].tolist()
        norm = [_normalize_col(x) for x in row_vals]
        score = sum(1 for x in norm if x in alias_keys)
        if score > best_score:
            best_score = score
            best_row = r

    header = df_raw.iloc[best_row].tolist()
    data = df_raw.iloc[best_row + 1 :].copy()
    data.columns = header
    data = data.dropna(axis=1, how="all").reset_index(drop=True)
    return data


@st.cache_data(show_spinner=False)
def get_sheet_names(data_bytes: bytes) -> list[str]:
    x = pd.ExcelFile(BytesIO(data_bytes))
    return x.sheet_names


@st.cache_data(show_spinner=False)
def load_sheet(
    data_bytes: bytes, sheet_name: str
) -> tuple[pd.DataFrame, dict[str, str], dict[str, str]]:
    # Tenta leitura padrão
    df_try = pd.read_excel(BytesIO(data_bytes), sheet_name=sheet_name, engine="openpyxl")
    df1, colmap1 = _rename_columns(df_try)

    need_autoheader = not any(
        col in df1.columns
        for col in ["net", "fator_risco", "rating", "subordinacao", "emissor"]
    )

    if need_autoheader:
        df_auto = _read_excel_autoheader(data_bytes, sheet_name)
        df, colmap = _rename_columns(df_auto)
        source = "autoheader"
    else:
        df, colmap = df1, colmap1
        source = "standard"

    # Conversões
    if "net" in df.columns:
        df["net"] = _to_number(df["net"])
    if "quantidade" in df.columns:
        df["quantidade"] = _to_number(df["quantidade"])
    if "data_vencimento" in df.columns:
        df["data_vencimento"] = _parse_dates(df["data_vencimento"])
    if "data_posicao" in df.columns:
        df["data_posicao"] = _parse_dates(df["data_posicao"])
    if "data_atualizacao" in df.columns:
        df["data_atualizacao"] = _parse_dates(df["data_atualizacao"])

    # Rating vazio => "Sem Rating"
    if "rating" in df.columns:
        r = df["rating"].astype("string")
        is_empty = r.isna() | (r.str.strip() == "") | (r.str.upper() == "N/A")
        df.loc[is_empty, "rating"] = "Sem Rating"

    meta = {"load_source": source}
    return df, colmap, meta


# ----------------------------- Preparação Dados ----------------------------- #
def _ensure_category(series: pd.Series, default_label: str) -> pd.Series:
    s = series.astype("string")
    is_empty = s.isna() | (s.str.strip() == "") | (s.str.upper() == "N/A")
    s = s.where(~is_empty, other=default_label)
    return s


def _bucket_vencimento(venc: pd.Series, ref_date: pd.Timestamp) -> pd.Series:
    s = pd.Series(index=venc.index, dtype="object")
    mask_nan = venc.isna()
    s.loc[mask_nan] = "Sem Vencimento"
    days = (venc - ref_date).dt.days
    s.loc[~mask_nan & (days < 0)] = "Vencidos"
    s.loc[~mask_nan & (days >= 0) & (days <= 365)] = "Até 1 ano"
    s.loc[~mask_nan & (days > 365) & (days <= 3 * 365)] = "1-3 anos"
    s.loc[~mask_nan & (days > 3 * 365) & (days <= 5 * 365)] = "3-5 anos"
    s.loc[~mask_nan & (days > 5 * 365)] = "Acima de 5 anos"
    s = pd.Categorical(
        s,
        categories=[
            "Vencidos",
            "Até 1 ano",
            "1-3 anos",
            "3-5 anos",
            "Acima de 5 anos",
            "Sem Vencimento",
        ],
        ordered=True,
    )
    return s


def prepare_aggregation(
    df: pd.DataFrame,
    dimension: Literal[
        "Ativo",
        "Sub Produto",
        "Emissor",
        "Fator de Risco",
        "Rating",
        "Subordinação",
        "Vencimento",
    ],
    venc_grouping: Literal["Ano", "Mes/Ano", "Buckets"],
    ref_date: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    if "net" not in df.columns:
        raise ValueError("Coluna 'NET' não encontrada (após normalização: 'net').")

    if dimension == "Ativo":
        cat = _ensure_category(df.get("ativo", pd.Series(index=df.index)), "Não informado")
    elif dimension == "Sub Produto":
        cat = _ensure_category(
            df.get("sub_produto", pd.Series(index=df.index)), "Não informado"
        )
    elif dimension == "Emissor":
        cat = _ensure_category(
            df.get("emissor", pd.Series(index=df.index)), "Não informado"
        )
    elif dimension == "Fator de Risco":
        cat = _ensure_category(
            df.get("fator_risco", pd.Series(index=df.index)), "Não informado"
        )
    elif dimension == "Rating":
        cat = _ensure_category(df.get("rating", pd.Series(index=df.index)), "Sem Rating")
    elif dimension == "Subordinação":
        cat = _ensure_category(
            df.get("subordinacao", pd.Series(index=df.index)), "Não informado"
        )
    elif dimension == "Vencimento":
        venc = df.get("data_vencimento", pd.Series(index=df.index, dtype="datetime64[ns]"))
        if ref_date is None:
            ref_date = pd.Timestamp.today().normalize()
        if venc_grouping == "Ano":
            cat = venc.dt.year.astype("Int64").astype("string")
            cat = _ensure_category(cat, "Sem Vencimento")
        elif venc_grouping == "Mes/Ano":
            cat = venc.dt.to_period("M").astype("string")
            cat = _ensure_category(cat, "Sem Vencimento")
        else:
            cat = _bucket_vencimento(venc, ref_date).astype("string")
    else:
        raise ValueError("Dimensão inválida.")

    agg = (
        pd.DataFrame({"categoria": cat, "net": df["net"]})
        .groupby("categoria", dropna=False, as_index=False)["net"]
        .sum()
    )

    agg = agg[(agg["categoria"].astype("string").str.strip() != "")]

    total = agg["net"].sum()
    if total == 0 or np.isclose(total, 0.0):
        agg["participacao"] = 0.0
    else:
        agg["participacao"] = agg["net"] / total

    agg = agg.sort_values("net", ascending=False, kind="mergesort").reset_index(drop=True)
    return agg


def apply_others_grouping(
    agg: pd.DataFrame,
    method: Literal["none", "pct", "topn"],
    pct_threshold: float = 0.01,
    top_n: int = 10,
    others_label: str = "Outros",
) -> pd.DataFrame:
    if method == "none":
        return agg

    if method == "pct":
        mask_small = agg["participacao"] < pct_threshold
        if not mask_small.any():
            return agg
        big = agg.loc[~mask_small].copy()
        small = agg.loc[mask_small]
        row_outros = pd.DataFrame(
            {
                "categoria": [others_label],
                "net": [small["net"].sum()],
                "participacao": [small["participacao"].sum()],
            }
        )
        out = pd.concat([big, row_outros], ignore_index=True)
        out = (
            out.sort_values(["categoria"], key=lambda s: s != others_label)
            .sort_values("net", ascending=False, kind="mergesort")
            .reset_index(drop=True)
        )
        return out

    if method == "topn":
        if top_n >= len(agg):
            return agg
        big = agg.iloc[:top_n].copy()
        small = agg.iloc[top_n:]
        row_outros = pd.DataFrame(
            {
                "categoria": [others_label],
                "net": [small["net"].sum()],
                "participacao": [small["participacao"].sum()],
            }
        )
        out = pd.concat([big, row_outros], ignore_index=True)
        out = (
            out.sort_values(["categoria"], key=lambda s: s != others_label)
            .sort_values("net", ascending=False, kind="mergesort")
            .reset_index(drop=True)
        )
        return out

    return agg


# ------------------------------ Gráficos ------------------------------------ #
def plot_bar(
    df: pd.DataFrame, value_mode: Literal["valor", "percent"], title: str
) -> "px.Figure":
    if value_mode == "valor":
        x = "net"
        hovertemplate = "%{y}<br>NET: R$ %{x:,.2f}<extra></extra>"
        fig = px.bar(df, x=x, y="categoria", orientation="h", text="net", title=title)
        fig.update_traces(texttemplate="R$ %{x:,.2f}", hovertemplate=hovertemplate)
        fig.update_xaxes(tickprefix="R$ ", tickformat=",.2f")
    else:
        df = df.copy()
        df["pct"] = df["participacao"]
        x = "pct"
        hovertemplate = (
            "%{y}<br>Participação: %{x:.2%}"
            "<br>NET: R$ %{customdata:,.2f}<extra></extra>"
        )
        fig = px.bar(
            df,
            x=x,
            y="categoria",
            orientation="h",
            text="pct",
            title=title,
            custom_data=["net"],
        )
        fig.update_traces(texttemplate="%{x:.1%}", hovertemplate=hovertemplate)
        fig.update_xaxes(tickformat=".0%")
    fig.update_layout(yaxis_title="", xaxis_title="", bargap=0.25, height=600)
    return fig


def plot_pie(
    df: pd.DataFrame, value_mode: Literal["valor", "percent"], title: str
) -> "px.Figure":
    fig = px.pie(df, names="categoria", values="net", title=title, hole=0.0)
    fig.update_traces(
        textposition="inside",
        textinfo="percent+label",
        hovertemplate="%{label}<br>NET: R$ %{value:,.2f} (%{percent})<extra></extra>",
    )
    fig.update_layout(height=600)
    return fig


# ----------------- Gerador de Excel consolidado (múltiplas abas) ------------ #
def _build_consolidated_xlsx(
    df_raw: pd.DataFrame,
    agg_view: pd.DataFrame,
    dimension: str,
    venc_grouping: Optional[str],
    include_all_dims: bool,
    include_diag: bool,
    colmap: dict[str, str],
    meta: dict[str, str],
    ref_date: Optional[pd.Timestamp],
) -> bytes:
    buf = BytesIO()
    writer = pd.ExcelWriter(buf, engine="xlsxwriter")
    wb = writer.book
    fmt_cur = wb.add_format({"num_format": "R$ #,##0.00"})
    fmt_pct = wb.add_format({"num_format": "0.00%"})

    # 1) Dados originais
    sheet_raw = _safe_sheet_name("Dados_Originais")
    df_raw.to_excel(writer, sheet_name=sheet_raw, index=False)
    ws_raw = writer.sheets[sheet_raw]
    ws_raw.freeze_panes(1, 0)
    ws_raw.set_column(0, max(0, len(df_raw.columns) - 1), 18)
    if "net" in df_raw.columns:
        net_idx = int(df_raw.columns.get_loc("net"))
        ws_raw.set_column(net_idx, net_idx, 18, fmt_cur)

    # 2) Agregado da visão atual
    sheet_view = _safe_sheet_name(f"Agregado_View_{dimension}")
    out_view = agg_view.rename(
        columns={
            "categoria": "Categoria",
            "net": "NET",
            "participacao": "Participacao",
        }
    )
    out_view.to_excel(writer, sheet_name=sheet_view, index=False)
    ws_view = writer.sheets[sheet_view]
    ws_view.freeze_panes(1, 0)
    ws_view.set_column(0, 0, 35)
    ws_view.set_column(1, 1, 18, fmt_cur)
    ws_view.set_column(2, 2, 14, fmt_pct)

    # 3) (Opcional) Agregações para todas as dimensões
    if include_all_dims:
        dims = [
            "Ativo",
            "Sub Produto",
            "Emissor",
            "Fator de Risco",
            "Rating",
            "Subordinação",
            "Vencimento",
        ]
        for dim in dims:
            vg = venc_grouping if dim == "Vencimento" else "Ano"
            try:
                agg_all = prepare_aggregation(
                    df=df_raw, dimension=dim, venc_grouping=vg, ref_date=ref_date
                )
            except Exception:
                continue
            out_all = agg_all.rename(
                columns={
                    "categoria": "Categoria",
                    "net": "NET",
                    "participacao": "Participacao",
                }
            )
            suffix = f"_{vg}" if dim == "Vencimento" else ""
            sheet_name = _safe_sheet_name(f"Agregado_{dim}{suffix}")
            out_all.to_excel(writer, sheet_name=sheet_name, index=False)
            ws = writer.sheets[sheet_name]
            ws.freeze_panes(1, 0)
            ws.set_column(0, 0, 35)
            ws.set_column(1, 1, 18, fmt_cur)
            ws.set_column(2, 2, 14, fmt_pct)

    # 4) (Opcional) Diagnóstico
    if include_diag:
        diag = pd.DataFrame(
            {"Original": list(colmap.keys()), "Normalizado": list(colmap.values())}
        )
        sheet_diag = _safe_sheet_name("Diagnostico_Mapeamento")
        diag.to_excel(writer, sheet_name=sheet_diag, index=False)
        ws_d = writer.sheets[sheet_diag]
        ws_d.freeze_panes(1, 0)
        ws_d.set_column(0, 1, 40)

        params = {
            "Dimensao_atual": dimension,
            "Vencimento_group": venc_grouping or "",
            "Fonte_leitura": meta.get("load_source", ""),
            "NET_total": float(df_raw["net"].sum(skipna=True)) if "net" in df_raw else 0.0,
            "Ref_date": str(ref_date.date()) if isinstance(ref_date, pd.Timestamp) else "",
        }
        sheet_par = _safe_sheet_name("Parametros")
        pd.DataFrame(list(params.items()), columns=["Parametro", "Valor"]).to_excel(
            writer, sheet_name=sheet_par, index=False
        )
        ws_p = writer.sheets[sheet_par]
        ws_p.freeze_panes(1, 0)
        ws_p.set_column(0, 0, 25)
        ws_p.set_column(1, 1, 40)

    writer.close()
    buf.seek(0)
    return buf.getvalue()


# ---------------------------- UI / Aplicação -------------------------------- #
st.title("📊 Dashboard NET por Dimensão")

with st.sidebar:
    st.header("1) Fonte de dados")
    uploaded = st.file_uploader(
        "Carregue seu arquivo .xlsx", type=["xlsx", "xls"], accept_multiple_files=False
    )

if not uploaded:
    st.info("Carregue um arquivo Excel com as colunas informadas.")
    st.stop()

data_bytes = uploaded.getvalue()
sheets = get_sheet_names(data_bytes)

with st.sidebar:
    sheet = st.selectbox("Aba (planilha):", options=sheets, index=0)

df, colmap, meta = load_sheet(data_bytes, sheet)

# Correção manual se alguma coluna não foi reconhecida
with st.sidebar:
    st.header("2) Correção de colunas (opcional)")
    if "rating" not in df.columns:
        st.info("Coluna 'Rating' não reconhecida automaticamente.")
        opt = ["(nenhuma)"] + list(df.columns)
        sel = st.selectbox("Coluna para Rating", options=opt, index=0, key="map_rate")
        if sel != "(nenhuma)":
            df["rating"] = df[sel]
    if "subordinacao" not in df.columns:
        st.info("Coluna 'Subordinação' não reconhecida automaticamente.")
        opt2 = ["(nenhuma)"] + list(df.columns)
        sel2 = st.selectbox(
            "Coluna para Subordinação", options=opt2, index=0, key="map_sub"
        )
        if sel2 != "(nenhuma)":
            df["subordinacao"] = df[sel2]
    if "fator_risco" not in df.columns:
        st.info("Coluna 'Fator de Risco' não reconhecida automaticamente.")
        opt3 = ["(nenhuma)"] + list(df.columns)
        sel3 = st.selectbox(
            "Coluna para Fator de Risco", options=opt3, index=0, key="map_fator"
        )
        if sel3 != "(nenhuma)":
            df["fator_risco"] = df[sel3]

# Reaplica normalização de Rating após possível override
if "rating" in df.columns:
    r = df["rating"].astype("string")
    is_empty = r.isna() | (r.str.strip() == "") | (r.str.upper() == "N/A")
    df.loc[is_empty, "rating"] = "Sem Rating"

# Métricas básicas
col_a, col_b, col_c = st.columns(3)
with col_a:
    st.metric("Linhas (dados carregados)", f"{len(df):,}".replace(",", "."))
with col_b:
    total_net = df["net"].sum(skipna=True) if "net" in df.columns else 0.0
    st.metric("NET Total", f"R$ {total_net:,.2f}")
with col_c:
    st.write("Colunas detectadas:")
    st.caption(", ".join(df.columns))

st.divider()

with st.sidebar:
    st.header("3) Configurações da visão")

    dimension = st.selectbox(
        "Dimensão",
        options=[
            "Ativo",
            "Sub Produto",
            "Emissor",
            "Fator de Risco",
            "Rating",
            "Subordinação",
            "Vencimento",
        ],
        index=0,
    )

    venc_grouping = None
    ref_date = None
    if dimension == "Vencimento":
        venc_grouping = st.selectbox(
            "Agrupar vencimento por", options=["Ano", "Mes/Ano", "Buckets"], index=2
        )
        if venc_grouping == "Buckets":
            ref_date = st.date_input(
                "Data de referência",
                value=pd.Timestamp.today().date(),
                help="Usada para definir os buckets (vencidos, <=1y, etc.)",
            )
            ref_date = pd.Timestamp(ref_date)

    value_mode = st.radio(
        "Mostrar", options=["Valores (R$)", "Percentual (%)"], horizontal=False, index=0
    )
    value_mode_key = "valor" if value_mode.startswith("Valores") else "percent"

    chart_type = st.radio(
        "Tipo de gráfico", options=["Barra", "Pizza"], horizontal=True, index=0
    )

    st.subheader('Agrupamento "Outros"')
    group_method = st.radio(
        "Método",
        options=[
            "Nenhum",
            "Por participação mínima (%)",
            "Top N",
        ],
        index=0,
    )
    pct_threshold = None
    top_n = None
    if group_method == "Por participação mínima (%)":
        pct_threshold = st.slider(
            "Itens com participação abaixo de (%)",
            min_value=0.1,
            max_value=10.0,
            value=1.0,
            step=0.1,
        )
    elif group_method == "Top N":
        top_n = st.slider(
            "Manter os Top N itens", min_value=3, max_value=50, value=10, step=1
        )

st.divider()

with st.expander("Diagnóstico"):
    st.caption(
        f"Fonte de leitura: {'autoheader' if meta.get('load_source') == 'autoheader' else 'standard'}"
    )
    map_df = pd.DataFrame(
        {"Original": list(colmap.keys()), "Normalizado": list(colmap.values())}
    )
    st.dataframe(map_df, use_container_width=True, hide_index=True)
    if "fator_risco" in df.columns:
        st.write("Amostra de 'Fator de Risco':")
        st.write(df["fator_risco"].astype("string").dropna().head(20).unique())
    else:
        st.write("Coluna 'fator_risco' ainda não está presente.")

# Agregação
try:
    agg = prepare_aggregation(
        df=df,
        dimension=dimension,
        venc_grouping=venc_grouping if venc_grouping else "Ano",
        ref_date=ref_date,
    )
except ValueError as e:
    st.error(str(e))
    st.stop()

# Aplica "Outros"
method_key = "none"
if group_method == "Por participação mínima (%)":
    method_key = "pct"
elif group_method == "Top N":
    method_key = "topn"

agg_final = apply_others_grouping(
    agg=agg,
    method=method_key,
    pct_threshold=(pct_threshold or 1.0) / 100.0,
    top_n=(top_n or 10),
)

# Exibição
title = f"NET por {dimension}"
if dimension == "Vencimento" and venc_grouping:
    title += f" ({venc_grouping})"

if chart_type == "Barra":
    fig = plot_bar(agg_final, value_mode=value_mode_key, title=title)
else:
    fig = plot_pie(agg_final, value_mode=value_mode_key, title=title)

st.plotly_chart(fig, use_container_width=True)

# ------------------------------- Downloads ---------------------------------- #
st.subheader("Downloads")

# Opção: Excel consolidado (múltiplas abas)
colx1, colx2 = st.columns([2, 1])
with colx1:
    include_all_dims = st.checkbox(
        "Incluir agregações para todas as dimensões no XLSX", value=False
    )
    include_diag = st.checkbox("Incluir aba de diagnóstico no XLSX", value=True)

xlsx_bytes = _build_consolidated_xlsx(
    df_raw=df,
    agg_view=agg_final,
    dimension=dimension,
    venc_grouping=venc_grouping if venc_grouping else "Ano",
    include_all_dims=include_all_dims,
    include_diag=include_diag,
    colmap=colmap,
    meta=meta,
    ref_date=ref_date,
)

xlsx_name = _safe_filename(f"NET_consolidado_{dimension}") + ".xlsx"
st.download_button(
    "Baixar Excel consolidado (XLSX)",
    data=xlsx_bytes,
    file_name=xlsx_name,
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    key="dl_xlsx_all",
)

# CSVs (mantidos como alternativa rápida)
col1, col2 = st.columns(2)
with col1:
    csv_agg = agg_final.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "Baixar dados agregados (CSV)",
        data=csv_agg,
        file_name=_safe_filename(f"NET_por_{dimension}") + ".csv",
        mime="text/csv",
        key="dl_agg",
    )

with col2:
    csv_raw = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "Baixar dados originais (CSV)",
        data=csv_raw,
        file_name="dados_originais.csv",
        mime="text/csv",
        key="dl_raw",
    )

st.caption(
    "Dica: se 'Fator de Risco' não aparecer, use a 'Correção de colunas' na "
    "sidebar. O XLSX consolidado inclui abas separadas para os dados e "
    "agregações."
)