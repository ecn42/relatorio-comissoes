import io
from typing import Dict, List, Optional, Sequence

import pandas as pd
import streamlit as st


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names: strip, lower, and replace spaces/accents.
    Returns a new dataframe with a mapping preserved in df.attrs["colmap"].
    """
    import unicodedata

    def normalize(s: str) -> str:
        s2 = (
            unicodedata.normalize("NFKD", s)
            .encode("ascii", "ignore")
            .decode("ascii")
        )
        s2 = s2.strip().lower().replace(" ", "_")
        s2 = s2.replace(";", "")  # in case headers came with delimiters
        return s2

    original = list(df.columns)
    new_cols = [normalize(c) for c in original]
    df2 = df.copy()
    df2.columns = new_cols
    df2.attrs["colmap"] = dict(zip(new_cols, original))
    return df2


def require_columns(df: pd.DataFrame, required: List[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(
            "Missing required columns after normalization: "
            + ", ".join(missing)
            + "\nFound: "
            + ", ".join(df.columns)
        )
        st.stop()


def safe_number(series: pd.Series) -> pd.Series:
    """
    Convert NET-like columns to numeric, handling commas and dots.
    """
    if series.dtype.kind in ("i", "u", "f"):
        return series
    s = (
        series.astype(str)
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False)
    )
    return pd.to_numeric(s, errors="coerce")


def aggregate_and_show(df: pd.DataFrame, by: str, net_col: str):
    st.subheader(f"Total NET by {by.replace('_', ' ').title()}")
    agg = df.groupby(by, dropna=False, observed=True)[net_col].sum().reset_index()
    agg = agg.sort_values(net_col, ascending=False)
    st.dataframe(agg, use_container_width=True)

    top_n = st.slider(
        f"Top categories for {by}", min_value=5, max_value=50, value=20, step=1
    )
    st.bar_chart(agg.set_index(by)[net_col].head(top_n))


# -------- Helpers for Análise de Crédito --------
def _normalize_text_value(s: str) -> str:
    """Uppercase, strip, remove accents, turn spaces into underscores."""
    import unicodedata

    if pd.isna(s):
        return s
    s2 = (
        unicodedata.normalize("NFKD", str(s))
        .encode("ascii", "ignore")
        .decode("ascii")
    )
    s2 = s2.strip().upper().replace(" ", "_")
    return s2


def add_produto_norm(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    if "produto" in df2.columns:
        df2["produto_norm"] = df2["produto"].map(_normalize_text_value)
    else:
        df2["produto_norm"] = pd.Series([pd.NA] * len(df2))
    return df2


def add_sub_produto_norm(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    if "sub_produto" in df2.columns:
        df2["sub_produto_norm"] = df2["sub_produto"].map(_normalize_text_value)
    else:
        df2["sub_produto_norm"] = pd.Series([pd.NA] * len(df2))
    return df2


def fill_missing_emissor(df: pd.DataFrame) -> pd.DataFrame:
    """
    If 'emissor' is missing/blank, fill with the first token of 'ativo'
    (substring up to first space). Leaves as NA if 'ativo' is missing.
    """
    df2 = df.copy()
    if "emissor" not in df2.columns:
        df2["emissor"] = pd.NA
    mask_missing = df2["emissor"].isna() | (
        df2["emissor"].astype(str).str.strip() == ""
    )
    if "ativo" in df2.columns:
        src = df2.loc[mask_missing, "ativo"]
        tokens = (
            src.where(src.notna(), "")
            .astype(str)
            .str.strip()
            .str.split()
            .str[0]
        )
        tokens = tokens.replace({"": pd.NA})
        df2.loc[mask_missing, "emissor"] = tokens
    else:
        st.warning(
            "Coluna 'Ativo' ausente; não foi possível inferir Emissor "
            "para linhas sem Emissor."
        )
    return df2


def normalize_emissor_specials(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map Emissor == 'NÃO' (accent/case-insensitive) to 'Compromissadas'.
    """
    import unicodedata

    if "emissor" not in df.columns:
        return df

    def norm(s: str) -> str:
        s2 = (
            unicodedata.normalize("NFKD", str(s))
            .encode("ascii", "ignore")
            .decode("ascii")
        )
        return s2.strip().upper()

    emis = df["emissor"]
    emis_norm = emis.astype(str).map(norm)
    mask_nao = emis_norm == "NAO"
    df.loc[mask_nao, "emissor"] = "Compromissadas"
    return df


def ensure_datetime(
    df: pd.DataFrame, col: str, dayfirst: bool = True
) -> pd.DataFrame:
    if col in df.columns:
        df = df.copy()
        df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=dayfirst)
    return df


def add_prazo_bucket(
    df: pd.DataFrame,
    pos_col: str = "data_posicao",
    venc_col: str = "data_de_vencimento",
) -> pd.DataFrame:
    """
    Compute days to maturity and bucket it.
    Uses data_posicao if available; otherwise assumes 'today'.
    """
    from datetime import datetime

    df2 = df.copy()
    df2 = ensure_datetime(df2, pos_col, dayfirst=True)
    df2 = ensure_datetime(df2, venc_col, dayfirst=True)

    today = pd.Timestamp(datetime.utcnow().date())
    if pos_col not in df2.columns or df2[pos_col].isna().all():
        pos_series = pd.Series([today] * len(df2), index=df2.index)
    else:
        pos_series = df2[pos_col].fillna(today)

    if venc_col not in df2.columns:
        df2["prazo_bucket"] = pd.Series(
            ["Sem Vencimento"] * len(df2), index=df2.index
        )
        return df2

    days = (df2[venc_col] - pos_series).dt.days

    def bucket(d):
        if pd.isna(d):
            return "Sem Vencimento"
        if d < 0:
            return "Vencido"
        if d <= 180:
            return "0-6m"
        if d <= 365:
            return "6-12m"
        if d <= 730:
            return "1-2y"
        if d <= 1095:
            return "2-3y"
        if d <= 1825:
            return "3-5y"
        return "5y+"

    df2["prazo_bucket"] = days.map(bucket)
    cat_order = [
        "Vencido",
        "0-6m",
        "6-12m",
        "1-2y",
        "2-3y",
        "3-5y",
        "5y+",
        "Sem Vencimento",
    ]
    df2["prazo_bucket"] = pd.Categorical(
        df2["prazo_bucket"], categories=cat_order, ordered=True
    )
    return df2


def dominant_product_per_subproduto(
    df: pd.DataFrame,
    sub_col: str = "sub_produto_norm",
    prod_col: str = "produto_norm",
) -> Dict[str, str]:
    """
    Map each sub_produto_norm to its dominant produto_norm (by row count).
    """
    if sub_col not in df.columns or prod_col not in df.columns:
        return {}
    tmp = (
        df.dropna(subset=[sub_col, prod_col])
        .groupby([sub_col, prod_col], dropna=False, observed=True)
        .size()
        .reset_index(name="n")
    )
    if tmp.empty:
        return {}
    dom = (
        tmp.sort_values("n", ascending=False)
        .drop_duplicates(sub_col, keep="first")
        .reset_index(drop=True)
    )
    return dict(zip(dom[sub_col], dom[prod_col]))


def exposure_table(
    df: pd.DataFrame,
    group_col: str,
    credit_total: float,
    grand_total: float,
    include_product_breakdown: bool = True,
    product_col: str = "produto_norm",
    product_order: Optional[Sequence[str]] = None,
    include_subprod_breakdown: bool = True,
    sub_produto_col: str = "sub_produto_norm",
) -> pd.DataFrame:
    """
    Returns a dataframe with columns:
    [group_col, valor, %_sobre_credito, %_sobre_total]
    Plus product breakdown and sub produto breakdown columns.
    """
    base_cols = [group_col, "valor", "%_sobre_credito", "%_sobre_total"]

    if df.empty:
        return pd.DataFrame(columns=base_cols)

    agg = (
        df.groupby(group_col, dropna=False, observed=True)["net"]
        .sum()
        .reset_index()
        .rename(columns={"net": "valor"})
    )

    if credit_total == 0:
        agg["%_sobre_credito"] = pd.NA
    else:
        agg["%_sobre_credito"] = agg["valor"] / credit_total

    if grand_total == 0:
        agg["%_sobre_total"] = pd.NA
    else:
        agg["%_sobre_total"] = agg["valor"] / grand_total

    out = agg

    # Product breakdown
    prod_cols: Sequence[str] = []
    if include_product_breakdown and product_col in df.columns:
        piv_prod = (
            df.groupby([group_col, product_col], dropna=False, observed=True)[
                "net"
            ]
            .sum()
            .unstack(fill_value=0)
        )
        if product_order is not None:
            existing = [p for p in product_order if p in piv_prod.columns]
            rest = [c for c in piv_prod.columns if c not in existing]
            piv_prod = piv_prod[existing + rest]

        out = out.merge(piv_prod.reset_index(), on=group_col, how="left")
        prod_cols = list(piv_prod.columns)
        if prod_cols:
            out[prod_cols] = out[prod_cols].fillna(0)

        if grand_total == 0:
            for p in prod_cols:
                out[f"{p}_%_sobre_total"] = pd.NA
        else:
            for p in prod_cols:
                out[f"{p}_%_sobre_total"] = out[p] / grand_total

    # Sub Produto breakdown
    if include_subprod_breakdown and sub_produto_col in df.columns:
        piv_sub = (
            df.groupby([group_col, sub_produto_col], dropna=False, observed=True)[
                "net"
            ]
            .sum()
            .unstack(fill_value=0)
        )
        sub_raw_cols = list(piv_sub.columns)
        pref_map = {c: f"sub_{c}" for c in sub_raw_cols}
        piv_sub = piv_sub.rename(columns=pref_map)

        out = out.merge(piv_sub.reset_index(), on=group_col, how="left")
        sub_cols = list(piv_sub.columns)
        if sub_cols:
            out[sub_cols] = out[sub_cols].fillna(0)

        sp_to_prod = dominant_product_per_subproduto(
            df, sub_col=sub_produto_col, prod_col=product_col
        )

        for sp_raw in sub_raw_cols:
            col_abs = f"sub_{sp_raw}"
            if credit_total == 0:
                out[f"{col_abs}_%_sobre_credito"] = pd.NA
            else:
                out[f"{col_abs}_%_sobre_credito"] = out[col_abs] / credit_total
            if grand_total == 0:
                out[f"{col_abs}_%_sobre_total"] = pd.NA
            else:
                out[f"{col_abs}_%_sobre_total"] = out[col_abs] / grand_total
            parent = sp_to_prod.get(sp_raw)
            if parent and parent in out.columns:
                denom = out[parent].replace(0, pd.NA)
                out[f"{col_abs}_%_sobre_produto"] = out[col_abs] / denom
            else:
                out[f"{col_abs}_%_sobre_produto"] = pd.NA

    out = out.sort_values("valor", ascending=False)
    return out


def show_exposure_section(
    df_credit: pd.DataFrame,
    group_col: str,
    label: str,
    credit_total: float,
    grand_total: float,
    default_top_n: int = 20,
    product_order: Optional[Sequence[str]] = (
        "FIXED_INCOME",
        "RENDA_FIXA",
        "TESOURO_DIRETO",
        "FUNDOS",
    ),
):
    st.markdown(f"### Exposição por {label}")
    if df_credit.empty or group_col not in df_credit.columns:
        st.warning(f"Coluna necessária ausente ou sem dados: {group_col}")
        return

    exp_df = exposure_table(
        df=df_credit,
        group_col=group_col,
        credit_total=credit_total,
        grand_total=grand_total,
        include_product_breakdown=True,
        product_col="produto_norm",
        product_order=product_order,
        include_subprod_breakdown=True,
        sub_produto_col="sub_produto_norm",
    )

    disp = exp_df.copy()

    def fmt_money(x):
        return f"{x:,.2f}"

    def fmt_pct(x):
        return f"{x:.2%}" if pd.notna(x) else "—"

    for col in disp.columns:
        if col == group_col:
            continue
        if (
            col.endswith("_%_sobre_total")
            or col.endswith("_%_sobre_credito")
            or col.endswith("_%_sobre_produto")
            or col in ("%_sobre_credito", "%_sobre_total")
        ):
            disp[col] = disp[col].apply(fmt_pct)
        else:
            disp[col] = disp[col].map(fmt_money)

    st.dataframe(disp, use_container_width=True)

    plot_choice = st.radio(
        f"O que mostrar no gráfico por {label}?",
        options=["Valor", "% sobre crédito", "% sobre total"],
        key=f"plot_choice_{group_col}",
        horizontal=True,
    )
    top_n = st.slider(
        f"Top categorias por {label}", 5, 50, default_top_n, 1, key=f"top_{group_col}"
    )

    plot_col = {
        "Valor": "valor",
        "% sobre crédito": "%_sobre_credito",
        "% sobre total": "%_sobre_total",
    }[plot_choice]

    series = exp_df.set_index(group_col)[plot_col].head(top_n)
    st.bar_chart(series)


def build_produto_sub_table(
    df_credit: pd.DataFrame, credit_total: float, grand_total: float
) -> pd.DataFrame:
    """
    Aggregate by Produto/Sub Produto pair and compute:
    valor, % over same Produto, % over credit, % over total.
    """
    grp = (
        df_credit.groupby(
            ["produto_norm", "sub_produto_norm", "produto", "sub_produto"],
            dropna=False,
            observed=True,
        )["net"]
        .sum()
        .reset_index()
        .rename(columns={"net": "valor"})
    )

    totals_prod = (
        df_credit.groupby("produto_norm", dropna=False)["net"].sum().rename("denom")
    )

    grp = grp.merge(totals_prod, on="produto_norm", how="left")
    grp["%_sobre_produto"] = grp["valor"] / grp["denom"].replace(0, pd.NA)

    if credit_total == 0:
        grp["%_sobre_credito"] = pd.NA
    else:
        grp["%_sobre_credito"] = grp["valor"] / credit_total

    if grand_total == 0:
        grp["%_sobre_total"] = pd.NA
    else:
        grp["%_sobre_total"] = grp["valor"] / grand_total

    p = grp["produto"].astype(str).str.strip().replace({"": "Sem Produto"})
    sp = grp["sub_produto"].astype(str).str.strip().replace({"": "Sem Sub Produto"})
    grp["Produto-Sub Produto"] = p + " - " + sp

    out = grp[
        [
            "Produto-Sub Produto",
            "produto",
            "sub_produto",
            "valor",
            "%_sobre_produto",
            "%_sobre_credito",
            "%_sobre_total",
        ]
    ].sort_values("valor", ascending=False)

    return out.reset_index(drop=True)


def show_produto_sub_view(
    df_credit: pd.DataFrame, credit_total: float, grand_total: float
):
    st.markdown("### Produto-Sub Produto")
    table = build_produto_sub_table(df_credit, credit_total, grand_total)

    disp = table.copy()

    def fmt_money(x):
        return f"{x:,.2f}"

    def fmt_pct(x):
        return f"{x:.2%}" if pd.notna(x) else "—"

    if not disp.empty:
        disp["valor"] = disp["valor"].map(fmt_money)
        for c in ["%_sobre_produto", "%_sobre_credito", "%_sobre_total"]:
            disp[c] = disp[c].apply(fmt_pct)

    st.dataframe(disp, use_container_width=True)

    plot_choice = st.radio(
        "O que mostrar no gráfico por Produto-Sub Produto?",
        options=["Valor", "% sobre produto", "% sobre crédito", "% sobre total"],
        key="plot_choice_produto_sub",
        horizontal=True,
    )
    top_n = st.slider(
        "Top Produto-Sub Produto", 5, 50, 20, 1, key="top_produto_sub"
    )

    plot_col = {
        "Valor": "valor",
        "% sobre produto": "%_sobre_produto",
        "% sobre crédito": "%_sobre_credito",
        "% sobre total": "%_sobre_total",
    }[plot_choice]

    series = table.set_index("Produto-Sub Produto")[plot_col].head(top_n)
    st.bar_chart(series)


def main():
    st.set_page_config(page_title="NET Aggregations", layout="wide")
    st.title("NET Aggregations Dashboard")

    st.markdown(
        "- Upload your Excel (.xlsx)\n"
        "- App will drop Assessor and Cliente\n"
        "- Shows total NET by Produto, Sub Produto, and Fator de Risco\n"
        "- Tab 'Análise de Crédito': FIXED INCOME, Renda Fixa, Tesouro Direto; "
        "e Fundos/Renda Fixa\n"
        "- Exposições por Emissor, Prazo, Fator de Risco, e Produto-Sub Produto\n"
        "- Em Emissor/Prazo/Fator: detalhamento por Produto e Sub Produto"
    )

    file = st.file_uploader("Upload Excel (.xlsx)", type=["xlsx"])
    if not file:
        st.info("Awaiting file upload...")
        st.stop()

    try:
        df_raw = pd.read_excel(file, engine="openpyxl")
    except Exception as e:
        st.error(f"Failed to read Excel: {e}")
        st.stop()

    if df_raw.empty:
        st.warning("The uploaded file contains no rows.")
        st.stop()

    df = normalize_columns(df_raw)

    required_cols = ["produto", "sub_produto", "fator_de_risco", "net"]
    require_columns(df, required_cols)

    drop_candidates = ["assessor", "cliente"]
    to_drop = [c for c in drop_candidates if c in df.columns]
    df = df.drop(columns=to_drop, errors="ignore")

    df["net"] = safe_number(df["net"])

    nan_count = df["net"].isna().sum()
    if nan_count > 0:
        with st.expander(
            "Rows with non-numeric NET were dropped (click to see count)"
        ):
            st.write(f"Non-numeric NET rows dropped: {nan_count}")
        df = df[df["net"].notna()]

    with st.expander("Data preview (after cleaning)"):
        st.dataframe(df.head(50), use_container_width=True)

    tab_resumo, tab_credito = st.tabs(["Resumo", "Análise de Crédito"])

    with tab_resumo:
        cols = st.columns(1)
        with cols[0]:
            total_net = df["net"].sum()
            st.metric(label="TOTAL NET", value=f"{total_net:,.2f}")
            aggregate_and_show(df, "produto", "net")
            aggregate_and_show(df, "sub_produto", "net")
            aggregate_and_show(df, "fator_de_risco", "net")

        with st.expander("Download aggregated results"):
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
                (
                    df.groupby("produto", dropna=False)["net"]
                    .sum()
                    .reset_index()
                    .to_excel(writer, index=False, sheet_name="by_produto")
                )
                (
                    df.groupby("sub_produto", dropna=False)["net"]
                    .sum()
                    .reset_index()
                    .to_excel(writer, index=False, sheet_name="by_sub_produto")
                )
                (
                    df.groupby("fator_de_risco", dropna=False)["net"]
                    .sum()
                    .reset_index()
                    .to_excel(writer, index=False, sheet_name="by_fator_de_risco")
                )
            buffer.seek(0)
            st.download_button(
                label="Download Aggregations (Excel)",
                data=buffer.getvalue(),
                file_name="aggregations.xlsx",
                mime=(
                    "application/vnd.openxmlformats-officedocument."
                    "spreadsheetml.sheet"
                ),
            )

    with tab_credito:
        st.subheader("Análise de Crédito")

        df2 = add_sub_produto_norm(add_produto_norm(df))

        # Toggle to include/exclude Fixed Income Funds from Credit analysis/export
        if hasattr(st, "toggle"):
            include_funds = st.toggle(
                "Incluir Fundos (Renda Fixa) no Crédito?", value=True
            )
        else:
            include_funds = st.checkbox(
                "Incluir Fundos (Renda Fixa) no Crédito?", value=True
            )
        st.caption(
            "Fundos de Renda Fixa estão "
            + ("incluídos" if include_funds else "excluídos")
            + " nas análises e exportações abaixo."
        )

        credit_codes = {"FIXED_INCOME", "RENDA_FIXA", "TESOURO_DIRETO"}
        mask_core_credit = df2["produto_norm"].isin(credit_codes)
        mask_funds_rf = (
            (df2["produto_norm"] == "FUNDOS")
            & (df2["sub_produto_norm"] == "RENDA_FIXA")
        )
        mask_credit = mask_core_credit | (mask_funds_rf & include_funds)
        df_credit = df2[mask_credit].copy()

        df_credit = fill_missing_emissor(df_credit)
        df_credit = normalize_emissor_specials(df_credit)
        df_credit = add_prazo_bucket(df_credit)

        total_net_all = df["net"].sum()
        total_net_credit = df_credit["net"].sum()

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("TOTAL NET (Carteira)", f"{total_net_all:,.2f}")
        with c2:
            st.metric("TOTAL NET (Crédito)", f"{total_net_credit:,.2f}")
        with c3:
            pct_credit_over_total = (
                total_net_credit / total_net_all if total_net_all != 0 else 0.0
            )
        st.metric("% Crédito / Carteira", f"{pct_credit_over_total:.2%}")

        if df_credit.empty:
            st.info(
                "Não há linhas em produtos de crédito selecionados. "
                "Verifique se o arquivo contém FIXED INCOME / Renda Fixa / "
                "Tesouro Direto e, se desejar, habilite a inclusão de "
                "Fundos de Renda Fixa no controle acima."
            )
            st.stop()

        show_exposure_section(
            df_credit=df_credit,
            group_col="emissor",
            label="Emissor",
            credit_total=total_net_credit,
            grand_total=total_net_all,
            default_top_n=20,
        )
        show_exposure_section(
            df_credit=df_credit,
            group_col="prazo_bucket",
            label="Prazo",
            credit_total=total_net_credit,
            grand_total=total_net_all,
            default_top_n=15,
        )
        show_exposure_section(
            df_credit=df_credit,
            group_col="fator_de_risco",
            label="Fator de Risco",
            credit_total=total_net_credit,
            grand_total=total_net_all,
            default_top_n=20,
        )

        show_produto_sub_view(
            df_credit=df_credit,
            credit_total=total_net_credit,
            grand_total=total_net_all,
        )

        with st.expander("Download exposições de crédito (Excel)"):
            by_emissor = exposure_table(
                df_credit,
                "emissor",
                total_net_credit,
                total_net_all,
                include_product_breakdown=True,
                product_col="produto_norm",
                include_subprod_breakdown=True,
                sub_produto_col="sub_produto_norm",
            )
            by_prazo = exposure_table(
                df_credit,
                "prazo_bucket",
                total_net_credit,
                total_net_all,
                include_product_breakdown=True,
                product_col="produto_norm",
                include_subprod_breakdown=True,
                sub_produto_col="sub_produto_norm",
            )
            by_fator = exposure_table(
                df_credit,
                "fator_de_risco",
                total_net_credit,
                total_net_all,
                include_product_breakdown=True,
                product_col="produto_norm",
                include_subprod_breakdown=True,
                sub_produto_col="sub_produto_norm",
            )
            by_prod_sub = build_produto_sub_table(
                df_credit, total_net_credit, total_net_all
            )

            buffer2 = io.BytesIO()
            with pd.ExcelWriter(buffer2, engine="xlsxwriter") as writer:
                by_emissor.to_excel(writer, index=False, sheet_name="by_emissor")
                by_prazo.to_excel(writer, index=False, sheet_name="by_prazo")
                by_fator.to_excel(
                    writer, index=False, sheet_name="by_fator_de_risco"
                )
                by_prod_sub.to_excel(
                    writer, index=False, sheet_name="by_produto_sub"
                )
            buffer2.seek(0)
            st.download_button(
                label="Download (Crédito) - Exposições",
                data=buffer2.getvalue(),
                file_name="credit_exposures.xlsx",
                mime=(
                    "application/vnd.openxmlformats-officedocument."
                    "spreadsheetml.sheet"
                ),
            )


if __name__ == "__main__":
    pd.set_option("display.max_colwidth", 200)
    main()