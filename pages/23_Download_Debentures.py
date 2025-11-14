# app.py
# Streamlit app to download the Debentures "Excel" export (actually TSV)
# from the exact URL provided and parse it into a pandas DataFrame.

from __future__ import annotations

import io
from typing import Tuple, Dict
import streamlit as st
import pandas as pd
import requests


URL = (
    "https://www.debentures.com.br/exploreosnd/consultaadados/"
    "emissoesdedebentures/caracteristicas_e.asp"
    "?tip_deb=publicas&op_exc=Nada"
)


def build_headers(referer: str | None = None) -> Dict[str, str]:
    h = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/plain,application/vnd.ms-excel,*/*;q=0.8",
        "Accept-Language": "pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7",
        "Connection": "keep-alive",
    }
    if referer:
        h["Referer"] = referer
    return h


@st.cache_data(ttl=1800, show_spinner=False)
def download_text(url: str) -> Tuple[str, Dict[str, str], str]:
    with requests.Session() as s:
        s.headers.update(build_headers(referer=url))
        r = s.get(url, timeout=60)
        r.raise_for_status()
        # Prefer a robust fallback if encoding is not declared
        enc = r.encoding or r.apparent_encoding or "latin-1"
        text = r.content.decode(enc, errors="replace")
        return text, dict(r.headers), r.url


def find_header_index(lines: list[str]) -> int:
    # Header line has lots of tabs and starts with "Codigo do Ativo"
    for i, line in enumerate(lines):
        if line.count("\t") >= 10 and line.lower().startswith("codigo do ativo"):
            return i
    # Fallback: first line with many tabs
    for i, line in enumerate(lines):
        if line.count("\t") >= 10:
            return i
    raise ValueError("Header line not found (no tab-delimited header detected).")


def parse_tsv_text_to_df(text: str) -> pd.DataFrame:
    # Normalize newlines
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = t.split("\n")

    header_idx = find_header_index(lines)
    table_text = "\n".join(lines[header_idx:])

    # Parse as TSV; keep everything as string to avoid type inference surprises
    df = pd.read_csv(
        io.StringIO(table_text),
        sep="\t",
        engine="python",
        dtype=str,
        on_bad_lines="skip",
        quoting=3,  # csv.QUOTE_NONE without importing csv
    )

    # Clean column names a bit
    df.columns = [c.strip().replace("\xa0", " ") for c in df.columns]
    # Drop fully empty rows
    df = df.dropna(how="all")
    # Strip whitespace in cells (optional; comment out if not desired)
    for c in df.columns:
        if pd.api.types.is_string_dtype(df[c]):
            df[c] = df[c].str.strip()

    return df


def main():
    st.set_page_config(page_title="Debentures → DataFrame", layout="wide")
    st.title("Debentures (TSV export) → DataFrame")

    st.caption("Baixa exatamente o URL informado e cria um DataFrame.")
    url = st.text_input("URL (não alterar):", value=URL, disabled=True)

    if st.button("Baixar e carregar", type="primary"):
        with st.spinner("Baixando conteúdo (TSV)..."):
            text, headers, final_url = download_text(url)

        st.info(
            f"Final URL: {final_url} | "
            f"Content-Type: {headers.get('Content-Type','unknown')}"
        )

        # Optional: show a short preview for debugging
        with st.expander("Prévia da resposta (primeiros ~2000 chars)"):
            st.text(text[:2000])

        with st.spinner("Convertendo para DataFrame..."):
            df = parse_tsv_text_to_df(text)

        st.success(f"DataFrame carregado: {df.shape[0]} linhas × {df.shape[1]} cols")
        st.dataframe(df, use_container_width=True)

        st.download_button(
            "Baixar CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="debentures.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()