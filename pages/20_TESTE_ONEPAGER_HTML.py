# pages/06_Generate_HTML_Factsheet.py
"""
Generate HTML Factsheet from Excel export and allow high-fidelity PDF export.
Reads the XLSX output from Export_Factsheet_Data and produces a styled HTML
factsheet matching the Ceres Wealth PDF layout. Adds a "Download PDF" button
that renders via headless Chromium (Playwright) honoring CSS @page.

Updates:
- PDF generation measures the rendered '.page' container and sets a custom
  PDF width/height, ensuring the entire factsheet prints on a single page.
- If the benchmark is IBOV (or mapping says so), the line chart uses 'ibov_cum'
  instead of 'cdi_cum' (with safe fallbacks) and labels the legend accordingly.
- Adds a second uploader to load a "Fund Catalog" Excel sheet with columns:
  [Fundos, AuM, Classe do Fundo, Benchmark, Tx. Administração,
   Tx Performance, Administrador, Resgate, Risco].
  At generation time, the app will auto-fill missing fund details using the
  catalog row matching the fund CNPJ. Any missing fields remain as defaults.
- Adds batch generation: upload many factsheet Excel files at once and
  generate all HTML/PDFs in one click. Provides batch download of all HTMLs,
  all PDFs, and the original Excel uploads as separate ZIP files.
- Adds a progress indicator during batch processing, showing which file is
  being processed and overall progress.
- Formats uploaded details for display:
  - AuM is displayed as 'R$ xxx.xxx.xxx,xx' (pt-BR style)
  - Tx. Administração and Tx. Performance come as decimals like '0,03' and
    are displayed as '3% a.a.' (values <= 1 treated as fraction of 1).
- Adds mapping dict (original_name, original_cnpj, master_cnpj, graph). For
  graphs and tables (line chart, allocation, Top 10), always use master_cnpj
  data when available; use original_name and original_cnpj at the header; and
  use original_cnpj for the fund-details table (AuM, fees, etc). For the line
  chart comparator, use the 'graph' field (CDI or IBOV).
- BATCH MODE: Uploaded sheets are keyed by master_cnpj. The mapping is used
  to find original_name/original_cnpj for display and catalog lookup.
- NEW: Adds an optional Word (.docx) uploader with a JSON array of funds info
  (by original CNPJ) to auto-fill the content sections:
  ["Descrição do Fundo", "Características Principais", "Equipe de Gestão",
   "Portfólio e Tese de Investimento", "Performance Commentary"].
  Single mode: if fields are empty, they'll be filled from Word by original CNPJ.
  Batch mode: per-fund texts are taken from Word if available; otherwise batch
  defaults apply.
"""

from __future__ import annotations

import base64
import io
import json
import math
import re
import unicodedata
import zipfile
from typing import Optional

import pandas as pd
import streamlit as st

# Authentication
if not st.session_state.get("authenticated", False):
    st.warning("Please enter the password on the Home page first.")
    st.stop()

# ---------------------- Constants ----------------------

BRAND_BROWN = "#825120"
BRAND_BROWN_DARK = "#6B4219"
LIGHT_GRAY = "#F5F5F5"
BLOCK_BG = "#F8F8F8"
TABLE_BORDER = "#E0D5CA"
TEXT_DARK = "#333333"


# ---------------------- FUND MAPPING ----------------------
# Provided mapping. We key by cleaned original_cnpj (digits only).
FUND_MAPPING_RAW = [
    {
        "original_name": "ABSOLUTE ATENAS P FICFIRF",
        "original_cnpj": "47.612.663/0001-20",
        "master_cnpj": "47.628.822/0001-85",
        "graph": "CDI",
    },
    {
        "original_name": "ABSOLUTE ATENAS PREV FICFIRF",
        "original_cnpj": "47.564.734/0001-67",
        "master_cnpj": "47.612.777/0001-70",
        "graph": "CDI",
    },
    {
        "original_name": "Absolute Creta P FICFIRF CrPr",
        "original_cnpj": "47.628.957/0001-40",
        "master_cnpj": "47.628.918/0001-43",
        "graph": "CDI",
    },
    {
        "original_name": "ABSOLUTE DELFOS FICFIRF LP CrPr",
        "original_cnpj": "52.324.414/0001-70",
        "master_cnpj": "52.324.362/0001-31",
        "graph": "CDI",
    },
    {
        "original_name": "Absolute Delfos Prev FICFIRF CrPr RL Access",
        "original_cnpj": "57.370.727/0001-88",
        "master_cnpj": "49.558.667/0001-01",
        "graph": "CDI",
    },
    {
        "original_name": "Absolute Hidra CDI Incentivado Infra FICFIRF CrPr",
        "original_cnpj": "47.612.737/0001-29",
        "master_cnpj": "47.628.885/0001-31",
        "graph": "CDI",
    },
    {
        "original_name": "Absolute Hidra CDI Plus P Incentivado Infra FICFIRF",
        "original_cnpj": "57.027.655/0001-70",
        "master_cnpj": "57.046.745/0001-09",
        "graph": "CDI",
    },
    {
        "original_name": "Absolute Hidra IPCA FIC de FIF RF",
        "original_cnpj": "51.112.363/0001-50",
        "master_cnpj": "51.112.915/0001-20",
        "graph": "CDI",
    },
    {
        "original_name": "ABSOLUTE OLIMPIA FICFIM CrPr",
        "original_cnpj": "48.986.106/0001-32",
        "master_cnpj": "48.986.296/0001-98",
        "graph": "CDI",
    },
    {
        "original_name": "Absolute Pace Long Biased FICFIA",
        "original_cnpj": "32.073.525/0001-43",
        "master_cnpj": "32.071.783/0001-90",
        "graph": "IBOV",
    },
    {
        "original_name": "Artesanal Credito Privado 30 FICFIM",
        "original_cnpj": "45.830.738/0001-14",
        "master_cnpj": "45.830.738/0001-14",
        "graph": "CDI",
    },
    {
        "original_name": "Artesanal Credito Privado 90 FICFIDC RL",
        "original_cnpj": "20.441.301/0001-68",
        "master_cnpj": "20.441.301/0001-68",
        "graph": "CDI",
    },
    {
        "original_name": "Artesanal Renda Fixa FIRF",
        "original_cnpj": "24.773.832/0001-09",
        "master_cnpj": "24.773.832/0001-09",
        "graph": "CDI",
    },
    {
        "original_name": "AZ Quest Luce FIC FIF RF CP LP",
        "original_cnpj": "23.556.185/0001-10",
        "master_cnpj": "23.556.204/0001-09",
        "graph": "CDI",
    },
    {
        "original_name": "AZ Quest Valore FIRF CrPr",
        "original_cnpj": "19.782.311/0001-88",
        "master_cnpj": "19.782.311/0001-88",
        "graph": "CDI",
    },
    {
        "original_name": "BNP Paribas Match DI FIRF RL",
        "original_cnpj": "09.636.393/0001-07",
        "master_cnpj": "09.636.393/0001-07",
        "graph": "CDI",
    },
    {
        "original_name": "BNP Paribas Targus FIC FIRF CrPr",
        "original_cnpj": "05.862.906/0001-39",
        "master_cnpj": "12.107.669/0001-66",
        "graph": "CDI",
    },
    {
        "original_name": "Bradesco Debentures Incentivadas Infra CDI FICFIRF",
        "original_cnpj": "32.388.012/0001-21",
        "master_cnpj": "32.388.012/0001-21",
        "graph": "CDI",
    },
    {
        "original_name": "Bradesco Sky DI FICFIRFRef",
        "original_cnpj": "04.831.907/0001-53",
        "master_cnpj": "32.312.120/0001-10",
        "graph": "CDI",
    },
    {
        "original_name": "Bradesco Ultra FICFIRF CrPr LP",
        "original_cnpj": "37.532.786/0001-06",
        "master_cnpj": "30.392.659/0001-00",
        "graph": "CDI",
    },
    {
        "original_name": "Bradesco Zupo FIC FIRF LP CP",
        "original_cnpj": "34.109.809/0001-78",
        "master_cnpj": "32.388.077/0001-77",
        "graph": "CDI",
    },
    {
        "original_name": "BTG CDB Plus FIRF",
        "original_cnpj": "27.717.359/0001-30",
        "master_cnpj": "27.717.359/0001-30",
        "graph": "CDI",
    },
    {
        "original_name": "BTG CDB PLUS PREV FI CP",
        "original_cnpj": "47.564.915/0001-93",
        "master_cnpj": "47.564.915/0001-93",
        "graph": "CDI",
    },
    {
        "original_name": "BTG Cred Corp FICFIRF CrPr",
        "original_cnpj": "14.171.644/0001-57",
        "master_cnpj": "14.557.317/0001-38",
        "graph": "CDI",
    },
    {
        "original_name": "BTG Cred Corp Plus FIM CrPr",
        "original_cnpj": "33.599.991/0001-20",
        "master_cnpj": "33.599.991/0001-20",
        "graph": "CDI",
    },
    {
        "original_name": "BTG Debentures Incentivadas Infra CDI FICFIRF",
        "original_cnpj": "30.934.757/0001-13",
        "master_cnpj": "31.094.141/0001-44",
        "graph": "CDI",
    },
    {
        "original_name": "BTG Multigestor Global Equities BRL FIA IE",
        "original_cnpj": "41.287.933/0001-99",
        "master_cnpj": "41.287.933/0001-99",
        "graph": "IBOV",
    },
    {
        "original_name": "BTG Pactual Tesouro IPCA Curto FIRF",
        "original_cnpj": "07.539.298/0001-51",
        "master_cnpj": "07.539.298/0001-51",
        "graph": "CDI",
    },
    {
        "original_name": "BTG Reference Ouro USD FICFIM",
        "original_cnpj": "36.727.910/0001-18",
        "master_cnpj": "34.979.818/0001-10",
        "graph": "CDI",
    },
    {
        "original_name": "BTG S&P 500 BRL FIM",
        "original_cnpj": "36.499.594/0001-74",
        "master_cnpj": "36.499.594/0001-74",
        "graph": "IBOV",
    },
    {
        "original_name": "BTG Yield DI FIRF Ref CrPr",
        "original_cnpj": "00.840.011/0001-80",
        "master_cnpj": "00.840.011/0001-80",
        "graph": "CDI",
    },
    {
        "original_name": "Constellation Inovação FIF em Cotas de FIA",
        "original_cnpj": "36.352.596/0001-36",
        "master_cnpj": "42.870.670/0001-09",
        "graph": "IBOV",
    },
    {
        "original_name": "Daycoval Classic FIRF CP",
        "original_cnpj": "10.783.480/0001-68",
        "master_cnpj": "53.802.999/0001-59",
        "graph": "CDI",
    },
    {
        "original_name": "Galloway Latam Bonds BRL FIM CrPr IE",
        "original_cnpj": "21.732.619/0001-60",
        "master_cnpj": "21.732.619/0001-60",
        "graph": "CDI",
    },
    {
        "original_name": "Genoa Arpa FICFIM",
        "original_cnpj": "37.495.383/0001-26",
        "master_cnpj": "37.487.877/0001-69",
        "graph": "IBOV",
    },
    {
        "original_name": "GENOA CAPITAL CRUISE BTG PREV FICFIM",
        "original_cnpj": "42.535.251/0001-10",
        "master_cnpj": "41.575.755/0001-00",
        "graph": "CDI",
    },
    {
        "original_name": "Genoa Capital Vestas B FICFIM",
        "original_cnpj": "49.240.295/0001-62",
        "master_cnpj": "48.966.484/0001-54",
        "graph": "CDI",
    },
    {
        "original_name": "GENOA RADAR FICFIM ACCESS",
        "original_cnpj": "37.287.419/0001-86",
        "master_cnpj": "35.828.652/0001-01",
        "graph": "CDI",
    },
    {
        "original_name": "GENOA SAGRES FICFIM",
        "original_cnpj": "51.012.457/0001-57",
        "master_cnpj": "51.012.134/0001-63",
        "graph": "CDI",
    },
    {
        "original_name": "Hashdex 100 Nasdaq Crypto Index FIM",
        "original_cnpj": "33.736.845/0001-07",
        "master_cnpj": "38.314.708/0001-90",
        "graph": "CDI",
    },
    {
        "original_name": "Ibiuna Credit Deb Inc Infra RF Subclasse",
        "original_cnpj": "54.518.925/0001-58",
        "master_cnpj": "54.518.925/0001-58",
        "graph": "CDI",
    },
    {
        "original_name": "Investo Previdencia Global FIM",
        "original_cnpj": "61.928.582/0001-65",
        "master_cnpj": "61.928.582/0001-65",
        "graph": "CDI",
    },
    {
        "original_name": "IP PARTICIPACOES FICFIA BDR NI",
        "original_cnpj": "29.544.764/0001-20",
        "master_cnpj": "11.435.298/0001-89",
        "graph": "IBOV",
    },
    {
        "original_name": "Itau Action Debentures Incentivadas Infra Dist FICFIRF CrPr LP",
        "original_cnpj": "42.826.922/0001-00",
        "master_cnpj": "42.378.579/0001-70",
        "graph": "CDI",
    },
    {
        "original_name": "Itau Artax Dist FICFIM RL",
        "original_cnpj": "44.983.745/0001-93",
        "master_cnpj": "42.717.511/0001-79",
        "graph": "CDI",
    },
    {
        "original_name": "Itau Debentures Incentivadas Infra CDI Dist FICFIRF CrPr",
        "original_cnpj": "45.512.145/0001-00",
        "master_cnpj": "42.411.555/0001-76",
        "graph": "CDI",
    },
    {
        "original_name": "Itau Global Dinamico RF Dist BTG Prev FICFI LP",
        "original_cnpj": "50.235.964/0001-97",
        "master_cnpj": "39.566.756/0001-38",
        "graph": "CDI",
    },
    {
        "original_name": "Itau Global Dinamico Ultra Dist FIM RL",
        "original_cnpj": "52.026.419/0001-16",
        "master_cnpj": "42.332.169/0001-99",
        "graph": "CDI",
    },
    {
        "original_name": "Itaú Hedge Plus Multimercado FIF",
        "original_cnpj": "29.993.583/0001-80",
        "master_cnpj": "11.419.627/0001-06",
        "graph": "CDI",
    },
    {
        "original_name": "ITAU HIGH YIELD PREV FICFIRF CPLP",
        "original_cnpj": "42.827.330/0001-03",
        "master_cnpj": "42.814.813/0001-65",
        "graph": "CDI",
    },
    {
        "original_name": "Itau Janeiro Flex Prev Vertice Multimercado Access",
        "original_cnpj": "51.102.626/0001-40",
        "master_cnpj": "51.769.756/0001-30",
        "graph": "CDI",
    },
    {
        "original_name": "ITAU JANEIRO RF PREV FICFI ACCESS",
        "original_cnpj": "54.254.287/0001-05",
        "master_cnpj": "51.768.648/0001-43",
        "graph": "CDI",
    },
    {
        "original_name": "Itaú Optimus Titan FIF Mult",
        "original_cnpj": "35.727.276/0001-50",
        "master_cnpj": "40.671.799/0001-62",
        "graph": "CDI",
    },
    {
        "original_name": "ITAU SINFONIA PREV FICFIM CRPR RL ACCESS",
        "original_cnpj": "54.304.929/0001-33",
        "master_cnpj": "54.643.280/0001-85",
        "graph": "CDI",
    },
    {
        "original_name": "Jive Bossanova HY FICFIDC",
        "original_cnpj": "46.322.919/0001-00",
        "master_cnpj": "46.322.919/0001-00",
        "graph": "CDI",
    },
    {
        "original_name": "KAPITALO K10 FIC FIM",
        "original_cnpj": "43.984.537/0001-46",
        "master_cnpj": "29.726.117/0001-39",
        "graph": "CDI",
    },
    {
        "original_name": "KAPITALO K10 GLOBAL PREV FICFIM",
        "original_cnpj": "41.498.482/0001-39",
        "master_cnpj": "41.575.707/0001-03",
        "graph": "CDI",
    },
    {
        "original_name": "Kinea Alpes Prev FICFIRF CP RL",
        "original_cnpj": "55.019.754/0001-85",
        "master_cnpj": "41.756.159/0001-18",
        "graph": "CDI",
    },
    {
        "original_name": "Kinea Andes Subclasse I FIRF CrPr LP",
        "original_cnpj": "41.993.797/0001-52",
        "master_cnpj": "41.978.506/0001-57",
        "graph": "CDI",
    },
    {
        "original_name": "Kinea Atlas FIM",
        "original_cnpj": "29.762.315/0001-58",
        "master_cnpj": "29.762.315/0001-58",
        "graph": "CDI",
    },
    {
        "original_name": "Kinea Chronos FIM RL",
        "original_cnpj": "21.624.757/0001-26",
        "master_cnpj": "21.624.757/0001-26",
        "graph": "CDI",
    },
    {
        "original_name": "Kinea IPCA Dinâmico II FIF RF RL",
        "original_cnpj": "39.586.858/0001-15",
        "master_cnpj": "39.586.858/0001-15",
        "graph": "CDI",
    },
    {
        "original_name": "Legacy Capital FIC FIM",
        "original_cnpj": "30.338.679/0001-94",
        "master_cnpj": "29.236.556/0001-63",
        "graph": "CDI",
    },
    {
        "original_name": "Legacy Capital Compound Deb Inc Infra CDI FIRF CP",
        "original_cnpj": "50.891.130/0001-30",
        "master_cnpj": "50.891.130/0001-30",
        "graph": "CDI",
    },
    {
        "original_name": "MAN GLG HIGH YIELD OPPORTUNITIES BRL FICFIM IE",
        "original_cnpj": "41.594.835/0001-02",
        "master_cnpj": "41.888.295/0001-61",
        "graph": "CDI",
    },
    {
        "original_name": "Mapfre RF FI",
        "original_cnpj": "07.906.349/0001-36",
        "master_cnpj": "07.906.349/0001-36",
        "graph": "CDI",
    },
    {
        "original_name": "Oaktree Credit FICFIM IE Access",
        "original_cnpj": "30.338.686/0001-96",
        "master_cnpj": "29.363.886/0001-10",
        "graph": "CDI",
    },
    {
        "original_name": "Opportunity Global Equity BRL FICFIA IE BDR NI",
        "original_cnpj": "46.351.969/0001-08",
        "master_cnpj": "46.372.615/0001-40",
        "graph": "IBOV",
    },
    {
        "original_name": "Pimco Global Financeiro Credit FICFIM IE",
        "original_cnpj": "29.066.331/0001-06",
        "master_cnpj": "29.066.638/0001-07",
        "graph": "CDI",
    },
    {
        "original_name": "Pimco Income FICFIM IE",
        "original_cnpj": "23.729.512/0001-99",
        "master_cnpj": "23.720.107/0001-00",
        "graph": "CDI",
    },
    {
        "original_name": "Porto Ipe FICFIRF CrPr LP",
        "original_cnpj": "35.378.376/0001-19",
        "master_cnpj": "35.377.796/0001-80",
        "graph": "CDI",
    },
    {
        "original_name": "Porto Seguro FIRF Referenciado DI CP",
        "original_cnpj": "18.719.154/0001-01",
        "master_cnpj": "18.719.154/0001-01",
        "graph": "CDI",
    },
    {
        "original_name": "Principal Global High Yield FIM IE",
        "original_cnpj": "17.302.010/0001-84",
        "master_cnpj": "17.302.010/0001-84",
        "graph": "CDI",
    },
    {
        "original_name": "Real Investor Credito Estruturado 30 FICFIDC",
        "original_cnpj": "50.251.934/0001-74",
        "master_cnpj": "50.251.934/0001-74",
        "graph": "CDI",
    },
    {
        "original_name": "Real Investor Credito Estruturado 90 FICFIDC",
        "original_cnpj": "46.521.568/0001-59",
        "master_cnpj": "46.521.568/0001-59",
        "graph": "CDI",
    },
    {
        "original_name": "Real Investor FIC de FIF Multimercado",
        "original_cnpj": "28.911.549/0001-57",
        "master_cnpj": "42.870.882/0001-96",
        "graph": "CDI",
    },
    {
        "original_name": "Real Investor FIC FIA BDR Nível I",
        "original_cnpj": "10.500.884/0001-05",
        "master_cnpj": "36.352.539/0001-57",
        "graph": "IBOV",
    },
    {
        "original_name": "Riza Lotus FIF RF - Referenciada DI CP RL",
        "original_cnpj": "36.498.670/0001-27",
        "master_cnpj": "36.498.670/0001-27",
        "graph": "CDI",
    },
    {
        "original_name": "Riza Lotus Prev FICFIRF CrP",
        "original_cnpj": "56.954.063/0001-31",
        "master_cnpj": "45.646.230/0001-60",
        "graph": "CDI",
    },
    {
        "original_name": "Riza Statheros FICFIM CrPr",
        "original_cnpj": "42.260.903/0001-51",
        "master_cnpj": "42.260.903/0001-51",
        "graph": "CDI",
    },
    {
        "original_name": "Safra Agilité FIRF CP",
        "original_cnpj": "12.796.232/0001-87",
        "master_cnpj": "12.796.232/0001-87",
        "graph": "CDI",
    },
    {
        "original_name": "Safra DI Master FIRF Referenciado DI LP",
        "original_cnpj": "02.536.364/0001-16",
        "master_cnpj": "02.536.364/0001-16",
        "graph": "CDI",
    },
    {
        "original_name": "Solis Antares FIC FIDC",
        "original_cnpj": "13.054.728/0001-48",
        "master_cnpj": "13.054.656/0001-39",
        "graph": "CDI",
    },
    {
        "original_name": "Solis Antares Light CrPr FICFIDC",
        "original_cnpj": "34.780.531/0001-66",
        "master_cnpj": "34.780.531/0001-66",
        "graph": "CDI",
    },
    {
        "original_name": "Sparta Debêntures Incentivadas",
        "original_cnpj": "39.959.025/0001-52",
        "master_cnpj": "39.723.106/0001-59",
        "graph": "CDI",
    },
    {
        "original_name": "SPARTA DEBENTURES INCENTIVADAS INFRA CDI FICFIRF",
        "original_cnpj": "26.759.909/0001-11",
        "master_cnpj": "30.676.315/0001-14",
        "graph": "CDI",
    },
    {
        "original_name": "Sparta Top Inflação FIC FIF RF CP LP",
        "original_cnpj": "38.026.926/0001-29",
        "master_cnpj": "38.026.869/0001-88",
        "graph": "CDI",
    },
    {
        "original_name": "SPX Patriot FIC FIA",
        "original_cnpj": "15.334.585/0001-53",
        "master_cnpj": "15.350.712/0001-08",
        "graph": "IBOV",
    },
    {
        "original_name": "SulAmerica Excellence FIRF LP",
        "original_cnpj": "04.899.128/0001-90",
        "master_cnpj": "04.899.128/0001-90",
        "graph": "CDI",
    },
    {
        "original_name": "SulAmérica Exclusive FIRF CP",
        "original_cnpj": "04.839.017/0001-98",
        "master_cnpj": "04.839.017/0001-98",
        "graph": "CDI",
    },
    {
        "original_name": "Tivio Institucional FIRF CrPR",
        "original_cnpj": "06.866.051/0001-87",
        "master_cnpj": "06.866.051/0001-87",
        "graph": "CDI",
    },
    {
        "original_name": "Verde AM Prev FIM",
        "original_cnpj": "23.339.936/0001-47",
        "master_cnpj": "23.339.924/0001-12",
        "graph": "CDI",
    },
    {
        "original_name": "Vinci Credito Estruturado Seleção FICFIDC",
        "original_cnpj": "22.282.621/0001-48",
        "master_cnpj": "22.282.621/0001-48",
        "graph": "CDI",
    },
    {
        "original_name": "Vinland 2 Debentures Incentivadas Infra Ativo FIRF CrPr LP Subclasse I",
        "original_cnpj": "61.543.507/0001-86",
        "master_cnpj": "61.543.507/0001-86",
        "graph": "CDI",
    },
    {
        "original_name": "Vinland Credito High Grade FICFIM CrPr",
        "original_cnpj": "47.684.408/0001-93",
        "master_cnpj": "47.670.400/0001-78",
        "graph": "CDI",
    },
    {
        "original_name": "VINLAND CREDITO P PREV T1 FIC FIM CRPR",
        "original_cnpj": "51.154.059/0001-75",
        "master_cnpj": "49.426.637/0001-33",
        "graph": "CDI",
    },
    {
        "original_name": "Vinland Debentures Incentivado Infra IPCA FIRF Subclasse I",
        "original_cnpj": "57.075.698/0001-21",
        "master_cnpj": "57.075.698/0001-21",
        "graph": "CDI",
    },
    {
        "original_name": "Vinland Incentivado Debêntures Infra Ativo FIF RF CP",
        "original_cnpj": "50.862.124/0001-54",
        "master_cnpj": "50.862.124/0001-54",
        "graph": "CDI",
    },
    {
        "original_name": "Vinland Long Short FICFIM Sub I",
        "original_cnpj": "58.650.505/0001-81",
        "master_cnpj": "58.650.505/0001-81",
        "graph": "CDI",
    },
    {
        "original_name": "Vinland Macro FICFIM",
        "original_cnpj": "29.148.660/0001-04",
        "master_cnpj": "28.581.145/0001-42",
        "graph": "CDI",
    },
    {
        "original_name": "Vinland Macro Plus FICFIM",
        "original_cnpj": "47.212.318/0001-08",
        "master_cnpj": "28.692.461/0001-91",
        "graph": "CDI",
    },
    {
        "original_name": "VINLAND RF ATIVO FICFI LP",
        "original_cnpj": "34.687.428/0001-76",
        "master_cnpj": "34.687.399/0001-42",
        "graph": "CDI",
    },
    {
        "original_name": "Western Asset Total Credit FIF RF CP",
        "original_cnpj": "28.320.857/0001-08",
        "master_cnpj": "28.320.857/0001-08",
        "graph": "CDI",
    },
    {
        "original_name": "Western Credito Bancario Plus FIRF",
        "original_cnpj": "49.983.964/0001-96",
        "master_cnpj": "49.983.964/0001-96",
        "graph": "CDI",
    },
    {
        "original_name": "WHG Global Long Biased BRL FICFIA IE",
        "original_cnpj": "41.409.761/0001-89",
        "master_cnpj": "40.921.195/0001-27",
        "graph": "IBOV",
    },
]


# ---------------------- Utility: Text/CNPJ normalization ----------------------


def strip_accents(text: str) -> str:
    if text is None:
        return ""
    return "".join(
        c
        for c in unicodedata.normalize("NFKD", str(text))
        if not unicodedata.combining(c)
    )


def norm_key(text: str) -> str:
    t = strip_accents(str(text)).lower()
    keep = []
    for ch in t:
        if ch.isalnum():
            keep.append(ch)
    return "".join(keep)


def clean_cnpj(text: str) -> str:
    if text is None:
        return ""
    return "".join(ch for ch in str(text) if ch.isdigit())


def is_missing(val) -> bool:
    if val is None:
        return True
    s = str(val).strip()
    if s == "":
        return True
    s_up = strip_accents(s).upper()
    return s_up in {"#N/D", "N/D", "NA", "NAN", "NONE"}


def risk_to_level(val: Optional[str], default: str = "MODERADO") -> str:
    if is_missing(val):
        return default
    v = strip_accents(str(val)).upper()
    if "CONSERV" in v:
        return "CONSERVADOR"
    if "MODER" in v:
        return "MODERADO"
    if "SOFIST" in v or "ARROJ" in v:
        return "SOFISTICADO"
    return default


def sanitize_filename(name: str) -> str:
    """Remove or replace characters that are invalid in filenames."""
    # Replace problematic characters with underscores
    sanitized = re.sub(r'[<>:"/\\|?*]', "_", name)
    # Replace multiple spaces/underscores with single underscore
    sanitized = re.sub(r"[\s_]+", "_", sanitized)
    # Remove leading/trailing underscores and spaces
    sanitized = sanitized.strip("_ ")
    return sanitized


def build_fund_map_by_original() -> dict[str, dict]:
    """Build mapping keyed by cleaned original_cnpj."""
    mp: dict[str, dict] = {}
    for row in FUND_MAPPING_RAW:
        oc = clean_cnpj(row.get("original_cnpj", ""))
        if not oc:
            continue
        mp[oc] = {
            "original_name": str(row.get("original_name", ""))
            .replace("\u00a0", " ")
            .strip(),
            "original_cnpj": str(row.get("original_cnpj", ""))
            .replace("\u00a0", " ")
            .strip(),
            "master_cnpj": str(row.get("master_cnpj", ""))
            .replace("\u00a0", " ")
            .strip(),
            "graph": str(row.get("graph", "CDI")).strip().upper(),
        }
    return mp


def build_fund_map_by_master() -> dict[str, dict]:
    """Build mapping keyed by cleaned master_cnpj."""
    mp: dict[str, dict] = {}
    for row in FUND_MAPPING_RAW:
        mc = clean_cnpj(row.get("master_cnpj", ""))
        if not mc:
            continue
        mp[mc] = {
            "original_name": str(row.get("original_name", ""))
            .replace("\u00a0", " ")
            .strip(),
            "original_cnpj": str(row.get("original_cnpj", ""))
            .replace("\u00a0", " ")
            .strip(),
            "master_cnpj": str(row.get("master_cnpj", ""))
            .replace("\u00a0", " ")
            .strip(),
            "graph": str(row.get("graph", "CDI")).strip().upper(),
        }
    return mp


FUND_MAP_BY_ORIGINAL = build_fund_map_by_original()
FUND_MAP_BY_MASTER = build_fund_map_by_master()


def get_fund_mapping_by_original(cnpj: str) -> Optional[dict]:
    """Look up mapping by original_cnpj."""
    return FUND_MAP_BY_ORIGINAL.get(clean_cnpj(cnpj))


def get_fund_mapping_by_master(cnpj: str) -> Optional[dict]:
    """Look up mapping by master_cnpj."""
    return FUND_MAP_BY_MASTER.get(clean_cnpj(cnpj))


# ---------------------- Number parsing/formatting (pt-BR) ----------------------


def parse_br_number(val) -> Optional[float]:
    """
    Parse numbers like '20.617.549.200,98' or '0,03' into float.
    Also handles values that are already numeric (from pandas).
    Returns None if parsing fails or value is missing/#N/D.
    """
    if is_missing(val):
        return None

    # If it's already a number (int or float), return directly
    if isinstance(val, (int, float)):
        # Check for pandas NaN
        try:
            if pd.isna(val):
                return None
        except (TypeError, ValueError):
            pass
        return float(val)

    s = str(val).strip()

    # Check for NaN string representations
    if s.lower() in ("nan", "none", ""):
        return None

    s = (
        s.replace("R$", "")
        .replace("%", "")
        .replace(" ", "")
        .replace("\u00a0", "")
        .replace("\u202f", "")
        .strip()
    )

    if not s:
        return None

    # Handle scientific notation (e.g., "6,36E+09" or "6.36E+09")
    if "e" in s.lower():
        # Replace comma with dot for scientific notation
        s = s.replace(",", ".")
        try:
            return float(s)
        except Exception:
            return None

    # Check if it looks like a US-formatted number (has dot but no comma)
    has_comma = "," in s
    has_dot = "." in s

    if has_dot and not has_comma:
        # Could be US decimal (0.006) or pt-BR thousands (1.000)
        #
        # pt-BR thousands pattern: starts with 1-9, then groups of .XXX
        # Examples: 1.000, 12.345, 123.456.789
        #
        # NOT pt-BR thousands: 0.006 (starts with 0), 1.23 (not 3 digits after dot)
        # These are US decimal format.

        import re as _re

        # pt-BR thousands must start with 1-9, not 0
        ptbr_thousands_pattern = r"^[1-9]\d{0,2}(\.\d{3})+$"
        if _re.match(ptbr_thousands_pattern, s):
            # It's pt-BR thousands format like "1.000" or "12.345.678"
            s = s.replace(".", "")
            try:
                return float(s)
            except Exception:
                return None
        else:
            # It's US decimal format like "0.006" or "1.5"
            try:
                return float(s)
            except Exception:
                return None

    # Standard pt-BR format with comma as decimal separator
    # Remove thousands '.' and convert ',' to '.'
    s = s.replace(".", "").replace(",", ".")
    try:
        return float(s)
    except Exception:
        return None


def format_ptbr_number(value: float, decimals: int = 2, trim: bool = True) -> str:
    """
    Format a float as pt-BR number with '.' thousands and ',' decimal.
    If trim is True, trailing zeros and trailing comma are removed.
    """
    s = f"{value:,.{decimals}f}"
    s = s.replace(",", "X").replace(".", ",").replace("X", ".")
    if trim:
        if "," in s:
            s = s.rstrip("0").rstrip(",")
    return s


def format_brl_currency_from_any(val) -> str:
    v = parse_br_number(val)
    if v is None:
        return "-"
    return f"R$ {format_ptbr_number(v, 2, trim=False)}"


def format_percent_aa_from_any(val) -> str:
    """
    Accepts values like '0,03' or '3', returns '3% a.a.'.
    Values <= 1 are interpreted as fraction of 1 (multiply by 100).
    """
    v = parse_br_number(val)
    if v is None:
        return "-"
    pct = v * 100 if v <= 1 else v
    # Keep up to 2 decimals, but trim trailing zeros
    txt = format_ptbr_number(pct, 2, trim=True)
    return f"{txt}% a.a."


def format_fund_details_for_display(details: dict) -> dict:
    """
    Returns a copy of details formatted for display:
    - AuM -> 'R$ x.xxx.xxx,xx'
    - Tx. Administração -> 'x% a.a.'
    - Tx. Performance -> 'x% a.a.'
    Other fields passed through or '-' if missing.
    """
    if details is None:
        return {}
    out = dict(details)
    out["AuM"] = format_brl_currency_from_any(details.get("AuM", ""))
    out["Tx. Administração"] = format_percent_aa_from_any(
        details.get("Tx. Administração", "")
    )
    out["Tx. Performance"] = format_percent_aa_from_any(
        details.get("Tx. Performance", "")
    )
    for k in ["Classe do Fundo", "Benchmark", "Administrador", "Resgate"]:
        val = out.get(k, "")
        if is_missing(val):
            out[k] = "-"
        else:
            out[k] = str(val).replace("\u00a0", " ").strip()
    return out


# ---------------------- Helper Functions ----------------------


def read_excel_sheets(uploaded_file) -> dict[str, pd.DataFrame]:
    """Read all sheets from uploaded Excel file."""
    xls = pd.ExcelFile(uploaded_file)
    sheets = {}
    for name in xls.sheet_names:
        if name == "meta":
            sheets[name] = pd.read_excel(xls, sheet_name=name, header=None)
        else:
            sheets[name] = pd.read_excel(xls, sheet_name=name)
    return sheets


def parse_meta(df: pd.DataFrame) -> dict:
    """Parse meta sheet into dictionary. Expects header=None format."""
    meta = {}
    if not df.empty and len(df.columns) >= 2:
        try:
            fund_name_val = df.iloc[0, 1]
            if pd.notna(fund_name_val):
                meta["fund_name"] = str(fund_name_val)
        except (IndexError, KeyError):
            pass
    for _, row in df.iterrows():
        key = str(row.iloc[0]).strip().upper() if pd.notna(row.iloc[0]) else ""
        val = row.iloc[1] if len(row) > 1 and pd.notna(row.iloc[1]) else ""
        if "CNPJ" in key:
            meta["cnpj"] = str(val)
        elif "SUBCLASSE" in key:
            meta["subclasse"] = str(val)
        elif "SNAPSHOT" in key:
            meta["snapshot_date"] = str(val)
        elif "SÉRIE" in key or "INF_DIARIO" in key or "RANGE" in key:
            meta["series_range"] = str(val)
        elif "COMPARADOR" in key or "BENCHMARK" in key:
            meta["benchmark"] = str(val)
    return meta


def format_pct(val, decimals: int = 2) -> str:
    if pd.isna(val):
        return "-"
    try:
        return f"{float(val) * 100:.{decimals}f}%"
    except (ValueError, TypeError):
        return str(val)


def format_number(val, decimals: int = 2) -> str:
    if pd.isna(val):
        return "-"
    try:
        return f"{float(val):.{decimals}f}"
    except (ValueError, TypeError):
        return str(val)


def clean_string_value(val) -> str:
    if pd.isna(val) or val is None:
        return ""
    str_val = str(val).strip()
    if str_val.lower() == "nan":
        return ""
    return str_val


def image_to_base64(uploaded_file) -> str:
    if uploaded_file is None:
        return ""
    bytes_data = uploaded_file.getvalue()
    base64_str = base64.b64encode(bytes_data).decode()
    return base64_str


def get_image_mime_type(filename: str) -> str:
    filename = filename.lower()
    if filename.endswith(".png"):
        return "image/png"
    elif filename.endswith(".jpg") or filename.endswith(".jpeg"):
        return "image/jpeg"
    elif filename.endswith(".svg"):
        return "image/svg+xml"
    elif filename.endswith(".gif"):
        return "image/gif"
    elif filename.endswith(".webp"):
        return "image/webp"
    return "image/png"


def generate_line_chart_svg(
    df: pd.DataFrame,
    comparator: Optional[str] = None,
    benchmark: Optional[str] = None,
) -> str:
    """
    Generate SVG line chart from line_index data with legend at bottom.

    Comparator priority:
      1) comparator (explicit), accepted values: 'IBOV', 'CDI'
      2) benchmark (meta), prefer IBOV if contains 'IBOV', else CDI

    Uses 'fundo_cum' and chosen 'ibov_cum' or 'cdi_cum' (with safe fallbacks).
    """
    if df.empty or "date" not in df.columns:
        return "<p>No chart data available</p>"

    fund_col = "fundo_cum" if "fundo_cum" in df.columns else "Fundo"

    comp = (comparator or "").strip().upper()
    if comp not in {"IBOV", "CDI"}:
        bmk_str = (benchmark or "").upper()
        comp = "IBOV" if "IBOV" in bmk_str else "CDI"

    def first_existing(cols: list[str]) -> Optional[str]:
        for c in cols:
            if c in df.columns:
                return c
        return None

    if comp == "IBOV":
        bench_col = first_existing(["ibov_cum", "IBOV", "ibov"])
        bench_label = "ibov"
    else:
        bench_col = first_existing(["cdi_cum", "CDI", "cdi"])
        bench_label = "cdi"

    if bench_col is None:
        # Fallback to the other comparator if preferred not found
        if comp == "IBOV":
            bench_col = first_existing(["cdi_cum", "CDI", "cdi"])
            bench_label = "cdi" if bench_col else "bench"
        else:
            bench_col = first_existing(["ibov_cum", "IBOV", "ibov"])
            bench_label = "ibov" if bench_col else "bench"

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    width, height = 700, 200
    margin = {"top": 10, "right": 10, "bottom": 50, "left": 35}
    plot_w = width - margin["left"] - margin["right"]
    plot_h = height - margin["top"] - margin["bottom"]

    fund_vals = pd.to_numeric(df.get(fund_col, pd.Series()), errors="coerce").fillna(0)
    if bench_col:
        bench_vals = pd.to_numeric(
            df.get(bench_col, pd.Series()), errors="coerce"
        ).fillna(0)
    else:
        bench_vals = pd.Series([0] * len(df))

    all_vals = pd.concat([fund_vals, bench_vals])
    y_min = float(all_vals.min()) if not all_vals.empty else 0
    y_max = float(all_vals.max()) if not all_vals.empty else 0.2
    y_range = y_max - y_min if y_max != y_min else 0.1

    y_min = max(0, y_min - y_range * 0.02)
    y_max += y_range * 0.05
    y_range = y_max - y_min

    def scale_y(v):
        return margin["top"] + plot_h - ((v - y_min) / y_range * plot_h)

    def scale_x(i):
        return margin["left"] + (i / max(len(df) - 1, 1)) * plot_w

    fund_points = []
    bench_points = []
    for i in range(len(df)):
        x = scale_x(i)
        fund_points.append(f"{x:.1f},{scale_y(fund_vals.iloc[i]):.1f}")
        bench_points.append(f"{x:.1f},{scale_y(bench_vals.iloc[i]):.1f}")

    fund_path = "M " + " L ".join(fund_points) if fund_points else ""
    bench_path = "M " + " L ".join(bench_points) if bench_points else ""

    # Y-axis labels and grid lines
    y_labels = ""
    num_y_ticks = 5
    for i in range(num_y_ticks + 1):
        y_val = y_min + (y_range * i / num_y_ticks)
        y_pos = scale_y(y_val)
        y_labels += f'''
            <text x="{margin['left'] - 5}" y="{y_pos + 3}" 
                  text-anchor="end" font-size="8" fill="#888" 
                  font-family="Arial, sans-serif">
                {y_val * 100:.0f}%
            </text>
            <line x1="{margin['left']}" y1="{y_pos}" 
                  x2="{width - margin['right']}" y2="{y_pos}" 
                  stroke="#E8E8E8" stroke-width="1"/>
        '''

    # X-axis labels - rotated
    x_labels = ""
    num_x_labels = min(14, len(df))
    step = max(len(df) // num_x_labels, 1)
    for i in range(0, len(df), step):
        x = scale_x(i)
        date_str = df["date"].iloc[i].strftime("%b-%y").lower()
        x_labels += f'''
            <text x="{x}" y="{margin['top'] + plot_h + 12}" 
                  text-anchor="end" font-size="7" fill="#888"
                  font-family="Arial, sans-serif"
                  transform="rotate(-45, {x}, {margin['top'] + plot_h + 12})">
                {date_str}
            </text>
        '''

    # Legend at the bottom center
    legend_y = height - 10
    legend_center_x = width / 2

    svg = f'''
    <svg width="100%" height="{height}" viewBox="0 0 {width} {height}" 
         xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="xMidYMid meet">
        
        {y_labels}
        {x_labels}
        
        <!-- Chart lines -->
        <path d="{fund_path}" fill="none" stroke="{BRAND_BROWN}" 
              stroke-width="2"/>
        <path d="{bench_path}" fill="none" stroke="#C4A484" 
              stroke-width="1.5"/>
        
        <!-- Legend at bottom center -->
        <line x1="{legend_center_x - 70}" y1="{legend_y}" 
              x2="{legend_center_x - 50}" y2="{legend_y}" 
              stroke="{BRAND_BROWN}" stroke-width="2"/>
        <text x="{legend_center_x - 45}" y="{legend_y + 4}" 
              font-size="9" fill="#333" 
              font-family="Arial, sans-serif">fundo</text>
        <line x1="{legend_center_x + 15}" y1="{legend_y}" 
              x2="{legend_center_x + 35}" y2="{legend_y}" 
              stroke="#C4A484" stroke-width="1.5"/>
        <text x="{legend_center_x + 40}" y="{legend_y + 4}" 
              font-size="9" fill="#333"
              font-family="Arial, sans-serif">{bench_label}</text>
    </svg>
    '''
    return svg


def generate_donut_chart_svg(df: pd.DataFrame) -> str:
    """Generate SVG donut/pie chart from alloc_donut data with legend below."""
    if df.empty:
        return "<p>No allocation data</p>"

    cat_col = None
    pct_col = None
    for c in df.columns:
        cl = str(c).upper()
        if "CATEGORIA" in cl or "ATIVO" in cl:
            cat_col = c
        if "%" in str(c) or "PCT" in cl:
            pct_col = c

    if cat_col is None or pct_col is None:
        cat_col = df.columns[0]
        pct_col = df.columns[-1]

    categories = df[cat_col].tolist()
    values = pd.to_numeric(df[pct_col], errors="coerce").fillna(0).tolist()

    total = sum(values)
    if total > 10:
        values = [v / 100 for v in values]
        total = sum(values)

    if total == 0:
        return "<p>No allocation data</p>"

    colors = ["#825120", "#9E6B3D", "#B8895A", "#D4B896", "#E8DDD0", "#F2EDE7"]

    valid_items = [(cat, val) for cat, val in zip(categories, values) if val > 0]
    num_items = len(valid_items)

    if num_items <= 3:
        r = 60
        inner_r = 28
    elif num_items <= 5:
        r = 55
        inner_r = 25
    else:
        r = 50
        inner_r = 22

    cx, cy = 75, r + 5

    svg_paths = []
    legend_items = []
    start_angle = -90
    color_idx = 0

    for cat, val in valid_items:
        angle = (val / total) * 360
        if angle >= 360:
            angle = 359.99
        end_angle = start_angle + angle

        start_rad = math.radians(start_angle)
        end_rad = math.radians(end_angle)

        x1_out = cx + r * math.cos(start_rad)
        y1_out = cy + r * math.sin(start_rad)
        x2_out = cx + r * math.cos(end_rad)
        y2_out = cy + r * math.sin(end_rad)

        x1_in = cx + inner_r * math.cos(end_rad)
        y1_in = cy + inner_r * math.sin(end_rad)
        x2_in = cx + inner_r * math.cos(start_rad)
        y2_in = cy + inner_r * math.sin(start_rad)

        large_arc = 1 if angle > 180 else 0
        color = colors[color_idx % len(colors)]

        path = f"""
            M {x1_out:.1f} {y1_out:.1f}
            A {r} {r} 0 {large_arc} 1 {x2_out:.1f} {y2_out:.1f}
            L {x1_in:.1f} {y1_in:.1f}
            A {inner_r} {inner_r} 0 {large_arc} 0 {x2_in:.1f} {y2_in:.1f}
            Z
        """
        svg_paths.append(f'<path d="{path}" fill="{color}"/>')

        legend_items.append((cat, color, val))
        color_idx += 1
        start_angle = end_angle

    legend_y_start = cy + r + 15
    legend_x = 10
    line_height = 14

    legend_svg = []
    for idx, (cat, color, val) in enumerate(legend_items):
        legend_y = legend_y_start + idx * line_height

        cat_label = str(cat)[:25] if len(str(cat)) > 25 else str(cat)
        pct_label = f"({val * 100:.1f}%)"

        legend_svg.append(
            f'''
            <rect x="{legend_x}" y="{legend_y}" width="10" height="10" 
                  fill="{color}" rx="2"/>
            <text x="{legend_x + 14}" y="{legend_y + 8}" font-size="7" 
                  fill="{TEXT_DARK}"
                  font-family="'Open Sans', 'Segoe UI', Arial, sans-serif">
                {cat_label} {pct_label}
            </text>
        '''
        )

    total_height = legend_y_start + len(legend_items) * line_height + 5
    svg_width = 150

    svg = f'''
    <svg width="100%" height="100%" viewBox="0 0 {svg_width} 
         {total_height}" 
         xmlns="http://www.w3.org/2000/svg" 
         preserveAspectRatio="xMidYMid meet">
        <g>
            {''.join(svg_paths)}
        </g>
        {''.join(legend_svg)}
    </svg>
    '''
    return svg


def build_perf_table_html(df: pd.DataFrame) -> str:
    if df.empty:
        return "<p>No performance data available</p>"

    cols = ["SÉRIE", "YTD", "3M", "6M", "12M", "24M", "36M"]

    df = df.copy()
    df.columns = [str(c).upper().strip() for c in df.columns]

    html = '<table class="perf-table"><thead><tr>'
    for idx, c in enumerate(cols):
        if idx == 0:
            html += f'<th class="first-th">{c}</th>'
        elif idx == len(cols) - 1:
            html += f'<th class="last-th">{c}</th>'
        else:
            html += f"<th>{c}</th>"
    html += "</tr></thead><tbody>"

    for idx, row in df.iterrows():
        row_class = "even-row" if idx % 2 == 1 else ""
        serie_name = str(row.get("SÉRIE", row.iloc[0])).upper()
        html += f"<td class='serie-cell'>{serie_name}</td>"

        for c in cols[1:]:
            if c in df.columns:
                val = row.get(c, "")
                if serie_name == "SHARPE":
                    formatted = format_number(val, 2)
                else:
                    formatted = format_pct(val, 2)
                html += f"<td>{formatted}</td>"

        html += "</tr>"

    html += "</tbody></table>"
    return html


def build_top_positions_html(df: pd.DataFrame) -> str:
    if df.empty:
        return "<p>No position data available</p>"

    df = df.copy()
    df.columns = [str(c).upper().strip() for c in df.columns]

    ativo_col = next((c for c in df.columns if "ATIVO" in c), df.columns[0])
    emissor_col = next(
        (c for c in df.columns if "EMISSOR" in c),
        df.columns[1] if len(df.columns) > 1 else None,
    )
    pct_col = next((c for c in df.columns if "%" in c or "PCT" in c), df.columns[-1])

    html = '<table class="top-table"><thead><tr>'
    html += '<th class="first-th">ATIVO</th><th>EMISSOR</th><th class="last-th">%</th>'
    html += "</tr></thead><tbody>"

    for idx, row in df.head(10).iterrows():
        row_class = "even-row" if idx % 2 == 1 else ""
        ativo = clean_string_value(row.get(ativo_col, ""))[:35]
        emissor = (
            clean_string_value(row.get(emissor_col, ""))[:45] if emissor_col else ""
        )
        pct = format_pct(row.get(pct_col, 0), 2)

        html += (
            f'<tr class="{row_class}"><td>{ativo}</td>'
            f"<td>{emissor}</td>"
            f'<td class="pct-cell">{pct}</td></tr>'
        )

    html += "</tbody></table>"
    return html


def build_risk_indicator(level: str = "MODERADO") -> str:
    levels = ["CONSERVADOR", "MODERADO", "SOFISTICADO"]
    level = level.upper()

    html = '<div class="risk-scale">'
    for idx, lv in enumerate(levels):
        active = "active" if lv == level else ""
        position_class = ""
        if idx == 0:
            position_class = "first"
        elif idx == len(levels) - 1:
            position_class = "last"
        html += f'<div class="risk-level {active} {position_class}">{lv}</div>'
    html += "</div>"
    return html


def build_fund_details_html(details: dict) -> str:
    icons = {
        "AuM": """<svg viewBox="0 0 24 24" fill="none" stroke="currentColor"
           stroke-width="2"><path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5
           10-5M2 12l10 5 10-5"/></svg>""",
        "Classe do Fundo": """<svg viewBox="0 0 24 24" fill="none"
           stroke="currentColor" stroke-width="2"><circle cx="12" cy="12"
           r="10"/><path d="M12 6v6l4 2"/></svg>""",
        "Benchmark": """<svg viewBox="0 0 24 24" fill="none"
           stroke="currentColor" stroke-width="2"><circle cx="12" cy="12"
           r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12"
           y1="16" x2="12" y2="16"/></svg>""",
        "Tx. Administração": """<svg viewBox="0 0 24 24" fill="none"
           stroke="currentColor" stroke-width="2"><rect x="2" y="7"
           width="20" height="14" rx="2"/><path d="M16 7V5a2 2 0 00-2-2h-4a2
           2 0 00-2 2v2"/></svg>""",
        "Tx. Performance": """<svg viewBox="0 0 24 24" fill="none"
           stroke="currentColor" stroke-width="2"><polyline points="23 6
           13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23
           12"/></svg>""",
        "Administrador": """<svg viewBox="0 0 24 24" fill="none"
           stroke="currentColor" stroke-width="2"><path d="M3 21h18M3 7v14M21
           7v14M6 7V4a1 1 0 011-1h10a1 1 0 011 1v3M9 21v-4a2 2 0 012-2h2a2 2 0
           012 2v4"/></svg>""",
        "Resgate": """<svg viewBox="0 0 24 24" fill="none" stroke="currentColor"
           stroke-width="2"><rect x="3" y="4" width="18" height="18"
           rx="2"/><line x1="16" y1="2" x2="16" y2="6"/><line x1="8" y1="2"
           x2="8" y2="6"/><line x1="3" y1="10" x2="21"
           y2="10"/></svg>""",
    }

    html = ""
    for key, value in details.items():
        icon_svg = icons.get(key, icons["AuM"])
        display_value = value if value else "-"
        html += f"""
        <div class="detail-row">
            <div class="detail-icon">{icon_svg}</div>
            <div class="detail-content">
                <div class="detail-label">{key}</div>
                <div class="detail-value">{display_value}</div>
            </div>
        </div>
        """
    return html


def build_combined_info_html(
    characteristics: str,
    team: str,
    portfolio_thesis: str,
    performance_commentary: str,
) -> str:
    sections = [
        (
            "Características Principais",
            characteristics
            or "The fund leverages on PIMCO's investment philosophy and "
            "process: long-term orientation, combines both top-down views "
            "with bottom-up credit research.",
        ),
        (
            "Equipe de Gestão",
            team
            or "The fund is managed by Mark Kiesel, CIO of Global Credit "
            "(25 years at PIMCO), Mohit Mittal (14) and Jelle Bronds (16), "
            "and benefits from the support of a team of 80 credit analysts.",
        ),
        (
            "Portfólio e Tese de Investimento",
            portfolio_thesis
            or "The portfolio is built according to security selection out "
            "of solid fundamentals, completed with a sector allocation "
            "process, and positioning on duration, curve and FX based on "
            "Pimco's macroeconomic view and top-down investment framework.",
        ),
        (
            "Performance",
            performance_commentary
            or "2024 – the fund outperformed its benchmark thanks to "
            "security selection in European corporates and some EM names. "
            "In terms of sector, OW in banks and brokers helped, UW in "
            "government-related sector also contributed. The managers "
            "prefer sectors with asset coverage and high barriers to entry.",
        ),
    ]

    html = ""
    for title, content in sections:
        html += f"""
        <div class="info-section">
            <div class="info-section-title">{title}</div>
            <div class="info-section-content">{content}</div>
        </div>
        """
    return html


def build_logo_html(logo_base64: str, logo_mime: str) -> str:
    if logo_base64:
        return f"""
        <div class="logo">
            <img src="data:{logo_mime};base64,{logo_base64}" alt="Logo"
                 class="logo-img"/>
        </div>
        """
    else:
        return """
        <div class="logo">
            <div class="logo-text">C E R E S</div>
            <div class="logo-subtext">W E A L T H</div>
        </div>
        """


def generate_html_factsheet(
    meta: dict,
    perf_df: pd.DataFrame,
    line_df: pd.DataFrame,
    alloc_df: pd.DataFrame,
    top_df: pd.DataFrame,
    description: str = "",
    characteristics: str = "",
    team: str = "",
    portfolio_thesis: str = "",
    performance_commentary: str = "",
    risk_level: str = "MODERADO",
    fund_details: Optional[dict] = None,
    logo_base64: str = "",
    logo_mime: str = "image/png",
    graph_indexer: Optional[str] = None,
    internal_use_only: bool = False,
) -> str:
    """Generate complete HTML factsheet."""

    fund_name = meta.get("fund_name", "FUND NAME")
    cnpj = meta.get("cnpj", "00.000.000/0000-00")
    series_range = meta.get("series_range", "2024-01-01 / 2025-01-01")

    if fund_details is None:
        fund_details = {
            "AuM": "",
            "Classe do Fundo": "",
            "Benchmark": meta.get("benchmark", "CDI"),
            "Tx. Administração": "",
            "Tx. Performance": "",
            "Administrador": "",
            "Resgate": "",
        }

    perf_table_html = build_perf_table_html(perf_df)
    top_table_html = build_top_positions_html(top_df)
    line_chart_svg = generate_line_chart_svg(
        line_df, comparator=graph_indexer, benchmark=meta.get("benchmark")
    )
    donut_chart_svg = generate_donut_chart_svg(alloc_df)
    risk_indicator = build_risk_indicator(risk_level)
    fund_details_html = build_fund_details_html(fund_details)
    combined_info_html = build_combined_info_html(
        characteristics, team, portfolio_thesis, performance_commentary
    )
    logo_html = build_logo_html(logo_base64, logo_mime)

    html = f'''
<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport"
          content="width=device-width, initial-scale=1.0">
    <title>{fund_name} - Factsheet</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Open+Sans:wght@400;600;700&display=swap');
        
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Open Sans', 'Segoe UI', Arial, sans-serif;
            font-size: 10px;
            line-height: 1.4;
            color: {TEXT_DARK};
            background: #f0f0f0;
        }}
        
        .page {{
            width: 297mm;
            margin: 0 auto;
            background: white;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        
        @media screen and (max-width: 1200px) {{
            .page {{
                width: 100%;
                margin: 0;
                box-shadow: none;
            }}
        }}
        
        @media print {{
            @page {{
                size: auto;
                margin: 0;
            }}
            body {{
                background: white;
                -webkit-print-color-adjust: exact !important;
                print-color-adjust: exact !important;
            }}
            .page {{
                width: 100%;
                margin: 0;
                box-shadow: none;
                page-break-after: auto;
            }}
        }}
        
        /* Header - NO padding, full width */
        .header {{
            background: {BRAND_BROWN};
            color: white;
            padding: 6px 12px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-radius: 0;
        }}
        
        .header-left h1 {{
            font-size: 14px;
            font-weight: 700;
            letter-spacing: 0.5px;
            margin-bottom: 2px;
        }}
        
        .header-left .cnpj {{
            font-size: 9px;
            opacity: 0.9;
        }}
        
        .logo {{
            text-align: right;
        }}
        
        .logo-text {{
            font-size: 13px;
            font-weight: 700;
            letter-spacing: 4px;
        }}
        
        .logo-subtext {{
            font-size: 8px;
            font-weight: 400;
            letter-spacing: 5px;
            margin-top: 1px;
        }}
        
        .logo-img {{
            max-height: 40px;
            max-width: 120px;
            object-fit: contain;
        }}
        
        /* Content wrapper - has padding */
        .content-wrapper {{
            padding: 2mm 2mm 5mm 5mm;
        }}
        
        /* Main grid - 2 columns */
        .main-grid {{
            display: grid;
            grid-template-columns: 58% 40%;
            gap: 6px;
            align-items: start;
        }}
        
        .left-column {{
            display: flex;
            flex-direction: column;
            gap: 4px;
        }}
        
        .right-column {{
            display: flex;
            flex-direction: column;
            gap: 4px;
        }}
        
        /* Section styling */
        .section-title {{
            color: {BRAND_BROWN};
            font-size: 12px;
            font-weight: 700;
            margin-bottom: 2px;
            text-transform: uppercase;
            letter-spacing: 0.3px;
        }}
        
        .section-content {{
            font-size: 9px;
            line-height: 1.5;
            text-align: justify;
            color: #444;
        }}
        
        /* Performance chart box - rounded corners on header */
        .chart-box {{
            background: {BLOCK_BG};
            flex-shrink: 0;
            border-radius: 6px;
            overflow: hidden;
        }}
        
        .chart-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 4px 6px;
            background: {BRAND_BROWN};
            color: white;
            border-radius: 6px 6px 0 0;
        }}
        
        .chart-title {{
            color: white;
            font-weight: 700;
            font-size: 10px;
            text-transform: uppercase;
        }}
        
        .chart-date {{
            font-size: 9px;
            color: white;
            opacity: 0.9;
        }}
        
        .chart-content {{
            padding: 0;
            width: 100%;
        }}
        
        .chart-content svg {{
            display: block;
            width: 100%;
        }}
        
        /* Performance Table - rounded corners on header */
        .perf-table {{
            width: 100%;
            border-collapse: separate;
            border-spacing: 0;
            font-size: 9px;
        }}
        
        .perf-table th {{
            background: {BRAND_BROWN};
            color: white;
            padding: 4px 4px;
            text-align: center;
            font-weight: 600;
            font-size: 9px;
        }}
        
        .perf-table th.first-th {{
            border-radius: 6px 0 0 0;
        }}
        
        .perf-table th.last-th {{
            border-radius: 0 6px 0 0;
        }}
        
        .perf-table td {{
            padding: 3px 4px;
            text-align: center;
            border-bottom: 1px solid {TABLE_BORDER};
        }}
        
        .perf-table .even-row {{
            background: {LIGHT_GRAY};
        }}
        
        .perf-table .serie-cell {{
            text-align: left;
            font-weight: 600;
            padding-left: 6px;
        }}
        
        /* Top positions table - rounded corners on header */
        .top-table {{
            width: 100%;
            border-collapse: separate;
            border-spacing: 0;
            font-size: 8px;
        }}
        
        .top-table th {{
            background: {BRAND_BROWN};
            color: white;
            padding: 3px 4px;
            text-align: left;
            font-weight: 600;
            font-size: 8px;
        }}
        
        .top-table th.first-th {{
            border-radius: 6px 0 0 0;
        }}
        
        .top-table th.last-th {{
            text-align: center;
            width: 50px;
            border-radius: 0 6px 0 0;
        }}
        
        .top-table td {{
            padding: 2px 4px;
            border-bottom: 1px solid {TABLE_BORDER};
        }}
        
        .top-table .pct-cell {{
            text-align: center;
        }}
        
        .top-table .even-row {{
            background: {LIGHT_GRAY};
        }}
        
        /* Styled block */
        .styled-block {{
            background: {BLOCK_BG};
            border-radius: 4px;
            padding: 4px 6px;
        }}
        
        /* Risk indicator */
        .risk-box {{
            background: {BLOCK_BG};
            border-radius: 4px;
            padding: 4px 6px;
        }}
        
        .risk-title {{
            color: {BRAND_BROWN};
            font-weight: 700;
            font-size: 12px;
            margin-bottom: 4px;
        }}
        
        .risk-scale {{
            display: flex;
            gap: 0;
            border-radius: 20px;
            overflow: hidden;
        }}
        
        .risk-level {{
            flex: 1;
            text-align: center;
            padding: 5px 3px;
            font-size: 9px;
            font-weight: 600;
            text-transform: uppercase;
            color: #666;
            background: #C4C4C4;
            border: none;
        }}
        
        .risk-level.first {{
            border-radius: 20px 0 0 20px;
        }}
        
        .risk-level.last {{
            border-radius: 0 20px 20px 0;
        }}
        
        .risk-level.active {{
            background: {BRAND_BROWN};
            color: white;
        }}
        
        /* Fund details and donut side by side */
        .details-donut-row {{
            display: flex;
            gap: 6px;
            align-items: stretch;
        }}
        
        /* Fund details - improved styling */
        .fund-details {{
            flex: 1;
            background: {BLOCK_BG};
            border-radius: 4px;
            padding: 6px 8px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }}
        
        .detail-row {{
            display: flex;
            align-items: flex-start;
            min-height: 28px;
            padding: 4px 0;
            border-bottom: 1px solid {TABLE_BORDER};
        }}
        
        .detail-row:last-child {{
            border-bottom: none;
        }}
        
        .detail-icon {{
            width: 18px;
            height: 18px;
            margin-right: 8px;
            color: {BRAND_BROWN};
            flex-shrink: 0;
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        
        .detail-icon svg {{
            width: 14px;
            height: 14px;
        }}
        
        .detail-content {{
            display: flex;
            flex-direction: column;
            align-items: flex-start;
        }}
        
        .detail-label {{
            font-size: 9px;
            color: #888;
            line-height: 1.2;
            margin-bottom: 1px;
        }}
        
        .detail-value {{
            font-size: 11px;
            color: {TEXT_DARK};
            font-weight: 600;
            text-align: left;
        }}
        
        /* Donut chart */
        .donut-box {{
            flex: 1;
            background: {BLOCK_BG};
            border-radius: 4px;
            padding: 4px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        }}
        
        /* Combined info block */
        .combined-info-box {{
            background: {BLOCK_BG};
            border-radius: 4px;
            padding: 4px 6px;
        }}
        
        .info-section {{
            margin-bottom: 4px;
            padding-bottom: 4px;
            border-bottom: 1px solid {TABLE_BORDER};
        }}
        
        .info-section:last-child {{
            margin-bottom: 0;
            padding-bottom: 0;
            border-bottom: none;
        }}
        
        .info-section-title {{
            color: {BRAND_BROWN};
            font-size: 11px;
            font-weight: 700;
            margin-bottom: 1px;
            text-transform: uppercase;
            letter-spacing: 0.3px;
        }}
        
        .info-section-content {{
            font-size: 8px;
            line-height: 1.4;
            color: #444;
            text-align: justify;
        }}
        
        /* Description section */
        .desc-section {{
            margin-bottom: 2px;
        }}
        
        @media print {{
            .combined-info-box {{
                page-break-inside: avoid;
            }}
            .styled-block, .risk-box, .fund-details, .donut-box,
            .combined-info-box, .chart-box {{
                -webkit-print-color-adjust: exact !important;
                print-color-adjust: exact !important;
            }}
        }}

        .internal-banner {{
            background: #ae0000;
            color: white;
            padding: 4px 0;
            text-align: center;
            font-size: 10px;
            font-weight: 700;
            letter-spacing: 1px;
            text-transform: uppercase;
        }}
    </style>
</head>
<body>
    <div class="page">
        {f'<div class="internal-banner">MATERIAL DE USO INTERNO - PROIBIDO O COMPARTILHAMENTO</div>' if internal_use_only else ''}
        <!-- Header - no padding wrapper -->
        <div class="header">
            <div class="header-left">
                <h1>{fund_name}</h1>
                <div class="cnpj">{cnpj}</div>
            </div>
            {logo_html}
        </div>
        
        <!-- Content with padding -->
        <div class="content-wrapper">
            <!-- Main Content Grid -->
            <div class="main-grid">
                <!-- Left Column -->
                <div class="left-column">
                    <!-- Description -->
                    <div class="desc-section">
                        <div class="section-title">Descrição do Fundo</div>
                        <div class="section-content">
                            {description or "The portfolio is built according to security selection out of solid fundamentals, completed with a sector allocation process, and positioning on duration, curve and FX based on Pimco's macroeconomic view and top-down investment framework."}
                        </div>
                    </div>
                    
                    <!-- Performance Chart -->
                    <div class="chart-box">
                        <div class="chart-header">
                            <div class="chart-title">Performance</div>
                            <div class="chart-date">{series_range}</div>
                        </div>
                        <div class="chart-content">
                            {line_chart_svg}
                        </div>
                    </div>
                    
                    <!-- Performance Table -->
                    <div>
                        <div class="section-title">Índices de Rentabilidade</div>
                        {perf_table_html}
                    </div>
                    
                    <!-- Top Positions -->
                    <div>
                        <div class="section-title">Top 10 Exposições</div>
                        {top_table_html}
                    </div>
                </div>
                
                <!-- Right Column -->
                <div class="right-column">
                    <!-- Risk Indicator -->
                    <div class="risk-box">
                        <div class="risk-title">RISCO</div>
                        {risk_indicator}
                    </div>
                    
                    <!-- Fund Details + Donut Chart side by side -->
                    <div class="details-donut-row">
                        <div class="fund-details">
                            {fund_details_html}
                        </div>
                        <div class="donut-box">
                            {donut_chart_svg}
                        </div>
                    </div>
                    
                    <!-- Combined Info Block -->
                    <div class="combined-info-box">
                        {combined_info_html}
                    </div>
                </div>
            </div>
        </div>
    </div>
</body>
</html>
'''
    return html


# ---------------------- Fund Catalog ingestion ----------------------


def parse_fund_catalog(uploaded_file) -> dict[str, dict]:
    """
    Parse an Excel file containing a catalog of funds with columns:
    Fundos | AuM | Classe do Fundo | Benchmark | Tx. Administração |
    Tx Performance | Administrador | Resgate | Risco

    Returns a dict keyed by cleaned CNPJ (digits only) with values:
    {
      "AuM": str,
      "Classe do Fundo": str,
      "Benchmark": str,
      "Tx. Administração": str,
      "Tx. Performance": str,
      "Administrador": str,
      "Resgate": str,
      "Risco": str,
    }
    """
    df = pd.read_excel(uploaded_file, dtype=str)
    original_cols = list(df.columns)
    norm_to_original = {norm_key(c): c for c in original_cols}

    def pick_col(aliases: list[str]) -> Optional[str]:
        for alias in aliases:
            if alias in norm_to_original:
                return norm_to_original[alias]
        return None

    col_fundos = pick_col(["fundos", "cnpj"])
    col_aum = pick_col(["aum"])
    col_class = pick_col(["classedofundo", "classefundo", "classe"])
    col_bench = pick_col(["benchmark", "comparador"])
    col_tx_admin = pick_col(["txadministracao", "txdeadministracao"])
    col_tx_perf = pick_col(["txperformance", "txdesempenho", "txperformancea"])
    col_admin = pick_col(["administrador"])
    col_resgate = pick_col(["resgate"])
    col_risco = pick_col(["risco", "perfilrisco"])

    required = [col_fundos]
    if any(c is None for c in required):
        return {}

    cat_map: dict[str, dict] = {}
    for _, row in df.iterrows():
        cnpj_raw = row.get(col_fundos, "")
        cnpj_clean = clean_cnpj(cnpj_raw)
        if not cnpj_clean:
            continue

        rec = {
            "AuM": row.get(col_aum, "") if col_aum else "",
            "Classe do Fundo": row.get(col_class, "") if col_class else "",
            "Benchmark": row.get(col_bench, "") if col_bench else "",
            "Tx. Administração": row.get(col_tx_admin, "") if col_tx_admin else "",
            "Tx. Performance": row.get(col_tx_perf, "") if col_tx_perf else "",
            "Administrador": row.get(col_admin, "") if col_admin else "",
            "Resgate": row.get(col_resgate, "") if col_resgate else "",
            "Risco": row.get(col_risco, "") if col_risco else "",
        }
        cat_map[cnpj_clean] = rec

    return cat_map


def apply_catalog_overrides(
    cnpj: str,
    fund_details: dict,
    risk_level: str,
    meta_benchmark: Optional[str],
    catalog_map: Optional[dict[str, dict]],
) -> tuple[dict, str, str]:
    """
    If catalog_map has a row for the CNPJ, fill any missing fields in
    fund_details with values from the catalog. Use catalog risk if present.
    Update benchmark if catalog provides one. Return (fund_details, risk, bench).
    """
    if not catalog_map:
        return fund_details, risk_level, meta_benchmark or ""

    cnpj_clean = clean_cnpj(cnpj)
    rec = catalog_map.get(cnpj_clean)
    if not rec:
        return fund_details, risk_level, meta_benchmark or ""

    for key in [
        "AuM",
        "Classe do Fundo",
        "Benchmark",
        "Tx. Administração",
        "Tx. Performance",
        "Administrador",
        "Resgate",
    ]:
        cur = fund_details.get(key, "")
        if (cur is None or str(cur).strip() == "") and not is_missing(rec.get(key)):
            fund_details[key] = str(rec.get(key))

    if not is_missing(rec.get("Risco")):
        risk_level = risk_to_level(rec.get("Risco"), default=risk_level)

    bench = fund_details.get("Benchmark", "") or (meta_benchmark or "")

    return fund_details, risk_level, bench


# ---------------------- Word (.docx) Funds Info ingestion ----------------------

# ---------------------- Word (.docx) Funds Info ingestion ----------------------

def _read_docx_text(uploaded_file) -> str:
    """
    Robustly read .docx file text content.
    """
    text = ""
    
    # Ensure we are at the start of the file
    uploaded_file.seek(0)
    file_bytes = uploaded_file.read()
    
    # 1. Try python-docx (best for preserving structure)
    try:
        from docx import Document
        with io.BytesIO(file_bytes) as bio:
            doc = Document(bio)
            # Join with newlines to ensure JSON doesn't mash together
            paras = [p.text for p in doc.paragraphs]
            text = "\n".join(paras)
    except Exception:
        # 2. Fallback: unzip and read XML (dirty extraction)
        try:
            with zipfile.ZipFile(io.BytesIO(file_bytes)) as zf:
                xml_bytes = zf.read("word/document.xml")
                xml_text = xml_bytes.decode("utf-8", errors="ignore")
                # Strip XML tags
                text = re.sub(r"<[^>]+>", "", xml_text)
        except Exception:
            text = ""

    return text

def _normalize_word_json(s: str) -> str:
    """
    Aggressively normalize Word-formatted text into valid JSON.
    """
    if not s:
        return s
        
    # 1. Replace all variations of smart quotes with straight quotes
    replacements = {
        '\u201c': '"',  # Left double quote
        '\u201d': '"',  # Right double quote
        '\u201e': '"',  # Double low-9 quote
        '\u201f': '"',  # Double high-reversed quote
        '“': '"', 
        '”': '"',
        '‟': '"',
        '„': '"',
        
        # Replace single smart quotes with straight single quotes (optional for JSON values)
        '\u2018': "'", 
        '\u2019': "'", 
        '‘': "'", 
        '’': "'",
        
        # Normalize spaces
        '\u00a0': ' ', # Non-breaking space
        '\u202f': ' ', # Narrow no-break space
    }
    
    for old, new in replacements.items():
        s = s.replace(old, new)

    # 2. Remove problematic trailing commas before closing braces/brackets
    # Matches: , (whitespace) }  OR  , (whitespace) ]
    s = re.sub(r",(\s*[}\]])", r"\1", s)
    
    return s

def parse_funds_info_docx(uploaded_file) -> tuple[dict[str, dict], list[dict]]:
    """
    Parse a Word (.docx) file containing a JSON array.
    """
    raw_text = _read_docx_text(uploaded_file)
    if not raw_text:
        st.error("Could not extract text from the Word file.")
        return {}, []

    # Extract the JSON block: Find first '[' and last ']'
    start = raw_text.find("[")
    end = raw_text.rfind("]")
    
    if start == -1 or end == -1 or end <= start:
        st.error("No JSON array brackets [...] found in the Word document.")
        return {}, []

    # Get the candidate string and clean it
    json_candidate = raw_text[start : end + 1]
    json_clean = _normalize_word_json(json_candidate)

    rows = []
    try:
        # Attempt 1: Parse cleaned text
        rows = json.loads(json_clean)
    except json.JSONDecodeError as e:
        # Attempt 2: Strict whitespace cleanup (Word sometimes adds weird tabs/newlines)
        try:
            # Replace control characters but keep spaces
            json_strict = "".join(ch if ch >= ' ' else ' ' for ch in json_clean)
            rows = json.loads(json_strict)
        except json.JSONDecodeError as e2:
            st.error(f"JSON Parsing Error: {e2}")
            # Show a snippet around the error to help user debug
            error_idx = e2.pos
            snippet_start = max(0, error_idx - 50)
            snippet_end = min(len(json_clean), error_idx + 50)
            st.code(f"...{json_clean[snippet_start:snippet_end]}...", language="text")
            return {}, []
    except Exception as e:
        st.error(f"Unexpected error parsing JSON: {e}")
        return {}, []

    # If parsing succeeded but didn't return a list
    if not isinstance(rows, list):
        st.error(f"Expected a JSON list [...], but got {type(rows).__name__}")
        return {}, []

    info_map: dict[str, dict] = {}

    # Helper to pick values case-insensitively
    def pick(obj: dict, keys: list[str]) -> str:
        # 1. Exact match
        for k in keys:
            if k in obj and obj[k] is not None:
                return str(obj[k]).strip()
        
        # 2. Case-insensitive / Accent-insensitive match
        obj_norm = {strip_accents(str(k)).lower(): v for k, v in obj.items()}
        for k in keys:
            k_norm = strip_accents(k).lower()
            if k_norm in obj_norm and obj_norm[k_norm] is not None:
                return str(obj_norm[k_norm]).strip()
        return ""

    for obj in rows:
        if not isinstance(obj, dict):
            continue
            
        cnpj_raw = pick(obj, ["cnpj"])
        cnpj_clean = clean_cnpj(cnpj_raw)
        
        if not cnpj_clean:
            continue
            
        rec = {
            "fund_name": pick(obj, ["fundo", "fundo_nome", "nome"]),
            "description": pick(obj, ["descrição", "descricao"]),
            "characteristics": pick(obj, ["características", "caracteristicas"]),
            "team": pick(obj, ["equipe"]),
            "portfolio_thesis": pick(obj, ["portfolio", "portfólio", "portafolio"]),
            "performance_commentary": pick(obj, ["performance", "comentarios"]),
        }
        info_map[cnpj_clean] = rec

    return info_map, rows
    """
    Parse a Word (.docx) file that contains a JSON array of fund info.

    Expected JSON structure (list of objects), each object may contain:
      - "fundo"
      - "cnpj"  (original CNPJ)
      - "descrição" | "descricao"
      - "características" | "caracteristicas"
      - "equipe"
      - "portfolio" | "portfólio"
      - "performance"

    Returns:
      (info_map, rows)
      info_map: dict keyed by cleaned original CNPJ with values {
          "fund_name",
          "description",
          "characteristics",
          "team",
          "portfolio_thesis",
          "performance_commentary",
      }
      rows: the original parsed rows (for preview)
    """
    text = _read_docx_text(uploaded_file)
    if not text:
        return {}, []

    # Extract the largest bracketed block (best-effort)
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return {}, []

    json_str = text[start : end + 1]
    json_str = _normalize_json_like_text(json_str)

    rows = []
    try:
        parsed = json.loads(json_str)
        if isinstance(parsed, list):
            rows = parsed
        else:
            rows = []
    except Exception:
        # Try another pass: collapse excessive whitespace
        json_str_try = re.sub(r"\s+", " ", json_str)
        json_str_try = _normalize_json_like_text(json_str_try)
        try:
            parsed = json.loads(json_str_try)
            rows = parsed if isinstance(parsed, list) else []
        except Exception:
            rows = []

    info_map: dict[str, dict] = {}

    def pick(obj: dict, keys: list[str]) -> str:
        for k in keys:
            if k in obj and obj[k] is not None:
                v = str(obj[k]).strip()
                if v:
                    return v
        # also try accentless lookups
        obj_noacc = {strip_accents(k).lower(): v for k, v in obj.items()}
        for k in keys:
            k_na = strip_accents(k).lower()
            if k_na in obj_noacc and obj_noacc[k_na] is not None:
                v = str(obj_noacc[k_na]).strip()
                if v:
                    return v
        return ""

    for obj in rows:
        if not isinstance(obj, dict):
            continue
        cnpj_raw = pick(obj, ["cnpj"])
        cnpj_clean = clean_cnpj(cnpj_raw)
        if not cnpj_clean:
            continue
        rec = {
            "fund_name": pick(obj, ["fundo", "fundo_nome", "nome"]),
            "description": pick(obj, ["descrição", "descricao"]),
            "characteristics": pick(obj, ["características", "caracteristicas"]),
            "team": pick(obj, ["equipe"]),
            "portfolio_thesis": pick(obj, ["portfolio", "portfólio", "portafolio"]),
            "performance_commentary": pick(obj, ["performance", "comentarios"]),
        }
        info_map[cnpj_clean] = rec

    return info_map, rows



# ---------------------- Text Generation Helper ----------------------

def generate_performance_text(perf_df: pd.DataFrame, benchmark_name: str) -> str:
    """
    Generate a natural language summary of the fund's performance
    based on the 'Índices de Rentabilidade' table (perf_df).
    
    Expected Columns in perf_df (normalized):
    [SERIE, YTD, 3M, 6M, 12M, 24M, 36M]
    
    Rows are expected to contain:
    - Fund row (first one)
    - Benchmark row (CDI or IBOV)
    - Volatility row
    - Sharpe row
    
    Format:
    "O fundo rendeu {ytd} neste ano, {%} do CDI (or {%} acima do IBOV). 
     Nos últimos 36 meses, rendeu {36m}, com volatilidade de {vol36m} e sharpe de {sharpe36m}.
     [Fluff]"
    """
    if perf_df.empty:
        return ""
    
    df = perf_df.copy()
    # Normalize columns to uppercase stripped
    df.columns = [str(c).upper().strip() for c in df.columns]
    
    # Identify the series column (usually index 0)
    serie_col = df.columns[0]
    
    # Helper to clean/parse values
    def get_val(row_idx, col_name):
        if row_idx < 0 or row_idx >= len(df):
            return None
        if col_name not in df.columns:
            return None
        val = df.iloc[row_idx][col_name]
        return parse_br_number(val)

    # Locate rows
    fund_idx = -1
    cdi_idx = -1
    ibov_idx = -1
    vol_idx = -1
    sharpe_idx = -1
    
    for idx, row in df.iterrows():
        s = str(row[serie_col]).upper()
        if "VOLATILIDADE" in s or "VOL" in s:
            vol_idx = idx
        elif "SHARPE" in s:
            sharpe_idx = idx
        elif "CDI" in s:
            cdi_idx = idx
        elif "IBOV" in s:
            ibov_idx = idx
        else:
            # Assume it is the fund if it hasn't been found yet and looks like a name
            if fund_idx == -1 and len(s) > 1:
                fund_idx = idx
                
    # Fallback: if fund_idx still -1, take 0 if 0 is not occupied by others?
    # Or just assume 0 if it wasn't identifed as benchmark/stats
    if fund_idx == -1:
         # Check if 0 is used
         if 0 not in (cdi_idx, ibov_idx, vol_idx, sharpe_idx):
             fund_idx = 0
            
    # Determine which benchmark to compare against based on benchmark_name
    bench_idx = -1
    is_cdi = True
    
    if benchmark_name:
        bname = benchmark_name.upper()
        if "IBOV" in bname:
            bench_idx = ibov_idx
            is_cdi = False
        elif "CDI" in bname:
            bench_idx = cdi_idx
            is_cdi = True
    
    # Fallback if benchmark_name was vague, try to pick one availability
    if bench_idx == -1:
        if cdi_idx != -1:
            bench_idx = cdi_idx
            is_cdi = True
        elif ibov_idx != -1:
            bench_idx = ibov_idx
            is_cdi = False

    # Extract Data
    ytd_val = get_val(fund_idx, "YTD")
    
    # 36M Data
    r36_val = get_val(fund_idx, "36M")
    
    vol36_val = None
    if vol_idx != -1:
        # Try 36M first, if missing fallback? No, prompt specifically asks for 36m stats.
        vol36_val = get_val(vol_idx, "36M")
        
    sharpe36_val = None
    if sharpe_idx != -1:
        sharpe36_val = get_val(sharpe_idx, "36M")

    # Formatting helper
    def fmt_pct(v):
        if v is None: return "-"
        return f"{v*100:.1f}%".replace(".", ",")

    def fmt_num(v):
        if v is None: return "-"
        return f"{v:.2f}".replace(".", ",")
        
    # Build Text
    parts = []
    
    # Part 1: YTD
    if ytd_val is not None:
        p1 = f"O fundo rendeu {fmt_pct(ytd_val)} neste ano"
        
        # Benchmark comparison
        if bench_idx != -1:
            bench_ytd = get_val(bench_idx, "YTD")
            if bench_ytd is not None:
                if is_cdi:
                    # % do CDI
                    # Avoid div by zero
                    if abs(bench_ytd) > 1e-9:
                        pct_of_cdi = (ytd_val / bench_ytd)
                        p1 += f", contra {pct_of_cdi*100:.0f}% do CDI"
                    else:
                        p1 += "" # Can't compute % of 0
                else:
                    # % above/below IBOV
                    diff = ytd_val - bench_ytd
                    direction = "acima" if diff >= 0 else "abaixo"
                    p1 += f", {fmt_pct(abs(diff))} {direction} do IBOV"
        
        p1 += "."
        parts.append(p1)
    
    # Part 2: 36 Months
    if r36_val is not None:
        p2 = f"Nos últimos 36 meses, a estratégia acumulou retorno de {fmt_pct(r36_val)}"
        stats = []
        if vol36_val is not None:
            stats.append(f"com volatilidade anualizada de {fmt_pct(vol36_val)}")
        if sharpe36_val is not None:
            stats.append(f"índice Sharpe de {fmt_num(sharpe36_val)}")
        
        if stats:
            p2 += ", " + " e ".join(stats)
        
        p2 += "."
        parts.append(p2)
        
    # Part 3: Fluff
    fluff = (
        "O portfólio segue posicionado de forma consistente com a tese de investimentos de longo prazo, "
        "buscando capturar oportunidades assimétricas e preservação de capital em diferentes cenários de mercado."
    )
    parts.append(fluff)
    
    return " ".join(parts)

# ---------------------- PDF Rendering (Playwright) ----------------------


def html_to_pdf_playwright(html_content: str) -> bytes:
    """
    Render HTML to a single-page PDF using headless Chromium via Playwright.

    This function measures the rendered `.page` container and sets a custom
    PDF width/height so the whole factsheet fits on exactly one PDF page.

    Requires:
      pip install playwright
      playwright install chromium

    The PDF honors print backgrounds/colors.
    """
    from playwright.sync_api import sync_playwright

    MAX_PX = 19000  # ~198 inches at 96 CSS px/in

    with sync_playwright() as p:
        browser = p.chromium.launch()
        context = browser.new_context()
        page = context.new_page()

        page.set_content(html_content, wait_until="load")
        page.wait_for_load_state("networkidle")

        page.emulate_media(media="screen")
        try:
            page.wait_for_function(
                'document.fonts && document.fonts.status === "loaded"',
                timeout=10000,
            )
        except Exception:
            pass

        dims_screen = page.evaluate(
            """
            () => {
              const el = document.querySelector('.page') || document.body;
              const rect = el.getBoundingClientRect();
              const width = Math.ceil(Math.max(el.scrollWidth, rect.width));
              const height = Math.ceil(Math.max(el.scrollHeight, rect.height));
              return { width, height };
            }
            """
        )

        width_px = int(max(600, min(dims_screen.get("width", 1122), MAX_PX)))
        height_px = int(max(600, min(dims_screen.get("height", 800), MAX_PX)))

        try:
            page.set_viewport_size({"width": width_px, "height": min(height_px, 5000)})
        except Exception:
            pass

        page.emulate_media(media="print")
        try:
            page.wait_for_function(
                'document.fonts && document.fonts.status === "loaded"',
                timeout=10000,
            )
        except Exception:
            pass

        dims_print = page.evaluate(
            """
            () => {
              const el = document.querySelector('.page') || document.body;
              const rect = el.getBoundingClientRect();
              const width = Math.ceil(Math.max(el.scrollWidth, rect.width));
              const height = Math.ceil(Math.max(el.scrollHeight, rect.height));
              return { width, height };
            }
            """
        )

        width_px = int(max(width_px, min(dims_print.get("width", width_px), MAX_PX)))
        height_px = int(
            max(height_px, min(dims_print.get("height", height_px), MAX_PX))
        )

        width_px += 2
        height_px += 2

        pdf_bytes = page.pdf(
            print_background=True,
            prefer_css_page_size=False,
            width=f"{width_px}px",
            height=f"{height_px}px",
            margin={"top": "0", "right": "0", "bottom": "0", "left": "0"},
        )

        context.close()
        browser.close()

    return pdf_bytes


# ---------------------- Streamlit App ----------------------


def make_zip(pairs: list[tuple[str, bytes]]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in pairs:
            zf.writestr(name, data)
    buf.seek(0)
    return buf.getvalue()


def main():
    st.set_page_config(
        page_title="Generate HTML Factsheet", page_icon="📄", layout="wide"
    )

    st.title("📄 Generate HTML Factsheet")
    st.markdown(
        "Upload the Excel file generated by "
        "**Export Factsheet Data** to create a styled HTML factsheet and "
        "download as PDF. Optionally upload a Fund Catalog to auto-fill "
        "fund details. Use Batch Generation to process multiple files at once."
    )

    with st.sidebar:
        st.subheader("Settings")
        internal_use_only = st.checkbox(
            "Internal Use Only Banner",
            value=False,
            help="Adds a red banner at the top of the factsheet: MATERIAL DE USO INTERNO - PROIBIDO O COMPARTILHAMENTO"
        )
        
        st.divider()
        override_performance = st.checkbox(
            "Override Performance Text",
            value=False,
            help="If checked, ignores Word/default text for 'Performance Commentary' and generates a summary from the Performance table (Perf, Vol, Sharpe)."
        )

    # ---------------------- Catalog uploader ----------------------
    st.subheader("Fund Catalog (optional)")
    catalog_file = st.file_uploader(
        "Upload Fund Catalog (Excel, .xlsx)",
        type=["xlsx"],
        key="catalog_uploader",
        help=(
            "Expected columns: Fundos | AuM | Classe do Fundo | Benchmark | "
            "Tx. Administração | Tx Performance | Administrador | Resgate | Risco"
        ),
    )
    catalog_map = None
    if catalog_file:
        try:
            catalog_map = parse_fund_catalog(catalog_file)
            st.success(f"Catalog loaded with {len(catalog_map)} rows.")
            with st.expander("Preview catalog (first 20 rows)"):
                catalog_file.seek(0)
                st.dataframe(
                    pd.read_excel(catalog_file).head(20), width='stretch'
                )
        except Exception as e:
            st.error(f"Failed to read catalog: {e}")
            catalog_map = None

    # ---------------------- Word (.docx) funds info uploader ----------------------
    st.subheader("Funds Info from Word (optional)")
    st.caption(
        "Upload a .docx containing a JSON array with items like: "
        '[{"fundo": "...", "cnpj": "00.000.000/0000-00", '
        '"descrição": "...", "características": "...", "equipe": "...", '
        '"portfolio": "...", "performance": "..."}]'
    )
    if "word_info_map" not in st.session_state:
        st.session_state["word_info_map"] = None
    if "word_info_rows" not in st.session_state:
        st.session_state["word_info_rows"] = []

    word_file = st.file_uploader(
        "Upload Funds Info (.docx)",
        type=["docx"],
        key="docx_uploader",
        help="A Word doc containing a JSON array of fund objects (by original CNPJ).",
    )
    if word_file:
        try:
            info_map, info_rows = parse_funds_info_docx(word_file)
            st.session_state["word_info_map"] = info_map if info_map else None
            st.session_state["word_info_rows"] = info_rows or []
            if info_map:
                st.success(
                    f"Word info loaded for {len(info_map)} funds (matched by original CNPJ)."
                )
                with st.expander("Preview Word info (first 10 funds)"):
                    preview = []
                    for row in info_rows[:10]:
                        preview.append(
                            {
                                "fundo": row.get("fundo", ""),
                                "cnpj": row.get("cnpj", ""),
                                "descrição (inicio)": str(
                                    row.get("descrição", row.get("descricao", ""))
                                )[:80],
                            }
                        )
                    if preview:
                        st.dataframe(pd.DataFrame(preview), width='stretch')
                    else:
                        st.info("No previewable rows.")
            else:
                st.warning(
                    "Could not parse a valid JSON array from the .docx. "
                    "Ensure the content contains a JSON list [ ... ] with fund objects."
                )
        except Exception as e:
            st.error(f"Failed to parse Word file: {e}")

    st.divider()

    # ---------------------- Single file mode ----------------------
    st.header("Single Generation")
    uploaded_file = st.file_uploader(
        "Upload Factsheet Excel (.xlsx)",
        type=["xlsx"],
        help="Upload the factsheet Excel export",
        key="single_factsheet",
    )

    # Logo (applies to single and batch)
    st.subheader("Logo (applies to single and batch)")
    logo_file = st.file_uploader(
        "Upload Logo (optional)",
        type=["png", "jpg", "jpeg", "svg", "webp"],
        help="Upload a logo to replace 'CERES WEALTH' text",
        key="logo_uploader",
    )
    if logo_file:
        st.image(logo_file, width=100, caption="Logo preview")

    if uploaded_file is None:
        st.info("Upload a factsheet Excel file for single generation.")
    else:
        try:
            sheets = read_excel_sheets(uploaded_file)
            st.success(f"Loaded {len(sheets)} sheets: {', '.join(sheets.keys())}")
        except Exception as e:
            st.error(f"Error reading Excel file: {e}")
            return

        meta = parse_meta(sheets.get("meta", pd.DataFrame()))
        perf_df = sheets.get("perf_table", pd.DataFrame())
        line_df = sheets.get("line_index", pd.DataFrame())
        alloc_df = sheets.get("alloc_donut", pd.DataFrame())
        top_df = sheets.get("top_positions", pd.DataFrame())

        # Mapping for display/master selection
        mapping = get_fund_mapping_by_original(meta.get("cnpj", ""))
        display_name = mapping["original_name"] if mapping else meta.get("fund_name", "")
        display_cnpj = mapping["original_cnpj"] if mapping else meta.get("cnpj", "")
        graph_indexer = mapping["graph"] if mapping else None
        master_cnpj = mapping["master_cnpj"] if mapping else display_cnpj

        st.divider()

        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Fund Information")
            st.text_input(
                "Fund Name (header)",
                value=display_name or "",
                disabled=True,
                key="fn",
            )
            cnpj_input = st.text_input(
                "CNPJ (original, used for details/catalog & Word match)",
                value=display_cnpj or "",
                key="cnpj",
            )
            risk_level = st.selectbox(
                "Risk Level",
                ["CONSERVADOR", "MODERADO", "SOFISTICADO"],
                index=1,
                key="risk",
            )

        with col2:
            st.subheader("Fund Details (sidebar)")
            aum = st.text_input(
                "AuM", value="", placeholder="R$ XX.XXX.XXX,XX", key="aum"
            )
            fund_class = st.text_input(
                "Classe do Fundo", value="", placeholder="Renda Fixa", key="class"
            )
            # Start with meta benchmark, may be overridden by catalog
            benchmark = st.text_input(
                "Benchmark", value=meta.get("benchmark", "CDI"), key="bench"
            )
            admin_fee = st.text_input(
                "Tx. Administração",
                value="",
                placeholder="0,30% a.a. (ex.: 0,003)",
                key="txadm",
            )

        with col3:
            st.subheader("More Details")
            perf_fee = st.text_input(
                "Tx. Performance",
                value="",
                placeholder="20% a.a. (ex.: 0,2)",
                key="txperf",
            )
            administrator = st.text_input(
                "Administrador",
                value="",
                placeholder="Nome do Administrador",
                key="adm",
            )
            redemption = st.text_input(
                "Resgate", value="", placeholder="D+XX", key="resg"
            )

        st.divider()

        # Content Sections (defaults from Word if available for this original CNPJ)
        st.subheader("Content Sections")
        st.caption("All info sections will be combined into a single block")

        # Determine defaults from Word info map for this original CNPJ
        word_info_map = st.session_state.get("word_info_map") or {}
        cnpj_for_content = clean_cnpj(cnpj_input or display_cnpj or meta.get("cnpj", ""))
        word_defaults = word_info_map.get(cnpj_for_content, {}) if cnpj_for_content else {}

        col_a, col_b = st.columns(2)

        with col_a:
            description = st.text_area(
                "Descrição do Fundo",
                value=word_defaults.get("description", ""),
                placeholder=(
                    "The portfolio is built according to security selection..."
                ),
                height=100,
                key="desc",
            )
            characteristics = st.text_area(
                "Características Principais",
                value=word_defaults.get("characteristics", ""),
                placeholder="The fund leverages on PIMCO's investment philosophy...",
                height=100,
                key="carac",
            )
            team = st.text_area(
                "Equipe de Gestão",
                value=word_defaults.get("team", ""),
                placeholder="The fund is managed by...",
                height=100,
                key="team",
            )

        with col_b:
            portfolio_thesis = st.text_area(
                "Portfólio e Tese de Investimento",
                value=word_defaults.get("portfolio_thesis", ""),
                placeholder=(
                    "The portfolio is built according to security selection..."
                ),
                height=100,
                key="thesis",
            )
            # Determine defaults for Performance Commentary
            # Priority: 
            # 1. If Override Checkbox -> Generated Text
            # 2. Else -> Word Defaults (or empty)
            
            val_perf_comm = word_defaults.get("performance_commentary", "")
            
            if override_performance:
                if perf_df.empty:
                     st.warning("⚠️ Tabela de performance (perf_table) está vazia.")
                else:
                    gen_text = generate_performance_text(perf_df, benchmark)
                    if gen_text:
                        val_perf_comm = gen_text
                    else:
                        st.warning("⚠️ Texto gerado vazio. Verifique se a tabela de performance tem as colunas 'YTD', '36M' e se as linhas de SERIE (Volatilidade/Sharpe/Indices) foram identificadas.")
                        st.dataframe(perf_df)

            performance_commentary = st.text_area(
                "Performance Commentary",
                value=val_perf_comm,
                placeholder="2024 – the fund outperformed its benchmark thanks to...",
                height=100,
                # Change key invocation if override matches to force refresh if user toggles?
                # Actually, simply using a stateful key logic or just letting Streamlit handle it.
                # If we change the key, it resets the widget.
                key=f"perfcomm_{override_performance}", 
            )

        st.divider()

        # Session persistence for single outputs
        if "factsheet_html" not in st.session_state:
            st.session_state["factsheet_html"] = None
        if "factsheet_pdf" not in st.session_state:
            st.session_state["factsheet_pdf"] = None
        if "factsheet_pdf_name" not in st.session_state:
            st.session_state["factsheet_pdf_name"] = "factsheet.pdf"
        if "factsheet_html_name" not in st.session_state:
            st.session_state["factsheet_html_name"] = "factsheet.html"

        generate_clicked = st.button(
            "🚀 Generate HTML Factsheet (Single)",
            type="primary",
            width='stretch',
            key="single_generate",
        )

        if generate_clicked:
            # Build initial details from UI (raw values)
            fund_details_raw = {
                "AuM": aum,
                "Classe do Fundo": fund_class,
                "Benchmark": benchmark,
                "Tx. Administração": admin_fee,
                "Tx. Performance": perf_fee,
                "Administrador": administrator,
                "Resgate": redemption,
            }

            # Process logo
            logo_base64 = ""
            logo_mime = "image/png"
            if logo_file:
                logo_base64 = image_to_base64(logo_file)
                logo_mime = get_image_mime_type(logo_file.name)

            # Apply catalog overrides using ORIGINAL CNPJ
            cnpj_for_details = cnpj_input or display_cnpj or meta.get("cnpj", "")
            fund_details_filled, risk_level_final, bench_final = apply_catalog_overrides(
                cnpj=cnpj_for_details,
                fund_details=fund_details_raw,
                risk_level=risk_level,
                meta_benchmark=meta.get("benchmark", ""),
                catalog_map=catalog_map,
            )

            # Format details for display
            fund_details_display = format_fund_details_for_display(
                fund_details_filled
            )

            # Prepare meta for display (header uses original name/cnpj)
            meta_display = dict(meta)
            meta_display["fund_name"] = (
                display_name or meta.get("fund_name", "")
            )
            meta_display["cnpj"] = (
                cnpj_for_details or display_cnpj or meta.get("cnpj", "")
            )
            meta_display["benchmark"] = bench_final or meta.get("benchmark", "CDI")

            # Content final values: prefer UI values, else Word info (by original CNPJ), else defaults
            wi = word_info_map.get(clean_cnpj(cnpj_for_details), {}) if word_info_map else {}
            desc_final = description or wi.get("description", "")
            carac_final = characteristics or wi.get("characteristics", "")
            team_final = team or wi.get("team", "")
            thesis_final = portfolio_thesis or wi.get("portfolio_thesis", "")
            perfcomm_final = (
                performance_commentary or wi.get("performance_commentary", "")
            )

            # Single mode: we only have current sheets; master-based data is not available.
            # Use current dataframes for charts/tables.
            html_content = generate_html_factsheet(
                meta=meta_display,
                perf_df=perf_df,
                line_df=line_df,
                alloc_df=alloc_df,
                top_df=top_df,
                description=desc_final,
                characteristics=carac_final,
                team=team_final,
                portfolio_thesis=thesis_final,
                performance_commentary=perfcomm_final,
                risk_level=risk_level_final,
                fund_details=fund_details_display,
                logo_base64=logo_base64,
                logo_mime=logo_mime,
                graph_indexer=graph_indexer,
                internal_use_only=internal_use_only,
            )

            safe_name = sanitize_filename(meta_display.get("fund_name", "fund"))
            safe_cnpj = clean_cnpj(meta_display.get("cnpj", ""))
            sub_id = meta_display.get("subclasse", "")
            sub_suffix = f"_{sub_id}" if sub_id else ""

            if safe_name and safe_cnpj:
                pdf_name = f"{safe_name}_{safe_cnpj}{sub_suffix}.pdf"
                html_name = f"{safe_name}_{safe_cnpj}{sub_suffix}.html"
            else:
                pdf_name = f"factsheet_{safe_cnpj or 'fund'}{sub_suffix}.pdf"
                html_name = f"factsheet_{safe_cnpj or 'fund'}{sub_suffix}.html"

            st.session_state["factsheet_html"] = html_content
            st.session_state["factsheet_html_name"] = html_name

            try:
                with st.spinner("Rendering PDF (Chromium headless)..."):
                    pdf_bytes = html_to_pdf_playwright(html_content)
                st.session_state["factsheet_pdf"] = pdf_bytes
                st.session_state["factsheet_pdf_name"] = pdf_name
                st.success("PDF rendered successfully (single-page size).")
            except Exception as e:
                st.session_state["factsheet_pdf"] = None
                st.error(
                    "PDF generation failed. Please ensure Playwright and "
                    "Chromium are installed: `pip install playwright` then "
                    "`playwright install chromium`. "
                    f"Error: {e}"
                )

        # Show preview and download buttons if available
        if st.session_state["factsheet_html"]:
            st.subheader("Preview")
            st.components.v1.html(
                st.session_state["factsheet_html"], height=900, scrolling=True
            )

            col_dl1, col_dl2 = st.columns(2)
            with col_dl1:
                st.download_button(
                    label="📥 Download HTML",
                    data=st.session_state["factsheet_html"],
                    file_name=st.session_state["factsheet_html_name"],
                    mime="text/html",
                    width='stretch',
                )
            with col_dl2:
                if st.session_state["factsheet_pdf"] is not None:
                    st.download_button(
                        label="📄 Download PDF",
                        data=st.session_state["factsheet_pdf"],
                        file_name=st.session_state["factsheet_pdf_name"],
                        mime="application/pdf",
                        width='stretch',
                    )
                else:
                    st.info(
                        "PDF not available. Generate again after installing "
                        "Playwright/Chromium."
                    )

        with st.expander("📊 Preview loaded data"):
            tab1, tab2, tab3, tab4 = st.tabs(
                ["Performance", "Line Index", "Allocation", "Top Positions"]
            )
            with tab1:
                st.dataframe(perf_df, width='stretch')
            with tab2:
                st.dataframe(
                    line_df.head(20) if not line_df.empty else line_df,
                    width='stretch',
                )
            with tab3:
                st.dataframe(alloc_df, width='stretch')
            with tab4:
                st.dataframe(top_df, width='stretch')

    st.divider()

    # ---------------------- Batch mode ----------------------
    st.header("Batch Generation")
    st.markdown(
        """
        **How batch mode works:**
        
        1. Upload Excel files that contain **master_cnpj** data (from Export Factsheet Data).
        2. The app looks up each file's CNPJ in the mapping dictionary.
        3. If found in the mapping (by master_cnpj), the app uses:
           - **original_name** and **original_cnpj** for the PDF header and filename
           - **original_cnpj** to look up fund details in the catalog
           - **master_cnpj** sheets for performance graph, rentabilidade table, and Top 10
           - **graph** field (CDI/IBOV) to determine the chart comparator
        4. If not found in mapping, the file's own data is used as-is.
        """
    )
    batch_files = st.file_uploader(
        "Upload multiple Factsheet Excel files (.xlsx)",
        type=["xlsx"],
        accept_multiple_files=True,
        key="batch_uploader",
    )

    # Content sections to apply to all in batch
    with st.expander("Batch content (applied to all funds)"):
        b_desc = st.text_area(
            "Descrição do Fundo (batch)",
            value="",
            height=80,
            key="b_desc",
        )
        b_carac = st.text_area(
            "Características Principais (batch)",
            value="",
            height=80,
            key="b_carac",
        )
        b_team = st.text_area(
            "Equipe de Gestão (batch)",
            value="",
            height=80,
            key="b_team",
        )
        b_thesis = st.text_area(
            "Portfólio e Tese de Investimento (batch)",
            value="",
            height=80,
            key="b_thesis",
        )
        b_perfcomm = st.text_area(
            "Performance Commentary (batch)",
            value="",
            height=80,
            key="b_perfcomm",
        )

    default_batch_risk = st.selectbox(
        "Default Risk for batch (used if catalog absent/missing)",
        ["CONSERVADOR", "MODERADO", "SOFISTICADO"],
        index=1,
        key="b_risk_default",
    )

    if "batch_results" not in st.session_state:
        st.session_state["batch_results"] = []

    generate_batch = st.button(
        "🚀 Generate Batch",
        type="primary",
        width='stretch',
        key="batch_generate_btn",
    )

    if generate_batch:
        if not batch_files:
            st.warning("Please upload at least one factsheet Excel file.")
        else:
            # Preload all files and index by cleaned CNPJ (the CNPJ in the file = master_cnpj)
            preloaded = []
            idx_by_master_cnpj: dict[str, dict] = {}

            for uf in batch_files:
                try:
                    sheets = read_excel_sheets(uf)
                    meta = parse_meta(sheets.get("meta", pd.DataFrame()))
                    # The CNPJ in the uploaded file IS the master_cnpj
                    file_cnpj_clean = clean_cnpj(meta.get("cnpj", ""))
                    try:
                        excel_bytes = uf.getvalue()
                    except Exception:
                        uf.seek(0)
                        excel_bytes = uf.read()
                    rec = {
                        "file": uf,
                        "sheets": sheets,
                        "meta": meta,
                        "bytes": excel_bytes,
                        "file_cnpj_clean": file_cnpj_clean,
                    }
                    preloaded.append(rec)
                    if file_cnpj_clean:
                        if file_cnpj_clean not in idx_by_master_cnpj:
                            idx_by_master_cnpj[file_cnpj_clean] = []
                        idx_by_master_cnpj[file_cnpj_clean].append(rec)
                except Exception as e:
                    st.error(f"Failed preloading {uf.name}: {e}")

            # Now we need to generate factsheets for each MAPPING entry
            # that has a master_cnpj present in idx_by_master_cnpj
            # Plus any uploaded files that are NOT in the mapping (use as-is)

            results = []
            processed_master_cnpjs = set()

            # First pass: process all mapping entries that have uploaded master data
            mappings_to_process = []
            for mapping in FUND_MAPPING_RAW:
                master_cnpj_clean = clean_cnpj(mapping.get("master_cnpj", ""))
                if master_cnpj_clean in idx_by_master_cnpj:
                    mappings_to_process.append(mapping)
                    processed_master_cnpjs.add(master_cnpj_clean)

            # Second pass: find uploaded files NOT covered by any mapping
            # (their file CNPJ is not a master_cnpj in mapping)
            unmapped_files = []
            for rec in preloaded:
                if rec["file_cnpj_clean"] not in processed_master_cnpjs:
                    # Check if this file's CNPJ is a master_cnpj for any mapping
                    is_master = any(
                        clean_cnpj(m.get("master_cnpj", "")) == rec["file_cnpj_clean"]
                        for m in FUND_MAPPING_RAW
                    )
                    if not is_master:
                        unmapped_files.append(rec)

            total = len(mappings_to_process) + len(unmapped_files)
            if total == 0:
                st.warning(
                    "No funds to process. Check that uploaded files match "
                    "mapping master_cnpj values."
                )
            else:
                progress = st.progress(0, text=f"Processing 0/{total} funds...")
                status_line = st.empty()
                log_container = st.container()

                # Process logo once
                logo_base64 = ""
                logo_mime = "image/png"
                if logo_file:
                    try:
                        logo_base64 = image_to_base64(logo_file)
                        logo_mime = get_image_mime_type(logo_file.name)
                    except Exception:
                        pass

                # Word info map (by original CNPJ)
                word_info_map = st.session_state.get("word_info_map") or {}

                idx = 0

                # Process mapped funds
                for mapping in mappings_to_process:
                    idx += 1
                    original_name = (
                        str(mapping.get("original_name", ""))
                        .replace("\u00a0", " ")
                        .strip()
                    )
                    original_cnpj = (
                        str(mapping.get("original_cnpj", ""))
                        .replace("\u00a0", " ")
                        .strip()
                    )
                    master_cnpj = (
                        str(mapping.get("master_cnpj", ""))
                        .replace("\u00a0", " ")
                        .strip()
                    )
                    graph_indexer = (
                        str(mapping.get("graph", "CDI")).strip().upper()
                    )

                    master_cnpj_clean = clean_cnpj(master_cnpj)
                    original_cnpj_clean = clean_cnpj(original_cnpj)

                    percent = int((idx - 1) / total * 100)
                    progress.progress(
                        percent,
                        text=f"Processing {idx}/{total}: {original_name}",
                    )
                    status_line.info(
                        f"Processing fund {idx} of {total}: "
                        f"{original_name} ({original_cnpj})"
                    )

                    try:
                        # Get master data sheets (list of one or more files/subclasses)
                        master_recs = idx_by_master_cnpj.get(master_cnpj_clean, [])
                        if not master_recs:
                            with log_container:
                                st.warning(
                                    f"Master data not found for {original_name} "
                                    f"(master: {master_cnpj})"
                                )
                            continue

                        for master_rec in master_recs:
                            sheets_master = master_rec["sheets"]
                            meta_master = master_rec["meta"]
                            sub_id = meta_master.get("subclasse", "")
                            sub_suffix = f"_{sub_id}" if sub_id else ""

                            # Use master sheets for chart, perf table, allocation, top positions
                            line_df = sheets_master.get("line_index", pd.DataFrame())
                            alloc_df = sheets_master.get("alloc_donut", pd.DataFrame())
                            top_df = sheets_master.get("top_positions", pd.DataFrame())
                            perf_df = sheets_master.get("perf_table", pd.DataFrame())

                            # Build details via catalog using ORIGINAL CNPJ
                            details_raw = {
                                "AuM": "",
                                "Classe do Fundo": "",
                                "Benchmark": meta_master.get("benchmark", "CDI"),
                                "Tx. Administração": "",
                                "Tx. Performance": "",
                                "Administrador": "",
                                "Resgate": "",
                            }

                            details_filled, risk_final, bench_final = (
                                apply_catalog_overrides(
                                    cnpj=original_cnpj,  # Use original_cnpj for catalog lookup
                                    fund_details=details_raw,
                                    risk_level=default_batch_risk,
                                    meta_benchmark=meta_master.get("benchmark", ""),
                                    catalog_map=catalog_map,
                                )
                            )
                            details_display = format_fund_details_for_display(
                                details_filled
                            )

                            # Build meta for display with original name/cnpj
                            meta_display = dict(meta_master)
                            meta_display["fund_name"] = original_name
                            meta_display["cnpj"] = original_cnpj
                            meta_display["benchmark"] = (
                                bench_final or meta_master.get("benchmark", "CDI")
                            )

                            # Per-fund content from Word by ORIGINAL CNPJ, else batch defaults
                            wi = word_info_map.get(original_cnpj_clean, {})
                            desc_use = wi.get("description", "") or b_desc
                            carac_use = wi.get("characteristics", "") or b_carac
                            team_use = wi.get("team", "") or b_team
                            thesis_use = wi.get("portfolio_thesis", "") or b_thesis
                            perfcomm_use = wi.get("performance_commentary", "") or b_perfcomm
                            
                            # Override performance text check
                            if override_performance and not perf_df.empty:
                                gen_text = generate_performance_text(perf_df, meta_display.get("benchmark", ""))
                                if gen_text:
                                    perfcomm_use = gen_text

                            html_content = generate_html_factsheet(
                                meta=meta_display,
                                perf_df=perf_df,
                                line_df=line_df,
                                alloc_df=alloc_df,
                                top_df=top_df,
                                description=desc_use,
                                characteristics=carac_use,
                                team=team_use,
                                portfolio_thesis=thesis_use,
                                performance_commentary=perfcomm_use,
                                risk_level=risk_final,
                                fund_details=details_display,
                                logo_base64=logo_base64,
                                logo_mime=logo_mime,
                                graph_indexer=graph_indexer,
                                internal_use_only=internal_use_only,
                            )

                            # Filename: {original_name}_{original_cnpj}_{sub}.pdf
                            safe_name = sanitize_filename(original_name)
                            safe_cnpj = original_cnpj_clean
                            
                            if safe_name and safe_cnpj:
                                html_name = f"{safe_name}_{safe_cnpj}{sub_suffix}.html"
                                pdf_name = f"{safe_name}_{safe_cnpj}{sub_suffix}.pdf"
                            else:
                                html_name = f"factsheet_{safe_cnpj or 'fund'}{sub_suffix}.html"
                                pdf_name = f"factsheet_{safe_cnpj or 'fund'}{sub_suffix}.pdf"

                            pdf_bytes = b""
                            try:
                                pdf_bytes = html_to_pdf_playwright(html_content)
                                with log_container:
                                    st.success(f"Rendered PDF: {pdf_name}")
                            except Exception as e:
                                with log_container:
                                    st.warning(
                                        f"PDF render failed for {original_name}{sub_suffix}: {e}. "
                                        "HTML still generated."
                                    )

                            results.append(
                                {
                                    "fund_name": original_name + (f" ({sub_id})" if sub_id else ""),
                                    "cnpj": original_cnpj,
                                    "html_name": html_name,
                                    "html_bytes": html_content.encode("utf-8"),
                                    "pdf_name": pdf_name,
                                    "pdf_bytes": pdf_bytes,
                                    "excel_name": master_rec["file"].name,
                                    "excel_bytes": master_rec["bytes"],
                                }
                            )
                            with log_container:
                                st.info(f"Generated HTML: {html_name}")

                    except Exception as e:
                        with log_container:
                            st.error(f"Failed to process {original_name}: {e}")

                    percent = int(idx / total * 100)
                    progress.progress(
                        percent, text=f"Processing {idx}/{total}: {original_name}"
                    )

                # Process unmapped files (use their own data as-is)
                for rec in unmapped_files:
                    idx += 1
                    meta = rec["meta"]
                    fund_name = meta.get("fund_name", "Unknown Fund")
                    fund_cnpj = meta.get("cnpj", "")

                    percent = int((idx - 1) / total * 100)
                    progress.progress(
                        percent, text=f"Processing {idx}/{total}: {fund_name}"
                    )
                    status_line.info(
                        f"Processing fund {idx} of {total}: {fund_name} (unmapped)"
                    )

                    try:
                        sheets = rec["sheets"]
                        line_df = sheets.get("line_index", pd.DataFrame())
                        alloc_df = sheets.get("alloc_donut", pd.DataFrame())
                        top_df = sheets.get("top_positions", pd.DataFrame())
                        perf_df = sheets.get("perf_table", pd.DataFrame())

                        details_raw = {
                            "AuM": "",
                            "Classe do Fundo": "",
                            "Benchmark": meta.get("benchmark", "CDI"),
                            "Tx. Administração": "",
                            "Tx. Performance": "",
                            "Administrador": "",
                            "Resgate": "",
                        }

                        details_filled, risk_final, bench_final = (
                            apply_catalog_overrides(
                                cnpj=fund_cnpj,
                                fund_details=details_raw,
                                risk_level=default_batch_risk,
                                meta_benchmark=meta.get("benchmark", ""),
                                catalog_map=catalog_map,
                            )
                        )
                        details_display = format_fund_details_for_display(
                            details_filled
                        )

                        meta_display = dict(meta)
                        meta_display["benchmark"] = (
                            bench_final or meta.get("benchmark", "CDI")
                        )

                        # Try Word content by this fund's original CNPJ (unmapped so we only have this cnpj)
                        wi = {}
                        if st.session_state.get("word_info_map"):
                            wi = st.session_state["word_info_map"].get(
                                clean_cnpj(fund_cnpj), {}
                            )
                        desc_use = wi.get("description", "") or b_desc
                        carac_use = wi.get("characteristics", "") or b_carac
                        team_use = wi.get("team", "") or b_team
                        thesis_use = wi.get("portfolio_thesis", "") or b_thesis
                        perfcomm_use = (
                            wi.get("performance_commentary", "") or b_perfcomm
                        )
                        
                        # Override performance text check
                        if override_performance and not perf_df.empty:
                            gen_text = generate_performance_text(perf_df, meta_display.get("benchmark", ""))
                            if gen_text:
                                perfcomm_use = gen_text

                        html_content = generate_html_factsheet(
                            meta=meta_display,
                            perf_df=perf_df,
                            line_df=line_df,
                            alloc_df=alloc_df,
                            top_df=top_df,
                            description=desc_use,
                            characteristics=carac_use,
                            team=team_use,
                            portfolio_thesis=thesis_use,
                            performance_commentary=perfcomm_use,
                            risk_level=risk_final,
                            fund_details=details_display,
                            logo_base64=logo_base64,
                            logo_mime=logo_mime,
                            graph_indexer=None,
                            internal_use_only=internal_use_only,
                        )

                        safe_name = sanitize_filename(fund_name)
                        safe_cnpj = clean_cnpj(fund_cnpj)
                        sub_id = meta.get("subclasse", "")
                        sub_suffix = f"_{sub_id}" if sub_id else ""

                        if safe_name and safe_cnpj:
                            html_name = f"{safe_name}_{safe_cnpj}{sub_suffix}.html"
                            pdf_name = f"{safe_name}_{safe_cnpj}{sub_suffix}.pdf"
                        else:
                            html_name = f"factsheet_{safe_cnpj or 'fund'}{sub_suffix}.html"
                            pdf_name = f"factsheet_{safe_cnpj or 'fund'}{sub_suffix}.pdf"

                        pdf_bytes = b""
                        try:
                            pdf_bytes = html_to_pdf_playwright(html_content)
                            with log_container:
                                st.success(f"Rendered PDF: {pdf_name}")
                        except Exception as e:
                            with log_container:
                                st.warning(
                                    f"PDF render failed for {fund_name}: {e}. "
                                    "HTML still generated."
                                )

                        results.append(
                            {
                                "fund_name": fund_name,
                                "cnpj": fund_cnpj,
                                "html_name": html_name,
                                "html_bytes": html_content.encode("utf-8"),
                                "pdf_name": pdf_name,
                                "pdf_bytes": pdf_bytes,
                                "excel_name": rec["file"].name,
                                "excel_bytes": rec["bytes"],
                            }
                        )
                        with log_container:
                            st.info(f"Generated HTML: {html_name}")

                    except Exception as e:
                        with log_container:
                            st.error(f"Failed to process {fund_name}: {e}")

                    percent = int(idx / total * 100)
                    progress.progress(
                        percent, text=f"Processing {idx}/{total}: {fund_name}"
                    )

                status_line.success(f"Completed processing {total} funds.")
                st.session_state["batch_results"] = results
                if results:
                    st.success(f"Generated {len(results)} factsheets in batch.")

    # Batch downloads
    if st.session_state.get("batch_results"):
        results = st.session_state["batch_results"]
        st.subheader("Batch Downloads")

        html_pairs = [(r["html_name"], r["html_bytes"]) for r in results]
        pdf_pairs = [
            (r["pdf_name"], r["pdf_bytes"]) for r in results if r["pdf_bytes"]
        ]
        excel_pairs = [(r["excel_name"], r["excel_bytes"]) for r in results]

        html_zip = make_zip(html_pairs) if html_pairs else b""
        pdf_zip = make_zip(pdf_pairs) if pdf_pairs else b""
        excel_zip = make_zip(excel_pairs) if excel_pairs else b""

        c1, c2, c3 = st.columns(3)
        with c1:
            st.download_button(
                "📦 Download all HTMLs (zip)",
                data=html_zip,
                file_name="factsheets_html.zip",
                mime="application/zip",
                width='stretch',
                disabled=not html_pairs,
            )
        with c2:
            st.download_button(
                "📦 Download all PDFs (zip)",
                data=pdf_zip,
                file_name="factsheets_pdf.zip",
                mime="application/zip",
                width='stretch',
                disabled=not pdf_pairs,
            )
        with c3:
            st.download_button(
                "📦 Download original Excel uploads (zip)",
                data=excel_zip,
                file_name="factsheets_excel.zip",
                mime="application/zip",
                width='stretch',
                disabled=not excel_pairs,
            )

        with st.expander("Batch summary"):
            df_sum = pd.DataFrame(
                [
                    {
                        "Fund Name": r["fund_name"],
                        "CNPJ": r["cnpj"],
                        "HTML": r["html_name"],
                        "PDF": r["pdf_name"] if r["pdf_bytes"] else "PDF failed",
                        "Excel": r["excel_name"],
                    }
                    for r in results
                ]
            )
            st.dataframe(df_sum, width='stretch')


if __name__ == "__main__":
    main()