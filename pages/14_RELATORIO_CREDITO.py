# app.py
# Streamlit app to generate "Relatório de Risco de Crédito"
# from a portfolio CSV (pipe-delimited) OR a local SQLite DB
# (gorila_positions.db in project folder).
# Now:
# - Choose data source: CSV or SQLite (.db)
# - When using DB, FIXED table: pmv_plus_gorila; list unique reference_date
#   values to pick from
# - Issuer concentration uses parsed_company_name (with safe fallback)
# - Toggle to hide rating section
# - All checks/percentages are based on PL em Renda Fixa (Crédito + Tesouro)
#   (and we also show "% do PL" as extra info)
# - Reference date uses ONLY the 'reference_date' column (no lookup in raw)
# - Funds: If asset_class indicates a fund (e.g., "Fixed income fund"),
#   treat it as Fund (not Crédito), even if security_type == "BONDS".
# - Country risk exposure uses ONLY the 'country_risk' column (raw)
# - New: Country Risk exposure and Currency exposure sections (dash + report)
# - New: Vencimentos separados em buckets: <1 ano, <3 anos, <5 anos, +5 anos
#   (dash + report)
# - New: Exposure by parsed_bond_type (dash + report) — exact values
# - New: Custódia em Fundos (apenas) no dashboard
# - New: Donut charts rendered as static PNG images (Matplotlib) embedded
#   in the exported HTML report for Moeda, Vencimento e Tipo de Título
#   (PL em Renda Fixa), so they are preserved when opening in Word/PDF
# - Update: Separate graphs for BRL and USD in Factor Risk and Bond Type
#   (dashboard and report)
# - Update (Nov/2025):
#   • Donuts in report are larger, white background, labels+percentages inside
#     wedges with white tiny font (configurable sliders).
#   • Sliders in sidebar control donut size/DPI/font/ring thickness/etc used
#     in HTML generation.
#   • Dashboard graph heights adjustable (larger by default).
#   • Report 3.2: add BRL and USD sub-sections with Top-10 issuer tables
#     (fit to A4).
#   • Country exposure: charts removed; use table in dashboard and report.
#   • Added CSV downloads for PL and PL em Renda Fixa (European format).
#
# - Customization (Dec/2025):
#   • Exclude Instituições S1 da análise de concentração (mas manter no PL em
#     Renda Fixa)
#   • Gráfico (Relatório): % do PL em Renda Fixa alocado em Risco Soberano e
#     Instituições S1 — formato BARRA
#   • Removido Stress Testing (seção)
#   • Adicionadas seções 1 (Objetivo) e 2 (Metodologia) no início do relatório
#   • Tabelas do relatório centralizadas (não ocupar 100% da largura)
# - Change (Dec/2025):
#   • Rating part: add bar graphs (BRL and USD) both in Dashboard and Report.
#   • Normalize rating value "NONE" (and common equivalents) to "Sem Rating".
#   • Order all rating bars by rating scale (AAA at top, Sem Rating at bottom).

import base64
import json
import re
import csv as _csv
import os
import sqlite3
import math
from io import BytesIO
from contextlib import closing
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, List
import unicodedata  # Para matching acento-insensível das S1

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from jinja2 import Template

# Matplotlib para imagens estáticas no relatório (robusto para Word/PDF)
try:
    import matplotlib

    matplotlib.use("Agg")  # backend headless
    import matplotlib.pyplot as plt

    HAS_MPL = True
except Exception:
    HAS_MPL = False

# ---------------------------- Config ---------------------------- #

st.set_page_config(
    page_title="Relatório de Risco de Crédito",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Autenticação simples
if not st.session_state.get("authenticated", False):
    st.warning("Please enter the password on the Home page first.")
    st.write("Status: Não Autenticado")
    st.stop()

st.write("Autenticado")

PT_MONTHS = {
    "jan": 1,
    "fev": 2,
    "mar": 3,
    "abr": 4,
    "mai": 5,
    "jun": 6,
    "jul": 7,
    "ago": 8,
    "set": 9,
    "out": 10,
    "nov": 11,
    "dez": 12,
}

CREDIT_TYPES = {
    "CORPORATE_BONDS_CDB",
    "CORPORATE_BONDS_LCA",
    "CORPORATE_BONDS_LCI",
    "CORPORATE_BONDS_LIG",
    "CORPORATE_BONDS_DEBENTURE",
    "CORPORATE_BONDS_CRI",
    "CORPORATE_BONDS_CRA",
    "CORPORATE_BONDS_LF",
    "BONDS",
}

PUBLIC_BOND_TYPES = {
    "TREASURY_LOCAL_NTNB",
    "TREASURY_LOCAL_NTNC",
    "TREASURY_LOCAL_NTNF",
    "TREASURY_LOCAL_LFT",
    "TREASURY_LOCAL_LTN",
}

# Tipos que denotam fundos
FUND_TYPES = {"FUNDQUOTE"}
FUND_TYPE_ALIASES = {
    "FUND",
    "MUTUAL_FUND",
    "INVESTMENT_FUND",
    "ETF",
    "REIT",
    "FUND_SHARE",
    "FUND_UNITS",
    "FUND_UNITHOLDING",
}
FUND_TOKENS = {
    "FUND",
    "FUNDO",
    "ETF",
    "FII",
    "FIP",
    "FIDC",
    "FMIEE",
    "FMIE",
    "FIAGRO",
    "PRIVATE EQUITY FUND",
    "FIXED INCOME FUND",
    "FUNDO DE INVESTIMENTO",
}

IGNORE_IN_LIMITS_AS_TESOURO = {"TESOURO NACIONAL"}

# NEW: Lista S1 (em CAIXA ALTA, conforme solicitado)
S1_Institutions = {
    "BANCO SANTANDER (BRASIL)",
    "CAIXA ECONOMICA FEDERAL",
    "ITAU UNIBANCO",
    "BANCO BTG PACTUAL",
    "BTG PACTUAL DTVM",
    "BCO BTG PACTUAL",
}

S2_Institutions = {
    "BANCO SICOOB",
    "BANCO XP SA",
    "BANCO SAFRA",
    "BNDES",
    "BANCO XP",
}
# Alias em CAIXA ALTA
S1_INSTITUTIONS = set(S1_Institutions)
S2_INSTITUTIONS = set(S2_Institutions)


def _normalize_no_accents_upper(s: Optional[str]) -> str:
    if s is None:
        return ""
    s = str(s)
    norm = unicodedata.normalize("NFKD", s)
    no_acc = "".join(c for c in norm if not unicodedata.combining(c))
    return no_acc.upper().strip()


S1_TOKENS_NORM = {_normalize_no_accents_upper(x) for x in S1_INSTITUTIONS}
S2_TOKENS_NORM = {_normalize_no_accents_upper(x) for x in S2_INSTITUTIONS}


def is_s1_from_parsed_company(name: Optional[str]) -> bool:
    if not isinstance(name, str) or not name.strip():
        return False
    u = _normalize_no_accents_upper(name)
    for tok in S1_TOKENS_NORM:
        if tok in u:
            return True
    return False


def is_s2_from_parsed_company(name: Optional[str]) -> bool:
    if not isinstance(name, str) or not name.strip():
        return False
    u = _normalize_no_accents_upper(name)
    for tok in S2_TOKENS_NORM:
        if tok in u:
            return True
    return False


# Limites padrão (editáveis)
DEFAULT_LIMITS = {
    "max_single_issuer": 0.15,
    "max_top10_issuers": 0.50,
    "max_prefixed": 0.10,
    "max_maturity_over_5y": 0.45,  # > 5 anos
    "min_sovereign_or_aaa": 0.40,
}

# ---------------------------- Paleta de Cores ---------------------------- #

PALETA_CORES = [
    "#013220",
    "#8c6239",
    "#57575a",
    "#b08568",
    "#09202e",
    "#582308",
    "#7a6200",
]

# ---------------------------- Rating Ordering ---------------------------- #

RATING_ORDER = [
    "AAA",
    "AA+",
    "AA",
    "AA-",
    "A+",
    "A",
    "A-",
    "BBB+",
    "BBB",
    "BBB-",
    "BB+",
    "BB",
    "BB-",
    "B+",
    "B",
    "B-",
    "CCC+",
    "CCC",
    "CCC-",
    "CC",
    "C",
    "D",
    "Sem Rating",
]


def normalize_rating_label(x: Optional[str]) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip()
    if not s or s == "########":
        return None
    u = s.upper()
    if u in {
        "NONE",
        "NENHUM",
        "SEM RATING",
        "SEM_RATING",
        "NA",
        "N/A",
        "-",
        "NR",
        "UNRATED",
        "NOT RATED",
        "SEM CLASSIFICAÇÃO",
    }:
        return None
    return u


def rating_sort_key(label: str) -> int:
    """Return order index; unknown ratings go before 'Sem Rating', and 'Sem Rating' last."""
    if label is None or str(label).strip() == "":
        label = "Sem Rating"
    lab = str(label).strip()
    # unify casing for matching
    if lab.upper() in {"NONE", "NR", "UNRATED", "NOT RATED"}:
        lab = "Sem Rating"
    # Keep exact "Sem Rating" case at bottom
    if lab == "Sem Rating":
        return len(RATING_ORDER) - 1
    try:
        return RATING_ORDER.index(lab)
    except ValueError:
        # Place unknown just before 'Sem Rating'
        return len(RATING_ORDER) - 2


# ---------------------------- CSV Reader ---------------------------- #


def read_portfolio_csv(uploaded_file) -> pd.DataFrame:
    """
    Leitor robusto para CSV com delimitador variável.
    """
    if uploaded_file is None:
        return pd.DataFrame()

    head = uploaded_file.read(131072)
    uploaded_file.seek(0)
    sample = head.decode("utf-8-sig", errors="replace")
    try:
        dialect = _csv.Sniffer().sniff(sample, delimiters=[",", ";", "|", "\t"])
        sep = dialect.delimiter
    except Exception:
        sep = "|"

    try:
        df = pd.read_csv(
            uploaded_file,
            sep=sep,
            engine="python",
            dtype=str,
            keep_default_na=False,
            skipinitialspace=True,
        )
    except pd.errors.ParserError:
        uploaded_file.seek(0)
        df = pd.read_csv(
            uploaded_file,
            sep="|",
            engine="python",
            dtype=str,
            keep_default_na=False,
            skipinitialspace=True,
        )

    df = df.loc[:, [c for c in df.columns if c.strip() != ""]]
    df.columns = [c.strip() for c in df.columns]
    return df


# ---------------------------- SQLite Reader ---------------------------- #


def _sqlite_list_tables(conn: sqlite3.Connection) -> List[str]:
    q = (
        "SELECT name FROM sqlite_master WHERE type='table' "
        "AND name NOT LIKE 'sqlite_%'"
    )
    with closing(conn.cursor()) as cur:
        cur.execute(q)
        rows = cur.fetchall()
    return [r[0] for r in rows]


def _sqlite_table_columns(conn: sqlite3.Connection, table: str) -> List[str]:
    with closing(conn.cursor()) as cur:
        cur.execute(f"PRAGMA table_info('{table}')")
        rows = cur.fetchall()
        return [row[1] for row in rows]


def detect_tables_with_reference_date(conn: sqlite3.Connection) -> List[str]:
    """
    Retorna tabelas com coluna 'reference_date'.
    """
    tables = _sqlite_list_tables(conn)
    out = []
    for t in tables:
        cols = [c.lower() for c in _sqlite_table_columns(conn, t)]
        if "reference_date" in cols:
            out.append(t)
    return out


def list_unique_reference_dates(conn: sqlite3.Connection, table: str) -> List[str]:
    """
    Retorna valores distintos de reference_date.
    """
    q = f"""
        SELECT DISTINCT reference_date
        FROM "{table}"
        WHERE reference_date IS NOT NULL
          AND TRIM(reference_date) <> ''
    """
    df = pd.read_sql_query(q, conn)
    vals = (
        df["reference_date"].astype(str).tolist()
        if "reference_date" in df.columns
        else []
    )
    seen = set()
    ordered = []
    for v in vals:
        if v not in seen:
            seen.add(v)
            ordered.append(v)
    return ordered


def read_positions_from_db_path(db_path: str, table: str) -> pd.DataFrame:
    """
    Lê tabela do SQLite para DataFrame.
    """
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(f'SELECT * FROM "{table}"', conn)
    df.columns = [c.strip() for c in df.columns]
    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = df[c].astype(str)
    return df


# ---------------------------- Helpers ---------------------------- #


def normalize_cnpj(x: Optional[str]) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip()
    if not s or s == "########":
        return None
    digits = re.sub(r"\D", "", s)
    if len(digits) == 14:
        return digits
    return None


def to_date_safe(x: str) -> Optional[datetime]:
    if pd.isna(x):
        return None
    if isinstance(x, datetime):
        return x
    s = str(x).strip()
    if not s or s == "########":
        return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            pass
    for fmt in ("%d/%m/%Y", "%d/%m/%y"):
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            pass
    if re.fullmatch(r"\d{8}", s):
        try:
            return datetime.strptime(s, "%Y%m%d")
        except Exception:
            pass
    m = re.search(r"(20\d{2})[_/-](\d{2})[_/-](\d{2})", s)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            return datetime(y, mo, d)
        except Exception:
            pass
    m2 = re.search(
        r"\b(jan|fev|mar|abr|mai|jun|jul|ago|set|out|nov|dez)[/ -](\d{2,4})\b",
        s.lower(),
    )
    if m2:
        mon_txt = m2.group(1)
        yr = int(m2.group(2))
        if yr < 100:
            yr = 2000 + yr
        mo = PT_MONTHS.get(mon_txt)
        if mo:
            try:
                return datetime(yr, mo, 1)
            except Exception:
                pass
    return None


def pct(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def as_pct_str(x: float) -> str:
    return f"{x * 100:.2f}%"


def brl(v: float) -> str:
    """Formata valor em formato brasileiro (R$ com vírgula decimal)."""
    return f"{v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def safe_float(x) -> float:
    if isinstance(x, (int, float, np.number)):
        return float(x)
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return 0.0
    s = str(x).strip().replace(".", "").replace(",", ".")
    try:
        return float(s)
    except Exception:
        return 0.0


def first_nonempty(*vals) -> Optional[str]:
    for v in vals:
        if v is None:
            continue
        if isinstance(v, str) and v.strip() and v.strip() != "########":
            return v.strip()
    return None


def clean_issuer_name(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    s = name.strip().upper()
    repl = {
        "ITAU UNIBANCO": "ITAÚ UNIBANCO",
        "ITAU": "ITAÚ UNIBANCO",
        "BANCO ITAU": "ITAÚ UNIBANCO",
        "BANCO DO BRASILIA": "BRB BANCO DE BRASÍLIA",
        "BRB BANCO DE BRASILIA": "BRB BANCO DE BRASÍLIA",
        "TESOURO": "TESOURO NACIONAL",
    }
    return repl.get(s, s)


def _normalize_text_for_fund_detection(s: str) -> str:
    if not isinstance(s, str):
        return ""
    u = s.upper()
    u = re.sub(r"[\s\-_]+", " ", u)
    return u.strip()


def is_fund_row(row: pd.Series) -> bool:
    # """
    # Heurísticas para classificar fundos.
    # """
    # t = _normalize_text_for_fund_detection(row.get("security_type", ""))
    # if t in FUND_TYPES or t in FUND_TYPE_ALIASES:
    #     return True
    # if re.search(r"\bFUND\b", t):
    #     return True

    # class_fields = [
    #     "asset_class",
    #     "asset_category",
    #     "asset_subclass",
    #     "product_type",
    #     "class",
    #     "subclass",
    #     "category",
    # ]
    # for f in class_fields:
    #     val = _normalize_text_for_fund_detection(row.get(f, ""))
    #     if not val:
    #         continue
    #     for tok in FUND_TOKENS:
    #         if tok in val:
    #             return True
    #     if re.search(r"\bFUND(O|S)?\b", val):
    #         return True
    #     if re.search(r"\bFUNDO(S)?\b", val):
    #         return True

    # text = " ".join(
    #     [
    #         _normalize_text_for_fund_detection(row.get("security_name", "")),
    #         _normalize_text_for_fund_detection(row.get("security_description", "")),
    #     ]
    # )
    # if re.search(r"\b(FUNDO|FUND|ETF|FII|FIDC|FIP)\b", text):
    #     return True
    # if re.search(r"\b(COTA|COTAS|QUOTA|QUOTAS)\b", text):
    #     return True
    val = _normalize_text_for_fund_detection(row.get("asset_class", ""))
    if val == "FIXED INCOME FUND":
        return True
    return False


def is_credit_row(row: pd.Series) -> bool:
    # Fundos nunca são crédito (inclui 'Fixed income fund')
    if is_fund_row(row):
        return False
    
    # Exclusão manual sugerida pelo usuário (FGTS e DFO)
    bt = str(row.get("parsed_bond_type", "")).strip().upper()
    if bt in ["FGTS", "DFO"]:
        return False

    t = str(row.get("security_type", "")).strip().upper()
    if t in CREDIT_TYPES:
        return True
    if t == "CUSTOM" and str(row.get("asset_class", "")).upper() == "FIXED_INCOME":
        return True
    return False


def is_treasury_row(row: pd.Series) -> bool:
    # Fundos nunca são Tesouro
    if is_fund_row(row):
        return False

    # Exclusão manual sugerida pelo usuário (FGTS e DFO)
    bt = str(row.get("parsed_bond_type", "")).strip().upper()
    if bt in ["FGTS", "DFO"]:
        return False

    t = str(row.get("security_type", "")).strip().upper()
    if t in PUBLIC_BOND_TYPES:
        return True
    nm = str(row.get("issuer_name_norm", "")).upper()
    return nm == "TESOURO NACIONAL"


def indexer_bucket(row: pd.Series) -> str:
    idx = first_nonempty(str(row.get("indexer", "")))
    if idx:
        u = idx.strip().upper()
        if "CDI" in u:
            return "CDI"
        if "IPCA" in u:
            return "IPCA"
        if "PRÉ" in u or "PRE" in u or "PREFIX" in u:
            return "PRÉ"
    text = (
        f"{row.get('security_name', '')} "
        f"{row.get('raw', '')} "
        f"{row.get('security_description', '')}"
    ).upper()
    if " IPCA" in text:
        return "IPCA"
    if " CDI" in text:
        return "CDI"
    if " PRÉ" in text or " PREFIX" in text or re.search(r"\bPRE[_ ]", text):
        return "PRÉ"
    return "OUTROS"


def parse_maturity(row: pd.Series) -> Optional[datetime]:
    mtxt = first_nonempty(row.get("parsed_maturity_date"), row.get("parsed_maturity"))
    d = to_date_safe(mtxt) if mtxt else None
    if d:
        return d
    for f in ("security_name", "security_description", "raw", "parsed_maturity"):
        d = to_date_safe(row.get(f, None))
        if d:
            return d
    return None


def issuer_from_row(row: pd.Series) -> Tuple[Optional[str], Optional[str]]:
    cnpj_raw = first_nonempty(
        row.get("security_issuer_raw"),
        row.get("issuer"),
        row.get("issuer_from_raw"),
        row.get("parsed_issuer_cnpj"),
    )
    cnpj = normalize_cnpj(cnpj_raw)

    nm = first_nonempty(
        row.get("parsed_issuer_name"),
        row.get("issuer_name_from_json"),
        row.get("issuer_name"),
        row.get("issuer_from_raw_name"),
    )

    is_public = (
        str(row.get("security_type", "")).upper() in PUBLIC_BOND_TYPES
        or str(row.get("security_name", "")).upper().startswith("NTN")
    )
    if is_public:
        nm = "TESOURO NACIONAL"

    nm = clean_issuer_name(nm) if nm else None
    return (cnpj, nm)


def assign_market_value(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    mv_col = None
    for c in ["market_value_amount", "market_value", "valor_mercado"]:
        if c in df.columns:
            mv_col = c
            break
    if mv_col:
        df["mv"] = df[mv_col].apply(safe_float)
    else:
        if "security_market_value_raw" in df.columns:
            df["mv"] = df["security_market_value_raw"].apply(safe_float)
        else:
            df["mv"] = 0.0
    return df


def normalize_currency_value(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    u = str(s).strip().upper().replace(" ", "")
    sym_map = {
        "R$": "BRL",
        "US$": "USD",
        "$": "USD",
        "€": "EUR",
        "£": "GBP",
        "¥": "JPY",
        "C$": "CAD",
        "A$": "AUD",
    }
    if u in sym_map:
        return sym_map[u]
    text_map = {
        "REAL": "BRL",
        "REAIS": "BRL",
        "DOLAR": "USD",
        "DÓLAR": "USD",
        "EURO": "EUR",
        "LIBRA": "GBP",
        "IENE": "JPY",
        "FRANCOSUICO": "CHF",
        "FRANCOSUÍÇO": "CHF",
        "YUAN": "CNY",
        "RENMINBI": "CNY",
    }
    if u in text_map:
        return text_map[u]
    if len(u) == 3 and u.isalpha():
        return u
    return None


def currency_from_row(row: pd.Series) -> str:
    raw = first_nonempty(
        row.get("security_currency"),
        row.get("currency"),
        row.get("position_currency"),
        row.get("trade_currency"),
        row.get("moeda"),
        row.get("account_currency"),
    )
    code = normalize_currency_value(raw)
    return code if code else "BRL"


def country_from_row(row: pd.Series) -> str:
    """
    País de risco vem EXCLUSIVAMENTE de 'country_risk'.
    """
    return row.get("country_risk")


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    if "raw" in df.columns:

        def unpack_raw(x):
            try:
                if isinstance(x, dict):
                    return x
                if pd.isna(x) or not str(x).strip():
                    return {}
                return json.loads(x)
            except Exception:
                return {}

        raw = df["raw"].apply(unpack_raw)
        sec = raw.apply(lambda r: r.get("security", {}))

        df["security_description"] = sec.apply(lambda s: s.get("description"))
        df["security_currency"] = sec.apply(lambda s: s.get("currency"))
        df["security_type_raw"] = sec.apply(lambda s: s.get("type"))
        df["security_asset_class_raw"] = sec.apply(lambda s: s.get("assetClass"))
        df["security_issuer_raw"] = sec.apply(lambda s: s.get("issuer"))
        df["security_market_value_raw"] = sec.apply(lambda s: s.get("marketValue"))

    if "security_type" not in df.columns:
        if "security_type_raw" in df.columns:
            df["security_type"] = df["security_type_raw"]
        else:
            df["security_type"] = ""

    if "asset_class" not in df.columns:
        if "security_asset_class_raw" in df.columns:
            df["asset_class"] = df["security_asset_class_raw"]
        else:
            df["asset_class"] = ""

    df["indexer_bucket"] = df.apply(indexer_bucket, axis=1)
    df["maturity_date"] = df.apply(parse_maturity, axis=1)

    # Garantir parsed_company_name
    if "parsed_company_name" not in df.columns:
        df["parsed_company_name"] = None

    issuers = df.apply(issuer_from_row, axis=1, result_type="expand")
    df["issuer_cnpj"] = issuers[0]
    df["issuer_name_norm"] = issuers[1].fillna("DESCONHECIDO")

    # reference_date exclusivamente
    if "reference_date" in df.columns:
        df["reference_dt"] = df["reference_date"].apply(to_date_safe)
    else:
        df["reference_dt"] = None

    df = assign_market_value(df)

    # Flags
    df["is_fund"] = df.apply(is_fund_row, axis=1)
    df["is_credit"] = df.apply(is_credit_row, axis=1)
    df["is_treasury"] = df.apply(is_treasury_row, axis=1)

    # Bucket emissor
    def issuer_bucket_name(row: pd.Series) -> str:
        if bool(row.get("is_treasury", False)):
            return "TESOURO NACIONAL"
        pc = row.get("parsed_company_name")
        if isinstance(pc, str) and pc.strip() and pc.strip() != "########":
            return clean_issuer_name(pc) or "DESCONHECIDO"
        return row.get("issuer_name_norm", "DESCONHECIDO") or "DESCONHECIDO"

    df["issuer_bucket"] = df.apply(issuer_bucket_name, axis=1)

    # Moeda e País
    df["currency_code"] = df.apply(currency_from_row, axis=1)
    df["country_bucket"] = df.apply(country_from_row, axis=1)

    # Bucket de fundo (nome)
    def fund_bucket_name(row: pd.Series) -> Optional[str]:
        if not bool(row.get("is_fund", False)):
            return None
        return (
            first_nonempty(
                row.get("fund_name"),
                row.get("security_name"),
                row.get("security_description"),
                row.get("ticker"),
                row.get("symbol"),
                row.get("asset_name"),
                row.get("product_name"),
                row.get("name"),
            )
            or "Fundo (sem nome)"
        )

    df["fund_bucket"] = df.apply(fund_bucket_name, axis=1)

    # NEW: Flag S1 and S2 via parsed_company_name
    df["is_s1"] = df["parsed_company_name"].apply(is_s1_from_parsed_company)
    df["is_s2"] = df["parsed_company_name"].apply(is_s2_from_parsed_company)

    # Normalizar coluna de rating (preferir 'rating'; fallback para 'ratings')
    if "rating" not in df.columns and "ratings" in df.columns:
        df["rating"] = df["ratings"]

    if "rating" in df.columns:
        df["rating"] = df["rating"].apply(normalize_rating_label)
    else:
        df["rating"] = None

    return df


def compute_nav(df: pd.DataFrame) -> float:
    return float(df["mv"].sum())


def filter_bonds_by_type(
    df: pd.DataFrame, bond_types: List[str], currency: Optional[str] = None
) -> pd.DataFrame:
    """
    Filtra bonds por tipo e moeda.
    bond_types: lista de security_type values
    (ex: ["CORPORATE_BONDS_CRI", "CORPORATE_BONDS_CRA"])
    currency: "BRL", "USD", ou None para todas
    """
    filtered = df.copy()
    if bond_types:
        filtered = filtered.loc[
            filtered["security_type"].astype(str).str.upper().isin(
                [bt.upper() for bt in bond_types]
            )
        ]
    if currency:
        filtered = filtered.loc[filtered["currency_code"] == currency]
    return filtered


def compute_rating_distribution_by_group(
    df_subset: pd.DataFrame, group_total: float
) -> List[Dict[str, str]]:
    """
    Calcula distribuição de ratings para um grupo específico.
    Retorna lista de dicts com rating, mv, pct (relativo ao group_total).
    """
    if df_subset.empty or group_total <= 0:
        return []

    # Agrupar por rating
    if "rating" not in df_subset.columns:
        return []

    # Mapeia para labels finais e agrega
    tmp = df_subset.copy()
    tmp["rating_label"] = tmp["rating"].apply(
        lambda x: "Sem Rating" if normalize_rating_label(x) is None else normalize_rating_label(x)
    )
    rating_groups = tmp.groupby("rating_label", dropna=False)["mv"].sum().reset_index()
    rating_groups.columns = ["rating", "mv"]

    # Calcular percentuais
    rating_groups["pct"] = rating_groups["mv"].apply(
        lambda v: pct(float(v), group_total)
    )

    # Ordenar por nossa escala (AAA topo, Sem Rating fundo)
    rating_groups["order"] = rating_groups["rating"].apply(rating_sort_key)
    rating_groups = rating_groups.sort_values(["order", "mv"], ascending=[True, False])

    # Converter para lista de dicts
    rows = []
    for _, row in rating_groups.iterrows():
        rows.append(
            {
                "rating": str(row["rating"]),
                "mv": float(row["mv"]),
                "pct": as_pct_str(float(row["pct"])),
            }
        )

    return rows


def compute_sovereign_or_aaa_share(
    df: pd.DataFrame,
    denom_total: float,
    ratings_map: Optional[pd.DataFrame] = None,
) -> Tuple[float, pd.DataFrame]:
    """
    Share de Soberano/AAA relativo a denom_total.
    """
    if denom_total <= 0:
        return 0.0, pd.DataFrame()

    sovereign_mv = df.loc[df["issuer_bucket"] == "TESOURO NACIONAL", "mv"].sum()

    aaa_mv = 0.0
    join_df = pd.DataFrame()

    # Primeiro tentar usar ratings da coluna do dataframe (rating)
    if "rating" in df.columns:
        aaa_mv = df.loc[df["rating"].astype(str).str.upper() == "AAA", "mv"].sum()
        join_df = df[
            ["issuer_cnpj", "issuer_name_norm", "issuer_bucket", "mv", "rating"]
        ].copy()
        join_df["rating_bucket"] = join_df["rating"]
    # Fallback para ratings_map (compatibilidade)
    elif ratings_map is not None and not ratings_map.empty:
        m = ratings_map.copy()
        m.columns = [c.strip().lower() for c in m.columns]
        if "issuer_name" in m.columns:
            m["issuer_name_norm"] = m["issuer_name"].map(
                lambda x: clean_issuer_name(x) or "DESCONHECIDO"
            )
        else:
            m["issuer_name_norm"] = "DESCONHECIDO"

        left = df[["issuer_cnpj", "issuer_name_norm", "issuer_bucket", "mv"]].copy()
        left["issuer_cnpj"] = left["issuer_cnpj"].fillna("")

        right = m[["issuer_cnpj", "issuer_name_norm", "rating_bucket"]].copy()
        if "issuer_cnpj" not in right.columns:
            right["issuer_cnpj"] = ""
        right["issuer_cnpj"] = right["issuer_cnpj"].fillna("")

        join_df = left.merge(right, on=["issuer_cnpj", "issuer_name_norm"], how="left")

        no_hit = join_df["rating_bucket"].isna()
        if no_hit.any():
            j2 = left[no_hit].drop(columns=["issuer_cnpj"]).merge(
                right.drop(columns=["issuer_cnpj"]).drop_duplicates(),
                on=["issuer_name_norm"],
                how="left",
            )
            join_df.loc[no_hit, "rating_bucket"] = j2["rating_bucket"].values

        aaa_mv = join_df.loc[
            join_df["rating_bucket"].astype(str).str.upper() == "AAA", "mv"
        ].sum()

    share = pct(sovereign_mv + aaa_mv, denom_total)
    return share, join_df


def compute_concentration_by_bucket(
    df_subset: pd.DataFrame,
    denom_total: float,
    bucket_col: str = "issuer_bucket",
    exclude_set: Optional[set] = None,
) -> Dict[str, object]:
    """
    Concentração por bucket (ex.: emissor via parsed_company_name),
    excluindo nomes no exclude_set (ex.: Tesouro, S1).
    """
    if exclude_set is None:
        exclude_set = set()
    ex = df_subset.loc[~df_subset[bucket_col].isin(exclude_set)].copy()
    by = ex.groupby(bucket_col, dropna=False)["mv"].sum().sort_values(ascending=False)
    largest_mv = float(by.iloc[0]) if len(by) > 0 else 0.0
    top10_mv = float(by.head(10).sum()) if len(by) > 0 else 0.0
    largest_share = pct(largest_mv, denom_total)
    top10_share = pct(top10_mv, denom_total)
    return {
        "largest_mv": largest_mv,
        "top10_mv": top10_mv,
        "largest_share": largest_share,
        "top10_share": top10_share,
        "table": by,
    }


def compute_factor_risk_shares(
    df_subset: pd.DataFrame, denom_total: float
) -> Dict[str, float]:
    if denom_total <= 0 or df_subset.empty:
        return {"PRÉ": 0.0, "CDI": 0.0, "IPCA": 0.0, "OUTROS": 0.0}
    fr = df_subset.groupby("indexer_bucket")["mv"].sum()
    out = {"PRÉ": 0.0, "CDI": 0.0, "IPCA": 0.0, "OUTROS": 0.0}
    for k in out.keys():
        out[k] = pct(float(fr.get(k, 0.0)), denom_total)
    return out


def compute_maturity_share_over_5y(
    credit_df: pd.DataFrame, ref_date: datetime, denom_total: float
) -> Tuple[float, pd.DataFrame]:
    """
    Share de vencimentos > 5 anos relativo a denom_total (crédito privado).
    """
    five_years = ref_date + timedelta(days=int(365.25 * 5))
    if credit_df.empty:
        return 0.0, credit_df
    tmp = credit_df.copy()
    tmp["over_5y"] = tmp["maturity_date"].map(
        lambda d: d is not None and d > five_years
    )
    over_mv = tmp.loc[tmp["over_5y"], "mv"].sum()
    share = pct(over_mv, denom_total)
    return share, tmp[["issuer_bucket", "maturity_date", "mv", "over_5y"]]


def maturity_bucket_for_date(
    d: Optional[datetime], ref_date: datetime
) -> Optional[str]:
    if d is None:
        return None
    one = ref_date + timedelta(days=int(365.25))
    three = ref_date + timedelta(days=int(365.25 * 3))
    five = ref_date + timedelta(days=int(365.25 * 5))
    if d <= one:
        return "<1 ano"
    if d <= three:
        return "Entre 1 e 3 anos"
    if d <= five:
        return "Entre 3 e 5 anos"
    return "+5 anos"


def compute_maturity_buckets_exposure(
    credit_df: pd.DataFrame,
    ref_date: datetime,
    base_total: float,
    pl_total: float,
) -> pd.DataFrame:
    """
    Buckets de vencimento para crédito: <1, <3, <5, +5.
    """
    if credit_df.empty:
        data = {
            "bucket": ["<1 ano", "Entre 1 e 3 anos", "Entre 3 e 5 anos", "+5 anos"],
            "mv_credit": [0.0, 0.0, 0.0, 0.0],
        }
        out = pd.DataFrame(data)
        out["share_base"] = [0.0] * 4
        out["share_pl"] = [0.0] * 4
        return out

    tmp = credit_df.copy()
    tmp["maturity_bucket"] = tmp["maturity_date"].map(
        lambda d: maturity_bucket_for_date(d, ref_date)
    )
    tmp = tmp.loc[tmp["maturity_bucket"].notna()]

    by = tmp.groupby("maturity_bucket")["mv"].sum().reset_index()
    order = ["<1 ano", "Entre 1 e 3 anos", "Entre 3 e 5 anos", "+5 anos"]
    all_rows = []
    for b in order:
        mv = float(by.loc[by["maturity_bucket"] == b, "mv"].sum())
        all_rows.append({"bucket": b, "mv_credit": mv})
    out = pd.DataFrame(all_rows)
    out["share_base"] = out["mv_credit"].apply(lambda v: pct(v, base_total))
    out["share_pl"] = out["mv_credit"].apply(lambda v: pct(v, pl_total))
    return out


def status_tag(ok: bool) -> str:
    return "em conformidade" if ok else "fora do limite"


def exposure_table_by_bucket(
    df_pl: pd.DataFrame,
    df_base: pd.DataFrame,
    pl_total: float,
    base_total: float,
    bucket_col: str,
) -> pd.DataFrame:
    """
    Exposição por bucket para PL e PL em Renda Fixa.
    """
    base = df_base.groupby(bucket_col, dropna=False)["mv"].sum().rename("mv_base")
    pl = df_pl.groupby(bucket_col, dropna=False)["mv"].sum().rename("mv_pl")
    exp = pd.concat([base, pl], axis=1).fillna(0.0).reset_index()
    exp.columns = [bucket_col, "mv_base", "mv_pl"]
    exp["share_base"] = exp["mv_base"].apply(lambda v: pct(float(v), float(base_total)))
    exp["share_pl"] = exp["mv_pl"].apply(lambda v: pct(float(v), float(pl_total)))
    exp = exp.sort_values(["share_base", "share_pl"], ascending=False)
    return exp


def top_list_from_exposure(
    exp_df: pd.DataFrame, bucket_col: str, top_n: int = 5
) -> List[Dict[str, str]]:
    """
    Lista top_n buckets + 'Outros' agregado (para relatório).
    """
    rows = []
    df_sorted = exp_df.copy()
    df_sorted = df_sorted[df_sorted["share_base"] + df_sorted["share_pl"] > 0]
    top = df_sorted.head(top_n)
    rest = df_sorted.iloc[top_n:]
    for _, r in top.iterrows():
        rows.append(
            {
                "name": str(r[bucket_col]),
                "base_pct": as_pct_str(float(r["share_base"])),
                "pl_pct": as_pct_str(float(r["share_pl"])),
            }
        )
    if len(rest) > 0:
        base_rest = float(rest["share_base"].sum())
        pl_rest = float(rest["share_pl"].sum())
        if base_rest > 0 or pl_rest > 0:
            rows.append(
                {
                    "name": "Outros",
                    "base_pct": as_pct_str(base_rest),
                    "pl_pct": as_pct_str(pl_rest),
                }
            )
    return rows


# ---------------------------- Template (HTML Report) ---------------------------- #

REPORT_HTML = """
<!DOCTYPE html>
<html lang="pt-br">
<head>
  <meta charset="utf-8" />
  <style>
    body {
      font-family: Arial, sans-serif;
      font-size: 12px;
      color: #222;
      background: #fff;
    }
    h1 { font-size: 18px; margin-bottom: 4px; }
    h2 { font-size: 16px; margin-top: 16px; }
    h3 { font-size: 14px; margin-top: 12px; }
    h4 { font-size: 13px; margin-top: 10px; }
    .small { font-size: 11px; color: #555; }
    .ok { color: #0a7a0a; font-weight: bold; }
    .bad { color: #b00020; font-weight: bold; }
    .section { margin: 8px 0 14px 0; }
    ul { margin: 4px 0 6px 16px; }
    table { border-collapse: collapse; margin: 0 auto; }
    th, td {
      border-bottom: 1px solid #ddd;
      padding: 4px 6px;
      text-align: left;
      font-size: 12px;
    }
    .muted { color: #666; }
    .imgcell {
      width: 33%;
      vertical-align: top;
      padding: 4px;
      text-align: center;
      display: inline-block;
    }
    .imgcell-wide { width: 66%; }
    .imgcell img {
      max-width: 80%;
      height: auto;
      display: block;
      margin: 0 auto;
    }
    .nodata { color: #666; font-style: italic; padding: 8px 0 0 0; }

    /* Tabelas compactas, centralizadas e não em largura total */
    table.tbl-small {
      table-layout: fixed;
      width: 80%;
      max-width: 900px;
      font-size: 10px;
      page-break-inside: avoid;
      margin: 0 auto;
    }
    table.tbl-small th, table.tbl-small td {
      padding: 3px 4px;
      border-bottom: 1px solid #ddd;
    }
    .col-name {
      width: 70%;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .col-pct { width: 30%; text-align: right; }
  </style>
</head>
<body>
  <h1>RELATÓRIO DE MONITORAMENTO DE RISCO DE CRÉDITO</h1>
  <div class="small">
    {{ manager_name }} | CNPJ: {{ manager_cnpj }}<br/>
    Data de Emissão: {{ emission_date }}<br/>
    Período de Referência: {{ period_label }}<br/>
    Responsável: {{ responsible_name }} – Diretor de Risco
  </div>

  <div class="section">
    <h2>1. Objetivo</h2>
    Este relatório apresenta a avaliação e o monitoramento do risco de crédito
    das carteiras administradas pela {{ manager_name }}, em conformidade com as
    diretrizes da CVM, Bacen e melhores práticas.
  </div>

  <div class="section">
    <h2>2. Metodologia</h2>
    <ul>
      <li>Comitê de Investimentos: limites e estratégias por emissor e ativo.</li>
      <li>Rating: uso de ratings de agências quando disponível.</li>
      <li>Limites de Exposição: por emissor e por top 10 emissores.</li>
      <li>Monitoramento Contínuo: posições e eventos relevantes.</li>
      <li>Stress Testing: cenários adversos (liquidez e recessão).</li>
    </ul>
  </div>

  <div class="section">
    <h2>3. Exposição ao Risco de Crédito</h2>
    Total da carteira (PL): R$ {{ pl_total_fmt }}<br/>
    PL em Renda Fixa (Crédito + Tesouro): R$ {{ base_total_fmt }}<br/>
    Total de crédito: R$ {{ pl_credit_fmt }}
    ({{ credit_share_base }} do PL em Renda Fixa; {{ credit_share_pl }} do PL)

    {% if show_ratings_section %}
    <h3>3.1 Exposição por Classe e Rating</h3>
    <ul>
      <li>
        Limite por Rating: mínimo {{ min_aaa_pct }} do PL em Renda Fixa em Soberano/AAA.
      </li>
      <li>
        Exposição Atual Soberano/AAA:
        {{ aaa_share_base_pct }} do PL em Renda Fixa ({{ aaa_share_pl_pct }} do PL)
        <span class="{{ 'ok' if ok_aaa else 'bad' }}">
          ({{ conf_aaa }})
        </span>
      </li>
    </ul>
    <div class="small muted">
      Nota: Ratings vêm da coluna 'rating' no banco de dados. Soberano = Tesouro Nacional.
      Valores "NONE" são tratados como "Sem Rating".
    </div>

    <h4>3.1.A Análise de Crédito Privado – BRL (CRI + CRA + DEBENTURE)</h4>
    {% if brl_private_ratings_rows %}
      <div class="small">
        Total CRI+CRA+DEBENTURE (BRL): R$ {{ brl_private_total_fmt }}
      </div>
      <div class="imgcell imgcell-wide">{{ bar_ratings_brl | safe }}</div>
    {% else %}
      <div class="nodata">Sem dados para CRI+CRA+DEBENTURE em BRL.</div>
    {% endif %}

    <h4>3.1.B Análise de Crédito Privado – USD (Corporate)</h4>
    {% if usd_corporate_ratings_rows %}
      <div class="small">
        Total Corporate (USD): $ {{ usd_corporate_total_fmt }}
      </div>
      <div class="imgcell imgcell-wide">{{ bar_ratings_usd | safe }}</div>
    {% else %}
      <div class="nodata">Sem dados para Corporate em USD.</div>
    {% endif %}

    {% else %}
    <h3>3.1 Exposição por Classe</h3>
    <div class="small muted">Seção de Rating desativada.</div>
    {% endif %}

    <h4>3.1.C Análise de Crédito Bancário</h4>
    <div class="small">
      Total Crédito Bancário: R$ {{ bank_credit_total_fmt }}
    </div>

    <div class="imgcell imgcell-wide">{{ bar_sov_s1 | safe }}</div>
    
  </div>

  <div class="section">
    <h3>3.2 Limites de Exposição</h3>
    <ul>
      <li>
        Máximo {{ max_single_issuer_pct }} do PL em Renda Fixa em emissor único
        (Tesouro, Instituições S1 e S2 excluídos). Atual: {{ largest_issuer_share_base }}
        do PL em Renda Fixa ({{ largest_issuer_share_pl }} do PL)
        <span class="{{ 'ok' if ok_single else 'bad' }}">
          ({{ conf_single }})
        </span>
      </li>
      <li>
        Máximo {{ max_top10_issuers_pct }} do PL em Renda Fixa nos 10 maiores emissores
        (Tesouro, Instituições S1 e S2 excluídos). Atual: {{ top10_issuers_share_base }}
        do PL em Renda Fixa ({{ top10_issuers_share_pl }} do PL)
        <span class="{{ 'ok' if ok_top10 else 'bad' }}">
          ({{ conf_top10 }})
        </span>
      </li>
    </ul>

    <h4>3.2.A BRL</h4>
    {% if has_brl %}
      <div class="small">
        Emissor único (PL em Renda Fixa BRL): {{ max_single_issuer_pct }} — Atual:
        {{ largest_issuer_share_base_brl }}
        <span class="{{ 'ok' if ok_single_brl else 'bad' }}">
          ({{ conf_single_brl }})
        </span>
        <br/>
        Top-10 (PL em Renda Fixa BRL): {{ max_top10_issuers_pct }} — Atual:
        {{ top10_issuers_share_base_brl }}
        <span class="{{ 'ok' if ok_top10_brl else 'bad' }}">
          ({{ conf_top10_brl }})
        </span>
      </div>
      {% if issuer_brl_rows %}
        <table class="tbl-small">
          <thead>
            <tr>
              <th class="col-name">Emissor</th>
              <th class="col-pct">% do PL em Renda Fixa BRL</th>
            </tr>
          </thead>
          <tbody>
          {% for r in issuer_brl_rows %}
            <tr>
              <td class="col-name">{{ r.name }}</td>
              <td class="col-pct">{{ r.base_pct }}</td>
            </tr>
          {% endfor %}
          </tbody>
        </table>
      {% else %}
        <div class="nodata">Sem dados no PL em Renda Fixa BRL.</div>
      {% endif %}
    {% else %}
      <div class="nodata">Sem dados no PL em Renda Fixa BRL.</div>
    {% endif %}

    <h4>3.2.B USD</h4>
    {% if has_usd %}
      <div class="small">
        Emissor único (PL em Renda Fixa USD): {{ max_single_issuer_pct }} — Atual:
        {{ largest_issuer_share_base_usd }}
        <span class="{{ 'ok' if ok_single_usd else 'bad' }}">
          ({{ conf_single_usd }})
        </span>
        <br/>
        Top-10 (PL em Renda Fixa USD): {{ max_top10_issuers_pct }} — Atual:
        {{ top10_issuers_share_base_usd }}
        <span class="{{ 'ok' if ok_top10_usd else 'bad' }}">
          ({{ conf_top10_usd }})
        </span>
      </div>
      {% if issuer_usd_rows %}
        <table class="tbl-small">
          <thead>
            <tr>
              <th class="col-name">Emissor</th>
              <th class="col-pct">% do PL em Renda Fixa USD</th>
            </tr>
          </thead>
          <tbody>
          {% for r in issuer_usd_rows %}
            <tr>
              <td class="col-name">{{ r.name }}</td>
              <td class="col-pct">{{ r.base_pct }}</td>
            </tr>
          {% endfor %}
          </tbody>
        </table>
      {% else %}
        <div class="nodata">Sem dados no PL em Renda Fixa USD.</div>
      {% endif %}
    {% else %}
      <div class="nodata">Sem dados no PL em Renda Fixa USD.</div>
    {% endif %}
  </div>

  <div class="section">
    <h3>3.3 Concentração por Fator de Risco</h3>
    <div class="imgcell">{{ donut_factor_brl | safe }}</div>
    <div class="imgcell">{{ donut_factor_usd | safe }}</div>
  </div>

  <div class="section">
    <h3>3.4 Concentração por Vencimento</h3>
    <div class="imgcell">{{ donut_maturity_base | safe }}</div>
    <div style="margin-top:6px;">
      Limite: máximo {{ max_maturity_over_5y_pct }} do PL em Renda Fixa em vencimentos
      &gt; 5 anos. Atual: {{ over5y_pct_base }} do PL em Renda Fixa
      ({{ over5y_pct_pl }} do PL)
      <span class="{{ 'ok' if ok_over5y else 'bad' }}">
        ({{ conf_over5y }})
      </span>
    </div>
  </div>

  <div class="section">
    <h3>3.5 Exposição por País de Risco</h3>
    {% if country_rows %}
      <table class="tbl-small">
        <thead>
          <tr>
            <th class="col-name">País</th>
            <th class="col-pct">% do PL em Renda Fixa</th>
          </tr>
        </thead>
        <tbody>
        {% for r in country_rows %}
          <tr>
            <td class="col-name">{{ r.name }}</td>
            <td class="col-pct">{{ r.base_pct }}</td>
          </tr>
        {% endfor %}
        </tbody>
      </table>
    {% else %}
      <div class="nodata">Sem dados para exposição por país.</div>
    {% endif %}
  </div>

  <div class="section">
    <h3>3.6 Exposição Cambial</h3>
    <div class="small muted">
      PL em Renda Fixa = Crédito + Tesouro. Fundos não entram no PL em Renda Fixa.
    </div>
    <div class="imgcell">{{ donut_currency_base | safe }}</div>
  </div>

  <div class="section">
    <h3>3.7 Exposição por Tipo de Título</h3>
    <div class="imgcell">{{ donut_bondtype_brl | safe }}</div>
    <div class="imgcell">{{ donut_bondtype_usd | safe }}</div>
  </div>

  <div class="section">
    <h3>3.8 Inadimplência</h3>
    Índice de Inadimplência no período: {{ default_rate }}
  </div>

  <div class="section">
    <h2>4. Monitoramento de Eventos Relevantes</h2>
    <div>{{ events_html }}</div>
  </div>

  <div class="section">
    <h2>5. Conformidade Regulatória</h2>
    <ul>
      <li>CVM: Instrução CVM nº 555/2014; Resolução CVM nº 50/21.</li>
      <li>Bacen: Circular nº 3.930/2019.</li>
      <li>ANBIMA: Código de Regulação e Melhores Práticas.</li>
    </ul>
  </div>

  <div class="section">
    <h2>6. Conclusões</h2>
    <ul>
      <li>Os níveis de risco de crédito são compatíveis com o perfil estabelecido.</li>
      <li>Não foram identificadas exposições que comprometam a estabilidade
          financeira das carteiras.</li>
      <li>As políticas de risco são seguidas com monitoramento contínuo.</li>
    </ul>
  </div>

  <div class="section">
    <h2>7. Recomendações</h2>
    <ul>
      <li>Revisões trimestrais das métricas e limites.</li>
      <li>Treinamentos periódicos da equipe de risco.</li>
      <li>Atualização anual dos cenários de stress.</li>
    </ul>
  </div>

  <div class="section">
    Assinatura: {{ responsible_name }} – Diretor de Risco
  </div>

  <div class="section">
    <h3>Anexos</h3>
    <div class="small">
      Figuras geradas no app (por classe, por rating, por emissor, por fator,
      por vencimento, por país, por moeda e por tipo) podem ser exportadas
      separadamente.
    </div>
  </div>
</body>
</html>
"""

# ---------------------------- Streamlit UI ---------------------------- #

st.title("Relatório de Monitoramento de Risco de Crédito")

# Fonte de dados
st.sidebar.header("Fonte de dados")
data_source = st.sidebar.radio(
    "Origem dos dados",
    options=["SQLite (.db)", "CSV"],
    index=0,
)

csv_file = None
db_path = None
db_table_selected = None
ref_date_from_db_display = None  # dd/mm/aaaa
db_dates_display_sorted: List[str] = []

if data_source == "CSV":
    st.sidebar.subheader("Arquivos")
    csv_file = st.sidebar.file_uploader(
        "CSV da carteira (formato com '|')", type=["csv"]
    )
else:
    st.sidebar.subheader("Banco de Dados Local (SQLite)")
    db_path = st.sidebar.text_input(
        "Caminho do banco (relativo ao projeto)", value="databases/gorila_positions.db"
    )
    st.sidebar.markdown("Tabela fixa: pmv_plus_gorila")
    if not db_path or not os.path.exists(db_path):
        st.sidebar.warning(
            "Arquivo de banco não encontrado. Verifique o caminho: "
            f"{os.path.abspath(db_path or '')}"
        )
    else:
        try:
            with sqlite3.connect(db_path) as conn:
                candidates = detect_tables_with_reference_date(conn)
                if "pmv_plus_gorila" not in candidates:
                    st.sidebar.error(
                        "Tabela 'pmv_plus_gorila' não encontrada no banco "
                        "ou sem coluna 'reference_date'. "
                        f"Tabelas disponíveis: {', '.join(candidates) or '-'}"
                    )
                else:
                    db_table_selected = "pmv_plus_gorila"
                    raw_dates = list_unique_reference_dates(conn, db_table_selected)
                    pairs = []
                    for s in raw_dates:
                        d = to_date_safe(s)
                        if d is not None:
                            disp = d.strftime("%d/%m/%Y")
                            pairs.append((d, disp))
                        else:
                            pairs.append((None, s))
                    seen_disp = set()
                    display_vals = []
                    parsed_map = {}
                    for d, disp in pairs:
                        if disp not in seen_disp:
                            seen_disp.add(disp)
                            display_vals.append(disp)
                            parsed_map[disp] = d
                    display_vals_sorted = sorted(
                        display_vals,
                        key=lambda x: (parsed_map[x] is None, parsed_map[x]),
                    )
                    db_dates_display_sorted = display_vals_sorted
                    if len(display_vals_sorted) > 0:
                        default_idx = len(display_vals_sorted) - 1
                        ref_date_from_db_display = st.sidebar.selectbox(
                            "Data de referência (do DB) - únicas:",
                            options=display_vals_sorted,
                            index=default_idx,
                        )
                    else:
                        st.sidebar.info(
                            "Nenhuma 'reference_date' encontrada "
                            "na tabela pmv_plus_gorila."
                        )
        except Exception as e:
            st.sidebar.error(f"Erro ao abrir o banco: {e}")

st.sidebar.header("Filtros")
portfolio_id = st.sidebar.text_input("portfolio_id (opcional)", value="")

# Filtro de data de referência
if data_source == "CSV":
    ref_date_filter = st.sidebar.text_input(
        "Data de referência (opcional, dd/mm/aaaa)", value=""
    )
else:
    ref_date_filter = ref_date_from_db_display or ""

st.sidebar.header("Seções do Relatório")
show_ratings_section = st.sidebar.checkbox(
    "Exibir informações de rating (Soberano/AAA)", value=False
)

st.sidebar.header("Limites (política) – denominador: PL em Renda Fixa)")
max_single_issuer = st.sidebar.number_input(
    "Máx emissor único (PL em Renda Fixa)",
    min_value=0.0,
    max_value=1.0,
    value=DEFAULT_LIMITS["max_single_issuer"],
    step=0.01,
)
max_top10_issuers = st.sidebar.number_input(
    "Máx 10 maiores emissores (PL em Renda Fixa)",
    min_value=0.0,
    max_value=1.0,
    value=DEFAULT_LIMITS["max_top10_issuers"],
    step=0.01,
)
max_prefixed = st.sidebar.number_input(
    "Máx PRÉ (PL em Renda Fixa)",
    min_value=0.0,
    max_value=1.0,
    value=DEFAULT_LIMITS["max_prefixed"],
    step=0.01,
)
max_over5y = st.sidebar.number_input(
    "Máx vencimentos > 5 anos (PL em Renda Fixa)",
    min_value=0.0,
    max_value=1.0,
    value=DEFAULT_LIMITS["max_maturity_over_5y"],
    step=0.01,
)

st.sidebar.header("Cabeçalho do Relatório")
manager_name = st.sidebar.text_input(
    "Gestora", value="Ceres Asset Gestão de Investimentos Ltda"
)
manager_cnpj = st.sidebar.text_input("CNPJ", value="40.962.925/0001-38")
responsible_name = st.sidebar.text_input("Responsável", value="Brenno Melo")

# ---------------------------- Controles de Gráficos -------------------- #

st.sidebar.header("Gráficos do Relatório (Donuts)")
donut_size_in = st.sidebar.slider(
    "Tamanho do donut (pol.)", min_value=2.0, max_value=6.0, value=3.2, step=0.1
)
donut_dpi = st.sidebar.slider(
    "DPI (qualidade)", min_value=120, max_value=400, value=240, step=10
)
donut_ring_width = st.sidebar.slider(
    "Espessura do anel (0.2 fino – 0.9 grosso)",
    min_value=0.2,
    max_value=0.9,
    value=0.75,
    step=0.05,
)
donut_font_size = st.sidebar.slider(
    "Tamanho da fonte interna", min_value=4, max_value=14, value=6, step=1
)
donut_top_n = st.sidebar.slider(
    "Top N categorias (demais = 'Outros')", min_value=3, max_value=10, value=6, step=1
)
donut_min_pct_label = st.sidebar.slider(
    "Ocultar rótulos abaixo de (%)", min_value=0.0, max_value=10.0, value=1.0, step=0.5
)
donut_decimals = st.sidebar.slider(
    "Decimais nos %", min_value=0, max_value=2, value=1, step=1
)

DONUT_CFG = {
    "size_in": donut_size_in,
    "dpi": donut_dpi,
    "ring_width": donut_ring_width,
    "font_size": donut_font_size,
    "min_label_pct": donut_min_pct_label / 100.0,  # [0-1]
    "decimals": donut_decimals,
}

st.sidebar.header("Gráficos do Dashboard")
dash_height = st.sidebar.slider(
    "Altura (px)", min_value=240, max_value=900, value=420, step=10
)

# ---------------------------- Data Load ---------------------------- #

if data_source == "CSV":
    if not csv_file:
        st.info("Carregue o CSV para iniciar.")
        st.stop()
else:
    if not db_path or not os.path.exists(db_path) or not db_table_selected:
        st.info(
            "Informe o caminho do .db existente. O app usará a tabela "
            "'pmv_plus_gorila'."
        )
        st.stop()

# Carregar dados
if data_source == "CSV":
    df_raw = read_portfolio_csv(csv_file)
else:
    df_raw = read_positions_from_db_path(db_path, db_table_selected)

df = normalize_dataframe(df_raw)

if portfolio_id and "portfolio_id" in df.columns:
    df = df.loc[df["portfolio_id"] == portfolio_id].copy()
    if df.empty:
        st.warning("Nenhuma linha encontrada para o portfolio_id informado.")
        st.stop()

if ref_date_filter:
    try:
        dt_filter = datetime.strptime(ref_date_filter, "%d/%m/%Y")
        df = df.loc[
            df["reference_dt"].notna() & (df["reference_dt"] == dt_filter)
        ].copy()
    except Exception:
        st.warning("Data inválida. Use dd/mm/aaaa.")
        st.stop()

if df.empty:
    st.warning("Após filtros, nenhum dado disponível.")
    st.stop()

# --- NOVO: Editor de Emissões (DESCONHECIDO, NONE ou None) ---
mask_unknown = (
    df["issuer_bucket"].isna()
    | (df["issuer_bucket"].str.strip().str.upper().isin(["", "DESCONHECIDO", "NONE"]))
) & df["is_credit"]

unknown_issuers = df[mask_unknown].copy()
if not unknown_issuers.empty:
    st.warning(
        f"⚠️ Encontradas {len(unknown_issuers)} linhas com emissor não identificado. "
        "Por favor, preencha o campo 'parsed_company_name' abaixo para corrigi-los."
    )

    # Definir colunas para edição
    cols_to_edit = ["security_name", "issuer_name_norm", "parsed_company_name"]
    # Garantir que as colunas existem
    for c in cols_to_edit:
        if c not in unknown_issuers.columns:
            unknown_issuers[c] = ""

    edited_df = st.data_editor(
        unknown_issuers[cols_to_edit],
        key="issuer_editor",
        use_container_width=True,
        num_rows="fixed",
        disabled=["security_name", "issuer_name_norm"],
    )

    # Se houve alteração, atualizar df original
    if not edited_df.equals(unknown_issuers[cols_to_edit]):
        for idx, row in edited_df.iterrows():
            new_val = row["parsed_company_name"]
            if pd.notna(new_val) and str(new_val).strip():
                df.at[idx, "parsed_company_name"] = str(new_val).strip()

        # Re-normalizar / Re-bucketizar (apenas campos que dependem de parsed_company_name)
        def _reupdate_issuer_bucket(r):
            if bool(r.get("is_treasury", False)):
                return "TESOURO NACIONAL"
            pc = r.get("parsed_company_name")
            if isinstance(pc, str) and pc.strip() and pc.strip() != "########":
                return clean_issuer_name(pc) or "DESCONHECIDO"
            return r.get("issuer_name_norm", "DESCONHECIDO") or "DESCONHECIDO"

        df["issuer_bucket"] = df.apply(_reupdate_issuer_bucket, axis=1)
        # Também atualizar is_s1 que depende de parsed_company_name
        df["is_s1"] = df["parsed_company_name"].apply(is_s1_from_parsed_company)

events_text = st.text_area(
    "Eventos Relevantes (HTML ou texto simples):",
    value=("<ul><li>Destaques de resultados trimestrais (preencher).</li></ul>"),
    height=150,
)

# Ratings agora vêm da coluna 'rating' no banco de dados

# Métricas principais
pl_total = compute_nav(df)

credit_df = df.loc[df["is_credit"]].copy()
treasury_df = df.loc[df["is_treasury"]].copy()
funds_df = df.loc[df["is_fund"]].copy()
base_df = df.loc[df["is_credit"] | df["is_treasury"]].copy()

pl_credit = float(credit_df["mv"].sum())
pl_treasury = float(treasury_df["mv"].sum())
pl_funds = float(funds_df["mv"].sum())
base_total = float(base_df["mv"].sum())

st.write("Destaques do Portfólio")

# Sub-bases por moeda
base_df_brl = base_df.loc[base_df["currency_code"] == "BRL"].copy()
base_df_usd = base_df.loc[base_df["currency_code"] == "USD"].copy()
base_total_brl = float(base_df_brl["mv"].sum())
base_total_usd = float(base_df_usd["mv"].sum())

pl_total_brl = compute_nav(df.loc[df["currency_code"] == "BRL"])
pl_total_usd = compute_nav(df.loc[df["currency_code"] == "USD"])

# Período de referência
ref_dates = df["reference_dt"].dropna().sort_values().unique()
if len(ref_dates) > 0:
    ref_dt = pd.to_datetime(ref_dates[-1]).to_pydatetime()
else:
    ref_dt = datetime.today()
period_label = ref_dt.strftime("%B de %Y").title()

# Rating share (relativo ao PL em Renda Fixa e PL)
if show_ratings_section:
    aaa_share_base, _ = compute_sovereign_or_aaa_share(base_df, base_total, None)
    aaa_share_pl, _ = compute_sovereign_or_aaa_share(df, pl_total, None)
    ok_aaa = (
        aaa_share_base >= DEFAULT_LIMITS["min_sovereign_or_aaa"]
        if base_total > 0
        else True
    )
else:
    aaa_share_base, aaa_share_pl, ok_aaa = 0.0, 0.0, True

# Excluir Tesouro + S1 + S2 da análise de concentração
s1_s2_bucket_names = set(
    base_df.loc[base_df["is_s1"] | base_df["is_s2"], "issuer_bucket"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)
EXCLUDE_IN_LIMITS = set(IGNORE_IN_LIMITS_AS_TESOURO) | s1_s2_bucket_names

# Concentração por emissor (overall, BRL, USD)
conc_base = compute_concentration_by_bucket(
    base_df, base_total, bucket_col="issuer_bucket", exclude_set=EXCLUDE_IN_LIMITS
)
largest_share_pl = pct(conc_base["largest_mv"], pl_total)
top10_share_pl = pct(conc_base["top10_mv"], pl_total)

if base_total_brl > 0:
    conc_brl = compute_concentration_by_bucket(
        base_df_brl,
        base_total_brl,
        bucket_col="issuer_bucket",
        exclude_set=EXCLUDE_IN_LIMITS,
    )
else:
    conc_brl = {
        "largest_share": 0.0,
        "top10_share": 0.0,
        "table": pd.Series(dtype=float),
    }

if base_total_usd > 0:
    conc_usd = compute_concentration_by_bucket(
        base_df_usd,
        base_total_usd,
        bucket_col="issuer_bucket",
        exclude_set=EXCLUDE_IN_LIMITS,
    )
else:
    conc_usd = {
        "largest_share": 0.0,
        "top10_share": 0.0,
        "table": pd.Series(dtype=float),
    }

# Fator de risco
factor_shares_base = compute_factor_risk_shares(base_df, base_total)
factor_shares_pl = compute_factor_risk_shares(df, pl_total)  # info extra
factor_shares_brl = compute_factor_risk_shares(base_df_brl, base_total_brl)
factor_shares_usd = compute_factor_risk_shares(base_df_usd, base_total_usd)

# Vencimentos
over5y_share_base, over5y_table = compute_maturity_share_over_5y(
    credit_df, ref_dt, base_total
)
over5y_share_pl, _ = compute_maturity_share_over_5y(credit_df, ref_dt, pl_total)
mat_buckets_df = compute_maturity_buckets_exposure(
    credit_df, ref_dt, base_total, pl_total
)

# ---------------------------- Análise de Rating por Grupo ---------------------------- #

# BRL: CRI + CRA + DEBENTURE
brl_private_bond_types = [
    "CORPORATE_BONDS_CRI",
    "CORPORATE_BONDS_CRA",
    "CORPORATE_BONDS_DEBENTURE",
]
brl_private_df = filter_bonds_by_type(
    credit_df, brl_private_bond_types, currency="BRL"
)
brl_private_total = float(brl_private_df["mv"].sum())
brl_private_ratings_rows = compute_rating_distribution_by_group(
    brl_private_df, brl_private_total
)

# USD: Corporate
corp_type_mask = (
    credit_df["security_type"]
    .astype(str)
    .str.upper()
    .str.contains("CORPORATE", na=False)
)
if "parsed_bond_type" in credit_df.columns:
    corp_pbt_mask = (
        credit_df["parsed_bond_type"].astype(str).str.upper() == "CORPORATE"
    )
else:
    corp_pbt_mask = pd.Series(False, index=credit_df.index)

usd_corporate_df = credit_df.loc[
    (credit_df["currency_code"] == "USD") & (corp_type_mask | corp_pbt_mask)
].copy()
usd_corporate_total = float(usd_corporate_df["mv"].sum())
usd_corporate_ratings_rows = compute_rating_distribution_by_group(
    usd_corporate_df, usd_corporate_total
)

# Checks de conformidade
max_single_issuer = float(max_single_issuer)
max_top10_issuers = float(max_top10_issuers)
max_prefixed = float(max_prefixed)
max_over5y = float(max_over5y)

ok_single = conc_base["largest_share"] <= max_single_issuer
ok_top10 = conc_base["top10_share"] <= max_top10_issuers
ok_prefixed = factor_shares_base.get("PRÉ", 0.0) <= max_prefixed
ok_over5y = over5y_share_base <= max_over5y

# ---------------------------- Exposições (Novas) ---------------------------- #

# País (tabela)
country_exp_table = exposure_table_by_bucket(
    df_pl=df,
    df_base=base_df,
    pl_total=pl_total,
    base_total=base_total,
    bucket_col="country_bucket",
)
country_top_list = top_list_from_exposure(
    country_exp_table, bucket_col="country_bucket", top_n=5
)

# Moeda
currency_exp_table = exposure_table_by_bucket(
    df_pl=df,
    df_base=base_df,
    pl_total=pl_total,
    base_total=base_total,
    bucket_col="currency_code",
)
currency_top_list = top_list_from_exposure(
    currency_exp_table, bucket_col="currency_code", top_n=5
)

# Tipo de Título (parsed_bond_type)
if "parsed_bond_type" in df.columns:
    bond_type_exp_table = exposure_table_by_bucket(
        df_pl=df,
        df_base=base_df,
        pl_total=pl_total,
        base_total=base_total,
        bucket_col="parsed_bond_type",
    )
    bond_type_exp_table_brl = exposure_table_by_bucket(
        df_pl=df.loc[df["currency_code"] == "BRL"],
        df_base=base_df_brl,
        pl_total=pl_total_brl,
        base_total=base_total_brl,
        bucket_col="parsed_bond_type",
    )
    bond_type_exp_table_usd = exposure_table_by_bucket(
        df_pl=df.loc[df["currency_code"] == "USD"],
        df_base=base_df_usd,
        pl_total=pl_total_usd,
        base_total=base_total_usd,
        bucket_col="parsed_bond_type",
    )
    bond_type_top_list = top_list_from_exposure(
        bond_type_exp_table, bucket_col="parsed_bond_type", top_n=5
    )
else:
    bond_type_exp_table = pd.DataFrame()
    bond_type_exp_table_brl = pd.DataFrame()
    bond_type_exp_table_usd = pd.DataFrame()
    bond_type_top_list = []

# ---------------------------- Gráficos (Dashboard) -------------------------- #


def center_chart(fig):
    c_left, c_mid, c_right = st.columns([1, 8, 1])
    with c_mid:
        st.plotly_chart(fig, use_container_width=True)


st.subheader("Resumo")
c1, c2, c3, c4 = st.columns(4)
c1.metric(
    "PL Total (R$)",
    f"{pl_total:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."),
)
c2.metric(
    "PL em Renda Fixa (Crédito + Tesouro) (R$)",
    f"{base_total:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."),
)
c3.metric(
    "PL Crédito (R$)",
    f"{pl_credit:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."),
)
c4.metric(
    "% Crédito/PL | % Crédito/PL em Renda Fixa",
    f"{as_pct_str(pct(pl_credit, pl_total))} | "
    f"{as_pct_str(pct(pl_credit, base_total))}",
)

st.markdown("Exposição por Classe de Ativo")
df["class_bucket"] = np.select(
    [
        df["is_treasury"],
        df["is_fund"],
        df["is_credit"],
        df["asset_class"].astype(str).str.upper().eq("STOCKS"),
        df["asset_class"].astype(str).str.upper().eq("CASH"),
    ],
    ["Tesouro", "Fundos", "Crédito Privado", "Ações/ETFs", "Caixa"],
    default="Outros",
)
by_class = df.groupby("class_bucket")["mv"].sum().reset_index()
fig1 = px.pie(
    by_class, names="class_bucket", values="mv", title="Patrimônio por Classe", hole=0.2
)
fig1.update_layout(
    font=dict(size=9),
    legend_font_size=9,
    colorway=PALETA_CORES,
    height=dash_height,
)
fig1.update_traces(textfont_size=9)
center_chart(fig1)

# Custódia em Fundos (apenas)
st.markdown("Custódia em Fundos (apenas)")
cfa1, cfa2 = st.columns(2)
cfa1.metric(
    "PL Fundos (R$)",
    f"{pl_funds:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."),
)
cfa2.metric("% Fundos/PL", as_pct_str(pct(pl_funds, pl_total)))

if not funds_df.empty:
    by_fund = (
        funds_df.groupby("fund_bucket", dropna=False)["mv"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )
    by_fund.columns = ["Fundo", "Valor (R$)"]
    by_fund["% do PL"] = by_fund["Valor (R$)"].apply(lambda x: pct(x, pl_total))

    top_n = 10
    top_funds = by_fund.head(top_n).copy()
    fig_funds = px.bar(
        top_funds,
        x="Fundo",
        y=top_funds["Valor (R$)"] / 1e6,
        title="Top Fundos por Valor (R$ milhões)",
        text=(top_funds["Valor (R$)"] / 1e6).map(lambda v: f"{v:.2f}"),
        labels={"y": "R$ mi"},
    )
    fig_funds.update_layout(
        font=dict(size=9),
        showlegend=False,
        colorway=PALETA_CORES,
        height=dash_height,
    )
    fig_funds.update_traces(textfont_size=9)
    center_chart(fig_funds)
else:
    st.info("Sem posições classificadas como Fundos.")

# Fator de Risco – BRL
st.markdown("Concentração por Fator de Risco (% do PL em Renda Fixa) – BRL")
factor_plot_brl = pd.DataFrame(
    {
        "Fator": ["PRÉ", "CDI", "IPCA", "OUTROS"],
        "Percentual": [
            factor_shares_brl.get("PRÉ", 0.0) * 100,
            factor_shares_brl.get("CDI", 0.0) * 100,
            factor_shares_brl.get("IPCA", 0.0) * 100,
            factor_shares_brl.get("OUTROS", 0.0) * 100,
        ],
    }
)
fig2_brl = px.bar(
    factor_plot_brl,
    x="Fator",
    y="Percentual",
    color="Fator",
    title="Fator de Risco – BRL (PL em Renda Fixa BRL)",
    text=factor_plot_brl["Percentual"].map(lambda v: f"{v:.2f}%"),
    color_discrete_sequence=PALETA_CORES,
)
fig2_brl.update_traces(texttemplate="%{text}", textfont_size=9)
fig2_brl.update_layout(font=dict(size=9), showlegend=False, height=dash_height)
center_chart(fig2_brl)

# Fator de Risco – USD
st.markdown("Concentração por Fator de Risco (% do PL em Renda Fixa) – USD")
factor_plot_usd = pd.DataFrame(
    {
        "Fator": ["PRÉ", "CDI", "IPCA", "OUTROS"],
        "Percentual": [
            factor_shares_usd.get("PRÉ", 0.0) * 100,
            factor_shares_usd.get("CDI", 0.0) * 100,
            factor_shares_usd.get("IPCA", 0.0) * 100,
            factor_shares_usd.get("OUTROS", 0.0) * 100,
        ],
    }
)
fig2_usd = px.bar(
    factor_plot_usd,
    x="Fator",
    y="Percentual",
    color="Fator",
    title="Fator de Risco – USD (PL em Renda Fixa USD)",
    text=factor_plot_usd["Percentual"].map(lambda v: f"{v:.2f}%"),
    color_discrete_sequence=PALETA_CORES,
)
fig2_usd.update_traces(texttemplate="%{text}", textfont_size=9)
fig2_usd.update_layout(font=dict(size=9), showlegend=False, height=dash_height)
center_chart(fig2_usd)

st.markdown(
    "Concentração por Emissor (parsed_company_name) – PL em Renda Fixa; Tesouro, S1 e S2 excluídos"
)
issuer_table = conc_base["table"].reset_index()
issuer_table.columns = ["Emissor (parsed_company_name)", "Valor (R$)"]
issuer_table["% do PL em Renda Fixa"] = issuer_table["Valor (R$)"].apply(
    lambda x: pct(x, base_total)
)
issuer_table["% do PL"] = issuer_table["Valor (R$)"].apply(lambda x: pct(x, pl_total))
st.dataframe(
    issuer_table.assign(
        **{
            "Valor (R$)": issuer_table["Valor (R$)"]
            .map(
                lambda v: f"{v:,.2f}"
                .replace(",", "X")
                .replace(".", ",")
                .replace("X", ".")
            )
        }
    ).assign(
        **{
            "% do PL em Renda Fixa": issuer_table["% do PL em Renda Fixa"].map(
                as_pct_str
            ),
            "% do PL": issuer_table["% do PL"].map(as_pct_str),
        }
    )
)

# País (tabela - dashboard)
st.markdown("Exposição por País de Risco (% do PL em Renda Fixa)")
if not country_exp_table.empty:
    top_n = 10
    top_country = country_exp_table.head(top_n).copy()
    tbl_country = pd.DataFrame(
        {
            "País": list(top_country["country_bucket"]),
            "% do PL em Renda Fixa": list(top_country["share_base"].map(as_pct_str)),
        }
    )
    st.dataframe(tbl_country)
else:
    st.info("Sem dados para exposição por país.")

# Moeda - dashboard
st.markdown("Exposição Cambial (% do PL em Renda Fixa)")
if not currency_exp_table.empty:
    top_n = 10
    top_curr = currency_exp_table.head(top_n).copy()
    plot_curr = pd.DataFrame(
        {
            "Categoria": list(top_curr["currency_code"]),
            "Percentual": list(top_curr["share_base"] * 100.0),
        }
    )
    fig_curr = px.bar(
        plot_curr,
        x="Categoria",
        y="Percentual",
        color="Categoria",
        title="Top Moedas (% do PL em Renda Fixa)",
        text=plot_curr["Percentual"].map(lambda v: f"{v:.2f}%"),
        color_discrete_sequence=PALETA_CORES,
    )
    fig_curr.update_layout(font=dict(size=9), showlegend=False, height=dash_height)
    center_chart(fig_curr)
else:
    st.info("Sem dados para exposição cambial.")

# Tipo de Título – BRL
st.markdown("Exposição por Tipo de Título – % do PL em Renda Fixa BRL")
if "parsed_bond_type" in df.columns:
    if not bond_type_exp_table_brl.empty and base_total_brl > 0:
        top_n = 10
        top_bt_brl = bond_type_exp_table_brl.head(top_n).copy()
        plot_bt_brl = pd.DataFrame(
            {
                "Categoria": list(top_bt_brl["parsed_bond_type"].fillna("")),
                "Percentual": list(top_bt_brl["share_base"] * 100.0),
            }
        )
        fig_bt_brl = px.bar(
            plot_bt_brl,
            x="Categoria",
            y="Percentual",
            color="Categoria",
            title="Top Tipos – BRL (% do PL em Renda Fixa BRL)",
            text=plot_bt_brl["Percentual"].map(lambda v: f"{v:.2f}%"),
            color_discrete_sequence=PALETA_CORES,
        )
        fig_bt_brl.update_layout(
            font=dict(size=9), showlegend=False, height=dash_height
        )
        center_chart(fig_bt_brl)
    else:
        st.info("Sem dados agregados de tipos em BRL.")

    # Tipo de Título – USD
    st.markdown(
        "Exposição por Tipo de Título (parsed_bond_type) – % do PL em Renda Fixa USD"
    )
    if not bond_type_exp_table_usd.empty and base_total_usd > 0:
        top_bt_usd = bond_type_exp_table_usd.head(top_n).copy()
        plot_bt_usd = pd.DataFrame(
            {
                "Categoria": list(top_bt_usd["parsed_bond_type"].fillna("")),
                "Percentual": list(top_bt_usd["share_base"] * 100.0),
            }
        )
        fig_bt_usd = px.bar(
            plot_bt_usd,
            x="Categoria",
            y="Percentual",
            color="Categoria",
            title="Top Tipos – USD (% do PL em Renda Fixa USD)",
            text=plot_bt_usd["Percentual"].map(lambda v: f"{v:.2f}%"),
            color_discrete_sequence=PALETA_CORES,
        )
        fig_bt_usd.update_layout(
            font=dict(size=9), showlegend=False, height=dash_height
        )
        center_chart(fig_bt_usd)
    else:
        st.info("Sem dados agregados de tipos em USD.")
else:
    st.info(
        "Coluna 'parsed_bond_type' não encontrada no dataset. "
        "Seção de 'Tipo de Título' e debug desabilitada."
    )

# Vencimentos (gráfico - dashboard)
st.markdown("Vencimentos (Crédito Privado) – Buckets (% do PL em Renda Fixa)")
mat_plot = pd.DataFrame(
    {
        "Faixa": list(mat_buckets_df["bucket"]),
        "Percentual": list(mat_buckets_df["share_base"] * 100.0),
    }
)
fig_m = px.bar(
    mat_plot,
    x="Faixa",
    y="Percentual",
    color="Faixa",
    barmode="group",
    title="Exposição por Vencimento (Crédito): % do PL em Renda Fixa",
    text=mat_plot["Percentual"].map(lambda v: f"{v:.2f}%"),
    color_discrete_sequence=PALETA_CORES,
)
fig_m.update_traces(texttemplate="%{text}", textfont_size=9)
fig_m.update_layout(font=dict(size=9), showlegend=False, height=dash_height)
center_chart(fig_m)

# Scatter de vencimentos (opcional)
st.markdown("Distribuição de Vencimentos (pontos)")
if not credit_df.empty:
    cred_plot = credit_df.copy()
    cred_plot["Maturity"] = cred_plot["maturity_date"]
    cred_plot["Issuer"] = cred_plot["issuer_bucket"]
    cred_plot["Valor"] = pd.to_numeric(cred_plot["mv"], errors="coerce")
    cred_plot = cred_plot.loc[cred_plot["Maturity"].notna()].copy()
    cred_plot["Valor"] = (
        cred_plot["Valor"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .clip(lower=0.0)
    )
    cred_plot_nz = cred_plot.loc[cred_plot["Valor"] > 0.0].copy()
    if not cred_plot_nz.empty:
        fig3 = px.scatter(
            cred_plot_nz,
            x="Maturity",
            y="Valor",
            color="Issuer",
            size="Valor",
            title="Distribuição de Vencimentos",
            size_max=30,
        )
        fig3.update_layout(
            font=dict(size=9), colorway=PALETA_CORES, height=dash_height
        )
        center_chart(fig3)
    else:
        st.info("Sem valores válidos para o gráfico de pontos (size).")
else:
    st.info("Sem crédito privado para plotar.")

# Análise de Rating por Grupo (Dashboard) — BAR GRAPHS ORDERED BY RATING
if show_ratings_section:
    st.markdown("---")
    st.subheader("Análise de Rating por Grupo")

    # BRL: CRI + CRA + DEBENTURE
    st.markdown("**BRL: CRI + CRA + DEBENTURE**")
    if brl_private_ratings_rows:
        st.write(f"Total CRI+CRA+DEBENTURE (BRL): R$ {brl(brl_private_total)}")

        # Tabela
        brl_rating_df = pd.DataFrame(brl_private_ratings_rows)
        brl_rating_df["Valor (R$)"] = brl_rating_df["mv"].apply(lambda x: brl(x))
        brl_rating_df = brl_rating_df[["rating", "Valor (R$)", "pct"]].rename(
            columns={"rating": "Rating", "pct": "% do Total CRI+CRA+DEBENTURE"}
        )
        # Ordenar tabela pela escala
        brl_rating_df["order"] = brl_rating_df["Rating"].apply(rating_sort_key)
        brl_rating_df = brl_rating_df.sort_values("order").drop(columns="order")
        st.dataframe(brl_rating_df, use_container_width=True)

        # Gráfico de barras horizontal (percentual), ordenado por escala
        plot_brl = pd.DataFrame(brl_private_ratings_rows)
        plot_brl["Percentual"] = (
            plot_brl["mv"] / max(brl_private_total, 1e-9) * 100.0
        )
        plot_brl["order"] = plot_brl["rating"].apply(rating_sort_key)
        plot_brl = plot_brl.sort_values("order")
        present = [r for r in RATING_ORDER if r in plot_brl["rating"].tolist()]
        fig_brl_rating = px.bar(
            plot_brl,
            x="Percentual",
            y="rating",
            color="rating",
            orientation="h",
            title="Distribuição de Rating – BRL (% do total CRI+CRA+DEBENTURE)",
            text=plot_brl["Percentual"].map(lambda v: f"{v:.2f}%"),
            color_discrete_sequence=PALETA_CORES,
        )
        fig_brl_rating.update_layout(
            font=dict(size=9),
            showlegend=False,
            height=dash_height,
            yaxis=dict(categoryorder="array", categoryarray=present),
            xaxis_title="% do Total",
            yaxis_title="Rating",
        )
        fig_brl_rating.update_traces(texttemplate="%{text}", textfont_size=9)
        center_chart(fig_brl_rating)
    else:
        st.info("Sem dados para CRI+CRA+DEBENTURE em BRL.")

    # USD: Corporate
    st.markdown("**USD: Corporate**")
    if usd_corporate_ratings_rows:
        st.write(f"Total Corporate (USD): $ {brl(usd_corporate_total)}")

        # Tabela
        usd_rating_df = pd.DataFrame(usd_corporate_ratings_rows)
        usd_rating_df["Valor (USD)"] = usd_rating_df["mv"].apply(lambda x: brl(x))
        usd_rating_df = usd_rating_df[["rating", "Valor (USD)", "pct"]].rename(
            columns={"rating": "Rating", "pct": "% do Total Corporate USD"}
        )
        usd_rating_df["order"] = usd_rating_df["Rating"].apply(rating_sort_key)
        usd_rating_df = usd_rating_df.sort_values("order").drop(columns="order")
        st.dataframe(usd_rating_df, use_container_width=True)

        # Gráfico de barras horizontal (percentual), ordenado por escala
        plot_usd = pd.DataFrame(usd_corporate_ratings_rows)
        plot_usd["Percentual"] = (
            plot_usd["mv"] / max(usd_corporate_total, 1e-9) * 100.0
        )
        plot_usd["order"] = plot_usd["rating"].apply(rating_sort_key)
        plot_usd = plot_usd.sort_values("order")
        present_u = [r for r in RATING_ORDER if r in plot_usd["rating"].tolist()]
        fig_usd_rating = px.bar(
            plot_usd,
            x="Percentual",
            y="rating",
            color="rating",
            orientation="h",
            title="Distribuição de Rating – USD (% do total Corporate USD)",
            text=plot_usd["Percentual"].map(lambda v: f"{v:.2f}%"),
            color_discrete_sequence=PALETA_CORES,
        )
        fig_usd_rating.update_layout(
            font=dict(size=9),
            showlegend=False,
            height=dash_height,
            yaxis=dict(categoryorder="array", categoryarray=present_u),
            xaxis_title="% do Total",
            yaxis_title="Rating",
        )
        fig_usd_rating.update_traces(texttemplate="%{text}", textfont_size=9)
        center_chart(fig_usd_rating)
    else:
        st.info("Sem dados para Corporate em USD.")

# ------------------- Gráficos Estáticos (Relatório) -------------------- #


def _img_html_from_matplotlib(fig, dpi: int) -> str:
    """Converte figura Matplotlib para <img> base64 PNG (fundo branco)."""
    if not HAS_MPL or fig is None:
        return '<div class="nodata">Gráfico indisponível.</div>'
    buf = BytesIO()
    fig.savefig(
        buf,
        format="png",
        dpi=dpi,
        bbox_inches="tight",
        transparent=False,
        facecolor="white",
    )
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("ascii")
    return f'<img src="data:image/png;base64,{b64}" alt="chart" />'


def _build_donut_figure(
    labels: List[str],
    values: List[float],
    title: str,
    *,
    size_in: float,
    dpi: int,
    ring_width: float,
    font_size: float,
    min_label_pct: float,
    decimals: int,
) -> Optional["plt.Figure"]:
    """Donut com rótulos brancos dentro das fatias."""
    if not HAS_MPL:
        return None
    pairs = [(l, float(v)) for l, v in zip(labels, values) if float(v) > 0]
    if len(pairs) == 0 or sum(v for _, v in pairs) <= 0:
        return None

    labels = [str(l) if l is not None else "—" for l, _ in pairs]
    values = [float(v) for _, v in pairs]
    total = sum(values)

    fig, ax = plt.subplots(figsize=(size_in, size_in), dpi=dpi)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    colors = [PALETA_CORES[i % len(PALETA_CORES)] for i in range(len(values))]
    wedges, _texts = ax.pie(
        values,
        labels=None,
        startangle=90,
        counterclock=False,
        wedgeprops=dict(width=ring_width, edgecolor="white"),
        colors=colors,
    )

    inner_r = 1.0 - ring_width
    r_mid = inner_r + ring_width / 2.0

    for i, w in enumerate(wedges):
        share = values[i] / total if total else 0.0
        if share < min_label_pct:
            continue
        ang = (w.theta2 + w.theta1) / 2.0
        ang_rad = math.radians(ang)
        x = r_mid * math.cos(ang_rad)
        y = r_mid * math.sin(ang_rad)
        pct_text = f"{share * 100:.{decimals}f}%"
        txt = f"{labels[i]}\n{pct_text}"
        ax.text(
            x,
            y,
            txt,
            ha="center",
            va="center",
            color="white",
            fontsize=font_size,
            fontweight="bold",
        )

    ax.set_title(title, fontsize=8, color="#222")
    ax.axis("equal")
    return fig


def _donut_img_html_from_exposure_table(
    exp_df: pd.DataFrame,
    bucket_col: str,
    share_col: str,
    title: str,
    top_n: int,
    cfg: Dict[str, float],
) -> str:
    if exp_df is None or exp_df.empty:
        return '<div class="nodata">Sem dados.</div>'
    df = exp_df.copy()
    df = df[[bucket_col, share_col]].rename(
        columns={bucket_col: "categoria", share_col: "share"}
    )
    df["categoria"] = df["categoria"].fillna("—").astype(str)
    df = df.sort_values("share", ascending=False)

    df_top = df.head(top_n).copy()
    rest_share = float(df["share"].iloc[top_n:].sum()) if len(df) > top_n else 0.0
    if rest_share > 0:
        df_top = pd.concat(
            [df_top, pd.DataFrame([{"categoria": "Outros", "share": rest_share}])],
            ignore_index=True,
        )

    fig = _build_donut_figure(
        labels=df_top["categoria"].tolist(),
        values=df_top["share"].tolist(),
        title=title,
        size_in=cfg["size_in"],
        dpi=cfg["dpi"],
        ring_width=cfg["ring_width"],
        font_size=cfg["font_size"],
        min_label_pct=cfg["min_label_pct"],
        decimals=int(cfg["decimals"]),
    )
    if fig is None:
        return '<div class="nodata">Sem dados.</div>'
    return _img_html_from_matplotlib(fig, dpi=cfg["dpi"])


def _donut_img_html_from_maturity(
    mat_df: pd.DataFrame, share_col: str, title: str, cfg: Dict[str, float]
) -> str:
    if mat_df is None or mat_df.empty:
        return '<div class="nodata">Sem dados.</div>'
    order = ["<1 ano", "Entre 1 e 3 anos", "Entre 3 e 5 anos", "+5 anos"]
    tmp = mat_df.set_index("bucket").reindex(order).reset_index()
    labels = tmp["bucket"].fillna("—").tolist()
    values = tmp[share_col].fillna(0.0).tolist()

    fig = _build_donut_figure(
        labels=labels,
        values=values,
        title=title,
        size_in=cfg["size_in"],
        dpi=cfg["dpi"],
        ring_width=cfg["ring_width"],
        font_size=cfg["font_size"],
        min_label_pct=cfg["min_label_pct"],
        decimals=int(cfg["decimals"]),
    )
    if fig is None:
        return '<div class="nodata">Sem dados.</div>'
    return _img_html_from_matplotlib(fig, dpi=cfg["dpi"])


def _donut_img_html_from_factor_shares(
    shares: Dict[str, float], title: str, cfg: Dict[str, float]
) -> str:
    labels = ["PRÉ", "CDI", "IPCA", "OUTROS"]
    values = [
        shares.get("PRÉ", 0.0),
        shares.get("CDI", 0.0),
        shares.get("IPCA", 0.0),
        shares.get("OUTROS", 0.0),
    ]
    fig = _build_donut_figure(
        labels=labels,
        values=values,
        title=title,
        size_in=cfg["size_in"],
        dpi=cfg["dpi"],
        ring_width=cfg["ring_width"],
        font_size=cfg["font_size"],
        min_label_pct=cfg["min_label_pct"],
        decimals=int(cfg["decimals"]),
    )
    if fig is None:
        return '<div class="nodata">Sem dados.</div>'
    return _img_html_from_matplotlib(fig, dpi=cfg["dpi"])


# NEW: Barra para Soberano + S1 (% do PL em Renda Fixa)
def _bar_img_html_sov_s1(
    base_df: pd.DataFrame, base_total: float, cfg: Dict[str, float]
) -> str:
    if base_total <= 0 or not HAS_MPL:
        return '<div class="nodata">Sem dados.</div>'
    sov_mv = float(
        base_df.loc[base_df["issuer_bucket"] == "TESOURO NACIONAL", "mv"].sum()
    )
    s1_mv = float(base_df.loc[base_df["is_s1"], "mv"].sum())
    s2_mv = float(base_df.loc[base_df["is_s2"], "mv"].sum())
    sov_share = pct(sov_mv, base_total) * 100.0
    s1_share = pct(s1_mv, base_total) * 100.0
    s2_share = pct(s2_mv, base_total) * 100.0
    other_share = max(0.0, 100.0 - sov_share - s1_share - s2_share)

    labels = ["Soberano", "Instituições S1", "Instituições S2", "Outros"]
    values = [sov_share, s1_share, s2_share, other_share]
    colors = [PALETA_CORES[0], PALETA_CORES[1], PALETA_CORES[2], PALETA_CORES[3]]

    # Figura mais larga para rótulos
    fig, ax = plt.subplots(
        figsize=(cfg["size_in"] * 1.8, cfg["size_in"]), dpi=cfg["dpi"]
    )
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    bars = ax.bar(labels, values, color=colors)
    ax.set_ylim(0, 100)
    ax.set_ylabel("% do PL em Renda Fixa", fontsize=9)
    ax.set_title("Soberano, S1 e S2 – % do PL em Renda Fixa", fontsize=10)

    for bar, v in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.0,
            f"{v:.{int(cfg['decimals'])}f}%",
            ha="center",
            va="bottom",
            fontsize=max(8, int(cfg["font_size"])),
            color="#222",
            fontweight="bold",
        )

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    return _img_html_from_matplotlib(fig, dpi=cfg["dpi"])


# NEW: Barras para distribuição de rating (Relatório)
def _bar_img_html_ratings(
    rows: List[Dict[str, str]], title: str, cfg: Dict[str, float]
) -> str:
    if not HAS_MPL or not rows:
        return '<div class="nodata">Sem dados.</div>'
    df = pd.DataFrame(rows)
    if "rating" not in df.columns or "mv" not in df.columns:
        return '<div class="nodata">Sem dados.</div>'
    total = float(df["mv"].sum())
    if total <= 0:
        return '<div class="nodata">Sem dados.</div>'
    df["rating"] = df["rating"].apply(
        lambda x: "Sem Rating"
        if normalize_rating_label(x) is None
        else normalize_rating_label(x)
    )
    df["pct_val"] = df["mv"] / total * 100.0
    df["order"] = df["rating"].apply(rating_sort_key)
    df = df.sort_values("order")

    fig, ax = plt.subplots(
        figsize=(cfg["size_in"] * 1.6, cfg["size_in"]), dpi=cfg["dpi"]
    )
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    y = df["rating"].tolist()
    x = df["pct_val"].tolist()
    colors = [PALETA_CORES[i % len(PALETA_CORES)] for i in range(len(x))]
    bars = ax.barh(y, x, color=colors)
    ax.set_xlabel("% do Total", fontsize=9)
    ax.set_xlim(0, max(100, max(x) * 1.15))
    ax.set_title(title, fontsize=10)

    for bar, v in zip(bars, x):
        ax.text(
            bar.get_width() + max(0.5, 0.01 * max(x)),
            bar.get_y() + bar.get_height() / 2,
            f"{v:.{int(cfg['decimals'])}f}%",
            va="center",
            fontsize=max(8, int(cfg["font_size"])),
            color="#222",
            fontweight="bold",
        )

    # AAA no topo, Sem Rating no rodapé
    ax.invert_yaxis()

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    return _img_html_from_matplotlib(fig, dpi=cfg["dpi"])


# ---------------------------- CSV Export Helper ---------------------------- #


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """
    Exporta CSV em formato europeu; protege colunas id-like como strings.
    """
    df_copy = df.copy()

    protect_re = re.compile(r"(taxid|cnpj|cpf|codigo|cetip|isin|id$)", re.I)
    for col in df_copy.columns:
        if protect_re.search(col):
            df_copy[col] = df_copy[col].apply(
                lambda x: "" if pd.isna(x) else str(x)
            )

    numeric_cols = df_copy.select_dtypes(include=["number"]).columns
    for col in numeric_cols:
        df_copy[col] = df_copy[col].apply(
            lambda x: str(x).replace(".", ",") if pd.notna(x) else "",
            convert_dtype=False,
        )

    csv_str = df_copy.to_csv(index=False, sep=";", encoding="utf-8")
    return csv_str.encode("utf-8-sig")


# ---------------------------- Render do Relatório ---------------------------- #


def render_report_html() -> str:
    # Linhas de maturidade para contexto
    order = ["<1 ano", "Entre 1 e 3 anos", "Entre 3 e 5 anos", "+5 anos"]
    mat_rows = []
    for b in order:
        row = mat_buckets_df.loc[mat_buckets_df["bucket"] == b]
        base_pct = as_pct_str(float(row["share_base"].sum()))
        pl_pct = as_pct_str(float(row["share_pl"].sum()))
        mat_rows.append({"name": b, "base_pct": base_pct, "pl_pct": pl_pct})

    # Donut: Vencimento (base)
    donut_maturity_base = _donut_img_html_from_maturity(
        mat_buckets_df, "share_base", "Vencimentos – % do PL em Renda Fixa", DONUT_CFG
    )

    # Donut: Moeda
    donut_currency_base = _donut_img_html_from_exposure_table(
        currency_exp_table,
        "currency_code",
        "share_base",
        "Moeda – % do PL em Renda Fixa",
        top_n=donut_top_n,
        cfg=DONUT_CFG,
    )

    # Donut: Tipo (BRL/USD)
    if "parsed_bond_type" in df.columns and not bond_type_exp_table_brl.empty:
        donut_bondtype_brl = _donut_img_html_from_exposure_table(
            bond_type_exp_table_brl,
            "parsed_bond_type",
            "share_base",
            "Tipo de Título – BRL (% PL em Renda Fixa BRL)",
            top_n=donut_top_n,
            cfg=DONUT_CFG,
        )
    else:
        donut_bondtype_brl = '<div class="nodata">Sem dados.</div>'

    if "parsed_bond_type" in df.columns and not bond_type_exp_table_usd.empty:
        donut_bondtype_usd = _donut_img_html_from_exposure_table(
            bond_type_exp_table_usd,
            "parsed_bond_type",
            "share_base",
            "Tipo de Título – USD (% PL em Renda Fixa USD)",
            top_n=donut_top_n,
            cfg=DONUT_CFG,
        )
    else:
        donut_bondtype_usd = '<div class="nodata">Sem dados.</div>'

    # Donut: Fator de Risco (BRL/USD)
    donut_factor_brl = _donut_img_html_from_factor_shares(
        factor_shares_brl, "Fator de Risco – BRL (% PL em Renda Fixa BRL)", DONUT_CFG
    )
    donut_factor_usd = _donut_img_html_from_factor_shares(
        factor_shares_usd, "Fator de Risco – USD (% PL em Renda Fixa USD)", DONUT_CFG
    )

    # Tabelas Top-10 emissores por BRL/USD (ex Tesouro e S1)
    issuer_brl_rows = []
    issuer_usd_rows = []
    if base_total_brl > 0:
        exp_brl = exposure_table_by_bucket(
            df_pl=df.loc[df["currency_code"] == "BRL"],
            df_base=base_df_brl,
            pl_total=pl_total_brl,
            base_total=base_total_brl,
            bucket_col="issuer_bucket",
        )
        exp_brl = exp_brl.loc[~exp_brl["issuer_bucket"].isin(EXCLUDE_IN_LIMITS)]
        top_brl = exp_brl.head(10)
        issuer_brl_rows = [
            {
                "name": str(r["issuer_bucket"]),
                "base_pct": as_pct_str(float(r["share_base"])),
            }
            for _, r in top_brl.iterrows()
        ]

    if base_total_usd > 0:
        exp_usd = exposure_table_by_bucket(
            df_pl=df.loc[df["currency_code"] == "USD"],
            df_base=base_df_usd,
            pl_total=pl_total_usd,
            base_total=base_total_usd,
            bucket_col="issuer_bucket",
        )
        exp_usd = exp_usd.loc[~exp_usd["issuer_bucket"].isin(EXCLUDE_IN_LIMITS)]
        top_usd = exp_usd.head(10)
        issuer_usd_rows = [
            {
                "name": str(r["issuer_bucket"]),
                "base_pct": as_pct_str(float(r["share_base"])),
            }
            for _, r in top_usd.iterrows()
        ]

    # País (Top-10 + Outros)
    country_rows = []
    if not country_exp_table.empty:
        top_ctry = country_exp_table.head(10).copy()
        rest_base = float(country_exp_table["share_base"].iloc[10:].sum())
        for _, r in top_ctry.iterrows():
            country_rows.append(
                {
                    "name": str(r["country_bucket"]),
                    "base_pct": as_pct_str(float(r["share_base"])),
                }
            )
        if rest_base > 0:
            country_rows.append({"name": "Outros", "base_pct": as_pct_str(rest_base)})

    # BRL/USD statuses
    has_brl = base_total_brl > 0
    has_usd = base_total_usd > 0
    if has_brl:
        ok_single_brl = conc_brl["largest_share"] <= max_single_issuer
        ok_top10_brl = conc_brl["top10_share"] <= max_top10_issuers
        largest_brl = as_pct_str(conc_brl["largest_share"])
        top10_brl = as_pct_str(conc_brl["top10_share"])
    else:
        ok_single_brl = True
        ok_top10_brl = True
        largest_brl = "—"
        top10_brl = "—"

    if has_usd:
        ok_single_usd = conc_usd["largest_share"] <= max_single_issuer
        ok_top10_usd = conc_usd["top10_share"] <= max_top10_issuers
        largest_usd = as_pct_str(conc_usd["largest_share"])
        top10_usd = as_pct_str(conc_usd["top10_share"])
    else:
        ok_single_usd = True
        ok_top10_usd = True
        largest_usd = "—"
        top10_usd = "—"

    # NEW: Barra Soberano + S1 + S2
    sov_mv = float(
        base_df.loc[base_df["issuer_bucket"] == "TESOURO NACIONAL", "mv"].sum()
    )
    s1_mv = float(base_df.loc[base_df["is_s1"], "mv"].sum())
    s2_mv = float(base_df.loc[base_df["is_s2"], "mv"].sum())
    bank_credit_total = sov_mv + s1_mv + s2_mv

    bar_sov_s1 = _bar_img_html_sov_s1(base_df, base_total, DONUT_CFG)

    # Preparar dados de rating para template e gráficos (Relatório)
    brl_private_ratings_rows_formatted = []
    for r in brl_private_ratings_rows:
        brl_private_ratings_rows_formatted.append(
            {"rating": r["rating"], "mv_fmt": brl(r["mv"]), "pct": r["pct"], "mv": r["mv"]}
        )
    usd_corporate_ratings_rows_formatted = []
    for r in usd_corporate_ratings_rows:
        usd_corporate_ratings_rows_formatted.append(
            {"rating": r["rating"], "mv_fmt": brl(r["mv"]), "pct": r["pct"], "mv": r["mv"]}
        )

    # NEW: Gráficos de barras (estáticos) para ratings no relatório (ordenados)
    bar_ratings_brl = _bar_img_html_ratings(
        brl_private_ratings_rows_formatted,
        "Distribuição de Rating – BRL (% do total CRI+CRA+DEBENTURE)",
        DONUT_CFG,
    )
    bar_ratings_usd = _bar_img_html_ratings(
        usd_corporate_ratings_rows_formatted,
        "Distribuição de Rating – USD (% do total Corporate USD)",
        DONUT_CFG,
    )

    t = Template(REPORT_HTML)
    html = t.render(
        manager_name=manager_name,
        manager_cnpj=manager_cnpj,
        emission_date=datetime.today().strftime("%d/%m/%Y"),
        period_label=period_label,
        responsible_name=responsible_name,
        pl_total_fmt=brl(pl_total),
        base_total_fmt=brl(base_total),
        pl_credit_fmt=brl(pl_credit),
        credit_share_base=as_pct_str(pct(pl_credit, base_total)),
        credit_share_pl=as_pct_str(pct(pl_credit, pl_total)),
        show_ratings_section=show_ratings_section,
        min_aaa_pct=as_pct_str(DEFAULT_LIMITS["min_sovereign_or_aaa"]),
        aaa_share_base_pct=as_pct_str(aaa_share_base),
        aaa_share_pl_pct=as_pct_str(aaa_share_pl),
        ok_aaa=ok_aaa,
        conf_aaa=status_tag(ok_aaa),
        max_single_issuer_pct=as_pct_str(max_single_issuer),
        largest_issuer_share_base=as_pct_str(conc_base["largest_share"]),
        largest_issuer_share_pl=as_pct_str(largest_share_pl),
        ok_single=ok_single,
        conf_single=status_tag(ok_single),
        max_top10_issuers_pct=as_pct_str(max_top10_issuers),
        top10_issuers_share_base=as_pct_str(conc_base["top10_share"]),
        top10_issuers_share_pl=as_pct_str(top10_share_pl),
        ok_top10=ok_top10,
        conf_top10=status_tag(ok_top10),
        max_maturity_over_5y_pct=as_pct_str(max_over5y),
        over5y_pct_base=as_pct_str(over5y_share_base),
        over5y_pct_pl=as_pct_str(over5y_share_pl),
        ok_over5y=ok_over5y,
        conf_over5y=status_tag(ok_over5y),
        default_rate="0.00%",
        events_html=events_text,
        country_top=country_top_list,
        currency_top=currency_top_list,
        bond_type_top=bond_type_top_list,
        maturity_rows=mat_rows,
        donut_maturity_base=donut_maturity_base,
        donut_currency_base=donut_currency_base,
        donut_bondtype_brl=donut_bondtype_brl,
        donut_bondtype_usd=donut_bondtype_usd,
        donut_factor_brl=donut_factor_brl,
        donut_factor_usd=donut_factor_usd,
        has_brl=has_brl,
        has_usd=has_usd,
        issuer_brl_rows=issuer_brl_rows,
        issuer_usd_rows=issuer_usd_rows,
        largest_issuer_share_base_brl=largest_brl,
        top10_issuers_share_base_brl=top10_brl,
        ok_single_brl=ok_single_brl,
        ok_top10_brl=ok_top10_brl,
        conf_single_brl=status_tag(ok_single_brl),
        conf_top10_brl=status_tag(ok_top10_brl),
        largest_issuer_share_base_usd=largest_usd,
        top10_issuers_share_base_usd=top10_usd,
        ok_single_usd=ok_single_usd,
        ok_top10_usd=ok_top10_usd,
        conf_single_usd=status_tag(ok_single_usd),
        conf_top10_usd=status_tag(ok_top10_usd),
        country_rows=country_rows,
        bar_sov_s1=bar_sov_s1,
        bank_credit_total_fmt=brl(bank_credit_total),
        brl_private_ratings_rows=brl_private_ratings_rows_formatted,
        brl_private_total_fmt=brl(brl_private_total),
        usd_corporate_ratings_rows=usd_corporate_ratings_rows_formatted,
        usd_corporate_total_fmt=brl(usd_corporate_total),
        # Imagens dos gráficos de rating (ordenados por escala)
        bar_ratings_brl=bar_ratings_brl,
        bar_ratings_usd=bar_ratings_usd,
    )
    return html


st.subheader("Exportar Relatório")
if not HAS_MPL:
    st.warning(
        "Matplotlib não está instalado. Os gráficos estáticos no relatório "
        "serão substituídos por placeholders de texto."
    )

html_report = render_report_html()
st.download_button(
    "Baixar HTML",
    data=html_report.encode("utf-8"),
    file_name="relatorio_risco_credito.html",
    mime="text/html",
)

# ---------------------------- CSV Downloads ---------------------------- #

st.subheader("Exportar CSV (PL e PL em Renda Fixa)")
csv_pl_bytes = df_to_csv_bytes(df)
st.download_button(
    "Baixar CSV – PL (todas as posições)",
    data=csv_pl_bytes,
    file_name=f"portfolio_pl_{ref_dt.strftime('%Y%m%d')}.csv",
    mime="text/csv",
)
csv_base_bytes = df_to_csv_bytes(base_df)
st.download_button(
    "Baixar CSV – PL em Renda Fixa (Crédito + Tesouro)",
    data=csv_base_bytes,
    file_name=f"portfolio_base_{ref_dt.strftime('%Y%m%d')}.csv",
    mime="text/csv",
)

st.caption(
    "Observação: limites e cálculos usam o PL em Renda Fixa (Crédito + Tesouro). "
    "Os percentuais versus PL são exibidos como informação adicional. "
    "A concentração por emissor usa parsed_company_name e EXCLUI Tesouro, "
    "S1 e S2. Ao usar o banco SQLite local, a tabela fixa é "
    "'pmv_plus_gorila' e as datas de referência únicas são listadas para seleção. "
    "A referência de data usa exclusivamente 'reference_date'. Fundos (incluindo "
    "'Fixed income fund') não entram em Crédito mesmo se security_type indicar "
    "'BONDS'. País de risco vem diretamente da coluna 'country_risk'. "
    "O relatório inclui gráficos estáticos para Moeda, Vencimento, Tipo de Título, "
    "Fator de Risco BRL/USD, um gráfico de barras para Soberano, S1 e S2, "
    "e gráficos de barras para as distribuições de rating (BRL e USD), "
    "sempre ordenados por escala (AAA no topo, Sem Rating no rodapé). "
    "As tabelas do relatório são centralizadas e não ocupam 100% da largura."
)