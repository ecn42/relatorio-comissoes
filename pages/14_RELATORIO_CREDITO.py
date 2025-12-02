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

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from jinja2 import Template

# Matplotlib for static donuts in report (robust for Word/PDF)
try:
    import matplotlib

    matplotlib.use("Agg")  # headless backend for servers
    import matplotlib.pyplot as plt

    HAS_MPL = True
except Exception:
    HAS_MPL = False

try:
    import pdfkit  # Optional (wkhtmltopdf must be installed)

    HAS_PDFKIT = True
except Exception:
    HAS_PDFKIT = False

# ---------------------------- Config ---------------------------- #

st.set_page_config(
    page_title="Relatório de Risco de Crédito",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Simple Authentication
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

# Security-type aliases/tokens that denote funds in different datasets
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

# Tokens to detect funds in asset class/category strings or names
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

# Default policy thresholds (editable in sidebar)
DEFAULT_LIMITS = {
    "max_single_issuer": 0.05,
    "max_top10_issuers": 0.25,
    "max_prefixed": 0.10,
    "max_maturity_over_3y": 0.45,
    "min_sovereign_or_aaa": 0.80,
}

# ---------------------------- Color Palette ---------------------------- #

PALETA_CORES = [
    "#013220",
    "#8c6239",
    "#57575a",
    "#b08568",
    "#09202e",
    "#582308",
    "#7a6200",
]

# ---------------------------- CSV Reader ---------------------------- #


def read_portfolio_csv(uploaded_file) -> pd.DataFrame:
    """
    Robust reader for pipe-like CSV:
    - Detects delimiter, fallback '|' if needed.
    - Drops empty edge columns.
    - Strips columns names.
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
        cols = [row[1] for row in cur.fetchall()]
    return cols


def detect_tables_with_reference_date(conn: sqlite3.Connection) -> List[str]:
    """
    Returns tables that contain a 'reference_date' column (case-insensitive).
    """
    tables = _sqlite_list_tables(conn)
    out = []
    for t in tables:
        cols = [c.lower() for c in _sqlite_table_columns(conn, t)]
        if "reference_date" in cols:
            out.append(t)
    return out


def list_unique_reference_dates(
    conn: sqlite3.Connection, table: str
) -> List[str]:
    """
    Returns the raw distinct reference_date values from the table.
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
    Reads entire table from a local SQLite DB into a DataFrame.
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
    """
    Heuristics to classify funds.
    """
    t = _normalize_text_for_fund_detection(row.get("security_type", ""))
    if t in FUND_TYPES or t in FUND_TYPE_ALIASES:
        return True
    if re.search(r"\bFUND\b", t):
        return True

    class_fields = [
        "asset_class",
        "asset_category",
        "asset_subclass",
        "product_type",
        "class",
        "subclass",
        "category",
    ]
    for f in class_fields:
        val = _normalize_text_for_fund_detection(row.get(f, ""))
        if not val:
            continue
        for tok in FUND_TOKENS:
            if tok in val:
                return True
        if re.search(r"\bFUND(O|S)?\b", val):
            return True
        if re.search(r"\bFUNDO(S)?\b", val):
            return True

    text = " ".join(
        [
            _normalize_text_for_fund_detection(row.get("security_name", "")),
            _normalize_text_for_fund_detection(row.get("security_description", "")),
        ]
    )
    if re.search(r"\b(FUNDO|FUND|ETF|FII|FIDC|FIP)\b", text):
        return True
    if re.search(r"\b(COTA|COTAS|QUOTA|QUOTAS)\b", text):
        return True
    return False


def is_credit_row(row: pd.Series) -> bool:
    # Fundos nunca são crédito (inclui 'Fixed income fund')
    if is_fund_row(row):
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
    mtxt = first_nonempty(
        row.get("parsed_maturity_date"), row.get("parsed_maturity")
    )
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
    Country risk is sourced EXCLUSIVELY and DIRECTLY from 'country_risk'.
    No normalization, no fallback.
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
        df["security_asset_class_raw"] = sec.apply(
            lambda s: s.get("assetClass")
        )
        df["security_issuer_raw"] = sec.apply(lambda s: s.get("issuer"))
        df["security_market_value_raw"] = sec.apply(
            lambda s: s.get("marketValue")
        )

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

    issuers = df.apply(issuer_from_row, axis=1, result_type="expand")
    df["issuer_cnpj"] = issuers[0]
    df["issuer_name_norm"] = issuers[1].fillna("DESCONHECIDO")

    # Reference date: ONLY from 'reference_date' column
    if "reference_date" in df.columns:
        df["reference_dt"] = df["reference_date"].apply(to_date_safe)
    else:
        df["reference_dt"] = None

    df = assign_market_value(df)

    # Flags (fund, credit, treasury) — fund overrides credit/treasury
    df["is_fund"] = df.apply(is_fund_row, axis=1)
    df["is_credit"] = df.apply(is_credit_row, axis=1)
    df["is_treasury"] = df.apply(is_treasury_row, axis=1)

    # Issuer bucket for concentration: prefer parsed_company_name, fallback
    def issuer_bucket_name(row: pd.Series) -> str:
        if bool(row.get("is_treasury", False)):
            return "TESOURO NACIONAL"
        pc = row.get("parsed_company_name")
        if isinstance(pc, str) and pc.strip() and pc.strip() != "########":
            return clean_issuer_name(pc) or "DESCONHECIDO"
        return row.get("issuer_name_norm", "DESCONHECIDO") or "DESCONHECIDO"

    df["issuer_bucket"] = df.apply(issuer_bucket_name, axis=1)

    # Currency and Country buckets
    df["currency_code"] = df.apply(currency_from_row, axis=1)
    df["country_bucket"] = df.apply(country_from_row, axis=1)

    # Fund bucket (name) for dashboard
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
    return df


def compute_nav(df: pd.DataFrame) -> float:
    return float(df["mv"].sum())


def compute_sovereign_or_aaa_share(
    df: pd.DataFrame,
    denom_total: float,
    ratings_map: Optional[pd.DataFrame],
) -> Tuple[float, pd.DataFrame]:
    """
    Share de Soberano/AAA relativo a denom_total.
    Soberano: issuer_bucket == TESOURO NACIONAL.
    AAA: via ratings_map (issuer_cnpj/issuer_name).
    """
    if denom_total <= 0:
        return 0.0, pd.DataFrame()

    sovereign_mv = df.loc[df["issuer_bucket"] == "TESOURO NACIONAL", "mv"].sum()

    aaa_mv = 0.0
    join_df = pd.DataFrame()
    if ratings_map is not None and not ratings_map.empty:
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
    excluindo nomes no exclude_set (ex.: Tesouro).
    Retorna shares e tabela agregada.
    """
    if exclude_set is None:
        exclude_set = set()
    ex = df_subset.loc[~df_subset[bucket_col].isin(exclude_set)].copy()
    by = ex.groupby(bucket_col, dropna=False)["mv"].sum().sort_values(
        ascending=False
    )
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


def compute_maturity_share_over_3y(
    credit_df: pd.DataFrame, ref_date: datetime, denom_total: float
) -> Tuple[float, pd.DataFrame]:
    three_years = ref_date + timedelta(days=int(365.25 * 3))
    if credit_df.empty:
        return 0.0, credit_df
    tmp = credit_df.copy()
    tmp["over_3y"] = tmp["maturity_date"].map(
        lambda d: d is not None and d > three_years
    )
    over_mv = tmp.loc[tmp["over_3y"], "mv"].sum()
    share = pct(over_mv, denom_total)
    return share, tmp[["issuer_bucket", "maturity_date", "mv", "over_3y"]]


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
    Retorna DF com colunas:
    bucket, mv_credit, share_base, share_pl
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


def stress_loss_pct(
    df_credit: pd.DataFrame, denom_total: float, pd_shock: float, lgd: float
) -> float:
    if denom_total <= 0:
        return 0.0
    credit_ead = df_credit["mv"].sum()
    return pct(credit_ead, denom_total) * pd_shock * lgd


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
    Produz tabela de exposição por bucket para PL e PL em Renda Fixa.
    Retorna DataFrame com colunas:
    [bucket, mv_base, share_base, mv_pl, share_pl]
    Ordenado por share_base desc.
    """
    base = df_base.groupby(bucket_col, dropna=False)["mv"].sum().rename("mv_base")
    pl = df_pl.groupby(bucket_col, dropna=False)["mv"].sum().rename("mv_pl")
    exp = pd.concat([base, pl], axis=1).fillna(0.0).reset_index()
    exp.columns = [bucket_col, "mv_base", "mv_pl"]
    exp["share_base"] = exp["mv_base"].apply(
        lambda v: pct(float(v), float(base_total))
    )
    exp["share_pl"] = exp["mv_pl"].apply(lambda v: pct(float(v), float(pl_total)))
    exp = exp.sort_values(["share_base", "share_pl"], ascending=False)
    return exp


def top_list_from_exposure(
    exp_df: pd.DataFrame, bucket_col: str, top_n: int = 5
) -> List[Dict[str, str]]:
    """
    Constrói lista (para o relatório) com top_n buckets + 'Outros' agregado.
    Formata percentuais como strings.
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


# ---------------------------- Template ---------------------------- #

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
    table { border-collapse: collapse; width: 100%; }
    th, td {
      border-bottom: 1px solid #ddd;
      padding: 4px 6px;
      text-align: left;
      font-size: 12px;
    }
    .muted { color: #666; }
    .figure-note {
      margin-top: 4px; color: #666; font-size: 9px;
    }

    /* Centralizar as imagens (donuts) no relatório */
    .imgcell {
      width: 33%;
      vertical-align: top;
      padding: 4px;
      text-align: center;
      display: inline-block;
    }
    .imgcell img {
      max-width: 60%;
      height: auto;
      display: block;
      margin: 0 auto;
    }
    .nodata { color: #666; font-style: italic; padding: 8px 0 0 0; }

    /* Compact tables that fit on A4 */
    table.tbl-small {
      table-layout: fixed;
      width: 100%;
      font-size: 10px;
      page-break-inside: avoid;
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
    das carteiras administradas pela {{ manager_name }}, em conformidade com
    as diretrizes da CVM, Bacen e melhores práticas.
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
      Nota: "AAA" depende de arquivo de ratings opcional. Soberano =
      Tesouro Nacional.
    </div>
    {% else %}
    <h3>3.1 Exposição por Classe</h3>
    <div class="small muted">Seção de Rating desativada.</div>
    {% endif %}
  </div>

  <div class="section">
    <h3>3.2 Limites de Exposição</h3>
    <ul>
      <li>
        Máximo {{ max_single_issuer_pct }} do PL em Renda Fixa em emissor único
        (Tesouro excluído). Atual: {{ largest_issuer_share_base }}
        do PL em Renda Fixa ({{ largest_issuer_share_pl }} do PL)
        <span class="{{ 'ok' if ok_single else 'bad' }}">
          ({{ conf_single }})
        </span>
      </li>
      <li>
        Máximo {{ max_top10_issuers_pct }} do PL em Renda Fixa nos 10 maiores emissores
        (Tesouro excluído). Atual: {{ top10_issuers_share_base }}
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
      Limite: máximo {{ max_maturity_over_3y_pct }} do PL em Renda Fixa em vencimentos
      &gt; 3 anos. Atual: {{ over3y_pct_base }} do PL em Renda Fixa
      ({{ over3y_pct_pl }} do PL)
      <span class="{{ 'ok' if ok_over3y else 'bad' }}">
        ({{ conf_over3y }})
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
    <h2>5. Stress Testing</h2>
    <ul>
      <li>
        Crise de Liquidez: PD = {{ st1_pd }}, LGD = {{ st1_lgd }}.
        Perda potencial: {{ st1_loss_base }} do PL em Renda Fixa
        ({{ st1_loss_pl }} do PL)
      </li>
      <li>
        Recessão: PD = {{ st2_pd }}, LGD = {{ st2_lgd }}.
        Perda potencial: {{ st2_loss_base }} do PL em Renda Fixa
        ({{ st2_loss_pl }} do PL)
      </li>
    </ul>
  </div>

  <div class="section">
    <h2>6. Conformidade Regulatória</h2>
    <ul>
      <li>CVM: Instrução CVM nº 555/2014; Resolução CVM nº 50/21.</li>
      <li>Bacen: Circular nº 3.930/2019.</li>
      <li>ANBIMA: Código de Regulação e Melhores Práticas.</li>
    </ul>
  </div>

  <div class="section">
    <h2>7. Conclusões</h2>
    <ul>
      <li>
        Os níveis de risco de crédito são compatíveis com o perfil
        estabelecido.
      </li>
      <li>
        Não foram identificadas exposições que comprometam a estabilidade
        financeira das carteiras.
      </li>
      <li>
        As políticas de risco são seguidas com monitoramento contínuo.
      </li>
    </ul>
  </div>

  <div class="section">
    <h2>8. Recomendações</h2>
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
        "Caminho do banco (relativo ao projeto)", value="gorila_positions.db"
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
                    raw_dates = list_unique_reference_dates(
                        conn, db_table_selected
                    )
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

# Reference date filter
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
max_over3y = st.sidebar.number_input(
    "Máx vencimentos > 3 anos (PL em Renda Fixa)",
    min_value=0.0,
    max_value=1.0,
    value=DEFAULT_LIMITS["max_maturity_over_3y"],
    step=0.01,
)
min_aaa = st.sidebar.number_input(
    "Mín Soberano/AAA (PL em Renda Fixa)",
    min_value=0.0,
    max_value=1.0,
    value=DEFAULT_LIMITS["min_sovereign_or_aaa"],
    step=0.01,
)

st.sidebar.header("Stress Tests")
st1_pd = st.sidebar.number_input(
    "Cenário 1 PD (Crise Liquidez)", 0.0, 1.0, 0.05, 0.01
)
st1_lgd = st.sidebar.number_input("Cenário 1 LGD", 0.0, 1.0, 0.40, 0.05)
st2_pd = st.sidebar.number_input(
    "Cenário 2 PD (Recessão)", 0.0, 1.0, 0.10, 0.01
)
st2_lgd = st.sidebar.number_input("Cenário 2 LGD", 0.0, 1.0, 0.50, 0.05)

st.sidebar.header("Cabeçalho do Relatório")
manager_name = st.sidebar.text_input(
    "Gestora", value="Ceres Asset Gestão de Investimentos Ltda"
)
manager_cnpj = st.sidebar.text_input("CNPJ", value="40.962.925/0001-38")
responsible_name = st.sidebar.text_input("Responsável", value="Brenno Melo")

# ---------------------------- NEW: Graph Controls -------------------- #

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
    "Top N categorias (demais = 'Outros')",
    min_value=3,
    max_value=10,
    value=6,
    step=1,
)
donut_min_pct_label = st.sidebar.slider(
    "Ocultar rótulos abaixo de (%)",
    min_value=0.0,
    max_value=10.0,
    value=1.0,
    step=0.5,
)
donut_decimals = st.sidebar.slider(
    "Decimais nos %", min_value=0, max_value=2, value=1, step=1
)

DONUT_CFG = {
    "size_in": donut_size_in,
    "dpi": donut_dpi,
    "ring_width": donut_ring_width,
    "font_size": donut_font_size,
    "min_label_pct": donut_min_pct_label / 100.0,  # convert to [0-1]
    "decimals": donut_decimals,
}

st.sidebar.header("Gráficos do Dashboard")
dash_height = st.sidebar.slider(
    "Altura (px)", min_value=240, max_value=900, value=420, step=10
)

events_text = st.text_area(
    "Eventos Relevantes (HTML ou texto simples):",
    value=("<ul><li>Destaques de resultados trimestrais (preencher).</li></ul>"),
    height=150,
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

# Load data
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

# Load ratings map if provided
ratings_file = st.sidebar.file_uploader(
    "Opcional: ratings.csv (issuer_cnpj, issuer_name, rating_bucket)",
    type=["csv"],
)
ratings_df = None
if ratings_file:
    ratings_df = pd.read_csv(ratings_file, dtype=str, keep_default_na=False)

# Metrics: PL total, Crédito, Tesouro, PL em Renda Fixa
pl_total = compute_nav(df)

credit_df = df.loc[df["is_credit"]].copy()
treasury_df = df.loc[df["is_treasury"]].copy()
funds_df = df.loc[df["is_fund"]].copy()
base_df = df.loc[df["is_credit"] | df["is_treasury"]].copy()

pl_credit = float(credit_df["mv"].sum())
pl_treasury = float(treasury_df["mv"].sum())
pl_funds = float(funds_df["mv"].sum())
base_total = float(base_df["mv"].sum())

# Currency sub-bases for BRL and USD
base_df_brl = base_df.loc[base_df["currency_code"] == "BRL"].copy()
base_df_usd = base_df.loc[base_df["currency_code"] == "USD"].copy()
base_total_brl = float(base_df_brl["mv"].sum())
base_total_usd = float(base_df_usd["mv"].sum())

pl_total_brl = compute_nav(df.loc[df["currency_code"] == "BRL"])
pl_total_usd = compute_nav(df.loc[df["currency_code"] == "USD"])

# Reference period label
ref_dates = df["reference_dt"].dropna().sort_values().unique()
if len(ref_dates) > 0:
    ref_dt = pd.to_datetime(ref_dates[-1]).to_pydatetime()
else:
    ref_dt = datetime.today()
period_label = ref_dt.strftime("%B de %Y").title()

# Rating share (relative to PL em Renda Fixa and PL)
if show_ratings_section:
    aaa_share_base, _ = compute_sovereign_or_aaa_share(
        base_df, base_total, ratings_df
    )
    aaa_share_pl, _ = compute_sovereign_or_aaa_share(df, pl_total, ratings_df)
    ok_aaa = aaa_share_base >= min_aaa if base_total > 0 else True
else:
    aaa_share_base, aaa_share_pl, ok_aaa = 0.0, 0.0, True

# Concentration by issuer_bucket (parsed_company_name), ex-Tesouro (overall)
conc_base = compute_concentration_by_bucket(
    base_df,
    base_total,
    bucket_col="issuer_bucket",
    exclude_set=IGNORE_IN_LIMITS_AS_TESOURO,
)
largest_share_pl = pct(conc_base["largest_mv"], pl_total)
top10_share_pl = pct(conc_base["top10_mv"], pl_total)

# Concentration by issuer_bucket for BRL and USD (for report 3.2 tables)
if base_total_brl > 0:
    conc_brl = compute_concentration_by_bucket(
        base_df_brl,
        base_total_brl,
        bucket_col="issuer_bucket",
        exclude_set=IGNORE_IN_LIMITS_AS_TESOURO,
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
        exclude_set=IGNORE_IN_LIMITS_AS_TESOURO,
    )
else:
    conc_usd = {
        "largest_share": 0.0,
        "top10_share": 0.0,
        "table": pd.Series(dtype=float),
    }

# Factor risk shares
factor_shares_base = compute_factor_risk_shares(base_df, base_total)
factor_shares_pl = compute_factor_risk_shares(
    df, pl_total
)  # info only (not graphed)
factor_shares_brl = compute_factor_risk_shares(base_df_brl, base_total_brl)
factor_shares_usd = compute_factor_risk_shares(base_df_usd, base_total_usd)

# Maturity >3y: numerator = crédito, denominators = PL em Renda Fixa and PL
over3y_share_base, over3y_table = compute_maturity_share_over_3y(
    credit_df, ref_dt, base_total
)
over3y_share_pl, _ = compute_maturity_share_over_3y(credit_df, ref_dt, pl_total)

# New: maturity buckets (credit only) vs PL em Renda Fixa and PL
mat_buckets_df = compute_maturity_buckets_exposure(
    credit_df, ref_dt, base_total, pl_total
)

# Compliance checks (vs PL em Renda Fixa)
ok_single = conc_base["largest_share"] <= max_single_issuer
ok_top10 = conc_base["top10_share"] <= max_top10_issuers
ok_prefixed = factor_shares_base.get("PRÉ", 0.0) <= max_prefixed
ok_over3y = over3y_share_base <= max_over3y

# Stress tests (% do PL em Renda Fixa e % do PL)
st1_loss_base = stress_loss_pct(credit_df, base_total, st1_pd, st1_lgd)
st2_loss_base = stress_loss_pct(credit_df, base_total, st2_pd, st2_lgd)
st1_loss_pl = stress_loss_pct(credit_df, pl_total, st1_pd, st1_lgd)
st2_loss_pl = stress_loss_pct(credit_df, pl_total, st2_pd, st2_lgd)

# ---------------------------- New Exposures ---------------------------- #

# Country risk exposure (uses raw country_risk)
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

# Currency exposure
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

# parsed_bond_type exposure overall and by BRL/USD (exactly as in the column)
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

# ---------------------------- Charts (Dashboard) -------------------------- #

def center_chart(fig):
    """
    Centraliza um gráfico (Plotly) usando colunas.
    Ajuste as proporções [1, 8, 1] se quiser mais/menos largura.
    """
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
    by_class,
    names="class_bucket",
    values="mv",
    title="Patrimônio por Classe",
    hole=0.2,
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

# Factor Risk – split BRL and USD
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
    "Concentração por Emissor (parsed_company_name) – PL em Renda Fixa; Tesouro excluído"
)
issuer_table = conc_base["table"].reset_index()
issuer_table.columns = ["Emissor (parsed_company_name)", "Valor (R$)"]
issuer_table["% do PL em Renda Fixa"] = issuer_table["Valor (R$)"].apply(
    lambda x: pct(x, base_total)
)
issuer_table["% do PL"] = issuer_table["Valor (R$)"].apply(
    lambda x: pct(x, pl_total)
)
st.dataframe(
    issuer_table.assign(
        **{
            "Valor (R$)": issuer_table["Valor (R$)"].map(
                lambda v: f"{v:,.2f}"
                .replace(",", "X")
                .replace(".", ",")
                .replace("X", ".")
            ),
            "% do PL em Renda Fixa": issuer_table["% do PL em Renda Fixa"].map(as_pct_str),
            "% do PL": issuer_table["% do PL"].map(as_pct_str),
        }
    )
)

# Country Risk Exposure (TABLE instead of chart)
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

# Currency Exposure (charts only)
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
    fig_curr.update_layout(
        font=dict(size=9), showlegend=False, height=dash_height
    )
    center_chart(fig_curr)
else:
    st.info("Sem dados para exposição cambial.")

# parsed_bond_type Exposure – split BRL and USD
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

    st.markdown("Exposição por Tipo de Título (parsed_bond_type) – % do PL em Renda Fixa USD")
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

# Maturities buckets (credit only) – charts only
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

# Optional: scatter distribution
st.markdown("Distribuição de Vencimentos (pontos)")
if not credit_df.empty:
    cred_plot = credit_df.copy()
    cred_plot["Maturity"] = cred_plot["maturity_date"]
    cred_plot["Issuer"] = cred_plot["issuer_bucket"]
    cred_plot["Valor"] = cred_plot["mv"]
    cred_plot = cred_plot.loc[cred_plot["Maturity"].notna()]
    if not cred_plot.empty:
        fig3 = px.scatter(
            cred_plot,
            x="Maturity",
            y="Valor",
            color="Issuer",
            size="Valor",
            title="Distribuição de Vencimentos",
        )
        fig3.update_layout(
            font=dict(size=9), colorway=PALETA_CORES, height=dash_height
        )
        center_chart(fig3)
    else:
        st.info("Sem datas de vencimento parseadas em crédito.")
else:
    st.info("Sem crédito privado para plotar.")

# ------------------- Report Donuts (Static PNG) -------------------- #


def _img_html_from_matplotlib(fig, dpi: int) -> str:
    """Convert a Matplotlib figure to base64 PNG <img> HTML (white bg)."""
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
    """
    Build a donut chart (white background) with tiny white text labels
    and percentages INSIDE each wedge.
    """
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


# ---------------------------- CSV Export Helper ---------------------------- #


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """
    Export to CSV with European format; protect id-like columns as strings.
    """
    df_copy = df.copy()

    # Protect id-like columns (avoid scientific notation)
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


# ---------------------------- Report Rendering ---------------------------- #


def brl(v: float) -> str:
    return f"{v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def render_report_html() -> str:
    # Maturity rows kept for context (not directly shown as table)
    order = ["<1 ano", "Entre 1 e 3 anos", "Entre 3 e 5 anos", "+5 anos"]
    mat_rows = []
    for b in order:
        row = mat_buckets_df.loc[mat_buckets_df["bucket"] == b]
        base_pct = as_pct_str(float(row["share_base"].sum()))
        pl_pct = as_pct_str(float(row["share_pl"].sum()))
        mat_rows.append({"name": b, "base_pct": base_pct, "pl_pct": pl_pct})

    # Donuts: Maturity (PL em Renda Fixa only)
    donut_maturity_base = _donut_img_html_from_maturity(
        mat_buckets_df, "share_base", "Vencimentos – % do PL em Renda Fixa", DONUT_CFG
    )

    # Donuts: Moeda (PL em Renda Fixa only)
    donut_currency_base = _donut_img_html_from_exposure_table(
        currency_exp_table,
        "currency_code",
        "share_base",
        "Moeda – % do PL em Renda Fixa",
        top_n=donut_top_n,
        cfg=DONUT_CFG,
    )

    # Donuts: Tipo de Título (PL em Renda Fixa BRL/USD)
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

    # Donuts: Fator de Risco (PL em Renda Fixa BRL/USD)
    donut_factor_brl = _donut_img_html_from_factor_shares(
        factor_shares_brl, "Fator de Risco – BRL (% PL em Renda Fixa BRL)", DONUT_CFG
    )
    donut_factor_usd = _donut_img_html_from_factor_shares(
        factor_shares_usd, "Fator de Risco – USD (% PL em Renda Fixa USD)", DONUT_CFG
    )

    # Build issuer Top-10 tables for BRL/USD (exclude Tesouro)
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
        exp_brl = exp_brl.loc[
            ~exp_brl["issuer_bucket"].isin(IGNORE_IN_LIMITS_AS_TESOURO)
        ]
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
        exp_usd = exp_usd.loc[
            ~exp_usd["issuer_bucket"].isin(IGNORE_IN_LIMITS_AS_TESOURO)
        ]
        top_usd = exp_usd.head(10)
        issuer_usd_rows = [
            {
                "name": str(r["issuer_bucket"]),
                "base_pct": as_pct_str(float(r["share_base"])),
            }
            for _, r in top_usd.iterrows()
        ]

    # Country table rows (Top-10 + Outros)
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
            country_rows.append(
                {"name": "Outros", "base_pct": as_pct_str(rest_base)}
            )

    # BRL/USD limit statuses
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
        min_aaa_pct=as_pct_str(min_aaa),
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
        max_prefixed_pct=as_pct_str(max_prefixed),
        prefixed_pct_base=as_pct_str(factor_shares_base.get("PRÉ", 0.0)),
        prefixed_pct_pl=as_pct_str(factor_shares_pl.get("PRÉ", 0.0)),
        cdi_pct_base=as_pct_str(factor_shares_base.get("CDI", 0.0)),
        cdi_pct_pl=as_pct_str(factor_shares_pl.get("CDI", 0.0)),
        ipca_pct_base=as_pct_str(factor_shares_base.get("IPCA", 0.0)),
        ipca_pct_pl=as_pct_str(factor_shares_pl.get("IPCA", 0.0)),
        outros_pct_base=as_pct_str(factor_shares_base.get("OUTROS", 0.0)),
        outros_pct_pl=as_pct_str(factor_shares_pl.get("OUTROS", 0.0)),
        ok_prefixed=ok_prefixed,
        conf_prefixed=status_tag(ok_prefixed),
        max_maturity_over_3y_pct=as_pct_str(max_over3y),
        over3y_pct_base=as_pct_str(over3y_share_base),
        over3y_pct_pl=as_pct_str(over3y_share_pl),
        ok_over3y=ok_over3y,
        conf_over3y=status_tag(ok_over3y),
        default_rate="0.00%",
        events_html=events_text,
        st1_pd=as_pct_str(st1_pd),
        st1_lgd=as_pct_str(st1_lgd),
        st1_loss_base=as_pct_str(st1_loss_base),
        st1_loss_pl=as_pct_str(st1_loss_pl),
        st2_pd=as_pct_str(st2_pd),
        st2_lgd=as_pct_str(st2_lgd),
        st2_loss_base=as_pct_str(st2_loss_base),
        st2_loss_pl=as_pct_str(st2_loss_pl),
        # Lists and images for sections
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
        # New issuer tables per currency
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
        # Country table rows
        country_rows=country_rows,
    )
    return html


st.subheader("Exportar Relatório")
if not HAS_MPL:
    st.warning(
        "Matplotlib não está instalado. Os gráficos donut no relatório "
        "serão substituídos por placeholders de texto."
    )

html_report = render_report_html()
st.download_button(
    "Baixar HTML",
    data=html_report.encode("utf-8"),
    file_name="relatorio_risco_credito.html",
    mime="text/html",
)

if HAS_PDFKIT:
    if st.button("Gerar PDF (requer wkhtmltopdf instalado)"):

        try:
            pdf_bytes = pdfkit.from_string(html_report, False)
            st.download_button(
                "Baixar PDF",
                data=pdf_bytes,
                file_name="relatorio_risco_credito.pdf",
                mime="application/pdf",
            )
        except Exception as e:
            st.error(
                "Falha ao gerar PDF. Detalhes: "
                f"{e}. Dica: verifique a instalação do wkhtmltopdf."
            )
else:
    st.info(
        "Para exportar PDF, instale wkhtmltopdf e o pacote pdfkit. "
        "Alternativamente, abra o HTML no Word e exporte para PDF. "
        "Os gráficos do relatório são imagens (PNG) e serão preservados."
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
    "A concentração por emissor usa parsed_company_name. Ao usar o banco "
    "SQLite local, a tabela fixa é 'pmv_plus_gorila' e as datas de referência "
    "únicas são listadas para seleção. A referência de data usa "
    "exclusivamente 'reference_date'. Fundos (incluindo 'Fixed income fund') "
    "não entram em Crédito mesmo se security_type indicar 'BONDS'. País de "
    "risco vem diretamente da coluna 'country_risk'. O dashboard e o "
    "relatório incluem exposições por Moeda, Vencimento (buckets) e Tipo de "
    "Título. Fator de Risco e Tipo de Título possuem análises separadas para "
    "BRL e USD. Donuts do relatório têm fundo branco, rótulos e percentuais "
    "brancos dentro das fatias (fonte pequena configurável). As tabelas de "
    "Top-10 emissores por BRL/USD e a tabela de País foram otimizadas para "
    "cabimento em folha A4. Arquivos CSV podem ser baixados para PL ou PL em Renda Fixa."
)