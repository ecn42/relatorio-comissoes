# app.py
# Streamlit app to generate "Relatório de Monitoramento de Risco de Crédito"
# from a portfolio CSV (pipe-delimited). Now:
# - Issuer concentration uses parsed_company_name (with safe fallback)
# - Toggle to hide rating section
# - All checks/percentages are based on Base = Crédito + Tesouro
#   (and we also show "% do PL" as extra info)

import json
import re
import csv as _csv
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from jinja2 import Template

try:
    import pdfkit  # Optional (wkhtmltopdf must be installed)
    HAS_PDFKIT = True
except Exception:
    HAS_PDFKIT = False

### Simple Authentication
if not st.session_state.get("authenticated", False):
    st.warning("Please enter the password on the Home page first.")
    st.write("Status: Não Autenticado")
    st.stop()
    
  # prevent the rest of the page from running
st.write("Autenticado")

# ---------------------------- Config ---------------------------- #

st.set_page_config(
    page_title="Relatório de Risco de Crédito",
    layout="wide",
    initial_sidebar_state="expanded",
)

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
}

PUBLIC_BOND_TYPES = {
    "TREASURY_LOCAL_NTNB",
    "TREASURY_LOCAL_NTNC",
    "TREASURY_LOCAL_NTNF",
    "TREASURY_LOCAL_LFT",
    "TREASURY_LOCAL_LTN",
}

FUND_TYPES = {"FUNDQUOTE"}

IGNORE_IN_LIMITS_AS_TESOURO = {"TESOURO NACIONAL"}

# Default policy thresholds (editable in sidebar)
DEFAULT_LIMITS = {
    "max_single_issuer": 0.05,  # 5% da BASE (Crédito + Tesouro)
    "max_top10_issuers": 0.25,  # 25% da BASE
    "max_prefixed": 0.10,  # 10% da BASE
    "max_maturity_over_3y": 0.45,  # 45% da BASE
    "min_sovereign_or_aaa": 0.80,  # 80% da BASE
}


# ---------------------------- CSV Reader ---------------------------- #

def read_portfolio_csv(uploaded_file) -> pd.DataFrame:
    """
    Leitor robusto para arquivos com separador '|', com pipes nas bordas.
    - Detecta delimitador (entre ',', ';', '|', '\t'), fallback para '|'.
    - Remove colunas vazias criadas por pipes no início/fim da linha.
    - Normaliza nomes das colunas (strip).
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


def is_credit_row(row: pd.Series) -> bool:
    t = str(row.get("security_type", "")).strip().upper()
    if t in CREDIT_TYPES:
        return True
    if t == "CUSTOM" and str(row.get("asset_class", "")).upper() == "FIXED_INCOME":
        return True
    return False


def is_treasury_row(row: pd.Series) -> bool:
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
    for field in ("security_name", "security_description", "raw", "parsed_maturity"):
        d = to_date_safe(row.get(field, None))
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

    issuers = df.apply(issuer_from_row, axis=1, result_type="expand")
    df["issuer_cnpj"] = issuers[0]
    df["issuer_name_norm"] = issuers[1].fillna("DESCONHECIDO")

    df["reference_dt"] = df.apply(
        lambda row: (
            to_date_safe(
                (json.loads(row["raw"]).get("referenceDate"))
                if isinstance(row.get("raw", ""), str) and row.get("raw", "").strip()
                else None
            )
            or to_date_safe(row.get("reference_date", None))
        ),
        axis=1,
    )

    df = assign_market_value(df)

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

        left = df[
            ["issuer_cnpj", "issuer_name_norm", "issuer_bucket", "mv"]
        ].copy()
        left["issuer_cnpj"] = left["issuer_cnpj"].fillna("")

        right = m[["issuer_cnpj", "issuer_name_norm", "rating_bucket"]].copy()
        if "issuer_cnpj" not in right.columns:
            right["issuer_cnpj"] = ""
        right["issuer_cnpj"] = right["issuer_cnpj"].fillna("")

        join_df = left.merge(
            right, on=["issuer_cnpj", "issuer_name_norm"], how="left"
        )

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


def stress_loss_pct(
    df_credit: pd.DataFrame,
    denom_total: float,
    pd_shock: float,
    lgd: float,
) -> float:
    if denom_total <= 0:
        return 0.0
    credit_ead = df_credit["mv"].sum()
    return pct(credit_ead, denom_total) * pd_shock * lgd


def status_tag(ok: bool) -> str:
    return "em conformidade" if ok else "fora do limite"


# ---------------------------- Template ---------------------------- #

REPORT_HTML = """
<!DOCTYPE html>
<html lang="pt-br">
<head>
  <meta charset="utf-8" />
  <style>
    body { font-family: Arial, sans-serif; font-size: 12px; color: #222; }
    h1 { font-size: 18px; margin-bottom: 4px; }
    h2 { font-size: 16px; margin-top: 16px; }
    h3 { font-size: 14px; margin-top: 12px; }
    .small { font-size: 11px; color: #555; }
    .ok { color: #0a7a0a; font-weight: bold; }
    .bad { color: #b00020; font-weight: bold; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
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
    Base (Crédito + Tesouro): R$ {{ base_total_fmt }}<br/>
    Total de crédito: R$ {{ pl_credit_fmt }}
    ({{ credit_share_base }} da base; {{ credit_share_pl }} do PL)

    {% if show_ratings_section %}
    <h3>3.1 Exposição por Classe e Rating</h3>
    <ul>
      <li>
        Limite por Rating: mínimo {{ min_aaa_pct }} da base em Soberano/AAA.
      </li>
      <li>
        Exposição Atual Soberano/AAA:
        {{ aaa_share_base_pct }} da base ({{ aaa_share_pl_pct }} do PL)
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
        Máximo {{ max_single_issuer_pct }} da base em emissor único
        (Tesouro excluído). Atual: {{ largest_issuer_share_base }}
        da base ({{ largest_issuer_share_pl }} do PL)
        <span class="{{ 'ok' if ok_single else 'bad' }}">
          ({{ conf_single }})
        </span>
      </li>
      <li>
        Máximo {{ max_top10_issuers_pct }} da base nos 10 maiores emissores
        (Tesouro excluído). Atual: {{ top10_issuers_share_base }}
        da base ({{ top10_issuers_share_pl }} do PL)
        <span class="{{ 'ok' if ok_top10 else 'bad' }}">
          ({{ conf_top10 }})
        </span>
      </li>
    </ul>
  </div>

  <div class="section">
    <h3>3.3 Concentração por Fator de Risco</h3>
    <ul>
      <li>
        Pré-fixado: máximo {{ max_prefixed_pct }} da base.
        Atual (PRÉ): {{ prefixed_pct_base }} da base
        ({{ prefixed_pct_pl }} do PL)
        <span class="{{ 'ok' if ok_prefixed else 'bad' }}">
          ({{ conf_prefixed }})
        </span>
      </li>
      <li>CDI: {{ cdi_pct_base }} da base ({{ cdi_pct_pl }} do PL)</li>
      <li>IPCA: {{ ipca_pct_base }} da base ({{ ipca_pct_pl }} do PL)</li>
      <li>Outros: {{ outros_pct_base }} da base ({{ outros_pct_pl }} do PL)</li>
    </ul>
  </div>

  <div class="section">
    <h3>3.4 Concentração por Vencimento</h3>
    <ul>
      <li>
        Máximo {{ max_maturity_over_3y_pct }} da base em vencimentos > 3 anos
        (apenas crédito direto). Atual: {{ over3y_pct_base }} da base
        ({{ over3y_pct_pl }} do PL)
        <span class="{{ 'ok' if ok_over3y else 'bad' }}">
          ({{ conf_over3y }})
        </span>
      </li>
    </ul>
  </div>

  <div class="section">
    <h3>3.5 Inadimplência</h3>
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
        Perda potencial: {{ st1_loss_base }} da base
        ({{ st1_loss_pl }} do PL)
      </li>
      <li>
        Recessão: PD = {{ st2_pd }}, LGD = {{ st2_lgd }}.
        Perda potencial: {{ st2_loss_base }} da base
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
        Os níveis de risco de crédito são compatíveis com o perfil estabelecido.
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
      por vencimento) podem ser exportadas separadamente.
    </div>
  </div>
</body>
</html>
"""

# ---------------------------- Streamlit UI ---------------------------- #

st.title("Relatório de Monitoramento de Risco de Crédito")

st.sidebar.header("Arquivos")
csv_file = st.sidebar.file_uploader(
    "CSV da carteira (formato com '|')", type=["csv"]
)
ratings_file = st.sidebar.file_uploader(
    "Opcional: ratings.csv (issuer_cnpj, issuer_name, rating_bucket)",
    type=["csv"],
)

st.sidebar.header("Filtros")
portfolio_id = st.sidebar.text_input("portfolio_id (opcional)", value="")
ref_date_filter = st.sidebar.text_input(
    "Data de referência (opcional, dd/mm/aaaa)", value=""
)

st.sidebar.header("Seções do Relatório")
show_ratings_section = st.sidebar.checkbox(
    "Exibir informações de rating (Soberano/AAA)", value=False
)

st.sidebar.header("Limites (política) – denominador: Base (Crédito + Tesouro)")
max_single_issuer = st.sidebar.number_input(
    "Máx emissor único (BASE)",
    min_value=0.0,
    max_value=1.0,
    value=DEFAULT_LIMITS["max_single_issuer"],
    step=0.01,
)
max_top10_issuers = st.sidebar.number_input(
    "Máx 10 maiores emissores (BASE)",
    min_value=0.0,
    max_value=1.0,
    value=DEFAULT_LIMITS["max_top10_issuers"],
    step=0.01,
)
max_prefixed = st.sidebar.number_input(
    "Máx PRÉ (BASE)",
    min_value=0.0,
    max_value=1.0,
    value=DEFAULT_LIMITS["max_prefixed"],
    step=0.01,
)
max_over3y = st.sidebar.number_input(
    "Máx vencimentos > 3 anos (BASE)",
    min_value=0.0,
    max_value=1.0,
    value=DEFAULT_LIMITS["max_maturity_over_3y"],
    step=0.01,
)
min_aaa = st.sidebar.number_input(
    "Mín Soberano/AAA (BASE)",
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

events_text = st.text_area(
    "Eventos Relevantes (HTML ou texto simples):",
    value=(
        "<ul>"
        "<li>Destaques de resultados trimestrais (preencher).</li>"
        "</ul>"
    ),
    height=150,
)

if not csv_file:
    st.info("Carregue o CSV para iniciar.")
    st.stop()

# Load data
df_raw = read_portfolio_csv(csv_file)
df = normalize_dataframe(df_raw)

if portfolio_id:
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
ratings_df = None
if ratings_file:
    ratings_df = pd.read_csv(ratings_file, dtype=str, keep_default_na=False)

# Metrics: PL total, Crédito, Tesouro, Base
pl_total = compute_nav(df)

credit_df = df.loc[df["is_credit"]].copy()
treasury_df = df.loc[df["is_treasury"]].copy()
base_df = df.loc[df["is_credit"] | df["is_treasury"]].copy()

pl_credit = float(credit_df["mv"].sum())
pl_treasury = float(treasury_df["mv"].sum())
base_total = float(base_df["mv"].sum())

# Reference period label
ref_dates = df["reference_dt"].dropna().sort_values().unique()
if len(ref_dates) > 0:
    ref_dt = pd.to_datetime(ref_dates[-1]).to_pydatetime()
else:
    ref_dt = datetime.today()
period_label = ref_dt.strftime("%B de %Y").title()

# Rating share (relative to base and PL)
if show_ratings_section:
    aaa_share_base, _ = compute_sovereign_or_aaa_share(
        base_df, base_total, ratings_df
    )
    aaa_share_pl, _ = compute_sovereign_or_aaa_share(df, pl_total, ratings_df)
    ok_aaa = aaa_share_base >= min_aaa if base_total > 0 else True
else:
    aaa_share_base, aaa_share_pl, ok_aaa = 0.0, 0.0, True

# Concentration by issuer_bucket (parsed_company_name), ex-Tesouro
conc_base = compute_concentration_by_bucket(
    base_df, base_total, bucket_col="issuer_bucket",
    exclude_set=IGNORE_IN_LIMITS_AS_TESOURO
)
# Same numerators, different denominator for "% do PL"
largest_share_pl = pct(conc_base["largest_mv"], pl_total)
top10_share_pl = pct(conc_base["top10_mv"], pl_total)

# Factor risk shares (on base and on PL, for extra info)
factor_shares_base = compute_factor_risk_shares(base_df, base_total)
factor_shares_pl = compute_factor_risk_shares(df, pl_total)

# Maturity >3y: numerator = crédito, denominators = base and PL
over3y_share_base, over3y_table = compute_maturity_share_over_3y(
    credit_df, ref_dt, base_total
)
over3y_share_pl, _ = compute_maturity_share_over_3y(
    credit_df, ref_dt, pl_total
)

# Compliance checks (vs BASE)
ok_single = conc_base["largest_share"] <= max_single_issuer
ok_top10 = conc_base["top10_share"] <= max_top10_issuers
ok_prefixed = factor_shares_base.get("PRÉ", 0.0) <= max_prefixed
ok_over3y = over3y_share_base <= max_over3y

# Stress tests (% da base e % do PL)
st1_loss_base = stress_loss_pct(credit_df, base_total, st1_pd, st1_lgd)
st2_loss_base = stress_loss_pct(credit_df, base_total, st2_pd, st2_lgd)
st1_loss_pl = stress_loss_pct(credit_df, pl_total, st1_pd, st1_lgd)
st2_loss_pl = stress_loss_pct(credit_df, pl_total, st2_pd, st2_lgd)

# ---------------------------- Charts ---------------------------- #

st.subheader("Resumo")
c1, c2, c3, c4 = st.columns(4)
c1.metric(
    "PL Total (R$)",
    f"{pl_total:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."),
)
c2.metric(
    "Base (Crédito + Tesouro) (R$)",
    f"{base_total:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."),
)
c3.metric(
    "PL Crédito (R$)",
    f"{pl_credit:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."),
)
c4.metric(
    "% Crédito/PL | % Crédito/Base",
    f"{as_pct_str(pct(pl_credit, pl_total))} | "
    f"{as_pct_str(pct(pl_credit, base_total))}",
)

st.markdown("Exposição por Classe de Ativo")
df["class_bucket"] = np.select(
    [
        df["security_type"].astype(str).str.upper().isin(PUBLIC_BOND_TYPES),
        df["is_credit"],
        df["security_type"].astype(str).str.upper().isin(FUND_TYPES),
        df["asset_class"].astype(str).str.upper().eq("STOCKS"),
        df["asset_class"].astype(str).str.upper().eq("CASH"),
    ],
    ["Tesouro", "Crédito Privado", "Fundos", "Ações/ETFs", "Caixa"],
    default="Outros",
)
by_class = df.groupby("class_bucket")["mv"].sum().reset_index()
fig1 = px.pie(
    by_class, names="class_bucket", values="mv", title="Patrimônio por Classe",
    hole=0.4
)
st.plotly_chart(fig1, use_container_width=True)

st.markdown("Exposição por Fator de Risco (Base: Crédito + Tesouro)")
factor_plot = (
    pd.DataFrame(
        {
            "Fator": ["PRÉ", "CDI", "IPCA", "OUTROS"],
            "Base": [
                factor_shares_base.get("PRÉ", 0.0) * 100,
                factor_shares_base.get("CDI", 0.0) * 100,
                factor_shares_base.get("IPCA", 0.0) * 100,
                factor_shares_base.get("OUTROS", 0.0) * 100,
            ],
            "PL": [
                factor_shares_pl.get("PRÉ", 0.0) * 100,
                factor_shares_pl.get("CDI", 0.0) * 100,
                factor_shares_pl.get("IPCA", 0.0) * 100,
                factor_shares_pl.get("OUTROS", 0.0) * 100,
            ],
        }
    )
)
fig2 = px.bar(
    factor_plot, x="Fator", y="Base", text="Base",
    title="PL por Fator de Risco (Base)"
)
fig2.update_traces(texttemplate="%{text:.2f}%")
st.plotly_chart(fig2, use_container_width=True)
st.dataframe(
    factor_plot.assign(
        **{
            "Base": factor_plot["Base"].map(lambda v: f"{v:.2f}%"),
            "PL": factor_plot["PL"].map(lambda v: f"{v:.2f}%"),
        }
    )
)

st.markdown("Concentração por Emissor (parsed_company_name) – Base; Tesouro excluído")
issuer_table = conc_base["table"].reset_index()
issuer_table.columns = ["Emissor (parsed_company_name)", "Valor (R$)"]
issuer_table["% da Base"] = issuer_table["Valor (R$)"].apply(
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
            "% da Base": issuer_table["% da Base"].map(as_pct_str),
            "% do PL": issuer_table["% do PL"].map(as_pct_str),
        }
    )
)

st.markdown("Vencimentos (Crédito Privado)")
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
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("Sem datas de vencimento parseadas em crédito.")
else:
    st.info("Sem crédito privado para plotar.")

# ---------------------------- Report Rendering ---------------------------- #

def brl(v: float) -> str:
    return f"{v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def render_report_html() -> str:
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
        default_rate="0.00%",  # ajuste se rastrear atrasos/inadimplência
        events_html=events_text,
        st1_pd=as_pct_str(st1_pd),
        st1_lgd=as_pct_str(st1_lgd),
        st1_loss_base=as_pct_str(st1_loss_base),
        st1_loss_pl=as_pct_str(st1_loss_pl),
        st2_pd=as_pct_str(st2_pd),
        st2_lgd=as_pct_str(st2_lgd),
        st2_loss_base=as_pct_str(st2_loss_base),
        st2_loss_pl=as_pct_str(st2_loss_pl),
    )
    return html


st.subheader("Exportar Relatório")
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
            st.error(f"Falha ao gerar PDF: {e}")
else:
    st.info(
        "Para exportar PDF, instale wkhtmltopdf e o pacote pdfkit. "
        "Alternativamente, abra o HTML no Word e exporte para PDF."
    )

st.caption(
    "Observação: limites e cálculos usam a Base (Crédito + Tesouro). "
    "Os percentuais versus PL são exibidos como informação adicional. "
    "A concentração por emissor usa parsed_company_name."
)