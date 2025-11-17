import os
import json
import time
from datetime import date, timedelta
from typing import Dict, Iterable, List, Optional, Tuple
import re
import io
import unicodedata

import pandas as pd
import requests
import streamlit as st
import sqlite3

### Simple Authentication
if not st.session_state.get("authenticated", False):
    st.warning("Please enter the password on the Home page first.")
    st.write("Status: Não Autenticado")
    st.stop()
    
  # prevent the rest of the page from running
st.write("Autenticado")

# -------------------------------
# Config / Secrets
# -------------------------------


def get_api_key() -> Optional[str]:
    if "GORILA_API_KEY" in st.secrets:
        return st.secrets["GORILA_API_KEY"]
    return os.getenv("GORILA_API_KEY")


def default_reference_date() -> date:
    """
    Return the date 3 working days before today.
    Saturday and Sunday are considered non-working days.
    """
    today = date.today()
    d = today
    business_days = 0
    while business_days < 3:
        d = d - timedelta(days=1)
        if d.weekday() < 5:  # Monday=0, Sunday=6
            business_days += 1
    return d


# -------------------------------
# HTTP Client (with retries/backoff)
# -------------------------------


class GorilaClient:
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://core.gorila.com.br",
        timeout: int = 30,
        max_retries: int = 5,
        backoff_base: float = 0.6,
    ):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({"Authorization": api_key})
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_base = backoff_base

    def _safe_json(self, resp: requests.Response) -> Dict:
        if not resp.text:
            return {}
        try:
            return resp.json()
        except ValueError:
            return {}

    def _request(self, method: str, url: str, params: Dict = None) -> Dict:
        retryable_status = {408, 429, 500, 502, 503, 504}
        last_err = None

        for attempt in range(self.max_retries):
            try:
                r = self.session.request(
                    method, url, params=params, timeout=self.timeout
                )

                if r.status_code in retryable_status:
                    last_err = Exception(f"HTTP {r.status_code}: {r.text[:200]}")
                    wait = self.backoff_base * (2**attempt)
                    time.sleep(wait)
                    continue

                if 400 <= r.status_code < 500:
                    r.raise_for_status()

                return self._safe_json(r)

            except requests.HTTPError:
                raise
            except requests.RequestException as e:
                last_err = e
                wait = self.backoff_base * (2**attempt)
                time.sleep(wait)

        raise RuntimeError(f"Request failed after retries: {last_err}") from last_err

    def _get(self, path_or_url: str, params: Dict = None) -> Dict:
        url = (
            path_or_url
            if str(path_or_url).startswith("http")
            else f"{self.base_url}/{path_or_url.lstrip('/')}"
        )
        return self._request("GET", url, params=params or {})

    def _iter_records(self, payload):
        if isinstance(payload, dict) and "records" in payload:
            for r in payload.get("records", []) or []:
                yield r
        elif isinstance(payload, list):
            for r in payload:
                yield r
        else:
            if payload:
                yield payload

    def _extract_next(self, payload) -> Optional[str]:
        if isinstance(payload, dict):
            nxt = payload.get("next")
            return str(nxt).strip() if nxt else None
        return None

    def _paginate(self, path: str, params: Dict = None) -> Iterable[Dict]:
        payload = self._get(path, params=params)
        for r in self._iter_records(payload):
            yield r

        next_url = self._extract_next(payload)
        seen = set()
        empty_pages = 0

        while next_url:
            if next_url in seen:
                break
            seen.add(next_url)

            try:
                payload = self._get(next_url)
            except Exception:
                break

            count = 0
            for r in self._iter_records(payload):
                yield r
                count += 1

            if count == 0:
                empty_pages += 1
                if empty_pages >= 1:
                    break
            else:
                empty_pages = 0

            next_url = self._extract_next(payload)

    # Endpoints
    def list_portfolios(self) -> Iterable[Dict]:
        return self._paginate("/portfolios")

    def list_issuers(self) -> Iterable[Dict]:
        return self._paginate("/issuers")

    def list_positions(
        self, portfolio_id: str, reference_date: Optional[str] = None
    ) -> Iterable[Dict]:
        params = {}
        if reference_date:
            params["referenceDate"] = reference_date
        path = f"/portfolios/{portfolio_id}/positions"
        return self._paginate(path, params=params)

    def list_position_market_values(
        self, portfolio_id: str, reference_date: Optional[str] = None
    ) -> Iterable[Dict]:
        params = {}
        if reference_date:
            params["referenceDate"] = reference_date
        path = f"/portfolios/{portfolio_id}/positions/market-values"
        return self._paginate(path, params=params)


# -------------------------------
# Minimal flatteners (view/export)
# -------------------------------


def flatten_portfolio(p: Dict) -> Dict:
    return {
        "id": p.get("id"),
        "name": p.get("name"),
        "autoRnC": p.get("autoRnC"),
        "raw": json.dumps(p, ensure_ascii=False),
    }


def flatten_issuer(i: Dict) -> Dict:
    return {
        "id": i.get("id"),
        "name": i.get("name"),
        "taxId": i.get("taxId") or i.get("cnpj"),
        "raw": json.dumps(i, ensure_ascii=False),
    }


def flatten_position(portfolio_id: str, pos: Dict) -> Dict:
    sec = pos.get("security", {}) or {}
    bro = pos.get("broker", {}) or {}

    return {
        "portfolio_id": portfolio_id,
        "position_id": pos.get("id") or pos.get("positionId"),
        "reference_date": pos.get("referenceDate"),
        "quantity": pos.get("quantity"),
        "currency": pos.get("currency"),
        "security_id": sec.get("id"),
        "security_name": sec.get("name"),
        "security_type": sec.get("type"),
        "asset_class": sec.get("assetClass"),
        "isin": sec.get("isin"),
        "maturity_date": sec.get("maturityDate"),
        "validity_date": sec.get("validityDate"),
        "issuer_id": sec.get("issuerId"),
        "broker_id": bro.get("id"),
        "broker_name": bro.get("name"),
        "raw": json.dumps(pos, ensure_ascii=False),
    }


def flatten_position_market_value(portfolio_id: str, item: Dict) -> Dict:
    sec = item.get("security", {}) or {}

    mv_field = item.get("marketValue")
    if isinstance(mv_field, dict):
        mv_amount = mv_field.get("amount")
        mv_curr = mv_field.get("currency")
    else:
        mv_amount = item.get("marketValue") or item.get("value")
        mv_curr = item.get("currency")

    price_field = item.get("price") or item.get("unitPrice")
    if isinstance(price_field, dict):
        price_amount = price_field.get("amount")
        price_curr = price_field.get("currency")
    else:
        price_amount = price_field
        price_curr = item.get("currency")

    return {
        "portfolio_id": portfolio_id,
        "position_id": item.get("positionId") or item.get("position_id"),
        "reference_date": item.get("referenceDate"),
        "security_id": sec.get("id"),
        "security_name": sec.get("name"),
        "security_type": sec.get("type"),
        "asset_class": sec.get("assetClass"),
        "isin": sec.get("isin"),
        "quantity": item.get("quantity"),
        "currency": item.get("currency"),
        "price_amount": price_amount,
        "price_currency": price_curr,
        "market_value_amount": mv_amount,
        "market_value_currency": mv_curr,
        "raw": json.dumps(item, ensure_ascii=False),
    }


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Export to CSV with European format; protect id-like columns as strings."""
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


# -------------------------------
# Issuer JSON helpers (id + name)
# -------------------------------


def load_issuer_json_df(path: str) -> pd.DataFrame:
    if not path or not os.path.exists(path):
        return pd.DataFrame(columns=["id", "name"])
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return pd.DataFrame(columns=["id", "name"])

    rows = []
    if isinstance(data, dict):
        for k, v in data.items():
            rows.append({"id": str(k).strip(), "name": str(v) if v else None})
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                iid = item.get("id")
                nm = item.get("name")
                if iid is not None:
                    rows.append({"id": str(iid).strip(), "name": nm})
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["id", "name"])
    df = df.drop_duplicates(subset=["id"]).sort_values("id").reset_index(
        drop=True
    )
    return df[["id", "name"]]


def save_issuer_json_df(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    out = (
        df[["id", "name"]]
        .dropna(subset=["id"])
        .drop_duplicates(subset=["id"])
        .sort_values("id")
    )
    arr = [{"id": str(r.id), "name": r.name} for r in out.itertuples()]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(arr, f, ensure_ascii=False, indent=2)


def update_issuer_json_from_df(
    path: str, df_src: pd.DataFrame
) -> Tuple[int, int, int]:
    """
    Append new issuers (by id) from df_src (expects columns 'id','name').
    Does not overwrite names for existing IDs.
    Returns (existing_count, added_count, final_count).
    """
    df_src_norm = (
        df_src.rename(columns={"id": "id", "name": "name"})
        .copy()[["id", "name"]]
        .dropna(subset=["id"])
    )
    df_src_norm["id"] = df_src_norm["id"].astype(str).str.strip()

    df_existing = load_issuer_json_df(path)
    existing_ids = set(df_existing["id"]) if not df_existing.empty else set()

    df_new = df_src_norm[~df_src_norm["id"].isin(existing_ids)]
    added = len(df_new)

    if df_existing.empty:
        df_final = df_src_norm.drop_duplicates(subset=["id"])
    else:
        df_final = pd.concat(
            [df_existing, df_new], ignore_index=True
        ).drop_duplicates(subset=["id"])

    save_issuer_json_df(df_final, path)
    return (len(existing_ids), added, len(df_final))


def issuer_json_bytes(path: str) -> bytes:
    if not path or not os.path.exists(path):
        return b"[]"
    with open(path, "rb") as f:
        return f.read()


# -------------------------------
# PMV issuer extraction from raw
# -------------------------------


def extract_issuer_from_raw(raw_str: str) -> Optional[str]:
    if raw_str is None or raw_str == "":
        return None
    try:
        obj = json.loads(raw_str)
    except Exception:
        return None
    try:
        sec = obj.get("security") or {}
        issuer = sec.get("issuer")
        if issuer is None:
            return None
        return str(issuer).strip()
    except Exception:
        return None


def map_issuer_name_from_json(
    df_pmv: pd.DataFrame, issuer_json_path: str
) -> pd.DataFrame:
    if df_pmv.empty or "raw" not in df_pmv.columns:
        return df_pmv

    df = df_pmv.copy()

    stype = df.get("security_type", pd.Series(index=df.index, dtype="object"))
    mask_cb = stype.astype(str).str.startswith("CORPORATE_BONDS_", na=False)

    df["issuer_from_raw"] = None
    df.loc[mask_cb, "issuer_from_raw"] = df.loc[mask_cb, "raw"].apply(
        extract_issuer_from_raw
    )

    df_ij = load_issuer_json_df(issuer_json_path)
    if df_ij.empty:
        df["issuer_name_from_json"] = None
        return df

    id_to_name = (
        df_ij.dropna(subset=["id"])
        .drop_duplicates(subset=["id"])
        .set_index("id")["name"]
    )
    df["issuer_name_from_json"] = df["issuer_from_raw"].map(id_to_name)

    return df


# -------------------------------
# CRA/CRI parsing (from security.name)
# -------------------------------


def _extract_security_name_from_raw(raw_str: Optional[str]) -> Optional[str]:
    if raw_str is None or raw_str == "":
        return None
    try:
        obj = json.loads(raw_str)
        return (obj.get("security") or {}).get("name")
    except Exception:
        return None


def _extract_broker_name_from_raw(raw_str: Optional[str]) -> Optional[str]:
    if raw_str is None or raw_str == "":
        return None
    try:
        obj = json.loads(raw_str)
        return (obj.get("broker") or {}).get("name")
    except Exception:
        return None


def _extract_security_description_from_raw(
    raw_str: Optional[str],
) -> Optional[str]:
    if raw_str is None or raw_str == "":
        return None
    try:
        obj = json.loads(raw_str)
        return (obj.get("security") or {}).get("description")
    except Exception:
        return None


_MONTH_MAP = {
    # Portuguese
    "JAN": 1,
    "FEV": 2,
    "MAR": 3,
    "ABR": 4,
    "MAI": 5,
    "JUN": 6,
    "JUL": 7,
    "AGO": 8,
    "SET": 9,
    "OUT": 10,
    "NOV": 11,
    "DEZ": 12,
    # English
    "FEB": 2,
    "APR": 4,
    "MAY": 5,
    "AUG": 8,
    "SEP": 9,
    "OCT": 10,
    "DEC": 12,
}


def _parse_maturity_token_to_date(tok: Optional[str]) -> Optional[str]:
    """
    Convert e.g. 'NOV/2029' or '11/2029' to '2029-11-01'.
    Returns None if format is not recognized.
    """
    if not tok:
        return None
    s = str(tok).strip().upper()
    m = re.match(r"^(\d{1,2})/(\d{4})$", s)
    if m:
        month = int(m.group(1))
        year = int(m.group(2))
        if 1 <= month <= 12:
            return f"{year:04d}-{month:02d}-01"
        return None
    m = re.match(r"^([A-ZÀ-Ý]{3})/(\d{4})$", s)
    if m:
        abbr = m.group(1)[:3]
        year = int(m.group(2))
        month = _MONTH_MAP.get(abbr)
        if month:
            return f"{year:04d}-{month:02d}-01"
    return None


def _parse_ddmmyyyy_to_date(tok: Optional[str]) -> Optional[str]:
    """
    Convert 'DD/MM/YYYY' to 'YYYY-MM-DD'.
    """
    if not tok:
        return None
    s = str(tok).strip()
    m = re.match(r"^(\d{1,2})/(\d{1,2})/(\d{4})$", s)
    if not m:
        return None
    dd = int(m.group(1))
    mm = int(m.group(2))
    yyyy = int(m.group(3))
    try:
        _ = date(yyyy, mm, dd)
    except Exception:
        return None
    return f"{yyyy:04d}-{mm:02d}-{dd:02d}"


def _last_ddmmyyyy_from_text(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    matches = re.findall(r"(\d{1,2}/\d{1,2}/\d{4})", str(text))
    return matches[-1] if matches else None


def _parse_cra_cri_name(sec_name: Optional[str]) -> Dict[str, Optional[str]]:
    out = {
        "parsed_bond_type": None,
        "parsed_company_name": None,
        "parsed_maturity": None,
        "parsed_maturity_date": None,
        "parsed_cetip_code": None,
    }
    if not sec_name:
        return out

    tokens = re.findall(r"\S+", str(sec_name).strip())
    if len(tokens) < 3:
        return out

    bond_type = tokens[0] if tokens[0] in {"CRA", "CRI"} else None
    cetip_code = tokens[-1]

    mat_idx = None
    for i in range(len(tokens) - 2, 0, -1):
        if "/" in tokens[i]:
            mat_idx = i
            break
    if mat_idx is None and len(tokens) >= 3:
        mat_idx = len(tokens) - 2

    maturity_tok = tokens[mat_idx] if mat_idx is not None else None
    company_tokens = tokens[1:mat_idx] if mat_idx is not None else tokens[1:-1]
    company_name = " ".join(company_tokens).strip() if company_tokens else None

    out["parsed_bond_type"] = bond_type
    out["parsed_company_name"] = company_name or None
    out["parsed_maturity"] = maturity_tok or None
    out["parsed_maturity_date"] = _parse_maturity_token_to_date(maturity_tok)
    out["parsed_cetip_code"] = cetip_code or None
    return out


def add_cra_cri_parsed_cols(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df2 = df.copy()

    mask = df2["security_type"].isin(
        ["CORPORATE_BONDS_CRA", "CORPORATE_BONDS_CRI"]
    )
    if not mask.any():
        return df2

    sec_names = df2.loc[mask, "security_name"].astype("object")
    fallback_names = df2.loc[mask, "raw"].apply(_extract_security_name_from_raw)
    sec_names = sec_names.where(
        sec_names.notna() & (sec_names.str.strip() != ""), fallback_names
    )

    parsed = sec_names.apply(_parse_cra_cri_name)
    parsed_df = pd.DataFrame(list(parsed), index=sec_names.index)

    for col in parsed_df.columns:
        df2.loc[mask, col] = parsed_df[col]

    return df2


# -------------------------------
# CDB/LCI/LCA/LIG/LCD/LC parsing
# -------------------------------


def _parse_indexer_from_sec_name(sec_name: Optional[str]) -> Optional[str]:
    if not sec_name:
        return None
    s = str(sec_name).upper().strip()

    if re.search(r"\bPOS[_ ]?CDI\b", s):
        return "POS CDI"
    if re.search(r"\bPOS[_ ]?IPCA\b", s):
        return "POS IPCA"
    if re.search(r"\bPRE\b", s):
        return "PRE"

    parts = re.split(r"[_\s]+", s)
    if len(parts) >= 2:
        p1 = parts[1]
        if p1 == "PRE":
            return "PRE"
        if p1 == "POS" and len(parts) >= 3:
            if parts[2] in {"CDI", "IPCA"}:
                return f"POS {parts[2]}"
            return "POS"
    return None


def add_cdb_lci_lca_parsed_cols(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df2 = df.copy()
    stype = df2.get("security_type", pd.Series(index=df2.index, dtype="object"))
    mask = stype.isin(
        [
            "CORPORATE_BONDS_CDB",
            "CORPORATE_BONDS_LCA",
            "CORPORATE_BONDS_LCI",
            "CORPORATE_BONDS_LIG",
            "CORPORATE_BONDS_LCD",
            "CORPORATE_BONDS_LC",
        ]
    )
    if not mask.any():
        return df2

    type_map = {
        "CORPORATE_BONDS_CDB": "CDB",
        "CORPORATE_BONDS_LCA": "LCA",
        "CORPORATE_BONDS_LCI": "LCI",
        "CORPORATE_BONDS_LIG": "LIG",
        "CORPORATE_BONDS_LCD": "LCD",
        "CORPORATE_BONDS_LC": "LC",
    }
    df2.loc[mask, "parsed_bond_type"] = stype.map(type_map)

    broker_names = df2.loc[mask, "raw"].apply(_extract_broker_name_from_raw)
    df2.loc[mask, "parsed_company_name"] = broker_names

    descs = df2.loc[mask, "raw"].apply(_extract_security_description_from_raw)
    maturity_tokens = descs.apply(_last_ddmmyyyy_from_text)
    df2.loc[mask, "parsed_maturity"] = maturity_tokens
    df2.loc[mask, "parsed_maturity_date"] = maturity_tokens.apply(
        _parse_ddmmyyyy_to_date
    )

    sec_names = df2.loc[mask, "security_name"].astype("object")
    fallback_names = df2.loc[mask, "raw"].apply(_extract_security_name_from_raw)
    sec_names = sec_names.where(
        sec_names.notna() & (sec_names.str.strip() != ""), fallback_names
    )
    df2.loc[mask, "indexer"] = sec_names.apply(_parse_indexer_from_sec_name)

    df2.loc[mask, "parsed_cetip_code"] = None

    return df2


# -------------------------------
# Treasury (TESOURO) parsing
# -------------------------------


def _parse_treasury_sec_name(sec_name: Optional[str]) -> Dict[str, Optional[str]]:
    out = {
        "parsed_bond_type": "TÍTULO PÚBLICO",
        "parsed_company_name": "TESOURO NACIONAL",
        "issuer_name_from_json": "TESOURO NACIONAL",
        "parsed_maturity": None,
        "parsed_maturity_date": None,
        "parsed_cetip_code": None,
        "indexer": None,
    }
    if not sec_name:
        return out

    s = str(sec_name).strip().upper()
    m = re.match(r"^\s*([A-Z\-]+)\s*-\s*(\d{2}/\d{2}/\d{4})\s*$", s)
    if not m:
        return out

    code_raw = m.group(1)
    date_str = m.group(2)

    code = re.sub(r"[^A-Z]", "", code_raw)

    indexer_map = {
        "LFT": "POS CDI",
        "LTN": "PRE",
        "NTNB": "POS IPCA",
    }
    out["parsed_maturity"] = date_str
    out["parsed_maturity_date"] = _parse_ddmmyyyy_to_date(date_str)
    out["indexer"] = indexer_map.get(code)
    return out


def add_treasury_local_parsed_cols(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df2 = df.copy()

    stype = df2.get("security_type", pd.Series(index=df2.index, dtype="object"))
    mask = stype.astype(str).str.startswith("TREASURY_LOCAL_", na=False)
    if not mask.any():
        return df2

    sec_names = df2.loc[mask, "security_name"].astype("object")
    fallback_names = df2.loc[mask, "raw"].apply(_extract_security_name_from_raw)
    sec_names = sec_names.where(
        sec_names.notna() & (sec_names.str.strip() != ""), fallback_names
    )

    parsed = sec_names.apply(_parse_treasury_sec_name)
    parsed_df = pd.DataFrame(list(parsed), index=sec_names.index)

    for col in parsed_df.columns:
        df2.loc[mask, col] = parsed_df[col]

    df2.loc[mask, "parsed_cetip_code"] = None

    return df2


# -------------------------------
# Debentures (B3 TSV export) helpers
# -------------------------------

DEBENTURES_URL = (
    "https://www.debentures.com.br/exploreosnd/consultaadados/"
    "emissoesdedebentures/caracteristicas_e.asp"
    "?tip_deb=publicas&op_exc=Nada"
)

# Canonical debentures columns to keep (names preserved as requested)
DEB_CANON_COLS = [
    "Codigo do Ativo",
    "Empresa        ",
    "ISIN",
    "Registro CVM da Emissao",
    "Data de Emissao",
    " Data de Vencimento",
    "indice",
]


def _strip_accents(s: str) -> str:
    return "".join(
        ch
        for ch in unicodedata.normalize("NFD", s or "")
        if unicodedata.category(ch) != "Mn"
    )


def _norm_colname(s: str) -> str:
    s2 = (s or "").replace("\xa0", " ")
    s2 = _strip_accents(s2).lower()
    s2 = re.sub(r"\s+", " ", s2).strip()
    return s2


def deb_build_headers(referer: Optional[str] = None) -> Dict[str, str]:
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
def deb_download_text(url: str) -> Tuple[str, Dict[str, str], str]:
    with requests.Session() as s:
        s.headers.update(deb_build_headers(referer=url))
        r = s.get(url, timeout=60)
        r.raise_for_status()
        enc = r.encoding or r.apparent_encoding or "latin-1"
        text = r.content.decode(enc, errors="replace")
        return text, dict(r.headers), r.url


def deb_find_header_index(lines: List[str]) -> int:
    for i, line in enumerate(lines):
        if line.count("\t") >= 10 and line.lower().startswith("codigo do ativo"):
            return i
    for i, line in enumerate(lines):
        if line.count("\t") >= 10:
            return i
    raise ValueError("Header line not found (no tab-delimited header detected).")


def deb_parse_tsv_text_to_df(text: str) -> pd.DataFrame:
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = t.split("\n")

    header_idx = deb_find_header_index(lines)
    table_text = "\n".join(lines[header_idx:])

    df = pd.read_csv(
        io.StringIO(table_text),
        sep="\t",
        engine="python",
        dtype=str,
        on_bad_lines="skip",
        quoting=3,  # csv.QUOTE_NONE
    )

    # Preserve original spacing in column names (only replace NBSP by space)
    df.columns = [c.replace("\xa0", " ") for c in df.columns]

    # Drop fully empty rows
    df = df.dropna(how="all")

    # Strip whitespace in cell values (keep header spacing intact)
    for c in df.columns:
        if pd.api.types.is_string_dtype(df[c]):
            df[c] = df[c].astype(str).str.strip()

    return df


def deb_select_required_cols(df_full: pd.DataFrame) -> pd.DataFrame:
    if df_full.empty:
        return pd.DataFrame(columns=DEB_CANON_COLS)

    # Build map from normalized header -> actual header
    norm_to_actual: Dict[str, str] = {}
    for col in df_full.columns:
        n = _norm_colname(col)
        if n not in norm_to_actual:
            norm_to_actual[n] = col

    # Normalized needles for required canonical names
    needles = {
        "Codigo do Ativo": _norm_colname("Codigo do Ativo"),
        "Empresa        ": _norm_colname("Empresa"),
        "ISIN": _norm_colname("ISIN"),
        "Registro CVM da Emissao": _norm_colname("Registro CVM da Emissao"),
        "Data de Emissao": _norm_colname("Data de Emissao"),
        " Data de Vencimento": _norm_colname("Data de Vencimento"),
        "indice": _norm_colname("indice"),
    }

    actual_map: Dict[str, Optional[str]] = {}
    for canon, needle in needles.items():
        actual_map[canon] = norm_to_actual.get(needle)

    cols_found = [c for c in DEB_CANON_COLS if actual_map.get(c)]
    df_out = pd.DataFrame()
    if cols_found:
        df_out = df_full[[actual_map[c] for c in cols_found]].copy()
        rename_map = {actual_map[c]: c for c in cols_found}
        df_out = df_out.rename(columns=rename_map)

    for c in DEB_CANON_COLS:
        if c not in df_out.columns:
            df_out[c] = None

    df_out = df_out[DEB_CANON_COLS]
    return df_out


def _norm_deb_code(s: str) -> str:
    # Normalize to uppercase alphanumeric (remove spaces, hyphens, etc.)
    return re.sub(r"[^A-Z0-9]", "", str(s or "").upper())


def _norm_isin(s: str) -> str:
    return re.sub(r"\s+", "", str(s or "").upper())


def _find_deb_code_in_sec_name(
    sec_name: Optional[str], code_set: set
) -> Optional[str]:
    if not sec_name:
        return None
    s = _strip_accents(str(sec_name)).upper()
    # Tokenize by non-alphanumerics
    tokens = re.split(r"[^A-Z0-9]+", s)
    for tok in tokens:
        if tok and tok in code_set:
            return tok
    # Fallback: try full normalized string
    full = _norm_deb_code(sec_name)
    if full in code_set:
        return full
    return None


def add_debentures_parsed_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    For CORPORATE_BONDS_DEBENTURES:
      - match PMV security_name to debentures['Codigo do Ativo'] (robust)
      - fallback match by ISIN
      - parsed_company_name = debentures['Empresa        ']
      - parsed_maturity = debentures[' Data de Vencimento'] (DD/MM/YYYY)
      - parsed_maturity_date = YYYY-MM-DD
      - indexer = debentures['indice']
    """
    if df.empty:
        return df

    # Always set parsed_bond_type for singular debenture type
    df2 = df.copy()
    stype = df2.get("security_type", pd.Series(index=df2.index, dtype="object"))
    stype_u = stype.astype(str).str.upper()
    mask_exact_type = stype_u.eq("CORPORATE_BONDS_DEBENTURE")
    if mask_exact_type.any():
        df2.loc[mask_exact_type, "parsed_bond_type"] = "DEBENTURE"

    # If we don't have debenture reference data, return early with any
    # parsed_bond_type we may have set above.
    deb = st.session_state.get("df_debentures")
    if deb is None or deb.empty:
        return df2

    required = [
        "Codigo do Ativo",
        "Empresa        ",
        " Data de Vencimento",
        "indice",
        "ISIN",
    ]
    if any(c not in deb.columns for c in required):
        return df2

    stype_u = stype.astype(str).str.upper()
    mask = stype_u.eq("CORPORATE_BONDS_DEBENTURES") | stype_u.str.contains(
        "DEBENTUR", na=False
    )
    if not mask.any():
        return df2

    deb_small = deb[required].copy()

    # Code-key dicts
    deb_small["__code_key"] = deb_small["Codigo do Ativo"].apply(_norm_deb_code)
    deb_code = (
        deb_small.dropna(subset=["__code_key"])
        .drop_duplicates(subset=["__code_key"], keep="first")
        .set_index("__code_key")
    )
    code_set = set(deb_code.index)
    code_to_empresa = deb_code["Empresa        "].to_dict()
    code_to_venc = deb_code[" Data de Vencimento"].to_dict()
    code_to_indice = deb_code["indice"].to_dict()

    # ISIN-key dicts
    deb_small["__isin_key"] = deb_small["ISIN"].apply(_norm_isin)
    deb_isin = (
        deb_small.dropna(subset=["__isin_key"])
        .drop_duplicates(subset=["__isin_key"], keep="first")
        .set_index("__isin_key")
    )
    isin_to_empresa = deb_isin["Empresa        "].to_dict()
    isin_to_venc = deb_isin[" Data de Vencimento"].to_dict()
    isin_to_indice = deb_isin["indice"].to_dict()

    # PMV side: get security_name with fallback from raw
    sec_names = df2.loc[mask, "security_name"].astype("object")
    fallback_names = df2.loc[mask, "raw"].apply(_extract_security_name_from_raw)
    sec_names = sec_names.where(
        sec_names.notna() & (sec_names.str.strip() != ""), fallback_names
    )

    # Try to find a valid code in the security name
    sec_code_keys = sec_names.apply(
        lambda s: _find_deb_code_in_sec_name(s, code_set)
    )

    # ISIN series normalized
    isin_series = df2.loc[mask, "isin"].apply(_norm_isin)

    # Map by code first, fallback to ISIN
    comp_by_code = sec_code_keys.map(code_to_empresa)
    venc_by_code = sec_code_keys.map(code_to_venc)
    idxr_by_code = sec_code_keys.map(code_to_indice)

    comp_by_isin = isin_series.map(isin_to_empresa)
    venc_by_isin = isin_series.map(isin_to_venc)
    idxr_by_isin = isin_series.map(isin_to_indice)

    comp_final = comp_by_code.combine_first(comp_by_isin)
    venc_final = venc_by_code.combine_first(venc_by_isin)
    idxr_final = idxr_by_code.combine_first(idxr_by_isin)

    # Assign back
    df2.loc[comp_final.index, "parsed_company_name"] = comp_final
    df2.loc[venc_final.index, "parsed_maturity"] = venc_final
    df2.loc[venc_final.index, "parsed_maturity_date"] = venc_final.apply(
        _parse_ddmmyyyy_to_date
    )
    df2.loc[idxr_final.index, "indexer"] = idxr_final

    return df2


# -------------------------------
# Indexer normalization (post-processing)
# -------------------------------


def _normalize_indexer_value(val: Optional[str]) -> Optional[str]:
    """
    Normalize indexer values AFTER PMV df is ready:
      - POS CDI -> CDI
      - POS IPCA -> IPCA
      - IPC-A -> IPCA
      - IGP-M -> IPCA
      - IGMP -> IPCA
      - PRE -> PRÉ
      - PREFIXADO -> PRÉ
      - PRÉFIXADO -> PRÉ
    """
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    s = str(val).strip()
    if s == "":
        return None

    # Create normalization keys
    s_no_acc = _strip_accents(s).upper()
    s_nospace = re.sub(r"\s+", "", s_no_acc)
    s_clean = re.sub(r"[^A-Z0-9]", "", s_nospace)  # remove hyphens, etc.

    # Mappings by cleaned token
    if s_clean in {"DI"}:
        return "CDI"
    if s_clean in {"POSCDI"}:
        return "CDI"
    if s_clean in {"POSIPCA"}:
        return "IPCA"
    if s_clean in {"IPCA", "IPCA"}:
        return "IPCA"
    if s_clean in {"IPCA", "IPCA"}:
        return "IPCA"
    if s_clean in {"IPCA", "IPCA"}:
        return "IPCA"
    # IPC-A -> IPCA (cleaned "IPCA" already)
    if s_clean in {"IPCA"}:
        return "IPCA"
    # IGP-M/IGPM or IGMP -> IPCA (as per instruction)
    if s_clean in {"IGPM", "IGPM", "IGMP"}:
        return "IPCA"
    # PRE + PREFIXADO variants -> PRÉ
    if s_clean in {"PRE", "PREFIXADO", "PREFIXADO"}:
        return "PRÉ"

    # If input already "PRÉ", keep it
    if s.upper() == "PRÉ":
        return "PRÉ"

    # Default: return original trimmed string
    return s


# -------------------------------
# Debentures persistence (SQLite)
# -------------------------------


def deb_save_df_to_db(
    df: pd.DataFrame, db_path: str, table_name: str = "debentures"
) -> None:
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        df.to_sql(table_name, conn, if_exists="replace", index=False)


def deb_load_df_from_db(
    db_path: str, table_name: str = "debentures"
) -> pd.DataFrame:
    if not db_path or not os.path.exists(db_path):
        return pd.DataFrame(columns=DEB_CANON_COLS)
    try:
        with sqlite3.connect(db_path) as conn:
            df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    except Exception:
        return pd.DataFrame(columns=DEB_CANON_COLS)
    # Ensure canonical columns present and ordered
    for c in DEB_CANON_COLS:
        if c not in df.columns:
            df[c] = None
    return df[DEB_CANON_COLS]


# -------------------------------
# Streamlit UI (raw + robust fetch)
# -------------------------------


st.set_page_config(page_title="Gorila Loader (Raw+Issuers)", layout="wide")
st.title("Gorila CORE → Raw Data (with issuer JSON + mapping)")

api_key = get_api_key()
with st.sidebar:
    st.subheader("Configuration")
    local_key_input = st.text_input(
        "API key (if not using st.secrets / env)", type="password", value=""
    )
    if not api_key and local_key_input:
        api_key = local_key_input

    base_url = st.text_input("Base URL", value="https://core.gorila.com.br")

    issuer_json_path = st.text_input("Issuer JSON path", value="issuers.json")

if not api_key:
    st.warning(
        "Provide your Gorila API key via st.secrets['GORILA_API_KEY'], "
        "the GORILA_API_KEY env var, or the sidebar field."
    )
    st.stop()

client = GorilaClient(api_key=api_key, base_url=base_url)

# -------------------------------
# Reference date (outside sidebar)
# -------------------------------

st.header("Reference date")
ref_col1, ref_col2 = st.columns(2)

with ref_col1:
    use_ref_date = st.checkbox("Use reference date", value=True)

with ref_col2:
    default_ref = default_reference_date()
    ref_date_opt = st.date_input("Reference date", value=default_ref)

ref_date_str = ref_date_opt.isoformat() if use_ref_date else None

# -------------------------------
# Portfolios
# -------------------------------

st.header("Portfolios")
if st.button("Fetch portfolios"):
    try:
        rows = [flatten_portfolio(p) for p in client.list_portfolios()]
        df_port = pd.DataFrame(rows)
        st.session_state["df_portfolios"] = df_port
        st.dataframe(df_port, use_container_width=True)
    except Exception as e:
        st.error(f"Error fetching portfolios: {e}")

if st.session_state.get("df_portfolios") is not None:
    st.download_button(
        label="Download portfolios (CSV)",
        data=df_to_csv_bytes(st.session_state["df_portfolios"]),
        file_name="portfolios.csv",
        mime="text/csv;charset=utf-8",
    )

# Issuers
st.header("Issuers")
cols_iss = st.columns(2)
with cols_iss[0]:
    if st.button("Fetch issuers"):
        try:
            rows = [flatten_issuer(i) for i in client.list_issuers()]
            df_iss = pd.DataFrame(rows)
            st.session_state["df_issuers"] = df_iss
            st.dataframe(df_iss, use_container_width=True)
        except Exception as e:
            st.error(f"Error fetching issuers: {e}")

with cols_iss[1]:
    if st.button("Update issuer JSON from current issuer table"):
        try:
            if st.session_state.get("df_issuers") is None:
                rows = [flatten_issuer(i) for i in client.list_issuers()]
                st.session_state["df_issuers"] = pd.DataFrame(rows)
            df_iss_curr = st.session_state["df_issuers"]

            df_min = df_iss_curr[["id", "name"]].copy()
            exist_cnt, added_cnt, final_cnt = update_issuer_json_from_df(
                issuer_json_path, df_min
            )
            st.success(
                f"Issuer JSON updated. Existing={exist_cnt}, "
                f"added={added_cnt}, total={final_cnt}"
            )
            df_prev = load_issuer_json_df(issuer_json_path)
            st.dataframe(df_prev, use_container_width=True, height=240)
        except Exception as e:
            st.error(f"Error updating issuer JSON: {e}")

if st.session_state.get("df_issuers") is not None:
    st.download_button(
        label="Download issuers (CSV)",
        data=df_to_csv_bytes(st.session_state["df_issuers"]),
        file_name="issuers.csv",
        mime="text/csv;charset=utf-8",
    )

st.download_button(
    label="Download issuer JSON file",
    data=issuer_json_bytes(issuer_json_path),
    file_name=os.path.basename(issuer_json_path) or "issuers.json",
    mime="application/json",
)

# -------------------------------
# Debentures (B3 TSV export) BEFORE PMV
# -------------------------------

st.header("Debentures (B3 TSV export)")
st.caption("Baixa o TSV público e converte em DataFrame (colunas reduzidas).")
deb_url = st.text_input(
    "URL (padrão B3; pode manter)", value=DEBENTURES_URL, disabled=True
)

deb_db_path = st.text_input("Salvar/Carregar de (DB local)", value="df_debentures.db")
st.session_state["deb_db_path"] = deb_db_path
cols_deb = st.columns(2)

with cols_deb[0]:
    if st.button("Carregar do DB local"):
        try:
            df_deb = deb_load_df_from_db(deb_db_path)
            if df_deb.empty:
                st.warning("DB local vazio ou ausente.")
            else:
                st.session_state["df_debentures"] = df_deb
                st.success(
                    f"Debêntures carregado do DB: "
                    f"{df_deb.shape[0]} linhas × {df_deb.shape[1]} cols"
                )
                st.dataframe(df_deb, use_container_width=True)
        except Exception as e:
            st.error(f"Erro ao carregar DB local: {e}")

with cols_deb[1]:
    if st.button("Baixar TSV e sobrescrever DB", type="primary"):
        try:
            with st.spinner("Baixando conteúdo (TSV)..."):
                text, headers, final_url = deb_download_text(deb_url)

            st.info(
                f"Final URL: {final_url} | "
                f"Content-Type: {headers.get('Content-Type','unknown')}"
            )

            with st.expander("Prévia da resposta (primeiros ~2000 chars)"):
                st.text(text[:2000])

            with st.spinner("Convertendo para DataFrame..."):
                df_deb_full = deb_parse_tsv_text_to_df(text)
                df_deb = deb_select_required_cols(df_deb_full)

            st.session_state["df_debentures"] = df_deb
            # Save/overwrite local DB
            try:
                deb_save_df_to_db(df_deb, deb_db_path)
                st.info(f"DB salvo em: {deb_db_path} (tabela 'debentures').")
            except Exception as e:
                st.warning(f"Não foi possível salvar o DB local: {e}")

            st.success(
                f"Debêntures carregado: {df_deb.shape[0]} linhas × "
                f"{df_deb.shape[1]} cols"
            )
            st.dataframe(df_deb, use_container_width=True)

        except Exception as e:
            st.error(f"Erro ao carregar Debêntures: {e}")

if st.session_state.get("df_debentures") is not None:
    st.download_button(
        label="Download Debêntures (CSV)",
        data=df_to_csv_bytes(st.session_state["df_debentures"]),
        file_name="debentures.csv",
        mime="text/csv;charset=utf-8",
    )

# -------------------------------
# Positions
# -------------------------------

st.header("Positions")
use_all_portfolios_for_positions = st.toggle(
    "Use all portfolios for positions", value=False
)

selected_ids: List[str] = []
portfolios_loaded = st.session_state.get("df_portfolios") is not None

if use_all_portfolios_for_positions:
    if portfolios_loaded:
        selected_ids = (
            st.session_state["df_portfolios"]["id"].dropna().astype(str).tolist()
        )
        st.caption(f"Selected all portfolios: {len(selected_ids)}")
    else:
        st.info("Load portfolios first to use the 'all portfolios' option.")
else:
    if portfolios_loaded:
        df_port = st.session_state["df_portfolios"]
        st.caption("Select portfolios:")
        options = (df_port["id"] + " | " + df_port["name"].fillna("")).tolist()
        selected = st.multiselect("Portfolios", options=options)
        selected_ids = [s.split(" | ", 1)[0] for s in selected]
    else:
        st.info("Load portfolios to select via UI, or paste IDs below.")

    manual_ids = st.text_input("Or paste portfolio IDs (comma-separated)", value="")
    if manual_ids.strip():
        selected_ids += [s.strip() for s in manual_ids.split(",") if s.strip()]

if st.button("Fetch positions for selected portfolios"):
    if not selected_ids:
        st.warning("No portfolio IDs selected.")
    else:
        errors: List[Tuple[str, str]] = []
        dfs: List[pd.DataFrame] = []
        progress = st.progress(0.0)
        total = len(selected_ids)

        for idx, pid in enumerate(selected_ids, start=1):
            try:
                rows = [
                    flatten_position(pid, p)
                    for p in client.list_positions(pid, ref_date_str)
                ]
                dfs.append(pd.DataFrame(rows))
            except Exception as e:
                errors.append((pid, str(e)))
            finally:
                progress.progress(min(idx / total, 1.0))

        df_pos = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
        st.session_state["df_positions"] = df_pos
        st.dataframe(df_pos, use_container_width=True)

        if errors:
            st.warning(
                "Some portfolios failed while fetching positions:\n"
                + "\n".join([f"- {pid}: {msg}" for pid, msg in errors])
            )

if st.session_state.get("df_positions") is not None:
    st.download_button(
        label="Download positions (CSV)",
        data=df_to_csv_bytes(st.session_state["df_positions"]),
        file_name="positions.csv",
        mime="text/csv;charset=utf-8",
    )

# -------------------------------
# Position Market Values
# -------------------------------

st.header("Position Market Values")
use_all_portfolios_for_pmv = st.toggle(
    "Use all portfolios for market values", value=False
)

pmv_selected_ids: List[str] = []
if use_all_portfolios_for_pmv:
    if portfolios_loaded:
        pmv_selected_ids = (
            st.session_state["df_portfolios"]["id"].dropna().astype(str).tolist()
        )
        st.caption(f"Selected all portfolios: {len(pmv_selected_ids)}")
    else:
        st.info("Load portfolios first to use the 'all portfolios' option.")
else:
    if portfolios_loaded:
        df_port = st.session_state["df_portfolios"]
        st.caption("Select portfolios (PMV):")
        options_pmv = (df_port["id"] + " | " + df_port["name"].fillna("")).tolist()
        selected_pmv = st.multiselect(
            "Portfolios (market values)", options=options_pmv, key="pmv_ms"
        )
        pmv_selected_ids = [s.split(" | ", 1)[0] for s in selected_pmv]

    manual_pmv_ids = st.text_input(
        "Or paste portfolio IDs for PMV (comma-separated)", value=""
    )
    if manual_pmv_ids.strip():
        pmv_selected_ids += [
            s.strip() for s in manual_pmv_ids.split(",") if s.strip()
        ]

if st.button("Fetch position market values"):
    if not pmv_selected_ids:
        st.warning("No portfolio IDs selected for market values.")
    else:
        # Ensure debentures df loaded (prefer local DB; fallback to download)
        if st.session_state.get("df_debentures") is None:
            deb_db_path_ss = st.session_state.get("deb_db_path", "df_debentures.db")
            df_deb_local = deb_load_df_from_db(deb_db_path_ss)
            if not df_deb_local.empty:
                st.session_state["df_debentures"] = df_deb_local
            else:
                try:
                    text, _, _ = deb_download_text(DEBENTURES_URL)
                    df_deb_full = deb_parse_tsv_text_to_df(text)
                    df_deb = deb_select_required_cols(df_deb_full)
                    st.session_state["df_debentures"] = df_deb
                    try:
                        deb_save_df_to_db(df_deb, deb_db_path_ss)
                    except Exception:
                        pass
                except Exception:
                    pass

        errors: List[Tuple[str, str]] = []
        dfs: List[pd.DataFrame] = []
        progress = st.progress(0.0)
        total = len(pmv_selected_ids)

        for idx, pid in enumerate(pmv_selected_ids, start=1):
            try:
                rows = [
                    flatten_position_market_value(pid, it)
                    for it in client.list_position_market_values(
                        pid, ref_date_str
                    )
                ]
                dfs.append(pd.DataFrame(rows))
            except Exception as e:
                errors.append((pid, str(e)))
            finally:
                progress.progress(min(idx / total, 1.0))

        df_pmv = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

        if not df_pmv.empty:
            # Map issuers if available
            df_pmv = map_issuer_name_from_json(df_pmv, issuer_json_path)
            # Debentures parsing (uses df_debentures loaded earlier)
            df_pmv = add_debentures_parsed_cols(df_pmv)
            # Other instrument families
            df_pmv = add_cra_cri_parsed_cols(df_pmv)
            df_pmv = add_cdb_lci_lca_parsed_cols(df_pmv)
            df_pmv = add_treasury_local_parsed_cols(df_pmv)

            # Normalize indexer values ONLY AFTER df_pmv is ready/enriched
            if "indexer" not in df_pmv.columns:
                df_pmv["indexer"] = None
            df_pmv["indexer"] = df_pmv["indexer"].apply(_normalize_indexer_value)

        st.session_state["df_position_market_values"] = df_pmv
        st.dataframe(df_pmv, use_container_width=True)

        if errors:
            st.warning(
                "Some portfolios failed while fetching PMV:\n"
                + "\n".join([f"- {pid}: {msg}" for pid, msg in errors])
            )

if st.session_state.get("df_position_market_values") is not None:
    st.download_button(
        label="Download position market values (CSV)",
        data=df_to_csv_bytes(st.session_state["df_position_market_values"]),
        file_name="position_market_values.csv",
        mime="text/csv;charset=utf-8",
    )