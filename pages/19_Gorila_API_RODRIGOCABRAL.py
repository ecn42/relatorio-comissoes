import os
import json
import time
from datetime import date, timedelta
from typing import Dict, Iterable, List, Optional, Tuple
import re
import io
import unicodedata
import sqlite3

import pandas as pd
import requests
import streamlit as st

# -------------------------------
# Streamlit page config
# -------------------------------
st.set_page_config(page_title="Gorila API - Rodrigo Cabral", layout="wide")

# -------------------------------
# Simple Authentication
# -------------------------------
if not st.session_state.get("authenticated", False):
    st.warning("Please enter the password on the Home page first.")
    st.write("Status: Não Autenticado")
    st.stop()

st.write("Autenticado")

# -------------------------------
# Portfolio/Broker to Conta Mapping
# -------------------------------

PORTFOLIO_BROKER_CONTA_MAP = {
('14b315dc-8298-414a-89d3-e128fd292add','34904571000172'):'2538001',
('14b315dc-8298-414a-89d3-e128fd292add','90400888000142'):'290274117',
('b974d0f2-74fc-45be-b2e5-bce4e4dcea0e','39582666000130'):'2538003',
('b974d0f2-74fc-45be-b2e5-bce4e4dcea0e','OUTROS'):'2538002',
('dc44f575-66e8-426a-862a-b3949f64a27b','90400888000142'):'10075814',
('dc44f575-66e8-426a-862a-b3949f64a27b','60701190000104'):'6093',
('8ebea19c-1806-4141-80a3-b8c31a1f957e','33264668000103'):'1020910999',
('dc44f575-66e8-426a-862a-b3949f64a27b','33264668000103'):'1020910',
('8ebea19c-1806-4141-80a3-b8c31a1f957e','32062580000138'):'719431999',
('dc44f575-66e8-426a-86-2a-b3949f64a27b','32062580000138'):'719431',
('8ebea19c-1806-4141-80a3-b8c31a1f957e','-'):'30435001',
('8ebea19c-1806-4141-80a3-b8c31a1f957e','62285390000140'):'30435002',
('8ebea19c-1806-4141-80a3-b8c31a1f957e','00360305000104'):'30435003',
('dc44f575-66e8-426a-862a-b3949f64a27b','62232889000190'):'30435004',
('8ebea19c-1806-4141-80a3-b8c31a1f957e','33336454000197'):'30435005',
('8ebea19c-1806-4141-80a3-b8c31a1f957e','57061997000107'):'30435006',
('8ebea19c-1806-4141-80a3-b8c31a1f957e','OUTROS'):'30435007',
('fbfb1464-92ef-4630-affc-608930e02f9e','06332955000122'):'30435008',
('e860e317-584f-43fe-8a0a-512a8660cf4a','34904571000172'):'76001',
('e617307e-908f-4284-8f69-6f962deedc7b','34904571000172'):'368931',
('2e41fcb9-5b28-403d-872f-213dd54f610f','CHARLES'):'75013579',
('2e41fcb9-5b28-403d-872f-213dd54f610f','34904571000172'):'424471',
('2c868362-69a0-4884-9ba3-856f4547db72','06332955000122'):'30435009',

}


# Asset Class Mapping
ASSET_CLASS_MAP = {
    "FIXED_INCOME": "Renda Fixa",
    "MULTIMARKET": "Multimercado",
    "STOCKS": "Renda Variável",
    "OFFSHORE": "Offshore",
    "CASH": "Caixa",
    "TANGIBLE": "Imobiliário",
    "CURRENCY": "Outros",
}


def add_conta_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 'Conta' column to dataframe by matching portfolio_id and broker_id.
    """
    if df.empty:
        return df
    
    df_copy = df.copy()
    
    # Create Conta column with NaN initially
    df_copy["Conta"] = None
    
    # Apply mapping for each row
    for idx, row in df_copy.iterrows():
        portfolio_id = row.get("portfolio_id")
        broker_id = row.get("broker_id")
        
        if pd.notna(portfolio_id) and pd.notna(broker_id):
            key = (str(portfolio_id), str(broker_id))
            if key in PORTFOLIO_BROKER_CONTA_MAP:
                df_copy.at[idx, "Conta"] = PORTFOLIO_BROKER_CONTA_MAP[key]
    
    return df_copy


# -------------------------------
# Config / Secrets
# -------------------------------


def get_api_key() -> Optional[str]:
    # 1. Try Streamlit secrets (nested under [apis])
    if "apis" in st.secrets and "GORILA_API_KEY" in st.secrets["apis"]:
        return st.secrets["apis"]["GORILA_API_KEY"]
    # 2. Fallback to environment variable (for local/dev)
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
                    last_err = Exception(
                        f"HTTP {r.status_code}: {r.text[:200]}"
                    )
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

        raise RuntimeError(
            f"Request failed after retries: {last_err}"
        ) from last_err

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

    def list_brokers(self) -> Iterable[Dict]:
        # According to docs, brokers are under /brokers
        return self._paginate("/brokers")

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


def flatten_broker(b: Dict) -> Dict:
    return {
        "id": b.get("id"),
        "name": b.get("name"),
        "taxId": b.get("taxId") or b.get("cnpj"),
        "raw": json.dumps(b, ensure_ascii=False),
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
    bro = item.get("broker", {}) or {}

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
        "broker_id": bro.get("id"),
        "broker_name": bro.get("name"),
        "raw": json.dumps(item, ensure_ascii=False),
    }


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


def dataframes_to_excel_bytes(sheets: Dict[str, pd.DataFrame]) -> bytes:
    """
    Build an XLSX workbook with multiple sheets.
    Ensures 'issuers' sheet has only id and name.
    """
    def _prep_df(name: str, df_in: pd.DataFrame) -> pd.DataFrame:
        if df_in is None or df_in.empty:
            return pd.DataFrame()
        df = df_in.copy()
        if name.lower() == "issuers":
            keep = [c for c in ["id", "name"] if c in df.columns]
            if keep:
                df = df[keep]
            else:
                df = pd.DataFrame(columns=["id", "name"])
        if name.lower() == "brokers":
            keepb = [c for c in ["id", "name"] if c in df.columns]
            if keepb:
                df = df[keepb]
        # Ensure string type for id/name
        for c in ["id", "name"]:
            if c in df.columns:
                df[c] = df[c].astype(str)
        return df

    buf = io.BytesIO()
    try:
        with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
            for nm, df in sheets.items():
                _df = _prep_df(nm, df)
                _df.to_excel(writer, sheet_name=nm[:31], index=False)
        return buf.getvalue()
    except Exception:
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            for nm, df in sheets.items():
                _df = _prep_df(nm, df)
                _df.to_excel(writer, sheet_name=nm[:31], index=False)
        return buf.getvalue()


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
# PMV issuer/broker extraction from raw
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


def _extract_security_name_from_raw(
    raw_str: Optional[str],
) -> Optional[str]:
    if raw_str is None or raw_str == "":
        return None
    try:
        obj = json.loads(raw_str)
        return (obj.get("security") or {}).get("name")
    except Exception:
        return None


def _extract_broker_name_from_raw(
    raw_str: Optional[str],
) -> Optional[str]:
    if raw_str is None or raw_str == "":
        return None
    try:
        obj = json.loads(raw_str)
        return (obj.get("broker") or {}).get("name")
    except Exception:
        return None


def _extract_broker_id_from_raw(raw_str: Optional[str]) -> Optional[str]:
    if raw_str is None or raw_str == "":
        return None
    try:
        obj = json.loads(raw_str)
        bid = (obj.get("broker") or {}).get("id")
        return str(bid).strip() if bid is not None else None
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


def ensure_broker_cols_from_raw(df_pmv: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure broker_id and broker_name columns exist on PMV and, where missing,
    fill them from the raw JSON.
    """
    if df_pmv is None or df_pmv.empty:
        return df_pmv
    df = df_pmv.copy()
    if "raw" not in df.columns:
        return df

    # Initialize columns if missing
    if "broker_id" not in df.columns:
        df["broker_id"] = None
    if "broker_name" not in df.columns:
        df["broker_name"] = None

    # Fill missing values from raw
    mask_id = df["broker_id"].isna() | (df["broker_id"].astype(str) == "")
    mask_nm = df["broker_name"].isna() | (df["broker_name"].astype(str) == "")
    if mask_id.any():
        df.loc[mask_id, "broker_id"] = df.loc[mask_id, "raw"].apply(
            _extract_broker_id_from_raw
        )
    if mask_nm.any():
        df.loc[mask_nm, "broker_name"] = df.loc[mask_nm, "raw"].apply(
            _extract_broker_name_from_raw
        )
    return df


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

    # Extract indexer from raw JSON for CRA/CRI
    indexer_from_raw = df2.loc[mask, "raw"].apply(
        _extract_indexer_from_raw_json
    )
    df2.loc[mask, "indexer"] = indexer_from_raw

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


def _extract_indexer_from_raw_json(
    raw_json_str: Optional[str],
) -> Optional[str]:
    """
    Extract indexer from raw JSON's CLASSIFICATION_L2 parentId field.
    Maps:
      - FIXED_INCOME_PREFIXED -> PRE
      - FIXED_INCOME_INFLATION_INDEXED -> POS IPCA
      - FIXED_INCOME_INTEREST_INDEXED -> POS CDI
    """
    if not raw_json_str:
        return None

    try:
        data = json.loads(raw_json_str)
    except (json.JSONDecodeError, TypeError):
        return None

    # Navigate to classifications list
    sec = data.get("security", {})
    if not isinstance(sec, dict):
        return None

    classifications = sec.get("classifications", [])
    if not isinstance(classifications, list):
        return None

    # Find CLASSIFICATION_L2 entry
    for c in classifications:
        if not isinstance(c, dict):
            continue
        if c.get("level") == "CLASSIFICATION_L2":
            parent_id = c.get("parentId")
            if parent_id == "FIXED_INCOME_PREFIXED":
                return "PRE"
            elif parent_id == "FIXED_INCOME_INFLATION_INDEXED":
                return "POS IPCA"
            elif parent_id == "FIXED_INCOME_INTEREST_INDEXED":
                return "POS CDI"

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

    descs = df2.loc[mask, "raw"].apply(
        _extract_security_description_from_raw
    )
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
    # Try to extract indexer from security name first
    indexer_from_name = sec_names.apply(_parse_indexer_from_sec_name)
    # For entries where indexer_from_name is None, try extracting from raw
    indexer_from_raw = df2.loc[mask, "raw"].apply(
        _extract_indexer_from_raw_json
    )
    # Combine: use name extraction first, fallback to raw JSON extraction
    df2.loc[mask, "indexer"] = indexer_from_name.where(
        indexer_from_name.notna(), indexer_from_raw
    )

    df2.loc[mask, "parsed_cetip_code"] = None

    return df2


# -------------------------------
# Treasury (TESOURO) parsing
# -------------------------------


def _parse_treasury_sec_name(
    sec_name: Optional[str],
) -> Dict[str, Optional[str]]:
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
        if line.count("\t") >= 10 and line.lower().startswith(
            "codigo do ativo"
        ):
            return i
    for i, line in enumerate(lines):
        if line.count("\t") >= 10:
            return i
    raise ValueError(
        "Header line not found (no tab-delimited header detected)."
    )


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
        quoting=3,
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
        "Registro CVM da Emissao": _norm_colname(
            "Registro CVM da Emissao"
        ),
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
    tokens = re.split(r"[^A-Z0-9]+", s)
    for tok in tokens:
        if tok and tok in code_set:
            return tok
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
      - indexer = debentures['indice'], fallback from raw JSON classifications
    """
    if df.empty:
        return df

    df2 = df.copy()
    stype = df2.get("security_type", pd.Series(index=df2.index, dtype="object"))
    stype_u = stype.astype(str).str.upper()
    mask_exact_type = stype_u.eq("CORPORATE_BONDS_DEBENTURE")
    if mask_exact_type.any():
        df2.loc[mask_exact_type, "parsed_bond_type"] = "DEBENTURE"
        indexer_from_raw = df2.loc[
            mask_exact_type, "raw"
        ].apply(_extract_indexer_from_raw_json)
        df2.loc[mask_exact_type, "indexer"] = indexer_from_raw

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

    deb_small["__code_key"] = deb_small["Codigo do Ativo"].apply(
        _norm_deb_code
    )
    deb_code = (
        deb_small.dropna(subset=["__code_key"])
        .drop_duplicates(subset=["__code_key"], keep="first")
        .set_index("__code_key")
    )
    code_set = set(deb_code.index)
    code_to_empresa = deb_code["Empresa        "].to_dict()
    code_to_venc = deb_code[" Data de Vencimento"].to_dict()
    code_to_indice = deb_code["indice"].to_dict()

    deb_small["__isin_key"] = deb_small["ISIN"].apply(_norm_isin)
    deb_isin = (
        deb_small.dropna(subset=["__isin_key"])
        .drop_duplicates(subset=["__isin_key"], keep="first")
        .set_index("__isin_key")
    )
    isin_to_empresa = deb_isin["Empresa        "].to_dict()
    isin_to_venc = deb_isin[" Data de Vencimento"].to_dict()
    isin_to_indice = deb_isin["indice"].to_dict()

    sec_names = df2.loc[mask, "security_name"].astype("object")
    fallback_names = df2.loc[mask, "raw"].apply(_extract_security_name_from_raw)
    sec_names = sec_names.where(
        sec_names.notna() & (sec_names.str.strip() != ""), fallback_names
    )

    sec_code_keys = sec_names.apply(
        lambda s: _find_deb_code_in_sec_name(s, code_set)
    )

    isin_series = df2.loc[mask, "isin"].apply(_norm_isin)

    comp_by_code = sec_code_keys.map(code_to_empresa)
    venc_by_code = sec_code_keys.map(code_to_venc)
    idxr_by_code = sec_code_keys.map(code_to_indice)

    comp_by_isin = isin_series.map(isin_to_empresa)
    venc_by_isin = isin_series.map(isin_to_venc)
    idxr_by_isin = isin_series.map(isin_to_indice)

    comp_final = comp_by_code.combine_first(comp_by_isin)
    venc_final = venc_by_code.combine_first(venc_by_isin)
    idxr_final = idxr_by_code.combine_first(idxr_by_isin)

    df2.loc[comp_final.index, "parsed_company_name"] = comp_final
    df2.loc[venc_final.index, "parsed_maturity"] = venc_final
    df2.loc[venc_final.index, "parsed_maturity_date"] = venc_final.apply(
        _parse_ddmmyyyy_to_date
    )

    indexer_from_raw = df2.loc[mask, "raw"].apply(
        _extract_indexer_from_raw_json
    )
    idxr_final_with_fallback = idxr_final.combine_first(indexer_from_raw)
    df2.loc[idxr_final_with_fallback.index, "indexer"] = (
        idxr_final_with_fallback
    )

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

    s_no_acc = _strip_accents(s).upper()
    s_nospace = re.sub(r"\s+", "", s_no_acc)
    s_clean = re.sub(r"[^A-Z0-9]", "", s_nospace)

    if s_clean in {"DI", "POSCDI"}:
        return "CDI"
    if s_clean in {"POSIPCA", "IPCA"}:
        return "IPCA"
    if s_clean in {"IGPM", "IGMP"}:
        return "IPCA"
    if s_clean in {"PRE", "PREFIXADO"}:
        return "PRÉ"
    if s.upper() == "PRÉ":
        return "PRÉ"
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
    for c in DEB_CANON_COLS:
        if c not in df.columns:
            df[c] = None
    return df[DEB_CANON_COLS]


# -------------------------------
# Positions/PMV persistence (SQLite)
# -------------------------------


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (name,),
    )
    return cur.fetchone() is not None


def posdb_refdate_exists(
    db_path: str, table: str, ref_date: str
) -> bool:
    if not db_path or not os.path.exists(db_path):
        return False
    with sqlite3.connect(db_path) as conn:
        if not _table_exists(conn, table):
            return False
        cur = conn.execute(
            f"SELECT 1 FROM {table} WHERE reference_date = ? LIMIT 1",
            (ref_date,),
        )
        return cur.fetchone() is not None


def posdb_count_for_date(
    db_path: str, table: str, ref_date: str
) -> int:
    if not db_path or not os.path.exists(db_path):
        return 0
    with sqlite3.connect(db_path) as conn:
        if not _table_exists(conn, table):
            return 0
        cur = conn.execute(
            f"SELECT COUNT(*) FROM {table} WHERE reference_date = ?",
            (ref_date,),
        )
        row = cur.fetchone()
        return int(row[0]) if row else 0


def posdb_delete_date(
    conn: sqlite3.Connection, table: str, ref_date: str
) -> None:
    if _table_exists(conn, table):
        conn.execute(
            f"DELETE FROM {table} WHERE reference_date = ?",
            (ref_date,),
        )


def posdb_ensure_index(
    conn: sqlite3.Connection, table: str, col: str = "reference_date"
) -> None:
    idx_name = f"idx_{table}_{col}"
    try:
        conn.execute(
            f"CREATE INDEX IF NOT EXISTS {idx_name} "
            f"ON {table} ({col})"
        )
    except Exception:
        pass


def posdb_save_all(
    db_path: str,
    df_pos: Optional[pd.DataFrame],
    df_pmv: Optional[pd.DataFrame],
    ref_date: str,
    overwrite: bool,
) -> None:
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        if overwrite:
            posdb_delete_date(conn, "positions", ref_date)
            posdb_delete_date(conn, "pmv", ref_date)

        if df_pos is not None and not df_pos.empty:
            df_pos.to_sql(
                "positions", conn, if_exists="append", index=False
            )
            posdb_ensure_index(conn, "positions", "reference_date")

        if df_pmv is not None and not df_pmv.empty:
            df_pmv.to_sql("pmv", conn, if_exists="append", index=False)
            posdb_ensure_index(conn, "pmv", "reference_date")


# -------------------------------
# Streamlit UI (raw + robust fetch)
# -------------------------------

st.title("Gorila - Rodrigo Cabral")

api_key = get_api_key()
with st.sidebar:
    st.subheader("Configuration")

    local_key_input = st.text_input(
        "API key (if not using st.secrets / env)",
        type="password",
        value="",
    )
    if not api_key and local_key_input:
        api_key = local_key_input

    base_url = st.text_input(
        "Base URL", value="https://core.gorila.com.br"
    )

    issuer_json_path = st.text_input(
        "Issuer JSON path", value="issuers.json"
    )

    positions_db_path = st.text_input(
        "Positions DB path",
        value="databases/gorila_positions.db",
        help="Where to store positions/PMV snapshots per reference date.",
    )

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
# RUN ALL (end-to-end)
# -------------------------------

st.header("Run All")
st.caption(
    "Runs: Portfolios → Issuers → Brokers → Update Issuer JSON → "
    "Positions → PMV (for ALL portfolios using the selected reference date). "
    "Debentures are NOT downloaded; only current in-memory data is used. "
    "A consolidated Excel workbook is also available, with Issuers on a separate sheet "
    "(id and name)."
)

if st.button("Run All", type="primary"):
    try:
        with st.spinner("Fetching portfolios..."):
            rows = [flatten_portfolio(p) for p in client.list_portfolios()]
            df_port = pd.DataFrame(rows)
            st.session_state["df_portfolios"] = df_port
        st.success(f"Portfolios fetched: {len(df_port)}")

        with st.spinner("Fetching issuers..."):
            rows_iss = [flatten_issuer(i) for i in client.list_issuers()]
            df_iss = pd.DataFrame(rows_iss)
            st.session_state["df_issuers"] = df_iss
        st.success(f"Issuers fetched: {len(df_iss)}")

        with st.spinner("Fetching brokers..."):
            rows_bro = [flatten_broker(b) for b in client.list_brokers()]
            df_bro = pd.DataFrame(rows_bro)
            st.session_state["df_brokers"] = df_bro
        st.success(f"Brokers fetched: {len(df_bro)}")

        with st.spinner("Updating issuer JSON from issuer table..."):
            df_min = df_iss[["id", "name"]].copy()
            exist_cnt, added_cnt, final_cnt = update_issuer_json_from_df(
                issuer_json_path, df_min
            )
        st.info(
            f"Issuer JSON updated. Existing={exist_cnt}, "
            f"added={added_cnt}, total={final_cnt}"
        )

        all_ids: List[str] = (
            st.session_state["df_portfolios"]["id"]
            .dropna()
            .astype(str)
            .tolist()
        )
        if not all_ids:
            st.error("No portfolio IDs found after fetching portfolios.")
            st.stop()

        # Fetch positions (all portfolios)
        with st.spinner("Fetching positions for all portfolios..."):
            errors_pos: List[Tuple[str, str]] = []
            dfs_pos: List[pd.DataFrame] = []
            prog_pos = st.progress(0.0)
            total_pos = len(all_ids)

            for idx, pid in enumerate(all_ids, start=1):
                try:
                    rows = [
                        flatten_position(pid, p)
                        for p in client.list_positions(
                            pid, ref_date_str
                        )
                    ]
                    dfs_pos.append(pd.DataFrame(rows))
                except Exception as e:
                    errors_pos.append((pid, str(e)))
                finally:
                    prog_pos.progress(min(idx / total_pos, 1.0))

            df_pos_all = (
                pd.concat(dfs_pos, ignore_index=True)
                if dfs_pos
                else pd.DataFrame()
            )
            st.session_state["df_positions"] = df_pos_all

        st.success(
            "Positions fetched: "
            f"{df_pos_all.shape[0]} rows from {len(all_ids)} portfolios"
        )
        

        if errors_pos:
            st.warning(
                "Some portfolios failed while fetching positions:\n"
                + "\n".join([f"- {pid}: {msg}" for pid, msg in errors_pos])
            )

        # Fetch PMV (all portfolios)
        with st.spinner(
            "Fetching position market values for all portfolios..."
        ):
            errors_pmv: List[Tuple[str, str]] = []
            dfs_pmv: List[pd.DataFrame] = []
            prog_pmv = st.progress(0.0)
            total_pmv = len(all_ids)

            for idx, pid in enumerate(all_ids, start=1):
                try:
                    rows = [
                        flatten_position_market_value(pid, it)
                        for it in client.list_position_market_values(
                            pid, ref_date_str
                        )
                    ]
                    dfs_pmv.append(pd.DataFrame(rows))
                except Exception as e:
                    errors_pmv.append((pid, str(e)))
                finally:
                    prog_pmv.progress(min(idx / total_pmv, 1.0))

            df_pmv_all = (
                pd.concat(dfs_pmv, ignore_index=True)
                if dfs_pmv
                else pd.DataFrame()
            )

        # Enrich PMV
        if not df_pmv_all.empty:
            df_pmv_all = map_issuer_name_from_json(
                df_pmv_all, issuer_json_path
            )
            df_pmv_all = add_debentures_parsed_cols(df_pmv_all)
            df_pmv_all = add_cra_cri_parsed_cols(df_pmv_all)
            df_pmv_all = add_cdb_lci_lca_parsed_cols(df_pmv_all)
            df_pmv_all = add_treasury_local_parsed_cols(df_pmv_all)

            if "indexer" not in df_pmv_all.columns:
                df_pmv_all["indexer"] = None
            df_pmv_all["indexer"] = df_pmv_all["indexer"].apply(
                _normalize_indexer_value
            )

            # Ensure broker_id / broker_name from raw if missing
            df_pmv_all = ensure_broker_cols_from_raw(df_pmv_all)

        st.session_state["df_position_market_values"] = df_pmv_all

        st.success(
            "PMV fetched and enriched: "
            f"{df_pmv_all.shape[0]} rows from {len(all_ids)} portfolios"
        )

        # Add Conta column based on portfolio_id and broker_id mapping
        df_pmv_all = add_conta_column(df_pmv_all)
        
        # Map asset_class values to Portuguese names
        if "asset_class" in df_pmv_all.columns:
            df_pmv_all["asset_class"] = df_pmv_all["asset_class"].map(
                lambda x: ASSET_CLASS_MAP.get(x, x) if pd.notna(x) else x
            )
        

        if errors_pmv:
            st.warning(
                "Some portfolios failed while fetching PMV:\n"
                + "\n".join([f"- {pid}: {msg}" for pid, msg in errors_pmv])
            )

        # CSV (PMV) and consolidated XLSX workbook
        # Filter PMV columns for CSV export
        pmv_csv_columns = [
            "portfolio_id",
            "reference_date",
            "security_type",
            "asset_class",
            "price_currency",
            "market_value_amount",
            "broker_id",
            "broker_name",
            "parsed_bond_type",
            "indexer",
            "parsed_company_name",
            "parsed_maturity_date",
            "Conta",
        ]
        df_pmv_for_csv = df_pmv_all[[col for col in pmv_csv_columns if col in df_pmv_all.columns]]

        st.download_button(
            label="Download PMV (CSV) - Run All",
            data=df_to_csv_bytes(df_pmv_for_csv),
            file_name="position_market_values.csv",
            mime="text/csv;charset=utf-8",
        )

        # Download CSV without BTG PACTUAL DTVM
        df_pmv_sem_btg = df_pmv_for_csv[df_pmv_for_csv.get("broker_name", "") != "BTG PACTUAL DTVM"]
        st.download_button(
            label="Download CSV sem BTG",
            data=df_to_csv_bytes(df_pmv_sem_btg),
            file_name="position_market_values_sem_btg.csv",
            mime="text/csv;charset=utf-8",
        )

        wb_sheets = {}
        if st.session_state.get("df_portfolios") is not None:
            wb_sheets["portfolios"] = st.session_state["df_portfolios"]
        if st.session_state.get("df_issuers") is not None:
            wb_sheets["issuers"] = st.session_state["df_issuers"]
        if st.session_state.get("df_brokers") is not None:
            wb_sheets["brokers"] = st.session_state["df_brokers"]
        if st.session_state.get("df_positions") is not None:
            wb_sheets["positions"] = st.session_state["df_positions"]
        if st.session_state.get("df_position_market_values") is not None:
            wb_sheets["pmv"] = st.session_state["df_position_market_values"]

        if wb_sheets:
            xlsx_bytes = dataframes_to_excel_bytes(wb_sheets)
            st.download_button(
                label="Download consolidated workbook (XLSX)",
                data=xlsx_bytes,
                file_name="gorila_core_snapshot.xlsx",
                mime=(
                    "application/vnd.openxmlformats-officedocument."
                    "spreadsheetml.sheet"
                ),
            )

    except Exception as e:
        st.error(f"Run All failed: {e}")