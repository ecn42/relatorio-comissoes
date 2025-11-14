import os
import io
import re
import time
import json
import sqlite3
import logging
import zipfile
from pathlib import Path
from logging.handlers import RotatingFileHandler
from datetime import date, timedelta
from typing import Dict, Iterable, List, Optional
from urllib.parse import urljoin

import pandas as pd
import requests
import streamlit as st
import unicodedata

# -------------------------------
# Constants and Holidays
# -------------------------------

CVM_CACHE_DIR = "cvm_cache"

BRAZIL_HOLIDAYS_2025 = {
    date(2025, 1, 1),
    date(2025, 3, 3),
    date(2025, 3, 4),
    date(2025, 4, 18),
    date(2025, 4, 21),
    date(2025, 5, 1),
    date(2025, 6, 19),
    date(2025, 9, 7),
    date(2025, 10, 12),
    date(2025, 11, 2),
    date(2025, 11, 15),
    date(2025, 11, 20),
    date(2025, 12, 25),
}

# -------------------------------
# Utility Functions
# -------------------------------


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def get_3_working_days_ago() -> date:
    today = date.today()
    days_back = 0
    current = today
    while days_back < 3:
        current = current - timedelta(days=1)
        if current.weekday() < 5 and current not in BRAZIL_HOLIDAYS_2025:
            days_back += 1
    return current


def normalize_taxid_str(x) -> Optional[str]:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    s = str(x)
    s = re.sub(r"\D", "", s)
    return s or None


def normalize_codigo_cetip(x) -> Optional[str]:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    s = str(x).strip().upper()
    s = re.sub(r"[^A-Z0-9]", "", s)
    return s or None


def normalize_id_str(x) -> Optional[str]:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    s = str(x).strip()
    digits = re.sub(r"\D", "", s)
    return digits if digits else (s or None)


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Export to CSV with European format; protect id-like columns as strings."""
    df_copy = df.copy()

    # Protect id-like columns (avoid scientific notation)
    protect_re = re.compile(r"(taxid|cnpj|cpf|codigo|cetip|isin|id$)", re.I)
    for col in df_copy.columns:
        if protect_re.search(col):
            df_copy[col] = df_copy[col].apply(lambda x: "" if pd.isna(x) else str(x))

    numeric_cols = df_copy.select_dtypes(include=["number"]).columns
    for col in numeric_cols:
        df_copy[col] = df_copy[col].apply(
            lambda x: str(x).replace(".", ",") if pd.notna(x) else "",
            convert_dtype=False,
        )

    csv_str = df_copy.to_csv(index=False, sep=";", encoding="utf-8")
    return csv_str.encode("utf-8-sig")


# -------------------------------
# Debug sink
# -------------------------------


class DebugSink:
    def __init__(self, enabled: bool):
        self.enabled = enabled
        self._placeholder = st.empty()
        self._buffer: List[str] = []

        self.logger = logging.getLogger("gorila_debug")
        self.logger.setLevel(logging.INFO if enabled else logging.WARNING)

        if enabled:
            has_rotating = any(
                isinstance(h, RotatingFileHandler) for h in self.logger.handlers
            )
            if not has_rotating:
                handler = RotatingFileHandler(
                    "gorila_debug.log",
                    maxBytes=2_000_000,
                    backupCount=2,
                    encoding="utf-8",
                )
                fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
                handler.setFormatter(fmt)
                self.logger.addHandler(handler)

    def log(self, msg: str):
        if not self.enabled:
            return
        self._buffer.append(msg)
        self._buffer = self._buffer[-200:]
        self._placeholder.code("\n".join(self._buffer), language="text")
        try:
            self.logger.info(msg)
        except Exception:
            pass


# -------------------------------
# HTTP Client
# -------------------------------


class GorilaClient:
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://core.gorila.com.br",
        timeout: int = 30,
        max_retries: int = 5,
        backoff_base: float = 0.6,
        debug_sink: Optional[DebugSink] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({"Authorization": api_key})
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.debug = debug_sink

    def _dbg(self, msg: str):
        if self.debug:
            self.debug.log(msg)

    def _request(self, method: str, path_or_url: str, params: Dict = None) -> Dict:
        url = (
            path_or_url
            if str(path_or_url).startswith("http")
            else urljoin(self.base_url + "/", str(path_or_url).lstrip("/"))
        )

        retryable_status = {408, 429, 500, 502, 503, 504}
        last_err = None

        for attempt in range(self.max_retries):
            start = time.perf_counter()
            try:
                resp = self.session.request(
                    method, url, params=params, timeout=self.timeout
                )
                latency = (time.perf_counter() - start) * 1000
                self._dbg(
                    f"GET {resp.status_code} {int(latency)}ms | {url} | "
                    f"params={params}"
                )

                if resp.status_code in retryable_status:
                    wait = self.backoff_base * (2**attempt)
                    self._dbg(
                        f"Transient {resp.status_code}. Backing off "
                        f"{wait:.2f}s (attempt {attempt+1}/{self.max_retries})"
                    )
                    last_err = Exception(f"HTTP {resp.status_code}: {resp.text[:200]}")
                    time.sleep(wait)
                    continue

                if 400 <= resp.status_code < 500:
                    preview = resp.text[:200].replace("\n", " ")
                    self._dbg(f"Non-retryable {resp.status_code}. Body: {preview}")
                    resp.raise_for_status()

                if not resp.text:
                    return {}
                try:
                    return resp.json()
                except ValueError:
                    preview = resp.text[:200].replace("\n", " ")
                    self._dbg(f"Non-JSON response preview: {preview}")
                    return {}

            except requests.HTTPError:
                raise
            except requests.RequestException as e:
                latency = (time.perf_counter() - start) * 1000
                self._dbg(f"RequestException after {int(latency)}ms on {url}: {e}")
                last_err = e
                wait = self.backoff_base * (2**attempt)
                self._dbg(
                    f"Backing off {wait:.2f}s (attempt {attempt+1}/{self.max_retries})"
                )
                time.sleep(wait)

        raise RuntimeError(f"Request failed after retries: {last_err}") from last_err

    def _normalize_next(self, nxt: Optional[str]) -> Optional[str]:
        if not nxt:
            return None
        nxt = str(nxt).strip()
        return nxt or None

    def _extract_page_token(self, url_or_path: str) -> str:
        try:
            from urllib.parse import parse_qs, urlparse

            q = urlparse(url_or_path).query or ""
            token = parse_qs(q).get("pageToken", [""])[0]
            return token
        except Exception:
            return ""

    def _iter_records(self, payload: Dict):
        if isinstance(payload, dict) and "records" in payload:
            for r in payload.get("records", []):
                yield r
        elif isinstance(payload, list):
            for r in payload:
                yield r
        else:
            if payload:
                yield payload

    def _describe_and_count_page(self, payload: Dict, page_idx: int, total_before: int):
        if isinstance(payload, dict) and "records" in payload:
            recs = payload.get("records", []) or []
            nxt = payload.get("next")
            self._dbg(
                f"Page {page_idx}: {len(recs)} records "
                f"(cumulative ~{total_before + len(recs)}) "
                f"next_token={self._extract_page_token(nxt) if nxt else ''}"
            )
            return len(recs), nxt
        elif isinstance(payload, list):
            self._dbg(f"Page {page_idx}: list payload of {len(payload)} (no next)")
            return len(payload), None
        else:
            nxt = payload.get("next") if isinstance(payload, dict) else None
            self._dbg(f"Page {page_idx}: single/non-standard payload; next={bool(nxt)}")
            return 1 if payload else 0, nxt

    def _paginate(self, path: str, params: Dict = None) -> Iterable[Dict]:
        page_idx = 1
        total = 0
        seen_cursors = set()
        empty_page_streak = 0

        payload = self._request("GET", path, params=params)
        count, nxt = self._describe_and_count_page(payload, page_idx, total)
        total += count
        for r in self._iter_records(payload):
            yield r

        nxt = self._normalize_next(nxt)
        while nxt:
            token = self._extract_page_token(nxt) or nxt
            if token in seen_cursors:
                self._dbg(
                    "Detected repeated cursor on page "
                    f"{page_idx+1}; breaking to avoid loop. token={token}"
                )
                break
            seen_cursors.add(token)

            page_idx += 1
            payload = self._request("GET", nxt, params=None)
            count, nxt = self._describe_and_count_page(payload, page_idx, total)
            total += count

            if count == 0:
                empty_page_streak += 1
                if empty_page_streak >= 1:
                    self._dbg("Empty page encountered; stopping to avoid loop.")
                    break
            else:
                empty_page_streak = 0

            for r in self._iter_records(payload):
                yield r

            nxt = self._normalize_next(nxt)

    # API endpoints
    def list_portfolios(self) -> Iterable[Dict]:
        return self._paginate("/portfolios")

    def list_issuers(self) -> Iterable[Dict]:
        return self._paginate("/issuers")

    def list_positions(self, portfolio_id: str, reference_date: Optional[str] = None):
        params = {}
        if reference_date:
            params["referenceDate"] = reference_date
        path = f"/portfolios/{portfolio_id}/positions"
        return self._paginate(path, params=params)

    def list_position_market_values(
        self, portfolio_id: str, reference_date: Optional[str] = None
    ):
        params = {}
        if reference_date:
            params["referenceDate"] = reference_date
        path = f"/portfolios/{portfolio_id}/positions/market-values"
        return self._paginate(path, params=params)

    def read_issuer(self, issuer_id: str) -> Dict:
        return self._request("GET", f"/issuers/{issuer_id}")


# -------------------------------
# Flatten helpers
# -------------------------------


def flatten_portfolio(p: Dict) -> Dict:
    return {"id": p.get("id"), "name": p.get("name"), "autoRnC": p.get("autoRnC")}


def flatten_issuer(i: Dict) -> Dict:
    return {
        "id": i.get("id"),
        "name": i.get("name"),
        "taxId": normalize_taxid_str(i.get("taxId") or i.get("cnpj")),
    }


def flatten_position(portfolio_id: str, pos: Dict) -> Dict:
    sec = pos.get("security", {}) or {}
    bro = pos.get("broker", {}) or {}

    issuer_id = sec.get("issuerId")
    issuer_tax = normalize_taxid_str(sec.get("issuer"))

    return {
        "portfolio_id": portfolio_id,
        "position_id": pos.get("id") or pos.get("positionId"),
        "reference_date": pos.get("referenceDate"),
        "quantity": pos.get("quantity"),
        "currency": pos.get("currency"),
        "security_id": normalize_id_str(sec.get("id")),
        "security_name": sec.get("name"),
        "security_type": sec.get("type"),
        "asset_class": sec.get("assetClass"),
        "isin": sec.get("isin"),
        "maturity_date": sec.get("maturityDate"),
        "validity_date": sec.get("validityDate"),
        "broker_id": bro.get("id"),
        "broker_name": bro.get("name"),
        "issuer_id": issuer_id,
        "issuer_taxid": issuer_tax,
        "issuer_name": None,
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

    issuer_id = sec.get("issuerId")
    issuer_taxid = normalize_taxid_str(sec.get("issuer"))

    return {
        "portfolio_id": portfolio_id,
        "position_id": item.get("positionId") or item.get("position_id"),
        "reference_date": item.get("referenceDate"),
        "security_id": normalize_id_str(sec.get("id")),
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
        "issuer_id": issuer_id,
        "issuer_taxid": issuer_taxid,
        "issuer_name": None,
        "raw": json.dumps(item, ensure_ascii=False),
    }


# -------------------------------
# Persistence
# -------------------------------


def persist_df(db_path: str, table: str, df: pd.DataFrame) -> None:
    if df.empty:
        return
    with sqlite3.connect(db_path) as conn:
        df.to_sql(table, conn, if_exists="replace", index=False)


def read_table(db_path: str, table: str) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        try:
            return pd.read_sql_query(f"SELECT * FROM {table}", conn)
        except Exception:
            return pd.DataFrame()


# -------------------------------
# Streamlit Caching
# -------------------------------


@st.cache_data(show_spinner=False, ttl=600)
def cached_list_portfolios(api_key: str, enable_debug: bool) -> pd.DataFrame:
    client = GorilaClient(api_key, debug_sink=debug_sink if enable_debug else None)
    rows = [flatten_portfolio(p) for p in client.list_portfolios()]
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False, ttl=3600)
def cached_list_issuers(api_key: str, enable_debug: bool) -> pd.DataFrame:
    client = GorilaClient(api_key, debug_sink=debug_sink if enable_debug else None)
    rows = [flatten_issuer(i) for i in client.list_issuers()]
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False, ttl=600)
def cached_list_positions(
    api_key: str, portfolio_id: str, reference_date: Optional[str], enable_debug: bool
) -> pd.DataFrame:
    client = GorilaClient(api_key, debug_sink=debug_sink if enable_debug else None)
    rows = [
        flatten_position(portfolio_id, p)
        for p in client.list_positions(portfolio_id, reference_date)
    ]
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False, ttl=600)
def cached_list_position_market_values(
    api_key: str, portfolio_id: str, reference_date: Optional[str], enable_debug: bool
) -> pd.DataFrame:
    client = GorilaClient(api_key, debug_sink=debug_sink if enable_debug else None)
    rows = [
        flatten_position_market_value(portfolio_id, p)
        for p in client.list_position_market_values(portfolio_id, reference_date)
    ]
    return pd.DataFrame(rows)


# -------------------------------
# Issuers JSON sync (taxId + name only)
# -------------------------------


def load_issuers_json(path: str = "issuers.json") -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(columns=["taxId", "name"])
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
        rows = []

        if isinstance(obj, dict):
            # Accept {"123...": "NAME"} or {"123...": {"name": "...", "taxId": "..."}} or {"id": {...}}
            for k, v in obj.items():
                if isinstance(v, dict):
                    tax = normalize_taxid_str(v.get("taxId") or k)
                    nm = v.get("name")
                else:
                    tax = normalize_taxid_str(k)
                    nm = str(v)
                rows.append({"taxId": tax, "name": nm})
        elif isinstance(obj, list):
            for rec in obj:
                if not isinstance(rec, dict):
                    continue
                tax = normalize_taxid_str(rec.get("taxId") or rec.get("id"))
                nm = rec.get("name")
                rows.append({"taxId": tax, "name": nm})

        df = pd.DataFrame(rows)
        if df.empty:
            return pd.DataFrame(columns=["taxId", "name"])
        if "taxId" not in df.columns:
            df["taxId"] = None
        if "name" not in df.columns:
            df["name"] = None
        df["taxId"] = df["taxId"].apply(normalize_taxid_str)
        df = df.dropna(subset=["taxId"]).drop_duplicates(subset=["taxId"], keep="last")
        return df[["taxId", "name"]]
    except Exception:
        return pd.DataFrame(columns=["taxId", "name"])


def save_issuers_json(df: pd.DataFrame, path: str = "issuers.json") -> None:
    if df is None or df.empty:
        return
    df = df.copy()
    if "taxId" not in df.columns:
        df["taxId"] = None
    if "name" not in df.columns:
        df["name"] = None
    df["taxId"] = df["taxId"].apply(normalize_taxid_str)
    out = (
        df[["taxId", "name"]]
        .dropna(subset=["taxId"])
        .drop_duplicates(subset=["taxId"], keep="last")
        .to_dict("records")
    )
    Path(path).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


def sync_issuers_json_from_table(db_path: str, path: str = "issuers.json") -> None:
    df_tbl = read_table(db_path, "issuers")
    if df_tbl.empty:
        return
    df_tbl = df_tbl[["name", "taxId"]].copy()
    df_tbl["taxId"] = df_tbl["taxId"].apply(normalize_taxid_str)

    df_old = load_issuers_json(path)
    all_df = pd.concat([df_old, df_tbl], ignore_index=True)
    all_df["taxId"] = all_df["taxId"].apply(normalize_taxid_str)
    all_df = all_df.dropna(subset=["taxId"]).drop_duplicates(subset=["taxId"], keep="last")
    save_issuers_json(all_df, path)


# -------------------------------
# CRI/CRA CETIP and CVM loaders (download + unzip to disk)
# -------------------------------


def _norm_header(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = s.strip().lower()
    s = re.sub(r"[\s]+", "_", s)
    s = s.replace("__", "_")
    return s


def _read_csv_relaxed_file(path: Path) -> pd.DataFrame:
    for enc in ["utf-8-sig", "latin1", "cp1252"]:
        for sep in [";", ",", "\t"]:
            try:
                with path.open("r", encoding=enc, errors="replace") as f:
                    df = pd.read_csv(
                        f,
                        sep=sep,
                        engine="python",
                        dtype=str,
                        quoting=3,
                        on_bad_lines="skip",
                    )
                df.columns = [_norm_header(c) for c in df.columns]
                return df
            except Exception:
                continue
    return pd.DataFrame()


def _aggregate_cvm(df_raw: pd.DataFrame) -> pd.DataFrame:
    if df_raw is None or df_raw.empty:
        return pd.DataFrame(
            columns=["codigo_cetip_norm", "vencimento", "rating", "cnpj_emissora"]
        )

    cols = set(df_raw.columns)
    cand_codigo = ["codigo_cetip", "codigocetip", "codigo"]
    cand_venc = ["data_vencimento", "dt_vencimento", "vencimento", "data_venc", "venc"]
    cand_rating = [
        "classificacao_risco_atual",
        "classificacao_de_risco_atual",
        "classificacao_risco",
        "rating",
    ]
    cand_cnpj = ["cnpj_emissora", "cnpj", "cnpj_emissora_classe", "cnpj_emissor"]

    def pick(candidates):
        for c in candidates:
            if c in cols:
                return c
        return None

    c_codigo = pick(cand_codigo)
    if not c_codigo:
        return pd.DataFrame(
            columns=["codigo_cetip_norm", "vencimento", "rating", "cnpj_emissora"]
        )
    c_venc = pick(cand_venc)
    c_rating = pick(cand_rating)
    c_cnpj = pick(cand_cnpj)

    w = df_raw.copy()
    w["codigo_cetip_norm"] = (
        w[c_codigo].astype(str).str.upper().str.replace(r"[^A-Z0-9]", "", regex=True)
    )

    if c_venc:
        raw_v = w[c_venc].astype(str).str.strip()
        v1 = pd.to_datetime(raw_v, format="%d/%m/%Y", errors="coerce", dayfirst=True)
        v2 = pd.to_datetime(raw_v, errors="coerce", dayfirst=True)
        v = v1.combine_first(v2)
        w["vencimento_iso"] = v.dt.strftime("%Y-%m-%d")
    else:
        w["vencimento_iso"] = pd.NA

    if c_rating:
        w["rating_clean"] = w[c_rating].astype(str).str.strip().replace({"": pd.NA})
    else:
        w["rating_clean"] = pd.NA

    if c_cnpj:
        w["cnpj_digits"] = (
            w[c_cnpj].astype(str).str.replace(r"\D", "", regex=True).replace("", pd.NA)
        )
    else:
        w["cnpj_digits"] = pd.NA

    def first_non_null(s: pd.Series):
        s = s.dropna()
        return s.iloc[0] if len(s) else pd.NA

    agg = (
        w.groupby("codigo_cetip_norm", sort=False)
        .agg(
            vencimento=("vencimento_iso", first_non_null),
            rating=("rating_clean", first_non_null),
            cnpj_emissora=("cnpj_digits", first_non_null),
        )
        .reset_index()
    )
    return agg


def _download_to(path: Path, url: str) -> None:
    ensure_dir(path.parent)
    r = requests.get(url, timeout=180)
    r.raise_for_status()
    path.write_bytes(r.content)


def _extract_zip(zip_path: Path, extract_dir: Path) -> None:
    ensure_dir(extract_dir)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_dir)


def _find_csv(extract_dir: Path, kind: str, year: int, flavor: str) -> Optional[Path]:
    # Try strict names, else search
    strict = extract_dir / f"inf_mensal_{kind.lower()}_{flavor}_{year}.csv"
    if strict.exists():
        return strict
    # Search recursively
    for p in extract_dir.rglob("*.csv"):
        if re.search(rf"inf_mensal_{kind.lower()}_{flavor}_{year}\.csv$", p.name, re.I):
            return p
    return None


def load_cvm_map(kind: str, enable_debug: bool = False) -> pd.DataFrame:
    """
    Download latest ZIP to cvm_cache, unzip, then read classe+serie CSVs from disk.
    Aggregate per CETIP code and merge (classe preferred).
    """
    base = f"https://dados.cvm.gov.br/dados/SECURIT/DOC/INF_MENSAL_{kind}/DADOS/"
    idx = requests.get(base, timeout=60)
    idx.raise_for_status()
    html = idx.text
    years = re.findall(rf"inf_mensal_{kind.lower()}_(\d{{4}})\.zip", html, flags=re.I)
    if not years:
        return pd.DataFrame(
            columns=["codigo_cetip_norm", "vencimento", "rating", "cnpj_emissora"]
        )
    year = max(int(y) for y in years)

    # Local paths
    base_dir = Path(CVM_CACHE_DIR) / kind.upper() / str(year)
    ensure_dir(base_dir)
    zip_path = base_dir / f"inf_mensal_{kind.lower()}_{year}.zip"
    url = urljoin(base, zip_path.name)

    if enable_debug:
        try:
            debug_sink.log(f"CVM {kind}: local dir={base_dir}")
        except Exception:
            pass

    # Download if not present
    if not zip_path.exists():
        if enable_debug:
            try:
                debug_sink.log(f"Downloading {url} -> {zip_path}")
            except Exception:
                pass
        _download_to(zip_path, url)

    # Extract if not yet extracted (heuristic: look for classe/serie)
    classe_file = _find_csv(base_dir, kind, year, "classe")
    serie_file = _find_csv(base_dir, kind, year, "serie")
    if classe_file is None and serie_file is None:
        if enable_debug:
            try:
                debug_sink.log(f"Extracting {zip_path} to {base_dir}")
            except Exception:
                pass
        _extract_zip(zip_path, base_dir)
        classe_file = _find_csv(base_dir, kind, year, "classe")
        serie_file = _find_csv(base_dir, kind, year, "serie")

    df_c = _read_csv_relaxed_file(classe_file) if classe_file else pd.DataFrame()
    df_s = _read_csv_relaxed_file(serie_file) if serie_file else pd.DataFrame()

    agg_c = _aggregate_cvm(df_c)
    agg_s = _aggregate_cvm(df_s)

    if agg_c.empty and agg_s.empty:
        return pd.DataFrame(
            columns=["codigo_cetip_norm", "vencimento", "rating", "cnpj_emissora"]
        )
    if agg_c.empty:
        merged = agg_s.copy()
    elif agg_s.empty:
        merged = agg_c.copy()
    else:
        merged = agg_c.merge(
            agg_s, on="codigo_cetip_norm", how="outer", suffixes=("_c", "_s")
        )
        for tgt, ccol, scol in [
            ("vencimento", "vencimento_c", "vencimento_s"),
            ("rating", "rating_c", "rating_s"),
            ("cnpj_emissora", "cnpj_emissora_c", "cnpj_emissora_s"),
        ]:
            merged[tgt] = merged[ccol].combine_first(merged[scol])
        merged = merged[["codigo_cetip_norm", "vencimento", "rating", "cnpj_emissora"]]
    return merged


@st.cache_data(show_spinner=False, ttl=43200)
def cached_load_cvm_cri_map(enable_debug: bool) -> pd.DataFrame:
    return load_cvm_map("CRI", enable_debug)


@st.cache_data(show_spinner=False, ttl=43200)
def cached_load_cvm_cra_map(enable_debug: bool) -> pd.DataFrame:
    return load_cvm_map("CRA", enable_debug)


def add_codigo_cetip_from_name(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "security_name" not in df.columns:
        return df
    out = df.copy()
    sname = out["security_name"].astype(str)
    extracted = sname.str.extract(r"([A-Z0-9]{6,16})\s*$", expand=False)
    if "codigo_cetip" not in out.columns:
        out["codigo_cetip"] = extracted
    else:
        out["codigo_cetip"] = out["codigo_cetip"].fillna(extracted)
    out["codigo_cetip_norm"] = out["codigo_cetip"].apply(normalize_codigo_cetip)
    return out


# -------------------------------
# Issuer enrichment (robust + CVM override for CRI/CRA)
# -------------------------------


def attach_issuer_fields_and_names(
    df_pmv: pd.DataFrame,
    df_positions: Optional[pd.DataFrame],
    df_issuers: Optional[pd.DataFrame],
    issuers_json_path: str = "issuers.json",
) -> pd.DataFrame:
    if df_pmv.empty:
        return df_pmv

    out = df_pmv.copy()

    # Normalize keys used for merging/mapping
    if "security_id" in out.columns:
        out["security_id"] = out["security_id"].apply(normalize_id_str)
    if "issuer_taxid" in out.columns:
        out["issuer_taxid"] = out["issuer_taxid"].apply(normalize_taxid_str)

    # Fill issuer_id/issuer_taxid from positions by security_id
    if df_positions is not None and not df_positions.empty:
        pos = df_positions.copy()
        if "security_id" in pos.columns:
            pos["security_id"] = pos["security_id"].apply(normalize_id_str)
        if "issuer_taxid" in pos.columns:
            pos["issuer_taxid"] = pos["issuer_taxid"].apply(normalize_taxid_str)
        lookup = (
            pos[["security_id", "issuer_id", "issuer_taxid"]]
            .dropna(subset=["security_id"])
            .drop_duplicates(subset=["security_id"])
        )
        out = out.merge(lookup, on="security_id", how="left", suffixes=("", "_from_pos"))
        out["issuer_id"] = out["issuer_id"].combine_first(out.get("issuer_id_from_pos"))
        out["issuer_taxid"] = out["issuer_taxid"].combine_first(
            out.get("issuer_taxid_from_pos")
        )
        out.drop(
            columns=["issuer_id_from_pos", "issuer_taxid_from_pos"],
            inplace=True,
            errors="ignore",
        )

    # Build name maps: JSON (taxId->name) and table (id->name, taxId->name)
    taxid_to_name_json = {}
    json_df = load_issuers_json(issuers_json_path)
    if not json_df.empty:
        j = json_df.copy()
        j["taxId"] = j["taxId"].apply(normalize_taxid_str)
        taxid_to_name_json = (
            j.dropna(subset=["taxId"])
            .drop_duplicates(subset=["taxId"])
            .set_index("taxId")["name"]
            .to_dict()
        )

    id_to_name_tbl = {}
    taxid_to_name_tbl = {}
    if df_issuers is not None and not df_issuers.empty:
        di = df_issuers.copy()
        di["id"] = di["id"].astype(str)
        di["taxId"] = di["taxId"].apply(normalize_taxid_str)
        id_to_name_tbl = (
            di.dropna(subset=["id"]).drop_duplicates(subset=["id"]).set_index("id")["name"].to_dict()
        )
        taxid_to_name_tbl = (
            di.dropna(subset=["taxId"])
            .drop_duplicates(subset=["taxId"])
            .set_index("taxId")["name"]
            .to_dict()
        )

    if "issuer_name" not in out.columns:
        out["issuer_name"] = None

    # Prefer table for issuer_id->name
    if "issuer_id" in out.columns:
        out["issuer_id"] = out["issuer_id"].astype(str)
        out["issuer_name"] = out["issuer_name"].fillna(out["issuer_id"].map(id_to_name_tbl))

    # Then fill by issuer_taxid, prefer table over JSON
    if "issuer_taxid" in out.columns:
        name_by_taxid = {**taxid_to_name_json, **taxid_to_name_tbl}
        out["issuer_name"] = out["issuer_name"].fillna(out["issuer_taxid"].map(name_by_taxid))

    return out


def enrich_with_cvm_maps(
    df_pmv: pd.DataFrame,
    df_cvm_cri: Optional[pd.DataFrame],
    df_cvm_cra: Optional[pd.DataFrame],
    prefer_cvm_issuer_taxid: bool = True,
) -> pd.DataFrame:
    """
    - Join by codigo_cetip_norm for CRI/CRA rows.
    - Fill vencimento and rating.
    - For CRI/CRA, override issuer_taxid with CVM cnpj_emissora if available
      (to stabilize issuer across brokers).
    """
    if df_pmv.empty:
        return df_pmv
    out = df_pmv.copy()
    out = add_codigo_cetip_from_name(out)

    if "codigo_cetip_norm" in out.columns:
        out["codigo_cetip_norm"] = out["codigo_cetip_norm"].apply(normalize_codigo_cetip)

    stype = out.get("security_type", pd.Series(index=out.index, dtype="object")).astype(str).str.upper()

    def apply_map(kind_df: pd.DataFrame, mask: pd.Series):
        if kind_df is None or kind_df.empty:
            return
        kind_df = kind_df.copy()
        kind_df["codigo_cetip_norm"] = kind_df["codigo_cetip_norm"].apply(normalize_codigo_cetip)
        subset_idx = out.index[mask]
        tmp = out.loc[subset_idx].merge(
            kind_df,
            on="codigo_cetip_norm",
            how="left",
            suffixes=("", "_from_cvm"),
        )

        if "vencimento" not in out.columns:
            out["vencimento"] = pd.NA
        if "rating" not in out.columns:
            out["rating"] = pd.NA
        if "issuer_taxid" not in out.columns:
            out["issuer_taxid"] = pd.NA

        out.loc[subset_idx, "vencimento"] = out.loc[subset_idx, "vencimento"].fillna(
            tmp["vencimento"]
        )
        out.loc[subset_idx, "rating"] = out.loc[subset_idx, "rating"].fillna(tmp["rating"])

        cvm_tax = tmp["cnpj_emissora"].apply(normalize_taxid_str)
        if prefer_cvm_issuer_taxid:
            # Override with CVM where present
            ov_mask = cvm_tax.notna()
            out.loc[subset_idx[ov_mask], "issuer_taxid"] = cvm_tax[ov_mask]
        else:
            # Only fill if missing
            out.loc[subset_idx, "issuer_taxid"] = out.loc[subset_idx, "issuer_taxid"].fillna(
                cvm_tax
            )

    apply_map(df_cvm_cri, stype == "CORPORATE_BONDS_CRI")
    apply_map(df_cvm_cra, stype == "CORPORATE_BONDS_CRA")

    return out


# -------------------------------
# Config / Secrets
# -------------------------------


def get_api_key() -> Optional[str]:
    if "GORILA_API_KEY" in st.secrets:
        return st.secrets["GORILA_API_KEY"]
    return os.getenv("GORILA_API_KEY")


# -------------------------------
# Streamlit UI
# -------------------------------

st.set_page_config(page_title="Gorila Loader", layout="wide")
st.title("Gorila CORE → DB Loader")

api_key = get_api_key()
with st.sidebar:
    st.subheader("Configuration")
    local_key_input = st.text_input(
        "API key (if not using st.secrets / env)", type="password", value=""
    )
    if not api_key and local_key_input:
        api_key = local_key_input

    db_path = st.text_input("SQLite DB path", value="gorila.db")

    default_ref_date = get_3_working_days_ago()
    ref_date = st.date_input(
        "Positions/PMV reference date (optional)",
        value=default_ref_date,
        min_value=date(1990, 1, 1),
    )
    ref_date_str = ref_date.isoformat() if ref_date else None

    debug = st.toggle("Debug mode", value=False)

debug_sink = DebugSink(debug)

if not api_key:
    st.warning(
        "Provide your Gorila API key via st.secrets['GORILA_API_KEY'], "
        "the GORILA_API_KEY env var, or the sidebar field."
    )
    st.stop()

st.write("Authentication looks good.")

st.subheader("Debug output")
with st.expander("Show debug log", expanded=debug):
    pass

# Portfolios
st.header("Portfolios")
download_all_portfolios = st.toggle("Download all (CSV) — portfolios", value=False)

if st.button("Fetch portfolios"):
    t0 = time.perf_counter()
    with st.spinner("Loading portfolios..."):
        df_port = cached_list_portfolios(api_key, debug)
    st.dataframe(df_port, use_container_width=True)
    persist_df(db_path, "portfolios", df_port)
    st.session_state["df_portfolios"] = df_port
    t1 = time.perf_counter()
    debug_sink.log(
        f"Portfolios: fetched {len(df_port)} rows in {int((t1 - t0) * 1000)}ms "
        f"and persisted to SQLite"
    )
    st.caption("Saved to table: portfolios")

if download_all_portfolios and st.session_state.get("df_portfolios") is not None:
    st.download_button(
        label="Download all portfolios (CSV)",
        data=df_to_csv_bytes(st.session_state["df_portfolios"]),
        file_name="portfolios.csv",
        mime="text/csv;charset=utf-8",
    )

# Issuers
st.header("Issuers")
if st.button("Fetch issuers"):
    t0 = time.perf_counter()
    with st.spinner("Loading issuers (may take a while)..."):
        df_iss = cached_list_issuers(api_key, debug)
    st.dataframe(df_iss, use_container_width=True)
    persist_df(db_path, "issuers", df_iss)
    st.session_state["df_issuers"] = df_iss
    # Sync issuers.json after updating table
    sync_issuers_json_from_table(db_path, "issuers.json")
    t1 = time.perf_counter()
    debug_sink.log(
        f"Issuers: fetched {len(df_iss)} rows in {int((t1 - t0) * 1000)}ms, "
        f"persisted to SQLite, and synced issuers.json"
    )
    st.caption("Saved to table: issuers (and synced issuers.json)")

if st.session_state.get("df_issuers") is not None:
    st.download_button(
        label="Download all issuers (CSV)",
        data=df_to_csv_bytes(st.session_state["df_issuers"]),
        file_name="issuers.csv",
        mime="text/csv;charset=utf-8",
    )

# Positions
st.header("Positions")
use_all_portfolios_for_positions = st.toggle("Use all portfolios for positions", value=False)

df_port_tbl = read_table(db_path, "portfolios")
selected_ids: List[str] = []
if not use_all_portfolios_for_positions:
    if not df_port_tbl.empty:
        st.caption("Select one or more portfolios to fetch positions for:")
        choices = df_port_tbl["id"] + " | " + df_port_tbl["name"].fillna("")
        selected = st.multiselect("Portfolios", options=choices.tolist())
        selected_ids = [s.split(" | ", 1)[0] for s in selected]
    else:
        st.info(
            "Load portfolios first to enable a selection here. "
            "Alternatively, paste portfolio IDs below."
        )

    manual_ids = st.text_input("Or paste portfolio IDs (comma-separated)", value="")
    if manual_ids.strip():
        selected_ids += [s.strip() for s in manual_ids.split(",") if s.strip()]
else:
    if not df_port_tbl.empty:
        selected_ids = df_port_tbl["id"].dropna().astype(str).tolist()
        debug_sink.log(f"Using all portfolios for positions: {len(selected_ids)}")
    else:
        st.info("Load portfolios first to use all.")

if st.button("Fetch positions for selected portfolios"):
    if not selected_ids:
        st.warning("No portfolio IDs selected.")
    else:
        all_pos: List[pd.DataFrame] = []
        progress = st.progress(0)
        total = len(selected_ids)
        for idx, pid in enumerate(selected_ids, start=1):
            debug_sink.log(f"Starting positions fetch for {pid} ref_date={ref_date_str}")
            with st.spinner(f"Loading positions for {pid}..."):
                t0 = time.perf_counter()
                df = cached_list_positions(api_key, pid, ref_date_str, debug)
                t1 = time.perf_counter()
                debug_sink.log(
                    f"Finished positions fetch for {pid}: {len(df)} rows "
                    f"in {int((t1 - t0) * 1000)}ms"
                )
            all_pos.append(df)
            progress.progress(min(idx / total, 1.0))

        df_pos = pd.concat(all_pos, ignore_index=True) if all_pos else pd.DataFrame()
        st.dataframe(df_pos, use_container_width=True)
        persist_df(db_path, "positions", df_pos)
        st.session_state["df_positions"] = df_pos
        st.caption("Saved to table: positions")

# Position Market Values
st.header("Position Market Values")
use_all_portfolios_for_pmv = st.toggle("Use all portfolios for market values", value=False)

pmv_selected_ids: List[str] = []
if not use_all_portfolios_for_pmv:
    if not df_port_tbl.empty:
        st.caption("Select portfolios for position market values:")
        choices_pmv = df_port_tbl["id"] + " | " + df_port_tbl["name"].fillna("")
        selected_pmv = st.multiselect(
            "Portfolios (market values)", options=choices_pmv.tolist(), key="pmv_ms"
        )
        pmv_selected_ids = [s.split(" | ", 1)[0] for s in selected_pmv]

    manual_pmv_ids = st.text_input(
        "Or paste portfolio IDs for market values (comma-separated)", value=""
    )
    if manual_pmv_ids.strip():
        pmv_selected_ids += [s.strip() for s in manual_pmv_ids.split(",") if s.strip()]
else:
    if not df_port_tbl.empty:
        pmv_selected_ids = df_port_tbl["id"].dropna().astype(str).tolist()
        debug_sink.log(
            "Using all portfolios for position market values: "
            f"{len(pmv_selected_ids)} portfolios"
        )
    else:
        st.info("Load portfolios first to use all.")

if st.button("Fetch position market values"):
    target_portfolio_ids = pmv_selected_ids

    if not target_portfolio_ids:
        st.warning("No portfolio IDs selected to fetch market values.")
    else:
        rows: List[pd.DataFrame] = []
        progress = st.progress(0)
        total = len(target_portfolio_ids)
        for idx, pid in enumerate(target_portfolio_ids, start=1):
            debug_sink.log(f"Reading position market values for {pid} ref_date={ref_date_str}")
            try:
                t0 = time.perf_counter()
                df_one = cached_list_position_market_values(api_key, pid, ref_date_str, debug)
                t1 = time.perf_counter()
                debug_sink.log(
                    f"PMV for {pid}: {len(df_one)} row(s) in {int((t1 - t0) * 1000)}ms"
                )
                rows.append(df_one)
            except requests.HTTPError as e:
                status = e.response.status_code if getattr(e, "response", None) else "?"
                body = (
                    e.response.text[:200].replace("\n", " ")
                    if getattr(e, "response", None)
                    else ""
                )
                debug_sink.log(
                    f"HTTPError fetching PMV for {pid}: status={status} body={body}"
                )
            except Exception as e:
                debug_sink.log(f"Error fetching PMV for {pid}: {e}")

            progress.progress(min(idx / total, 1.0))

        df_pmv = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

        # Load positions and issuers for enrichment
        df_positions_local = st.session_state.get("df_positions")
        if df_positions_local is None or df_positions_local.empty:
            df_positions_local = read_table(db_path, "positions")

        df_issuers_tbl = read_table(db_path, "issuers")
        df_issuers_json = load_issuers_json("issuers.json")
        if not df_issuers_json.empty:
            # merge json into table-like structure for enrichment (taxId + name)
            j = df_issuers_json.rename(columns={"taxId": "taxId_json"})
            df_issuers_tbl = pd.concat(
                [df_issuers_tbl, j.assign(id=None).rename(columns={"taxId_json": "taxId"})],
                ignore_index=True,
            )

        # 1) Attach issuer fields and names (consistent for a security_id)
        df_pmv = attach_issuer_fields_and_names(df_pmv, df_positions_local, df_issuers_tbl)

        # 2) CETIP + CVM enrichment (download/unzip to disk, then parse)
        try:
            df_cvm_cri = cached_load_cvm_cri_map(debug)
        except Exception as e:
            debug_sink.log(f"CVM CRI map failed: {e}")
            df_cvm_cri = pd.DataFrame()

        try:
            df_cvm_cra = cached_load_cvm_cra_map(debug)
        except Exception as e:
            debug_sink.log(f"CVM CRA map failed: {e}")
            df_cvm_cra = pd.DataFrame()

        before_venc = df_pmv["vencimento"].notna().sum() if "vencimento" in df_pmv.columns else 0
        before_rating = df_pmv["rating"].notna().sum() if "rating" in df_pmv.columns else 0

        df_pmv = enrich_with_cvm_maps(
            df_pmv, df_cvm_cri, df_cvm_cra, prefer_cvm_issuer_taxid=True
        )

        after_venc = df_pmv["vencimento"].notna().sum() if "vencimento" in df_pmv.columns else 0
        after_rating = df_pmv["rating"].notna().sum() if "rating" in df_pmv.columns else 0

        debug_sink.log(
            f"CVM enrichment: venc +{after_venc - before_venc}, "
            f"rating +{after_rating - before_rating}"
        )

        # 3) Re-attach issuer names (issuer_taxid may have been set/overridden by CVM)
        df_pmv = attach_issuer_fields_and_names(df_pmv, None, df_issuers_tbl)

        st.dataframe(df_pmv, use_container_width=True)
        persist_df(db_path, "position_market_values", df_pmv)
        st.session_state["df_position_market_values"] = df_pmv
        st.caption("Saved to table: position_market_values")

if st.session_state.get("df_position_market_values") is not None:
    st.download_button(
        label="Download all position market values (CSV)",
        data=df_to_csv_bytes(st.session_state["df_position_market_values"]),
        file_name="position_market_values.csv",
        mime="text/csv;charset=utf-8",
    )

# Database preview
st.header("Database preview")
col1, col2 = st.columns(2)
with col1:
    st.subheader("portfolios")
    st.dataframe(read_table(db_path, "portfolios"), height=240)
with col2:
    st.subheader("issuers")
    st.dataframe(read_table(db_path, "issuers"), height=240)

row2_col1, row2_col2 = st.columns(2)
with row2_col1:
    st.subheader("positions")
    st.dataframe(read_table(db_path, "positions"), height=240)
with row2_col2:
    st.subheader("position_market_values")
    st.dataframe(read_table(db_path, "position_market_values"), height=240)