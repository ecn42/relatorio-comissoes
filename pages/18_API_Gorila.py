import os
import time
import json
import sqlite3
import logging
from logging.handlers import RotatingFileHandler
from datetime import date, timedelta
from typing import Dict, Iterable, List, Optional
from urllib.parse import urljoin

import pandas as pd
import requests
import streamlit as st

# Brazilian national holidays (non-working days)
BRAZIL_HOLIDAYS_2025 = {
    date(2025, 1, 1),   # New Year
    date(2025, 3, 3),   # Carnival Monday
    date(2025, 3, 4),   # Carnival Tuesday
    date(2025, 4, 18),  # Good Friday
    date(2025, 4, 21),  # Tiradentes Day
    date(2025, 5, 1),   # Labour Day
    date(2025, 6, 19),  # Corpus Christi
    date(2025, 9, 7),   # Independence Day
    date(2025, 10, 12), # Our Lady of Aparecida
    date(2025, 11, 2),  # All Souls' Day
    date(2025, 11, 15), # Republic Day
    date(2025, 11, 20), # Black Consciousness Day
    date(2025, 12, 25), # Christmas
}

# -------------------------------
# Utility Functions
# -------------------------------


def get_3_working_days_ago() -> date:
    """Get the date 3 working days ago (excluding weekends and
    Brazilian holidays)."""
    today = date.today()
    days_back = 0
    current = today

    while days_back < 3:
        current = current - timedelta(days=1)
        # Monday=0, Sunday=6
        if current.weekday() < 5 and current not in BRAZIL_HOLIDAYS_2025:
            days_back += 1

    return current


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Export to CSV with European format: ; separator and ,
    decimal point."""
    df_copy = df.copy()

    # Format numeric columns with , as decimal separator
    numeric_cols = df_copy.select_dtypes(
        include=["number"]
    ).columns
    for col in numeric_cols:
        df_copy[col] = df_copy[col].apply(
            lambda x: str(x).replace(".", ",")
            if pd.notna(x)
            else "",
            convert_dtype=False,
        )

    # Generate CSV string (encoding param is ignored when no file output)
    csv_str = df_copy.to_csv(
        index=False, sep=";", encoding="utf-8"
    )
    # Encode with UTF-8 BOM to ensure proper handling in Excel/other apps
    return csv_str.encode("utf-8-sig")


def enrich_positions_with_issuer_info(
    df_pos: pd.DataFrame, df_issuers: pd.DataFrame
) -> pd.DataFrame:
    """Add issuer_name to positions dataframe from issuers using issuer_taxid as key to issuers' id."""
    if df_pos.empty or df_issuers.empty:
        return df_pos

    # Ensure issuer_taxid columns exist and are strings for matching
    if 'issuer_taxid' not in df_pos.columns:
        return df_pos

    df_pos['issuer_taxid'] = df_pos['issuer_taxid'].astype(str).str.strip()
    df_issuers['id'] = df_issuers['id'].astype(str).str.strip()

    # Drop NaN issuer_taxid from positions for merge
    df_pos_merge = df_pos.dropna(subset=['issuer_taxid'])

    # Merge with issuers on issuer_taxid to id
    df_pos_merged = df_pos_merge.merge(
        df_issuers[['id', 'name']],
        left_on='issuer_taxid',
        right_on='id',
        how='left',
        suffixes=('', '_iss')
    )

    # Add issuer_name
    df_pos_merged['issuer_name'] = df_pos_merged['name']

    # Clean up columns
    df_pos_merged.drop(['name', 'id'], axis=1, inplace=True, errors='ignore')

    # Re-merge back to original df_pos (including those without issuer_taxid)
    merge_keys = ['portfolio_id', 'security_id', 'issuer_taxid']
    df_pos = df_pos.merge(
        df_pos_merged[merge_keys + ['issuer_name']],
        on=merge_keys,
        how='left',
        suffixes=('', '_new')
    )

    # Prioritize new values
    if 'issuer_name_new' in df_pos.columns:
        df_pos['issuer_name'] = df_pos['issuer_name'].fillna(df_pos['issuer_name_new'])
        df_pos.drop('issuer_name_new', axis=1, inplace=True)

    # Log enrichment stats for debugging
    enriched_count = df_pos['issuer_name'].notna().sum()
    print(f"Positions enrichment: {len(df_pos)} rows, {enriched_count} matched with issuer name")

    return df_pos


def enrich_pmv_with_issuer_info(
    df_pmv: pd.DataFrame, df_positions: pd.DataFrame
) -> pd.DataFrame:
    """Add issuer_id, issuer_taxid, and issuer_name to PMV dataframe from
    positions using security_id as key."""
    if df_pmv.empty or df_positions.empty:
        return df_pmv

    # Ensure security_id columns exist and are strings for matching
    if 'security_id' not in df_pmv.columns or 'security_id' not in df_positions.columns:
        return df_pmv

    df_pmv['security_id'] = df_pmv['security_id'].astype(str).str.strip()
    df_positions['security_id'] = df_positions['security_id'].astype(str).str.strip()

    # Create lookup from security_id to issuer info
    # (keep first occurrence per security, handle NaNs)
    issuer_lookup = (
        df_positions[["security_id", "issuer_id", "issuer_taxid", "issuer_name"]]
        .dropna(subset=['security_id'])
        .drop_duplicates(subset=["security_id"])
        .set_index("security_id")
        .reindex(df_pmv['security_id'].unique(), fill_value=pd.NA)  # Ensure all PMV security_ids are considered
    )

    # Merge with PMV on security_id
    df_pmv_merged = df_pmv.merge(
        issuer_lookup[["issuer_id", "issuer_taxid", "issuer_name"]],
        left_on="security_id",
        right_index=True,
        how="left",
        suffixes=("", "_from_pos"),
    )

    # Log merge stats for debugging
    merged_count = df_pmv_merged['issuer_id'].notna().sum()
    print(f"PMV enrichment: {len(df_pmv)} rows, {len(issuer_lookup)} unique securities, {merged_count} matched with issuer info")

    return df_pmv_merged


# -------------------------------
# Config / Secrets
# -------------------------------


def get_api_key() -> Optional[str]:
    # Streamlit Cloud (secrets)
    if "GORILA_API_KEY" in st.secrets:
        return st.secrets["GORILA_API_KEY"]
    # Local/env
    return os.getenv("GORILA_API_KEY")


# -------------------------------
# Debug sink
# -------------------------------


class DebugSink:
    def __init__(self, enabled: bool):
        self.enabled = enabled
        self._placeholder = st.empty()
        self._buffer: List[str] = []

        # Optional file logger when enabled
        self.logger = logging.getLogger("gorila_debug")
        self.logger.setLevel(
            logging.INFO if enabled else logging.WARNING
        )

        if enabled:
            has_rotating = any(
                isinstance(h, RotatingFileHandler)
                for h in self.logger.handlers
            )
            if not has_rotating:
                handler = RotatingFileHandler(
                    "gorila_debug.log",
                    maxBytes=2_000_000,
                    backupCount=2,
                    encoding="utf-8",
                )
                fmt = logging.Formatter(
                    "%(asctime)s | %(levelname)s | %(message)s"
                )
                handler.setFormatter(fmt)
                self.logger.addHandler(handler)

    def log(self, msg: str):
        if not self.enabled:
            return
        self._buffer.append(msg)
        # Keep last 200 lines to avoid UI overload
        self._buffer = self._buffer[-200:]
        # Show in-app
        self._placeholder.code("\n".join(self._buffer), language="text")
        # File
        try:
            self.logger.info(msg)
        except Exception:
            pass  # never break the app due to logging issues


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

    def _request(
        self, method: str, path_or_url: str, params: Dict = None
    ) -> Dict:
        url = (
            path_or_url
            if str(path_or_url).startswith("http")
            else urljoin(
                self.base_url + "/", str(path_or_url).lstrip("/")
            )
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
                        f"{wait:.2f}s (attempt {attempt+1}/"
                        f"{self.max_retries})"
                    )
                    time.sleep(wait)
                    last_err = Exception(
                        f"HTTP {resp.status_code}: {resp.text[:200]}"
                    )
                    continue

                if 400 <= resp.status_code < 500:
                    preview = resp.text[:200].replace("\n", " ")
                    self._dbg(
                        f"Non-retryable {resp.status_code}. "
                        f"Body: {preview}"
                    )
                    resp.raise_for_status()

                if not resp.text:
                    return {}
                try:
                    return resp.json()
                except ValueError:
                    preview = resp.text[:200].replace("\n", " ")
                    self._dbg(
                        f"Non-JSON response preview: {preview}"
                    )
                    return {}

            except requests.HTTPError:
                raise
            except requests.RequestException as e:
                latency = (time.perf_counter() - start) * 1000
                self._dbg(
                    f"RequestException after {int(latency)}ms on "
                    f"{url}: {e}"
                )
                last_err = e
                wait = self.backoff_base * (2**attempt)
                self._dbg(
                    f"Backing off {wait:.2f}s (attempt "
                    f"{attempt+1}/{self.max_retries})"
                )
                time.sleep(wait)

        raise RuntimeError(
            f"Request failed after retries: {last_err}"
        ) from last_err

    def _normalize_next(self, nxt: Optional[str]) -> Optional[str]:
        if not nxt:
            return None
        nxt = str(nxt).strip()
        return nxt or None

    def _extract_page_token(self, url_or_path: str) -> str:
        try:
            from urllib.parse import urlparse, parse_qs

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

    def _describe_and_count_page(
        self, payload: Dict, page_idx: int, total_before: int
    ):
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
            self._dbg(
                f"Page {page_idx}: list payload of {len(payload)} "
                f"(no next)"
            )
            return len(payload), None
        else:
            nxt = (
                payload.get("next")
                if isinstance(payload, dict)
                else None
            )
            self._dbg(
                f"Page {page_idx}: single/non-standard payload; "
                f"next={bool(nxt)}"
            )
            return 1 if payload else 0, nxt

    def _paginate(
        self, path: str, params: Dict = None
    ) -> Iterable[Dict]:
        page_idx = 1
        total = 0
        seen_cursors = set()
        empty_page_streak = 0

        payload = self._request("GET", path, params=params)
        count, nxt = self._describe_and_count_page(
            payload, page_idx, total
        )
        total += count
        for r in self._iter_records(payload):
            yield r

        nxt = self._normalize_next(nxt)
        while nxt:
            token = self._extract_page_token(nxt) or nxt
            if token in seen_cursors:
                self._dbg(
                    f"Detected repeated cursor on page "
                    f"{page_idx+1}; breaking to avoid loop. "
                    f"token={token}"
                )
                break
            seen_cursors.add(token)

            page_idx += 1
            payload = self._request("GET", nxt, params=None)
            count, nxt = self._describe_and_count_page(
                payload, page_idx, total
            )
            total += count

            if count == 0:
                empty_page_streak += 1
                if empty_page_streak >= 1:
                    self._dbg(
                        "Empty page encountered during pagination; "
                        "stopping to avoid loop."
                    )
                    break
            else:
                empty_page_streak = 0

            for r in self._iter_records(payload):
                yield r

            nxt = self._normalize_next(nxt)

    # -------- Specific endpoints --------

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
        """
        List Position Market Values for a portfolio.

        Correct path per docs:
          GET /portfolios/{portfolioId}/positions/market-values

        Optional: referenceDate=YYYY-MM-DD
        """
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
    return {
        "id": p.get("id"),
        "name": p.get("name"),
        "autoRnC": p.get("autoRnC"),
    }


def flatten_issuer(i: Dict) -> Dict:
    return {
        "id": i.get("id"),
        "name": i.get("name"),
        "taxId": i.get("taxId") or i.get("cnpj"),
    }


def flatten_position(portfolio_id: str, pos: Dict) -> Dict:
    sec = pos.get("security", {}) or {}
    bro = pos.get("broker", {}) or {}

    issuer_id = sec.get("issuerId")
    issuer_tax = sec.get("issuer")

    return {
        "portfolio_id": portfolio_id,
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
        "broker_id": bro.get("id"),
        "broker_name": bro.get("name"),
        "issuer_id": issuer_id,
        "issuer_taxid": issuer_tax,
        "issuer_name": None,
        "raw": json.dumps(pos, ensure_ascii=False),
    }


def flatten_position_market_value(
    portfolio_id: str, item: Dict
) -> Dict:
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
        "issuer_id": None,
        "issuer_taxid": None,
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
def cached_list_portfolios(
    api_key: str, enable_debug: bool
) -> pd.DataFrame:
    client = GorilaClient(
        api_key,
        debug_sink=debug_sink if enable_debug else None,
    )
    rows = [flatten_portfolio(p) for p in client.list_portfolios()]
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False, ttl=3600)
def cached_list_issuers(
    api_key: str, enable_debug: bool
) -> pd.DataFrame:
    client = GorilaClient(
        api_key,
        debug_sink=debug_sink if enable_debug else None,
    )
    rows = [flatten_issuer(i) for i in client.list_issuers()]
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False, ttl=600)
def cached_list_positions(
    api_key: str,
    portfolio_id: str,
    reference_date: Optional[str],
    enable_debug: bool,
) -> pd.DataFrame:
    client = GorilaClient(
        api_key,
        debug_sink=debug_sink if enable_debug else None,
    )
    rows = [
        flatten_position(portfolio_id, p)
        for p in client.list_positions(portfolio_id, reference_date)
    ]
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False, ttl=600)
def cached_list_position_market_values(
    api_key: str,
    portfolio_id: str,
    reference_date: Optional[str],
    enable_debug: bool,
) -> pd.DataFrame:
    client = GorilaClient(
        api_key,
        debug_sink=debug_sink if enable_debug else None,
    )
    rows = [
        flatten_position_market_value(portfolio_id, p)
        for p in client.list_position_market_values(
            portfolio_id, reference_date
        )
    ]
    return pd.DataFrame(rows)


# -------------------------------
# Streamlit UI
# -------------------------------


st.set_page_config(page_title="Gorila Loader", layout="wide")
st.title("Gorila CORE → DB Loader")

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

    db_path = st.text_input("SQLite DB path", value="gorila.db")

    # Default reference date to 3 working days ago
    default_ref_date = get_3_working_days_ago()
    ref_date = st.date_input(
        "Positions/PMV reference date (optional)",
        value=default_ref_date,
        min_value=date(1990, 1, 1),
    )
    ref_date_str = ref_date.isoformat() if ref_date else None

    # Debug toggle
    debug = st.toggle("Debug mode", value=False)

# Initialize debug sink
debug_sink = DebugSink(debug)

if not api_key:
    st.warning(
        "Provide your Gorila API key via st.secrets['GORILA_API_KEY'], "
        "the GORILA_API_KEY env var, or the sidebar field."
    )
    st.stop()

st.write("Authentication looks good.")

# Collapsible panel to show debug logs
st.subheader("Debug output")
with st.expander("Show debug log", expanded=debug):
    pass

# Portfolios
st.header("Portfolios")
download_all_portfolios = st.toggle(
    "Download all (CSV) — portfolios", value=False
)

if st.button("Fetch portfolios"):
    t0 = time.perf_counter()
    with st.spinner("Loading portfolios..."):
        df_port = cached_list_portfolios(api_key, debug)
    st.dataframe(df_port, use_container_width=True)
    persist_df(db_path, "portfolios", df_port)
    st.session_state["df_portfolios"] = df_port
    t1 = time.perf_counter()
    debug_sink.log(
        f"Portfolios: fetched {len(df_port)} rows in "
        f"{int((t1 - t0) * 1000)}ms and persisted to SQLite"
    )
    st.caption("Saved to table: portfolios")

if (
    download_all_portfolios
    and st.session_state.get("df_portfolios") is not None
):
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
    t1 = time.perf_counter()
    debug_sink.log(
        f"Issuers: fetched {len(df_iss)} rows in "
        f"{int((t1 - t0) * 1000)}ms and persisted to SQLite"
    )
    st.caption("Saved to table: issuers")

if (
    st.session_state.get("df_issuers") is not None
):
    st.download_button(
        label="Download all issuers (CSV)",
        data=df_to_csv_bytes(st.session_state["df_issuers"]),
        file_name="issuers.csv",
        mime="text/csv;charset=utf-8",
    )

# Positions
st.header("Positions")
use_all_portfolios_for_positions = st.toggle(
    "Use all portfolios for positions", value=False
)

df_port_tbl = read_table(db_path, "portfolios")
selected_ids = []
if not use_all_portfolios_for_positions:
    if not df_port_tbl.empty:
        st.caption("Select one or more portfolios to fetch positions for:")
        choices = (
            df_port_tbl["id"]
            + " | "
            + df_port_tbl["name"].fillna("")
        )
        selected = st.multiselect(
            "Portfolios", options=choices.tolist()
        )
        selected_ids = [s.split(" | ", 1)[0] for s in selected]
    else:
        st.info(
            "Load portfolios first to enable a selection here. "
            "Alternatively, paste portfolio IDs below."
        )

    manual_ids = st.text_input(
        "Or paste portfolio IDs (comma-separated)", value=""
    )
    if manual_ids.strip():
        selected_ids += [
            s.strip() for s in manual_ids.split(",") if s.strip()
        ]
else:
    if not df_port_tbl.empty:
        selected_ids = df_port_tbl["id"].dropna().astype(str).tolist()
        debug_sink.log(
            f"Using all portfolios for positions: "
            f"{len(selected_ids)} portfolios"
        )
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
            debug_sink.log(
                f"Starting positions fetch for portfolio {pid} "
                f"ref_date={ref_date_str}"
            )
            with st.spinner(f"Loading positions for {pid}..."):
                t0 = time.perf_counter()
                df = cached_list_positions(
                    api_key, pid, ref_date_str, debug
                )
                t1 = time.perf_counter()
                debug_sink.log(
                    f"Finished positions fetch for {pid}: {len(df)} rows "
                    f"in {int((t1 - t0) * 1000)}ms"
                )
            all_pos.append(df)
            progress.progress(min(idx / total, 1.0))

        t_concat = time.perf_counter()
        df_pos = (
            pd.concat(all_pos, ignore_index=True)
            if all_pos
            else pd.DataFrame()
        )
        t_persist_start = time.perf_counter()
        debug_sink.log(
            f"Concat positions took "
            f"{int((t_persist_start - t_concat) * 1000)}ms "
            f"total_rows={len(df_pos)}"
        )

        st.dataframe(df_pos, use_container_width=True)
        persist_df(db_path, "positions", df_pos)
        st.session_state["df_positions"] = df_pos
        t_done = time.perf_counter()
        debug_sink.log(
            f"Persist positions took "
            f"{int((t_done - t_persist_start) * 1000)}ms"
        )
        st.caption("Saved to table: positions")

# Position Market Values
st.header("Position Market Values")
use_all_portfolios_for_pmv = st.toggle(
    "Use all portfolios for market values", value=False
)

pmv_selected_ids = []
if not use_all_portfolios_for_pmv:
    if not df_port_tbl.empty:
        st.caption("Select portfolios for position market values:")
        choices_pmv = (
            df_port_tbl["id"]
            + " | "
            + df_port_tbl["name"].fillna("")
        )
        selected_pmv = st.multiselect(
            "Portfolios (market values)",
            options=choices_pmv.tolist(),
            key="pmv_ms",
        )
        pmv_selected_ids = [
            s.split(" | ", 1)[0] for s in selected_pmv
        ]

    manual_pmv_ids = st.text_input(
        "Or paste portfolio IDs for market values "
        "(comma-separated)",
        value="",
    )
    if manual_pmv_ids.strip():
        pmv_selected_ids += [
            s.strip()
            for s in manual_pmv_ids.split(",")
            if s.strip()
        ]
else:
    if not df_port_tbl.empty:
        pmv_selected_ids = df_port_tbl["id"].dropna().astype(str).tolist()
        debug_sink.log(
            f"Using all portfolios for position market values: "
            f"{len(pmv_selected_ids)} portfolios"
        )
    else:
        st.info("Load portfolios first to use all.")

if st.button("Fetch position market values"):
    target_portfolio_ids = pmv_selected_ids

    if not target_portfolio_ids:
        st.warning(
            "No portfolio IDs selected to fetch market values."
        )
    else:
        rows: List[pd.DataFrame] = []
        progress = st.progress(0)
        total = len(target_portfolio_ids)
        for idx, pid in enumerate(target_portfolio_ids, start=1):
            debug_sink.log(
                f"Reading position market values for {pid} "
                f"ref_date={ref_date_str}"
            )
            try:
                t0 = time.perf_counter()
                df_one = cached_list_position_market_values(
                    api_key, pid, ref_date_str, debug
                )
                t1 = time.perf_counter()
                debug_sink.log(
                    f"PMV for {pid}: {len(df_one)} row(s) "
                    f"in {int((t1 - t0) * 1000)}ms"
                )
                rows.append(df_one)
            except requests.HTTPError as e:
                status = (
                    e.response.status_code
                    if getattr(e, "response", None)
                    else "?"
                )
                body = (
                    e.response.text[:200].replace("\n", " ")
                    if getattr(e, "response", None)
                    else ""
                )
                debug_sink.log(
                    f"HTTPError fetching PMV for {pid}: "
                    f"status={status} body={body}"
                )
            except Exception as e:
                debug_sink.log(f"Error fetching PMV for {pid}: {e}")

            progress.progress(min(idx / total, 1.0))

        df_pmv = (
            pd.concat(rows, ignore_index=True)
            if rows
            else pd.DataFrame()
        )

        # Enrich PMV with issuer info from positions
        enriched = False
        if (
            not df_pmv.empty
            and st.session_state.get("df_positions") is not None
            and not st.session_state["df_positions"].empty
        ):
            df_positions_local = st.session_state["df_positions"]
            df_pmv = enrich_pmv_with_issuer_info(
                df_pmv, df_positions_local
            )
            if df_pmv['issuer_id'].notna().any():
                enriched = True
                debug_sink.log("Enriched PMV with issuer info from positions")
            else:
                debug_sink.log("Enrichment attempted but no matches found - check if positions were fetched for the same portfolios/reference date.")
        else:
            st.info(
                "Positions not yet fetched or empty; PMV will not have issuer "
                "info. Fetch positions first (for the same portfolios and reference date) to enrich market values."
            )

        if enriched:
            st.success("PMV enriched with issuer information.")

        st.dataframe(df_pmv, use_container_width=True)
        persist_df(db_path, "position_market_values", df_pmv)
        st.session_state["df_position_market_values"] = df_pmv
        st.caption("Saved to table: position_market_values")

if (
    st.session_state.get("df_position_market_values") is not None
):
    st.download_button(
        label="Download all position market values (CSV)",
        data=df_to_csv_bytes(
            st.session_state["df_position_market_values"]
        ),
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