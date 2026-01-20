# app.py
# Streamlit app to generate "Relatório de Mercado"
# Focusing on Variable Income assets (Stocks, FIIs, REITs, etc.)

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
import unicodedata
import yfinance as yf

import numpy as np
import pandas as pd
from scipy import stats
import plotly.express as px
import streamlit as st
from jinja2 import Template

# Add after existing imports at the top
from pathlib import Path
import html as _html

# PDF Generation with Playwright
try:
    from playwright.sync_api import sync_playwright
    HAS_PLAYWRIGHT = True
except ImportError:
    HAS_PLAYWRIGHT = False


def html_to_pdf_a4(html_content: str) -> bytes:
    """Render HTML to A4 PDF using Playwright."""
    if not HAS_PLAYWRIGHT:
        raise ImportError("Playwright not installed. Run: pip install playwright && playwright install chromium")
    
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.set_viewport_size({"width": 794, "height": 1123})
        page.set_content(html_content, wait_until="load")
        page.wait_for_load_state("networkidle")
        
        # Wait for fonts
        try:
            page.wait_for_function(
                'document.fonts && document.fonts.status === "loaded"',
                timeout=10000
            )
        except:
            pass
        
        pdf_bytes = page.pdf(
            format="A4",
            print_background=True,
            margin={
                "top": "15mm",
                "right": "12mm",
                "bottom": "15mm",
                "left": "12mm"
            },
        )
        browser.close()
    
    return pdf_bytes

def html_to_pdf_single_page(html_content: str) -> bytes:
    """Render HTML to single-page PDF with dynamic height."""
    if not HAS_PLAYWRIGHT:
        raise ImportError("Playwright not installed")
    
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.set_viewport_size({"width": 794, "height": 1123})
        page.set_content(html_content, wait_until="load")
        page.wait_for_load_state("networkidle")
        
        try:
            page.wait_for_function(
                'document.fonts && document.fonts.status === "loaded"',
                timeout=10000
            )
        except:
            pass
        
        # Measure content height
        height = page.evaluate(
            "() => Math.ceil(document.querySelector('.page').scrollHeight + 20)"
        )
        
        pdf_bytes = page.pdf(
            print_background=True,
            prefer_css_page_size=False,
            width="210mm",
            height=f"{max(297, height)}px",
            margin={"top": "0", "right": "0", "bottom": "0", "left": "0"},
        )
        browser.close()
    
    return pdf_bytes

# Matplotlib for static images in report
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False

# Add these imports at the top (after existing imports)
import json
from pathlib import Path

# ---------------------------- Sector Override System ---------------------------- #

SECTOR_OVERRIDES_FILE = "sector_overrides.json"


def load_sector_overrides() -> Dict[str, str]:
    """Load custom sector mappings from file."""
    if os.path.exists(SECTOR_OVERRIDES_FILE):
        try:
            with open(SECTOR_OVERRIDES_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_sector_overrides(overrides: Dict[str, str]) -> None:
    """Save custom sector mappings to file."""
    with open(SECTOR_OVERRIDES_FILE, "w", encoding="utf-8") as f:
        json.dump(overrides, f, ensure_ascii=False, indent=2)


def get_sector_with_override(ticker: str, yf_sector: str) -> str:
    """Get sector, checking for user override first."""
    overrides = load_sector_overrides()
    clean_ticker = str(ticker).split()[0].upper().replace(".SA", "")
    
    if clean_ticker in overrides:
        return overrides[clean_ticker]
    
    return yf_sector if yf_sector and yf_sector != "Outros" else "Outros"


def save_sector_override(ticker: str, sector: str) -> None:
    """Save a single sector override."""
    overrides = load_sector_overrides()
    clean_ticker = str(ticker).split()[0].upper().replace(".SA", "")
    overrides[clean_ticker] = sector
    save_sector_overrides(overrides)


# Modify get_stock_info_yf to use overrides:
def get_stock_info_yf(ticker: str, asset_class: str = "") -> Dict:
    if not ticker or ticker == "nan":
        return {"name": "DESCONHECIDO", "sector": "Outros"}

    clean_ticker = str(ticker).split()[0]
    ac = str(asset_class).strip().upper()

    # Check for override first
    overrides = load_sector_overrides()
    override_key = clean_ticker.upper().replace(".SA", "")

    if ac == "OFFSHORE" or "^" in clean_ticker or clean_ticker.endswith(".SA"):
        yf_t = clean_ticker
    else:
        yf_t = f"{clean_ticker}.SA"

    try:
        stock = yf.Ticker(yf_t)
        info = stock.info
        name = info.get("longName") or info.get("shortName") or clean_ticker
        yf_sector = info.get("sector") or info.get("category") or "Outros"

        # Use override if exists
        sector = overrides.get(override_key, yf_sector) if overrides else yf_sector

        return {"name": name, "sector": sector}
    except:
        sector = overrides.get(override_key, "Outros") if overrides else "Outros"
        return {"name": clean_ticker, "sector": sector}
# ---------------------------- Config ---------------------------- #

st.set_page_config(
    page_title="Relatório de Mercado",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Authentication
if not st.session_state.get("authenticated", False):
    st.warning("Please enter the password on the Home page first.")
    st.write("Status: Não Autenticado")
    st.stop()

st.write("Autenticado")

PT_MONTHS = {
    "jan": 1, "fev": 2, "mar": 3, "abr": 4, "mai": 5, "jun": 6,
    "jul": 7, "ago": 8, "set": 9, "out": 10, "nov": 11, "dez": 12,
}

PALETA_CORES = ["#013220", "#8c6239", "#57575a", "#b08568", "#09202e", "#582308", "#7a6200"]

# Benchmark definitions
BENCHMARKS = {
    "local_stocks": {"ticker": "^BVSP", "label": "IBOV"},
    "offshore_stocks": {"ticker": "^GSPC", "label": "S&P 500"},
    "local_fiis": {"ticker": "XFIX11.SA", "label": "XFIX11"},
    "offshore_reits": {"ticker": "VNQ", "label": "VNQ"},
}

# ---------------------------- Helpers ---------------------------- #

def brl(v: float) -> str:
    return f"{v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

def as_pct_str(x: float) -> str:
    return f"{x * 100:.2f}%"

def pct(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator != 0 else 0.0

def safe_float(x) -> float:
    if isinstance(x, (int, float, np.number)): return float(x)
    if x is None or (isinstance(x, float) and np.isnan(x)): return 0.0
    s = str(x).strip().replace(".", "").replace(",", ".")
    try: return float(s)
    except: return 0.0

def first_nonempty(*vals) -> Optional[str]:
    for v in vals:
        if v is not None and str(v).strip() and str(v).strip() != "########":
            return str(v).strip()
    return None

def to_date_safe(x: str) -> Optional[datetime]:
    if pd.isna(x): return None
    if isinstance(x, datetime): return x
    s = str(x).strip()
    if not s or s == "########": return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y", "%d/%m/%y", "%Y%m%d"):
        try: return datetime.strptime(s, fmt)
        except: pass
    return None

# ---------------------------- Data Processing ---------------------------- #

def is_market_row(row: pd.Series) -> bool:
    ac = str(row.get("asset_class", "")).strip().upper()
    st_val = str(row.get("security_type", "")).strip().upper()
    if ac in ["STOCKS", "OFFSHORE"] and st_val not in ["FUNDQUOTE", "CUSTOM"]:
        return True
    return False



def get_stock_name_yf(ticker: str, asset_class: str = "") -> str:
    return get_stock_info_yf(ticker, asset_class)["name"]


def build_yf_ticker(ticker: str, asset_class: str) -> str:
    """Convert a ticker to Yahoo Finance format."""
    if not ticker:
        return ""
    
    clean = str(ticker).split()[0].strip().upper()
    ac = str(asset_class).strip().upper()
    
    # Already has suffix or is an index
    if clean.endswith(".SA") or "^" in clean:
        return clean
    
    # Offshore assets - don't add .SA
    if ac == "OFFSHORE":
        return clean
    
    # Brazilian assets
    return f"{clean}.SA"


def extract_ticker_from_row(row: pd.Series) -> str:
    """Extract ticker from row, checking multiple possible column names."""
    candidates = [
        row.get("security_name_clean"),
        row.get("ticker"),
        row.get("symbol"),
        row.get("security_name"),
        row.get("asset_name"),
        row.get("nome"),
    ]
    
    for val in candidates:
        if val is not None and str(val).strip() and str(val).strip().lower() != "nan":
            return str(val).split()[0].strip()
    
    return ""


def download_price_data(
    tickers: List[str], 
    benchmark: str, 
    start_date: datetime, 
    end_date: datetime,
    debug: bool = False
) -> Optional[pd.DataFrame]:
    """Download price data for tickers and benchmark, returning Adj Close DataFrame."""
    
    valid_tickers = list(set([t for t in tickers if t and t.strip()]))
    
    if not valid_tickers:
        if debug:
            st.warning("No valid tickers to download")
        return None
    
    all_tickers = valid_tickers + [benchmark]
    
    if debug:
        st.info(f"Downloading data for {len(valid_tickers)} tickers + benchmark ({benchmark})")
        st.caption(f"Tickers: {valid_tickers[:10]}{'...' if len(valid_tickers) > 10 else ''}")
    
    try:
        data = yf.download(
            all_tickers, 
            start=start_date.strftime("%Y-%m-%d"), 
            end=end_date.strftime("%Y-%m-%d"), 
            progress=False,
            auto_adjust=False,
            threads=True
        )
        
        if data.empty:
            if debug:
                st.warning("yfinance returned empty data")
            return None
        
        if isinstance(data.columns, pd.MultiIndex):
            adj_close = data["Adj Close"].copy()
        else:
            adj_close = pd.DataFrame(data["Adj Close"])
            adj_close.columns = [all_tickers[0]]
        
        if debug:
            st.success(f"Downloaded {len(adj_close)} rows, {len(adj_close.columns)} columns")
            st.caption(f"Available columns: {list(adj_close.columns)[:10]}")
        
        return adj_close
        
    except Exception as e:
        if debug:
            st.error(f"Error downloading data: {e}")
        return None


def calculate_portfolio_returns(
    adj_close: pd.DataFrame,
    tickers_data: List[Dict],
    benchmark: str,
    debug: bool = False
) -> Tuple[Optional[pd.Series], Optional[pd.Series], List[Dict], Dict]:
    """
    Calculate portfolio and benchmark returns.
    Returns: (portfolio_returns, benchmark_returns, valid_items_used, var_data)
    var_data contains: portfolio_weights, returns_matrix, portfolio_tickers, total_mv
    """
    
    empty_var_data = {"weights": None, "returns_matrix": None, "tickers": [], "total_mv": 0}
    
    if adj_close is None or adj_close.empty:
        return None, None, [], empty_var_data
    
    if benchmark not in adj_close.columns:
        if debug:
            st.warning(f"Benchmark {benchmark} not found in data")
        return None, None, [], empty_var_data
    
    ticker_mv = {}
    for item in tickers_data:
        ticker = item["yf_ticker"]
        if ticker and ticker in adj_close.columns:
            ticker_mv[ticker] = ticker_mv.get(ticker, 0) + item["mv"]
    
    if debug:
        missing = [x["yf_ticker"] for x in tickers_data 
                   if x["yf_ticker"] and x["yf_ticker"] not in adj_close.columns]
        missing = list(set(missing))
        if missing:
            st.caption(f"Missing tickers: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    
    if not ticker_mv:
        if debug:
            st.warning("No portfolio tickers found in downloaded data")
        return None, None, [], empty_var_data
    
    total_mv = sum(ticker_mv.values())
    if total_mv <= 0:
        return None, None, [], empty_var_data
    
    portfolio_tickers = list(ticker_mv.keys())
    portfolio_weights = np.array([ticker_mv[t] / total_mv for t in portfolio_tickers])
    
    cols_needed = portfolio_tickers + [benchmark]
    price_df = adj_close[cols_needed].ffill().bfill()
    price_df = price_df.dropna(subset=[benchmark])
    
    if len(price_df) < 5:
        if debug:
            st.warning(f"Not enough data points after cleaning: {len(price_df)}")
        return None, None, [], empty_var_data
    
    returns = price_df.pct_change().iloc[1:].fillna(0)
    
    if returns.empty or len(returns) < 5:
        if debug:
            st.warning(f"Not enough return data points: {len(returns)}")
        return None, None, [], empty_var_data
    
    returns_matrix = returns[portfolio_tickers].values
    
    if debug:
        st.caption(f"Matrix shape: {returns_matrix.shape}, weights shape: {portfolio_weights.shape}")
        st.caption(f"Total weight: {portfolio_weights.sum():.2%} across {len(portfolio_tickers)} unique tickers")
    
    port_returns_arr = returns_matrix @ portfolio_weights
    port_returns = pd.Series(port_returns_arr, index=returns.index)
    bench_returns = returns[benchmark]
    
    valid_items = [{"yf_ticker": t, "mv": ticker_mv[t]} for t in portfolio_tickers]
    
    # Data for VaR calculation
    var_data = {
        "weights": portfolio_weights,
        "returns_matrix": returns_matrix,
        "tickers": portfolio_tickers,
        "total_mv": total_mv,
    }
    
    return port_returns, bench_returns, valid_items, var_data


def calculate_backtest_metrics(
    port_returns: pd.Series,
    bench_returns: pd.Series,
    period_name: str,
    benchmark_label: str
) -> Dict:
    """Calculate all backtest metrics from returns series."""
    
    if port_returns is None or bench_returns is None:
        return {}
    
    if len(port_returns) < 5:
        return {}
    
    # Cumulative returns
    cum_port = (1 + port_returns).cumprod()
    cum_bench = (1 + bench_returns).cumprod()
    
    # Portfolio Drawdown
    rolling_max_port = cum_port.cummax()
    drawdown_port = (cum_port - rolling_max_port) / rolling_max_port
    drawdown_port = drawdown_port.replace([np.inf, -np.inf], 0).fillna(0)
    
    # Benchmark Drawdown
    rolling_max_bench = cum_bench.cummax()
    drawdown_bench = (cum_bench - rolling_max_bench) / rolling_max_bench
    drawdown_bench = drawdown_bench.replace([np.inf, -np.inf], 0).fillna(0)
    
    max_drawdown = float(drawdown_port.min()) if not drawdown_port.empty else 0.0
    avg_drawdown = float(drawdown_port.mean()) if not drawdown_port.empty else 0.0
    max_drawdown_bench = float(drawdown_bench.min()) if not drawdown_bench.empty else 0.0
    
    # Last 252 days drawdown
    if len(drawdown_port) > 252:
        drawdown_12m = float(drawdown_port.iloc[-252:].min())
        drawdown_12m_bench = float(drawdown_bench.iloc[-252:].min())
    else:
        drawdown_12m = max_drawdown
        drawdown_12m_bench = max_drawdown_bench
    
    # Beta and Volatility
    beta = 0.0
    if len(port_returns) > 1 and len(bench_returns) > 1:
        try:
            aligned = pd.DataFrame({"port": port_returns, "bench": bench_returns}).dropna()
            if len(aligned) > 1:
                matrix = np.cov(aligned["port"].values, aligned["bench"].values)
                if matrix[1, 1] != 0:
                    beta = matrix[0, 1] / matrix[1, 1]
        except:
            pass
    
    vol = float(port_returns.std() * np.sqrt(252)) if len(port_returns) > 1 else 0
    vol_bench = float(bench_returns.std() * np.sqrt(252)) if len(bench_returns) > 1 else 0
    
    # Sharpe (assuming 0 risk-free)
    mean_return = port_returns.mean() * 252
    sharpe = mean_return / vol if vol != 0 else 0
    
    return {
        "period": period_name,
        "dates": cum_port.index.tolist(), 
        "cum_port": cum_port.tolist(), 
        "cum_bench": cum_bench.tolist(),
        "drawdown_port": drawdown_port.tolist(),
        "drawdown_bench": drawdown_bench.tolist(),
        "drawdown_dates": drawdown_port.index.tolist(),
        "benchmark_name": benchmark_label,
        "max_drawdown": max_drawdown, 
        "max_drawdown_bench": max_drawdown_bench,
        "avg_drawdown": avg_drawdown,
        "drawdown_12m": drawdown_12m,
        "drawdown_12m_bench": drawdown_12m_bench,
        "beta": beta, 
        "volatility": vol,
        "volatility_bench": vol_bench,
        "sharpe": sharpe,
        "final_return": float(cum_port.iloc[-1] - 1) if not cum_port.empty else 0.0, 
        "final_bench_return": float(cum_bench.iloc[-1] - 1) if not cum_bench.empty else 0.0,
        "data_points": len(cum_port),
        "start_date": cum_port.index[0] if not cum_port.empty else None,
        "end_date": cum_port.index[-1] if not cum_port.empty else None,
    }


def calculate_var_metrics(
    port_returns: pd.Series,
    portfolio_weights: np.ndarray,
    returns_matrix: np.ndarray,
    portfolio_tickers: List[str],
    total_mv: float,
    confidence_levels: List[float] = [0.95, 0.99],
    n_simulations: int = 10000,
) -> Dict:
    """
    Calculate comprehensive VaR metrics using multiple methodologies.
    
    Returns a dict with:
    - Historical VaR (percentile-based)
    - Parametric VaR (assumes normal distribution)
    - Monte Carlo VaR (simulated scenarios)
    - Component VaR (contribution by asset)
    """
    
    if port_returns is None or len(port_returns) < 20:
        return {}
    
    result = {}
    
    # Daily portfolio mean and std
    port_mean = port_returns.mean()
    port_std = port_returns.std()
    
    for conf in confidence_levels:
        conf_pct = int(conf * 100)
        alpha = 1 - conf  # e.g., 0.05 for 95%
        
        # =====================================================
        # 1. HISTORICAL VaR - Uses actual percentile of returns
        # =====================================================
        historical_var = float(np.percentile(port_returns, alpha * 100))
        historical_var_pct = historical_var  # Already as daily return %
        historical_var_brl = abs(historical_var) * total_mv
        
        # =====================================================
        # 2. PARAMETRIC VaR - Assumes normal distribution
        # =====================================================
        z_score = stats.norm.ppf(alpha)  # e.g., -1.645 for 95%, -2.326 for 99%
        parametric_var = port_mean + z_score * port_std
        parametric_var_pct = float(parametric_var)
        parametric_var_brl = abs(parametric_var) * total_mv
        
        # =====================================================
        # 3. MONTE CARLO VaR - Simulated scenarios
        # =====================================================
        np.random.seed(42)  # For reproducibility
        simulated_returns = np.random.normal(port_mean, port_std, n_simulations)
        monte_carlo_var = float(np.percentile(simulated_returns, alpha * 100))
        monte_carlo_var_pct = monte_carlo_var
        monte_carlo_var_brl = abs(monte_carlo_var) * total_mv
        
        # =====================================================
        # 4. COMPONENT VaR - Contribution by asset
        # =====================================================
        component_var = []
        if returns_matrix is not None and len(returns_matrix) > 0:
            try:
                # Calculate covariance matrix
                cov_matrix = np.cov(returns_matrix.T)
                if cov_matrix.ndim == 0:
                    cov_matrix = np.array([[cov_matrix]])
                
                # Portfolio variance
                port_variance = portfolio_weights @ cov_matrix @ portfolio_weights
                port_vol = np.sqrt(port_variance)
                
                # Marginal VaR for each asset
                marginal_var = (cov_matrix @ portfolio_weights) / port_vol
                
                # Component VaR = weight * marginal VaR * z_score
                comp_var = portfolio_weights * marginal_var * abs(z_score)
                
                # Normalize to sum to total VaR
                total_comp = comp_var.sum()
                if total_comp != 0:
                    comp_var_normalized = comp_var / total_comp * abs(parametric_var)
                else:
                    comp_var_normalized = comp_var
                
                for i, ticker in enumerate(portfolio_tickers):
                    component_var.append({
                        "ticker": ticker.replace(".SA", ""),
                        "weight": float(portfolio_weights[i]),
                        "marginal_var": float(marginal_var[i]),
                        "component_var_pct": float(comp_var_normalized[i]),
                        "component_var_brl": float(abs(comp_var_normalized[i]) * total_mv),
                        "contribution_pct": float(comp_var[i] / total_comp * 100) if total_comp != 0 else 0,
                    })
                
                # Sort by absolute contribution
                component_var.sort(key=lambda x: abs(x["component_var_brl"]), reverse=True)
                
            except Exception as e:
                pass  # If calculation fails, just skip component VaR
        
        # Store results for this confidence level
        result[f"var_{conf_pct}"] = {
            "confidence": conf,
            "confidence_pct": conf_pct,
            # Historical VaR
            "historical_var_pct": historical_var_pct,
            "historical_var_brl": historical_var_brl,
            # Parametric VaR
            "parametric_var_pct": parametric_var_pct,
            "parametric_var_brl": parametric_var_brl,
            # Monte Carlo VaR
            "monte_carlo_var_pct": monte_carlo_var_pct,
            "monte_carlo_var_brl": monte_carlo_var_brl,
            # Component VaR (top 10)
            "component_var": component_var[:10] if component_var else [],
            # Metadata
            "z_score": float(z_score),
            "n_simulations": n_simulations,
        }
    
    # Add summary statistics
    result["var_summary"] = {
        "portfolio_mean_daily": float(port_mean),
        "portfolio_std_daily": float(port_std),
        "portfolio_mean_annual": float(port_mean * 252),
        "portfolio_std_annual": float(port_std * np.sqrt(252)),
        "total_mv": total_mv,
        "n_observations": len(port_returns),
    }
    
    return result


def calculate_backtest_multi_period(
    df: pd.DataFrame, 
    ref_date: datetime, 
    benchmark: str = "^BVSP",
    benchmark_label: str = "IBOV",
    debug: bool = False
) -> Dict[str, Dict]:
    """
    Calculate backtests for 1Y, 3Y, and 5Y periods.
    Returns dict with keys '1Y', '3Y', '5Y' containing results for each period.
    """
    
    if df.empty:
        if debug:
            st.warning("DataFrame is empty")
        return {}
    
    df = df.copy()
    
    if debug:
        st.write(f"**Debug: DataFrame has {len(df)} rows**")
        st.write(f"Columns: {list(df.columns)}")
    
    tickers_data = []
    for idx, r in df.iterrows():
        ticker = extract_ticker_from_row(r)
        if not ticker:
            continue
        
        mv = r.get("mv", 0)
        if mv <= 0:
            continue
            
        ac = str(r.get("asset_class", "")).strip()
        yf_ticker = build_yf_ticker(ticker, ac)
        
        if yf_ticker:
            tickers_data.append({
                "ticker": ticker, 
                "yf_ticker": yf_ticker, 
                "mv": mv,
                "name": r.get("parsed_company_name", ticker),
                "sector": r.get("sector", "Outros"),
            })
    
    if debug:
        st.write(f"**Built {len(tickers_data)} ticker entries**")
        if tickers_data:
            st.write(f"Sample: {tickers_data[:3]}")
    
    if not tickers_data:
        if debug:
            st.warning("No valid tickers found in DataFrame")
        return {}
    
    yf_tickers = list(set([x["yf_ticker"] for x in tickers_data]))
    
    periods = {
        "5Y": timedelta(days=int(365.25 * 5)),
        "3Y": timedelta(days=int(365.25 * 3)),
        "1Y": timedelta(days=365),
    }
    
    max_start = ref_date - periods["5Y"]
    
    if debug:
        st.write(f"**Downloading data from {max_start.date()} to {ref_date.date()}**")
    
    adj_close = download_price_data(yf_tickers, benchmark, max_start, ref_date, debug=debug)
    
    if adj_close is None or adj_close.empty:
        if debug:
            st.error("Failed to download price data")
        return {}
    
    results = {}
    
    for period_name, delta in periods.items():
        period_start = ref_date - delta
        period_data = adj_close[adj_close.index >= period_start].copy()
        
        if period_data.empty:
            if debug:
                st.warning(f"No data for period {period_name}")
            continue
        
        port_ret, bench_ret, valid_items, var_data = calculate_portfolio_returns(
            period_data, 
            tickers_data, 
            benchmark,
            debug=(debug and period_name == "1Y")
        )
        
        if port_ret is None or bench_ret is None:
            continue
        
        result = calculate_backtest_metrics(port_ret, bench_ret, period_name, benchmark_label)
        
        if result:
            total_mv = sum(x["mv"] for x in valid_items)
            composition = []
            for item in valid_items:
                orig = next((t for t in tickers_data if t["yf_ticker"] == item["yf_ticker"]), {})
                composition.append({
                    "ticker": item["yf_ticker"].replace(".SA", ""),
                    "yf_ticker": item["yf_ticker"],
                    "name": orig.get("name", item["yf_ticker"]),
                    "sector": orig.get("sector", "Outros"),
                    "mv": item["mv"],
                    "weight": item["mv"] / total_mv if total_mv > 0 else 0,
                })
            
            composition.sort(key=lambda x: x["weight"], reverse=True)
            
            result["composition"] = composition
            result["total_mv"] = total_mv
            result["n_assets"] = len(composition)
            
            # Calculate VaR metrics
            if var_data.get("weights") is not None:
                var_metrics = calculate_var_metrics(
                    port_returns=port_ret,
                    portfolio_weights=var_data["weights"],
                    returns_matrix=var_data["returns_matrix"],
                    portfolio_tickers=var_data["tickers"],
                    total_mv=var_data["total_mv"],
                    confidence_levels=[0.95, 0.99],
                    n_simulations=10000,
                )
                result["var"] = var_metrics
            
            results[period_name] = result
            if debug:
                st.success(f"✓ {period_name}: {result['data_points']} data points, return={as_pct_str(result['final_return'])}")
    
    return results


def classify_asset_detailed(row: pd.Series) -> str:
    st_val = str(row.get("security_type", "")).strip().upper()
    ac = str(row.get("asset_class", "")).strip().upper()
    
    if ac == "OFFSHORE":
        if "STOCK" in st_val: return "Ações US"
        if "REIT" in st_val: return "REIT"
        if "ETF" in st_val: return "ETF US"
        return "Offshore Outros"
    elif ac == "STOCKS":
        if "STOCK_LOCAL" in st_val: return "Ações"
        if "FII" in st_val: return "FII"
        if "STOCK_ETF" in st_val: return "ETF"
        if "STOCK_RECEIPT" in st_val: return "Recibos"
        return "Ações Outros"
    return "Outros"


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    
    mv_col = next((c for c in ["market_value_amount", "market_value", "valor_mercado"] if c in df.columns), None)
    df["mv"] = df[mv_col].apply(safe_float) if mv_col else 0.0
    
    df["is_market"] = df.apply(is_market_row, axis=1)
    df["is_fund"] = df["asset_class"].astype(str).str.upper().isin(["FIXED INCOME FUND", "FUNDQUOTE"])
    
    df["detailed_class"] = df.apply(classify_asset_detailed, axis=1)
    df["exposure_group"] = np.where(
        df["asset_class"].astype(str).str.upper() == "OFFSHORE", 
        "Exposição Offshore", 
        "Exposição Local"
    )
    
    df["reference_dt"] = df["reference_date"].apply(to_date_safe)
    
    def get_curr(r):
        v = first_nonempty(r.get("currency"), r.get("moeda"), "BRL")
        return "BRL" if "REAL" in v.upper() or "BRL" in v.upper() else ("USD" if "DOL" in v.upper() or "USD" in v.upper() else v)
    df["currency_code"] = df.apply(get_curr, axis=1)
    
    df["class_bucket"] = np.select(
        [df["is_fund"], df["is_market"]],
        ["Fundos", "Ações/REITs/FIIs"],
        default="Outros"
    )
    return df


# ---------------------------- UI / Logic ---------------------------- #

st.sidebar.header("Fonte de dados")
data_source = st.sidebar.radio("Origem", ["SQLite (.db)", "CSV"])

if data_source == "CSV":
    uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if not uploaded:
        st.stop()
    df_raw = pd.read_csv(uploaded, sep=None, engine='python')
else:
    db_path = st.sidebar.text_input("DB Path", "databases/gorila_positions.db")
    if not os.path.exists(db_path):
        st.info("DB not found")
        st.stop()
    with sqlite3.connect(db_path) as conn:
        df_raw = pd.read_sql_query('SELECT * FROM pmv_plus_gorila', conn)

df = normalize_dataframe(df_raw)

# Filter Reference Date
ref_dates = sorted(df["reference_dt"].dropna().unique())
if not ref_dates:
    st.error("No dates found")
    st.stop()
ref_dt = st.sidebar.selectbox("Data de Referência", ref_dates, index=len(ref_dates)-1)
df = df[df["reference_dt"] == ref_dt]
period_label = ref_dt.strftime("%B de %Y").title()

# Debug mode toggle
debug_mode = st.sidebar.checkbox("Debug Mode", value=False)

# Base logic
base_df = df[df["is_market"]].copy()
pl_total = df["mv"].sum()
pl_market = base_df["mv"].sum()

# Track results and DataFrames initialization
local_stocks_results = {}
offshore_stocks_results = {}
local_fiis_results = {}
offshore_reits_results = {}

# Initialize DataFrames
local_stocks_df = pd.DataFrame()
offshore_stocks_df = pd.DataFrame()
local_fiis_df = pd.DataFrame()
offshore_reits_df = pd.DataFrame()
local_df = pd.DataFrame()
offshore_df = pd.DataFrame()

if not base_df.empty:
    with st.spinner("Resolvendo nomes, setores e calculando backtests..."):
        resolved_data = []
        for _, row in base_df.iterrows():
            ticker_raw = str(row.get("security_name", ""))
            ticker_clean = ticker_raw.split()[0] if ticker_raw else ""
            ac = str(row.get("asset_class", ""))
            info = get_stock_info_yf(ticker_clean, ac)
            resolved_data.append({
                "security_name_clean": ticker_clean,
                "parsed_company_name": info["name"],
                "sector": info["sector"]
            })
        
        base_df["security_name_clean"] = [x["security_name_clean"] for x in resolved_data]
        base_df["parsed_company_name"] = [x["parsed_company_name"] for x in resolved_data]
        base_df["sector"] = [x["sector"] for x in resolved_data]
        
        # Split into 4 Tracks
        # Track 1: Local Stocks/ETFs
        local_stocks_df = base_df[
            (base_df["detailed_class"].isin(["Ações", "Recibos", "ETF", "Ações Outros"])) &
            (base_df["exposure_group"] == "Exposição Local")
        ].copy()
        
        # Track 2: Offshore Stocks/ETFs
        offshore_stocks_df = base_df[
            (base_df["detailed_class"].isin(["Ações US", "ETF US", "Offshore Outros"])) &
            (base_df["exposure_group"] == "Exposição Offshore")
        ].copy()
        
        # Track 3: Local FIIs
        local_fiis_df = base_df[
            (base_df["detailed_class"] == "FII") &
            (base_df["exposure_group"] == "Exposição Local")
        ].copy()
        
        # Track 4: Offshore REITs
        offshore_reits_df = base_df[
            (base_df["detailed_class"] == "REIT") &
            (base_df["exposure_group"] == "Exposição Offshore")
        ].copy()
        
        # Combined DataFrames for Sector Visualization
        local_df = pd.concat([local_stocks_df, local_fiis_df], ignore_index=True)
        offshore_df = pd.concat([offshore_stocks_df, offshore_reits_df], ignore_index=True)
        
        if debug_mode:
            st.markdown("### Debug: Data Overview")
            st.write(f"Base DF: {len(base_df)} rows")
            st.write(f"Local Stocks DF: {len(local_stocks_df)} rows")
            st.write(f"Offshore Stocks DF: {len(offshore_stocks_df)} rows")
            st.write(f"Local FIIs DF: {len(local_fiis_df)} rows")
            st.write(f"Offshore REITs DF: {len(offshore_reits_df)} rows")
        
        # Performance Calculations - Multi Period for each track
        
        # Track 1: Local Stocks vs IBOV
        if debug_mode:
            st.markdown("### Debug: Local Stocks Backtest")
        local_stocks_results = calculate_backtest_multi_period(
            local_stocks_df, 
            ref_dt, 
            benchmark=BENCHMARKS["local_stocks"]["ticker"],
            benchmark_label=BENCHMARKS["local_stocks"]["label"],
            debug=debug_mode
        )
        
        # Track 2: Offshore Stocks vs S&P 500
        if debug_mode:
            st.markdown("### Debug: Offshore Stocks Backtest")
        offshore_stocks_results = calculate_backtest_multi_period(
            offshore_stocks_df, 
            ref_dt, 
            benchmark=BENCHMARKS["offshore_stocks"]["ticker"],
            benchmark_label=BENCHMARKS["offshore_stocks"]["label"],
            debug=debug_mode
        )
        
        # Track 3: Local FIIs vs XFIX11
        if debug_mode:
            st.markdown("### Debug: Local FIIs Backtest")
        local_fiis_results = calculate_backtest_multi_period(
            local_fiis_df, 
            ref_dt, 
            benchmark=BENCHMARKS["local_fiis"]["ticker"],
            benchmark_label=BENCHMARKS["local_fiis"]["label"],
            debug=debug_mode
        )
        
        # Track 4: Offshore REITs vs VNQ
        if debug_mode:
            st.markdown("### Debug: Offshore REITs Backtest")
        offshore_reits_results = calculate_backtest_multi_period(
            offshore_reits_df, 
            ref_dt, 
            benchmark=BENCHMARKS["offshore_reits"]["ticker"],
            benchmark_label=BENCHMARKS["offshore_reits"]["label"],
            debug=debug_mode
        )


def display_backtest_section(results: Dict[str, Dict], title: str, df_track: pd.DataFrame, default_period: str = "1Y"):
    """Display backtest results with period selector."""
    
    if not results:
        st.warning(f"Sem dados suficientes para backtest de {title}.")
        
        if not df_track.empty:
            st.markdown("**Ativos na carteira:**")
            summary = df_track.groupby("security_name_clean")["mv"].sum().sort_values(ascending=False).head(10)
            st.dataframe(summary.reset_index())
        return
    
    available_periods = list(results.keys())
    
    selected_period = st.selectbox(
        f"Período - {title}",
        available_periods,
        index=available_periods.index(default_period) if default_period in available_periods else 0,
        key=f"period_{title}"
    )
    
    res = results.get(selected_period, {})
    
    if not res:
        st.warning(f"Dados não disponíveis para {selected_period}")
        return
    
    bench_name = res.get('benchmark_name', 'Benchmark')
    
    # Metrics row
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Retorno Acum", as_pct_str(res.get('final_return', 0)))
    m2.metric(f"Retorno {bench_name}", as_pct_str(res.get('final_bench_return', 0)))
    m3.metric("Sharpe", f"{res.get('sharpe', 0):.2f}")
    m4.metric("Volatilidade", as_pct_str(res.get('volatility', 0)))
    m5.metric("Max DD", as_pct_str(res.get('max_drawdown', 0)))
    
    # Additional info row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Beta", f"{res.get('beta', 0):.2f}")
    col2.metric("DD Médio", as_pct_str(res.get('avg_drawdown', 0)))
    col3.metric("DD 12m", as_pct_str(res.get('drawdown_12m', 0)))
    col4.metric(f"Max DD {bench_name}", as_pct_str(res.get('max_drawdown_bench', 0)))
    
    # VaR Section with tooltips
    var_data = res.get('var', {})
    if var_data:
        st.markdown("---")
        st.markdown("### 📊 Value at Risk (VaR)")
        
        # Methodology tooltips
        methodology_info = """
        **Metodologias de VaR:**
        
        🔹 **VaR Histórico**: Utiliza o percentil real dos retornos históricos. Não assume distribuição normal, 
        capturando eventos de cauda ("fat tails"). É o método mais conservador em mercados voláteis.
        
        🔹 **VaR Paramétrico** (Variância-Covariância): Assume que os retornos seguem distribuição normal. 
        Usa a média e desvio padrão dos retornos multiplicados pelo z-score. Mais simples, mas pode 
        subestimar riscos em eventos extremos.
        
        🔹 **VaR Monte Carlo**: Simula 10.000 cenários de retornos baseados na distribuição observada. 
        Combina características dos métodos anteriores e é útil para portfólios complexos.
        
        🔹 **VaR por Componente**: Mostra a contribuição de cada ativo para o VaR total do portfólio.
        Calculado via VaR Marginal × Peso de cada ativo.
        
        **Interpretação**: Um VaR 95% de -2% significa que há 95% de confiança de que a perda diária 
        não excederá 2% (ou, equivalentemente, 5% de chance de perder mais que 2%).
        """
        
        with st.expander("ℹ️ Entenda as Metodologias de VaR", expanded=False):
            st.markdown(methodology_info)
        
        # VaR 95% and 99% side by side
        var_col1, var_col2 = st.columns(2)
        
        with var_col1:
            st.markdown("#### VaR 95% (Diário)")
            var_95 = var_data.get("var_95", {})
            if var_95:
                v1, v2, v3 = st.columns(3)
                v1.metric(
                    "Histórico", 
                    as_pct_str(var_95.get('historical_var_pct', 0)),
                    help="Percentil 5% dos retornos históricos reais"
                )
                v2.metric(
                    "Paramétrico", 
                    as_pct_str(var_95.get('parametric_var_pct', 0)),
                    help=f"μ + z×σ (z = {var_95.get('z_score', 0):.3f})"
                )
                v3.metric(
                    "Monte Carlo", 
                    as_pct_str(var_95.get('monte_carlo_var_pct', 0)),
                    help=f"Simulação de {var_95.get('n_simulations', 10000):,} cenários"
                )
                
                # VaR in BRL
                st.caption(f"💰 VaR em R$: Hist. {brl(var_95.get('historical_var_brl', 0))} | "
                          f"Param. {brl(var_95.get('parametric_var_brl', 0))} | "
                          f"MC {brl(var_95.get('monte_carlo_var_brl', 0))}")
        
        with var_col2:
            st.markdown("#### VaR 99% (Diário)")
            var_99 = var_data.get("var_99", {})
            if var_99:
                v1, v2, v3 = st.columns(3)
                v1.metric(
                    "Histórico", 
                    as_pct_str(var_99.get('historical_var_pct', 0)),
                    help="Percentil 1% dos retornos históricos reais"
                )
                v2.metric(
                    "Paramétrico", 
                    as_pct_str(var_99.get('parametric_var_pct', 0)),
                    help=f"μ + z×σ (z = {var_99.get('z_score', 0):.3f})"
                )
                v3.metric(
                    "Monte Carlo", 
                    as_pct_str(var_99.get('monte_carlo_var_pct', 0)),
                    help=f"Simulação de {var_99.get('n_simulations', 10000):,} cenários"
                )
                
                # VaR in BRL
                st.caption(f"💰 VaR em R$: Hist. {brl(var_99.get('historical_var_brl', 0))} | "
                          f"Param. {brl(var_99.get('parametric_var_brl', 0))} | "
                          f"MC {brl(var_99.get('monte_carlo_var_brl', 0))}")
        
        # Component VaR
        var_95 = var_data.get("var_95", {})
        component_var = var_95.get("component_var", [])
        if component_var:
            with st.expander("🔍 VaR por Componente (Top 10 contribuintes)", expanded=False):
                st.markdown("""
                **Contribuição de cada ativo para o VaR total:**
                - **VaR Comp. (%)**: Contribuição percentual do ativo para o VaR diário
                - **VaR Comp. (R$)**: Valor absoluto de contribuição para o VaR
                - **Contrib. (%)**: Participação relativa no VaR total (soma = 100%)
                """)
                
                comp_df = pd.DataFrame([
                    {
                        "Ticker": c["ticker"],
                        "Peso (%)": as_pct_str(c["weight"]),
                        "VaR Comp. (%)": as_pct_str(c["component_var_pct"]),
                        "VaR Comp. (R$)": brl(c["component_var_brl"]),
                        "Contrib. (%)": f"{c['contribution_pct']:.1f}%",
                    }
                    for c in component_var
                ])
                st.dataframe(comp_df, use_container_width=True, hide_index=True)
        
        # Summary statistics
        var_summary = var_data.get("var_summary", {})
        if var_summary:
            st.caption(
                f"📈 Retorno médio diário: {as_pct_str(var_summary.get('portfolio_mean_daily', 0))} | "
                f"Vol. diária: {as_pct_str(var_summary.get('portfolio_std_daily', 0))} | "
                f"Observações: {var_summary.get('n_observations', 0)}"
            )
        
        st.markdown("---")
    
    # Data availability info
    start_dt = res.get('start_date')
    end_dt = res.get('end_date')
    data_points = res.get('data_points', 0)
    
    if start_dt and end_dt:
        start_str = start_dt.strftime("%d/%m/%Y") if hasattr(start_dt, 'strftime') else str(start_dt)[:10]
        end_str = end_dt.strftime("%d/%m/%Y") if hasattr(end_dt, 'strftime') else str(end_dt)[:10]
        st.caption(f"Dados: {start_str} a {end_str} ({data_points} pontos)")
    
    # Portfolio Composition Expander
    with st.expander(f"📊 Composição do Portfólio ({res.get('n_assets', 0)} ativos)", expanded=False):
        composition = res.get("composition", [])
        if composition:
            c1, c2, c3 = st.columns(3)
            c1.metric("Nº de Ativos", res.get('n_assets', 0))
            c2.metric("PL Total", brl(res.get('total_mv', 0)))
            c3.metric("Maior Peso", as_pct_str(composition[0]['weight']) if composition else "N/A")
            
            st.markdown("**Top 10 Ativos por Peso:**")
            top_10 = composition[:10]
            top_df = pd.DataFrame([
                {
                    "Ticker": c["ticker"],
                    "Nome": c["name"][:40] + "..." if len(c["name"]) > 40 else c["name"],
                    "Setor": c["sector"],
                    "Valor (R$)": brl(c["mv"]),
                    "Peso (%)": as_pct_str(c["weight"]),
                }
                for c in top_10
            ])
            st.dataframe(top_df, use_container_width=True, hide_index=True)
            
            st.markdown("**Concentração por Setor:**")
            sector_data = {}
            for c in composition:
                sector = c["sector"]
                sector_data[sector] = sector_data.get(sector, 0) + c["mv"]
            
            total_mv = res.get('total_mv', 1)
            sector_df = pd.DataFrame([
                {"Setor": s, "Valor (R$)": brl(v), "Peso (%)": as_pct_str(v / total_mv)}
                for s, v in sorted(sector_data.items(), key=lambda x: x[1], reverse=True)
            ])
            st.dataframe(sector_df, use_container_width=True, hide_index=True)
            
            if len(composition) > 10:
                with st.expander(f"Ver todos os {len(composition)} ativos"):
                    full_df = pd.DataFrame([
                        {
                            "Ticker": c["ticker"],
                            "Nome": c["name"],
                            "Setor": c["sector"],
                            "Valor (R$)": brl(c["mv"]),
                            "Peso (%)": as_pct_str(c["weight"]),
                        }
                        for c in composition
                    ])
                    st.dataframe(full_df, use_container_width=True, hide_index=True)
            
            csv_data = pd.DataFrame([
                {
                    "ticker": c["ticker"],
                    "yf_ticker": c["yf_ticker"],
                    "name": c["name"],
                    "sector": c["sector"],
                    "mv": c["mv"],
                    "weight": c["weight"],
                }
                for c in composition
            ])
            csv_str = csv_data.to_csv(index=False)
            st.download_button(
                f"⬇️ Download Composição CSV",
                csv_str,
                f"composition_{title.replace(' ', '_')}_{selected_period}.csv",
                "text/csv",
                key=f"download_comp_{title}_{selected_period}"
            )
        else:
            st.info("Composição não disponível")
    
    # Performance chart
    if res.get("dates") and res.get("cum_port") and res.get("cum_bench"):
        plot_df = pd.DataFrame({
            "Data": res["dates"],
            "Carteira": res["cum_port"],
            bench_name: res["cum_bench"]
        })
        
        fig = px.line(
            plot_df, 
            x="Data", 
            y=["Carteira", bench_name],
            title=f"{title} - Performance {selected_period}",
            labels={"value": "Fator Acumulado", "variable": "Série"},
            color_discrete_sequence=PALETA_CORES
        )
        fig.update_layout(hovermode="x unified", colorway=PALETA_CORES)
        st.plotly_chart(fig, use_container_width=True)
    
    # Drawdown chart - Now with benchmark comparison
    if res.get("drawdown_dates") and res.get("drawdown_port") and res.get("drawdown_bench"):
        dd_df = pd.DataFrame({
            "Data": res["drawdown_dates"],
            "Carteira": res["drawdown_port"],
            bench_name: res["drawdown_bench"]
        })
        
        fig_dd = px.line(
            dd_df,
            x="Data",
            y=["Carteira", bench_name],
            title=f"Drawdown - {title} vs {bench_name} ({selected_period})",
            labels={"value": "Drawdown (%)", "variable": "Série"},
            color_discrete_sequence=[PALETA_CORES[0], PALETA_CORES[2]]
        )
        fig_dd.update_traces(selector=dict(name="Carteira"), fill='tozeroy', line_color=PALETA_CORES[0])
        fig_dd.update_traces(selector=dict(name=bench_name), line_color=PALETA_CORES[2], line_dash='dash')
        fig_dd.update_layout(hovermode="x unified", colorway=PALETA_CORES)
        st.plotly_chart(fig_dd, use_container_width=True)
    
    # Comparison table for all periods
    st.markdown("**Comparativo de Períodos:**")
    comparison_data = []
    for p in ["1Y", "3Y", "5Y"]:
        r = results.get(p)
        if r:
            comparison_data.append({
                "Período": p,
                "Retorno": as_pct_str(r.get('final_return', 0)),
                f"Ret {r.get('benchmark_name', 'Bench')}": as_pct_str(r.get('final_bench_return', 0)),
                "Sharpe": f"{r.get('sharpe', 0):.2f}",
                "Vol": as_pct_str(r.get('volatility', 0)),
                "Max DD": as_pct_str(r.get('max_drawdown', 0)),
                f"Max DD {r.get('benchmark_name', 'Bench')}": as_pct_str(r.get('max_drawdown_bench', 0)),
                "Beta": f"{r.get('beta', 0):.2f}",
                "Ativos": r.get('n_assets', 0),
            })
    
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)


# Dashboard
st.title("Relatório de Mercado")
st.subheader("Resumo")
c1, c2, c3 = st.columns(3)
c1.metric("PL Total", brl(pl_total))
c2.metric("PL Mercado", brl(pl_market))
c3.metric("% Mercado/PL", as_pct_str(pct(pl_market, pl_total)))

# ---------------------------- Sector Editor ---------------------------- #

st.markdown("---")
st.subheader("Editor de Setores")

# Find assets with "Outros" sector
outros_assets = base_df[base_df["sector"] == "Outros"][
    ["security_name_clean", "parsed_company_name", "exposure_group", "mv"]
].drop_duplicates(subset=["security_name_clean"]).copy()

if not outros_assets.empty:
    st.info(f"**{len(outros_assets)} ativos** com setor não identificado. Edite abaixo para corrigir.")

    # Common sector options
    SECTOR_OPTIONS = [
        "Outros",
        "Financeiro",
        "Tecnologia",
        "Saúde",
        "Consumo Cíclico",
        "Consumo Não-Cíclico",
        "Energia",
        "Utilities",
        "Materiais Básicos",
        "Industrial",
        "Imobiliário",
        "Comunicação",
        "Real Estate",
        "Healthcare",
        "Technology",
        "Financial Services",
        "Consumer Cyclical",
        "Consumer Defensive",
        "Energy",
        "Basic Materials",
        "Industrials",
        "Communication Services",
    ]

    # Initialize session state for edits
    if "sector_edits" not in st.session_state:
        st.session_state.sector_edits = {}

    # Create editable table
    edited_sectors = []
    
    with st.form("sector_editor_form"):
        for idx, row in outros_assets.iterrows():
            ticker = row["security_name_clean"]
            col1, col2, col3 = st.columns([2, 3, 3])
            
            with col1:
                st.text(ticker)
            with col2:
                st.text(row["parsed_company_name"][:30] if row["parsed_company_name"] else "")
            with col3:
                new_sector = st.selectbox(
                    f"Setor para {ticker}",
                    options=SECTOR_OPTIONS,
                    index=0,
                    key=f"sector_{ticker}",
                    label_visibility="collapsed"
                )
                if new_sector != "Outros":
                    edited_sectors.append((ticker, new_sector))

        submitted = st.form_submit_button("💾 Salvar Setores")
        
        if submitted and edited_sectors:
            for ticker, sector in edited_sectors:
                save_sector_override(ticker, sector)
            st.success(f"✅ {len(edited_sectors)} setor(es) salvo(s)! Recarregue a página para aplicar.")
            st.rerun()

else:
    st.success("✅ Todos os ativos têm setor identificado.")

# Show current overrides
with st.expander("📋 Ver/Editar Setores Salvos"):
    overrides = load_sector_overrides()
    if overrides:
        override_df = pd.DataFrame([
            {"Ticker": k, "Setor": v} for k, v in sorted(overrides.items())
        ])
        st.dataframe(override_df, use_container_width=True, hide_index=True)
        
        # Option to delete overrides
        ticker_to_delete = st.selectbox(
            "Remover override:",
            options=[""] + list(overrides.keys()),
            key="delete_override"
        )
        if ticker_to_delete and st.button("🗑️ Remover"):
            del overrides[ticker_to_delete]
            save_sector_overrides(overrides)
            st.success(f"Removido override para {ticker_to_delete}")
            st.rerun()
    else:
        st.info("Nenhum setor customizado salvo.")

# Track 1: Local Stocks
st.markdown("---")
st.header("1. Ações e ETFs Locais")
st.caption(f"Benchmark: {BENCHMARKS['local_stocks']['label']} ({BENCHMARKS['local_stocks']['ticker']})")
display_backtest_section(local_stocks_results, "Ações Locais", local_stocks_df, default_period="1Y")

# Track 2: Offshore Stocks
st.markdown("---")
st.header("2. Ações e ETFs Offshore")
st.caption(f"Benchmark: {BENCHMARKS['offshore_stocks']['label']} ({BENCHMARKS['offshore_stocks']['ticker']})")
display_backtest_section(offshore_stocks_results, "Ações Offshore", offshore_stocks_df, default_period="1Y")

# Track 3: Local FIIs
st.markdown("---")
st.header("3. FIIs Locais")
st.caption(f"Benchmark: {BENCHMARKS['local_fiis']['label']} ({BENCHMARKS['local_fiis']['ticker']})")
display_backtest_section(local_fiis_results, "FIIs Locais", local_fiis_df, default_period="1Y")

# Track 4: Offshore REITs
st.markdown("---")
st.header("4. REITs Offshore")
st.caption(f"Benchmark: {BENCHMARKS['offshore_reits']['label']} ({BENCHMARKS['offshore_reits']['ticker']})")
display_backtest_section(offshore_reits_results, "REITs Offshore", offshore_reits_df, default_period="1Y")

# Concentration Analysis
st.markdown("---")
st.subheader("Concentração por Ativo")
if not base_df.empty:
    iss_df = base_df.groupby(["security_name_clean", "parsed_company_name", "exposure_group"])["mv"].sum().sort_values(ascending=False).reset_index()
    iss_df.columns = ["Ticker", "Nome", "Exposição", "Valor (R$)"]
    iss_df["% do PL"] = iss_df["Valor (R$)"].apply(lambda x: as_pct_str(pct(x, pl_total)))
    iss_df["Valor (R$)"] = iss_df["Valor (R$)"].apply(brl)
    st.dataframe(iss_df, use_container_width=True)



# ---------------------------- Sector Visualization ---------------------------- #

st.markdown("---")
st.subheader("Concentração por Setor")

col_local, col_offshore = st.columns(2)

with col_local:
    st.markdown("**Setores - Exposição Local**")
    if not local_df.empty:
        sec_local = local_df.groupby("sector")["mv"].sum().sort_values(ascending=False).reset_index()
        sec_local.columns = ["Setor", "Valor"]
        sec_local["% PL"] = sec_local["Valor"] / sec_local["Valor"].sum() * 100
        
        fig_sec_local = px.bar(
            sec_local,
            x="Valor",
            y="Setor",
            orientation="h",
            text=sec_local["% PL"].apply(lambda x: f"{x:.1f}%"),
            color="Setor",
            color_discrete_sequence=PALETA_CORES
        )
        fig_sec_local.update_layout(
            showlegend=False,
            yaxis=dict(categoryorder='total ascending'),
            height=400,
            colorway=PALETA_CORES
        )
        fig_sec_local.update_traces(textposition="outside")
        st.plotly_chart(fig_sec_local, use_container_width=True)
    else:
        st.info("Sem dados locais")

with col_offshore:
    st.markdown("**Setores - Exposição Offshore**")
    if not offshore_df.empty:
        sec_off = offshore_df.groupby("sector")["mv"].sum().sort_values(ascending=False).reset_index()
        sec_off.columns = ["Setor", "Valor"]
        sec_off["% PL"] = sec_off["Valor"] / sec_off["Valor"].sum() * 100
        
        fig_sec_off = px.bar(
            sec_off,
            x="Valor",
            y="Setor",
            orientation="h",
            text=sec_off["% PL"].apply(lambda x: f"{x:.1f}%"),
            color="Setor",
            color_discrete_sequence=PALETA_CORES
        )
        fig_sec_off.update_layout(
            showlegend=False,
            yaxis=dict(categoryorder='total ascending'),
            height=400,
            colorway=PALETA_CORES
        )
        fig_sec_off.update_traces(textposition="outside")
        st.plotly_chart(fig_sec_off, use_container_width=True)
    else:
        st.info("Sem dados offshore")

# ---------------------------- Report Generation ---------------------------- #

# ---------------------------- Report Generation ---------------------------- #

REPORT_HTML = """
<!DOCTYPE html>
<html lang="pt-br">
<head>
  <meta charset="UTF-8" />
  <title>Relatório de Monitoramento de Risco de Mercado</title>
  <link href="https://fonts.googleapis.com/css2?family=Open+Sans:wght@400;600;700&display=swap" rel="stylesheet">
  <style>
    @page {
      size: A4;
      margin: 15mm 12mm 15mm 12mm;
    }

    :root {
      --brand: #825120;
      --brand-dark: #6B4219;
      --bg: #F8F8F8;
      --text: #333333;
      --tbl-border: #E0D5CA;
      --light: #F5F5F5;
    }

    * { box-sizing: border-box; }

    html, body {
      margin: 0;
      padding: 0;
      -webkit-print-color-adjust: exact;
      print-color-adjust: exact;
    }

    body {
      font-family: 'Open Sans', 'Segoe UI', Arial, sans-serif;
      background: #f0f0f0;
      color: var(--text);
      padding: 10px;
      font-size: 10px;
    }

    .page {
      width: 210mm;
      min-height: 297mm;
      margin: 0 auto;
      background: white;
      box-shadow: 0 2px 15px rgba(0,0,0,0.1);
      overflow: hidden;
      border: 1px solid var(--tbl-border);
      padding: 0 0 20px 0;
    }

    .main-header {
      background: var(--brand);
      color: #fff;
      padding: 14px 24px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
    }

    .header-logo {
      height: 45px;
      max-width: 120px;
      object-fit: contain;
    }

    .header-text {
      flex: 1;
      text-align: right;
    }

    .main-title {
      font-size: 18px;
      font-weight: 700;
      letter-spacing: 0.5px;
      margin-bottom: 4px;
    }

    .main-subtitle {
      font-size: 11px;
      opacity: 0.9;
      line-height: 1.4;
    }

    /* ===== PAGE BREAK CONTROLS ===== */
    .section-header {
      background: var(--bg);
      padding: 10px 24px;
      border-bottom: 1px solid var(--tbl-border);
      border-top: 1px solid var(--tbl-border);
      font-weight: 700;
      color: var(--brand-dark);
      font-size: 13px;
      margin-top: 8px;
      
      /* CRITICAL: Keep header with following content */
      page-break-after: avoid;
      break-after: avoid;
      page-break-inside: avoid;
      break-inside: avoid;
    }

    .section-header.sub {
      font-size: 12px;
      background: #fafafa;
      padding-left: 34px;
    }

    .content-block {
      padding: 12px 24px;
      
      /* Avoid breaking content blocks */
      page-break-inside: avoid;
      break-inside: avoid;
    }

    .content-block p {
      margin: 5px 0;
      line-height: 1.5;
    }

    .content-block ul {
      margin: 5px 0 10px 0;
      padding-left: 20px;
    }

    .content-block li {
      margin: 3px 0;
      line-height: 1.4;
    }

    .chart-container {
      padding: 12px 20px;
      border-bottom: 1px solid var(--tbl-border);
      text-align: center;
      
      /* Avoid breaking charts */
      page-break-inside: avoid;
      break-inside: avoid;
    }

    .chart-container img {
      max-width: 100%;
      height: auto;
      border: 1px solid #eee;
    }

    .chart-row {
      display: flex;
      gap: 12px;
      padding: 12px 20px;
      border-bottom: 1px solid var(--tbl-border);
      
      /* Avoid breaking chart rows */
      page-break-inside: avoid;
      break-inside: avoid;
    }

    .chart-half {
      flex: 1;
      text-align: center;
    }

    .chart-half img {
      max-width: 100%;
      height: auto;
      border: 1px solid #eee;
    }

    .chart-title {
      font-size: 10px;
      font-weight: 600;
      color: var(--brand-dark);
      margin-bottom: 5px;
    }

    /* ===== TABLE STYLING WITH PAGE BREAK CONTROL ===== */
    .table-wrap {
      width: 100%;
      overflow-x: auto;
      padding: 12px 24px;
      
      /* Avoid breaking tables */
      page-break-inside: avoid;
      break-inside: avoid;
    }

    table {
      width: 100%;
      border-collapse: separate;
      border-spacing: 0;
      font-size: 10px;
      margin-top: 4px;
      
      /* Avoid breaking tables */
      page-break-inside: avoid;
      break-inside: avoid;
    }

    thead {
      display: table-header-group;
    }
    
    tbody {
      display: table-row-group;
    }

    thead th {
      position: sticky;
      top: 0;
      z-index: 2;
      background: var(--light);
      color: var(--text);
      text-align: center;
      padding: 6px 5px;
      font-weight: 700;
      border-bottom: 2px solid var(--tbl-border);
      white-space: nowrap;
    }

    thead th:first-child {
      text-align: left;
    }

    tbody td {
      padding: 5px 5px;
      border-bottom: 1px solid var(--tbl-border);
      text-align: right;
      white-space: nowrap;
    }

    tbody td:first-child {
      text-align: left;
      font-weight: 600;
    }

    tbody tr:nth-child(even) td {
      background: var(--light);
    }
    
    /* Avoid breaking table rows */
    tbody tr {
      page-break-inside: avoid;
      break-inside: avoid;
    }

    tr.year-header td {
      background-color: #e0e0e0;
      font-weight: 800;
      text-align: left;
      padding-left: 10px;
      font-size: 11px;
      color: var(--brand-dark);
    }

    tr.main-row td {
      font-weight: 700;
      background-color: #fff !important;
      color: var(--text);
      border-bottom: 2px solid #eee;
    }

    tr.sub-row td {
      font-size: 9px;
      color: #666;
      background-color: #fcfcfc !important;
      border-bottom: 1px solid #f0f0f0;
    }

    tr.sub-row td:first-child {
      padding-left: 20px;
      font-weight: 400;
    }

    .positive { color: #0a7a0a; }
    .negative { color: #b00020; }
    .ok { color: #0a7a0a; font-weight: bold; }
    .bad { color: #b00020; font-weight: bold; }

    .metrics-grid {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 10px;
      padding: 12px 24px;
      
      /* Avoid breaking metrics grid */
      page-break-inside: avoid;
      break-inside: avoid;
    }

    .metric-box {
      background: var(--light);
      border: 1px solid var(--tbl-border);
      border-radius: 4px;
      padding: 8px 10px;
      text-align: center;
    }

    .metric-label {
      font-size: 9px;
      color: #666;
      margin-bottom: 3px;
    }

    .metric-value {
      font-size: 14px;
      font-weight: 700;
      color: var(--brand-dark);
    }

    /* ===== TRACK SECTIONS - KEEP TOGETHER ===== */
    .track-section {
      margin-top: 8px;
      
      /* Try to keep entire track section together, but allow break if needed */
      page-break-inside: auto;
      break-inside: auto;
    }
    
    /* Keep section header + first table together */
    .track-section > .section-header {
      page-break-after: avoid;
      break-after: avoid;
    }
    
    .track-section > .section-header + .table-wrap {
      page-break-before: avoid;
      break-before: avoid;
    }

    .var-table {
      margin-top: 8px;
    }

    .var-table th, .var-table td {
      padding: 4px 6px;
    }

    .summary-list {
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 5px 20px;
      margin: 10px 0;
    }

    .summary-item {
      display: flex;
      justify-content: space-between;
      padding: 3px 0;
      border-bottom: 1px dotted #ddd;
    }

    .summary-item .label {
      color: #666;
    }

    .summary-item .value {
      font-weight: 600;
    }

    .footer {
      padding: 12px 24px;
      color: #666;
      font-size: 10px;
      border-top: 2px solid var(--tbl-border);
      background: #fff;
      text-align: center;
      margin-top: 20px;
      
      /* Footer can break to new page if needed */
      page-break-inside: avoid;
      break-inside: avoid;
    }

    .signature-block {
      margin-top: 30px;
      padding: 20px 24px;
      text-align: center;
      
      /* Keep signature together */
      page-break-inside: avoid;
      break-inside: avoid;
    }

    .signature-line {
      width: 250px;
      border-top: 1px solid #333;
      margin: 0 auto 5px;
      padding-top: 5px;
    }

    /* ===== PRINT-SPECIFIC RULES ===== */
    @media print {
      body {
        padding: 0;
        background: white;
      }
      
      .page {
        box-shadow: none;
        border: none;
        width: 100%;
        min-height: auto;
      }
      
      /* Reinforce page break rules for print */
      .section-header {
        page-break-after: avoid !important;
        break-after: avoid !important;
      }
      
      .table-wrap,
      table,
      .chart-row,
      .chart-container,
      .metrics-grid,
      .content-block {
        page-break-inside: avoid !important;
        break-inside: avoid !important;
      }
      
      /* If a section is too long, allow break but keep header with some content */
      .track-section .table-wrap:first-of-type {
        page-break-before: avoid !important;
        break-before: avoid !important;
      }
    }
  </style>
</head>
<body>
  <div class="page">
    
    <!-- HEADER -->
    <div class="main-header">
      <img src="data:image/png;base64,{{ logo_base64 }}" class="header-logo" alt="Logo" onerror="this.style.display='none'" />
      <div class="header-text">
        <div class="main-title">RELATÓRIO DE MONITORAMENTO DE RISCO DE MERCADO</div>
        <div class="main-subtitle">
          {{ manager_name }} | CNPJ: {{ manager_cnpj }}<br/>
          Data de Emissão: {{ emission_date }} | Período: {{ period_label }}
        </div>
      </div>
    </div>

    <!-- SECTION 1: Objetivo -->
    <div class="section-header">1. Objetivo</div>
    <div class="content-block">
      <p>Este relatório tem como objetivo apresentar a avaliação e monitoramento do risco de mercado das carteiras 
      administradas pela {{ manager_name }}, em conformidade com as diretrizes da CVM, Bacen e melhores práticas de mercado.</p>
    </div>

    <!-- SECTION 2: Metodologia -->
    <div class="section-header">2. Metodologia</div>
    <div class="content-block">
      <p>A análise é conduzida sobre quatro classes de ativos separadas por exposição geográfica:</p>
      <ul>
        <li><strong>Ações Locais:</strong> Benchmark IBOV (^BVSP)</li>
        <li><strong>Ações Offshore:</strong> Benchmark S&P 500 (^GSPC)</li>
        <li><strong>FIIs Locais:</strong> Benchmark XFIX11</li>
        <li><strong>REITs Offshore:</strong> Benchmark VNQ</li>
      </ul>
      <p><strong>Indicadores calculados:</strong></p>
      <ul>
        <li>Limites de concentração: máximo 15% em ativos únicos</li>
        <li>Drawdown: maior queda entre topo e fundo, comparado ao benchmark</li>
        <li>Métricas de risco: Sharpe, Beta, Volatilidade e VaR (95% e 99%)</li>
        <li>Períodos analisados: 1 ano, 3 anos e 5 anos (quando disponível)</li>
      </ul>
    </div>

    <!-- SECTION 3: Exposição -->
    <div class="section-header">3. Exposição ao Risco de Mercado</div>
    <div class="content-block">
      <div class="metrics-grid">
        <div class="metric-box">
          <div class="metric-label">PL Total Mercado</div>
          <div class="metric-value">R$ {{ pl_market_fmt }}</div>
        </div>
        <div class="metric-box">
          <div class="metric-label">Ações Locais</div>
          <div class="metric-value">R$ {{ local_stocks_mv_fmt }}</div>
        </div>
        <div class="metric-box">
          <div class="metric-label">Ações Offshore</div>
          <div class="metric-value">R$ {{ offshore_stocks_mv_fmt }}</div>
        </div>
        <div class="metric-box">
          <div class="metric-label">FIIs Locais</div>
          <div class="metric-value">R$ {{ local_fiis_mv_fmt }}</div>
        </div>
      </div>
      
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Exposição</th>
              <th>Maior Posição</th>
              <th>Limite (15%)</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>Local</td>
              <td>{{ max_pos_local_pct }}</td>
              <td>15.00%</td>
              <td class="{{ 'ok' if ok_pos_local else 'bad' }}">{{ 'Conforme' if ok_pos_local else 'Não Conforme' }}</td>
            </tr>
            <tr>
              <td>Offshore</td>
              <td>{{ max_pos_offshore_pct }}</td>
              <td>15.00%</td>
              <td class="{{ 'ok' if ok_pos_offshore else 'bad' }}">{{ 'Conforme' if ok_pos_offshore else 'Não Conforme' }}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>

    <!-- Charts: Asset Distribution -->
    <div class="chart-row">
      <div class="chart-half">
        <div class="chart-title">Top 20 Ativos - Exposição Local</div>
        <img src="data:image/png;base64,{{ fig_local_assets }}" alt="Ativos Local"/>
      </div>
      <div class="chart-half">
        <div class="chart-title">Setores - Exposição Local</div>
        <img src="data:image/png;base64,{{ fig_local_sectors }}" alt="Setores Local"/>
      </div>
    </div>
    <div class="chart-row">
      <div class="chart-half">
        <div class="chart-title">Top 20 Ativos - Exposição Offshore</div>
        <img src="data:image/png;base64,{{ fig_offshore_assets }}" alt="Ativos Offshore"/>
      </div>
      <div class="chart-half">
        <div class="chart-title">Setores - Exposição Offshore</div>
        <img src="data:image/png;base64,{{ fig_offshore_sectors }}" alt="Setores Offshore"/>
      </div>
    </div>

    <!-- TRACK 1: Local Stocks -->
    <div class="track-section">
      <div class="section-header">3.1 Ações Locais (vs IBOV)</div>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Período</th>
              <th>Retorno</th>
              <th>IBOV</th>
              <th>Sharpe</th>
              <th>Volatilidade</th>
              <th>Max DD</th>
              <th>Max DD IBOV</th>
              <th>Beta</th>
            </tr>
          </thead>
          <tbody>
            {% for p in local_stocks_periods %}
            <tr class="main-row">
              <td>{{ p.period }}</td>
              <td class="{{ 'positive' if p.ret_val >= 0 else 'negative' }}">{{ p.ret }}</td>
              <td>{{ p.bench_ret }}</td>
              <td>{{ p.sharpe }}</td>
              <td>{{ p.vol }}</td>
              <td class="negative">{{ p.max_dd }}</td>
              <td>{{ p.max_dd_bench }}</td>
              <td>{{ p.beta }}</td>
            </tr>
            {% endfor %}
            {% if not local_stocks_periods %}
            <tr><td colspan="8" style="text-align:center; color:#999;">Dados insuficientes</td></tr>
            {% endif %}
          </tbody>
        </table>
      </div>
      
      <div class="chart-row">
        <div class="chart-half">
          <div class="chart-title">Performance Acumulada (1Y)</div>
          <img src="data:image/png;base64,{{ fig_local_stocks_bt_1y }}" alt="Performance Local Stocks"/>
        </div>
        <div class="chart-half">
          <div class="chart-title">Drawdown (1Y)</div>
          <img src="data:image/png;base64,{{ fig_local_stocks_dd_1y }}" alt="Drawdown Local Stocks"/>
        </div>
      </div>
      
      <div class="table-wrap">
        <p style="font-weight:600; margin-bottom:5px;">VaR Diário - Ações Locais (1Y)</p>
        <table class="var-table">
          <thead>
            <tr>
              <th>Confiança</th>
              <th>VaR Histórico</th>
              <th>VaR Paramétrico</th>
              <th>VaR Monte Carlo</th>
              <th>VaR Hist. (R$)</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td><strong>95%</strong></td>
              <td class="negative">{{ local_stocks_1y.var_95_hist }}</td>
              <td class="negative">{{ local_stocks_1y.var_95_param }}</td>
              <td class="negative">{{ local_stocks_1y.var_95_mc }}</td>
              <td>{{ local_stocks_1y.var_95_hist_brl }}</td>
            </tr>
            <tr>
              <td><strong>99%</strong></td>
              <td class="negative">{{ local_stocks_1y.var_99_hist }}</td>
              <td class="negative">{{ local_stocks_1y.var_99_param }}</td>
              <td class="negative">{{ local_stocks_1y.var_99_mc }}</td>
              <td>{{ local_stocks_1y.var_99_hist_brl }}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>

    <!-- TRACK 2: Offshore Stocks -->
    <div class="track-section">
      <div class="section-header">3.2 Ações Offshore (vs S&P 500)</div>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Período</th>
              <th>Retorno</th>
              <th>S&P 500</th>
              <th>Sharpe</th>
              <th>Volatilidade</th>
              <th>Max DD</th>
              <th>Max DD S&P</th>
              <th>Beta</th>
            </tr>
          </thead>
          <tbody>
            {% for p in offshore_stocks_periods %}
            <tr class="main-row">
              <td>{{ p.period }}</td>
              <td class="{{ 'positive' if p.ret_val >= 0 else 'negative' }}">{{ p.ret }}</td>
              <td>{{ p.bench_ret }}</td>
              <td>{{ p.sharpe }}</td>
              <td>{{ p.vol }}</td>
              <td class="negative">{{ p.max_dd }}</td>
              <td>{{ p.max_dd_bench }}</td>
              <td>{{ p.beta }}</td>
            </tr>
            {% endfor %}
            {% if not offshore_stocks_periods %}
            <tr><td colspan="8" style="text-align:center; color:#999;">Dados insuficientes</td></tr>
            {% endif %}
          </tbody>
        </table>
      </div>
      
      <div class="chart-row">
        <div class="chart-half">
          <div class="chart-title">Performance Acumulada (1Y)</div>
          <img src="data:image/png;base64,{{ fig_offshore_stocks_bt_1y }}" alt="Performance Offshore Stocks"/>
        </div>
        <div class="chart-half">
          <div class="chart-title">Drawdown (1Y)</div>
          <img src="data:image/png;base64,{{ fig_offshore_stocks_dd_1y }}" alt="Drawdown Offshore Stocks"/>
        </div>
      </div>
      
      <div class="table-wrap">
        <p style="font-weight:600; margin-bottom:5px;">VaR Diário - Ações Offshore (1Y)</p>
        <table class="var-table">
          <thead>
            <tr>
              <th>Confiança</th>
              <th>VaR Histórico</th>
              <th>VaR Paramétrico</th>
              <th>VaR Monte Carlo</th>
              <th>VaR Hist. (R$)</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td><strong>95%</strong></td>
              <td class="negative">{{ offshore_stocks_1y.var_95_hist }}</td>
              <td class="negative">{{ offshore_stocks_1y.var_95_param }}</td>
              <td class="negative">{{ offshore_stocks_1y.var_95_mc }}</td>
              <td>{{ offshore_stocks_1y.var_95_hist_brl }}</td>
            </tr>
            <tr>
              <td><strong>99%</strong></td>
              <td class="negative">{{ offshore_stocks_1y.var_99_hist }}</td>
              <td class="negative">{{ offshore_stocks_1y.var_99_param }}</td>
              <td class="negative">{{ offshore_stocks_1y.var_99_mc }}</td>
              <td>{{ offshore_stocks_1y.var_99_hist_brl }}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>

    <!-- TRACK 3: Local FIIs -->
    <div class="track-section">
      <div class="section-header">3.3 FIIs Locais (vs XFIX11)</div>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Período</th>
              <th>Retorno</th>
              <th>XFIX11</th>
              <th>Sharpe</th>
              <th>Volatilidade</th>
              <th>Max DD</th>
              <th>Max DD XFIX</th>
              <th>Beta</th>
            </tr>
          </thead>
          <tbody>
            {% for p in local_fiis_periods %}
            <tr class="main-row">
              <td>{{ p.period }}</td>
              <td class="{{ 'positive' if p.ret_val >= 0 else 'negative' }}">{{ p.ret }}</td>
              <td>{{ p.bench_ret }}</td>
              <td>{{ p.sharpe }}</td>
              <td>{{ p.vol }}</td>
              <td class="negative">{{ p.max_dd }}</td>
              <td>{{ p.max_dd_bench }}</td>
              <td>{{ p.beta }}</td>
            </tr>
            {% endfor %}
            {% if not local_fiis_periods %}
            <tr><td colspan="8" style="text-align:center; color:#999;">Dados insuficientes</td></tr>
            {% endif %}
          </tbody>
        </table>
      </div>
      
      <div class="chart-row">
        <div class="chart-half">
          <div class="chart-title">Performance Acumulada (1Y)</div>
          <img src="data:image/png;base64,{{ fig_local_fiis_bt_1y }}" alt="Performance FIIs"/>
        </div>
        <div class="chart-half">
          <div class="chart-title">Drawdown (1Y)</div>
          <img src="data:image/png;base64,{{ fig_local_fiis_dd_1y }}" alt="Drawdown FIIs"/>
        </div>
      </div>
      
      <div class="table-wrap">
        <p style="font-weight:600; margin-bottom:5px;">VaR Diário - FIIs (1Y)</p>
        <table class="var-table">
          <thead>
            <tr>
              <th>Confiança</th>
              <th>VaR Histórico</th>
              <th>VaR Paramétrico</th>
              <th>VaR Monte Carlo</th>
              <th>VaR Hist. (R$)</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td><strong>95%</strong></td>
              <td class="negative">{{ local_fiis_1y.var_95_hist }}</td>
              <td class="negative">{{ local_fiis_1y.var_95_param }}</td>
              <td class="negative">{{ local_fiis_1y.var_95_mc }}</td>
              <td>{{ local_fiis_1y.var_95_hist_brl }}</td>
            </tr>
            <tr>
              <td><strong>99%</strong></td>
              <td class="negative">{{ local_fiis_1y.var_99_hist }}</td>
              <td class="negative">{{ local_fiis_1y.var_99_param }}</td>
              <td class="negative">{{ local_fiis_1y.var_99_mc }}</td>
              <td>{{ local_fiis_1y.var_99_hist_brl }}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>

    <!-- TRACK 4: Offshore REITs -->
    <div class="track-section">
      <div class="section-header">3.4 REITs Offshore (vs VNQ)</div>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Período</th>
              <th>Retorno</th>
              <th>VNQ</th>
              <th>Sharpe</th>
              <th>Volatilidade</th>
              <th>Max DD</th>
              <th>Max DD VNQ</th>
              <th>Beta</th>
            </tr>
          </thead>
          <tbody>
            {% for p in offshore_reits_periods %}
            <tr class="main-row">
              <td>{{ p.period }}</td>
              <td class="{{ 'positive' if p.ret_val >= 0 else 'negative' }}">{{ p.ret }}</td>
              <td>{{ p.bench_ret }}</td>
              <td>{{ p.sharpe }}</td>
              <td>{{ p.vol }}</td>
              <td class="negative">{{ p.max_dd }}</td>
              <td>{{ p.max_dd_bench }}</td>
              <td>{{ p.beta }}</td>
            </tr>
            {% endfor %}
            {% if not offshore_reits_periods %}
            <tr><td colspan="8" style="text-align:center; color:#999;">Dados insuficientes</td></tr>
            {% endif %}
          </tbody>
        </table>
      </div>
      
      <div class="chart-row">
        <div class="chart-half">
          <div class="chart-title">Performance Acumulada (1Y)</div>
          <img src="data:image/png;base64,{{ fig_offshore_reits_bt_1y }}" alt="Performance REITs"/>
        </div>
        <div class="chart-half">
          <div class="chart-title">Drawdown (1Y)</div>
          <img src="data:image/png;base64,{{ fig_offshore_reits_dd_1y }}" alt="Drawdown REITs"/>
        </div>
      </div>
      
      <div class="table-wrap">
        <p style="font-weight:600; margin-bottom:5px;">VaR Diário - REITs (1Y)</p>
        <table class="var-table">
          <thead>
            <tr>
              <th>Confiança</th>
              <th>VaR Histórico</th>
              <th>VaR Paramétrico</th>
              <th>VaR Monte Carlo</th>
              <th>VaR Hist. (R$)</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td><strong>95%</strong></td>
              <td class="negative">{{ offshore_reits_1y.var_95_hist }}</td>
              <td class="negative">{{ offshore_reits_1y.var_95_param }}</td>
              <td class="negative">{{ offshore_reits_1y.var_95_mc }}</td>
              <td>{{ offshore_reits_1y.var_95_hist_brl }}</td>
            </tr>
            <tr>
              <td><strong>99%</strong></td>
              <td class="negative">{{ offshore_reits_1y.var_99_hist }}</td>
              <td class="negative">{{ offshore_reits_1y.var_99_param }}</td>
              <td class="negative">{{ offshore_reits_1y.var_99_mc }}</td>
              <td>{{ offshore_reits_1y.var_99_hist_brl }}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>

    <!-- Summary Table -->
    <div class="section-header">3.5 Resumo Comparativo (1Y)</div>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Métrica</th>
            <th>Ações Locais</th>
            <th>Ações Offshore</th>
            <th>FIIs</th>
            <th>REITs</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>Retorno</td>
            <td class="{{ 'positive' if local_stocks_1y.ret_val >= 0 else 'negative' }}">{{ local_stocks_1y.ret }}</td>
            <td class="{{ 'positive' if offshore_stocks_1y.ret_val >= 0 else 'negative' }}">{{ offshore_stocks_1y.ret }}</td>
            <td class="{{ 'positive' if local_fiis_1y.ret_val >= 0 else 'negative' }}">{{ local_fiis_1y.ret }}</td>
            <td class="{{ 'positive' if offshore_reits_1y.ret_val >= 0 else 'negative' }}">{{ offshore_reits_1y.ret }}</td>
          </tr>
          <tr>
            <td>Sharpe</td>
            <td>{{ local_stocks_1y.sharpe }}</td>
            <td>{{ offshore_stocks_1y.sharpe }}</td>
            <td>{{ local_fiis_1y.sharpe }}</td>
            <td>{{ offshore_reits_1y.sharpe }}</td>
          </tr>
          <tr>
            <td>Volatilidade</td>
            <td>{{ local_stocks_1y.vol }}</td>
            <td>{{ offshore_stocks_1y.vol }}</td>
            <td>{{ local_fiis_1y.vol }}</td>
            <td>{{ offshore_reits_1y.vol }}</td>
          </tr>
          <tr>
            <td>Beta</td>
            <td>{{ local_stocks_1y.beta }}</td>
            <td>{{ offshore_stocks_1y.beta }}</td>
            <td>{{ local_fiis_1y.beta }}</td>
            <td>{{ offshore_reits_1y.beta }}</td>
          </tr>
          <tr>
            <td>Max Drawdown</td>
            <td class="negative">{{ local_stocks_1y.max_dd }}</td>
            <td class="negative">{{ offshore_stocks_1y.max_dd }}</td>
            <td class="negative">{{ local_fiis_1y.max_dd }}</td>
            <td class="negative">{{ offshore_reits_1y.max_dd }}</td>
          </tr>
          <tr>
            <td>VaR 95%</td>
            <td class="negative">{{ local_stocks_1y.var_95_hist }}</td>
            <td class="negative">{{ offshore_stocks_1y.var_95_hist }}</td>
            <td class="negative">{{ local_fiis_1y.var_95_hist }}</td>
            <td class="negative">{{ offshore_reits_1y.var_95_hist }}</td>
          </tr>
        </tbody>
      </table>
    </div>

    <!-- Section 4: Compliance -->
    <div class="section-header">4. Conformidade Regulatória</div>
    <div class="content-block">
      <p>Este fundo opera em conformidade com as seguintes normas e regulamentos:</p>
      <ul>
        <li>Resolução CVM 175/2022 (nova regulação de fundos de investimento)</li>
        <li>Resolução CVM 50/2021 (prestação de serviços fiduciários)</li>
        <li>Resolução BCB 4.557/2017 (gerenciamento de risco)</li>
        <li>Código ANBIMA de Regulação e Melhores Práticas</li>
      </ul>
    </div>

    <!-- Section 5: Conclusions -->
    <div class="section-header">5. Conclusões</div>
    <div class="content-block">
      <p>{{ conclusions }}</p>
    </div>

    <!-- Signature -->
    <div class="signature-block">
      <div class="signature-line">
        <strong>{{ responsible_name }}</strong><br/>
        Diretor de Risco
      </div>
    </div>

    <!-- Footer -->
    <div class="footer">
      {{ manager_name }} | Documento gerado em {{ emission_date }} | Uso Interno
    </div>
    
  </div>
</body>
</html>
"""


def render_report_html():
    if not HAS_MPL:
        return "Matplotlib não disponível para geração de gráficos."
    
    def _gen_img(fig):
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor='white')
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def _get_track_metrics(res):
        if not res:
            return {
                "period": "N/A", "ret": "N/A", "ret_val": 0, "bench_ret": "N/A", 
                "max_dd": "N/A", "max_dd_bench": "N/A", "avg_dd": "N/A", 
                "dd_12m": "N/A", "sharpe": "N/A", "vol": "N/A", "beta": "N/A",
                "var_95_hist": "N/A", "var_95_param": "N/A", "var_95_mc": "N/A", "var_95_hist_brl": "N/A",
                "var_99_hist": "N/A", "var_99_param": "N/A", "var_99_mc": "N/A", "var_99_hist_brl": "N/A",
            }
        
        ret_val = res.get("final_return", 0)
        
        metrics = {
            "period": res.get("period", "N/A"),
            "ret": as_pct_str(ret_val),
            "ret_val": ret_val,  # Raw value for color coding
            "bench_ret": as_pct_str(res.get("final_bench_return", 0)),
            "max_dd": as_pct_str(res.get("max_drawdown", 0)),
            "max_dd_bench": as_pct_str(res.get("max_drawdown_bench", 0)),
            "avg_dd": as_pct_str(res.get("avg_drawdown", 0)),
            "dd_12m": as_pct_str(res.get("drawdown_12m", 0)),
            "sharpe": f"{res.get('sharpe', 0):.2f}",
            "vol": as_pct_str(res.get("volatility", 0)),
            "beta": f"{res.get('beta', 0):.2f}"
        }
        
        # Add VaR data if available
        var_data = res.get("var", {})
        if var_data:
            var_95 = var_data.get("var_95", {})
            var_99 = var_data.get("var_99", {})
            metrics.update({
                "var_95_hist": as_pct_str(var_95.get("historical_var_pct", 0)),
                "var_95_param": as_pct_str(var_95.get("parametric_var_pct", 0)),
                "var_95_mc": as_pct_str(var_95.get("monte_carlo_var_pct", 0)),
                "var_95_hist_brl": brl(var_95.get("historical_var_brl", 0)),
                "var_99_hist": as_pct_str(var_99.get("historical_var_pct", 0)),
                "var_99_param": as_pct_str(var_99.get("parametric_var_pct", 0)),
                "var_99_mc": as_pct_str(var_99.get("monte_carlo_var_pct", 0)),
                "var_99_hist_brl": brl(var_99.get("historical_var_brl", 0)),
            })
        else:
            metrics.update({
                "var_95_hist": "N/A", "var_95_param": "N/A", "var_95_mc": "N/A", "var_95_hist_brl": "N/A",
                "var_99_hist": "N/A", "var_99_param": "N/A", "var_99_mc": "N/A", "var_99_hist_brl": "N/A",
            })
        
        return metrics


    # Build period comparison tables for all 4 tracks
    def _build_periods(results):
        periods = []
        for p in ["1Y", "3Y", "5Y"]:
            if p in results:
                periods.append(_get_track_metrics(results[p]))
        return periods
    
    local_stocks_periods = _build_periods(local_stocks_results)
    offshore_stocks_periods = _build_periods(offshore_stocks_results)
    local_fiis_periods = _build_periods(local_fiis_results)
    offshore_reits_periods = _build_periods(offshore_reits_results)

    # Get 1Y metrics for summary
    local_stocks_1y = _get_track_metrics(local_stocks_results.get("1Y", {}))
    offshore_stocks_1y = _get_track_metrics(offshore_stocks_results.get("1Y", {}))
    local_fiis_1y = _get_track_metrics(local_fiis_results.get("1Y", {}))
    offshore_reits_1y = _get_track_metrics(offshore_reits_results.get("1Y", {}))

    # Local Assets Chart
    local_df = base_df[base_df["exposure_group"] == "Exposição Local"].copy()
    fig1, ax1 = plt.subplots(figsize=(6, 3))
    if not local_df.empty:
        top_local = local_df.groupby("security_name_clean")["mv"].sum().sort_values(ascending=False).head(20)
        top_local.plot(kind="bar", ax=ax1, color=PALETA_CORES[0])
    ax1.set_title("Top 20 Ativos Local")
    ax1.tick_params(axis='x', rotation=45)
    fig_local_assets = _gen_img(fig1)

    # Replace the sector pie charts with bar charts in render_report_html()

    # Local Sectors Chart - BAR instead of PIE
    fig2, ax2 = plt.subplots(figsize=(6, 3))
    if not local_df.empty:
        sec_local = local_df.groupby("sector")["mv"].sum().sort_values(ascending=False)
        colors = [PALETA_CORES[i % len(PALETA_CORES)] for i in range(len(sec_local))]
        sec_local.plot(kind="barh", ax=ax2, color=colors)
        ax2.set_xlabel("Valor (R$)")
        # Add percentage labels
        total = sec_local.sum()
        for i, (idx, v) in enumerate(sec_local.items()):
            ax2.text(v, i, f" {v/total*100:.1f}%", va='center', fontsize=8)
    ax2.set_title("Setores Local")
    ax2.set_ylabel("")
    fig_local_sectors = _gen_img(fig2)

    # Offshore Assets Chart
    offshore_df = base_df[base_df["exposure_group"] == "Exposição Offshore"].copy()
    fig3, ax3 = plt.subplots(figsize=(6, 3))
    if not offshore_df.empty:
        top_off = offshore_df.groupby("security_name_clean")["mv"].sum().sort_values(ascending=False).head(20)
        top_off.plot(kind="bar", ax=ax3, color=PALETA_CORES[4])
    ax3.set_title("Top 20 Ativos Offshore")
    ax3.tick_params(axis='x', rotation=45)
    fig_offshore_assets = _gen_img(fig3)

    # Offshore Sectors Chart
    fig4, ax4 = plt.subplots(figsize=(6, 3))
    if not offshore_df.empty:
        sec_off = offshore_df.groupby("sector")["mv"].sum().sort_values(ascending=False)
        colors = [PALETA_CORES[i % len(PALETA_CORES)] for i in range(len(sec_off))]
        sec_off.plot(kind="barh", ax=ax4, color=colors)
        ax4.set_xlabel("Valor (R$)")
        # Add percentage labels
        total = sec_off.sum()
        for i, (idx, v) in enumerate(sec_off.items()):
            ax4.text(v, i, f" {v/total*100:.1f}%", va='center', fontsize=8)
    ax4.set_title("Setores Offshore")
    ax4.set_ylabel("")
    fig_offshore_sectors = _gen_img(fig4)

    # Performance and Drawdown Chart generators
    def _gen_line_chart(res, title, label_port, label_bench):
        fig, ax = plt.subplots(figsize=(6, 3))
        if res and "dates" in res and res["dates"]:
            ax.plot(res["dates"], res["cum_port"], label=label_port, color=PALETA_CORES[0])
            ax.plot(res["dates"], res["cum_bench"], label=label_bench, color=PALETA_CORES[2], linestyle="--")
            ax.legend()
        ax.set_title(title)
        return _gen_img(fig)

    def _gen_dd_chart_comparison(res, title, label_port, label_bench):
        """Generate drawdown chart comparing portfolio vs benchmark."""
        fig, ax = plt.subplots(figsize=(6, 3))
        if res and "drawdown_dates" in res and res["drawdown_dates"]:
            dates = res["drawdown_dates"]
            dd_port = res.get("drawdown_port", [])
            dd_bench = res.get("drawdown_bench", [])
            
            if dd_port:
                ax.fill_between(dates, dd_port, 0, alpha=0.3, color=PALETA_CORES[0], label=label_port)
                ax.plot(dates, dd_port, color=PALETA_CORES[0], linewidth=1)
            if dd_bench:
                ax.plot(dates, dd_bench, color=PALETA_CORES[2], linestyle="--", linewidth=1, label=label_bench)
            
            ax.legend()
        ax.set_title(title)
        ax.set_ylabel("Drawdown")
        return _gen_img(fig)

    # Generate charts for all 4 tracks
    
    # Track 1: Local Stocks
    local_stocks_1y_data = local_stocks_results.get("1Y", {})
    fig_local_stocks_bt_1y = _gen_line_chart(
        local_stocks_1y_data, 
        "Performance Ações Locais vs IBOV (1Y)", 
        "Carteira", 
        "IBOV"
    )
    fig_local_stocks_dd_1y = _gen_dd_chart_comparison(
        local_stocks_1y_data, 
        "Drawdown Ações Locais vs IBOV (1Y)", 
        "Carteira", 
        "IBOV"
    )
    
    # Track 2: Offshore Stocks
    offshore_stocks_1y_data = offshore_stocks_results.get("1Y", {})
    fig_offshore_stocks_bt_1y = _gen_line_chart(
        offshore_stocks_1y_data, 
        "Performance Ações Offshore vs S&P 500 (1Y)", 
        "Carteira", 
        "S&P 500"
    )
    fig_offshore_stocks_dd_1y = _gen_dd_chart_comparison(
        offshore_stocks_1y_data, 
        "Drawdown Ações Offshore vs S&P 500 (1Y)", 
        "Carteira", 
        "S&P 500"
    )
    
    # Track 3: Local FIIs
    local_fiis_1y_data = local_fiis_results.get("1Y", {})
    fig_local_fiis_bt_1y = _gen_line_chart(
        local_fiis_1y_data, 
        "Performance FIIs vs XFIX11 (1Y)", 
        "Carteira", 
        "XFIX11"
    )
    fig_local_fiis_dd_1y = _gen_dd_chart_comparison(
        local_fiis_1y_data, 
        "Drawdown FIIs vs XFIX11 (1Y)", 
        "Carteira", 
        "XFIX11"
    )
    
    # Track 4: Offshore REITs
    offshore_reits_1y_data = offshore_reits_results.get("1Y", {})
    fig_offshore_reits_bt_1y = _gen_line_chart(
        offshore_reits_1y_data, 
        "Performance REITs vs VNQ (1Y)", 
        "Carteira", 
        "VNQ"
    )
    fig_offshore_reits_dd_1y = _gen_dd_chart_comparison(
        offshore_reits_1y_data, 
        "Drawdown REITs vs VNQ (1Y)", 
        "Carteira", 
        "VNQ"
    )

    # Compliance logic
    max_pos_local = local_df["mv"].max() / local_df["mv"].sum() if not local_df.empty and local_df["mv"].sum() > 0 else 0
    max_pos_offshore = offshore_df["mv"].max() / offshore_df["mv"].sum() if not offshore_df.empty and offshore_df["mv"].sum() > 0 else 0
    
    ok_pos_local = max_pos_local <= 0.15
    ok_pos_offshore = max_pos_offshore <= 0.15
    
    # Conclusions
    if ok_pos_local and ok_pos_offshore:
        conclusions = "As carteiras administradas apresentam níveis de risco de mercado compatíveis com o perfil de risco estabelecido. Não foram identificadas exposições significativas que possam comprometer a estabilidade financeira das carteiras."
    else:
        conclusions = "As carteiras administradas pela Ceres Asset Gestão de Investimentos Ltda apresentam, temporariamente, níveis de risco de mercado não compatíveis com o perfil de risco estabelecido. O período atual é de construção e tombamento de carteiras."

    t = Template(REPORT_HTML)
    return t.render(
        logo_base64="",
        manager_name="Ceres Asset Gestão de Investimentos Ltda",
        manager_cnpj="40.962.925/0001-38",
        emission_date=datetime.now().strftime("%d/%m/%Y"),
        period_label=period_label,
        responsible_name="Brenno Melo",
        pl_market_fmt=brl(pl_market),
        local_stocks_mv_fmt=brl(local_stocks_df["mv"].sum()) if not local_stocks_df.empty else "0,00",
        offshore_stocks_mv_fmt=brl(offshore_stocks_df["mv"].sum()) if not offshore_stocks_df.empty else "0,00",
        local_fiis_mv_fmt=brl(local_fiis_df["mv"].sum()) if not local_fiis_df.empty else "0,00",
        offshore_reits_mv_fmt=brl(offshore_reits_df["mv"].sum()) if not offshore_reits_df.empty else "0,00",
        max_pos_local_pct=as_pct_str(max_pos_local),
        ok_pos_local=ok_pos_local,
        max_pos_offshore_pct=as_pct_str(max_pos_offshore),
        ok_pos_offshore=ok_pos_offshore,
        fig_local_assets=fig_local_assets,
        fig_local_sectors=fig_local_sectors,
        fig_offshore_assets=fig_offshore_assets,
        fig_offshore_sectors=fig_offshore_sectors,
        # Track 1: Local Stocks
        fig_local_stocks_bt_1y=fig_local_stocks_bt_1y,
        fig_local_stocks_dd_1y=fig_local_stocks_dd_1y,
        local_stocks_periods=local_stocks_periods,
        local_stocks_1y=local_stocks_1y,
        # Track 2: Offshore Stocks
        fig_offshore_stocks_bt_1y=fig_offshore_stocks_bt_1y,
        fig_offshore_stocks_dd_1y=fig_offshore_stocks_dd_1y,
        offshore_stocks_periods=offshore_stocks_periods,
        offshore_stocks_1y=offshore_stocks_1y,
        # Track 3: Local FIIs
        fig_local_fiis_bt_1y=fig_local_fiis_bt_1y,
        fig_local_fiis_dd_1y=fig_local_fiis_dd_1y,
        local_fiis_periods=local_fiis_periods,
        local_fiis_1y=local_fiis_1y,
        # Track 4: Offshore REITs
        fig_offshore_reits_bt_1y=fig_offshore_reits_bt_1y,
        fig_offshore_reits_dd_1y=fig_offshore_reits_dd_1y,
        offshore_reits_periods=offshore_reits_periods,
        offshore_reits_1y=offshore_reits_1y,
        conclusions=conclusions
    )


# ---------------------------- Auto Report Generation ---------------------------- #

st.markdown("---")
st.subheader("📄 Relatórios Gerados")

# Store generated reports in session state to avoid regenerating on every interaction
if "report_html" not in st.session_state:
    st.session_state.report_html = None
    st.session_state.report_pdf = None
    st.session_state.report_generated_at = None

# Auto-generate reports if we have data
if not base_df.empty:
    # Check if we need to regenerate (new data or first run)
    current_data_hash = f"{ref_dt}_{len(base_df)}_{pl_market}"
    
    if (st.session_state.report_html is None or 
        st.session_state.get("data_hash") != current_data_hash):
        
        with st.spinner("Gerando relatórios HTML e PDF automaticamente..."):
            try:
                # Generate HTML
                html_content = render_report_html()
                st.session_state.report_html = html_content
                st.session_state.data_hash = current_data_hash
                st.session_state.report_generated_at = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
                
                # Generate PDF if Playwright is available
                if HAS_PLAYWRIGHT:
                    try:
                        pdf_bytes = html_to_pdf_a4(html_content)
                        st.session_state.report_pdf = pdf_bytes
                    except Exception as e:
                        st.warning(f"PDF generation failed: {e}")
                        st.session_state.report_pdf = None
                else:
                    st.session_state.report_pdf = None
                    
            except Exception as e:
                st.error(f"Erro ao gerar relatório: {e}")
                st.session_state.report_html = None
                st.session_state.report_pdf = None

    # Display download buttons if reports are ready
    if st.session_state.report_html:
        st.success(f"✅ Relatórios gerados automaticamente em {st.session_state.report_generated_at}")
        
        col_dl1, col_dl2, col_dl3 = st.columns(3)
        
        # HTML Download
        with col_dl1:
            filename_base = f"relatorio_mercado_{ref_dt.strftime('%Y%m%d')}"
            st.download_button(
                "📥 Download HTML",
                st.session_state.report_html,
                f"{filename_base}.html",
                "text/html",
                key="dl_html_auto"
            )
        
        # PDF Download
        with col_dl2:
            if st.session_state.report_pdf:
                st.download_button(
                    "📥 Download PDF",
                    st.session_state.report_pdf,
                    f"{filename_base}.pdf",
                    "application/pdf",
                    key="dl_pdf_auto"
                )
            elif HAS_PLAYWRIGHT:
                st.caption("PDF em processamento...")
            else:
                st.caption("⚠️ Playwright não instalado")
        
        # Regenerate button
        with col_dl3:
            if st.button("🔄 Regenerar Relatórios"):
                st.session_state.report_html = None
                st.session_state.report_pdf = None
                st.rerun()
        
        # Preview HTML in expander
        with st.expander("👁️ Pré-visualizar Relatório HTML", expanded=False):
            st.components.v1.html(st.session_state.report_html, height=800, scrolling=True)

else:
    st.info("Carregue dados para gerar os relatórios automaticamente.")

# Optional: Manual regeneration in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### Relatórios")
if st.sidebar.button("🔄 Forçar Regeneração"):
    st.session_state.report_html = None
    st.session_state.report_pdf = None
    st.rerun()

if not HAS_PLAYWRIGHT:
    st.sidebar.warning("⚠️ Playwright não instalado. PDF não disponível.")
    st.sidebar.code("pip install playwright\nplaywright install chromium", language="bash")