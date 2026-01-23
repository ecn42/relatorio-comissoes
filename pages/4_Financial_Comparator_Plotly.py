import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import base64
import io
from pages.ceres_logo import LOGO_BASE64
    

### Simple Authentication
if not st.session_state.get("authenticated", False):
    st.warning("Please enter the password on the Home page first.")
    st.write("Status: Não Autenticado")
    st.stop()
    
  # prevent the rest of the page from running
st.write("Autenticado")

# Ceres Wealth Color Scheme
CERES_COLORS = [
    
    "#8c6239",
    "#7F7F85",
    "#dedede",
    "#1e5f88",
    "#7a6200",
    "#013220",
    "#582308",
]

# Language translations
TRANSLATIONS = {
    "en": {
        "title": "📈 Stock Financial Comparator",
        "tickers_input": "Enter stock tickers (comma-separated)",
        "tickers_help": "e.g., AAPL,MSFT,GOOGL",
        "years_input": "Number of years to analyze",
        "analyze_btn": "Analyze Stocks",
        "fetching": "Fetching financial data...",
        "no_data": "No data available for",
        "error_fetching": "Error fetching",
        "no_common_years": "No common years found across tickers. Showing available data.",
        "financial_metrics": "Financial Metrics",
        "download_metrics": "Download Financial Metrics CSV",
        "current_quotes": "Current Stock Quotes",
        "download_quotes": "Download Quotes CSV",
        "historical_prices": "",
        "select_period": "Select time period",
        "comparison_dashboard": "📊 Company Comparison Dashboard",
        "select_companies": "Select companies to compare",
        "currency_unit": "Currency/Unit for revenue",
        "scale": "Scale",
        "select_at_least_2": "Select at least 2 companies to compare",
        "add_at_least_2": "Add at least 2 tickers to enable comparison",
        "detailed_comparison": "Detailed Comparison",
        "error_creating": "Error creating comparison",
        "select_different": "Please select two different companies",
        "data_unavailable": "Data unavailable",
        "revenue": "Revenue",
        "ebit_margin": "EBIT Margin",
        "ebitda_margin": "EBITDA Margin",
        "roe": "Return on Equity",
        "pe_ratio": "Price to Earnings",
        "debt_equity": "Leverage Ratio",
        "stock_performance": "1Y Stock Performance",
        "market_cap": "Market Cap",
        "current_price": "Current Price",
    },
    "pt": {
        "title": "📈 Comparador de Ações Financeiras",
        "tickers_input": "Digite os símbolos das ações (separados por vírgula)",
        "tickers_help": "ex: AAPL,MSFT,GOOGL",
        "years_input": "Número de anos para análise",
        "analyze_btn": "Analisar Ações",
        "fetching": "Buscando dados financeiros...",
        "no_data": "Sem dados disponíveis para",
        "error_fetching": "Erro ao buscar",
        "no_common_years": "Nenhum ano comum encontrado entre as ações. Mostrando dados disponíveis.",
        "financial_metrics": "Métricas Financeiras",
        "download_metrics": "Baixar Métricas em CSV",
        "current_quotes": "Cotações Atuais",
        "download_quotes": "Baixar Cotações em CSV",
        "historical_prices": "",
        "select_period": "Selecione o período",
        "comparison_dashboard": "📊 Painel de Comparação de Empresas",
        "select_companies": "Selecione empresas para comparar",
        "currency_unit": "Moeda/Unidade para receita",
        "scale": "Escala",
        "select_at_least_2": "Selecione pelo menos 2 empresas para comparar",
        "add_at_least_2": "Adicione pelo menos 2 símbolos de ações para habilitar comparação",
        "detailed_comparison": "Comparação Detalhada",
        "error_creating": "Erro ao criar comparação",
        "select_different": "Por favor, selecione duas empresas diferentes",
        "data_unavailable": "Dados indisponíveis",
        "revenue": "Receita",
        "ebit_margin": "Margem EBIT",
        "ebitda_margin": "Margem EBITDA",
        "roe": "Retorno sobre Patrimônio",
        "pe_ratio": "Preço/Lucro",
        "debt_equity": "Razão de Alavancagem",
        "stock_performance": "Desempenho 1Y",
        "market_cap": "Valor de Mercado",
        "current_price": "Preço Atual",
    }
}

def get_text(lang, key):
    """Get translated text"""
    return TRANSLATIONS[lang].get(key, key)


# ---------------------- PDF Logic (Playwright) ----------------------

def html_to_pdf_playwright(html_content: str, width_px: int = 1080, height_px: int = 1080) -> bytes:
    """
    Render HTML to a PDF using headless Chromium via Playwright.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        st.error("Playwright not installed. Please install it to use PDF generation features.")
        return None

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={"width": width_px, "height": height_px},
            device_scale_factor=2,  # Higher pixel density for better quality
        )
        page = context.new_page()

        page.set_content(html_content, wait_until="load")
        
        try:
           page.wait_for_load_state("networkidle", timeout=5000)
        except:
           pass

        pdf_bytes = page.pdf(
            print_background=True,
            width=f"{width_px}px",
            height=f"{height_px}px",
            page_ranges="1",
            margin={"top": "0", "right": "0", "bottom": "0", "left": "0"},
        )

        context.close()
        browser.close()

    return pdf_bytes


def html_to_png_playwright(html_content: str, width_px: int = 1080, height_px: int = 1080) -> bytes:
    """
    Render HTML to a Transparent PNG using headless Chromium via Playwright.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        st.error("Playwright not installed. Please install it to use features.")
        return None

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={"width": width_px, "height": height_px},
            device_scale_factor=2, 
        )
        page = context.new_page()

        # Set content with transparent background explicitly in body/html style if not already
        page.set_content(html_content, wait_until="load")
        
        try:
           page.wait_for_load_state("networkidle", timeout=5000)
        except:
           pass

        png_bytes = page.screenshot(
            omit_background=True,
            type="png",
        )

        context.close()
        browser.close()

    return png_bytes



def generate_bar_chart_svg(
    data: dict, 
    title: str, 
    formatter: callable = lambda x: f"{x:.2f}"
) -> str:
    """
    Generate a simple SVG bar chart.
    data: dict {ticker: value}
    title: Chart title
    formatter: Function to format values
    """
    if not data:
        return ""

    width = 320
    height = 220
    margin_top = 40
    margin_bottom = 30
    margin_left = 10
    margin_right = 10
    
    chart_h = height - margin_top - margin_bottom
    chart_w = width - margin_left - margin_right
    
    tickers = list(data.keys())
    values = list(data.values())
    
    # Handle NaNs or Nones
    values = [0 if pd.isna(v) or v is None else v for v in values]
    
    max_val = max(values) if values else 1
    if max_val == 0: max_val = 1
    
    # Bar width calculations
    n = len(tickers)
    bar_gap = 20
    # Total width available for bars = chart_w - (n-1)*gap
    # But usually we want fixed bar width or proportional
    # Let's do simple proportional
    step = chart_w / n
    bar_w = step * 0.6
    
    svg_parts = []
    
    # Title
    svg_parts.append(
        f'<text x="{width/2}" y="20" text-anchor="middle" font-family="sans-serif" font-size="14" font-weight="bold" fill="#333">{title}</text>'
    )
    
    for i, (ticker, val) in enumerate(zip(tickers, values)):
        # Normalize height
        # If negative values exist, this logic needs update, but for now assuming mostly positive metrics 
        # (Margin, Revenue, etc. can be neg, but let's stick to positive base or simple projection)
        # For simplicity in this bespoke chart: base at bottom. 
        # If val < 0, simple implementation might look weird, but acceptable for MVP replacement.
        
        # Scale 0 to max_val * 1.1 top
        scale_max = max(max_val, 0) * 1.1
        if scale_max == 0: scale_max = 1
        
        bar_h = (max(val, 0) / scale_max) * chart_h
        
        x = margin_left + (i * step) + (step - bar_w)/2
        y = margin_top + (chart_h - bar_h)
        
        color = CERES_COLORS[i % len(CERES_COLORS)]
        
        # Bar
        svg_parts.append(
            f'<rect x="{x}" y="{y}" width="{bar_w}" height="{bar_h}" fill="{color}" rx="2" />'
        )
        
        # Value Label (above bar)
        label = formatter(val)
        svg_parts.append(
            f'<text x="{x + bar_w/2}" y="{y - 5}" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#333">{label}</text>'
        )
        
        # X Axis Label (Ticker)
        svg_parts.append(
            f'<text x="{x + bar_w/2}" y="{height - 10}" text-anchor="middle" font-family="sans-serif" font-size="12" font-weight="600" fill="#555">{ticker}</text>'
        )

    return f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" style="background-color: #fff; border-radius: 8px; border: 1px solid #eee;">{"".join(svg_parts)}</svg>'


def generate_social_export(
    metrics_dict: dict, 
    selected_tickers: list,
    title_str: str,
    scale_label: str,
    scale_divisor: float,
    currency_unit: str,
    fmt: str = "Square",
    file_type: str = "pdf",
    transparent: bool = False
) -> bytes:
    """
    Generate a PDF or PNG suitable for social media sharing using SVG charts.
    file_type: 'pdf' or 'png'
    """
    
    # Dimensions
    if fmt == "Story":
        width, height = 1080, 1920
    else: # Square
        width, height = 1080, 1080
        
    bg_color = "#f4f4f4" if not transparent else "transparent"
    card_color = "#ffffff" if not transparent else "rgba(255,255,255,0.9)"
    brand_color = CERES_COLORS[0]
    
    # If transparent PNG, maybe we want the card to be semi-transparent or just the background transparent?
    # User said "transparent background", usually implies the outer padding is transparent.
    # The container logic below adds padding and a card. 
    # If transparent, let's make the BODY transparent, but keep the CONTAINER opaque 
    # so the charts are readable, but the surrounding 'margin' is clear?
    # Or does the user want NO container background?
    # Assuming standard "sticker" style: Transparent background, Opaque Card.
    
    if transparent:
        bg_color = "transparent"
        # We can keep card_color white
    
    # 1. Generate SVGs for each metric
    svgs = []
    
    # Helper to extract values
    def get_vals(key):
        return {t: metrics_dict[t].get(key, 0) for t in selected_tickers}
        
    # Revenue
    rev_data = {t: metrics_dict[t].get("revenue", 0)/scale_divisor for t in selected_tickers}
    svgs.append(generate_bar_chart_svg(
        rev_data, 
        f"Revenue ({scale_label} {currency_unit})", 
        lambda x: f"{x:.1f}"
    ))
    
    # EBIT Margin
    svgs.append(generate_bar_chart_svg(
        get_vals("ebit_margin"), "EBIT Margin (%)", lambda x: f"{x:.1f}%"
    ))
    
    # ROE
    svgs.append(generate_bar_chart_svg(
        get_vals("roe"), "ROE (%)", lambda x: f"{x:.1f}%"
    ))
    
    # P/E
    svgs.append(generate_bar_chart_svg(
        get_vals("pe_ratio"), "P/E Ratio", lambda x: f"{x:.1f}x"
    ))
    
    # Debt/equity
    svgs.append(generate_bar_chart_svg(
        get_vals("debt_equity"), "Debt/Equity", lambda x: f"{x:.2f}x"
    ))
    
    # EBITDA Margin
    svgs.append(generate_bar_chart_svg(
        get_vals("ebitda_margin"), "EBITDA Margin (%)", lambda x: f"{x:.1f}%"
    ))
    
    # CSS Grid Logic
    charts_html = "".join([f'<div class="chart-wrapper">{svg}</div>' for svg in svgs])
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            body {{
                margin: 0;
                padding: 0;
                background-color: {bg_color};
                font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
                width: {width}px;
                height: {height}px;
                display: flex;
                flex-direction: column;
                align-items: center;
                box-sizing: border-box;
                padding: 40px;
            }}
            .header {{
                text-align: center;
                margin-bottom: 30px;
                width: 100%;
            }}
            .logo {{
                color: {brand_color};
                font-size: 28px;
                font-weight: bold;
                letter-spacing: 4px;
                margin-bottom: 10px;
                text-transform: uppercase;
            }}
            .title {{
                color: #333;
                font-size: 36px;
                font-weight: 700;
            }}
            .grid-container {{
                display: grid;
                grid-template-columns: repeat(2, 1fr); 
                gap: 20px;
                width: 100%;
                flex: 1;
                justify-items: center;
                align-content: start;
            }}
            .chart-wrapper {{
                width: 100%;
                display: flex;
                justify-content: center;
            }}
            /* Adjust SVG size via CSS if needed, but they have intrinsic size */
            svg {{
                width: 100% !important;
                height: auto !important;
                max-width: 450px; 
                box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            }}
            
            .footer {{
                margin-top: auto;
                padding-top: 20px;
                color: #888;
                font-size: 16px;
                width: 100%;
                text-align: center;
                border-top: 1px solid #ddd;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <img src="data:image/png;base64,{LOGO_BASE64}" style="height: 50px; max-width: 200px; object-fit: contain; margin-bottom: 10px;" alt="Logo" />
            <div class="title">{title_str}</div>
        </div>
        
        <div class="grid-container">
            {charts_html}
        </div>
        
        <div class="footer">
            Generated by Ceres Wealth Analytics
        </div>
    </body>
    </html>
    """
    
    if file_type == "png":
        return html_to_png_playwright(html, width_px=width, height_px=height)
    else:
        return html_to_pdf_playwright(html, width_px=width, height_px=height)


    

def _format_dividend_yield(x):
    """Format dividend yield correctly"""
    if pd.notna(x):
        if x > 1:
            return f"{x:.2f}%"
        else:
            return f"{x*100:.2f}%"
    return "N/A"


def _get_historical_price(stock, date):
    """Get historical price at fiscal year end"""
    try:
        hist_data = stock.history(
            start=date,
            end=(
                date
                + pd.Timedelta(days=30)
            ),
            interval="1d",
        )
        if not hist_data.empty:
            return (
                hist_data["Close"].iloc[0]
            )
    except Exception:
        pass
    return None


def _calc_ttm(
    quarterly_income,
    quarterly_balance,
):
    """Calculate TTM metrics"""
    if (
        quarterly_income.empty
        or quarterly_balance.empty
    ):
        return None

    try:
        quarters_available = min(
            4, len(quarterly_income.columns)
        )
        ttm_revenue = (
            quarterly_income.loc[
                "Total Revenue"
            ]
            .iloc[:quarters_available]
            .sum()
        )
        ttm_ebit = (
            quarterly_income.loc[
                "Operating Income"
            ]
            .iloc[:quarters_available]
            .sum()
        )
        ttm_ebitda = (
            quarterly_income.loc[
                "EBITDA"
            ]
            .iloc[:quarters_available]
            .sum()
        )
        ttm_net_income = (
            quarterly_income.loc[
                "Net Income"
            ]
            .iloc[:quarters_available]
            .sum()
        )

        latest_balance = (
            quarterly_balance.iloc[:, 0]
        )

        cash_keys = [
            "Cash And Cash "
            "Equivalents",
            "Cash",
        ]
        ttm_cash = None
        for key in cash_keys:
            if key in latest_balance.index:
                ttm_cash = (
                    latest_balance[key]
                )
                break
        if ttm_cash is None:
            ttm_cash = 0

        debt_keys = [
            "Total Debt",
            "Long Term Debt",
        ]
        ttm_debt = None
        for key in debt_keys:
            if (
                key
                in latest_balance.index
            ):
                ttm_debt = (
                    latest_balance[key]
                )
                break
        if ttm_debt is None:
            ttm_debt = 0

        equity_keys = [
            "Total Stockholders "
            "Equity",
            "Stockholders Equity",
            "Total Equity",
        ]
        ttm_equity = None
        for key in equity_keys:
            if (
                key
                in latest_balance.index
            ):
                ttm_equity = (
                    latest_balance[key]
                )
                break
        if ttm_equity is None:
            ttm_equity = 0

        ttm_ebit_margin = (
            (ttm_ebit / ttm_revenue
             * 100)
            if ttm_revenue > 0
            else np.nan
        )
        ttm_ebitda_margin = (
            (ttm_ebitda / ttm_revenue
             * 100)
            if ttm_revenue > 0
            else np.nan
        )
        ttm_de = (
            (ttm_debt / ttm_equity)
            if ttm_equity > 0
            else np.nan
        )
        ttm_roe = (
            (ttm_net_income
             / ttm_equity * 100)
            if ttm_equity > 0
            else np.nan
        )

        return {
            "Ticker": None,
            "Fiscal Year": "TTM",
            "Revenue": ttm_revenue,
            "EBIT": ttm_ebit,
            "EBITDA": ttm_ebitda,
            "EBIT Margin (%)": (
                ttm_ebit_margin
            ),
            "EBITDA Margin (%)": (
                ttm_ebitda_margin
            ),
            "Cash & Equivalents": (
                ttm_cash
            ),
            "Total Debt": ttm_debt,
            "Total Equity": ttm_equity,
            "Debt/Equity": ttm_de,
            "ROE (%)": ttm_roe,
            "P/E Ratio": np.nan,
        }
    except Exception:
        return None


def _extract_metrics(
    stock, info, income_stmt,
    balance_sheet, latest_date
):
    """Extract key metrics for comparison"""
    try:
        revenue = income_stmt.loc[
            "Total Revenue", latest_date
        ]
        ebit = income_stmt.loc[
            "Operating Income", latest_date
        ]
        ebitda = income_stmt.loc[
            "EBITDA", latest_date
        ]
        net_income = income_stmt.loc[
            "Net Income", latest_date
        ]

        eps_keys = [
            "Basic EPS", "Diluted EPS"
        ]
        eps = None
        for key in eps_keys:
            if key in income_stmt.index:
                eps = income_stmt.loc[
                    key, latest_date
                ]
                break

        equity_keys = [
            "Total Stockholders Equity",
            "Stockholders Equity",
            "Total Equity",
        ]
        total_equity = None
        for key in equity_keys:
            if key in balance_sheet.index:
                total_equity = (
                    balance_sheet.loc[
                        key, latest_date
                    ]
                )
                break

        debt_keys = [
            "Total Debt",
            "Long Term Debt",
        ]
        total_debt = None
        for key in debt_keys:
            if key in balance_sheet.index:
                total_debt = (
                    balance_sheet.loc[
                        key, latest_date
                    ]
                )
                break

        if total_debt is None:
            total_debt = 0
        if total_equity is None:
            total_equity = 0

        ebit_margin = (
            (ebit / revenue * 100)
            if revenue > 0
            else 0
        )
        ebitda_margin = (
            (ebitda / revenue * 100)
            if revenue > 0
            else 0
        )
        roe = (
            (net_income / total_equity * 100)
            if total_equity > 0
            else 0
        )
        debt_equity = (
            (total_debt / total_equity)
            if total_equity > 0
            else 0
        )

        current_price = info.get(
            "currentPrice", 0
        )

        pe_ratio = info.get(
            "trailingPE", 0
        )

        return {
            "revenue": revenue,
            "ebit_margin": ebit_margin,
            "ebitda_margin": ebitda_margin,
            "roe": roe,
            "pe_ratio": pe_ratio,
            "debt_equity": debt_equity,
            "current_price": (
                current_price
            ),
            "market_cap": info.get(
                "marketCap", 0
            ),
            "52w_high": info.get(
                "fiftyTwoWeekHigh", 0
            ),
            "52w_low": info.get(
                "fiftyTwoWeekLow", 0
            ),
        }
    except Exception:
        return None


st.set_page_config(
    page_title="Stock Comparator",
    layout="wide",
)

# Language selector
lang = st.selectbox(
    "🌐 Language / Idioma",
    options=["English", "Português"],
    index=0,
)
lang_code = "en" if lang == "English" else "pt"

st.title(get_text(lang_code, "title"))

# Initialize session state
if "analysis_data" not in st.session_state:
    st.session_state.analysis_data = None
if "comparison_data" not in st.session_state:
    st.session_state.comparison_data = None
if "quotes_data" not in st.session_state:
    st.session_state.quotes_data = None
if "tickers" not in st.session_state:
    st.session_state.tickers = None

# User inputs
col1, col2 = st.columns(2)
with col1:
    ticker_input = st.text_input(
        get_text(lang_code, "tickers_input"),
        "AAPL,MSFT,GOOGL",
        help=get_text(lang_code, "tickers_help"),
    )

with col2:
    num_years = st.number_input(
        get_text(lang_code, "years_input"),
        min_value=1,
        max_value=10,
        value=3,
    )

tickers = [
    t.strip().upper()
    for t in ticker_input.split(",")
]

if st.button(
    get_text(lang_code, "analyze_btn"),
    width='content',
):
    with st.spinner(
        get_text(lang_code, "fetching")
    ):
        comparison_data = []
        quotes_data = []
        all_years = set()

        ticker_years = {}
        ticker_data_raw = {}

        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                income_stmt = (
                    stock.income_stmt
                )

                if income_stmt.empty:
                    st.warning(
                        f"{get_text(lang_code, 'no_data')} "
                        f"{ticker}"
                    )
                    continue

                years = [
                    date.year
                    for date in income_stmt.columns
                ]
                ticker_years[ticker] = (
                    set(years)
                )
                ticker_data_raw[ticker] = (
                    stock
                )

            except Exception as e:
                st.error(
                    f"{get_text(lang_code, 'error_fetching')} "
                    f"{ticker}: {str(e)}"
                )

        if ticker_years:
            all_years = set.intersection(
                *ticker_years.values()
            )
            if not all_years:
                st.warning(
                    get_text(
                        lang_code, "no_common_years"
                    )
                )
                all_years = set.union(
                    *ticker_years.values()
                )

        years_to_show = sorted(
            all_years, reverse=True
        )[:num_years]

        for ticker in tickers:
            try:
                stock = (
                    ticker_data_raw[ticker]
                )

                balance_sheet = (
                    stock.balance_sheet
                )
                income_stmt = (
                    stock.income_stmt
                )
                quarterly_income = (
                    stock.quarterly_income_stmt
                )
                quarterly_balance = (
                    stock.quarterly_balance_sheet
                )
                info = stock.info

                if (
                    balance_sheet.empty
                    or income_stmt.empty
                ):
                    continue

                current_price = info.get(
                    "currentPrice", None
                )
                market_cap = info.get(
                    "marketCap", None
                )
                pe_current = info.get(
                    "trailingPE", None
                )
                dividend_yield = info.get(
                    "dividendYield", None
                )
                fifty_two_week_high = (
                    info.get(
                        "fiftyTwoWeekHigh", None
                    )
                )
                fifty_two_week_low = (
                    info.get(
                        "fiftyTwoWeekLow", None
                    )
                )

                five_yr_avg_div_yield = (
                    info.get(
                        "fiveYearAvgDividendYield",
                        None,
                    )
                )

                quotes_data.append(
                    {
                        "Ticker": ticker,
                        "Current Price": (
                            current_price
                        ),
                        "Market Cap": market_cap,
                        "Trailing P/E": (
                            pe_current
                        ),
                        "52W High": (
                            fifty_two_week_high
                        ),
                        "52W Low": (
                            fifty_two_week_low
                        ),
                        "Dividend Yield": (
                            dividend_yield
                        ),
                        "5Y Avg Div Yield": (
                            five_yr_avg_div_yield
                        ),
                    }
                )

                ttm_data = _calc_ttm(
                    quarterly_income,
                    quarterly_balance,
                )
                if ttm_data:
                    ttm_data["Ticker"] = ticker
                    comparison_data.append(
                        ttm_data
                    )

                for year in years_to_show:
                    year_dates = [
                        date
                        for date in (
                            income_stmt.columns
                        )
                        if date.year == year
                    ]

                    if not year_dates:
                        continue

                    latest_date = (
                        year_dates[0]
                    )

                    revenue = (
                        income_stmt.loc[
                            "Total Revenue",
                            latest_date,
                        ]
                    )

                    ebit = income_stmt.loc[
                        "Operating Income",
                        latest_date,
                    ]

                    ebitda = (
                        income_stmt.loc[
                            "EBITDA",
                            latest_date,
                        ]
                    )

                    net_income = (
                        income_stmt.loc[
                            "Net Income",
                            latest_date,
                        ]
                    )

                    eps_keys = [
                        "Basic EPS",
                        "Diluted EPS",
                    ]
                    eps = None
                    for key in eps_keys:
                        if (
                            key
                            in income_stmt.index
                        ):
                            eps = (
                                income_stmt.loc[
                                    key,
                                    latest_date,
                                ]
                            )
                            break

                    hist_price = (
                        _get_historical_price(
                            stock,
                            latest_date,
                        )
                    )

                    cash_keys = [
                        "Cash And Cash "
                        "Equivalents",
                        "Cash",
                    ]
                    cash = None
                    for key in cash_keys:
                        if (
                            key
                            in balance_sheet.index
                        ):
                            cash = (
                                balance_sheet.loc[
                                    key,
                                    latest_date,
                                ]
                            )
                            break

                    if cash is None:
                        cash = 0

                    equity_keys = [
                        "Total Stockholders "
                        "Equity",
                        "Stockholders Equity",
                        "Total Equity",
                    ]
                    total_equity = None
                    for key in equity_keys:
                        if (
                            key
                            in balance_sheet.index
                        ):
                            total_equity = (
                                balance_sheet.loc[
                                    key,
                                    latest_date,
                                ]
                            )
                            break

                    debt_keys = [
                        "Total Debt",
                        "Long Term Debt",
                    ]
                    total_debt = None
                    for key in debt_keys:
                        if (
                            key
                            in balance_sheet.index
                        ):
                            total_debt = (
                                balance_sheet.loc[
                                    key,
                                    latest_date,
                                ]
                            )
                            break

                    if total_debt is None:
                        total_debt = 0
                    if total_equity is None:
                        total_equity = 0

                    ebit_margin = (
                        (ebit / revenue * 100)
                        if revenue > 0
                        else np.nan
                    )
                    ebitda_margin = (
                        (ebitda / revenue * 100)
                        if revenue > 0
                        else np.nan
                    )
                    debt_to_equity = (
                        (total_debt
                         / total_equity)
                        if total_equity > 0
                        else np.nan
                    )

                    roe = (
                        (net_income
                         / total_equity * 100)
                        if total_equity > 0
                        else np.nan
                    )

                    hist_pe_ratio = (
                        (hist_price / eps)
                        if (
                            eps
                            and eps > 0
                            and hist_price
                        )
                        else np.nan
                    )

                    comparison_data.append(
                        {
                            "Ticker": ticker,
                            "Fiscal Year": year,
                            "Revenue": revenue,
                            "EBIT": ebit,
                            "EBITDA": ebitda,
                            "EBIT Margin (%)": (
                                ebit_margin
                            ),
                            "EBITDA Margin (%)": (
                                ebitda_margin
                            ),
                            "Cash & Equivalents": (
                                cash
                            ),
                            "Total Debt": (
                                total_debt
                            ),
                            "Total Equity": (
                                total_equity
                            ),
                            "Debt/Equity": (
                                debt_to_equity
                            ),
                            "ROE (%)": roe,
                            "P/E Ratio": (
                                hist_pe_ratio
                            ),
                        }
                    )

            except Exception as e:
                st.error(
                    f"{get_text(lang_code, 'error_fetching')} "
                    f"{ticker}: {str(e)}"
                )

        st.session_state.analysis_data = (
            comparison_data
        )
        st.session_state.quotes_data = (
            quotes_data
        )
        st.session_state.tickers = tickers

# Display results if available
if (
    st.session_state.analysis_data
    is not None
):
    comparison_data = (
        st.session_state.analysis_data
    )
    quotes_data = (
        st.session_state.quotes_data
    )
    tickers = st.session_state.tickers

    # Financial Metrics Table
    st.subheader(
        get_text(lang_code, "financial_metrics")
    )
    df = pd.DataFrame(
        comparison_data
    )

    df = df.sort_values(
        ["Ticker", "Fiscal Year"],
        ascending=[True, False],
    )

    display_df = df.copy()
    numeric_cols = [
        "Revenue",
        "EBIT",
        "EBITDA",
        "Cash & Equivalents",
        "Total Debt",
        "Total Equity",
    ]
    for col in numeric_cols:
        display_df[col] = (
            display_df[col].apply(
                lambda x: (
                    f"${x/1e9:.2f}B"
                    if (
                        pd.notna(x)
                        and x != 0
                    )
                    else "N/A"
                )
            )
        )

    percentage_cols = [
        "EBIT Margin (%)",
        "EBITDA Margin (%)",
        "ROE (%)",
    ]
    for col in percentage_cols:
        display_df[col] = (
            display_df[col].apply(
                lambda x: (
                    f"{x:.2f}%"
                    if pd.notna(x)
                    else "N/A"
                )
            )
        )

    display_df["Debt/Equity"] = (
        display_df[
            "Debt/Equity"
        ].apply(
            lambda x: (
                f"{x:.2f}x"
                if pd.notna(x)
                else "N/A"
            )
        )
    )

    display_df["P/E Ratio"] = (
        display_df["P/E Ratio"].apply(
            lambda x: (
                f"{x:.2f}x"
                if pd.notna(x)
                else "N/A"
            )
        )
    )

    st.dataframe(
        display_df,
        width='content',
        hide_index=True,
    )

    csv = display_df.to_csv(
        index=False
    )
    st.download_button(
        label=get_text(
            lang_code, "download_metrics"
        ),
        data=csv,
        file_name=(
            "stock_comparison.csv"
        ),
        mime="text/csv",
    )

    # Quotes Table
    if quotes_data:
        st.subheader(
            get_text(lang_code, "current_quotes")
        )
        quotes_df = pd.DataFrame(
            quotes_data
        )

        display_quotes = (
            quotes_df.copy()
        )
        display_quotes[
            "Current Price"
        ] = display_quotes[
            "Current Price"
        ].apply(
            lambda x: (
                f"${x:.2f}"
                if pd.notna(x)
                else "N/A"
            )
        )
        display_quotes[
            "Market Cap"
        ] = display_quotes[
            "Market Cap"
        ].apply(
            lambda x: (
                f"${x/1e9:.2f}B"
                if pd.notna(x)
                else "N/A"
            )
        )
        display_quotes[
            "Trailing P/E"
        ] = display_quotes[
            "Trailing P/E"
        ].apply(
            lambda x: (
                f"{x:.2f}x"
                if pd.notna(x)
                else "N/A"
            )
        )
        display_quotes["52W High"] = (
            display_quotes[
                "52W High"
            ].apply(
                lambda x: (
                    f"${x:.2f}"
                    if pd.notna(x)
                    else "N/A"
                )
            )
        )
        display_quotes["52W Low"] = (
            display_quotes[
                "52W Low"
            ].apply(
                lambda x: (
                    f"${x:.2f}"
                    if pd.notna(x)
                    else "N/A"
                )
            )
        )

        display_quotes[
            "Dividend Yield"
        ] = display_quotes[
            "Dividend Yield"
        ].apply(
            _format_dividend_yield
        )
        display_quotes[
            "5Y Avg Div Yield"
        ] = display_quotes[
            "5Y Avg Div Yield"
        ].apply(
            _format_dividend_yield
        )

        st.dataframe(
            display_quotes,
            width='content',
            hide_index=True,
        )

        csv_quotes = (
            display_quotes.to_csv(
                index=False
            )
        )
        st.download_button(
            label=get_text(
                lang_code, "download_quotes"
            ),
            data=csv_quotes,
            file_name="stock_quotes.csv",
            mime="text/csv",
        )

    # Price Chart with Plotly
    if tickers:
        st.subheader(
            get_text(lang_code, "historical_prices")
        )

        period_options = {
            "1 Month": "1mo",
            "3 Months": "3mo",
            "6 Months": "6mo",
            "1 Year": "1y",
            "2 Years": "2y",
            "5 Years": "5y",
            "10 Years": "10y",
            "Max": "max",
        }

        selected_period = st.selectbox(
            get_text(lang_code, "select_period"),
            options=list(
                period_options.keys()
            ),
            index=3,
        )

        period_key = period_options[
            selected_period
        ]

        try:
            price_data = yf.download(
                tickers,
                period=period_key,
                progress=False,
            )

            fig = go.Figure()

            if len(tickers) == 1:
                if (
                    isinstance(
                        price_data,
                        pd.DataFrame,
                    )
                    and "Close"
                    in price_data.columns
                ):
                    close_prices = (
                        price_data["Close"]
                    )
                else:
                    close_prices = price_data

                fig.add_trace(
                    go.Scatter(
                        x=close_prices.index,
                        y=close_prices.values,
                        name=tickers[0],
                        line=dict(
                            color=CERES_COLORS[0],
                            width=3,
                        ),
                        hovertemplate=(
                            "<b>"
                            + tickers[0]
                            + "</b><br>"
                            "Date: %{x|%Y-%m-%d}<br>"
                            "Price: $%{y:.2f}"
                            "<extra></extra>"
                        ),
                    )
                )
            else:
                for idx, ticker in enumerate(
                    tickers
                ):
                    fig.add_trace(
                        go.Scatter(
                            x=(
                                price_data.index
                            ),
                            y=(
                                price_data[
                                    "Close"
                                ][ticker].values
                            ),
                            name=ticker,
                            line=dict(
                                color=(
                                    CERES_COLORS[
                                        idx
                                        % len(
                                            CERES_COLORS
                                        )
                                    ]
                                ),
                                width=3,
                            ),
                            hovertemplate=(
                                "<b>" + ticker + "</b><br>"
                                "Date: %{x|%Y-%m-%d}<br>"
                                "Price: $%{y:.2f}"
                                "<extra></extra>"
                            ),
                        )
                    )

            fig.update_layout(
                title=dict(
                    text=get_text(
                        lang_code,
                        "historical_prices",
                    ),
                    font=dict(
                        size=16, color="#ffffff"
                    ),
                ),
                xaxis=dict(
                    title=dict(
                        text="Date",
                        font=dict(color="#ffffff"),
                    ),
                    showgrid=True,
                    gridwidth=1,
                    gridcolor="rgba(255,255,255,0.1)",
                    showline=True,
                    linewidth=1,
                    linecolor="rgba(255,255,255,0.2)",
                    tickfont=dict(
                        color="#ffffff"
                    ),
                ),
                yaxis=dict(
                    title=dict(
                        text="Price ($)",
                        font=dict(color="#ffffff"),
                    ),
                    showgrid=True,
                    gridwidth=1,
                    gridcolor="rgba(255,255,255,0.1)",
                    showline=True,
                    linewidth=1,
                    linecolor="rgba(255,255,255,0.2)",
                    tickfont=dict(
                        color="#ffffff"
                    ),
                ),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                hovermode="x unified",
                height=515,
                width=1199,
                margin=dict(
                    l=60, r=40, t=80, b=60
                ),
                font=dict(color="#ffffff"),
                legend=dict(
                    font=dict(color="#ffffff"),
                    orientation="h",
                    x=0.5,
                    y=-0.2,
                    xanchor="center",
                    yanchor="top",
                )
            )

            st.plotly_chart(
                fig,
                width='content',
                config={
                    "displayModeBar": True,
                    "displaylogo": False,
                },
            )

        except Exception as e:
            st.error(
                f"{get_text(lang_code, 'error_fetching')} "
                f"price data: {str(e)}"
            )

    # Company Comparison Dashboard
    st.divider()
    st.subheader(
        get_text(lang_code, "comparison_dashboard")
    )

    if len(tickers) >= 2:
        selected_tickers = (
            st.multiselect(
                get_text(
                    lang_code, "select_companies"
                ),
                options=tickers,
                default=tickers[:2],
                key="comparison_tickers",
            )
        )

        if len(selected_tickers) >= 2:
            col1, col2 = st.columns([3, 1])
            with col1:
                currency_unit = st.text_input(
                    get_text(
                        lang_code, "currency_unit"
                    ),
                    value="USD",
                )
            with col2:
                scale = st.selectbox(
                    get_text(lang_code, "scale"),
                    ["Billion", "Million"],
                    index=0,
                )

            scale_divisor = (
                1e9 if scale == "Billion"
                else 1e6
            )
            scale_label = (
                "B" if scale == "Billion"
                else "M"
            )

            try:
                metrics_dict = {}
                for ticker in selected_tickers:
                    stock = yf.Ticker(ticker)
                    info = stock.info
                    income_stmt = (
                        stock.income_stmt
                    )
                    balance_sheet = (
                        stock.balance_sheet
                    )

                    if (
                        not income_stmt.empty
                    ):
                        latest_date = (
                            income_stmt.columns[0]
                        )
                        metrics = (
                            _extract_metrics(
                                stock,
                                info,
                                income_stmt,
                                balance_sheet,
                                latest_date,
                            )
                        )
                        metrics_dict[ticker] = (
                            metrics
                        )

                num_companies = len(
                    selected_tickers
                )

                # Create subplots with Plotly
                fig = make_subplots(
                    rows=2,
                    cols=3,
                    subplot_titles=(
                        get_text(
                            lang_code, "revenue"
                        ),
                        get_text(
                            lang_code, "ebit_margin"
                        ),
                        get_text(lang_code, "roe"),
                        get_text(
                            lang_code, "pe_ratio"
                        ),
                        get_text(
                            lang_code, "debt_equity"
                        ),
                        get_text(
                            lang_code,
                            "ebitda_margin",
                        ),
                    ),
                    specs=[
                        [
                            {"type": "bar"},
                            {"type": "bar"},
                            {"type": "bar"},
                        ],
                        [
                            {"type": "bar"},
                            {"type": "bar"},
                            {"type": "bar"},
                        ],
                    ],
                )

                # 1. Revenue
                revenues = [
                    metrics_dict[ticker][
                        "revenue"
                    ]
                    / scale_divisor
                    for ticker in selected_tickers
                ]
                for i, (ticker, value) in enumerate(
                    zip(selected_tickers, revenues)
                ):
                    fig.add_trace(
                        go.Bar(
                            x=[ticker],
                            y=[value],
                            marker=dict(
                                color=(
                                    CERES_COLORS[
                                        i
                                        % len(
                                            CERES_COLORS
                                        )
                                    ]
                                ),
                            ),
                            text=f"{value:.2f}",
                            textposition=(
                                "outside"
                            ),
                            textfont=dict(
                                color="#ffffff"
                            ),
                            showlegend=False,
                            hovertemplate=(
                                "<b>%{x}</b><br>"
                                f"{scale_label}: "
                                "%{y:.2f}"
                                "<extra></extra>"
                            ),
                        ),
                        row=1,
                        col=1,
                    )

                # 2. EBIT Margin
                ebit_margins = [
                    metrics_dict[ticker][
                        "ebit_margin"
                    ]
                    for ticker in selected_tickers
                ]
                for i, (ticker, value) in enumerate(
                    zip(
                        selected_tickers,
                        ebit_margins,
                    )
                ):
                    fig.add_trace(
                        go.Bar(
                            x=[ticker],
                            y=[value],
                            marker=dict(
                                color=(
                                    CERES_COLORS[
                                        i
                                        % len(
                                            CERES_COLORS
                                        )
                                    ]
                                ),
                            ),
                            text=f"{value:.2f}%",
                            textposition=(
                                "outside"
                            ),
                            textfont=dict(
                                color="#ffffff"
                            ),
                            showlegend=False,
                            hovertemplate=(
                                "<b>%{x}</b><br>"
                                "Margin: %{y:.2f}%"
                                "<extra></extra>"
                            ),
                        ),
                        row=1,
                        col=2,
                    )

                # 3. ROE
                roes = [
                    metrics_dict[ticker]["roe"]
                    for ticker in selected_tickers
                ]
                for i, (ticker, value) in enumerate(
                    zip(selected_tickers, roes)
                ):
                    fig.add_trace(
                        go.Bar(
                            x=[ticker],
                            y=[value],
                            marker=dict(
                                color=(
                                    CERES_COLORS[
                                        i
                                        % len(
                                            CERES_COLORS
                                        )
                                    ]
                                ),
                            ),
                            text=f"{value:.2f}%",
                            textposition=(
                                "outside"
                            ),
                            textfont=dict(
                                color="#ffffff"
                            ),
                            showlegend=False,
                            hovertemplate=(
                                "<b>%{x}</b><br>"
                                "ROE: %{y:.2f}%"
                                "<extra></extra>"
                            ),
                        ),
                        row=1,
                        col=3,
                    )

                # 4. P/E Ratio
                pe_ratios = [
                    metrics_dict[ticker][
                        "pe_ratio"
                    ]
                    for ticker in selected_tickers
                ]
                for i, (ticker, value) in enumerate(
                    zip(selected_tickers, pe_ratios)
                ):
                    fig.add_trace(
                        go.Bar(
                            x=[ticker],
                            y=[value],
                            marker=dict(
                                color=(
                                    CERES_COLORS[
                                        i
                                        % len(
                                            CERES_COLORS
                                        )
                                    ]
                                ),
                            ),
                            text=f"{value:.2f}x",
                            textposition=(
                                "outside"
                            ),
                            textfont=dict(
                                color="#ffffff"
                            ),
                            showlegend=False,
                            hovertemplate=(
                                "<b>%{x}</b><br>"
                                "P/E: %{y:.2f}x"
                                "<extra></extra>"
                            ),
                        ),
                        row=2,
                        col=1,
                    )

                # 5. Debt/Equity
                de_ratios = [
                    metrics_dict[ticker][
                        "debt_equity"
                    ]
                    for ticker in selected_tickers
                ]
                for i, (ticker, value) in enumerate(
                    zip(selected_tickers, de_ratios)
                ):
                    fig.add_trace(
                        go.Bar(
                            x=[ticker],
                            y=[value],
                            marker=dict(
                                color=(
                                    CERES_COLORS[
                                        i
                                        % len(
                                            CERES_COLORS
                                        )
                                    ]
                                ),
                            ),
                            text=f"{value:.2f}x",
                            textposition=(
                                "outside"
                            ),
                            textfont=dict(
                                color="#ffffff"
                            ),
                            showlegend=False,
                            hovertemplate=(
                                "<b>%{x}</b><br>"
                                "D/E: %{y:.2f}x"
                                "<extra></extra>"
                            ),
                        ),
                        row=2,
                        col=2,
                    )

                # 6. EBITDA Margin
                ebitda_margins = [
                    metrics_dict[ticker][
                        "ebitda_margin"
                    ]
                    for ticker in selected_tickers
                ]
                for i, (ticker, value) in enumerate(
                    zip(
                        selected_tickers,
                        ebitda_margins,
                    )
                ):
                    fig.add_trace(
                        go.Bar(
                            x=[ticker],
                            y=[value],
                            marker=dict(
                                color=(
                                    CERES_COLORS[
                                        i
                                        % len(
                                            CERES_COLORS
                                        )
                                    ]
                                ),
                            ),
                            text=f"{value:.2f}%",
                            textposition=(
                                "outside"
                            ),
                            textfont=dict(
                                color="#ffffff"
                            ),
                            showlegend=False,
                            hovertemplate=(
                                "<b>%{x}</b><br>"
                                "Margin: %{y:.2f}%"
                                "<extra></extra>"
                            ),
                        ),
                        row=2,
                        col=3,
                    )

                fig.update_layout(
                    height=530,
                    width=1210,
                    showlegend=False,
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(
                        size=11, color="#ffffff"
                    ),
                    margin=dict(
                        l=60, r=40, t=100, b=60
                    ),
                    title=dict(
                        text=(
                            ""
                            ""
                        ),
                        font=dict(
                            size=16,
                            color="#ffffff",
                        ),
                    ),
                )

                fig.update_xaxes(
                    showgrid=False,
                    showline=True,
                    linewidth=1,
                    linecolor="rgba(255,255,255,0.2)",
                    tickfont=dict(
                        color="#ffffff"
                    ),
                    title=dict(
                        font=dict(
                            color="#ffffff"
                        )
                    ),
                )
                fig.update_yaxes(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor="rgba(255,255,255,0.1)",
                    showline=True,
                    linewidth=1,
                    linecolor="rgba(255,255,255,0.2)",
                    tickfont=dict(
                        color="#ffffff"
                    ),
                    title=dict(
                        font=dict(
                            color="#ffffff"
                        )
                    ),
                )

                st.plotly_chart(
                    fig,
                    width='content',
                    config={
                        "displayModeBar": True,
                        "displaylogo": False,
                    },
                )

                # Social Media PDF Export
                st.divider()
                st.subheader("📱 Social Media Export")
                
                col_exp1, col_exp2, col_exp3 = st.columns([1, 1, 1])
                with col_exp1:
                    social_fmt = st.selectbox(
                        "Format", 
                        ["Square (1:1 - Post)", "Vertical (9:16 - Story)"],
                        key="social_fmt_select"
                    )
                
                with col_exp2:
                    file_fmt = st.selectbox(
                        "File Type",
                        ["PDF", "PNG (Transparent)"],
                        key="file_fmt_val"
                    )

                with col_exp3:
                    st.write("") # Spacer
                    st.write("") # Spacer
                    generate_btn = st.button("Generate Export")
                
                if generate_btn:
                    fmt_key = "Story" if "9:16" in social_fmt else "Square"
                    is_png = "PNG" in file_fmt
                    file_ext = "png" if is_png else "pdf"
                    mime_type = "image/png" if is_png else "application/pdf"
                    
                    with st.spinner(f"Generating {fmt_key} {file_ext.upper()}..."):
                        title_str = f"Comparison: {', '.join(selected_tickers)}"
                        
                        file_bytes = generate_social_export(
                            metrics_dict,
                            selected_tickers,
                            title_str,
                            scale_label,
                            scale_divisor,
                            currency_unit,
                            fmt=fmt_key,
                            file_type=file_ext,
                            transparent=True if is_png else False
                        )
                        
                        if file_bytes:
                            st.download_button(
                                label=f"📥 Download {fmt_key} {file_ext.upper()}",
                                data=file_bytes,
                                file_name=f"comparison_{fmt_key.lower()}_{'_'.join(selected_tickers)}.{file_ext}",
                                mime=mime_type,
                            )




                # Detailed Comparison
                # Table
                st.subheader(
                    get_text(
                        lang_code,
                        "detailed_comparison",
                    )
                )

                comparison_rows = {
                    "Metric": [
                        f"{get_text(lang_code, 'revenue')} "
                        f"({scale_label} "
                        f"{currency_unit})",
                        f"{get_text(lang_code, 'ebit_margin')} "
                        f"(%)",
                        f"{get_text(lang_code, 'ebitda_margin')} "
                        f"(%)",
                        f"{get_text(lang_code, 'roe')} "
                        f"(%)",
                        get_text(
                            lang_code, "pe_ratio"
                        ),
                        get_text(
                            lang_code, "debt_equity"
                        ),
                        get_text(
                            lang_code, "current_price"
                        ),
                        f"{get_text(lang_code, 'market_cap')} "
                        f"({scale_label})",
                        "52W High",
                        "52W Low",
                    ]
                }

                for ticker in (
                    selected_tickers
                ):
                    metrics = (
                        metrics_dict[ticker]
                    )
                    comparison_rows[ticker] = [
                        f"{metrics['revenue']/scale_divisor:.2f}",
                        f"{metrics['ebit_margin']:.2f}",
                        f"{metrics['ebitda_margin']:.2f}",
                        f"{metrics['roe']:.2f}",
                        f"{metrics['pe_ratio']:.2f}",
                        f"{metrics['debt_equity']:.2f}",
                        f"${metrics['current_price']:.2f}",
                        f"{metrics['market_cap']/scale_divisor:.2f}",
                        f"${metrics['52w_high']:.2f}",
                        f"${metrics['52w_low']:.2f}",
                    ]

                comparison_df = (
                    pd.DataFrame(
                        comparison_rows
                    )
                )
                st.dataframe(
                    comparison_df,
                    width='content',
                    hide_index=True,
                )

            except Exception as e:
                st.error(
                    f"{get_text(lang_code, 'error_creating')}: "
                    f"{str(e)}"
                )
        else:
            st.warning(
                get_text(
                    lang_code, "select_at_least_2"
                )
            )
    else:
        st.info(
            get_text(lang_code, "add_at_least_2")
        )