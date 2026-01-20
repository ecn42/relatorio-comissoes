import streamlit as st
import yfinance as yf
import pandas as pd
import sqlite3
import os
from datetime import datetime
from jinja2 import Template
import base64
from pathlib import Path
import io
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

# --- Configuration & Styling ---
st.set_page_config(page_title="Gerador de Trade Ideas", layout="wide")

PALETA_CORES = ["#825120", "#013220", "#8c6239", "#57575a", "#b08568", "#09202e", "#582308", "#7a6200"]
BRAND_BROWN = "#825120"
BRAND_BROWN_DARK = "#6B4219"
BLOCK_BG = "#F8F8F8"
TABLE_BORDER = "#E0D5CA"
TEXT_DARK = "#333333"

# --- Authentication Check ---
if not st.session_state.get("authenticated", False):
    st.warning("Por favor, faça o login na página Home primeiro.")
    st.stop()

# --- Database Setup ---
DB_PATH = "databases/trade_ideas.db"
ECO_DB_PATH = "databases/dados_economatico.db"

# Column Mapping for Economatica
ECONOMATICA_COLS = {
    "Valor de Mercado": "Valor de\nMercado\nAtual\nem milhares",
    "P/L 12m": "Preço / Lucro\n12 meses\n\nem vezes",
    "P/VP": "P / VPA\n\n\nem vezes",
    "ROE 12m": "ROE\n12 meses\n\nem %",
    "Dividend Yield 12m": "Dividend Yield\n12 meses\n\nem %",
    "Dívida Líq/EBITDA": "Dívida Líquida\nvs EBITDA\n12 meses\nem vezes",
    "Margem Líquida": "Margem Líquida\n12 meses\n\nem %",
    "Margem EBITDA": "Margem EBITDA\n12 meses\n\nem %",
    "Liquidez Vol (3m)": "Volume Médio\ndiário\n3 meses\nem milhares"
}


def format_df_for_html(df, decimal_places=2):
    """Format all numeric columns in a DataFrame to specified decimal places for HTML export."""
    df_formatted = df.copy()
    for col in df_formatted.columns:
        if col == "Código":  # Skip the ticker column
            continue
        try:
            # Convert to numeric, then format as string
            df_formatted[col] = pd.to_numeric(df_formatted[col], errors='coerce')
            df_formatted[col] = df_formatted[col].apply(
                lambda x: f"{x:.{decimal_places}f}" if pd.notna(x) else "-"
            )
        except:
            pass  # Keep original if conversion fails
    return df_formatted

def get_economatica_data(ticker):
    """Fetch data for a specific ticker from Economatica DB."""
    try:
        if not ticker: return None
        
        # Ticker format cleaning (remove .SA if present for this specific DB match, usually)
        # But user said "match 'Código' in the db to the ticker"
        # Let's clean it just in case
        t_clean = ticker.upper().replace(".SA", "").strip()
        
        conn = sqlite3.connect(ECO_DB_PATH)
        # Try exact match first
        df = pd.read_sql_query(f"SELECT * FROM sheet1 WHERE \"Código\" = '{t_clean}'", conn)
        
        if df.empty:
            # Try with .SA or without depending on what failed
             df = pd.read_sql_query(f"SELECT * FROM sheet1 WHERE \"Código\" LIKE '{t_clean}%'", conn)
             
        conn.close()
        
        if not df.empty:
            return df.iloc[0]
        return None
    except Exception as e:
        print(f"Error fetching Economatica data: {e}")
        return None

def get_peers_data(subsetor, columns_to_fetch=None):
    """Fetch peers in the same subsetor."""
    try:
        conn = sqlite3.connect(ECO_DB_PATH)
        
        cols_sql = "*"
        if columns_to_fetch:
            # Always include Código and Subsetor Bovespa
            needed = ["Código", "Subsetor Bovespa"] + list(columns_to_fetch)
            cols_sql = ", ".join([f'"{c}"' for c in set(needed)]) # handle duplicates
            
        query = f"SELECT {cols_sql} FROM sheet1 WHERE \"Subsetor Bovespa\" = '{subsetor}'"
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    except Exception as e:
        print(f"Error fetching peers: {e}")
        return pd.DataFrame()

def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # Updated schema with status, exit_price, and dates
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS trade_ideas (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT,
            entry_price REAL,
            target_price REAL,
            stop_price REAL,
            current_price REAL,
            exit_price REAL,
            upside_pct REAL,
            downside_pct REAL,
            result_pct REAL,
            notes TEXT,
            status TEXT DEFAULT 'ABERTA',
            start_date TEXT,
            end_date TEXT
        )
    """)
    # Add columns if they don't exist (simple migration)
    cursor.execute("PRAGMA table_info(trade_ideas)")
    columns = [col[1] for col in cursor.fetchall()]
    if 'exit_price' not in columns:
        cursor.execute("ALTER TABLE trade_ideas ADD COLUMN exit_price REAL")
    if 'result_pct' not in columns:
        cursor.execute("ALTER TABLE trade_ideas ADD COLUMN result_pct REAL")
    if 'status' not in columns:
        cursor.execute("ALTER TABLE trade_ideas ADD COLUMN status TEXT DEFAULT 'ABERTA'")
    if 'start_date' not in columns:
        cursor.execute("ALTER TABLE trade_ideas ADD COLUMN start_date TEXT")
    if 'end_date' not in columns:
        cursor.execute("ALTER TABLE trade_ideas ADD COLUMN end_date TEXT")
    if 'trade_type' not in columns:
        cursor.execute("ALTER TABLE trade_ideas ADD COLUMN trade_type TEXT DEFAULT 'LONG'")
    if 'ticker_short' not in columns:
        cursor.execute("ALTER TABLE trade_ideas ADD COLUMN ticker_short TEXT")
    if 'entry_price_short' not in columns:
        cursor.execute("ALTER TABLE trade_ideas ADD COLUMN entry_price_short REAL")
    if 'exit_price_short' not in columns:
        cursor.execute("ALTER TABLE trade_ideas ADD COLUMN exit_price_short REAL")
    if 'casa_analise' not in columns:
        cursor.execute("ALTER TABLE trade_ideas ADD COLUMN casa_analise TEXT")
    
    conn.commit()
    conn.close()

init_db()

# --- Helper Functions ---
def get_current_price(ticker):
    if not ticker:
        return 0.0
    
    # Add .SA if it's a Brazilian ticker and doesn't have it
    yf_ticker = ticker.upper().strip()
    if not yf_ticker.endswith(".SA") and not "^" in yf_ticker and "." not in yf_ticker:
        yf_ticker = f"{yf_ticker}.SA"
    
    try:
        stock = yf.Ticker(yf_ticker)
        # Using fast_info or regular info? regular info has 'currentPrice'
        price = stock.info.get("currentPrice") or stock.info.get("regularMarketPrice")
        if price is None:
            # Fallback to history if info fails
            hist = stock.history(period="1d")
            if not hist.empty:
                price = hist["Close"].iloc[-1]
        return price if price else 0.0
    except Exception as e:
        st.error(f"Erro ao buscar preço para {yf_ticker}: {e}")
        return 0.0

def brl(v: float) -> str:
    if v is None: return "0,00"
    return f"{v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

def pct(v: float) -> str:
    if v is None: return "0,00%"
    return f"{v * 100:.2f}%"

def image_to_base64(uploaded_file) -> str:
    """Convert uploaded image to base64 string for embedding in HTML."""
    if uploaded_file is None:
        return ""
    try:
        bytes_data = uploaded_file.getvalue()
        b64 = base64.b64encode(bytes_data).decode()
        file_type = uploaded_file.type if uploaded_file.type else "image/png"
        return f"data:{file_type};base64,{b64}"
    except:
        return ""

def generate_comparison_chart_b64(
    ticker,
    start_date_str,
    end_date_str=None,
    trade_type="LONG",
    entry_price=None,
    target_price=None,
    stop_price=None,
):
    """Generate a base64 line chart. For L&S, plots the ratio. Others, plots ticker vs IBOV dual axis with levels."""
    try:
        # Parse Dates safely
        def parse_any_date(d_str):
            if not d_str:
                return None
            for fmt in ("%Y-%m-%d", "%d/%m/%Y"):
                try:
                    return datetime.strptime(str(d_str).strip().split(" ")[0], fmt)
                except:
                    continue
            return None

        trade_start = parse_any_date(start_date_str)
        trade_end = parse_any_date(end_date_str)

        if not trade_start:
            trade_start = datetime.now() - pd.Timedelta(days=180)

        # 12 months lookback
        fetch_start = datetime.now() - pd.Timedelta(days=365)
        if trade_start < fetch_start:
            fetch_start = trade_start - pd.Timedelta(days=30)

        fetch_end = trade_end if trade_end else datetime.now()
        fetch_end += pd.Timedelta(days=5)

        if trade_type == "LS" and "/" in ticker:
            t_long, t_short = ticker.split("/")
            yf_long = (
                t_long
                if t_long.endswith(".SA") or "^" in t_long
                else f"{t_long}.SA"
            )
            yf_short = (
                t_short
                if t_short.endswith(".SA") or "^" in t_short
                else f"{t_short}.SA"
            )

            # Download each ticker separately to avoid column issues
            data_long = yf.download(
                yf_long, start=fetch_start, end=fetch_end, progress=False
            )
            data_short = yf.download(
                yf_short, start=fetch_start, end=fetch_end, progress=False
            )

            if data_long.empty or data_short.empty:
                st.warning(f"Dados vazios para {yf_long} ou {yf_short}")
                return ""

            # Use 'Adj Close' or 'Close'
            col_name = "Adj Close" if "Adj Close" in data_long.columns else "Close"

            # Handle MultiIndex columns (newer yfinance versions)
            if isinstance(data_long.columns, pd.MultiIndex):
                price_long = data_long[col_name].iloc[:, 0]
                price_short = data_short[col_name].iloc[:, 0]
            else:
                price_long = data_long[col_name]
                price_short = data_short[col_name]

            # Align indexes
            combined = pd.DataFrame(
                {"long": price_long, "short": price_short}
            ).dropna()

            if combined.empty:
                st.warning("Não foi possível alinhar os dados dos tickers")
                return ""

            df_plot = combined["long"] / combined["short"]

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(
                df_plot.index,
                df_plot,
                label=f"Ratio {ticker}",
                color=BRAND_BROWN,
                linewidth=2.5,
            )
            ax.axvline(
                x=trade_start, color=BRAND_BROWN, linestyle="-", alpha=0.3, label="Início"
            )
            if trade_end:
                ax.axvline(
                    x=trade_end, color="#b91c1c", linestyle="-", alpha=0.3, label="Fim"
                )

            ax.set_title(f"Evolução do Ratio: {ticker}", fontsize=12, fontweight="bold")
            ax.set_ylabel("Ratio", fontsize=10)
            ax.grid(True, alpha=0.2)
            ax.legend(loc="best", fontsize=9)
            fig.autofmt_xdate()

        else:
            # Clean ticker
            ticker_clean = ticker.strip().upper()
            yf_ticker = (
                ticker_clean
                if ticker_clean.endswith(".SA") or "^" in ticker_clean
                else f"{ticker_clean}.SA"
            )

            # Download each ticker separately
            data_ticker = yf.download(
                yf_ticker, start=fetch_start, end=fetch_end, progress=False
            )
            data_ibov = yf.download(
                "^BVSP", start=fetch_start, end=fetch_end, progress=False
            )

            if data_ticker.empty:
                st.warning(f"Sem dados para {yf_ticker}")
                return ""
            if data_ibov.empty:
                st.warning("Sem dados para IBOV")
                return ""

            # Use 'Adj Close' or 'Close'
            col_name = "Adj Close" if "Adj Close" in data_ticker.columns else "Close"

            # Handle MultiIndex columns (newer yfinance versions)
            if isinstance(data_ticker.columns, pd.MultiIndex):
                price_ticker = data_ticker[col_name].iloc[:, 0]
                price_ibov = data_ibov[col_name].iloc[:, 0]
            else:
                price_ticker = data_ticker[col_name]
                price_ibov = data_ibov[col_name]

            # Dual Axis Plot
            fig, ax1 = plt.subplots(figsize=(10, 5))

            # Plot Ticker on Left Axis
            ax1.plot(
                price_ticker.index,
                price_ticker,
                label=ticker_clean,
                color=BRAND_BROWN,
                linewidth=2.0,
            )
            ax1.set_ylabel(
                f"Preço ({ticker_clean})",
                color=BRAND_BROWN,
                fontsize=10,
                fontweight="bold",
            )
            ax1.tick_params(axis="y", labelcolor=BRAND_BROWN)

            # Plot BVSP on Right Axis
            ax2 = ax1.twinx()
            ax2.plot(
                price_ibov.index,
                price_ibov,
                label="IBOV",
                color="#999999",
                linestyle="--",
                alpha=0.6,
                linewidth=1.5,
            )
            ax2.set_ylabel("Pontos (IBOV)", color="#999999", fontsize=10)
            ax2.tick_params(axis="y", labelcolor="#999999")

            # Horizontal Lines (Levels)
            if entry_price and entry_price > 0:
                ax1.axhline(
                    y=entry_price,
                    color="#333",
                    linestyle=":",
                    linewidth=1.5,
                    alpha=0.8,
                    label="Entrada",
                )

            if target_price and target_price > 0:
                ax1.axhline(
                    y=target_price,
                    color="#15803d",
                    linestyle="--",
                    linewidth=1.5,
                    alpha=0.9,
                    label="Alvo",
                )

            if stop_price and stop_price > 0:
                ax1.axhline(
                    y=stop_price,
                    color="#b91c1c",
                    linestyle="--",
                    linewidth=1.5,
                    alpha=0.9,
                    label="Stop",
                )

            # Vertical lines for Trade Start/End
            ax1.axvline(x=trade_start, color=BRAND_BROWN, linestyle="-", alpha=0.2)
            if trade_end:
                ax1.axvline(x=trade_end, color="#b91c1c", linestyle="-", alpha=0.2)

            ax1.set_title(
                f"Evolução de Preço: {ticker_clean} vs IBOV (12 Meses)",
                fontsize=12,
                fontweight="bold",
            )
            ax1.grid(True, alpha=0.15)

            # Combine legends
            lines_1, labels_1 = ax1.get_legend_handles_labels()
            lines_2, labels_2 = ax2.get_legend_handles_labels()
            ax1.legend(
                lines_1 + lines_2,
                labels_1 + labels_2,
                loc="upper left",
                fontsize=8,
                framealpha=0.9,
            )

            fig.autofmt_xdate()

        # Convert to B64
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)  # Important: reset buffer position
        b64_str = base64.b64encode(buf.getvalue()).decode()
        return f"data:image/png;base64,{b64_str}"

    except Exception as e:
        st.error(f"Erro ao gerar gráfico: {e}")
        import traceback
        st.code(traceback.format_exc())  # Show full traceback for debugging
        return ""

def generate_consolidated_performance_chart_b64(df_perf):
    """Generate a bar chart showing result % for each finished trade."""
    try:
        finished_trades = df_perf[df_perf["status"] == "ENCERRADA"].copy()
        if finished_trades.empty:
            return ""
        
        # Sort by date
        finished_trades = finished_trades.sort_values("dt_end")
        
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = [BRAND_BROWN if x >= 0 else "#b91c1c" for x in finished_trades["result_pct"]]
        
        bars = ax.bar(finished_trades["ticker"] + " (" + finished_trades["id"].astype(str) + ")", 
                      finished_trades["result_pct"] * 100, 
                      color=colors, alpha=0.8)
        
        ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
        ax.set_title("Resultado por Operação (%)", fontsize=14, fontweight="bold")
        ax.set_ylabel("Retorno %", fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        
        # Add labels on top of bars
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, yval + (0.5 if yval >= 0 else -1.5), 
                    f"{yval:.1f}%", ha='center', va='bottom', fontsize=9, fontweight='bold')

        plt.xticks(rotation=45, ha='right')
        
        # Convert to B64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        plt.close(fig)
        return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"
    except Exception as e:
        print(f"Error generating consolidated chart: {e}")
        return ""

# --- Sidebar Configuration ---
st.sidebar.header("🎨 Identidade Visual")
logo_upload = st.sidebar.file_uploader("Logo da Empresa (opcional)", type=["png", "jpg", "jpeg"])
logo_b64 = image_to_base64(logo_upload) if logo_upload else ""
st.sidebar.markdown(f'<div style="background-color: {BRAND_BROWN}; height: 5px; margin: 10px 0;"></div>', unsafe_allow_html=True)

# --- PDF Generation (Playwright) ---
def html_to_pdf(html_content):
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        st.error("Playwright não encontrado no ambiente Python.")
        return None

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.set_content(html_content, wait_until="networkidle")
            
            # Calculate content height dynamically
            content_height = page.evaluate("() => document.body.scrollHeight")
            # Add margins (top 15mm + bottom 20mm = 35mm ≈ 132px at 96dpi)
            total_height_px = content_height + 140
            # Convert to mm (96 dpi: 1mm ≈ 3.78px)
            total_height_mm = total_height_px / 3.78
            
            pdf = page.pdf(
                width="210mm",  # A4 width
                height=f"{total_height_mm}mm",  # Dynamic height based on content
                print_background=True,
                margin={
                    "top": "15mm",
                    "right": "12mm",
                    "bottom": "20mm",
                    "left": "12mm"
                },
                prefer_css_page_size=False,  # Use our calculated size instead of CSS
            )
            browser.close()
            return pdf
    except Exception as e:
        st.error(f"Erro ao gerar PDF com Playwright: {e}")
        return None

# --- HTML Template ---
# --- HTML Templates ---

# Template for Performance Report (Consolidated)
PERFORMANCE_REPORT_TEMPLATE = """
<!DOCTYPE html>
<html lang="pt-br">
<head>
    <meta charset="UTF-8">
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Open+Sans:wght@400;600;700&display=swap');
        
        @page { margin: 15mm 12mm 20mm 12mm; }
        
        * { box-sizing: border-box; }
        
        html, body { 
            font-family: 'Open Sans', 'Segoe UI', Arial, sans-serif; 
            color: #333; 
            margin: 0; 
            padding: 0; 
            background-color: white;
            font-size: 11px;
            line-height: 1.4;
        }
        
        :root {
            --brand: {{ color_primary }};
            --brand-dark: #6B4219;
            --bg: #F8F8F8;
            --text: #333333;
            --tbl-border: #E0D5CA;
            --light: #F5F5F5;
        }

        .header { 
            background: var(--brand); 
            color: white; 
            padding: 18px 24px; 
            display: flex; 
            align-items: center; 
            justify-content: space-between;
            margin-bottom: 20px;
            border-radius: 4px;
            page-break-inside: avoid;
        }
        .header-logo { height: 50px; max-width: 150px; object-fit: contain; }
        .header-text { text-align: right; }
        .title { font-size: 22px; font-weight: 700; margin: 0; }
        .period { font-size: 12px; opacity: 0.9; }

        .section-header { 
            background: var(--bg); 
            padding: 8px 16px; 
            border-left: 4px solid var(--brand); 
            font-weight: 700; 
            color: var(--brand); 
            margin: 20px 0 15px 0;
            font-size: 14px;
            page-break-after: avoid;
            page-break-inside: avoid;
        }

        .metrics-grid { 
            display: flex; 
            flex-wrap: wrap;
            gap: 12px; 
            margin-bottom: 25px;
            page-break-inside: avoid;
        }
        .metric-card { 
            flex: 1 1 calc(33.33% - 12px);
            min-width: 140px;
            padding: 15px; 
            border: 1px solid var(--tbl-border); 
            border-radius: 6px; 
            background-color: #fcfcfc; 
            text-align: center;
            page-break-inside: avoid;
        }
        .metric-label { font-size: 9px; text-transform: uppercase; color: #888; font-weight: 600; margin-bottom: 8px; }
        .metric-value { font-size: 18px; font-weight: 700; }
        
        .positive { color: #15803d; }
        .negative { color: #b91c1c; }

        .chart-container { 
            margin: 20px 0; 
            text-align: center; 
            border: 1px solid var(--tbl-border); 
            border-radius: 6px; 
            padding: 12px; 
            background: #fff;
            page-break-inside: avoid;
            max-height: 300px;
            overflow: hidden;
        }
        .chart-container img { 
            max-width: 100%; 
            max-height: 280px;
            height: auto;
            object-fit: contain;
        }

        /* Enhanced Table Styling */
        .table-wrapper { width: 100%; overflow-x: auto; margin-top: 10px; }
        
        table { width: 100%; border-collapse: separate; border-spacing: 0; font-size: 11px; }
        thead { display: table-header-group; }

        .peer-table, table.dataframe {
            width: 100%; border-collapse: separate; border-spacing: 0; font-size: 11px; margin-bottom: 0;
        }
        
        th { 
            background-color: var(--light); color: var(--text); padding: 10px 8px; 
            text-align: right; border-bottom: 2px solid var(--tbl-border); font-weight: 700; 
        }
        th:first-child { text-align: left; }

        td { padding: 8px; border-bottom: 1px solid var(--tbl-border); text-align: right; }
        td:first-child { text-align: left; font-weight: 600; }
        
        tr { page-break-inside: avoid; }
        tr:nth-child(even) td { background-color: var(--light); }

        .footer { 
            margin-top: 30px; text-align: center; font-size: 9px; color: #999; 
            border-top: 1px solid #eee; padding-top: 15px; page-break-inside: avoid;
        }
        
        @media print {
            body { -webkit-print-color-adjust: exact !important; print-color-adjust: exact !important; }
        }
    </style>
</head>
<body>
    <div class="page">
        <div class="header">
            {% if logo %}
            <img src="{{ logo }}" class="header-logo" alt="Logo">
            {% else %}<div style="width: 1px;"></div>{% endif %}
            <div class="header-text">
                <div class="title">Relatório de Performance</div>
                <div class="period">{{ report_period }}</div>
            </div>
        </div>

        <div class="section-header">Métricas Consolidadas</div>
        <div class="metrics-grid">
            <div class="metric-card"><div class="metric-label">Total de Ideias</div><div class="metric-value">{{ total_trades }}</div></div>
            <div class="metric-card"><div class="metric-label">Taxa de Acerto</div><div class="metric-value positive">{{ win_rate }}</div></div>
            <div class="metric-card"><div class="metric-label">Média Lucro/Perda</div><div class="metric-value {% if avg_pl_raw >= 0 %}positive{% else %}negative{% endif %}">{{ avg_pl }}</div></div>
            <div class="metric-card"><div class="metric-label">Trades Encerradas</div><div class="metric-value">{{ closed_trades }}</div></div>
            <div class="metric-card"><div class="metric-label">Duração Média</div><div class="metric-value">{{ avg_duration }} dias</div></div>
            <div class="metric-card"><div class="metric-label">Exp. Matemática</div><div class="metric-value">{{ exp_math }}</div></div>
        </div>

        {% if performance_chart %}
        <div class="section-header">Desempenho por Ativo</div>
        <div class="chart-container">
            <img src="{{ performance_chart }}" alt="Consolidated Performance Chart">
        </div>
        {% endif %}

        <div class="section-header">Histórico de Operações</div>
        <div class="table-wrapper">
            <table>
                <thead>
                    <tr><th>Ativo</th><th>Início</th><th>Fim</th><th>Resultado</th></tr>
                </thead>
                <tbody>
                    {% for trade in trades_list %}
                    <tr>
                        <td style="font-weight: 600;">{{ trade.ticker }}</td>
                        <td>{{ trade.start }}</td>
                        <td>{{ trade.end }}</td>
                        <td class="{% if trade.raw_result >= 0 %}positive{% else %}negative{% endif %}" style="font-weight: 700;">{{ trade.result }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>

        <div class="footer">
            Este relatório consolida a performance histórica das Trade Ideas.<br>
            Ceres Wealth Management | Gerado em {{ generated_at }}
        </div>
    </div>
</body>
</html>
"""

# Template for Single Trade Idea Report
REPORT_HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="pt-br">
<head>
    <meta charset="UTF-8">
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Open+Sans:wght@400;600;700&display=swap');
        
        @page { margin: 15mm 12mm 20mm 12mm; }
        
        * { box-sizing: border-box; }
        
        html, body { 
            font-family: 'Open Sans', 'Segoe UI', Arial, sans-serif; 
            color: #333; 
            margin: 0; 
            padding: 0; 
            background-color: white;
            font-size: 11px;
            line-height: 1.4;
        }
        
        :root {
            --brand: {{ color_primary }};
            --brand-dark: #6B4219;
            --bg: #F8F8F8;
            --text: #333333;
            --tbl-border: #E0D5CA;
            --light: #F5F5F5;
        }

        .header { 
            background: var(--brand); 
            color: white; 
            padding: 18px 24px; 
            display: flex; 
            align-items: center; 
            justify-content: space-between;
            margin-bottom: 20px;
            border-radius: 4px;
            page-break-inside: avoid;
        }
        .header-logo { height: 50px; max-width: 150px; object-fit: contain; }
        .header-text { text-align: right; }
        .title { font-size: 22px; font-weight: 700; margin: 0; }
        .subtitle { font-size: 14px; opacity: 0.9; margin-top: 2px;}

        .section-header { 
            background: var(--bg); 
            padding: 8px 16px; 
            border-left: 4px solid var(--brand); 
            font-weight: 700; 
            color: var(--brand); 
            margin: 20px 0 15px 0;
            font-size: 14px;
            page-break-after: avoid;
            page-break-inside: avoid;
        }

        .metrics-grid { 
            display: flex; 
            flex-wrap: wrap;
            gap: 12px; 
            margin-bottom: 25px;
            page-break-inside: avoid;
        }
        .metric-card { 
            flex: 1 1 calc(25% - 12px);
            min-width: 120px;
            padding: 15px; 
            border: 1px solid var(--tbl-border); 
            border-radius: 6px; 
            background-color: #fcfcfc; 
            text-align: center;
            page-break-inside: avoid;
        }
        .metric-label { font-size: 9px; text-transform: uppercase; color: #888; font-weight: 600; margin-bottom: 8px; }
        .metric-value { font-size: 16px; font-weight: 700; }
        
        .positive { color: #15803d; }
        .negative { color: #b91c1c; }

        .chart-container { 
            margin: 20px 0; 
            text-align: center; 
            border: 1px solid var(--tbl-border); 
            border-radius: 6px; 
            padding: 12px; 
            background: #fff;
            page-break-inside: avoid;
            max-height: 350px;
            overflow: hidden;
        }
        .chart-container img { 
            max-width: 100%; 
            max-height: 320px;
            height: auto;
            object-fit: contain;
        }

        /* Enhanced Table Styling */
        .table-wrapper { width: 100%; overflow-x: auto; margin-top: 10px; }
        
        table { width: 100%; border-collapse: separate; border-spacing: 0; font-size: 11px; }
        thead { display: table-header-group; }

        .peer-table, table.dataframe {
            width: 100%; border-collapse: separate; border-spacing: 0; font-size: 11px; margin-bottom: 0;
        }
        
        th { 
            background-color: var(--light); color: var(--text); padding: 10px 8px; 
            text-align: right; border-bottom: 2px solid var(--tbl-border); font-weight: 700; 
        }
        th:first-child { text-align: left; }

        td { padding: 8px; border-bottom: 1px solid var(--tbl-border); text-align: right; }
        td:first-child { text-align: left; font-weight: 600; }
        
        tr { page-break-inside: avoid; }
        tr:nth-child(even) td { background-color: var(--light); }

        .notes-section { 
            padding: 15px; background: #fafafa; border: 1px solid #eee; 
            border-radius: 5px; line-height: 1.5; page-break-inside: avoid; margin-bottom: 15px;
        }
        .notes-section h3 { margin: 0 0 8px 0; color: #444; border-bottom: 1px solid #ddd; padding-bottom: 5px; font-size: 13px; }
        .notes-section p { margin: 0; font-size: 11px; white-space: pre-wrap; }

        .footer { 
            margin-top: 30px; text-align: center; font-size: 9px; color: #999; 
            border-top: 1px solid #eee; padding-top: 15px; page-break-inside: avoid;
        }
        
        @media print {
            body { -webkit-print-color-adjust: exact !important; print-color-adjust: exact !important; }
        }
    </style>
</head>
<body>
    <div class="page">
        <div class="header">
            {% if logo %}
            <img src="{{ logo }}" class="header-logo" alt="Logo">
            {% else %}<div style="width: 1px;"></div>{% endif %}
            <div class="header-text">
                <div class="title">{{ trade_type_label }}</div>
                <div class="subtitle">{{ ticker }} | {{ date }}{% if casa_analise %} | {{ casa_analise }}{% endif %}</div>
            </div>
        </div>

        <div class="section-header">Parâmetros da Operação</div>
        <div class="metrics-grid">
            <div class="metric-card"><div class="metric-label">{{ label_entry }}</div><div class="metric-value">{{ prefix }} {{ entry_price }}</div></div>
            <div class="metric-card"><div class="metric-label">{{ label_target }}</div><div class="metric-value positive">{{ prefix }} {{ target_price }}</div></div>
            <div class="metric-card"><div class="metric-label">{{ label_stop }}</div><div class="metric-value negative">{{ prefix }} {{ stop_price }}</div></div>
            <div class="metric-card"><div class="metric-label">Upside Est.</div><div class="metric-value positive">{{ upside_pct }}</div></div>
            <div class="metric-card"><div class="metric-label">Downside Est.</div><div class="metric-value negative">{{ downside_pct }}</div></div>
            <div class="metric-card"><div class="metric-label">Status</div><div class="metric-value">{{ status }}</div></div>
        </div>

        {% if comparison_chart %}
        <div class="section-header">Análise Gráfica</div>
        <div class="chart-container">
            <img src="{{ comparison_chart }}" alt="Trade Analysis Chart">
        </div>
        {% endif %}

        {% if comparison_table %}
        <div class="section-header">{{ comparison_title }}</div>
        <div class="table-wrapper">
            {{ comparison_table }}
        </div>
        {% endif %}

        {% if notes %}
        <div class="section-header">Tese de Investimento / Notas</div>
        <div class="notes-section">
            <p>{{ notes }}</p>
        </div>
        {% endif %}

        <div class="footer">
            Este material é apenas para fins informativos e não constitui recomendação de investimento.<br>
            Ceres Wealth Management | Gerado em {{ date }}
        </div>
    </div>
</body>
</html>
"""

# --- Main UI ---
st.title("🚀 Gerador de Trade Ideas")
st.write("Preencha os dados abaixo para gerar um relatório de Trade Idea.")

# --- Trade Type selection ---
trade_type = st.radio("Tipo de Operação", ["LONG (Compra)", "SHORT (Venda)", "LONG & SHORT (Pair)"], horizontal=True)

col1, col2 = st.columns(2)

if trade_type == "LONG & SHORT (Pair)":
    with col1:
        ticker_long = st.text_input("Ticker LONG (comprado)", "").upper().strip()
        ticker_short = st.text_input("Ticker SHORT (vendido)", "").upper().strip()
        
        if st.button("Buscar Preços Atuais") and ticker_long and ticker_short:
            with st.spinner("Buscando preços..."):
                p_long = get_current_price(ticker_long)
                p_short = get_current_price(ticker_short)
                if p_long > 0 and p_short > 0:
                    st.session_state["p_long"] = p_long
                    st.session_state["p_short"] = p_short
                    st.session_state["current_ratio"] = p_long / p_short
                    st.success(f"Ratio atual {ticker_long}/{ticker_short}: {st.session_state['current_ratio']:.4f}")
                else:
                    st.error("Erro ao buscar um ou ambos os preços.")

        p_long_val = st.number_input("Preço Ativo LONG (R$)", value=st.session_state.get("p_long", 0.0), format="%.2f")
        p_short_val = st.number_input("Preço Ativo SHORT (R$)", value=st.session_state.get("p_short", 0.0), format="%.2f")
        
        current_ratio = 0.0
        if p_long_val > 0 and p_short_val > 0:
            current_ratio = p_long_val / p_short_val
        st.info(f"Ratio Atual: **{current_ratio:.4f}**")

        entry_ratio = st.number_input("Ratio de Entrada", value=current_ratio if current_ratio > 0 else 0.0, format="%.4f")

    with col2:
        entry_date = st.date_input("Data de Entrada", value=datetime.today())
        target_ratio = st.number_input("Ratio Alvo", value=entry_ratio * 1.1 if entry_ratio > 0 else 0.0, format="%.4f")
        stop_ratio = st.number_input("Ratio Stop", value=entry_ratio * 0.95 if entry_ratio > 0 else 0.0, format="%.4f")
        
    ticker = f"{ticker_long}/{ticker_short}"
    entry_price = entry_ratio
    target_price = target_ratio
    stop_price = stop_ratio
    current_price = current_ratio
    
else: # LONG or SHORT
    is_short = (trade_type == "SHORT (Venda)")
    with col1:
        ticker = st.text_input("Ticker (ex: PETR4, ITUB4, AAPL, ^BVSP)", "").upper().strip()
        
        if st.button("Buscar Preço Atual") and ticker:
            with st.spinner("Buscando preço..."):
                price = get_current_price(ticker)
                if price > 0:
                    st.session_state["current_price"] = price
                    st.success(f"Preço atual de {ticker}: R$ {brl(price)}")
                else:
                    st.error("Não foi possível encontrar o preço.")

        # --- SECTOR DATA & PEERS ---
        eco_data = get_economatica_data(ticker)
        sector_name = None
        peers_df = pd.DataFrame()
        
        if eco_data is not None:
            sector_name = eco_data.get("Subsetor Bovespa")
            st.info(f"🏢 **Setor Identificado:** {sector_name}")
            
            with st.expander("📊 Comparação com Pares (Setorial)", expanded=False):
                default_cols = ["P/L 12m", "P/VP", "ROE 12m", "Dividend Yield 12m"]
                sel_cols = st.multiselect("Métricas para Comparar:", 
                                          options=list(ECONOMATICA_COLS.keys()),
                                          default=default_cols)
                
                # Fetch peers
                if sector_name:
                    # Always include Market Cap for sorting
                    mcap_col_db = ECONOMATICA_COLS["Valor de Mercado"]
                    
                    # Map selected friendly names to DB columns
                    db_cols_selected = [ECONOMATICA_COLS[c] for c in sel_cols]
                    
                    # Ensure mcap is in the fetch list
                    db_cols_fetch = list(set(db_cols_selected + [mcap_col_db]))
                    
                    raw_peers = get_peers_data(sector_name, db_cols_fetch)
                    
                    if not raw_peers.empty:
                        # Clean and Sort by Market Cap
                        # Attempt to convert market cap to numeric, coercing errors
                        raw_peers[mcap_col_db] = pd.to_numeric(raw_peers[mcap_col_db], errors='coerce').fillna(0)
                        raw_peers = raw_peers.sort_values(by=mcap_col_db, ascending=False)
                        
                        # Identify Top 5 + Current
                        all_tickers = raw_peers["Código"].unique().tolist()
                        
                        # Top 5
                        top_5 = raw_peers.head(5)["Código"].tolist()
                        
                        # Ensure current ticker is in default if available
                        # Note: ticker variable from input might not match exactly DB "Código" if not cleaned same way
                        # But let's assume raw_peers has the correct codes
                        # Try to match fuzzy or exact
                        current_clean = ticker.upper().replace(".SA", "").strip()
                        
                        default_selection = set(top_5)
                        if current_clean in all_tickers:
                            default_selection.add(current_clean)
                        
                        # Multiselect for peers
                        selected_peers_list = st.multiselect("Selecionar Pares (Default: Top 5 Market Cap):", 
                                                             options=all_tickers,
                                                             default=list(default_selection))
                        
                        # Filter DataFrame based on selection
                        raw_peers_filtered = raw_peers[raw_peers["Código"].isin(selected_peers_list)].copy()
                        
                        # Rename columns for display
                        rev_map = {v: k for k, v in ECONOMATICA_COLS.items()}
                        raw_peers_renamed = raw_peers_filtered.rename(columns=rev_map)
                        
                        # Filter only selected columns (remove mcap if not selected by user)
                        # Ensure "Código" is there
                        final_cols = ["Código"] + sel_cols
                        final_peers = raw_peers_renamed[final_cols].copy()
                        
                        # Round numeric columns for display
                        numeric_cols = final_peers.select_dtypes(include=['float64', 'float32', 'int64', 'int32']).columns
                        final_peers[numeric_cols] = final_peers[numeric_cols].round(2)
                        
                        st.dataframe(final_peers, hide_index=True, use_container_width=True)
                        
                        # Option to include in report
                        include_comp = st.checkbox("Incluir tabela de comparação no relatório PDF", value=True)
                        if include_comp:
                            peer_html = format_df_for_html(final_peers).to_html(
                                classes="table peer-table", index=False, border=0
                            )
                            st.session_state["peer_comparison_html"] = peer_html
                            st.session_state["peer_comparison_title"] = f"Comparação Setorial: {sector_name}"
                        else:
                             st.session_state.pop("peer_comparison_html", None)

        current_price = st.number_input("Preço Atual (R$)", 
                                       value=st.session_state.get("current_price", 0.0), 
                                       step=0.01, format="%.2f")

        entry_price = st.number_input("Preço de Entrada (R$)", value=current_price if current_price > 0 else 0.0, step=0.01, format="%.2f")

    with col2:
        entry_date = st.date_input("Data de Entrada", value=datetime.today())
        if not is_short:
            target_price = st.number_input("Preço Alvo (R$)", value=entry_price * 1.2 if entry_price > 0 else 0.0, step=0.01, format="%.2f")
            stop_price = st.number_input("Preço Stop (Loss) (R$)", value=entry_price * 0.9 if entry_price > 0 else 0.0, step=0.01, format="%.2f")
        else: # SHORT
            target_price = st.number_input("Preço Alvo (R$)", value=entry_price * 0.8 if entry_price > 0 else 0.0, step=0.01, format="%.2f")
            stop_price = st.number_input("Preço Stop (Loss) (R$)", value=entry_price * 1.1 if entry_price > 0 else 0.0, step=0.01, format="%.2f")

# --- Calculations ---
upside_pct = 0.0
downside_pct = 0.0
if entry_price > 0:
    if trade_type == "SHORT (Venda)":
        upside_pct = 1 - (target_price / entry_price)
        downside_pct = 1 - (stop_price / entry_price)
    else: # LONG or L&S Ratio
        upside_pct = (target_price / entry_price) - 1
        downside_pct = (stop_price / entry_price) - 1

st.markdown("---")
m1, m2, m3 = st.columns(3)
m1.metric("Upside Estimado", pct(upside_pct), delta=f"{upside_pct*100:.2f}%", delta_color="normal")
m2.metric("Downside Estimado", pct(downside_pct), delta=f"{downside_pct*100:.2f}%", delta_color="inverse")
risk_reward = abs(upside_pct / downside_pct) if downside_pct != 0 else 0
m3.metric("Relação Risco/Retorno", f"{risk_reward:.2f}")

casa_analise = st.text_input("Casa de Análise (ex: XP, BTG, Itaú BBA, etc.)")
notes = st.text_area("Notas / Tese de Investimento", height=150)

if st.button("Gerar Trade Idea e Salvar"):
    if not ticker or entry_price <= 0:
        st.error("Por favor, preencha o ticker e o preço de entrada.")
    else:
        # Save to DB
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        entry_date_str = entry_date.strftime("%Y-%m-%d")
        
        t_type = "LONG"
        if trade_type == "SHORT (Venda)": t_type = "SHORT"
        elif trade_type == "LONG & SHORT (Pair)": t_type = "LS"
        
        t_long = ticker
        t_short = ""
        e_p_short = 0.0
        
        if t_type == "LS":
            t_long = ticker_long
            t_short = ticker_short
            e_p_short = p_short_val
        
        cursor.execute("""
            INSERT INTO trade_ideas 
            (ticker, entry_price, target_price, stop_price, current_price, upside_pct, downside_pct, notes, start_date, status, trade_type, ticker_short, entry_price_short, casa_analise)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (ticker, entry_price, target_price, stop_price, current_price, upside_pct, downside_pct, notes, entry_date_str, "ABERTA", t_type, t_short, e_p_short, casa_analise))
        conn.commit()
        conn.close()
        st.success("Trade Idea salva no banco de dados com sucesso!")

        # Generate HTML for preview
        with st.spinner("Gerando gráfico de performance..."):
            t_type_val = "LONG"
            if trade_type == "SHORT (Venda)": t_type_val = "SHORT"
            elif trade_type == "LONG & SHORT (Pair)": t_type_val = "LS"
            
            chart_b64 = generate_comparison_chart_b64(ticker, entry_date_str, None, t_type_val, entry_price, target_price, stop_price)

        template = Template(REPORT_HTML_TEMPLATE)
        
        # Labels and formatting for report
        is_ls = (t_type_val == "LS")
        prefix = "" if is_ls else "R$"
        l_entry = "Ratio de Entrada" if is_ls else "Preço de Entrada"
        l_target = "Ratio Alvo" if is_ls else "Preço Alvo"
        l_stop = "Ratio Stop" if is_ls else "Preço Stop"
        l_exit = "Ratio de Saída" if is_ls else "Preço de Saída"
        
        t_label = "Trade Idea - Compra"
        if t_type_val == "SHORT": t_label = "Trade Idea - Venda"
        elif t_type_val == "LS": t_label = "Trade Idea - Long & Short"

        html_content = template.render(
            ticker=ticker,
            entry_price=brl(entry_price) if not is_ls else f"{entry_price:.4f}",
            target_price=brl(target_price) if not is_ls else f"{target_price:.4f}",
            stop_price=brl(stop_price) if not is_ls else f"{stop_price:.4f}",
            exit_price="0,00",
            upside_pct=pct(upside_pct),
            downside_pct=pct(downside_pct),
            result_pct="0,00%",
            result_raw=0,
            notes=notes,
            date=entry_date.strftime("%d/%m/%Y"),
            end_date="",
            color_primary=BRAND_BROWN,
            status="ABERTA",
            logo=logo_b64,
            duration_days=0,
            comparison_chart=chart_b64,
            trade_type_label=t_label,
            prefix=prefix,
            label_entry=l_entry,
            label_target=l_target,
            label_stop=l_stop,
            label_exit=l_exit,
            comparison_table=st.session_state.get("peer_comparison_html", ""),
            comparison_title=st.session_state.get("peer_comparison_title", "Comparação Setorial"),
            casa_analise=casa_analise
        )
        
        # Save to session state for persistence
        st.session_state["generated_html"] = html_content
        st.session_state["generated_ticker"] = ticker

# --- Display persistent results if they exist ---
if "generated_html" in st.session_state:
    html_content = st.session_state["generated_html"]
    ticker = st.session_state["generated_ticker"]
    
    st.subheader("Preview do Relatório")
    st.components.v1.html(html_content, height=600, scrolling=True)

    # Download Links
    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        st.download_button(
            "Baixar HTML", 
            html_content, 
            file_name=f"Trade_Idea_{ticker}.html", 
            mime="text/html",
            key="dl_html"
        )
    
    with col_dl2:
        if st.button("Gerar PDF", key="gen_pdf_btn"):
            with st.spinner("Gerando PDF..."):
                pdf_bytes = html_to_pdf(html_content)
                if pdf_bytes:
                    st.session_state["pdf_bytes"] = pdf_bytes
                    st.success("PDF gerado com sucesso!")
        
        if "pdf_bytes" in st.session_state:
            st.download_button(
                "Baixar PDF", 
                st.session_state["pdf_bytes"], 
                file_name=f"Trade_Idea_{ticker}.pdf", 
                mime="application/pdf",
                key="dl_pdf"
            )

# --- Trade Ideas Management ---
st.markdown("---")
st.subheader("📂 Gerenciamento de Trade Ideas")

conn = sqlite3.connect(DB_PATH)
df_all = pd.read_sql_query("SELECT * FROM trade_ideas ORDER BY id DESC", conn)
conn.close()

if not df_all.empty:
    with st.expander("Gerenciar Registros (Editar / Encerrar / PDF)"):
        selected_id = st.selectbox("Selecione uma Trade Idea para gerenciar", 
                                   options=df_all["id"].tolist(),
                                   format_func=lambda x: f"ID {x} - {df_all[df_all['id']==x]['ticker'].values[0]} ({df_all[df_all['id']==x]['status'].values[0]})")
        
        row = df_all[df_all["id"] == selected_id].iloc[0]
        r_type = row.get("trade_type", "LONG")
        is_ls = (r_type == "LS")
        
        col_ed1, col_ed2 = st.columns(2)
        with col_ed1:
            m_ticker = st.text_input("Ticker", value=row["ticker"], key=f"ed_ticker_{selected_id}")
            m_type = st.selectbox("Tipo", options=["LONG", "SHORT", "LS"], 
                                  index=["LONG", "SHORT", "LS"].index(r_type) if r_type in ["LONG", "SHORT", "LS"] else 0,
                                  key=f"ed_type_{selected_id}")
            m_status = st.selectbox("Status", options=["ABERTA", "ENCERRADA"], 
                                    index=0 if row["status"] == "ABERTA" else 1, 
                                    key=f"ed_status_{selected_id}")
            label_e = "Ratio de Entrada" if m_type == "LS" else "Entrada (R$)"
            m_entry = st.number_input(label_e, value=float(row["entry_price"] or 0), format="%.4f" if m_type == "LS" else "%.2f", key=f"ed_entry_{selected_id}")
            m_start_date = st.text_input("Data Início", value=row["start_date"] or "", key=f"ed_start_{selected_id}")

        with col_ed2:
            label_t = "Ratio Alvo" if m_type == "LS" else "Alvo (R$)"
            label_s = "Ratio Stop" if m_type == "LS" else "Stop (R$)"
            label_ex = "Ratio de Saída" if m_type == "LS" else "Saída (R$)"
            
            m_target = st.number_input(label_t, value=float(row["target_price"] or 0), format="%.4f" if m_type == "LS" else "%.2f", key=f"ed_target_{selected_id}")
            m_stop = st.number_input(label_s, value=float(row["stop_price"] or 0), format="%.4f" if m_type == "LS" else "%.2f", key=f"ed_stop_{selected_id}")
            
            # Fields for finished trades
            m_exit = st.number_input(label_ex, value=float(row["exit_price"] or 0), format="%.4f" if m_type == "LS" else "%.2f", key=f"ed_exit_{selected_id}")
            m_end_date = st.text_input("Data Fim", value=row["end_date"] or "", key=f"ed_end_{selected_id}")

        m_casa_analise = st.text_input("Casa de Análise", value=row.get("casa_analise", "") or "", key=f"ed_casa_{selected_id}")
        m_notes = st.text_area("Notas", value=row["notes"] or "", key=f"ed_notes_{selected_id}")

        # --- MANAGMENT SECTOR DATA & PEERS ---
        with st.expander("📊 Comparação com Pares (Setorial) - PDF", expanded=False):
            eco_data_mgr = get_economatica_data(m_ticker)
            if eco_data_mgr is not None:
                sector_name_mgr = eco_data_mgr.get("Subsetor Bovespa")
                st.info(f"Setor: {sector_name_mgr}")
                
                default_cols_mgr = ["P/L 12m", "P/VP", "ROE 12m", "Dividend Yield 12m"]
                sel_cols_mgr = st.multiselect("Métricas:", 
                                            options=list(ECONOMATICA_COLS.keys()),
                                            default=default_cols_mgr,
                                            key=f"mgr_cols_{selected_id}")
                
                if sector_name_mgr:
                    # Always include Market Cap for sorting
                    mcap_col_db_mgr = ECONOMATICA_COLS["Valor de Mercado"]
                    
                    db_cols_selected_mgr = [ECONOMATICA_COLS[c] for c in sel_cols_mgr]
                    
                    # Ensure mcap is in fetch
                    db_cols_fetch_mgr = list(set(db_cols_selected_mgr + [mcap_col_db_mgr]))
                    
                    raw_peers_mgr = get_peers_data(sector_name_mgr, db_cols_fetch_mgr)
                    
                    if not raw_peers_mgr.empty:
                        # Clean and Sort by Market Cap
                        raw_peers_mgr[mcap_col_db_mgr] = pd.to_numeric(raw_peers_mgr[mcap_col_db_mgr], errors='coerce').fillna(0)
                        raw_peers_mgr = raw_peers_mgr.sort_values(by=mcap_col_db_mgr, ascending=False)
                        
                        # Identify Top 5 + Current
                        all_tickers_mgr = raw_peers_mgr["Código"].unique().tolist()
                        
                        # Top 5
                        top_5_mgr = raw_peers_mgr.head(5)["Código"].tolist()
                        
                        # Ensure current ticker is in default if available
                        current_clean_mgr = m_ticker.upper().replace(".SA", "").strip()
                        
                        default_selection_mgr = set(top_5_mgr)
                        if current_clean_mgr in all_tickers_mgr:
                            default_selection_mgr.add(current_clean_mgr)
                        
                        # Multiselect for peers
                        selected_peers_list_mgr = st.multiselect("Selecionar Pares (Default: Top 5 Market Cap):", 
                                                             options=all_tickers_mgr,
                                                             default=list(default_selection_mgr),
                                                             key=f"mgr_peers_sel_{selected_id}")
                                                             
                        # Filter DataFrame based on selection
                        raw_peers_filtered_mgr = raw_peers_mgr[raw_peers_mgr["Código"].isin(selected_peers_list_mgr)].copy()
                        
                        # Rename columns
                        rev_map_mgr = {v: k for k, v in ECONOMATICA_COLS.items()}
                        raw_mgr_renamed = raw_peers_filtered_mgr.rename(columns=rev_map_mgr)
                        
                        # Filter cols
                        final_cols_mgr = ["Código"] + sel_cols_mgr
                        final_peers_mgr = raw_mgr_renamed[final_cols_mgr].copy()
                        
                        # Round numeric columns for display
                        numeric_cols_mgr = final_peers_mgr.select_dtypes(include=['float64', 'float32', 'int64', 'int32']).columns
                        final_peers_mgr[numeric_cols_mgr] = final_peers_mgr[numeric_cols_mgr].round(2)
                        
                        st.dataframe(final_peers_mgr, hide_index=True, use_container_width=True)
                        
                        # Store for PDF generation
                        include_comp_mgr = st.checkbox("Incluir no PDF", value=True, key=f"mgr_inc_{selected_id}")
                        if include_comp_mgr:
                            peer_html_mgr = format_df_for_html(final_peers_mgr).to_html(
                                classes="table peer-table", index=False, border=0
                            )
                            st.session_state[f"peer_html_{selected_id}"] = peer_html_mgr
                            st.session_state[f"peer_title_{selected_id}"] = f"Comparação Setorial: {sector_name_mgr}"
                        else:
                            st.session_state.pop(f"peer_html_{selected_id}", None)
            else:
                st.warning("Dados não encontrados no Economatica para este ticker.")

        # Buttons for selected record
        btn_col1, btn_col2, btn_col3 = st.columns(3)

        if btn_col1.button("💾 Salvar Alterações", key=f"save_btn_{selected_id}"):
            if m_type == "SHORT":
                r_pct = 1 - (m_exit / m_entry) if m_entry > 0 and m_exit > 0 else 0
            else: # LONG or LS
                r_pct = (m_exit / m_entry) - 1 if m_entry > 0 and m_exit > 0 else 0
            
            final_status = m_status
            final_end_date = m_end_date
            
            # Logic: If exit price is provided, auto-close the trade
            if m_exit > 0:
                final_status = "ENCERRADA"
                # If no end date provided, default to today
                if not final_end_date:
                    final_end_date = datetime.now().strftime("%Y-%m-%d")
                elif "/" in final_end_date: # Conversion helper
                    try:
                        final_end_date = datetime.strptime(final_end_date, "%d/%m/%Y").strftime("%Y-%m-%d")
                    except: pass
            
            conn = sqlite3.connect(DB_PATH)
            conn.execute("""
                UPDATE trade_ideas SET 
                ticker=?, status=?, entry_price=?, target_price=?, stop_price=?, 
                exit_price=?, start_date=?, end_date=?, notes=?, result_pct=?, trade_type=?, casa_analise=?
                WHERE id=?
            """, (m_ticker, final_status, m_entry, m_target, m_stop, m_exit, m_start_date, final_end_date, m_notes, r_pct, m_type, m_casa_analise, selected_id))
            conn.commit()
            conn.close()
            st.success(f"Registro atualizado! Status: {final_status}")
            st.rerun()

        if btn_col2.button("📄 Gerar PDF do Registro", key=f"pdf_btn_{selected_id}"):
            # Re-calculate or use stored values
            if m_type == "SHORT":
                r_pct_val = 1 - (m_exit / m_entry) if m_entry > 0 and m_exit > 0 else 0
                u_pct_val = 1 - (m_target / m_entry) if m_entry > 0 else 0
                d_pct_val = 1 - (m_stop / m_entry) if m_entry > 0 else 0
            else: # LONG or LS
                r_pct_val = (m_exit / m_entry) - 1 if m_entry > 0 and m_exit > 0 else 0
                u_pct_val = (m_target / m_entry) - 1 if m_entry > 0 else 0
                d_pct_val = (m_stop / m_entry) - 1 if m_entry > 0 else 0
            
            # Logic: If exit price is in UI, treat as ENCERRADA for the report
            preview_status = m_status
            preview_end_date = m_end_date
            if m_exit > 0:
                preview_status = "ENCERRADA"
                if not preview_end_date:
                    preview_end_date = datetime.now().strftime("%Y-%m-%d")
            
            # Duration calculation helper
            def calc_dur(s, e):
                if not s or not e: return 0
                try:
                    fmt = "%Y-%m-%d" if "-" in s else "%d/%m/%Y"
                    d1 = datetime.strptime(s.split(' ')[0], fmt)
                    fmt2 = "%Y-%m-%d" if "-" in e else "%d/%m/%Y"
                    d2 = datetime.strptime(e.split(' ')[0], fmt2)
                    return (d2 - d1).days
                except: return 0

            template = Template(REPORT_HTML_TEMPLATE)
            with st.spinner("Gerando gráfico de performance..."):
                chart_b64 = generate_comparison_chart_b64(m_ticker, m_start_date, preview_end_date, m_type, m_entry, m_target, m_stop)

            # Labels and formatting for report
            is_ls_report = (m_type == "LS")
            prefix = "" if is_ls_report else "R$"
            l_entry = "Ratio de Entrada" if is_ls_report else "Preço de Entrada"
            l_target = "Ratio Alvo" if is_ls_report else "Preço Alvo"
            l_stop = "Ratio Stop" if is_ls_report else "Preço Stop"
            l_exit = "Ratio de Saída" if is_ls_report else "Preço de Saída"
            
            t_label = "Trade Idea - Compra"
            if m_type == "SHORT": t_label = "Trade Idea - Venda"
            elif m_type == "LS": t_label = "Trade Idea - Long & Short"

            h_content = template.render(
                ticker=m_ticker,
                entry_price=brl(m_entry) if not is_ls_report else f"{m_entry:.4f}",
                target_price=brl(m_target) if not is_ls_report else f"{m_target:.4f}",
                stop_price=brl(m_stop) if not is_ls_report else f"{m_stop:.4f}",
                exit_price=brl(m_exit) if not is_ls_report else f"{m_exit:.4f}",
                upside_pct=pct(u_pct_val),
                downside_pct=pct(d_pct_val),
                result_pct=pct(r_pct_val),
                result_raw=r_pct_val,
                notes=m_notes,
                date=m_start_date[:10] if m_start_date else "",
                end_date=preview_end_date[:10] if preview_end_date else "",
                color_primary=BRAND_BROWN,
                status=preview_status,
                logo=logo_b64,
                duration_days=calc_dur(m_start_date, preview_end_date),
                comparison_chart=chart_b64,
                trade_type_label=t_label,
                prefix=prefix,
                label_entry=l_entry,
                label_target=l_target,
                label_stop=l_stop,
                label_exit=l_exit,
                comparison_table=st.session_state.get(f"peer_html_{selected_id}", ""),
                comparison_title=st.session_state.get(f"peer_title_{selected_id}", "Comparação Setorial"),
                casa_analise=m_casa_analise
            )
            
            with st.spinner("Gerando PDF..."):
                p_bytes = html_to_pdf(h_content)
                if p_bytes:
                    st.session_state[f"pdf_history_{selected_id}"] = p_bytes
                    st.success("PDF pronto para download!")
        
        if f"pdf_history_{selected_id}" in st.session_state:
            st.download_button("⬇️ Baixar PDF Gerado", 
                               st.session_state[f"pdf_history_{selected_id}"], 
                               file_name=f"Trade_Idea_{m_ticker}_{selected_id}.pdf",
                               mime="application/pdf")

        if btn_col3.button("⚠️ Excluir Registro", key=f"del_btn_{selected_id}"):
            conn = sqlite3.connect(DB_PATH)
            conn.execute("DELETE FROM trade_ideas WHERE id = ?", (selected_id,))
            conn.commit()
            conn.close()
            st.warning("Registro excluído.")
            st.rerun()

# --- Performance Analysis ---
st.markdown("---")
st.subheader("📊 Análise de Performance Histórica")

if not df_all.empty:
    with st.expander("Consolidado de Performance"):
        # Filters
        perf_col1, perf_col2, perf_col3 = st.columns(3)
        
        # We need to process dates for filtering
        df_perf = df_all.copy()
        
        # Helper to parse messy dates
        def safe_parse_date(d):
            if not d: return None
            for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%Y-%m-%d %H:%M:%S"):
                try:
                    return datetime.strptime(d.split(' ')[0], fmt)
                except: continue
            return None

        df_perf["dt_start"] = df_perf["start_date"].apply(safe_parse_date)
        df_perf["dt_end"] = df_perf["end_date"].apply(safe_parse_date)
        
        # Filter Options
        years = sorted(df_perf["dt_start"].dropna().dt.year.unique(), reverse=True)
        if not years: years = [datetime.now().year]
        
        sel_year = perf_col1.selectbox("Ano", options=years)
        report_type = perf_col2.selectbox("Tipo de Relatório", options=["Semestral", "Anual", "Mensal"])
        
        if report_type == "Semestral":
            sel_period = perf_col3.selectbox("Semestre", options=["1º Semestre (Jan-Jun)", "2º Semestre (Jul-Dez)"])
            period_label = f"{sel_period} de {sel_year}"
            if "1º" in sel_period:
                mask = (df_perf["dt_start"].dt.year == sel_year) & (df_perf["dt_start"].dt.month <= 6)
            else:
                mask = (df_perf["dt_start"].dt.year == sel_year) & (df_perf["dt_start"].dt.month > 6)
        elif report_type == "Anual":
            period_label = f"Ano Completo {sel_year}"
            mask = (df_perf["dt_start"].dt.year == sel_year)
        else: # Mensal
            sel_month = perf_col3.selectbox("Mês", options=list(range(1, 13)), format_func=lambda x: datetime(2000, x, 1).strftime("%B"))
            period_label = f"{datetime(2000, sel_month, 1).strftime('%B')} de {sel_year}"
            mask = (df_perf["dt_start"].dt.year == sel_year) & (df_perf["dt_start"].dt.month == sel_month)
            
        df_filtered = df_perf[mask].copy()
        
        if not df_filtered.empty:
            # Calculations
            total_trades = len(df_filtered)
            df_closed = df_filtered[df_filtered["status"] == "ENCERRADA"].copy()
            closed_count = len(df_closed)
            
            if closed_count > 0:
                pos_trades = df_closed[df_closed["result_pct"] > 0]
                win_rate = len(pos_trades) / closed_count
                avg_pl = df_closed["result_pct"].mean()
                
                # Duration
                df_closed["duration"] = (df_closed["dt_end"] - df_closed["dt_start"]).dt.days
                avg_duration = df_closed["duration"].mean()
                
                # Math Expectancy: (Win Rate * Avg Win) - (Loss Rate * Avg Loss)
                avg_win = pos_trades["result_pct"].mean() if not pos_trades.empty else 0
                neg_trades = df_closed[df_closed["result_pct"] <= 0]
                avg_loss = abs(neg_trades["result_pct"].mean()) if not neg_trades.empty else 0
                exp_math = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
            else:
                win_rate = 0
                avg_pl = 0
                avg_duration = 0
                exp_math = 0
            
            # Display Metrics
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            m_col1.metric("Total de Ideias", total_trades)
            m_col2.metric("Win Rate", pct(win_rate))
            m_col3.metric("Média Lucro/Perda", pct(avg_pl))
            m_col4.metric("Exp. Matemática", f"{exp_math:.4f}")
            
            # PDF Generation for Performance
            if st.button("📄 Gerar Relatório de Performance (PDF)"):
                # Prepare data for template
                trades_list = []
                for _, r in df_closed.iterrows():
                    trades_list.append({
                        "ticker": r["ticker"],
                        "start": r["dt_start"].strftime("%d/%m/%Y") if r["dt_start"] else "-",
                        "end": r["dt_end"].strftime("%d/%m/%Y") if r["dt_end"] else "-",
                        "result": pct(r["result_pct"]),
                        "raw_result": r["result_pct"]
                    })
                
                # Prepare consolidated results chart
                with st.spinner("Gerando gráficos comparativos..."):
                    perf_chart_b64 = generate_consolidated_performance_chart_b64(df_perf)

                template = Template(PERFORMANCE_REPORT_TEMPLATE)
                perf_html = template.render(
                    report_period=period_label,
                    total_trades=total_trades,
                    win_rate=pct(win_rate),
                    avg_pl=pct(avg_pl),
                    avg_pl_raw=avg_pl,
                    closed_trades=closed_count,
                    avg_duration=round(avg_duration, 1),
                    exp_math=round(exp_math, 4),
                    trades_list=trades_list,
                    generated_at=datetime.now().strftime("%d/%m/%Y %H:%M"),
                    color_primary=BRAND_BROWN,
                    logo=logo_b64,
                    performance_chart=perf_chart_b64
                )
                
                with st.spinner("Gerando Relatório Consolidados..."):
                    pdf_perf = html_to_pdf(perf_html)
                    if pdf_perf:
                        st.session_state["pdf_perf"] = pdf_perf
                        st.success("Relatório de performance pronto!")
            
            if "pdf_perf" in st.session_state:
                st.download_button("⬇️ Baixar Relatório de Performance", 
                                   st.session_state["pdf_perf"], 
                                   file_name=f"Performance_Trade_Ideas_{sel_year}.pdf",
                                   mime="application/pdf")
            
            st.dataframe(df_filtered[["status", "ticker", "start_date", "end_date", "result_pct"]], use_container_width=True, hide_index=True)

        else:
            st.warning("Nenhum dado encontrado para o período selecionado.")

else:
    st.info("Nenhuma trade idea salva ainda.")
