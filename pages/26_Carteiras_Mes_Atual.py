import streamlit as st
import yfinance as yf
import pandas as pd
import sqlite3
from datetime import datetime
from pathlib import Path
import base64
import html as _html
import plotly.graph_objects as go
import io
from pypdf import PdfWriter, PdfReader
from playwright.sync_api import sync_playwright

# Database setup
DB_PATH = Path("databases/carteiras_rv_mes_atual.db")
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# Theme constants
BRAND_BROWN = "#825120"
BRAND_BROWN_DARK = "#6B4219"
LIGHT_GRAY = "#F5F5F5"
BLOCK_BG = "#F8F8F8"
TABLE_BORDER = "#E0D5CA"
TEXT_DARK = "#333333"


def _html_escape(x: object) -> str:
    return _html.escape("" if pd.isna(x) else str(x))


def image_to_base64(uploaded_file) -> str:
    """Convert uploaded image to base64 string for embedding in HTML."""
    if uploaded_file is None:
        return ""
    bytes_data = uploaded_file.getvalue()
    b64 = base64.b64encode(bytes_data).decode()
    # Determine mime type
    file_type = uploaded_file.type if uploaded_file.type else "image/png"
    return f"data:{file_type};base64,{b64}"


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS portfolios (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS portfolio_stocks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            portfolio_id INTEGER NOT NULL,
            ticker TEXT NOT NULL,
            weight REAL NOT NULL DEFAULT 0,
            FOREIGN KEY (portfolio_id) REFERENCES portfolios(id) ON DELETE CASCADE,
            UNIQUE(portfolio_id, ticker)
        )
    """)
    conn.commit()
    conn.close()


def get_portfolios():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM portfolios ORDER BY name", conn)
    conn.close()
    return df


def get_portfolio_stocks(portfolio_id):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT ticker, weight FROM portfolio_stocks WHERE portfolio_id = ?",
        conn,
        params=(portfolio_id,),
    )
    conn.close()
    return df


def get_portfolio_total_weight(portfolio_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT COALESCE(SUM(weight), 0) FROM portfolio_stocks WHERE portfolio_id = ?",
        (portfolio_id,),
    )
    total = cursor.fetchone()[0]
    conn.close()
    return total


def add_portfolio(name):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO portfolios (name) VALUES (?)", (name,))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()


def delete_portfolio(portfolio_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM portfolio_stocks WHERE portfolio_id = ?", (portfolio_id,))
    cursor.execute("DELETE FROM portfolios WHERE id = ?", (portfolio_id,))
    conn.commit()
    conn.close()


def add_stock_to_portfolio(portfolio_id, ticker, weight):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO portfolio_stocks (portfolio_id, ticker, weight) VALUES (?, ?, ?)",
            (portfolio_id, ticker.upper(), weight),
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()


def add_stocks_batch(portfolio_id, stocks_list):
    """Add multiple stocks at once. Returns (success_count, errors_list)."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    success = 0
    errors = []

    for ticker, weight in stocks_list:
        try:
            cursor.execute(
                "INSERT INTO portfolio_stocks (portfolio_id, ticker, weight) VALUES (?, ?, ?)",
                (portfolio_id, ticker.upper(), weight),
            )
            success += 1
        except sqlite3.IntegrityError:
            errors.append(f"{ticker} (já existe)")

    conn.commit()
    conn.close()
    return success, errors


def update_stock_weight(portfolio_id, ticker, weight):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE portfolio_stocks SET weight = ? WHERE portfolio_id = ? AND ticker = ?",
        (weight, portfolio_id, ticker),
    )
    conn.commit()
    conn.close()


def remove_stock_from_portfolio(portfolio_id, ticker):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "DELETE FROM portfolio_stocks WHERE portfolio_id = ? AND ticker = ?",
        (portfolio_id, ticker),
    )
    conn.commit()
    conn.close()


def clear_portfolio_stocks(portfolio_id):
    """Remove all stocks from a portfolio."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM portfolio_stocks WHERE portfolio_id = ?", (portfolio_id,))
    conn.commit()
    conn.close()


def parse_weight(weight_str: str) -> float | None:
    """
    Parse weight from various formats:
    - XX% or XX,XX% or XX.XX% → percentage directly (15% → 15)
    - 0,xx or 0.xx or 0,xxx or 0.xxx → decimal to percentage (0,15 → 15)
    - XX or XX,XX or XX.XX (no %) → percentage directly if >= 1, else decimal
    """
    weight_str = weight_str.strip()
    has_percent = "%" in weight_str
    weight_str = weight_str.replace("%", "").strip()

    # Normalize decimal separator (comma to dot)
    weight_str = weight_str.replace(",", ".")

    try:
        value = float(weight_str)
    except ValueError:
        return None

    # If has % symbol, treat as percentage directly
    if has_percent:
        return value

    # No % symbol: if value < 1, assume it's decimal (0.15 = 15%)
    if value < 1:
        return value * 100

    # Otherwise treat as percentage
    return value


def parse_batch_input(text: str) -> tuple[list, list]:
    """
    Parse batch input in format STOCK;WEIGHT (one per line).
    Accepts multiple weight formats.
    Returns (parsed_list, errors_list).
    """
    parsed = []
    errors = []

    # Normalize separators: replace multiple spaces with newlines
    text = text.strip()
    lines = [line.strip() for line in text.split("\n") if line.strip()]

    for line in lines:
        # Try semicolon separator
        if ";" in line:
            parts = line.split(";")
        else:
            errors.append(f"'{line}' - formato inválido (use TICKER;PESO)")
            continue

        if len(parts) != 2:
            errors.append(f"'{line}' - formato inválido")
            continue

        ticker = parts[0].strip().upper()
        weight_str = parts[1].strip()

        if not ticker:
            errors.append(f"'{line}' - ticker vazio")
            continue

        weight = parse_weight(weight_str)

        if weight is None:
            errors.append(f"'{ticker}' - peso inválido: {parts[1]}")
            continue

        if weight <= 0 or weight > 100:
            errors.append(f"'{ticker}' - peso deve ser entre 0 e 100 (recebido: {weight:.2f}%)")
            continue

        parsed.append((ticker, weight))

    return parsed, errors


def get_stock_return_mtd(ticker: str) -> dict:
    """Get stock return from first trading day of current month until now."""
    # Determine the correct Yahoo Finance ticker
    if ticker.startswith("^") or ticker.endswith(".SA"):
        yf_ticker = ticker
    else:
        yf_ticker = f"{ticker}.SA"
        
    today = datetime.now()
    first_day_of_month = today.replace(day=1)

    try:
        stock = yf.Ticker(yf_ticker)
        hist = stock.history(start=first_day_of_month, end=today, interval="1d")

        if hist.empty:
            return {"ticker": ticker, "error": "No data available"}

        # Try to get stock name
        try:
            name = stock.info.get("longName") or stock.info.get("shortName") or ticker
        except Exception:
            name = ticker

        first_close = hist["Close"].iloc[0]
        last_close = hist["Close"].iloc[-1]
        return_pct = ((last_close - first_close) / first_close) * 100

        return {
            "ticker": ticker,
            "name": name,
            "yf_ticker": yf_ticker,
            "first_date": hist.index[0].strftime("%Y-%m-%d"),
            "last_date": hist.index[-1].strftime("%Y-%m-%d"),
            "first_close": first_close,
            "last_close": last_close,
            "return_pct": return_pct,
        }
    except Exception as e:
        return {"ticker": ticker, "error": str(e)}


def calculate_portfolio_returns(stocks_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate returns for all stocks in a portfolio with weights."""
    results = []
    for _, row in stocks_df.iterrows():
        result = get_stock_return_mtd(row["ticker"])
        result["weight"] = row["weight"]
        results.append(result)
    return pd.DataFrame(results)


def get_ibov_return_mtd() -> dict:
    """Get IBOV return from first trading day of current month until now."""
    return get_stock_return_mtd("^BVSP")


def get_ifix_return_mtd() -> dict:
    """Get IFIX return from first trading day of current month until now."""
    return get_stock_return_mtd("IFIX.SA")


def generate_multi_portfolio_mtd_html(
    summaries: list[dict],
    ibov_data: dict,
    ifix_data: dict,
    title: str = "Relatório Diário de Carteiras RV",
    subtitle: str = "",
    logo_base64: str = "",
) -> str:
    """Generate a combined HTML report for multiple portfolios."""
    if not summaries:
        return "<p>Nenhuma carteira selecionada para o relatório.</p>"

    # --- CSS Styles ---
    css = f"""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Open+Sans:wght@400;600;700&display=swap');
      @page {{
        size: auto;
        margin: 0;
      }}
      :root {{
        --brand: {BRAND_BROWN};
        --brand-dark: {BRAND_BROWN_DARK};
        --bg: {BLOCK_BG};
        --text: {TEXT_DARK};
        --tbl-border: {TABLE_BORDER};
        --light: {LIGHT_GRAY};
      }}
      * {{ box-sizing: border-box; }}
      html, body {{
        margin: 0;
        padding: 0;
        -webkit-print-color-adjust: exact;
        print-color-adjust: exact;
      }}
      body {{
        font-family: 'Open Sans', 'Segoe UI', Arial, sans-serif;
        background: #f0f0f0;
        color: var(--text);
        padding: 20px;
      }}
      .page {{
        max-width: 1000px;
        margin: 0 auto 30px auto;
        background: white;
        box-shadow: 0 2px 15px rgba(0,0,0,0.1);
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid var(--tbl-border);
        page-break-after: always;
      }}
      .main-header {{
        background: var(--brand);
        color: #fff;
        padding: 15px 25px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 15px;
      }}
      .header-logo {{
        height: 50px;
        max-width: 150px;
        object-fit: contain;
      }}
      .header-text {{
        flex: 1;
        text-align: right;
      }}
      .main-title {{ font-size: 24px; font-weight: 700; letter-spacing: 0.5px; }}
      .main-subtitle {{ font-size: 14px; opacity: 0.9; }}
      
      .section-header {{
        background: var(--bg);
        padding: 8px 25px;
        border-bottom: 1px solid var(--tbl-border);
        border-top: 1px solid var(--tbl-border);
        font-weight: 700;
        color: var(--brand-dark);
        font-size: 16px;
        margin-top: 0;
      }}
      
      .content-block {{
        padding: 20px 25px;
      }}

      .stats-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 15px;
        margin-bottom: 20px;
      }}
      .stat-card {{
        background: var(--light);
        padding: 15px;
        border-radius: 6px;
        text-align: center;
        border: 1px solid var(--tbl-border);
      }}
      .stat-label {{ font-size: 11px; text-transform: uppercase; color: #666; font-weight: 600; }}
      .stat-value {{ font-size: 20px; font-weight: 700; margin-top: 5px; }}
      
      .table-wrap {{ 
        width: 100%; 
        overflow-x: auto; 
        margin-bottom: 20px;
      }}
      table {{
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        font-size: 12px;
      }}
      thead th {{
        background: var(--light); 
        color: var(--text);
        text-align: right; 
        padding: 10px; 
        font-weight: 700;
        border-bottom: 2px solid var(--tbl-border);
      }}
      thead th:first-child {{ text-align: left; }}
      
      tbody td {{
        padding: 8px 10px; 
        border-bottom: 1px solid var(--tbl-border);
        text-align: right; 
      }}
      tbody td:first-child {{ text-align: left; font-weight: 600; }}
      tbody tr:nth-child(even) td {{ background: var(--light); }}
      
      .positive {{ color: #15803d; }}
      .negative {{ color: #b91c1c; }}
      
      .footer {{
        padding: 10px 25px; 
        color: #666; 
        font-size: 12px;
        border-top: 1px solid var(--tbl-border);
        background: #fff;
        text-align: center;
      }}

      @media print {{
        body {{ padding: 0; background: white; }}
        .page {{ box-shadow: none; border: none; margin: 0; border-radius: 0; }}
      }}
    </style>
    """

    logo_html = f'<img src="{logo_base64}" class="header-logo" alt="Logo" />' if logo_base64 else ""

    pages_html = []
    for summary in summaries:
        # Determine benchmark based on portfolio name
        is_fii = "FII" in summary["name"].upper()
        bench_data = ifix_data if is_fii else ibov_data
        bench_name = "IFIX" if is_fii else "IBOV"

        bench_ret = bench_data.get("return_pct")
        bench_txt = f"{bench_ret:+.2f}%" if bench_ret is not None else "N/A"
        bench_class = "positive" if (bench_ret or 0) >= 0 else "negative"

        port_ret = summary.get("weighted_return")
        port_txt = f"{port_ret:+.2f}%" if port_ret is not None else "N/A"
        port_class = "positive" if (port_ret or 0) >= 0 else "negative"
        
        diff = port_ret - bench_ret if port_ret is not None and bench_ret is not None else None
        diff_txt = f"{diff:+.2f}%" if diff is not None else "N/A"
        diff_class = "positive" if (diff or 0) >= 0 else "negative"

        # Sort stocks by contribution
        stocks_data = summary.get("stocks_data")
        if stocks_data is not None:
            stocks_df = stocks_data.sort_values("contribution", ascending=False)
            best = stocks_df.iloc[0]
            worst = stocks_df.iloc[-1]
            
            rows = []
            for _, s in stocks_df.iterrows():
                rows.append(f"""
                <tr>
                    <td>{_html_escape(s['ticker'])}</td>
                    <td style="text-align: left; font-size: 11px; color: #666;">{_html_escape(s['name'])}</td>
                    <td>{s['weight']:.2f}%</td>
                    <td class="{'positive' if s['return_pct'] >= 0 else 'negative'}">{s['return_pct']:+.2f}%</td>
                    <td class="{'positive' if s['contribution'] >= 0 else 'negative'}">{s['contribution']:+.3f}%</td>
                </tr>
                """)
            table_rows_html = "".join(rows)
        else:
            table_rows_html = "<tr><td colspan='5'>Nenhum dado de ação disponível.</td></tr>"
            best = worst = {"ticker": "N/A", "return_pct": 0, "contribution": 0}

        page = f"""
        <div class="page">
          <div class="main-header">
            {logo_html}
            <div class="header-text">
              <div class="main-title">{_html_escape(summary['name'])}</div>
              <div class="main-subtitle">{_html_escape(title)} | {subtitle}</div>
            </div>
          </div>
          
          <div class="section-header">Resumo Mensal (MTD)</div>
          <div class="content-block">
            <div class="stats-grid">
              <div class="stat-card">
                <div class="stat-label">Retorno Carteira</div>
                <div class="stat-value {port_class}">{port_txt}</div>
              </div>
              <div class="stat-card">
                <div class="stat-label">Retorno {bench_name}</div>
                <div class="stat-value {bench_class}">{bench_txt}</div>
              </div>
              <div class="stat-card">
                <div class="stat-label">Alpha (vs {bench_name})</div>
                <div class="stat-value {diff_class}">{diff_txt}</div>
              </div>
            </div>
            
            <div class="stats-grid">
              <div class="stat-card">
                <div class="stat-label">Qtd. Ações</div>
                <div class="stat-value">{summary['num_stocks']}</div>
              </div>
              <div class="stat-card">
                <div class="stat-label">Melhor Contribuidor</div>
                <div class="stat-value positive" style="font-size: 16px;">{_html_escape(best['ticker'])} ({best['contribution']:+.2f}%)</div>
              </div>
              <div class="stat-card">
                <div class="stat-label">Pior Contribuidor</div>
                <div class="stat-value negative" style="font-size: 16px;">{_html_escape(worst['ticker'])} ({worst['contribution']:+.2f}%)</div>
              </div>
            </div>
          </div>

          <div class="section-header">Análise de Ativos</div>
          <div class="content-block">
            <div class="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Ativo</th>
                    <th style="text-align: left;">Empresa</th>
                    <th>Peso (%)</th>
                    <th>Retorno (%)</th>
                    <th>Contribuição (%)</th>
                  </tr>
                </thead>
                <tbody>
                  {table_rows_html}
                </tbody>
              </table>
            </div>
          </div>
          
          <div class="footer">
            Uso Interno - Não Compartilhar | Dados via Yahoo Finance | Gerado em {datetime.now().strftime('%d/%m/%Y %H:%M')}
          </div>
        </div>
        """
        pages_html.append(page)

    full_html = f"""
    <!DOCTYPE html>
    <html lang="pt-br">
    <head>
      <meta charset="UTF-8" />
      <title>{_html_escape(title)}</title>
      {css}
    </head>
    <body>
      {''.join(pages_html)}
    </body>
    </html>
    """
    return full_html


def html_to_pdf_multi_page(
    summaries: list[dict],
    ibov_data: dict,
    ifix_data: dict,
    title: str = "Relatório Diário de Carteiras RV",
    subtitle: str = "",
    logo_base64: str = "",
) -> bytes:
    """
    Render each portfolio to a separate PDF page with dynamic height and merge them.
    """
    MAX_PX = 32000
    writer = PdfWriter()

    with sync_playwright() as p:
        browser = p.chromium.launch()
        context = browser.new_context()
        page = context.new_page()

        for summary in summaries:
            # Generate HTML for a single portfolio
            single_html = generate_multi_portfolio_mtd_html(
                [summary],
                ibov_data,
                ifix_data,
                title=title,
                subtitle=subtitle,
                logo_base64=logo_base64,
            )

            # In generate_multi_portfolio_mtd_html, the .page class has margin: 0 auto 30px auto;
            # For PDF generation of single pages, we might want to ensure it looks tight.
            # We can inject a small CSS override if needed, but let's see how it behaves.

            page.set_viewport_size({"width": 1000, "height": 800})
            page.set_content(single_html, wait_until="load")
            page.wait_for_load_state("networkidle")

            # Wait for fonts
            try:
                page.wait_for_function(
                    'document.fonts && document.fonts.status === "loaded"',
                    timeout=5000,
                )
            except Exception:
                pass

            # Measure full content height
            dims = page.evaluate(
                """
                () => {
                  const el = document.querySelector('.page');
                  if (el) {
                    return {
                      width: Math.ceil(el.scrollWidth + 10),
                      height: Math.ceil(el.scrollHeight + 10)
                    };
                  }
                  return {
                    width: Math.ceil(document.body.scrollWidth + 10),
                    height: Math.ceil(document.body.scrollHeight + 10)
                  };
                }
                """
            )

            width_px = min(max(800, dims.get("width", 1000)), MAX_PX)
            height_px = min(max(400, dims.get("height", 800)), MAX_PX)

            # Generate PDF for this specific page
            page_pdf_bytes = page.pdf(
                print_background=True,
                prefer_css_page_size=False,
                width=f"{width_px}px",
                height=f"{height_px}px",
                margin={"top": "0", "right": "0", "bottom": "0", "left": "0"},
            )

            # Add to writer
            reader = PdfReader(io.BytesIO(page_pdf_bytes))
            writer.add_page(reader.pages[0])

        context.close()
        browser.close()

    output_pdf = io.BytesIO()
    writer.write(output_pdf)
    return output_pdf.getvalue()


def get_portfolio_summary(portfolio_id: int, portfolio_name: str) -> dict:
    """Get summary info for a portfolio including weighted return."""
    stocks_df = get_portfolio_stocks(portfolio_id)
    total_weight = get_portfolio_total_weight(portfolio_id)

    summary = {
        "id": portfolio_id,
        "name": portfolio_name,
        "num_stocks": len(stocks_df),
        "total_weight": total_weight,
        "weighted_return": None,
        "stocks_data": None,
        "errors": [],
    }

    if stocks_df.empty:
        return summary

    returns_df = calculate_portfolio_returns(stocks_df)
    success_df = returns_df[~returns_df["return_pct"].isna()].copy()
    error_df = returns_df[returns_df["return_pct"].isna()]

    if not success_df.empty:
        success_df["contribution"] = success_df["return_pct"] * success_df["weight"] / 100
        summary["weighted_return"] = success_df["contribution"].sum()
        summary["stocks_data"] = success_df

    if not error_df.empty:
        summary["errors"] = error_df["ticker"].tolist()

    return summary


# Initialize database
init_db()

# Streamlit UI
st.set_page_config(page_title="Carteiras RV - Retorno Mensal", layout="wide")
st.title("📊 Carteiras de Renda Variável - Retorno MTD")
st.caption("Retorno ponderado desde o primeiro dia de negociação do mês atual")

# Initialize PDF in session state
if "multi_portfolio_pdf" not in st.session_state:
    st.session_state["multi_portfolio_pdf"] = None

# Main tabs
tab_dashboard, tab_manage = st.tabs(["📈 Dashboard", "⚙️ Gerenciar Carteiras"])

# Sidebar for report configuration
st.sidebar.header("📁 Configuração do Relatório")
logo_upload = st.sidebar.file_uploader("Logo da Empresa (opcional)", type=["png", "jpg", "jpeg"])
logo_b64 = image_to_base64(logo_upload) if logo_upload else ""

report_title = st.sidebar.text_input("Título do Relatório", "Relatório de Perfomance - Mensal")
report_subtitle = st.sidebar.text_input("Subtítulo", datetime.now().strftime("%B / %Y"))

# ============== TAB 1: DASHBOARD ==============
with tab_dashboard:
    portfolios = get_portfolios()

    if portfolios.empty:
        st.info("Nenhuma carteira criada. Vá para a aba 'Gerenciar Carteiras' para criar uma.")
    else:
        col_btn1, col_btn2 = st.columns([1, 1])
        with col_btn1:
            if st.button("🔄 Atualizar Retornos de Todas as Carteiras", type="primary", use_container_width=True):
                st.session_state["dashboard_updated"] = True
        
        with col_btn2:
            # Multi-portfolio selection for report
            selected_for_report = st.multiselect(
                "Selecionar para Relatório",
                options=portfolios["name"].tolist(),
                default=portfolios["name"].tolist(),
                help="Selecione as carteiras que deseja incluir no relatório HTML"
            )

        if st.session_state.get("dashboard_updated", False):
            with st.spinner("Buscando dados (Ações + IBOV + IFIX)..."):
                summaries = []
                ibov_data = get_ibov_return_mtd()
                ifix_data = get_ifix_return_mtd()
                
                for _, portfolio in portfolios.iterrows():
                    summary = get_portfolio_summary(portfolio["id"], portfolio["name"])
                    summaries.append(summary)

                st.session_state["summaries"] = summaries
                st.session_state["ibov_data"] = ibov_data
                st.session_state["ifix_data"] = ifix_data
                st.session_state["dashboard_updated"] = False # Reset
                st.session_state["last_update"] = datetime.now().strftime("%d/%m/%Y %H:%M")

        if "summaries" in st.session_state:
            summaries = st.session_state["summaries"]
            ibov_data = st.session_state["ibov_data"]
            ifix_data = st.session_state.get("ifix_data", {})

            # Add Generate HTML button
            st.divider()
            col_rep1, col_rep2 = st.columns([3, 1])
            with col_rep1:
                st.subheader("📋 Relatório Consolidado")
            with col_rep2:
                # Filter summaries based on selection
                report_summaries = [s for s in summaries if s["name"] in selected_for_report]
                
                if st.button("📄 Gerar Relatório HTML", type="secondary", use_container_width=True):
                    html_content = generate_multi_portfolio_mtd_html(
                        report_summaries, 
                        ibov_data, 
                        ifix_data,
                        title=report_title, 
                        subtitle=report_subtitle,
                        logo_base64=logo_b64
                    )
                    
                    b64_html = base64.b64encode(html_content.encode()).decode()
                    href = f'<a href="data:text/html;base64,{b64_html}" download="relatorio_carteiras_{datetime.now().strftime("%Y%m%d")}.html" style="text-decoration:none; color:inherit;">📥 Baixar Relatório HTML</a>'
                    st.markdown(
                        f"""
                        <div style="background-color: #825120; color: white; padding: 10px; border-radius: 5px; text-align: center; cursor: pointer;">
                            {href}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            # PDF Export Section
            st.divider()
            col_pdf1, col_pdf2 = st.columns([1, 1])
            with col_pdf1:
                if st.button("🚀 Gerar Relatório PDF", type="primary", use_container_width=True):
                    try:
                        with st.spinner("Gerando PDF (pode levar alguns segundos)..."):
                            report_summaries = [s for s in summaries if s["name"] in selected_for_report]
                            pdf_bytes = html_to_pdf_multi_page(
                                report_summaries,
                                ibov_data,
                                ifix_data,
                                title=report_title,
                                subtitle=report_subtitle,
                                logo_base64=logo_b64
                            )
                            st.session_state["multi_portfolio_pdf"] = pdf_bytes
                            st.success("PDF gerado com sucesso!")
                    except Exception as e:
                        st.error(f"Erro ao gerar PDF: {e}")
                        st.info("Verifique se o Playwright está instalado corretamente.")

            with col_pdf2:
                if st.session_state["multi_portfolio_pdf"]:
                    st.download_button(
                        label="📥 Baixar Relatório PDF",
                        data=st.session_state["multi_portfolio_pdf"],
                        file_name=f"relatorio_carteiras_{datetime.now().strftime('%Y%m%d')}.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )

            # Display cards in grid
            num_cols = 3
            for i in range(0, len(summaries), num_cols):
                cols = st.columns(num_cols)
                for j, col in enumerate(cols):
                    if i + j < len(summaries):
                        summary = summaries[i + j]
                        with col:
                            with st.container(border=True):
                                st.subheader(f"📁 {summary['name']}")

                                # Weight status
                                weight_ok = abs(summary["total_weight"] - 100) < 0.01
                                weight_icon = "✅" if weight_ok else "⚠️"

                                col_info1, col_info2 = st.columns(2)
                                with col_info1:
                                    st.metric("Ações", summary["num_stocks"])
                                with col_info2:
                                    st.metric(
                                        "Peso Total",
                                        f"{summary['total_weight']:.1f}% {weight_icon}",
                                    )

                                st.divider()

                                # Return
                                if summary["weighted_return"] is not None:
                                    ret = summary["weighted_return"]
                                    color = "green" if ret >= 0 else "red"
                                    st.markdown(
                                        f"### :{color}[{ret:+.2f}%]",
                                        help="Retorno ponderado MTD",
                                    )

                                    # Top performers
                                    if summary["stocks_data"] is not None:
                                        df = summary["stocks_data"].copy()
                                        df = df.sort_values(
                                            "return_pct", ascending=False
                                        )

                                        with st.expander("📊 Detalhes"):
                                            # Best and worst
                                            if len(df) > 0:
                                                best = df.iloc[0]
                                                worst = df.iloc[-1]

                                                st.caption("**Melhor:**")
                                                st.write(
                                                    f"🟢 {best['ticker']}: "
                                                    f"{best['return_pct']:+.2f}%"
                                                )

                                                st.caption("**Pior:**")
                                                st.write(
                                                    f"🔴 {worst['ticker']}: "
                                                    f"{worst['return_pct']:+.2f}%"
                                                )

                                            # Full table
                                            st.caption("**Todas as ações:**")
                                            display_df = df[
                                                ["ticker", "weight", "return_pct", "contribution"]
                                            ].copy()
                                            display_df.columns = [
                                                "Ticker",
                                                "Peso %",
                                                "Retorno %",
                                                "Contrib %",
                                            ]
                                            st.dataframe(
                                                display_df.style.format(
                                                    {
                                                        "Peso %": "{:.1f}",
                                                        "Retorno %": "{:+.2f}",
                                                        "Contrib %": "{:+.3f}",
                                                    }
                                                ),
                                                use_container_width=True,
                                                hide_index=True,
                                            )
                                elif summary["num_stocks"] == 0:
                                    st.info("Carteira vazia")
                                else:
                                    st.warning("Sem dados disponíveis")

                                # Errors
                                if summary["errors"]:
                                    st.caption(
                                        f"⚠️ Erros: {', '.join(summary['errors'])}"
                                    )

            st.session_state["last_update"] = datetime.now().strftime("%d/%m/%Y %H:%M")

        if "last_update" in st.session_state:
            st.caption(f"Última atualização: {st.session_state['last_update']}")
        else:
            st.info("Clique no botão acima para carregar os retornos das carteiras.")


# ============== TAB 2: MANAGE ==============
with tab_manage:
    # Sidebar-like section for portfolio creation/deletion
    with st.expander("➕ Criar / 🗑️ Excluir Carteira", expanded=False):
        col_create, col_delete = st.columns(2)

        with col_create:
            st.subheader("Nova Carteira")
            new_portfolio_name = st.text_input("Nome da carteira", key="new_portfolio_name")
            if st.button("Criar Carteira", type="primary", key="create_btn"):
                if new_portfolio_name:
                    if add_portfolio(new_portfolio_name):
                        st.success(f"Carteira '{new_portfolio_name}' criada!")
                        st.rerun()
                    else:
                        st.error("Carteira já existe!")
                else:
                    st.warning("Digite um nome para a carteira")

        with col_delete:
            st.subheader("Excluir Carteira")
            portfolios = get_portfolios()
            if not portfolios.empty:
                portfolio_to_delete = st.selectbox(
                    "Selecionar carteira",
                    portfolios["id"].tolist(),
                    format_func=lambda x: portfolios[portfolios["id"] == x]["name"].values[0],
                    key="delete_select",
                )
                if st.button("🗑️ Excluir", type="secondary", key="delete_btn"):
                    delete_portfolio(portfolio_to_delete)
                    st.success("Carteira excluída!")
                    st.rerun()
            else:
                st.info("Nenhuma carteira para excluir")

    st.divider()

    # Main content
    portfolios = get_portfolios()

    if portfolios.empty:
        st.info("Nenhuma carteira criada. Use o painel acima para criar uma nova carteira.")
    else:
        # Portfolio selector
        selected_portfolio_id = st.selectbox(
            "Selecione a Carteira",
            portfolios["id"].tolist(),
            format_func=lambda x: portfolios[portfolios["id"] == x]["name"].values[0],
            key="manage_portfolio_select",
        )

        selected_portfolio_name = portfolios[portfolios["id"] == selected_portfolio_id][
            "name"
        ].values[0]

        # Show total weight
        total_weight = get_portfolio_total_weight(selected_portfolio_id)
        weight_color = "green" if abs(total_weight - 100) < 0.01 else "red"
        st.markdown(
            f"**Peso total alocado:** :{weight_color}[{total_weight:.2f}%]"
            f" {'✅' if abs(total_weight - 100) < 0.01 else '⚠️ (deve somar 100%)'}"
        )

        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader(f"Ações em '{selected_portfolio_name}'")

            # Tabs for single add vs batch add
            tab_single, tab_batch = st.tabs(["➕ Adicionar Uma", "📋 Adicionar em Lote"])

            with tab_single:
                with st.form("add_stock_form"):
                    new_stock = st.text_input(
                        "Ticker (sem .SA)",
                        placeholder="Ex: PETR4, VALE3, ITUB4",
                    )
                    new_weight = st.number_input(
                        "Peso (%)",
                        min_value=0.0,
                        max_value=100.0,
                        value=0.0,
                        step=0.5,
                        format="%.2f",
                    )
                    submitted = st.form_submit_button("Adicionar")
                    if submitted and new_stock:
                        if new_weight <= 0:
                            st.error("Peso deve ser maior que 0!")
                        elif add_stock_to_portfolio(
                            selected_portfolio_id, new_stock.strip(), new_weight
                        ):
                            st.success(f"{new_stock.upper()} ({new_weight}%) adicionada!")
                            st.rerun()
                        else:
                            st.error("Ação já existe na carteira!")

            with tab_batch:
                st.markdown("**Formato:** `TICKER;PESO` (uma por linha)")
                st.caption(
                    "Formatos de peso aceitos:\n"
                    "• 15% ou 15,50% ou 15.50%\n"
                    "• 0,15 ou 0.15 (decimal → 15%)\n"
                    "• 15 ou 15,50 ou 15.50"
                )

                batch_input = st.text_area(
                    "Cole as ações aqui",
                    height=150,
                    placeholder="ITUB4;15%\nRDOR3;0,10\nVALE3;20.5%",
                )

                col_add, col_clear = st.columns(2)
                with col_add:
                    if st.button(
                        "📥 Adicionar Todas", type="primary", use_container_width=True
                    ):
                        if batch_input.strip():
                            parsed, parse_errors = parse_batch_input(batch_input)

                            if parse_errors:
                                st.error("Erros de formato:")
                                for err in parse_errors:
                                    st.write(f"• {err}")

                            if parsed:
                                success, db_errors = add_stocks_batch(
                                    selected_portfolio_id, parsed
                                )
                                if success > 0:
                                    st.success(f"{success} ação(ões) adicionada(s)!")
                                if db_errors:
                                    st.warning("Não adicionadas:")
                                    for err in db_errors:
                                        st.write(f"• {err}")
                                st.rerun()
                        else:
                            st.warning("Cole as ações no campo acima")

                with col_clear:
                    if st.button(
                        "🗑️ Limpar Carteira", type="secondary", use_container_width=True
                    ):
                        clear_portfolio_stocks(selected_portfolio_id)
                        st.success("Todas as ações removidas!")
                        st.rerun()

            # List current stocks with weights
            st.divider()
            current_stocks_df = get_portfolio_stocks(selected_portfolio_id)
            if not current_stocks_df.empty:
                st.write("**Ações na carteira:**")
                for _, row in current_stocks_df.iterrows():
                    col_stock, col_weight, col_btn = st.columns([2, 1.5, 0.5])
                    with col_stock:
                        st.write(f"**{row['ticker']}**")
                    with col_weight:
                        new_w = st.number_input(
                            "Peso",
                            min_value=0.0,
                            max_value=100.0,
                            value=float(row["weight"]),
                            step=0.5,
                            format="%.2f",
                            key=f"weight_{row['ticker']}",
                            label_visibility="collapsed",
                        )
                        if new_w != row["weight"]:
                            update_stock_weight(
                                selected_portfolio_id, row["ticker"], new_w
                            )
                            st.rerun()
                    with col_btn:
                        if st.button("❌", key=f"remove_{row['ticker']}"):
                            remove_stock_from_portfolio(
                                selected_portfolio_id, row["ticker"]
                            )
                            st.rerun()
            else:
                st.info("Nenhuma ação na carteira")

        with col2:
            st.subheader("📈 Retornos MTD")

            if not current_stocks_df.empty:
                if st.button("🔄 Calcular Retornos", type="primary", key="calc_returns"):
                    with st.spinner("Buscando dados do Yahoo Finance..."):
                        returns_df = calculate_portfolio_returns(current_stocks_df)

                        # Separate successful and failed
                        success_df = returns_df[~returns_df["return_pct"].isna()].copy()
                        error_df = returns_df[returns_df["return_pct"].isna()]

                        if not success_df.empty:
                            # Calculate weighted return contribution
                            success_df["contribution"] = (
                                success_df["return_pct"] * success_df["weight"] / 100
                            )

                            # Format for display
                            display_df = success_df[
                                [
                                    "ticker",
                                    "weight",
                                    "first_date",
                                    "last_date",
                                    "first_close",
                                    "last_close",
                                    "return_pct",
                                    "contribution",
                                ]
                            ].copy()
                            display_df.columns = [
                                "Ticker",
                                "Peso (%)",
                                "Primeira Data",
                                "Última Data",
                                "Preço Inicial",
                                "Preço Atual",
                                "Retorno (%)",
                                "Contribuição (%)",
                            ]

                            # Style the dataframe
                            st.dataframe(
                                display_df.style.format(
                                    {
                                        "Peso (%)": "{:.2f}%",
                                        "Preço Inicial": "R$ {:.2f}",
                                        "Preço Atual": "R$ {:.2f}",
                                        "Retorno (%)": "{:+.2f}%",
                                        "Contribuição (%)": "{:+.4f}%",
                                    }
                                ).map(
                                    lambda x: (
                                        "color: green"
                                        if isinstance(x, (int, float)) and x > 0
                                        else (
                                            "color: red"
                                            if isinstance(x, (int, float)) and x < 0
                                            else ""
                                        )
                                    ),
                                    subset=["Retorno (%)", "Contribuição (%)"],
                                ),
                                use_container_width=True,
                            )

                            # Portfolio weighted return
                            weighted_return = success_df["contribution"].sum()
                            simple_avg = success_df["return_pct"].mean()

                            col_m1, col_m2 = st.columns(2)
                            with col_m1:
                                st.metric(
                                    "🎯 Retorno Ponderado da Carteira",
                                    f"{weighted_return:+.4f}%",
                                )
                            with col_m2:
                                st.metric(
                                    "📊 Retorno Médio Simples",
                                    f"{simple_avg:+.2f}%",
                                )

                            # Warning if weights don't sum to 100
                            total_w = success_df["weight"].sum()
                            if abs(total_w - 100) > 0.01:
                                st.warning(
                                    f"⚠️ Os pesos das ações com dados somam {total_w:.2f}%. "
                                    "O retorno ponderado considera apenas as ações com dados disponíveis."
                                )

                        if not error_df.empty:
                            st.warning("Erros ao buscar dados:")
                            for _, row in error_df.iterrows():
                                st.write(
                                    f"• {row['ticker']} ({row['weight']:.2f}%): "
                                    f"{row.get('error', 'Erro desconhecido')}"
                                )
            else:
                st.info("Adicione ações à carteira para calcular os retornos")

# Footer
st.divider()
st.caption(
    f"Dados via Yahoo Finance (yfinance). Última atualização: {datetime.now().strftime('%d/%m/%Y %H:%M')}"
)