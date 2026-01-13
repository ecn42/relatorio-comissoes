import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import sqlite3
import re
from pathlib import Path
import base64


st.set_page_config(page_title="Portfolio Analyzer", layout="wide")
st.title("📊 Portfolio Return Analyzer")

# Database configuration
DB_DIR = Path("databases")
DB_PATH = DB_DIR / "rentabilidade_carteiras_btg.db"
BENCHMARK_DB_PATH = DB_DIR / "data_cdi_ibov.db"


def init_db():
    """Ensure database directory exists."""
    DB_DIR.mkdir(parents=True, exist_ok=True)


def db_exists():
    """Check if database file exists."""
    return DB_PATH.exists()


def save_to_db(df: pd.DataFrame):
    """Save DataFrame to SQLite database."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    try:
        # Store with index (Date) as a column
        df_to_save = df.reset_index()
        df_to_save["Data"] = df_to_save["Data"].astype(str)
        df_to_save.to_sql("returns", conn, if_exists="replace", index=False)
        conn.commit()
    finally:
        conn.close()


def load_from_db() -> pd.DataFrame:
    """Load DataFrame from SQLite database."""
    if not db_exists():
        return pd.DataFrame()

    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql("SELECT * FROM returns", conn)
        df["Data"] = pd.to_datetime(df["Data"])
        df = df.set_index("Data").sort_index()
        return df
    except Exception:
        return pd.DataFrame()
    finally:
        conn.close()


def load_benchmark_data(benchmark_name: str) -> pd.Series:
    """Load benchmark data (CDI or IBOV) and convert to monthly returns."""
    if not BENCHMARK_DB_PATH.exists():
        return pd.Series(dtype=float)

    table_map = {"CDI": "cdi", "IBOV": "ibov"}
    if benchmark_name not in table_map:
        return pd.Series(dtype=float)

    conn = sqlite3.connect(BENCHMARK_DB_PATH)
    try:
        query = f"SELECT date, daily_return FROM {table_map[benchmark_name]}"
        df = pd.read_sql(query, conn)
        df["date"] = pd.to_datetime(df["date"])
        
        # Calculate monthly returns: product(1 + r) - 1
        # Group by Year-Month and align to 1st of the month
        df["month_date"] = df["date"].apply(lambda x: x.replace(day=1))
        
        monthly_returns = df.groupby("month_date")["daily_return"].apply(
            lambda x: (1 + x).prod() - 1
        )
        return monthly_returns
        
    except Exception as e:
        st.error(f"Error loading benchmark: {e}")
        return pd.Series(dtype=float)
    finally:
        conn.close()





def image_to_base64(uploaded_file) -> str:
    """Convert uploaded image to base64 string for embedding in HTML."""
    if uploaded_file is None:
        return ""
    bytes_data = uploaded_file.getvalue()
    b64 = base64.b64encode(bytes_data).decode()
    # Determine mime type
    file_type = uploaded_file.type if uploaded_file.type else "image/png"
    return f"data:{file_type};base64,{b64}"


def parse_excel(uploaded_file) -> pd.DataFrame:
    """Parse Excel file and return cleaned DataFrame."""
    df = pd.read_excel(uploaded_file)

    # Rename first column to 'Data' if needed
    df.rename(columns={df.columns[0]: "Data"}, inplace=True)

    # Parse Brazilian date format (e.g., 'out-09' -> datetime)
    month_map = {
        "jan": "01",
        "fev": "02",
        "mar": "03",
        "abr": "04",
        "mai": "05",
        "jun": "06",
        "jul": "07",
        "ago": "08",
        "set": "09",
        "out": "10",
        "nov": "11",
        "dez": "12",
    }

    def parse_date(val):
        if pd.isna(val):
            return pd.NaT
        if isinstance(val, (pd.Timestamp, pd.DatetimeIndex)):
            return val
        if not isinstance(val, str):
            return pd.NaT
        val = val.strip().lower()
        parts = val.split("-")
        if len(parts) != 2:
            return pd.NaT
        month_str, year_str = parts
        if month_str not in month_map:
            return pd.NaT
        try:
            year = int(year_str)
            year = 2000 + year if year < 50 else 1900 + year
            return pd.Timestamp(year=year, month=int(month_map[month_str]), day=1)
        except ValueError:
            return pd.NaT

    df["Data"] = df["Data"].apply(parse_date)

    # Remove rows with invalid dates
    df = df[df["Data"].notna()]
    df = df.set_index("Data").sort_index()

    # Convert percentage strings to floats (handles "1,5%" format)
    # Convert values to numeric and handle percentage scaling
    for col in df.columns:
        # If column is object type, clean strings first
        if df[col].dtype == "object":
            df[col] = (
                df[col]
                .astype(str)
                .str.replace("%", "", regex=False)
                .str.replace(",", ".", regex=False)
                .str.strip()
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Heuristic: Check magnitude to decide if we need to divide by 100
        # Monthly returns are typically < 1.0 (100%).
        # If mean|val| > 1.0, it's likely in 0-100 scale (e.g., 5.2 for 5.2%).
        # If mean|val| <= 1.0, it's likely in 0-1 scale (e.g., 0.052 for 5.2%).
        # Exception: Extreme volatility, but >100% avg monthly return is rare.
        valid_vals = df[col].dropna()
        if len(valid_vals) > 0:
            if valid_vals.abs().mean() > 1.0:
                df[col] = df[col] / 100

    return df


def compare_dataframes(existing_df: pd.DataFrame, new_df: pd.DataFrame) -> dict:
    """Compare existing and new DataFrames, return differences."""
    result = {
        "new_months": [],
        "new_strategies": [],
        "updated_cells": 0,
        "has_changes": False,
    }

    if existing_df.empty:
        result["new_months"] = list(new_df.index)
        result["new_strategies"] = list(new_df.columns)
        result["has_changes"] = True
        return result

    # Find new months
    existing_dates = set(existing_df.index)
    new_dates = set(new_df.index)
    result["new_months"] = sorted(list(new_dates - existing_dates))

    # Find new strategies
    existing_cols = set(existing_df.columns)
    new_cols = set(new_df.columns)
    result["new_strategies"] = list(new_cols - existing_cols)

    # Check for updated values in overlapping data
    common_dates = existing_dates & new_dates
    common_cols = existing_cols & new_cols

    for date in common_dates:
        for col in common_cols:
            old_val = existing_df.loc[date, col]
            new_val = new_df.loc[date, col]

            # Handle NaN comparisons
            old_is_nan = pd.isna(old_val)
            new_is_nan = pd.isna(new_val)

            if old_is_nan and new_is_nan:
                continue
            elif old_is_nan != new_is_nan:
                result["updated_cells"] += 1
            elif abs(old_val - new_val) > 1e-9:
                result["updated_cells"] += 1

    result["has_changes"] = (
        len(result["new_months"]) > 0
        or len(result["new_strategies"]) > 0
        or result["updated_cells"] > 0
    )

    return result


def merge_dataframes(existing_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    """Merge new data into existing DataFrame, preferring new values."""
    if existing_df.empty:
        return new_df

    # Combine all columns
    all_cols = list(set(existing_df.columns) | set(new_df.columns))

    # Combine all dates
    all_dates = sorted(set(existing_df.index) | set(new_df.index))

    # Create merged DataFrame
    merged = pd.DataFrame(index=all_dates, columns=all_cols)

    # Fill with existing data first
    for col in existing_df.columns:
        for date in existing_df.index:
            merged.loc[date, col] = existing_df.loc[date, col]

    # Overwrite/add with new data
    for col in new_df.columns:
        for date in new_df.index:
            merged.loc[date, col] = new_df.loc[date, col]

    merged.index.name = "Data"
    return merged


def calculate_cumulative_returns(df):
    """Calculate cumulative returns from monthly returns."""
    return (1 + df).cumprod() - 1


def calculate_portfolio_return(df, weights):
    """Calculate weighted portfolio return."""
    portfolio = pd.Series(0.0, index=df.index)
    for col, weight in weights.items():
        if col in df.columns:
            portfolio += df[col].fillna(0) * weight
    return portfolio


def get_first_valid_date(series):
    """Get the first date with valid data for a series."""
    valid = series.dropna()
    return valid.index.min() if len(valid) > 0 else None


def get_smart_start_date(df, strategies, min_date):
    """Calculate the smart start date based on selected strategies."""
    if len(strategies) >= 2:
        start_dates = []
        for strat in strategies:
            first_date = get_first_valid_date(df[strat])
            if first_date:
                start_dates.append(first_date)
        return max(start_dates) if start_dates else min_date
    elif len(strategies) == 1:
        first_date = get_first_valid_date(df[strategies[0]])
        return first_date if first_date else min_date
    return min_date


def format_pct(val):
    """Format value as percentage string."""
    if pd.isna(val):
        return "-"
    return f"{val * 100:.2f}%"



def sanitize_filename(name: str) -> str:
    """Remove or replace characters that are invalid in filenames."""
    sanitized = re.sub(r'[<>:"/\\|?*]', "_", str(name))
    sanitized = re.sub(r"[\s_]+", "_", sanitized)
    sanitized = sanitized.strip("_ ")
    return sanitized


def html_to_pdf_playwright(html_content: str) -> bytes:
    """
    Render HTML to a single-page PDF using headless Chromium via Playwright.
    """
    from playwright.sync_api import sync_playwright

    MAX_PX = 32000  # Increased limit

    with sync_playwright() as p:
        browser = p.chromium.launch()
        context = browser.new_context()
        page = context.new_page()

        # Set a wide viewport initially
        page.set_viewport_size({"width": 1200, "height": 800})
        page.set_content(html_content, wait_until="load")
        page.wait_for_load_state("networkidle")

        # Wait for fonts
        try:
            page.wait_for_function(
                'document.fonts && document.fonts.status === "loaded"',
                timeout=10000,
            )
        except Exception:
            pass

        # Measure full content height
        dims = page.evaluate(
            """
            () => {
              const page = document.querySelector('.page');
              if (page) {
                return {
                  width: Math.ceil(page.scrollWidth + 40),
                  height: Math.ceil(page.scrollHeight + 40)
                };
              }
              return {
                width: Math.ceil(document.body.scrollWidth + 40),
                height: Math.ceil(document.body.scrollHeight + 40)
              };
            }
            """
        )

        width_px = min(max(800, dims.get("width", 1200)), MAX_PX)
        height_px = min(max(600, dims.get("height", 800)), MAX_PX)

        # Generate PDF with exact dimensions
        pdf_bytes = page.pdf(
            print_background=True,
            prefer_css_page_size=False,
            width=f"{width_px}px",
            height=f"{height_px}px",
            margin={"top": "0", "right": "0", "bottom": "0", "left": "0"},
        )

        context.close()
        browser.close()

    return pdf_bytes

# =====================================================
# HTML REPORT GENERATION (Adapted from Comparador)
# =====================================================

import html as _html

# Theme constants
BRAND_BROWN = "#825120"
BRAND_BROWN_DARK = "#6B4219"
LIGHT_GRAY = "#F5F5F5"
BLOCK_BG = "#F8F8F8"
TABLE_BORDER = "#E0D5CA"
TEXT_DARK = "#333333"


def _html_escape(x: object) -> str:
    return _html.escape("" if pd.isna(x) else str(x))


def generate_portfolio_html(
    stats_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    fig: go.Figure,
    fig_drawdown: go.Figure = None,
    contribution_df: pd.DataFrame = None,
    fig_pie: go.Figure = None,
    title: str = "Portfolio Analysis Report",
    subtitle: str = "",
    logo_base64: str = "",  # NEW PARAMETER
) -> str:
    if stats_df.empty:
        return "<p>No data to render.</p>"

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
        padding: 10px;
      }}
      .page {{
        max-width: 1200px;
        margin: 0 auto;
        background: white;
        box-shadow: 0 2px 15px rgba(0,0,0,0.1);
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid var(--tbl-border);
        page-break-inside: avoid;
      }}
      .main-header {{
        background: var(--brand);
        color: #fff;
        padding: 10px 20px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 15px;
      }}
      .header-logo {{
        height: 50px;
        max-width: 120px;
        object-fit: contain;
      }}
      .header-text {{
        flex: 1;
        text-align: right;
      }}
      .main-title {{ font-size: 22px; font-weight: 700; letter-spacing: 0.5px; }}
      .main-subtitle {{ font-size: 13px; opacity: 0.9; }}
      
      .section-header {{
        background: var(--bg);
        padding: 6px 20px;
        border-bottom: 1px solid var(--tbl-border);
        border-top: 1px solid var(--tbl-border);
        font-weight: 700;
        color: var(--brand-dark);
        font-size: 14px;
        margin-top: 0;
        page-break-after: avoid;
      }}
      
      .chart-container {{
        padding: 5px;
        border-bottom: 1px solid var(--tbl-border);
        page-break-inside: avoid;
      }}

      .table-wrap {{ 
        width: 100%; 
        overflow-x: auto; 
        padding: 0 20px; 
        margin-bottom: 10px;
        page-break-inside: avoid;
      }}
      table {{
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        font-size: 11px;
        margin-top: 5px;
      }}
      thead th {{
        position: sticky; top: 0; z-index: 2;
        background: var(--light); 
        color: var(--text);
        text-align: center; 
        padding: 4px 4px; 
        font-weight: 700;
        border-bottom: 2px solid var(--tbl-border);
        white-space: nowrap;
      }}
      thead th:first-child {{ text-align: left; }}
      
      tbody td {{
        padding: 4px 4px; 
        border-bottom: 1px solid var(--tbl-border);
        text-align: right; 
        white-space: nowrap;
      }}
      tbody td:first-child {{ text-align: left; font-weight: 600; white-space: nowrap; }}
      tbody tr:nth-child(even) td {{ background: var(--light); }}
      
      .footer {{
        padding: 6px 20px; 
        color: #666; 
        font-size: 11px;
        border-top: 1px solid var(--tbl-border);
        background: #fff;
        text-align: center;
      }}
      
      tr.year-header td {{
        background-color: #e0e0e0;
        font-weight: 800;
        text-align: left;
        padding-left: 10px;
        font-size: 12px;
        color: var(--brand-dark);
      }}
      
      tr.main-row td {{
        font-weight: 700;
        background-color: #fff;
        color: var(--text);
        border-bottom: 2px solid #eee;
      }}
      
      tr.sub-row td {{
        font-size: 10px;
        color: #777;
        background-color: #fcfcfc;
        border-bottom: 1px solid #f0f0f0;
      }}
      tr.sub-row td:first-child {{
        padding-left: 20px;
        font-weight: 400;
      }}

      /* Force single page in print */
      @media print {{
        body {{ padding: 0; background: white; }}
        .page {{ 
          box-shadow: none; 
          border-radius: 0;
          page-break-inside: avoid;
          break-inside: avoid;
        }}
      }}
    </style>
    """

    # --- Build Logo HTML ---
    logo_html = ""
    if logo_base64:
        logo_html = f'<img src="{logo_base64}" class="header-logo" alt="Logo" />'

    # ... (rest of the stats_html, chart_html, etc. building code stays the same)

    # --- Build Stats Table ---
    stats_cols = list(stats_df.columns)
    stats_ths = "".join(f"<th>{_html_escape(c)}</th>" for c in stats_cols)
    
    stats_rows = []
    for _, row in stats_df.iterrows():
        tds = []
        for i, c in enumerate(stats_cols):
            val = row[c]
            align = 'left' if i == 0 else 'right'
            tds.append(f'<td style="text-align:{align}">{_html_escape(val)}</td>')
        stats_rows.append(f"<tr>{''.join(tds)}</tr>")

    stats_html = f"""
        <div class="section-header">Resumo Estatístico</div>
        <div class="table-wrap">
            <table>
                <thead><tr>{stats_ths}</tr></thead>
                <tbody>{''.join(stats_rows)}</tbody>
            </table>
        </div>
    """

    # --- Build Chart HTML ---
    chart_html = ""
    if fig:
        chart_content = fig.to_html(full_html=False, include_plotlyjs='cdn')
        chart_html = f"""
            <div class="section-header">Gráfico de Retorno Acumulado</div>
            <div class="chart-container">
                {chart_content}
            </div>
        """

    # --- Build Drawdown Chart HTML ---
    drawdown_html = ""
    if fig_drawdown:
        dd_content = fig_drawdown.to_html(full_html=False, include_plotlyjs='cdn')
        drawdown_html = f"""
            <div class="section-header">Gráfico de Drawdown</div>
            <div class="chart-container">
                {dd_content}
            </div>
        """

    # --- Build Contribution HTML ---
    contrib_html = ""
    if contribution_df is not None and not contribution_df.empty:
        # Build Table
        c_cols = list(contribution_df.columns)
        c_ths = "".join(f"<th>{_html_escape(c)}</th>" for c in c_cols)
        c_rows = []
        for _, row in contribution_df.iterrows():
             tds = []
             for i, c in enumerate(c_cols):
                 val = row[c]
                 # First col align left, others right
                 align = 'left' if i == 0 else 'right'
                 tds.append(f'<td style="text-align:{align}">{_html_escape(val)}</td>')
             c_rows.append(f"<tr>{''.join(tds)}</tr>")
        
        contrib_table_html = f"""
        <div class="table-wrap" style="flex:1; min-width:300px; margin-right:10px; padding: 0 10px;">
            <table style="width:100%">
                <thead><tr>{c_ths}</tr></thead>
                <tbody>{''.join(c_rows)}</tbody>
            </table>
        </div>
        """
        
        # Build Pie Chart
        # Build Pie Chart
        pie_html = ""
        if fig_pie:
            # Make pie chart smaller for report
            fig_pie_copy = go.Figure(fig_pie)
            fig_pie_copy.update_layout(
                height=220,
                width=280,
                margin=dict(t=5, b=5, l=5, r=5),
                showlegend=False,
            )
            pie_content = fig_pie_copy.to_html(full_html=False, include_plotlyjs='cdn')
            pie_html = f"""
            <div class="chart-container" style="flex:0 0 300px; border-bottom:none; padding: 0 10px;">
                {pie_content}
            </div>
            """
            
        contrib_html = f"""
            <div class="section-header">Análise de Contribuição</div>
            <div style="display:flex; flex-wrap:wrap; padding:10px 20px; align-items:flex-start;">
                {contrib_table_html}
                {pie_html}
            </div>
        """

    # --- Build Pivot Table (Year x Month) ---
    months = ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun", "Jul", "Ago", "Set", "Out", "Nov", "Dez"]
    pivot_ths = "<th>Ano / Estratégia</th>" + "".join(f"<th>{m}</th>" for m in months) + "<th>Acum. Ano</th>"
    
    pivot_rows = []
    
    years = sorted(raw_df.index.year.unique(), reverse=True)
    strategies = list(raw_df.columns)
    
    # Sort strategies: Portfolio first, then Benchmark, then others
    def strat_sort_key(s):
        s = s.lower()
        if "portfolio" in s: return 0
        if "benchmark" in s or "cdi" in s or "ibov" in s: return 1
        return 2
    
    strategies = sorted(strategies, key=strat_sort_key)

    for year in years:
        # 1. Year Header
        pivot_rows.append(f'<tr class="year-header"><td colspan="14">{year}</td></tr>')
        
        df_year = raw_df[raw_df.index.year == year]
        
        # Helper to generate cells
        def build_row_cells(strategy_name, data_series, is_percent_of_bench=False, benchmark_series=None):
             tds = []
             # YTD Calculation
             if is_percent_of_bench:
                 ytd_port = (1 + data_series.fillna(0)).prod() - 1
                 ytd_bench = (1 + benchmark_series.fillna(0)).prod() - 1
                 
                 if abs(ytd_bench) > 1e-6:
                     val = (ytd_port / ytd_bench) * 100
                     txt = f"{val:.1f}%"
                 else:
                     txt = "-"
                 ytd_html = f"<b>{txt}</b>"
             else:
                 ytd_val = (1 + data_series.fillna(0)).prod() - 1
                 color_ytd = "green" if ytd_val >= 0 else "red"
                 ytd_html = f'<span style="color:{color_ytd}">{ytd_val*100:.2f}%</span>'

             # Months Generation
             month_cells = []
             for m_idx in range(1, 13):
                 try:
                     # Filter for month
                     val_s = data_series[data_series.index.month == m_idx]
                     if not val_s.empty and pd.notna(val_s.iloc[0]):
                         v = val_s.iloc[0]
                         
                         if is_percent_of_bench:
                             # Benchmark value for this month
                             b_s = benchmark_series[benchmark_series.index.month == m_idx]
                             if not b_s.empty and pd.notna(b_s.iloc[0]):
                                 b = b_s.iloc[0]
                                 if abs(b) > 1e-9:
                                     p_val = (v / b) * 100
                                     cell_txt = f"{p_val:.0f}%"
                                     month_cells.append(f'<td style="font-size:10px">{cell_txt}</td>')
                                 else:
                                     month_cells.append('<td>-</td>')
                             else:
                                 month_cells.append('<td>-</td>')
                         else:
                             txt = f"{v * 100:.2f}%"
                             color = "green" if v >= 0 else "red"
                             month_cells.append(f'<td style="color:{color}">{txt}</td>')
                     else:
                         month_cells.append("<td>-</td>")
                 except Exception:
                     month_cells.append("<td>-</td>")
             
             label_cell = f"<td>{_html_escape(strategy_name)}</td>"
             return [label_cell] + month_cells + [f"<td>{ytd_html}</td>"]

        # 2. Main Rows
        # Identify key columns anew to be safe inside loop (though constant)
        all_cols = list(raw_df.columns)
        portfolio_col = next((c for c in all_cols if "Portfolio" in c), None)
        benchmark_col = next((c for c in all_cols if "Benchmark" in c or "CDI" in c or "IBOV" in c), None)
        sub_strats = [c for c in all_cols if c != portfolio_col and c != benchmark_col]
        sub_strats.sort()

        if portfolio_col:
            cells = build_row_cells(portfolio_col, df_year[portfolio_col])
            pivot_rows.append(f'<tr class="main-row">{"".join(cells)}</tr>')
            
        if benchmark_col:
            cells = build_row_cells(benchmark_col, df_year[benchmark_col])
            pivot_rows.append(f'<tr class="main-row">{"".join(cells)}</tr>')
            
            # 3. % of Benchmark Row (Derived)
            if portfolio_col:
                cells = build_row_cells(
                    "% do Benchmark", 
                    df_year[portfolio_col], 
                    is_percent_of_bench=True, 
                    benchmark_series=df_year[benchmark_col]
                )
                pivot_rows.append(f'<tr class="main-row" style="font-style:italic; border-bottom:2px solid #ddd;">{"".join(cells)}</tr>')

        # 4. Sub Rows
        for strat in sub_strats:
             cells = build_row_cells(strat, df_year[strat])
             pivot_rows.append(f'<tr class="sub-row">{"".join(cells)}</tr>')

    pivot_table_html = f"""
         <div class="section-header">Matriz de Performance Mensal</div>
        <div class="table-wrap">
            <table>
                <thead><tr>{pivot_ths}</tr></thead>
                <tbody>{''.join(pivot_rows)}</tbody>
            </table>
        </div>
    """

    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8" />
      <title>{_html_escape(title)}</title>
      {css}
    </head>
    <body>
      <div class="page">
        <div class="main-header">
          {logo_html}
          <div class="header-text">
            <div class="main-title">{_html_escape(title)}</div>
            <div class="main-subtitle">{_html_escape(subtitle)}</div>
          </div>
        </div>
        
        {stats_html}
        
        {contrib_html}
        
        {chart_html}
        
        {drawdown_html}
        
        {pivot_table_html}
        
        <div class="footer">
          Uso Interno - Não Compartilhar
        </div>
      </div>
    </body>
    </html>
    """
    return html_content


# =====================================================
# DATABASE MANAGEMENT SECTION
# =====================================================

st.sidebar.header("💾 Data Management")

# Check if DB exists and load
existing_df = load_from_db()
has_existing_data = not existing_df.empty

if has_existing_data:
    st.sidebar.success(f"✅ Database loaded")
    st.sidebar.caption(
        f"📅 {existing_df.index.min().strftime('%b-%Y')} to "
        f"{existing_df.index.max().strftime('%b-%Y')}"
    )
    st.sidebar.caption(f"📊 {len(existing_df.columns)} strategies, {len(existing_df)} months")
else:
    st.sidebar.warning("⚠️ No database found")
    st.sidebar.caption("Upload an Excel file to create one")

# File upload in sidebar
uploaded_file = st.sidebar.file_uploader(
    "Upload Excel file" if not has_existing_data else "Upload to update",
    type=["xlsx", "xls"],
)

# Handle file upload
if uploaded_file:
    new_df = parse_excel(uploaded_file)

    if new_df.empty:
        st.sidebar.error("No valid data found in file")
    else:
        # Compare with existing data
        diff = compare_dataframes(existing_df, new_df)

        if diff["has_changes"]:
            st.sidebar.markdown("---")
            st.sidebar.subheader("📋 Changes Detected")

            if diff["new_months"]:
                st.sidebar.write(f"**New months:** {len(diff['new_months'])}")
                with st.sidebar.expander("View new months"):
                    for d in diff["new_months"][:10]:
                        st.write(f"• {d.strftime('%b-%Y')}")
                    if len(diff["new_months"]) > 10:
                        st.write(f"... and {len(diff['new_months']) - 10} more")

            if diff["new_strategies"]:
                st.sidebar.write(f"**New strategies:** {len(diff['new_strategies'])}")
                with st.sidebar.expander("View new strategies"):
                    for s in diff["new_strategies"]:
                        st.write(f"• {s}")

            if diff["updated_cells"]:
                st.sidebar.write(f"**Updated values:** {diff['updated_cells']}")

            # Confirmation buttons
            col1, col2 = st.sidebar.columns(2)
            with col1:
                if st.button("✅ Update DB", type="primary", use_container_width=True):
                    merged_df = merge_dataframes(existing_df, new_df)
                    save_to_db(merged_df)
                    st.sidebar.success("Database updated!")
                    st.rerun()

            with col2:
                if st.button("❌ Cancel", use_container_width=True):
                    st.rerun()
        else:
            st.sidebar.info("No changes detected in uploaded file")

# Option to delete database
if has_existing_data:
    st.sidebar.markdown("---")
    with st.sidebar.expander("⚠️ Danger Zone"):
        if st.button("🗑️ Delete Database", type="secondary"):
            st.session_state.confirm_delete = True

        if st.session_state.get("confirm_delete", False):
            st.warning("Are you sure? This cannot be undone.")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Yes, delete", type="primary"):
                    DB_PATH.unlink(missing_ok=True)
                    st.session_state.confirm_delete = False
                    st.rerun()
            with col2:
                if st.button("No, keep it"):
                    st.session_state.confirm_delete = False
                    st.rerun()

st.sidebar.header("🖼️ Report Branding")
uploaded_logo = st.sidebar.file_uploader(
    "Upload Logo (for PDF/HTML report)",
    type=["png", "jpg", "jpeg", "svg"],
    key="logo_uploader",
)

if uploaded_logo:
    st.sidebar.image(uploaded_logo, width=150, caption="Logo preview")
    
# =====================================================
# MAIN APP LOGIC
# =====================================================

# Use existing data from DB
df = existing_df

if df.empty:
    st.info("👆 Upload your Excel file in the sidebar to get started")

    st.subheader("Expected Format")
    st.code(
        """
| Data    | 10SIM  | FII    | IFIX   | DIVIDENDOS | ETF STRATEGY | BDR    |
|---------|--------|--------|--------|------------|--------------|--------|
| out-09  | 0,20%  |        |        |            |              |        |
| nov-09  | 13,30% |        |        |            |              |        |
        """,
        language="text",
    )
    st.stop()

# Get available strategies (columns with data)
available_strategies = [col for col in df.columns if df[col].notna().sum() > 0]

min_date = df.index.min()
max_date = df.index.max()

st.divider()

# --- SETTINGS ON MAIN PAGE ---
st.subheader("⚙️ Settings")

col_strat, col_empty = st.columns([2, 1])

with col_strat:
    selected_strategies = st.multiselect(
        "Select Strategies",
        available_strategies,
        default=available_strategies[:2]
        if len(available_strategies) >= 2
        else available_strategies,
        key="selected_strategies",
    )

# =====================================================
# AUTO-DETECT DATE LOGIC
# =====================================================

if "trigger_auto_detect" not in st.session_state:
    st.session_state.trigger_auto_detect = False

if st.session_state.trigger_auto_detect:
    strategies = st.session_state.get("selected_strategies", [])
    if strategies:
        smart_date = get_smart_start_date(df, strategies, min_date)
        st.session_state.start_date_input = smart_date.date()
    st.session_state.trigger_auto_detect = False

if "start_date_input" not in st.session_state:
    st.session_state.start_date_input = min_date.date()

col_date1, col_btn, col_date2 = st.columns([2, 1, 2])

with col_date1:
    start_date = st.date_input(
        "Start Date",
        min_value=min_date.date(),
        max_value=max_date.date(),
        key="start_date_input",
    )

with col_btn:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button(
        "🔄 Auto-detect",
        help="Set start date to when all selected strategies have data",
    ):
        if selected_strategies:
            st.session_state.trigger_auto_detect = True
            st.rerun()
        else:
            st.warning("Select at least one strategy first")

with col_date2:
    end_date = st.date_input(
        "End Date",
        value=max_date.date(),
        min_value=min_date.date(),
        max_value=max_date.date(),
        key="end_date_input",
    )

st.subheader("📊 Benchmark")
benchmark_option = st.selectbox(
    "Select Benchmark for Comparison",
    options=["None", "CDI", "IBOV"],
    index=1  # Default to CDI
)

# =====================================================
# WEIGHTS / REBALANCE LOGIC
# =====================================================

weights = {}
if selected_strategies:
    if "trigger_rebalance" not in st.session_state:
        st.session_state.trigger_rebalance = False

    if st.session_state.trigger_rebalance:
        equal_weight = 100.0 / len(selected_strategies)
        for strat in selected_strategies:
            st.session_state[f"weight_{strat}"] = equal_weight
        st.session_state.trigger_rebalance = False

    for strat in selected_strategies:
        if f"weight_{strat}" not in st.session_state:
            st.session_state[f"weight_{strat}"] = 100.0 / len(selected_strategies)

    col_weights, col_rebalance = st.columns([3, 1])

    with col_weights:
        st.markdown("**Portfolio Weights (%)**")

    with col_rebalance:
        if st.button(
            "⚖️ Equal Weights",
            help="Set all weights equally to sum to 100%",
        ):
            st.session_state.trigger_rebalance = True
            st.rerun()

    weight_cols = st.columns(min(len(selected_strategies), 4))
    for i, strategy in enumerate(selected_strategies):
        with weight_cols[i % len(weight_cols)]:
            new_weight = st.number_input(
                strategy,
                min_value=0.0,
                max_value=100.0,
                step=5.0,
                key=f"weight_{strategy}",
            )
            weights[strategy] = new_weight / 100

    total_weight = sum(weights.values())
    if abs(total_weight - 1.0) > 0.001:
        st.warning(f"⚠️ Weights sum to {total_weight*100:.1f}% (should be 100%)")
    else:
        st.success("✅ Weights sum to 100%")

st.divider()

# Filter data by date range
mask = (df.index >= pd.Timestamp(start_date)) & (df.index <= pd.Timestamp(end_date))
df_filtered = df.loc[mask]

if selected_strategies and not df_filtered.empty:
    # Calculate returns
    df_strategies = df_filtered[selected_strategies].copy()

    # Portfolio monthly return
    portfolio_monthly = calculate_portfolio_return(df_strategies, weights)

    # Cumulative returns
    cum_strategies = calculate_cumulative_returns(df_strategies)
    cum_portfolio = calculate_cumulative_returns(
        portfolio_monthly.to_frame("Portfolio")
    )["Portfolio"]

    # --- BENCHMARK DATA PREPARATION ---
    benchmark_series = None
    if benchmark_option != "None":
        bm_data = load_benchmark_data(benchmark_option)
        # Filter benchmark to match df_filtered date range
        # Reindex to match portfolio index, filling missing with 0 (or NaN if you prefer rigid alignment)
        benchmark_series = bm_data.reindex(df_filtered.index).fillna(0)
    
    # --- CHARTS ---
    st.header("📈 Cumulative Returns")

    fig = go.Figure()

    # Custom Ceres colors
    CERES_COLORS = [
        "#013220",
        "#57575a",
        "#b08568",
        "#09202e",
        "#582308",
        "#7a6200",
    ]
    colors = CERES_COLORS
    for i, col in enumerate(selected_strategies):
        fig.add_trace(
            go.Scatter(
                x=cum_strategies.index,
                y=cum_strategies[col] * 100,
                name=col,
                line=dict(color=colors[i % len(colors)]),
                hovertemplate="%{x|%b-%Y}: %{y:.2f}%<extra>" + col + "</extra>",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=cum_portfolio.index,
            y=cum_portfolio * 100,
            name="Portfolio",
            line=dict(color=BRAND_BROWN, width=3),
            hovertemplate="%{x|%b-%Y}: %{y:.2f}%<extra>Portfolio</extra>",
        )
    )
    
    if benchmark_series is not None:
        cum_benchmark = calculate_cumulative_returns(benchmark_series)
        fig.add_trace(
            go.Scatter(
                x=cum_benchmark.index,
                y=cum_benchmark * 100,
                name=f"Benchmark ({benchmark_option})",
                line=dict(color="gray", width=2, dash="dash"),
                hovertemplate="%{x|%b-%Y}: %{y:.2f}%<extra>Benchmark</extra>",
            )
        )

    fig.update_layout(
        yaxis_title="Cumulative Return (%)",
        xaxis_title="Date",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=380,
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- DRAWDOWN CHART ---
    st.header("📉 Drawdown")
    
    fig_drawdown = go.Figure()
    
    # Calculate and plot drawdowns
    # Strategies
    for i, col in enumerate(selected_strategies):
        dd_series = (1 + df_strategies[col]).cumprod() / (1 + df_strategies[col]).cumprod().cummax() - 1
        fig_drawdown.add_trace(
            go.Scatter(
                x=dd_series.index,
                y=dd_series * 100,
                name=col,
                line=dict(color=colors[i % len(colors)], width=1),
                fill='tozeroy',
                # fillcolor=... removed to avoid parsing errors with non-hex colors
                hovertemplate="%{x|%b-%Y}: %{y:.2f}%<extra>" + col + "</extra>",
            )
        )
    
    # Portfolio
    dd_portfolio = (1 + portfolio_monthly).cumprod() / (1 + portfolio_monthly).cumprod().cummax() - 1
    # Convert BRAND_BROWN to rgba for fill
    # brand_brown_rgb = hex to rgb... simpler to just use fill='tozeroy' and let plotly handle opacity or set simple color
    fig_drawdown.add_trace(
        go.Scatter(
            x=dd_portfolio.index,
            y=dd_portfolio * 100,
            name="Portfolio",
            line=dict(color=BRAND_BROWN, width=2),
            fill='tozeroy',
            hovertemplate="%{x|%b-%Y}: %{y:.2f}%<extra>Portfolio</extra>",
        )
    )
    
    # Benchmark
    if benchmark_series is not None:
         dd_bm = (1 + benchmark_series.fillna(0)).cumprod() / (1 + benchmark_series.fillna(0)).cumprod().cummax() - 1
         fig_drawdown.add_trace(
            go.Scatter(
                x=dd_bm.index,
                y=dd_bm * 100,
                name=f"Benchmark ({benchmark_option})",
                line=dict(color="gray", width=1, dash="dash"),
                # No fill for benchmark usually to keep it clean, or very light
                hovertemplate="%{x|%b-%Y}: %{y:.2f}%<extra>Benchmark</extra>",
            )
        )

    fig_drawdown.update_layout(
        yaxis_title="Drawdown (%)",
        xaxis_title="Date",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=300,
    )
    st.plotly_chart(fig_drawdown, use_container_width=True)

    # --- CONTRIBUTION ANALYSIS ---
    st.header("🎯 Contribution Analysis")

    contributions = {}
    for strategy in selected_strategies:
        strategy_contrib = df_strategies[strategy].fillna(0) * weights[strategy]
        contributions[strategy] = (1 + strategy_contrib).prod() - 1

    total_return = (1 + portfolio_monthly).prod() - 1

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Final Returns")
        metrics_df = pd.DataFrame(
            {
                "Strategy": list(contributions.keys()) + ["Portfolio"],
                "Weight": [f"{weights[s]*100:.1f}%" for s in contributions] + ["100%"],
                "Contribution": [f"{v*100:.2f}%" for v in contributions.values()]
                + [f"{total_return*100:.2f}%"],
            }
        )
        st.dataframe(metrics_df, hide_index=True, use_container_width=True)

    with col2:
        st.subheader("Contribution Breakdown")
        
        # Colors with BRAND_BROWN first
        pie_colors = [BRAND_BROWN] + [c for c in CERES_COLORS if c != BRAND_BROWN]
        
        fig_pie = go.Figure(
            data=[
                go.Pie(
                    labels=list(contributions.keys()),
                    values=[max(0, v) for v in contributions.values()],
                    hole=0.4,
                    marker=dict(colors=pie_colors),
                    textinfo='label+percent',
                    textfont_size=11,
                )
            ]
        )
        fig_pie.update_layout(
            height=280,  # Smaller height
            margin=dict(t=5, b=5, l=5, r=5),
            showlegend=False,  # Cleaner without legend since labels are on chart
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # --- STATISTICS ---
    st.header("📊 Statistics")

    stats = []
    
    # Calculate Benchmark Stats if available (for display)
    if benchmark_series is not None:
        bm_series_clean = benchmark_series.dropna()
        if len(bm_series_clean) > 0:
            bm_cum = (1 + bm_series_clean).prod() - 1
            bm_ann_ret_display = (1 + bm_cum) ** (12 / len(bm_series_clean)) - 1
            bm_vol = bm_series_clean.std() * (12**0.5)
            bm_max_dd = ((1 + bm_series_clean).cumprod() / (1 + bm_series_clean).cumprod().cummax() - 1).min()
            
            stats.append({
                "Strategy": f"Benchmark ({benchmark_option})",
                "Total Return": f"{bm_cum*100:.2f}%",
                "Ann. Return": f"{bm_ann_ret_display*100:.2f}%",
                "Ann. Vol": f"{bm_vol*100:.2f}%",
                "Sharpe": "-",
                "Max Drawdown": f"{bm_max_dd*100:.2f}%",
            })

    # Load CDI for correct Sharpe Calculation (Risk Free Rate)
    # Regardless of selected benchmark, Sharpe should be excess return over Risk Free (CDI)
    try:
        cdi_data = load_benchmark_data("CDI")
        cdi_series = cdi_data.reindex(df_filtered.index).fillna(0)
        cdi_cum = (1 + cdi_series).prod() - 1
        # Calculate annualized CDI specifically for the period
        days_in_period = (cdi_series.index.max() - cdi_series.index.min()).days
        if days_in_period > 0:
            cdi_ann_ret = (1 + cdi_cum) ** (365 / days_in_period) - 1
        else:
             # Fallback if days calculation fails (e.g. single month)
            cdi_ann_ret = (1 + cdi_cum) ** (12 / len(cdi_series)) - 1 if len(cdi_series) > 0 else 0.10 # Fallback 10%
    except Exception:
        cdi_ann_ret = 0.10 # Robust fallback (10% approx historical avg)

    strategies_to_calc = selected_strategies + ["Portfolio"]
    for col in strategies_to_calc:
        series = portfolio_monthly if col == "Portfolio" else df_strategies[col]
        series = series.dropna()
        if len(series) == 0:
            continue
        cum_ret = (1 + series).prod() - 1
        ann_ret = (1 + cum_ret) ** (12 / len(series)) - 1 if len(series) > 0 else 0
        vol = series.std() * (12**0.5)
        
        # Sharpe Calculation (Excess return over CDI)
        excess_ret = ann_ret - cdi_ann_ret
        sharpe = excess_ret / vol if vol > 0 else 0
            
        max_dd = ((1 + series).cumprod() / (1 + series).cumprod().cummax() - 1).min()

        stats.append(
            {
                "Strategy": col,
                "Total Return": f"{cum_ret*100:.2f}%",
                "Ann. Return": f"{ann_ret*100:.2f}%",
                "Ann. Vol": f"{vol*100:.2f}%",
                "Sharpe": f"{sharpe:.2f}",
                "Max Drawdown": f"{max_dd*100:.2f}%",
            }
        )

    stats_df = pd.DataFrame(stats)
    st.dataframe(stats_df, hide_index=True, use_container_width=True)

    # --- DETAILED MONTHLY TABLE ---
    st.header("📅 Detailed Monthly Returns")

    monthly_table = pd.DataFrame(index=df_filtered.index)
    monthly_table.index.name = "Date"

    for strat in selected_strategies:
        monthly_table[f"{strat} (Monthly)"] = df_strategies[strat]

    monthly_table["Portfolio (Monthly)"] = portfolio_monthly
    
    if benchmark_series is not None:
        monthly_table[f"{benchmark_option} (Monthly)"] = benchmark_series

    for strat in selected_strategies:
        monthly_table[f"{strat} (Cumul.)"] = cum_strategies[strat]

    monthly_table["Portfolio (Cumul.)"] = cum_portfolio
    
    if benchmark_series is not None:
        monthly_table[f"{benchmark_option} (Cumul.)"] = cum_benchmark

    for strat in selected_strategies:
        monthly_table[f"{strat} Contrib."] = (
            df_strategies[strat].fillna(0) * weights[strat]
        )

    display_table = monthly_table.copy()
    display_table.index = display_table.index.strftime("%b-%Y")

    for col in display_table.columns:
        display_table[col] = display_table[col].apply(format_pct)

    col_order = []
    for strat in selected_strategies:
        col_order.append(f"{strat} (Monthly)")
    col_order.append("Portfolio (Monthly)")
    
    if benchmark_series is not None:
        col_order.append(f"{benchmark_option} (Monthly)")
        
    for strat in selected_strategies:
        col_order.append(f"{strat} (Cumul.)")
    col_order.append("Portfolio (Cumul.)")
    
    if benchmark_series is not None:
        col_order.append(f"{benchmark_option} (Cumul.)")
        
    for strat in selected_strategies:
        col_order.append(f"{strat} Contrib.")

    display_table = display_table[col_order]

    tab1, tab2, tab3 = st.tabs(
        ["📊 Monthly Returns", "📈 Cumulative Returns", "🎯 Contributions"]
    )

    with tab1:
        monthly_cols = [f"{s} (Monthly)" for s in selected_strategies] + [
            "Portfolio (Monthly)"
        ]
        if benchmark_series is not None:
            monthly_cols.append(f"{benchmark_option} (Monthly)")
            
        st.dataframe(
            display_table[monthly_cols],
            use_container_width=True,
            height=400,
        )

    with tab2:
        cumul_cols = [f"{s} (Cumul.)" for s in selected_strategies] + [
            "Portfolio (Cumul.)"
        ]
        if benchmark_series is not None:
            cumul_cols.append(f"{benchmark_option} (Cumul.)")
            
        st.dataframe(
            display_table[cumul_cols],
            use_container_width=True,
            height=400,
        )

    with tab3:
        contrib_cols = [f"{s} Contrib." for s in selected_strategies]
        st.dataframe(
            display_table[contrib_cols],
            use_container_width=True,
            height=400,
        )

    st.download_button(
        label="📥 Download Full Table (CSV)",
        data=monthly_table.to_csv(),
        file_name="portfolio_analysis.csv",
        mime="text/csv",
    )

    # Prepare Raw Data for HTML Report
    # Combine selected strategies + Portfolio + Benchmark (if present)
    raw_html_df = df_strategies[selected_strategies].copy()
    raw_html_df["Portfolio"] = portfolio_monthly
    if benchmark_series is not None:
        raw_html_df[f"Benchmark ({benchmark_option})"] = benchmark_series
        
    # Convert logo to base64 if uploaded
    logo_b64 = ""
    if uploaded_logo:
        logo_b64 = image_to_base64(uploaded_logo)

    # Generate HTML Report
    html_report = generate_portfolio_html(
        stats_df, 
        raw_html_df,
        fig,
        fig_drawdown=fig_drawdown,
        contribution_df=metrics_df,
        fig_pie=fig_pie,
        title="Relatório de Carteiras - Renda Variável",
        subtitle=f"{start_date.strftime('%b %Y')} - {end_date.strftime('%b %Y')}",
        logo_base64=logo_b64,  # NEW
    )
    
    st.download_button(
        label="📄 Download Report (HTML)",
        data=html_report,
        file_name="portfolio_report.html",
        mime="text/html",
    )

    # PDF Export Section
    st.markdown("---")
    st.subheader("📄 PDF Export")
    
    if "portfolio_pdf" not in st.session_state:
        st.session_state["portfolio_pdf"] = None
    
    col_pdf1, col_pdf2 = st.columns([1, 2])
    
    with col_pdf1:
        if st.button("🚀 Generate PDF Report", type="primary", use_container_width=True):
            try:
                with st.spinner("Rendering PDF (this may take a moment)..."):
                    # Use the same HTML content
                    pdf_bytes = html_to_pdf_playwright(html_report)
                    st.session_state["portfolio_pdf"] = pdf_bytes
                    st.success("PDF generated successfully!")
            except Exception as e:
                st.error(f"Failed to generate PDF: {e}")
                st.info("Ensure Playwright is installed: `pip install playwright && playwright install chromium`")
    
    with col_pdf2:
        if st.session_state["portfolio_pdf"]:
            st.download_button(
                label="📥 Download PDF Report",
                data=st.session_state["portfolio_pdf"],
                file_name="portfolio_report.pdf",
                mime="application/pdf",
                use_container_width=True,
            )