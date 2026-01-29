import streamlit as st
import pandas as pd
import sqlite3
import re
import unicodedata
import html as _html
from datetime import datetime
import json
import os
from pages.ceres_logo import LOGO_BASE64

PRESETS_FILE = "databases/presets_economatica.json"

# -------------------- Configuration & Theme --------------------
st.set_page_config(page_title="Análise Ações Economatica", layout="wide")

BRAND_BROWN = "#825120"
BRAND_BROWN_DARK = "#6B4219"
LIGHT_GRAY = "#F5F5F5"
BLOCK_BG = "#F8F8F8"
TABLE_BORDER = "#E0D5CA"
TEXT_DARK = "#333333"

DB_PATH = "databases/dados_economatico.db"
LOGO_URL = "https://docs.ceresinvestimentos.info/remote.php/dav/files/eduardo.nascimento/LogoCERESWealth.png"

# -------------------- Helpers --------------------
def _norm_text(x: object) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip().replace("\n", " ")
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return s.lower()

def parse_br_number(x: object) -> float | object:
    if pd.isna(x):
        return pd.NA
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip().lower()
    if s in ("", "nan", "na", "n/a", "n.a", "n.a.", "-", "–", "—"):
        return pd.NA

    is_negative = False
    if s.startswith("(") and s.endswith(")"):
        is_negative = True
        s = s[1:-1]

    s = s.replace("r$", "").replace("mn", "").replace("mi", "").replace("x", "").replace("%", "")
    s = re.sub(r"[^0-9\-.,]", "", s)
    if s == "" or s in ("-", "–", "—"):
        return pd.NA

    # SMART PARSING: Handle both 1.234,56 (BR) and 1234.56 (US/Standard)
    if "," in s and "." in s:
        # Both present. Usually dot is thousands, comma is decimal in BR: 1.234,56
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else: # US: 1,234.56
            s = s.replace(",", "")
    elif "," in s:
        # Only comma. Likely decimal: 1234,56
        s = s.replace(",", ".")
    # If only dot, it's already standard float format: 1234.56

    try:
        val = float(s)
        if is_negative:
            val = -val
        return val
    except Exception:
        return pd.NA

def _format_ptbr_number(value: float, decimals: int = 2, trim: bool = True) -> str:
    try:
        s = f"{float(value):,.{decimals}f}"
    except Exception:
        return "-"
    s = s.replace(",", "X").replace(".", ",").replace("X", ".")
    if trim and "," in s:
        s = s.rstrip("0").rstrip(",")
    return s

def _is_percent_col(colname: str) -> bool:
    n = str(colname).strip().lower()
    tokens = ["%", "yld", "yield", "margem", "margin", "roe", "roa", "roic", "growth", "cagr", "var.", "variação"]
    return any(t in n for t in tokens)

def _is_multiple_col(colname: str) -> bool:
    n = str(colname).strip().lower()
    tokens = ["p/l", "p\\l", "p/e", "p\\e", "p/b", "p\\b", "p/vp", "ev/ebitda", "ev ebitda", "net debt/ebitda", "dívida líquida/ebitda", "divida liquida/ebitda", "vezes"]
    return any(t in n for t in tokens)

def _best_is_higher(colname: str) -> bool:
    if _is_multiple_col(colname):
        return False
    n = str(colname).strip().lower()
    higher_good_tokens = ["market cap", "volume", "vol.", "vol 3m", "roe", "roa", "roic", "yld", "yield", "dy", "margem", "margin", "crescimento", "growth", "cagr", "valor de mercado"]
    return any(t in n for t in higher_good_tokens) or _is_percent_col(colname)

def _html_escape(x: object) -> str:
    return _html.escape("" if pd.isna(x) else str(x))

# -------------------- Data Loading --------------------
@st.cache_data
def load_economatica_data():
    if not sqlite3.sqlite_version:
        st.error("SQLite3 not available.")
        return pd.DataFrame()
    
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Detect available tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [t[0] for t in cursor.fetchall()]
        
        if not tables:
            conn.close()
            st.warning("Nenhuma tabela encontrada no banco de dados.")
            return pd.DataFrame()
            
        # Priority: 'sheet1', then the first available table
        table_name = "sheet1" if "sheet1" in tables else tables[0]
        
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        conn.close()
        
        # Clean column names
        def clean_col(c):
            # Remove newlines and multiple spaces
            c = str(c).replace("\n", " ").strip()
            c = re.sub(r"\s+", " ", c)
            # Comprehensive list of units to remove
            units = [
                "em milhares", "em R$", "em %", "em vezes", "em units", 
                "em dif p.p.", "em unidades", "consolid:sim*", "consolid:não*",
                "Mais recente", "Em moeda orig", "de 12 meses"
            ]
            for u in units:
                # Use regex to match units case-insensitively, potentially surrounded by spaces or parentheses
                pattern = rf"\(?\s*{re.escape(u)}\s*\)?"
                c = re.sub(pattern, "", c, flags=re.IGNORECASE).strip()
            
            # Final touch: clean multiple spaces again if unit removal created them
            c = re.sub(r"\s+", " ", c).strip()
            return c
            
        df.columns = [clean_col(c) for c in df.columns]
        
        # Identify Ticker and Name columns
        ticker_choices = ['Ticker', 'Código', 'Ativo', 'Papel', 'Ativo / Papel']
        name_choices = ['Nome', 'Empresa', 'Razão Social', 'Nome Empresa']
        
        # Map first match found
        for tc in ticker_choices:
            if tc in df.columns:
                df = df.rename(columns={tc: 'Ticker'})
                break
        
        for nc in name_choices:
            if nc in df.columns:
                df = df.rename(columns={nc: 'Nome'})
                break
        
        # Ensure Ticker exists and is cleaned
        if 'Ticker' not in df.columns:
            st.error("Coluna de Ticker não identificada. Verifique os nomes das colunas no banco de dados.")
            return pd.DataFrame()
            
        df['Ticker'] = df['Ticker'].astype(str).str.strip()
        
        # Robust Sector Detection/Fallback
        if 'Setor Economatica' not in df.columns:
            sector_fallbacks = ['Subsetor Bovespa', 'Segmento listagem Bovespa', 'Setor NAICS ult disponiv']
            for sf in sector_fallbacks:
                if sf in df.columns:
                    df['Setor Economatica'] = df[sf]
                    break
            else:
                df['Setor Economatica'] = "N/A"
        
        return df
    except Exception as e:
        st.error(f"Erro ao carregar banco de dados: {e}")
        return pd.DataFrame()

# -------------------- Presets Logic --------------------
def load_presets():
    if not os.path.exists(PRESETS_FILE):
        return {}
    try:
        with open(PRESETS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return {}

def save_presets(presets):
    with open(PRESETS_FILE, "w", encoding="utf-8") as f:
        json.dump(presets, f, indent=4, ensure_ascii=False)

# -------------------- PDF Generator --------------------
def html_to_pdf_playwright(html_content: str) -> bytes:
    """
    Render HTML to a single-page PDF using headless Chromium via Playwright.
    Adapted from pages/20_TESTE_ONEPAGER_HTML.py
    """
    from playwright.sync_api import sync_playwright

    MAX_PX = 19000  # ~198 inches at 96 CSS px/in

    with sync_playwright() as p:
        browser = p.chromium.launch()
        context = browser.new_context()
        page = context.new_page()

        # Set content and wait for it to load
        page.set_content(html_content, wait_until="load")
        page.wait_for_load_state("networkidle")

        # Emulate screen media to ensure charts render as seen on screen
        page.emulate_media(media="screen")
        
        # Wait for fonts to load
        try:
            page.wait_for_function('document.fonts && document.fonts.status === "loaded"', timeout=5000)
        except: pass

        # Measure the .page container
        dims = page.evaluate("""
            () => {
              const el = document.querySelector('.page') || document.body;
              const rect = el.getBoundingClientRect();
              const width = Math.ceil(Math.max(el.scrollWidth, rect.width));
              const height = Math.ceil(Math.max(el.scrollHeight, rect.height));
              return { width, height };
            }
        """)

        width_px = int(max(800, min(dims.get("width", 1200), MAX_PX)))
        height_px = int(max(600, min(dims.get("height", 1000), MAX_PX)))

        # Add a buffer to height to avoid splitting onto a second page (spilling footer)
        # Adding 15px is usually enough to account for browser rounding errors
        width_px += 2
        height_px += 15

        pdf_bytes = page.pdf(
            print_background=True,
            prefer_css_page_size=False,
            width=f"{width_px}px",
            height=f"{height_px}px",
            margin={"top": "0px", "right": "0px", "bottom": "0px", "left": "0px"},
            scale=1.0
        )
        browser.close()
        return pdf_bytes

# -------------------- HTML Generator --------------------
def generate_consolidated_report_html(
    df_source: pd.DataFrame,
    tickers: list[str],
    metrics_table: list[str],
    metrics_bar: list[str] = None,
    metrics_hbar: list[str] = None,
    metrics_radar: list[str] = None,
    title: str = "Relatório Consolidado de Ações",
    subtitle: str = "Dados Economatica",
    show_graphs: list[str] = None,
    section_order: str = "graphs_first",
    transpose: bool = False,
) -> str:
    if show_graphs is None:
        show_graphs = ["bar", "hbar", "radar"]
    if metrics_bar is None: metrics_bar = metrics_table
    if metrics_hbar is None: metrics_hbar = metrics_table
    if metrics_radar is None: metrics_radar = metrics_table

    if df_source.empty or not tickers or (not metrics_table and not any([metrics_bar, metrics_hbar, metrics_radar])):
        return "<p>Nenhum dado para exibir.</p>"

    # Data Filtering
    base_cols = ["Ticker", "Nome"]
    if "Setor Economatica" in df_source.columns:
        base_cols.append("Setor Economatica")
    
    cols_to_use = [c for c in base_cols if c in df_source.columns]
    
    df = df_source[df_source["Ticker"].isin(tickers)].copy()
    df = df.drop_duplicates(subset=["Ticker"], keep="first")
    df = df.sort_values(by="Ticker")
    
    # All unique metrics across all components
    unique_metrics = list(set(metrics_table) | set(metrics_bar or []) | set(metrics_hbar or []) | set(metrics_radar or []))
    
    # -------------------- DATA PREP (Shared for all) --------------------
    num_map = {}
    rank_map = {}
    for m in unique_metrics:
        ser = df[m].apply(parse_br_number)
        num_map[m] = ser
        if ser.notna().sum() >= 2:
            rank = ser.rank(ascending=not _best_is_higher(m), method="min")
        else:
            rank = pd.Series([pd.NA] * len(ser), index=ser.index)
        rank_map[m] = rank

    def cell_bg_css(metric, idx_in_df):
        r = rank_map[metric].get(idx_in_df, pd.NA)
        if pd.isna(r): return ""
        total = rank_map[metric].dropna().max()
        if not total or pd.isna(total): return ""
        val = (r - 1) / max(total - 1, 1)
        alpha = 0.22 * (1.0 - float(val))
        return f"background: rgba(130,81,32,{alpha:.2f}); color: {TEXT_DARK};"

    rows_html = []
    ths = []

    if not transpose:
        ths = [f'<th class="sticky">{_html_escape(c)}</th>' for c in cols_to_use]
        for m in metrics_table:
            ths.append(f'<th class="sticky sortable" data-type="num">{_html_escape(m)}</th>')
            
        for i, row in df.iterrows():
            tds = []
            for c in cols_to_use:
                tds.append(f"<td>{_html_escape(row.get(c, ''))}</td>")
            for m in metrics_table:
                val = num_map[m].get(i, pd.NA)
                if not pd.isna(val):
                    txt = _format_ptbr_number(val, 2) + ("%" if _is_percent_col(m) else "x" if _is_multiple_col(m) else "")
                    if not _is_percent_col(m) and not _is_multiple_col(m):
                        decimals = 0 if abs(val) > 1000 else 2
                        txt = _format_ptbr_number(val, decimals)
                else:
                    txt = "-"
                style = cell_bg_css(m, i)
                tds.append(f'<td style="{style}">{txt}</td>')
            rows_html.append("<tr>" + "".join(tds) + "</tr>")
    else:
        # TRANSPOSED: Metrics as rows, Tickers as columns
        ths = [f'<th class="sticky">Métrica</th>']
        for _, row in df.iterrows():
            ths.append(f'<th class="sticky">{_html_escape(row["Ticker"])}</th>')
        
        for m in metrics_table:
            tds = [f'<td style="text-align:left; font-weight:600; position:sticky; left:0; background:white; z-index:3;">{_html_escape(m)}</td>']
            for i, row in df.iterrows():
                val = num_map[m].get(i, pd.NA)
                if not pd.isna(val):
                    txt = _format_ptbr_number(val, 2) + ("%" if _is_percent_col(m) else "x" if _is_multiple_col(m) else "")
                    if not _is_percent_col(m) and not _is_multiple_col(m):
                        decimals = 0 if abs(val) > 1000 else 2
                        txt = _format_ptbr_number(val, decimals)
                else:
                    txt = "-"
                style = cell_bg_css(m, i)
                tds.append(f'<td style="{style}">{txt}</td>')
            rows_html.append("<tr>" + "".join(tds) + "</tr>")

    # -------------------- CHART DATA FUNCTIONS --------------------
    # Color palette
    brand_colors = [
        'rgba(130, 81, 32, 0.85)', 'rgba(127, 140, 141, 0.85)', 'rgba(46, 125, 50, 0.85)',
        'rgba(25, 118, 210, 0.85)', 'rgba(211, 47, 47, 0.85)', 'rgba(245, 124, 0, 0.85)',
        'rgba(123, 31, 162, 0.85)', 'rgba(0, 151, 167, 0.85)'
    ]

    def prep_chart_json(metrics_to_use):
        if not metrics_to_use: return "[]", "[]"
        
        m_max = {m: max([abs(v) for v in num_map[m].dropna()] or [0.001]) for m in metrics_to_use}
        
        ds_list = []
        for idx, (i, row) in enumerate(df.iterrows()):
            ticker, name = str(row["Ticker"]), str(row["Nome"])
            raw_v, norm_v = [], []
            for m in metrics_to_use:
                val = num_map[m].get(i, pd.NA)
                v_f = float(val) if not pd.isna(val) else None
                raw_v.append(v_f)
                if v_f is None:
                    norm_v.append(0)
                else:
                    max_v = m_max[m]
                    if _is_multiple_col(m):
                        norm_v.append(round(100 * (max_v / (abs(v_f) + 0.001)), 1) if abs(v_f) > 0 else 100)
                    else:
                        norm_v.append(round(100 * abs(v_f) / (max_v + 0.0001), 1))
            
            color = brand_colors[idx % len(brand_colors)]
            ds_list.append({
                "label": f"{name} ({ticker})", "data": norm_v, "raw": raw_v,
                "backgroundColor": color, "borderColor": color,
                "backgroundColorLight": color.replace('0.85', '0.25'),
                "borderWidth": 2, "fill": True
            })
        return json.dumps(metrics_to_use), json.dumps(ds_list)

    labels_bar, datasets_bar = prep_chart_json(metrics_bar)
    labels_hbar, datasets_hbar = prep_chart_json(metrics_hbar)
    labels_radar, datasets_radar = prep_chart_json(metrics_radar)

    css = f"""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Open+Sans:wght@400;600;700&display=swap');
      :root {{ --brand: {BRAND_BROWN}; --brand-dark: {BRAND_BROWN_DARK}; --bg: {BLOCK_BG}; --text: {TEXT_DARK}; --tbl-border: {TABLE_BORDER}; --light: {LIGHT_GRAY}; }}
      * {{ box-sizing: border-box; }}
      body {{ font-family: 'Open Sans', sans-serif; background: #f0f0f0; color: var(--text); margin: 0; padding: 0; }}
      .page {{ max-width: 1400px; margin: 12px auto; background: white; box-shadow: 0 2px 10px rgba(0,0,0,0.08); border-radius: 6px; overflow: hidden; }}
      .header {{ background: var(--brand); color: #fff; padding: 14px 20px; display: flex; align-items: baseline; justify-content: space-between; }}
      .title {{ font-size: 18px; font-weight: 700; }}
      .subtitle {{ font-size: 12px; opacity: 0.9; }}
      .section {{ padding: 20px; border-bottom: 1px solid var(--tbl-border); }}
      .section-title {{ font-size: 16px; font-weight: 700; color: var(--brand); margin-bottom: 15px; border-left: 4px solid var(--brand); padding-left: 10px; }}
      
      .table-wrap {{ width: 100%; overflow-x: auto; }}
      table {{ width: 100%; border-collapse: separate; border-spacing: 0; font-size: 12px; }}
      thead th {{ position: sticky; top: 0; z-index: 2; background: var(--brand); color: #fff; text-align: center; padding: 10px 8px; font-weight: 700; border-bottom: 1px solid var(--tbl-border); white-space: nowrap; }}
      tbody td {{ padding: 10px 8px; border-bottom: 1px solid var(--tbl-border); text-align: center; white-space: nowrap; }}
      tbody tr:nth-child(even) td {{ background: var(--light); }}
      tbody td:first-child, thead th:first-child {{ text-align: left; position: sticky; left: 0; z-index: 3; background: white; }}
      thead th:first-child {{ z-index: 4; background: var(--brand); }}
      th.sortable {{ cursor: pointer; }}
      
      .charts-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px; }}
      @media (max-width: 900px) {{ .charts-grid {{ grid-template-columns: 1fr; }} }}
      .chart-container {{ background: var(--light); border-radius: 8px; padding: 16px; height: 400px; position: relative; }}
      .chart-container h3 {{ margin: 0 0 10px 0; font-size: 14px; text-align: center; color: var(--brand-dark); }}
      .chart-wrapper {{ position: relative; width: 100%; height: calc(100% - 30px); }}
      
      .radar-wrap {{ display: flex; justify-content: center; }}
      .radar-container {{ background: var(--light); border-radius: 8px; padding: 20px; width: 100%; max-width: 600px; height: 500px; }}
      
      .footer {{ padding: 10px 20px; color: #666; font-size: 11px; background: #fff; text-align: center; }}
    </style>
    """

    # Sortable only if not transposed (transposed sorting is weird)
    sort_js = "" if transpose else f"""
        const tbl = document.getElementById('cmp-table');
        let sortState = {{ col: -1, asc: true }};
        function getCellNum(td) {{
          const txt = td.textContent.trim().replace(/\\./g, '').replace(',', '.').replace('%', '').replace('x', '');
          const v = parseFloat(txt);
          return isNaN(v) ? NaN : v;
        }}
        function sortBy(thIdx) {{
          const tbody = tbl.tBodies[0];
          const rows = Array.from(tbody.rows);
          const asc = (sortState.col === thIdx) ? !sortState.asc : true;
          rows.sort((a, b) => {{
            const va = getCellNum(a.cells[thIdx]);
            const vb = getCellNum(b.cells[thIdx]);
            if (isNaN(va) && isNaN(vb)) return 0;
            if (isNaN(va)) return 1;
            if (isNaN(vb)) return -1;
            return asc ? va - vb : vb - va;
          }});
          rows.forEach(r => tbody.appendChild(r));
          sortState = {{ col: thIdx, asc }};
        }}
        tbl.tHead.rows[0].cells.forEach((th, idx) => {{
          if (th.classList.contains('sortable')) {{
            th.addEventListener('click', () => sortBy(idx));
          }}
        }});
    """

    js = f"""
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
    <script>
      (function() {{
        const commonOpts = {{
          responsive: true, maintainAspectRatio: false,
          plugins: {{
            legend: {{ position: 'top', labels: {{ boxWidth: 12, font: {{ size: 11 }} }} }},
            tooltip: {{
              callbacks: {{
                label: function(ctx) {{
                  const dsIdx = ctx.datasetIndex;
                  const dataIdx = ctx.dataIndex;
                  const chartId = ctx.chart.canvas.id.replace('Chart', '');
                  let rawVal = null;
                  if (chartId === 'bar') rawVal = {datasets_bar}[dsIdx].raw[dataIdx];
                  else if (chartId === 'hbar') rawVal = {datasets_hbar}[dsIdx].raw[dataIdx];
                  else if (chartId === 'radar') rawVal = {datasets_radar}[dsIdx].raw[dataIdx];
                  const formatted = rawVal !== null ? rawVal.toLocaleString('pt-BR', {{ minimumFractionDigits: 2, maximumFractionDigits: 2 }}) : 'N/A';
                  return ctx.dataset.label + ': ' + formatted;
                }}
              }}
            }}
          }}
        }};

        if (document.getElementById('barChart')) {{
          new Chart(document.getElementById('barChart'), {{
            type: 'bar',
            data: {{ labels: {labels_bar}, datasets: {datasets_bar} }},
            options: {{ ...commonOpts, scales: {{ y: {{ beginAtZero: true, max: 100, title: {{ display: true, text: 'Score' }} }} }} }}
          }});
        }}
        if (document.getElementById('hbarChart')) {{
          new Chart(document.getElementById('hbarChart'), {{
            type: 'bar',
            data: {{ labels: {labels_hbar}, datasets: {datasets_hbar} }},
            options: {{ ...commonOpts, indexAxis: 'y', scales: {{ x: {{ beginAtZero: true, max: 100 }} }} }}
          }});
        }}
        if (document.getElementById('radarChart')) {{
          const radarDatasets = {datasets_radar}.map(d => ({{ ...d, backgroundColor: d.backgroundColorLight, pointRadius: 3 }}));
          new Chart(document.getElementById('radarChart'), {{
            type: 'radar',
            data: {{ labels: {labels_radar}, datasets: radarDatasets }},
            options: {{ ...commonOpts, scales: {{ r: {{ suggestedMin: 0, suggestedMax: 100, ticks: {{ display: false }} }} }} }}
          }});
        }}
        {sort_js}
      }})();
    </script>
    """

    # Smart Layout Logic for Graphs
    bar_enabled = 'bar' in show_graphs
    hbar_enabled = 'hbar' in show_graphs
    radar_enabled = 'radar' in show_graphs
    
    # Decide where to put charts
    grid_charts = []
    below_charts_html = ""
    
    if bar_enabled and hbar_enabled:
        grid_charts.append(('bar', "Comparação de Performance (Barras)"))
        grid_charts.append(('hbar', "Distribuição de Métricas (Horizontal)"))
        if radar_enabled:
            below_charts_html = f"""
                <div class="radar-wrap">
                  <div class="radar-container">
                    <h3 style="text-align:center; color:var(--brand); margin-bottom:10px;">Visão 360° (Radar)</h3>
                    <div class="chart-wrapper"><canvas id="radarChart"></canvas></div>
                    <p style="font-size:10px; color:#888; text-align:center; margin-top:5px;">* Valores normalizados de 0-100. Para múltiplos, maior score indica valor mais atrativo (menor múltiplo).</p>
                  </div>
                </div>
            """
    elif (bar_enabled or hbar_enabled) and radar_enabled:
        # One bar + Radar -> Put both in grid!
        if bar_enabled: grid_charts.append(('bar', "Comparação de Performance (Barras)"))
        if hbar_enabled: grid_charts.append(('hbar', "Distribuição de Métricas (Horizontal)"))
        grid_charts.append(('radar', "Visão 360° (Radar)"))
    else:
        # Just bar(s) or just radar
        if bar_enabled: grid_charts.append(('bar', "Comparação de Performance (Barras)"))
        if hbar_enabled: grid_charts.append(('hbar', "Distribuição de Métricas (Horizontal)"))
        if radar_enabled:
            # If ONLY radar is enabled, put it in the wide wrap
            below_charts_html = f"""
                <div class="radar-wrap">
                  <div class="radar-container">
                    <h3 style="text-align:center; color:var(--brand); margin-bottom:10px;">Visão 360° (Radar)</h3>
                    <div class="chart-wrapper"><canvas id="radarChart"></canvas></div>
                    <p style="font-size:10px; color:#888; text-align:center; margin-top:5px;">* Valores normalizados de 0-100. Para múltiplos, maior score indica valor mais atrativo (menor múltiplo).</p>
                  </div>
                </div>
            """

    grid_html = ""
    if grid_charts:
        items = []
        for chart_id, chart_title in grid_charts:
            items.append(f"""
                <div class="chart-container">
                    <h3>{chart_title}</h3>
                    <div class="chart-wrapper"><canvas id="{chart_id}Chart"></canvas></div>
                </div>
            """)
        grid_html = f'<div class="charts-grid">{"".join(items)}</div>'

    graphs_html = f"""
        <div class="section">
          <div class="section-title">Visualização Gráfica (Scores Históricos)</div>
          {grid_html}
          {below_charts_html}
        </div>
    """ if (grid_html or below_charts_html) else ""

    table_html = f"""
        <div class="section">
          <div class="section-title">Tabela Comparativa Detalhada</div>
          <div class="table-wrap">
            <table id="cmp-table">
              <thead><tr>{''.join(ths)}</tr></thead>
              <tbody>{''.join(rows_html)}</tbody>
            </table>
          </div>
        </div>
    """

    content_html = ""
    if section_order == "graphs_first":
        content_html = graphs_html + table_html
    else:
        content_html = table_html + graphs_html

    html = f"""
    <!DOCTYPE html>
    <html lang="pt-BR">
    <head><meta charset="UTF-8" /><title>{_html_escape(title)}</title>{css}</head>
    <body>
      <div class="page">
        <div class="header">
          <div style="display: flex; align-items: center; gap: 15px;">
            <img src="data:image/png;base64,{LOGO_BASE64.strip()}" alt="Logo" style="height: 35px; border-radius: 4px;" />
            <div>
              <div class="title">{_html_escape(title)}</div>
              <div class="subtitle">{_html_escape(subtitle)}</div>
            </div>
          </div>
        </div>
        {content_html}
        <div class="footer">Relatório gerado automaticamente — Dados Economatica — {datetime.now().strftime('%d/%m/%Y %H:%M')}</div>
      </div>
      {js}
    </body>
    </html>
    """
    return html

# -------------------- Streamlit UI --------------------
st.title("📊 Análise de Ações - Economatica")
st.markdown("Comparador de ações utilizando os dados extraídos do Economatica.")

df_raw = load_economatica_data()

if df_raw.empty:
    st.warning("Banco de dados não encontrado ou vazio. Vá em 'Database Economatica' para fazer o upload.")
    st.stop()

# -------------------- PRESETS MANAGER --------------------
with st.expander("📂 Gerenciador de Presets (Salvar/Carregar Configurações)", expanded=False):
    col_p1, col_p2, col_p3 = st.columns([2, 1, 1])
    
    presets_data = load_presets()
    preset_names = list(presets_data.keys())
    
    with col_p1:
        sel_preset = st.selectbox("Selecione um Preset", [""] + preset_names, index=0)
        
    with col_p2:
        st.write("") # Spacer
        st.write("")
        if st.button("📥 Carregar Preset", use_container_width=True):
            if sel_preset and sel_preset in presets_data:
                p = presets_data[sel_preset]
                
                # Apply Metrics (Validate against available)
                valid_metrics = [m for m in p.get("metrics", []) if m in df_raw.columns]
                st.session_state["widget_sel_metrics"] = valid_metrics
                
                # Apply Sectors
                valid_sectors = [s for s in p.get("setores", []) if s in df_raw["Setor Economatica"].unique()]
                st.session_state["widget_sel_setor"] = valid_sectors
                
                # Apply Tickers (override auto-select)
                # Note: sel_tickers uses 'sel_tickers_widget' key
                valid_tickers = [t for t in p.get("tickers", []) if t in df_raw["Ticker"].values]
                st.session_state["sel_tickers_widget"] = valid_tickers
                st.session_state.sel_tickers = valid_tickers # Sync logic variable
                
                # Apply Filters
                for m, rng in p.get("filters", {}).items():
                    if len(rng) == 2:
                        st.session_state[f"filter_{m}"] = tuple(rng)
                
                # Apply MC Range
                mc_rng = p.get("market_cap_range", [])
                if len(mc_rng) == 2:
                    st.session_state["widget_sel_mc"] = tuple(mc_rng)

                # Apply Settings
                settings = p.get("settings", {})
                st.session_state["widget_show_bar"] = settings.get("show_bar", True)
                st.session_state["widget_show_hbar"] = settings.get("show_hbar", True)
                st.session_state["widget_show_radar"] = settings.get("show_radar", True)
                st.session_state["widget_order_opt"] = "Gráficos" if settings.get("order", "graphs_first") == "graphs_first" else "Tabela"
                st.session_state["widget_transpose_table"] = settings.get("transpose", False)
                
                # Custom Metrics
                st.session_state["widget_custom_metrics"] = settings.get("custom_metrics", False)
                cust_lists = p.get("custom_lists", {})
                st.session_state["widget_m_table"] = [m for m in cust_lists.get("table", []) if m in df_raw.columns]
                st.session_state["widget_m_bar"] = [m for m in cust_lists.get("bar", []) if m in df_raw.columns]
                st.session_state["widget_m_hbar"] = [m for m in cust_lists.get("hbar", []) if m in df_raw.columns]
                st.session_state["widget_m_radar"] = [m for m in cust_lists.get("radar", []) if m in df_raw.columns]
                
                st.success(f"Preset '{sel_preset}' carregado!")
                st.rerun()

    with col_p3:
        st.write("") # Spacer
        st.write("") 
        
    st.markdown("---")
    c_save1, c_save2 = st.columns([2, 1])
    with c_save1:
        new_preset_name = st.text_input("Nome para Salvar/Atualizar Preset")
    with c_save2:
        st.write("")
        st.write("")
        if st.button("💾 Salvar Configuração Atual", use_container_width=True):
            if not new_preset_name:
                st.error("Digite um nome para o preset.")
            else:
                # Gather state
                current_config = {
                    "metrics": st.session_state.get("widget_sel_metrics", []),
                    "setores": st.session_state.get("widget_sel_setor", []),
                    "tickers": st.session_state.get("sel_tickers_widget", []),
                    "market_cap_range": st.session_state.get("widget_sel_mc", []),
                    "filters": {}, # Will populate from filters in session state matching current metrics
                    "settings": {
                        "show_bar": st.session_state.get("widget_show_bar", True),
                        "show_hbar": st.session_state.get("widget_show_hbar", True),
                        "show_radar": st.session_state.get("widget_show_radar", True),
                        "order": "graphs_first" if st.session_state.get("widget_order_opt", "Gráficos") == "Gráficos" else "table_first",
                        "transpose": st.session_state.get("widget_transpose_table", False),
                        "custom_metrics": st.session_state.get("widget_custom_metrics", False)
                    },
                    "custom_lists": {
                        "table": st.session_state.get("widget_m_table", []),
                        "bar": st.session_state.get("widget_m_bar", []),
                        "hbar": st.session_state.get("widget_m_hbar", []),
                        "radar": st.session_state.get("widget_m_radar", [])
                    }
                }
                
                # Gather active filters
                # We iterate over selected metrics and check if a filter key exists
                for m in current_config["metrics"]:
                    f_key = f"filter_{m}"
                    if f_key in st.session_state:
                        current_config["filters"][m] = st.session_state[f_key]
                
                presets_data[new_preset_name] = current_config
                save_presets(presets_data)
                st.success(f"Preset '{new_preset_name}' salvo com sucesso!")
                st.rerun()

# Filters on Main Page
col_f1, col_f2 = st.columns(2)
with col_f1:
    setores = sorted(df_raw["Setor Economatica"].dropna().unique().tolist())
    sel_setor = st.multiselect("Filtrar por Setor", setores, key="widget_sel_setor")

with col_f2:
    # Market Cap Filter
    mc_col = "Valor de Mercado Atual"
    if mc_col in df_raw.columns:
        # Convert to float for slider
        mc_series = df_raw[mc_col].apply(parse_br_number).dropna()
        if not mc_series.empty:
            min_mc = float(mc_series.min())
            max_mc = float(mc_series.max())
            
            # SAFEGUARD: Clamp session state value if present
            if "widget_sel_mc" in st.session_state:
                curr = st.session_state["widget_sel_mc"]
                # Ensure tuple has 2 elements
                if isinstance(curr, (tuple, list)) and len(curr) == 2:
                    safe_min = max(min_mc, min(curr[0], max_mc))
                    safe_max = min(max_mc, max(curr[1], min_mc))
                    st.session_state["widget_sel_mc"] = (safe_min, safe_max)
            else:
                 # Ensure default is set if not present
                 st.session_state["widget_sel_mc"] = (min_mc, max_mc)

            sel_mc = st.slider("Filtrar por Valor de Mercado (milhares)", min_mc, max_mc, key="widget_sel_mc")
        else:
            sel_mc = None
    else:
        sel_mc = None

df_filtered = df_raw.copy()
if sel_setor:
    df_filtered = df_filtered[df_filtered["Setor Economatica"].isin(sel_setor)]
if sel_mc:
    # Apply MC filter
    df_filtered["_mc_tmp"] = df_filtered[mc_col].apply(parse_br_number).fillna(0)
    df_filtered = df_filtered[(df_filtered["_mc_tmp"] >= sel_mc[0]) & (df_filtered["_mc_tmp"] <= sel_mc[1])]
    df_filtered = df_filtered.drop(columns=["_mc_tmp"])

# Select Base Metrics First (needed for range filtering)
base_cols_info = ["Ticker", "Nome", "Setor Economatica", "Setor NAICS ult disponiv", "Classe", "Bolsa / Fonte", "Tipo de Ativo", "Ativo / Cancelado", "Data da Últ Cotação", "Data do Último Balanço", "Unnamed: 0", "Consolidado", "Link da Última Nota Explicativa"]
metric_candidates = [c for c in df_raw.columns if c not in base_cols_info]
select_defaults = ["P / VPA", "Preço / Lucro 12 meses", "Dividend Yield 12 meses", "EV vs EBITDA 12 meses", "ROE 12 meses", "Dívida Líquida vs EBITDA 12 meses"]
select_defaults = [d for d in select_defaults if d in metric_candidates]

st.markdown("### 1. Seleção de Métricas e Filtros de Valor")
sel_metrics = st.multiselect("Selecionar Métricas Base", metric_candidates, default=select_defaults, key="widget_sel_metrics")

# -------------------- Metric Range Filtering --------------------
with st.expander("🔍 Filtrar por Valores das Métricas", expanded=False):
    if not sel_metrics:
        st.info("Selecione métricas acima para filtrar por seus valores.")
    else:
        st.write("Ajuste os intervalos para filtrar as ações:")
        metric_filters = {}
        # Layout in columns for sliders
        s_cols = st.columns(2)
        for idx, m in enumerate(sel_metrics):
            with s_cols[idx % 2]:
                # Get numeric series for this metric
                m_series = df_filtered[m].apply(parse_br_number).dropna()
                if not m_series.empty:
                    m_min = float(m_series.min())
                    m_max = float(m_series.max())
                    if m_min == m_max:
                        st.write(f"**{m}**: Valor único ({m_min})")
                        metric_filters[m] = (m_min, m_max)
                    else:
                        suffix = "%" if _is_percent_col(m) else "x" if _is_multiple_col(m) else ""
                        
                        f_key = f"filter_{m}"
                        # SAFEGUARD: Clamp session state value if present
                        if f_key in st.session_state:
                            curr = st.session_state[f_key]
                            if isinstance(curr, (tuple, list)) and len(curr) == 2:
                                safe_min = max(m_min, min(curr[0], m_max))
                                safe_max = min(m_max, max(curr[1], m_min))
                                st.session_state[f_key] = (safe_min, safe_max)
                        
                        metric_filters[m] = st.slider(
                            f"{m} ({suffix})", 
                            m_min, m_max, (m_min, m_max),
                            key=f_key
                        )
                else:
                    st.write(f"**{m}**: Sem dados numéricos para filtrar")

        # Apply metric filters to df_filtered
        for m, (f_min, f_max) in metric_filters.items():
            df_filtered[f"_{m}_tmp"] = df_filtered[m].apply(parse_br_number).fillna(-999999) # Handle NAs if needed
            df_filtered = df_filtered[(df_filtered[f"_{m}_tmp"] >= f_min) & (df_filtered[f"_{m}_tmp"] <= f_max)]
            df_filtered = df_filtered.drop(columns=[f"_{m}_tmp"])

st.markdown("### 2. Seleção de Ações")

# -------------------- Smart Auto-Selection Logic --------------------
# Define session state key for previous sector selection to detect changes
if 'prev_sel_setor' not in st.session_state:
    st.session_state.prev_sel_setor = []

if sel_setor and sel_setor != st.session_state.prev_sel_setor:
    # Trigger auto-selection
    temp_df = df_filtered.copy()
    mc_metric = "Valor de Mercado Atual"
    vol_metric = "Volume Médio diário 3 meses"
    
    # Parse metrics for comparison
    temp_df['_mc'] = temp_df[mc_metric].apply(parse_br_number).fillna(0)
    temp_df['_vol'] = temp_df[vol_metric].apply(parse_br_number).fillna(0)
    
    # Identify unique stocks by first 4 characters of Ticker (e.g., PETR)
    temp_df['_stock_base'] = temp_df['Ticker'].str[:4]
    
    # Within each stock base, pick the ticker with the highest volume
    # Use sort + head(1) per group for safety
    idx = temp_df.sort_values('_vol', ascending=False).groupby('_stock_base').head(1).index
    best_tickers_df = temp_df.loc[idx]
    
    # Sort by Market Cap and pick top 5
    top_5 = best_tickers_df.sort_values(by='_mc', ascending=False).head(5)
    auto_selected = top_5['Ticker'].tolist()
    
    # Update current tickers in session state
    st.session_state.sel_tickers = auto_selected
    st.session_state.prev_sel_setor = sel_setor
elif not sel_setor:
    # If secteur is cleared, we don't necessarily clear tickers, 
    # but we reset the tracking so re-selecting sector triggers again
    st.session_state.prev_sel_setor = []

# Use session state for multiselect to allow auto-updates
if 'sel_tickers' not in st.session_state:
    st.session_state.sel_tickers = []

all_tickers = sorted(df_filtered["Ticker"].unique().tolist())
sel_tickers = st.multiselect(
    "Selecionar Ações (após filtros)", 
    all_tickers, 
    key='sel_tickers_widget', 
    default=st.session_state.sel_tickers if all(t in all_tickers for t in st.session_state.sel_tickers) else []
)
# Sync back to session state
st.session_state.sel_tickers = sel_tickers

# Additional Report Settings
with st.expander("⚙️ Configurações do Relatório", expanded=True):
    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        st.write("**Gráficos a incluir:**")
        show_bar = st.checkbox("Barras Verticais", value=True, key="widget_show_bar")
        show_hbar = st.checkbox("Barras Horizontais", value=True, key="widget_show_hbar")
        show_radar = st.checkbox("Radar", value=True, key="widget_show_radar")
        selected_graphs = []
        if show_bar: selected_graphs.append("bar")
        if show_hbar: selected_graphs.append("hbar")
        if show_radar: selected_graphs.append("radar")
    
    with col_s2:
        st.write("**Ordem das Seções:**")
        order_opt = st.radio("Mostrar primeiro:", ["Gráficos", "Tabela"], index=0, key="widget_order_opt")
        report_order = "graphs_first" if order_opt == "Gráficos" else "table_first"
        
    with col_s3:
        st.write("**Estrutura da Tabela:**")
        transpose_table = st.toggle("Inverter Tabela (Transpor)", value=False, key="widget_transpose_table")
        custom_metrics = st.toggle("Customizar métricas por componente", value=False, key="widget_custom_metrics")

if custom_metrics:
    st.info("Personalize as métricas para cada parte do relatório abaixo:")
    m_col1, m_col2 = st.columns(2)
    with m_col1:
        m_table = st.multiselect("Métricas da Tabela", metric_candidates, default=sel_metrics, key="widget_m_table")
        m_bar = st.multiselect("Métricas do Gráfico Barras", metric_candidates, default=sel_metrics, key="widget_m_bar")
    with m_col2:
        m_hbar = st.multiselect("Métricas do Gráfico Horizontal", metric_candidates, default=sel_metrics, key="widget_m_hbar")
        m_radar = st.multiselect("Métricas do Gráfico Radar", metric_candidates, default=sel_metrics, key="widget_m_radar")
else:
    m_table = m_bar = m_hbar = m_radar = sel_metrics

tab1, = st.tabs(["📊 Relatório Consolidado"])

with tab1:
    if st.button("Gerar Relatório Completo") or (sel_tickers and sel_metrics):
        if not sel_tickers:
            st.info("Selecione pelo menos uma ação.")
        elif not m_table and not any([m_bar, m_hbar, m_radar]):
            st.info("Selecione pelo menos uma métrica.")
        else:
            html_content = generate_consolidated_report_html(
                df_raw, 
                sel_tickers, 
                metrics_table=m_table,
                metrics_bar=m_bar if show_bar else None,
                metrics_hbar=m_hbar if show_hbar else None,
                metrics_radar=m_radar if show_radar else None,
                show_graphs=selected_graphs,
                section_order=report_order,
                transpose=transpose_table
            )
            st.components.v1.html(html_content, height=1400, scrolling=True)
            
            c1, c2 = st.columns(2)
            with c1:
                st.download_button("📩 Baixar Relatório (HTML)", html_content, f"relatorio_economatica_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html", "text/html", use_container_width=True)
            with c2:
                if st.button("📄 Gerar e Baixar PDF", use_container_width=True):
                    with st.spinner("Gerando PDF com fidelidade total..."):
                        try:
                            pdf_bytes = html_to_pdf_playwright(html_content)
                            st.download_button(
                                "🎁 Clique aqui para Salvar PDF",
                                pdf_bytes,
                                f"relatorio_economatica_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                "application/pdf",
                                use_container_width=True
                            )
                        except Exception as e:
                            st.error(f"Erro ao gerar PDF: {e}. Certifique-se que o Playwright está instalado: `pip install playwright` e `playwright install chromium`.")
