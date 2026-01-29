================================================================================
                    HTML/PDF REPORT STYLE GUIDE - A4 FORMAT
================================================================================

This guide documents the design system for generating consistent HTML/PDF reports.
Use as a reference for LLMs when generating financial reports.

--------------------------------------------------------------------------------
1. COLOR PALETTE
--------------------------------------------------------------------------------

# Primary Brand Colors
BRAND_BROWN = "#825120"        # Primary accent, headers, highlights
BRAND_BROWN_DARK = "#6B4219"   # Darker variant for text emphasis

# Neutrals
LIGHT_GRAY = "#F5F5F5"         # Table header backgrounds, alternating rows
BLOCK_BG = "#F8F8F8"           # Section header backgrounds
TABLE_BORDER = "#E0D5CA"       # Warm-toned borders (softer than gray)
TEXT_DARK = "#333333"          # Primary text color

# Chart Colors (Ceres Palette)
CERES_COLORS = [
    "#013220",  # Dark green
    "#57575a",  # Gray
    "#b08568",  # Tan/brown
    "#09202e",  # Navy
    "#582308",  # Dark red-brown
    "#7a6200",  # Olive/gold
]

# Semantic Colors
POSITIVE = "green"             # Positive returns
NEGATIVE = "red"               # Negative returns
BENCHMARK_LINE = "gray"        # Benchmark series (dashed)

--------------------------------------------------------------------------------
2. TYPOGRAPHY
--------------------------------------------------------------------------------

Font Family: 'Open Sans', 'Segoe UI', Arial, sans-serif
Import URL: https://fonts.googleapis.com/css2?family=Open+Sans:wght@400;600;700&display=swap

Font Sizes:
  - Title:          20px (main report title)
  - Subtitle:       12px (secondary info)
  - Section Header: 13px (section titles)
  - Table Header:   10px (column headers)
  - Table Body:     10px (cell content)
  - Sub Row:        9px  (secondary/nested rows)
  - Footer:         10px (footer text)

Font Weights:
  - Normal:    400
  - Semibold:  600
  - Bold:      700
  - Heavy:     800 (year headers in pivot tables)

--------------------------------------------------------------------------------
3. A4 DIMENSIONS
--------------------------------------------------------------------------------

Page Size: 210mm × 297mm (A4)
Page Width (content): 190mm (with 10mm margins)
Pixel Equivalent: ~794px × 1123px at 96 DPI

Spacing:
  - Page padding: 10mm
  - Horizontal content padding: 15px
  - Section header padding: 5px 15px
  - Chart container padding: 5px
  - Table cell padding: 3px 4px

--------------------------------------------------------------------------------
4. LAYOUT STRUCTURE
--------------------------------------------------------------------------------

┌─────────────────────────────────────────────────────────┐
│  HEADER (brand color background)                        │
│  [Logo]                    [Title + Subtitle aligned R] │
├─────────────────────────────────────────────────────────┤
│  SECTION HEADER (light bg, border top/bottom)           │
├─────────────────────────────────────────────────────────┤
│  CONTENT BLOCK (tables, charts, etc.)                   │
├─────────────────────────────────────────────────────────┤
│  SECTION HEADER                                         │
├─────────────────────────────────────────────────────────┤
│  CONTENT BLOCK                                          │
├─────────────────────────────────────────────────────────┤
│  FOOTER (centered, muted text)                          │
└─────────────────────────────────────────────────────────┘

--------------------------------------------------------------------------------
5. COMPLETE CSS TEMPLATE
--------------------------------------------------------------------------------

@import url('https://fonts.googleapis.com/css2?family=Open+Sans:wght@400;600;700&display=swap');

@page {
  size: A4;
  margin: 0;
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
}

.main-header {
  background: var(--brand);
  color: #fff;
  padding: 8px 15px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}

.header-logo {
  height: 40px;
  max-width: 100px;
  object-fit: contain;
}

.header-text {
  flex: 1;
  text-align: right;
}

.main-title {
  font-size: 20px;
  font-weight: 700;
  letter-spacing: 0.5px;
}

.main-subtitle {
  font-size: 12px;
  opacity: 0.9;
}

.section-header {
  background: var(--bg);
  padding: 5px 15px;
  border-bottom: 1px solid var(--tbl-border);
  border-top: 1px solid var(--tbl-border);
  font-weight: 700;
  color: var(--brand-dark);
  font-size: 13px;
  margin-top: 0;
  page-break-after: avoid;
}

.chart-container {
  padding: 5px;
  border-bottom: 1px solid var(--tbl-border);
  page-break-inside: avoid;
}

.table-wrap {
  width: 100%;
  overflow-x: auto;
  padding: 0 15px;
  margin-bottom: 8px;
  page-break-inside: avoid;
}

table {
  width: 100%;
  border-collapse: separate;
  border-spacing: 0;
  font-size: 10px;
  margin-top: 4px;
}

thead th {
  position: sticky;
  top: 0;
  z-index: 2;
  background: var(--light);
  color: var(--text);
  text-align: center;
  padding: 3px 4px;
  font-weight: 700;
  border-bottom: 2px solid var(--tbl-border);
  white-space: nowrap;
}

thead th:first-child {
  text-align: left;
}

tbody td {
  padding: 3px 4px;
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
  background-color: #fff;
  color: var(--text);
  border-bottom: 2px solid #eee;
}

tr.sub-row td {
  font-size: 9px;
  color: #777;
  background-color: #fcfcfc;
  border-bottom: 1px solid #f0f0f0;
}

tr.sub-row td:first-child {
  padding-left: 15px;
  font-weight: 400;
}

.footer {
  padding: 5px 15px;
  color: #666;
  font-size: 10px;
  border-top: 1px solid var(--tbl-border);
  background: #fff;
  text-align: center;
}

@media print {
  body {
    padding: 0;
    background: white;
  }
  .page {
    box-shadow: none;
    page-break-inside: avoid;
    break-inside: avoid;
  }
}

--------------------------------------------------------------------------------
6. HTML STRUCTURE TEMPLATE
--------------------------------------------------------------------------------

<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>Report Title</title>
  <style>
    /* INSERT CSS FROM SECTION 5 HERE */
  </style>
</head>
<body>
  <div class="page">
    
    <!-- HEADER -->
    <div class="main-header">
      <img src="data:image/png;base64,..." class="header-logo" alt="Logo" />
      <div class="header-text">
        <div class="main-title">Report Title</div>
        <div class="main-subtitle">Period: Jan 2024 - Dec 2024</div>
      </div>
    </div>
    
    <!-- SECTION: Stats Summary -->
    <div class="section-header">Resumo Estatístico</div>
    <div class="table-wrap">
      <table>
        <thead>
          <tr><th>Strategy</th><th>Total Return</th><th>Ann. Return</th><th>Sharpe</th></tr>
        </thead>
        <tbody>
          <tr><td>Portfolio</td><td>25.30%</td><td>12.50%</td><td>1.25</td></tr>
          <tr><td>Benchmark</td><td>18.20%</td><td>9.10%</td><td>0.95</td></tr>
        </tbody>
      </table>
    </div>
    
    <!-- SECTION: Chart -->
    <div class="section-header">Gráfico de Retorno Acumulado</div>
    <div class="chart-container">
      <!-- Plotly HTML embed: fig.to_html(full_html=False, include_plotlyjs='cdn') -->
    </div>
    
    <!-- SECTION: Pivot Table (Year x Month) -->
    <div class="section-header">Matriz de Performance Mensal</div>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Ano / Estratégia</th>
            <th>Jan</th><th>Fev</th><th>Mar</th><th>Abr</th><th>Mai</th><th>Jun</th>
            <th>Jul</th><th>Ago</th><th>Set</th><th>Out</th><th>Nov</th><th>Dez</th>
            <th>Acum. Ano</th>
          </tr>
        </thead>
        <tbody>
          <!-- Year Header -->
          <tr class="year-header"><td colspan="14">2024</td></tr>
          
          <!-- Main Rows (Portfolio, Benchmark) -->
          <tr class="main-row">
            <td>Portfolio</td>
            <td style="color:green">1.20%</td>
            <td style="color:red">-0.50%</td>
            <!-- ... more months ... -->
            <td><span style="color:green">12.50%</span></td>
          </tr>
          <tr class="main-row">
            <td>Benchmark (CDI)</td>
            <td>0.95%</td>
            <td>0.87%</td>
            <!-- ... more months ... -->
            <td><span style="color:green">9.10%</span></td>
          </tr>
          <tr class="main-row" style="font-style:italic; border-bottom:2px solid #ddd;">
            <td>% do Benchmark</td>
            <td>126%</td>
            <td>-57%</td>
            <!-- ... more months ... -->
            <td><b>137%</b></td>
          </tr>
          
          <!-- Sub Rows (Individual Strategies) -->
          <tr class="sub-row">
            <td>Strategy A</td>
            <td style="color:green">0.80%</td>
            <td style="color:red">-0.30%</td>
            <!-- ... more months ... -->
            <td>8.20%</td>
          </tr>
          <tr class="sub-row">
            <td>Strategy B</td>
            <td style="color:green">1.50%</td>
            <td style="color:green">0.20%</td>
            <!-- ... more months ... -->
            <td>15.30%</td>
          </tr>
        </tbody>
      </table>
    </div>
    
    <!-- FOOTER -->
    <div class="footer">
      Uso Interno - Não Compartilhar
    </div>
    
  </div>
</body>
</html>

--------------------------------------------------------------------------------
7. TABLE FORMATTING CONVENTIONS
--------------------------------------------------------------------------------

Element                 | Alignment | Style
------------------------|-----------|----------------------------------
Row labels (1st col)    | Left      | font-weight: 600
Numeric values          | Right     | Normal weight
Percentages (returns)   | Right     | Color-coded (green/red)
Headers                 | Center*   | Bold, sticky (*first col: left)
YTD/Total columns       | Right     | Bold, wrapped in <b>

Color Coding for Returns:
  Positive: <td style="color:green">1.25%</td>
  Negative: <td style="color:red">-0.50%</td>
  Neutral:  <td>105%</td>

--------------------------------------------------------------------------------
8. CHART CONFIGURATION (PLOTLY)
--------------------------------------------------------------------------------

# Chart Heights (A4 optimized)
MAIN_CHART_HEIGHT = 320      # Cumulative returns
SECONDARY_CHART_HEIGHT = 250 # Drawdown
SMALL_CHART_HEIGHT = 180     # Pie charts

# Common Layout
fig.update_layout(
    height=320,
    yaxis_title="Cumulative Return (%)",
    xaxis_title="Date",
    hovermode="x unified",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        font=dict(size=9)
    ),
    margin=dict(t=30, b=40, l=50, r=20),
)

# Line Styling
Portfolio:  line=dict(color="#825120", width=3)
Benchmark:  line=dict(color="gray", width=2, dash="dash")
Strategies: line=dict(color=CERES_COLORS[i], width=1.5)

# Hover Template
hovertemplate="%{x|%b-%Y}: %{y:.2f}%<extra>Series Name</extra>"

# Pie Chart (small, for reports)
fig_pie.update_layout(
    height=180,
    width=220,
    margin=dict(t=5, b=5, l=5, r=5),
    showlegend=False,
)

--------------------------------------------------------------------------------
9. PDF GENERATION (PLAYWRIGHT - A4)
--------------------------------------------------------------------------------

from playwright.sync_api import sync_playwright

def html_to_pdf_a4(html_content: str) -> bytes:
    """Render HTML to A4 PDF using Playwright."""
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
                "top": "10mm",
                "right": "10mm",
                "bottom": "10mm",
                "left": "10mm"
            },
        )
        browser.close()
    
    return pdf_bytes

# For single-page reports (auto-height), use:
def html_to_pdf_single_page(html_content: str) -> bytes:
    """Render HTML to single-page PDF with dynamic height."""
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

--------------------------------------------------------------------------------
10. PYTHON HELPER FUNCTIONS
--------------------------------------------------------------------------------

import html as _html
import pandas as pd

def html_escape(x) -> str:
    """Safely escape HTML entities."""
    return _html.escape("" if pd.isna(x) else str(x))

def format_pct(val, decimals=2) -> str:
    """Format value as percentage string."""
    if pd.isna(val):
        return "-"
    return f"{val * 100:.{decimals}f}%"

def format_pct_colored(val, decimals=2) -> str:
    """Format percentage with color coding."""
    if pd.isna(val):
        return "<td>-</td>"
    color = "green" if val >= 0 else "red"
    return f'<td style="color:{color}">{val * 100:.{decimals}f}%</td>'

def image_to_base64(file_path_or_bytes) -> str:
    """Convert image to base64 data URI."""
    import base64
    if isinstance(file_path_or_bytes, (str, Path)):
        with open(file_path_or_bytes, "rb") as f:
            data = f.read()
        ext = str(file_path_or_bytes).split(".")[-1].lower()
    else:
        data = file_path_or_bytes
        ext = "png"
    
    mime_map = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg", "svg": "image/svg+xml"}
    mime = mime_map.get(ext, "image/png")
    b64 = base64.b64encode(data).decode()
    return f"data:{mime};base64,{b64}"

def build_table_html(df: pd.DataFrame, first_col_left=True) -> str:
    """Convert DataFrame to HTML table string."""
    cols = list(df.columns)
    ths = "".join(f"<th>{html_escape(c)}</th>" for c in cols)
    
    rows = []
    for _, row in df.iterrows():
        tds = []
        for i, c in enumerate(cols):
            align = "left" if (i == 0 and first_col_left) else "right"
            tds.append(f'<td style="text-align:{align}">{html_escape(row[c])}</td>')
        rows.append(f"<tr>{''.join(tds)}</tr>")
    
    return f"""
    <table>
      <thead><tr>{ths}</tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
    """

--------------------------------------------------------------------------------
11. QUICK REFERENCE SUMMARY
--------------------------------------------------------------------------------

COLORS:
  Brand:     #825120 (brown)
  Dark:      #6B4219
  Light BG:  #F5F5F5
  Borders:   #E0D5CA
  Text:      #333333
  Positive:  green
  Negative:  red

FONTS:
  Family:    Open Sans, Segoe UI, Arial
  Body:      10px
  Title:     20px
  Section:   13px

DIMENSIONS (A4):
  Page:      210mm × 297mm
  Padding:   10mm or 15px
  Charts:    320px (main), 250px (secondary), 180px (small)

CSS CLASSES:
  .page           - Main container
  .main-header    - Brown header with logo/title
  .section-header - Gray section dividers
  .chart-container- Chart wrapper
  .table-wrap     - Table wrapper with padding
  .footer         - Bottom footer
  
TABLE CLASSES:
  tr.year-header  - Year separator row (colspan full)
  tr.main-row     - Primary data rows (bold)
  tr.sub-row      - Secondary/nested rows (smaller, indented)

================================================================================