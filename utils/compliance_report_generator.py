import base64
import pandas as pd
import datetime
from pages.ceres_logo import LOGO_BASE64

# Try importing playwright, but don't crash if missing (though report generation will fail)
try:
    from playwright.sync_api import sync_playwright
except ImportError:
    sync_playwright = None

# ---------------------- HTML/CSS Template ----------------------
# Logic adapted from report_format_guide.md

CSS_STYLES = """
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
  padding: 20px;
  font-size: 11px; /* Slightly larger base for readability */
}

.page {
  width: 210mm;
  min-height: 297mm;
  margin: 0 auto;
  background: white;
  box-shadow: 0 2px 15px rgba(0,0,0,0.1);
  overflow: hidden;
  border: 1px solid var(--tbl-border);
  padding-bottom: 30px;
}

.main-header {
  background: var(--brand);
  color: #fff;
  padding: 15px 25px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}

.header-logo {
  height: 50px;
  max-width: 150px;
  object-fit: contain;
  filter: brightness(0) invert(1); /* Make logo white if transparent */
}

.header-text {
  flex: 1;
  text-align: right;
}

.main-title {
  font-size: 22px;
  font-weight: 700;
  letter-spacing: 0.5px;
  text-transform: uppercase;
}

.main-subtitle {
  font-size: 12px;
  opacity: 0.9;
  margin-top: 4px;
}

.section-header {
  background: var(--bg);
  padding: 8px 25px;
  border-bottom: 1px solid var(--tbl-border);
  border-top: 1px solid var(--tbl-border);
  font-weight: 700;
  color: var(--brand-dark);
  font-size: 14px;
  margin-top: 15px;
  page-break-after: avoid;
  text-transform: uppercase;
}

.content-block {
    padding: 15px 25px;
}

.info-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
    margin-bottom: 15px;
}

.info-item {
    font-size: 12px;
}

.info-label {
    font-weight: 600;
    color: #666;
}

.status-box {
    padding: 10px;
    border-radius: 4px;
    font-weight: bold;
    text-align: center;
    margin-top: 10px;
    margin-bottom: 10px;
}

.status-ok {
    background-color: #d4edda;
    color: #155724;
    border: 1px solid #c3e6cb;
}

.status-warning {
    background-color: #fff3cd;
    color: #856404;
    border: 1px solid #ffeeba;
}

.violation-box {
    border-left: 4px solid #dc3545;
    background-color: #fff5f5;
    padding: 10px;
    margin-bottom: 8px;
}

.violation-title {
    font-weight: 700;
    color: #dc3545;
    margin-bottom: 4px;
}

.table-wrap {
  width: 100%;
  overflow-x: auto;
  margin-top: 5px;
}

table {
  width: 100%;
  border-collapse: collapse;
  font-size: 11px;
}

thead th {
  background: var(--light);
  color: var(--text);
  text-align: right;
  padding: 6px 8px;
  font-weight: 700;
  border-bottom: 2px solid var(--tbl-border);
}

thead th:first-child {
  text-align: left;
}

tbody td {
  padding: 6px 8px;
  border-bottom: 1px solid #eee;
  text-align: right;
  color: #444;
}

tbody td:first-child {
  text-align: left;
  font-weight: 600;
  color: var(--text);
}

tbody tr:last-child td {
    border-bottom: none;
}

.footer {
  padding: 15px 25px;
  color: #999;
  font-size: 9px;
  border-top: 1px solid var(--tbl-border);
  background: #fff;
  text-align: center;
  margin-top: auto;
}

@media print {
  body {
    padding: 0;
    background: white;
  }
  .page {
    box-shadow: none;
    border: none;
    width: 100%;
    margin: 0;
  }
}
"""

def generate_compliance_html(client_name, portfolio_id, total_equity, compliance_data, profile):
    """
    Generates an HTML report string for Client Compliance.
    """
    
    # 1. Prepare Data
    report_date = datetime.datetime.now().strftime("%d/%m/%Y")
    
    has_issues = compliance_data.get('has_issues', False)
    allocation_violations = compliance_data.get('allocation_violations', [])
    exception_violations = compliance_data.get('exception_violations', [])
    df_comparison = compliance_data.get('df_comparison', pd.DataFrame())

    # Status Box Logic
    if has_issues:
        status_class = "status-warning"
        status_text = "⚠️ PERFIL EM DESENQUADRAMENTO"
        if allocation_violations and exception_violations:
            status_desc = "Foram identificados desvios de alocação e violações de regras restritivas."
        elif allocation_violations:
            status_desc = "Foram identificados desvios nas faixas de alocação por classe de ativo."
        else:
            status_desc = "Foram identificadas violações em regras específicas (exceções)."
    else:
        status_class = "status-ok"
        status_text = "✅ PERFIL ADEQUADO"
        status_desc = "A carteira está em conformidade com os parâmetros estabelecidos."

    # Logo (ensure prefix)
    logo_src = LOGO_BASE64 if LOGO_BASE64.startswith("data:") else f"data:image/png;base64,{LOGO_BASE64}"

    # 2. Build HTML Content
    html_parts = []
    
    html_parts.append(f"""
    <!DOCTYPE html>
    <html lang="pt-BR">
    <head>
      <meta charset="UTF-8" />
      <title>Relatório de Adequação - {client_name}</title>
      <style>{CSS_STYLES}</style>
    </head>
    <body>
      <div class="page">
        <!-- HEADER -->
        <div class="main-header">
          <img src="{logo_src}" class="header-logo" alt="Logo Ceres" />
          <div class="header-text">
            <div class="main-title">Relatório de Adequação</div>
            <div class="main-subtitle">Data: {report_date}</div>
          </div>
        </div>

        <!-- INFO BLOCK -->
        <div class="content-block">
            <div class="info-grid">
                <div class="info-item"><span class="info-label">Cliente:</span> {client_name}</div>
                <div class="info-item"><span class="info-label">ID Portfólio:</span> {portfolio_id}</div>
                <div class="info-item"><span class="info-label">Perfil de Risco:</span> {profile.name}</div>
                <div class="info-item"><span class="info-label">Patrimônio Total:</span> R$ {total_equity:,.2f}</div>
            </div>
            
            <div class="status-box {status_class}">
                {status_text}
                <div style="font-weight: normal; font-size: 11px; margin-top: 4px;">{status_desc}</div>
            </div>
        </div>
    """)

    # 3. Violations Section (if any)
    if has_issues:
        html_parts.append('<div class="section-header">Pontos de Atenção</div>')
        html_parts.append('<div class="content-block">')
        
        if allocation_violations:
            html_parts.append('<div class="violation-box">')
            html_parts.append('<div class="violation-title">Desvios de Alocação</div>')
            html_parts.append('<ul>')
            for v in allocation_violations:
                html_parts.append(f'<li>{v}</li>')
            html_parts.append('</ul></div>')

        if exception_violations:
            html_parts.append('<div class="violation-box">')
            html_parts.append('<div class="violation-title">Violações de Regras Específicas</div>')
            html_parts.append('<ul>')
            for v in exception_violations:
                html_parts.append(f'<li>{v}</li>')
            html_parts.append('</ul></div>')
            
        html_parts.append('</div>')

    # 4. Allocation Table Section
    html_parts.append('<div class="section-header">Análise de Alocação</div>')
    html_parts.append('<div class="content-block">')
    
    if not df_comparison.empty:
        # Prepare Rows
        rows_html = ""
        for _, row in df_comparison.iterrows():
            asset_class = row.get('Classe de Ativos', '')
            target = row.get('Alvo (%)', 0)
            current = row.get('Atual (%)', 0)
            min_val = row.get('Min (%)', 0)
            max_val = row.get('Max (%)', 0)
            
            # Formatting
            diff = current - target
            
            # Status check for row coloring (optional, or just text color)
            is_out = not (min_val <= current <= max_val)
            pct_style = 'color: #dc3545; font-weight: bold;' if is_out else 'color: #155724;'
            
            rows_html += f"""
            <tr>
                <td>{asset_class}</td>
                <td>{min_val:.1f}%</td>
                <td>{target:.1f}%</td>
                <td>{max_val:.1f}%</td>
                <td style="{pct_style}">{current:.2f}%</td>
                <td>{diff:+.2f}%</td>
            </tr>
            """
            
        html_parts.append(f"""
        <div class="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Classe de Ativos</th>
                <th>Mínimo</th>
                <th>Alvo</th>
                <th>Máximo</th>
                <th>Atual</th>
                <th>Desvio</th>
              </tr>
            </thead>
            <tbody>
              {rows_html}
            </tbody>
          </table>
        </div>
        """)
    else:
        html_parts.append("<p>Nenhum dado de alocação disponível.</p>")
    
    html_parts.append('</div>')
    
    # Footer
    html_parts.append(f"""
        <div class="footer">
          Gerado em {datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")} • Ceres Wealth
        </div>
      </div>
    </body>
    </html>
    """)
    
    return "".join(html_parts)


def html_to_pdf(html_content: str) -> bytes:
    """
    Renders HTML to PDF bytes using Playwright.
    Must be called within a supported environment.
    """
    if sync_playwright is None:
        raise ImportError("Playwright is not installed.")

    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--no-sandbox", "--disable-setuid-sandbox"])
        page = browser.new_page()
        
        # Set content
        page.set_content(html_content, wait_until="load")
        
        # Simple wait to ensure rendering
        # In a real scenario, might want to wait for fonts or specific elements
        
        pdf_bytes = page.pdf(
            format="A4",
            print_background=True,
            margin={
                "top": "0", # Handled by CSS
                "right": "0",
                "bottom": "0",
                "left": "0"
            }
        )
        browser.close()
        
    return pdf_bytes
