import base64
import datetime
import pandas as pd
from io import BytesIO
from pages.ceres_logo import LOGO_BASE64

# Try importing playwright
try:
    from playwright.sync_api import sync_playwright
except ImportError:
    sync_playwright = None

# ---------------------- HTML/CSS Template ----------------------
# Based on report_format_guide.md

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

h1, h2, h3, h4, h5, h6 {
  page-break-after: avoid;
  break-after: avoid;
  margin-bottom: 5px; /* Ensure close proximity to content */
}

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
  padding: 0;
  font-size: 10px;
}

.page {
  width: 210mm;
  min-height: 297mm;
  margin: 0 auto;
  background: white;
  box-shadow: 0 2px 15px rgba(0,0,0,0.1);
  overflow: hidden;
  /* Padding acts as margin for content, allowing header to bleed */
  padding: 10mm; 
  page-break-after: always;
}

.page:last-child {
  page-break-after: auto;
}

.main-header {
  background: var(--brand);
  color: #fff;
  /* Top/Bottom padding, Left/Right matching page padding for alignment */
  padding: 20px 10mm;
  display: flex;
  align-items: center;
  gap: 15px;
  /* Negative margin pulls it to the very edge of the page */
  margin: -10mm -10mm 15px -10mm;
}

.header-logo {
  height: 50px;
  max-width: 120px;
  object-fit: contain;
  filter: brightness(0) invert(1);
}

.header-content {
  flex: 1;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: flex-end; /* Right align flex items */
  text-align: right;     /* Right align text */
}

.main-title {
  font-size: 16px;
  font-weight: 700;
  letter-spacing: 0.5px;
  text-transform: uppercase;
  margin-bottom: 4px;
}

.header-info {
  font-size: 9px;
  line-height: 1.2;
  opacity: 0.9;
  text-align: right;
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

.content-block {
    padding: 8px 5px;
}

.chart-container {
  padding: 5px;
  border-bottom: 1px solid var(--tbl-border);
  page-break-inside: avoid;
  text-align: center;
}

.chart-container img {
    max-width: 100%;
    height: auto;
}

.table-wrap {
  width: 100%;
  overflow-x: auto;
  padding: 0 5px;
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

.footer {
  padding: 5px 15px;
  color: #666;
  font-size: 10px;
  border-top: 1px solid var(--tbl-border);
  background: #fff;
  text-align: center;
}

/* Specific Utilities */
.status-ok { color: green; font-weight: bold; }
.status-bad { color: red; font-weight: bold; }
.small-note { font-size: 9px; color: #777; font-style: italic; margin-top: 4px; }

/* Enforce max height on images to prevent them from being too huge */
.content-block img, .chart-container img {
  max-width: 100%;
  max-height: 250px;
  height: auto;
  object-fit: contain;
  display: inline-block;
}

@media print {
  body {
    padding: 0;
    background: white;
  }
  .page {
    box-shadow: none;
    page-break-inside: avoid; /* Prevent page breaking inside container if possible */
    break-inside: avoid;
    border: none;
    margin: 0;
    padding: 10mm; /* Print padding */
  }
  .main-header {
      /* Re-apply bleed margin for print context - matches padding */
      margin: -10mm -10mm 15px -10mm;
      padding: 20px 10mm;
      -webkit-print-color-adjust: exact;
      print-color-adjust: exact;
  }
}
"""

def _img_tag(base64_str: str) -> str:
    if not base64_str or "nodata" in base64_str:
        return '<div style="color:#999;font-style:italic;padding:10px;">Sem dados visuais.</div>'
    # Check if it's already an img tag or raw base64
    if base64_str.startswith("<img"):
        return base64_str
    
    # Clean up standard matplotlib base64 if needed, though usually passthrough
    # Assuming the input is the src value or the full tag. 
    # If the input ALREADY contains 'data:image...', use it as src.
    if base64_str.startswith("data:image"):
        return f'<img src="{base64_str}" style="max-height:300px;width:auto;" />'
    
    return base64_str

def generate_formatted_report_html(
    manager_name: str,
    manager_cnpj: str,
    responsible_name: str,
    period_label: str,
    emission_date: str,
    metrics: dict,
    compliance_checks: dict,
    tables: dict,
    plots: dict,
    show_ratings: bool
) -> str:
    """
    Generates the full HTML report string using the strictly defined template.
    Structure follows the original UNFORMATTED report exactly, but with new CSS.
    """
    
    # --- Prepare Header ---
    logo_src = LOGO_BASE64 if LOGO_BASE64.startswith("data:") else f"data:image/png;base64,{LOGO_BASE64}"
    
    html = f"""
    <!DOCTYPE html>
    <html lang="pt-BR">
    <head>
      <meta charset="UTF-8" />
      <title>Relatório de Monitoramento de Risco de Crédito</title>
      <style>{CSS_STYLES}</style>
    </head>
    <body>
      <div class="page">
        <!-- HEADER -->
        <div class="main-header">
          <img src="{logo_src}" class="header-logo" alt="Logo" />
          <div class="header-content">
            <div class="main-title">RELATÓRIO DE MONITORAMENTO DE RISCO DE CRÉDITO</div>
            <div class="header-info">
                 {manager_name} | CNPJ: {manager_cnpj}<br/>
                 Data de Emissão: {emission_date}<br/>
                 Período de Referência: {period_label}<br/>
                 Responsável: {responsible_name} – Diretor de Risco
            </div>
          </div>
        </div>
        
        <!-- 1. OBJETIVO -->
        <div class="section-header">1. OBJETIVO</div>
        <div class="content-block">
            Este relatório apresenta a avaliação e o monitoramento do risco de crédito
            das carteiras administradas pela {manager_name}, em conformidade com as
            diretrizes da CVM, Bacen e melhores práticas.
        </div>

        <!-- 2. METODOLOGIA -->
        <div class="section-header">2. METODOLOGIA</div>
        <div class="content-block">
            <ul>
              <li>Comitê de Investimentos: limites e estratégias por emissor e ativo.</li>
              <li>Rating: uso de ratings de agências quando disponível.</li>
              <li>Limites de Exposição: por emissor e por top 10 emissores.</li>
              <li>Monitoramento Contínuo: posições e eventos relevantes.</li>
              <li>Stress Testing: cenários adversos (liquidez e recessão).</li>
            </ul>
        </div>

        <!-- 3. EXPOSIÇÃO AO RISCO DE CRÉDITO -->
        <div class="section-header">3. EXPOSIÇÃO AO RISCO DE CRÉDITO</div>
        <div class="content-block">
            Total da carteira (PL): R$ {metrics['pl_total']}<br/>
            PL em Renda Fixa (Crédito + Tesouro): R$ {metrics['base_total']}<br/>
            Total de crédito: R$ {metrics['pl_credit']}
            ({metrics['credit_share_base']} do PL em Renda Fixa; {metrics['credit_share_pl']} do PL)
        </div>
    """

    # 3.1 Ratings
    if show_ratings:
        html += f"""
        <div class="content-block">
            <h3>3.1 Exposição por Classe e Rating</h3>
            <ul>
              <li>
                Limite por Rating: mínimo {metrics['min_aaa_pct']} do PL em Renda Fixa em Soberano/AAA.
              </li>
              <li>
                Exposição Atual Soberano/AAA:
                {metrics['aaa_share_base_pct']} do PL em Renda Fixa ({metrics['aaa_share_pl_pct']} do PL)
                <span class="{'status-ok' if compliance_checks['ok_aaa'] else 'status-bad'}">
                  ({compliance_checks['conf_aaa']})
                </span>
              </li>
            </ul>
            <div class="small-note">
              Nota: Ratings vêm da coluna 'rating' no banco de dados. Soberano = Tesouro Nacional.
              Valores "NONE" são tratados como "Sem Rating".
            </div>

            <h4>3.1.A Análise de Crédito Privado – BRL (CRI + CRA + DEBENTURE)</h4>
            <div class="small-note">
              Total CRI+CRA+DEBENTURE (BRL): R$ {metrics['brl_private_total']}
            </div>
            <div class="chart-container">{_img_tag(plots.get('ratings_brl', ''))}</div>

            <h4>3.1.B Análise de Crédito Privado – USD (Corporate)</h4>
            <div class="small-note">
                Total Corporate (USD): $ {metrics['usd_corporate_total']}
            </div>
            <div class="chart-container">{_img_tag(plots.get('ratings_usd', ''))}</div>
        </div>
        """
    else:
        html += """
        <div class="content-block">
            <h3>3.1 Exposição por Classe</h3>
            <div class="small-note">Seção de Rating desativada.</div>
        </div>
        """

    html += f"""
        <div class="content-block">
            <h4>3.1.C Análise de Crédito Bancário</h4>
            <div class="small-note">
              Total Crédito Bancário: R$ {metrics['bank_credit_total']}
            </div>
            <div class="chart-container">{_img_tag(plots.get('sov_s1', ''))}</div>
        </div>
    """

    # 3.2 Limites
    html += f"""
        <div class="content-block">
            <h3>3.2 Limites de Exposição</h3>
            <ul>
              <li>
                Máximo {metrics['max_single_pct']} do PL em Renda Fixa em emissor único
                (Tesouro, Instituições S1 e S2 excluídos). Atual: {metrics['largest_share_base']}
                do PL em Renda Fixa ({metrics['largest_share_pl']} do PL)
                <span class="{'status-ok' if compliance_checks['ok_single'] else 'status-bad'}">
                  ({compliance_checks['conf_single']})
                </span>
              </li>
              <li>
                Máximo {metrics['max_top10_pct']} do PL em Renda Fixa nos 10 maiores emissores
                (Tesouro, Instituições S1 e S2 excluídos). Atual: {metrics['top10_share_base']}
                do PL em Renda Fixa ({metrics['top10_share_pl']} do PL)
                <span class="{'status-ok' if compliance_checks['ok_top10'] else 'status-bad'}">
                  ({compliance_checks['conf_top10']})
                </span>
              </li>
            </ul>

            <h4>3.2.A BRL</h4>
            <div class="small-note">
                Emissor único (PL em Renda Fixa BRL): {metrics['max_single_pct']} — Atual:
                {metrics['largest_share_base_brl']}
                <span class="{'status-ok' if compliance_checks['ok_single_brl'] else 'status-bad'}">
                  ({compliance_checks['conf_single_brl']})
                </span>
                <br/>
                Top-10 (PL em Renda Fixa BRL): {metrics['max_top10_pct']} — Atual:
                {metrics['top10_share_base_brl']}
                <span class="{'status-ok' if compliance_checks['ok_top10_brl'] else 'status-bad'}">
                  ({compliance_checks['conf_top10_brl']})
                </span>
            </div>
            {tables.get('issuers_brl', '<div class="small-note">Sem dados no PL em Renda Fixa BRL.</div>')}
            
            <h4>3.2.B USD</h4>
             <div class="small-note">
                Emissor único (PL em Renda Fixa USD): {metrics['max_single_pct']} — Atual:
                {metrics['largest_share_base_usd']}
                <span class="{'status-ok' if compliance_checks['ok_single_usd'] else 'status-bad'}">
                  ({compliance_checks['conf_single_usd']})
                </span>
                <br/>
                Top-10 (PL em Renda Fixa USD): {metrics['max_top10_pct']} — Atual:
                {metrics['top10_share_base_usd']}
                <span class="{'status-ok' if compliance_checks['ok_top10_usd'] else 'status-bad'}">
                  ({compliance_checks['conf_top10_usd']})
                </span>
            </div>
            {tables.get('issuers_usd', '<div class="small-note">Sem dados no PL em Renda Fixa USD.</div>')}
        </div>
        
        <!-- 3.3 FATOR DE RISCO -->
        <div class="content-block">
            <h3>3.3 Concentração por Fator de Risco</h3>
            <div style="text-align:center;">
                 <div style="display:inline-block; width:45%; vertical-align:top;">{_img_tag(plots.get('factor_brl', ''))}</div>
                 <div style="display:inline-block; width:45%; vertical-align:top;">{_img_tag(plots.get('factor_usd', ''))}</div>
            </div>
        </div>

        <!-- 3.4 VENCIMENTO -->
        <div class="content-block">
            <h3>3.4 Concentração por Vencimento</h3>
            <div style="text-align:center;">{_img_tag(plots.get('maturity_base', ''))}</div>
            <div style="margin-top:6px;">
              Limite: máximo {metrics['max_over5y_pct']} do PL em Renda Fixa em vencimentos
              &gt; 5 anos. Atual: {metrics['over5y_pct_base']} do PL em Renda Fixa
              ({metrics['over5y_pct_pl']} do PL)
              <span class="{'status-ok' if compliance_checks['ok_over5y'] else 'status-bad'}">
                ({compliance_checks['conf_over5y']})
              </span>
            </div>
        </div>
        
        <!-- 3.5 PAIS -->
        <div class="content-block">
            <h3>3.5 Exposição por País de Risco</h3>
            {tables.get('country_table', '<div class="small-note">Sem dados para exposição por país.</div>')}
        </div>

        <!-- 3.6 MOEDA -->
        <div class="content-block">
            <h3>3.6 Exposição Cambial</h3>
            <div class="small-note">
              PL em Renda Fixa = Crédito + Tesouro. Fundos não entram no PL em Renda Fixa.
            </div>
            <div style="text-align:center;">{_img_tag(plots.get('currency_base', ''))}</div>
        </div>

        <!-- 3.7 TIPO -->
        <div class="content-block">
            <h3>3.7 Exposição por Tipo de Título</h3>
            <div style="text-align:center;">
                 <div style="display:inline-block; width:45%; vertical-align:top;">{_img_tag(plots.get('bondtype_brl', ''))}</div>
                 <div style="display:inline-block; width:45%; vertical-align:top;">{_img_tag(plots.get('bondtype_usd', ''))}</div>
            </div>
        </div>
        
        <!-- 3.8 INADIMPLENCIA -->
        <div class="content-block">
            <h3>3.8 Inadimplência</h3>
            Índice de Inadimplência no período: {metrics['default_rate']}
        </div>
        
        <!-- 4. EVENTOS -->
        <div class="section-header">4. MONITORAMENTO DE EVENTOS RELEVANTES</div>
        <div class="content-block">
            {tables.get('events_text', '<div></div>')}
        </div>
        
        <!-- 5. CONFORMIDADE -->
        <div class="section-header">5. CONFORMIDADE REGULATÓRIA</div>
        <div class="content-block">
            <ul>
              <li>CVM: Instrução CVM nº 555/2014; Resolução CVM nº 50/21.</li>
              <li>Bacen: Circular nº 3.930/2019.</li>
              <li>ANBIMA: Código de Regulação e Melhores Práticas.</li>
            </ul>
        </div>

        <!-- 6. CONCLUSOES -->
        <div class="section-header">6. CONCLUSÕES</div>
        <div class="content-block">
            <ul>
              <li>Os níveis de risco de crédito são compatíveis com o perfil estabelecido.</li>
              <li>Não foram identificadas exposições que comprometam a estabilidade financeira das carteiras.</li>
              <li>As políticas de risco são seguidas com monitoramento contínuo.</li>
            </ul>
        </div>
        
        <!-- 7. RECOMENDACOES -->
        <div class="section-header">7. RECOMENDAÇÕES</div>
        <div class="content-block">
            <ul>
              <li>Revisões trimestrais das métricas e limites.</li>
              <li>Treinamentos periódicos da equipe de risco.</li>
              <li>Atualização anual dos cenários de stress.</li>
            </ul>
        </div>
        
        <div class="content-block" style="margin-top:20px;">
            Assinatura: {responsible_name} – Diretor de Risco
        </div>
        
        <div class="content-block" style="border-top:1px solid #ccc; margin-top:15px; padding-top:10px;">
            <h3>Anexos</h3>
            <div class="small-note">
              Figuras geradas no app (por classe, por rating, por emissor, por fator,
              por vencimento, por país, por moeda e por tipo) podem ser exportadas
              separadamente.
            </div>
        </div>
        
      </div>
    </body>
    </html>
    """
    
    return html



def html_to_pdf_formatted(html_content: str) -> bytes:
    """
    Renders HTML to PDF bytes using Playwright (A4).
    """
    if sync_playwright is None:
        raise ImportError("Playwright is not installed.")

    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--no-sandbox", "--disable-setuid-sandbox"])
        page = browser.new_page()
        
        # Set content
        page.set_content(html_content, wait_until="load")
        
        # We enforce A4 via PDF options
        pdf_bytes = page.pdf(
            format="A4",
            print_background=True,
            # We use CSS margins/padding for content, so PDF margin is minimal
            margin={
                "top": "0",
                "right": "0",
                "bottom": "0",
                "left": "0"
            }
        )
        browser.close()
        
    return pdf_bytes
