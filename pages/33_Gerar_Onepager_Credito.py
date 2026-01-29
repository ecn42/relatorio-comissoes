import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
import plotly.io as pio
import os
import base64
import zipfile
import io
from pages.ceres_logo import LOGO_BASE64

# Try to import Playwright
try:
    from playwright.sync_api import sync_playwright
    HAS_PLAYWRIGHT = True
except ImportError:
    HAS_PLAYWRIGHT = False

# --- Configuration ---
st.set_page_config(page_title="Relatório de Crédito", layout="wide")
DB_PATH = "databases/onepager_credito.db"

# --- Styling Constants ---
BRAND_BROWN = "#825120"
BRAND_BROWN_DARK = "#6B4219"
LIGHT_GRAY = "#F5F5F5"
CERES_COLORS = ["#013220", "#57575a", "#b08568", "#09202e", "#582308", "#7a6200"]

# --- Custom CSS for Streamlit UI ---
st.markdown(f"""
    <style>
    .main-header {{
        background-color: {BRAND_BROWN};
        color: white;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }}
    .main-title {{
        font-family: 'Century Gothic', sans-serif;
        font-size: 24px;
        font-weight: bold;
    }}
    .sub-title {{
        font-family: 'Century Gothic', sans-serif;
        font-size: 14px;
        opacity: 0.9;
    }}
    .section-header {{
        background-color: {LIGHT_GRAY};
        color: {BRAND_BROWN_DARK};
        padding: 8px 12px;
        font-weight: 700;
        border-bottom: 2px solid #E0D5CA;
        margin-top: 20px;
        margin-bottom: 10px;
        font-family: 'Segoe UI', sans-serif;
    }}
    .metric-table {{
        width: 100%;
        border-collapse: collapse;
        font-size: 14px;
    }}
    .metric-table td {{
        padding: 6px;
        border-bottom: 1px solid #f0f0f0;
    }}
    .metric-label {{
        font-weight: 600;
        color: #555;
        width: 40%;
    }}
    .metric-value {{
        text-align: right;
        font-weight: 700;
        color: #333;
    }}
    </style>
""", unsafe_allow_html=True)

# --- Helpers ---

def clean_series_data(df, col_name):
    """Robust cleaning: Coerce numeric, drop NaNs/Zeros, sort by Data."""
    if col_name not in df.columns:
        return pd.DataFrame()
    df = df.copy()
    df[col_name] = pd.to_numeric(df[col_name], errors='coerce')
    df = df.dropna(subset=[col_name, "Data"])
    df = df[df[col_name] != 0]
    return df.sort_values("Data")

def safe_format_date(val):
    if pd.isna(val) or str(val).strip() in ["", "-", "0", "NaT"]:
        return "-"
    try:
        dt = pd.to_datetime(val, errors='coerce', dayfirst=True)
        if pd.isna(dt):
            return str(val)
        return dt.strftime('%d/%m/%Y')
    except:
        return str(val)

def format_currency_brl(val):
    if pd.isna(val) or val == "":
        return "-"
    try:
        return f"R$ {float(val):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except:
        return str(val)

def format_percent(val):
    if pd.isna(val) or val == "":
        return "-"
    try:
        return f"{float(val):.2f}%"
    except:
        return str(val)

def filter_data(data):
    """Remove fields with empty/placeholder values."""
    clean = {}
    for k, v in data.items():
        s_val = str(v).strip()
        if s_val not in ["-", "", "N/A", "nan", "None", "NaT"]:
            clean[k] = v
    return clean

# --- Database & Data Loading ---

def get_connection():
    if not os.path.exists(DB_PATH):
        return None
    return sqlite3.connect(DB_PATH)

def load_main_df(conn):
    try:
        main_df = pd.read_sql_query("SELECT * FROM main_cred", conn)
        main_df.columns = [" ".join(str(c).split()) for c in main_df.columns]
        return main_df
    except Exception:
        return pd.DataFrame()

# --- Core Generation Logic (Reusable) ---

def get_asset_metadata(df_main, asset_code):
    """Retrieve and format metadata for a single asset."""
    subset = df_main[df_main["Código"] == asset_code]
    if subset.empty:
        return None, {}, {}
    
    asset_meta = subset.iloc[0]
    
    raw_info_main = {
        "Código": asset_code,
        "Nome da Emissão": asset_meta.get("Nome da Emissao", "N/A"),
        "Classe": asset_meta.get("Classe", "N/A"),
        "Emissor": asset_meta.get("Nome", "N/A"),
        "CNPJ": asset_meta.get("CNPJ", "N/A"),
        "Espécie": asset_meta.get("Espécie", "N/A"),
        "Setor": asset_meta.get("Setor NAICS ult disponiv", "N/A"),
        "Rating": asset_meta.get("Classificação de Risco Atual", "N/A"),
        "Indexador": asset_meta.get("Índice de correção", "N/A"),
    }

    raw_info_details = {
        "Emissão": safe_format_date(asset_meta.get("Data de emissão")),
        "Vencimento": safe_format_date(asset_meta.get("Vencimento/Repactuação")),
        "Remuneração": str(asset_meta.get("Remuneração", "-")),
        "Duration": str(asset_meta.get("Duration (DU) 27Jan26", "-")).split('.')[0],
        "Taxa YTM": format_percent(asset_meta.get("Taxa (YTM) em %", 0)),
        "PU Atual": format_currency_brl(asset_meta.get("PU em R$", 0)),
    }
    
    return asset_meta, filter_data(raw_info_main), filter_data(raw_info_details)

def generate_asset_charts_html(conn, asset_code):
    """Generate HTML strings for the 3 charts."""
    chart_configs = [
        ("YTM", "Evolução da Taxa (YTM)", "Taxa (%)", 0),
        ("PU", "Evolução do Preço Unitário (PU)", "R$", 1),
        ("PU_Percent", "Evolução % do PU Par", "% Par", 2)
    ]
    
    figs_html = []
    
    for table, title, ylabel, color_idx in chart_configs:
        try:
            query = f'SELECT Data, "{asset_code}" FROM {table}'
            df_chart = pd.read_sql_query(query, conn)
            df_chart["Data"] = pd.to_datetime(df_chart["Data"])
            
            df_clean = clean_series_data(df_chart, asset_code)
            
            if not df_clean.empty:
                fig = px.line(df_clean, x="Data", y=asset_code)
                fig.update_layout(
                    title=dict(text=title, font=dict(size=14)),
                    template="plotly_white",
                    height=280,
                    margin=dict(l=40, r=40, t=40, b=40),
                    yaxis_title=ylabel,
                    xaxis_title=None,
                    hovermode="x unified"
                )
                fig.update_traces(line=dict(color=CERES_COLORS[color_idx], width=2))
                figs_html.append(pio.to_html(fig, full_html=False, include_plotlyjs='cdn'))
            else:
                figs_html.append("") # Empty if no data
        except:
            figs_html.append("") # Empty on error
            
    return figs_html

def create_report_html(asset_meta, info_main, info_details, figs_html):
    """Builds the full HTML string."""
    style_css = f"""
    @import url('https://fonts.googleapis.com/css2?family=Open+Sans:wght@400;600;700&display=swap');
    @page {{ size: A4; margin: 0; }}
    html, body {{ margin: 0; padding: 0; -webkit-print-color-adjust: exact; }}
    body {{ font-family: 'Open Sans', sans-serif; background: white; color: #333; font-size: 10px; }}
    .page {{ width: 210mm; min-height: 297mm; margin: 0 auto; background: white; overflow: hidden; position: relative; }}
    .main-header {{ background: {BRAND_BROWN}; color: white; padding: 10px 20px; display: flex; align-items: center; justify-content: space-between; }}
    .header-logo {{ height: 40px; object-fit: contain; filter: brightness(0) invert(1); }}
    .header-text {{ text-align: right; }}
    .main-title {{ font-size: 20px; font-weight: 700; }}
    .main-subtitle {{ font-size: 11px; opacity: 0.95; }}
    .section-header {{ background: #F8F8F8; padding: 5px 15px; border-bottom: 1px solid #E0D5CA; border-top: 1px solid #E0D5CA; font-weight: 700; color: {BRAND_BROWN_DARK}; font-size: 12px; margin-top: 15px; }}
    .tables-container {{ display: flex; gap: 20px; padding: 10px 15px; }}
    .info-table {{ width: 100%; border-collapse: collapse; font-size: 10px; }}
    .info-table td {{ padding: 4px; border-bottom: 1px solid #eee; }}
    .info-table td:first-child {{ font-weight: 600; color: #555; width: 45%; }}
    .info-table td:last-child {{ text-align: right; font-weight: 700; color: #000; }}
    .chart-box {{ padding: 10px 15px; page-break-inside: avoid; }}
    .chart-title {{ font-size: 11px; font-weight: 700; margin-bottom: 5px; color: #333; }}
    .footer {{ position: absolute; bottom: 0; width: 100%; text-align: center; font-size: 9px; color: #999; padding: 10px; border-top: 1px solid #eee; }}
    """
    
    header_html = f"""
    <div class="main-header">
        <img src="data:image/png;base64,{LOGO_BASE64}" class="header-logo" style="filter: none;">
        <div class="header-text">
            <div class="main-title">Relatório de Crédito</div>
            <div class="main-subtitle">{asset_meta.get('Nome', '')} | {asset_meta.get('Código', '')}</div>
        </div>
    </div>
    """
    
    def make_table_html(data):
        rows = ""
        for k, v in data.items():
            rows += f"<tr><td>{k}</td><td>{v}</td></tr>"
        return f"<table class='info-table'>{rows}</table>"
    
    meta_html = f"""
    <div class="tables-container">
        <div style="flex: 1;">
            <div style="font-weight:700; color:{BRAND_BROWN}; margin-bottom:5px; font-size:11px;">Informações Principais</div>
            {make_table_html(info_main)}
        </div>
        <div style="flex: 1;">
            <div style="font-weight:700; color:{BRAND_BROWN}; margin-bottom:5px; font-size:11px;">Detalhes da Emissão</div>
            {make_table_html(info_details)}
        </div>
    </div>
    """
    
    charts_section = ""
    titles = ["Evolução da Taxa (YTM)", "Evolução do Preço Unitário (PU)", "Evolução % do PU Par"]
    for i, fig_html in enumerate(figs_html):
        if fig_html:
            charts_section += f"""
            <div class="chart-box">
                <div class="chart-title">{titles[i]}</div>
                {fig_html}
            </div>
            """
    
    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head><style>{style_css}</style></head>
    <body>
        <div class="page">
            {header_html}
            {meta_html}
            <div class="section-header">Análise Gráfica - Histórico</div>
            {charts_section}
            <div class="footer">Gerado via Ceres Wealth System | {pd.Timestamp.now().strftime('%d/%m/%Y %H:%M')}</div>
        </div>
    </body>
    </html>
    """
    return full_html

def generate_pdf_from_html(html_content):
    if not HAS_PLAYWRIGHT: return None
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.set_viewport_size({"width": 794, "height": 1123})
        page.set_content(html_content, wait_until="networkidle")
        pdf_bytes = page.pdf(format="A4", print_background=True, margin={"top": "10mm", "right": "10mm", "bottom": "10mm", "left": "10mm"})
        browser.close()
        return pdf_bytes

# --- Main App ---

if not st.session_state.get("authenticated", False):
    st.warning("Please enter the password on the Home page first.")
    st.stop()

conn = get_connection()
if not conn:
    st.error(f"Banco de dados não encontrado em {DB_PATH}.")
    st.stop()

df_main = load_main_df(conn)
if "Código" not in df_main.columns:
    st.error("Coluna 'Código' não encontrada.")
    st.stop()

asset_list = sorted(df_main["Código"].dropna().astype(str).unique())

with st.sidebar:
    st.markdown("### 🔍 Visualização Individual")
    selected_asset = st.selectbox("Ativo", asset_list)
    st.markdown("---")
    
    st.markdown("### 📦 Geração em Lote")
    with st.expander("Baixar Todos (ZIP)"):
        st.info(f"Total de ativos: {len(asset_list)}")
        # Default to HTML since PDF is slow logic
        fmt_option = st.radio("Formato", ["HTML", "PDF"], index=0)
        
        if st.button("Gerar ZIP com Todos"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            zip_buffer = io.BytesIO()
            
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                for idx, asset_code in enumerate(asset_list):
                    status_text.text(f"Processando {idx+1}/{len(asset_list)}: {asset_code}...")
                    
                    # Generate content
                    am, im, details = get_asset_metadata(df_main, asset_code)
                    if am is None: continue
                    figs = generate_asset_charts_html(conn, asset_code)
                    html_content = create_report_html(am, im, details, figs)
                    
                    if fmt_option == "HTML":
                        zf.writestr(f"{asset_code}.html", html_content)
                    else:
                        pdf_data = generate_pdf_from_html(html_content)
                        if pdf_data:
                            zf.writestr(f"{asset_code}.pdf", pdf_data)
                    
                    progress_bar.progress((idx + 1) / len(asset_list))
            
            progress_bar.empty()
            status_text.success("Geração concluída!")
            st.download_button(
                label="⬇️ Baixar ZIP",
                data=zip_buffer.getvalue(),
                file_name=f"OnePagers_{fmt_option}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.zip",
                mime="application/zip"
            )

# --- Render Individual View ---

asset_meta, info_main, info_details = get_asset_metadata(df_main, selected_asset)

# Header
st.markdown(f"""
    <div class="main-header">
        <div>
            <div class="main-title">Relatório de Crédito</div>
            <div class="sub-title">{asset_meta.get('Nome', 'N/A')} | {selected_asset}</div>
        </div>
        <img src="data:image/png;base64,{LOGO_BASE64}" width="120">
    </div>
""", unsafe_allow_html=True)

# Metadata
if info_main or info_details:
    c1, c2 = st.columns(2)
    with c1:
        if info_main:
            st.markdown('<div class="section-header">Informações Principais</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <table class="metric-table">
                {''.join([f'<tr><td class="metric-label">{k}</td><td class="metric-value">{v}</td></tr>' for k,v in info_main.items()])}
            </table>
            """, unsafe_allow_html=True)

    with c2:
        if info_details:
            st.markdown('<div class="section-header">Detalhes da Emissão</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <table class="metric-table">
                {''.join([f'<tr><td class="metric-label">{k}</td><td class="metric-value">{v}</td></tr>' for k,v in info_details.items()])}
            </table>
            """, unsafe_allow_html=True)

# Charts for UI
st.markdown('<div class="section-header">Análise Gráfica</div>', unsafe_allow_html=True)

chart_configs = [
    ("YTM", "Evolução da Taxa (YTM)", "Taxa (%)", 0),
    ("PU", "Evolução do Preço Unitário (PU)", "R$", 1),
    ("PU_Percent", "Evolução % do PU Par", "% Par", 2)
]

figs_html_for_single_export = []

for table, title, ylabel, color_idx in chart_configs:
    try:
        query = f'SELECT Data, "{selected_asset}" FROM {table}'
        df_chart = pd.read_sql_query(query, conn)
        df_chart["Data"] = pd.to_datetime(df_chart["Data"])
        df_clean = clean_series_data(df_chart, selected_asset)
        
        if not df_clean.empty:
            fig = px.line(df_clean, x="Data", y=selected_asset)
            fig.update_layout(
                title=dict(text=title, font=dict(size=14)),
                template="plotly_white",
                height=280,
                margin=dict(l=40, r=40, t=40, b=40),
                yaxis_title=ylabel,
                xaxis_title=None,
                hovermode="x unified"
            )
            fig.update_traces(line=dict(color=CERES_COLORS[color_idx], width=2))
            st.plotly_chart(fig, use_container_width=True)
            figs_html_for_single_export.append(pio.to_html(fig, full_html=False, include_plotlyjs='cdn'))
        else:
            figs_html_for_single_export.append("")
            st.info(f"Dados insuficientes para {title}.")
    except:
        figs_html_for_single_export.append("")

conn.close()

# --- Single ExportButtons ---

single_html = create_report_html(asset_meta, info_main, info_details, figs_html_for_single_export)

with st.sidebar:
    st.markdown("### Exportar Atual")
    st.download_button("📄 Baixar HTML", single_html, f"{selected_asset}.html", "text/html")
    if HAS_PLAYWRIGHT:
        if st.button("🖨️ Baixar PDF"):
            with st.spinner("Gerando PDF..."):
                pdf = generate_pdf_from_html(single_html)
                if pdf:
                    b64_pdf = base64.b64encode(pdf).decode()
                    href = f'<a href="data:application/pdf;base64,{b64_pdf}" download="{selected_asset}.pdf" style="text-decoration:none; color:white; background:#825120; padding:8px 16px; border-radius:5px; display:block; text-align:center;">⬇️ Baixar PDF</a>'
                    st.markdown(href, unsafe_allow_html=True)

st.markdown("---")
st.markdown("<div style='text-align:center; color:#999; font-size:10px;'>Ceres Wealth System</div>", unsafe_allow_html=True)
