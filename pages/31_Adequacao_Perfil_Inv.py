import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
import os
from datetime import datetime
import json

# New Imports
from databases.models.client_profile import ClientProfile
import databases.profile_manager as profile_manager
import utils.compliance_report_generator as report_gen

# --- Configuration ---
st.set_page_config(page_title="Adequação de Perfil", layout="wide")

# Authentication check (common in this project)
if not st.session_state.get("authenticated", False):
    # Fallback for local testing if needed
    pass

# --- Database & Overrides Setup ---
DB_PATH = "databases/gorila_positions.db"
GORILA_DB_PATH = "databases/gorila.db"
PATHS_TO_CHECK = [DB_PATH, GORILA_DB_PATH]
OVERRIDES_FILE = "databases/adequacao_overrides.json"

def load_overrides():
    if os.path.exists(OVERRIDES_FILE):
        try:
            with open(OVERRIDES_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_overrides(overrides):
    # Ensure directory exists
    os.makedirs(os.path.dirname(OVERRIDES_FILE), exist_ok=True)
    with open(OVERRIDES_FILE, "w", encoding="utf-8") as f:
        json.dump(overrides, f, ensure_ascii=False, indent=2)

def get_connection(path):
    return sqlite3.connect(path)

@st.cache_data(ttl=600)
def load_portfolio_names():
    if not os.path.exists(GORILA_DB_PATH):
        return {}
    try:
        with get_connection(GORILA_DB_PATH) as conn:
            df = pd.read_sql_query("SELECT id, name FROM portfolios", conn)
            return dict(zip(df['id'], df['name']))
    except Exception as e:
        st.error(f"Erro ao carregar nomes dos portfólios: {e}")
        return {}

@st.cache_data(ttl=600)
def get_available_dates():
    if not os.path.exists(DB_PATH):
        return []
    try:
        with get_connection(DB_PATH) as conn:
            query = "SELECT DISTINCT reference_date FROM pmv_plus_gorila ORDER BY reference_date DESC"
            df = pd.read_sql_query(query, conn)
            # Filter valid dates (exclude None/Null/Empty)
            dates = [d for d in df['reference_date'].tolist() if d and str(d).lower() != 'none']
            return dates
    except Exception:
        return []

@st.cache_data(ttl=600)
def load_data(target_date=None):
    if not os.path.exists(DB_PATH):
        st.error(f"Banco de dados não encontrado: {DB_PATH}")
        return pd.DataFrame()
    
    try:
        with get_connection(DB_PATH) as conn:
            if target_date:
                query = "SELECT * FROM pmv_plus_gorila WHERE reference_date = ?"
                df = pd.read_sql_query(query, conn, params=(target_date,))
            else:
                query = "SELECT * FROM pmv_plus_gorila"
                df = pd.read_sql_query(query, conn)
        return df
    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")
        return pd.DataFrame()

# --- Asset Mapping Logic ---
def classify_asset(row, overrides=None):
    security_name_raw = str(row.get('security_name', ''))
    security_type = str(row.get('security_type', '')).upper()
    asset_class = str(row.get('asset_class', '')).upper()
    security_name = security_name_raw.lower()
    
    # Check manual overrides first
    if overrides and security_name_raw in overrides:
        return overrides[security_name_raw]
    
    # 1. Liquidez
    if security_type in ['CASH', 'CURRENCY', 'REPO', 'TREASURY_LOCAL_LFT'] or \
       any(k in security_name for k in ['liquidez', 'lft', 'caixa', 'compromissada', 'money market']):
        return "Liquidez"
    
    # 2. Previdência (check before others as it can be RF/MM)
    if 'previdência' in security_name or 'prev ' in security_name or ' bg ' in security_name:
        if asset_class == 'MULTIMARKET' or 'mult' in security_name:
            return "Previdência Privada - Multimercado"
        return "Previdência Privada - Renda Fixa"

    # 3. Exterior
    if asset_class == 'OFFSHORE' or 'offshore' in security_name or 'exterior' in security_name:
        if security_type in ['OFFSHORE_STOCK', 'OFFSHORE_ETF', 'OFFSHORE_REIT'] or \
           any(k in security_name for k in ['equity', 'stock', 'reit', 'ações exterior']):
            return "Investimentos no Exterior - Renda Variável"
        return "Investimentos no Exterior - Renda Fixa"

    # 4. Renda Fixa - Títulos Públicos
    if security_type.startswith('TREASURY_LOCAL') or \
       any(k in security_name for k in ['tesouro', 'ntn', 'ltn']):
        return "Renda Fixa - Títulos Públicos"
    
    # 5. Renda Fixa - Crédito Bancário
    if security_type in ['CORPORATE_BONDS_CDB', 'CORPORATE_BONDS_LCI', 'CORPORATE_BONDS_LCA', 'CORPORATE_BONDS_LIG', 'CORPORATE_BONDS_LCD'] or \
       any(k in security_name for k in ['cdb', 'lci', 'lca', 'lig', 'lcd', 'letra de crédito', 'bancário']):
        return "Renda Fixa - Crédito Bancário"
    
    # 6. Renda Fixa - Crédito Privado
    if security_type in ['CORPORATE_BONDS_CRA', 'CORPORATE_BONDS_CRI', 'CORPORATE_BONDS_DEBENTURE', 'CORPORATE_BONDS_CDCA'] or \
       any(k in security_name for k in ['debênture', 'cra ', 'cri ', 'cdca', 'deb ', 'financiamento', 'imobiliário']):
        return "Renda Fixa - Crédito Privado"
    
    # 7. Renda Fixa - Fundos de Crédito
    if (security_type == 'FUNDQUOTE' and asset_class == 'FIXED_INCOME') or \
       ('fundo' in security_name and 'crédito' in security_name):
        return "Renda Fixa - Fundos de Crédito"
    
    # 8. Multimercado
    if asset_class == 'MULTIMARKET' or 'multimercado' in security_name:
        if 'exterior' in security_name or 'offshore' in security_name:
            return "Fundos Multimercado Exterior"
        return "Fundos Multimercado"
    
    # 9. Renda Variável
    if security_type == 'STOCK_LOCAL' or 'ações' in security_name:
        return "Renda Variável - Ações"
    if security_type == 'FII' or 'fii ' in security_name or 'fundo imobiliário' in security_name:
        return "Renda Variável - Fundos Imobiliários"
    if security_type == 'FUNDQUOTE' and asset_class == 'STOCKS':
        return "Renda Variável - Fundos de Ações"
    
    # 10. Criptomoedas
    if security_type == 'CRYPTOCURRENCY' or any(k in security_name for k in ['cripto', 'bitcoin', 'eth ', ' btc ']):
        return "Criptomoedas"
    
    # 11. Estruturados / FIDC
    if 'fidc' in security_name or 'estruturado' in security_name or 'fip' in security_name:
        return "Produtos Estruturados (FIDCs)"
    
    # Fallback for remaining Fixed Income
    if asset_class == 'FIXED_INCOME':
        return "Renda Fixa - Crédito Privado"
    
    return "Outros / Não Classificado"

def check_violation(current_pct, current_money, ex):
    """Check if violation occurs based on limit_mode."""
    pct_violated = ex.limit_pct is not None and current_pct > ex.limit_pct
    money_violated = ex.limit_money is not None and current_money > ex.limit_money
    
    if ex.limit_mode == "pct":
        return pct_violated
    elif ex.limit_mode == "money":
        return money_violated
    elif ex.limit_mode == "and":
        return pct_violated and money_violated
    else:  # "or"
        return pct_violated or money_violated

def format_violation_msg(current_pct, current_money, ex, label):
    """Format violation message based on limit_mode."""
    if ex.limit_mode == "pct":
        return f"⚠️ {label}: ocupa {current_pct:.2f}% (Limite: {ex.limit_pct:.1f}%)"
    elif ex.limit_mode == "money":
        return f"⚠️ {label}: ocupa R$ {current_money:,.0f} (Limite: R$ {ex.limit_money:,.0f})"
    elif ex.limit_mode == "and":
        return f"⚠️ {label}: {current_pct:.2f}% E R$ {current_money:,.0f} > Limites"
    else:  # "or"
        return f"⚠️ {label}: {current_pct:.2f}% OU R$ {current_money:,.0f} > Limites"

def check_client_compliance(portfolio_id, client_name, df_positions, overrides, profile):
    # Filter for this portfolio
    df_filtered = df_positions[df_positions['portfolio_id'].astype(str) == str(portfolio_id)].copy()
    
    if df_filtered.empty:
        return {
            "portfolio_id": portfolio_id,
            "client_name": client_name,
            "total_equity": 0.0,
            "has_issues": False,
            "allocation_violations": [],
            "exception_violations": [],
            "df_comparison": pd.DataFrame(),
            "df_filtered": df_filtered
        }

    # classify
    df_filtered['Classe_Adequacao'] = df_filtered.apply(lambda row: classify_asset(row, overrides), axis=1)
    total_mv = df_filtered['market_value_amount'].sum()
    
    # 2. Group by Class
    df_current = df_filtered.groupby('Classe_Adequacao')['market_value_amount'].sum().reset_index()
    if total_mv > 0:
        df_current['Atual (%)'] = (df_current['market_value_amount'] / total_mv) * 100
    else:
        df_current['Atual (%)'] = 0.0
    df_current.rename(columns={'Classe_Adequacao': 'Classe de Ativos'}, inplace=True)
    
    # 3. Merge with Target
    df_target = profile.to_dataframe()
    df_comparison = pd.merge(df_target, df_current[['Classe de Ativos', 'Atual (%)']], on='Classe de Ativos', how='left').fillna(0)
    
    # 4. Check Allocation Violations
    allocation_violations = []
    
    for _, row in df_comparison.iterrows():
        if not (row['Min (%)'] <= row['Atual (%)'] <= row['Max (%)']):
             allocation_violations.append(f"{row['Classe de Ativos']}: {row['Atual (%)']:.2f}% (Meta: {row['Min (%)']:.1f}-{row['Max (%)']:.1f}%)")

    # 5. Check Exceptions
    exception_violations = []
    if profile.exceptions:
        for ex in profile.exceptions:
            target_vals_upper = [v.upper() for v in ex.target_values]
            
            if ex.target_type == "security_type":
                relevant_rows = df_filtered[df_filtered['security_type'].str.upper().isin(target_vals_upper)]
            elif ex.target_type == "asset_class":
                relevant_rows = df_filtered[df_filtered['asset_class'].str.upper().isin(target_vals_upper)]
            else:
                relevant_rows = pd.DataFrame()
            
            if relevant_rows.empty:
                continue
                
            total_value_ex = relevant_rows['market_value_amount'].sum()
            current_pct_ex = (total_value_ex / total_mv) * 100 if total_mv > 0 else 0.0
            
            target_label = ", ".join(ex.target_values)
            
            if ex.rule_type == "proibida" and current_pct_ex > 0.001:
                exception_violations.append(f"❌ Proibido: {target_label} ({current_pct_ex:.2f}%)")
                
            elif ex.rule_type == "max_limit":
                if check_violation(current_pct_ex, total_value_ex, ex):
                    exception_violations.append(format_violation_msg(current_pct_ex, total_value_ex, ex, f"Classe {target_label}"))
            
            elif ex.rule_type == "max_asset_limit":
                for _, asset_row in relevant_rows.iterrows():
                    asset_value = asset_row['market_value_amount']
                    asset_pct = (asset_value / total_mv) * 100 if total_mv > 0 else 0.0
                    if check_violation(asset_pct, asset_value, ex):
                        exception_violations.append(format_violation_msg(asset_pct, asset_value, ex, asset_row['security_name']))
            
            elif ex.rule_type == "max_issuer_limit":
                issuer_col = 'parsed_company_name'
                if issuer_col in relevant_rows.columns:
                    issuer_groups = relevant_rows.groupby(issuer_col, dropna=False)['market_value_amount'].sum()
                    for issuer, issuer_value in issuer_groups.items():
                        issuer_pct = (issuer_value / total_mv) * 100 if total_mv > 0 else 0.0
                        issuer_name = issuer if issuer else "Emissor Desconhecido"
                        if check_violation(issuer_pct, issuer_value, ex):
                            exception_violations.append(format_violation_msg(issuer_pct, issuer_value, ex, f"Emissor: {issuer_name}"))

    return {
        "portfolio_id": portfolio_id,
        "client_name": client_name,
        "total_equity": total_mv,
        "has_issues": bool(allocation_violations or exception_violations),
        "allocation_violations": allocation_violations,
        "exception_violations": exception_violations,
        "df_comparison": df_comparison,
        "df_filtered": df_filtered
    }

# --- UI Layout ---
st.title("🛡️ Adequação de Perfil de Investimento")
st.markdown("---")

df_positions = pd.DataFrame()

# --- Date Selection ---
available_dates = get_available_dates()
selected_date = None

if available_dates:
    col_date_1, col_date_2 = st.columns([1, 3])
    with col_date_1:
        selected_date = st.selectbox(
            "📅 Data de Referência", 
            available_dates, 
            index=0,
            help="Selecione a data base para a análise das carteiras."
        )
else:
    st.warning("⚠️ Nenhuma data encontrada na base de dados.")

# Load Data based on selection
if selected_date:
    df_positions = load_data(selected_date)
else:
    # Fallback if no dates found or selection issue, though logic above handles it
    df_positions = load_data()

if df_positions.empty:
    st.info("Nenhum dado encontrado na tabela 'pmv_plus_gorila'.")
    st.stop()

# Load custom overrides and portfolio names
overrides = load_overrides()
portfolio_names = load_portfolio_names()

# Portfolio Selection Logic
df_positions_all = df_positions.copy()
portfolios_raw = df_positions['portfolio_id'].unique()
valid_p_ids = [str(p) for p in portfolios_raw if p is not None]

# --- Global Filters (Main Body) ---
col_gf_1, col_gf_2 = st.columns([3, 1])
with col_gf_1:
    only_configured_global = st.checkbox(
        "⚙️ Somente Perfis Configurados (Filtro Global)", 
        value=True, 
        help="Mostrar apenas clientes que já possuem perfil salvo. Isso afeta todas as abas."
    )

if only_configured_global:
    configured_ids = profile_manager.list_profiles()
    valid_p_ids = [pid for pid in valid_p_ids if str(pid) in configured_ids]
    
    if not valid_p_ids:
        st.warning("Nenhum cliente encontrado com perfil configurado.")


# Format options as "Name - ID"
selection_options = []
id_to_option = {}
id_to_name = {}
for p_id in valid_p_ids:
    name = portfolio_names.get(p_id, "Desconhecido")
    option = f"{name} - {p_id}"
    selection_options.append(option)
    id_to_option[p_id] = option
    id_to_name[p_id] = name

selection_options.sort()

# --- Tabs ---
tab_overview, tab_analysis, tab_manage = st.tabs(["🔍 Visão Geral", "📊 Análise de Adequação", "⚙️ Gerenciar Perfis"])

# --- TAB 1: Overview ---
with tab_overview:
    st.header("Visão Geral de Conformidade")
    
    col_ov_1, col_ov_2 = st.columns([3, 1])
    with col_ov_1:
         # Remove checkbox as it moved to sidebar
         only_issues = st.checkbox("⚠️ Apenas com Pontos de Atenção", value=True)
    with col_ov_2:
         if st.button("🔄 Atualizar Análise"):
             st.rerun()

    if st.button("Iniciar Análise Completa (Pode demorar um pouco)"):
        all_results = []
        progress_bar = st.progress(0)
        
        # Determine list of portfolios (Use Global Filter)
        ids_to_check = valid_p_ids
        total_p = len(ids_to_check)
        
        if total_p == 0:
            st.warning("Nenhum cliente disponível para análise com os filtros atuais.")
            st.stop()
        
        for idx, p_id_str in enumerate(ids_to_check):

            c_name = portfolio_names.get(p_id_str, "Desconhecido") # portfolio_names loaded globally
            
            # Load Profile
            c_profile = profile_manager.load_profile(p_id_str, client_name=c_name)
            
            # Run Check
            res = check_client_compliance(p_id_str, c_name, df_positions, overrides, c_profile)
            all_results.append(res)
            
            progress_bar.progress((idx + 1) / total_p)
            
        progress_bar.empty()
        
        # Build DataFrame for display
        summary_rows = []
        for res in all_results:
            if not res: continue
            
            if only_issues and not res['has_issues']:
                continue
                
            status_icon = "⚠️" if res['has_issues'] else "✅"
            
            violations_text = ""
            if res['allocation_violations']:
                violations_text += "Alocação: " + "; ".join(res['allocation_violations']) + ". "
            if res['exception_violations']:
                violations_text += "Regras: " + "; ".join(res['exception_violations'])
            
            summary_rows.append({
                "Status": status_icon,
                "Cliente": res['client_name'],
                "PL Total": res['total_equity'],
                "Detalhes": violations_text
            })
            
        df_summary = pd.DataFrame(summary_rows)
        if not df_summary.empty:
            st.dataframe(
                df_summary,
                column_config={
                    "Status": st.column_config.TextColumn("Status", width="small"),
                    "Cliente": st.column_config.TextColumn("Cliente", width="medium"),
                    "PL Total": st.column_config.NumberColumn("PL Total", format="R$ %.2f"),
                    "Detalhes": st.column_config.TextColumn("Detalhes", width="large"),
                },
                use_container_width=True,
                hide_index=True
            )
        else:
            st.success("Tudo certo! Nenhum cliente com pendências encontrado (nos filtros atuais).")

# --- TAB 2: Analysis ---
with tab_analysis:
    col_sel_1, col_sel_2 = st.columns(2)
    with col_sel_1:
        selected_option = st.selectbox("Selecione o Portfólio para Análise", selection_options, key="select_analysis")
    
    selected_portfolio_id = selected_option.rsplit(" - ", 1)[-1]
    client_name = id_to_name.get(selected_portfolio_id, "Cliente Desconhecido")
    
    # Load Profile
    profile = profile_manager.load_profile(selected_portfolio_id, client_name=client_name)
    df_target = profile.to_dataframe()
    
    # Run Check (using shared function)
    res = check_client_compliance(selected_portfolio_id, client_name, df_positions, overrides, profile)
    
    df_filtered = res['df_filtered']
    total_mv = res['total_equity']
    df_comparison = res['df_comparison']
    
    # Display Status/Violations immediately
    if res['has_issues']:
        with st.expander("🚨 Detalhes dos Pontos de Atenção", expanded=True):
            if res['allocation_violations']:
                st.write("**Alocação Fora da Meta:**")
                for v in res['allocation_violations']:
                    st.write(f"- {v}")
            
            if res['exception_violations']:
                if res['allocation_violations']: st.write("---")
                st.write("**Violação de Regras de Exceção:**")
                for v in res['exception_violations']:
                     st.write(f"- {v}")
    
    # Calculate Deviations
    df_comparison['Status'] = df_comparison.apply(
        lambda x: "✅ OK" if x['Min (%)'] <= x['Atual (%)'] <= x['Max (%)'] else "⚠️ Fora", axis=1
    )
    df_comparison['Diferença (%)'] = df_comparison['Atual (%)'] - df_comparison['Alvo (%)']
    
    # Display Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Patrimônio Total", f"R$ {total_mv:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
    col2.metric("Ativos Classificados", len(df_filtered))
    col3.metric("Status Geral", "Revisão Necessária" if (df_comparison['Status'] == "⚠️ Fora").any() else "Adequado")
    
    # Report Generation
    st.markdown("---")
    col_rep_1, col_rep_2 = st.columns([1, 1])
    
    report_html = None
    
    with col_rep_1:
        if st.button("📄 Gerar Relatório de Conformidade"):
            with st.spinner("Gerando relatório..."):
                try:
                    # Pass the COMPOSITE object `res` which contains violations lists
                    # Construct specific compliance_data dict expected by generator
                    compliance_data = {
                        'has_issues': res['has_issues'],
                        'allocation_violations': res['allocation_violations'],
                        'exception_violations': res['exception_violations'],
                        'df_comparison': df_comparison
                    }
                    
                    report_html = report_gen.generate_compliance_html(
                        client_name=client_name,
                        portfolio_id=selected_portfolio_id,
                        total_equity=total_mv,
                        compliance_data=compliance_data,
                        profile=profile
                    )
                    
                    # Store in session state to persist after rerun (if needed, though button press acts as trigger)
                    st.session_state['last_report_html'] = report_html
                    st.success("Relatório gerado com sucesso!")
                    
                except Exception as e:
                    st.error(f"Erro ao gerar relatório: {e}")

    with col_rep_2:
        if st.session_state.get('last_report_html'):
            report_data = st.session_state['last_report_html']
            
            # HTML Download
            st.download_button(
                label="⬇️ Baixar Relatório (HTML)",
                data=report_data,
                file_name=f"Relatorio_Adequacao_{selected_portfolio_id}.html",
                mime="text/html"
            )
            
            # PDF Download (Try/Except for Playwright)
            try:
                if st.button("⬇️ Baixar Relatório (PDF)"):
                    with st.spinner("Renderizando PDF... (isso pode levar alguns segundos)"):
                        pdf_data = report_gen.html_to_pdf(report_data)
                        st.download_button(
                            label="Clique para Salvar PDF",
                            data=pdf_data,
                            file_name=f"Relatorio_Adequacao_{selected_portfolio_id}.pdf",
                            mime="application/pdf"
                        )
            except ImportError:
               st.warning("Biblioteca de geração de PDF não disponível (Playwright). Apenas HTML disponível.")
            except Exception as e:
               st.error(f"Erro na geração de PDF: {e}")

    st.markdown("---")
    # Formatting for display
    df_display = df_comparison.copy()
    for col in ['Alvo (%)', 'Atual (%)', 'Min (%)', 'Max (%)', 'Diferença (%)']:
        df_display[col] = df_display[col].apply(lambda x: f"{x:.2f}%")
    
    df_display['Faixa Tolerada (%)'] = df_comparison.apply(lambda x: f"{x['Min (%)']:.1f}% - {x['Max (%)']:.1f}%", axis=1)
    
    cols_to_show = ['Classe de Ativos', 'Alvo (%)', 'Atual (%)', 'Faixa Tolerada (%)', 'Diferença (%)', 'Status']
    st.table(df_display[cols_to_show])
    
    # Visual analysis
    st.subheader("Análise Visual")
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Atual',
        x=df_comparison['Classe de Ativos'],
        y=df_comparison['Atual (%)'],
        marker_color='#582308'
    ))
    fig.add_trace(go.Scatter(
        name='Alvo',
        x=df_comparison['Classe de Ativos'],
        y=df_comparison['Alvo (%)'],
        mode='markers',
        marker=dict(size=12, color='#013220', symbol='diamond')
    ))
    
    fig.update_layout(
        title=f"Alocação Atual vs Alvo ({profile.name})",
        xaxis_title="Classe de Ativos",
        yaxis_title="Percentual (%)",
        barmode='group',
        height=600
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Detalhamento de Ativos por Classe
    st.markdown("---")
    st.subheader("Detalhamento por Classe de Ativo")
    class_options_sorted = sorted(df_target['Classe de Ativos'].unique())
    selected_class = st.selectbox("Selecione uma Classe para ver os ativos", class_options_sorted)
    
    df_details = df_filtered[df_filtered['Classe_Adequacao'] == selected_class][['security_name', 'security_type', 'market_value_amount']].copy()
    df_details['% do Portfólio'] = (df_details['market_value_amount'] / total_mv) * 100
    st.dataframe(df_details.style.format({'market_value_amount': '{:,.2f}', '% do Portfólio': '{:.2f}%'}), use_container_width=True)

    # Asset Classification Editor
    st.markdown("---")
    col_ed_h, col_ed_b = st.columns([3, 1])
    with col_ed_h:
        st.subheader("✏️ Editor de Classificação de Ativos (Global)")
        st.info("Classifique ativos manualmente. Estas regras são GLOBAIS (valem para todos os clientes).")
    
    # Prepare data for editor
    df_editor = df_filtered[['security_name', 'Classe_Adequacao']].drop_duplicates().sort_values('security_name')
    df_editor.rename(columns={'security_name': 'Ativo', 'Classe_Adequacao': 'Classificação Atual'}, inplace=True)
    
    classes_available = sorted(df_target['Classe de Ativos'].tolist() + ["Outros / Não Classificado"])
    
    edited_df = st.data_editor(
        df_editor,
        column_config={
            "Classificação Atual": st.column_config.SelectboxColumn(
                "Classificação Manual",
                help="Selecione a classe correta para este ativo",
                width="large",
                options=classes_available,
                required=True,
            )
        },
        disabled=["Ativo"],
        hide_index=True,
        use_container_width=True,
        key="asset_editor"
    )
    
    with col_ed_b:
        st.write("") # Spacer
        st.write("")
        if st.button("Salvar Classificações", use_container_width=True, type="primary", key="save_overrides_btn"):
            new_overrides = load_overrides()
            has_changes = False
            
            for _, row_ed in edited_df.iterrows():
                asset_name = row_ed['Ativo']
                new_class = row_ed['Classificação Atual']
                
                # Get current heuristic class to see if it changed
                # We need to find the original row in df_filtered to get its data
                orig_row = df_filtered[df_filtered['security_name'] == asset_name].iloc[0]
                # Heuristic class is what classify_asset returns without overrides
                heuristic_class = classify_asset(orig_row, overrides={}) 
                
                if new_class != heuristic_class:
                    new_overrides[asset_name] = new_class
                    has_changes = True
                elif asset_name in new_overrides:
                    # If set back to heuristic value, remove override
                    del new_overrides[asset_name]
                    has_changes = True
            
            if has_changes:
                save_overrides(new_overrides)
                st.success("Salvo!")
                st.rerun()
            else:
                st.info("Sem alterações.")


# --- TAB 2: Management ---
with tab_manage:
    st.header("Gerenciamento de Perfis de Investimento")
    st.info("Aqui você pode definir metas de alocação específicas para cada cliente.")
    
    col_m_1, col_m_2 = st.columns(2)
    with col_m_1:
         selected_manage_option = st.selectbox("Selecione o Cliente para Editar", selection_options, key="select_manage")
    
    manage_id = selected_manage_option.rsplit(" - ", 1)[-1]
    manage_name = id_to_name.get(manage_id, "Desconhecido")
    
    # Load (or create default) profile
    profile_edit = profile_manager.load_profile(manage_id, client_name=manage_name)
    
    st.subheader(f"Editando regras para: {profile_edit.name}")
    
    # Convert rules to DataFrame for editing
    # We need to restructure it slightly for data_editor to perform validation
    rules_data = []
    for asset_class, rule in profile_edit.rules.items():
        rules_data.append({
            "Classe de Ativos": asset_class,
            "Alvo (%)": float(rule.target),
            "Min (%)": float(rule.min_val),
            "Max (%)": float(rule.max_val)
        })
    df_rules_edit = pd.DataFrame(rules_data)
    
    edited_rules_df = st.data_editor(
        df_rules_edit,
        column_config={
            "Classe de Ativos": st.column_config.TextColumn("Classe", disabled=True),
            "Alvo (%)": st.column_config.NumberColumn("Meta (%)", min_value=0.0, max_value=100.0, format="%.2f"),
            "Min (%)": st.column_config.NumberColumn("Mín (%)", min_value=0.0, max_value=100.0, format="%.2f"),
            "Max (%)": st.column_config.NumberColumn("Máx (%)", min_value=0.0, max_value=100.0, format="%.2f"),
        },
        hide_index=True,
        use_container_width=True,
        key="rules_editor"
    )
    
    if st.button("Salvar Perfil", type="primary"):
        # validate logic (Min <= Alvo <= Max)
        has_error = False
        for _, row in edited_rules_df.iterrows():
            if not (row['Min (%)'] <= row['Alvo (%)'] <= row['Max (%)']):
                st.error(f"Erro na classe '{row['Classe de Ativos']}': Min ({row['Min (%)']}) deve ser <= Alvo ({row['Alvo (%)']}) <= Max ({row['Max (%)']})")
                has_error = True
        
        if not has_error:
            # Update profile object
            for _, row in edited_rules_df.iterrows():
                profile_edit.update_rule(
                    row['Classe de Ativos'],
                    row['Alvo (%)'],
                    row['Min (%)'],
                    row['Max (%)']
                )
            
            # Persist
            try:
                profile_manager.save_profile(profile_edit)
                st.success(f"Perfil de {profile_edit.name} salvo com sucesso!")
            except Exception as e:
                st.error(f"Erro ao salvar perfil: {e}")
            
    # --- Exceptions Management Section ---
    st.markdown("---")
    st.subheader("⚠️ Regras de Exceção Personalizadas")
    st.info("Adicione regras específicas para proibir ou limitar ativos, tipos de ativos, ou concentração por ativo/emissor.")

    col_ex_1, col_ex_2 = st.columns([1, 2])
    
    with col_ex_1:
        st.markdown("##### Adicionar Nova Regra")
        ex_target_type = st.selectbox("Tipo de Alvo", ["security_type", "asset_class"], key="ex_type")
        
        # Get unique options based on selection - now using MULTISELECT
        if ex_target_type == "security_type":
             unique_options = sorted([str(x).upper() for x in df_positions['security_type'].dropna().unique()])
        else:
             unique_options = sorted([str(x).upper() for x in df_positions['asset_class'].dropna().unique()])
             
        ex_target_vals = st.multiselect("Valor(es) do Alvo", unique_options, key="ex_vals")
        
        ex_rule_type = st.selectbox(
            "Tipo de Regra", 
            ["proibida", "max_limit", "max_asset_limit", "max_issuer_limit"],
            format_func=lambda x: {
                "proibida": "Proibido",
                "max_limit": "Limite Máximo (Classe)",
                "max_asset_limit": "Limite por Ativo Único",
                "max_issuer_limit": "Limite por Emissor"
            }.get(x, x),
            key="ex_rule"
        )
        
        # Limit mode and inputs (only show for non-prohibited rules)
        ex_limit_pct = None
        ex_limit_money = None
        ex_limit_mode = "pct"
        
        if ex_rule_type != "proibida":
            ex_limit_mode = st.selectbox(
                "Modo de Limite",
                ["pct", "money", "and", "or"],
                format_func=lambda x: {
                    "pct": "Somente % do PL",
                    "money": "Somente R$",
                    "and": "% E R$ (ambos devem ser violados)",
                    "or": "% OU R$ (qualquer um violado)"
                }.get(x, x),
                key="ex_limit_mode"
            )
            
            # Show appropriate inputs based on mode
            if ex_limit_mode in ["pct", "and", "or"]:
                ex_limit_pct = st.number_input("Limite Máximo (%)", min_value=0.0, max_value=100.0, value=0.0, key="ex_limit_pct")
            if ex_limit_mode in ["money", "and", "or"]:
                ex_limit_money = st.number_input("Limite Máximo (R$)", min_value=0.0, value=0.0, format="%.2f", key="ex_limit_money")
        
        if st.button("Adicionar Regra"):
            if not ex_target_vals:
                st.error("Selecione pelo menos um valor alvo.")
            else:
                # Create one rule per selected value for easier management
                for val in ex_target_vals:
                    profile_edit.add_exception(
                        ex_target_type, 
                        [val],  # Single value per rule
                        ex_rule_type, 
                        limit_pct=ex_limit_pct,
                        limit_money=ex_limit_money,
                        limit_mode=ex_limit_mode
                    )
                profile_manager.save_profile(profile_edit)
                st.success(f"{len(ex_target_vals)} regra(s) adicionada(s)!")
                st.rerun()

    with col_ex_2:
        st.markdown("##### Regras Ativas")
        if not profile_edit.exceptions:
            st.info("Nenhuma regra de exceção cadastrada.")
        else:
            ex_data = []
            for i, ex in enumerate(profile_edit.exceptions):
                # Format limit display based on mode
                if ex.rule_type == "proibida":
                    limit_str = "Proibido"
                else:
                    rule_label = {
                        "max_limit": "Máx Classe",
                        "max_asset_limit": "Máx/Ativo",
                        "max_issuer_limit": "Máx/Emissor"
                    }.get(ex.rule_type, ex.rule_type)
                    
                    if ex.limit_mode == "pct":
                        limit_str = f"{rule_label}: {ex.limit_pct:.1f}%"
                    elif ex.limit_mode == "money":
                        limit_str = f"{rule_label}: R$ {ex.limit_money:,.0f}"
                    elif ex.limit_mode == "and":
                        limit_str = f"{rule_label}: {ex.limit_pct:.1f}% E R$ {ex.limit_money:,.0f}"
                    else:  # or
                        limit_str = f"{rule_label}: {ex.limit_pct:.1f}% OU R$ {ex.limit_money:,.0f}"
                
                ex_data.append({
                    "Alvo": ex.target_type,
                    "Valores": ", ".join(ex.target_values),
                    "Regra": limit_str,
                    "Index": i
                })
            
            df_ex = pd.DataFrame(ex_data)
            st.dataframe(df_ex[["Alvo", "Valores", "Regra"]], use_container_width=True, hide_index=True)
            
            # Remove rule
            rule_to_remove = st.selectbox("Selecione a regra para remover", 
                                          options=[f"{r['Valores']} ({r['Regra']})" for r in ex_data],
                                          index=None,
                                          placeholder="Selecione para remover...")
            
            if rule_to_remove and st.button("Remover Regra Selecionada"):
                selected_idx = [f"{r['Valores']} ({r['Regra']})" for r in ex_data].index(rule_to_remove)
                target_to_remove = profile_edit.exceptions[selected_idx]
                
                profile_edit.remove_exception(target_to_remove.target_type, target_to_remove.target_values)
                profile_manager.save_profile(profile_edit)
                st.success("Regra removida!")
                st.rerun()


