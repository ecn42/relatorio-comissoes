import streamlit as st

st.set_page_config(page_title="Relatórios", layout="wide")

PASSWORD = st.secrets["auth"]["PASSWORD"]  # use st.secrets or env var

st.title("Dashboard Ceres Wealth")

# Simple auth gate
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    pwd = st.text_input("Password", type="password")
    if st.button("Login"):
        if pwd == PASSWORD:
            st.session_state["authenticated"] = True
            st.success("Logged in. You can now open other pages.")
        else:
            st.error("Incorrect password")
else:
    st.success("You are already logged in.")

# ---------- PAGES (definitions first) ----------


def home():
    st.title("Dashboard de Relatórios")

    st.write(
        "Bem-vindo! Abaixo está um resumo das seções e páginas disponíveis. "
        "Use o menu superior para navegar."
    )

    st.divider()

    st.subheader("Acesso Rápido")
    
    # Ferramentas
    st.markdown("##### 🛠️ Ferramentas")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.page_link("pages/3_Graph_Studio.py", label="Estúdio de Gráficos", icon="📊")
        st.page_link("pages/25_Carteiras_RV.py", label="Carteiras RV", icon="💼")
    with c2:
        st.page_link("pages/26_Carteiras_Mes_Atual.py", label="Carteiras Mês Atual", icon="📅")
        st.page_link("pages/28_Gerador_Trade_Ideas.py", label="Gerador de Trade Ideas", icon="🚀")
    with c3:
        st.page_link("pages/30_Analise_Acoes_Economatica.py", label="Análise Ações Economatica", icon="📈")
        st.page_link("pages/31_Adequacao_Perfil_Inv.py", label="Adequação Perfil", icon="⚖️")
    with c4:
        st.page_link("pages/34_Ferramentas_Payoff.py", label="Calculadora Payoff", icon="🔢")
        
    st.divider()

    # Relatórios
    st.markdown("##### 📑 Relatórios")
    r1, r2, r3, r4 = st.columns(4)
    with r1:
         st.page_link("pages/14_RELATORIO_CREDITO.py", label="Relatório de Crédito", icon="🏦")
    with r2:
         st.page_link("pages/27_Relatorio_Mercado.py", label="Relatório de Mercado", icon="📉")
         
    st.divider()

    # Formatação
    st.markdown("##### 📝 Formatação")
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        st.page_link("pages/15_Texto_Carteira_Acoes.py", label="Texto Carteira Ações", icon="📄")
    with f2:
        st.page_link("pages/9_NEW_Portfolio_parser.py", label="Texto Asset Allocation", icon="📥")
    with f3:
        st.page_link("pages/10_tabela_fornico.py", label="Tabela Asset Allocation", icon="📊")
        
    st.divider()



# ---------- NAV SETUP (the real page map for navigation) ----------

home_page = st.Page(
    home,
    title="Home",
    icon=":material/home:",
    default=True,
)

pages = {
    "": [home_page],  # aparece primeiro no menu topo, fora dos grupos
    "Legado (XP)": [
        st.Page(
            "pages/1_comissoes.py",
            title="Relatório Comissões",
            icon=":material/trending_up:",
        ),
        st.Page(
            "pages/2_Relatorio_Positivador.py",
            title="Relatório Positivador",
            icon=":material/trending_up:",
        ),
    ],
    "Ferramentas": [
        st.Page(
            "pages/3_Graph_Studio.py",
            title="Estúdio de Gráficos",
            icon=":material/insert_chart:",
        ),
        st.Page(
            "pages/25_Carteiras_RV.py",
            title="Carteiras RV",
            icon=":material/account_balance_wallet:",
        ),

        st.Page(
            "pages/26_Carteiras_Mes_Atual.py",
            title="Carteiras Mes Atual",
            icon=":material/account_balance_wallet:",
        ),
        st.Page(
            "pages/28_Gerador_Trade_Ideas.py",
            title="Gerador de Trade Ideas",
            icon=":material/rocket_launch:",
        ),
        st.Page(
            "pages/30_Analise_Acoes_Economatica.py",
            title="Análise Ações Economatica",
            icon=":material/bar_chart:",
        ),
        st.Page(
            "pages/31_Adequacao_Perfil_Inv.py",
            title="Adequação Perfil Investidor",
            icon=":material/bar_chart:",
        ),
        st.Page(
            "pages/34_Ferramentas_Payoff.py",
            title="Calculadora Payoff",
            icon=":material/functions:",
        )
    ],
    "One Pager Fundos": [
        st.Page(
            "pages/5_baixar_cdi.py",
            title="1. Baixar CDI",
            icon=":material/download:",
        ),
        st.Page(
            "pages/11_carteira_fundos_FINAL.py",
            title="2. Carteira Fundos",
            icon=":material/account_balance_wallet:",
        ),
        st.Page(
            "pages/6_rent_fundos_gt.py",
            title="3. Rent. Fundos",
            icon=":material/trending_up:",
        ),
        st.Page(
            "pages/7_factsheet.py",
            title="4. Gerar Excel One Pager",
            icon=":material/description:",
        ),

        st.Page(
            "pages/20_TESTE_ONEPAGER_HTML.py",
            title="5. Gerar HTML/PDF",
            icon=":material/html:",
        ),
        st.Page(
            "pages/23_Puxar_FIDCS.py",
            title="6. Puxar FIDCS",
            icon=":material/description:",
        )
    ],
    "Formatação": [
        st.Page(
            "pages/15_Texto_Carteira_Acoes.py",
            title="Texto Carteira Ações",
            icon=":material/article:",
        ),
        st.Page(
            "pages/9_NEW_Portfolio_parser.py",
            title="Texto Asset Allocation",
            icon=":material/file_download:",
        ),
        st.Page(
            "pages/10_tabela_fornico.py",
            title="Tabela Asset Allocation",
            icon=":material/table_chart:",
        ),
    ],
    "Gorila/Relatórios de Risco": [
        st.Page(
            "pages/12_Gorila_API_novo.py",
            title="Gorila API Novo",
            icon=":material/api:",
        ),

        
        st.Page(
            "pages/17_Pictet_to_PMV.py",
            title="Pictet → PMV Mapper",
            icon=":material/sync_alt:",
        ),

        st.Page(
            "pages/18_CUSTOM_ASSETS_MANUAL_PARSER.py",
            title="Editar ativos CUSTOM",
            icon=":material/edit:",
        ),
        
        st.Page(
            "pages/14_RELATORIO_CREDITO.py",
            title="Relatório de Crédito",
            icon=":material/account_balance:",
        ),

        st.Page(
            "pages/19_Gorila_API_RODRIGOCABRAL.py",
            title="Dados API - Rodrigo Cabral",
            icon=":material/api:",
        ),

        st.Page(
            "pages/21_Add_Ratings_BR.py",
            title="Add Ratings BR",
            icon=":material/star:",
        ),

        st.Page(
            "pages/22_Rating_To_Fitch.py",
            title="Rating to Fitch",
            icon=":material/translate:",
        ),
        
        st.Page(
            "pages/27_Relatorio_Mercado.py",
            title="Relatório Mercado",
            icon=":material/vital_signs:",
        )

    ],
    "One Pager Crédito": [
        st.Page(
            "pages/33_Gerar_Onepager_Credito.py",
            title="Gerar OnePager Crédito",
            icon=":material/description:",
        ),
    ],
    "Databases": [
        st.Page(
            "pages/29_Database_Economatica.py",
            title="Database Economatica",
            icon=":material/database:",
        ),
        st.Page(
            "pages/32_Database_OnePager_Credito.py",
            title="Database OnePager Crédito",
            icon=":material/database:",
        ),
    ],
}

pg = st.navigation(pages, position="top")
pg.run()