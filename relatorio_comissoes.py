import streamlit as st

st.set_page_config(page_title="Relatórios", layout="wide")

PASSWORD = st.secrets["auth"]["PASSWORD"]  # ideally use st.secrets or env var instead of hardcoding

st.title("Dashboard Ceres Wealth")

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

def home():
    st.title("Dashboard de Relatórios")

    st.write(
        "Bem-vindo! Use o menu superior para acessar relatórios, "
        "ferramentas, carteiras e integrações de dados."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Relatórios")
        st.markdown(
            "- Relatório Comissões\n"
            "- Relatório Positivador\n"
            "- Relatório de Crédito"
        )

    with col2:
        st.subheader("Outras seções")
        st.markdown(
            "- Ferramentas auxiliares (CDI, Factsheet, etc.)\n"
            "- Carteiras recomendadas e fundos\n"
            "- APIs e dados externos"
        )


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
            icon=":material/insert_chart:",),
        
        st.Page(
            "pages/4_Financial_Comparator_Plotly.py",
            title="Comparador de Ações",
            icon=":material/bar_chart:",
        ),
        
    ],
    "One Pager Fundos": [
        st.Page("pages/5_baixar_cdi.py", 
                title="1. Baixar CDI",
                icon=":material/download:"
                ),
        st.Page(
            "pages/11_carteira_fundos_FINAL.py",
            title="2. Carteira Fundos",
            icon=":material/account_balance_wallet:",
            ),
        st.Page("pages/6_rent_fundos_gt.py", 
                title="3. Rent. Fundos",
                icon=":material/trending_up:",
                ),
        
        st.Page("pages/7_factsheet.py", 
                title="4. Gerar Excel One Pager",
                icon=":material/description:",
                ),


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
            title="Tabela Fórnico",
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
            "pages/14_RELATORIO_CREDITO.py",
            title="Relatório de Crédito",
            icon=":material/account_balance:",  
              ),
    ],
}

pg = st.navigation(pages, position="top")
pg.run()