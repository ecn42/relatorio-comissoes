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

    # Estrutura do resumo (sem paths)
    sections = {
        "": [
            {"title": "Home", "icon": ":material/home:", "default": True},
        ],
        "Legado (XP)": [
            {"title": "Relatório Comissões", "icon": ":material/trending_up:"},
            {"title": "Relatório Positivador", "icon": ":material/trending_up:"},
        ],
        "Ferramentas": [
            {"title": "Estúdio de Gráficos", "icon": ":material/insert_chart:"},
            {"title": "Comparador de Ações", "icon": ":material/bar_chart:"},
            {"title": "Gerador de Trade Ideas", "icon": ":material/rocket_launch:"},
            {"title": "API 4intelligence", "icon": ":material/api:"},
            {"title": "Database Economatica", "icon": ":material/database:"},
        ],
        "One Pager Fundos": [
            {"title": "1. Baixar CDI", "icon": ":material/download:"},
            {"title": "2. Carteira Fundos", "icon": ":material/account_balance_wallet:"},
            {"title": "3. Rent. Fundos", "icon": ":material/trending_up:"},
            {"title": "4. Gerar Excel One Pager", "icon": ":material/description:"},
            {"title": "5. Gerar HTML/PDF", "icon": ":material/html:"},
            {"title": "6. Puxar Lâmina", "icon": ":material/description:"},
        ],
        "Formatação": [
            {"title": "Texto Carteira Ações", "icon": ":material/article:"},
            {"title": "Texto Asset Allocation", "icon": ":material/file_download:"},
            {"title": "Tabela Asset Allocation", "icon": ":material/table_chart:"},
        ],
        "Gorila/Relatórios de Risco": [
            {"title": "Gorila API Novo", "icon": ":material/api:"},
            {"title": "Relatório de Crédito", "icon": ":material/account_balance:"},
            {"title": "Add Ratings BR", "icon": ":material/star:"},
            {"title": "Rating to Fitch", "icon": ":material/translate:"},
        ],
    }

    # Renderiza SEÇÕES em colunas (cada coluna é uma seção)
    # Ajuste n_cols conforme preferir (2 ou 3).
    n_cols = 3
    section_items = list(sections.items())

    # Quebra a lista de seções em linhas de n_cols colunas
    for row_start in range(0, len(section_items), n_cols):
        row = section_items[row_start : row_start + n_cols]
        cols = st.columns(len(row))
        for c, (section_name, items) in zip(cols, row):
            with c:
                # Título da seção
                if section_name == "":
                    st.subheader("Home e Início")
                else:
                    st.subheader(section_name)

                # Lista das páginas da seção (subseções apenas como bullet list)
                for it in items:
                    icon = it.get("icon", "")
                    title = it.get("title", "Sem título")
                    c.markdown(f"- {icon} {title}")

    st.info(
        "Dica: Utilize o menu superior para abrir as páginas. "
        "Este resumo é apenas informativo."
    )


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
            "pages/4_Financial_Comparator_Plotly.py",
            title="Comparador de Ações",
            icon=":material/bar_chart:",
        ),
        st.Page(
            "pages/24_Comparador_Acoes_Novo.py",
            title="Comparador de Ações Novo",
            icon=":material/bar_chart:",
        ),
        st.Page(
            "pages/16_API_4int.py",
            title="API 4intelligence",
            icon=":material/api:",
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
            "pages/29_Database_Economatica.py",
            title="Database Economatica",
            icon=":material/database:",
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
            icon=":material/translate:",
        )

    ],
}

pg = st.navigation(pages, position="top")
pg.run()