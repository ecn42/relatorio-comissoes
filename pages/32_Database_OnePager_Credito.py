import streamlit as st
import pandas as pd
import sqlite3
import os
import time

# --- Configuration ---
st.set_page_config(page_title="Database OnePager Credito", layout="wide")
DB_PATH = "databases/onepager_credito.db"

# --- Sidebar info ---
if os.path.exists(DB_PATH):
    creation_time = os.path.getmtime(DB_PATH)
    dt_creation = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(creation_time))
    size_mb = os.path.getsize(DB_PATH) / (1024 * 1024)
    
    st.sidebar.info(f"**Estado Atual do DB**\n\n📅 Data: {dt_creation}\n💾 Tamanho: {size_mb:.2f} MB")
    
    # Show tables in sidebar
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        conn.close()
        
        if tables:
            st.sidebar.markdown("### Tabelas Atuais")
            for t in tables:
                st.sidebar.text(f"• {t[0]}")
    except:
        pass
else:
    st.sidebar.warning("Banco de dados ainda não existe.")

st.title("Onepager de Crédito")
st.markdown("---")

st.info("ℹ️ Utilize esta ferramenta para atualizar a base de dados do Onepager de Crédito.")

# --- File Uploader ---
uploaded_file = st.file_uploader("Selecione o arquivo Excel (.xlsx, .xls)", type=["xlsx", "xls"])

if uploaded_file is not None:
    try:
        # Read Excel skipping first 3 rows, using 4th row (index 3) as header
        # Using openpyxl engine is standard for xlsx
        df = pd.read_excel(uploaded_file, header=3)
        
        # Determine target table based on columns
        # Normalize columns: remove newlines and extra spaces for matching
        cleaned_columns = [" ".join(str(c).split()) for c in df.columns]
        
        # Check if ANY column contains the target string (substring match)
        def has_column_match(target_substring, columns):
            target_clean = " ".join(target_substring.split())
            for col in columns:
                if target_clean.lower() in col.lower():
                    return True
            return False

        if has_column_match("NAICS", cleaned_columns):
            target_table = "main_cred"
            table_desc = "Tabela Principal (main_cred)"
        elif has_column_match("PU ajust p/ prov", cleaned_columns):
            target_table = "PU"
            table_desc = "Tabela de PU (PU ajust p/ prov)"
        elif has_column_match("% PU Par", cleaned_columns):
            target_table = "PU_Percent"
            table_desc = "Tabela de % PU Par"
        elif has_column_match("Taxa (YTM)", cleaned_columns):
            target_table = "YTM"
            table_desc = "Tabela de YTM (Taxa (YTM))"
        else:
            target_table = "main_cred"
            table_desc = "Tabela Principal (main_cred)"

            
        # Rename columns for specific tables (Last token only)
        if target_table in ["PU", "PU_Percent", "YTM"]:
            new_columns = []
            for col in df.columns:
                # Take the last token (e.g. "Taxa (YTM)\nVALE3" -> "VALE3")
                # "Data" remains "Data"
                new_col = str(col).strip().split()[-1]
                new_columns.append(new_col)
            df.columns = new_columns
            st.info(f"ℹ️ Colunas renomeadas automaticamente para simplificação (ex: {new_columns[1] if len(new_columns)>1 else '...'})")

        st.success(f"Arquivo identificado para atualização de: **{table_desc}**")
        st.write(f"Tabela Alvo: `{target_table}`")
        
        st.markdown("### Pré-visualização (Primeiras 5 linhas)")
        st.dataframe(df.head())
        
        st.markdown(f"Tem certeza que deseja substituir a tabela **{target_table}** com os dados acima?")
        
        if st.button("🚨 CONFIRMAR E ATUALIZAR TABELA", type="primary"):
            with st.status(f"Atualizando {target_table}...", expanded=True) as status:
                
                # Ensure dir exists
                os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
                
                try:
                    conn = sqlite3.connect(DB_PATH)
                    
                    # Write to SQL - replace functionality
                    df.to_sql(target_table, conn, if_exists='replace', index=False)
                    
                    conn.close()
                    
                    status.update(label="Concluído!", state="complete", expanded=False)
                    st.success(f"✅ Tabela `{target_table}` atualizada com sucesso no banco de dados `{DB_PATH}`!")
                    
                    # Show summary of new DB state for that table
                    conn = sqlite3.connect(DB_PATH)
                    cursor = conn.cursor()
                    cursor.execute(f"SELECT COUNT(*) FROM {target_table}")
                    row_count = cursor.fetchone()[0]
                    conn.close()
                    st.info(f"A nova tabela `{target_table}` possui {row_count} linhas.")
                    
                except Exception as e:
                    st.error(f"Erro crítico durante a atualização do banco de dados: {e}")
                    if 'conn' in locals(): conn.close()
                    
    except Exception as e:
        st.error(f"Erro ao processar o arquivo: {e}")
