import streamlit as st
import pandas as pd
import sqlite3
import os
import time

# --- Configuration ---
st.set_page_config(page_title="Database Economatica", layout="wide")
DB_PATH = "databases/dados_economatico.db"

# --- Configuration ---
st.set_page_config(page_title="Database Economatica", layout="wide")
DB_PATH = "databases/dados_economatico.db"

st.title("📂 Database Manager: Economatica")
st.markdown("---")

st.info("ℹ️ Utilize esta ferramenta para atualizar a base de dados do Economatica. O arquivo Excel enviado irá **SUBSTITUIR COMPLETAMENTE** o banco de dados atual.")

# --- File Uploader ---
uploaded_file = st.file_uploader("Selecione o arquivo Excel (.xlsx, .xls)", type=["xlsx", "xls"])

if uploaded_file is not None:
    st.markdown("### Pré-visualização")
    
    try:
        # Read all sheets to show info
        xl = pd.ExcelFile(uploaded_file)
        sheet_names = xl.sheet_names
        
        st.write(f"**Abas encontradas:** {', '.join(sheet_names)}")
        
        # Preview first sheet/rows - Skip first 3 rows (0, 1, 2), header is row 3
        df_preview = pd.read_excel(uploaded_file, sheet_name=sheet_names[0], header=3, nrows=5)
        st.dataframe(df_preview)
        
        st.markdown("---")
        col1, col2 = st.columns([1, 4])
        
        if col1.button("🚨 PROCESSAR E SUBSTITUIR DB", type="primary"):
            with st.status("Processando...", expanded=True) as status:
                progress_bar = st.progress(0)
                
                # 1. Close existing connections/Ensure dir
                st.write("Preparando ambiente...")
                os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
                
                # 2. Delete old DB if exists
                if os.path.exists(DB_PATH):
                    try:
                        os.remove(DB_PATH)
                        st.write(f"Banco de dados antigo removido: `{DB_PATH}`")
                    except Exception as e:
                        st.error(f"Erro ao remover banco antigo: {e}")
                        st.stop()
                
                # 3. Process Excel
                st.write("Lendo arquivo Excel e processando abas (Header: Linha 3)...")
                progress_step = 100 / len(sheet_names)
                current_progress = 0
                
                try:
                    conn = sqlite3.connect(DB_PATH)
                    
                    # Iterate through all sheets
                    for sheet in sheet_names:
                        st.write(f"Importando aba: **{sheet}**")
                        # Clean table name (remove spaces, special chars)
                        table_name = sheet.strip().lower().replace(" ", "_").replace("-", "_")
                        
                        # Read with header=3 (skipping 0, 1, 2)
                        df = pd.read_excel(uploaded_file, sheet_name=sheet, header=3)
                        
                        # Write to SQL
                        df.to_sql(table_name, conn, if_exists='replace', index=False)
                        
                        current_progress += progress_step
                        progress_bar.progress(int(min(current_progress, 100)))
                    
                    conn.close()
                    progress_bar.progress(100)
                    status.update(label="Concluído!", state="complete", expanded=False)
                    
                    st.success(f"✅ Banco de dados atualizado com sucesso! O arquivo `{DB_PATH}` foi recriado com {len(sheet_names)} tabela(s).")
                    
                    # Show summary of new DB
                    with st.expander("Verificar novo banco de dados"):
                        conn = sqlite3.connect(DB_PATH)
                        cursor = conn.cursor()
                        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                        tables = cursor.fetchall()
                        conn.close()
                        
                        st.write(" **Tabelas criadas:**")
                        for t in tables:
                            st.code(t[0])
                            
                except Exception as e:
                    st.error(f"Erro crítico durante o processamento: {e}")
                    if 'conn' in locals(): conn.close()
                    
    except Exception as e:
        st.error(f"Erro ao ler o arquivo Excel: {e}")

else:
    # Check if DB exists and show status
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
