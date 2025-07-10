import streamlit as st
import pandas as pd
import sqlite3
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import os
from streamlit_extras.dataframe_explorer import dataframe_explorer

# Enhanced color palette for better visualizations
COLORS = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e', 
    'success': '#2ca02c',
    'warning': '#d62728',
    'info': '#9467bd',
    'light': '#17becf',
    'dark': '#8c564b'
}

# Brazilian number formatting
def format_currency(value):
    """Format currency in Brazilian format"""
    if pd.isna(value) or value is None:
        return "R$ 0,00"
    return f"R$ {value:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.')

def format_number(value):
    """Format numbers in Brazilian format"""
    if pd.isna(value) or value is None:
        return "0"
    return f"{value:,.0f}".replace(',', '.')

class CommissionDataManager:
    """
    This class handles all database operations for our commission data.
    We use a class to organize our code better and avoid repeating database connection logic.
    """
    
    def __init__(self, db_path="/mnt/databases/commission_data.db"):
        """
        Initialize the database connection and create tables if they don't exist.
        
        Args:
            db_path (str): Path to the SQLite database file
        """
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """
        Create the main table if it doesn't exist.
        We define all columns with appropriate data types.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check if table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='commission_data'")
        table_exists = cursor.fetchone()
        
        if table_exists:
            # Check if document_type column exists
            cursor.execute("PRAGMA table_info(commission_data)")
            existing_columns = [col[1] for col in cursor.fetchall()]
            
            if 'document_type' not in existing_columns:
                cursor.execute("ALTER TABLE commission_data ADD COLUMN document_type TEXT DEFAULT 'original'")
                st.info("✅ Coluna 'document_type' adicionada à tabela")
            
            # Check if the table has the problematic UNIQUE constraint
            cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='commission_data'")
            table_sql = cursor.fetchone()[0]
            
            if "UNIQUE(" in table_sql:
                st.warning("🔧 Esquema antigo do banco detectado com restrições. Recriando tabela...")
                
                # Backup existing data
                cursor.execute("SELECT * FROM commission_data")
                existing_data = cursor.fetchall()
                
                # Get column names
                cursor.execute("PRAGMA table_info(commission_data)")
                columns_info = cursor.fetchall()
                column_names = [col[1] for col in columns_info if col[1] != 'id']  # Exclude auto-increment ID
                
                # Drop the old table
                cursor.execute("DROP TABLE commission_data")
                
                # Create new table WITHOUT unique constraints
                cursor.execute('''
                    CREATE TABLE commission_data (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        categoria TEXT,
                        produto TEXT,
                        nivel_1 TEXT,
                        nivel_2 TEXT,
                        nivel_3 TEXT,
                        cod_cliente TEXT,
                        data TEXT,
                        receita_rs REAL,
                        receita_liquida_rs REAL,
                        repasse_perc_escritorio REAL,
                        comissao_bruta_rs_escritorio REAL,
                        cod_assessor_direto TEXT,
                        repasse_perc_assessor_direto REAL,
                        comissao_rs_assessor_direto REAL,
                        cod_assessor_indireto_i TEXT,
                        repasse_perc_assessor_indireto_i REAL,
                        comissao_rs_assessor_indireto_i REAL,
                        cod_assessor_indireto_ii TEXT,
                        repasse_perc_assessor_indireto_ii REAL,
                        comissao_rs_assessor_indireto_ii REAL,
                        cod_assessor_indireto_iii TEXT,
                        repasse_perc_assessor_indireto_iii REAL,
                        comissao_rs_assessor_indireto_iii REAL,
                        month_year TEXT,
                        document_type TEXT DEFAULT 'original'
                    )
                ''')
                
                # Restore data if any existed
                if existing_data:
                    # Prepare insert statement (excluding the auto-increment id)
                    placeholders = ','.join(['?' for _ in column_names])
                    insert_sql = f"INSERT INTO commission_data ({','.join(column_names)}) VALUES ({placeholders})"
                    
                    # Insert data
                    for row in existing_data:
                        # Skip the ID column (first column) from the old data
                        cursor.execute(insert_sql, row[1:])
                    
                    st.success(f"✅ Tabela recriada e {len(existing_data)} registros restaurados")
                else:
                    st.success("✅ Nova tabela criada com esquema melhorado")
            else:
                st.info("Esquema do banco de dados já está atualizado")
        else:
            # Create new table without unique constraints
            cursor.execute('''
                CREATE TABLE commission_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    categoria TEXT,
                    produto TEXT,
                    nivel_1 TEXT,
                    nivel_2 TEXT,
                    nivel_3 TEXT,
                    cod_cliente TEXT,
                    data TEXT,
                    receita_rs REAL,
                    receita_liquida_rs REAL,
                    repasse_perc_escritorio REAL,
                    comissao_bruta_rs_escritorio REAL,
                    cod_assessor_direto TEXT,
                    repasse_perc_assessor_direto REAL,
                    comissao_rs_assessor_direto REAL,
                    cod_assessor_indireto_i TEXT,
                    repasse_perc_assessor_indireto_i REAL,
                    comissao_rs_assessor_indireto_i REAL,
                    cod_assessor_indireto_ii TEXT,
                    repasse_perc_assessor_indireto_ii REAL,
                    comissao_rs_assessor_indireto_ii REAL,
                    cod_assessor_indireto_iii TEXT,
                    repasse_perc_assessor_indireto_iii REAL,
                    comissao_rs_assessor_indireto_iii REAL,
                    month_year TEXT,
                    document_type TEXT DEFAULT 'original'
                )
            ''')
            st.success("✅ Nova tabela do banco de dados criada")
        
        conn.commit()
        conn.close()

    def detect_file_type(self, df):
        """Detect if the file is P2 format or original format with improved logic."""
        
        # Convert column names to lowercase for comparison
        columns_lower = [col.lower().strip() for col in df.columns]
        
        # P2 format indicators (more specific)
        p2_indicators = [
            'código assessor', 'codigo assessor',
            'comissão escritório', 'comissao escritorio', 
            'receita bruta'
        ]
        
        # Original format indicators  
        original_indicators = [
            'cód. assessor direto', 'cod. assessor direto',
            'comissão bruta (r$) escritório', 'comissao bruta (r$) escritorio'
        ]
        
        # Count matches
        p2_matches = sum(1 for indicator in p2_indicators if any(indicator in col for col in columns_lower))
        original_matches = sum(1 for indicator in original_indicators if any(indicator in col for col in columns_lower))
        
        # Debug output
        st.write(f"**Detecção de formato:**")
        st.write(f"- Indicadores P2 encontrados: {p2_matches}")
        st.write(f"- Indicadores Original encontrados: {original_matches}")
        
        return "p2" if p2_matches >= 1 else "original"
    
    def fix_client_codes_data_types(self):
        """
        Fix client codes that are stored as floats (with .0) to be integers.
        This resolves filtering and comparison issues.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            st.info("🔧 Verificando e corrigindo tipos de dados dos códigos de cliente...")
            
            # First, let's see what we're dealing with
            cursor.execute("SELECT DISTINCT cod_cliente FROM commission_data WHERE cod_cliente IS NOT NULL LIMIT 20")
            sample_codes = cursor.fetchall()
            
            st.write("**Códigos de cliente antes da correção:**")
            st.write([code[0] for code in sample_codes])
            
            # Update all client codes to remove .0 if they're stored as floats
            cursor.execute("""
                UPDATE commission_data 
                SET cod_cliente = CAST(CAST(cod_cliente AS REAL) AS INTEGER)
                WHERE cod_cliente IS NOT NULL 
                AND cod_cliente LIKE '%.0'
            """)
            
            updated_rows = cursor.rowcount
            
            # Also fix the assessor codes that might have the same issue
            cursor.execute("""
                UPDATE commission_data 
                SET cod_assessor_direto = CAST(CAST(cod_assessor_direto AS REAL) AS INTEGER)
                WHERE cod_assessor_direto IS NOT NULL 
                AND cod_assessor_direto LIKE '%.0'
            """)
            
            updated_assessor_rows = cursor.rowcount
            
            # Fix indirect assessor codes too
            for col in ['cod_assessor_indireto_i', 'cod_assessor_indireto_ii', 'cod_assessor_indireto_iii']:
                cursor.execute(f"""
                    UPDATE commission_data 
                    SET {col} = CAST(CAST({col} AS REAL) AS INTEGER)
                    WHERE {col} IS NOT NULL 
                    AND {col} LIKE '%.0'
                """)
            
            conn.commit()
            
            # Show results after fix
            cursor.execute("SELECT DISTINCT cod_cliente FROM commission_data WHERE cod_cliente IS NOT NULL LIMIT 20")
            sample_codes_after = cursor.fetchall()
            
            st.success(f"✅ {updated_rows} códigos de cliente corrigidos")
            st.success(f"✅ {updated_assessor_rows} códigos de assessor corrigidos")
            
            st.write("**Códigos de cliente após correção:**")
            st.write([code[0] for code in sample_codes_after])
            
            return True
            
        except Exception as e:
            st.error(f"Erro ao corrigir códigos de cliente: {str(e)}")
            conn.rollback()
            return False
        finally:
            conn.close()

    def clean_data_types_on_insert(self, df):
        """
        Clean data types before inserting to prevent float/int issues and mixed type problems.
        
        Args:
            df (DataFrame): DataFrame to clean
            
        Returns:
            DataFrame: Cleaned DataFrame
        """
        df_clean = df.copy()
        
        # List of columns that should be integers (client and assessor codes)
        code_columns = [
            'cod_cliente', 'cod_assessor_direto', 'cod_assessor_indireto_i', 
            'cod_assessor_indireto_ii', 'cod_assessor_indireto_iii'
        ]
        
        for col in code_columns:
            if col in df_clean.columns:
                # Convert to numeric first, then to integer, handling NaN values
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                # Convert to integer, but keep NaN as None for database
                df_clean[col] = df_clean[col].apply(
                    lambda x: str(int(x)) if pd.notna(x) else None
                )
        
        # List of columns that should be strings (text columns)
        text_columns = [
            'categoria', 'produto', 'nivel_1', 'nivel_2', 'nivel_3'
        ]
        
        for col in text_columns:
            if col in df_clean.columns:
                # Ensure these are strings and handle NaN properly
                df_clean[col] = df_clean[col].astype(str)
                df_clean[col] = df_clean[col].replace('nan', None)
                df_clean[col] = df_clean[col].replace('NaN', None)
                df_clean[col] = df_clean[col].where(df_clean[col] != 'None', None)
        
        # List of columns that should be numeric (financial data)
        numeric_columns = [
            'receita_rs', 'receita_liquida_rs', 'repasse_perc_escritorio',
            'comissao_bruta_rs_escritorio', 'repasse_perc_assessor_direto',
            'comissao_rs_assessor_direto', 'repasse_perc_assessor_indireto_i',
            'comissao_rs_assessor_indireto_i', 'repasse_perc_assessor_indireto_ii',
            'comissao_rs_assessor_indireto_ii', 'repasse_perc_assessor_indireto_iii',
            'comissao_rs_assessor_indireto_iii'
        ]
        
        for col in numeric_columns:
            if col in df_clean.columns:
                # Convert to numeric, coercing errors to NaN
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
        return df_clean
    
    def process_xlsx_file(self, file_path_or_buffer):
        """
        Process an Excel file and extract month/year information.
        Now supports both original and P2 formats with improved error handling.
        """
        try:
            # Read the Excel file
            df = pd.read_excel(file_path_or_buffer)
            
            # Debug: Let's see what columns we actually have
            st.write("**Colunas encontradas no arquivo:**")
            st.write(list(df.columns))
            
            # Clean column names
            df.columns = df.columns.str.strip()
            
            # Detect file type
            file_type = self.detect_file_type(df)
            st.info(f"📄 **Tipo de arquivo detectado:** {file_type.upper()}")
            
            # Process based on file type
            if file_type == "p2":
                return self._process_p2_file(df)
            else:
                return self._process_original_file(df)
                
        except Exception as e:
            st.error(f"Erro ao processar arquivo: {str(e)}")
            st.exception(e)  # Show full traceback for debugging
            return None, None, 0

    def _process_original_file(self, df):
        """Process original format files (your existing logic)."""
        # Check if we have the expected 'Data' column
        if 'Data' not in df.columns:
            date_columns = [col for col in df.columns if 'data' in col.lower() or 'date' in col.lower()]
            if date_columns:
                st.warning(f"Coluna 'Data' não encontrada. Encontradas estas colunas similares: {date_columns}")
                st.info("Por favor, verifique os nomes das colunas no seu arquivo Excel.")
            else:
                st.error("Nenhuma coluna de data encontrada no arquivo.")
            return None, None, 0
        
        # Convert 'Data' column to datetime
        df['Data'] = pd.to_datetime(df['Data'], format='%d/%m/%Y', errors='coerce')
        
        # Check if we have any valid dates
        valid_dates = df['Data'].dropna()
        if len(valid_dates) == 0:
            st.error("Nenhuma data válida encontrada na coluna 'Data'.")
            return None, None, 0
        
        # Extract month and year from the first valid date
        first_date = valid_dates.iloc[0]
        month_year = first_date.strftime('%Y-%m')
        
        # Add month_year and document_type columns
        df['month_year'] = month_year
        df['document_type'] = 'original'
        
        # Show some statistics about the data
        st.write(f"**Prévia dos dados ORIGINAIS para {month_year}:**")
        st.write(f"- Total de linhas: {format_number(len(df))}")
        st.write(f"- Datas válidas: {format_number(len(valid_dates))}")
        st.write(f"- Período: {valid_dates.min().strftime('%d/%m/%Y')} até {valid_dates.max().strftime('%d/%m/%Y')}")
        
        return df, month_year, len(df)

    def _process_p2_file(self, df):
        """Process P2 format files with improved error handling and column detection."""
        try:
            # Debug: Show actual columns in P2 file
            st.write("**Colunas encontradas no arquivo P2:**")
            for i, col in enumerate(df.columns):
                st.write(f"{i+1}. '{col}'")
            
            # Find date column with more flexible matching
            date_column = None
            possible_date_names = ['data', 'date', 'dt']
            
            for col in df.columns:
                col_lower = col.lower().strip()
                if any(date_name in col_lower for date_name in possible_date_names):
                    date_column = col
                    st.info(f"✅ Coluna de data encontrada: '{col}'")
                    break
            
            if date_column is None:
                st.error("❌ Nenhuma coluna de data encontrada no arquivo P2.")
                st.write("**Colunas disponíveis:**", list(df.columns))
                return None, None, 0
            
            # Show sample of date values before conversion
            st.write(f"**Amostra de valores na coluna '{date_column}' (primeiros 10):**")
            sample_dates = df[date_column].head(10).tolist()
            for i, date_val in enumerate(sample_dates):
                st.write(f"{i+1}. '{date_val}' (tipo: {type(date_val).__name__})")
            
            # Store original date column for backup
            original_date_values = df[date_column].copy()
            
            # Convert date column to datetime with multiple format attempts
            date_formats = [
                '%d/%m/%Y',      # 31/12/2023
                '%Y-%m-%d',      # 2023-12-31
                '%d-%m-%Y',      # 31-12-2023
                '%m/%d/%Y',      # 12/31/2023
                '%d/%m/%y',      # 31/12/23
                '%Y/%m/%d',      # 2023/12/31
                '%d.%m.%Y',      # 31.12.2023
                '%Y.%m.%d',      # 2023.12.31
            ]
            
            converted_dates = None
            successful_format = None
            
            for fmt in date_formats:
                try:
                    st.write(f"🔄 Tentando formato: {fmt}")
                    test_conversion = pd.to_datetime(original_date_values, format=fmt, errors='coerce')
                    valid_count = test_conversion.notna().sum()
                    st.write(f"   - Datas válidas com este formato: {valid_count}/{len(original_date_values)}")
                    
                    if valid_count > 0:
                        converted_dates = test_conversion
                        successful_format = fmt
                        st.success(f"✅ Formato de data detectado: {fmt} ({valid_count} datas válidas)")
                        break
                except Exception as e:
                    st.write(f"   - Erro com formato {fmt}: {str(e)}")
                    continue
            
            # If no specific format worked, try automatic detection
            if converted_dates is None or converted_dates.notna().sum() == 0:
                st.write("🔄 Tentando detecção automática de formato...")
                try:
                    converted_dates = pd.to_datetime(original_date_values, errors='coerce', infer_datetime_format=True)
                    valid_count = converted_dates.notna().sum()
                    st.write(f"   - Datas válidas com detecção automática: {valid_count}/{len(original_date_values)}")
                    if valid_count > 0:
                        successful_format = "detecção automática"
                        st.success(f"✅ Detecção automática funcionou: {valid_count} datas válidas")
                except Exception as e:
                    st.write(f"   - Erro na detecção automática: {str(e)}")
            
            # If still no success, try converting to string first and then parsing
            if converted_dates is None or converted_dates.notna().sum() == 0:
                st.write("🔄 Tentando conversão via string...")
                try:
                    # Convert to string first, then try parsing
                    string_dates = original_date_values.astype(str)
                    
                    # Show sample of string conversion
                    st.write("**Amostra após conversão para string:**")
                    for i, str_date in enumerate(string_dates.head(5)):
                        st.write(f"{i+1}. '{str_date}'")
                    
                    # Try parsing the string dates
                    for fmt in date_formats:
                        try:
                            test_conversion = pd.to_datetime(string_dates, format=fmt, errors='coerce')
                            valid_count = test_conversion.notna().sum()
                            if valid_count > 0:
                                converted_dates = test_conversion
                                successful_format = f"{fmt} (via string)"
                                st.success(f"✅ Conversão via string funcionou com formato {fmt}: {valid_count} datas válidas")
                                break
                        except:
                            continue
                    
                    # If still no success, try automatic on strings
                    if converted_dates is None or converted_dates.notna().sum() == 0:
                        converted_dates = pd.to_datetime(string_dates, errors='coerce', infer_datetime_format=True)
                        valid_count = converted_dates.notna().sum()
                        if valid_count > 0:
                            successful_format = "detecção automática via string"
                            st.success(f"✅ Detecção automática via string funcionou: {valid_count} datas válidas")
                            
                except Exception as e:
                    st.write(f"   - Erro na conversão via string: {str(e)}")
            
            # Final check
            if converted_dates is None:
                converted_dates = pd.Series([pd.NaT] * len(df))
            
            # Update the dataframe with converted dates
            df[date_column] = converted_dates
            
            # Rename the date column to 'Data' for consistency
            if date_column != 'Data':
                df = df.rename(columns={date_column: 'Data'})
            
            # Check for valid dates
            valid_dates = df['Data'].dropna()
            if len(valid_dates) == 0:
                st.error("❌ Nenhuma data válida encontrada após todas as tentativas de conversão.")
                
                # Show detailed debugging info
                st.write("**Debugging - Valores originais que falharam:**")
                unique_values = original_date_values.unique()[:10]  # Show first 10 unique values
                for val in unique_values:
                    st.write(f"- '{val}' (tipo: {type(val).__name__})")
                
                # Ask user to check the file
                st.info("💡 **Sugestões:**")
                st.write("1. Verifique se a coluna de data contém valores válidos")
                st.write("2. Formatos suportados: DD/MM/YYYY, YYYY-MM-DD, DD-MM-YYYY, etc.")
                st.write("3. Certifique-se de que não há células vazias ou texto na coluna de data")
                
                return None, None, 0
            
            # Extract month and year
            first_date = valid_dates.iloc[0]
            month_year = first_date.strftime('%Y-%m')
            
            # Add month_year and document_type columns
            df['month_year'] = month_year
            df['document_type'] = 'p2'
            
            # Show statistics
            st.write(f"**Prévia dos dados P2 para {month_year}:**")
            st.write(f"- Total de linhas: {format_number(len(df))}")
            st.write(f"- Datas válidas: {format_number(len(valid_dates))}")
            st.write(f"- Formato usado: {successful_format}")
            st.write(f"- Período: {valid_dates.min().strftime('%d/%m/%Y')} até {valid_dates.max().strftime('%d/%m/%Y')}")
            
            # Show sample of successfully converted dates
            st.write("**Amostra de datas convertidas com sucesso:**")
            sample_converted = valid_dates.head(5)
            for i, date_val in enumerate(sample_converted):
                st.write(f"{i+1}. {date_val.strftime('%d/%m/%Y')}")
            
            return df, month_year, len(df)
            
        except Exception as e:
            st.error(f"❌ Erro ao processar arquivo P2: {str(e)}")
            st.exception(e)  # Show full traceback for debugging
            return None, None, 0

    def insert_data(self, df, month_year):
        """
        Insert data into the database, handling month-level duplicates properly.
        """
        conn = sqlite3.connect(self.db_path)
        
        try:
            # Check if this month already exists
            existing_months_query = "SELECT DISTINCT month_year FROM commission_data WHERE month_year = ?"
            existing_check = pd.read_sql_query(existing_months_query, conn, params=[month_year])
            month_exists = len(existing_check) > 0
            
            if month_exists:
                # Get count of existing records
                count_query = "SELECT COUNT(*) as count FROM commission_data WHERE month_year = ?"
                existing_count = pd.read_sql_query(count_query, conn, params=[month_year]).iloc[0]['count']
                
                st.warning(f"⚠️ **Mês {month_year} já existe no banco de dados!**")
                st.info(f"📊 Registros existentes para {month_year}: {format_number(existing_count)}")
                st.info(f"📊 Novos registros para inserir: {format_number(len(df))}")
                
                # Create unique operation key
                import time
                unique_id = f"{month_year}_{len(df)}_{int(time.time() * 1000) % 10000}"
                operation_key = f"db_choice_{unique_id}"
                
                # Check if user has made a choice
                user_choice = st.session_state.get(operation_key, None)
                
                if user_choice is None:
                    # Show choice buttons
                    st.markdown("**Escolha uma ação:**")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if st.button(f"🔄 Substituir {month_year}", key=f"replace_{unique_id}"):
                            st.session_state[operation_key] = "replace"
                            st.rerun()
                    
                    with col2:
                        if st.button(f"➕ Adicionar a {month_year}", key=f"append_{unique_id}"):
                            st.session_state[operation_key] = "append"
                            st.rerun()
                    
                    with col3:
                        if st.button(f"⏭️ Pular {month_year}", key=f"skip_{unique_id}"):
                            st.session_state[operation_key] = "skip"
                            st.rerun()
                    
                    # Return 0 to indicate waiting for user choice
                    return 0
                
                else:
                    # User has made a choice, execute it
                    if user_choice == "replace":
                        st.info(f"🔄 Substituindo dados para {month_year}...")
                        cursor = conn.cursor()
                        cursor.execute("DELETE FROM commission_data WHERE month_year = ?", (month_year,))
                        deleted_count = cursor.rowcount
                        conn.commit()
                        st.success(f"🗑️ {format_number(deleted_count)} registros deletados.")
                        
                        # Insert new records
                        inserted_count = self._insert_all_records(df, month_year, conn)
                        
                    elif user_choice == "append":
                        st.info(f"➕ Adicionando dados para {month_year}...")
                        inserted_count = self._insert_all_records(df, month_year, conn)
                        
                    elif user_choice == "skip":
                        st.info(f"⏭️ Dados para {month_year} foram pulados.")
                        inserted_count = 0
                    
                    # Clear the choice from session state
                    if operation_key in st.session_state:
                        del st.session_state[operation_key]
                    
                    return inserted_count
            
            else:
                # New month, insert directly
                st.success(f"✅ Novo mês {month_year} detectado. Inserindo registros...")
                return self._insert_all_records(df, month_year, conn)
                        
        except Exception as e:
            st.error(f"Erro ao inserir dados: {str(e)}")
            conn.rollback()
            return 0
        finally:
            conn.close()
            
    def _insert_all_records(self, df, month_year, conn):
        """
        Insert all records without duplicate checking (for new months or after replacement).
        
        Args:
            df (DataFrame): The processed data
            month_year (str): Month-year string for this data
            conn: Database connection
            
        Returns:
            int: Number of records inserted
        """
        try:
            # Prepare the DataFrame
            df_final = self._prepare_dataframe_for_insertion(df, month_year)
            
            if df_final is None or df_final.empty:
                st.error("❌ Erro na preparação dos dados para inserção.")
                return 0
            
            st.write("**Dados a serem inseridos (primeiras 3 linhas):**")
            # Create a safe display version
            display_df = df_final.head(3).copy()
            for col in display_df.columns:
                if display_df[col].dtype == 'object':
                    display_df[col] = display_df[col].astype(str)
            st.dataframe(display_df)
            
            # Use pandas to_sql for bulk insert (much faster)
            df_final.to_sql('commission_data', conn, if_exists='append', index=False)
            
            inserted_count = len(df_final)
            st.success(f"✅ {format_number(inserted_count)} registros inseridos com sucesso para {month_year}")
            
            return inserted_count
            
        except Exception as e:
            st.error(f"❌ Erro na inserção em lote: {str(e)}")
            st.exception(e)  # Show full traceback for debugging
            return 0

    def _prepare_dataframe_for_insertion(self, df, month_year):
        """
        Prepare DataFrame for database insertion by handling data types and mapping columns.
        Now supports both original and P2 formats with improved error handling.
        """
        try:
            # Check document type
            document_type = df['document_type'].iloc[0] if 'document_type' in df.columns else 'original'
            
            if document_type == 'p2':
                return self._prepare_p2_dataframe_for_insertion(df, month_year)
            else:
                return self._prepare_original_dataframe_for_insertion(df, month_year)
        except Exception as e:
            st.error(f"❌ Erro na preparação do DataFrame: {str(e)}")
            st.exception(e)
            return None

    def _prepare_original_dataframe_for_insertion(self, df, month_year):
        """Prepare original format DataFrame (your existing logic)."""
        try:
            # Clean data types first
            df = self.clean_data_types_on_insert(df)
            
            # Your existing column mapping
            column_mapping = {
                'Categoria': 'categoria',
                'Produto': 'produto',
                'Nível 1': 'nivel_1',
                'Nível 2': 'nivel_2',
                'Nível 3': 'nivel_3',
                'Cód. Cliente': 'cod_cliente',
                'Data': 'data',
                'Receita (R$)': 'receita_rs',
                'Receita Líquida (R$)': 'receita_liquida_rs',
                'Repasse (%) Escritório': 'repasse_perc_escritorio',
                'Comissão Bruta (R$) Escritório': 'comissao_bruta_rs_escritorio',
                'Cód. Assessor Direto': 'cod_assessor_direto',
                'Repasse (%) Assessor Direto': 'repasse_perc_assessor_direto',
                'Comissão (R$) Assessor Direto': 'comissao_rs_assessor_direto',
                'Cód. Assessor Indireto I': 'cod_assessor_indireto_i',
                'Repasse (%) Assessor Indireto I': 'repasse_perc_assessor_indireto_i',
                'Comissão (R$) Assessor Indireto I': 'comissao_rs_assessor_indireto_i',
                'Cód. Assessor Indireto II': 'cod_assessor_indireto_ii',
                'Repasse (%) Assessor Indireto II': 'repasse_perc_assessor_indireto_ii',
                'Comissão (R$) Assessor Indireto II': 'comissao_rs_assessor_indireto_ii',
                'Cód. Assessor Indireto III': 'cod_assessor_indireto_iii',
                'Repasse (%) Assessor Indireto III': 'repasse_perc_assessor_indireto_iii',
                'Comissão (R$) Assessor Indireto III': 'comissao_rs_assessor_indireto_iii',
                'month_year': 'month_year',
                'document_type': 'document_type'
            }
            
            # Only map columns that exist in the DataFrame
            existing_mapping = {k: v for k, v in column_mapping.items() if k in df.columns}
            
            # Rename columns to match database schema
            df_mapped = df.rename(columns=existing_mapping)
            
            # Add missing columns with None values for database compatibility
            db_columns = [
                'categoria', 'produto', 'nivel_1', 'nivel_2', 'nivel_3', 'cod_cliente',
                'data', 'receita_rs', 'receita_liquida_rs', 'repasse_perc_escritorio',
                'comissao_bruta_rs_escritorio', 'cod_assessor_direto', 'repasse_perc_assessor_direto',
                'comissao_rs_assessor_direto', 'cod_assessor_indireto_i', 'repasse_perc_assessor_indireto_i',
                'comissao_rs_assessor_indireto_i', 'cod_assessor_indireto_ii', 'repasse_perc_assessor_indireto_ii',
                'comissao_rs_assessor_indireto_ii', 'cod_assessor_indireto_iii', 'repasse_perc_assessor_indireto_iii',
                'comissao_rs_assessor_indireto_iii', 'month_year', 'document_type'
            ]
            
            for col in db_columns:
                if col not in df_mapped.columns:
                    df_mapped[col] = None
            
            # Select only the columns that exist in our database schema
            df_final = df_mapped[db_columns].copy()
            
            # Convert datetime columns to string format for SQLite
            if 'data' in df_final.columns and df_final['data'].dtype == 'datetime64[ns]':
                df_final['data'] = df_final['data'].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # Handle NaN values properly
            df_final = self._clean_final_dataframe(df_final)
            
            return df_final
            
        except Exception as e:
            st.error(f"❌ Erro na preparação do DataFrame original: {str(e)}")
            st.exception(e)
            return None

    def _prepare_p2_dataframe_for_insertion(self, df, month_year):
        """Prepare P2 DataFrame for database insertion with improved column mapping and data cleaning."""
        
        try:
            # Clean data types first
            df = self.clean_data_types_on_insert(df)
            
            # Show columns before mapping for debugging
            st.write("**Colunas disponíveis no DataFrame P2:**")
            for i, col in enumerate(df.columns):
                st.write(f"{i+1}. '{col}'")
            
            # Improved P2 column mapping with exact matching first, then fuzzy matching
            p2_column_mapping = {}
            
            # Define exact mappings first (case-insensitive)
            exact_mappings = {
                'categoria': 'categoria',
                'código cliente': 'cod_cliente',
                'codigo cliente': 'cod_cliente',
                'código assessor': 'cod_assessor_direto',
                'codigo assessor': 'cod_assessor_direto',
                'data': 'data',
                'receita bruta': 'receita_rs',
                'receita líquida': 'receita_liquida_rs',
                'receita liquida': 'receita_liquida_rs',
                'comissão (%) escritório': 'repasse_perc_escritorio',
                'comissao (%) escritorio': 'repasse_perc_escritorio',
                'comissão escritório': 'comissao_bruta_rs_escritorio',
                'comissao escritorio': 'comissao_bruta_rs_escritorio',
                'nível 1': 'nivel_1',
                'nivel 1': 'nivel_1',
                'nível 2': 'nivel_2',
                'nivel 2': 'nivel_2',
                'nível 3': 'nivel_3',
                'nivel 3': 'nivel_3',
                'month_year': 'month_year',
                'document_type': 'document_type'
            }
            
            # First pass: exact matching
            for col in df.columns:
                col_lower = col.lower().strip()
                if col_lower in exact_mappings:
                    p2_column_mapping[col] = exact_mappings[col_lower]
                    st.info(f"✅ Mapeamento exato: '{col}' → '{exact_mappings[col_lower]}'")
            
            # Second pass: fuzzy matching for unmapped columns
            for col in df.columns:
                if col in p2_column_mapping:
                    continue  # Already mapped
                    
                col_lower = col.lower().strip()
                
                # Fuzzy matching rules
                if 'cliente' in col_lower and 'código' in col_lower:
                    p2_column_mapping[col] = 'cod_cliente'
                elif 'assessor' in col_lower and 'código' in col_lower:
                    p2_column_mapping[col] = 'cod_assessor_direto'
                elif 'receita' in col_lower and 'bruta' in col_lower:
                    p2_column_mapping[col] = 'receita_rs'
                elif 'receita' in col_lower and ('líquida' in col_lower or 'liquida' in col_lower):
                    p2_column_mapping[col] = 'receita_liquida_rs'
                elif 'comissão' in col_lower and 'escritório' in col_lower and '%' in col_lower:
                    p2_column_mapping[col] = 'repasse_perc_escritorio'
                elif 'comissão' in col_lower and 'escritório' in col_lower and '%' not in col_lower:
                    p2_column_mapping[col] = 'comissao_bruta_rs_escritorio'
                elif 'nível' in col_lower and '1' in col_lower:
                    p2_column_mapping[col] = 'nivel_1'
                elif 'nível' in col_lower and '2' in col_lower:
                    p2_column_mapping[col] = 'nivel_2'
                elif 'nível' in col_lower and '3' in col_lower:
                    p2_column_mapping[col] = 'nivel_3'
            
            st.write("**Mapeamento final de colunas P2:**")
            for original, mapped in p2_column_mapping.items():
                st.write(f"- '{original}' → '{mapped}'")
            
            # Apply the mapping
            df_mapped = df.rename(columns=p2_column_mapping)
            
            # Set produto = categoria for P2 files (if categoria exists)
            if 'categoria' in df_mapped.columns:
                df_mapped['produto'] = df_mapped['categoria']
            else:
                df_mapped['produto'] = 'P2 - Produto não especificado'
            
            # Define all required database columns
            db_columns = [
                'categoria', 'produto', 'nivel_1', 'nivel_2', 'nivel_3', 'cod_cliente',
                'data', 'receita_rs', 'receita_liquida_rs', 'repasse_perc_escritorio',
                'comissao_bruta_rs_escritorio', 'cod_assessor_direto', 'repasse_perc_assessor_direto',
                'comissao_rs_assessor_direto', 'cod_assessor_indireto_i', 'repasse_perc_assessor_indireto_i',
                'comissao_rs_assessor_indireto_i', 'cod_assessor_indireto_ii', 'repasse_perc_assessor_indireto_ii',
                'comissao_rs_assessor_indireto_ii', 'cod_assessor_indireto_iii', 'repasse_perc_assessor_indireto_iii',
                'comissao_rs_assessor_indireto_iii', 'month_year', 'document_type'
            ]
            
            # Add missing columns with None values
            for col in db_columns:
                if col not in df_mapped.columns:
                    df_mapped[col] = None
            
            # Select only the database columns
            df_final = df_mapped[db_columns].copy()
            
            # Handle data type conversions with proper cleaning
            if 'data' in df_final.columns and df_final['data'].dtype == 'datetime64[ns]':
                df_final['data'] = df_final['data'].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # Clean the final dataframe
            df_final = self._clean_final_dataframe(df_final)
            
            st.write("**Colunas com dados não-nulos:**")
            non_null_info = []
            for col in df_final.columns:
                non_null_count = df_final[col].notna().sum()
                if non_null_count > 0:
                    non_null_info.append(f"{col}: {non_null_count}")
            st.write(non_null_info)
            
            return df_final
            
        except Exception as e:
            st.error(f"❌ Erro na preparação do DataFrame P2: {str(e)}")
            st.exception(e)
            return None

    def _clean_final_dataframe(self, df_final):
        """
        Clean the final dataframe to ensure consistent data types and handle NaN values properly.
        
        Args:
            df_final (DataFrame): The dataframe to clean
            
        Returns:
            DataFrame: Cleaned dataframe
        """
        try:
            # Define column types
            numeric_columns = [
                'receita_rs', 'receita_liquida_rs', 'repasse_perc_escritorio', 
                'comissao_bruta_rs_escritorio', 'repasse_perc_assessor_direto',
                'comissao_rs_assessor_direto', 'repasse_perc_assessor_indireto_i',
                'comissao_rs_assessor_indireto_i', 'repasse_perc_assessor_indireto_ii',
                'comissao_rs_assessor_indireto_ii', 'repasse_perc_assessor_indireto_iii',
                'comissao_rs_assessor_indireto_iii'
            ]
            
            text_columns = [
                'categoria', 'produto', 'nivel_1', 'nivel_2', 'nivel_3', 'cod_cliente',
                'cod_assessor_direto', 'cod_assessor_indireto_i', 'cod_assessor_indireto_ii',
                'cod_assessor_indireto_iii', 'month_year', 'document_type', 'data'
            ]
            
            # Handle numeric columns
            for col in numeric_columns:
                if col in df_final.columns:
                    # Convert to numeric, coercing errors to NaN
                    df_final[col] = pd.to_numeric(df_final[col], errors='coerce')
                    # Replace NaN with None for SQLite compatibility
                    df_final[col] = df_final[col].where(pd.notna(df_final[col]), None)
            
            # Handle text columns
            for col in text_columns:
                if col in df_final.columns:
                    # Convert to string first
                    df_final[col] = df_final[col].astype(str)
                    # Replace various NaN representations with None
                    nan_values = ['nan', 'NaN', 'None', '<NA>', 'null', 'NULL']
                    for nan_val in nan_values:
                        df_final[col] = df_final[col].replace(nan_val, None)
                    # Replace empty strings with None
                    df_final[col] = df_final[col].replace('', None)
                    # Final check for pandas NaN
                    df_final[col] = df_final[col].where(pd.notna(df_final[col]), None)
            
            # Ensure all columns have consistent types
            for col in df_final.columns:
                if col in text_columns:
                    df_final[col] = df_final[col].astype('object')
                elif col in numeric_columns:
                    # Keep as float64 for numeric columns
                    df_final[col] = pd.to_numeric(df_final[col], errors='coerce')
            
            return df_final
            
        except Exception as e:
            st.error(f"❌ Erro na limpeza final do DataFrame: {str(e)}")
            st.exception(e)
            return df_final  # Return original if cleaning fails
    
    def get_available_months(self):
        """
        Get all available months in the database.
        
        Returns:
            list: List of month-year strings
        """
        conn = sqlite3.connect(self.db_path)
        try:
            query = "SELECT DISTINCT month_year FROM commission_data ORDER BY month_year"
            df = pd.read_sql_query(query, conn)
            return df['month_year'].tolist()
        finally:
            conn.close()
    
    def get_data_for_analysis(self, months=None, filters=None, document_types=None):
        """
        Retrieve data for analysis with optional filters.
        
        Args:
            months (list): List of months to include (None for all)
            filters (dict): Dictionary of filters to apply
            document_types (list): List of document types to include ["original", "p2"]
            
        Returns:
            DataFrame: Filtered data for analysis
        """
        conn = sqlite3.connect(self.db_path)
        try:
            query = "SELECT * FROM commission_data WHERE 1=1"
            params = []
            
            if months:
                placeholders = ','.join(['?' for _ in months])
                query += f" AND month_year IN ({placeholders})"
                params.extend(months)
            
            if document_types:
                placeholders = ','.join(['?' for _ in document_types])
                query += f" AND document_type IN ({placeholders})"
                params.extend(document_types)
            
            if filters:
                for column, values in filters.items():
                    if values:  # Only apply filter if values are selected
                        placeholders = ','.join(['?' for _ in values])
                        query += f" AND {column} IN ({placeholders})"
                        params.extend(values)
            
            df = pd.read_sql_query(query, conn, params=params)
            return df
        finally:
            conn.close()

def create_product_analysis(df):
    """Create enhanced analysis by product with Brazilian formatting."""
    st.subheader("📦 Análise de Comissões por Produto")
    df['cod_cliente'] = df['cod_cliente'].astype(str).str.replace('\.0$', '', regex=True)
    
    # Group by product and sum commissions
    product_summary = df.groupby('produto')['comissao_bruta_rs_escritorio'].agg(['sum', 'count', 'mean']).reset_index()
    product_summary.columns = ['Produto', 'Comissão Total', 'Qtd Transações', 'Comissão Média']
    
    # Sort by total commission
    product_summary = product_summary.sort_values('Comissão Total', ascending=False)
    
    # Format currency columns for display
    product_summary['Comissão Total Formatada'] = product_summary['Comissão Total'].apply(format_currency)
    product_summary['Comissão Média Formatada'] = product_summary['Comissão Média'].apply(format_currency)
    
    # Create enhanced bar chart
    fig = px.bar(
        product_summary.head(10),
        x='Produto',
        y='Comissão Total',
        title='Top 10 Produtos por Comissão Total',
        labels={'Comissão Total': 'Comissão (R$)', 'Produto': 'Produto'},
        color='Comissão Total',
        color_continuous_scale='Blues',
        text='Comissão Total'
    )
    
    # Enhanced formatting
    fig.update_traces(
        texttemplate='%{text:,.0f}',
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Comissão: R$ %{y:,.2f}<extra></extra>'
    )
    
    fig.update_layout(
        xaxis_tickangle=45,
        showlegend=False,
        height=500,
        font=dict(size=12),
        title_font_size=16,
        xaxis_title_font_size=14,
        yaxis_title_font_size=14
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show summary table with formatted values
    display_df = product_summary[['Produto', 'Comissão Total Formatada', 'Qtd Transações', 'Comissão Média Formatada']].copy()
    display_df.columns = ['Produto', 'Comissão Total', 'Transações', 'Comissão Média']
    st.dataframe(display_df, use_container_width=True)

def create_level1_analysis(df):
    """Create enhanced analysis by Level 1 with Brazilian formatting."""
    st.subheader("🏢 Análise de Comissões por Nível 1")
    df['cod_cliente'] = df['cod_cliente'].astype(str).str.replace('\.0$', '', regex=True)
    
    level1_summary = df.groupby('nivel_1')['comissao_bruta_rs_escritorio'].agg(['sum', 'count', 'mean']).reset_index()
    level1_summary.columns = ['Nível 1', 'Comissão Total', 'Qtd Transações', 'Comissão Média']
    level1_summary = level1_summary.sort_values('Comissão Total', ascending=False)
    
    # Create enhanced pie chart
    fig = px.pie(
        level1_summary,
        values='Comissão Total',
        names='Nível 1',
        title='Distribuição de Comissões por Nível 1',
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        hovertemplate='<b>%{label}</b><br>Comissão: R$ %{value:,.2f}<br>Percentual: %{percent}<extra></extra>'
    )
    
    fig.update_layout(
        height=500,
        font=dict(size=12),
        title_font_size=16
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Format and display table
    level1_summary['Comissão Total Formatada'] = level1_summary['Comissão Total'].apply(format_currency)
    level1_summary['Comissão Média Formatada'] = level1_summary['Comissão Média'].apply(format_currency)
    
    display_df = level1_summary[['Nível 1', 'Comissão Total Formatada', 'Qtd Transações', 'Comissão Média Formatada']].copy()
    display_df.columns = ['Nível 1', 'Comissão Total', 'Transações', 'Comissão Média']
    st.dataframe(display_df, use_container_width=True)

def create_client_analysis(df):
    """Create enhanced analysis by client code with Brazilian formatting."""
    st.subheader("👥 Análise de Comissões por Cliente")
    df['cod_cliente'] = df['cod_cliente'].astype(str).str.replace('\.0$', '', regex=True)
    
    client_summary = df.groupby('cod_cliente')['comissao_bruta_rs_escritorio'].agg(['sum', 'count']).reset_index()
    client_summary.columns = ['Código Cliente', 'Comissão Total', 'Qtd Transações']
    client_summary = client_summary.sort_values('Comissão Total', ascending=False)
    
    # Show top clients
    fig = px.bar(
        client_summary.head(20),
        x='Código Cliente',
        y='Comissão Total',
        title='Top 20 Clientes por Comissão Total',
        labels={'Comissão Total': 'Comissão (R$)', 'Código Cliente': 'Código do Cliente'},
        color='Comissão Total',
        color_continuous_scale='Viridis',
        text='Comissão Total'
    )
    
    fig.update_traces(
        texttemplate='%{text:,.0f}',
        textposition='outside',
        hovertemplate='<b>Cliente: %{x}</b><br>Comissão: R$ %{y:,.2f}<br>Transações: %{customdata}<extra></extra>',
        customdata=client_summary.head(20)['Qtd Transações']
    )
    
    fig.update_layout(
        xaxis_tickangle=45,
        showlegend=False,
        height=500,
        font=dict(size=12),
        title_font_size=16
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Format and display table
    client_summary['Comissão Total Formatada'] = client_summary['Comissão Total'].apply(format_currency)
    display_df = client_summary.head(50)[['Código Cliente', 'Comissão Total Formatada', 'Qtd Transações']].copy()
    display_df.columns = ['Código Cliente', 'Comissão Total', 'Transações']
    st.dataframe(display_df, use_container_width=True)

def create_assessor_analysis(df):
    """Create enhanced analysis by assessor code with Brazilian formatting."""
    st.subheader("🎯 Análise de Comissões por Assessor")
    df['cod_cliente'] = df['cod_cliente'].astype(str).str.replace('\.0$', '', regex=True)
    
    assessor_summary = df.groupby('cod_assessor_direto')['comissao_bruta_rs_escritorio'].agg(['sum', 'count', 'mean']).reset_index()
    assessor_summary.columns = ['Código Assessor', 'Comissão Total', 'Qtd Transações', 'Comissão Média']
    assessor_summary = assessor_summary.sort_values('Comissão Total', ascending=False)
    
    # Create enhanced scatter plot
    fig = px.scatter(
        assessor_summary,
        x='Qtd Transações',
        y='Comissão Total',
        size='Comissão Média',
        hover_data=['Código Assessor'],
        title='Performance dos Assessores: Volume vs Comissão',
        labels={
            'Comissão Total': 'Comissão Total (R$)',
            'Qtd Transações': 'Número de Transações',
            'Comissão Média': 'Comissão Média (R$)'
        },
        color='Comissão Média',
        color_continuous_scale='Plasma'
    )
    
    fig.update_traces(
        hovertemplate='<b>Assessor: %{customdata[0]}</b><br>' +
                     'Transações: %{x}<br>' +
                     'Comissão Total: R$ %{y:,.2f}<br>' +
                     'Comissão Média: R$ %{marker.size:,.2f}<extra></extra>'
    )
    
    fig.update_layout(
        height=500,
        font=dict(size=12),
        title_font_size=16
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Format and display table
    assessor_summary['Comissão Total Formatada'] = assessor_summary['Comissão Total'].apply(format_currency)
    assessor_summary['Comissão Média Formatada'] = assessor_summary['Comissão Média'].apply(format_currency)
    
    display_df = assessor_summary[['Código Assessor', 'Comissão Total Formatada', 'Qtd Transações', 'Comissão Média Formatada']].copy()
    display_df.columns = ['Código Assessor', 'Comissão Total', 'Transações', 'Comissão Média']
    st.dataframe(display_df, use_container_width=True)

def create_time_evolution_analysis(df):
    """Create enhanced time evolution analysis with Brazilian formatting."""
    st.subheader("📈 Evolução das Comissões ao Longo do Tempo")
    df['cod_cliente'] = df['cod_cliente'].astype(str).str.replace('\.0$', '', regex=True)
    
    # Group by month
    time_summary = df.groupby('month_year')['comissao_bruta_rs_escritorio'].agg(['sum', 'count', 'mean']).reset_index()
    time_summary.columns = ['Mês', 'Comissão Total', 'Qtd Transações', 'Comissão Média']
    time_summary = time_summary.sort_values('Mês')
    
    # Create enhanced line chart
    fig = px.line(
        time_summary,
        x='Mês',
        y='Comissão Total',
        title='Evolução das Comissões ao Longo do Tempo',
        markers=True,
        labels={'Comissão Total': 'Comissão (R$)', 'Mês': 'Período'},
        line_shape='spline'
    )
    
    # Add area fill
    fig.update_traces(
        fill='tonexty',
        fillcolor='rgba(31, 119, 180, 0.2)',
        line=dict(color='rgba(31, 119, 180, 1)', width=3),
        marker=dict(size=8, color='rgba(31, 119, 180, 1)'),
        hovertemplate='<b>%{x}</b><br>Comissão: R$ %{y:,.2f}<extra></extra>'
    )
    
    fig.update_layout(
        hovermode='x unified',
        height=500,
        font=dict(size=12),
        title_font_size=16,
        xaxis_title_font_size=14,
        yaxis_title_font_size=14
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show growth rates
    time_summary['Taxa de Crescimento (%)'] = time_summary['Comissão Total'].pct_change() * 100
    time_summary['Taxa de Crescimento (%)'] = time_summary['Taxa de Crescimento (%)'].round(2)
    
    # Format currency columns
    time_summary['Comissão Total Formatada'] = time_summary['Comissão Total'].apply(format_currency)
    time_summary['Comissão Média Formatada'] = time_summary['Comissão Média'].apply(format_currency)
    
    display_df = time_summary[['Mês', 'Comissão Total Formatada', 'Qtd Transações', 'Comissão Média Formatada', 'Taxa de Crescimento (%)']].copy()
    display_df.columns = ['Mês', 'Comissão Total', 'Transações', 'Comissão Média', 'Crescimento (%)']
    st.dataframe(display_df, use_container_width=True)

def create_renda_variavel_analysis(df):
    """Create comprehensive analysis for Renda Variável data with enhanced visualizations."""
    
    # Convert client codes to string for consistent comparison
    df['cod_cliente'] = df['cod_cliente'].astype(str).str.replace('\.0$', '', regex=True)
    
    # Load cross-sell clients to exclude them
    cross_sell_clients_to_exclude = load_cross_sell_clients()
    
    # ADD TOGGLE AT THE BEGINNING
    st.subheader("🔄 Seleção de Tipo de Cliente")
    
    # Create toggle for client type selection
    client_type = st.radio(
        "Selecione o tipo de cliente para análise:",
        options=["Clientes Normais", "Clientes Cross-sell"],
        index=0,  # Default to normal clients
        horizontal=True,
        help="Escolha entre analisar apenas clientes normais ou apenas clientes cross-sell"
    )
    
    # Apply filters based on toggle selection
    base_filter = (
        (df['categoria'].isin(['Renda Variável', 'Fundos Imobiliários', 'Produtos Financeiros']) & ~df['produto'].isin(['COE'])) |
        (df['produto'] == 'BTC')
    )
    
    if client_type == "Clientes Normais":
        # Filter for normal clients (exclude cross-sell clients)
        df_filtered = df[
            (~df['cod_cliente'].isin(cross_sell_clients_to_exclude)) & base_filter
        ].copy()
        client_type_label = "Clientes Normais"
        client_type_emoji = "👥"
        commission_percentage = 0.10  # 10% for normal clients
    else:
        # Filter for cross-sell clients only
        df_filtered = df[
            (df['cod_cliente'].isin(cross_sell_clients_to_exclude)) & base_filter
        ].copy()
        client_type_label = "Clientes Cross-sell"
        client_type_emoji = "🔄"
        commission_percentage = 0.01  # 1% for cross-sell clients
    
    if df_filtered.empty:
        st.warning(f"⚠️ Nenhum dado encontrado para {client_type_label} após aplicar os filtros de Renda Variável.")
        return
    
    # Show filtering summary with enhanced metrics (updated to show client type)
    st.info(f"📊 **Resumo da Análise de Renda Variável - {client_type_emoji} {client_type_label}:**")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total de Registros", format_number(len(df_filtered)))
    with col2:
        total_commission = df_filtered['comissao_bruta_rs_escritorio'].sum()
        st.metric("Comissão Total", format_currency(total_commission))
    with col3:
        avg_commission = df_filtered['comissao_bruta_rs_escritorio'].mean()
        st.metric("Comissão Média", format_currency(avg_commission))
    with col4:
        unique_clients = df_filtered['cod_cliente'].nunique()
        st.metric("Clientes Únicos", format_number(unique_clients))
    
    # Create analysis sections
    st.markdown("---")
    
    # Update chart titles to reflect client type
    chart_title_suffix = f" - {client_type_label}"
    
    # 1. Enhanced Time Evolution Analysis
    st.subheader("📈 Evolução das Comissões ao Longo do Tempo")
    
    # Total evolution with enhanced styling
    time_total = df_filtered.groupby('month_year')['comissao_bruta_rs_escritorio'].sum().reset_index()
    time_total.columns = ['Mês', 'Comissão Total']
    time_total = time_total.sort_values('Mês')
    
    fig_total = px.line(
        time_total,
        x='Mês',
        y='Comissão Total',
        title=f'Evolução Total das Comissões - Renda Variável{chart_title_suffix}',
        markers=True,
        labels={'Comissão Total': 'Comissão (R$)', 'Mês': 'Período'},
        line_shape='spline'
    )
    
    # Set different colors based on client type
    line_color = 'rgba(31, 119, 180, 1)' if client_type == "Clientes Normais" else 'rgba(255, 127, 14, 1)'
    fill_color = 'rgba(31, 119, 180, 0.2)' if client_type == "Clientes Normais" else 'rgba(255, 127, 14, 0.2)'
    
    fig_total.update_traces(
        fill='tonexty',
        fillcolor=fill_color,
        line=dict(color=line_color, width=3),
        marker=dict(size=10, color=line_color),
        hovertemplate='<b>%{x}</b><br>Comissão: R$ %{y:,.2f}<extra></extra>'
    )
    
    fig_total.update_layout(
        hovermode='x unified',
        height=500,
        font=dict(size=12),
        title_font_size=16
    )
    st.plotly_chart(fig_total, use_container_width=True)
    
    # Evolution by Category with enhanced colors
    st.subheader("📊 Evolução das Comissões por Categoria")
    
    time_category = df_filtered.groupby(['month_year', 'categoria'])['comissao_bruta_rs_escritorio'].sum().reset_index()
    time_category.columns = ['Mês', 'Categoria', 'Comissão']
    
    fig_cat = px.line(
        time_category,
        x='Mês',
        y='Comissão',
        color='Categoria',
        title=f'Evolução das Comissões por Categoria{chart_title_suffix}',
        markers=True,
        labels={'Comissão': 'Comissão (R$)', 'Mês': 'Período'},
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    
    fig_cat.update_traces(
        line=dict(width=3),
        marker=dict(size=8),
        hovertemplate='<b>%{fullData.name}</b><br>%{x}<br>Comissão: R$ %{y:,.2f}<extra></extra>'
    )
    
    fig_cat.update_layout(
        hovermode='x unified',
        height=500,
        font=dict(size=12),
        title_font_size=16
    )
    st.plotly_chart(fig_cat, use_container_width=True)
    
    # Evolution by Product (Top 10) with enhanced styling
    st.subheader("🏷️ Evolução das Comissões por Principais Produtos")
    
    # Get top 10 products by total commission
    top_products = df_filtered.groupby('produto')['comissao_bruta_rs_escritorio'].sum().nlargest(10).index.tolist()
    
    df_top_products = df_filtered[df_filtered['produto'].isin(top_products)]
    time_product = df_top_products.groupby(['month_year', 'produto'])['comissao_bruta_rs_escritorio'].sum().reset_index()
    time_product.columns = ['Mês', 'Produto', 'Comissão']
    
    fig_prod = px.line(
        time_product,
        x='Mês',
        y='Comissão',
        color='Produto',
        title=f'Evolução das Comissões - Top 10 Produtos{chart_title_suffix}',
        markers=True,
        labels={'Comissão': 'Comissão (R$)', 'Mês': 'Período'},
        color_discrete_sequence=px.colors.qualitative.Plotly
    )
    
    fig_prod.update_traces(
        line=dict(width=2),
        marker=dict(size=6)
    )
    
    fig_prod.update_layout(
        hovermode='x unified',
        legend=dict(orientation="v", x=1.02, y=1),
        height=500,
        font=dict(size=12),
        title_font_size=16
    )
    st.plotly_chart(fig_prod, use_container_width=True)
    

    # 3. Enhanced Category Deep Dive
    st.markdown("---")
    st.subheader("🎯 Análise Detalhada por Categoria")
    
    # Category summary with better formatting
    category_summary = df_filtered.groupby('categoria').agg({
        'comissao_bruta_rs_escritorio': ['sum', 'count', 'mean'],
        'receita_rs': 'sum',
        'cod_cliente': 'nunique'
    }).round(2)
    
    category_summary.columns = ['Comissão Total', 'Transações', 'Comissão Média', 'Receita Total', 'Clientes Únicos']
    category_summary = category_summary.reset_index()
    
    # Enhanced category pie chart
    fig_cat_pie = px.pie(
        category_summary,
        values='Comissão Total',
        names='categoria',
        title=f'Distribuição de Comissões por Categoria{chart_title_suffix}',
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    fig_cat_pie.update_traces(
        textposition='inside',
        textinfo='percent+label',
        hovertemplate='<b>%{label}</b><br>Comissão: R$ %{value:,.2f}<br>Percentual: %{percent}<extra></extra>'
    )
    
    fig_cat_pie.update_layout(
        height=500,
        font=dict(size=12),
        title_font_size=16
    )
    st.plotly_chart(fig_cat_pie, use_container_width=True)
    
    # Format and display category table
    for col in ['Comissão Total', 'Comissão Média', 'Receita Total']:
        category_summary[f'{col} Formatado'] = category_summary[col].apply(format_currency)
    
    display_cat = category_summary[['categoria', 'Comissão Total Formatado', 'Transações', 
                                   'Comissão Média Formatado', 'Receita Total Formatado', 'Clientes Únicos']].copy()
    display_cat.columns = ['Categoria', 'Comissão Total', 'Transações', 'Comissão Média', 'Receita Total', 'Clientes Únicos']
    st.dataframe(display_cat, use_container_width=True)
    
    # 4. Mesa RV Commission Calculation - Individual Analysis
    st.markdown("---")
    st.subheader("💰 Comissão Aproximada Mesa RV - Análise Individual")
    
    # Calculate Mesa RV commission based on current client type
    percentage_text = "10%" if client_type == "Clientes Normais" else "1%"
    
    st.info(f"📋 **Análise Individual:** {percentage_text} da comissão total para {client_type_label}")
    
    # Create monthly Mesa RV commission table for current selection
    mesa_rv_summary = df_filtered.groupby('month_year')['comissao_bruta_rs_escritorio'].sum().reset_index()
    mesa_rv_summary.columns = ['Mês', 'Comissão Total']
    mesa_rv_summary = mesa_rv_summary.sort_values('Mês')
    
    # Calculate Mesa RV commission for current selection
    mesa_rv_summary['Comissão Mesa RV'] = mesa_rv_summary['Comissão Total'] * commission_percentage
    
    # Calculate cumulative values for current selection
    mesa_rv_summary['Comissão Total Acumulada'] = mesa_rv_summary['Comissão Total'].cumsum()
    mesa_rv_summary['Comissão Mesa RV Acumulada'] = mesa_rv_summary['Comissão Mesa RV'].cumsum()
    
    # Format currency columns for display
    mesa_rv_summary['Comissão Total Formatada'] = mesa_rv_summary['Comissão Total'].apply(format_currency)
    mesa_rv_summary['Comissão Mesa RV Formatada'] = mesa_rv_summary['Comissão Mesa RV'].apply(format_currency)
    mesa_rv_summary['Comissão Total Acumulada Formatada'] = mesa_rv_summary['Comissão Total Acumulada'].apply(format_currency)
    mesa_rv_summary['Comissão Mesa RV Acumulada Formatada'] = mesa_rv_summary['Comissão Mesa RV Acumulada'].apply(format_currency)
    
    # Display the individual table
    display_mesa_rv = mesa_rv_summary[[
        'Mês', 
        'Comissão Total Formatada', 
        'Comissão Mesa RV Formatada',
        'Comissão Total Acumulada Formatada',
        'Comissão Mesa RV Acumulada Formatada'
    ]].copy()
    
    display_mesa_rv.columns = [
        'Mês', 
        'Comissão Total', 
        f'Mesa RV ({percentage_text})',
        'Total Acumulado',
        'Mesa RV Acumulada'
    ]
    
    st.dataframe(display_mesa_rv, use_container_width=True)
    
    # Summary metrics for current selection
    total_mesa_rv = mesa_rv_summary['Comissão Mesa RV'].sum()
    avg_mesa_rv = mesa_rv_summary['Comissão Mesa RV'].mean()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            f"Total Mesa RV ({percentage_text})", 
            format_currency(total_mesa_rv)
        )
    with col2:
        st.metric(
            f"Média Mensal Mesa RV", 
            format_currency(avg_mesa_rv)
        )
    with col3:
        total_original = mesa_rv_summary['Comissão Total'].sum()
        st.metric(
            "Comissão Total Original", 
            format_currency(total_original)
        )
    
    # Optional: Add a chart showing Mesa RV evolution for current selection
    st.subheader("📊 Evolução da Comissão Mesa RV")
    
    fig_mesa_rv = px.bar(
        mesa_rv_summary,
        x='Mês',
        y='Comissão Mesa RV',
        title=f'Evolução Mensal da Comissão Mesa RV ({percentage_text}) - {client_type_label}',
        labels={'Comissão Mesa RV': 'Comissão Mesa RV (R$)', 'Mês': 'Período'},
        color_discrete_sequence=[line_color]  # Use same color as the line chart
    )
    
    fig_mesa_rv.update_traces(
        hovertemplate='<b>%{x}</b><br>Mesa RV: R$ %{y:,.2f}<extra></extra>'
    )
    
    fig_mesa_rv.update_layout(
        height=400,
        font=dict(size=12),
        title_font_size=16
    )
    
    st.plotly_chart(fig_mesa_rv, use_container_width=True)
    
    # 5. COMPREHENSIVE COMPARISON TABLE
    st.markdown("---")
    st.subheader("📊 Tabela de Comissões (Normal e Cross Sell)")
    
    st.info("📋 **Análise Completa:** Análise lado a lado de ambos os tipos de cliente")
    
    # Calculate data for NORMAL CLIENTS
    df_normal = df[
        (~df['cod_cliente'].isin(cross_sell_clients_to_exclude)) & base_filter
    ].copy()
    
    # Calculate data for CROSS-SELL CLIENTS  
    df_cross_sell = df[
        (df['cod_cliente'].isin(cross_sell_clients_to_exclude)) & base_filter
    ].copy()
    
    # Get all unique months from both datasets
    all_months = sorted(set(df_normal['month_year'].unique()) | set(df_cross_sell['month_year'].unique()))
    
    # Create comparison dataframe
    comparison_data = []
    
    for month in all_months:
        # Normal clients data
        normal_month_data = df_normal[df_normal['month_year'] == month]
        normal_total = normal_month_data['comissao_bruta_rs_escritorio'].sum()
        normal_mesa_rv = normal_total * 0.10  # 10% for normal clients
        
        # Cross-sell clients data
        cross_sell_month_data = df_cross_sell[df_cross_sell['month_year'] == month]
        cross_sell_total = cross_sell_month_data['comissao_bruta_rs_escritorio'].sum()
        cross_sell_mesa_rv = cross_sell_total * 0.01  # 1% for cross-sell clients
        
        # Combined totals
        combined_total = normal_total + cross_sell_total
        combined_mesa_rv = normal_mesa_rv + cross_sell_mesa_rv
        
        comparison_data.append({
            'Mês': month,
            'Normal_Total': normal_total,
            'Normal_Mesa_RV': normal_mesa_rv,
            'CrossSell_Total': cross_sell_total,
            'CrossSell_Mesa_RV': cross_sell_mesa_rv,
            'Combined_Total': combined_total,
            'Combined_Mesa_RV': combined_mesa_rv
        })
    
    # Create DataFrame
    comparison_df = pd.DataFrame(comparison_data)
    
    # Calculate cumulative values
    comparison_df['Normal_Total_Acum'] = comparison_df['Normal_Total'].cumsum()
    comparison_df['Normal_Mesa_RV_Acum'] = comparison_df['Normal_Mesa_RV'].cumsum()
    comparison_df['CrossSell_Total_Acum'] = comparison_df['CrossSell_Total'].cumsum()
    comparison_df['CrossSell_Mesa_RV_Acum'] = comparison_df['CrossSell_Mesa_RV'].cumsum()
    comparison_df['Combined_Total_Acum'] = comparison_df['Combined_Total'].cumsum()
    comparison_df['Combined_Mesa_RV_Acum'] = comparison_df['Combined_Mesa_RV'].cumsum()
    
    # Format all currency columns
    currency_columns = [
        'Normal_Total', 'Normal_Mesa_RV', 'CrossSell_Total', 'CrossSell_Mesa_RV',
        'Combined_Total', 'Combined_Mesa_RV', 'Normal_Total_Acum', 'Normal_Mesa_RV_Acum',
        'CrossSell_Total_Acum', 'CrossSell_Mesa_RV_Acum', 'Combined_Total_Acum', 'Combined_Mesa_RV_Acum'
    ]
    
    for col in currency_columns:
        comparison_df[f'{col}_Formatted'] = comparison_df[col].apply(format_currency)
    
    # Create display table
    display_comparison = comparison_df[[
        'Mês',
        'Normal_Total_Formatted',
        'Normal_Mesa_RV_Formatted', 
        'CrossSell_Total_Formatted',
        'CrossSell_Mesa_RV_Formatted',
        'Combined_Total_Formatted',
        'Combined_Mesa_RV_Formatted'
    ]].copy()
    
    display_comparison.columns = [
        'Mês',
        'Normal - Total',
        'Normal - Mesa RV (10%)',
        'Cross-sell - Total', 
        'Cross-sell - Mesa RV (1%)',
        'Combinado - Total',
        'Combinado - Mesa RV'
    ]
    
    st.dataframe(display_comparison, use_container_width=True)
    
    # Summary metrics comparison
    st.subheader("📈 Resumo")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("**👥 Clientes Normais**")
        normal_total_commission = comparison_df['Normal_Total'].sum()
        normal_total_mesa_rv = comparison_df['Normal_Mesa_RV'].sum()
        st.metric("Total Comissão", format_currency(normal_total_commission))
        st.metric("Total Mesa RV (10%)", format_currency(normal_total_mesa_rv))
        
    with col2:
        st.markdown("**🔄 Clientes Cross-sell**")
        cross_sell_total_commission = comparison_df['CrossSell_Total'].sum()
        cross_sell_total_mesa_rv = comparison_df['CrossSell_Mesa_RV'].sum()
        st.metric("Total Comissão", format_currency(cross_sell_total_commission))
        st.metric("Total Mesa RV (1%)", format_currency(cross_sell_total_mesa_rv))
        
    with col3:
        st.markdown("**🔗 Combinado**")
        combined_total_commission = comparison_df['Combined_Total'].sum()
        combined_total_mesa_rv = comparison_df['Combined_Mesa_RV'].sum()
        st.metric("Total Comissão", format_currency(combined_total_commission))
        st.metric("Total Mesa RV", format_currency(combined_total_mesa_rv))
        
    with col4:
        st.markdown("**📊 Proporções**")
        if combined_total_commission > 0:
            normal_percentage = (normal_total_commission / combined_total_commission) * 100
            cross_sell_percentage = (cross_sell_total_commission / combined_total_commission) * 100
            st.metric("% Normal", f"{normal_percentage:.1f}%")
            st.metric("% Cross-sell", f"{cross_sell_percentage:.1f}%")
        else:
            st.metric("% Normal", "0%")
            st.metric("% Cross-sell", "0%")
    

    

def create_cross_sell_analysis(df_filtered):
    """Create comprehensive analysis for the specified list of Cross-Sell clients with enhanced visualizations."""
    
    if df_filtered.empty:
        st.warning("⚠️ Nenhum dado encontrado para os clientes de cross-sell especificados no período selecionado.")
        return
    
    # Show filtering summary with enhanced metrics
    st.info(f"📊 **Resumo da Análise de Cross-Sell:**")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total de Registros", format_number(len(df_filtered)))
    with col2:
        total_commission = df_filtered['comissao_bruta_rs_escritorio'].sum()
        st.metric("Comissão Total", format_currency(total_commission))
    with col3:
        avg_commission = df_filtered['comissao_bruta_rs_escritorio'].mean()
        st.metric("Comissão Média", format_currency(avg_commission))
    with col4:
        unique_clients = df_filtered['cod_cliente'].nunique()
        st.metric("Clientes Únicos", format_number(unique_clients))
    
    # Create analysis sections
    st.markdown("---")
    
    # 1. Enhanced Time Evolution Analysis
    st.subheader("📈 Evolução das Comissões ao Longo do Tempo (Clientes Cross-Sell)")
    
    time_total = df_filtered.groupby('month_year')['comissao_bruta_rs_escritorio'].sum().reset_index()
    time_total.columns = ['Mês', 'Comissão Total']
    time_total = time_total.sort_values('Mês')
    
    fig_total = px.line(
        time_total, x='Mês', y='Comissão Total',
        title='Evolução Total das Comissões - Clientes Cross-Sell',
        markers=True, labels={'Comissão Total': 'Comissão (R$)', 'Mês': 'Período'},
        line_shape='spline'
    )
    
    fig_total.update_traces(
        fill='tonexty',
        fillcolor='rgba(255, 127, 14, 0.2)',
        line=dict(color='rgba(255, 127, 14, 1)', width=3),
        marker=dict(size=10, color='rgba(255, 127, 14, 1)'),
        hovertemplate='<b>%{x}</b><br>Comissão: R$ %{y:,.2f}<extra></extra>'
    )
    
    fig_total.update_layout(
        hovermode='x unified', height=500, font=dict(size=12), title_font_size=16
    )
    st.plotly_chart(fig_total, use_container_width=True)
    
    # Evolution by Category with enhanced styling
    st.subheader("📊 Evolução das Comissões por Produto")
    time_category = df_filtered.groupby(['month_year', 'produto'])['comissao_bruta_rs_escritorio'].sum().reset_index()
    time_category.columns = ['Mês', 'Produto', 'Comissão']
    
    fig_cat = px.line(
        time_category, x='Mês', y='Comissão', color='Produto',
        title='Evolução das Comissões por Produto', markers=True,
        labels={'Comissão': 'Comissão (R$)', 'Mês': 'Período'},
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    
    fig_cat.update_traces(line=dict(width=3), marker=dict(size=8))
    fig_cat.update_layout(hovermode='x unified', height=500, font=dict(size=12), title_font_size=16)
    st.plotly_chart(fig_cat, use_container_width=True)
    
    
    # 3. Enhanced Category Deep Dive
    st.markdown("---")
    st.subheader("🎯 Análise Detalhada por Produto")
    category_summary = df_filtered.groupby('produto').agg({
        'comissao_bruta_rs_escritorio': ['sum', 'count', 'mean'],
        'receita_rs': 'sum', 'cod_cliente': 'nunique'
    }).round(2)
    category_summary.columns = ['Comissão Total', 'Transações', 'Comissão Média', 'Receita Total', 'Clientes Únicos']
    category_summary = category_summary.reset_index()
    
    fig_cat_pie = px.pie(
        category_summary, values='Comissão Total', names='produto', 
        title='Distribuição de Comissões por Produto',
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    fig_cat_pie.update_traces(
        textposition='inside', textinfo='percent+label',
        hovertemplate='<b>%{label}</b><br>Comissão: R$ %{value:,.2f}<br>Percentual: %{percent}<extra></extra>'
    )
    fig_cat_pie.update_layout(height=500, font=dict(size=12), title_font_size=16)
    st.plotly_chart(fig_cat_pie, use_container_width=True)
    
    # Format and display category table
    for col in ['Comissão Total', 'Comissão Média', 'Receita Total']:
        category_summary[f'{col} Formatado'] = category_summary[col].apply(format_currency)
    
    display_cat = category_summary[['produto', 'Comissão Total Formatado', 'Transações', 
                                   'Comissão Média Formatado', 'Receita Total Formatado', 'Clientes Únicos']].copy()
    display_cat.columns = ['Produto', 'Comissão Total', 'Transações', 'Comissão Média', 'Receita Total', 'Clientes Únicos']
    st.dataframe(display_cat, use_container_width=True)
    
    # 4. Enhanced Product Analysis
    st.markdown("---")
    st.subheader("📦 Análise Detalhada por Produto")
    product_summary = df_filtered.groupby('produto').agg({
        'comissao_bruta_rs_escritorio': ['sum', 'count', 'mean'],
        'receita_rs': 'sum', 'cod_cliente': 'nunique'
    }).round(2)
    product_summary.columns = ['Comissão Total', 'Transações', 'Comissão Média', 'Receita Total', 'Clientes Únicos']
    product_summary = product_summary.reset_index().sort_values('Comissão Total', ascending=False)
    
    fig_prod_bar = px.bar(
        product_summary.head(15), x='produto', y='Comissão Total',
        title='Top 15 Produtos por Comissão Total', 
        labels={'Comissão Total': 'Comissão (R$)', 'produto': 'Produto'},
        color='Comissão Total', color_continuous_scale='Viridis', text='Comissão Total'
    )
    fig_prod_bar.update_traces(
        texttemplate='%{text:,.0f}', textposition='outside',
        hovertemplate='<b>%{x}</b><br>Comissão: R$ %{y:,.2f}<extra></extra>'
    )
    fig_prod_bar.update_layout(xaxis_tickangle=45, showlegend=False, height=600, font=dict(size=12), title_font_size=16)
    st.plotly_chart(fig_prod_bar, use_container_width=True)
    
    # Format and display product table
    for col in ['Comissão Total', 'Comissão Média', 'Receita Total']:
        product_summary[f'{col} Formatado'] = product_summary[col].apply(format_currency)
    
    display_prod = product_summary[['produto', 'Comissão Total Formatado', 'Transações', 
                                   'Comissão Média Formatado', 'Receita Total Formatado', 'Clientes Únicos']].copy()
    display_prod.columns = ['Produto', 'Comissão Total', 'Transações', 'Comissão Média', 'Receita Total', 'Clientes Únicos']
    st.dataframe(display_prod, use_container_width=True)
    
    # 5. Enhanced Client Analysis (within the cross-sell group)
    st.markdown("---")
    st.subheader("👥 Análise de Clientes (dentro do Grupo Cross-Sell)")
    client_summary = df_filtered.groupby('cod_cliente').agg({
        'comissao_bruta_rs_escritorio': ['sum', 'count', 'mean'],
        'receita_rs': 'sum'
    }).round(2)
    client_summary.columns = ['Comissão Total', 'Transações', 'Comissão Média', 'Receita Total']
    client_summary = client_summary.reset_index().sort_values('Comissão Total', ascending=False)
    client_summary.reset_index(drop=True, inplace=True)
    
    st.write("**Performance dos Clientes (da lista monitorada):**")
    fig_client = px.bar(
        client_summary.head(20), x=client_summary.head(20).index, y='Comissão Total',
        title='Top 20 Clientes Cross-Sell por Comissão Total',
        labels={'Comissão Total': 'Comissão (R$)', 'cod_cliente': 'Código do Cliente'},
        color='Comissão Total', color_continuous_scale='Plasma', text='Comissão Total',
        custom_data=['cod_cliente'], hover_name='cod_cliente'
    )
    fig_client.update_traces(
        texttemplate='%{text:,.0f}', textposition='outside',
        hovertemplate='<b>Cliente: %{customdata[0]}</b><br>Comissão: R$ %{y:,.2f}<extra></extra>'
    )
    fig_client.update_layout(xaxis_tickvals= client_summary.head(20).index, xaxis_ticktext=client_summary['cod_cliente'], xaxis_tickangle=45, showlegend=False, height=500, font=dict(size=12), title_font_size=16)
    st.plotly_chart(fig_client, use_container_width=True)
    
    # Format and display client table
    for col in ['Comissão Total', 'Comissão Média', 'Receita Total']:
        client_summary[f'{col} Formatado'] = client_summary[col].apply(format_currency)
    
    display_client = client_summary[['cod_cliente', 'Comissão Total Formatado', 'Transações', 
                                    'Comissão Média Formatado', 'Receita Total Formatado']].copy()
    display_client.columns = ['Código Cliente', 'Comissão Total', 'Transações', 'Comissão Média', 'Receita Total']
    st.dataframe(display_client, use_container_width=True)
    
    # 6. Enhanced Assessor Analysis
    st.markdown("---")
    st.subheader("🎯 Performance dos Assessores (para Clientes Cross-Sell)")
    assessor_summary = df_filtered.groupby('cod_assessor_direto').agg({
        'comissao_bruta_rs_escritorio': ['sum', 'count', 'mean'],
        'receita_rs': 'sum', 'cod_cliente': 'nunique'
    }).round(2)
    assessor_summary.columns = ['Comissão Total', 'Transações', 'Comissão Média', 'Receita Total', 'Clientes Únicos']
    assessor_summary = assessor_summary.reset_index().sort_values('Comissão Total', ascending=False)
    
    fig_assessor = px.scatter(
        assessor_summary.head(30), x='Transações', y='Comissão Total',
        size='Comissão Média', hover_data=['cod_assessor_direto', 'Clientes Únicos'],
        title='Performance dos Assessores: Volume vs Comissão (Top 30)',
        labels={'Comissão Total': 'Comissão (R$)', 'Transações': 'Número de Transações'},
        color='Clientes Únicos', color_continuous_scale='Turbo'
    )
    fig_assessor.update_traces(
        hovertemplate='<b>Assessor: %{customdata[0]}</b><br>' +
                     'Transações: %{x}<br>' +
                     'Comissão Total: R$ %{y:,.2f}<br>' +
                     'Clientes Únicos: %{color}<br>' +
                     'Comissão Média: R$ %{marker.size:,.2f}<extra></extra>'
    )
    fig_assessor.update_layout(height=500, font=dict(size=12), title_font_size=16)
    st.plotly_chart(fig_assessor, use_container_width=True)
    
    # Format and display assessor table
    for col in ['Comissão Total', 'Comissão Média', 'Receita Total']:
        assessor_summary[f'{col} Formatado'] = assessor_summary[col].apply(format_currency)
    
    display_assessor = assessor_summary.head(20)[['cod_assessor_direto', 'Comissão Total Formatado', 'Transações', 
                                                 'Comissão Média Formatado', 'Receita Total Formatado', 'Clientes Únicos']].copy()
    display_assessor.columns = ['Código Assessor', 'Comissão Total', 'Transações', 'Comissão Média', 'Receita Total', 'Clientes Únicos']
    st.dataframe(display_assessor, use_container_width=True)

def load_cross_sell_clients(file_path="/mnt/databases/cross_sell_clients.txt"):
    """
    Loads client codes from a text file for cross-sell analysis.
    
    Args:
        file_path (str): The path to the text file.
        
    Returns:
        list: A list of unique client codes as strings.
    """
    try:
        with open(file_path, 'r') as f:
            # Read lines, strip whitespace, filter out empty lines, and get unique codes
            clients = {line.strip() for line in f if line.strip()}
        return list(clients)
    except FileNotFoundError:
        st.error(f"Erro: O arquivo de lista de clientes '{file_path}' não foi encontrado.")
        return []
    
def create_centralized_month_selector(available_months):
    """
    Create a single, centralized month selector in the sidebar that all tabs can use.
    
    Args:
        available_months (list): List of available month-year strings (format: 'YYYY-MM')
        
    Returns:
        dict: Dictionary with selections for each tab
    """
    if not available_months:
        return {}
    
    # Parse and organize months by year
    months_by_year = {}
    for month_str in available_months:
        try:
            year, month = month_str.split('-')
            year = int(year)
            month = int(month)
            
            if year not in months_by_year:
                months_by_year[year] = []
            months_by_year[year].append((month, month_str))
        except ValueError:
            continue  # Skip invalid format
    
    # Sort years and months
    sorted_years = sorted(months_by_year.keys(), reverse=True)  # Most recent first
    for year in months_by_year:
        months_by_year[year].sort(key=lambda x: x[0], reverse=True)  # Most recent month first
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("📅 Seleção de Período")
    
    # Initialize session state for the main selection if not exists
    main_session_key = "main_selected_months"
    if main_session_key not in st.session_state:
        st.session_state[main_session_key] = available_months.copy()  # Default: all selected
    
    # Quick selection options
    st.sidebar.markdown("**🚀 Seleção Rápida:**")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("✅ Todos", key="select_all_main", 
                     help="Selecionar todos os meses disponíveis"):
            st.session_state[main_session_key] = available_months.copy()
            st.rerun()
    with col2:
        if st.button("❌ Limpar", key="clear_all_main",
                     help="Desmarcar todos os meses"):
            st.session_state[main_session_key] = []
            st.rerun()
    
    # Year-based selection
    st.sidebar.markdown("**📊 Seleção por Ano:**")
    
    selected_months = st.session_state[main_session_key].copy()
    
    for year in sorted_years:
        year_months = [month_str for _, month_str in months_by_year[year]]
        year_selected_count = len([m for m in year_months if m in selected_months])
        
        # Year header with selection info
        year_header = f"**{year}** ({year_selected_count}/{len(year_months)} meses)"
        st.sidebar.markdown(year_header)
        
        # Year-level controls
        col1, col2, col3 = st.sidebar.columns([1, 1, 2])
        
        with col1:
            if st.button("✅", key=f"select_year_{year}_main",
                        help=f"Selecionar todos os meses de {year}"):
                for month_str in year_months:
                    if month_str not in selected_months:
                        selected_months.append(month_str)
                st.session_state[main_session_key] = selected_months
                st.rerun()
        
        with col2:
            if st.button("❌", key=f"deselect_year_{year}_main",
                        help=f"Desmarcar todos os meses de {year}"):
                selected_months = [m for m in selected_months if m not in year_months]
                st.session_state[main_session_key] = selected_months
                st.rerun()
        
        # Individual month checkboxes
        month_names = {
            1: "Jan", 2: "Fev", 3: "Mar", 4: "Abr", 5: "Mai", 6: "Jun",
            7: "Jul", 8: "Ago", 9: "Set", 10: "Out", 11: "Nov", 12: "Dez"
        }
        
        # Create a container for better spacing
        with st.sidebar.container():
            for month_num, month_str in months_by_year[year]:
                month_label = f"{month_names[month_num]} {year}"
                
                # Check if this month is currently selected
                is_selected = month_str in selected_months
                
                # Create checkbox with callback
                checkbox_result = st.checkbox(
                    month_label,
                    value=is_selected,
                    key=f"month_{month_str}_main",
                    on_change=update_main_month_selection,
                    args=(month_str,)
                )
        
        st.sidebar.markdown("")  # Add some spacing
    
    # Get the current selection from session state
    current_selection = st.session_state[main_session_key]
    
    # Show selection summary
    if current_selection:
        st.sidebar.success(f"📊 **{len(current_selection)} mês(es) selecionado(s)**")
        
        # Show selected months in a compact format
        with st.sidebar.expander("Ver meses selecionados"):
            # Group selected months by year for display
            selected_by_year = {}
            month_names = {
                1: "Jan", 2: "Fev", 3: "Mar", 4: "Abr", 5: "Mai", 6: "Jun",
                7: "Jul", 8: "Ago", 9: "Set", 10: "Out", 11: "Nov", 12: "Dez"
            }
            
            for month_str in sorted(current_selection, reverse=True):
                try:
                    year, month = month_str.split('-')
                    year = int(year)
                    month = int(month)
                    
                    if year not in selected_by_year:
                        selected_by_year[year] = []
                    selected_by_year[year].append(month_names[month])
                except ValueError:
                    continue
            
            for year in sorted(selected_by_year.keys(), reverse=True):
                months_list = ", ".join(selected_by_year[year])
                st.write(f"**{year}:** {months_list}")
    else:
        st.sidebar.warning("⚠️ Nenhum mês selecionado")
    
    return current_selection

def update_main_month_selection(month_str):
    """
    Callback function to update the main month selection when checkbox is clicked.
    
    Args:
        month_str (str): The month string being toggled
    """
    session_key = "main_selected_months"
    checkbox_key = f"month_{month_str}_main"
    
    # Get current checkbox state
    checkbox_state = st.session_state.get(checkbox_key, False)
    
    # Get current selection
    current_selection = st.session_state.get(session_key, [])
    
    # Update selection based on checkbox state
    if checkbox_state and month_str not in current_selection:
        current_selection.append(month_str)
    elif not checkbox_state and month_str in current_selection:
        current_selection.remove(month_str)
    
    # Update session state
    st.session_state[session_key] = current_selection

def get_selected_months():
    """
    Get the currently selected months from the centralized selector.
    
    Returns:
        list: Currently selected months
    """
    return st.session_state.get("main_selected_months", [])

def main():
    """
    Main Streamlit application function with centralized month selection.
    """
    st.set_page_config(
        page_title="Analisador de Dados de Comissão",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("📊 Analisador de Dados de Comissão")
    st.markdown("Faça upload dos seus arquivos Excel mensais e analise os dados de comissão ao longo do tempo.")

    # Initialize our data manager
    data_manager = CommissionDataManager()

    # Get available months once for all tabs
    available_months = data_manager.get_available_months()

    # Show database status in sidebar
    st.sidebar.title("📊 Dashboard de Comissões")

    if available_months:
        st.sidebar.success(f"✅ **{len(available_months)} mês(es) disponível(is)**")

        # Show quick stats
        conn = sqlite3.connect(data_manager.db_path)
        total_records = pd.read_sql_query("SELECT COUNT(*) as count FROM commission_data", conn).iloc[0]['count']
        total_commission = pd.read_sql_query("SELECT SUM(comissao_bruta_rs_escritorio) as total FROM commission_data", conn).iloc[0]['total']
        
        # Show document type breakdown
        doc_type_query = "SELECT document_type, COUNT(*) as count FROM commission_data GROUP BY document_type"
        doc_type_stats = pd.read_sql_query(doc_type_query, conn)
        conn.close()

        st.sidebar.metric("Total de Registros", format_number(total_records))
        if total_commission:
            st.sidebar.metric("Comissão Total", format_currency(total_commission))
        
        # Show document type breakdown
        if not doc_type_stats.empty:
            st.sidebar.markdown("**📄 Tipos de Documento:**")
            for _, row in doc_type_stats.iterrows():
                doc_type = row['document_type'] if row['document_type'] else 'original'
                count = row['count']
                st.sidebar.write(f"- {doc_type.upper()}: {format_number(count)}")

        # Create the centralized month selector (only once)
        selected_months = create_centralized_month_selector(available_months)
        
        # Add document type selector
        st.sidebar.markdown("---")
        st.sidebar.subheader("📄 Tipo de Documento")
        
        # Get available document types
        conn = sqlite3.connect(data_manager.db_path)
        doc_types_query = "SELECT DISTINCT document_type FROM commission_data WHERE document_type IS NOT NULL"
        available_doc_types_df = pd.read_sql_query(doc_types_query, conn)
        conn.close()
        
        if not available_doc_types_df.empty:
            available_doc_types = available_doc_types_df['document_type'].tolist()
            # Replace None with 'original' for display
            available_doc_types = ['original' if x is None else x for x in available_doc_types]
            available_doc_types = list(set(available_doc_types))  # Remove duplicates
            
            selected_doc_types = st.sidebar.multiselect(
                "Selecione os tipos de documento:",
                options=available_doc_types,
                default=available_doc_types,
                help="Escolha quais tipos de documento incluir na análise"
            )
        else:
            selected_doc_types = ["original"]
    else:
        st.sidebar.warning("⚠️ Nenhum dado disponível")
        st.sidebar.info("Faça upload de arquivos na aba 'Upload de Dados'")
        selected_months = []
        selected_doc_types = ["original"]

    # Create tabs - MODIFIED ORDER HERE
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Análise de Dados",
        "💹 Renda Variável",
        "🔄 Cross-Sell",
        "🔍 Explorar Dados",
        "📁 Upload de Dados" # This tab is now last
    ])

    with tab1: # This is now "Análise de Dados"
        st.header("Análise de Dados")

        if not available_months:
            st.warning("⚠️ Nenhum dado disponível. Por favor, faça upload de alguns arquivos primeiro!")
            return

        if not selected_months:
            st.warning("Por favor, selecione pelo menos um mês na barra lateral.")
            return

        # Get data for selected months and document types
        df_analysis = data_manager.get_data_for_analysis(
            months=selected_months, 
            document_types=selected_doc_types
        )

        if df_analysis.empty:
            st.warning("Nenhum dado encontrado para os meses e tipos de documento selecionados.")
            return

        st.success(f"📊 Analisando {format_number(len(df_analysis))} registros de {len(selected_months)} mês(es)")
        
        # Show document type breakdown in the analysis
        if 'document_type' in df_analysis.columns:
            doc_breakdown = df_analysis['document_type'].value_counts()
            st.info(f"📄 **Breakdown por tipo:** " + " | ".join([f"{k.upper()}: {format_number(v)}" for k, v in doc_breakdown.items()]))

        # Analysis options
        analysis_type = st.selectbox(
            "Escolha o tipo de análise:",
            ["Por Produto", "Por Nível 1", "Por Código de Cliente", "Por Código de Assessor", "Evolução Temporal"]
        )

        # Create visualizations based on selection
        if analysis_type == "Por Produto":
            create_product_analysis(df_analysis)
        elif analysis_type == "Por Nível 1":
            create_level1_analysis(df_analysis)
        elif analysis_type == "Por Código de Cliente":
            create_client_analysis(df_analysis)
        elif analysis_type == "Por Código de Assessor":
            create_assessor_analysis(df_analysis)
        elif analysis_type == "Evolução Temporal":
            create_time_evolution_analysis(df_analysis)

    with tab2: # This is now "Renda Variável"
        st.header("💹 Análise de Renda Variável")

        if not available_months:
            st.warning("⚠️ Nenhum dado disponível. Por favor, faça upload de alguns arquivos primeiro!")
            return

        if not selected_months:
            st.warning("Por favor, selecione pelo menos um mês na barra lateral.")
            return

        # Get data for selected months and document types
        df_rv = data_manager.get_data_for_analysis(
            months=selected_months, 
            document_types=selected_doc_types
        )

        if df_rv.empty:
            st.warning("Nenhum dado encontrado para os meses e tipos de documento selecionados.")
            return

        # Apply Renda Variável analysis
        create_renda_variavel_analysis(df_rv)

    with tab3: # This is now "Cross-Sell"
        st.header("🔄 Análise de Cross-Sell")
        st.markdown("Análise dos dados de comissão para uma lista específica de clientes de cross-sell, carregada do arquivo `cross_sell_clients.txt`.")

        if not available_months:
            st.warning("⚠️ Nenhum dado disponível. Por favor, faça upload de alguns arquivos primeiro!")
        else:
            # Load cross-sell clients
            cross_sell_clients = load_cross_sell_clients()

            if not cross_sell_clients:
                st.warning("Lista de clientes está vazia ou arquivo não encontrado. Não é possível realizar a análise.")
            else:
                st.info(f"Carregados {format_number(len(cross_sell_clients))} clientes únicos para análise de cross-sell.")
                with st.expander("Visualizar Clientes Monitorados"):
                    # Display clients in a more organized way
                    clients_df = pd.DataFrame(cross_sell_clients, columns=['Código do Cliente'])
                    st.dataframe(clients_df, use_container_width=True)

                if not selected_months:
                    st.warning("Por favor, selecione pelo menos um mês na barra lateral.")
                else:
                    # Get data and filter
                    df_all = data_manager.get_data_for_analysis(
                        months=selected_months, 
                        document_types=selected_doc_types
                    )

                    if not df_all.empty:
                        # Important: Ensure client codes are strings for matching
                        df_all['cod_cliente'] = df_all['cod_cliente'].astype(str).str.replace('\.0$', '', regex=True)
                        df_cross_sell = df_all[df_all['cod_cliente'].isin(cross_sell_clients)].copy()

                        # Call the enhanced analysis function
                        create_cross_sell_analysis(df_cross_sell)

    with tab4: # This is now "Explorar Dados"
        st.header("🔍 Explorador de Dados")
        st.markdown("Use esta ferramenta para explorar e filtrar os dados de forma interativa.")

        if not available_months:
            st.warning("⚠️ Nenhum dado disponível. Por favor, faça upload de alguns arquivos primeiro!")
            return

        if selected_months:
            df_explorer = data_manager.get_data_for_analysis(
                months=selected_months, 
                document_types=selected_doc_types
            )
            if not df_explorer.empty:
                st.info(f"📊 Explorando {format_number(len(df_explorer))} registros")

                # Add some basic statistics before the explorer
                with st.expander("📈 Estatísticas Rápidas"):
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        total_commission = df_explorer['comissao_bruta_rs_escritorio'].sum()
                        st.metric("Comissão Total", format_currency(total_commission))
                    with col2:
                        avg_commission = df_explorer['comissao_bruta_rs_escritorio'].mean()
                        st.metric("Comissão Média", format_currency(avg_commission))
                    with col3:
                        unique_clients = df_explorer['cod_cliente'].nunique()
                        st.metric("Clientes Únicos", format_number(unique_clients))
                    with col4:
                        unique_products = df_explorer['produto'].nunique()
                        st.metric("Produtos Únicos", format_number(unique_products))

                # Use the dataframe explorer
                filtered_df = dataframe_explorer(df_explorer, case=False)

                # Show filtered results summary
                if len(filtered_df) != len(df_explorer):
                    st.info(f"🔍 Filtro aplicado: {format_number(len(filtered_df))} de {format_number(len(df_explorer))} registros mostrados")

                st.dataframe(filtered_df, use_container_width=True)

                # Add download button for filtered data
                if not filtered_df.empty:
                    csv = filtered_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Baixar dados filtrados como CSV",
                        data=csv,
                        file_name=f"dados_comissao_filtrados_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
        else:
            st.warning("Por favor, selecione pelo menos um mês na barra lateral.")

    with tab5: # This is now "Upload de Dados"
        st.header("Upload de Arquivos Excel")

        # Add a section for database maintenance
        with st.expander("🔧 Manutenção do Banco de Dados"):
            st.subheader("Corrigir Tipos de Dados")
            st.write("Se você está tendo problemas com códigos de cliente sendo armazenados como decimais (ex: '12345.0'), clique abaixo para corrigi-los:")

            if st.button("Corrigir Tipos de Dados dos Códigos de Cliente"):
                data_manager.fix_client_codes_data_types()

        # File uploader - allows multiple files
        uploaded_files = st.file_uploader(
            "Escolha os arquivos Excel",
            type=['xlsx', 'xls'],
            accept_multiple_files=True,
            help="Faça upload de um ou mais arquivos Excel contendo dados de comissão (formato original ou P2)"
        )

        if uploaded_files:
            for uploaded_file in uploaded_files:
                st.subheader(f"Processando: {uploaded_file.name}")

                # Process the file
                df, month_year, record_count = data_manager.process_xlsx_file(uploaded_file)

                # In the Upload tab, replace the existing button logic with:
                if df is not None:
                    st.success(f"✅ Arquivo processado com sucesso!")
                    st.info(f"📅 Mês/Ano detectado: {month_year}")
                    st.info(f"📊 Registros encontrados: {format_number(record_count)}")
                    
                    # Check if month exists before showing process button
                    conn = sqlite3.connect(data_manager.db_path)
                    existing_check = pd.read_sql_query(
                        "SELECT COUNT(*) as count FROM commission_data WHERE month_year = ?", 
                        conn, params=[month_year]
                    )
                    conn.close()
                    
                    month_exists = existing_check.iloc[0]['count'] > 0
                    
                    if month_exists:
                        existing_count = existing_check.iloc[0]['count']
                        st.warning(f"⚠️ Mês {month_year} já existe ({format_number(existing_count)} registros)")
                        
                        # Create choice buttons
                        choice_key = f"choice_{uploaded_file.name}_{month_year}"
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            if st.button(f"🔄 Substituir", key=f"replace_{choice_key}"):
                                st.session_state[choice_key] = "replace"
                        with col2:
                            if st.button(f"➕ Adicionar", key=f"append_{choice_key}"):
                                st.session_state[choice_key] = "append"
                        with col3:
                            if st.button(f"⏭️ Pular", key=f"skip_{choice_key}"):
                                st.session_state[choice_key] = "skip"
                        
                        # Process based on choice
                        user_choice = st.session_state.get(choice_key)
                        if user_choice:
                            if st.button(f"✅ Confirmar: {user_choice.upper()}", key=f"confirm_{choice_key}"):
                                with st.spinner("Processando..."):
                                    if user_choice == "replace":
                                        # Delete existing data first
                                        conn = sqlite3.connect(data_manager.db_path)
                                        cursor = conn.cursor()
                                        cursor.execute("DELETE FROM commission_data WHERE month_year = ?", (month_year,))
                                        conn.commit()
                                        conn.close()
                                    
                                    if user_choice != "skip":
                                        inserted_count = data_manager._insert_all_records(df, month_year, sqlite3.connect(data_manager.db_path))
                                        st.success(f"✅ {format_number(inserted_count)} registros processados!")
                                    else:
                                        st.info("Processamento pulado.")
                                    
                                    # Clear choice and refresh
                                    del st.session_state[choice_key]
                                    st.rerun()
                    else:
                        # New month, show simple process button
                        if st.button(f"Processar dados de {uploaded_file.name}", key=f"process_{uploaded_file.name}"):
                            with st.spinner("Processando dados..."):
                                inserted_count = data_manager._insert_all_records(df, month_year, sqlite3.connect(data_manager.db_path))
                                if inserted_count > 0:
                                    st.success(f"✅ {format_number(inserted_count)} registros processados!")
                                    st.rerun()

        # Show current database status
        st.subheader("Status do Banco de Dados")
        if available_months:
            st.success(f"📅 Meses disponíveis: {', '.join(available_months)}")

            # Show record count per month and document type
            conn = sqlite3.connect(data_manager.db_path)
            month_counts = pd.read_sql_query(
                """SELECT 
                    month_year as 'Mês', 
                    document_type as 'Tipo',
                    COUNT(*) as 'Qtd Registros' 
                FROM commission_data 
                GROUP BY month_year, document_type 
                ORDER BY month_year, document_type""",
                conn
            )
            conn.close()

            # Format the count column and document type
            month_counts['Qtd Registros Formatada'] = month_counts['Qtd Registros'].apply(format_number)
            month_counts['Tipo'] = month_counts['Tipo'].fillna('original').str.upper()
            display_counts = month_counts[['Mês', 'Tipo', 'Qtd Registros Formatada']].copy()
            display_counts.columns = ['Mês', 'Tipo de Documento', 'Quantidade de Registros']

            st.dataframe(display_counts, use_container_width=True)
        else:
            st.info("Ainda não há dados no banco. Faça upload de alguns arquivos!")

if __name__ == "__main__":
    main()