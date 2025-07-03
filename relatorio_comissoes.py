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
    return f"R$ {value:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.')

def format_number(value):
    """Format numbers in Brazilian format"""
    return f"{value:,.0f}".replace(',', '.')

class CommissionDataManager:
    """
    This class handles all database operations for our commission data.
    We use a class to organize our code better and avoid repeating database connection logic.
    """
    
    def __init__(self, db_path="commission_data.db"):
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
                        month_year TEXT
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
                    month_year TEXT
                )
            ''')
            st.success("✅ Nova tabela do banco de dados criada")
        
        conn.commit()
        conn.close()
    
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
            # This SQL will convert '12345.0' to '12345' but leave '12345' as '12345'
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
        Clean data types before inserting to prevent float/int issues.
        
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
        
        return df_clean
    
    def recreate_table_if_needed(self):
        """
        Recreate the table with the correct schema if there are constraint issues.
        This is a helper method to fix existing databases.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Backup existing data
            cursor.execute("SELECT * FROM commission_data")
            existing_data = cursor.fetchall()
            
            # Get column names
            cursor.execute("PRAGMA table_info(commission_data)")
            columns_info = cursor.fetchall()
            column_names = [col[1] for col in columns_info]
            
            # Drop the old table
            cursor.execute("DROP TABLE commission_data")
            
            # Create new table with correct constraint
            cursor.execute('''
                CREATE TABLE commission_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    categoria TEXT,
                    produto TEXT,
                    nivel_1 TEXT,
                    nivel_2 TEXT,
                    nivel_3 TEXT,
                    cod_cliente TEXT,
                    data DATE,
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
                    UNIQUE(cod_cliente, data, produto, cod_assessor_direto)
                )
            ''')
            
            # Restore data if any existed
            if existing_data:
                # Prepare insert statement
                placeholders = ','.join(['?' for _ in column_names])
                insert_sql = f"INSERT INTO commission_data ({','.join(column_names)}) VALUES ({placeholders})"
                
                # Insert data, ignoring duplicates
                for row in existing_data:
                    try:
                        cursor.execute(insert_sql, row)
                    except sqlite3.IntegrityError:
                        # Skip duplicates
                        continue
                
                st.success(f"Tabela recriada e {len(existing_data)} registros restaurados")
            
            conn.commit()
            return True
            
        except Exception as e:
            st.error(f"Erro ao recriar tabela: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()
    
    def process_xlsx_file(self, file_path_or_buffer):
        """
        Process an Excel file and extract month/year information.
        
        Args:
            file_path_or_buffer: Either a file path or a file buffer from Streamlit
            
        Returns:
            tuple: (DataFrame, month_year_string, records_count)
        """
        try:
            # Read the Excel file - no 'sep' parameter needed for Excel files
            df = pd.read_excel(file_path_or_buffer)
            
            # Debug: Let's see what columns we actually have
            st.write("**Colunas encontradas no arquivo:**")
            st.write(list(df.columns))
            
            # Clean column names (remove extra spaces and standardize)
            df.columns = df.columns.str.strip()
            
            # Check if we have the expected 'Data' column
            if 'Data' not in df.columns:
                # Let's look for similar column names
                date_columns = [col for col in df.columns if 'data' in col.lower() or 'date' in col.lower()]
                if date_columns:
                    st.warning(f"Coluna 'Data' não encontrada. Encontradas estas colunas similares: {date_columns}")
                    st.info("Por favor, verifique os nomes das colunas no seu arquivo Excel.")
                else:
                    st.error("Nenhuma coluna de data encontrada no arquivo.")
                return None, None, 0
            
            # Convert 'Data' column to datetime
            # This handles different date formats that might exist in your files
            df['Data'] = pd.to_datetime(df['Data'], format='%d/%m/%Y', errors='coerce')
            
            # Check if we have any valid dates
            valid_dates = df['Data'].dropna()
            if len(valid_dates) == 0:
                st.error("Nenhuma data válida encontrada na coluna 'Data'.")
                return None, None, 0
            
            # Extract month and year from the first valid date
            # We assume all data in one file belongs to the same month
            first_date = valid_dates.iloc[0]
            month_year = first_date.strftime('%Y-%m')
            
            # Add month_year column for easier querying
            df['month_year'] = month_year
            
            # Show some statistics about the data
            st.write(f"**Prévia dos dados para {month_year}:**")
            st.write(f"- Total de linhas: {format_number(len(df))}")
            st.write(f"- Datas válidas: {format_number(len(valid_dates))}")
            st.write(f"- Período: {valid_dates.min().strftime('%d/%m/%Y')} até {valid_dates.max().strftime('%d/%m/%Y')}")
            
            return df, month_year, len(df)
            
        except Exception as e:
            st.error(f"Erro ao processar arquivo: {str(e)}")
            st.write("**Informações de debug:**")
            st.write(f"Tipo do erro: {type(e).__name__}")
            
            # Try to read just the first few rows to see the structure
            try:
                df_sample = pd.read_excel(file_path_or_buffer, nrows=5)
                st.write("**Primeiras 5 linhas do arquivo:**")
                st.dataframe(df_sample)
            except:
                st.write("Não foi possível ler nem uma amostra do arquivo.")
            
            return None, None, 0
    
    def insert_data(self, df, month_year):
        """
        Insert data into the database, handling month-level duplicates properly.
        
        Args:
            df (DataFrame): The processed data
            month_year (str): Month-year string for this data
            
        Returns:
            int: Number of records inserted
        """
        conn = sqlite3.connect(self.db_path)
        
        try:
            # First, check if this month already exists in the database
            existing_months_query = "SELECT DISTINCT month_year FROM commission_data WHERE month_year = ?"
            existing_check = pd.read_sql_query(existing_months_query, conn, params=[month_year])
            
            month_exists = len(existing_check) > 0
            
            if month_exists:
                # Get count of existing records for this month
                count_query = "SELECT COUNT(*) as count FROM commission_data WHERE month_year = ?"
                existing_count = pd.read_sql_query(count_query, conn, params=[month_year]).iloc[0]['count']
                
                st.warning(f"⚠️ **Mês {month_year} já existe no banco de dados!**")
                st.info(f"📊 Registros existentes para {month_year}: {format_number(existing_count)}")
                st.info(f"📊 Novos registros para inserir: {format_number(len(df))}")
                
                # Create columns for the choice buttons
                col1, col2 = st.columns(2)
                
                with col1:
                    replace_month = st.button(
                        f"🔄 Substituir {month_year}",
                        key=f"replace_{month_year}",
                        help="Deletar dados existentes deste mês e inserir novos dados"
                    )
                
                with col2:
                    skip_month = st.button(
                        f"⏭️ Pular {month_year}",
                        key=f"skip_{month_year}",
                        help="Manter dados existentes e ignorar o novo arquivo"
                    )
                
                if replace_month:
                    # Delete existing data for this month
                    delete_query = "DELETE FROM commission_data WHERE month_year = ?"
                    cursor = conn.cursor()
                    cursor.execute(delete_query, (month_year,))
                    deleted_count = cursor.rowcount
                    conn.commit()
                    
                    st.success(f"🗑️ {format_number(deleted_count)} registros existentes deletados para {month_year}")
                    
                    # Now insert all new data
                    return self._insert_all_records(df, month_year, conn)
                    
                elif skip_month:
                    st.info(f"⏭️ Inserção de dados para {month_year} foi pulada")
                    return 0
                
                else:
                    st.info("👆 Por favor, escolha uma ação acima para prosseguir.")
                    return 0
            
            else:
                # Month doesn't exist, insert all records
                st.success(f"✅ Novo mês {month_year} detectado. Inserindo todos os registros...")
                return self._insert_all_records(df, month_year, conn)
                
        except Exception as e:
            st.error(f"Erro ao verificar/inserir dados: {str(e)}")
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
            
            st.write("**Dados a serem inseridos (primeiras 3 linhas):**")
            st.dataframe(df_final.head(3))
            
            # Use pandas to_sql for bulk insert (much faster)
            df_final.to_sql('commission_data', conn, if_exists='append', index=False)
            
            inserted_count = len(df_final)
            st.success(f"✅ {format_number(inserted_count)} registros inseridos com sucesso para {month_year}")
            
            return inserted_count
            
        except Exception as e:
            st.error(f"Erro na inserção em lote: {str(e)}")
            return 0

    def _prepare_dataframe_for_insertion(self, df, month_year):
        """
        Prepare DataFrame for database insertion by handling data types and mapping columns.
        
        Args:
            df (DataFrame): The processed data
            month_year (str): Month-year string for this data
            
        Returns:
            DataFrame: Prepared DataFrame ready for insertion
        """
        # Clean data types first
        df = self.clean_data_types_on_insert(df)
        
        # Map DataFrame columns to database columns
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
            'month_year': 'month_year'
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
            'comissao_rs_assessor_indireto_iii', 'month_year'
        ]
        
        for col in db_columns:
            if col not in df_mapped.columns:
                df_mapped[col] = None
        
        # Select only the columns that exist in our database schema
        df_final = df_mapped[db_columns].copy()
        
        # Convert datetime columns to string format for SQLite
        if 'data' in df_final.columns and df_final['data'].dtype == 'datetime64[ns]':
            df_final['data'] = df_final['data'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Handle NaN values in numeric columns
        numeric_columns = [
            'receita_rs', 'receita_liquida_rs', 'repasse_perc_escritorio',
            'comissao_bruta_rs_escritorio', 'repasse_perc_assessor_direto',
            'comissao_rs_assessor_direto', 'repasse_perc_assessor_indireto_i',
            'comissao_rs_assessor_indireto_i', 'repasse_perc_assessor_indireto_ii',
            'comissao_rs_assessor_indireto_ii', 'repasse_perc_assessor_indireto_iii',
            'comissao_rs_assessor_indireto_iii'
        ]
        
        for col in numeric_columns:
            if col in df_final.columns:
                df_final[col] = df_final[col].where(pd.notna(df_final[col]), None)
        
        # Handle text columns - replace NaN with None
        text_columns = [
            'categoria', 'produto', 'nivel_1', 'nivel_2', 'nivel_3', 'cod_cliente',
            'cod_assessor_direto', 'cod_assessor_indireto_i', 'cod_assessor_indireto_ii',
            'cod_assessor_indireto_iii', 'month_year'
        ]
        
        for col in text_columns:
            if col in df_final.columns:
                df_final[col] = df_final[col].where(pd.notna(df_final[col]), None)
        
        return df_final
    
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
    
    def get_data_for_analysis(self, months=None, filters=None):
        """
        Retrieve data for analysis with optional filters.
        
        Args:
            months (list): List of months to include (None for all)
            filters (dict): Dictionary of filters to apply
            
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
    
    # Apply filters
    df_filtered = df[
        (~df['cod_cliente'].isin(['2733563', '5901578'])) &
        (df['categoria'].isin(['Renda Variável', 'Fundos Imobiliários', 'Produtos Financeiros'])) &
        (~df['produto'].isin(['OPERAÇÕES ESTRUTURADAS - FICC', 'COE']))
    ].copy()

    
    if df_filtered.empty:
        st.warning("⚠️ Nenhum dado encontrado após aplicar os filtros de Renda Variável.")
        return
    
    # Show filtering summary with enhanced metrics
    st.info(f"📊 **Resumo da Análise de Renda Variável:**")
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
    st.subheader("📈 Evolução das Comissões ao Longo do Tempo")
    
    # Total evolution with enhanced styling
    time_total = df_filtered.groupby('month_year')['comissao_bruta_rs_escritorio'].sum().reset_index()
    time_total.columns = ['Mês', 'Comissão Total']
    time_total = time_total.sort_values('Mês')
    
    fig_total = px.line(
        time_total,
        x='Mês',
        y='Comissão Total',
        title='Evolução Total das Comissões - Renda Variável',
        markers=True,
        labels={'Comissão Total': 'Comissão (R$)', 'Mês': 'Período'},
        line_shape='spline'
    )
    
    fig_total.update_traces(
        fill='tonexty',
        fillcolor='rgba(31, 119, 180, 0.2)',
        line=dict(color='rgba(31, 119, 180, 1)', width=3),
        marker=dict(size=10, color='rgba(31, 119, 180, 1)'),
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
        title='Evolução das Comissões por Categoria',
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
        title='Evolução das Comissões - Top 10 Produtos',
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
    
    # 2. Enhanced Monthly Analysis
    st.markdown("---")
    st.subheader("📅 Análise Mensal Detalhada")
    
    # Monthly summary table with better formatting
    monthly_summary = df_filtered.groupby('month_year').agg({
        'comissao_bruta_rs_escritorio': ['sum', 'count', 'mean', 'std'],
        'receita_rs': 'sum',
        'cod_cliente': 'nunique'
    }).round(2)
    
    # Flatten column names
    monthly_summary.columns = [
        'Comissão Total', 'Qtd Transações', 'Comissão Média', 'Desvio Padrão',
        'Receita Total', 'Clientes Únicos'
    ]
    monthly_summary = monthly_summary.reset_index()
    monthly_summary.columns = [
        'Mês', 'Comissão Total', 'Transações', 'Comissão Média', 
        'Desvio Padrão', 'Receita Total', 'Clientes Únicos'
    ]
    
    # Calculate growth rates
    monthly_summary['Crescimento Comissão (%)'] = monthly_summary['Comissão Total'].pct_change() * 100
    monthly_summary['Crescimento Comissão (%)'] = monthly_summary['Crescimento Comissão (%)'].round(2)
    
    # Format currency columns
    for col in ['Comissão Total', 'Comissão Média', 'Desvio Padrão', 'Receita Total']:
        monthly_summary[f'{col} Formatado'] = monthly_summary[col].apply(format_currency)
    
    # Display formatted table
    display_cols = ['Mês', 'Comissão Total Formatado', 'Transações', 'Comissão Média Formatado', 
                   'Receita Total Formatado', 'Clientes Únicos', 'Crescimento Comissão (%)']
    display_monthly = monthly_summary[display_cols].copy()
    display_monthly.columns = ['Mês', 'Comissão Total', 'Transações', 'Comissão Média', 
                              'Receita Total', 'Clientes Únicos', 'Crescimento (%)']
    
    st.dataframe(display_monthly, use_container_width=True)
    
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
        title='Distribuição de Comissões por Categoria',
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
    
    # Continue with remaining sections...
    # [The rest of the function continues with similar enhancements]

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
    
    # Continue with remaining sections using similar enhanced formatting...
    # [Rest of the function continues with similar improvements]

def load_cross_sell_clients(file_path="cross_sell_clients.txt"):
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

def create_date_range_selector(available_months, key_suffix=""):
    """
    Alternative: Create a date range selector for easier period selection.
    
    Args:
        available_months (list): List of available month-year strings
        key_suffix (str): Suffix for unique widget keys
        
    Returns:
        list: Selected months within the date range
    """
    if not available_months:
        return []
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("📅 Seleção por Período")
    
    # Convert month strings to dates for range selection
    month_dates = []
    for month_str in available_months:
        try:
            year, month = month_str.split('-')
            # Use first day of each month for date representation
            date_obj = datetime(int(year), int(month), 1)
            month_dates.append((date_obj, month_str))
        except ValueError:
            continue
    
    month_dates.sort(key=lambda x: x[0])  # Sort by date
    
    if not month_dates:
        return []
    
    min_date = month_dates[0][0]
    max_date = month_dates[-1][0]
    
    # Date range selector
    st.sidebar.markdown("**📊 Selecione o período:**")
    
    date_range = st.sidebar.date_input(
        "Período de análise:",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
        key=f"date_range_{key_suffix}",
        help="Selecione o período inicial e final para análise"
    )
    
    # Handle single date selection
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    elif isinstance(date_range, datetime.date):
        start_date = end_date = date_range
    else:
        return []
    
    # Filter months within the selected range
    selected_months = []
    for date_obj, month_str in month_dates:
        if start_date <= date_obj.date() <= end_date:
            selected_months.append(month_str)
    
    # Show selection summary
    if selected_months:
        st.sidebar.success(f"📊 **{len(selected_months)} mês(es) no período**")
        
        # Show period summary
        start_month = selected_months[0] if selected_months else ""
        end_month = selected_months[-1] if selected_months else ""
        
        if start_month and end_month:
            if start_month == end_month:
                st.sidebar.info(f"📅 Período: {start_month}")
            else:
                st.sidebar.info(f"📅 Período: {start_month} até {end_month}")
    else:
        st.sidebar.warning("⚠️ Nenhum mês no período selecionado")
    
    return selected_months

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
        conn.close()
        
        st.sidebar.metric("Total de Registros", format_number(total_records))
        if total_commission:
            st.sidebar.metric("Comissão Total", format_currency(total_commission))
        
        # Create the centralized month selector (only once)
        selected_months = create_centralized_month_selector(available_months)
    else:
        st.sidebar.warning("⚠️ Nenhum dado disponível")
        st.sidebar.info("Faça upload de arquivos na aba 'Upload de Dados'")
        selected_months = []
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📁 Upload de Dados", 
        "📈 Análise de Dados", 
        "💹 Renda Variável", 
        "🔄 Cross-Sell", 
        "🔍 Explorar Dados"
    ])
    
    with tab1:
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
            help="Faça upload de um ou mais arquivos Excel contendo dados de comissão"
        )
        
        if uploaded_files:
            for uploaded_file in uploaded_files:
                st.subheader(f"Processando: {uploaded_file.name}")
                
                # Process the file
                df, month_year, record_count = data_manager.process_xlsx_file(uploaded_file)
                
                if df is not None:
                    st.success(f"✅ Arquivo processado com sucesso!")
                    st.info(f"📅 Mês/Ano detectado: {month_year}")
                    st.info(f"📊 Registros encontrados: {format_number(record_count)}")
                    # Show a preview of the data
                    with st.expander("Visualizar dados"):
                        st.dataframe(df.head())
                    
                    # Insert data into database
                    if st.button(f"Processar dados de {uploaded_file.name}", key=f"process_{uploaded_file.name}"):
                        inserted_count = data_manager.insert_data(df, month_year)
                        if inserted_count > 0:
                            st.success(f"✅ {format_number(inserted_count)} registros processados com sucesso!")
                            # Clear the cache and rerun to refresh available months
                            st.cache_data.clear()
                            st.rerun()
                        else:
                            st.info("ℹ️ Nenhum registro foi inserido.")
        
        # Show current database status
        st.subheader("Status do Banco de Dados")
        if available_months:
            st.success(f"📅 Meses disponíveis: {', '.join(available_months)}")
            
            # Show record count per month
            conn = sqlite3.connect(data_manager.db_path)
            month_counts = pd.read_sql_query(
                "SELECT month_year as 'Mês', COUNT(*) as 'Qtd Registros' FROM commission_data GROUP BY month_year ORDER BY month_year",
                conn
            )
            conn.close()
            
            # Format the count column
            month_counts['Qtd Registros Formatada'] = month_counts['Qtd Registros'].apply(format_number)
            display_counts = month_counts[['Mês', 'Qtd Registros Formatada']].copy()
            display_counts.columns = ['Mês', 'Quantidade de Registros']
            
            st.dataframe(display_counts, use_container_width=True)
        else:
            st.info("Ainda não há dados no banco. Faça upload de alguns arquivos!")
    
    with tab2:
        st.header("Análise de Dados")
        
        if not available_months:
            st.warning("⚠️ Nenhum dado disponível. Por favor, faça upload de alguns arquivos primeiro!")
            return
        
        if not selected_months:
            st.warning("Por favor, selecione pelo menos um mês na barra lateral.")
            return
        
        # Get data for selected months
        df_analysis = data_manager.get_data_for_analysis(months=selected_months)
        
        if df_analysis.empty:
            st.warning("Nenhum dado encontrado para os meses selecionados.")
            return
        
        st.success(f"📊 Analisando {format_number(len(df_analysis))} registros de {len(selected_months)} mês(es)")
        
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

    with tab3:
        st.header("💹 Análise de Renda Variável")
        
        if not available_months:
            st.warning("⚠️ Nenhum dado disponível. Por favor, faça upload de alguns arquivos primeiro!")
            return
        
        if not selected_months:
            st.warning("Por favor, selecione pelo menos um mês na barra lateral.")
            return
        
        # Get data for selected months
        df_rv = data_manager.get_data_for_analysis(months=selected_months)
        
        if df_rv.empty:
            st.warning("Nenhum dado encontrado para os meses selecionados.")
            return
        
        # Apply Renda Variável analysis
        create_renda_variavel_analysis(df_rv)

    with tab4:
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
                    df_all = data_manager.get_data_for_analysis(months=selected_months)
                    
                    if not df_all.empty:
                        # Important: Ensure client codes are strings for matching
                        df_all['cod_cliente'] = df_all['cod_cliente'].astype(str).str.replace('\.0$', '', regex=True)
                        df_cross_sell = df_all[df_all['cod_cliente'].isin(cross_sell_clients)].copy()
                        
                        # Call the enhanced analysis function
                        create_cross_sell_analysis(df_cross_sell)
    
    with tab5:
        st.header("🔍 Explorador de Dados")
        st.markdown("Use esta ferramenta para explorar e filtrar os dados de forma interativa.")
        
        if not available_months:
            st.warning("⚠️ Nenhum dado disponível. Por favor, faça upload de alguns arquivos primeiro!")
            return
        
        if selected_months:
            df_explorer = data_manager.get_data_for_analysis(months=selected_months)
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


def create_renda_variavel_analysis(df):
    """Create comprehensive analysis for Renda Variável data with enhanced visualizations."""
    
    # Convert client codes to string for consistent comparison
    df['cod_cliente'] = df['cod_cliente'].astype(str).str.replace('\.0$', '', regex=True)
    
    # Apply filters
    df_filtered = df[
        (~df['cod_cliente'].isin(['2733563', '5901578'])) &
        (df['categoria'].isin(['Renda Variável', 'Fundos Imobiliários', 'Produtos Financeiros'])) &
        (~df['produto'].isin(['OPERAÇÕES ESTRUTURADAS - FICC', 'COE']))
    ].copy()

    
    if df_filtered.empty:
        st.warning("⚠️ Nenhum dado encontrado após aplicar os filtros de Renda Variável.")
        return
    
    # Show filtering summary with enhanced metrics
    st.info(f"📊 **Resumo da Análise de Renda Variável:**")
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
    st.subheader("📈 Evolução das Comissões ao Longo do Tempo")
    
    # Total evolution with enhanced styling
    time_total = df_filtered.groupby('month_year')['comissao_bruta_rs_escritorio'].sum().reset_index()
    time_total.columns = ['Mês', 'Comissão Total']
    time_total = time_total.sort_values('Mês')
    
    fig_total = px.line(
        time_total,
        x='Mês',
        y='Comissão Total',
        title='Evolução Total das Comissões - Renda Variável',
        markers=True,
        labels={'Comissão Total': 'Comissão (R$)', 'Mês': 'Período'},
        line_shape='spline'
    )
    
    fig_total.update_traces(
        fill='tonexty',
        fillcolor='rgba(31, 119, 180, 0.2)',
        line=dict(color='rgba(31, 119, 180, 1)', width=3),
        marker=dict(size=10, color='rgba(31, 119, 180, 1)'),
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
        title='Evolução das Comissões por Categoria',
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
        title='Evolução das Comissões - Top 10 Produtos',
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
    
    # 2. Enhanced Monthly Analysis
    st.markdown("---")
    st.subheader("📅 Análise Mensal Detalhada")
    
    # Monthly summary table with better formatting
    monthly_summary = df_filtered.groupby('month_year').agg({
        'comissao_bruta_rs_escritorio': ['sum', 'count', 'mean', 'std'],
        'receita_rs': 'sum',
        'cod_cliente': 'nunique'
    }).round(2)
    
    # Flatten column names
    monthly_summary.columns = [
        'Comissão Total', 'Qtd Transações', 'Comissão Média', 'Desvio Padrão',
        'Receita Total', 'Clientes Únicos'
    ]
    monthly_summary = monthly_summary.reset_index()
    monthly_summary.columns = [
        'Mês', 'Comissão Total', 'Transações', 'Comissão Média', 
        'Desvio Padrão', 'Receita Total', 'Clientes Únicos'
    ]
    
    # Calculate growth rates
    monthly_summary['Crescimento Comissão (%)'] = monthly_summary['Comissão Total'].pct_change() * 100
    monthly_summary['Crescimento Comissão (%)'] = monthly_summary['Crescimento Comissão (%)'].round(2)
    
    # Format currency columns
    for col in ['Comissão Total', 'Comissão Média', 'Desvio Padrão', 'Receita Total']:
        monthly_summary[f'{col} Formatado'] = monthly_summary[col].apply(format_currency)
    
    # Display formatted table
    display_cols = ['Mês', 'Comissão Total Formatado', 'Transações', 'Comissão Média Formatado', 
                   'Receita Total Formatado', 'Clientes Únicos', 'Crescimento Comissão (%)']
    display_monthly = monthly_summary[display_cols].copy()
    display_monthly.columns = ['Mês', 'Comissão Total', 'Transações', 'Comissão Média', 
                              'Receita Total', 'Clientes Únicos', 'Crescimento (%)']
    
    st.dataframe(display_monthly, use_container_width=True)
    
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
        title='Distribuição de Comissões por Categoria',
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
    
    # 4. Enhanced Product Analysis
    st.markdown("---")
    st.subheader("📦 Análise Detalhada por Produto")
    
    product_summary = df_filtered.groupby('produto').agg({
        'comissao_bruta_rs_escritorio': ['sum', 'count', 'mean'],
        'receita_rs': 'sum',
        'cod_cliente': 'nunique'
    }).round(2)
    
    product_summary.columns = ['Comissão Total', 'Transações', 'Comissão Média', 'Receita Total', 'Clientes Únicos']
    product_summary = product_summary.reset_index().sort_values('Comissão Total', ascending=False)
    
    # Top 15 products bar chart with enhanced styling
    fig_prod_bar = px.bar(
        product_summary.head(15),
        x='produto',
        y='Comissão Total',
        title='Top 15 Produtos por Comissão Total',
        labels={'Comissão Total': 'Comissão (R$)', 'produto': 'Produto'},
        color='Comissão Total',
        color_continuous_scale='Viridis',
        text='Comissão Total'
    )
    
    fig_prod_bar.update_traces(
        texttemplate='%{text:,.0f}',
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Comissão: R$ %{y:,.2f}<extra></extra>'
    )
    
    fig_prod_bar.update_layout(
        xaxis_tickangle=45,
        showlegend=False,
        height=600,
        font=dict(size=12),
        title_font_size=16
    )
    st.plotly_chart(fig_prod_bar, use_container_width=True)
    
    # Format and display product table
    for col in ['Comissão Total', 'Comissão Média', 'Receita Total']:
        product_summary[f'{col} Formatado'] = product_summary[col].apply(format_currency)
    
    display_prod = product_summary[['produto', 'Comissão Total Formatado', 'Transações', 
                                   'Comissão Média Formatado', 'Receita Total Formatado', 'Clientes Únicos']].copy()
    display_prod.columns = ['Produto', 'Comissão Total', 'Transações', 'Comissão Média', 'Receita Total', 'Clientes Únicos']
    st.dataframe(display_prod, use_container_width=True)
    
    # 5. Enhanced Client Analysis
    st.markdown("---")
    st.subheader("👥 Análise Detalhada por Cliente")
    
    client_summary = df_filtered.groupby('cod_cliente').agg({
        'comissao_bruta_rs_escritorio': ['sum', 'count', 'mean'],
        'receita_rs': 'sum'
    }).round(2)
    
    client_summary.columns = ['Comissão Total', 'Transações', 'Comissão Média', 'Receita Total']
    client_summary = client_summary.reset_index().sort_values('Comissão Total', ascending=False)
    
    # Client distribution with enhanced styling
    st.write("**Top 20 Clientes por Comissão:**")
    fig_client = px.bar(
        client_summary.head(20),
        x='cod_cliente',
        y='Comissão Total',
        title='Top 20 Clientes por Comissão Total',
        labels={'Comissão Total': 'Comissão (R$)', 'cod_cliente': 'Código do Cliente'},
        color='Comissão Total',
        color_continuous_scale='Plasma',
        text='Comissão Total'
    )
    
    fig_client.update_traces(
        texttemplate='%{text:,.0f}',
        textposition='outside',
        hovertemplate='<b>Cliente: %{x}</b><br>Comissão: R$ %{y:,.2f}<extra></extra>'
    )
    
    fig_client.update_layout(
        xaxis_tickangle=45,
        showlegend=False,
        height=500,
        font=dict(size=12),
        title_font_size=16
    )
    st.plotly_chart(fig_client, use_container_width=True)
    
    # Client concentration analysis
    total_clients = len(client_summary)
    top_10_pct = client_summary.head(int(total_clients * 0.1))['Comissão Total'].sum()
    total_commission_all = client_summary['Comissão Total'].sum()
    concentration_10 = (top_10_pct / total_commission_all) * 100
    
    st.info(f"📊 **Análise de Concentração de Clientes:**")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total de Clientes Únicos", format_number(total_clients))
    with col2:
        st.metric("Concentração Top 10%", f"{concentration_10:.1f}%")
    
    # 6. Enhanced Assessor Analysis
    st.markdown("---")
    st.subheader("🎯 Performance dos Assessores")
    
    assessor_summary = df_filtered.groupby('cod_assessor_direto').agg({
        'comissao_bruta_rs_escritorio': ['sum', 'count', 'mean'],
        'receita_rs': 'sum',
        'cod_cliente': 'nunique'
    }).round(2)
    
    assessor_summary.columns = ['Comissão Total', 'Transações', 'Comissão Média', 'Receita Total', 'Clientes Únicos']
    assessor_summary = assessor_summary.reset_index().sort_values('Comissão Total', ascending=False)
    
    # Assessor performance scatter plot with enhanced styling
    fig_assessor = px.scatter(
        assessor_summary.head(30),  # Top 30 assessors
        x='Transações',
        y='Comissão Total',
        size='Comissão Média',
        hover_data=['cod_assessor_direto', 'Clientes Únicos'],
        title='Performance dos Assessores: Volume vs Comissão (Top 30)',
        labels={'Comissão Total': 'Comissão (R$)', 'Transações': 'Número de Transações'},
        color='Clientes Únicos',
        color_continuous_scale='Turbo'
    )
    
    fig_assessor.update_traces(
        hovertemplate='<b>Assessor: %{customdata[0]}</b><br>' +
                     'Transações: %{x}<br>' +
                     'Comissão Total: R$ %{y:,.2f}<br>' +
                     'Clientes Únicos: %{color}<br>' +
                     'Comissão Média: R$ %{marker.size:,.2f}<extra></extra>'
    )
    
    fig_assessor.update_layout(
        height=500,
        font=dict(size=12),
        title_font_size=16
    )
    st.plotly_chart(fig_assessor, use_container_width=True)
    
    # Format and display assessor table
    for col in ['Comissão Total', 'Comissão Média', 'Receita Total']:
        assessor_summary[f'{col} Formatado'] = assessor_summary[col].apply(format_currency)
    
    display_assessor = assessor_summary.head(20)[['cod_assessor_direto', 'Comissão Total Formatado', 'Transações', 
                                                 'Comissão Média Formatado', 'Receita Total Formatado', 'Clientes Únicos']].copy()
    display_assessor.columns = ['Código Assessor', 'Comissão Total', 'Transações', 'Comissão Média', 'Receita Total', 'Clientes Únicos']
    st.dataframe(display_assessor, use_container_width=True)
    
    # 7. Enhanced Statistical Summary
    st.markdown("---")
    st.subheader("📊 Resumo Estatístico")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Estatísticas de Comissão:**")
        commission_stats = df_filtered['comissao_bruta_rs_escritorio'].describe()
        # Format the statistics
        formatted_stats = pd.DataFrame({
            'Estatística': ['Contagem', 'Média', 'Desvio Padrão', 'Mínimo', '25%', '50% (Mediana)', '75%', 'Máximo'],
            'Valor': [
                format_number(commission_stats['count']),
                format_currency(commission_stats['mean']),
                format_currency(commission_stats['std']),
                format_currency(commission_stats['min']),
                format_currency(commission_stats['25%']),
                format_currency(commission_stats['50%']),
                format_currency(commission_stats['75%']),
                format_currency(commission_stats['max'])
            ]
        })
        st.dataframe(formatted_stats, use_container_width=True)
    
    with col2:
        st.write("**Estatísticas de Receita:**")
        revenue_stats = df_filtered['receita_rs'].describe()
        # Format the statistics
        formatted_revenue_stats = pd.DataFrame({
            'Estatística': ['Contagem', 'Média', 'Desvio Padrão', 'Mínimo', '25%', '50% (Mediana)', '75%', 'Máximo'],
            'Valor': [
                format_number(revenue_stats['count']),
                format_currency(revenue_stats['mean']),
                format_currency(revenue_stats['std']),
                format_currency(revenue_stats['min']),
                format_currency(revenue_stats['25%']),
                format_currency(revenue_stats['50%']),
                format_currency(revenue_stats['75%']),
                format_currency(revenue_stats['max'])
            ]
        })
        st.dataframe(formatted_revenue_stats, use_container_width=True)
    
    # Commission distribution histogram with enhanced styling
    fig_hist = px.histogram(
        df_filtered,
        x='comissao_bruta_rs_escritorio',
        nbins=50,
        title='Distribuição das Comissões',
        labels={'comissao_bruta_rs_escritorio': 'Comissão (R$)', 'count': 'Frequência'},
        color_discrete_sequence=['#1f77b4']
    )
    
    fig_hist.update_traces(
        hovertemplate='Faixa: R$ %{x:,.2f}<br>Frequência: %{y}<extra></extra>'
    )
    
    fig_hist.update_layout(
        height=400,
        font=dict(size=12),
        title_font_size=16,
        showlegend=False
    )
    st.plotly_chart(fig_hist, use_container_width=True)

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
    
    # 2. Enhanced Monthly Analysis
    st.markdown("---")
    st.subheader("📅 Análise Mensal Detalhada")
    monthly_summary = df_filtered.groupby('month_year').agg({
        'comissao_bruta_rs_escritorio': ['sum', 'count', 'mean'],
        'receita_rs': 'sum', 'cod_cliente': 'nunique'
    }).round(2)
    monthly_summary.columns = ['Comissão Total', 'Transações', 'Comissão Média', 'Receita Total', 'Clientes Únicos']
    monthly_summary = monthly_summary.reset_index()
    monthly_summary.columns = ['Mês', 'Comissão Total', 'Transações', 'Comissão Média', 'Receita Total', 'Clientes Únicos']
    monthly_summary['Crescimento Comissão (%)'] = (monthly_summary['Comissão Total'].pct_change() * 100).round(2)
    
    # Format currency columns
    for col in ['Comissão Total', 'Comissão Média', 'Receita Total']:
        monthly_summary[f'{col} Formatado'] = monthly_summary[col].apply(format_currency)
    
    display_monthly = monthly_summary[['Mês', 'Comissão Total Formatado', 'Transações', 'Comissão Média Formatado', 
                                     'Receita Total Formatado', 'Clientes Únicos', 'Crescimento Comissão (%)']].copy()
    display_monthly.columns = ['Mês', 'Comissão Total', 'Transações', 'Comissão Média', 'Receita Total', 'Clientes Únicos', 'Crescimento (%)']
    st.dataframe(display_monthly, use_container_width=True)
    
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

if __name__ == "__main__":
    main()