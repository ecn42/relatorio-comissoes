import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import calendar
import os

# ============================================================================
# DATABASE AND UTILITY FUNCTIONS (from upload code)
# ============================================================================

def create_database_table(conn):
    """Create the table if it doesn't exist"""
    cursor = conn.cursor()
    
    create_table_query = """
    CREATE TABLE IF NOT EXISTS financial_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        Assessor TEXT,
        Cliente TEXT,
        Profissao TEXT,
        Sexo TEXT,
        Segmento TEXT,
        Data_de_Cadastro TEXT,
        Fez_Segundo_Aporte TEXT,
        Data_de_Nascimento TEXT,
        Status TEXT,
        Ativou_em_M TEXT,
        Evadiu_em_M TEXT,
        Operou_Bolsa TEXT,
        Operou_Fundo TEXT,
        Operou_Renda_Fixa TEXT,
        Aplicacao_Financeira_Declarada_Ajustada REAL,
        Receita_no_Mes REAL,
        Receita_Bovespa REAL,
        Receita_Futuros REAL,
        Receita_RF_Bancarios REAL,
        Receita_RF_Privados REAL,
        Receita_RF_Publicos REAL,
        Captacao_Bruta_em_M REAL,
        Resgate_em_M REAL,
        Captacao_Liquida_em_M REAL,
        Captacao_TED REAL,
        Captacao_ST REAL,
        Captacao_OTA REAL,
        Captacao_RF REAL,
        Captacao_TD REAL,
        Captacao_PREV REAL,
        Net_em_M_1 REAL,
        Net_Em_M REAL,
        Net_Renda_Fixa REAL,
        Net_Fundos_Imobiliarios REAL,
        Net_Renda_Variavel REAL,
        Net_Fundos REAL,
        Net_Financeiro REAL,
        Net_Previdencia REAL,
        Net_Outros REAL,
        Receita_Aluguel REAL,
        Receita_Complemento_Pacote_Corretagem REAL,
        Tipo_Pessoa TEXT,
        Data_Posicao TEXT,
        Data_Atualizacao TEXT
    )
    """
    
    cursor.execute(create_table_query)
    conn.commit()

def create_estruturadas_table(conn):
    """Create the estruturadas table if it doesn't exist"""
    cursor = conn.cursor()
    
    create_table_query = """
    CREATE TABLE IF NOT EXISTS estruturadas (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        Cliente TEXT,
        Data TEXT,
        Origem TEXT,
        Ativo TEXT,
        Estrategia TEXT,
        Comissao REAL,
        Cod_Matriz TEXT,
        Cod_A TEXT,
        Status_da_Operacao TEXT
    )
    """
    
    cursor.execute(create_table_query)
    conn.commit()

def parse_date(date_value):
    """Parse date from various formats and return datetime object"""
    if pd.isna(date_value):
        return None
    
    # If it's already a datetime object
    if isinstance(date_value, datetime):
        return date_value
    
    # Convert to string and try different formats
    date_str = str(date_value).strip()
    
    # Common date formats
    formats = [
        '%Y-%m-%d',
        '%d/%m/%Y',
        '%m/%d/%Y',
        '%Y-%m-%d %H:%M:%S',
        '%d/%m/%Y %H:%M:%S'
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    
    # If none of the formats work, try pandas to_datetime
    try:
        return pd.to_datetime(date_value)
    except:
        return None

def get_month_year_key(date_obj):
    """Get month-year key from datetime object"""
    if date_obj is None:
        return None
    return f"{date_obj.year}-{date_obj.month:02d}"

def load_cross_sell_clients(file_path="/mnt/databases/cross_sell_clients.txt"):
    """
    Loads client codes from a text file for cross-sell analysis.
    
    Args:
        file_path (str): The path to the text file.
        
    Returns:
        list: A list of unique client codes as strings.
    """
    if not os.path.exists(file_path):
        st.sidebar.warning(f"Arquivo de cross-sell '{os.path.basename(file_path)}' não encontrado.")
        return []
    try:
        with open(file_path, 'r') as f:
            # Read lines, strip whitespace, filter out empty lines, and get unique codes
            clients = {line.strip() for line in f if line.strip()}
        return list(clients)
    except Exception as e:
        st.sidebar.error(f"Erro ao carregar clientes de cross-sell: {e}")
        return []

# ============================================================================
# UPLOAD FUNCTIONALITY FUNCTIONS
# ============================================================================

def check_for_duplicates(df_new_cleaned, conn):
    """Ultra-safe duplicate detection using composite keys"""
    st.info("🔍 Starting duplicate detection...")
    
    # Load existing data from database
    existing_df = load_estruturadas_from_db(conn)
    
    if len(existing_df) == 0:
        st.info("🆕 Database is empty - all rows will be treated as new")
        return False, df_new_cleaned, []
    
    # Remove the 'id' column from existing data for comparison
    if 'id' in existing_df.columns:
        existing_df = existing_df.drop('id', axis=1)
    
    # Get matching columns
    new_columns = set(df_new_cleaned.columns)
    existing_columns = set(existing_df.columns)
    comparison_columns = list(new_columns.intersection(existing_columns))
    
    if not comparison_columns:
        st.error("❌ No matching columns found for comparison!")
        return False, df_new_cleaned, []
    
    st.info(f"🔍 Comparing on columns: {', '.join(comparison_columns)}")
    
    # Create composite keys for existing data
    def create_composite_key(row):
        values = []
        for col in comparison_columns:
            val = row[col]
            if pd.isna(val) or val is None:
                values.append('')
            else:
                # Normalize the value
                str_val = str(val).strip()
                try:
                    float_val = float(str_val)
                    if float_val.is_integer():
                        values.append(str(int(float_val)))
                    else:
                        values.append(f"{float_val:.10g}")
                except (ValueError, TypeError):
                    values.append(str_val.lower().strip())
        return '|'.join(values)
    
    # Get existing composite keys
    existing_keys = set()
    for _, row in existing_df.iterrows():
        key = create_composite_key(row)
        existing_keys.add(key)
    
    # Check new data and separate unique from duplicates
    unique_rows = []
    duplicate_details = []
    
    for idx, row in df_new_cleaned.iterrows():
        key = create_composite_key(row)
        
        if key in existing_keys:
            # This is a duplicate
            duplicate_details.append({
                'index': idx,
                'Cliente': row.get('Cliente', 'N/A'),
                'Data': row.get('Data', 'N/A'),
                'Ativo': row.get('Ativo', 'N/A'),
                'composite_key': key,
                'row_data': row.to_dict()
            })
        else:
            # This is unique
            unique_rows.append(row)
    
    # Create DataFrame from unique rows
    if unique_rows:
        df_unique = pd.DataFrame(unique_rows).reset_index(drop=True)
    else:
        df_unique = pd.DataFrame(columns=df_new_cleaned.columns)
    
    duplicates_found = len(duplicate_details) > 0
    
    st.info(f"🔍 Found {len(duplicate_details)} duplicates, {len(unique_rows)} unique rows")
    
    return duplicates_found, df_unique, duplicate_details

def display_duplicate_analysis(duplicates_found, duplicate_details, df_original, df_unique):
    """Display detailed duplicate analysis without nested expanders"""
    st.subheader("🔍 Duplicate Analysis")
    
    if duplicates_found:
        st.warning(f"⚠️ Found {len(duplicate_details)} duplicate row(s) that already exist in the database!")
        
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Processed Rows", len(df_original))
        with col2:
            st.metric("Duplicate Rows", len(duplicate_details))
        with col3:
            st.metric("Unique Rows", len(df_unique))
        
        # Show duplicate details
        st.markdown("### 📋 Duplicate Rows Details")
        
        # Create tabs for better organization if there are many duplicates
        if len(duplicate_details) <= 5:
            # Show all duplicates directly if there are few
            for i, dup in enumerate(duplicate_details, 1):
                st.markdown(f"**Duplicate {i}:**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write(f"Cliente: {dup['Cliente']}")
                with col2:
                    st.write(f"Data: {dup['Data']}")
                with col3:
                    st.write(f"Ativo: {dup['Ativo']}")
                
                # Show composite key for debugging
                if 'composite_key' in dup:
                    st.code(f"Key: {dup['composite_key']}", language=None)
                
                # Show full row data in a code block instead of expander
                st.markdown(f"**Full row data for duplicate {i}:**")
                st.json(dup['row_data'])
                
                if i < len(duplicate_details):  # Don't add divider after last item
                    st.divider()
        else:
            # For many duplicates, use a more compact display
            st.info(f"Showing first 5 of {len(duplicate_details)} duplicates:")
            for i, dup in enumerate(duplicate_details[:5], 1):
                with st.container():
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.write(f"**{i}.** {dup['Cliente']}")
                    with col2:
                        st.write(dup['Data'])
                    with col3:
                        st.write(dup['Ativo'])
                    with col4:
                        # Add a button to show details
                        if st.button(f"Show Details", key=f"dup_details_{i}"):
                            st.json(dup['row_data'])
            
            if len(duplicate_details) > 5:
                st.info(f"... and {len(duplicate_details) - 5} more duplicates")
        
        # Show what will be processed
        if len(df_unique) > 0:
            st.success(f"✅ {len(df_unique)} unique rows will be processed (duplicates excluded)")
        else:
            st.error("⚠️ No unique rows to process - all rows are duplicates!")
    
    else:
        st.success("✅ No duplicates found! All rows are unique.")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Processed Rows", len(df_original))
        with col2:
            st.metric("Unique Rows", len(df_unique))

def process_estruturadas_data(df):
    """Process estruturadas data according to specific requirements"""
    # Make a copy to avoid modifying the original
    df_processed = df.copy()
    
    # Remove last 3 rows
    if len(df_processed) > 3:
        df_processed = df_processed.iloc[:-3]
        st.info(f"🗑️ Removed last 3 rows from the data")
    
    # Filter only "Totalmente executado" status
    if 'Status da Operação' in df_processed.columns:
        initial_count = len(df_processed)
        df_processed = df_processed[
            df_processed['Status da Operação'] == 'Totalmente executado'
        ]
        filtered_count = len(df_processed)
        st.info(f"📊 Filtered data: {initial_count} → {filtered_count} rows (keeping only 'Totalmente executado')")
    else:
        st.warning("⚠️ Column 'Status da Operação' not found!")
    
    # Process "Cod A" column - remove first character from each row
    if 'Cod A' in df_processed.columns:
        def remove_first_char(value):
            if pd.isna(value):
                return value
            str_value = str(value)
            return str_value[1:] if len(str_value) > 0 else str_value
        
        df_processed['Cod A'] = df_processed['Cod A'].apply(remove_first_char)
        st.info("✂️ Removed first character from 'Cod A' column")
    else:
        st.warning("⚠️ Column 'Cod A' not found!")
    
    return df_processed

def clean_estruturadas_column_names(df):
    """Clean column names for estruturadas table"""
    column_mapping = {
        'Cliente': 'Cliente',
        'Data': 'Data',
        'Origem': 'Origem',
        'Ativo': 'Ativo',
        'Estratégia': 'Estrategia',
        'Comissão': 'Comissao',
        'Cod Matriz': 'Cod_Matriz',
        'Cod A': 'Cod_A',
        'Status da Operação': 'Status_da_Operacao'
    }
    
    df_cleaned = df.rename(columns=column_mapping)
    return df_cleaned

def clean_column_names(df):
    """Clean column names to match database schema"""
    column_mapping = {
        'Assessor': 'Assessor',
        'Cliente': 'Cliente',
        'Profissão': 'Profissao',
        'Sexo': 'Sexo',
        'Segmento': 'Segmento',
        'Data de Cadastro': 'Data_de_Cadastro',
        'Fez Segundo Aporte?': 'Fez_Segundo_Aporte',
        'Data de Nascimento': 'Data_de_Nascimento',
        'Status': 'Status',
        'Ativou em M?': 'Ativou_em_M',
        'Evadiu em M?': 'Evadiu_em_M',
        'Operou Bolsa?': 'Operou_Bolsa',
        'Operou Fundo?': 'Operou_Fundo',
        'Operou Renda Fixa?': 'Operou_Renda_Fixa',
        'Aplicação Financeira Declarada Ajustada': 'Aplicacao_Financeira_Declarada_Ajustada',
        'Receita no Mês': 'Receita_no_Mes',
        'Receita Bovespa': 'Receita_Bovespa',
        'Receita Futuros': 'Receita_Futuros',
        'Receita RF Bancários': 'Receita_RF_Bancarios',
        'Receita RF Privados': 'Receita_RF_Privados',
        'Receita RF Públicos': 'Receita_RF_Publicos',
        'Captação Bruta em M': 'Captacao_Bruta_em_M',
        'Resgate em M': 'Resgate_em_M',
        'Captação Líquida em M': 'Captacao_Liquida_em_M',
        'Captação TED': 'Captacao_TED',
        'Captação ST': 'Captacao_ST',
        'Captação OTA': 'Captacao_OTA',
        'Captação RF': 'Captacao_RF',
        'Captação TD': 'Captacao_TD',
        'Captação PREV': 'Captacao_PREV',
        'Net em M 1': 'Net_em_M_1',
        'Net Em M': 'Net_Em_M',
        'Net Renda Fixa': 'Net_Renda_Fixa',
        'Net Fundos Imobiliários': 'Net_Fundos_Imobiliarios',
        'Net Renda Variável': 'Net_Renda_Variavel',
        'Net Fundos': 'Net_Fundos',
        'Net Financeiro': 'Net_Financeiro',
        'Net Previdência': 'Net_Previdencia',
        'Net Outros': 'Net_Outros',
        'Receita Aluguel': 'Receita_Aluguel',
        'Receita Complemento Pacote Corretagem': 'Receita_Complemento_Pacote_Corretagem',
        'Tipo Pessoa': 'Tipo_Pessoa',
        'Data Posição': 'Data_Posicao',
        'Data Atualização': 'Data_Atualizacao'
    }
    
    df_cleaned = df.rename(columns=column_mapping)
    return df_cleaned

def insert_data_to_db(conn, df):
    """Insert DataFrame data into SQLite database"""
    try:
        df.to_sql('financial_data', conn, if_exists='append', index=False)
        return True
    except Exception as e:
        st.error(f"Error inserting data: {str(e)}")
        return False

def insert_estruturadas_to_db(conn, df):
    """Insert estruturadas DataFrame data into SQLite database"""
    try:
        df.to_sql('estruturadas', conn, if_exists='append', index=False)
        return True
    except Exception as e:
        st.error(f"Error inserting estruturadas data: {str(e)}")
        return False

def load_estruturadas_from_db(conn, cross_sell_clients=None):
    """Load all estruturadas data from database"""
    try:
        query = "SELECT * FROM estruturadas ORDER BY Data DESC"
        df = pd.read_sql_query(query, conn)
        return df
    except Exception as e:
        # Return empty dataframe if table doesn't exist or other error
        return pd.DataFrame()

def load_filtered_estruturadas_data(db_path, cross_sell_clients=None):
    """Load estruturadas data from database, filtered by cross-sell clients"""
    if not os.path.exists(db_path):
        return None

    try:
        conn = sqlite3.connect(db_path)
        if cross_sell_clients:
            # Filter by cross-sell clients
            placeholders = ','.join(['?' for _ in cross_sell_clients])
            query = f"""
            SELECT *
            FROM estruturadas
            WHERE Cliente IN ({placeholders})
            ORDER BY Data DESC
            """
            df = pd.read_sql_query(query, conn, params=cross_sell_clients)
        else:
            df = load_estruturadas_from_db(conn)  # Load all if no cross-sell clients
        conn.close()
        return df
    except Exception as e:
        return pd.DataFrame()

def clear_estruturadas_table(conn):
    """Clear all data from estruturadas table"""
    cursor = conn.cursor()
    cursor.execute("DELETE FROM estruturadas")
    conn.commit()
    return cursor.rowcount

def get_existing_data_posicao_by_month(conn):
    """Get existing Data Posição values grouped by month/year"""
    cursor = conn.cursor()
    query = "SELECT DISTINCT Data_Posicao FROM financial_data"
    cursor.execute(query)
    
    existing_dates = cursor.fetchall()
    month_year_dict = {}
    
    for (date_str,) in existing_dates:
        date_obj = parse_date(date_str)
        if date_obj:
            month_year = get_month_year_key(date_obj)
            if month_year not in month_year_dict:
                month_year_dict[month_year] = []
            month_year_dict[month_year].append({
                'date_str': date_str,
                'date_obj': date_obj
            })
    
    # Sort dates within each month and keep only the latest
    latest_by_month = {}
    for month_year, dates in month_year_dict.items():
        latest_date = max(dates, key=lambda x: x['date_obj'])
        latest_by_month[month_year] = latest_date
    
    return month_year_dict, latest_by_month

def delete_old_data_for_month_year(conn, month_year, keep_date_str):
    """Delete all data for a specific month/year except the one to keep"""
    cursor = conn.cursor()
    
    # Get all Data_Posicao values for this month/year
    query = "SELECT DISTINCT Data_Posicao FROM financial_data"
    cursor.execute(query)
    all_dates = cursor.fetchall()
    
    dates_to_delete = []
    for (date_str,) in all_dates:
        date_obj = parse_date(date_str)
        if date_obj and get_month_year_key(date_obj) == month_year and date_str != keep_date_str:
            dates_to_delete.append(date_str)
    
    # Delete old records
    deleted_count = 0
    for date_to_delete in dates_to_delete:
        delete_query = "DELETE FROM financial_data WHERE Data_Posicao = ?"
        cursor.execute(delete_query, (date_to_delete,))
        deleted_count += cursor.rowcount
    
    conn.commit()
    return deleted_count, dates_to_delete

def analyze_new_data_positions(df):
    """Analyze new data positions and group by month/year"""
    new_data_analysis = {}
    
    for _, row in df.iterrows():
        date_obj = parse_date(row['Data Posição'])
        if date_obj:
            month_year = get_month_year_key(date_obj)
            if month_year not in new_data_analysis:
                new_data_analysis[month_year] = []
            
            new_data_analysis[month_year].append({
                'date_str': str(row['Data Posição']),
                'date_obj': date_obj,
                'row_count': 1
            })
    
    # Find the latest date for each month/year in new data
    latest_new_by_month = {}
    for month_year, dates in new_data_analysis.items():
        latest_date = max(dates, key=lambda x: x['date_obj'])
        latest_new_by_month[month_year] = latest_date
    
    return new_data_analysis, latest_new_by_month

def display_data_posicao_summary(df, title="Data Posição Summary"):
    """Display a summary of Data Posição values organized by month/year"""
    st.subheader(f"📅 {title}")
    
    if 'Data_Posicao' in df.columns:
        column_name = 'Data_Posicao'
    elif 'Data Posição' in df.columns:
        column_name = 'Data Posição'
    else:
        st.warning("No Data Posição column found")
        return
    
    # Group by month/year
    month_year_summary = {}
    for _, row in df.iterrows():
        date_obj = parse_date(row[column_name])
        if date_obj:
            month_year = get_month_year_key(date_obj)
            if month_year not in month_year_summary:
                month_year_summary[month_year] = {
                    'dates': [],
                    'count': 0
                }
            month_year_summary[month_year]['dates'].append({
                'date_str': str(row[column_name]),
                'date_obj': date_obj
            })
            month_year_summary[month_year]['count'] += 1
    
    # Sort by year-month and display
    sorted_months = sorted(month_year_summary.keys(), reverse=True)
    
    for month_year in sorted_months:
        year, month = month_year.split('-')
        month_name = calendar.month_name[int(month)]
        summary = month_year_summary[month_year]
        
        # Find latest date in this month
        latest_date = max(summary['dates'], key=lambda x: x['date_obj'])
        
        with st.expander(f"📊 {month_name} {year} - {summary['count']} records - Latest: {latest_date['date_str']}"):
            # Sort dates within month
            sorted_dates = sorted(summary['dates'], key=lambda x: x['date_obj'], reverse=True)
            
            for i, date_info in enumerate(sorted_dates):
                if i == 0:  # Latest date
                    st.success(f"🟢 **Latest**: {date_info['date_str']}")
                else:
                    st.info(f"🔵 {date_info['date_str']}")

# ============================================================================
# DASHBOARD FUNCTIONS (from visualization code)
# ============================================================================

def fix_data_types(df):
    """Fix data types for proper sorting and calculations"""
    # Columns that should be numeric
    numeric_columns = [
        'Net_Em_M', 'Net_em_M_1', 'Net_Renda_Variavel', 'Net_Fundos_Imobiliarios',
        'Net_Financeiro', 'Receita_no_Mes', 'Receita_Bovespa', 'Receita_Futuros',
        'Receita_RF_Bancarios', 'Receita_RF_Privados', 'Receita_RF_Publicos'
    ]
    
    # Convert numeric columns
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Boolean/categorical columns that should be standardized
    boolean_columns = ['Operou_Bolsa', 'Ativou_em_M']
    for col in boolean_columns:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    
    return df

def load_estruturadas_data(db_path):
    """Load estruturadas data from database"""
    if not os.path.exists(db_path):
        return None
    
    try:
        conn = sqlite3.connect(db_path)
        query = """
        SELECT 
            Cliente,
            Data,
            Cod_A as Assessor,
            Comissao,
            Ativo,
            Estrategia,
            Status_da_Operacao
        FROM estruturadas 
        WHERE Cod_A IS NOT NULL AND Cod_A != ''
        """
        df_estruturadas = pd.read_sql_query(query, conn)
        conn.close()
        
        # Apply 0.8 multiplier to Comissao
        df_estruturadas['Comissao_Estruturada'] = pd.to_numeric(
            df_estruturadas['Comissao'], errors='coerce'
        ) * 0.8
        
        # Parse dates
        df_estruturadas['Data_Parsed'] = df_estruturadas['Data'].apply(parse_date)
        df_estruturadas['Month_Year'] = df_estruturadas['Data_Parsed'].apply(
            lambda x: get_month_year_key(x) if x else None
        )
        
        return df_estruturadas
    except Exception as e:
        st.error(f"Error loading estruturadas data: {str(e)}")
        return None

def load_data_from_db(db_path):
    """Load all data from database into a pandas DataFrame"""
    if not os.path.exists(db_path):
        return None
    
    try:
        conn = sqlite3.connect(db_path)
        query = """
        SELECT * FROM financial_data 
        ORDER BY 
            CASE 
                WHEN Data_Posicao GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]*' 
                THEN Data_Posicao 
                ELSE '9999-12-31' 
            END DESC
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        # Fix data types
        df = fix_data_types(df)
        
        return df
    except Exception as e:
        st.error(f"Error loading database: {str(e)}")
        return None

def apply_receita_multiplier(df):
    """Apply 0.5 * 0.8 multiplier to all Receita columns"""
    df_copy = df.copy()
    
    # Find all columns that start with "Receita"
    receita_columns = [col for col in df_copy.columns if col.startswith('Receita')]
    
    # Apply multiplier (0.5 * 0.8 = 0.4)
    for col in receita_columns:
        df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce') * 0.4
    
    return df_copy, receita_columns

def prepare_data_with_dates(df):
    """Add parsed date columns and month-year information"""
    df_copy = df.copy()
    
    # Parse Data_Posicao
    df_copy['Data_Posicao_Parsed'] = df_copy['Data_Posicao'].apply(parse_date)
    
    # Create month-year columns
    df_copy['Month_Year'] = df_copy['Data_Posicao_Parsed'].apply(
        lambda x: get_month_year_key(x) if x else None
    )
    
    # Create readable month-year labels
    df_copy['Month_Year_Label'] = df_copy['Data_Posicao_Parsed'].apply(
        lambda x: f"{calendar.month_name[x.month]} {x.year}" if x else None
    )
    
    # Create separate year and month columns
    df_copy['Year'] = df_copy['Data_Posicao_Parsed'].apply(
        lambda x: x.year if x else None
    )
    df_copy['Month'] = df_copy['Data_Posicao_Parsed'].apply(
        lambda x: x.month if x else None
    )
    df_copy['Month_Name'] = df_copy['Month'].apply(
        lambda x: calendar.month_name[x] if x else None
    )
    
    return df_copy

def get_available_months(df):
    """Get list of available months from the data"""
    if 'Month_Year' not in df.columns:
        return []
    
    available_months = df['Month_Year'].dropna().unique()
    
    # Sort months chronologically
    month_data = []
    for month_year in available_months:
        if month_year:
            year, month = month_year.split('-')
            month_data.append({
                'month_year': month_year,
                'year': int(year),
                'month': int(month),
                'label': f"{calendar.month_name[int(month)]} {year}"
            })
    
    # Sort by year and month (newest first)
    month_data.sort(key=lambda x: (x['year'], x['month']), reverse=True)
    
    return month_data

def get_operou_bolsa_analysis(df_filtered):
    """Analyze clients who did not operate in stock market (Operou_Bolsa = Não)"""
    
    # Filter clients who did not operate in stock market
    nao_operou = df_filtered[df_filtered['Operou_Bolsa'] == 'Não']
    
    if nao_operou.empty:
        return pd.DataFrame(), pd.DataFrame()
    
    # Count by assessor
    operou_summary = df_filtered.groupby('Assessor').agg({
        'Cliente': 'count',  # Total clients
        'Net_Em_M': 'sum'    # Total patrimony
    }).reset_index()
    operou_summary = operou_summary.rename(columns={'Cliente': 'Total_Clientes'})
    
    # Count clients who didn't operate
    nao_operou_summary = nao_operou.groupby('Assessor').agg({
        'Cliente': 'count',  # Clients who didn't operate
        'Net_Em_M': 'sum'    # Their patrimony
    }).reset_index()
    nao_operou_summary = nao_operou_summary.rename(columns={
        'Cliente': 'Clientes_Nao_Operou',
        'Net_Em_M': 'Patrimonio_Nao_Operou'
    })
    
    # Merge summaries
    operou_analysis = operou_summary.merge(
        nao_operou_summary, 
        on='Assessor', 
        how='left'
    )
    
    # Fill NaN values
    operou_analysis['Clientes_Nao_Operou'] = operou_analysis['Clientes_Nao_Operou'].fillna(0)
    operou_analysis['Patrimonio_Nao_Operou'] = operou_analysis['Patrimonio_Nao_Operou'].fillna(0)
    
    # Calculate percentages
    operou_analysis['Percentual_Nao_Operou'] = (
        operou_analysis['Clientes_Nao_Operou'] / operou_analysis['Total_Clientes'] * 100
    ).fillna(0)
    
    # Sort by percentage
    operou_analysis = operou_analysis.sort_values('Percentual_Nao_Operou', ascending=False)
    
    return operou_analysis, nao_operou

def get_ativou_analysis(df_filtered):
    """Analyze clients who activated in the month (Ativou_em_M = Sim)"""
    
    # Filter clients who activated
    ativou = df_filtered[df_filtered['Ativou_em_M'] == 'Sim']
    
    if ativou.empty:
        return pd.DataFrame()
    
    # Select relevant columns
    ativou_details = ativou[[
        'Assessor', 'Cliente', 'Net_Em_M', 'Net_Renda_Variavel', 
        'Net_Financeiro', 'Tipo_Pessoa', 'Data_Posicao'
    ]].copy()
    
    # Sort by Net_Em_M descending
    ativou_details = ativou_details.sort_values('Net_Em_M', ascending=False)
    
    return ativou_details

def calculate_new_metrics(df_filtered, estruturadas_summary):
    """Calculate the new metrics: Variação PL, ROA Total, ROA Estruturadas, ROA RV, Alocação RV"""
    
    # Group by Assessor to get aggregated values
    metrics_data = df_filtered.groupby('Assessor').agg({
        'Net_Em_M': 'sum',
        'Net_em_M_1': 'sum',
        'Net_Renda_Variavel': 'sum',
        'Net_Fundos_Imobiliarios': 'sum',
        'Net_Financeiro': 'sum',
        'Receita_no_Mes': 'sum',
        'Receita_Bovespa': 'sum',
        'Receita_Futuros': 'sum'
    }).reset_index()
    
    # Merge with estruturadas summary
    if not estruturadas_summary.empty:
        metrics_data = metrics_data.merge(
            estruturadas_summary[['Assessor', 'Comissao_Estruturada']], 
            on='Assessor', 
            how='left'
        )
    else:
        metrics_data['Comissao_Estruturada'] = 0
    
    metrics_data['Comissao_Estruturada'] = metrics_data['Comissao_Estruturada'].fillna(0)
    
    # Calculate new metrics
    # 1. Variação PL
    metrics_data['Variacao_PL'] = metrics_data['Net_Em_M'] - metrics_data['Net_em_M_1']
    
    # 2. ROA Total = Receita_Total / Net_Em_M
    metrics_data['Receita_Total'] = metrics_data['Receita_no_Mes'] + metrics_data['Comissao_Estruturada']
    metrics_data['ROA_Total'] = (metrics_data['Receita_Total'] / metrics_data['Net_Em_M'] * 100).replace([float('inf'), -float('inf')], 0)
    
    # 3. ROA Estruturadas = Comissao_Estruturada / Net_Em_M
    metrics_data['ROA_Estruturadas'] = (metrics_data['Comissao_Estruturada'] / metrics_data['Net_Em_M'] * 100).replace([float('inf'), -float('inf')], 0)
    
    # 4. ROA RV = (Receita_Bovespa + Receita_Futuros) / (Net_Renda_Variavel + Net_Fundos_Imobiliarios)
    metrics_data['Receita_RV'] = metrics_data['Receita_Bovespa'] + metrics_data['Receita_Futuros']
    metrics_data['Net_RV_Total'] = metrics_data['Net_Renda_Variavel'] + metrics_data['Net_Fundos_Imobiliarios']
    metrics_data['ROA_RV'] = (metrics_data['Receita_RV'] / metrics_data['Net_RV_Total'] * 100).replace([float('inf'), -float('inf')], 0)
    
    # 5. Alocação RV = (Net_Renda_Variavel + Net_Fundos_Imobiliarios) / Net_Em_M
    metrics_data['Alocacao_RV'] = (metrics_data['Net_RV_Total'] / metrics_data['Net_Em_M'] * 100).replace([float('inf'), -float('inf')], 0)
    
    # Handle NaN values
    numeric_columns = ['Variacao_PL', 'ROA_Total', 'ROA_Estruturadas', 'ROA_RV', 'Alocacao_RV']
    for col in numeric_columns:
        metrics_data[col] = metrics_data[col].fillna(0)
    
    return metrics_data

def get_estruturadas_summary(df_estruturadas, selected_months, selected_assessores, cross_sell_clients=None, client_type_filter="Todos"):
    """Get estruturadas summary for selected filters, including cross-sell client filtering"""
    if df_estruturadas is None or df_estruturadas.empty:
        return pd.DataFrame()
    
    # Filter estruturadas data by month and assessor
    df_estruturadas_filtered = df_estruturadas[
        (df_estruturadas['Month_Year'].isin(selected_months)) &
        (df_estruturadas['Assessor'].isin(selected_assessores))
    ]
    
    if df_estruturadas_filtered.empty:
        return pd.DataFrame()
    
    # Apply client type filter if specified
    if client_type_filter != "Todos" and cross_sell_clients:
        # Ensure client codes are strings for comparison
        df_estruturadas_filtered['Cliente'] = df_estruturadas_filtered['Cliente'].astype(str)
        
        if client_type_filter == "Apenas Cross-Sell":
            df_estruturadas_filtered = df_estruturadas_filtered[
                df_estruturadas_filtered['Cliente'].isin(cross_sell_clients)
            ]
        elif client_type_filter == "Apenas Normais":
            df_estruturadas_filtered = df_estruturadas_filtered[
                ~df_estruturadas_filtered['Cliente'].isin(cross_sell_clients)
            ]
    
    if df_estruturadas_filtered.empty:
        return pd.DataFrame()
    
    # Group by Assessor and sum
    estruturadas_summary = df_estruturadas_filtered.groupby('Assessor').agg({
        'Comissao_Estruturada': 'sum',
        'Ativo': 'count'  # Count of operations
    }).reset_index()
    estruturadas_summary = estruturadas_summary.rename(columns={'Ativo': 'Operacoes_Estruturadas'})
    
    return estruturadas_summary

def create_receita_by_assessor_chart(df_filtered, estruturadas_summary, chart_type="bar"):
    """Create chart showing Receita Total by Assessor"""
    
    # Group traditional revenue by Assessor
    receita_by_assessor = df_filtered.groupby('Assessor')['Receita_no_Mes'].sum().reset_index()
    
    # Merge with estruturadas summary
    if not estruturadas_summary.empty:
        receita_by_assessor = receita_by_assessor.merge(
            estruturadas_summary[['Assessor', 'Comissao_Estruturada', 'Operacoes_Estruturadas']], 
            on='Assessor', 
            how='left'
        )
    else:
        receita_by_assessor['Comissao_Estruturada'] = 0
        receita_by_assessor['Operacoes_Estruturadas'] = 0
    
    # Fill NaN values
    receita_by_assessor['Comissao_Estruturada'] = receita_by_assessor['Comissao_Estruturada'].fillna(0)
    receita_by_assessor['Operacoes_Estruturadas'] = receita_by_assessor['Operacoes_Estruturadas'].fillna(0)
    
    # Calculate total revenue
    receita_by_assessor['Receita_Total'] = receita_by_assessor['Receita_no_Mes'] + receita_by_assessor['Comissao_Estruturada']
    
    # Sort by total revenue
    receita_by_assessor = receita_by_assessor.sort_values('Receita_Total', ascending=False)
    
    # Format values for display
    receita_by_assessor['Receita_Formatted'] = receita_by_assessor['Receita_Total'].apply(
        lambda x: f"R$ {x:,.2f}" if pd.notna(x) else "R$ 0,00"
    )
    
    if chart_type == "bar":
        fig = px.bar(
            receita_by_assessor,
            x='Assessor',
            y='Receita_Total',
            title='Receita Total por Assessor (Tradicional × 0.4 + Estruturadas × 0.8)',
            labels={
                'Receita_Total': 'Receita Total (R$)',
                'Assessor': 'Assessor'
            },
            text='Receita_Formatted'
        )
        
        fig.update_traces(
            texttemplate='%{text}',
            textposition='outside'
        )
        
        fig.update_layout(
            xaxis_tickangle=-45,
            height=600,
            showlegend=False
        )
        
    elif chart_type == "pie":
        fig = px.pie(
            receita_by_assessor,
            values='Receita_Total',
            names='Assessor',
            title='Distribuição da Receita Total por Assessor (Tradicional × 0.4 + Estruturadas × 0.8)'
        )
        
        fig.update_traces(
            textposition='inside',
            textinfo='percent+label',
            hovertemplate='<b>%{label}</b><br>Receita: R$ %{value:,.2f}<br>Percentual: %{percent}<extra></extra>'
        )
        
        fig.update_layout(height=600)
    
    elif chart_type == "horizontal_bar":
        fig = px.bar(
            receita_by_assessor,
            x='Receita_Total',
            y='Assessor',
            orientation='h',
            title='Receita Total por Assessor (Tradicional × 0.4 + Estruturadas × 0.8)',
            labels={
                'Receita_Total': 'Receita Total (R$)',
                'Assessor': 'Assessor'
            },
            text='Receita_Formatted'
        )
        
        fig.update_traces(
            texttemplate='%{text}',
            textposition='outside'
        )
        
        fig.update_layout(
            height=max(400, len(receita_by_assessor) * 40),
            showlegend=False
        )
    
    # Format y-axis to show currency
    fig.update_layout(
        yaxis_tickformat=',.0f' if chart_type != "horizontal_bar" else None,
        xaxis_tickformat=',.0f' if chart_type == "horizontal_bar" else None,
        template='plotly_white'
    )
    
    return fig, receita_by_assessor

def create_summary_metrics(df_filtered, estruturadas_summary):
    """Create summary metrics for the filtered data"""
    total_receita_tradicional = df_filtered['Receita_no_Mes'].sum()
    total_comissao_estruturada = estruturadas_summary['Comissao_Estruturada'].sum() if not estruturadas_summary.empty else 0
    total_receita = total_receita_tradicional + total_comissao_estruturada
    
    total_assessores = df_filtered['Assessor'].nunique()
    avg_receita_per_assessor = total_receita / total_assessores if total_assessores > 0 else 0
    total_clients = df_filtered['Cliente'].nunique()
    total_operacoes_estruturadas = estruturadas_summary['Operacoes_Estruturadas'].sum() if not estruturadas_summary.empty else 0
    
    return {
        'total_receita': total_receita,
        'total_receita_tradicional': total_receita_tradicional,
        'total_comissao_estruturada': total_comissao_estruturada,
        'total_assessores': total_assessores,
        'avg_receita_per_assessor': avg_receita_per_assessor,
        'total_clients': total_clients,
        'total_operacoes_estruturadas': total_operacoes_estruturadas
    }

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    st.set_page_config(
        page_title="Dash Positivador",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("📊 Dashboard Positivador - Acompanhamento mensal P1")
    st.markdown("Mostrando dados Positivador + Estruturadas")
    
    # Database file path
    db_path = "/mnt/databases/financial_data.db"
    
    # Connect to database and create tables
    conn = sqlite3.connect(db_path)
    create_database_table(conn)
    create_estruturadas_table(conn)
    conn.close()
    
# Create main tabs - UPDATED with upload tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
        "📊 Dashboard", 
        "📈 Análise de Receita",  # Translated from "Revenue Analysis"
        "🎯 Portfólio & ROA",    # Translated from "Portfolio & ROA"
        "🚫 Não Operaram",       # Translated from "Non-Operators"
        "✅ Clientes Ativados",  # Translated from "Activated Clients"
        "📤 Upload Dados Financeiros",  # Translated from "Upload Financial Data"
        "🏗️ Upload Estruturadas",   # Translated from "Upload Estruturadas"
        "🗄️ Gerenciador BD",      # Translated from "Database Manager" (abbreviated for tab space)
        "ℹ️ Ajuda & Info"            # Translated from "Help & Info"
    ])
    
    # Check if database exists and has data
    db_exists = os.path.exists(db_path)
    has_data = False
    
    if db_exists:
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM financial_data")
            count = cursor.fetchone()[0]
            has_data = count > 0
            conn.close()
        except:
            has_data = False
    
    # ============================================================================
    # TAB 1: DASHBOARD (Main Overview)
    # ============================================================================
# ============================================================================
# TAB 1: DASHBOARD (Main Overview) - TRANSLATED
# ============================================================================
    with tab1:
        st.header("📊 Visão Geral do Dashboard Financeiro")
        
        if not has_data:
            st.warning("⚠️ Nenhum dado encontrado no banco de dados! Por favor, faça upload dos dados primeiro usando as abas de upload.")
            st.info("💡 Use as abas 'Upload Dados Financeiros' ou 'Upload Estruturadas' para começar.")
            
            # Show quick stats about database
            if db_exists:
                st.info("✅ Arquivo do banco de dados existe mas está vazio")
            else:
                st.info("❌ Arquivo do banco de dados ainda não existe")
            
            return
        
        # Load data for dashboard
        with st.spinner("Carregando dados para o dashboard..."):
            df = load_data_from_db(db_path)
            df_estruturadas = load_estruturadas_data(db_path)
        
        if df is None or df.empty:
            st.error("❌ Falha ao carregar dados do banco de dados!")
            return
        
        # Apply receita multiplier
        df_adjusted, receita_columns = apply_receita_multiplier(df)
        
        # Prepare data with date parsing
        df_prepared = prepare_data_with_dates(df_adjusted)
        
        # Get available months
        available_months = get_available_months(df_prepared)
        
        if not available_months:
            st.error("❌ Nenhuma data válida encontrada nos dados!")
            return
        
        # Quick filters for dashboard
        st.subheader("🔍 Filtros Rápidos")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Month selection (simplified for dashboard)
            month_options = {month['label']: month['month_year'] for month in available_months}
            selected_month_label = st.selectbox(
                "Selecionar Mês:",
                options=list(month_options.keys()),
                index=0,
                key="dashboard_month"
            )
            selected_months = [month_options[selected_month_label]]
        
        with col2:
            # Assessor filter (top 10 for dashboard)
            available_assessores = sorted(df_prepared['Assessor'].dropna().unique())
            # top_assessores = available_assessores[:10] if len(available_assessores) > 10 else available_assessores
            selected_assessores = st.multiselect(
                "Selecionar Assessores:",
                options=available_assessores,
                default=available_assessores,
                key="dashboard_assessors"
            )
        
        with col3:
            # Tipo Pessoa filter
            available_tipo_pessoa = sorted(df_prepared['Tipo_Pessoa'].dropna().unique())
            selected_tipo_pessoa = st.multiselect(
                "Selecionar Tipo Pessoa:",
                options=available_tipo_pessoa,
                default=available_tipo_pessoa,
                key="dashboard_tipo"
            )
        
        # Filter data
        df_filtered = df_prepared[
            (df_prepared['Month_Year'].isin(selected_months)) &
            (df_prepared['Assessor'].isin(selected_assessores)) &
            (df_prepared['Tipo_Pessoa'].isin(selected_tipo_pessoa)) &
            (df_prepared['Receita_no_Mes'].notna())
        ]
        
        if df_filtered.empty:
            st.warning("⚠️ Nenhum dado encontrado para os filtros selecionados!")
            return
        
        # Get estruturadas summary
        # Get estruturadas summary
        estruturadas_summary = get_estruturadas_summary(df_estruturadas, selected_months, selected_assessores)
        
        # Create summary metrics
        summary_metrics = create_summary_metrics(df_filtered, estruturadas_summary)
        
        # Display key metrics
        st.subheader("📊 Métricas Principais")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                "Receita Total",
                f"R$ {summary_metrics['total_receita']:,.2f}"
            )
        
        with col2:
            st.metric(
                "Total Assessores",
                f"{summary_metrics['total_assessores']:,}"
            )
        
        with col3:
            st.metric(
                "Total Clientes",
                f"{summary_metrics['total_clients']:,}"
            )
        
        with col4:
            structured_percentage = (summary_metrics['total_comissao_estruturada'] / summary_metrics['total_receita'] * 100) if summary_metrics['total_receita'] > 0 else 0
            st.metric(
                "% Estruturadas",
                f"{structured_percentage:.1f}%"
            )
        
        with col5:
            st.metric(
                "Receita Média/Assessor",
                f"R$ {summary_metrics['avg_receita_per_assessor']:,.2f}"
            )
        
        # Quick charts
        st.subheader("📈 Visão Geral Rápida da Receita")
        
        # Create revenue chart
        fig, receita_data = create_receita_by_assessor_chart(df_filtered, estruturadas_summary, "bar")
        st.plotly_chart(fig, use_container_width=True, key='receita_by_assessor_chart_1')
        
        # Show top performers table
        st.subheader("🏆 Melhores Desempenhos")
        top_performers = receita_data.head(10).copy()
        
        # Format for display
        top_performers['Receita_Total_Display'] = top_performers['Receita_Total'].apply(
            lambda x: f"R$ {x:,.2f}"
        )
        top_performers['Receita_Tradicional_Display'] = top_performers['Receita_no_Mes'].apply(
            lambda x: f"R$ {x:,.2f}"
        )
        top_performers['Estruturadas_Display'] = top_performers['Comissao_Estruturada'].apply(
            lambda x: f"R$ {x:,.2f}"
        )
        
        display_top = top_performers[['Assessor', 'Receita_Total_Display', 'Receita_Tradicional_Display', 'Estruturadas_Display']].rename(columns={
            'Assessor': 'Assessor',
            'Receita_Total_Display': 'Receita Total',
            'Receita_Tradicional_Display': 'Receita Tradicional',
            'Estruturadas_Display': 'Receita Estruturada'
        })
        
        st.dataframe(display_top, use_container_width=True, hide_index=True)
        
        # Navigation help
        st.info("💡 **Dica**: Use as outras abas para análises detalhadas, upload de dados e gerenciamento do banco de dados.")


        # TAB 2: REVENUE ANALYSIS (Detailed revenue analysis) - TRANSLATED
        # ============================================================================
        with tab2:
            st.header("📈 Análise Detalhada de Receita")
            
            if not has_data:
                st.warning("⚠️ Nenhum dado disponível. Por favor, faça upload dos dados primeiro.")
                return
            
            # Load data
            with st.spinner("Carregando dados para análise de receita..."):
                df = load_data_from_db(db_path)
                df_estruturadas = load_estruturadas_data(db_path)
            
            if df is None or df.empty:
                st.error("❌ Falha ao carregar dados!")
                return
            
            # Apply receita multiplier and prepare data
            df_adjusted, receita_columns = apply_receita_multiplier(df)
            df_prepared = prepare_data_with_dates(df_adjusted)
            available_months = get_available_months(df_prepared)
            
            # Detailed filters (similar to original dashboard)
            st.sidebar.header("🔍 Filtros de Análise de Receita")
            
            # Month selection
            st.sidebar.subheader("📅 Selecionar Mês(es)")
            month_selection_type = st.sidebar.radio(
                "Tipo de Seleção:",
                ["Mês Único", "Múltiplos Meses", "Todos os Meses"],
                key="revenue_month_type"
            )
            
            if month_selection_type == "Todos os Meses":
                selected_months = [month['month_year'] for month in available_months]
                st.sidebar.info(f"✅ Selecionado: Todos os {len(available_months)} meses")
            elif month_selection_type == "Mês Único":
                month_options = {month['label']: month['month_year'] for month in available_months}
                selected_month_label = st.sidebar.selectbox(
                    "Escolher Mês:",
                    options=list(month_options.keys()),
                    index=0,
                    key="revenue_single_month"
                )
                selected_months = [month_options[selected_month_label]]
            else:  # Multiple Months
                month_options = {month['label']: month['month_year'] for month in available_months}
                selected_month_labels = st.sidebar.multiselect(
                    "Escolher Múltiplos Meses:",
                    options=list(month_options.keys()),
                    default=[list(month_options.keys())[0]],
                    key="revenue_multi_months"
                )
                selected_months = [month_options[label] for label in selected_month_labels]
            
            # Assessor and Tipo Pessoa filters
            available_assessores = sorted(df_prepared['Assessor'].dropna().unique())
            available_tipo_pessoa = sorted(df_prepared['Tipo_Pessoa'].dropna().unique())
            
            assessor_filter_type = st.sidebar.radio(
                "Filtro de Assessor:",
                ["Todos os Assessores", "Selecionar Específicos"],
                key="revenue_assessor_type"
            )
            
            if assessor_filter_type == "Selecionar Específicos":
                selected_assessores = st.sidebar.multiselect(
                    "Escolher Assessores:",
                    options=available_assessores,
                    default=available_assessores[:10] if len(available_assessores) > 10 else available_assessores,
                    key="revenue_assessors"
                )
            else:
                selected_assessores = available_assessores
            
            selected_tipo_pessoa = st.sidebar.multiselect(
                "Escolher Tipo Pessoa:",
                options=available_tipo_pessoa,
                default=available_tipo_pessoa,
                key="revenue_tipo"
            )
            
            # Client type filter
            st.sidebar.subheader("👥 Filtro de Cliente")
            client_type_filter = st.sidebar.radio(
                "Tipo de Cliente:",
                ["Todos", "Apenas Cross-Sell", "Apenas Normais"],
                key="revenue_client_type",
                help="Filtra clientes com base na lista de cross-sell."
            )

            # Chart type
            chart_type = st.sidebar.selectbox(
                "Tipo de Gráfico:",
                ["bar", "horizontal_bar", "pie"],
                format_func=lambda x: {
                    "bar": "📊 Gráfico de Barras Vertical",
                    "horizontal_bar": "📈 Gráfico de Barras Horizontal", 
                    "pie": "🥧 Gráfico de Pizza"
                }[x],
                key="revenue_chart_type"
            )
            
            # Filter data
            df_filtered = df_prepared[
                (df_prepared['Month_Year'].isin(selected_months)) &
                (df_prepared['Assessor'].isin(selected_assessores)) &
                (df_prepared['Tipo_Pessoa'].isin(selected_tipo_pessoa)) &
                (df_prepared['Receita_no_Mes'].notna())
            ].copy()
            
            # Load cross-sell clients for filtering
            cross_sell_clients = load_cross_sell_clients() if client_type_filter != "Todos" else None
            
            # Apply client type filter to traditional data
            if client_type_filter != "Todos":
                if cross_sell_clients:
                    # Ensure client codes are strings for comparison
                    df_filtered['Cliente'] = df_filtered['Cliente'].astype(str)
                    
                    if client_type_filter == "Apenas Cross-Sell":
                        df_filtered = df_filtered[df_filtered['Cliente'].isin(cross_sell_clients)]
                    elif client_type_filter == "Apenas Normais":
                        df_filtered = df_filtered[~df_filtered['Cliente'].isin(cross_sell_clients)]

            # Show filter status
            if client_type_filter != "Todos":
                if cross_sell_clients:
                    if client_type_filter == "Apenas Cross-Sell":
                        st.info(f"🎯 Filtro ativo: Mostrando apenas clientes Cross-Sell ({len(cross_sell_clients)} clientes na lista)")
                    elif client_type_filter == "Apenas Normais":
                        st.info(f"👥 Filtro ativo: Mostrando apenas clientes Normais (excluindo {len(cross_sell_clients)} clientes Cross-Sell)")
                else:
                    st.warning("⚠️ Lista de clientes Cross-Sell não encontrada. Filtro não aplicado.")
            
            if df_filtered.empty:
                st.warning("⚠️ Nenhum dado encontrado para os filtros selecionados!")
                return
            
            # Get estruturadas summary with cross-sell filtering
            estruturadas_summary = get_estruturadas_summary(
                df_estruturadas, 
                selected_months, 
                selected_assessores, 
                cross_sell_clients, 
                client_type_filter
            )
            
            # Create summary metrics
            summary_metrics = create_summary_metrics(df_filtered, estruturadas_summary)
            
            # Display metrics
            st.subheader("📊 Resumo da Receita")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Receita Total", f"R$ {summary_metrics['total_receita']:,.2f}")
            with col2:
                st.metric("Receita Tradicional", f"R$ {summary_metrics['total_receita_tradicional']:,.2f}")
            with col3:
                st.metric("Receita Estruturada", f"R$ {summary_metrics['total_comissao_estruturada']:,.2f}")
            with col4:
                structured_pct = (summary_metrics['total_comissao_estruturada'] / summary_metrics['total_receita'] * 100) if summary_metrics['total_receita'] > 0 else 0
                st.metric("% Estruturada", f"{structured_pct:.1f}%")
            
            # Revenue chart
            fig, receita_data = create_receita_by_assessor_chart(df_filtered, estruturadas_summary, chart_type)
            st.plotly_chart(fig, use_container_width=True, key='receita_by_assessor_chart_2')
            
            # Detailed table
            st.subheader("📋 Dados Detalhados de Receita")
            
            # Format data for display
            display_data = receita_data.copy()
            for col in ['Receita_Total', 'Receita_no_Mes', 'Comissao_Estruturada']:
                display_data[f'{col}_Display'] = display_data[col].apply(
                    lambda x: f"R$ {x:,.2f}" if pd.notna(x) else "R$ 0,00"
                )
            
            display_columns = {
                'Assessor': 'Assessor',
                'Receita_Total_Display': 'Receita Total',
                'Receita_no_Mes_Display': 'Receita Tradicional',
                'Comissao_Estruturada_Display': 'Receita Estruturada',
                'Operacoes_Estruturadas': 'Operações Estruturadas'
            }
            
            display_df = display_data.rename(columns=display_columns)
            st.dataframe(display_df[list(display_columns.values())], use_container_width=True, hide_index=True)
            
            # Download option
            csv_data = receita_data.to_csv(sep=';', index=False)
            st.download_button(
                label="📥 Baixar Dados de Receita",
                data=csv_data,
                file_name=f"analise_receita_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            # ============================================================================
    # TAB 3: PORTFOLIO & ROA ANALYSIS
    # ============================================================================
# ============================================================================
# TAB 3: PORTFOLIO & ROA ANALYSIS - TRANSLATED
# ============================================================================
    with tab3:
        st.header("🎯 Análise de Portfólio & ROA")
        
        if not has_data:
            st.warning("⚠️ Nenhum dado disponível. Por favor, faça upload dos dados primeiro.")
            return
        
        # Load and prepare data (similar to revenue analysis)
        with st.spinner("Carregando dados para análise de portfólio..."):
            df = load_data_from_db(db_path)
            df_estruturadas = load_estruturadas_data(db_path)
        
        if df is None or df.empty:
            st.error("❌ Falha ao carregar dados!")
            return
        
        df_adjusted, _ = apply_receita_multiplier(df)
        df_prepared = prepare_data_with_dates(df_adjusted)
        available_months = get_available_months(df_prepared)
        
        # Use latest month by default for portfolio analysis
        if available_months:
            selected_months = [available_months[0]['month_year']]
            selected_assessores = sorted(df_prepared['Assessor'].dropna().unique())
            selected_tipo_pessoa = sorted(df_prepared['Tipo_Pessoa'].dropna().unique())
            
            # Filter data
            df_filtered = df_prepared[
                (df_prepared['Month_Year'].isin(selected_months)) &
                (df_prepared['Assessor'].isin(selected_assessores)) &
                (df_prepared['Tipo_Pessoa'].isin(selected_tipo_pessoa)) &
                (df_prepared['Receita_no_Mes'].notna())
            ]
            
            if not df_filtered.empty:
                # Get estruturadas summary and calculate metrics
                estruturadas_summary = get_estruturadas_summary(df_estruturadas, selected_months, selected_assessores)
                # Get estruturadas summary
                
                metrics_data = calculate_new_metrics(df_filtered, estruturadas_summary)
                
                # ROA Explanation
                with st.expander("📚 Como os ROAs são Calculados"):
                    st.markdown("""
                    ### 📊 **Fórmulas dos ROAs (Return on Assets)**
                    
                    **🔹 ROA Total:**
                    ```
                    ROA Total = (Receita Total ÷ Patrimônio Líquido) × 100
                    ```
                    - **Receita Total** = Receita Tradicional (×0.4) + Comissões Estruturadas (×0.8)
                    - **Patrimônio Líquido** = Net_Em_M (valor atual do patrimônio)
                    
                    **🔹 ROA Estruturadas:**
                    ```
                    ROA Estruturadas = (Comissões Estruturadas ÷ Patrimônio Líquido) × 100
                    ```
                    
                    **🔹 ROA RV:**
                    ```
                    ROA RV = (Receita RV ÷ Patrimônio RV) × 100
                    ```
                    - **Receita RV** = Receita Bovespa (×0.4) + Receita Futuros (×0.4)
                    - **Patrimônio RV** = Net_Renda_Variavel + Net_Fundos_Imobiliarios
                    
                    **🔹 Alocação RV:**
                    ```
                    Alocação RV = (Patrimônio RV ÷ Patrimônio Total) × 100
                    ```
                    - Indica o percentual do patrimônio alocado em renda variável
                    
                    **🔹 Variação PL:**
                    ```
                    Variação PL = Patrimônio Atual - Patrimônio Mês Anterior
                    ```
                    - **Patrimônio Atual** = Net_Em_M
                    - **Patrimônio Mês Anterior** = Net_em_M_1
                    """)
                
                # Display metrics
                st.subheader("📊 Resumo das Métricas de Portfólio")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    avg_roa_total = metrics_data['ROA_Total'].mean()
                    st.metric("ROA Total Médio", f"{avg_roa_total:.2f}%")
                
                with col2:
                    avg_alocacao_rv = metrics_data['Alocacao_RV'].mean()
                    st.metric("Alocação RV Média", f"{avg_alocacao_rv:.2f}%")
                
                with col3:
                    total_patrimonio = metrics_data['Net_Em_M'].sum()
                    st.metric("Patrimônio Total", f"R$ {total_patrimonio:,.2f}")
                
                with col4:
                    total_variacao = metrics_data['Variacao_PL'].sum()
                    st.metric("Variação PL Total", f"R$ {total_variacao:,.2f}")
                
                # ROA Comparison Chart
                st.subheader("📊 Comparação de ROA por Assessor")
                
                # Create ROA comparison chart
                metrics_sorted = metrics_data.sort_values('ROA_Total', ascending=False)
                
                fig_roa = go.Figure()
                
                # ROA Total
                fig_roa.add_trace(go.Bar(
                    name='ROA Total',
                    x=metrics_sorted['Assessor'],
                    y=metrics_sorted['ROA_Total'],
                    marker_color='#1f77b4'
                ))
                
                # ROA Estruturadas
                fig_roa.add_trace(go.Bar(
                    name='ROA Estruturadas',
                    x=metrics_sorted['Assessor'],
                    y=metrics_sorted['ROA_Estruturadas'],
                    marker_color='#ff7f0e'
                ))
                
                # ROA RV
                fig_roa.add_trace(go.Bar(
                    name='ROA Renda Variável',
                    x=metrics_sorted['Assessor'],
                    y=metrics_sorted['ROA_RV'],
                    marker_color='#2ca02c'
                ))
                
                fig_roa.update_layout(
                    title='Comparação de ROA por Assessor (%)',
                    xaxis_title='Assessor',
                    yaxis_title='ROA (%)',
                    
                    barmode='group',
                    xaxis_tickangle=-45,
                    height=600,
                    template='plotly_white',
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
                st.plotly_chart(fig_roa, use_container_width=True, key='roa_fig')
                
                # Portfolio allocation chart
                st.subheader("📊 Análise de Alocação de Portfólio")
                
                fig_alocacao = go.Figure()
                
                # Alocação RV (bar chart)
                fig_alocacao.add_trace(go.Bar(
                    name='Alocação RV (%)',
                    x=metrics_sorted['Assessor'],
                    y=metrics_sorted['Alocacao_RV'],
                    marker_color='#9467bd',
                    yaxis='y',
                    text=[f"{x:.1f}%" for x in metrics_sorted['Alocacao_RV']],
                    textposition='outside'
                ))
                
                # Net Financeiro (line chart on secondary y-axis)
                fig_alocacao.add_trace(go.Scatter(
                    name='Net Financeiro (R$)',
                    x=metrics_sorted['Assessor'],
                    y=metrics_sorted['Net_Financeiro'],
                    mode='lines+markers',
                    marker_color='#d62728',
                    yaxis='y2',
                    line=dict(width=3)
                ))
                
                fig_alocacao.update_layout(
                    title='Alocação em Renda Variável (%) e Net Financeiro por Assessor',
                    xaxis_title='Assessor',
                    xaxis_tickangle=-45,
                    height=600,
                    template='plotly_white',
                    yaxis=dict(
                        title='Alocação RV (%)',
                        side='left'
                    ),
                    yaxis2=dict(
                        title='Net Financeiro (R$)',
                        side='right',
                        overlaying='y',
                        tickformat=',.0f'
                    ),
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
                st.plotly_chart(fig_alocacao, use_container_width=True, key='alocacao_fig')
                
                # Detailed metrics table
                st.subheader("📋 Métricas Detalhadas de Portfólio")
                
                # Format metrics for display
                display_metrics = metrics_data.copy()
                
                # Format monetary values
                for col in ['Net_Em_M', 'Net_RV_Total', 'Net_Financeiro', 'Receita_Total', 'Variacao_PL']:
                    display_metrics[f'{col}_Display'] = display_metrics[col].apply(
                        lambda x: f"R$ {x:,.2f}" if pd.notna(x) else "R$ 0,00"
                    )
                
                # Format percentage values
                for col in ['ROA_Total', 'ROA_Estruturadas', 'ROA_RV', 'Alocacao_RV']:
                    display_metrics[f'{col}_Display'] = display_metrics[col].apply(
                        lambda x: f"{x:.2f}%" if pd.notna(x) else "0.00%"
                    )
                
                # Create display dataframe
                portfolio_display = display_metrics[[
                    'Assessor', 'ROA_Total_Display', 'ROA_Estruturadas_Display', 
                    'ROA_RV_Display', 'Alocacao_RV_Display', 'Net_Em_M_Display',
                    'Net_RV_Total_Display', 'Net_Financeiro_Display', 'Variacao_PL_Display'
                ]].rename(columns={
                    'Assessor': 'Assessor',
                    'ROA_Total_Display': 'ROA Total (%)',
                    'ROA_Estruturadas_Display': 'ROA Estruturadas (%)',
                    'ROA_RV_Display': 'ROA RV (%)',
                    'Alocacao_RV_Display': 'Alocação RV (%)',
                    'Net_Em_M_Display': 'Patrimônio Total',
                    'Net_RV_Total_Display': 'Patrimônio RV',
                    'Net_Financeiro_Display': 'Net Financeiro',
                    'Variacao_PL_Display': 'Variação PL'
                })
                
                st.dataframe(portfolio_display, use_container_width=True, hide_index=True)
                
            else:
                st.warning("⚠️ Nenhum dado disponível para análise de portfólio.")
        else:
            st.warning("⚠️ Nenhum mês disponível para análise.")
    
    # ============================================================================
    # TAB 4: NON-OPERATORS ANALYSIS
    # ============================================================================
# ============================================================================
# TAB 4: NON-OPERATORS ANALYSIS - TRANSLATED
# ============================================================================
    with tab4:
        st.header("🚫 Análise de Clientes que Não Operaram")
        
        if not has_data:
            st.warning("⚠️ Nenhum dado disponível. Por favor, faça upload dos dados primeiro.")
            return
        
        # Load and prepare data
        with st.spinner("Carregando dados para análise de não operadores..."):
            df = load_data_from_db(db_path)
        
        if df is None or df.empty:
            st.error("❌ Falha ao carregar dados!")
            return
        
        df_adjusted, _ = apply_receita_multiplier(df)
        df_prepared = prepare_data_with_dates(df_adjusted)
        available_months = get_available_months(df_prepared)
        
        if available_months:
            # Use latest month by default
            selected_months = [available_months[0]['month_year']]
            selected_assessores = sorted(df_prepared['Assessor'].dropna().unique())
            selected_tipo_pessoa = sorted(df_prepared['Tipo_Pessoa'].dropna().unique())
            
            # Filter data
            df_filtered = df_prepared[
                (df_prepared['Month_Year'].isin(selected_months)) &
                (df_prepared['Assessor'].isin(selected_assessores)) &
                (df_prepared['Tipo_Pessoa'].isin(selected_tipo_pessoa))
            ]
            
            if not df_filtered.empty:
                # Get analysis
                operou_analysis, nao_operou_details = get_operou_bolsa_analysis(df_filtered)
                
                if not operou_analysis.empty:
                    # Summary metrics
                    st.subheader("📊 Resumo de Clientes Não Operadores")
                    
                    total_clients = operou_analysis['Total_Clientes'].sum()
                    total_nao_operou = operou_analysis['Clientes_Nao_Operou'].sum()
                    total_patrimonio_nao_operou = operou_analysis['Patrimonio_Nao_Operou'].sum()
                    overall_percentage = operou_analysis['Percentual_Nao_Operou'].mean()
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Clientes", f"{int(total_clients):,}")
                    with col2:
                        st.metric("Clientes Não Operaram", f"{int(total_nao_operou):,}")
                    with col3:
                        st.metric("% Geral", f"{overall_percentage:.1f}%")
                    with col4:
                        st.metric("Patrimônio Não Operou", f"R$ {total_patrimonio_nao_operou:,.2f}")
                    
                    # Chart
                    st.subheader("📊 Clientes Não Operadores por Assessor")
                    
                    fig_operou = go.Figure()
                    
                    # Bar chart for percentage
                    fig_operou.add_trace(go.Bar(
                        name='% Clientes Não Operaram',
                        x=operou_analysis['Assessor'],
                        y=operou_analysis['Percentual_Nao_Operou'],
                        marker_color='#ff7f0e',
                        yaxis='y',
                        text=[f"{x:.1f}%" for x in operou_analysis['Percentual_Nao_Operou']],
                        textposition='outside'
                    ))
                    
                    # Line chart for absolute numbers
                    fig_operou.add_trace(go.Scatter(
                        name='Qtd Clientes Não Operaram',
                        x=operou_analysis['Assessor'],
                        y=operou_analysis['Clientes_Nao_Operou'],
                        mode='lines+markers',
                        marker_color='#d62728',
                        yaxis='y2',
                        line=dict(width=3)
                    ))
                    
                    fig_operou.update_layout(
                        title='Clientes que Não Operaram na Bolsa por Assessor',
                        xaxis_title='Assessor',
                        xaxis_tickangle=-45,
                        height=600,
                        template='plotly_white',
                        yaxis=dict(
                            title='Percentual (%)',
                            side='left'
                        ),
                        yaxis2=dict(
                            title='Quantidade de Clientes',
                            side='right',
                            overlaying='y'
                        ),
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        )
                    )
                    
                    st.plotly_chart(fig_operou, use_container_width=True, key='operou_chart')
                    
                    # Summary table
                    st.subheader("📋 Resumo por Assessor")
                    
                    # Format display data
                    operou_display = operou_analysis.copy()
                    operou_display['Patrimonio_Display'] = operou_display['Patrimonio_Nao_Operou'].apply(
                        lambda x: f"R$ {x:,.2f}" if pd.notna(x) else "R$ 0,00"
                    )
                    operou_display['Percentual_Display'] = operou_display['Percentual_Nao_Operou'].apply(
                        lambda x: f"{x:.1f}%" if pd.notna(x) else "0.0%"
                    )
                    
                    display_operou = operou_display[[
                        'Assessor', 'Total_Clientes', 'Clientes_Nao_Operou', 
                        'Percentual_Display', 'Patrimonio_Display'
                    ]].rename(columns={
                        'Assessor': 'Assessor',
                        'Total_Clientes': 'Total Clientes',
                        'Clientes_Nao_Operou': 'Clientes Não Operaram',
                        'Percentual_Display': 'Percentual (%)',
                        'Patrimonio_Display': 'Patrimônio Não Operou'
                    })
                    
                    st.dataframe(display_operou, use_container_width=True, hide_index=True)
                    
                    # Detailed client list
                    if not nao_operou_details.empty:
                        st.subheader("📋 Lista Detalhada de Clientes")
                        
                        # Format client details
                        client_display = nao_operou_details[[
                            'Assessor', 'Cliente', 'Net_Em_M', 'Net_Renda_Variavel', 
                            'Net_Financeiro', 'Tipo_Pessoa'
                        ]].copy()
                        
                        # Sort by patrimony
                       
                        client_display = client_display.sort_values('Net_Em_M', ascending=False)
                        
                        # Format monetary values
                        for col in ['Net_Em_M', 'Net_Renda_Variavel', 'Net_Financeiro']:
                            client_display[f'{col}_Display'] = client_display[col].apply(
                                lambda x: f"R$ {x:,.2f}" if pd.notna(x) else "R$ 0,00"
                            )
                        
                        final_client_display = client_display[[
                            'Assessor', 'Cliente', 'Net_Em_M_Display', 
                            'Net_Renda_Variavel_Display', 'Net_Financeiro_Display', 'Tipo_Pessoa'
                        ]].rename(columns={
                            'Assessor': 'Assessor',
                            'Cliente': 'Cliente',
                            'Net_Em_M_Display': 'Patrimônio Total',
                            'Net_Renda_Variavel_Display': 'Patrimônio RV',
                            'Net_Financeiro_Display': 'Net Financeiro',
                            'Tipo_Pessoa': 'Tipo Pessoa'
                        })
                        
                        st.dataframe(final_client_display, use_container_width=True, hide_index=True)
                        
                        # Download option
                        csv_data = nao_operou_details.to_csv(sep=';', index=False)
                        st.download_button(
                            label="📥 Baixar Lista de Não Operadores",
                            data=csv_data,
                            file_name=f"nao_operadores_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                    else:
                        st.success("✅ Todos os clientes operaram no mercado de ações!")
                else:
                    st.info("📊 Nenhum dado de análise de não operadores disponível.")
            else:
                st.warning("⚠️ Nenhum dado disponível para análise.")
        else:
            st.warning("⚠️ Nenhum mês disponível para análise.")
    
    # ============================================================================
    # TAB 5: ACTIVATED CLIENTS ANALYSIS
    # ============================================================================
# ============================================================================
# TAB 5: ACTIVATED CLIENTS ANALYSIS - TRANSLATED
# ============================================================================
    with tab5:
        st.header("✅ Análise de Clientes Ativados")
        
        if not has_data:
            st.warning("⚠️ Nenhum dado disponível. Por favor, faça upload dos dados primeiro.")
            return
        
        # Load and prepare data
        with st.spinner("Carregando dados para análise de clientes ativados..."):
            df = load_data_from_db(db_path)
        
        if df is None or df.empty:
            st.error("❌ Falha ao carregar dados!")
            return
        
        df_adjusted, _ = apply_receita_multiplier(df)
        df_prepared = prepare_data_with_dates(df_adjusted)
        available_months = get_available_months(df_prepared)
        
        if available_months:
            # Use latest month by default
            selected_months = [available_months[0]['month_year']]
            selected_assessores = sorted(df_prepared['Assessor'].dropna().unique())
            selected_tipo_pessoa = sorted(df_prepared['Tipo_Pessoa'].dropna().unique())
            
            # Filter data
            df_filtered = df_prepared[
                (df_prepared['Month_Year'].isin(selected_months)) &
                (df_prepared['Assessor'].isin(selected_assessores)) &
                (df_prepared['Tipo_Pessoa'].isin(selected_tipo_pessoa))
            ]
            
            if not df_filtered.empty:
                # Get activated clients analysis
                ativou_details = get_ativou_analysis(df_filtered)
                
                if not ativou_details.empty:
                    # Summary metrics
                    st.subheader("📊 Resumo de Clientes Ativados")
                    
                    total_ativados = len(ativou_details)
                    patrimonio_total_ativados = ativou_details['Net_Em_M'].sum()
                    patrimonio_medio_ativados = ativou_details['Net_Em_M'].mean()
                    assessores_com_ativacao = ativou_details['Assessor'].nunique()
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Ativados", f"{total_ativados:,}")
                    with col2:
                        st.metric("Patrimônio Total", f"R$ {patrimonio_total_ativados:,.2f}")
                    with col3:
                        st.metric("Patrimônio Médio", f"R$ {patrimonio_medio_ativados:,.2f}")
                    with col4:
                        st.metric("Assessores com Ativações", f"{assessores_com_ativacao:,}")
                    
                    # Chart: Activations by Assessor
                    st.subheader("📊 Ativações por Assessor")
                    
                    ativacao_por_assessor = ativou_details.groupby('Assessor').agg({
                        'Cliente': 'count',
                        'Net_Em_M': ['sum', 'mean']
                    }).reset_index()
                    
                    # Flatten column names
                    ativacao_por_assessor.columns = ['Assessor', 'Qtd_Ativacoes', 'Patrimonio_Total', 'Patrimonio_Medio']
                    ativacao_por_assessor = ativacao_por_assessor.sort_values('Qtd_Ativacoes', ascending=False)
                    
                    # Create chart
                    fig_ativacao = go.Figure()
                    
                    # Bar chart for quantity
                    fig_ativacao.add_trace(go.Bar(
                        name='Quantidade de Ativações',
                        x=ativacao_por_assessor['Assessor'],
                        y=ativacao_por_assessor['Qtd_Ativacoes'],
                        marker_color='#2ca02c',
                        yaxis='y',
                        text=ativacao_por_assessor['Qtd_Ativacoes'],
                        textposition='outside'
                    ))
                    
                    # Line chart for average patrimony
                    fig_ativacao.add_trace(go.Scatter(
                        name='Patrimônio Médio (R$)',
                        x=ativacao_por_assessor['Assessor'],
                        y=ativacao_por_assessor['Patrimonio_Medio'],
                        mode='lines+markers',
                        marker_color='#ff7f0e',
                        yaxis='y2',
                        line=dict(width=3)
                    ))
                    
                    fig_ativacao.update_layout(
                        title='Ativações por Assessor (Quantidade e Patrimônio Médio)',
                        xaxis_title='Assessor',
                        xaxis_tickangle=-45,
                        height=600,
                        template='plotly_white',
                        yaxis=dict(
                            title='Quantidade de Ativações',
                            side='left'
                        ),
                        yaxis2=dict(
                            title='Patrimônio Médio (R$)',
                            side='right',
                            overlaying='y',
                            tickformat=',.0f'
                        ),
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        )
                    )
                    
                    st.plotly_chart(fig_ativacao, use_container_width=True, key='ativacao_chart')
                    
                    # Summary table by assessor
                    st.subheader("📋 Resumo por Assessor")
                    
                    # Format summary data
                    ativacao_display = ativacao_por_assessor.copy()
                    ativacao_display['Patrimonio_Total_Display'] = ativacao_display['Patrimonio_Total'].apply(
                        lambda x: f"R$ {x:,.2f}" if pd.notna(x) else "R$ 0,00"
                    )
                    ativacao_display['Patrimonio_Medio_Display'] = ativacao_display['Patrimonio_Medio'].apply(
                        lambda x: f"R$ {x:,.2f}" if pd.notna(x) else "R$ 0,00"
                    )
                    
                    display_ativacao = ativacao_display[[
                        'Assessor', 'Qtd_Ativacoes', 'Patrimonio_Total_Display', 'Patrimonio_Medio_Display'
                    ]].rename(columns={
                        'Assessor': 'Assessor',
                        'Qtd_Ativacoes': 'Quantidade de Ativações',
                        'Patrimonio_Total_Display': 'Patrimônio Total',
                        'Patrimonio_Medio_Display': 'Patrimônio Médio'
                    })
                    
                    st.dataframe(display_ativacao, use_container_width=True, hide_index=True)
                    
                    # Detailed client list
                    st.subheader("📋 Lista Detalhada de Clientes Ativados")
                    
                    # Format client details
                    ativou_display = ativou_details.copy()
                    
                    # Format monetary values
                    for col in ['Net_Em_M', 'Net_Renda_Variavel', 'Net_Financeiro']:
                        ativou_display[f'{col}_Display'] = ativou_display[col].apply(
                            lambda x: f"R$ {x:,.2f}" if pd.notna(x) else "R$ 0,00"
                        )
                    
                    final_ativou_display = ativou_display[[
                        'Assessor', 'Cliente', 'Net_Em_M_Display', 
                        'Net_Renda_Variavel_Display', 'Net_Financeiro_Display', 
                        'Tipo_Pessoa', 'Data_Posicao'
                    ]].rename(columns={
                        'Assessor': 'Assessor',
                        'Cliente': 'Cliente',
                        'Net_Em_M_Display': 'Patrimônio Total',
                        'Net_Renda_Variavel_Display': 'Patrimônio RV',
                        'Net_Financeiro_Display': 'Net Financeiro',
                        'Tipo_Pessoa': 'Tipo Pessoa',
                        'Data_Posicao': 'Data Posição'
                    })
                    
                    st.dataframe(final_ativou_display, use_container_width=True, hide_index=True)
                    
                    # Distribution by Tipo_Pessoa
                    st.subheader("📊 Distribuição por Tipo Pessoa")
                    
                    tipo_pessoa_dist = ativou_details['Tipo_Pessoa'].value_counts().reset_index()
                    tipo_pessoa_dist.columns = ['Tipo_Pessoa', 'Quantidade']
                    
                    fig_tipo = px.pie(
                        tipo_pessoa_dist,
                        values='Quantidade',
                        names='Tipo_Pessoa',
                        title='Distribuição de Clientes Ativados por Tipo Pessoa'
                    )
                    
                    fig_tipo.update_traces(
                        textposition='inside',
                        textinfo='percent+label',
                        hovertemplate='<b>%{label}</b><br>Quantidade: %{value}<br>Percentual: %{percent}<extra></extra>'
                    )
                    
                    fig_tipo.update_layout(height=400)
                    st.plotly_chart(fig_tipo, use_container_width=True, key='tipo_pessoa_chart')
                    
                    # Download option
                    csv_data = ativou_details.to_csv(sep=';', index=False)
                    st.download_button(
                        label="📥 Baixar Lista de Clientes Ativados",
                        data=csv_data,
                        file_name=f"clientes_ativados_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                    
                else:
                    st.info("📊 Nenhum cliente ativado no período selecionado (Ativou_em_M = Sim).")
            else:
                st.warning("⚠️ Nenhum dado disponível para análise.")
        else:
            st.warning("⚠️ Nenhum mês disponível para análise.")
    
    # ============================================================================
    # TAB 6: UPLOAD FINANCIAL DATA
    # ============================================================================
# ============================================================================
# TAB 6: UPLOAD FINANCIAL DATA - TRANSLATED
# ============================================================================
    with tab6:
        st.header("📤 Upload de Dados Financeiros")
        st.markdown("**Lógica de Data Posição Mais Recente**: Apenas a data mais recente por mês/ano é mantida")
        st.markdown("**Organização dos Dados**: Todos os dados são ordenados por Data Posição (mais novos primeiro)")
        
        # File uploader for financial data
        uploaded_file = st.file_uploader(
            "Escolha um arquivo Excel (Dados Financeiros)", 
            type=['xlsx', 'xls'],
            help="Faça upload do seu arquivo Excel com dados financeiros",
            key="financial_uploader"
        )
        
        if uploaded_file is not None:
            try:
                # Read Excel file
                with st.spinner("Lendo arquivo Excel..."):
                    df = pd.read_excel(uploaded_file)
                
                st.success(f"✅ Arquivo carregado com sucesso! Encontradas {len(df)} linhas")
                
                # Display basic info about the file
                st.subheader("📋 Informações do Arquivo")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total de Linhas", len(df))
                with col2:
                    st.metric("Total de Colunas", len(df.columns))
                with col3:
                    if 'Data Posição' in df.columns:
                        unique_dates = df['Data Posição'].nunique()
                        st.metric("Datas Posição Únicas", unique_dates)
                
                # Show Data Posição summary for uploaded file
                display_data_posicao_summary(df, "Arquivo Carregado - Resumo de Data Posição")
                
                # Show preview of data
                st.subheader("🔍 Visualização dos Dados")
                st.dataframe(df.head(10), use_container_width=True)
                
                # Check for Data Posição column
                if 'Data Posição' not in df.columns:
                    st.error("❌ Coluna 'Data Posição' não encontrada no arquivo!")
                    return
                
                # Connect to database for analysis
                conn = sqlite3.connect(db_path)
                
                # Analyze new data positions
                new_data_analysis, latest_new_by_month = analyze_new_data_positions(df)
                
                # Get existing data positions by month
                existing_by_month, latest_existing_by_month = get_existing_data_posicao_by_month(conn)
                
                # Analysis results
                st.subheader("📅 Análise de Data Posição")
                
                updates_needed = []
                new_months = []
                no_action_needed = []
                
                for month_year, new_latest in latest_new_by_month.items():
                    if month_year in latest_existing_by_month:
                        existing_latest = latest_existing_by_month[month_year]
                        if new_latest['date_obj'] > existing_latest['date_obj']:
                            updates_needed.append({
                                'month_year': month_year,
                                'existing_date': existing_latest['date_str'],
                                'new_date': new_latest['date_str'],
                                'existing_obj': existing_latest['date_obj'],
                                'new_obj': new_latest['date_obj']
                            })
                        else:
                            no_action_needed.append({
                                'month_year': month_year,
                                'existing_date': existing_latest['date_str'],
                                'new_date': new_latest['date_str'],
                                'reason': 'Data existente é mais nova ou igual'
                            })
                    else:
                        new_months.append({
                            'month_year': month_year,
                            'new_date': new_latest['date_str']
                        })
                
                # Display analysis results
                if updates_needed:
                    st.warning(f"🔄 **{len(updates_needed)} mês(es) precisam de atualização** (dados mais novos disponíveis):")
                    for update in updates_needed:
                        month_name = calendar.month_name[int(update['month_year'].split('-')[1])]
                        year = update['month_year'].split('-')[0]
                        st.write(f"- **{month_name} {year}**: {update['existing_date']} → {update['new_date']}")
                
                if new_months:
                    st.info(f"🆕 **{len(new_months)} novo(s) mês(es)** serão adicionados:")
                    for new_month in new_months:
                        month_name = calendar.month_name[int(new_month['month_year'].split('-')[1])]
                        year = new_month['month_year'].split('-')[0]
                        st.write(f"- **{month_name} {year}**: {new_month['new_date']}")
                
                if no_action_needed:
                    st.success(f"✅ **{len(no_action_needed)} mês(es) já estão atualizados:**")
                    for no_action in no_action_needed:
                        month_name = calendar.month_name[int(no_action['month_year'].split('-')[1])]
                        year = no_action['month_year'].split('-')[0]
                        st.write(f"- **{month_name} {year}**: {no_action['existing_date']} (mantendo existente)")
                
                # Process data button
                if updates_needed or new_months:
                    if st.button("💾 Processar Atualizações de Dados", type="primary", key="process_financial"):
                        with st.spinner("Processando atualizações de dados..."):
                            total_deleted = 0
                            total_inserted = 0
                            
                            # Handle updates (delete old, insert new)
                            for update in updates_needed:
                                # Delete old data for this month/year
                                deleted_count, deleted_dates = delete_old_data_for_month_year(
                                    conn, 
                                    update['month_year'], 
                                    update['new_date']
                                )
                                total_deleted += deleted_count
                                
                                if deleted_count > 0:
                                    st.info(f"🗑️ Excluídos {deleted_count} registros antigos para {update['month_year']}")
                            
                            # Prepare data to insert (only latest dates per month)
                            df_to_insert_list = []
                            for month_year in list(set([u['month_year'] for u in updates_needed] + [n['month_year'] for n in new_months])):
                                if month_year in latest_new_by_month:
                                    latest_date_str = latest_new_by_month[month_year]['date_str']
                                    df_month = df[df['Data Posição'].astype(str) == latest_date_str]
                                    df_to_insert_list.append(df_month)
                            
                            if df_to_insert_list:
                                df_to_insert = pd.concat(df_to_insert_list, ignore_index=True)
                                
                                # Clean column names and insert
                                df_cleaned = clean_column_names(df_to_insert)
                                
                                if insert_data_to_db(conn, df_cleaned):
                                    total_inserted = len(df_to_insert)
                                    st.success(f"✅ Dados processados com sucesso!")
                                    st.success(f"📊 Resumo: Excluídos {total_deleted} registros antigos, Inseridos {total_inserted} novos registros")
                                else:
                                    st.error("❌ Falha ao inserir novos dados")
                else:
                    st.info("ℹ️ Nenhuma atualização necessária. Todos os dados já estão atualizados.")
                
                conn.close()
                
            except Exception as e:
                st.error(f"❌ Erro ao processar arquivo: {str(e)}")
                st.exception(e)
    
    # ============================================================================
    # TAB 7: UPLOAD ESTRUTURADAS DATA
    # ============================================================================
# ============================================================================
# TAB 7: UPLOAD ESTRUTURADAS DATA - TRANSLATED
# ============================================================================
    with tab7:
        st.header("🏗️ Upload de Dados Estruturados")
        st.markdown("**Regras de Processamento**:")
        st.markdown("- ✂️ Remove as 3 últimas linhas da planilha")
        st.markdown("- 🔍 Mantém apenas as linhas onde 'Status da Operação' = 'Totalmente executado'")
        st.markdown("- ✏️ Remove o primeiro caractere da coluna 'Cod A'")
        st.markdown("- 🔍 **Verifica duplicatas exatas** em relação aos registros existentes no banco de dados")
        
        # File uploader for estruturadas data
        uploaded_estruturadas_file = st.file_uploader(
            "Escolha um arquivo Excel (Dados Estruturados)", 
            type=['xlsx', 'xls'],
            help="Faça upload do seu arquivo Excel com dados de estruturadas",
            key="estruturadas_uploader"
        )
        
        if uploaded_estruturadas_file is not None:
            try:
                # Read Excel file
                with st.spinner("Lendo arquivo Excel de Estruturadas..."):
                    df_estruturadas = pd.read_excel(uploaded_estruturadas_file)
                
                st.success(f"✅ Arquivo de Estruturadas carregado com sucesso! Encontradas {len(df_estruturadas)} linhas")
                
                # Display expected columns
                expected_columns = ['Cliente', 'Data', 'Origem', 'Ativo', 'Estratégia', 'Comissão', 'Cod Matriz', 'Cod A', 'Status da Operação']
                st.subheader("📋 Colunas Esperadas")
                st.write("; ".join(expected_columns))
                
                # Display actual columns
                st.subheader("📋 Colunas Atuais no Arquivo")
                st.write("; ".join(df_estruturadas.columns.tolist()))
                
                # Check if all expected columns are present
                missing_columns = [col for col in expected_columns if col not in df_estruturadas.columns]
                if missing_columns:
                    st.error(f"❌ Colunas ausentes: {'; '.join(missing_columns)}")
                else:
                    st.success("✅ Todas as colunas esperadas encontradas!")
                
                # Show preview of original data
                st.subheader("🔍 Visualização dos Dados Originais")
                st.dataframe(df_estruturadas.head(10), use_container_width=True)
                
                # Process the data
                df_processed = process_estruturadas_data(df_estruturadas)
                
                # Show preview of processed data
                st.subheader("🔍 Visualização dos Dados Processados")
                st.dataframe(df_processed.head(10), use_container_width=True)
                
                # Display processing summary
                st.subheader("📊 Resumo do Processamento")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Linhas Originais", len(df_estruturadas))
                with col2:
                    st.metric("Linhas Processadas", len(df_processed))
                with col3:
                    st.metric("Linhas Removidas", len(df_estruturadas) - len(df_processed))
                
                # Check for duplicates AFTER processing and column cleaning
                if len(df_processed) > 0:
                    with st.spinner("Verificando duplicatas em relação ao banco de dados existente..."):
                        # Clean column names FIRST, then check for duplicates
                        conn = sqlite3.connect(db_path)
                        df_cleaned = clean_estruturadas_column_names(df_processed)
                        duplicates_found, df_unique, duplicate_details = check_for_duplicates(df_cleaned, conn)
                        conn.close()
                    
                    # Display duplicate analysis
                    display_duplicate_analysis(duplicates_found, duplicate_details, df_cleaned, df_unique)
                    
                    # Option to replace all data or add unique data
                    st.subheader("💾 Operações do Banco de Dados")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if st.button("🔄 Substituir Todos os Dados Estruturados", type="primary", key="replace_estruturadas"):
                            with st.spinner("Substituindo dados estruturados..."):
                                conn = sqlite3.connect(db_path)
                                # Clear existing data
                                deleted_count = clear_estruturadas_table(conn)
                                st.info(f"🗑️ Limpos {deleted_count} registros existentes")
                                
                                # Insert all processed data (df_cleaned already has clean column names)
                                if insert_estruturadas_to_db(conn, df_cleaned):
                                    st.success(f"✅ Inseridos {len(df_cleaned)} novos registros com sucesso!")
                                else:
                                    st.error("❌ Falha ao inserir novos dados")
                                conn.close()
                    
                    with col2:
                        # Only show add button if there are unique rows to add
                        if len(df_unique) > 0:
                            if st.button("➕ Adicionar Apenas Dados Únicos", type="secondary", key="add_unique_estruturadas"):
                                with st.spinner("Adicionando dados estruturados únicos..."):
                                    conn = sqlite3.connect(db_path)
                                    if insert_estruturadas_to_db(conn, df_unique):
                                        st.success(f"✅ Adicionados {len(df_unique)} registros únicos com sucesso!")
                                        if duplicates_found:
                                            st.info(f"ℹ️ Pulados {len(duplicate_details)} registros duplicados")
                                    else:
                                        st.error("❌ Falha ao adicionar novos dados")
                                    conn.close()
                        else:
                            st.button("➕ Adicionar Apenas Dados Únicos", disabled=True, help="Nenhum dado único para adicionar - todas as linhas são duplicatas")
                
                else:
                    st.warning("⚠️ Nenhum dado para processar após a filtragem!")
                
            except Exception as e:
                st.error(f"❌ Erro ao processar arquivo de estruturadas: {str(e)}")
                st.exception(e)


    # ============================================================================
    # TAB 8: DATABASE MANAGER
    # ============================================================================
# ============================================================================
# TAB 8: DATABASE MANAGER - TRANSLATED
# ============================================================================
    with tab8:
        st.header("🗄️ Gerenciador de Banco de Dados")
        st.markdown("Visualize e gerencie o conteúdo do seu banco de dados")
        
        # Database info
        st.subheader("📊 Informações do Banco de Dados")
        
        if os.path.exists(db_path):
            file_size = os.path.getsize(db_path)
            st.info(f"Tamanho do banco de dados: {file_size:,} bytes")
            
            # Quick stats from database
            try:
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                
                # Financial data stats
                cursor.execute("SELECT COUNT(*) FROM financial_data")
                total_financial_records = cursor.fetchone()[0]
                
                # Estruturadas data stats
                cursor.execute("SELECT COUNT(*) FROM estruturadas")
                total_estruturadas_records = cursor.fetchone()[0]
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Registros Financeiros", total_financial_records)
                with col2:
                    st.metric("Registros Estruturados", total_estruturadas_records)
                with col3:
                    st.metric("Total de Registros", total_financial_records + total_estruturadas_records)
                
                # Show unique months in financial database (sorted)
                if total_financial_records > 0:
                    cursor.execute("""
                        SELECT DISTINCT Data_Posicao FROM financial_data 
                        ORDER BY 
                            CASE 
                                WHEN Data_Posicao GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]*' 
                                THEN Data_Posicao 
                                ELSE '9999-12-31' 
                            END DESC
                    """)
                    dates_in_db = cursor.fetchall()
                    if dates_in_db:
                        st.subheader("📅 Períodos de Dados Financeiros Disponíveis")
                        for (date_str,) in dates_in_db[:10]:  # Show last 10
                            date_obj = parse_date(date_str)
                            if date_obj:
                                month_name = calendar.month_name[date_obj.month]
                                st.text(f"{month_name} {date_obj.year}: {date_str}")
                        
                        if len(dates_in_db) > 10:
                            st.text(f"... e mais {len(dates_in_db) - 10} períodos")
                
                conn.close()
                
            except Exception as e:
                st.error(f"Erro no banco de dados: {str(e)}")
        else:
            st.warning("Banco de dados não encontrado")
        
        # Load and display database content
        st.subheader("📋 Visualizador de Conteúdo do Banco de Dados")
        
        # Tabs for different tables
        db_tab1, db_tab2 = st.tabs(["📈 Dados Financeiros", "🏗️ Dados Estruturados"])
        
        with db_tab1:
            if st.button("📊 Carregar Conteúdo do Banco de Dados Financeiro", type="secondary", key="load_financial_db"):
                with st.spinner("Carregando dados financeiros do banco de dados..."):
                    df_from_db = load_data_from_db(db_path)
                    
                    if df_from_db is not None and not df_from_db.empty:
                        st.success(f"✅ Carregados {len(df_from_db)} registros financeiros")
                        
                        # Show summary statistics
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Total de Registros", len(df_from_db))
                        
                        with col2:
                            if 'Data_Posicao' in df_from_db.columns:
                                unique_db_dates = df_from_db['Data_Posicao'].nunique()
                                st.metric("Datas Posição Únicas", unique_db_dates)
                        
                        with col3:
                            if 'Assessor' in df_from_db.columns:
                                unique_assessors = df_from_db['Assessor'].nunique()
                                st.metric("Assessores Únicos", unique_assessors)
                        
                        # Display data
                        st.subheader("📋 Registros do Banco de Dados Financeiro")
                        st.dataframe(df_from_db, use_container_width=True)
                        
                        # Option to download as CSV
                        csv_data = df_from_db.to_csv(sep=';', index=False)
                        st.download_button(
                            label="📥 Baixar Banco de Dados Financeiro como CSV",
                            data=csv_data,
                            file_name=f"exportacao_dados_financeiros_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            key="download_financial_db"
                        )
                    else:
                        st.info("Nenhum dado financeiro encontrado no banco de dados.")
        
        with db_tab2:
            if st.button("📊 Carregar Conteúdo do Banco de Dados Estruturados", type="secondary", key="load_estruturadas_db"):
                with st.spinner("Carregando dados estruturados do banco de dados..."):
                    conn = sqlite3.connect(db_path)
                    df_estruturadas_from_db = load_estruturadas_from_db(conn)
                    conn.close()
                    
                    if not df_estruturadas_from_db.empty:
                        st.success(f"✅ Carregados {len(df_estruturadas_from_db)} registros estruturados")
                        
                        # Show summary statistics
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Total de Registros", len(df_estruturadas_from_db))
                        
                        with col2:
                            if 'Cliente' in df_estruturadas_from_db.columns:
                                unique_clients = df_estruturadas_from_db['Cliente'].nunique()
                                st.metric("Clientes Únicos", unique_clients)
                        
                        with col3:
                            if 'Data' in df_estruturadas_from_db.columns:
                                unique_dates = df_estruturadas_from_db['Data'].nunique()
                                st.metric("Datas Únicas", unique_dates)
                        
                        # Display data
                        st.subheader("📋 Registros do Banco de Dados Estruturados")
                        st.dataframe(df_estruturadas_from_db, use_container_width=True)
                        
                        # Option to download as CSV
                        csv_data = df_estruturadas_from_db.to_csv(sep=';', index=False)
                        st.download_button(
                            label="📥 Baixar Banco de Dados Estruturados como CSV",
                            data=csv_data,
                            file_name=f"exportacao_dados_estruturados_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            key="download_estruturadas_db"
                        )
                    else:
                        st.info("Nenhum dado estruturado encontrado no banco de dados.")
        
        # Database maintenance
        st.subheader("🔧 Manutenção do Banco de Dados")
        st.warning("⚠️ **Atenção**: Estas operações não podem ser desfeitas!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🗑️ Limpar Dados Financeiros", type="secondary", key="clear_financial"):
                if st.checkbox("Eu entendo que isso irá apagar todos os dados financeiros", key="confirm_clear_financial"):
                    try:
                        conn = sqlite3.connect(db_path)
                        cursor = conn.cursor()
                        cursor.execute("DELETE FROM financial_data")
                        deleted_count = cursor.rowcount
                        conn.commit()
                        conn.close()
                        st.success(f"✅ Apagados {deleted_count} registros financeiros")
                    except Exception as e:
                        st.error(f"❌ Erro ao limpar dados financeiros: {str(e)}")
        
        with col2:
            if st.button("🗑️ Limpar Dados Estruturados", type="secondary", key="clear_estruturadas"):
                if st.checkbox("Eu entendo que isso irá apagar todos os dados estruturados", key="confirm_clear_estruturadas"):
                    try:
                        conn = sqlite3.connect(db_path)
                        deleted_count = clear_estruturadas_table(conn)
                        conn.close()
                        st.success(f"✅ Apagados {deleted_count} registros estruturados")
                    except Exception as e:
                        st.error(f"❌ Erro ao limpar dados estruturados: {str(e)}")
    
    # ============================================================================
    # TAB 9: HELP & INFO
    # ============================================================================
# ============================================================================
# TAB 9: HELP & INFO - TRANSLATED
# ============================================================================
    with tab9:
        st.header("ℹ️ Ajuda e Informações")
        
        # Application overview
        st.subheader("📊 Visão Geral da Aplicação")
        st.markdown("""
        Este completo Gerenciador e Dashboard de Dados Financeiros oferece:
        
        **🔹 Gerenciamento de Dados:**
        - Faça upload e processe dados financeiros de arquivos Excel
        - Faça upload e processe dados de produtos estruturados (estruturadas)
        - Detecção automática de duplicatas e validação de dados
        - Atualizações inteligentes de dados baseadas em data (mantém apenas o mais recente por mês)
        
        **🔹 Análise e Visualização:**
        - Análise de receita com produtos tradicionais e estruturados
        - Análise de portfólio e ROA (Retorno sobre Ativos)
        - Análise de comportamento do cliente (não operadores, clientes ativados)
        - Gráficos interativos e tabelas detalhadas
        
        **🔹 Gerenciamento de Banco de Dados:**
        - Banco de dados SQLite para armazenamento confiável
        - Capacidades de exportação de dados
        - Ferramentas de manutenção do banco de dados
        """)
        
        # Data processing rules
        st.subheader("📋 Regras de Processamento de Dados")
        
        with st.expander("📈 Processamento de Dados Financeiros"):
            st.markdown("""
            **Mapeamento de Colunas:**
            - Nomes de colunas em Português são automaticamente mapeados para o esquema do banco de dados em Inglês
            - Exemplo: "Data Posição" → "Data_Posicao"
            
            **Lógica de Data:**
            - Apenas a "Data Posição" mais recente por mês/ano é mantida
            - Dados mais antigos para o mesmo mês são automaticamente substituídos
            
            **Ajustes de Receita:**
            - Todas as colunas "Receita" são multiplicadas por 0.4 (0.5 × 0.8)
            - Isso representa a estrutura de comissão ajustada
            """)
        
        with st.expander("🏗️ Processamento de Dados Estruturados"):
            st.markdown("""
            **Etapas de Processamento:**
            1. Remove as 3 últimas linhas da planilha carregada
            2. Mantém apenas as linhas onde "Status da Operação" = "Totalmente executado"
            3. Remove o primeiro caractere da coluna "Cod A"
            4. Aplica o multiplicador de 0.8 aos valores de comissão
            
            **Detecção de Duplicatas:**
            - Verificação abrangente de duplicatas em relação ao banco de dados existente
            - Usa chaves compostas para correspondência precisa
            - Opção de adicionar apenas registros únicos ou substituir todos os dados
            """)
        
        # ROA calculations
        st.subheader("🎯 Cálculos de ROA")
        
        with st.expander("📊 Fórmulas de ROA Explicadas"):
            st.markdown("""
            **ROA Total:**
            ```
            ROA Total = (Receita Total ÷ Patrimônio Líquido) × 100
            ```
            - Receita Total = Receita Tradicional (×0.4) + Comissões Estruturadas (×0.8)
            
            **ROA Estruturadas:**
            ```
            ROA Estruturadas = (Comissões Estruturadas ÷ Patrimônio Líquido) × 100
            ```
            
            **ROA Renda Variável:**
            ```
            ROA RV = (Receita RV ÷ Patrimônio RV) × 100
            ```
            - Receita RV = Receita Bovespa + Receita Futuros
            - Patrimônio RV = Renda Variável + Fundos Imobiliários
            
            **Alocação RV:**
            ```
            Alocação RV = (Patrimônio RV ÷ Patrimônio Total) × 100
            ```
            
            **Variação PL:**
            ```
            Variação PL = Patrimônio Atual - Patrimônio Mês Anterior
            ```
            """)
        
        # Usage tips
        st.subheader("💡 Dicas de Uso")
        
        with st.expander("🚀 Primeiros Passos"):
            st.markdown("""
            **Configuração Inicial:**
            1. Comece pela aba "Upload Dados Financeiros"
            2. Faça upload do seu arquivo Excel principal de dados financeiros
            3. Revise a análise de dados e processe as atualizações
            4. Opcionalmente, faça upload dos dados de estruturadas
            5. Use as abas do dashboard para análise
            
            **Atualizações Regulares:**
            1. Faça upload de novos arquivos de dados mensais
            2. O sistema detectará automaticamente o que precisa ser atualizado
            3. Apenas dados mais novos substituirão registros existentes
            4. Use o dashboard para análise contínua
            """)
        
        with st.expander("📊 Navegação no Dashboard"):
            st.markdown("""
            **Aba Dashboard:** Visão geral rápida com métricas chave e melhores desempenhos
            
            **Aba Análise de Receita:** Detalhamento da receita com filtros flexíveis
            
            **Aba Portfólio & ROA:** Análise avançada de portfólio e cálculos de ROA
            
            **Aba Não Operaram:** Análise de clientes que não negociaram ações
            
            **Aba Clientes Ativados:** Análise de clientes recém-ativados
            
            **Abas de Upload:** Gerenciamento de dados e processamento de arquivos
            
            **Aba Gerenciador de Banco de Dados:** Visualize, exporte e mantenha o conteúdo do banco de dados
            """)
        
        with st.expander("🔍 Filtragem e Análise"):
            st.markdown("""
            **Seleção de Período:**
            - Mês Único: Analisa um mês específico
            - Múltiplos Meses: Compara vários meses
            - Todos os Meses: Análise abrangente de todos os dados disponíveis
            
            **Filtragem por Assessor:**
            - Selecione assessores específicos para análise focada
            - Use "Todos os Assessores" para uma visão geral completa
            
            **Tipos de Gráfico:**
            - Barras Verticais: Visualização padrão de comparação
            - Barras Horizontais: Melhor para muitos assessores
            - Gráfico de Pizza: Visualização de distribuição percentual
            """)
        
        # Troubleshooting
        st.subheader("🔧 Solução de Problemas")
        
        with st.expander("❌ Problemas Comuns"):
            st.markdown("""
            **Problemas de Upload de Arquivo:**
            - Certifique-se de que os arquivos Excel tenham os nomes de coluna esperados
            - Verifique se a coluna "Data Posição" existe nos dados financeiros
            - Verifique se os arquivos de estruturadas possuem todas as colunas necessárias
            
            **Problemas de Processamento de Dados:**
            - Verifique se há caracteres especiais nos dados
            - Certifique-se de que os formatos de data sejam consistentes
            - Verifique se as colunas numéricas não contêm texto
            
            **Problemas do Dashboard:**
            - Se nenhum dado aparecer, verifique se os dados foram enviados com sucesso
            - Tente recarregar a página se os gráficos não carregarem
            - Verifique as seleções de filtro se os resultados parecerem vazios
            
            **Problemas de Desempenho:**
            - Grandes conjuntos de dados podem levar tempo para serem processados
            - Considere filtrar os dados para melhor desempenho
            - Use a análise de mês único para resultados mais rápidos
            """)
        
        # Technical information
        st.subheader("🔧 Informações Técnicas")
        
        with st.expander("💻 Requisitos do Sistema"):
            st.markdown("""
            **Dependências:**
            - Streamlit (interface web)
            - Pandas (processamento de dados)
            - SQLite3 (banco de dados)
            - Plotly (gráficos interativos)
            - OpenPyXL (leitura de arquivos Excel)
            
            **Banco de Dados:**
            - Arquivo de banco de dados SQLite: `financial_data.db`
            - Duas tabelas principais: `financial_data` e `estruturadas`
            - Criação automática de tabelas na primeira execução
            
            **Formatos de Arquivo:**
            - Suportados: .xlsx, .xls
            - Formato de exportação: CSV com separador de ponto e vírgula
            """)
        
        # Contact and support
        st.subheader("📞 Suporte")
        st.markdown("""
        **Para suporte técnico ou dúvidas:**
        - Verifique esta seção de ajuda primeiro
        - Revise as mensagens de erro para orientação específica
        - Certifique-se de que seus arquivos de dados correspondam ao formato esperado
        - Tente a aba Gerenciador de Banco de Dados para verificar a integridade dos dados
        """)
        
        # Version information
        st.subheader("📋 Informações da Versão")
        st.info("Gerenciador e Dashboard de Dados Financeiros v2.0 - Sistema Integrado de Upload e Análise")

if __name__ == "__main__":
    main()