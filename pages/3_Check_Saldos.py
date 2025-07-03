import streamlit as st
import pandas as pd
import io
import re # We'll use regex for a more robust cleaning if needed, but not strictly necessary for simple replacements

def main():
    st.title("📊 Financial Dashboard - Client Analysis")
    st.markdown("---")
    
    # File upload section
    st.header("📁 Upload Excel File")
    uploaded_file = st.file_uploader(
        "Choose your Excel file", 
        type=['xlsx', 'xls'],
        help="Upload the Excel file with client data"
    )
    
    if uploaded_file is not None:
        try:
            # Read the Excel file
            df = load_and_process_data(uploaded_file)
            
            if df is not None:
                st.success(f"✅ File loaded successfully! Found {len(df)} records.")
                
                # Display the analysis
                display_analysis(df)
                
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
            st.exception(e) # Show full traceback for debugging
    else:
        st.info("👆 Please upload an Excel file to begin analysis")

def convert_brazilian_number(value):
    """
    Robustly converts Brazilian number format string (e.g., '1.234.567,89') to a float.
    Handles thousands separators (dots) and decimal commas.
    Returns 0.0 for non-numeric or invalid inputs.
    """
    if pd.isna(value) or str(value).strip() == '':
        return 0.0
    
    str_value = str(value).strip()
    
    # Replace dots (thousands separators) with nothing
    # Replace comma (decimal separator) with a dot
    # Then attempt to convert to float
    try:
        # Check if the string contains a comma. If it does, it's the decimal separator.
        # If it doesn't, it might be an integer, or uses dots as decimal (unlikely for BR)
        if ',' in str_value:
            # First remove all thousands separators (dots)
            cleaned_value = str_value.replace('.', '')
            # Then replace the decimal comma with a dot
            cleaned_value = cleaned_value.replace(',', '.')
        else:
            # If no comma, assume it's either an integer or uses dot as decimal (e.g. 1234.56 or just 1234)
            # Given the problem context, if no comma, it's likely a whole number or already in standard float format
            # We'll just try to convert directly.
            cleaned_value = str_value
        
        return float(cleaned_value)
    except ValueError:
        # If conversion fails, return 0.0
        return 0.0

def load_and_process_data(uploaded_file):
    """
    Load and process the Excel file with proper Brazilian number conversion
    """
    try:
        # Read Excel file
        df = pd.read_excel(uploaded_file)
        
        # Display basic info about the file
        st.subheader("📋 File Overview")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", len(df))
        with col2:
            st.metric("Total Assessors", df['Assessor'].nunique())
        with col3:
            st.metric("Total Clients", df['Cliente'].nunique())
        
        # Show a sample of original data for verification
        st.subheader("🔍 Sample Data (Original Format)")
        # Display the full original column to see the raw input strings
        st.dataframe(df[['Conta', 'Cliente', 'D0', 'Total']].head(5), use_container_width=True)
        
        # Clean numeric columns using proper Brazilian format conversion
        numeric_columns = ['D0', 'D+1', 'D+2', 'D+3', 'Total']
        
        for col in numeric_columns:
            if col in df.columns:
                # Apply Brazilian number conversion
                # It's good practice to ensure the column is treated as strings before applying string methods
                df[col] = df[col].astype(str).apply(convert_brazilian_number)
            else:
                st.warning(f"Column '{col}' not found in the uploaded file.")
                
        # Show converted data for verification
        st.subheader("✅ Sample Data (After Conversion)")
        sample_converted = df[['Conta', 'Cliente', 'D0', 'Total']].head(5)
        
        # Format the numbers for display in the verification table
        for col in ['D0', 'Total']:
            if col in sample_converted.columns:
                sample_converted[col] = sample_converted[col].apply(lambda x: f"R$ {x:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.'))

        st.dataframe(sample_converted, use_container_width=True)
        
        # Show some statistics to verify conversion
        # st.subheader("📊 Conversion Verification (Calculated on converted floats)")
        # col1, col2, col3, col4 = st.columns(4)
        # with col1:
        #     st.metric("Max D0", f"R$ {df['D0'].max():,.2f}")
        # with col2:
        #     st.metric("Max Total", f"R$ {df['Total'].max():,.2f}")
        # with col3:
        #     st.metric("Min D0", f"R$ {df['D0'].min():,.2f}")
        # with col4:
        #     st.metric("Min Total", f"R$ {df['Total'].min():,.2f}")
        
        return df
        
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        st.exception(e) # Show full traceback for debugging
        return None

def display_analysis(df):
    """
    Display the main analysis: clients where Total == D0 and Total != 0
    """
    st.header("🎯 Analysis: Clients where Total = D0 (excluding zeros)")
    
    # Filter clients where Total == D0 AND Total != 0
    # Use a small tolerance for floating point comparison to account for precision issues
    tolerance = 1e-6 # A very small number like 0.000001
    filtered_df = df[
        (abs(df['Total'] - df['D0']) < tolerance) & 
        (abs(df['Total']) > tolerance) # Ensure total is not zero (within tolerance)
    ].copy()
    
    if len(filtered_df) == 0:
        st.warning("⚠️ No clients found where Total equals D0 (excluding zeros)")
        
        # Debug information to help understand why no clients were found
        st.subheader("🔧 Debug Information")
        total_non_zero = df[abs(df['Total']) > tolerance]
        st.write(f"Total records after initial non-zero filter: {len(total_non_zero)}")
        
        # Check how many are 'close' based on D0 and Total
        close_values = df[abs(df['Total'] - df['D0']) < tolerance]
        st.write(f"Total records where Total is approximately equal to D0: {len(close_values)}")
        
        # Show actual values for some problematic rows if they exist
        st.write("First 5 rows where Total is close to D0 but might be zero or other issues:")
        st.dataframe(df[['Conta', 'Cliente', 'D0', 'Total']].head(5))
        
        return
    
    st.success(f"Found {len(filtered_df)} clients where Total = D0 (excluding zeros)")
    
    # Get unique assessors and create multiselect
    assessors = sorted(filtered_df['Assessor'].unique())
    
    st.subheader("👥 Select Assessors")
    selected_assessors = st.multiselect(
        "Choose one or more assessors to analyze:",
        options=assessors,
        default=assessors,
        help="You can select multiple assessors to generate reports for each one"
    )
    
    if not selected_assessors:
        st.warning("⚠️ Please select at least one assessor")
        return
    
    # Display data for each selected assessor
    for assessor in selected_assessors:
        st.markdown("---")
        display_assessor_data(filtered_df, assessor)

def sort_data_for_whatsapp(data):
    """
    Sort data for WhatsApp message:
    1. Negative values first (most negative to least negative)
    2. Then positive values (largest to smallest)
    """
    # Separate negative and positive values
    negative_data = data[data['Total'] < 0].copy()
    positive_data = data[data['Total'] >= 0].copy() # Use >= to include 0 if it somehow slips through (though we filtered it out)
    
    # Sort negative values: most negative first (ascending order for negatives, e.g., -100, -50, -10)
    negative_data = negative_data.sort_values('Total', ascending=True)
    
    # Sort positive values: largest first (descending order, e.g., 1000, 500, 100)
    positive_data = positive_data.sort_values('Total', ascending=False)
    
    # Combine: negatives first, then positives
    sorted_data = pd.concat([negative_data, positive_data], ignore_index=True)
    
    return sorted_data

def display_assessor_data(df, assessor):
    """
    Display data for a specific assessor with copy-friendly format
    """
    assessor_data = df[df['Assessor'] == assessor].copy()
    
    if len(assessor_data) == 0:
        st.info(f"No data found for Assessor {assessor}")
        return
    
    st.subheader(f"👤 Assessor: {assessor}")
    
    # Show summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Clients", len(assessor_data))
    with col2:
        negative_count = len(assessor_data[assessor_data['Total'] < 0])
        st.metric("Negative Values", negative_count)
    with col3:
        positive_count = len(assessor_data[assessor_data['Total'] > 0])
        st.metric("Positive Values", positive_count)
    
    # Display table (sorted by Total descending for easy viewing)
    display_columns = ['Conta', 'Cliente', 'Assessor', 'D0', 'Total']
    table_data = assessor_data[display_columns].sort_values('Total', ascending=False)
    
    # Format numbers for display in the Streamlit table
    table_display = table_data.copy()
    # Apply Brazilian currency formatting
    table_display['D0'] = table_display['D0'].apply(lambda x: f"R$ {x:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.'))
    table_display['Total'] = table_display['Total'].apply(lambda x: f"R$ {x:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.'))
    
    st.dataframe(
        table_display,
        use_container_width=True,
        hide_index=True
    )
    
    # Generate WhatsApp message with custom sorting
    st.subheader("📱 WhatsApp Message")
    whatsapp_message = generate_whatsapp_message(assessor_data, assessor)
    
    # Display in a text area for easy copying
    st.text_area(
        f"Copy message for Assessor {assessor}:",
        whatsapp_message,
        height=300,
        key=f"whatsapp_{assessor}",
        help="Click in the text area and press Ctrl+A to select all, then Ctrl+C to copy"
    )
    
    # Download button for Excel
    excel_buffer = create_excel_download(table_data) # Use the numeric table_data for Excel download
    st.download_button(
        label=f"📥 Download Excel - Assessor {assessor}",
        data=excel_buffer,
        file_name=f"assessor_{assessor}_clients.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key=f"download_{assessor}"
    )

def generate_whatsapp_message(data, assessor):
    """
    Generate a formatted message for WhatsApp with custom sorting
    """
    # Sort data according to requirements
    sorted_data = sort_data_for_whatsapp(data)
    
    message = f"🏦 *RELATÓRIO - ASSESSOR {assessor}*\n"
    message += f"📊 Total de clientes: {len(data)}\n"
    
    # Add summary of negative and positive values
    negative_count = len(data[data['Total'] < 0])
    positive_count = len(data[data['Total'] > 0])
    
    if negative_count > 0:
        message += f"⚠️ Valores negativos: {negative_count}\n"
    if positive_count > 0:
        message += f"✅ Valores positivos: {positive_count}\n"
    
    message += "=" * 40 + "\n\n"
    
    # Add section headers
    current_section = None
    
    for _, row in sorted_data.iterrows():
        # Format the Total value as Brazilian currency for the message
        total_formatted = f"R$ {row['Total']:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.')
        
        # Add section header if needed
        # This logic checks if we're transitioning from negative to positive or vice-versa
        is_negative = row['Total'] < 0
        if current_section is None or current_section != is_negative:
            current_section = is_negative
            if current_section:
                message += "🔴 *VALORES NEGATIVOS (PRIORIDADE)*\n"
                message += "-" * 35 + "\n"
            else:
                message += "\n🟢 *VALORES POSITIVOS*\n"
                message += "-" * 25 + "\n"
        
        # Add client information
        message += f"👤 *{row['Cliente']}*\n"
        message += f"📋 Conta: {row['Conta']}\n"
        message += f"💰 Valor: {total_formatted}\n"
        
        # Add priority indicator for negative values
        if row['Total'] < 0:
            message += "⚠️ *ATENÇÃO: VALOR NEGATIVO*\n"
        
        message += "-" * 30 + "\n"
    
    message += f"\n📅 Relatório gerado em: {pd.Timestamp.now().strftime('%d/%m/%Y às %H:%M')}"
    
    return message

def create_excel_download(data):
    """
    Create an Excel file in memory for download
    """
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        data.to_excel(writer, index=False, sheet_name='Clients')
    
    return output.getvalue()

# Run the app
if __name__ == "__main__":
    # Configure Streamlit page
    st.set_page_config(
        page_title="Financial Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    main()