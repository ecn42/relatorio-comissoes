import streamlit as st
import pandas as pd
import io
import re # We'll use regex for a more robust cleaning if needed, but not strictly necessary for simple replacements
import streamlit.components.v1 as components

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
    
    How it works:
    1. First checks if the value is empty or NaN
    2. Converts to string and removes whitespace
    3. If there's a comma, treats it as decimal separator and removes dots (thousands separators)
    4. If no comma, assumes it's already in standard format or an integer
    5. Returns 0.0 if conversion fails
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
    
    This function:
    1. Reads the Excel file into a pandas DataFrame
    2. Shows basic statistics about the file
    3. Converts Brazilian number format to standard float format
    4. Displays before/after samples for verification
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
        
        # # Show a sample of original data for verification
        # st.subheader("🔍 Sample Data (Original Format)")
        # # Display the full original column to see the raw input strings
        # st.dataframe(df[['Conta', 'Cliente', 'D0', 'Total']].head(5), use_container_width=True)
        
        # Clean numeric columns using proper Brazilian format conversion
        numeric_columns = ['D0', 'D+1', 'D+2', 'D+3', 'Total']
        
        for col in numeric_columns:
            if col in df.columns:
                # Apply Brazilian number conversion
                # It's good practice to ensure the column is treated as strings before applying string methods
                df[col] = df[col].astype(str).apply(convert_brazilian_number)
            else:
                st.warning(f"Column '{col}' not found in the uploaded file.")
                
        # # Show converted data for verification
        # st.subheader("✅ Sample Data (After Conversion)")
        # sample_converted = df[['Conta', 'Cliente', 'D0', 'D+1', 'D+2', 'D+3',  'Total']].head(5)
        
        # # Format the numbers for display in the verification table
        # for col in ['D0','D+1', 'D+2', 'D+3', 'Total']:
        #     if col in sample_converted.columns:
        #         sample_converted[col] = sample_converted[col].apply(lambda x: f"R$ {x:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.'))

        # st.dataframe(sample_converted, use_container_width=True)
        
        return df
        
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        st.exception(e) # Show full traceback for debugging
        return None

def create_copy_button(text_to_copy, button_text, button_key):
    """
    Creates a copy-to-clipboard button using JavaScript
    
    This function:
    1. Creates a unique HTML element ID for each button
    2. Embeds JavaScript code that copies text to clipboard
    3. Provides visual feedback when copy is successful
    4. Handles errors gracefully if clipboard access fails
    
    Parameters:
    - text_to_copy: The text content to be copied
    - button_text: The label displayed on the button
    - button_key: Unique identifier for the button
    """
    # Escape the text for JavaScript (handle quotes and newlines)
    escaped_text = text_to_copy.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n').replace('\r', '\\r')
    
    # Create unique IDs for this button
    button_id = f"copy_btn_{button_key}"
    
    # HTML and JavaScript for the copy button
    copy_button_html = f"""
    <div style="margin: 10px 0;">
        <button 
            id="{button_id}"
            onclick="copyToClipboard_{button_key}()"
            style="
                background-color: #ff4b4b;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                cursor: pointer;
                font-size: 14px;
                font-weight: 500;
                transition: background-color 0.3s;
            "
            onmouseover="this.style.backgroundColor='#ff6b6b'"
            onmouseout="this.style.backgroundColor='#ff4b4b'"
        >
            📋 {button_text}
        </button>
        <span id="status_{button_key}" style="margin-left: 10px; color: green; font-weight: bold;"></span>
    </div>
    
    <script>
        function copyToClipboard_{button_key}() {{
            const text = "{escaped_text}";
            const button = document.getElementById("{button_id}");
            const status = document.getElementById("status_{button_key}");
            
            // Try to use the modern Clipboard API
            if (navigator.clipboard && window.isSecureContext) {{
                navigator.clipboard.writeText(text).then(function() {{
                    // Success feedback
                    status.textContent = "✅ Copied!";
                    button.style.backgroundColor = "#28a745";
                    button.textContent = "✅ Copied!";
                    
                    // Reset after 2 seconds
                    setTimeout(function() {{
                        status.textContent = "";
                        button.style.backgroundColor = "#ff4b4b";
                        button.textContent = "📋 {button_text}";
                    }}, 2000);
                }}).catch(function(err) {{
                    // Fallback if Clipboard API fails
                    fallbackCopy_{button_key}(text);
                }});
            }} else {{
                // Fallback for older browsers or non-secure contexts
                fallbackCopy_{button_key}(text);
            }}
        }}
        
        function fallbackCopy_{button_key}(text) {{
            const textArea = document.createElement("textarea");
            textArea.value = text;
            textArea.style.position = "fixed";
            textArea.style.left = "-999999px";
            textArea.style.top = "-999999px";
            document.body.appendChild(textArea);
            textArea.focus();
            textArea.select();
            
            try {{
                const successful = document.execCommand('copy');
                const button = document.getElementById("{button_id}");
                const status = document.getElementById("status_{button_key}");
                
                if (successful) {{
                    status.textContent = "✅ Copied!";
                    button.style.backgroundColor = "#28a745";
                    button.textContent = "✅ Copied!";
                    
                    setTimeout(function() {{
                        status.textContent = "";
                        button.style.backgroundColor = "#ff4b4b";
                        button.textContent = "📋 {button_text}";
                    }}, 2000);
                }} else {{
                    status.textContent = "❌ Copy failed";
                    setTimeout(function() {{
                        status.textContent = "";
                    }}, 2000);
                }}
            }} catch (err) {{
                const status = document.getElementById("status_{button_key}");
                status.textContent = "❌ Copy not supported";
                setTimeout(function() {{
                    status.textContent = "";
                }}, 2000);
            }}
            
            document.body.removeChild(textArea);
        }}
    </script>
    """
    
    # Render the HTML component
    components.html(copy_button_html, height=60)

def display_analysis(df):
    """
    Display the main analysis: clients where Total == D0 and Total != 0
    Now includes threshold configuration for filtering values in the WhatsApp message
    """
    st.header("🎯 Analysis: Clients where Total = D0 (excluding zeros)")
    
    # Add threshold configuration section
    st.subheader("⚙️ Message Threshold Configuration")
    st.markdown("Configure which values should appear in the WhatsApp message based on thresholds:")
    
    # Create columns for threshold inputs
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**D0 Values (Main Analysis)**")
        # Negative threshold for D0 (values BELOW this threshold will be included)
        d0_negative_threshold = st.number_input(
            "D0 Negative Threshold",
            value=-1000.0,
            step=100.0,
            help="Include D0 values below this amount (e.g., -1000 includes all values less than -1000)"
        )
        
        # Positive threshold for D0 (values ABOVE this threshold will be included)
        d0_positive_threshold = st.number_input(
            "D0 Positive Threshold",
            value=1000.0,
            step=100.0,
            help="Include D0 values above this amount (e.g., 1000 includes all values greater than 1000)"
        )
    
    with col2:
        st.markdown("**D+1, D+2, D+3 Values (Additional Analysis)**")
        # Threshold for D+1, D+2, D+3 negative values
        d_plus_negative_threshold = st.number_input(
            "D+1/D+2/D+3 Negative Threshold",
            value=-500.0,
            step=100.0,
            help="Include D+1, D+2, D+3 negative values below this amount"
        )
    
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
        display_assessor_data(
            filtered_df, 
            assessor, 
            d0_negative_threshold, 
            d0_positive_threshold, 
            d_plus_negative_threshold
        )

def sort_data_for_whatsapp(data):
    """
    Sort data for WhatsApp message:
    1. Negative values first (most negative to least negative)
    2. Then positive values (largest to smallest)
    
    This function separates the data into negative and positive values,
    sorts each group appropriately, then combines them back together.
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

def get_d_plus_negatives(df, assessor, threshold):
    """
    Get clients with negative values in D+1, D+2, or D+3 columns that are below the threshold.
    
    This function:
    1. Filters data for the specific assessor
    2. Checks each D+ column for values below the threshold
    3. Returns a list of dictionaries with client info and which columns have negative values
    
    Parameters:
    - df: The main dataframe
    - assessor: The assessor name to filter by
    - threshold: The threshold value (negative values below this will be included)
    
    Returns:
    - List of dictionaries containing client information and negative D+ values
    """
    assessor_data = df[df['Assessor'] == assessor].copy()
    d_plus_negatives = []
    
    # Check each row for negative values in D+1, D+2, D+3
    for _, row in assessor_data.iterrows():
        negative_columns = []
        negative_values = []
        
        # Check each D+ column
        for col in ['D+1', 'D+2', 'D+3']:
            if col in df.columns and row[col] < threshold:
                negative_columns.append(col)
                negative_values.append(row[col])
        
        # If any negative values found, add to list
        if negative_columns:
            d_plus_negatives.append({
                'Conta': row['Conta'],
                'Cliente': row['Cliente'],
                'negative_columns': negative_columns,
                'negative_values': negative_values,
                'most_negative': min(negative_values)  # For sorting purposes
            })
    
    # Sort by most negative value first
    d_plus_negatives.sort(key=lambda x: x['most_negative'])
    
    return d_plus_negatives

def display_assessor_data(df, assessor, d0_negative_threshold, d0_positive_threshold, d_plus_negative_threshold):
    """
    Display data for a specific assessor with copy-friendly format
    Now includes threshold filtering for the WhatsApp message and copy buttons
    
    Parameters:
    - df: The filtered dataframe (where Total == D0)
    - assessor: The assessor name
    - d0_negative_threshold: Threshold for D0 negative values
    - d0_positive_threshold: Threshold for D0 positive values  
    - d_plus_negative_threshold: Threshold for D+1/D+2/D+3 negative values
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
    
    # Generate WhatsApp message with thresholds
    st.subheader("📱 WhatsApp Message")
    whatsapp_message = generate_whatsapp_message(
        assessor_data, 
        assessor, 
        d0_negative_threshold, 
        d0_positive_threshold, 
        d_plus_negative_threshold,
        df  # Pass the original df to check D+ columns
    )
    
    # Create two columns: one for the text area, one for the copy button
    col1, col2 = st.columns([4, 1])
    
    with col1:
        # Display in a text area for easy copying
        st.text_area(
            f"Message for Assessor {assessor}:",
            whatsapp_message,
            height=400,  # Increased height to accommodate more content
            key=f"whatsapp_{assessor}",
            help="You can also select text manually and copy with Ctrl+C"
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)  # Add some spacing
        # Add the copy button using our custom function
        create_copy_button(
            text_to_copy=whatsapp_message,
            button_text=f"Copy Message",
            button_key=f"assessor_{assessor}"
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

def generate_whatsapp_message(data, assessor, d0_negative_threshold, d0_positive_threshold, d_plus_negative_threshold, original_df):
    """
    Generate a formatted message for WhatsApp with threshold filtering and D+ negative values
    
    This function creates a comprehensive WhatsApp message that includes:
    1. D0 values that meet the threshold criteria
    2. D+1, D+2, D+3 negative values that meet their threshold
    3. Proper sorting and formatting for easy reading
    
    Parameters:
    - data: Filtered data where Total == D0
    - assessor: Assessor name
    - d0_negative_threshold: Threshold for D0 negative values
    - d0_positive_threshold: Threshold for D0 positive values
    - d_plus_negative_threshold: Threshold for D+ negative values
    - original_df: Original dataframe to check D+ columns
    """
    # Filter D0 data based on thresholds
    d0_filtered = data[
        (data['Total'] < d0_negative_threshold) |  # Negative values below threshold
        (data['Total'] > d0_positive_threshold)    # Positive values above threshold
    ].copy()
    
    # Get D+ negative values
    d_plus_negatives = get_d_plus_negatives(original_df, assessor, d_plus_negative_threshold)
    
    # Sort D0 data according to requirements
    d0_sorted = sort_data_for_whatsapp(d0_filtered)
    
    # Start building the message
    message = f"🏦 *RELATÓRIO - ASSESSOR {assessor}*\n"
    message += f"📊 Total de clientes analisados: {len(data)}\n"
    message += f"📋 Clientes no relatório: {len(d0_filtered) + len(d_plus_negatives)}\n"
    
    # Add threshold information
    message += f"⚙️ Filtros aplicados:\n"
    message += f"   • D0 negativos < R$ {d0_negative_threshold:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.') + "\n"
    message += f"   • D0 positivos > R$ {d0_positive_threshold:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.') + "\n"
    message += f"   • D+1/D+2/D+3 negativos < R$ {d_plus_negative_threshold:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.') + "\n"
    
    message += "=" * 40 + "\n\n"
    
    # Add D0 analysis section
    if len(d0_sorted) > 0:
        message += "📈 *ANÁLISE PRINCIPAL (Total = D0)*\n"
        message += "=" * 35 + "\n"
        
        # Add summary of negative and positive values for D0
        d0_negative_count = len(d0_sorted[d0_sorted['Total'] < 0])
        d0_positive_count = len(d0_sorted[d0_sorted['Total'] > 0])
        
        if d0_negative_count > 0:
            message += f"⚠️ Valores negativos D0: {d0_negative_count}\n"
        if d0_positive_count > 0:
            message += f"✅ Valores positivos D0: {d0_positive_count}\n"
        
        message += "\n"
        
        # Add D0 section headers and data
        current_section = None
        
        for _, row in d0_sorted.iterrows():
            # Format the Total value as Brazilian currency for the message
            total_formatted = f"R$ {row['Total']:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.')
            
            # Add section header if needed
            is_negative = row['Total'] < 0
            if current_section is None or current_section != is_negative:
                current_section = is_negative
                if current_section:
                    message += "🔴 *VALORES NEGATIVOS D0 (PRIORIDADE MÁXIMA)*\n"
                    message += "-" * 40 + "\n"
                else:
                    message += "\n🟢 *VALORES POSITIVOS D0*\n"
                    message += "-" * 25 + "\n"
            
            # Add client information
            message += f"👤 *{row['Cliente']}*\n"
            message += f"📋 Conta: {row['Conta']}\n"
            message += f"💰 Valor D0: {total_formatted}\n"
            
            # Add priority indicator for negative values
            if row['Total'] < 0:
                message += "⚠️ *ATENÇÃO: VALOR NEGATIVO D0*\n"
            
            message += "-" * 30 + "\n"
    
    # Add D+ negative values section
    if d_plus_negatives:
        message += "\n🔍 *ANÁLISE ADICIONAL (D+1, D+2, D+3 NEGATIVOS)*\n"
        message += "=" * 45 + "\n"
        message += f"⚠️ Clientes com valores negativos em D+: {len(d_plus_negatives)}\n\n"
        
        message += "🔴 *VALORES NEGATIVOS D+1/D+2/D+3 (ALTA PRIORIDADE)*\n"
        message += "-" * 45 + "\n"
        
        for client in d_plus_negatives:
            message += f"👤 *{client['Cliente']}*\n"
            message += f"📋 Conta: {client['Conta']}\n"
            
            # Add each negative D+ value
            for i, col in enumerate(client['negative_columns']):
                value_formatted = f"R$ {client['negative_values'][i]:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.')
                message += f"💰 {col}: {value_formatted}\n"
            
            message += "⚠️ *ATENÇÃO: VALORES NEGATIVOS EM D+*\n"
            message += "-" * 30 + "\n"
    
    # Add summary if no data meets criteria
    if len(d0_sorted) == 0 and len(d_plus_negatives) == 0:
        message += "✅ *NENHUM CLIENTE ATENDE OS CRITÉRIOS DEFINIDOS*\n"
        message += "Todos os valores estão dentro dos limites estabelecidos.\n"
    
    message += f"\n📅 Relatório gerado em: {pd.Timestamp.now().strftime('%d/%m/%Y às %H:%M')}"
    
    return message

def create_excel_download(data):
    """
    Create an Excel file in memory for download
    
    This function:
    1. Creates a BytesIO buffer to store the Excel file in memory
    2. Uses pandas ExcelWriter to write the data to the buffer
    3. Returns the buffer contents as bytes for download
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