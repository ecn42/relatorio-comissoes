import streamlit as st
import pandas as pd
import os
from pathlib import Path
import re
from collections import defaultdict
import zipfile
import io

def extract_date_from_filename(filename):
    """
    Extract the YYYYMM date from the filename.
    
    Args:
        filename (str): The filename to extract date from
        
    Returns:
        str or None: The extracted date (YYYYMM) or None if not found
    """
    # Pattern to match YYYYMM in the filename
    # Looking for 6 consecutive digits after the underscore
    pattern = r'_(\d{6})_'
    match = re.search(pattern, filename)
    
    if match:
        return match.group(1)  # Return the captured group (the date)
    return None

def get_file_type(filename):
    """
    Extract the file type from filename (everything after the last underscore, before .xlsx).
    
    Args:
        filename (str): The filename to extract type from
        
    Returns:
        str: The file type
    """
    # Remove the .xlsx extension and get everything after the last underscore
    base_name = filename.replace('.xlsx', '')
    return base_name.split('_')[-1]

def process_uploaded_files(uploaded_files):
    """
    Process uploaded files and merge them by date.
    
    Args:
        uploaded_files: List of uploaded file objects from Streamlit
        
    Returns:
        dict: Dictionary with date as key and merged DataFrame as value
    """
    # Dictionary to group files by date
    files_by_date = defaultdict(list)
    
    # Process each uploaded file
    for uploaded_file in uploaded_files:
        filename = uploaded_file.name
        date = extract_date_from_filename(filename)
        
        if date:
            files_by_date[date].append(uploaded_file)
            st.write(f"✓ File: {filename} -> Date: {date}")
        else:
            st.warning(f"⚠️ Could not extract date from {filename}")
    
    # Dictionary to store merged dataframes
    merged_data = {}
    
    # Process each date group
    for date, file_list in files_by_date.items():
        st.write(f"\n**Processing date: {date} ({len(file_list)} files)**")
        
        # List to store all dataframes for this date
        all_dataframes = []
        
        for uploaded_file in file_list:
            try:
                # Read the Excel file from uploaded file object
                df = pd.read_excel(uploaded_file)
                
                # Get file type to add as a column
                file_type = get_file_type(uploaded_file.name)
                
                # Add a column to identify the source file type
                df['Tipo_Arquivo'] = file_type
                
                # Add the dataframe to our list
                all_dataframes.append(df)
                
                st.write(f"  ✓ Loaded {uploaded_file.name} ({len(df)} rows) - Type: {file_type}")
                
            except Exception as e:
                st.error(f"  ✗ Error processing {uploaded_file.name}: {str(e)}")
        
        # Combine all dataframes for this date
        if all_dataframes:
            try:
                # Concatenate all dataframes
                combined_df = pd.concat(all_dataframes, ignore_index=True, sort=False)
                merged_data[date] = combined_df
                
                st.success(f"  📁 Successfully merged data for {date}")
                st.write(f"     Total rows: {len(combined_df)}")
                st.write(f"     Total columns: {len(combined_df.columns)}")
                
            except Exception as e:
                st.error(f"  ✗ Error combining data for date {date}: {str(e)}")
        else:
            st.warning(f"  ⚠️  No valid data found for date {date}")
    
    return merged_data

def create_download_zip(merged_data):
    """
    Create a ZIP file containing all merged Excel files.
    
    Args:
        merged_data (dict): Dictionary with date as key and DataFrame as value
        
    Returns:
        bytes: ZIP file content as bytes
    """
    # Create a BytesIO object to store the ZIP file in memory
    zip_buffer = io.BytesIO()
    
    # Create a ZIP file
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for date, df in merged_data.items():
            # Create Excel file in memory
            excel_buffer = io.BytesIO()
            df.to_excel(excel_buffer, index=False, sheet_name='Dados_Combinados')
            excel_buffer.seek(0)
            
            # Add Excel file to ZIP
            filename = f"MERGED_{date}_CERESCAPIT.xlsx"
            zip_file.writestr(filename, excel_buffer.getvalue())
    
    zip_buffer.seek(0)
    return zip_buffer.getvalue()

def main():
    """
    Main Streamlit application function.
    """
    # Set page configuration
    st.set_page_config(
        page_title="Excel Files Merger",
        page_icon="📊",
        layout="wide"
    )
    
    # Main title and description
    st.title("📊 Excel Files Merger by Date")
    st.markdown("---")
    
    st.markdown("""
    ### How it works:
    1. **Upload your Excel files** using the file uploader below
    2. **Files are automatically grouped by date** (extracted from filename pattern `_YYYYMM_`)
    3. **Each date group is merged** into a single Excel file
    4. **Download individual files** or get all files in a ZIP archive
    
    **Expected filename format:** `something_202401_type.xlsx`
    """)
    
    # File uploader
    st.markdown("### 📁 Upload Excel Files")
    uploaded_files = st.file_uploader(
        "Choose Excel files to merge",
        type=['xlsx'],
        accept_multiple_files=True,
        help="Select multiple Excel files that follow the naming pattern with dates"
    )
    
    # Process files when uploaded
    if uploaded_files:
        st.markdown("---")
        st.markdown("### 🔄 Processing Files")
        
        # Show file count
        st.info(f"Found {len(uploaded_files)} Excel files to process")
        
        # Process the files
        with st.spinner("Processing files..."):
            merged_data = process_uploaded_files(uploaded_files)
        
        # Show results
        if merged_data:
            st.markdown("---")
            st.markdown("### 📋 Results Summary")
            
            # Create summary table
            summary_data = []
            for date, df in merged_data.items():
                summary_data.append({
                    'Date': date,
                    'Total Rows': len(df),
                    'Total Columns': len(df.columns),
                    'File Types': ', '.join(df['Tipo_Arquivo'].unique())
                })
            
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True)
            
            # Download options
            st.markdown("---")
            st.markdown("### 💾 Download Options")
            
            # Create two columns for download options
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Individual Files:**")
                # Individual file downloads
                for date, df in merged_data.items():
                    # Convert DataFrame to Excel in memory
                    excel_buffer = io.BytesIO()
                    df.to_excel(excel_buffer, index=False, sheet_name='Dados_Combinados')
                    excel_buffer.seek(0)
                    
                    filename = f"MERGED_{date}_CERESCAPIT.xlsx"
                    st.download_button(
                        label=f"📄 Download {date}",
                        data=excel_buffer.getvalue(),
                        file_name=filename,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
            
            with col2:
                st.markdown("**All Files (ZIP):**")
                # ZIP download with all files
                zip_data = create_download_zip(merged_data)
                st.download_button(
                    label="📦 Download All Files (ZIP)",
                    data=zip_data,
                    file_name="merged_excel_files.zip",
                    mime="application/zip"
                )
            
            # Preview section
            st.markdown("---")
            st.markdown("### 👀 Data Preview")
            
            # Let user select which date to preview
            selected_date = st.selectbox(
                "Select a date to preview:",
                options=list(merged_data.keys()),
                help="Choose which merged dataset you want to preview"
            )
            
            if selected_date:
                preview_df = merged_data[selected_date]
                
                # Show basic info
                st.write(f"**Preview for {selected_date}:**")
                st.write(f"Shape: {preview_df.shape[0]} rows × {preview_df.shape[1]} columns")
                
                # Show column info
                with st.expander("📋 Column Information"):
                    col_info = pd.DataFrame({
                        'Column': preview_df.columns,
                        'Data Type': preview_df.dtypes,
                        'Non-Null Count': preview_df.count(),
                        'Null Count': preview_df.isnull().sum()
                    })
                    st.dataframe(col_info, use_container_width=True)
                
                # Show data preview
                st.dataframe(preview_df.head(100), use_container_width=True)
        
        else:
            st.error("❌ No valid data could be processed from the uploaded files.")
            st.markdown("""
            **Possible issues:**
            - Files don't follow the expected naming pattern `_YYYYMM_`
            - Files are corrupted or not valid Excel files
            - Files are empty
            """)
    
    else:
        st.info("👆 Please upload Excel files to get started")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <small>Excel Files Merger Dashboard | Built with Streamlit</small>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()