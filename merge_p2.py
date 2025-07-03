import pandas as pd
import os
from pathlib import Path
import re
from collections import defaultdict

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

def merge_excel_files_by_date(folder_path, output_folder=None):
    """
    Merge Excel files by date, creating one file per date with all data in a single sheet.
    
    Args:
        folder_path (str): Path to the folder containing Excel files
        output_folder (str, optional): Path to save merged files. If None, saves in input folder
    """
    folder_path = Path(folder_path)
    
    if output_folder is None:
        output_folder = folder_path
    else:
        output_folder = Path(output_folder)
        output_folder.mkdir(exist_ok=True)  # Create output folder if it doesn't exist
    
    # Dictionary to group files by date
    files_by_date = defaultdict(list)
    
    # Get all Excel files and group them by date
    excel_files = list(folder_path.glob('*.xlsx'))
    
    print(f"Found {len(excel_files)} Excel files in {folder_path}")
    
    for file_path in excel_files:
        filename = file_path.name
        date = extract_date_from_filename(filename)
        
        if date:
            files_by_date[date].append(file_path)
            print(f"File: {filename} -> Date: {date}")
        else:
            print(f"Warning: Could not extract date from {filename}")
    
    print(f"\nFound {len(files_by_date)} unique dates to process")
    
    # Process each date group
    for date, file_list in files_by_date.items():
        print(f"\nProcessing date: {date} ({len(file_list)} files)")
        
        # List to store all dataframes for this date
        all_dataframes = []
        
        for file_path in file_list:
            try:
                # Read the Excel file
                df = pd.read_excel(file_path)
                
                # Get file type to add as a column
                file_type = get_file_type(file_path.name)
                
                # Add a column to identify the source file type
                # This helps you know which data came from which file
                df['Tipo_Arquivo'] = file_type
                
                # Add the dataframe to our list
                all_dataframes.append(df)
                
                print(f"  ✓ Loaded {file_path.name} ({len(df)} rows) - Type: {file_type}")
                
            except Exception as e:
                print(f"  ✗ Error processing {file_path.name}: {str(e)}")
        
        # Combine all dataframes for this date
        if all_dataframes:
            try:
                # Concatenate all dataframes
                # ignore_index=True creates a new continuous index
                # sort=False preserves the original column order
                combined_df = pd.concat(all_dataframes, ignore_index=True, sort=False)
                
                # Create output filename
                output_filename = f"MERGED_{date}_CERESCAPIT.xlsx"
                output_path = output_folder / output_filename
                
                # Save the combined dataframe to Excel
                combined_df.to_excel(output_path, index=False, sheet_name='Dados_Combinados')
                
                print(f"  📁 Created: {output_filename}")
                print(f"     Total rows: {len(combined_df)}")
                print(f"     Total columns: {len(combined_df.columns)}")
                
                # Show column names for reference
                print(f"     Columns: {list(combined_df.columns)}")
                
            except Exception as e:
                print(f"  ✗ Error combining data for date {date}: {str(e)}")
        else:
            print(f"  ⚠️  No valid data found for date {date}")
    
    print(f"\n🎉 Process completed! Merged files saved in: {output_folder}")

# Main execution
if __name__ == "__main__":
    # Your specific paths
    input_folder = r"C:\Users\Eduardo\Documents\GitHub\relatorios-comissoes\P2"
    output_folder = r"C:\Users\Eduardo\Documents\GitHub\relatorios-comissoes\P2\merged_files"
    
    print("=" * 60)
    print("EXCEL FILES MERGER BY DATE - SINGLE SHEET")
    print("=" * 60)
    print(f"Input folder:  {input_folder}")
    print(f"Output folder: {output_folder}")
    print("=" * 60)
    
    # Check if input folder exists
    if not Path(input_folder).exists():
        print(f"❌ Error: Input folder does not exist: {input_folder}")
        print("Please check the path and try again.")
    else:
        # Run the merge
        merge_excel_files_by_date(input_folder, output_folder)