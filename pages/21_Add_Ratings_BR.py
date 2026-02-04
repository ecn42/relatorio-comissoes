import os
import sqlite3
from typing import Optional, Dict, List, Tuple

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Add Ratings BR", layout="wide")

# Simple Authentication
if not st.session_state.get("authenticated", False):
    st.warning("Please enter the password on the Home page first.")
    st.write("Status: Não Autenticado")
    st.stop()

st.write("Autenticado")

DB_PATH = "databases/gorila_positions.db"
TARGET_TABLE = "pmv_plus_gorila"


def load_sheet(uploaded_file) -> Optional[pd.DataFrame]:
    """
    Load a CSV or Excel file into a DataFrame.
    """
    if uploaded_file is None:
        return None

    name = uploaded_file.name.lower()
    if name.endswith((".xls", ".xlsx", ".xlsm", ".xlsb", ".ods")):
        return pd.read_excel(uploaded_file)

    # Assume CSV, try to sniff separator
    return pd.read_csv(uploaded_file, sep=None, engine="python")


def read_pmv_plus_gorila(db_path: str = DB_PATH) -> pd.DataFrame:
    """
    Read the 'pmv_plus_gorila' table from gorila_positions.db.
    Returns DataFrame with rowid column for updates.
    """
    if not os.path.exists(db_path):
        st.error(f"Database not found: {db_path}")
        return pd.DataFrame()

    try:
        conn = sqlite3.connect(db_path)
        # Include rowid for updates
        query = f"SELECT rowid AS rowid, * FROM {TARGET_TABLE}"
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Error reading database: {e}")
        return pd.DataFrame()


def ensure_rating_column(db_path: str = DB_PATH) -> bool:
    """
    Ensure the 'rating' column exists in pmv_plus_gorila table.
    Returns True if column exists or was created successfully.
    """
    if not os.path.exists(db_path):
        return False

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if rating column exists
        cursor.execute(f"PRAGMA table_info({TARGET_TABLE})")
        columns = [row[1] for row in cursor.fetchall()]
        
        if "rating" not in columns:
            # Add rating column
            cursor.execute(f"ALTER TABLE {TARGET_TABLE} ADD COLUMN rating TEXT")
            conn.commit()
        
        conn.close()
        return True
    except Exception as e:
        st.error(f"Error ensuring rating column: {e}")
        return False


def update_ratings(
    df_db: pd.DataFrame,
    df_excel: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, int], List[str]]:
    """
    Match Excel data to database records and update rating column.
    
    Args:
        df_db: DataFrame from pmv_plus_gorila with rowid
        df_excel: DataFrame from uploaded Excel file
        
    Returns:
        Tuple of (updated_df, stats_dict, unmatched_codes)
        stats_dict: {'cri_cra_matched': int, 'deb_matched': int, 'total_updated': int}
        unmatched_codes: List of CÓDIGO values that didn't match
    """
    df_updated = df_db.copy()
    
    # Ensure rating column exists in DataFrame
    if "rating" not in df_updated.columns:
        df_updated["rating"] = pd.NA
    
    stats = {
        "cri_cra_matched": 0,
        "deb_matched": 0,
        "total_updated": 0,
    }
    unmatched_codes = []
    
    # Normalize column names (strip whitespace, handle case)
    df_excel.columns = df_excel.columns.str.strip()
    
    # Check required columns
    required_cols = ["TIPO", "CÓDIGO", "AGÊNCIA"]
    missing_cols = [col for col in required_cols if col not in df_excel.columns]
    if missing_cols:
        st.error(f"Missing required columns in Excel: {', '.join(missing_cols)}")
        return df_updated, stats, unmatched_codes
    
    # Process each row in Excel
    for _, excel_row in df_excel.iterrows():
        tipo = str(excel_row["TIPO"]).strip() if pd.notna(excel_row["TIPO"]) else ""
        codigo = str(excel_row["CÓDIGO"]).strip() if pd.notna(excel_row["CÓDIGO"]) else ""
        agencia = str(excel_row["AGÊNCIA"]).strip() if pd.notna(excel_row["AGÊNCIA"]) else ""
        
        if not codigo or not agencia:
            continue
        
        # Skip if tipo doesn't match our criteria
        tipo_upper = tipo.upper()
        is_cri_cra = tipo_upper in ["CRI", "CRA"]
        is_deb = tipo_upper in ["DEB.", "DEB. ISENTA", "DEBENTURE", "DEBENTURE ISENTA"]
        
        if not (is_cri_cra or is_deb):
            continue
        
        matched = False
        
        if is_cri_cra:
            # Match by parsed_cetip_code
            if "parsed_cetip_code" in df_updated.columns:
                # Normalize for comparison (strip, handle NaN)
                mask = (
                    df_updated["parsed_cetip_code"]
                    .astype(str)
                    .str.strip()
                    .str.upper()
                    == codigo.upper()
                )
                if mask.any():
                    df_updated.loc[mask, "rating"] = agencia
                    matched_count = mask.sum()
                    stats["cri_cra_matched"] += matched_count
                    stats["total_updated"] += matched_count
                    matched = True
        
        elif is_deb:
            # Match by security_name
            if "security_name" in df_updated.columns:
                # Normalize for comparison (strip, handle NaN)
                mask = (
                    df_updated["security_name"]
                    .astype(str)
                    .str.strip()
                    .str.upper()
                    == codigo.upper()
                )
                if mask.any():
                    df_updated.loc[mask, "rating"] = agencia
                    matched_count = mask.sum()
                    stats["deb_matched"] += matched_count
                    stats["total_updated"] += matched_count
                    matched = True
        
        if not matched:
            unmatched_codes.append(f"{tipo}: {codigo}")
    
    return df_updated, stats, unmatched_codes


def save_ratings(df: pd.DataFrame, db_path: str = DB_PATH) -> bool:
    """
    Save updated ratings to database using rowid-based updates.
    Only updates the rating column.
    """
    if not os.path.exists(db_path):
        st.error(f"Database not found: {db_path}")
        return False
    
    if "rowid" not in df.columns:
        st.error("Internal error: 'rowid' column missing in DataFrame.")
        return False
    
    if "rating" not in df.columns:
        st.error("Internal error: 'rating' column missing in DataFrame.")
        return False
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        updated_count = 0
        for _, row in df.iterrows():
            rowid_val = row["rowid"]
            rating_val = row["rating"]
            
            # Skip rows without a valid rowid
            if pd.isna(rowid_val):
                continue
            
            # Update only rating column
            cursor.execute(
                f"UPDATE {TARGET_TABLE} SET rating = ? WHERE rowid = ?",
                (rating_val if pd.notna(rating_val) else None, int(rowid_val)),
            )
            updated_count += 1
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Error saving to database: {e}")
        return False


from io import BytesIO

def get_template_excel() -> bytes:
    """
    Generate an Excel template for ratings upload.
    Returns bytes of the Excel file.
    """
    # Create sample data
    data = {
        "TIPO": ["CRI", "DEB.", "CRA", "DEBENTURE ISENTA"],
        "CÓDIGO": ["CRI11", "DEB11", "CRA11", "DEB12"],
        "AGÊNCIA": ["AAA", "AA+", "A-", "BBB+"],
    }
    df = pd.DataFrame(data)
    
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Ratings")
        
        # Adjust column width
        workbook = writer.book
        worksheet = writer.sheets["Ratings"]
        for i, col in enumerate(df.columns):
            column_len = max(df[col].astype(str).map(len).max(), len(col)) + 2
            worksheet.set_column(i, i, column_len)
            
    return output.getvalue()


def main() -> None:
    st.title("Add Ratings BR")
    
    st.markdown(
        "This app will:\n"
        "1. Load the **pmv_plus_gorila** table from `gorila_positions.db`.\n"
        "2. Load an Excel file with rating information.\n"
        "3. Match records:\n"
        "   - **CRI/CRA**: Match **CÓDIGO** to `parsed_cetip_code` in database\n"
        "   - **Deb./Deb. Isenta**: Match **CÓDIGO** to `security_name` in database\n"
        "4. Update the **rating** column with **AGÊNCIA** value.\n"
        "5. Save updated ratings back to the database."
    )
    
    # Check if database exists
    if not os.path.exists(DB_PATH):
        st.error(f"Database not found: {DB_PATH}")
        return
    
    # Ensure rating column exists
    if not ensure_rating_column():
        st.error("Failed to ensure rating column exists in database.")
        return
    
    # Load database table
    with st.spinner("Loading pmv_plus_gorila table from database..."):
        df_db = read_pmv_plus_gorila()
    
    if df_db.empty:
        st.error("The pmv_plus_gorila table is empty or does not exist.")
        return
    
    st.markdown(
        f"Loaded **pmv_plus_gorila** table with **{len(df_db)}** rows and "
        f"**{len(df_db.columns)}** columns."
    )
    
    # Download Template Button
    st.subheader("1. Download Template")
    st.markdown("Download the template sheet below to fill in your rating updates.")
    
    template_bytes = get_template_excel()
    st.download_button(
        label="Download Template Excel",
        data=template_bytes,
        file_name="ratings_template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    
    st.divider()
    

    # Upload Excel file
    st.subheader("2. Upload Filled Template OR Paste Data")
    
    col_upload, col_paste = st.columns(2)
    
    with col_upload:
        excel_file = st.file_uploader(
            "Upload Excel file",
            type=["xls", "xlsx", "xlsm", "xlsb", "ods"],
            key="ratings_excel_file",
        )
        
    with col_paste:
        paste_text = st.text_area(
            "Paste Data (TIPO, CÓDIGO, AGÊNCIA)", 
            height=150, 
            placeholder="CRI\tCRI11\tAAA\nDEB.\tDEB11\tAA+"
        )
        st.caption("Copy from Excel (tab-separated) or use semicolon (;). Must have 3 columns.")

    df_excel = None
    
    if excel_file is not None:
        # Load Excel file
        df_excel = load_sheet(excel_file)
        if df_excel is None or df_excel.empty:
            st.error("Could not read the uploaded Excel file or file is empty.")
            return
            
    elif paste_text:
        # Process pasted text
        from io import StringIO
        import csv
        
        try:
            # Try to sniff separator
            sniffer = csv.Sniffer()
            # Sample first few lines
            sample = "\n".join(paste_text.splitlines()[:5])
            try:
                dialect = sniffer.sniff(sample)
                sep = dialect.delimiter
            except:
                # Fallback to tab
                sep = "\t"
                
            # Read CSV
            df_excel = pd.read_csv(StringIO(paste_text), sep=sep, engine="python", header=None)
            
            # If header is likely missing (first row doesn't look like header), try to assign checks
            # But usually paste from Excel won't have headers if user just selects data, 
            # OR they will copy headers.
            # Let's try to detect if first row matches our expected headers
            first_row_vals = [str(x).upper().strip() for x in df_excel.iloc[0].tolist()]
            expected_headers = ["TIPO", "CÓDIGO", "AGÊNCIA"] # Agência might be tricky with accent
            
            # Simple check: if first row contains "TIPO", use it as header
            if "TIPO" in first_row_vals:
                df_excel.columns = df_excel.iloc[0]
                df_excel = df_excel[1:]
            else:
                # Assign default headers if 3 columns
                if len(df_excel.columns) >= 3:
                     # Rename first 3 columns
                     cols = list(df_excel.columns)
                     cols[0] = "TIPO"
                     cols[1] = "CÓDIGO"
                     cols[2] = "AGÊNCIA"
                     df_excel.columns = cols
                else:
                    st.error(f"Pasted data has {len(df_excel.columns)} columns, expected at least 3 (TIPO, CÓDIGO, AGÊNCIA).")
                    return
                    
        except Exception as e:
            st.error(f"Error parsing text: {e}")
            return

    if df_excel is None:
        st.info("Upload an Excel file or paste data to proceed.")
        return
    
    st.markdown(
        f"Loaded data with **{len(df_excel)}** rows and "
        f"**{len(df_excel.columns)}** columns."
    )
    
    # Show preview of Excel data
    st.subheader("Preview of Excel Data")
    st.dataframe(df_excel.head(20))
    
    # Show Excel columns
    st.subheader("Excel Columns")
    st.write(", ".join(df_excel.columns.tolist()))
    
    # Validate required columns
    required_cols = ["TIPO", "CÓDIGO", "AGÊNCIA"]
    df_excel.columns = df_excel.columns.str.strip()
    missing_cols = [col for col in required_cols if col not in df_excel.columns]
    
    if missing_cols:
        st.error(f"Missing required columns: {', '.join(missing_cols)}")
        st.info(f"Available columns: {', '.join(df_excel.columns.tolist())}")
        return
    
    # Process matching
    with st.spinner("Processing matches..."):
        df_updated, stats, unmatched = update_ratings(df_db, df_excel)
    
    # Show summary
    st.subheader("Matching Summary")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("CRI/CRA Matched", stats["cri_cra_matched"])
    with col2:
        st.metric("Debentures Matched", stats["deb_matched"])
    with col3:
        st.metric("Total Records Updated", stats["total_updated"])
    
    if unmatched:
        st.warning(
            f"**{len(unmatched)}** codes did not match any records in the database:\n"
            + "\n".join([f"- {code}" for code in unmatched[:20]])
            + (f"\n... and {len(unmatched) - 20} more" if len(unmatched) > 20 else "")
        )
    
    if stats["total_updated"] == 0:
        st.info("No records were matched. Please check your Excel data and database.")
        return
    
    # Show preview of records that will be updated
    st.subheader("Preview of Records to be Updated")
    # Find rows that were updated (rating changed from original)
    if "rating" in df_db.columns:
        df_db_rating = df_db["rating"].fillna("")
        df_updated_rating = df_updated["rating"].fillna("")
        changed_mask = df_db_rating != df_updated_rating
    else:
        changed_mask = df_updated["rating"].notna()
    
    df_preview = df_updated[changed_mask].head(20)
    if not df_preview.empty:
        preview_cols = ["rowid", "security_name", "parsed_cetip_code", "rating"]
        available_cols = [col for col in preview_cols if col in df_preview.columns]
        st.dataframe(df_preview[available_cols])
    else:
        st.info("No preview available.")
    
    # Update button
    if st.button("Update Ratings in Database", type="primary"):
        with st.spinner("Saving updates to database..."):
            # Only save rows that were actually changed
            if "rating" in df_db.columns:
                df_db_rating = df_db["rating"].fillna("")
                df_updated_rating = df_updated["rating"].fillna("")
                changed_mask = df_db_rating != df_updated_rating
            else:
                changed_mask = df_updated["rating"].notna()
            
            df_to_save = df_updated[changed_mask].copy()
            
            if df_to_save.empty:
                st.warning("No changes to save.")
            else:
                success = save_ratings(df_to_save)
                if success:
                    st.success(
                        f"Successfully updated **{len(df_to_save)}** records in the database!"
                    )
                    st.balloons()
                else:
                    st.error("Failed to save updates to database.")


if __name__ == "__main__":
    main()

