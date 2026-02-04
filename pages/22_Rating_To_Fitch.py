import os
import sqlite3
from typing import Optional, Dict, List, Tuple

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Rating to Fitch", layout="wide")

# Simple Authentication
if not st.session_state.get("authenticated", False):
    st.warning("Please enter the password on the Home page first.")
    st.write("Status: Não Autenticado")
    st.stop()

st.write("Autenticado")

DB_PATH = "databases/gorila_positions.db"
TARGET_TABLE = "pmv_plus_gorila"



from utils.rating_utils import translate_to_fitch

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


def update_ratings_to_fitch(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, int], List[str]]:
    """
    Apply Fitch translations to ratings column.
    
    Args:
        df: DataFrame from pmv_plus_gorila with rowid and rating column
        
    Returns:
        Tuple of (updated_df, stats_dict, unmatched_ratings)
        stats_dict: {
            'moodys_converted': int,
            'sp_converted': int,
            'fitch_normalized': int,
            'unchanged': int,
            'unmatched': int,
            'total_updated': int
        }
        unmatched_ratings: List of rating values that couldn't be matched
    """
    df_updated = df.copy()
    
    # Ensure rating column exists
    if "rating" not in df_updated.columns:
        df_updated["rating"] = pd.NA
    
    stats = {
        "moodys_converted": 0,
        "sp_converted": 0,
        "fitch_normalized": 0,
        "unchanged": 0,
        "unmatched": 0,
        "total_updated": 0,
    }
    
    unmatched_ratings = []
    
    # Define original mappings to detect source
    moodys_ratings = {
        "AAA", "AA1", "AA2", "AA3", "A1", "A2", "A3",
        "BAA1", "BAA2", "BAA3", "BA1", "BA2", "BA3",
        "B1", "B2", "B3", "CAA1", "CAA2", "CAA3", "CA", "C"
    }
    
    sp_ratings = {
        "AAA", "AA+", "AA", "AA-", "A+", "A", "A-",
        "BBB+", "BBB", "BBB-", "BB+", "BB", "BB-",
        "B+", "B", "B-", "CCC+", "CCC", "CCC-", "CC", "C", "D"
    }
    
    valid_fitch = {
        "AAA", "AA+", "AA", "AA-",
        "A+", "A", "A-",
        "BBB+", "BBB", "BBB-",
        "BB+", "BB", "BB-",
        "B+", "B", "B-",
        "CCC", "DDD", "DD", "D"
    }
    
    # Process each row
    for idx, row in df_updated.iterrows():
        original_rating = row["rating"]
        
        # Skip if no rating
        if pd.isna(original_rating) or not str(original_rating).strip() or str(original_rating).strip() == "-":
            stats["unchanged"] += 1
            continue
        
        original_str = str(original_rating).strip()
        original_upper = original_str.upper()
        
        # Translate to Fitch
        fitch_rating = translate_to_fitch(original_str)
        
        if fitch_rating is None:
            # Couldn't match
            stats["unmatched"] += 1
            if original_upper not in [u.upper() for u in unmatched_ratings]:
                unmatched_ratings.append(original_str)
            continue
        
        # Check if already in correct Fitch format (no change needed)
        if original_upper == fitch_rating:
            # Already correct Fitch format
            stats["unchanged"] += 1
            continue
        
        # Determine source for statistics (rating will be changed)
        if original_upper in moodys_ratings:
            stats["moodys_converted"] += 1
        elif original_upper in sp_ratings:
            stats["sp_converted"] += 1
        elif original_upper in valid_fitch:
            # Was already Fitch but needed normalization (case change)
            stats["fitch_normalized"] += 1
        else:
            # Shouldn't happen, but count as converted
            pass  # Will be counted in total_updated below
        
        # Update the rating
        df_updated.at[idx, "rating"] = fitch_rating
        stats["total_updated"] += 1
    
    return df_updated, stats, unmatched_ratings


def save_fitch_ratings(df_updated: pd.DataFrame, df_original: pd.DataFrame, db_path: str = DB_PATH) -> bool:
    """
    Save updated ratings to database using rowid-based updates.
    Only updates rows where the rating actually changed.
    """
    if not os.path.exists(db_path):
        st.error(f"Database not found: {db_path}")
        return False
    
    if "rowid" not in df_updated.columns:
        st.error("Internal error: 'rowid' column missing in DataFrame.")
        return False
    
    if "rating" not in df_updated.columns:
        st.error("Internal error: 'rating' column missing in DataFrame.")
        return False
    
    try:
        # Find rows that changed
        df_orig_rating = df_original["rating"].fillna("")
        df_upd_rating = df_updated["rating"].fillna("")
        changed_mask = df_orig_rating.astype(str) != df_upd_rating.astype(str)
        
        df_to_save = df_updated[changed_mask].copy()
        
        if df_to_save.empty:
            return True  # Nothing to update
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        updated_count = 0
        for _, row in df_to_save.iterrows():
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


def main() -> None:
    st.title("Rating to Fitch Translator")
    
    st.markdown(
        "This app will:\n"
        "1. Load the **pmv_plus_gorila** table from `gorila_positions.db`.\n"
        "2. Translate Moody's and S&P ratings to Fitch format in the **rating** column.\n"
        "3. Normalize existing Fitch ratings to uppercase format.\n"
        "4. Show statistics and preview before applying changes.\n"
        "5. Update the database with translated ratings."
    )
    
    # Check if database exists
    if not os.path.exists(DB_PATH):
        st.error(f"Database not found: {DB_PATH}")
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
    
    # Check if rating column exists
    if "rating" not in df_db.columns:
        st.error("The 'rating' column does not exist in the table.")
        return
    
    # Show current rating statistics
    st.subheader("Current Rating Statistics")
    df_with_ratings = df_db[df_db["rating"].notna() & (df_db["rating"].astype(str).str.strip() != "") & (df_db["rating"].astype(str).str.strip() != "-")]
    st.write(f"Records with ratings: **{len(df_with_ratings)}** out of **{len(df_db)}** total")
    
    if len(df_with_ratings) > 0:
        rating_counts = df_with_ratings["rating"].value_counts().head(20)
        st.write("**Top 20 current ratings:**")
        st.dataframe(rating_counts.reset_index().rename(columns={"index": "Rating", "rating": "Count"}))
    
    # Process translations
    with st.spinner("Processing rating translations..."):
        df_updated, stats, unmatched = update_ratings_to_fitch(df_db)
    
    # Show summary statistics
    st.subheader("Translation Summary")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Moody's Converted", stats["moodys_converted"])
    with col2:
        st.metric("S&P Converted", stats["sp_converted"])
    with col3:
        st.metric("Fitch Normalized", stats["fitch_normalized"])
    with col4:
        st.metric("Total Updated", stats["total_updated"])
    
    col5, col6 = st.columns(2)
    with col5:
        st.metric("Unchanged", stats["unchanged"])
    with col6:
        st.metric("Unmatched", stats["unmatched"])
    
    # Show unmatched ratings warning
    if unmatched:
        st.warning(
            f"**{len(unmatched)}** unique rating(s) could not be matched and will be left unchanged:\n"
            + "\n".join([f"- {rating}" for rating in unmatched[:20]])
            + (f"\n... and {len(unmatched) - 20} more" if len(unmatched) > 20 else "")
        )
    
    if stats["total_updated"] == 0:
        st.info("No ratings need to be updated. All ratings are already in Fitch format or couldn't be matched.")
        return
    
    # Show preview of records that will be updated
    st.subheader("Preview of Records to be Updated")
    
    # Find rows that were updated (rating changed from original)
    df_db_rating = df_db["rating"].fillna("").astype(str)
    df_updated_rating = df_updated["rating"].fillna("").astype(str)
    changed_mask = df_db_rating != df_updated_rating
    
    df_preview = df_updated[changed_mask].head(50)
    if not df_preview.empty:
        preview_cols = ["rowid", "security_name", "parsed_cetip_code", "rating"]
        available_cols = [col for col in preview_cols if col in df_preview.columns]
        
        # Add original rating for comparison
        df_preview_display = df_preview[available_cols].copy()
        df_preview_display["original_rating"] = df_db.loc[df_preview.index, "rating"].values
        df_preview_display["new_rating"] = df_preview["rating"].values
        df_preview_display = df_preview_display[["rowid", "security_name", "parsed_cetip_code", "original_rating", "new_rating"]]
        
        st.dataframe(df_preview_display)
        if len(df_preview) < len(df_updated[changed_mask]):
            st.info(f"Showing first 50 of {len(df_updated[changed_mask])} records to be updated.")
    else:
        st.info("No preview available.")
    
    # Update button
    if st.button("Apply Fitch Translations to Database", type="primary"):
        with st.spinner("Saving updates to database..."):
            success = save_fitch_ratings(df_updated, df_db)
            if success:
                st.success(
                    f"Successfully updated **{stats['total_updated']}** ratings in the database!"
                )
                st.balloons()
            else:
                st.error("Failed to save updates to database.")


if __name__ == "__main__":
    main()

