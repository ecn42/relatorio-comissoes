import os
import sqlite3
from datetime import date
from typing import Optional

import pandas as pd
import streamlit as st

# -------------------------------
# Streamlit page config
# -------------------------------
st.set_page_config(
    page_title="Custom Assets Manual Parser",
    layout="wide",
)

# -------------------------------
# Simple Authentication
# -------------------------------
if not st.session_state.get("authenticated", False):
    st.warning("Please enter the password on the Home page first.")
    st.write("Status: Não Autenticado")
    st.stop()

st.write("Autenticado")

# -------------------------------
# Database path
# -------------------------------
DB_PATH = "gorila_positions.db"
TARGET_TABLE = "pmv_plus_gorila"

# -------------------------------
# Load CUSTOM assets from DB
# -------------------------------


def get_available_dates() -> list[str]:
    """Get list of unique reference_date values from the table."""
    if not os.path.exists(DB_PATH):
        return []

    try:
        conn = sqlite3.connect(DB_PATH)
        query = (
            f"SELECT DISTINCT reference_date "
            f"FROM {TARGET_TABLE} "
            f"ORDER BY reference_date DESC"
        )
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df["reference_date"].tolist() if not df.empty else []
    except Exception as e:
        st.error(f"Error fetching dates: {e}")
        return []


def find_matching_records(
    security_id: int,
    current_date: str,
    columns_to_check: list[str],
) -> pd.DataFrame:
    """Find records with same security_id in other dates."""
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()

    try:
        conn = sqlite3.connect(DB_PATH)
        query = f"""
        SELECT rowid AS rowid, *
        FROM {TARGET_TABLE}
        WHERE security_id = ?
          AND reference_date != ?
        ORDER BY reference_date DESC
        """
        df = pd.read_sql_query(query, conn, params=(security_id, current_date))
        conn.close()
        return df
    except Exception as e:
        st.error(f"Error finding matching records: {e}")
        return pd.DataFrame()


def load_custom_assets(reference_date: Optional[str] = None) -> pd.DataFrame:
    """Load all CUSTOM security_type records from gorila_positions.db.

    Returns:
        DataFrame with at least a 'rowid' column for safe updates.
    """
    if not os.path.exists(DB_PATH):
        st.error(f"Database not found: {DB_PATH}")
        return pd.DataFrame()

    try:
        conn = sqlite3.connect(DB_PATH)

        if reference_date:
            query = f"""
            SELECT
                rowid AS rowid,
                *
            FROM {TARGET_TABLE}
            WHERE security_type = 'CUSTOM'
              AND reference_date = ?
            ORDER BY position_id
            """
            df = pd.read_sql_query(query, conn, params=(reference_date,))
        else:
            query = f"""
            SELECT
                rowid AS rowid,
                *
            FROM {TARGET_TABLE}
            WHERE security_type = 'CUSTOM'
            ORDER BY position_id
            """
            df = pd.read_sql_query(query, conn)

        conn.close()
        return df
    except Exception as e:
        st.error(f"Error loading database: {e}")
        return pd.DataFrame()


def save_custom_assets(df: pd.DataFrame) -> bool:
    """Save edited CUSTOM assets back to database.

    Uses SQLite rowid as the unique, non-editable key for each row.
    """
    if not os.path.exists(DB_PATH):
        st.error(f"Database not found: {DB_PATH}")
        return False

    if "rowid" not in df.columns:
        st.error(
            "Internal error: 'rowid' column missing in DataFrame. "
            "Reload the data before saving."
        )
        return False

    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # Columns that can be updated (exclude the row key and PK)
        updatable_columns = [
            col
            for col in df.columns
            if col not in ("rowid", "position_id")
        ]

        for _, row in df.iterrows():
            rowid_val = row["rowid"]

            # Defensive: skip rows without a valid rowid
            if pd.isna(rowid_val):
                continue

            set_clause = ", ".join(
                [f"{col} = ?" for col in updatable_columns]
            )
            values = [row[col] for col in updatable_columns]
            values.append(int(rowid_val))

            query = (
                f"UPDATE {TARGET_TABLE} "
                f"SET {set_clause} "
                f"WHERE rowid = ?"
            )
            cursor.execute(query, values)

        conn.commit()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Error saving to database: {e}")
        return False


# -------------------------------
# Main UI
# -------------------------------

st.header("Custom Assets Manual Parser")
st.caption(
    "Edit CUSTOM security_type records from the Gorila database. "
    "Changes will be saved back to gorila_positions.db."
)

st.write(f"Using table: **{TARGET_TABLE}**")

# Get available dates
available_dates = get_available_dates()

if not available_dates:
    st.info("No reference dates found in database")
    st.stop()

# Date selector
selected_date = st.selectbox(
    "Select Reference Date",
    options=available_dates,
    key="date_selector",
)

# Load button
if st.button("📂 Load CUSTOM Assets from DB", key="load_btn"):
    df_custom = load_custom_assets(selected_date)
    if not df_custom.empty:
        # Keep both original (for diff) and working copy
        st.session_state["df_custom_assets_original"] = df_custom.copy()
        st.session_state["df_custom_assets"] = df_custom
        st.session_state["selected_date"] = selected_date
        st.success(
            f"Loaded {len(df_custom)} CUSTOM asset(s) for {selected_date}"
        )
    else:
        st.info(f"No CUSTOM assets found for {selected_date}")

# Display and edit
if st.session_state.get("df_custom_assets") is not None:
    df_custom = st.session_state["df_custom_assets"]
    selected_date = st.session_state.get("selected_date")

    if df_custom.empty:
        st.info("No records to display")
    else:
        st.subheader(f"Editing {len(df_custom)} CUSTOM Asset(s)")

        # Show columns info
        st.caption(f"Columns: {', '.join(df_custom.columns.tolist())}")

        # Editable data editor
        edited_df = st.data_editor(
            df_custom,
            use_container_width=True,
            height=400,
            key="custom_assets_editor",
            num_rows="fixed",
            column_config={
                # Internal key: never editable
                "rowid": st.column_config.Column(
                    "rowid",
                    help="Internal SQLite rowid (do not edit)",
                    disabled=True,
                ),
                # Treat position_id as a stable identifier
                "position_id": st.column_config.Column(
                    "position_id",
                    help="Position identifier (do not edit)",
                    disabled=True,
                ),
            },
        )

        # Save button
        col1, col2 = st.columns([1, 3])

        with col1:
            if st.button(
                "💾 Save Changes to DB",
                key="save_btn",
                type="primary",
            ):
                if save_custom_assets(edited_df):
                    st.success("✓ Changes saved successfully!")
                    # Update working copy, but keep 'original' for diff
                    st.session_state["df_custom_assets"] = edited_df
                else:
                    st.error("✗ Failed to save changes")

        with col2:
            st.caption("Click 'Save Changes to DB' to persist your edits")

        # Download CSV
        csv_data = edited_df.to_csv(index=False)
        st.download_button(
            label="📥 Download as CSV",
            data=csv_data,
            file_name=f"custom_assets_{selected_date}.csv",
            mime="text/csv",
        )

        # Section for applying changes to other dates
        st.divider()
        st.subheader("Apply Changes to Other Dates")
        st.caption(
            "Find same security_id in other dates and apply the changes"
        )

        # Use the original DF from when it was loaded for diff
        original_df = st.session_state.get("df_custom_assets_original")

        if original_df is None or original_df.empty:
            st.info(
                "Original data not found in session. "
                "Reload the data to enable change tracking."
            )
        else:
            original_df = original_df.reset_index(drop=True)
            edited_df_reset = edited_df.reset_index(drop=True)

            changes: dict[int, dict[str, object]] = {}

            for idx in range(len(edited_df_reset)):
                row_changes: dict[str, object] = {}
                for col in edited_df_reset.columns:
                    # We never propagate rowid changes and we do not expect
                    # position_id edits, so skip them in the logical diff.
                    if col in ("rowid", "position_id"):
                        continue

                    orig_val = original_df.loc[idx, col]
                    new_val = edited_df_reset.loc[idx, col]

                    orig_is_nan = pd.isna(orig_val)
                    new_is_nan = pd.isna(new_val)

                    if orig_is_nan and new_is_nan:
                        continue
                    elif orig_is_nan != new_is_nan:
                        row_changes[col] = new_val
                    elif not (orig_is_nan or new_is_nan) and str(
                        orig_val
                    ) != str(new_val):
                        row_changes[col] = new_val

                if row_changes:
                    changes[idx] = row_changes

            if changes:
                st.write(f"Found {len(changes)} row(s) with changes")

                # Show summary of changes
                with st.expander("View changes summary"):
                    for idx, row_changes in changes.items():
                        sec_id = original_df.loc[idx, "security_id"]
                        st.write(f"**Row {idx} (security_id: {sec_id})**")
                        for col, new_val in row_changes.items():
                            old_val = original_df.loc[idx, col]
                            st.write(f"  - {col}: `{old_val}` → `{new_val}`")

                # Option to apply to other dates
                if st.button(
                    "🔍 Find & Apply Changes to Other Dates",
                    key="apply_other_dates_btn",
                ):
                    total_applied = 0

                    for idx, row_changes in changes.items():
                        sec_id = original_df.loc[idx, "security_id"]

                        # Find matching records in other dates
                        matching_df = find_matching_records(
                            sec_id,
                            selected_date,
                            list(row_changes.keys()),
                        )

                        if not matching_df.empty:
                            st.write(
                                "Found "
                                f"{len(matching_df)} matching record(s) "
                                f"for security_id {sec_id}"
                            )

                            try:
                                conn = sqlite3.connect(DB_PATH)
                                cursor = conn.cursor()

                                for _, match_row in (
                                    matching_df.iterrows()
                                ):
                                    set_clause = ", ".join(
                                        [
                                            f"{col} = ?"
                                            for col in row_changes.keys()
                                        ]
                                    )
                                    values = list(row_changes.values())
                                    values.append(
                                        match_row["position_id"]
                                    )

                                    query = (
                                        f"UPDATE {TARGET_TABLE} "
                                        f"SET {set_clause} "
                                        f"WHERE position_id = ?"
                                    )
                                    cursor.execute(query, values)
                                    total_applied += 1

                                conn.commit()
                                conn.close()
                            except Exception as e:
                                st.error(
                                    f"Error applying changes: {e}"
                                )

                    if total_applied > 0:
                        st.success(
                            "✓ Applied changes to "
                            f"{total_applied} record(s) in other dates!"
                        )
                    else:
                        st.info(
                            "No matching records found in other dates"
                        )
            else:
                st.info(
                    "No changes detected. Edit the records and save "
                    "to see options here."
        )
else:
    st.info("Click 'Load CUSTOM Assets from DB' to get started")