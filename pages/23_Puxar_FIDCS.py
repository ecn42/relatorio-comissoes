import streamlit as st
import requests
import pandas as pd
import sqlite3
import zipfile
import io
import re
from pathlib import Path
from datetime import datetime

# Constants
BASE_URL = "https://dados.cvm.gov.br/dados/FIDC/DOC/INF_MENSAL/DADOS/"
DB_PATH = Path("databases/historico_fidcs.db")


def ensure_db_directory():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def get_available_files():
    response = requests.get(BASE_URL)
    response.raise_for_status()

    # More flexible pattern to catch different naming conventions
    patterns = [
        r'href="(inf_mensal_fidc_(\d{6})\.zip)"',
        r'href="(inf_mensal_fidc_(\d{4}_\d{2})\.zip)"',  # YYYY_MM format
        r'href="(inf_mensal_fidc_(\d{4}-\d{2})\.zip)"',  # YYYY-MM format
    ]

    files = []
    seen = set()

    for pattern in patterns:
        matches = re.findall(pattern, response.text, re.IGNORECASE)
        for filename, date_part in matches:
            if filename.lower() in seen:
                continue
            seen.add(filename.lower())

            # Normalize date_part to YYYYMM
            yyyymm = re.sub(r"[_-]", "", date_part)

            if len(yyyymm) == 6 and yyyymm.isdigit():
                year = int(yyyymm[:4])
                month = int(yyyymm[4:])
                if 1 <= month <= 12 and 2000 <= year <= 2100:
                    files.append({
                        "filename": filename,
                        "yyyymm": yyyymm,
                        "year": year,
                        "month": month,
                        "date": datetime(year, month, 1),
                    })

    return sorted(files, key=lambda x: x["yyyymm"])


def find_tab_x_6_file(file_list: list[str]) -> str | None:
    """Find the tab_X_6 CSV file using multiple patterns."""
    patterns = [
        r"inf_mensal_fidc_tab_X_6_\d+\.csv",
        r"inf_mensal_fidc_tab_x_6_\d+\.csv",
        r".*tab.*X.*6.*\.csv",
        r".*tab.*x.*6.*\.csv",
        r".*tab.*6.*\.csv",
    ]

    for pattern in patterns:
        for name in file_list:
            if re.match(pattern, name, re.IGNORECASE):
                return name

    for name in file_list:
        if name.endswith(".csv") and "6" in name:
            return name

    return None


def download_and_extract_csv(
    filename: str,
) -> tuple[pd.DataFrame | None, list[str], str | None]:
    """Download a ZIP file and extract the tab_X_6 CSV."""
    url = BASE_URL + filename
    response = requests.get(url, timeout=60)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        all_files = zf.namelist()
        target_csv = find_tab_x_6_file(all_files)

        if target_csv:
            with zf.open(target_csv) as f:
                df = pd.read_csv(f, sep=";", encoding="latin-1")

                # Unify CNPJ column: 2024+ uses CNPJ_FUNDO_CLASSE, rename to old standard
                if "CNPJ_FUNDO_CLASSE" in df.columns:
                    df.rename(columns={"CNPJ_FUNDO_CLASSE": "CNPJ_FUNDO"}, inplace=True)

                return df, all_files, target_csv

        return None, all_files, None


def init_database():
    ensure_db_directory()
    conn = sqlite3.connect(DB_PATH)
    return conn


def get_existing_columns(conn: sqlite3.Connection, table_name: str) -> set[str]:
    try:
        cursor = conn.execute(f"PRAGMA table_info({table_name})")
        return {row[1] for row in cursor.fetchall()}
    except sqlite3.OperationalError:
        return set()


def add_missing_columns(
    conn: sqlite3.Connection, table_name: str, df: pd.DataFrame
):
    existing_cols = get_existing_columns(conn, table_name)

    if not existing_cols:
        return

    for col in df.columns:
        if col not in existing_cols:
            dtype = df[col].dtype
            if pd.api.types.is_integer_dtype(dtype):
                sql_type = "INTEGER"
            elif pd.api.types.is_float_dtype(dtype):
                sql_type = "REAL"
            else:
                sql_type = "TEXT"

            conn.execute(f'ALTER TABLE {table_name} ADD COLUMN "{col}" {sql_type}')
            conn.commit()


def get_existing_periods(conn: sqlite3.Connection) -> set[str]:
    """Get existing periods in YYYYMM format."""
    try:
        cursor = conn.execute("SELECT DISTINCT DT_COMPTC FROM fidc_tab_x_6")
        periods = set()
        for row in cursor.fetchall():
            if row[0]:
                # Normalize to YYYYMM format
                dt_str = str(row[0])
                # Handle formats: "2024-01-01", "2024-01", "202401"
                clean = re.sub(r"[^0-9]", "", dt_str)[:6]
                if len(clean) == 6:
                    periods.add(clean)
        return periods
    except sqlite3.OperationalError:
        return set()


def delete_period_data(conn: sqlite3.Connection, yyyymm: str):
    """Delete existing data for a given period to prevent duplicates."""
    try:
        # Build patterns to match different date formats
        year = yyyymm[:4]
        month = yyyymm[4:6]

        # Match various formats: "2024-01-01", "2024-01", "202401", etc.
        patterns = [
            f"{year}-{month}%",  # 2024-01-01 or 2024-01
            f"{year}/{month}%",  # 2024/01/01
            f"{yyyymm}%",  # 202401
        ]

        for pattern in patterns:
            conn.execute(
                "DELETE FROM fidc_tab_x_6 WHERE DT_COMPTC LIKE ?", (pattern,)
            )

        conn.commit()
    except sqlite3.OperationalError:
        pass  # Table doesn't exist yet


def get_all_cnpjs(conn: sqlite3.Connection) -> list[str]:
    try:
        cursor = conn.execute(
            "SELECT DISTINCT CNPJ_FUNDO FROM fidc_tab_x_6 ORDER BY CNPJ_FUNDO"
        )
        return [row[0] for row in cursor.fetchall() if row[0]]
    except sqlite3.OperationalError:
        return []


def get_filtered_data(
    conn: sqlite3.Connection,
    cnpjs: list[str] | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    query = "SELECT * FROM fidc_tab_x_6"

    if cnpjs:
        placeholders = ",".join(["?" for _ in cnpjs])
        query += f" WHERE CNPJ_FUNDO IN ({placeholders})"

    query += " ORDER BY DT_COMPTC DESC, CNPJ_FUNDO"

    if limit:
        query += f" LIMIT {limit}"

    if cnpjs:
        return pd.read_sql(query, conn, params=cnpjs)
    return pd.read_sql(query, conn)


def save_to_database(
    conn: sqlite3.Connection, df: pd.DataFrame, yyyymm: str, replace: bool = True
):
    """Save DataFrame to database, optionally replacing existing period data."""
    table_name = "fidc_tab_x_6"

    if replace:
        delete_period_data(conn, yyyymm)

    add_missing_columns(conn, table_name, df)
    df.to_sql(table_name, conn, if_exists="append", index=False)
    conn.commit()


def main():
    st.set_page_config(page_title="CVM FIDC Data Loader", layout="wide")
    st.title("📊 CVM FIDC Monthly Data Loader")

    tab_download, tab_query, tab_debug = st.tabs(
        ["📥 Download Data", "🔍 Query & Export", "🔧 Debug"]
    )

    # ==================== DEBUG TAB ====================
    with tab_debug:
        st.subheader("🔧 Inspect ZIP Contents & Raw HTML")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("🌐 Show Raw File List from CVM"):
                with st.spinner("Fetching..."):
                    response = requests.get(BASE_URL, timeout=30)
                    # Show all href matches
                    all_hrefs = re.findall(r'href="([^"]+)"', response.text)
                    zip_files = [h for h in all_hrefs if h.endswith(".zip")]

                    st.markdown(f"**Found {len(zip_files)} ZIP files:**")
                    for zf in sorted(zip_files):
                        st.code(zf)

        with col2:
            if st.button("📋 Show Parsed Files"):
                with st.spinner("Parsing..."):
                    try:
                        files = get_available_files()
                        st.success(f"Parsed {len(files)} files")
                        df = pd.DataFrame(files)
                        st.dataframe(df, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error: {e}")

        st.markdown("---")
        st.subheader("🔍 Inspect Individual ZIP")

        with st.spinner("Fetching file list..."):
            try:
                available_files = get_available_files()
            except Exception as e:
                st.error(f"Error: {e}")
                available_files = []

        if available_files:
            selected_file = st.selectbox(
                "Select a ZIP file to inspect",
                [f["filename"] for f in available_files],
                index=len(available_files) - 1,
            )

            if st.button("🔍 Inspect ZIP Contents", key="inspect"):
                with st.spinner(f"Downloading {selected_file}..."):
                    url = BASE_URL + selected_file
                    response = requests.get(url, timeout=60)

                    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
                        files_in_zip = zf.namelist()

                        st.markdown(f"**{len(files_in_zip)} files inside:**")

                        for f in files_in_zip:
                            if f.endswith(".csv"):
                                if "6" in f.lower():
                                    st.success(f"✅ {f}")
                                else:
                                    st.info(f"📄 {f}")
                            else:
                                st.text(f)

                        target = find_tab_x_6_file(files_in_zip)
                        st.markdown("---")
                        if target:
                            st.success(f"**Would select:** `{target}`")
                        else:
                            st.error("**No matching file found!**")

    # ==================== DOWNLOAD TAB ====================
    with tab_download:
        st.markdown(
            "Downloads `inf_mensal_fidc_tab_X_6` from CVM and stores in SQLite."
        )

        with st.spinner("Fetching available files from CVM..."):
            try:
                available_files = get_available_files()
            except Exception as e:
                st.error(f"Error fetching file list: {e}")
                return

        if not available_files:
            st.warning("No files found. Check the Debug tab for raw file list.")
            return

        st.success(
            f"Found {len(available_files)} files "
            f"({available_files[0]['yyyymm']} to {available_files[-1]['yyyymm']})"
        )

        # Date range selection
        col1, col2 = st.columns(2)
        min_date = available_files[0]["date"]
        max_date = available_files[-1]["date"]

        with col1:
            start_date = st.date_input(
                "Start Date",
                value=max_date,
                min_value=min_date,
                max_value=max_date,
            )
        with col2:
            end_date = st.date_input(
                "End Date",
                value=max_date,
                min_value=min_date,
                max_value=max_date,
            )

        filtered_files = [
            f
            for f in available_files
            if start_date <= f["date"].date() <= end_date
        ]

        st.info(f"Selected {len(filtered_files)} file(s) to process.")

        # Database status
        conn = init_database()
        existing_periods = get_existing_periods(conn)
        existing_cols = get_existing_columns(conn, "fidc_tab_x_6")

        if existing_periods:
            st.markdown(f"**Database has {len(existing_periods)} periods loaded.**")
            with st.expander("Show loaded periods"):
                st.write(sorted(existing_periods))

        if existing_cols:
            with st.expander(f"📋 Current table columns ({len(existing_cols)})"):
                st.write(sorted(existing_cols))

        # Options
        col1, col2 = st.columns(2)
        with col1:
            skip_existing = st.checkbox("Skip already loaded periods", value=True)
        with col2:
            replace_existing = st.checkbox(
                "Replace if period exists (prevents duplicates)",
                value=True,
                help="Deletes existing data for a period before inserting new data",
            )

        if skip_existing:
            files_to_process = [
                f for f in filtered_files if f["yyyymm"] not in existing_periods
            ]
            skipped = len(filtered_files) - len(files_to_process)
            if skipped:
                st.info(f"Skipping {skipped} already loaded period(s).")
        else:
            files_to_process = filtered_files

        # Download button
        if st.button("🚀 Download and Load Data", type="primary"):
            if not files_to_process:
                st.warning("No new files to process.")
            else:
                progress_bar = st.progress(0)
                status_text = st.empty()
                results = []

                for i, file_info in enumerate(files_to_process):
                    status_text.text(f"Processing {file_info['filename']}...")

                    try:
                        df, zip_contents, csv_used = download_and_extract_csv(
                            file_info["filename"]
                        )
                        if df is not None:
                            save_to_database(
                                conn, df, file_info["yyyymm"], replace=replace_existing
                            )
                            results.append({
                                "File": file_info["filename"],
                                "Period": file_info["yyyymm"],
                                "Status": "✅ Success",
                                "CSV Found": csv_used,
                                "Rows": len(df),
                            })
                        else:
                            results.append({
                                "File": file_info["filename"],
                                "Period": file_info["yyyymm"],
                                "Status": "⚠️ No matching CSV",
                                "CSV Found": "None",
                                "Rows": 0,
                                "ZIP Contents": ", ".join(zip_contents[:5]),
                            })
                    except Exception as e:
                        results.append({
                            "File": file_info["filename"],
                            "Period": file_info["yyyymm"],
                            "Status": f"❌ {str(e)[:50]}",
                            "CSV Found": "Error",
                            "Rows": 0,
                        })

                    progress_bar.progress((i + 1) / len(files_to_process))

                status_text.text("Done!")
                st.dataframe(pd.DataFrame(results), use_container_width=True)

                total_rows = sum(r["Rows"] for r in results)
                success_count = sum(1 for r in results if "✅" in r["Status"])
                st.success(f"Loaded {total_rows:,} rows from {success_count} files.")

        conn.close()

    # ==================== QUERY TAB ====================
    with tab_query:
        st.markdown("Filter data by CNPJ and export to CSV.")

        conn = init_database()

        try:
            count = conn.execute("SELECT COUNT(*) FROM fidc_tab_x_6").fetchone()[0]
        except sqlite3.OperationalError:
            st.warning("Database is empty. Download some data first.")
            conn.close()
            return

        st.metric("Total Rows in Database", f"{count:,}")

        all_cnpjs = get_all_cnpjs(conn)

        if not all_cnpjs:
            st.warning("No data in database yet.")
            conn.close()
            return

        st.markdown(f"**{len(all_cnpjs)} unique CNPJs available**")

        st.subheader("🔍 Filter by CNPJ")

        filter_method = st.radio(
            "Filter method",
            ["Select from list", "Paste CNPJs"],
            horizontal=True,
        )

        selected_cnpjs = []

        if filter_method == "Select from list":
            selected_cnpjs = st.multiselect(
                "Select CNPJs",
                options=all_cnpjs,
                placeholder="Choose CNPJs to filter...",
            )
        else:
            cnpj_input = st.text_area(
                "Paste CNPJs (one per line or comma-separated)",
                placeholder="00.000.000/0001-00\n11.111.111/0001-11",
                height=150,
            )
            if cnpj_input:
                raw_cnpjs = re.split(r"[,\n]", cnpj_input)
                selected_cnpjs = [c.strip() for c in raw_cnpjs if c.strip()]

                found = [c for c in selected_cnpjs if c in all_cnpjs]
                not_found = [c for c in selected_cnpjs if c not in all_cnpjs]

                if found:
                    st.success(f"✅ Found {len(found)} CNPJs in database")
                if not_found:
                    with st.expander(f"⚠️ {len(not_found)} CNPJs not found"):
                        st.write(not_found)

                selected_cnpjs = found

        st.subheader("📤 Preview & Export")

        if selected_cnpjs:
            st.info(f"Filtering by {len(selected_cnpjs)} CNPJ(s)")

            preview_df = get_filtered_data(conn, selected_cnpjs, limit=100)
            placeholders = ",".join(["?" for _ in selected_cnpjs])
            total_filtered = conn.execute(
                f"SELECT COUNT(*) FROM fidc_tab_x_6 WHERE CNPJ_FUNDO IN ({placeholders})",
                selected_cnpjs,
            ).fetchone()[0]

            st.markdown(f"**Total rows matching filter: {total_filtered:,}**")
            st.markdown("**Preview (first 100 rows):**")
            st.dataframe(preview_df, use_container_width=True)

            if st.button("📥 Prepare Filtered Export", type="primary"):
                with st.spinner("Fetching all filtered data..."):
                    full_df = get_filtered_data(conn, selected_cnpjs)

                csv_data = full_df.to_csv(index=False, sep=";", encoding="utf-8-sig")
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                st.download_button(
                    label=f"⬇️ Download CSV ({len(full_df):,} rows)",
                    data=csv_data,
                    file_name=f"fidc_export_{timestamp}.csv",
                    mime="text/csv",
                )
        
        st.markdown("---")
        st.subheader("📦 Full Database Export")
        
        st.info(f"Export all {count:,} rows from the database.")
        
        if st.button("📥 Prepare Full Export (All Data)"):
            with st.spinner("Fetching all data..."):
                full_df = get_filtered_data(conn)

            csv_data = full_df.to_csv(index=False, sep=";", encoding="utf-8-sig")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            st.download_button(
                label=f"⬇️ Download All Data CSV ({len(full_df):,} rows)",
                data=csv_data,
                file_name=f"fidc_export_all_{timestamp}.csv",
                mime="text/csv",
            )

        conn.close()


if __name__ == "__main__":
    main()