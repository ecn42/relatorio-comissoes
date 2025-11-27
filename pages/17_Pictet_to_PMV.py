import re
import sqlite3
from typing import List, Tuple, Optional

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype
import streamlit as st
import yfinance as yf


st.set_page_config(page_title="Pictet → PMV Appender (DB)", layout="wide")

DB_PATH = "gorila_positions.db"


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """
    Export to CSV with European format; protect id-like columns as strings.
    """
    df_copy = df.copy()

    # Protect id-like columns (avoid scientific notation)
    protect_re = re.compile(r"(taxid|cnpj|cpf|codigo|cetip|isin|id$)", re.I)
    for col in df_copy.columns:
        if protect_re.search(col):
            df_copy[col] = df_copy[col].apply(
                lambda x: "" if pd.isna(x) else str(x)
            )

    numeric_cols = df_copy.select_dtypes(include=["number"]).columns
    for col in numeric_cols:
        df_copy[col] = df_copy[col].apply(
            lambda x: str(x).replace(".", ",") if pd.notna(x) else "",
            convert_dtype=False,
        )

    csv_str = df_copy.to_csv(index=False, sep=";", encoding="utf-8")
    return csv_str.encode("utf-8-sig")


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


def read_pmv_from_db(db_path: str = DB_PATH) -> pd.DataFrame:
    """
    Read the 'pmv' table from gorila_positions.db.
    """
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql("SELECT * FROM pmv", conn)


def read_table_if_exists(
    table_name: str, db_path: str = DB_PATH
) -> Optional[pd.DataFrame]:
    """
    Read a table from SQLite if it exists; otherwise return None.
    """
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name=?",
            (table_name,),
        )
        if cur.fetchone() is None:
            return None
        return pd.read_sql(f"SELECT * FROM {table_name}", conn)


def prepare_for_sqlite(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert datetime-like and Timestamp values to strings
    so that SQLite can store them.
    """
    df_sql = df.copy()
    for col in df_sql.columns:
        series = df_sql[col]

        # Proper datetime dtype
        if is_datetime64_any_dtype(series):
            df_sql[col] = series.dt.strftime("%Y-%m-%d %H:%M:%S")
        else:
            # Object columns that may contain Timestamp instances
            if series.map(lambda v: isinstance(v, pd.Timestamp)).any():
                df_sql[col] = series.astype(str)

    return df_sql


def write_pmv_plus_gorila(
    df: pd.DataFrame, db_path: str = DB_PATH
) -> None:
    """
    Write the combined DataFrame to 'pmv_plus_gorila' table
    (overwrite if exists), making it SQLite-friendly first.
    """
    df_sql = prepare_for_sqlite(df)

    with sqlite3.connect(db_path) as conn:
        df_sql.to_sql(
            "pmv_plus_gorila",
            conn,
            if_exists="replace",
            index=False,
        )


def fetch_usd_brl_rate() -> float:
    """
    Fetch the latest USD/BRL close price using yfinance.
    """
    ticker = yf.Ticker("USDBRL=X")
    hist = ticker.history(period="1d")
    if hist.empty or "Close" not in hist.columns:
        raise RuntimeError("No USD/BRL price data returned from yfinance.")
    return float(hist["Close"].iloc[-1])


# PMV base column order (must stay in this order)
PMV_BASE_ORDER: List[str] = [
    "portfolio_id",
    "position_id",
    "reference_date",
    "security_id",
    "security_name",
    "security_type",
    "asset_class",
    "isin",
    "quantity",
    "currency",
    "price_amount",
    "price_currency",
    "market_value_amount",
    "market_value_currency",
    "raw",
    "issuer_from_raw",
    "issuer_name_from_json",
    "parsed_bond_type",
    "parsed_company_name",
    "parsed_maturity",
    "parsed_maturity_date",
    "parsed_cetip_code",
    "indexer",
]

# Pictet column → list of PMV target columns
# "new:" means create a new column
MAPPING: List[Tuple[str, List[str]]] = [
    ("Account nr.", ["portfolio_id"]),
    ("Valuation date", ["reference_date"]),
    ("Asset class", ["security_type"]),
    ("Description", ["security_name"]),
    ("Quantity", ["quantity"]),
    (
        "Position currency",
        ["currency", "price_currency", "market_value_currency"],
    ),
    ("Valuation in position currency", ["market_value_amount"]),
    ("ISIN", ["isin"]),
    ("Financial instrument type", ["asset_class"]),
    ("Country risk", ["new:country_risk"]),
    ("Issuer", ["parsed_company_name"]),
    ("Telekurs Sector", ["new:sector"]),
    ("Bond Issuer Type", ["parsed_bond_type"]),
    ("Maturity", ["parsed_maturity", "parsed_maturity_date"]),
    ("Composite rating", ["new:rating"]),
]


def extract_new_target_columns(
    mapping: List[Tuple[str, List[str]]],
) -> List[str]:
    """
    From the mapping, extract all targets that are flagged as new:XXX.
    """
    new_cols: List[str] = []
    for _, targets in mapping:
        for t in targets:
            if t.startswith("new:"):
                name = t[4:]
                if name not in new_cols:
                    new_cols.append(name)
    return new_cols


def build_pmv_from_pictet(
    df_pictet: pd.DataFrame,
    mapping: List[Tuple[str, List[str]]],
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Create a PMV-style DataFrame from the Pictet sheet.

    Each row in df_pictet becomes a new row in the resulting DataFrame.
    Columns are created according to the mapping.
    """
    pictet_as_pmv = pd.DataFrame(index=df_pictet.index)
    missing_sources: List[str] = []

    for pictet_col, targets in mapping:
        if pictet_col not in df_pictet.columns:
            missing_sources.append(pictet_col)
            continue

        for target in targets:
            target_name = target[4:] if target.startswith("new:") else target
            pictet_as_pmv[target_name] = df_pictet[pictet_col]

    return pictet_as_pmv, missing_sources


def reorder_pmv_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reorder columns so that the PMV base columns come first in the
    specified order, and any other columns are appended at the end in
    their existing order.
    """
    existing_cols = list(df.columns)
    base_cols = [c for c in PMV_BASE_ORDER if c in existing_cols]
    other_cols = [c for c in existing_cols if c not in PMV_BASE_ORDER]
    final_cols = base_cols + other_cols
    return df.reindex(columns=final_cols)


def main() -> None:
    st.title("Pictet → PMV Appender (gorila_positions.db)")

    st.markdown(
        "This app will:\n"
        "1. Load the **pmv** table from `gorila_positions.db`.\n"
        "2. Load a Pictet sheet (Excel/CSV).\n"
        "3. Match **reference_date** (PMV) with **Valuation date** (Pictet).\n"
        "4. For matched dates, append Pictet positions to PMV positions.\n"
        "5. Fill `country_risk = \"Brazil\"` for original PMV rows.\n"
        "6. Convert `market_value_amount` from USD to BRL where "
        "`currency == 'USD'`, storing the original USD amount in "
        "`market_value_amount_original_currency`.\n"
        "7. Save/merge the result into **pmv_plus_gorila** in the DB.\n"
        "   If those dates already exist there, you choose whether to "
        "do nothing or overwrite them.\n"
        "8. If `asset_class == 'Bond'`, set `indexer = 'PRÉ'`."
    )

    # --- Load PMV table from DB ---
    try:
        df_pmv = read_pmv_from_db()
    except Exception as e:
        st.error(
            "Could not read 'pmv' table from gorila_positions.db. "
            f"Error: {e}"
        )
        return

    if "reference_date" not in df_pmv.columns:
        st.error("The 'pmv' table must contain a 'reference_date' column.")
        return

    # Normalize reference_date to 'YYYY-MM-DD' strings
    df_pmv["reference_date"] = pd.to_datetime(
        df_pmv["reference_date"], errors="coerce"
    ).dt.date.astype(str)

    # Ensure all PMV base columns exist (if missing, create as NA)
    for col in PMV_BASE_ORDER:
        if col not in df_pmv.columns:
            df_pmv[col] = pd.NA

    # Put PMV columns in the required base order (others follow)
    df_pmv = reorder_pmv_columns(df_pmv)

    st.markdown(
        f"Loaded **pmv** table from DB with **{len(df_pmv)}** rows and "
        f"**{len(df_pmv.columns)}** columns."
    )

    # --- Upload Pictet sheet ---
    pictet_file = st.file_uploader(
        "Upload Pictet sheet (CSV or Excel)",
        type=["csv", "xls", "xlsx", "xlsm", "xlsb", "ods"],
        key="pictet_file",
    )

    if pictet_file is None:
        st.info("Upload a Pictet sheet to proceed.")
        return

    df_pictet = load_sheet(pictet_file)
    if df_pictet is None:
        st.error("Could not read the uploaded Pictet file.")
        return

    if "Valuation date" not in df_pictet.columns:
        st.error("Pictet sheet must contain a 'Valuation date' column.")
        return

    # Normalize Valuation date to 'YYYY-MM-DD' strings
    df_pictet["Valuation date"] = pd.to_datetime(
        df_pictet["Valuation date"], errors="coerce"
    ).dt.date.astype(str)

    pictet_dates = sorted(
        d for d in df_pictet["Valuation date"].dropna().unique()
    )
    pmv_dates = set(df_pmv["reference_date"].dropna().unique())
    matched_dates = sorted(set(pictet_dates) & pmv_dates)

    if not matched_dates:
        st.error(
            "No matching dates between Pictet 'Valuation date' and "
            "PMV 'reference_date'."
        )
        return

    st.markdown(
        "Matched dates between Pictet and PMV (used for appending):\n"
        f"- {', '.join(matched_dates)}"
    )

    # --- Check existing pmv_plus_gorila for duplicate dates ---
    df_existing_plus = read_table_if_exists("pmv_plus_gorila")
    duplicate_dates: List[str] = []
    dup_action: str = "Overwrite data for these dates"  # default

    if df_existing_plus is not None:
        if "reference_date" in df_existing_plus.columns:
            df_existing_plus["reference_date"] = (
                df_existing_plus["reference_date"].astype(str)
            )
            existing_dates_plus = set(
                df_existing_plus["reference_date"].dropna().unique()
            )
            duplicate_dates = sorted(
                existing_dates_plus & set(matched_dates)
            )
            if duplicate_dates:
                st.warning(
                    "The table **pmv_plus_gorila** already has data for "
                    "the following dates:\n"
                    f"- {', '.join(duplicate_dates)}"
                )
                dup_action = st.radio(
                    "Data for these dates already exists in "
                    "`pmv_plus_gorila`. What do you want to do?",
                    options=[
                        "Do nothing",
                        "Overwrite data for these dates",
                    ],
                    index=0,
                    key="dup_action",
                )
            else:
                st.info(
                    "Table **pmv_plus_gorila** exists, but none of the "
                    "matched dates are present there. New data will be "
                    "appended."
                )
        else:
            st.warning(
                "Table **pmv_plus_gorila** exists but has no "
                "`reference_date` column; new data will be appended."
            )

    if st.button("Update pmv_plus_gorila table"):
        with st.spinner("Processing and preparing data..."):
            # Filter PMV and Pictet to matched dates
            df_pmv_filtered = df_pmv[
                df_pmv["reference_date"].isin(matched_dates)
            ].copy()
            df_pictet_filtered = df_pictet[
                df_pictet["Valuation date"].isin(matched_dates)
            ].copy()

            # 1. Ensure new columns from the mapping exist in PMV
            new_cols = extract_new_target_columns(MAPPING)
            df_pmv_extended = df_pmv_filtered.copy()

            for col in new_cols:
                if col not in df_pmv_extended.columns:
                    df_pmv_extended[col] = pd.NA

            # 2. Ensure country_risk exists and set it to "Brazil"
            if "country_risk" not in df_pmv_extended.columns:
                df_pmv_extended["country_risk"] = pd.NA
            df_pmv_extended.loc[:, "country_risk"] = "Brazil"

            # 3. Build PMV-style rows from Pictet positions (matched dates)
            df_pictet_pmv, missing_sources = build_pmv_from_pictet(
                df_pictet=df_pictet_filtered,
                mapping=MAPPING,
            )

            # If Pictet had no Country risk column, make sure the column exists
            if "country_risk" not in df_pictet_pmv.columns:
                df_pictet_pmv["country_risk"] = pd.NA

            # 4. Align columns: start with PMV extended columns in their
            #    current order, then append any extra columns from Pictet.
            all_cols_new = list(df_pmv_extended.columns) + [
                c
                for c in df_pictet_pmv.columns
                if c not in df_pmv_extended.columns
            ]

            df_pmv_aligned = df_pmv_extended.reindex(columns=all_cols_new)
            df_pictet_aligned = df_pictet_pmv.reindex(columns=all_cols_new)

            # 5. Append Pictet rows at the bottom of PMV rows
            df_result = pd.concat(
                [df_pmv_aligned, df_pictet_aligned],
                ignore_index=True,
                sort=False,
            )

            # 6. Final column ordering for this new batch
            df_result = reorder_pmv_columns(df_result)

            # 6b. Set 'indexer' = 'PRÉ' where asset_class == 'Bond'
            if "asset_class" in df_result.columns and "indexer" in df_result.columns:
                bond_mask = (
                    df_result["asset_class"]
                    .astype(str)
                    .str.strip()
                    .str.upper()
                    == "BOND"
                )
                df_result.loc[bond_mask, "indexer"] = "PRÉ"

            # 7. Fetch USD/BRL FX rate and convert USD market values
            try:
                usd_brl = fetch_usd_brl_rate()
            except Exception as e:
                st.error(
                    "Could not fetch USD/BRL rate from yfinance. "
                    f"Skipping FX conversion. Error: {e}"
                )
            else:
                # Ensure the new column exists; it should be at the end
                if (
                    "market_value_amount_original_currency"
                    not in df_result.columns
                ):
                    df_result["market_value_amount_original_currency"] = pd.NA

                if "market_value_amount" in df_result.columns:
                    # Convert to numeric for multiplication
                    df_result["market_value_amount"] = pd.to_numeric(
                        df_result["market_value_amount"],
                        errors="coerce",
                    )

                    # Identify USD rows by the 'currency' column
                    usd_mask = (
                        df_result["currency"]
                        .astype(str)
                        .str.upper()
                        .str.strip()
                        == "USD"
                    )

                    # Store original USD amount
                    df_result.loc[
                        usd_mask, "market_value_amount_original_currency"
                    ] = df_result.loc[usd_mask, "market_value_amount"]

                    # Convert market_value_amount from USD to BRL
                    df_result.loc[usd_mask, "market_value_amount"] = (
                        df_result.loc[usd_mask, "market_value_amount"] * usd_brl
                    )
                else:
                    st.warning(
                        "Column 'market_value_amount' not found; "
                        "skipping FX conversion."
                    )

            # --- Merge with existing pmv_plus_gorila if needed ---
            write_to_db = True
            if df_existing_plus is None:
                # No existing table: just use df_result
                combined = df_result
            else:
                # Ensure reference_date is string
                if "reference_date" in df_existing_plus.columns:
                    df_existing_plus["reference_date"] = (
                        df_existing_plus["reference_date"].astype(str)
                    )

                # Align columns between existing table and new batch
                all_cols = list(df_existing_plus.columns) + [
                    c
                    for c in df_result.columns
                    if c not in df_existing_plus.columns
                ]
                existing_aligned = df_existing_plus.reindex(columns=all_cols)
                new_aligned = df_result.reindex(columns=all_cols)

                if not duplicate_dates:
                    # No overlap: just append new data
                    combined = pd.concat(
                        [existing_aligned, new_aligned],
                        ignore_index=True,
                        sort=False,
                    )
                else:
                    if dup_action == "Do nothing":
                        # Keep existing table as-is; do not write
                        write_to_db = False
                        combined = existing_aligned
                    else:
                        # Overwrite data for these dates: drop old rows for
                        # duplicate dates, then append new rows
                        dup_set = set(duplicate_dates)
                        existing_no_dup = existing_aligned[
                            ~existing_aligned["reference_date"].isin(dup_set)
                        ]
                        combined = pd.concat(
                            [existing_no_dup, new_aligned],
                            ignore_index=True,
                            sort=False,
                        )

            # Ensure PMV base columns are first in final table
            combined = reorder_pmv_columns(combined)

            # 8. Write to DB (if allowed)
            if write_to_db:
                try:
                    write_pmv_plus_gorila(combined)
                except Exception as e:
                    st.error(
                        "Failed to write 'pmv_plus_gorila' table to "
                        f"gorila_positions.db. Error: {e}"
                    )
                    return
            else:
                st.info(
                    "You chose 'Do nothing' for existing dates, so "
                    "the database table 'pmv_plus_gorila' was not modified."
                )

        # --- UI feedback & preview ---
        if "missing_sources" in locals() and missing_sources:
            st.warning(
                "The following Pictet columns from the mapping were not "
                "found in the uploaded Pictet sheet and were skipped: "
                f"{', '.join(missing_sources)}"
            )

        if write_to_db:
            st.success(
                "Table 'pmv_plus_gorila' updated in gorila_positions.db. "
                "Preview of the combined data (including existing rows):"
            )
            preview_df = combined
        else:
            st.info(
                "Preview below is the new batch (not written to DB "
                "because you chose 'Do nothing' for existing dates)."
            )
            preview_df = df_result

        st.dataframe(preview_df.head(50))

        # Optional CSV download of the written/preview table
        csv_bytes = df_to_csv_bytes(preview_df)
        st.download_button(
            label="Download preview as CSV",
            data=csv_bytes,
            file_name=(
                "pmv_plus_gorila.csv"
                if write_to_db
                else "pmv_plus_gorila_preview_new_batch.csv"
            ),
            mime="text/csv",
        )


if __name__ == "__main__":
    main()