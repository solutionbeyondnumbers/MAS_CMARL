# scripts/16_dataset2_setup_audit.py
# ======================================================================================
# STEP 16: DATASET 2 SETUP + AUDIT
# Project: ResilientGraph-CMARL
# Dataset 2: Brazilian E-Commerce Public Dataset by Olist
#
# Purpose:
#   1. Create separate Dataset 2 folders.
#   2. Detect already extracted Olist CSV files OR extract archive.zip if available.
#   3. Audit all Olist CSV files.
#   4. Check shapes, missing values, duplicates, join keys, date columns, and distributions.
#   5. Save all Dataset 2 outputs separately under outputs_dataset2/.
# ======================================================================================

from __future__ import annotations

import json
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


# --------------------------------------------------------------------------------------
# Project paths
# --------------------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET2_RAW_DIR = PROJECT_ROOT / "data" / "dataset2_raw"
DATASET2_EXTRACTED_DIR = DATASET2_RAW_DIR / "olist_extracted"
DATASET2_PROCESSED_DIR = PROJECT_ROOT / "data" / "dataset2_processed"
DATASET2_GRAPH_DIR = PROJECT_ROOT / "data" / "dataset2_graph"
DATASET2_SIMULATION_DIR = PROJECT_ROOT / "data" / "dataset2_simulation"

OUTPUTS2_DIR = PROJECT_ROOT / "outputs_dataset2"
TABLES2_DIR = OUTPUTS2_DIR / "tables"
FIGURES2_DIR = OUTPUTS2_DIR / "figures"
MODELS2_DIR = OUTPUTS2_DIR / "models"
LOGS2_DIR = OUTPUTS2_DIR / "logs"
REPORTS2_DIR = OUTPUTS2_DIR / "reports"
EXPLAIN2_DIR = OUTPUTS2_DIR / "explainability"
STRESS2_DIR = OUTPUTS2_DIR / "stress_tests"
DASHBOARD2_DIR = OUTPUTS2_DIR / "dashboard"

LOG_FILE = LOGS2_DIR / "16_dataset2_setup_audit.log"


REQUIRED_OLIST_FILES = {
    "customers": "olist_customers_dataset.csv",
    "geolocation": "olist_geolocation_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "order_payments": "olist_order_payments_dataset.csv",
    "order_reviews": "olist_order_reviews_dataset.csv",
    "orders": "olist_orders_dataset.csv",
    "products": "olist_products_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "category_translation": "product_category_name_translation.csv",
}


DATE_COLUMNS_BY_TABLE = {
    "orders": [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ],
    "order_items": [
        "shipping_limit_date",
    ],
    "order_reviews": [
        "review_creation_date",
        "review_answer_timestamp",
    ],
}


JOIN_KEY_TESTS = [
    {
        "test_name": "orders.customer_id -> customers.customer_id",
        "left_table": "orders",
        "right_table": "customers",
        "left_key": "customer_id",
        "right_key": "customer_id",
    },
    {
        "test_name": "order_items.order_id -> orders.order_id",
        "left_table": "order_items",
        "right_table": "orders",
        "left_key": "order_id",
        "right_key": "order_id",
    },
    {
        "test_name": "order_payments.order_id -> orders.order_id",
        "left_table": "order_payments",
        "right_table": "orders",
        "left_key": "order_id",
        "right_key": "order_id",
    },
    {
        "test_name": "order_reviews.order_id -> orders.order_id",
        "left_table": "order_reviews",
        "right_table": "orders",
        "left_key": "order_id",
        "right_key": "order_id",
    },
    {
        "test_name": "order_items.product_id -> products.product_id",
        "left_table": "order_items",
        "right_table": "products",
        "left_key": "product_id",
        "right_key": "product_id",
    },
    {
        "test_name": "order_items.seller_id -> sellers.seller_id",
        "left_table": "order_items",
        "right_table": "sellers",
        "left_key": "seller_id",
        "right_key": "seller_id",
    },
    {
        "test_name": "products.product_category_name -> category_translation.product_category_name",
        "left_table": "products",
        "right_table": "category_translation",
        "left_key": "product_category_name",
        "right_key": "product_category_name",
    },
]


# --------------------------------------------------------------------------------------
# Logging helpers
# --------------------------------------------------------------------------------------

def ensure_directories() -> None:
    dirs = [
        DATASET2_RAW_DIR,
        DATASET2_EXTRACTED_DIR,
        DATASET2_PROCESSED_DIR,
        DATASET2_GRAPH_DIR,
        DATASET2_SIMULATION_DIR,
        OUTPUTS2_DIR,
        TABLES2_DIR,
        FIGURES2_DIR,
        MODELS2_DIR,
        LOGS2_DIR,
        REPORTS2_DIR,
        EXPLAIN2_DIR,
        STRESS2_DIR,
        DASHBOARD2_DIR,
    ]

    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def reset_log() -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    LOG_FILE.write_text("", encoding="utf-8")


def log(message: str = "") -> None:
    print(message, flush=True)
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(str(message) + "\n")


def print_header(title: str) -> None:
    line = "=" * 100
    log("\n" + line)
    log(title)
    log(line)


def print_section(title: str) -> None:
    log("\n" + "-" * 100)
    log(title)
    log("-" * 100)


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    log(f"[SAVED] {path}")


def save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=4, default=str), encoding="utf-8")
    log(f"[SAVED] {path}")


def save_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    log(f"[SAVED] {path}")


# --------------------------------------------------------------------------------------
# Dataset discovery and extraction
# --------------------------------------------------------------------------------------

def candidate_search_dirs() -> List[Path]:
    return [
        DATASET2_RAW_DIR,
        DATASET2_EXTRACTED_DIR,
        PROJECT_ROOT / "data" / "raw" / "dataset2_raw",
        PROJECT_ROOT / "data" / "raw",
        PROJECT_ROOT,
    ]


def locate_required_files() -> Dict[str, Path | None]:
    file_map: Dict[str, Path | None] = {}

    for table_name, file_name in REQUIRED_OLIST_FILES.items():
        found_path = None

        for base_dir in candidate_search_dirs():
            candidate = base_dir / file_name
            if candidate.exists():
                found_path = candidate
                break

        if found_path is None:
            matches = list(PROJECT_ROOT.glob(f"**/{file_name}"))
            if matches:
                found_path = matches[0]

        file_map[table_name] = found_path

    return file_map


def required_csvs_available(file_map: Dict[str, Path | None]) -> bool:
    return all(path is not None and path.exists() for path in file_map.values())


def find_archive_zip() -> Path | None:
    candidates = [
        PROJECT_ROOT / "archive.zip",
        DATASET2_RAW_DIR / "archive.zip",
        DATASET2_EXTRACTED_DIR / "archive.zip",
        PROJECT_ROOT / "data" / "raw" / "archive.zip",
        PROJECT_ROOT / "data" / "raw" / "dataset2_raw" / "archive.zip",
    ]

    for path in candidates:
        if path.exists():
            return path

    matches = list(PROJECT_ROOT.glob("**/archive.zip"))
    if matches:
        return matches[0]

    return None


def extract_archive_if_needed() -> Tuple[bool, Path | None, Dict[str, Path | None]]:
    file_map = locate_required_files()

    if required_csvs_available(file_map):
        log("[OK] Required Olist CSV files already found. Archive extraction not required.")
        return True, None, file_map

    archive_path = find_archive_zip()

    if archive_path is None:
        log("[ERROR] Required Olist CSV files were not found and archive.zip was not found.")
        log("")
        log("Place either the extracted CSV files or archive.zip in one of these locations:")
        log(f"1. {DATASET2_RAW_DIR}")
        log(f"2. {DATASET2_EXTRACTED_DIR}")
        log(f"3. {PROJECT_ROOT / 'data' / 'raw' / 'dataset2_raw'}")
        log(f"4. {PROJECT_ROOT / 'data' / 'raw'}")
        log(f"5. {PROJECT_ROOT}")
        log("")
        log("Missing files:")
        for table_name, path in file_map.items():
            if path is None:
                log(f" - {REQUIRED_OLIST_FILES[table_name]}")
        return False, None, file_map

    print_section("Extracting Dataset 2 Archive")
    log(f"[ARCHIVE FOUND] {archive_path}")

    try:
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(DATASET2_EXTRACTED_DIR)
        log(f"[OK] Extracted archive to: {DATASET2_EXTRACTED_DIR}")
    except zipfile.BadZipFile:
        log("[ERROR] archive.zip is not a valid zip file.")
        return False, archive_path, file_map
    except Exception as exc:
        log(f"[ERROR] Could not extract archive: {exc}")
        return False, archive_path, file_map

    file_map = locate_required_files()

    if not required_csvs_available(file_map):
        log("[ERROR] Archive extracted, but some required CSV files are still missing.")
        for table_name, path in file_map.items():
            if path is None:
                log(f" - {REQUIRED_OLIST_FILES[table_name]}")
        return False, archive_path, file_map

    return True, archive_path, file_map


# --------------------------------------------------------------------------------------
# CSV loading and audit
# --------------------------------------------------------------------------------------

def read_csv_safe(path: Path) -> pd.DataFrame:
    encodings = ["utf-8", "utf-8-sig", "latin1"]

    last_error = None

    for enc in encodings:
        try:
            df = pd.read_csv(path, encoding=enc)
            log(f"[OK] Loaded {path.name} using encoding: {enc}")
            return df
        except Exception as exc:
            last_error = exc

    raise RuntimeError(f"Could not read {path}. Last error: {last_error}")


def load_olist_tables(file_map: Dict[str, Path | None]) -> Dict[str, pd.DataFrame]:
    tables: Dict[str, pd.DataFrame] = {}

    for table_name, path in file_map.items():
        if path is None:
            tables[table_name] = pd.DataFrame()
            log(f"[MISSING] {table_name}: {REQUIRED_OLIST_FILES[table_name]}")
            continue

        tables[table_name] = read_csv_safe(path)

    return tables


def build_file_inventory(file_map: Dict[str, Path | None]) -> pd.DataFrame:
    rows = []

    for table_name, expected_file in REQUIRED_OLIST_FILES.items():
        path = file_map.get(table_name)

        rows.append(
            {
                "table_name": table_name,
                "expected_file": expected_file,
                "file_found": path is not None,
                "file_path": str(path) if path is not None else "",
                "file_size_mb": round(path.stat().st_size / (1024 * 1024), 4) if path else None,
            }
        )

    return pd.DataFrame(rows)


def build_table_audit_summary(tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []

    for table_name, df in tables.items():
        if df.empty:
            rows.append(
                {
                    "table_name": table_name,
                    "rows": 0,
                    "columns": 0,
                    "memory_mb": 0,
                    "duplicate_rows": 0,
                    "total_missing_values": 0,
                    "missing_percent_total_cells": 0,
                    "status": "EMPTY_OR_NOT_LOADED",
                }
            )
            continue

        total_cells = df.shape[0] * df.shape[1]
        missing_count = int(df.isna().sum().sum())

        rows.append(
            {
                "table_name": table_name,
                "rows": int(df.shape[0]),
                "columns": int(df.shape[1]),
                "memory_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 4),
                "duplicate_rows": int(df.duplicated().sum()),
                "total_missing_values": missing_count,
                "missing_percent_total_cells": round((missing_count / total_cells) * 100, 4) if total_cells > 0 else 0,
                "status": "LOADED",
            }
        )

    return pd.DataFrame(rows)


def build_missing_values_report(tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []

    for table_name, df in tables.items():
        if df.empty:
            continue

        for col in df.columns:
            missing = int(df[col].isna().sum())

            rows.append(
                {
                    "table_name": table_name,
                    "column_name": col,
                    "dtype": str(df[col].dtype),
                    "missing_count": missing,
                    "missing_percent": round((missing / len(df)) * 100, 4) if len(df) else 0,
                    "unique_values": int(df[col].nunique(dropna=True)),
                }
            )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values(
        ["missing_percent", "missing_count"],
        ascending=False,
    )


def build_duplicate_report(tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    primary_keys = {
        "customers": ["customer_id"],
        "orders": ["order_id"],
        "products": ["product_id"],
        "sellers": ["seller_id"],
        "category_translation": ["product_category_name"],
        "order_items": ["order_id", "order_item_id"],
        "order_payments": ["order_id", "payment_sequential"],
        "order_reviews": ["review_id"],
        "geolocation": ["geolocation_zip_code_prefix", "geolocation_lat", "geolocation_lng"],
    }

    rows = []

    for table_name, df in tables.items():
        if df.empty:
            continue

        key_cols = primary_keys.get(table_name, [])
        available_key_cols = [c for c in key_cols if c in df.columns]

        duplicate_full_rows = int(df.duplicated().sum())

        duplicate_key_rows = None
        if available_key_cols:
            duplicate_key_rows = int(df.duplicated(subset=available_key_cols).sum())

        rows.append(
            {
                "table_name": table_name,
                "primary_key_checked": ", ".join(available_key_cols),
                "duplicate_full_rows": duplicate_full_rows,
                "duplicate_key_rows": duplicate_key_rows,
                "rows": int(len(df)),
            }
        )

    return pd.DataFrame(rows)


def build_column_dictionary(tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []

    for table_name, df in tables.items():
        if df.empty:
            continue

        for col in df.columns:
            sample_values = df[col].dropna().astype(str).head(5).tolist()

            rows.append(
                {
                    "table_name": table_name,
                    "column_name": col,
                    "dtype": str(df[col].dtype),
                    "non_null_count": int(df[col].notna().sum()),
                    "missing_count": int(df[col].isna().sum()),
                    "unique_values": int(df[col].nunique(dropna=True)),
                    "sample_values": " | ".join(sample_values),
                }
            )

    return pd.DataFrame(rows)


def build_join_key_integrity_report(tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []

    for test in JOIN_KEY_TESTS:
        test_name = test["test_name"]
        left_table = test["left_table"]
        right_table = test["right_table"]
        left_key = test["left_key"]
        right_key = test["right_key"]

        left_df = tables.get(left_table, pd.DataFrame())
        right_df = tables.get(right_table, pd.DataFrame())

        if left_df.empty or right_df.empty or left_key not in left_df.columns or right_key not in right_df.columns:
            rows.append(
                {
                    "test_name": test_name,
                    "left_table": left_table,
                    "right_table": right_table,
                    "left_key": left_key,
                    "right_key": right_key,
                    "left_rows": len(left_df),
                    "right_rows": len(right_df),
                    "matched_left_rows": 0,
                    "unmatched_left_rows": None,
                    "unmatched_left_percent": None,
                    "status": "SKIPPED_MISSING_TABLE_OR_KEY",
                }
            )
            continue

        right_keys = set(right_df[right_key].dropna().astype(str).unique())
        left_values = left_df[left_key].dropna().astype(str)

        matched = int(left_values.isin(right_keys).sum())
        total = int(left_values.shape[0])
        unmatched = total - matched
        unmatched_percent = round((unmatched / total) * 100, 4) if total else 0

        rows.append(
            {
                "test_name": test_name,
                "left_table": left_table,
                "right_table": right_table,
                "left_key": left_key,
                "right_key": right_key,
                "left_rows": int(len(left_df)),
                "right_rows": int(len(right_df)),
                "matched_left_rows": matched,
                "unmatched_left_rows": unmatched,
                "unmatched_left_percent": unmatched_percent,
                "status": "PASS" if unmatched == 0 else "CHECK",
            }
        )

    return pd.DataFrame(rows)


def build_date_quality_report(tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []

    for table_name, date_cols in DATE_COLUMNS_BY_TABLE.items():
        df = tables.get(table_name, pd.DataFrame())

        if df.empty:
            continue

        for col in date_cols:
            if col not in df.columns:
                rows.append(
                    {
                        "table_name": table_name,
                        "date_column": col,
                        "exists": False,
                        "valid_dates": 0,
                        "invalid_or_missing_dates": None,
                        "min_date": None,
                        "max_date": None,
                    }
                )
                continue

            parsed = pd.to_datetime(df[col], errors="coerce")
            valid = int(parsed.notna().sum())
            invalid_or_missing = int(parsed.isna().sum())

            rows.append(
                {
                    "table_name": table_name,
                    "date_column": col,
                    "exists": True,
                    "valid_dates": valid,
                    "invalid_or_missing_dates": invalid_or_missing,
                    "min_date": parsed.min(),
                    "max_date": parsed.max(),
                }
            )

    return pd.DataFrame(rows)


def value_count_report(
    df: pd.DataFrame,
    column: str,
    table_name: str,
    output_name: str,
    top_n: int = 50,
) -> pd.DataFrame:
    if df.empty or column not in df.columns:
        report = pd.DataFrame(
            columns=[
                "table_name",
                "column_name",
                "value",
                "count",
                "percent",
            ]
        )
        save_csv(report, TABLES2_DIR / output_name)
        return report

    counts = df[column].value_counts(dropna=False).head(top_n)
    total = len(df)

    report = counts.reset_index()
    report.columns = ["value", "count"]
    report.insert(0, "column_name", column)
    report.insert(0, "table_name", table_name)
    report["percent"] = (report["count"] / total * 100).round(4)

    save_csv(report, TABLES2_DIR / output_name)

    return report


def build_numeric_summary(tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []

    for table_name, df in tables.items():
        if df.empty:
            continue

        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

        for col in numeric_cols:
            series = df[col]

            rows.append(
                {
                    "table_name": table_name,
                    "column_name": col,
                    "count": int(series.notna().sum()),
                    "mean": float(series.mean()) if series.notna().any() else None,
                    "std": float(series.std()) if series.notna().any() else None,
                    "min": float(series.min()) if series.notna().any() else None,
                    "p25": float(series.quantile(0.25)) if series.notna().any() else None,
                    "median": float(series.median()) if series.notna().any() else None,
                    "p75": float(series.quantile(0.75)) if series.notna().any() else None,
                    "max": float(series.max()) if series.notna().any() else None,
                }
            )

    return pd.DataFrame(rows)


def print_key_terminal_summary(
    file_inventory: pd.DataFrame,
    table_summary: pd.DataFrame,
    join_report: pd.DataFrame,
    date_report: pd.DataFrame,
    missing_report: pd.DataFrame,
) -> None:
    print_section("Dataset 2 File Inventory")
    if file_inventory.empty:
        log("[WARNING] No file inventory created.")
    else:
        log(file_inventory.to_string(index=False))

    print_section("Dataset 2 Table Summary")
    if table_summary.empty:
        log("[WARNING] No tables loaded.")
    else:
        log(table_summary.to_string(index=False))

    print_section("Join-Key Integrity Summary")
    if join_report.empty:
        log("[WARNING] No join-key report created.")
    else:
        log(join_report.to_string(index=False))

    print_section("Date Quality Summary")
    if date_report.empty:
        log("[WARNING] No date quality report created.")
    else:
        log(date_report.to_string(index=False))

    print_section("Top Missing Columns")
    if missing_report.empty:
        log("[OK] No missing report created.")
    else:
        top_missing = missing_report[missing_report["missing_count"] > 0].head(30)

        if top_missing.empty:
            log("[OK] No missing values detected.")
        else:
            log(top_missing.to_string(index=False))


def save_clean_copy_to_dataset2_raw(file_map: Dict[str, Path | None]) -> None:
    """
    Keep Dataset 2 raw folder organized by copying file path references logically.
    This does not duplicate content unnecessarily. It only warns if files are located elsewhere.
    """
    rows = []

    for table_name, path in file_map.items():
        expected_local_path = DATASET2_RAW_DIR / REQUIRED_OLIST_FILES[table_name]

        rows.append(
            {
                "table_name": table_name,
                "current_path": str(path) if path else "",
                "preferred_path": str(expected_local_path),
                "already_in_preferred_raw_folder": bool(path is not None and path.resolve() == expected_local_path.resolve()),
            }
        )

    relocation_report = pd.DataFrame(rows)
    save_csv(relocation_report, TABLES2_DIR / "16_dataset2_file_location_preference_report.csv")


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

def main() -> None:
    ensure_directories()
    reset_log()

    print_header("STEP 16: DATASET 2 SETUP + AUDIT - OLIST E-COMMERCE")
    log(f"[TIME] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"[PROJECT ROOT] {PROJECT_ROOT}")
    log(f"[DATASET 2 RAW DIR] {DATASET2_RAW_DIR}")
    log(f"[DATASET 2 OUTPUT DIR] {OUTPUTS2_DIR}")

    print_section("Creating Dataset 2 Folder Structure")
    log(f"[OK] Raw directory: {DATASET2_RAW_DIR}")
    log(f"[OK] Extracted directory: {DATASET2_EXTRACTED_DIR}")
    log(f"[OK] Processed directory: {DATASET2_PROCESSED_DIR}")
    log(f"[OK] Graph directory: {DATASET2_GRAPH_DIR}")
    log(f"[OK] Simulation directory: {DATASET2_SIMULATION_DIR}")
    log(f"[OK] Outputs directory: {OUTPUTS2_DIR}")

    print_section("Checking Existing CSV Files or Archive")
    extraction_ok, archive_path, file_map = extract_archive_if_needed()

    if not extraction_ok:
        log("[STOPPED] Dataset 2 setup and audit could not continue.")
        sys.exit(1)

    print_section("Locating Required Olist CSV Files")
    file_map = locate_required_files()
    file_inventory = build_file_inventory(file_map)

    save_csv(file_inventory, TABLES2_DIR / "16_dataset2_file_inventory.csv")
    log(file_inventory.to_string(index=False))

    missing_files = file_inventory[file_inventory["file_found"] == False]
    if not missing_files.empty:
        log("")
        log("[ERROR] Some required Olist files are missing.")
        log(missing_files.to_string(index=False))
        sys.exit(1)

    save_clean_copy_to_dataset2_raw(file_map)

    print_section("Loading Olist CSV Files")
    tables = load_olist_tables(file_map)

    print_section("Building Audit Reports")
    table_summary = build_table_audit_summary(tables)
    missing_report = build_missing_values_report(tables)
    duplicate_report = build_duplicate_report(tables)
    column_dictionary = build_column_dictionary(tables)
    join_report = build_join_key_integrity_report(tables)
    date_report = build_date_quality_report(tables)
    numeric_summary = build_numeric_summary(tables)

    save_csv(table_summary, TABLES2_DIR / "16_dataset2_table_audit_summary.csv")
    save_csv(missing_report, TABLES2_DIR / "16_dataset2_missing_values_report.csv")
    save_csv(duplicate_report, TABLES2_DIR / "16_dataset2_duplicate_report.csv")
    save_csv(column_dictionary, TABLES2_DIR / "16_dataset2_column_dictionary.csv")
    save_csv(join_report, TABLES2_DIR / "16_dataset2_join_key_integrity_report.csv")
    save_csv(date_report, TABLES2_DIR / "16_dataset2_date_quality_report.csv")
    save_csv(numeric_summary, TABLES2_DIR / "16_dataset2_numeric_summary.csv")

    print_section("Saving Important Distribution Reports")

    value_count_report(
        tables.get("orders", pd.DataFrame()),
        "order_status",
        "orders",
        "16_dataset2_order_status_distribution.csv",
    )

    value_count_report(
        tables.get("customers", pd.DataFrame()),
        "customer_state",
        "customers",
        "16_dataset2_customer_state_distribution.csv",
    )

    value_count_report(
        tables.get("sellers", pd.DataFrame()),
        "seller_state",
        "sellers",
        "16_dataset2_seller_state_distribution.csv",
    )

    value_count_report(
        tables.get("order_payments", pd.DataFrame()),
        "payment_type",
        "order_payments",
        "16_dataset2_payment_type_distribution.csv",
    )

    value_count_report(
        tables.get("order_reviews", pd.DataFrame()),
        "review_score",
        "order_reviews",
        "16_dataset2_review_score_distribution.csv",
    )

    value_count_report(
        tables.get("products", pd.DataFrame()),
        "product_category_name",
        "products",
        "16_dataset2_product_category_distribution.csv",
    )

    print_key_terminal_summary(
        file_inventory=file_inventory,
        table_summary=table_summary,
        join_report=join_report,
        date_report=date_report,
        missing_report=missing_report,
    )

    total_rows_loaded = int(table_summary["rows"].sum()) if not table_summary.empty else 0
    total_tables_loaded = int((table_summary["status"] == "LOADED").sum()) if not table_summary.empty else 0
    total_missing_values = int(table_summary["total_missing_values"].sum()) if not table_summary.empty else 0
    total_duplicate_rows = int(table_summary["duplicate_rows"].sum()) if not table_summary.empty else 0

    join_tests_total = int(len(join_report))
    join_tests_passed = int((join_report["status"] == "PASS").sum()) if not join_report.empty else 0
    join_tests_check = int((join_report["status"] == "CHECK").sum()) if not join_report.empty else 0

    summary = {
        "step": "16_dataset2_setup_audit",
        "dataset": "Brazilian E-Commerce Public Dataset by Olist",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "project_root": str(PROJECT_ROOT),
        "archive_path": str(archive_path) if archive_path else None,
        "dataset2_raw_dir": str(DATASET2_RAW_DIR),
        "dataset2_extracted_dir": str(DATASET2_EXTRACTED_DIR),
        "outputs_dataset2_dir": str(OUTPUTS2_DIR),
        "required_files": REQUIRED_OLIST_FILES,
        "tables_loaded": total_tables_loaded,
        "total_rows_loaded_across_tables": total_rows_loaded,
        "total_missing_values_across_tables": total_missing_values,
        "total_duplicate_rows_across_tables": total_duplicate_rows,
        "join_tests_total": join_tests_total,
        "join_tests_passed": join_tests_passed,
        "join_tests_check": join_tests_check,
        "date_columns_checked": int(len(date_report)),
        "output_tables": [
            "16_dataset2_file_inventory.csv",
            "16_dataset2_file_location_preference_report.csv",
            "16_dataset2_table_audit_summary.csv",
            "16_dataset2_missing_values_report.csv",
            "16_dataset2_duplicate_report.csv",
            "16_dataset2_column_dictionary.csv",
            "16_dataset2_join_key_integrity_report.csv",
            "16_dataset2_date_quality_report.csv",
            "16_dataset2_numeric_summary.csv",
            "16_dataset2_order_status_distribution.csv",
            "16_dataset2_customer_state_distribution.csv",
            "16_dataset2_seller_state_distribution.csv",
            "16_dataset2_payment_type_distribution.csv",
            "16_dataset2_review_score_distribution.csv",
            "16_dataset2_product_category_distribution.csv",
        ],
    }

    save_json(summary, REPORTS2_DIR / "16_dataset2_setup_audit_summary.json")

    report_lines: List[str] = []
    report_lines.append("STEP 16: DATASET 2 SETUP + AUDIT REPORT")
    report_lines.append("=" * 100)
    report_lines.append("Dataset: Brazilian E-Commerce Public Dataset by Olist")
    report_lines.append(f"Timestamp: {summary['timestamp']}")
    report_lines.append(f"Project root: {PROJECT_ROOT}")
    report_lines.append(f"Dataset 2 raw directory: {DATASET2_RAW_DIR}")
    report_lines.append(f"Dataset 2 output directory: {OUTPUTS2_DIR}")
    report_lines.append("")
    report_lines.append("Audit Summary")
    report_lines.append("-" * 100)
    report_lines.append(f"Tables loaded: {total_tables_loaded}")
    report_lines.append(f"Total rows loaded across tables: {total_rows_loaded:,}")
    report_lines.append(f"Total missing values across tables: {total_missing_values:,}")
    report_lines.append(f"Total duplicate rows across tables: {total_duplicate_rows:,}")
    report_lines.append(f"Join tests passed: {join_tests_passed} / {join_tests_total}")
    report_lines.append(f"Join tests requiring check: {join_tests_check}")
    report_lines.append(f"Date columns checked: {summary['date_columns_checked']}")
    report_lines.append("")
    report_lines.append("Loaded Tables")
    report_lines.append("-" * 100)
    report_lines.append(table_summary.to_string(index=False))
    report_lines.append("")
    report_lines.append("Join Integrity")
    report_lines.append("-" * 100)
    report_lines.append(join_report.to_string(index=False))
    report_lines.append("")
    report_lines.append("Date Quality")
    report_lines.append("-" * 100)
    report_lines.append(date_report.to_string(index=False))
    report_lines.append("")
    report_lines.append("Top Missing Columns")
    report_lines.append("-" * 100)

    if missing_report.empty:
        report_lines.append("No missing-value report available.")
    else:
        top_missing = missing_report[missing_report["missing_count"] > 0].head(30)
        if top_missing.empty:
            report_lines.append("No missing values detected.")
        else:
            report_lines.append(top_missing.to_string(index=False))

    save_text(
        "\n".join(report_lines),
        REPORTS2_DIR / "16_dataset2_setup_audit_report.txt",
    )

    print_section("Step 16 Completed")
    log("[DONE] Dataset 2 setup and audit completed successfully.")
    log(f"[LOG SAVED] {LOG_FILE}")
    log(f"[TABLES SAVED] {TABLES2_DIR}")
    log(f"[REPORTS SAVED] {REPORTS2_DIR}")

if __name__ == "__main__":
    main()