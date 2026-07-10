import sys
from pathlib import Path
from datetime import datetime

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from resilientgraph_cmarl.m00_config import (
    RAW_DATA_FILE,
    TABLES_DIR,
    REPORTS_DIR,
    LOGS_DIR,
    EXPECTED_DATACO_COLUMNS,
    LEAKAGE_KEYWORDS,
)
from resilientgraph_cmarl.m01_utils_io import (
    print_header,
    print_subheader,
    load_csv_flexible,
    normalize_columns,
    save_csv,
    save_json,
    save_text,
    make_missing_report,
    make_dtype_report,
    make_categorical_summary,
    safe_numeric_summary,
    dataframe_memory_mb,
    infer_possible_date_columns,
    infer_possible_id_columns,
    infer_possible_leakage_columns,
    find_expected_columns,
    timestamp,
)


def main():
    print_header("STEP 02: DATA AUDIT - ResilientGraph-CMARL")

    print(f"[TIME] {timestamp()}")
    print(f"[PROJECT ROOT] {PROJECT_ROOT}")
    print(f"[RAW DATA FILE] {RAW_DATA_FILE}")

    if not RAW_DATA_FILE.exists():
        print("\n[ERROR] Raw dataset file not found.")
        print("Please place the dataset here:")
        print(RAW_DATA_FILE)
        raise FileNotFoundError(RAW_DATA_FILE)

    print_subheader("Loading Dataset")
    df = load_csv_flexible(RAW_DATA_FILE)
    df = normalize_columns(df)

    n_rows, n_cols = df.shape
    memory_mb = dataframe_memory_mb(df)
    duplicate_rows = int(df.duplicated().sum())

    print(f"[SHAPE] Rows: {n_rows:,} | Columns: {n_cols:,}")
    print(f"[MEMORY] {memory_mb} MB")
    print(f"[DUPLICATES] Full duplicate rows: {duplicate_rows:,}")

    print_subheader("Column Overview")
    for i, col in enumerate(df.columns, start=1):
        print(f"{i:03d}. {col} | dtype={df[col].dtype} | missing={df[col].isna().sum()}")

    print_subheader("Generating Reports")

    missing_report = make_missing_report(df)
    dtype_report = make_dtype_report(df)
    numeric_summary = safe_numeric_summary(df)
    categorical_summary = make_categorical_summary(df)

    possible_date_columns = infer_possible_date_columns(df)
    possible_id_columns = infer_possible_id_columns(df)
    possible_leakage_columns = infer_possible_leakage_columns(df, LEAKAGE_KEYWORDS)
    expected_column_matches = find_expected_columns(df.columns.tolist(), EXPECTED_DATACO_COLUMNS)

    expected_report = pd.DataFrame([
        {
            "required_role": role,
            "matched_column": matched_col if matched_col is not None else "NOT FOUND",
            "status": "FOUND" if matched_col is not None else "MISSING",
        }
        for role, matched_col in expected_column_matches.items()
    ])

    audit_summary = {
        "audit_time": timestamp(),
        "raw_data_file": str(RAW_DATA_FILE),
        "rows": n_rows,
        "columns": n_cols,
        "memory_mb": memory_mb,
        "duplicate_rows": duplicate_rows,
        "total_missing_values": int(df.isna().sum().sum()),
        "numeric_columns": int(len(df.select_dtypes(include="number").columns)),
        "categorical_columns": int(len(df.select_dtypes(include=["object", "category", "bool"]).columns)),
        "possible_date_columns": possible_date_columns,
        "possible_id_columns": possible_id_columns,
        "possible_leakage_columns": possible_leakage_columns,
        "expected_column_matches": expected_column_matches,
    }

    save_csv(missing_report, TABLES_DIR / "02_missing_values_report.csv")
    save_csv(dtype_report, TABLES_DIR / "02_dtype_column_report.csv")
    save_csv(expected_report, TABLES_DIR / "02_expected_dataco_columns_report.csv")

    if not numeric_summary.empty:
        save_csv(numeric_summary, TABLES_DIR / "02_numeric_summary.csv")
    else:
        print("[INFO] No numeric columns found. Numeric summary skipped.")

    if not categorical_summary.empty:
        save_csv(categorical_summary, TABLES_DIR / "02_categorical_summary.csv")
    else:
        print("[INFO] No categorical columns found. Categorical summary skipped.")

    save_json(audit_summary, REPORTS_DIR / "02_data_audit_summary.json")

    report_lines = []
    report_lines.append("STEP 02 DATA AUDIT REPORT")
    report_lines.append("=" * 80)
    report_lines.append(f"Audit time: {audit_summary['audit_time']}")
    report_lines.append(f"Raw file: {RAW_DATA_FILE}")
    report_lines.append(f"Rows: {n_rows:,}")
    report_lines.append(f"Columns: {n_cols:,}")
    report_lines.append(f"Memory MB: {memory_mb}")
    report_lines.append(f"Duplicate rows: {duplicate_rows:,}")
    report_lines.append(f"Total missing values: {audit_summary['total_missing_values']:,}")
    report_lines.append("")
    report_lines.append("Possible date columns:")
    report_lines.extend([f"- {c}" for c in possible_date_columns] or ["- None found"])
    report_lines.append("")
    report_lines.append("Possible leakage-sensitive columns:")
    report_lines.extend([f"- {c}" for c in possible_leakage_columns] or ["- None found"])
    report_lines.append("")
    report_lines.append("Expected DataCo column matching:")
    for role, col in expected_column_matches.items():
        report_lines.append(f"- {role}: {col if col else 'NOT FOUND'}")
    report_lines.append("")
    report_lines.append("Important note:")
    report_lines.append(
        "Leakage-sensitive columns are not automatically removed in Step 02. "
        "They are only flagged. Final removal/controlled use will be handled during preprocessing and target construction."
    )

    save_text("\n".join(report_lines), REPORTS_DIR / "02_data_audit_report.txt")

    log_text = (
        f"[{timestamp()}] Step 02 completed. "
        f"Rows={n_rows}, Columns={n_cols}, Duplicates={duplicate_rows}, Missing={audit_summary['total_missing_values']}\n"
    )
    save_text(log_text, LOGS_DIR / "02_data_audit.log")

    print_subheader("Audit Summary")
    print(f"[ROWS] {n_rows:,}")
    print(f"[COLUMNS] {n_cols:,}")
    print(f"[MEMORY MB] {memory_mb}")
    print(f"[DUPLICATE ROWS] {duplicate_rows:,}")
    print(f"[TOTAL MISSING] {audit_summary['total_missing_values']:,}")

    print("\n[EXPECTED DATACO COLUMN MATCHES]")
    print(expected_report.to_string(index=False))

    print("\n[POSSIBLE DATE COLUMNS]")
    for col in possible_date_columns:
        print(f"- {col}")

    print("\n[POSSIBLE LEAKAGE-SENSITIVE COLUMNS]")
    for col in possible_leakage_columns:
        print(f"- {col}")

    print_header("STEP 02 COMPLETED SUCCESSFULLY")
    print("[NEXT] Send me the terminal output, especially the expected-column matching table.")
    print("[NEXT FILE AFTER VERIFICATION] 03_preprocessing.py")


if __name__ == "__main__":
    main()