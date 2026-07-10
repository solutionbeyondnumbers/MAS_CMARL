import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from resilientgraph_cmarl.m00_config import (
    RAW_DATA_FILE,
    PROCESSED_DIR,
    TABLES_DIR,
    REPORTS_DIR,
    LOGS_DIR,
)
from resilientgraph_cmarl.m01_utils_io import (
    print_header,
    print_subheader,
    load_csv_flexible,
    normalize_columns,
    save_csv,
    save_json,
    save_text,
    timestamp,
)
from resilientgraph_cmarl.m02_audit_preprocess import preprocess_dataco


def main():
    print_header("STEP 03: PREPROCESSING - ResilientGraph-CMARL")

    print(f"[TIME] {timestamp()}")
    print(f"[PROJECT ROOT] {PROJECT_ROOT}")
    print(f"[RAW DATA FILE] {RAW_DATA_FILE}")

    if not RAW_DATA_FILE.exists():
        raise FileNotFoundError(f"Raw dataset not found: {RAW_DATA_FILE}")

    print_subheader("Loading Raw Dataset")
    raw_df = load_csv_flexible(RAW_DATA_FILE)
    raw_df = normalize_columns(raw_df)

    print(f"[RAW SHAPE] {raw_df.shape[0]:,} rows | {raw_df.shape[1]:,} columns")

    print_subheader("Running Preprocessing")
    results = preprocess_dataco(raw_df)

    cleaned_df = results["cleaned_df"]
    removed_columns_report = results["removed_columns_report"]
    date_parse_report = results["date_parse_report"]
    missing_fill_report = results["missing_fill_report"]
    column_role_report = results["column_role_report"]

    print(f"[CLEANED SHAPE] {cleaned_df.shape[0]:,} rows | {cleaned_df.shape[1]:,} columns")
    print(f"[REMOVED COLUMNS] {len(removed_columns_report)}")
    print(f"[REMAINING MISSING VALUES] {int(cleaned_df.isna().sum().sum()):,}")

    print_subheader("Saving Preprocessed Outputs")

    cleaned_path = PROCESSED_DIR / "03_dataco_cleaned.csv"
    save_csv(cleaned_df, cleaned_path)

    save_csv(removed_columns_report, TABLES_DIR / "03_removed_columns_report.csv")
    save_csv(date_parse_report, TABLES_DIR / "03_date_parse_report.csv")
    save_csv(missing_fill_report, TABLES_DIR / "03_missing_fill_report.csv")
    save_csv(column_role_report, TABLES_DIR / "03_column_role_report.csv")

    preprocessing_summary = {
        "time": timestamp(),
        "raw_rows": int(raw_df.shape[0]),
        "raw_columns": int(raw_df.shape[1]),
        "cleaned_rows": int(cleaned_df.shape[0]),
        "cleaned_columns": int(cleaned_df.shape[1]),
        "removed_columns": removed_columns_report["column"].tolist() if not removed_columns_report.empty else [],
        "remaining_missing_values": int(cleaned_df.isna().sum().sum()),
        "output_cleaned_file": str(cleaned_path),
        "date_parse_columns": date_parse_report.to_dict(orient="records") if not date_parse_report.empty else [],
    }

    save_json(preprocessing_summary, REPORTS_DIR / "03_preprocessing_summary.json")

    report_lines = []
    report_lines.append("STEP 03 PREPROCESSING REPORT")
    report_lines.append("=" * 80)
    report_lines.append(f"Time: {preprocessing_summary['time']}")
    report_lines.append(f"Raw shape: {raw_df.shape[0]:,} rows x {raw_df.shape[1]:,} columns")
    report_lines.append(f"Cleaned shape: {cleaned_df.shape[0]:,} rows x {cleaned_df.shape[1]:,} columns")
    report_lines.append("")
    report_lines.append("Removed columns:")
    if not removed_columns_report.empty:
        for col in removed_columns_report["column"].tolist():
            report_lines.append(f"- {col}")
    else:
        report_lines.append("- None")
    report_lines.append("")
    report_lines.append("Created basic operational features:")
    created_cols = [
        "order_year",
        "order_month",
        "order_week",
        "order_day",
        "order_dayofweek",
        "order_quarter",
        "computed_shipping_days",
        "shipping_delay_gap",
        "is_delayed_by_days",
        "sales_per_unit",
        "profit_margin",
    ]
    for col in created_cols:
        if col in cleaned_df.columns:
            report_lines.append(f"- {col}")
    report_lines.append("")
    report_lines.append("Important note:")
    report_lines.append(
        "Risk-source columns such as Late_delivery_risk, Delivery Status, Days for shipping "
        "(real), and Days for shipment (scheduled) are intentionally retained for Step 04 risk-score construction."
    )

    save_text("\n".join(report_lines), REPORTS_DIR / "03_preprocessing_report.txt")

    log_text = (
        f"[{timestamp()}] Step 03 completed. "
        f"Raw={raw_df.shape}, Cleaned={cleaned_df.shape}, "
        f"RemainingMissing={int(cleaned_df.isna().sum().sum())}\n"
    )
    save_text(log_text, LOGS_DIR / "03_preprocessing.log")

    print_subheader("Preprocessing Summary")
    print(f"[RAW SHAPE] {raw_df.shape}")
    print(f"[CLEANED SHAPE] {cleaned_df.shape}")
    print(f"[CLEANED FILE] {cleaned_path}")
    print("[REMOVED COLUMNS]")
    if not removed_columns_report.empty:
        print(removed_columns_report.to_string(index=False))
    else:
        print("None")

    print("\n[DATE PARSE REPORT]")
    if not date_parse_report.empty:
        print(date_parse_report.to_string(index=False))
    else:
        print("No date columns parsed.")

    print("\n[CREATED FEATURE CHECK]")
    for col in [
        "computed_shipping_days",
        "shipping_delay_gap",
        "is_delayed_by_days",
        "sales_per_unit",
        "profit_margin",
    ]:
        if col in cleaned_df.columns:
            print(f"[OK] {col}")

    print_header("STEP 03 COMPLETED SUCCESSFULLY")
    print("[NEXT] Send me the terminal output.")
    print("[NEXT FILE AFTER VERIFICATION] 04_feature_engineering_risk_labels.py")


if __name__ == "__main__":
    main()