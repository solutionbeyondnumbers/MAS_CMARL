import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from resilientgraph_cmarl.m00_config import (
    PROCESSED_DIR,
    TABLES_DIR,
    REPORTS_DIR,
    LOGS_DIR,
)
from resilientgraph_cmarl.m01_utils_io import (
    print_header,
    print_subheader,
    load_csv_flexible,
    save_csv,
    save_json,
    save_text,
    timestamp,
)
from resilientgraph_cmarl.m03_features_risk import build_feature_engineered_dataset


INPUT_FILE = PROCESSED_DIR / "03_dataco_cleaned.csv"
OUTPUT_FILE = PROCESSED_DIR / "04_dataco_features_risk_labels.csv"


def main():
    print_header("STEP 04: FEATURE ENGINEERING + COMPOSITE RISK LABELS")

    print(f"[TIME] {timestamp()}")
    print(f"[PROJECT ROOT] {PROJECT_ROOT}")
    print(f"[INPUT FILE] {INPUT_FILE}")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Step 03 cleaned file not found: {INPUT_FILE}")

    print_subheader("Loading Step 03 Cleaned Dataset")
    df = load_csv_flexible(INPUT_FILE)
    print(f"[INPUT SHAPE] {df.shape[0]:,} rows | {df.shape[1]:,} columns")

    print_subheader("Building Temporal Features and Risk Labels")
    results = build_feature_engineered_dataset(df)

    engineered_df = results["engineered_df"]
    risk_config_report = results["risk_config_report"]
    risk_distribution_report = results["risk_distribution_report"]
    feature_summary_report = results["feature_summary_report"]

    print(f"[OUTPUT SHAPE] {engineered_df.shape[0]:,} rows | {engineered_df.shape[1]:,} columns")

    print_subheader("Saving Step 04 Outputs")
    save_csv(engineered_df, OUTPUT_FILE)

    save_csv(risk_config_report, TABLES_DIR / "04_risk_score_weights_thresholds.csv")
    save_csv(risk_distribution_report, TABLES_DIR / "04_risk_label_distribution.csv")
    save_csv(feature_summary_report, TABLES_DIR / "04_engineered_feature_summary.csv")

    risk_counts = risk_distribution_report.to_dict(orient="records")

    summary = {
        "time": timestamp(),
        "input_file": str(INPUT_FILE),
        "output_file": str(OUTPUT_FILE),
        "input_rows": int(df.shape[0]),
        "input_columns": int(df.shape[1]),
        "output_rows": int(engineered_df.shape[0]),
        "output_columns": int(engineered_df.shape[1]),
        "risk_distribution": risk_counts,
        "created_core_columns": [
            "qty_lag_1",
            "qty_lag_3",
            "qty_lag_7",
            "qty_roll_mean_3",
            "qty_roll_mean_7",
            "qty_roll_std_7",
            "delay_lag_1",
            "delay_roll_mean_7",
            "delay_roll_std_7",
            "profit_lag_1",
            "profit_roll_mean_7",
            "profit_roll_std_7",
            "delay_risk",
            "late_delivery_component",
            "demand_volatility_risk",
            "anomaly_risk",
            "shortage_exposure_risk",
            "profit_loss_risk",
            "shipping_risk",
            "composite_disruption_risk_score",
            "risk_label",
            "risk_label_name",
        ],
    }

    save_json(summary, REPORTS_DIR / "04_feature_engineering_risk_summary.json")

    report_lines = []
    report_lines.append("STEP 04 FEATURE ENGINEERING AND RISK LABEL REPORT")
    report_lines.append("=" * 90)
    report_lines.append(f"Time: {summary['time']}")
    report_lines.append(f"Input shape: {df.shape[0]:,} rows x {df.shape[1]:,} columns")
    report_lines.append(f"Output shape: {engineered_df.shape[0]:,} rows x {engineered_df.shape[1]:,} columns")
    report_lines.append("")
    report_lines.append("Core proposed risk components:")
    report_lines.append("- Delay risk")
    report_lines.append("- Late-delivery component")
    report_lines.append("- Demand volatility risk")
    report_lines.append("- Anomaly-risk proxy")
    report_lines.append("- Shortage-exposure risk")
    report_lines.append("- Profit-loss risk")
    report_lines.append("- Shipping-mode historical risk")
    report_lines.append("")
    report_lines.append("Risk label distribution:")
    for row in risk_counts:
        report_lines.append(f"- {row['risk_label_name']}: {row['count']} ({row['percent']}%)")
    report_lines.append("")
    report_lines.append("Important note:")
    report_lines.append(
        "Risk labels are created using a composite disruption-risk score. "
        "Thresholds are based on 33rd and 66th percentiles to maintain class balance for modelling."
    )

    save_text("\n".join(report_lines), REPORTS_DIR / "04_feature_engineering_risk_report.txt")

    log_text = (
        f"[{timestamp()}] Step 04 completed. "
        f"Input={df.shape}, Output={engineered_df.shape}, "
        f"RiskDistribution={risk_counts}\n"
    )
    save_text(log_text, LOGS_DIR / "04_feature_engineering_risk.log")

    print_subheader("Risk Label Distribution")
    print(risk_distribution_report.to_string(index=False))

    print_subheader("Risk Score Component Summary")
    display_cols = [
        "feature",
        "missing",
        "mean",
        "std",
        "min",
        "max",
    ]
    print(feature_summary_report[display_cols].to_string(index=False))

    print_subheader("Saved Files")
    print(f"[FEATURED DATASET] {OUTPUT_FILE}")
    print(f"[RISK DISTRIBUTION] {TABLES_DIR / '04_risk_label_distribution.csv'}")
    print(f"[RISK CONFIG] {TABLES_DIR / '04_risk_score_weights_thresholds.csv'}")
    print(f"[FEATURE SUMMARY] {TABLES_DIR / '04_engineered_feature_summary.csv'}")

    print_header("STEP 04 COMPLETED SUCCESSFULLY")
    print("[NEXT] Send me the terminal output.")
    print("[NEXT FILE AFTER VERIFICATION] 05_graph_construction_risknet.py")


if __name__ == "__main__":
    main()