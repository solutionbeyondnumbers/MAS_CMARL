import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from resilientgraph_cmarl.m00_config import (
    PROCESSED_DIR,
    TABLES_DIR,
    REPORTS_DIR,
    LOGS_DIR,
    MODELS_DIR,
    FIGURES_DIR,
    RANDOM_STATE,
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
from resilientgraph_cmarl.m05_baseline_proposed_models import (
    TARGET_COL,
    make_temporal_split,
    get_baseline_feature_columns,
    split_x_y,
    train_evaluate_single_model,
    save_model,
    make_metrics_table,
    make_prediction_table,
)


INPUT_FILE = PROCESSED_DIR / "05_dataco_graph_risk_features.csv"

BASELINE_MODEL_NAMES = [
    "XGBoost",
    "LightGBM",
    "CatBoost",
]


def save_confusion_matrix_figure(cm, model_name: str, split_name: str):
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Low", "Moderate", "High"],
    )
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"{model_name} - {split_name} confusion matrix")

    path = FIGURES_DIR / f"06_{model_name.lower()}_{split_name}_confusion_matrix.png"
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)

    print(f"[SAVED] {path}")


def main():
    print_header("STEP 06: TRAIN BASELINE 3 ML MODELS")

    print(f"[TIME] {timestamp()}")
    print(f"[PROJECT ROOT] {PROJECT_ROOT}")
    print(f"[INPUT FILE] {INPUT_FILE}")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Step 05 graph-risk dataset not found: {INPUT_FILE}")

    print_subheader("Loading Step 05 Dataset")
    df = load_csv_flexible(INPUT_FILE)
    print(f"[INPUT SHAPE] {df.shape[0]:,} rows | {df.shape[1]:,} columns")

    if TARGET_COL not in df.columns:
        raise KeyError(f"Target column missing: {TARGET_COL}")

    print_subheader("Creating Leakage-Safe Temporal Split")
    split_data = make_temporal_split(df)

    for split_name, split_df in split_data.items():
        min_date = pd.to_datetime(split_df["order date (DateOrders)"]).min()
        max_date = pd.to_datetime(split_df["order date (DateOrders)"]).max()
        print(
            f"[{split_name.upper()}] "
            f"rows={split_df.shape[0]:,} | "
            f"date={min_date} to {max_date} | "
            f"label_counts={split_df[TARGET_COL].value_counts().sort_index().to_dict()}"
        )

    print_subheader("Selecting Baseline Features")
    feature_cols = get_baseline_feature_columns(df)

    print(f"[BASELINE FEATURE COUNT] {len(feature_cols)}")
    print("[FIRST 30 FEATURES]")
    for i, col in enumerate(feature_cols[:30], start=1):
        print(f"{i:02d}. {col}")

    feature_report = pd.DataFrame(
        {
            "feature": feature_cols,
            "dtype": [str(df[c].dtype) for c in feature_cols],
            "used_in": "baseline_models",
        }
    )
    save_csv(feature_report, TABLES_DIR / "06_baseline_feature_list.csv")

    split_xy = split_x_y(split_data, feature_cols)

    X_train = split_xy["X_train"]
    y_train = split_xy["y_train"]
    X_valid = split_xy["X_valid"]
    y_valid = split_xy["y_valid"]
    X_test = split_xy["X_test"]
    y_test = split_xy["y_test"]

    print_subheader("Training Baseline Models")
    all_results = {}
    all_prediction_tables = []

    for model_name in BASELINE_MODEL_NAMES:
        print("\n" + "-" * 100)
        print(f"[TRAINING] {model_name}")
        print("-" * 100)

        result = train_evaluate_single_model(
            model_name=model_name,
            X_train=X_train,
            y_train=y_train,
            X_valid=X_valid,
            y_valid=y_valid,
            X_test=X_test,
            y_test=y_test,
            random_state=RANDOM_STATE,
        )

        all_results[model_name] = result

        model_path = MODELS_DIR / f"06_baseline_{model_name.lower()}_model.joblib"
        save_model(result["model"], model_path)
        print(f"[SAVED] {model_path}")

        for split_name in ["train", "valid", "test"]:
            metrics = result["metrics"][split_name]
            print(
                f"[{model_name} | {split_name.upper()}] "
                f"ACC={metrics['accuracy']:.4f} | "
                f"BAL_ACC={metrics['balanced_accuracy']:.4f} | "
                f"F1_MACRO={metrics['f1_macro']:.4f} | "
                f"F1_WEIGHTED={metrics['f1_weighted']:.4f} | "
                f"ROC_AUC_OVR={metrics['roc_auc_ovr_macro']:.4f}"
            )

            save_confusion_matrix_figure(
                result["confusion_matrices"][split_name],
                model_name,
                split_name,
            )

            preds = result["predictions"][split_name]
            pred_table = make_prediction_table(
                model_name=model_name,
                split_name=split_name,
                y_true=preds["y_true"],
                y_pred=preds["y_pred"],
                y_proba=preds["y_proba"],
            )
            all_prediction_tables.append(pred_table)

        report_path = REPORTS_DIR / f"06_{model_name.lower()}_classification_report.json"
        save_json(result["classification_reports"], report_path)

    print_subheader("Saving Final Step 06 Tables")

    metrics_table = make_metrics_table(all_results)
    save_csv(metrics_table, TABLES_DIR / "06_baseline_model_metrics.csv")

    if all_prediction_tables:
        prediction_table = pd.concat(all_prediction_tables, ignore_index=True)
        save_csv(prediction_table, TABLES_DIR / "06_baseline_predictions.csv")

    best_test = (
        metrics_table[metrics_table["split"] == "test"]
        .sort_values(["f1_macro", "balanced_accuracy"], ascending=False)
        .head(1)
    )

    save_csv(best_test, TABLES_DIR / "06_best_baseline_model.csv")

    summary = {
        "time": timestamp(),
        "input_file": str(INPUT_FILE),
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "feature_count": int(len(feature_cols)),
        "baseline_models": BASELINE_MODEL_NAMES,
        "split_rows": {
            "train": int(split_data["train"].shape[0]),
            "valid": int(split_data["valid"].shape[0]),
            "test": int(split_data["test"].shape[0]),
        },
        "best_test_model": best_test.to_dict(orient="records"),
        "important_leakage_control": [
            "Risk label, risk score, risk components, delivery outcome, post-shipping columns, graph features, and IDs are excluded from baseline features.",
            "Temporal split is used instead of random split.",
            "Graph-enhanced proposed models will be trained separately in Step 07.",
        ],
    }

    save_json(summary, REPORTS_DIR / "06_baseline_training_summary.json")

    report_lines = []
    report_lines.append("STEP 06 BASELINE MODEL TRAINING REPORT")
    report_lines.append("=" * 90)
    report_lines.append(f"Time: {summary['time']}")
    report_lines.append(f"Input file: {INPUT_FILE}")
    report_lines.append(f"Input shape: {df.shape[0]:,} rows x {df.shape[1]:,} columns")
    report_lines.append(f"Baseline feature count: {len(feature_cols)}")
    report_lines.append("")
    report_lines.append("Baseline models trained:")
    for model_name in BASELINE_MODEL_NAMES:
        report_lines.append(f"- {model_name}")
    report_lines.append("")
    report_lines.append("Temporal split:")
    for split_name, split_df in split_data.items():
        report_lines.append(f"- {split_name}: {split_df.shape[0]:,} rows")
    report_lines.append("")
    report_lines.append("Leakage control:")
    report_lines.append("- Direct risk score, risk components, late-delivery label, delivery status, post-shipping duration, IDs, and graph features were excluded.")
    report_lines.append("- Graph-enhanced proposed versions will be trained in Step 07 using the same baseline feature set plus graph-risk features.")
    report_lines.append("")
    report_lines.append("Test-set model ranking:")
    test_metrics = metrics_table[metrics_table["split"] == "test"].sort_values(
        ["f1_macro", "balanced_accuracy"],
        ascending=False,
    )
    for _, row in test_metrics.iterrows():
        report_lines.append(
            f"- {row['model_name']}: "
            f"accuracy={row['accuracy']:.4f}, "
            f"balanced_accuracy={row['balanced_accuracy']:.4f}, "
            f"macro_f1={row['f1_macro']:.4f}, "
            f"weighted_f1={row['f1_weighted']:.4f}, "
            f"roc_auc_ovr={row['roc_auc_ovr_macro']:.4f}"
        )

    save_text("\n".join(report_lines), REPORTS_DIR / "06_baseline_training_report.txt")

    log_text = (
        f"[{timestamp()}] Step 06 completed. "
        f"Models={BASELINE_MODEL_NAMES}, FeatureCount={len(feature_cols)}, "
        f"BestTest={best_test.to_dict(orient='records')}\n"
    )
    save_text(log_text, LOGS_DIR / "06_baseline_training.log")

    print_subheader("Final Baseline Test Metrics")
    test_metrics = metrics_table[metrics_table["split"] == "test"].sort_values(
        ["f1_macro", "balanced_accuracy"],
        ascending=False,
    )
    print(test_metrics.to_string(index=False))

    print_subheader("Saved Files")
    print(f"[METRICS] {TABLES_DIR / '06_baseline_model_metrics.csv'}")
    print(f"[PREDICTIONS] {TABLES_DIR / '06_baseline_predictions.csv'}")
    print(f"[BEST BASELINE] {TABLES_DIR / '06_best_baseline_model.csv'}")
    print(f"[REPORT] {REPORTS_DIR / '06_baseline_training_report.txt'}")

    print_header("STEP 06 COMPLETED SUCCESSFULLY")
    print("[NEXT] Send me the terminal output.")
    print("[NEXT FILE AFTER VERIFICATION] 07_train_proposed_3_rgc_models.py")


if __name__ == "__main__":
    main()