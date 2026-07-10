import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from resilientgraph_cmarl.m00_config import (
    TABLES_DIR,
    REPORTS_DIR,
    LOGS_DIR,
    FIGURES_DIR,
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


STEP06_BASELINE_ML = TABLES_DIR / "06_baseline_model_metrics.csv"
STEP07_PROPOSED_ML = TABLES_DIR / "07_leakage_safe_proposed_rgc_model_metrics.csv"
STEP07_ML_IMPROVEMENT = TABLES_DIR / "07_leakage_safe_baseline_vs_proposed_rgc_improvement.csv"

STEP09_BASELINE_RL_SUMMARY = TABLES_DIR / "09_baseline_policy_summary.csv"
STEP09_BEST_BASELINE = TABLES_DIR / "09_best_baseline_policy.csv"

STEP10_PROPOSED_CMARL_SUMMARY = TABLES_DIR / "10_proposed_cmarl_policy_summary.csv"
STEP10_BEST_PROPOSED = TABLES_DIR / "10_best_proposed_cmarl_policy.csv"
STEP10_CMARL_IMPROVEMENT = TABLES_DIR / "10_baseline_vs_proposed_cmarl_improvement.csv"

OUT_ML_FINAL = TABLES_DIR / "11_final_ml_baseline_vs_proposed_summary.csv"
OUT_RL_FINAL = TABLES_DIR / "11_final_rl_baseline_vs_proposed_summary.csv"
OUT_MASTER = TABLES_DIR / "11_final_master_performance_summary.csv"
OUT_KEY_FINDINGS = TABLES_DIR / "11_key_findings_for_manuscript.csv"
OUT_ARTIFACT_INDEX = TABLES_DIR / "11_final_output_artifact_index.csv"

OUT_JSON = REPORTS_DIR / "11_final_comparison_summary.json"
OUT_REPORT = REPORTS_DIR / "11_final_baseline_vs_proposed_report.txt"
OUT_LOG = LOGS_DIR / "11_final_baseline_vs_proposed.log"


def require_file(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Required file missing: {path}")


def safe_float(value, default=0.0):
    value = pd.to_numeric(value, errors="coerce")
    if pd.isna(value) or np.isinf(value):
        return float(default)
    return float(value)


def load_required_csv(path: Path) -> pd.DataFrame:
    require_file(path)
    return load_csv_flexible(path)


def build_ml_summary() -> pd.DataFrame:
    improvement = load_required_csv(STEP07_ML_IMPROVEMENT)

    rows = []

    for _, row in improvement.iterrows():
        rows.append({
            "module": "Risk Prediction",
            "task_type": "Leakage-safe multi-class risk classification",
            "algorithm": row["algorithm"],
            "baseline_model": row["baseline_model"],
            "proposed_model": row["proposed_model"],
            "baseline_accuracy": safe_float(row["baseline_accuracy"]),
            "proposed_accuracy": safe_float(row["proposed_accuracy"]),
            "accuracy_delta": safe_float(row["accuracy_delta"]),
            "accuracy_relative_improvement_percent": safe_float(row["accuracy_relative_improvement_percent"]),
            "baseline_balanced_accuracy": safe_float(row["baseline_balanced_accuracy"]),
            "proposed_balanced_accuracy": safe_float(row["proposed_balanced_accuracy"]),
            "balanced_accuracy_delta": safe_float(row["balanced_accuracy_delta"]),
            "baseline_macro_f1": safe_float(row["baseline_f1_macro"]),
            "proposed_macro_f1": safe_float(row["proposed_f1_macro"]),
            "macro_f1_delta": safe_float(row["f1_macro_delta"]),
            "macro_f1_relative_improvement_percent": safe_float(row["f1_macro_relative_improvement_percent"]),
            "baseline_weighted_f1": safe_float(row["baseline_f1_weighted"]),
            "proposed_weighted_f1": safe_float(row["proposed_f1_weighted"]),
            "weighted_f1_delta": safe_float(row["f1_weighted_delta"]),
            "baseline_roc_auc": safe_float(row["baseline_roc_auc_ovr"]),
            "proposed_roc_auc": safe_float(row["proposed_roc_auc_ovr"]),
            "roc_auc_delta": safe_float(row["roc_auc_delta"]),
        })

    ml_summary = pd.DataFrame(rows)
    ml_summary = ml_summary.sort_values(
        ["proposed_macro_f1", "proposed_accuracy"],
        ascending=False,
    ).reset_index(drop=True)

    return ml_summary


def build_rl_summary() -> pd.DataFrame:
    best_baseline = load_required_csv(STEP09_BEST_BASELINE)
    best_proposed = load_required_csv(STEP10_BEST_PROPOSED)
    improvement = load_required_csv(STEP10_CMARL_IMPROVEMENT)

    base = best_baseline.iloc[0]
    prop = best_proposed.iloc[0]
    imp = improvement.iloc[0]

    rows = [
        {
            "module": "Digital Twin Decision Optimization",
            "task_type": "Best Step 09 baseline policy",
            "model_or_policy": base["policy_name"],
            "mean_reward": safe_float(base["mean_reward"]),
            "mean_risk": safe_float(base["mean_risk"]),
            "mean_delay": safe_float(base["mean_delay"]),
            "mean_service": safe_float(base["mean_service"]),
            "mean_resilience": safe_float(base["mean_resilience"]),
            "mean_profit": safe_float(base["mean_profit"]),
            "mean_cost": safe_float(base["mean_cost"]),
            "data_split": base.get("data_split", "test"),
            "scenario_name": base.get("scenario_name", "normal"),
        },
        {
            "module": "Digital Twin Decision Optimization",
            "task_type": "Proposed STHG-CMAPPO / CMARL policy",
            "model_or_policy": prop["policy_name"],
            "mean_reward": safe_float(prop["mean_reward"]),
            "mean_risk": safe_float(prop["mean_risk"]),
            "mean_delay": safe_float(prop["mean_delay"]),
            "mean_service": safe_float(prop["mean_service"]),
            "mean_resilience": safe_float(prop["mean_resilience"]),
            "mean_profit": safe_float(prop["mean_profit"]),
            "mean_cost": safe_float(prop["mean_cost"]),
            "data_split": prop.get("data_split", "test"),
            "scenario_name": prop.get("scenario_name", "normal"),
        },
    ]

    rl_summary = pd.DataFrame(rows)

    rl_summary["reward_improvement_vs_best_baseline_percent"] = ""
    rl_summary["risk_reduction_vs_best_baseline_percent"] = ""
    rl_summary["delay_reduction_vs_best_baseline_percent"] = ""
    rl_summary["service_improvement_vs_best_baseline_percent"] = ""
    rl_summary["resilience_improvement_vs_best_baseline_percent"] = ""
    rl_summary["profit_improvement_vs_best_baseline_percent"] = ""
    rl_summary["cost_change_vs_best_baseline_percent"] = ""

    proposed_idx = rl_summary["task_type"] == "Proposed STHG-CMAPPO / CMARL policy"

    rl_summary.loc[proposed_idx, "reward_improvement_vs_best_baseline_percent"] = safe_float(
        imp["mean_reward_relative_improvement_percent"]
    )
    rl_summary.loc[proposed_idx, "risk_reduction_vs_best_baseline_percent"] = safe_float(
        imp["risk_reduction_percent"]
    )
    rl_summary.loc[proposed_idx, "delay_reduction_vs_best_baseline_percent"] = safe_float(
        imp["delay_reduction_percent"]
    )
    rl_summary.loc[proposed_idx, "service_improvement_vs_best_baseline_percent"] = safe_float(
        imp["service_improvement_percent"]
    )
    rl_summary.loc[proposed_idx, "resilience_improvement_vs_best_baseline_percent"] = safe_float(
        imp["resilience_improvement_percent"]
    )
    rl_summary.loc[proposed_idx, "profit_improvement_vs_best_baseline_percent"] = safe_float(
        imp["profit_improvement_percent"]
    )
    rl_summary.loc[proposed_idx, "cost_change_vs_best_baseline_percent"] = safe_float(
        imp["cost_change_percent"]
    )

    return rl_summary


def build_master_summary(ml_summary: pd.DataFrame, rl_summary: pd.DataFrame) -> pd.DataFrame:
    best_ml = ml_summary.head(1).iloc[0]
    proposed_rl = rl_summary[rl_summary["task_type"] == "Proposed STHG-CMAPPO / CMARL policy"].iloc[0]

    master_rows = [
        {
            "analysis_stage": "Step 07",
            "module": "Risk Prediction",
            "best_baseline": best_ml["baseline_model"],
            "best_proposed": best_ml["proposed_model"],
            "primary_metric": "Macro-F1",
            "baseline_value": best_ml["baseline_macro_f1"],
            "proposed_value": best_ml["proposed_macro_f1"],
            "absolute_gain": best_ml["macro_f1_delta"],
            "relative_gain_percent": best_ml["macro_f1_relative_improvement_percent"],
            "secondary_metric_1": "Accuracy",
            "secondary_baseline_1": best_ml["baseline_accuracy"],
            "secondary_proposed_1": best_ml["proposed_accuracy"],
            "secondary_metric_2": "ROC-AUC",
            "secondary_baseline_2": best_ml["baseline_roc_auc"],
            "secondary_proposed_2": best_ml["proposed_roc_auc"],
            "interpretation": "Leakage-safe RGC-enhanced ML improved risk-state classification over conventional ML baseline.",
        },
        {
            "analysis_stage": "Step 10",
            "module": "Digital Twin Decision Optimization",
            "best_baseline": rl_summary.iloc[0]["model_or_policy"],
            "best_proposed": proposed_rl["model_or_policy"],
            "primary_metric": "Mean Reward",
            "baseline_value": rl_summary.iloc[0]["mean_reward"],
            "proposed_value": proposed_rl["mean_reward"],
            "absolute_gain": proposed_rl["mean_reward"] - rl_summary.iloc[0]["mean_reward"],
            "relative_gain_percent": proposed_rl["reward_improvement_vs_best_baseline_percent"],
            "secondary_metric_1": "Risk Reduction (%)",
            "secondary_baseline_1": rl_summary.iloc[0]["mean_risk"],
            "secondary_proposed_1": proposed_rl["mean_risk"],
            "secondary_metric_2": "Delay Reduction (%)",
            "secondary_baseline_2": rl_summary.iloc[0]["mean_delay"],
            "secondary_proposed_2": proposed_rl["mean_delay"],
            "interpretation": "Proposed STHG-CMAPPO/CMARL decision layer improved reward, risk, delay, service, resilience, and profit over the strongest baseline policy.",
        },
    ]

    return pd.DataFrame(master_rows)


def build_key_findings(ml_summary: pd.DataFrame, rl_summary: pd.DataFrame) -> pd.DataFrame:
    best_ml = ml_summary.head(1).iloc[0]
    baseline_rl = rl_summary.iloc[0]
    proposed_rl = rl_summary[rl_summary["task_type"] == "Proposed STHG-CMAPPO / CMARL policy"].iloc[0]

    findings = [
        {
            "finding_no": 1,
            "finding_type": "Risk prediction",
            "finding": (
                f"{best_ml['proposed_model']} achieved the strongest leakage-safe risk-classification "
                f"performance with accuracy={best_ml['proposed_accuracy']:.4f}, "
                f"macro-F1={best_ml['proposed_macro_f1']:.4f}, and ROC-AUC={best_ml['proposed_roc_auc']:.4f}."
            ),
        },
        {
            "finding_no": 2,
            "finding_type": "ML improvement",
            "finding": (
                f"Compared with {best_ml['baseline_model']}, {best_ml['proposed_model']} improved "
                f"macro-F1 by {best_ml['macro_f1_relative_improvement_percent']:.2f}% and "
                f"accuracy by {best_ml['accuracy_relative_improvement_percent']:.2f}%."
            ),
        },
        {
            "finding_no": 3,
            "finding_type": "Baseline policy",
            "finding": (
                f"The strongest Step 09 baseline policy was {baseline_rl['model_or_policy']}, "
                f"with mean reward={baseline_rl['mean_reward']:.4f}, mean risk={baseline_rl['mean_risk']:.4f}, "
                f"and mean resilience={baseline_rl['mean_resilience']:.4f}."
            ),
        },
        {
            "finding_no": 4,
            "finding_type": "CMARL improvement",
            "finding": (
                f"The proposed {proposed_rl['model_or_policy']} improved mean reward by "
                f"{float(proposed_rl['reward_improvement_vs_best_baseline_percent']):.2f}% over the strongest baseline."
            ),
        },
        {
            "finding_no": 5,
            "finding_type": "Resilience and risk",
            "finding": (
                f"The proposed CMARL policy reduced risk by "
                f"{float(proposed_rl['risk_reduction_vs_best_baseline_percent']):.2f}%, reduced delay by "
                f"{float(proposed_rl['delay_reduction_vs_best_baseline_percent']):.2f}%, improved service by "
                f"{float(proposed_rl['service_improvement_vs_best_baseline_percent']):.2f}%, and improved resilience by "
                f"{float(proposed_rl['resilience_improvement_vs_best_baseline_percent']):.2f}%."
            ),
        },
    ]

    return pd.DataFrame(findings)


def build_artifact_index() -> pd.DataFrame:
    artifact_paths = [
        STEP06_BASELINE_ML,
        STEP07_PROPOSED_ML,
        STEP07_ML_IMPROVEMENT,
        STEP09_BASELINE_RL_SUMMARY,
        STEP09_BEST_BASELINE,
        STEP10_PROPOSED_CMARL_SUMMARY,
        STEP10_BEST_PROPOSED,
        STEP10_CMARL_IMPROVEMENT,
        OUT_ML_FINAL,
        OUT_RL_FINAL,
        OUT_MASTER,
        OUT_KEY_FINDINGS,
    ]

    rows = []

    for path in artifact_paths:
        rows.append({
            "artifact_name": path.name,
            "artifact_path": str(path),
            "exists": path.exists(),
            "artifact_type": path.suffix.replace(".", ""),
        })

    return pd.DataFrame(rows)


def save_ml_comparison_figure(ml_summary: pd.DataFrame):
    plot_df = ml_summary.copy()

    x = np.arange(len(plot_df))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, plot_df["baseline_macro_f1"], width, label="Baseline Macro-F1")
    ax.bar(x + width / 2, plot_df["proposed_macro_f1"], width, label="Proposed Macro-F1")

    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["algorithm"], rotation=0)
    ax.set_ylabel("Macro-F1")
    ax.set_title("Step 07 leakage-safe ML baseline vs proposed RGC comparison")
    ax.legend()

    fig.tight_layout()
    fig_path = FIGURES_DIR / "11_ml_macro_f1_baseline_vs_proposed.png"
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)

    print(f"[SAVED] {fig_path}")


def save_rl_comparison_figure(rl_summary: pd.DataFrame):
    plot_df = rl_summary.copy()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(plot_df["task_type"], plot_df["mean_reward"])
    ax.set_ylabel("Mean Reward")
    ax.set_title("Step 10 proposed CMARL vs strongest Step 09 baseline")
    ax.tick_params(axis="x", rotation=20)

    fig.tight_layout()
    fig_path = FIGURES_DIR / "11_rl_reward_baseline_vs_proposed.png"
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)

    print(f"[SAVED] {fig_path}")


def save_master_gain_figure(master_summary: pd.DataFrame):
    plot_df = master_summary.copy()

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(plot_df["module"], plot_df["relative_gain_percent"])
    ax.set_ylabel("Relative Improvement (%)")
    ax.set_title("Final relative improvement summary")
    ax.tick_params(axis="x", rotation=15)

    fig.tight_layout()
    fig_path = FIGURES_DIR / "11_final_relative_improvement_summary.png"
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)

    print(f"[SAVED] {fig_path}")


def main():
    print_header("STEP 11: FINAL BASELINE VS PROPOSED COMPARISON")

    print(f"[TIME] {timestamp()}")
    print(f"[PROJECT ROOT] {PROJECT_ROOT}")

    print_subheader("Checking Required Input Files")

    required_files = [
        STEP06_BASELINE_ML,
        STEP07_PROPOSED_ML,
        STEP07_ML_IMPROVEMENT,
        STEP09_BASELINE_RL_SUMMARY,
        STEP09_BEST_BASELINE,
        STEP10_PROPOSED_CMARL_SUMMARY,
        STEP10_BEST_PROPOSED,
        STEP10_CMARL_IMPROVEMENT,
    ]

    for path in required_files:
        require_file(path)
        print(f"[OK] {path}")

    print_subheader("Building Final ML Comparison")
    ml_summary = build_ml_summary()
    save_csv(ml_summary, OUT_ML_FINAL)
    print(ml_summary.to_string(index=False))

    print_subheader("Building Final RL/CMARL Comparison")
    rl_summary = build_rl_summary()
    save_csv(rl_summary, OUT_RL_FINAL)
    print(rl_summary.to_string(index=False))

    print_subheader("Building Master Summary")
    master_summary = build_master_summary(ml_summary, rl_summary)
    save_csv(master_summary, OUT_MASTER)
    print(master_summary.to_string(index=False))

    print_subheader("Building Key Findings")
    key_findings = build_key_findings(ml_summary, rl_summary)
    save_csv(key_findings, OUT_KEY_FINDINGS)
    print(key_findings.to_string(index=False))

    print_subheader("Saving Artifact Index")
    artifact_index = build_artifact_index()
    save_csv(artifact_index, OUT_ARTIFACT_INDEX)

    print_subheader("Saving Figures")
    save_ml_comparison_figure(ml_summary)
    save_rl_comparison_figure(rl_summary)
    save_master_gain_figure(master_summary)

    best_ml = ml_summary.iloc[0]
    baseline_rl = rl_summary.iloc[0]
    proposed_rl = rl_summary[rl_summary["task_type"] == "Proposed STHG-CMAPPO / CMARL policy"].iloc[0]

    summary_json = {
        "time": timestamp(),
        "best_ml_model": {
            "baseline": best_ml["baseline_model"],
            "proposed": best_ml["proposed_model"],
            "baseline_accuracy": float(best_ml["baseline_accuracy"]),
            "proposed_accuracy": float(best_ml["proposed_accuracy"]),
            "baseline_macro_f1": float(best_ml["baseline_macro_f1"]),
            "proposed_macro_f1": float(best_ml["proposed_macro_f1"]),
            "macro_f1_relative_improvement_percent": float(best_ml["macro_f1_relative_improvement_percent"]),
            "proposed_roc_auc": float(best_ml["proposed_roc_auc"]),
        },
        "best_decision_policy": {
            "baseline_policy": baseline_rl["model_or_policy"],
            "proposed_policy": proposed_rl["model_or_policy"],
            "baseline_mean_reward": float(baseline_rl["mean_reward"]),
            "proposed_mean_reward": float(proposed_rl["mean_reward"]),
            "reward_improvement_percent": float(proposed_rl["reward_improvement_vs_best_baseline_percent"]),
            "risk_reduction_percent": float(proposed_rl["risk_reduction_vs_best_baseline_percent"]),
            "delay_reduction_percent": float(proposed_rl["delay_reduction_vs_best_baseline_percent"]),
            "service_improvement_percent": float(proposed_rl["service_improvement_vs_best_baseline_percent"]),
            "resilience_improvement_percent": float(proposed_rl["resilience_improvement_vs_best_baseline_percent"]),
            "profit_improvement_percent": float(proposed_rl["profit_improvement_vs_best_baseline_percent"]),
        },
        "output_files": {
            "ml_summary": str(OUT_ML_FINAL),
            "rl_summary": str(OUT_RL_FINAL),
            "master_summary": str(OUT_MASTER),
            "key_findings": str(OUT_KEY_FINDINGS),
            "artifact_index": str(OUT_ARTIFACT_INDEX),
            "report": str(OUT_REPORT),
        },
    }

    save_json(summary_json, OUT_JSON)

    report_lines = []
    report_lines.append("STEP 11 FINAL BASELINE VS PROPOSED COMPARISON REPORT")
    report_lines.append("=" * 95)
    report_lines.append(f"Time: {summary_json['time']}")
    report_lines.append("")
    report_lines.append("1. Final leakage-safe risk prediction result")
    report_lines.append("-" * 95)
    report_lines.append(
        f"Best proposed ML model: {best_ml['proposed_model']} compared with {best_ml['baseline_model']}."
    )
    report_lines.append(
        f"Accuracy improved from {best_ml['baseline_accuracy']:.4f} to {best_ml['proposed_accuracy']:.4f} "
        f"({best_ml['accuracy_relative_improvement_percent']:.2f}% relative improvement)."
    )
    report_lines.append(
        f"Macro-F1 improved from {best_ml['baseline_macro_f1']:.4f} to {best_ml['proposed_macro_f1']:.4f} "
        f"({best_ml['macro_f1_relative_improvement_percent']:.2f}% relative improvement)."
    )
    report_lines.append(
        f"ROC-AUC improved from {best_ml['baseline_roc_auc']:.4f} to {best_ml['proposed_roc_auc']:.4f}."
    )
    report_lines.append("")
    report_lines.append("2. Final digital-twin decision optimization result")
    report_lines.append("-" * 95)
    report_lines.append(
        f"Best Step 09 baseline policy: {baseline_rl['model_or_policy']} with mean reward "
        f"{baseline_rl['mean_reward']:.4f}."
    )
    report_lines.append(
        f"Selected proposed policy: {proposed_rl['model_or_policy']} with mean reward "
        f"{proposed_rl['mean_reward']:.4f}."
    )
    report_lines.append(
        f"Mean reward improvement: {float(proposed_rl['reward_improvement_vs_best_baseline_percent']):.2f}%."
    )
    report_lines.append(
        f"Risk reduction: {float(proposed_rl['risk_reduction_vs_best_baseline_percent']):.2f}%."
    )
    report_lines.append(
        f"Delay reduction: {float(proposed_rl['delay_reduction_vs_best_baseline_percent']):.2f}%."
    )
    report_lines.append(
        f"Service improvement: {float(proposed_rl['service_improvement_vs_best_baseline_percent']):.2f}%."
    )
    report_lines.append(
        f"Resilience improvement: {float(proposed_rl['resilience_improvement_vs_best_baseline_percent']):.2f}%."
    )
    report_lines.append(
        f"Profit improvement: {float(proposed_rl['profit_improvement_vs_best_baseline_percent']):.2f}%."
    )
    report_lines.append("")
    report_lines.append("3. Manuscript-ready key findings")
    report_lines.append("-" * 95)
    for _, row in key_findings.iterrows():
        report_lines.append(f"{int(row['finding_no'])}. {row['finding']}")
    report_lines.append("")
    report_lines.append("4. Output tables")
    report_lines.append("-" * 95)
    for _, row in artifact_index.iterrows():
        report_lines.append(f"- {row['artifact_name']}: {row['artifact_path']}")

    save_text("\n".join(report_lines), OUT_REPORT)

    log_text = (
        f"[{timestamp()}] Step 11 completed. "
        f"BestML={best_ml['proposed_model']}, "
        f"BestCMARL={proposed_rl['model_or_policy']}, "
        f"RewardImprovement={float(proposed_rl['reward_improvement_vs_best_baseline_percent']):.2f}%\n"
    )
    save_text(log_text, OUT_LOG)

    print_subheader("Saved Files")
    print(f"[ML FINAL] {OUT_ML_FINAL}")
    print(f"[RL FINAL] {OUT_RL_FINAL}")
    print(f"[MASTER SUMMARY] {OUT_MASTER}")
    print(f"[KEY FINDINGS] {OUT_KEY_FINDINGS}")
    print(f"[ARTIFACT INDEX] {OUT_ARTIFACT_INDEX}")
    print(f"[JSON SUMMARY] {OUT_JSON}")
    print(f"[REPORT] {OUT_REPORT}")

    print_header("STEP 11 COMPLETED SUCCESSFULLY")
    print("[NEXT] Send me the terminal output.")
    print("[NEXT FILE AFTER VERIFICATION] 12_ablation_study.py")


if __name__ == "__main__":
    main()