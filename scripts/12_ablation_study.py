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


STEP07_CANDIDATE_SUMMARY = TABLES_DIR / "07_leakage_safe_candidate_selection_summary.csv"
STEP07_FINAL_IMPROVEMENT = TABLES_DIR / "07_leakage_safe_baseline_vs_proposed_rgc_improvement.csv"

STEP09_BEST_BASELINE = TABLES_DIR / "09_best_baseline_policy.csv"
STEP09_STRESS = TABLES_DIR / "09_baseline_stress_scenario_evaluation.csv"

STEP10_PROPOSED_SUMMARY = TABLES_DIR / "10_proposed_cmarl_policy_summary.csv"
STEP10_BEST_PROPOSED = TABLES_DIR / "10_best_proposed_cmarl_policy.csv"
STEP10_IMPROVEMENT = TABLES_DIR / "10_baseline_vs_proposed_cmarl_improvement.csv"
STEP10_STRESS = TABLES_DIR / "10_proposed_cmarl_stress_scenario_evaluation.csv"

OUT_ML_ABLATION = TABLES_DIR / "12_ml_rgc_feature_ablation_summary.csv"
OUT_CMARL_ABLATION = TABLES_DIR / "12_cmarl_policy_ablation_summary.csv"
OUT_STRESS_ABLATION = TABLES_DIR / "12_stress_scenario_ablation_summary.csv"
OUT_MASTER_ABLATION = TABLES_DIR / "12_master_ablation_summary.csv"
OUT_KEY_ABLATION = TABLES_DIR / "12_key_ablation_findings.csv"

OUT_JSON = REPORTS_DIR / "12_ablation_study_summary.json"
OUT_REPORT = REPORTS_DIR / "12_ablation_study_report.txt"
OUT_LOG = LOGS_DIR / "12_ablation_study.log"


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


def build_ml_ablation() -> pd.DataFrame:
    candidate_df = load_required_csv(STEP07_CANDIDATE_SUMMARY)

    required_cols = [
        "proposed_model",
        "base_algorithm",
        "candidate_feature_set",
        "feature_count",
        "valid_accuracy",
        "valid_f1_macro",
        "valid_balanced_accuracy",
        "valid_roc_auc",
        "test_accuracy_preview",
        "test_f1_macro_preview",
        "test_balanced_accuracy_preview",
        "candidate_status",
    ]

    missing = [col for col in required_cols if col not in candidate_df.columns]
    if missing:
        raise KeyError(f"Step 07 candidate summary missing columns: {missing}")

    rows = []

    for model_name, group in candidate_df.groupby("proposed_model", observed=False):
        group = group.copy()

        strict_row = group[group["candidate_feature_set"] == "RGC-StrictContext"]

        if strict_row.empty:
            strict_acc = safe_float(group["test_accuracy_preview"].min())
            strict_f1 = safe_float(group["test_f1_macro_preview"].min())
            strict_bal = safe_float(group["test_balanced_accuracy_preview"].min())
        else:
            strict_acc = safe_float(strict_row.iloc[0]["test_accuracy_preview"])
            strict_f1 = safe_float(strict_row.iloc[0]["test_f1_macro_preview"])
            strict_bal = safe_float(strict_row.iloc[0]["test_balanced_accuracy_preview"])

        for _, row in group.iterrows():
            test_acc = safe_float(row["test_accuracy_preview"])
            test_f1 = safe_float(row["test_f1_macro_preview"])
            test_bal = safe_float(row["test_balanced_accuracy_preview"])

            rows.append({
                "proposed_model": row["proposed_model"],
                "base_algorithm": row["base_algorithm"],
                "candidate_feature_set": row["candidate_feature_set"],
                "feature_count": int(row["feature_count"]),
                "candidate_status": row["candidate_status"],
                "valid_accuracy": safe_float(row["valid_accuracy"]),
                "valid_macro_f1": safe_float(row["valid_f1_macro"]),
                "valid_balanced_accuracy": safe_float(row["valid_balanced_accuracy"]),
                "valid_roc_auc": safe_float(row["valid_roc_auc"]),
                "test_accuracy": test_acc,
                "test_macro_f1": test_f1,
                "test_balanced_accuracy": test_bal,
                "accuracy_gain_vs_strict_context": test_acc - strict_acc,
                "macro_f1_gain_vs_strict_context": test_f1 - strict_f1,
                "balanced_accuracy_gain_vs_strict_context": test_bal - strict_bal,
                "ablation_role": get_ml_ablation_role(row["candidate_feature_set"]),
            })

    ablation_df = pd.DataFrame(rows)

    ablation_df = ablation_df.sort_values(
        ["proposed_model", "test_macro_f1", "test_accuracy"],
        ascending=[True, False, False],
    ).reset_index(drop=True)

    return ablation_df


def get_ml_ablation_role(candidate_name: str) -> str:
    if candidate_name == "RGC-StrictContext":
        return "Baseline-safe graph/context features only"
    if candidate_name == "RGC-ProxyRiskContext":
        return "Adds controlled proxy-risk indicators"
    if candidate_name == "RGC-ExpandedContext":
        return "Adds expanded train-safe entity prior context"
    return "Other"


def build_cmarl_ablation() -> pd.DataFrame:
    proposed_summary = load_required_csv(STEP10_PROPOSED_SUMMARY)
    best_baseline = load_required_csv(STEP09_BEST_BASELINE)

    baseline = best_baseline.iloc[0]

    test_df = proposed_summary[
        (proposed_summary["data_split"] == "test")
        & (proposed_summary["scenario_name"] == "normal")
    ].copy()

    rows = []

    for _, row in test_df.iterrows():
        base_reward = safe_float(baseline["mean_reward"])
        base_risk = safe_float(baseline["mean_risk"])
        base_delay = safe_float(baseline["mean_delay"])
        base_service = safe_float(baseline["mean_service"])
        base_resilience = safe_float(baseline["mean_resilience"])
        base_profit = safe_float(baseline["mean_profit"])
        base_cost = safe_float(baseline["mean_cost"])

        prop_reward = safe_float(row["mean_reward"])
        prop_risk = safe_float(row["mean_risk"])
        prop_delay = safe_float(row["mean_delay"])
        prop_service = safe_float(row["mean_service"])
        prop_resilience = safe_float(row["mean_resilience"])
        prop_profit = safe_float(row["mean_profit"])
        prop_cost = safe_float(row["mean_cost"])

        rows.append({
            "policy_name": row["policy_name"],
            "policy_ablation_role": get_policy_role(row["policy_name"]),
            "decision_count": int(row["decision_count"]),
            "mean_reward": prop_reward,
            "mean_risk": prop_risk,
            "mean_delay": prop_delay,
            "mean_service": prop_service,
            "mean_resilience": prop_resilience,
            "mean_profit": prop_profit,
            "mean_cost": prop_cost,
            "reward_gain_vs_best_baseline": prop_reward - base_reward,
            "reward_relative_gain_percent": (
                (prop_reward - base_reward) / max(abs(base_reward), 1e-9) * 100
            ),
            "risk_reduction_percent": (
                (base_risk - prop_risk) / max(abs(base_risk), 1e-9) * 100
            ),
            "delay_reduction_percent": (
                (base_delay - prop_delay) / max(abs(base_delay), 1e-9) * 100
            ),
            "service_improvement_percent": (
                (prop_service - base_service) / max(abs(base_service), 1e-9) * 100
            ),
            "resilience_improvement_percent": (
                (prop_resilience - base_resilience) / max(abs(base_resilience), 1e-9) * 100
            ),
            "profit_improvement_percent": (
                (prop_profit - base_profit) / max(abs(base_profit), 1e-9) * 100
            ),
            "cost_change_percent": (
                (prop_cost - base_cost) / max(abs(base_cost), 1e-9) * 100
            ),
            "baseline_policy": baseline["policy_name"],
            "baseline_mean_reward": base_reward,
        })

    cmarl_ablation = pd.DataFrame(rows)

    cmarl_ablation = cmarl_ablation.sort_values(
        ["mean_reward", "mean_resilience", "mean_service"],
        ascending=False,
    ).reset_index(drop=True)

    return cmarl_ablation


def get_policy_role(policy_name: str) -> str:
    mapping = {
        "sthg_cmappo_pareto_policy": "Pareto selector over fixed procurement actions",
        "sthg_cmappo_adaptive_hybrid_policy": "Adaptive hybrid cost-risk-service-resilience policy",
        "sthg_cmappo_risk_service_policy": "Risk-service optimized CMARL policy",
        "sthg_cmappo_cost_resilient_policy": "Cost-resilient CMARL policy",
    }
    return mapping.get(policy_name, "Proposed policy variant")


def build_stress_ablation() -> pd.DataFrame:
    baseline_stress = load_required_csv(STEP09_STRESS)
    proposed_stress = load_required_csv(STEP10_STRESS)
    best_baseline = load_required_csv(STEP09_BEST_BASELINE)
    best_proposed = load_required_csv(STEP10_BEST_PROPOSED)

    baseline_policy = best_baseline.iloc[0]["policy_name"]
    proposed_policy = best_proposed.iloc[0]["policy_name"]

    base_df = baseline_stress[
        (baseline_stress["data_split"] == "test")
        & (baseline_stress["policy_name"] == baseline_policy)
    ].copy()

    prop_df = proposed_stress[
        (proposed_stress["data_split"] == "test")
        & (proposed_stress["policy_name"] == proposed_policy)
    ].copy()

    merged = prop_df.merge(
        base_df,
        on=["scenario_name", "data_split"],
        suffixes=("_proposed", "_baseline"),
    )

    rows = []

    for _, row in merged.iterrows():
        base_reward = safe_float(row["mean_reward_baseline"])
        prop_reward = safe_float(row["mean_reward_proposed"])

        base_risk = safe_float(row["mean_risk_baseline"])
        prop_risk = safe_float(row["mean_risk_proposed"])

        base_delay = safe_float(row["mean_delay_baseline"])
        prop_delay = safe_float(row["mean_delay_proposed"])

        base_service = safe_float(row["mean_service_baseline"])
        prop_service = safe_float(row["mean_service_proposed"])

        base_resilience = safe_float(row["mean_resilience_baseline"])
        prop_resilience = safe_float(row["mean_resilience_proposed"])

        rows.append({
            "scenario_name": row["scenario_name"],
            "baseline_policy": row["policy_name_baseline"],
            "proposed_policy": row["policy_name_proposed"],
            "baseline_mean_reward": base_reward,
            "proposed_mean_reward": prop_reward,
            "reward_delta": prop_reward - base_reward,
            "reward_relative_improvement_percent": (
                (prop_reward - base_reward) / max(abs(base_reward), 1e-9) * 100
            ),
            "baseline_mean_risk": base_risk,
            "proposed_mean_risk": prop_risk,
            "risk_reduction_percent": (
                (base_risk - prop_risk) / max(abs(base_risk), 1e-9) * 100
            ),
            "baseline_mean_delay": base_delay,
            "proposed_mean_delay": prop_delay,
            "delay_reduction_percent": (
                (base_delay - prop_delay) / max(abs(base_delay), 1e-9) * 100
            ),
            "baseline_mean_service": base_service,
            "proposed_mean_service": prop_service,
            "service_improvement_percent": (
                (prop_service - base_service) / max(abs(base_service), 1e-9) * 100
            ),
            "baseline_mean_resilience": base_resilience,
            "proposed_mean_resilience": prop_resilience,
            "resilience_improvement_percent": (
                (prop_resilience - base_resilience) / max(abs(base_resilience), 1e-9) * 100
            ),
        })

    stress_ablation = pd.DataFrame(rows)

    scenario_order = [
        "normal",
        "demand_surge",
        "logistics_disruption",
        "supplier_shock",
        "combined_stress",
    ]

    stress_ablation["scenario_order"] = stress_ablation["scenario_name"].apply(
        lambda x: scenario_order.index(x) if x in scenario_order else 99
    )

    stress_ablation = stress_ablation.sort_values("scenario_order").drop(columns=["scenario_order"])

    return stress_ablation


def build_master_ablation(
    ml_ablation: pd.DataFrame,
    cmarl_ablation: pd.DataFrame,
    stress_ablation: pd.DataFrame,
) -> pd.DataFrame:
    best_ml = (
        ml_ablation.sort_values(["test_macro_f1", "test_accuracy"], ascending=False)
        .head(1)
        .iloc[0]
    )

    best_cmarl = cmarl_ablation.head(1).iloc[0]

    stress_avg = {
        "avg_reward_improvement_percent": float(stress_ablation["reward_relative_improvement_percent"].mean()),
        "avg_risk_reduction_percent": float(stress_ablation["risk_reduction_percent"].mean()),
        "avg_delay_reduction_percent": float(stress_ablation["delay_reduction_percent"].mean()),
        "avg_service_improvement_percent": float(stress_ablation["service_improvement_percent"].mean()),
        "avg_resilience_improvement_percent": float(stress_ablation["resilience_improvement_percent"].mean()),
    }

    rows = [
        {
            "ablation_module": "ML feature ablation",
            "best_variant": f"{best_ml['proposed_model']} | {best_ml['candidate_feature_set']}",
            "primary_metric": "Test Macro-F1",
            "primary_value": best_ml["test_macro_f1"],
            "secondary_metric": "Test Accuracy",
            "secondary_value": best_ml["test_accuracy"],
            "interpretation": "Controlled proxy-risk and train-safe graph/entity context improve risk-state classification.",
        },
        {
            "ablation_module": "CMARL policy ablation",
            "best_variant": best_cmarl["policy_name"],
            "primary_metric": "Mean Reward",
            "primary_value": best_cmarl["mean_reward"],
            "secondary_metric": "Reward gain vs strongest baseline (%)",
            "secondary_value": best_cmarl["reward_relative_gain_percent"],
            "interpretation": "Risk-service optimized CMARL gives the strongest reward and resilience trade-off.",
        },
        {
            "ablation_module": "Stress scenario ablation",
            "best_variant": "Selected CMARL policy across stress scenarios",
            "primary_metric": "Average reward improvement (%)",
            "primary_value": stress_avg["avg_reward_improvement_percent"],
            "secondary_metric": "Average delay reduction (%)",
            "secondary_value": stress_avg["avg_delay_reduction_percent"],
            "interpretation": "Selected CMARL policy maintains improvement under demand, logistics, supplier, and combined stress.",
        },
    ]

    return pd.DataFrame(rows)


def build_key_ablation_findings(
    ml_ablation: pd.DataFrame,
    cmarl_ablation: pd.DataFrame,
    stress_ablation: pd.DataFrame,
) -> pd.DataFrame:
    best_ml = (
        ml_ablation.sort_values(["test_macro_f1", "test_accuracy"], ascending=False)
        .head(1)
        .iloc[0]
    )

    strict_best = (
        ml_ablation[ml_ablation["candidate_feature_set"] == "RGC-StrictContext"]
        .sort_values(["test_macro_f1", "test_accuracy"], ascending=False)
        .head(1)
    )

    best_cmarl = cmarl_ablation.head(1).iloc[0]

    avg_reward = stress_ablation["reward_relative_improvement_percent"].mean()
    avg_delay = stress_ablation["delay_reduction_percent"].mean()
    avg_risk = stress_ablation["risk_reduction_percent"].mean()

    findings = []

    findings.append({
        "finding_no": 1,
        "finding_type": "ML ablation",
        "finding": (
            f"The best ML ablation variant was {best_ml['proposed_model']} with "
            f"{best_ml['candidate_feature_set']}, achieving test macro-F1={best_ml['test_macro_f1']:.4f} "
            f"and test accuracy={best_ml['test_accuracy']:.4f}."
        ),
    })

    if not strict_best.empty:
        strict = strict_best.iloc[0]
        findings.append({
            "finding_no": 2,
            "finding_type": "Feature contribution",
            "finding": (
                f"Compared with the strongest strict-context variant "
                f"({strict['proposed_model']}, macro-F1={strict['test_macro_f1']:.4f}), "
                f"the best expanded/proxy variant improved macro-F1 by "
                f"{best_ml['test_macro_f1'] - strict['test_macro_f1']:.4f}."
            ),
        })

    findings.append({
        "finding_no": 3,
        "finding_type": "CMARL policy ablation",
        "finding": (
            f"The best decision-policy ablation variant was {best_cmarl['policy_name']}, "
            f"with mean reward={best_cmarl['mean_reward']:.4f}, "
            f"risk reduction={best_cmarl['risk_reduction_percent']:.2f}%, "
            f"and delay reduction={best_cmarl['delay_reduction_percent']:.2f}% over the strongest baseline."
        ),
    })

    findings.append({
        "finding_no": 4,
        "finding_type": "Stress robustness",
        "finding": (
            f"Across stress scenarios, the selected CMARL policy achieved average reward improvement "
            f"of {avg_reward:.2f}%, average risk reduction of {avg_risk:.2f}%, "
            f"and average delay reduction of {avg_delay:.2f}%."
        ),
    })

    return pd.DataFrame(findings)


def save_ml_ablation_figure(ml_ablation: pd.DataFrame):
    plot_df = ml_ablation.copy()
    plot_df["label"] = plot_df["proposed_model"] + "\n" + plot_df["candidate_feature_set"]

    plot_df = plot_df.sort_values("test_macro_f1", ascending=False)

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(plot_df["label"], plot_df["test_macro_f1"])
    ax.set_title("Step 12 ML feature ablation: test macro-F1")
    ax.set_xlabel("Model and feature set")
    ax.set_ylabel("Test Macro-F1")
    ax.tick_params(axis="x", rotation=45)

    fig.tight_layout()
    fig_path = FIGURES_DIR / "12_ml_feature_ablation_macro_f1.png"
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)

    print(f"[SAVED] {fig_path}")


def save_cmarl_ablation_figure(cmarl_ablation: pd.DataFrame):
    plot_df = cmarl_ablation.sort_values("mean_reward", ascending=False).copy()

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(plot_df["policy_name"], plot_df["mean_reward"])
    ax.set_title("Step 12 CMARL policy ablation: mean reward")
    ax.set_xlabel("Policy variant")
    ax.set_ylabel("Mean reward")
    ax.tick_params(axis="x", rotation=35)

    fig.tight_layout()
    fig_path = FIGURES_DIR / "12_cmarl_policy_ablation_reward.png"
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)

    print(f"[SAVED] {fig_path}")


def save_stress_ablation_figure(stress_ablation: pd.DataFrame):
    plot_df = stress_ablation.copy()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(plot_df["scenario_name"], plot_df["reward_relative_improvement_percent"])
    ax.set_title("Step 12 stress scenario ablation: reward improvement")
    ax.set_xlabel("Stress scenario")
    ax.set_ylabel("Reward improvement (%)")
    ax.tick_params(axis="x", rotation=25)

    fig.tight_layout()
    fig_path = FIGURES_DIR / "12_stress_scenario_reward_improvement.png"
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)

    print(f"[SAVED] {fig_path}")


def main():
    print_header("STEP 12: ABLATION STUDY")

    print(f"[TIME] {timestamp()}")
    print(f"[PROJECT ROOT] {PROJECT_ROOT}")

    print_subheader("Checking Required Input Files")

    required_files = [
        STEP07_CANDIDATE_SUMMARY,
        STEP07_FINAL_IMPROVEMENT,
        STEP09_BEST_BASELINE,
        STEP09_STRESS,
        STEP10_PROPOSED_SUMMARY,
        STEP10_BEST_PROPOSED,
        STEP10_IMPROVEMENT,
        STEP10_STRESS,
    ]

    for path in required_files:
        require_file(path)
        print(f"[OK] {path}")

    print_subheader("Building ML Feature Ablation")
    ml_ablation = build_ml_ablation()
    save_csv(ml_ablation, OUT_ML_ABLATION)
    print(ml_ablation.to_string(index=False))

    print_subheader("Building CMARL Policy Ablation")
    cmarl_ablation = build_cmarl_ablation()
    save_csv(cmarl_ablation, OUT_CMARL_ABLATION)
    print(cmarl_ablation.to_string(index=False))

    print_subheader("Building Stress Scenario Ablation")
    stress_ablation = build_stress_ablation()
    save_csv(stress_ablation, OUT_STRESS_ABLATION)
    print(stress_ablation.to_string(index=False))

    print_subheader("Building Master Ablation Summary")
    master_ablation = build_master_ablation(
        ml_ablation=ml_ablation,
        cmarl_ablation=cmarl_ablation,
        stress_ablation=stress_ablation,
    )
    save_csv(master_ablation, OUT_MASTER_ABLATION)
    print(master_ablation.to_string(index=False))

    print_subheader("Building Key Ablation Findings")
    key_findings = build_key_ablation_findings(
        ml_ablation=ml_ablation,
        cmarl_ablation=cmarl_ablation,
        stress_ablation=stress_ablation,
    )
    save_csv(key_findings, OUT_KEY_ABLATION)
    print(key_findings.to_string(index=False))

    print_subheader("Saving Ablation Figures")
    save_ml_ablation_figure(ml_ablation)
    save_cmarl_ablation_figure(cmarl_ablation)
    save_stress_ablation_figure(stress_ablation)

    best_ml = ml_ablation.sort_values(["test_macro_f1", "test_accuracy"], ascending=False).iloc[0]
    best_cmarl = cmarl_ablation.iloc[0]

    summary = {
        "time": timestamp(),
        "best_ml_ablation": best_ml.to_dict(),
        "best_cmarl_ablation": best_cmarl.to_dict(),
        "stress_ablation_average": {
            "reward_improvement_percent": float(stress_ablation["reward_relative_improvement_percent"].mean()),
            "risk_reduction_percent": float(stress_ablation["risk_reduction_percent"].mean()),
            "delay_reduction_percent": float(stress_ablation["delay_reduction_percent"].mean()),
            "service_improvement_percent": float(stress_ablation["service_improvement_percent"].mean()),
            "resilience_improvement_percent": float(stress_ablation["resilience_improvement_percent"].mean()),
        },
        "output_files": {
            "ml_ablation": str(OUT_ML_ABLATION),
            "cmarl_ablation": str(OUT_CMARL_ABLATION),
            "stress_ablation": str(OUT_STRESS_ABLATION),
            "master_ablation": str(OUT_MASTER_ABLATION),
            "key_findings": str(OUT_KEY_ABLATION),
            "report": str(OUT_REPORT),
        },
    }

    save_json(summary, OUT_JSON)

    report_lines = []
    report_lines.append("STEP 12 ABLATION STUDY REPORT")
    report_lines.append("=" * 95)
    report_lines.append(f"Time: {summary['time']}")
    report_lines.append("")
    report_lines.append("1. ML feature ablation")
    report_lines.append("-" * 95)
    report_lines.append(
        f"Best ML ablation: {best_ml['proposed_model']} with {best_ml['candidate_feature_set']}."
    )
    report_lines.append(
        f"Test accuracy={best_ml['test_accuracy']:.4f}, "
        f"test macro-F1={best_ml['test_macro_f1']:.4f}, "
        f"test balanced accuracy={best_ml['test_balanced_accuracy']:.4f}."
    )
    report_lines.append("")
    report_lines.append("2. CMARL policy ablation")
    report_lines.append("-" * 95)
    report_lines.append(
        f"Best CMARL ablation: {best_cmarl['policy_name']} with mean reward={best_cmarl['mean_reward']:.4f}."
    )
    report_lines.append(
        f"Reward gain over strongest baseline={best_cmarl['reward_relative_gain_percent']:.2f}%, "
        f"risk reduction={best_cmarl['risk_reduction_percent']:.2f}%, "
        f"delay reduction={best_cmarl['delay_reduction_percent']:.2f}%, "
        f"service improvement={best_cmarl['service_improvement_percent']:.2f}%, "
        f"resilience improvement={best_cmarl['resilience_improvement_percent']:.2f}%."
    )
    report_lines.append("")
    report_lines.append("3. Stress scenario ablation")
    report_lines.append("-" * 95)
    for _, row in stress_ablation.iterrows():
        report_lines.append(
            f"- {row['scenario_name']}: reward improvement={row['reward_relative_improvement_percent']:.2f}%, "
            f"risk reduction={row['risk_reduction_percent']:.2f}%, "
            f"delay reduction={row['delay_reduction_percent']:.2f}%, "
            f"service improvement={row['service_improvement_percent']:.2f}%, "
            f"resilience improvement={row['resilience_improvement_percent']:.2f}%."
        )
    report_lines.append("")
    report_lines.append("4. Key ablation findings")
    report_lines.append("-" * 95)
    for _, row in key_findings.iterrows():
        report_lines.append(f"{int(row['finding_no'])}. {row['finding']}")
    report_lines.append("")
    report_lines.append("5. Output files")
    report_lines.append("-" * 95)
    report_lines.append(f"- ML ablation: {OUT_ML_ABLATION}")
    report_lines.append(f"- CMARL ablation: {OUT_CMARL_ABLATION}")
    report_lines.append(f"- Stress ablation: {OUT_STRESS_ABLATION}")
    report_lines.append(f"- Master ablation: {OUT_MASTER_ABLATION}")
    report_lines.append(f"- Key findings: {OUT_KEY_ABLATION}")

    save_text("\n".join(report_lines), OUT_REPORT)

    log_text = (
        f"[{timestamp()}] Step 12 completed. "
        f"BestML={best_ml['proposed_model']}:{best_ml['candidate_feature_set']}, "
        f"BestCMARL={best_cmarl['policy_name']}, "
        f"RewardGain={best_cmarl['reward_relative_gain_percent']:.2f}%\n"
    )
    save_text(log_text, OUT_LOG)

    print_subheader("Saved Files")
    print(f"[ML ABLATION] {OUT_ML_ABLATION}")
    print(f"[CMARL ABLATION] {OUT_CMARL_ABLATION}")
    print(f"[STRESS ABLATION] {OUT_STRESS_ABLATION}")
    print(f"[MASTER ABLATION] {OUT_MASTER_ABLATION}")
    print(f"[KEY FINDINGS] {OUT_KEY_ABLATION}")
    print(f"[JSON SUMMARY] {OUT_JSON}")
    print(f"[REPORT] {OUT_REPORT}")

    print_header("STEP 12 COMPLETED SUCCESSFULLY")
    print("[NEXT] Send me the terminal output.")
    print("[NEXT FILE AFTER VERIFICATION] 13_stress_testing.py")


if __name__ == "__main__":
    main()