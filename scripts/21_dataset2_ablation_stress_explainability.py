# scripts/21_dataset2_ablation_stress_explainability.py
# ======================================================================================
# STEP 21: DATASET 2 ABLATION, STRESS ROBUSTNESS AND EXPLAINABILITY
# Project: ResilientGraph-CMARL
# Dataset 2: Brazilian E-Commerce Public Dataset by Olist
#
# IMPORTANT:
#   This script does not hardcode final performance values.
#   It reads actual saved outputs from Step 18, Step 19 and Step 20:
#
#   Step 18:
#       data/dataset2_processed/18_olist_graph_risk_features.csv
#
#   Step 19:
#       outputs_dataset2/tables/19_dataset2_ml_model_metrics_all_splits.csv
#       outputs_dataset2/tables/19_dataset2_test_model_ranking.csv
#       outputs_dataset2/tables/19_dataset2_baseline_vs_proposed_rgc_improvement.csv
#
#   Step 20:
#       data/dataset2_simulation/20_dataset2_digital_twin_dataset.csv
#       data/dataset2_simulation/20_dataset2_counterfactual_transition_table_normal.csv
#       outputs_dataset2/tables/20_dataset2_all_policy_evaluation.csv
#       outputs_dataset2/tables/20_dataset2_all_policy_action_distribution.csv
#       outputs_dataset2/tables/20_dataset2_best_baseline_policy.csv
#       outputs_dataset2/tables/20_dataset2_best_proposed_cmarl_policy.csv
#       outputs_dataset2/tables/20_dataset2_baseline_vs_proposed_cmarl_improvement.csv
#       outputs_dataset2/tables/20_dataset2_policy_strength_check.csv
#       outputs_dataset2/tables/20_dataset2_stress_scenario_summary.csv
#
# Outputs:
#   - Actual ML ablation summary
#   - Actual policy ablation summary
#   - Coordination ablation by action
#   - Stress robustness summary
#   - State-feature explainability correlations
#   - Action distribution explainability
#   - Final lock table
#   - Publication-ready figures
#   - Report and JSON summary
# ======================================================================================

from __future__ import annotations

import json
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET2_PROCESSED_DIR = PROJECT_ROOT / "data" / "dataset2_processed"
DATASET2_SIMULATION_DIR = PROJECT_ROOT / "data" / "dataset2_simulation"

OUTPUTS2_DIR = PROJECT_ROOT / "outputs_dataset2"
TABLES2_DIR = OUTPUTS2_DIR / "tables"
FIGURES2_DIR = OUTPUTS2_DIR / "figures"
LOGS2_DIR = OUTPUTS2_DIR / "logs"
REPORTS2_DIR = OUTPUTS2_DIR / "reports"
EXPLAIN2_DIR = OUTPUTS2_DIR / "explainability"
STRESS2_DIR = OUTPUTS2_DIR / "stress_tests"

LOG_FILE = LOGS2_DIR / "21_dataset2_ablation_stress_explainability.log"

STEP18_GRAPH_FILE = DATASET2_PROCESSED_DIR / "18_olist_graph_risk_features.csv"

STEP19_METRICS_FILE = TABLES2_DIR / "19_dataset2_ml_model_metrics_all_splits.csv"
STEP19_TEST_RANKING_FILE = TABLES2_DIR / "19_dataset2_test_model_ranking.csv"
STEP19_IMPROVEMENT_FILE = TABLES2_DIR / "19_dataset2_baseline_vs_proposed_rgc_improvement.csv"

STEP20_DT_FILE = DATASET2_SIMULATION_DIR / "20_dataset2_digital_twin_dataset.csv"
STEP20_TRANSITION_FILE = DATASET2_SIMULATION_DIR / "20_dataset2_counterfactual_transition_table_normal.csv"
STEP20_ALL_POLICY_FILE = TABLES2_DIR / "20_dataset2_all_policy_evaluation.csv"
STEP20_ACTION_DIST_FILE = TABLES2_DIR / "20_dataset2_all_policy_action_distribution.csv"
STEP20_BEST_BASELINE_FILE = TABLES2_DIR / "20_dataset2_best_baseline_policy.csv"
STEP20_BEST_PROPOSED_FILE = TABLES2_DIR / "20_dataset2_best_proposed_cmarl_policy.csv"
STEP20_POLICY_IMPROVEMENT_FILE = TABLES2_DIR / "20_dataset2_baseline_vs_proposed_cmarl_improvement.csv"
STEP20_STRENGTH_CHECK_FILE = TABLES2_DIR / "20_dataset2_policy_strength_check.csv"
STEP20_STRESS_SUMMARY_FILE = TABLES2_DIR / "20_dataset2_stress_scenario_summary.csv"


# --------------------------------------------------------------------------------------
# Logging helpers
# --------------------------------------------------------------------------------------

def ensure_directories() -> None:
    for d in [
        OUTPUTS2_DIR,
        TABLES2_DIR,
        FIGURES2_DIR,
        LOGS2_DIR,
        REPORTS2_DIR,
        EXPLAIN2_DIR,
        STRESS2_DIR,
    ]:
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
# Utility helpers
# --------------------------------------------------------------------------------------

def read_csv_required(path: Path, description: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required file missing for {description}: {path}")

    df = pd.read_csv(path)
    log(f"[LOADED] {description}: {path} | shape={df.shape[0]:,} rows × {df.shape[1]:,} columns")
    return df


def safe_numeric(series: pd.Series, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default)


def safe_pct_improvement(
    baseline_value: float,
    proposed_value: float,
    higher_better: bool = True,
) -> float:
    if baseline_value == 0:
        return np.nan

    if higher_better:
        return (proposed_value - baseline_value) / baseline_value * 100.0

    return (baseline_value - proposed_value) / baseline_value * 100.0


def extract_metric_value(df: pd.DataFrame, metric_name: str, value_col: str = "percent_improvement") -> float:
    sub = df[df["metric"] == metric_name]

    if sub.empty:
        return np.nan

    return float(sub.iloc[0][value_col])


def clean_float(value) -> Optional[float]:
    try:
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


# --------------------------------------------------------------------------------------
# ML ablation from Step 19 actual outputs
# --------------------------------------------------------------------------------------

def build_ml_ablation_summary(
    step19_metrics: pd.DataFrame,
    step19_test_ranking: pd.DataFrame,
    step19_improvement: pd.DataFrame,
) -> pd.DataFrame:
    test_df = step19_metrics[step19_metrics["split"] == "test"].copy()

    rows: List[Dict[str, object]] = []

    if not test_df.empty:
        for model_name in sorted(test_df["model_name"].unique()):
            model_df = test_df[test_df["model_name"] == model_name]

            baseline_df = model_df[model_df["feature_set"].astype(str).str.lower() == "baseline"]
            proposed_df = model_df[model_df["feature_set"].astype(str).str.contains("proposed", case=False, na=False)]

            if baseline_df.empty or proposed_df.empty:
                continue

            baseline_row = baseline_df.sort_values("macro_f1", ascending=False).iloc[0]
            proposed_row = proposed_df.sort_values("macro_f1", ascending=False).iloc[0]

            for metric, higher_better in [
                ("accuracy", True),
                ("balanced_accuracy", True),
                ("macro_f1", True),
                ("weighted_f1", True),
                ("roc_auc_ovr_macro", True),
            ]:
                b = float(baseline_row[metric])
                p = float(proposed_row[metric])

                rows.append(
                    {
                        "ablation_type": "ML feature ablation",
                        "model_name": model_name,
                        "baseline_feature_set": str(baseline_row["feature_set"]),
                        "proposed_feature_set": str(proposed_row["feature_set"]),
                        "metric": metric,
                        "baseline_value": b,
                        "proposed_value": p,
                        "absolute_improvement": p - b,
                        "percent_improvement": safe_pct_improvement(b, p, higher_better=higher_better),
                        "source": "Step 19 actual saved metrics",
                    }
                )

    # Add best-row summary from actual ranking.
    if not step19_test_ranking.empty:
        best_row = step19_test_ranking.iloc[0].to_dict()

        rows.append(
            {
                "ablation_type": "Best Step 19 test model",
                "model_name": best_row.get("model_name", ""),
                "baseline_feature_set": "",
                "proposed_feature_set": best_row.get("feature_set", ""),
                "metric": "best_test_macro_f1",
                "baseline_value": np.nan,
                "proposed_value": clean_float(best_row.get("macro_f1", np.nan)),
                "absolute_improvement": np.nan,
                "percent_improvement": np.nan,
                "source": "Step 19 actual test ranking",
            }
        )

        rows.append(
            {
                "ablation_type": "Best Step 19 test model",
                "model_name": best_row.get("model_name", ""),
                "baseline_feature_set": "",
                "proposed_feature_set": best_row.get("feature_set", ""),
                "metric": "best_test_accuracy",
                "baseline_value": np.nan,
                "proposed_value": clean_float(best_row.get("accuracy", np.nan)),
                "absolute_improvement": np.nan,
                "percent_improvement": np.nan,
                "source": "Step 19 actual test ranking",
            }
        )

    # Include original Step 19 improvement rows, if available.
    if not step19_improvement.empty:
        for row in step19_improvement.itertuples(index=False):
            rows.append(
                {
                    "ablation_type": "Step 19 paired improvement",
                    "model_name": getattr(row, "model_name", ""),
                    "baseline_feature_set": "Baseline",
                    "proposed_feature_set": "Proposed-RGC-Optimized",
                    "metric": getattr(row, "metric", ""),
                    "baseline_value": clean_float(getattr(row, "baseline_value", np.nan)),
                    "proposed_value": clean_float(getattr(row, "proposed_rgc_optimized_value", np.nan)),
                    "absolute_improvement": clean_float(getattr(row, "absolute_improvement", np.nan)),
                    "percent_improvement": clean_float(getattr(row, "percent_improvement", np.nan)),
                    "source": "Step 19 actual improvement table",
                }
            )

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Policy ablation from Step 20 actual outputs
# --------------------------------------------------------------------------------------

def build_policy_ablation_summary(
    policy_improvement: pd.DataFrame,
    strength_check: pd.DataFrame,
    best_baseline: pd.DataFrame,
    best_proposed: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    best_baseline_policy = str(best_baseline.iloc[0]["policy_name"]) if not best_baseline.empty else ""
    best_proposed_policy = str(best_proposed.iloc[0]["policy_name"]) if not best_proposed.empty else ""

    for row in policy_improvement.itertuples(index=False):
        rows.append(
            {
                "ablation_type": "Policy-level ablation",
                "baseline_policy": best_baseline_policy,
                "proposed_policy": best_proposed_policy,
                "metric": getattr(row, "metric", ""),
                "direction": getattr(row, "direction", ""),
                "baseline_value": clean_float(getattr(row, "baseline_value", np.nan)),
                "proposed_value": clean_float(getattr(row, "proposed_value", np.nan)),
                "absolute_improvement": clean_float(getattr(row, "absolute_improvement", np.nan)),
                "percent_improvement": clean_float(getattr(row, "percent_improvement", np.nan)),
                "source": "Step 20 actual baseline-vs-proposed improvement",
            }
        )

    if not strength_check.empty:
        for row in strength_check.itertuples(index=False):
            rows.append(
                {
                    "ablation_type": "Policy strength check",
                    "baseline_policy": best_baseline_policy,
                    "proposed_policy": best_proposed_policy,
                    "metric": getattr(row, "check", ""),
                    "direction": "target_check",
                    "baseline_value": clean_float(getattr(row, "target_percent", np.nan)),
                    "proposed_value": clean_float(getattr(row, "observed_percent", np.nan)),
                    "absolute_improvement": np.nan,
                    "percent_improvement": clean_float(getattr(row, "observed_percent", np.nan)),
                    "source": f"Step 20 actual strength check: {getattr(row, 'status', '')}",
                }
            )

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Coordination ablation using actual transition table
# --------------------------------------------------------------------------------------

def build_coordination_ablation(transition_df: pd.DataFrame) -> pd.DataFrame:
    required_cols = [
        "action_id",
        "action_name",
        "coordinated_action",
        "reward",
        "risk",
        "delay",
        "service",
        "resilience",
        "cost",
        "profit_proxy",
    ]

    missing = [c for c in required_cols if c not in transition_df.columns]

    if missing:
        raise ValueError(f"Transition table missing required columns: {missing}")

    group = (
        transition_df.groupby(["action_id", "action_name", "coordinated_action"], as_index=False)
        .agg(
            rows=("reward", "count"),
            mean_reward=("reward", "mean"),
            mean_risk=("risk", "mean"),
            mean_delay=("delay", "mean"),
            mean_service=("service", "mean"),
            mean_resilience=("resilience", "mean"),
            mean_cost=("cost", "mean"),
            mean_profit_proxy=("profit_proxy", "mean"),
        )
    )

    rows: List[Dict[str, object]] = []

    for action_id in sorted(group["action_id"].unique()):
        action_group = group[group["action_id"] == action_id]

        non_coord = action_group[action_group["coordinated_action"] == 0]
        coord = action_group[action_group["coordinated_action"] == 1]

        if non_coord.empty or coord.empty:
            continue

        b = non_coord.iloc[0]
        p = coord.iloc[0]

        action_name = str(b["action_name"])

        metric_specs = [
            ("mean_reward", True),
            ("mean_risk", False),
            ("mean_delay", False),
            ("mean_service", True),
            ("mean_resilience", True),
            ("mean_cost", False),
            ("mean_profit_proxy", True),
        ]

        for metric, higher_better in metric_specs:
            b_val = float(b[metric])
            p_val = float(p[metric])

            rows.append(
                {
                    "ablation_type": "Coordination-effect ablation",
                    "action_id": int(action_id),
                    "action_name": action_name,
                    "metric": metric,
                    "non_coordinated_value": b_val,
                    "graph_coordinated_value": p_val,
                    "absolute_improvement": (p_val - b_val) if higher_better else (b_val - p_val),
                    "percent_improvement": safe_pct_improvement(
                        baseline_value=b_val,
                        proposed_value=p_val,
                        higher_better=higher_better,
                    ),
                    "rows_non_coordinated": int(b["rows"]),
                    "rows_graph_coordinated": int(p["rows"]),
                    "source": "Step 20 actual counterfactual transition table",
                }
            )

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Stress robustness summary
# --------------------------------------------------------------------------------------

def build_stress_robustness_summary(stress_summary: pd.DataFrame) -> pd.DataFrame:
    required_cols = [
        "scenario",
        "reward_improvement_percent",
        "risk_reduction_percent",
        "delay_reduction_percent",
        "service_improvement_percent",
        "resilience_improvement_percent",
    ]

    missing = [c for c in required_cols if c not in stress_summary.columns]

    if missing:
        raise ValueError(f"Stress summary missing required columns: {missing}")

    rows: List[Dict[str, object]] = []

    metric_cols = [
        "reward_improvement_percent",
        "risk_reduction_percent",
        "delay_reduction_percent",
        "service_improvement_percent",
        "resilience_improvement_percent",
    ]

    for metric in metric_cols:
        s = safe_numeric(stress_summary[metric], np.nan)

        rows.append(
            {
                "metric": metric,
                "scenario_count": int(len(stress_summary)),
                "mean_improvement_percent": float(s.mean()),
                "median_improvement_percent": float(s.median()),
                "min_improvement_percent": float(s.min()),
                "max_improvement_percent": float(s.max()),
                "positive_scenario_count": int((s > 0).sum()),
                "positive_scenario_rate_percent": float((s > 0).mean() * 100),
                "source": "Step 20 actual stress scenario summary",
            }
        )

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Explainability correlations
# --------------------------------------------------------------------------------------

def build_state_explainability_correlations(dt: pd.DataFrame) -> pd.DataFrame:
    state_cols = [
        c for c in dt.columns
        if c.startswith("dt_") and pd.api.types.is_numeric_dtype(dt[c])
    ]

    target_cols = [
        "dt_base_risk",
        "dt_base_delay",
        "dt_base_service",
        "dt_base_resilience",
        "dt_base_cost",
        "dt_profit_proxy",
        "dt_state_pressure_index",
        "dt_coordination_opportunity",
    ]

    target_cols = [c for c in target_cols if c in dt.columns]

    rows: List[Dict[str, object]] = []

    for target in target_cols:
        target_series = safe_numeric(dt[target], np.nan)

        for feature in state_cols:
            if feature == target:
                continue

            feature_series = safe_numeric(dt[feature], np.nan)

            if feature_series.nunique(dropna=True) <= 1 or target_series.nunique(dropna=True) <= 1:
                corr_value = np.nan
            else:
                corr_value = feature_series.corr(target_series, method="spearman")

            rows.append(
                {
                    "explainability_type": "State-feature Spearman correlation",
                    "target_variable": target,
                    "feature_name": feature,
                    "spearman_correlation": corr_value,
                    "abs_spearman_correlation": abs(corr_value) if pd.notna(corr_value) else np.nan,
                    "interpretation": interpret_correlation(corr_value),
                    "source": "Step 20 actual digital-twin state dataset",
                }
            )

    corr_df = pd.DataFrame(rows)

    if not corr_df.empty:
        corr_df = corr_df.sort_values(
            ["target_variable", "abs_spearman_correlation"],
            ascending=[True, False],
        )

    return corr_df


def interpret_correlation(value: float) -> str:
    if pd.isna(value):
        return "not_available"

    abs_value = abs(float(value))

    if abs_value >= 0.70:
        strength = "strong"
    elif abs_value >= 0.40:
        strength = "moderate"
    elif abs_value >= 0.20:
        strength = "weak"
    else:
        strength = "very_weak"

    direction = "positive" if value >= 0 else "negative"

    return f"{strength}_{direction}"


def build_action_distribution_explainability(
    action_dist: pd.DataFrame,
    best_baseline: pd.DataFrame,
    best_proposed: pd.DataFrame,
) -> pd.DataFrame:
    best_baseline_policy = str(best_baseline.iloc[0]["policy_name"]) if not best_baseline.empty else ""
    best_proposed_policy = str(best_proposed.iloc[0]["policy_name"]) if not best_proposed.empty else ""

    sub = action_dist[
        (action_dist["split"] == "test")
        & (action_dist["scenario"] == "normal")
        & (action_dist["policy_name"].isin([best_baseline_policy, best_proposed_policy]))
    ].copy()

    if sub.empty:
        return pd.DataFrame()

    sub["explainability_type"] = "Policy action distribution"
    sub["source"] = "Step 20 actual action distribution table"

    return sub[
        [
            "explainability_type",
            "policy_name",
            "policy_group",
            "scenario",
            "split",
            "action_name",
            "count",
            "percent",
            "source",
        ]
    ].sort_values(["policy_name", "percent"], ascending=[True, False])


# --------------------------------------------------------------------------------------
# Final lock table
# --------------------------------------------------------------------------------------

def build_final_lock_table(
    graph_df: pd.DataFrame,
    dt: pd.DataFrame,
    step19_test_ranking: pd.DataFrame,
    policy_improvement: pd.DataFrame,
    strength_check: pd.DataFrame,
    stress_robustness: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    rows.append(
        {
            "component": "Step 18 graph-risk feature dataset",
            "criterion": "Rows, graph features and missing values",
            "actual_result": f"{graph_df.shape[0]:,} rows × {graph_df.shape[1]:,} columns; missing={int(graph_df.isna().sum().sum()):,}",
            "status": "LOCKED" if graph_df.shape[0] > 0 and int(graph_df.isna().sum().sum()) == 0 else "CHECK",
            "source": "Step 18 actual graph-risk feature file",
        }
    )

    rows.append(
        {
            "component": "Step 20 digital twin dataset",
            "criterion": "Rows, state features and missing values",
            "actual_result": f"{dt.shape[0]:,} rows × {dt.shape[1]:,} columns; missing={int(dt.isna().sum().sum()):,}",
            "status": "LOCKED" if dt.shape[0] > 0 and int(dt.isna().sum().sum()) == 0 else "CHECK",
            "source": "Step 20 actual digital-twin file",
        }
    )

    if not step19_test_ranking.empty:
        best_ml = step19_test_ranking.iloc[0]

        rows.append(
            {
                "component": "Step 19 ML proposed model",
                "criterion": "Best actual test model",
                "actual_result": (
                    f"{best_ml.get('model_name', '')} | {best_ml.get('feature_set', '')} | "
                    f"accuracy={float(best_ml.get('accuracy', np.nan)):.6f}, "
                    f"macro_f1={float(best_ml.get('macro_f1', np.nan)):.6f}"
                ),
                "status": "LOCKED" if str(best_ml.get("feature_set", "")).lower().startswith("proposed") else "CHECK",
                "source": "Step 19 actual test ranking",
            }
        )

    required_policy_metrics = {
        "mean_reward": 5.0,
        "mean_risk": 3.0,
        "mean_delay": 3.0,
        "mean_service": 3.0,
        "mean_resilience": 3.0,
    }

    for metric, target in required_policy_metrics.items():
        observed = extract_metric_value(policy_improvement, metric, "percent_improvement")

        rows.append(
            {
                "component": "Step 20 policy improvement",
                "criterion": f"{metric} target >= {target:.2f}%",
                "actual_result": f"{observed:.6f}%" if pd.notna(observed) else "not_available",
                "status": "LOCKED" if pd.notna(observed) and observed >= target else "CHECK",
                "source": "Step 20 actual policy improvement table",
            }
        )

    if not strength_check.empty:
        pass_rate = float((strength_check["status"].astype(str).str.upper() == "PASS").mean() * 100)

        rows.append(
            {
                "component": "Step 20 strength check",
                "criterion": "All targets pass",
                "actual_result": f"{pass_rate:.2f}% checks passed",
                "status": "LOCKED" if pass_rate == 100 else "CHECK",
                "source": "Step 20 actual strength check table",
            }
        )

    if not stress_robustness.empty:
        reward_row = stress_robustness[stress_robustness["metric"] == "reward_improvement_percent"]

        if not reward_row.empty:
            mean_reward_stress = float(reward_row.iloc[0]["mean_improvement_percent"])
            positive_rate = float(reward_row.iloc[0]["positive_scenario_rate_percent"])

            rows.append(
                {
                    "component": "Step 20 stress robustness",
                    "criterion": "Positive reward improvement across stress scenarios",
                    "actual_result": f"mean={mean_reward_stress:.6f}%, positive_rate={positive_rate:.2f}%",
                    "status": "LOCKED" if positive_rate >= 80 else "CHECK",
                    "source": "Step 20 actual stress summary",
                }
            )

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------

def plot_coordination_reward_by_action(coord_ablation: pd.DataFrame) -> None:
    sub = coord_ablation[coord_ablation["metric"] == "mean_reward"].copy()

    if sub.empty:
        return

    sub = sub.sort_values("percent_improvement", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(sub["action_name"], sub["percent_improvement"])
    ax.set_title("Dataset 2 Coordination Ablation: Reward Improvement by Action")
    ax.set_xlabel("Action")
    ax.set_ylabel("Reward improvement (%)")
    ax.tick_params(axis="x", rotation=35)
    plt.tight_layout()

    path = FIGURES2_DIR / "21_dataset2_coordination_reward_improvement_by_action.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    log(f"[SAVED] {path}")


def plot_stress_reward_improvement(stress_summary: pd.DataFrame) -> None:
    if "scenario" not in stress_summary.columns or "reward_improvement_percent" not in stress_summary.columns:
        return

    plot_df = stress_summary.sort_values("reward_improvement_percent", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(plot_df["scenario"], plot_df["reward_improvement_percent"])
    ax.set_title("Dataset 2 Stress Robustness: Reward Improvement by Scenario")
    ax.set_xlabel("Stress scenario")
    ax.set_ylabel("Reward improvement (%)")
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()

    path = FIGURES2_DIR / "21_dataset2_stress_reward_improvement_by_scenario.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    log(f"[SAVED] {path}")


def plot_policy_action_diversity(action_explain: pd.DataFrame) -> None:
    if action_explain.empty:
        return

    pivot = action_explain.pivot_table(
        index="policy_name",
        columns="action_name",
        values="percent",
        aggfunc="sum",
        fill_value=0,
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    pivot.plot(kind="bar", stacked=True, ax=ax)
    ax.set_title("Dataset 2 Best Baseline vs Proposed Action Distribution")
    ax.set_xlabel("Policy")
    ax.set_ylabel("Action share (%)")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(title="Action", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()

    path = FIGURES2_DIR / "21_dataset2_best_policy_action_distribution.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    log(f"[SAVED] {path}")


def plot_top_state_correlations(corr_df: pd.DataFrame, target_variable: str = "dt_coordination_opportunity") -> None:
    sub = corr_df[corr_df["target_variable"] == target_variable].copy()

    if sub.empty:
        return

    sub = sub.dropna(subset=["abs_spearman_correlation"]).head(12)

    if sub.empty:
        return

    sub = sub.sort_values("abs_spearman_correlation", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(sub["feature_name"], sub["abs_spearman_correlation"])
    ax.set_title("Dataset 2 Explainability: Top State Correlations")
    ax.set_xlabel("Absolute Spearman correlation")
    ax.set_ylabel("State feature")
    plt.tight_layout()

    path = FIGURES2_DIR / "21_dataset2_top_state_feature_correlations.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    log(f"[SAVED] {path}")


def plot_baseline_vs_proposed_metrics(policy_improvement: pd.DataFrame) -> None:
    metric_order = [
        "mean_reward",
        "mean_risk",
        "mean_delay",
        "mean_service",
        "mean_resilience",
        "mean_cost",
        "mean_profit_proxy",
    ]

    sub = policy_improvement[policy_improvement["metric"].isin(metric_order)].copy()

    if sub.empty:
        return

    sub["metric"] = pd.Categorical(sub["metric"], categories=metric_order, ordered=True)
    sub = sub.sort_values("metric")

    x = np.arange(len(sub))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width / 2, sub["baseline_value"], width, label="Best baseline")
    ax.bar(x + width / 2, sub["proposed_value"], width, label="Best proposed")
    ax.set_title("Dataset 2 Best Baseline vs Proposed Policy Metrics")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Actual value")
    ax.set_xticks(x)
    ax.set_xticklabels(sub["metric"], rotation=30, ha="right")
    ax.legend()
    plt.tight_layout()

    path = FIGURES2_DIR / "21_dataset2_best_baseline_vs_proposed_metrics.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    log(f"[SAVED] {path}")


# --------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------

def build_report_text(
    ml_ablation: pd.DataFrame,
    policy_ablation: pd.DataFrame,
    coord_ablation: pd.DataFrame,
    stress_robustness: pd.DataFrame,
    state_corr: pd.DataFrame,
    action_explain: pd.DataFrame,
    lock_table: pd.DataFrame,
) -> str:
    lines: List[str] = []

    lines.append("STEP 21: DATASET 2 ABLATION, STRESS ROBUSTNESS AND EXPLAINABILITY REPORT")
    lines.append("=" * 100)
    lines.append(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("This report is generated only from actual saved Step 18, Step 19 and Step 20 outputs.")
    lines.append("No final result values are manually hardcoded.")
    lines.append("")

    lines.append("Final Lock Table")
    lines.append("-" * 100)
    lines.append(lock_table.to_string(index=False))
    lines.append("")

    lines.append("ML Ablation Summary")
    lines.append("-" * 100)
    if ml_ablation.empty:
        lines.append("No ML ablation rows available.")
    else:
        display = ml_ablation[
            ml_ablation["ablation_type"].isin(["ML feature ablation", "Best Step 19 test model"])
        ].head(30)
        lines.append(display.to_string(index=False))
    lines.append("")

    lines.append("Policy Ablation Summary")
    lines.append("-" * 100)
    if policy_ablation.empty:
        lines.append("No policy ablation rows available.")
    else:
        display = policy_ablation[policy_ablation["ablation_type"] == "Policy-level ablation"].head(20)
        lines.append(display.to_string(index=False))
    lines.append("")

    lines.append("Coordination Ablation by Action")
    lines.append("-" * 100)
    if coord_ablation.empty:
        lines.append("No coordination ablation rows available.")
    else:
        display = coord_ablation[coord_ablation["metric"].isin(["mean_reward", "mean_risk", "mean_delay", "mean_service", "mean_resilience"])].head(40)
        lines.append(display.to_string(index=False))
    lines.append("")

    lines.append("Stress Robustness Summary")
    lines.append("-" * 100)
    if stress_robustness.empty:
        lines.append("No stress robustness rows available.")
    else:
        lines.append(stress_robustness.to_string(index=False))
    lines.append("")

    lines.append("Top Explainability Correlations for Coordination Opportunity")
    lines.append("-" * 100)
    if state_corr.empty:
        lines.append("No state correlation rows available.")
    else:
        display = state_corr[state_corr["target_variable"] == "dt_coordination_opportunity"].head(15)
        lines.append(display.to_string(index=False))
    lines.append("")

    lines.append("Action Distribution Explainability")
    lines.append("-" * 100)
    if action_explain.empty:
        lines.append("No action distribution rows available.")
    else:
        lines.append(action_explain.to_string(index=False))

    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

def main() -> None:
    ensure_directories()
    reset_log()

    print_header("STEP 21: DATASET 2 ABLATION, STRESS ROBUSTNESS AND EXPLAINABILITY")
    log(f"[TIME] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"[PROJECT ROOT] {PROJECT_ROOT}")
    log("[MODE] Actual saved outputs only. No hardcoded final metrics.")

    try:
        print_section("Loading Actual Saved Inputs")

        graph_df = read_csv_required(STEP18_GRAPH_FILE, "Step 18 graph-risk features")

        step19_metrics = read_csv_required(STEP19_METRICS_FILE, "Step 19 ML metrics")
        step19_test_ranking = read_csv_required(STEP19_TEST_RANKING_FILE, "Step 19 test model ranking")
        step19_improvement = read_csv_required(STEP19_IMPROVEMENT_FILE, "Step 19 ML improvement table")

        dt = read_csv_required(STEP20_DT_FILE, "Step 20 digital-twin dataset")
        transition_df = read_csv_required(STEP20_TRANSITION_FILE, "Step 20 counterfactual transition table")
        all_policy = read_csv_required(STEP20_ALL_POLICY_FILE, "Step 20 all policy evaluation")
        action_dist = read_csv_required(STEP20_ACTION_DIST_FILE, "Step 20 all policy action distribution")
        best_baseline = read_csv_required(STEP20_BEST_BASELINE_FILE, "Step 20 best baseline policy")
        best_proposed = read_csv_required(STEP20_BEST_PROPOSED_FILE, "Step 20 best proposed policy")
        policy_improvement = read_csv_required(STEP20_POLICY_IMPROVEMENT_FILE, "Step 20 policy improvement")
        strength_check = read_csv_required(STEP20_STRENGTH_CHECK_FILE, "Step 20 policy strength check")
        stress_summary = read_csv_required(STEP20_STRESS_SUMMARY_FILE, "Step 20 stress scenario summary")

        print_section("Running Step 19 Actual ML Ablation")
        ml_ablation = build_ml_ablation_summary(
            step19_metrics=step19_metrics,
            step19_test_ranking=step19_test_ranking,
            step19_improvement=step19_improvement,
        )
        save_csv(ml_ablation, TABLES2_DIR / "21_dataset2_ml_ablation_from_step19.csv")

        print_section("Running Step 20 Actual Policy Ablation")
        policy_ablation = build_policy_ablation_summary(
            policy_improvement=policy_improvement,
            strength_check=strength_check,
            best_baseline=best_baseline,
            best_proposed=best_proposed,
        )
        save_csv(policy_ablation, TABLES2_DIR / "21_dataset2_policy_ablation_from_step20.csv")

        print_section("Running Coordination Ablation from Actual Transition Table")
        coordination_ablation = build_coordination_ablation(transition_df)
        save_csv(coordination_ablation, TABLES2_DIR / "21_dataset2_coordination_ablation_by_action.csv")

        print_section("Running Stress Robustness Summary from Actual Stress Outputs")
        stress_robustness = build_stress_robustness_summary(stress_summary)
        save_csv(stress_robustness, TABLES2_DIR / "21_dataset2_stress_robustness_summary.csv")

        print_section("Running Explainability from Actual Digital-Twin States")
        state_corr = build_state_explainability_correlations(dt)
        action_explain = build_action_distribution_explainability(
            action_dist=action_dist,
            best_baseline=best_baseline,
            best_proposed=best_proposed,
        )

        save_csv(state_corr, EXPLAIN2_DIR / "21_dataset2_state_feature_explainability_correlations.csv")
        save_csv(action_explain, EXPLAIN2_DIR / "21_dataset2_action_distribution_explainability.csv")

        print_section("Building Final Lock Table")
        lock_table = build_final_lock_table(
            graph_df=graph_df,
            dt=dt,
            step19_test_ranking=step19_test_ranking,
            policy_improvement=policy_improvement,
            strength_check=strength_check,
            stress_robustness=stress_robustness,
        )

        save_csv(lock_table, TABLES2_DIR / "21_dataset2_final_lock_table.csv")
        log(lock_table.to_string(index=False))

        print_section("Generating Figures")
        plot_coordination_reward_by_action(coordination_ablation)
        plot_stress_reward_improvement(stress_summary)
        plot_policy_action_diversity(action_explain)
        plot_top_state_correlations(state_corr, target_variable="dt_coordination_opportunity")
        plot_baseline_vs_proposed_metrics(policy_improvement)

        print_section("Building Report and JSON Summary")

        all_locked = bool((lock_table["status"].astype(str).str.upper() == "LOCKED").all())

        summary = {
            "step": "21_dataset2_ablation_stress_explainability",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "actual_outputs_only": True,
            "manual_metric_hardcoding": False,
            "input_files": {
                "step18_graph": str(STEP18_GRAPH_FILE),
                "step19_metrics": str(STEP19_METRICS_FILE),
                "step19_ranking": str(STEP19_TEST_RANKING_FILE),
                "step20_digital_twin": str(STEP20_DT_FILE),
                "step20_transition": str(STEP20_TRANSITION_FILE),
                "step20_policy_metrics": str(STEP20_ALL_POLICY_FILE),
                "step20_policy_improvement": str(STEP20_POLICY_IMPROVEMENT_FILE),
                "step20_stress_summary": str(STEP20_STRESS_SUMMARY_FILE),
            },
            "output_rows": {
                "ml_ablation_rows": int(len(ml_ablation)),
                "policy_ablation_rows": int(len(policy_ablation)),
                "coordination_ablation_rows": int(len(coordination_ablation)),
                "stress_robustness_rows": int(len(stress_robustness)),
                "state_correlation_rows": int(len(state_corr)),
                "action_explainability_rows": int(len(action_explain)),
                "lock_table_rows": int(len(lock_table)),
            },
            "all_lock_checks_passed": all_locked,
            "lock_table": lock_table.to_dict(orient="records"),
        }

        save_json(summary, REPORTS2_DIR / "21_dataset2_ablation_stress_explainability_summary.json")

        report_text = build_report_text(
            ml_ablation=ml_ablation,
            policy_ablation=policy_ablation,
            coord_ablation=coordination_ablation,
            stress_robustness=stress_robustness,
            state_corr=state_corr,
            action_explain=action_explain,
            lock_table=lock_table,
        )

        save_text(report_text, REPORTS2_DIR / "21_dataset2_ablation_stress_explainability_report.txt")

        print_section("Step 21 Final Terminal Summary")

        log("[ACTUAL OUTPUT MODE] PASS")
        log("[MANUAL METRIC HARDCODING] NO")
        log(f"[LOCK TABLE STATUS] {'LOCKED' if all_locked else 'CHECK_REQUIRED'}")
        log("")

        log("[TOP LOCK TABLE]")
        log(lock_table.to_string(index=False))
        log("")

        log("[STRESS ROBUSTNESS SUMMARY]")
        log(stress_robustness.to_string(index=False))
        log("")

        log("[TOP COORDINATION ABLATION ROWS]")
        coord_display = coordination_ablation[
            coordination_ablation["metric"].isin(["mean_reward", "mean_risk", "mean_delay", "mean_service", "mean_resilience"])
        ].head(25)
        log(coord_display.to_string(index=False))
        log("")

        log("[TOP EXPLAINABILITY CORRELATIONS]")
        explain_display = state_corr[state_corr["target_variable"] == "dt_coordination_opportunity"].head(15)
        log(explain_display.to_string(index=False))

        print_section("Step 21 Completed")
        log("[DONE] Dataset 2 ablation, stress robustness and explainability completed successfully.")
        log(f"[TABLES SAVED] {TABLES2_DIR}")
        log(f"[EXPLAINABILITY SAVED] {EXPLAIN2_DIR}")
        log(f"[FIGURES SAVED] {FIGURES2_DIR}")
        log(f"[REPORT SAVED] {REPORTS2_DIR / '21_dataset2_ablation_stress_explainability_report.txt'}")
        log(f"[SUMMARY SAVED] {REPORTS2_DIR / '21_dataset2_ablation_stress_explainability_summary.json'}")
        log(f"[LOG SAVED] {LOG_FILE}")
        log("")
        log("NEXT STEP:")
        log("py -3.10 -u .\\scripts\\22_multidataset_comparison_dashboard.py")

    except Exception as exc:
        print_section("Step 21 Failed")
        log(f"[ERROR] {exc}")
        log(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()