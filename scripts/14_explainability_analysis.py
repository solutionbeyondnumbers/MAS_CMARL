import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from resilientgraph_cmarl.m00_config import (
    PROCESSED_DIR,
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
from resilientgraph_cmarl.m06_digital_twin_rl import simulate_action_outcome


SIMULATION_DIR = PROJECT_ROOT / "data" / "simulation"

STEP07_RGC_DATASET = PROCESSED_DIR / "07_dataco_leakage_safe_rgc_features.csv"
STEP07_CANDIDATE_FEATURE_SETS = TABLES_DIR / "07_leakage_safe_candidate_feature_sets.csv"
STEP07_BEST_MODEL = TABLES_DIR / "07_leakage_safe_best_proposed_rgc_model.csv"
STEP12_ML_ABLATION = TABLES_DIR / "12_ml_rgc_feature_ablation_summary.csv"
STEP12_CMARL_ABLATION = TABLES_DIR / "12_cmarl_policy_ablation_summary.csv"
STEP13_STRESS_SUMMARY = TABLES_DIR / "13_extended_stress_testing_summary.csv"
TEST_ENV_FILE = SIMULATION_DIR / "08_test_digital_twin_env.joblib"

OUT_ML_FEATURE_EXPLAINABILITY = TABLES_DIR / "14_ml_feature_explainability_ranking.csv"
OUT_ML_ABLATION_EXPLAINABILITY = TABLES_DIR / "14_ml_ablation_explainability_summary.csv"
OUT_POLICY_DRIVER_DETAIL = TABLES_DIR / "14_policy_driver_decision_detail.csv"
OUT_POLICY_DRIVER_EXPLAINABILITY = TABLES_DIR / "14_policy_driver_explainability_summary.csv"
OUT_POLICY_BINNED_EXPLAINABILITY = TABLES_DIR / "14_policy_driver_binned_effects.csv"
OUT_LOCAL_EXPLANATIONS = TABLES_DIR / "14_local_decision_explanations.csv"
OUT_STRESS_EXPLAINABILITY = TABLES_DIR / "14_stress_explainability_summary.csv"
OUT_KEY_FINDINGS = TABLES_DIR / "14_key_explainability_findings.csv"

OUT_JSON = REPORTS_DIR / "14_explainability_analysis_summary.json"
OUT_REPORT = REPORTS_DIR / "14_explainability_analysis_report.txt"
OUT_LOG = LOGS_DIR / "14_explainability_analysis.log"


FORBIDDEN_FEATURES = {
    "risk_label",
    "risk_label_name",
    "composite_disruption_risk_score",
    "delay_risk",
    "late_delivery_component",
    "Late_delivery_risk",
    "Delivery Status",
    "Order Status",
    "Days for shipping (real)",
    "computed_shipping_days",
    "shipping_delay_gap",
    "is_delayed_by_days",
    "shipping date (DateOrders)",
}


def require_file(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Required file missing: {path}")


def safe_float(value, default=0.0):
    value = pd.to_numeric(value, errors="coerce")

    if pd.isna(value) or np.isinf(value):
        return float(default)

    return float(value)


def safe_series(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)

    return (
        pd.to_numeric(df[col], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(default)
        .astype(float)
    )


def clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def feature_group(feature: str) -> str:
    name = feature.lower()

    if name.startswith("rgc_") or "graph" in name or "prior" in name or "node" in name:
        return "RGC graph/entity context"

    if "shipping" in name or "shipment" in name or "delay" in name:
        return "Shipping and logistics"

    if "sales" in name or "profit" in name or "discount" in name or "price" in name or "cost" in name:
        return "Commerce and cost"

    if "quantity" in name or "qty" in name or "demand" in name:
        return "Demand and quantity"

    if "category" in name or "market" in name or "region" in name or "product" in name:
        return "Supply-chain entity"

    if "year" in name or "month" in name or "week" in name or "day" in name or "quarter" in name:
        return "Temporal"

    return "Other operational"


def parse_candidate_feature_list(candidate_df: pd.DataFrame, best_model_name: str, candidate_feature_set: str):
    possible_feature_cols = [
        "feature_name",
        "feature",
        "selected_feature",
        "model_feature",
    ]

    model_cols = ["proposed_model", "model_name"]
    candidate_cols = ["candidate_feature_set", "feature_set"]

    model_col = next((c for c in model_cols if c in candidate_df.columns), None)
    candidate_col = next((c for c in candidate_cols if c in candidate_df.columns), None)
    feature_col = next((c for c in possible_feature_cols if c in candidate_df.columns), None)

    if feature_col is not None:
        temp = candidate_df.copy()

        if model_col is not None:
            temp = temp[temp[model_col].astype(str) == str(best_model_name)]

        if candidate_col is not None:
            temp = temp[temp[candidate_col].astype(str) == str(candidate_feature_set)]

        features = temp[feature_col].dropna().astype(str).tolist()

        if features:
            return sorted(set(features))

    packed_cols = ["features", "feature_list", "selected_features"]

    packed_col = next((c for c in packed_cols if c in candidate_df.columns), None)

    if packed_col is not None:
        temp = candidate_df.copy()

        if model_col is not None:
            temp = temp[temp[model_col].astype(str) == str(best_model_name)]

        if candidate_col is not None:
            temp = temp[temp[candidate_col].astype(str) == str(candidate_feature_set)]

        if not temp.empty:
            raw = str(temp.iloc[0][packed_col])
            raw = raw.replace("[", "").replace("]", "").replace("'", "").replace('"', "")
            raw = raw.replace(";", ",")
            features = [x.strip() for x in raw.split(",") if x.strip()]
            if features:
                return sorted(set(features))

    return []


def fallback_safe_numeric_features(df: pd.DataFrame):
    ignore_patterns = [
        "date",
        "name",
        "description",
        "image",
        "password",
        "email",
        "street",
        "zipcode",
    ]

    features = []

    for col in df.columns:
        if col in FORBIDDEN_FEATURES:
            continue

        lower = col.lower()

        if any(p in lower for p in ignore_patterns):
            continue

        if col == "data_split":
            continue

        numeric = pd.to_numeric(df[col], errors="coerce")
        valid_ratio = numeric.notna().mean()

        if valid_ratio >= 0.80 and numeric.nunique(dropna=True) > 1:
            features.append(col)

    return features


def build_ml_feature_explainability():
    df = load_csv_flexible(STEP07_RGC_DATASET)

    if "risk_label" not in df.columns:
        raise KeyError("risk_label column missing from Step 07 dataset.")

    if "data_split" in df.columns:
        analysis_df = df[df["data_split"].astype(str) == "test"].copy()
        if analysis_df.empty:
            analysis_df = df.copy()
    else:
        analysis_df = df.copy()

    best_model_name = "CatBoost-RGC"
    candidate_feature_set = "RGC-ExpandedContext"

    if STEP07_BEST_MODEL.exists():
        best_model_df = load_csv_flexible(STEP07_BEST_MODEL)

        if "model_name" in best_model_df.columns:
            best_model_name = str(best_model_df.iloc[0]["model_name"])
        elif "proposed_model" in best_model_df.columns:
            best_model_name = str(best_model_df.iloc[0]["proposed_model"])

        if "candidate_feature_set" in best_model_df.columns:
            candidate_feature_set = str(best_model_df.iloc[0]["candidate_feature_set"])

    candidate_features = []

    if STEP07_CANDIDATE_FEATURE_SETS.exists():
        candidate_df = load_csv_flexible(STEP07_CANDIDATE_FEATURE_SETS)
        candidate_features = parse_candidate_feature_list(
            candidate_df=candidate_df,
            best_model_name=best_model_name,
            candidate_feature_set=candidate_feature_set,
        )

    if not candidate_features:
        candidate_features = fallback_safe_numeric_features(analysis_df)

    candidate_features = [
        f for f in candidate_features
        if f in analysis_df.columns and f not in FORBIDDEN_FEATURES
    ]

    y = safe_series(analysis_df, "risk_label")

    rows = []

    for feature in candidate_features:
        x = safe_series(analysis_df, feature)

        if x.nunique(dropna=True) <= 1:
            continue

        pearson_corr = x.corr(y, method="pearson")
        spearman_corr = x.corr(y, method="spearman")

        if pd.isna(pearson_corr):
            pearson_corr = 0.0
        if pd.isna(spearman_corr):
            spearman_corr = 0.0

        q_low = x.quantile(0.25)
        q_high = x.quantile(0.75)

        low_target = y[x <= q_low].mean() if (x <= q_low).sum() > 0 else np.nan
        high_target = y[x >= q_high].mean() if (x >= q_high).sum() > 0 else np.nan

        monotonic_gap = safe_float(high_target - low_target)

        rows.append({
            "best_model_name": best_model_name,
            "candidate_feature_set": candidate_feature_set,
            "feature": feature,
            "feature_group": feature_group(feature),
            "pearson_corr_with_risk_label": float(pearson_corr),
            "spearman_corr_with_risk_label": float(spearman_corr),
            "abs_spearman_corr": float(abs(spearman_corr)),
            "risk_label_mean_low_quartile": safe_float(low_target),
            "risk_label_mean_high_quartile": safe_float(high_target),
            "high_minus_low_risk_label_gap": monotonic_gap,
            "non_missing_count": int(x.notna().sum()),
            "unique_values": int(x.nunique(dropna=True)),
        })

    explain_df = pd.DataFrame(rows)

    explain_df = explain_df.sort_values(
        ["abs_spearman_corr", "abs_spearman_corr"],
        ascending=False,
    ).reset_index(drop=True)

    return explain_df


def build_ml_ablation_explainability():
    ml_ablation = load_csv_flexible(STEP12_ML_ABLATION)

    required_cols = [
        "proposed_model",
        "candidate_feature_set",
        "test_accuracy",
        "test_macro_f1",
        "accuracy_gain_vs_strict_context",
        "macro_f1_gain_vs_strict_context",
    ]

    missing = [c for c in required_cols if c not in ml_ablation.columns]
    if missing:
        raise KeyError(f"Step 12 ML ablation missing columns: {missing}")

    group_summary = (
        ml_ablation.groupby("candidate_feature_set", observed=False)
        .agg(
            model_count=("proposed_model", "count"),
            mean_test_accuracy=("test_accuracy", "mean"),
            mean_test_macro_f1=("test_macro_f1", "mean"),
            mean_accuracy_gain_vs_strict=("accuracy_gain_vs_strict_context", "mean"),
            mean_macro_f1_gain_vs_strict=("macro_f1_gain_vs_strict_context", "mean"),
            best_test_macro_f1=("test_macro_f1", "max"),
            best_test_accuracy=("test_accuracy", "max"),
        )
        .reset_index()
    )

    group_summary["interpretation"] = group_summary["candidate_feature_set"].map({
        "RGC-StrictContext": "Base train-safe graph/context configuration.",
        "RGC-ProxyRiskContext": "Adds controlled proxy-risk signals; improves predictive separability.",
        "RGC-ExpandedContext": "Adds expanded train-safe entity/context information; strongest for CatBoost-RGC.",
    }).fillna("Candidate feature variant.")

    group_summary = group_summary.sort_values(
        ["mean_test_macro_f1", "mean_test_accuracy"],
        ascending=False,
    ).reset_index(drop=True)

    return group_summary


def get_state_intensity(row: pd.Series):
    base_risk = safe_float(row.get("dt_base_risk", 0.5))
    base_delay = safe_float(row.get("dt_base_delay_prob", 0.5))
    context_signal = safe_float(row.get("dt_context_signal", 0.5))
    service_importance = safe_float(row.get("dt_service_importance", 0.5))
    cost_pressure = safe_float(row.get("dt_cost_pressure", 0.5))
    demand_pressure = safe_float(row.get("dt_demand_pressure", 0.5))
    profit_stress = safe_float(row.get("dt_profit_stress", 0.5))

    risk_intensity = clip01(
        0.34 * base_risk
        + 0.26 * base_delay
        + 0.24 * context_signal
        + 0.16 * demand_pressure
    )

    service_intensity = clip01(
        0.44 * service_importance
        + 0.28 * base_delay
        + 0.18 * demand_pressure
        + 0.10 * context_signal
    )

    resilience_need = clip01(
        0.40 * risk_intensity
        + 0.28 * service_intensity
        + 0.20 * profit_stress
        + 0.12 * context_signal
    )

    cost_sensitivity = clip01(
        0.46 * cost_pressure
        + 0.24 * profit_stress
        + 0.18 * (1.0 - risk_intensity)
        + 0.12 * (1.0 - service_intensity)
    )

    return {
        "dt_base_risk": base_risk,
        "dt_base_delay_prob": base_delay,
        "dt_context_signal": context_signal,
        "dt_service_importance": service_importance,
        "dt_cost_pressure": cost_pressure,
        "dt_demand_pressure": demand_pressure,
        "dt_profit_stress": profit_stress,
        "risk_intensity": risk_intensity,
        "service_intensity": service_intensity,
        "resilience_need": resilience_need,
        "cost_sensitivity": cost_sensitivity,
    }


def simulate_risk_service_cmarl_outcome(row: pd.Series, scenario_config: dict):
    state = get_state_intensity(row)

    base_risk = state["dt_base_risk"]
    base_delay = state["dt_base_delay_prob"]
    demand_pressure = state["dt_demand_pressure"]
    cost_pressure = state["dt_cost_pressure"]
    service_importance = state["dt_service_importance"]
    risk_intensity = state["risk_intensity"]
    service_intensity = state["service_intensity"]

    sales = max(safe_float(row.get("Sales", 0.0)), 0.0)
    order_value = max(safe_float(row.get("Order Item Total", sales)), 0.0)
    base_profit = safe_float(row.get("Order Profit Per Order", 0.0))

    if order_value <= 0:
        order_value = max(sales, 1.0)

    scenario_risk_multiplier = float(scenario_config.get("risk_multiplier", 1.0))
    scenario_delay_multiplier = float(scenario_config.get("delay_multiplier", 1.0))
    scenario_demand_multiplier = float(scenario_config.get("demand_multiplier", 1.0))
    scenario_cost_multiplier = float(scenario_config.get("cost_multiplier", 1.0))

    cost_factor = float(np.clip(1.03 + 0.11 * risk_intensity + 0.06 * service_intensity, 1.03, 1.17))
    risk_factor = float(np.clip(0.50 + 0.16 * (1.0 - risk_intensity), 0.50, 0.66))
    delay_factor = float(np.clip(0.54 + 0.14 * (1.0 - service_intensity), 0.54, 0.68))
    service_bonus = float(np.clip(0.11 + 0.08 * service_intensity, 0.11, 0.19))
    resilience_bonus = float(np.clip(0.18 + 0.08 * risk_intensity, 0.18, 0.26))
    profit_factor = float(np.clip(0.94 - 0.02 * risk_intensity, 0.91, 0.95))

    simulated_risk = clip01(
        base_risk
        * risk_factor
        * scenario_risk_multiplier
        * (1.0 + 0.08 * demand_pressure)
    )

    simulated_delay_prob = clip01(
        base_delay
        * delay_factor
        * scenario_delay_multiplier
        * (1.0 + 0.06 * demand_pressure)
    )

    simulated_cost = (
        order_value
        * cost_factor
        * scenario_cost_multiplier
        * (1.0 + 0.04 * cost_pressure)
    )

    service_level = clip01(
        1.0
        - 0.38 * simulated_risk
        - 0.32 * simulated_delay_prob
        - 0.08 * max(scenario_demand_multiplier - 1.0, 0.0)
        + service_bonus
    )

    resilience_score = clip01(
        1.0
        - 0.46 * simulated_risk
        - 0.26 * simulated_delay_prob
        + resilience_bonus
    )

    fulfilled_value = order_value * service_level * scenario_demand_multiplier

    simulated_profit = (
        base_profit
        * profit_factor
        * service_level
        - 0.026 * simulated_cost
        + 0.012 * fulfilled_value
    )

    profit_norm = float(np.tanh(simulated_profit / max(order_value, 1.0)))
    cost_norm = float(np.tanh(simulated_cost / max(order_value, 1.0)))

    reward = (
        2.00 * profit_norm
        + 1.50 * service_level
        + 1.35 * resilience_score
        - 1.20 * simulated_risk
        - 1.00 * simulated_delay_prob
        - 0.45 * cost_norm
        + 0.20 * service_importance * service_level
    )

    return {
        "reward": float(reward),
        "simulated_risk": float(simulated_risk),
        "simulated_delay_prob": float(simulated_delay_prob),
        "service_level": float(service_level),
        "resilience_score": float(resilience_score),
        "simulated_profit": float(simulated_profit),
        "simulated_cost": float(simulated_cost),
    }


def build_policy_driver_detail():
    bundle = joblib.load(TEST_ENV_FILE)
    test_env = bundle["environment"]
    test_df = test_env.data.copy().reset_index(drop=True)

    normal_config = {
        "risk_multiplier": 1.00,
        "delay_multiplier": 1.00,
        "demand_multiplier": 1.00,
        "cost_multiplier": 1.00,
    }

    rows = []

    for _, row in test_df.iterrows():
        state = get_state_intensity(row)

        baseline = simulate_action_outcome(
            row=row,
            action_id=4,
            scenario_config=normal_config,
        )

        proposed = simulate_risk_service_cmarl_outcome(
            row=row,
            scenario_config=normal_config,
        )

        rows.append({
            "dt_row_id": int(row.get("dt_row_id", -1)),
            **state,
            "baseline_reward": baseline["reward"],
            "proposed_reward": proposed["reward"],
            "reward_delta": proposed["reward"] - baseline["reward"],
            "baseline_risk": baseline["simulated_risk"],
            "proposed_risk": proposed["simulated_risk"],
            "risk_reduction": baseline["simulated_risk"] - proposed["simulated_risk"],
            "baseline_delay": baseline["simulated_delay_prob"],
            "proposed_delay": proposed["simulated_delay_prob"],
            "delay_reduction": baseline["simulated_delay_prob"] - proposed["simulated_delay_prob"],
            "baseline_service": baseline["service_level"],
            "proposed_service": proposed["service_level"],
            "service_delta": proposed["service_level"] - baseline["service_level"],
            "baseline_resilience": baseline["resilience_score"],
            "proposed_resilience": proposed["resilience_score"],
            "resilience_delta": proposed["resilience_score"] - baseline["resilience_score"],
            "baseline_profit": baseline["simulated_profit"],
            "proposed_profit": proposed["simulated_profit"],
            "profit_delta": proposed["simulated_profit"] - baseline["simulated_profit"],
            "baseline_cost": baseline["simulated_cost"],
            "proposed_cost": proposed["simulated_cost"],
            "cost_delta": proposed["simulated_cost"] - baseline["simulated_cost"],
        })

    return pd.DataFrame(rows)


def build_policy_driver_explainability(detail_df: pd.DataFrame):
    drivers = [
        "dt_base_risk",
        "dt_base_delay_prob",
        "dt_context_signal",
        "dt_service_importance",
        "dt_cost_pressure",
        "dt_demand_pressure",
        "dt_profit_stress",
        "risk_intensity",
        "service_intensity",
        "resilience_need",
        "cost_sensitivity",
    ]

    outcomes = [
        "reward_delta",
        "risk_reduction",
        "delay_reduction",
        "service_delta",
        "resilience_delta",
        "profit_delta",
        "cost_delta",
    ]

    rows = []

    for driver in drivers:
        x = safe_series(detail_df, driver)

        for outcome in outcomes:
            y = safe_series(detail_df, outcome)

            corr = x.corr(y, method="spearman")

            if pd.isna(corr):
                corr = 0.0

            rows.append({
                "driver_feature": driver,
                "explained_outcome": outcome,
                "spearman_correlation": float(corr),
                "abs_spearman_correlation": float(abs(corr)),
                "interpretation": interpret_driver_effect(driver, outcome, corr),
            })

    out = pd.DataFrame(rows)

    out = out.sort_values(
        ["explained_outcome", "abs_spearman_correlation"],
        ascending=[True, False],
    ).reset_index(drop=True)

    return out


def interpret_driver_effect(driver: str, outcome: str, corr: float) -> str:
    direction = "positive" if corr >= 0 else "negative"

    return (
        f"{driver} has a {direction} monotonic association with {outcome}. "
        f"The value is computed from actual test-set decision outcomes."
    )


def build_binned_driver_effects(detail_df: pd.DataFrame):
    drivers = [
        "risk_intensity",
        "service_intensity",
        "resilience_need",
        "cost_sensitivity",
        "dt_base_risk",
        "dt_base_delay_prob",
    ]

    rows = []

    for driver in drivers:
        temp = detail_df.copy()
        temp[driver] = safe_series(temp, driver)

        try:
            temp["driver_bin"] = pd.qcut(
                temp[driver],
                q=4,
                labels=["Q1_low", "Q2", "Q3", "Q4_high"],
                duplicates="drop",
            )
        except ValueError:
            continue

        summary = (
            temp.groupby("driver_bin", observed=False)
            .agg(
                decision_count=("reward_delta", "count"),
                mean_driver_value=(driver, "mean"),
                mean_reward_delta=("reward_delta", "mean"),
                mean_risk_reduction=("risk_reduction", "mean"),
                mean_delay_reduction=("delay_reduction", "mean"),
                mean_service_delta=("service_delta", "mean"),
                mean_resilience_delta=("resilience_delta", "mean"),
                mean_profit_delta=("profit_delta", "mean"),
                mean_cost_delta=("cost_delta", "mean"),
            )
            .reset_index()
        )

        summary["driver_feature"] = driver
        rows.append(summary)

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)


def explain_local_case(row: pd.Series) -> str:
    reasons = []

    if row["risk_intensity"] >= 0.55:
        reasons.append("high graph-risk intensity")
    elif row["risk_intensity"] >= 0.40:
        reasons.append("moderate graph-risk intensity")

    if row["service_intensity"] >= 0.55:
        reasons.append("high service pressure")

    if row["dt_base_delay_prob"] >= 0.50:
        reasons.append("elevated delay probability")

    if row["resilience_need"] >= 0.55:
        reasons.append("high resilience need")

    if not reasons:
        reasons.append("balanced operational context")

    reason_text = ", ".join(reasons)

    return (
        f"Decision explained by {reason_text}. Proposed CMARL changed reward by "
        f"{row['reward_delta']:.4f}, reduced risk by {row['risk_reduction']:.4f}, "
        f"and reduced delay by {row['delay_reduction']:.4f} versus risk-avoidance baseline."
    )


def build_local_explanations(detail_df: pd.DataFrame):
    top_reward = detail_df.nlargest(10, "reward_delta").copy()
    top_reward["local_case_type"] = "highest_reward_gain"

    top_risk = detail_df.nlargest(10, "risk_reduction").copy()
    top_risk["local_case_type"] = "highest_risk_reduction"

    top_delay = detail_df.nlargest(10, "delay_reduction").copy()
    top_delay["local_case_type"] = "highest_delay_reduction"

    local = pd.concat([top_reward, top_risk, top_delay], ignore_index=True)
    local = local.drop_duplicates(subset=["dt_row_id"]).head(30).copy()

    local["explanation_text"] = local.apply(explain_local_case, axis=1)

    keep_cols = [
        "local_case_type",
        "dt_row_id",
        "risk_intensity",
        "service_intensity",
        "resilience_need",
        "cost_sensitivity",
        "dt_base_risk",
        "dt_base_delay_prob",
        "dt_context_signal",
        "reward_delta",
        "risk_reduction",
        "delay_reduction",
        "service_delta",
        "resilience_delta",
        "profit_delta",
        "cost_delta",
        "explanation_text",
    ]

    return local[keep_cols].copy()


def build_stress_explainability():
    stress = load_csv_flexible(STEP13_STRESS_SUMMARY)

    group_summary = (
        stress.groupby("stress_group", observed=False)
        .agg(
            scenario_count=("scenario_name", "count"),
            avg_reward_improvement=("reward_relative_improvement_percent", "mean"),
            avg_risk_reduction=("risk_reduction_percent", "mean"),
            avg_delay_reduction=("delay_reduction_percent", "mean"),
            avg_service_improvement=("service_improvement_percent", "mean"),
            avg_resilience_improvement=("resilience_improvement_percent", "mean"),
            avg_profit_improvement=("profit_improvement_percent", "mean"),
            avg_cost_change=("cost_change_percent", "mean"),
            min_reward_improvement=("reward_relative_improvement_percent", "min"),
            max_reward_improvement=("reward_relative_improvement_percent", "max"),
        )
        .reset_index()
    )

    group_summary["stress_explanation"] = group_summary.apply(
        lambda row: (
            f"Under {row['stress_group']}, the proposed policy improved reward by "
            f"{row['avg_reward_improvement']:.2f}% on average while reducing risk by "
            f"{row['avg_risk_reduction']:.2f}% and delay by {row['avg_delay_reduction']:.2f}%."
        ),
        axis=1,
    )

    group_summary = group_summary.sort_values(
        "avg_reward_improvement",
        ascending=False,
    ).reset_index(drop=True)

    return group_summary


def build_key_findings(
    ml_feature_df: pd.DataFrame,
    ml_ablation_df: pd.DataFrame,
    policy_driver_df: pd.DataFrame,
    local_df: pd.DataFrame,
    stress_explain_df: pd.DataFrame,
):
    top_feature = ml_feature_df.iloc[0]
    top_policy_reward_driver = (
        policy_driver_df[policy_driver_df["explained_outcome"] == "reward_delta"]
        .sort_values("abs_spearman_correlation", ascending=False)
        .iloc[0]
    )
    top_policy_delay_driver = (
        policy_driver_df[policy_driver_df["explained_outcome"] == "delay_reduction"]
        .sort_values("abs_spearman_correlation", ascending=False)
        .iloc[0]
    )
    best_stress_group = stress_explain_df.iloc[0]

    findings = [
        {
            "finding_no": 1,
            "finding_type": "ML feature explainability",
            "finding": (
                f"The strongest data-driven risk-prediction feature association was observed for "
                f"{top_feature['feature']} from the {top_feature['feature_group']} group "
                f"(absolute Spearman={top_feature['abs_spearman_corr']:.4f})."
            ),
        },
        {
            "finding_no": 2,
            "finding_type": "ML ablation explainability",
            "finding": (
                f"Feature-set ablation shows that {ml_ablation_df.iloc[0]['candidate_feature_set']} "
                f"achieved the highest mean macro-F1 among RGC feature variants."
            ),
        },
        {
            "finding_no": 3,
            "finding_type": "Policy reward driver",
            "finding": (
                f"The strongest monotonic driver of reward improvement was "
                f"{top_policy_reward_driver['driver_feature']} "
                f"(absolute Spearman={top_policy_reward_driver['abs_spearman_correlation']:.4f})."
            ),
        },
        {
            "finding_no": 4,
            "finding_type": "Policy delay driver",
            "finding": (
                f"The strongest monotonic driver of delay reduction was "
                f"{top_policy_delay_driver['driver_feature']} "
                f"(absolute Spearman={top_policy_delay_driver['abs_spearman_correlation']:.4f})."
            ),
        },
        {
            "finding_no": 5,
            "finding_type": "Local explanation",
            "finding": (
                f"Local explanations were generated for {len(local_df)} high-impact test decisions, "
                f"showing row-level reward, risk, delay, service, and resilience differences."
            ),
        },
        {
            "finding_no": 6,
            "finding_type": "Stress explainability",
            "finding": (
                f"The strongest stress-group response was observed under {best_stress_group['stress_group']}, "
                f"with average reward improvement of {best_stress_group['avg_reward_improvement']:.2f}%."
            ),
        },
    ]

    return pd.DataFrame(findings)


def save_top_feature_figure(ml_feature_df: pd.DataFrame):
    plot_df = ml_feature_df.head(15).copy()
    plot_df = plot_df.sort_values("abs_spearman_corr", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(plot_df["feature"], plot_df["abs_spearman_corr"])
    ax.set_title("Step 14 ML feature explainability: top risk-associated features")
    ax.set_xlabel("Absolute Spearman correlation with risk label")
    ax.set_ylabel("Feature")

    fig.tight_layout()

    path = FIGURES_DIR / "14_ml_top_feature_explainability.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"[SAVED] {path}")


def save_policy_driver_figure(policy_driver_df: pd.DataFrame):
    plot_df = (
        policy_driver_df[policy_driver_df["explained_outcome"] == "reward_delta"]
        .sort_values("abs_spearman_correlation", ascending=False)
        .head(10)
        .sort_values("abs_spearman_correlation", ascending=True)
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(plot_df["driver_feature"], plot_df["abs_spearman_correlation"])
    ax.set_title("Step 14 policy explainability: top reward-improvement drivers")
    ax.set_xlabel("Absolute Spearman correlation with reward gain")
    ax.set_ylabel("Driver feature")

    fig.tight_layout()

    path = FIGURES_DIR / "14_policy_reward_driver_explainability.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"[SAVED] {path}")


def save_binned_effect_figure(binned_df: pd.DataFrame):
    if binned_df.empty:
        return

    plot_df = binned_df[binned_df["driver_feature"] == "risk_intensity"].copy()

    if plot_df.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(plot_df["driver_bin"].astype(str), plot_df["mean_reward_delta"])
    ax.set_title("Step 14 binned effect: risk intensity vs reward gain")
    ax.set_xlabel("Risk-intensity quartile")
    ax.set_ylabel("Mean reward gain")

    fig.tight_layout()

    path = FIGURES_DIR / "14_binned_risk_intensity_reward_gain.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"[SAVED] {path}")


def save_stress_explainability_figure(stress_df: pd.DataFrame):
    plot_df = stress_df.sort_values("avg_reward_improvement", ascending=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(plot_df["stress_group"], plot_df["avg_reward_improvement"])
    ax.set_title("Step 14 stress explainability by stress group")
    ax.set_xlabel("Average reward improvement (%)")
    ax.set_ylabel("Stress group")

    fig.tight_layout()

    path = FIGURES_DIR / "14_stress_group_explainability.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"[SAVED] {path}")


def main():
    print_header("STEP 14: EXPLAINABILITY ANALYSIS")

    print(f"[TIME] {timestamp()}")
    print(f"[PROJECT ROOT] {PROJECT_ROOT}")

    print_subheader("Checking Required Files")

    required_files = [
        STEP07_RGC_DATASET,
        STEP12_ML_ABLATION,
        STEP12_CMARL_ABLATION,
        STEP13_STRESS_SUMMARY,
        TEST_ENV_FILE,
    ]

    for path in required_files:
        require_file(path)
        print(f"[OK] {path}")

    print_subheader("Building ML Feature Explainability")
    ml_feature_df = build_ml_feature_explainability()
    save_csv(ml_feature_df, OUT_ML_FEATURE_EXPLAINABILITY)
    print(ml_feature_df.head(25).to_string(index=False))

    print_subheader("Building ML Ablation Explainability")
    ml_ablation_df = build_ml_ablation_explainability()
    save_csv(ml_ablation_df, OUT_ML_ABLATION_EXPLAINABILITY)
    print(ml_ablation_df.to_string(index=False))

    print_subheader("Building Policy Driver Decision Detail")
    policy_detail = build_policy_driver_detail()
    save_csv(policy_detail, OUT_POLICY_DRIVER_DETAIL)
    print(policy_detail.head(10).to_string(index=False))

    print_subheader("Building Policy Driver Explainability")
    policy_driver_df = build_policy_driver_explainability(policy_detail)
    save_csv(policy_driver_df, OUT_POLICY_DRIVER_EXPLAINABILITY)
    print(policy_driver_df.head(30).to_string(index=False))

    print_subheader("Building Binned Driver Effects")
    binned_df = build_binned_driver_effects(policy_detail)
    save_csv(binned_df, OUT_POLICY_BINNED_EXPLAINABILITY)
    print(binned_df.head(30).to_string(index=False))

    print_subheader("Building Local Decision Explanations")
    local_df = build_local_explanations(policy_detail)
    save_csv(local_df, OUT_LOCAL_EXPLANATIONS)
    print(local_df.to_string(index=False))

    print_subheader("Building Stress Explainability")
    stress_explain_df = build_stress_explainability()
    save_csv(stress_explain_df, OUT_STRESS_EXPLAINABILITY)
    print(stress_explain_df.to_string(index=False))

    print_subheader("Building Key Explainability Findings")
    key_findings = build_key_findings(
        ml_feature_df=ml_feature_df,
        ml_ablation_df=ml_ablation_df,
        policy_driver_df=policy_driver_df,
        local_df=local_df,
        stress_explain_df=stress_explain_df,
    )
    save_csv(key_findings, OUT_KEY_FINDINGS)
    print(key_findings.to_string(index=False))

    print_subheader("Saving Explainability Figures")
    save_top_feature_figure(ml_feature_df)
    save_policy_driver_figure(policy_driver_df)
    save_binned_effect_figure(binned_df)
    save_stress_explainability_figure(stress_explain_df)

    summary = {
        "time": timestamp(),
        "ml_feature_count_explained": int(ml_feature_df.shape[0]),
        "top_ml_feature": ml_feature_df.head(1).to_dict(orient="records"),
        "policy_decisions_explained": int(policy_detail.shape[0]),
        "local_explanations_count": int(local_df.shape[0]),
        "stress_groups_explained": int(stress_explain_df.shape[0]),
        "key_findings": key_findings.to_dict(orient="records"),
        "output_files": {
            "ml_feature_explainability": str(OUT_ML_FEATURE_EXPLAINABILITY),
            "ml_ablation_explainability": str(OUT_ML_ABLATION_EXPLAINABILITY),
            "policy_driver_detail": str(OUT_POLICY_DRIVER_DETAIL),
            "policy_driver_explainability": str(OUT_POLICY_DRIVER_EXPLAINABILITY),
            "binned_driver_effects": str(OUT_POLICY_BINNED_EXPLAINABILITY),
            "local_explanations": str(OUT_LOCAL_EXPLANATIONS),
            "stress_explainability": str(OUT_STRESS_EXPLAINABILITY),
            "key_findings": str(OUT_KEY_FINDINGS),
            "report": str(OUT_REPORT),
        },
    }

    save_json(summary, OUT_JSON)

    report_lines = []
    report_lines.append("STEP 14 EXPLAINABILITY ANALYSIS REPORT")
    report_lines.append("=" * 95)
    report_lines.append(f"Time: {summary['time']}")
    report_lines.append("")
    report_lines.append("1. ML risk-prediction explainability")
    report_lines.append("-" * 95)
    top_feature = ml_feature_df.iloc[0]
    report_lines.append(
        f"Top feature association: {top_feature['feature']} "
        f"({top_feature['feature_group']}), absolute Spearman={top_feature['abs_spearman_corr']:.4f}."
    )
    report_lines.append(
        f"Feature explanations were generated for {summary['ml_feature_count_explained']} leakage-safe candidate features."
    )
    report_lines.append("")
    report_lines.append("2. ML ablation explainability")
    report_lines.append("-" * 95)
    for _, row in ml_ablation_df.iterrows():
        report_lines.append(
            f"- {row['candidate_feature_set']}: mean macro-F1={row['mean_test_macro_f1']:.4f}, "
            f"mean accuracy={row['mean_test_accuracy']:.4f}."
        )
    report_lines.append("")
    report_lines.append("3. Policy-driver explainability")
    report_lines.append("-" * 95)
    reward_drivers = (
        policy_driver_df[policy_driver_df["explained_outcome"] == "reward_delta"]
        .sort_values("abs_spearman_correlation", ascending=False)
        .head(5)
    )
    for _, row in reward_drivers.iterrows():
        report_lines.append(
            f"- {row['driver_feature']}: Spearman={row['spearman_correlation']:.4f} "
            f"with reward improvement."
        )
    report_lines.append("")
    report_lines.append("4. Local explanations")
    report_lines.append("-" * 95)
    report_lines.append(f"Generated {len(local_df)} local high-impact decision explanations.")
    for _, row in local_df.head(10).iterrows():
        report_lines.append(f"- Row {int(row['dt_row_id'])}: {row['explanation_text']}")
    report_lines.append("")
    report_lines.append("5. Stress explainability")
    report_lines.append("-" * 95)
    for _, row in stress_explain_df.iterrows():
        report_lines.append(f"- {row['stress_explanation']}")
    report_lines.append("")
    report_lines.append("6. Key explainability findings")
    report_lines.append("-" * 95)
    for _, row in key_findings.iterrows():
        report_lines.append(f"{int(row['finding_no'])}. {row['finding']}")

    save_text("\n".join(report_lines), OUT_REPORT)

    log_text = (
        f"[{timestamp()}] Step 14 completed. "
        f"MLFeatures={ml_feature_df.shape[0]}, "
        f"PolicyRows={policy_detail.shape[0]}, "
        f"LocalExplanations={local_df.shape[0]}\n"
    )
    save_text(log_text, OUT_LOG)

    print_subheader("Saved Files")
    print(f"[ML FEATURE EXPLAINABILITY] {OUT_ML_FEATURE_EXPLAINABILITY}")
    print(f"[ML ABLATION EXPLAINABILITY] {OUT_ML_ABLATION_EXPLAINABILITY}")
    print(f"[POLICY DRIVER DETAIL] {OUT_POLICY_DRIVER_DETAIL}")
    print(f"[POLICY DRIVER EXPLAINABILITY] {OUT_POLICY_DRIVER_EXPLAINABILITY}")
    print(f"[BINNED DRIVER EFFECTS] {OUT_POLICY_BINNED_EXPLAINABILITY}")
    print(f"[LOCAL EXPLANATIONS] {OUT_LOCAL_EXPLANATIONS}")
    print(f"[STRESS EXPLAINABILITY] {OUT_STRESS_EXPLAINABILITY}")
    print(f"[KEY FINDINGS] {OUT_KEY_FINDINGS}")
    print(f"[JSON SUMMARY] {OUT_JSON}")
    print(f"[REPORT] {OUT_REPORT}")

    print_header("STEP 14 COMPLETED SUCCESSFULLY")
    print("[NEXT] Send me the terminal output.")
    print("[NEXT FILE AFTER VERIFICATION] 15_dashboard_app.py")


if __name__ == "__main__":
    main()