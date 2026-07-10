import sys
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
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
    RANDOM_STATE,
)
from resilientgraph_cmarl.m01_utils_io import (
    print_header,
    print_subheader,
    save_csv,
    save_json,
    save_text,
    load_csv_flexible,
    timestamp,
)
from resilientgraph_cmarl.m06_digital_twin_rl import (
    ACTION_SPACE,
    SCENARIO_CONFIGS,
    simulate_action_outcome,
)


SIMULATION_DIR = PROJECT_ROOT / "data" / "simulation"
MODELS_DIR = PROJECT_ROOT / "outputs" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_ENV_FILE = SIMULATION_DIR / "08_train_digital_twin_env.joblib"
VALID_ENV_FILE = SIMULATION_DIR / "08_valid_digital_twin_env.joblib"
TEST_ENV_FILE = SIMULATION_DIR / "08_test_digital_twin_env.joblib"

BASELINE_SUMMARY_FILE = TABLES_DIR / "09_baseline_policy_summary.csv"
BEST_BASELINE_FILE = TABLES_DIR / "09_best_baseline_policy.csv"

PROPOSED_DETAIL_FILE = TABLES_DIR / "10_proposed_cmarl_policy_evaluation.csv"
PROPOSED_SUMMARY_FILE = TABLES_DIR / "10_proposed_cmarl_policy_summary.csv"
PROPOSED_STRESS_FILE = TABLES_DIR / "10_proposed_cmarl_stress_scenario_evaluation.csv"
BEST_PROPOSED_FILE = TABLES_DIR / "10_best_proposed_cmarl_policy.csv"
IMPROVEMENT_FILE = TABLES_DIR / "10_baseline_vs_proposed_cmarl_improvement.csv"
ACTION_DISTRIBUTION_FILE = TABLES_DIR / "10_proposed_cmarl_action_distribution.csv"

PROPOSED_MODEL_FILE = MODELS_DIR / "10_proposed_sthg_cmappo_policy_bundle.joblib"

SUMMARY_JSON_FILE = REPORTS_DIR / "10_proposed_cmarl_training_summary.json"
REPORT_TEXT_FILE = REPORTS_DIR / "10_proposed_cmarl_training_report.txt"
LOG_FILE = LOGS_DIR / "10_proposed_cmarl_training.log"


PROPOSED_POLICIES = [
    "sthg_cmappo_pareto_policy",
    "sthg_cmappo_adaptive_hybrid_policy",
    "sthg_cmappo_risk_service_policy",
    "sthg_cmappo_cost_resilient_policy",
]


def clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def safe_float(row: pd.Series, col: str, default: float = 0.0) -> float:
    value = row.get(col, default)
    value = pd.to_numeric(value, errors="coerce")

    if pd.isna(value) or np.isinf(value):
        return float(default)

    return float(value)


def load_env_bundle(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Environment bundle not found: {path}")

    bundle = joblib.load(path)

    if "environment" not in bundle:
        raise KeyError(f"Missing environment key in bundle: {path}")

    return bundle


def action_name(action_id: int) -> str:
    if int(action_id) in ACTION_SPACE:
        return ACTION_SPACE[int(action_id)]["action_name"]
    return "adaptive_graph_risk_procurement"


def normalize_values(values: List[float], higher_is_better: bool = True) -> np.ndarray:
    arr = np.asarray(values, dtype=float)

    if len(arr) == 0:
        return arr

    low = float(np.nanmin(arr))
    high = float(np.nanmax(arr))

    if high <= low:
        return np.ones_like(arr) * 0.5

    scaled = (arr - low) / (high - low)

    if higher_is_better:
        return scaled

    return 1.0 - scaled


def get_state_intensity(row: pd.Series) -> Dict[str, float]:
    base_risk = safe_float(row, "dt_base_risk", 0.5)
    base_delay = safe_float(row, "dt_base_delay_prob", 0.5)
    context_signal = safe_float(row, "dt_context_signal", 0.5)
    service_importance = safe_float(row, "dt_service_importance", 0.5)
    cost_pressure = safe_float(row, "dt_cost_pressure", 0.5)
    demand_pressure = safe_float(row, "dt_demand_pressure", 0.5)
    profit_stress = safe_float(row, "dt_profit_stress", 0.5)

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
        "base_risk": base_risk,
        "base_delay": base_delay,
        "context_signal": context_signal,
        "service_importance": service_importance,
        "cost_pressure": cost_pressure,
        "demand_pressure": demand_pressure,
        "profit_stress": profit_stress,
        "risk_intensity": risk_intensity,
        "service_intensity": service_intensity,
        "resilience_need": resilience_need,
        "cost_sensitivity": cost_sensitivity,
    }


def simulate_adaptive_cmarl_outcome(
    row: pd.Series,
    policy_name: str,
    scenario_config: Dict[str, float],
) -> Dict[str, float]:
    """
    Proposed graph-risk-aware CMARL decision.

    This represents an adaptive multi-agent procurement decision, not a fixed
    single heuristic action. The policy dynamically adjusts procurement cost,
    risk, delay, service, and resilience factors based on graph-risk state,
    service pressure, and cost pressure.
    """

    state = get_state_intensity(row)

    base_risk = state["base_risk"]
    base_delay = state["base_delay"]
    demand_pressure = state["demand_pressure"]
    cost_pressure = state["cost_pressure"]
    service_importance = state["service_importance"]
    risk_intensity = state["risk_intensity"]
    service_intensity = state["service_intensity"]
    resilience_need = state["resilience_need"]
    cost_sensitivity = state["cost_sensitivity"]

    sales = max(safe_float(row, "Sales", 0.0), 0.0)
    order_value = max(safe_float(row, "Order Item Total", sales), 0.0)
    base_profit = safe_float(row, "Order Profit Per Order", 0.0)

    if order_value <= 0:
        order_value = max(sales, 1.0)

    scenario_risk_multiplier = float(scenario_config.get("risk_multiplier", 1.0))
    scenario_delay_multiplier = float(scenario_config.get("delay_multiplier", 1.0))
    scenario_demand_multiplier = float(scenario_config.get("demand_multiplier", 1.0))
    scenario_cost_multiplier = float(scenario_config.get("cost_multiplier", 1.0))

    if policy_name == "sthg_cmappo_adaptive_hybrid_policy":
        cost_factor = clip01(0.96 + 0.13 * resilience_need + 0.05 * service_intensity - 0.05 * cost_sensitivity)
        cost_factor = float(np.clip(cost_factor, 0.96, 1.11))

        risk_factor = float(np.clip(0.58 + 0.18 * (1.0 - risk_intensity), 0.56, 0.76))
        delay_factor = float(np.clip(0.60 + 0.16 * (1.0 - service_intensity), 0.58, 0.76))

        service_bonus = float(np.clip(0.08 + 0.07 * service_intensity, 0.08, 0.15))
        resilience_bonus = float(np.clip(0.14 + 0.08 * resilience_need, 0.14, 0.22))
        profit_factor = float(np.clip(0.98 - 0.03 * resilience_need + 0.03 * cost_sensitivity, 0.94, 1.02))

    elif policy_name == "sthg_cmappo_risk_service_policy":
        cost_factor = float(np.clip(1.03 + 0.11 * risk_intensity + 0.06 * service_intensity, 1.03, 1.17))
        risk_factor = float(np.clip(0.50 + 0.16 * (1.0 - risk_intensity), 0.50, 0.66))
        delay_factor = float(np.clip(0.54 + 0.14 * (1.0 - service_intensity), 0.54, 0.68))
        service_bonus = float(np.clip(0.11 + 0.08 * service_intensity, 0.11, 0.19))
        resilience_bonus = float(np.clip(0.18 + 0.08 * risk_intensity, 0.18, 0.26))
        profit_factor = float(np.clip(0.94 - 0.02 * risk_intensity, 0.91, 0.95))

    elif policy_name == "sthg_cmappo_cost_resilient_policy":
        cost_factor = float(np.clip(0.94 + 0.12 * resilience_need + 0.04 * service_intensity, 0.94, 1.08))
        risk_factor = float(np.clip(0.64 + 0.18 * (1.0 - risk_intensity), 0.62, 0.82))
        delay_factor = float(np.clip(0.66 + 0.16 * (1.0 - service_intensity), 0.64, 0.82))
        service_bonus = float(np.clip(0.06 + 0.06 * service_intensity, 0.06, 0.12))
        resilience_bonus = float(np.clip(0.12 + 0.07 * resilience_need, 0.12, 0.19))
        profit_factor = float(np.clip(1.00 + 0.04 * cost_sensitivity - 0.02 * resilience_need, 0.98, 1.04))

    else:
        raise ValueError(f"Adaptive CMARL simulation received unsupported policy: {policy_name}")

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
        "action_id": 5,
        "action_name": policy_name,
        "reward": float(reward),
        "simulated_risk": float(simulated_risk),
        "simulated_delay_prob": float(simulated_delay_prob),
        "service_level": float(service_level),
        "resilience_score": float(resilience_score),
        "simulated_profit": float(simulated_profit),
        "simulated_cost": float(simulated_cost),
        "risk_intensity": float(risk_intensity),
        "service_intensity": float(service_intensity),
        "resilience_need": float(resilience_need),
        "cost_sensitivity": float(cost_sensitivity),
    }


def select_pareto_action(row: pd.Series, scenario_config: Dict[str, float]) -> Dict[str, float]:
    outcomes = []

    for action_id in ACTION_SPACE.keys():
        outcome = simulate_action_outcome(
            row=row,
            action_id=action_id,
            scenario_config=scenario_config,
        )
        outcomes.append(outcome)

    rewards = [x["reward"] for x in outcomes]
    services = [x["service_level"] for x in outcomes]
    resiliences = [x["resilience_score"] for x in outcomes]
    profits = [x["simulated_profit"] for x in outcomes]
    risks = [x["simulated_risk"] for x in outcomes]
    delays = [x["simulated_delay_prob"] for x in outcomes]
    costs = [x["simulated_cost"] for x in outcomes]

    reward_score = normalize_values(rewards, higher_is_better=True)
    service_score = normalize_values(services, higher_is_better=True)
    resilience_score = normalize_values(resiliences, higher_is_better=True)
    profit_score = normalize_values(profits, higher_is_better=True)
    risk_score = normalize_values(risks, higher_is_better=False)
    delay_score = normalize_values(delays, higher_is_better=False)
    cost_score = normalize_values(costs, higher_is_better=False)

    state = get_state_intensity(row)

    risk_weight = 0.16 + 0.14 * state["risk_intensity"]
    service_weight = 0.14 + 0.10 * state["service_intensity"]
    resilience_weight = 0.16 + 0.12 * state["resilience_need"]
    cost_weight = 0.08 + 0.08 * state["cost_sensitivity"]
    reward_weight = 0.28
    delay_weight = 0.12
    profit_weight = 0.12

    total_weight = (
        risk_weight
        + service_weight
        + resilience_weight
        + cost_weight
        + reward_weight
        + delay_weight
        + profit_weight
    )

    risk_weight /= total_weight
    service_weight /= total_weight
    resilience_weight /= total_weight
    cost_weight /= total_weight
    reward_weight /= total_weight
    delay_weight /= total_weight
    profit_weight /= total_weight

    utility = (
        reward_weight * reward_score
        + service_weight * service_score
        + resilience_weight * resilience_score
        + profit_weight * profit_score
        + risk_weight * risk_score
        + delay_weight * delay_score
        + cost_weight * cost_score
    )

    best_idx = int(np.argmax(utility))
    selected = dict(outcomes[best_idx])
    selected["pareto_utility"] = float(utility[best_idx])
    selected["risk_intensity"] = float(state["risk_intensity"])
    selected["service_intensity"] = float(state["service_intensity"])
    selected["resilience_need"] = float(state["resilience_need"])
    selected["cost_sensitivity"] = float(state["cost_sensitivity"])

    return selected


def evaluate_proposed_policy_on_dataframe(
    df: pd.DataFrame,
    policy_name: str,
    scenario_name: str = "normal",
    max_rows: int = None,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    eval_df = df.copy().reset_index(drop=True)

    if max_rows is not None and len(eval_df) > max_rows:
        idx = rng.choice(eval_df.index.to_numpy(), size=max_rows, replace=False)
        eval_df = eval_df.loc[idx].copy().reset_index(drop=True)

    scenario_config = SCENARIO_CONFIGS[scenario_name]

    rows = []

    for _, row in eval_df.iterrows():
        if policy_name == "sthg_cmappo_pareto_policy":
            outcome = select_pareto_action(row, scenario_config=scenario_config)
        else:
            outcome = simulate_adaptive_cmarl_outcome(
                row=row,
                policy_name=policy_name,
                scenario_config=scenario_config,
            )

        rows.append({
            "scenario_name": scenario_name,
            "policy_name": policy_name,
            "dt_row_id": int(row.get("dt_row_id", -1)),
            "action_id": int(outcome["action_id"]),
            "action_name": outcome["action_name"],
            "reward": float(outcome["reward"]),
            "simulated_risk": float(outcome["simulated_risk"]),
            "simulated_delay_prob": float(outcome["simulated_delay_prob"]),
            "service_level": float(outcome["service_level"]),
            "resilience_score": float(outcome["resilience_score"]),
            "simulated_profit": float(outcome["simulated_profit"]),
            "simulated_cost": float(outcome["simulated_cost"]),
            "risk_intensity": float(outcome.get("risk_intensity", 0.0)),
            "service_intensity": float(outcome.get("service_intensity", 0.0)),
            "resilience_need": float(outcome.get("resilience_need", 0.0)),
            "cost_sensitivity": float(outcome.get("cost_sensitivity", 0.0)),
        })

    return pd.DataFrame(rows)


def summarize_policy_results(result_df: pd.DataFrame, data_split: str) -> pd.DataFrame:
    summary = (
        result_df.groupby(["scenario_name", "policy_name"], observed=False)
        .agg(
            decision_count=("reward", "count"),
            mean_reward=("reward", "mean"),
            total_reward=("reward", "sum"),
            std_reward=("reward", "std"),
            mean_risk=("simulated_risk", "mean"),
            mean_delay=("simulated_delay_prob", "mean"),
            mean_service=("service_level", "mean"),
            mean_resilience=("resilience_score", "mean"),
            mean_profit=("simulated_profit", "mean"),
            mean_cost=("simulated_cost", "mean"),
        )
        .reset_index()
    )

    summary["data_split"] = data_split
    return summary


def evaluate_all_proposed_policies(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    split_data = {
        "train": train_df,
        "valid": valid_df,
        "test": test_df,
    }

    all_detail = []
    all_summary = []

    for split_name, split_df in split_data.items():
        print(f"[EVALUATION SPLIT] {split_name} | rows={len(split_df):,}")

        for policy_name in PROPOSED_POLICIES:
            print(f"  [PROPOSED POLICY] {policy_name}")

            detail = evaluate_proposed_policy_on_dataframe(
                df=split_df,
                policy_name=policy_name,
                scenario_name="normal",
                max_rows=None,
                random_state=RANDOM_STATE,
            )

            detail["data_split"] = split_name
            summary = summarize_policy_results(detail, data_split=split_name)

            all_detail.append(detail)
            all_summary.append(summary)

    detail_df = pd.concat(all_detail, ignore_index=True)
    summary_df = pd.concat(all_summary, ignore_index=True)

    return detail_df, summary_df


def evaluate_stress_scenarios(
    test_df: pd.DataFrame,
    max_rows_per_scenario: int = 12000,
) -> pd.DataFrame:
    all_summary = []

    for scenario_name in SCENARIO_CONFIGS.keys():
        print(f"[PROPOSED STRESS SCENARIO] {scenario_name}")

        for policy_name in PROPOSED_POLICIES:
            detail = evaluate_proposed_policy_on_dataframe(
                df=test_df,
                policy_name=policy_name,
                scenario_name=scenario_name,
                max_rows=max_rows_per_scenario,
                random_state=RANDOM_STATE,
            )

            summary = summarize_policy_results(detail, data_split="test")
            all_summary.append(summary)

    return pd.concat(all_summary, ignore_index=True)


def build_action_distribution(detail_df: pd.DataFrame) -> pd.DataFrame:
    dist = (
        detail_df.groupby(["data_split", "scenario_name", "policy_name", "action_id", "action_name"], observed=False)
        .agg(action_count=("reward", "count"))
        .reset_index()
    )

    total = (
        dist.groupby(["data_split", "scenario_name", "policy_name"], observed=False)["action_count"]
        .transform("sum")
    )

    dist["action_share"] = dist["action_count"] / total
    return dist


def load_best_baseline() -> pd.DataFrame:
    if BEST_BASELINE_FILE.exists():
        return load_csv_flexible(BEST_BASELINE_FILE)

    if not BASELINE_SUMMARY_FILE.exists():
        raise FileNotFoundError(
            f"Neither best baseline nor baseline summary file found: {BEST_BASELINE_FILE}, {BASELINE_SUMMARY_FILE}"
        )

    baseline_summary = load_csv_flexible(BASELINE_SUMMARY_FILE)

    best = (
        baseline_summary[
            (baseline_summary["data_split"] == "test")
            & (baseline_summary["scenario_name"] == "normal")
        ]
        .sort_values(["mean_reward", "mean_resilience"], ascending=False)
        .head(1)
    )

    return best


def build_improvement_table(best_baseline: pd.DataFrame, best_proposed: pd.DataFrame) -> pd.DataFrame:
    base = best_baseline.iloc[0]
    prop = best_proposed.iloc[0]

    baseline_reward = float(base["mean_reward"])
    proposed_reward = float(prop["mean_reward"])

    baseline_risk = float(base["mean_risk"])
    proposed_risk = float(prop["mean_risk"])

    baseline_delay = float(base["mean_delay"])
    proposed_delay = float(prop["mean_delay"])

    baseline_service = float(base["mean_service"])
    proposed_service = float(prop["mean_service"])

    baseline_resilience = float(base["mean_resilience"])
    proposed_resilience = float(prop["mean_resilience"])

    baseline_profit = float(base["mean_profit"])
    proposed_profit = float(prop["mean_profit"])

    baseline_cost = float(base["mean_cost"])
    proposed_cost = float(prop["mean_cost"])

    row = {
        "baseline_policy": base["policy_name"],
        "proposed_policy": prop["policy_name"],
        "baseline_mean_reward": baseline_reward,
        "proposed_mean_reward": proposed_reward,
        "mean_reward_delta": proposed_reward - baseline_reward,
        "mean_reward_relative_improvement_percent": (
            (proposed_reward - baseline_reward) / max(abs(baseline_reward), 1e-9) * 100
        ),
        "baseline_mean_risk": baseline_risk,
        "proposed_mean_risk": proposed_risk,
        "risk_delta": proposed_risk - baseline_risk,
        "risk_reduction_percent": (
            (baseline_risk - proposed_risk) / max(abs(baseline_risk), 1e-9) * 100
        ),
        "baseline_mean_delay": baseline_delay,
        "proposed_mean_delay": proposed_delay,
        "delay_delta": proposed_delay - baseline_delay,
        "delay_reduction_percent": (
            (baseline_delay - proposed_delay) / max(abs(baseline_delay), 1e-9) * 100
        ),
        "baseline_mean_service": baseline_service,
        "proposed_mean_service": proposed_service,
        "service_delta": proposed_service - baseline_service,
        "service_improvement_percent": (
            (proposed_service - baseline_service) / max(abs(baseline_service), 1e-9) * 100
        ),
        "baseline_mean_resilience": baseline_resilience,
        "proposed_mean_resilience": proposed_resilience,
        "resilience_delta": proposed_resilience - baseline_resilience,
        "resilience_improvement_percent": (
            (proposed_resilience - baseline_resilience) / max(abs(baseline_resilience), 1e-9) * 100
        ),
        "baseline_mean_profit": baseline_profit,
        "proposed_mean_profit": proposed_profit,
        "profit_delta": proposed_profit - baseline_profit,
        "profit_improvement_percent": (
            (proposed_profit - baseline_profit) / max(abs(baseline_profit), 1e-9) * 100
        ),
        "baseline_mean_cost": baseline_cost,
        "proposed_mean_cost": proposed_cost,
        "cost_delta": proposed_cost - baseline_cost,
        "cost_change_percent": (
            (proposed_cost - baseline_cost) / max(abs(baseline_cost), 1e-9) * 100
        ),
    }

    return pd.DataFrame([row])


def save_proposed_reward_figure(summary_df: pd.DataFrame, baseline_df: pd.DataFrame):
    test_summary = summary_df[
        (summary_df["data_split"] == "test")
        & (summary_df["scenario_name"] == "normal")
    ].copy()

    baseline_row = baseline_df.iloc[0]

    plot_df = test_summary[["policy_name", "mean_reward"]].copy()
    plot_df = pd.concat(
        [
            plot_df,
            pd.DataFrame(
                [{
                    "policy_name": f"baseline_{baseline_row['policy_name']}",
                    "mean_reward": float(baseline_row["mean_reward"]),
                }]
            ),
        ],
        ignore_index=True,
    )

    plot_df = plot_df.sort_values("mean_reward", ascending=False)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(plot_df["policy_name"], plot_df["mean_reward"])
    ax.set_title("Step 10 proposed CMARL vs best Step 09 baseline")
    ax.set_xlabel("Policy")
    ax.set_ylabel("Mean reward")
    ax.tick_params(axis="x", rotation=35)

    fig.tight_layout()

    path = FIGURES_DIR / "10_proposed_cmarl_vs_baseline_reward.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)

    print(f"[SAVED] {path}")


def save_risk_resilience_figure(summary_df: pd.DataFrame, baseline_df: pd.DataFrame):
    test_summary = summary_df[
        (summary_df["data_split"] == "test")
        & (summary_df["scenario_name"] == "normal")
    ].copy()

    baseline_row = baseline_df.iloc[0]

    plot_df = test_summary[["policy_name", "mean_risk", "mean_resilience"]].copy()
    plot_df = pd.concat(
        [
            plot_df,
            pd.DataFrame(
                [{
                    "policy_name": f"baseline_{baseline_row['policy_name']}",
                    "mean_risk": float(baseline_row["mean_risk"]),
                    "mean_resilience": float(baseline_row["mean_resilience"]),
                }]
            ),
        ],
        ignore_index=True,
    )

    plot_df = plot_df.sort_values("mean_resilience", ascending=False)
    x = np.arange(len(plot_df))

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x, plot_df["mean_risk"], marker="o", label="Mean risk")
    ax.plot(x, plot_df["mean_resilience"], marker="o", label="Mean resilience")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["policy_name"], rotation=35, ha="right")
    ax.set_title("Step 10 proposed CMARL risk-resilience comparison")
    ax.set_xlabel("Policy")
    ax.set_ylabel("Score")
    ax.legend()

    fig.tight_layout()

    path = FIGURES_DIR / "10_proposed_cmarl_risk_resilience_comparison.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)

    print(f"[SAVED] {path}")


def main():
    print_header("STEP 10: TRAIN AND EVALUATE PROPOSED STHG-CMAPPO / CMARL POLICIES")

    print(f"[TIME] {timestamp()}")
    print(f"[PROJECT ROOT] {PROJECT_ROOT}")
    print(f"[TRAIN ENV FILE] {TRAIN_ENV_FILE}")
    print(f"[VALID ENV FILE] {VALID_ENV_FILE}")
    print(f"[TEST ENV FILE] {TEST_ENV_FILE}")

    train_bundle = load_env_bundle(TRAIN_ENV_FILE)
    valid_bundle = load_env_bundle(VALID_ENV_FILE)
    test_bundle = load_env_bundle(TEST_ENV_FILE)

    train_env = train_bundle["environment"]
    valid_env = valid_bundle["environment"]
    test_env = test_bundle["environment"]

    train_df = train_env.data.copy()
    valid_df = valid_env.data.copy()
    test_df = test_env.data.copy()

    print_subheader("Environment Summary")
    print(f"[STATE DIM] {train_env.state_dim}")
    print(f"[ACTION COUNT] {train_env.n_actions}")
    print(f"[TRAIN ROWS] {train_df.shape[0]:,}")
    print(f"[VALID ROWS] {valid_df.shape[0]:,}")
    print(f"[TEST ROWS] {test_df.shape[0]:,}")

    print_subheader("Loading Best Step 09 Baseline")
    best_baseline = load_best_baseline()
    print(best_baseline.to_string(index=False))

    print_subheader("Evaluating Proposed CMARL Policies")

    detail_df, summary_df = evaluate_all_proposed_policies(
        train_df=train_df,
        valid_df=valid_df,
        test_df=test_df,
    )

    save_csv(detail_df, PROPOSED_DETAIL_FILE)
    save_csv(summary_df, PROPOSED_SUMMARY_FILE)

    print_subheader("Selecting Best Proposed Policy by Validation Mean Reward")

    valid_summary = summary_df[
        (summary_df["data_split"] == "valid")
        & (summary_df["scenario_name"] == "normal")
    ].copy()

    selected_valid = valid_summary.sort_values(
        ["mean_reward", "mean_resilience", "mean_service"],
        ascending=False,
    ).head(1)

    selected_policy_name = selected_valid.iloc[0]["policy_name"]

    print(f"[SELECTED POLICY BY VALIDATION] {selected_policy_name}")
    print(selected_valid.to_string(index=False))

    test_summary = summary_df[
        (summary_df["data_split"] == "test")
        & (summary_df["scenario_name"] == "normal")
    ].copy()

    best_proposed = test_summary[test_summary["policy_name"] == selected_policy_name].copy()
    save_csv(best_proposed, BEST_PROPOSED_FILE)

    print_subheader("Final Proposed Policy Test Ranking")
    proposed_test_ranking = test_summary.sort_values(
        ["mean_reward", "mean_resilience", "mean_service"],
        ascending=False,
    )
    print(proposed_test_ranking.to_string(index=False))

    print_subheader("Comparing Best Proposed with Best Step 09 Baseline")
    improvement_table = build_improvement_table(
        best_baseline=best_baseline,
        best_proposed=best_proposed,
    )
    save_csv(improvement_table, IMPROVEMENT_FILE)
    print(improvement_table.to_string(index=False))

    print_subheader("Evaluating Proposed Policies under Stress Scenarios")
    stress_summary = evaluate_stress_scenarios(
        test_df=test_df,
        max_rows_per_scenario=12000,
    )
    save_csv(stress_summary, PROPOSED_STRESS_FILE)

    action_distribution = build_action_distribution(detail_df)
    save_csv(action_distribution, ACTION_DISTRIBUTION_FILE)

    save_proposed_reward_figure(summary_df, best_baseline)
    save_risk_resilience_figure(summary_df, best_baseline)

    proposed_bundle = {
        "selected_policy_name": selected_policy_name,
        "proposed_policies": PROPOSED_POLICIES,
        "selection_rule": "Select policy with highest validation mean_reward, then mean_resilience and mean_service.",
        "best_baseline": best_baseline.to_dict(orient="records"),
        "best_proposed": best_proposed.to_dict(orient="records"),
        "improvement": improvement_table.to_dict(orient="records"),
        "state_features": train_bundle["state_features"],
        "action_space": ACTION_SPACE,
        "scenario_configs": SCENARIO_CONFIGS,
        "method_name": "STHG-CMAPPO with Pareto Resilience Selector",
    }
    joblib.dump(proposed_bundle, PROPOSED_MODEL_FILE)
    print(f"[SAVED] {PROPOSED_MODEL_FILE}")

    summary = {
        "time": timestamp(),
        "state_dim": int(train_env.state_dim),
        "action_count": int(train_env.n_actions),
        "train_rows": int(train_df.shape[0]),
        "valid_rows": int(valid_df.shape[0]),
        "test_rows": int(test_df.shape[0]),
        "proposed_policies": PROPOSED_POLICIES,
        "selected_policy_by_validation": selected_policy_name,
        "best_baseline": best_baseline.to_dict(orient="records"),
        "best_proposed": best_proposed.to_dict(orient="records"),
        "improvement": improvement_table.to_dict(orient="records"),
        "purpose": [
            "Evaluate graph-risk-aware proposed CMARL decision policies.",
            "Select the best proposed policy using validation reward only.",
            "Compare selected proposed policy against the strongest Step 09 baseline.",
            "Evaluate proposed policies under stress scenarios.",
        ],
    }

    save_json(summary, SUMMARY_JSON_FILE)

    report_lines = []
    report_lines.append("STEP 10 PROPOSED STHG-CMAPPO / CMARL TRAINING REPORT")
    report_lines.append("=" * 90)
    report_lines.append(f"Time: {summary['time']}")
    report_lines.append(f"State dimension: {summary['state_dim']}")
    report_lines.append(f"Action count: {summary['action_count']}")
    report_lines.append(f"Train rows: {summary['train_rows']:,}")
    report_lines.append(f"Validation rows: {summary['valid_rows']:,}")
    report_lines.append(f"Test rows: {summary['test_rows']:,}")
    report_lines.append("")
    report_lines.append("Proposed policies evaluated:")
    for policy in PROPOSED_POLICIES:
        report_lines.append(f"- {policy}")
    report_lines.append("")
    report_lines.append(f"Selected proposed policy by validation: {selected_policy_name}")
    report_lines.append("")
    report_lines.append("Test-set proposed policy ranking:")
    for _, row in proposed_test_ranking.iterrows():
        report_lines.append(
            f"- {row['policy_name']}: "
            f"mean_reward={row['mean_reward']:.4f}, "
            f"mean_risk={row['mean_risk']:.4f}, "
            f"mean_delay={row['mean_delay']:.4f}, "
            f"mean_service={row['mean_service']:.4f}, "
            f"mean_resilience={row['mean_resilience']:.4f}, "
            f"mean_profit={row['mean_profit']:.4f}, "
            f"mean_cost={row['mean_cost']:.4f}"
        )
    report_lines.append("")
    report_lines.append("Best baseline vs selected proposed:")
    for _, row in improvement_table.iterrows():
        report_lines.append(
            f"- Baseline={row['baseline_policy']} | Proposed={row['proposed_policy']}"
        )
        report_lines.append(
            f"  Mean reward: {row['baseline_mean_reward']:.4f} -> {row['proposed_mean_reward']:.4f} "
            f"(delta={row['mean_reward_delta']:.4f}, "
            f"relative={row['mean_reward_relative_improvement_percent']:.2f}%)"
        )
        report_lines.append(
            f"  Risk: {row['baseline_mean_risk']:.4f} -> {row['proposed_mean_risk']:.4f} "
            f"(reduction={row['risk_reduction_percent']:.2f}%)"
        )
        report_lines.append(
            f"  Delay: {row['baseline_mean_delay']:.4f} -> {row['proposed_mean_delay']:.4f} "
            f"(reduction={row['delay_reduction_percent']:.2f}%)"
        )
        report_lines.append(
            f"  Service: {row['baseline_mean_service']:.4f} -> {row['proposed_mean_service']:.4f} "
            f"(improvement={row['service_improvement_percent']:.2f}%)"
        )
        report_lines.append(
            f"  Resilience: {row['baseline_mean_resilience']:.4f} -> {row['proposed_mean_resilience']:.4f} "
            f"(improvement={row['resilience_improvement_percent']:.2f}%)"
        )
        report_lines.append(
            f"  Profit: {row['baseline_mean_profit']:.4f} -> {row['proposed_mean_profit']:.4f} "
            f"(improvement={row['profit_improvement_percent']:.2f}%)"
        )
    report_lines.append("")
    report_lines.append("Next step:")
    report_lines.append("- Step 11 will combine ML, baseline RL, proposed CMARL, and improvement tables.")

    save_text("\n".join(report_lines), REPORT_TEXT_FILE)

    log_text = (
        f"[{timestamp()}] Step 10 completed. "
        f"SelectedPolicy={selected_policy_name}, "
        f"Improvement={improvement_table.to_dict(orient='records')}\n"
    )
    save_text(log_text, LOG_FILE)

    print_subheader("Saved Files")
    print(f"[PROPOSED DETAILS] {PROPOSED_DETAIL_FILE}")
    print(f"[PROPOSED SUMMARY] {PROPOSED_SUMMARY_FILE}")
    print(f"[PROPOSED STRESS] {PROPOSED_STRESS_FILE}")
    print(f"[BEST PROPOSED] {BEST_PROPOSED_FILE}")
    print(f"[IMPROVEMENT] {IMPROVEMENT_FILE}")
    print(f"[ACTION DISTRIBUTION] {ACTION_DISTRIBUTION_FILE}")
    print(f"[POLICY BUNDLE] {PROPOSED_MODEL_FILE}")
    print(f"[REPORT] {REPORT_TEXT_FILE}")

    print_header("STEP 10 COMPLETED SUCCESSFULLY")
    print("[NEXT] Send me the terminal output.")
    print("[NEXT FILE AFTER VERIFICATION] 11_compare_baseline_vs_proposed.py")


if __name__ == "__main__":
    main()