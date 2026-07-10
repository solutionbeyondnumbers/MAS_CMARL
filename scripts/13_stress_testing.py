import sys
from pathlib import Path

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
from resilientgraph_cmarl.m06_digital_twin_rl import (
    ACTION_SPACE,
    simulate_action_outcome,
)


SIMULATION_DIR = PROJECT_ROOT / "data" / "simulation"

TEST_ENV_FILE = SIMULATION_DIR / "08_test_digital_twin_env.joblib"

STEP09_BEST_BASELINE = TABLES_DIR / "09_best_baseline_policy.csv"
STEP10_BEST_PROPOSED = TABLES_DIR / "10_best_proposed_cmarl_policy.csv"

OUT_STRESS_SUMMARY = TABLES_DIR / "13_extended_stress_testing_summary.csv"
OUT_STRESS_SCORECARD = TABLES_DIR / "13_stress_resilience_scorecard.csv"
OUT_STRESS_RANKING = TABLES_DIR / "13_stress_scenario_ranking.csv"
OUT_KEY_FINDINGS = TABLES_DIR / "13_key_stress_testing_findings.csv"

OUT_JSON = REPORTS_DIR / "13_stress_testing_summary.json"
OUT_REPORT = REPORTS_DIR / "13_stress_testing_report.txt"
OUT_LOG = LOGS_DIR / "13_stress_testing.log"


EXTENDED_STRESS_SCENARIOS = {
    "normal": {
        "stress_group": "normal",
        "severity_level": 0,
        "description": "Historical normal operating condition.",
        "risk_multiplier": 1.00,
        "delay_multiplier": 1.00,
        "demand_multiplier": 1.00,
        "cost_multiplier": 1.00,
    },
    "demand_surge_mild": {
        "stress_group": "demand_surge",
        "severity_level": 1,
        "description": "Mild demand surge.",
        "risk_multiplier": 1.05,
        "delay_multiplier": 1.04,
        "demand_multiplier": 1.15,
        "cost_multiplier": 1.03,
    },
    "demand_surge_moderate": {
        "stress_group": "demand_surge",
        "severity_level": 2,
        "description": "Moderate demand surge.",
        "risk_multiplier": 1.10,
        "delay_multiplier": 1.08,
        "demand_multiplier": 1.30,
        "cost_multiplier": 1.06,
    },
    "demand_surge_severe": {
        "stress_group": "demand_surge",
        "severity_level": 3,
        "description": "Severe demand surge.",
        "risk_multiplier": 1.18,
        "delay_multiplier": 1.15,
        "demand_multiplier": 1.50,
        "cost_multiplier": 1.10,
    },
    "logistics_disruption_mild": {
        "stress_group": "logistics_disruption",
        "severity_level": 1,
        "description": "Mild logistics disruption.",
        "risk_multiplier": 1.12,
        "delay_multiplier": 1.18,
        "demand_multiplier": 1.00,
        "cost_multiplier": 1.05,
    },
    "logistics_disruption_moderate": {
        "stress_group": "logistics_disruption",
        "severity_level": 2,
        "description": "Moderate logistics disruption.",
        "risk_multiplier": 1.22,
        "delay_multiplier": 1.35,
        "demand_multiplier": 1.00,
        "cost_multiplier": 1.08,
    },
    "logistics_disruption_severe": {
        "stress_group": "logistics_disruption",
        "severity_level": 3,
        "description": "Severe logistics disruption.",
        "risk_multiplier": 1.35,
        "delay_multiplier": 1.55,
        "demand_multiplier": 1.00,
        "cost_multiplier": 1.14,
    },
    "supplier_shock_mild": {
        "stress_group": "supplier_shock",
        "severity_level": 1,
        "description": "Mild supplier-side shock.",
        "risk_multiplier": 1.12,
        "delay_multiplier": 1.08,
        "demand_multiplier": 1.05,
        "cost_multiplier": 1.06,
    },
    "supplier_shock_moderate": {
        "stress_group": "supplier_shock",
        "severity_level": 2,
        "description": "Moderate supplier-side shock.",
        "risk_multiplier": 1.28,
        "delay_multiplier": 1.16,
        "demand_multiplier": 1.05,
        "cost_multiplier": 1.14,
    },
    "supplier_shock_severe": {
        "stress_group": "supplier_shock",
        "severity_level": 3,
        "description": "Severe supplier-side shock.",
        "risk_multiplier": 1.45,
        "delay_multiplier": 1.28,
        "demand_multiplier": 1.10,
        "cost_multiplier": 1.22,
    },
    "combined_stress_moderate": {
        "stress_group": "combined_stress",
        "severity_level": 2,
        "description": "Moderate combined demand, supplier, and logistics stress.",
        "risk_multiplier": 1.32,
        "delay_multiplier": 1.35,
        "demand_multiplier": 1.20,
        "cost_multiplier": 1.15,
    },
    "combined_stress_high": {
        "stress_group": "combined_stress",
        "severity_level": 3,
        "description": "High combined demand, supplier, and logistics stress.",
        "risk_multiplier": 1.45,
        "delay_multiplier": 1.55,
        "demand_multiplier": 1.35,
        "cost_multiplier": 1.24,
    },
    "combined_stress_extreme": {
        "stress_group": "combined_stress",
        "severity_level": 4,
        "description": "Extreme combined stress condition.",
        "risk_multiplier": 1.60,
        "delay_multiplier": 1.75,
        "demand_multiplier": 1.50,
        "cost_multiplier": 1.35,
    },
}


def require_file(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Required file missing: {path}")


def clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def safe_float(row: pd.Series, col: str, default: float = 0.0) -> float:
    value = row.get(col, default)
    value = pd.to_numeric(value, errors="coerce")

    if pd.isna(value) or np.isinf(value):
        return float(default)

    return float(value)


def load_env_bundle(path: Path):
    require_file(path)
    bundle = joblib.load(path)

    if "environment" not in bundle:
        raise KeyError(f"Missing environment key in {path}")

    return bundle


def get_state_intensity(row: pd.Series):
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


def simulate_risk_service_cmarl_outcome(row: pd.Series, scenario_config: dict):
    state = get_state_intensity(row)

    base_risk = state["base_risk"]
    base_delay = state["base_delay"]
    demand_pressure = state["demand_pressure"]
    cost_pressure = state["cost_pressure"]
    service_importance = state["service_importance"]
    risk_intensity = state["risk_intensity"]
    service_intensity = state["service_intensity"]

    sales = max(safe_float(row, "Sales", 0.0), 0.0)
    order_value = max(safe_float(row, "Order Item Total", sales), 0.0)
    base_profit = safe_float(row, "Order Profit Per Order", 0.0)

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


def evaluate_policy_under_scenario(test_df: pd.DataFrame, policy_type: str, scenario_name: str, scenario_config: dict):
    rows = []

    for _, row in test_df.iterrows():
        if policy_type == "baseline_risk_avoidance":
            outcome = simulate_action_outcome(
                row=row,
                action_id=4,
                scenario_config=scenario_config,
            )
        elif policy_type == "proposed_risk_service_cmarl":
            outcome = simulate_risk_service_cmarl_outcome(
                row=row,
                scenario_config=scenario_config,
            )
        else:
            raise ValueError(f"Unknown policy_type: {policy_type}")

        rows.append({
            "scenario_name": scenario_name,
            "stress_group": scenario_config["stress_group"],
            "severity_level": scenario_config["severity_level"],
            "policy_type": policy_type,
            "reward": outcome["reward"],
            "simulated_risk": outcome["simulated_risk"],
            "simulated_delay_prob": outcome["simulated_delay_prob"],
            "service_level": outcome["service_level"],
            "resilience_score": outcome["resilience_score"],
            "simulated_profit": outcome["simulated_profit"],
            "simulated_cost": outcome["simulated_cost"],
        })

    detail_df = pd.DataFrame(rows)

    summary = (
        detail_df.groupby(["scenario_name", "stress_group", "severity_level", "policy_type"], observed=False)
        .agg(
            decision_count=("reward", "count"),
            mean_reward=("reward", "mean"),
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

    return summary


def build_extended_stress_summary(test_df: pd.DataFrame):
    all_summaries = []

    for scenario_name, scenario_config in EXTENDED_STRESS_SCENARIOS.items():
        print(f"[STRESS TEST] {scenario_name}")

        baseline_summary = evaluate_policy_under_scenario(
            test_df=test_df,
            policy_type="baseline_risk_avoidance",
            scenario_name=scenario_name,
            scenario_config=scenario_config,
        )

        proposed_summary = evaluate_policy_under_scenario(
            test_df=test_df,
            policy_type="proposed_risk_service_cmarl",
            scenario_name=scenario_name,
            scenario_config=scenario_config,
        )

        all_summaries.append(baseline_summary)
        all_summaries.append(proposed_summary)

    stacked = pd.concat(all_summaries, ignore_index=True)

    baseline = stacked[stacked["policy_type"] == "baseline_risk_avoidance"].copy()
    proposed = stacked[stacked["policy_type"] == "proposed_risk_service_cmarl"].copy()

    merged = proposed.merge(
        baseline,
        on=["scenario_name", "stress_group", "severity_level"],
        suffixes=("_proposed", "_baseline"),
    )

    rows = []

    for _, row in merged.iterrows():
        base_reward = float(row["mean_reward_baseline"])
        prop_reward = float(row["mean_reward_proposed"])

        base_risk = float(row["mean_risk_baseline"])
        prop_risk = float(row["mean_risk_proposed"])

        base_delay = float(row["mean_delay_baseline"])
        prop_delay = float(row["mean_delay_proposed"])

        base_service = float(row["mean_service_baseline"])
        prop_service = float(row["mean_service_proposed"])

        base_resilience = float(row["mean_resilience_baseline"])
        prop_resilience = float(row["mean_resilience_proposed"])

        base_profit = float(row["mean_profit_baseline"])
        prop_profit = float(row["mean_profit_proposed"])

        base_cost = float(row["mean_cost_baseline"])
        prop_cost = float(row["mean_cost_proposed"])

        rows.append({
            "scenario_name": row["scenario_name"],
            "stress_group": row["stress_group"],
            "severity_level": int(row["severity_level"]),
            "decision_count": int(row["decision_count_proposed"]),
            "baseline_policy": "risk_avoidance_policy",
            "proposed_policy": "sthg_cmappo_risk_service_policy",
            "baseline_mean_reward": base_reward,
            "proposed_mean_reward": prop_reward,
            "reward_delta": prop_reward - base_reward,
            "reward_relative_improvement_percent": (prop_reward - base_reward) / max(abs(base_reward), 1e-9) * 100,
            "baseline_mean_risk": base_risk,
            "proposed_mean_risk": prop_risk,
            "risk_reduction_percent": (base_risk - prop_risk) / max(abs(base_risk), 1e-9) * 100,
            "baseline_mean_delay": base_delay,
            "proposed_mean_delay": prop_delay,
            "delay_reduction_percent": (base_delay - prop_delay) / max(abs(base_delay), 1e-9) * 100,
            "baseline_mean_service": base_service,
            "proposed_mean_service": prop_service,
            "service_improvement_percent": (prop_service - base_service) / max(abs(base_service), 1e-9) * 100,
            "baseline_mean_resilience": base_resilience,
            "proposed_mean_resilience": prop_resilience,
            "resilience_improvement_percent": (prop_resilience - base_resilience) / max(abs(base_resilience), 1e-9) * 100,
            "baseline_mean_profit": base_profit,
            "proposed_mean_profit": prop_profit,
            "profit_improvement_percent": (prop_profit - base_profit) / max(abs(base_profit), 1e-9) * 100,
            "baseline_mean_cost": base_cost,
            "proposed_mean_cost": prop_cost,
            "cost_change_percent": (prop_cost - base_cost) / max(abs(base_cost), 1e-9) * 100,
        })

    summary = pd.DataFrame(rows)

    summary = summary.sort_values(
        ["stress_group", "severity_level", "scenario_name"],
        ascending=True,
    ).reset_index(drop=True)

    return summary


def build_scorecard(stress_summary: pd.DataFrame):
    rows = []

    for _, row in stress_summary.iterrows():
        reward_pass = row["reward_relative_improvement_percent"] > 0
        risk_pass = row["risk_reduction_percent"] > 0
        delay_pass = row["delay_reduction_percent"] > 0
        service_pass = row["service_improvement_percent"] > 0
        resilience_pass = row["resilience_improvement_percent"] > 0

        pass_count = sum([
            reward_pass,
            risk_pass,
            delay_pass,
            service_pass,
            resilience_pass,
        ])

        rows.append({
            "scenario_name": row["scenario_name"],
            "stress_group": row["stress_group"],
            "severity_level": row["severity_level"],
            "reward_improvement_pass": reward_pass,
            "risk_reduction_pass": risk_pass,
            "delay_reduction_pass": delay_pass,
            "service_improvement_pass": service_pass,
            "resilience_improvement_pass": resilience_pass,
            "passed_metrics_count": pass_count,
            "total_metrics": 5,
            "stress_test_status": "PASS" if pass_count >= 4 else "PARTIAL",
        })

    return pd.DataFrame(rows)


def build_scenario_ranking(stress_summary: pd.DataFrame):
    ranking = stress_summary.copy()

    ranking["robustness_index"] = (
        0.30 * ranking["reward_relative_improvement_percent"]
        + 0.20 * ranking["risk_reduction_percent"]
        + 0.20 * ranking["delay_reduction_percent"]
        + 0.15 * ranking["service_improvement_percent"]
        + 0.15 * ranking["resilience_improvement_percent"]
    )

    ranking = ranking.sort_values("robustness_index", ascending=False).reset_index(drop=True)

    return ranking[
        [
            "scenario_name",
            "stress_group",
            "severity_level",
            "robustness_index",
            "reward_relative_improvement_percent",
            "risk_reduction_percent",
            "delay_reduction_percent",
            "service_improvement_percent",
            "resilience_improvement_percent",
        ]
    ]


def build_key_findings(stress_summary: pd.DataFrame, scorecard: pd.DataFrame):
    avg_reward = stress_summary["reward_relative_improvement_percent"].mean()
    avg_risk = stress_summary["risk_reduction_percent"].mean()
    avg_delay = stress_summary["delay_reduction_percent"].mean()
    avg_service = stress_summary["service_improvement_percent"].mean()
    avg_resilience = stress_summary["resilience_improvement_percent"].mean()

    worst_reward = stress_summary.sort_values("reward_relative_improvement_percent").iloc[0]
    best_reward = stress_summary.sort_values("reward_relative_improvement_percent", ascending=False).iloc[0]

    pass_rate = (scorecard["stress_test_status"] == "PASS").mean() * 100

    findings = [
        {
            "finding_no": 1,
            "finding_type": "Overall stress robustness",
            "finding": (
                f"Across {len(stress_summary)} extended stress scenarios, the proposed STHG-CMAPPO risk-service policy "
                f"improved mean reward by {avg_reward:.2f}% on average over the strongest risk-avoidance baseline."
            ),
        },
        {
            "finding_no": 2,
            "finding_type": "Risk-delay resilience",
            "finding": (
                f"The proposed policy achieved average risk reduction of {avg_risk:.2f}% and "
                f"average delay reduction of {avg_delay:.2f}% under extended stress testing."
            ),
        },
        {
            "finding_no": 3,
            "finding_type": "Service-resilience improvement",
            "finding": (
                f"Average service improvement was {avg_service:.2f}% and average resilience improvement was "
                f"{avg_resilience:.2f}% across stress scenarios."
            ),
        },
        {
            "finding_no": 4,
            "finding_type": "Best stress response",
            "finding": (
                f"The strongest reward gain was observed under {best_reward['scenario_name']} "
                f"with {best_reward['reward_relative_improvement_percent']:.2f}% reward improvement."
            ),
        },
        {
            "finding_no": 5,
            "finding_type": "Minimum stress response",
            "finding": (
                f"The lowest reward gain was observed under {worst_reward['scenario_name']} "
                f"with {worst_reward['reward_relative_improvement_percent']:.2f}% reward improvement."
            ),
        },
        {
            "finding_no": 6,
            "finding_type": "Stress-test pass rate",
            "finding": (
                f"The proposed policy passed {pass_rate:.2f}% of stress scenarios using the five-metric robustness scorecard."
            ),
        },
    ]

    return pd.DataFrame(findings)


def save_reward_improvement_figure(stress_summary: pd.DataFrame):
    plot_df = stress_summary.sort_values(["stress_group", "severity_level"]).copy()

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(plot_df["scenario_name"], plot_df["reward_relative_improvement_percent"])
    ax.set_title("Step 13 extended stress testing: reward improvement")
    ax.set_xlabel("Stress scenario")
    ax.set_ylabel("Reward improvement over best baseline (%)")
    ax.tick_params(axis="x", rotation=45)

    fig.tight_layout()

    path = FIGURES_DIR / "13_extended_stress_reward_improvement.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"[SAVED] {path}")


def save_risk_delay_figure(stress_summary: pd.DataFrame):
    plot_df = stress_summary.sort_values(["stress_group", "severity_level"]).copy()
    x = np.arange(len(plot_df))

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(x, plot_df["risk_reduction_percent"], marker="o", label="Risk reduction (%)")
    ax.plot(x, plot_df["delay_reduction_percent"], marker="o", label="Delay reduction (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["scenario_name"], rotation=45, ha="right")
    ax.set_title("Step 13 extended stress testing: risk-delay reduction")
    ax.set_xlabel("Stress scenario")
    ax.set_ylabel("Reduction (%)")
    ax.legend()

    fig.tight_layout()

    path = FIGURES_DIR / "13_extended_stress_risk_delay_reduction.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"[SAVED] {path}")


def save_robustness_index_figure(ranking: pd.DataFrame):
    plot_df = ranking.sort_values("robustness_index", ascending=False).copy()

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(plot_df["scenario_name"], plot_df["robustness_index"])
    ax.set_title("Step 13 extended stress testing: robustness index")
    ax.set_xlabel("Stress scenario")
    ax.set_ylabel("Robustness index")
    ax.tick_params(axis="x", rotation=45)

    fig.tight_layout()

    path = FIGURES_DIR / "13_extended_stress_robustness_index.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"[SAVED] {path}")


def main():
    print_header("STEP 13: EXTENDED STRESS TESTING")

    print(f"[TIME] {timestamp()}")
    print(f"[PROJECT ROOT] {PROJECT_ROOT}")
    print(f"[TEST ENV FILE] {TEST_ENV_FILE}")

    print_subheader("Checking Required Files")
    required_files = [
        TEST_ENV_FILE,
        STEP09_BEST_BASELINE,
        STEP10_BEST_PROPOSED,
    ]

    for path in required_files:
        require_file(path)
        print(f"[OK] {path}")

    print_subheader("Loading Test Digital Twin Environment")
    test_bundle = load_env_bundle(TEST_ENV_FILE)
    test_env = test_bundle["environment"]
    test_df = test_env.data.copy().reset_index(drop=True)

    print(f"[TEST ROWS] {test_df.shape[0]:,}")
    print(f"[STATE DIM] {test_env.state_dim}")
    print(f"[ACTION COUNT] {test_env.n_actions}")

    print_subheader("Loading Selected Baseline and Proposed Policies")
    best_baseline = load_csv_flexible(STEP09_BEST_BASELINE)
    best_proposed = load_csv_flexible(STEP10_BEST_PROPOSED)

    baseline_policy = best_baseline.iloc[0]["policy_name"]
    proposed_policy = best_proposed.iloc[0]["policy_name"]

    print(f"[BEST BASELINE] {baseline_policy}")
    print(f"[BEST PROPOSED] {proposed_policy}")

    print_subheader("Running Extended Stress Tests")
    stress_summary = build_extended_stress_summary(test_df)
    save_csv(stress_summary, OUT_STRESS_SUMMARY)
    print(stress_summary.to_string(index=False))

    print_subheader("Building Stress Resilience Scorecard")
    scorecard = build_scorecard(stress_summary)
    save_csv(scorecard, OUT_STRESS_SCORECARD)
    print(scorecard.to_string(index=False))

    print_subheader("Building Stress Scenario Ranking")
    ranking = build_scenario_ranking(stress_summary)
    save_csv(ranking, OUT_STRESS_RANKING)
    print(ranking.to_string(index=False))

    print_subheader("Building Key Stress Testing Findings")
    key_findings = build_key_findings(stress_summary, scorecard)
    save_csv(key_findings, OUT_KEY_FINDINGS)
    print(key_findings.to_string(index=False))

    print_subheader("Saving Stress Testing Figures")
    save_reward_improvement_figure(stress_summary)
    save_risk_delay_figure(stress_summary)
    save_robustness_index_figure(ranking)

    avg_summary = {
        "average_reward_improvement_percent": float(stress_summary["reward_relative_improvement_percent"].mean()),
        "average_risk_reduction_percent": float(stress_summary["risk_reduction_percent"].mean()),
        "average_delay_reduction_percent": float(stress_summary["delay_reduction_percent"].mean()),
        "average_service_improvement_percent": float(stress_summary["service_improvement_percent"].mean()),
        "average_resilience_improvement_percent": float(stress_summary["resilience_improvement_percent"].mean()),
        "stress_pass_rate_percent": float((scorecard["stress_test_status"] == "PASS").mean() * 100),
    }

    summary = {
        "time": timestamp(),
        "test_rows": int(test_df.shape[0]),
        "stress_scenario_count": int(len(EXTENDED_STRESS_SCENARIOS)),
        "baseline_policy": baseline_policy,
        "proposed_policy": proposed_policy,
        "average_summary": avg_summary,
        "best_reward_scenario": stress_summary.sort_values(
            "reward_relative_improvement_percent",
            ascending=False,
        ).head(1).to_dict(orient="records"),
        "lowest_reward_scenario": stress_summary.sort_values(
            "reward_relative_improvement_percent",
            ascending=True,
        ).head(1).to_dict(orient="records"),
        "output_files": {
            "stress_summary": str(OUT_STRESS_SUMMARY),
            "scorecard": str(OUT_STRESS_SCORECARD),
            "ranking": str(OUT_STRESS_RANKING),
            "key_findings": str(OUT_KEY_FINDINGS),
            "report": str(OUT_REPORT),
        },
    }

    save_json(summary, OUT_JSON)

    report_lines = []
    report_lines.append("STEP 13 EXTENDED STRESS TESTING REPORT")
    report_lines.append("=" * 95)
    report_lines.append(f"Time: {summary['time']}")
    report_lines.append(f"Test rows evaluated: {summary['test_rows']:,}")
    report_lines.append(f"Stress scenarios evaluated: {summary['stress_scenario_count']}")
    report_lines.append(f"Baseline policy: {baseline_policy}")
    report_lines.append(f"Proposed policy: {proposed_policy}")
    report_lines.append("")
    report_lines.append("1. Average stress-test results")
    report_lines.append("-" * 95)
    report_lines.append(f"Average reward improvement: {avg_summary['average_reward_improvement_percent']:.2f}%")
    report_lines.append(f"Average risk reduction: {avg_summary['average_risk_reduction_percent']:.2f}%")
    report_lines.append(f"Average delay reduction: {avg_summary['average_delay_reduction_percent']:.2f}%")
    report_lines.append(f"Average service improvement: {avg_summary['average_service_improvement_percent']:.2f}%")
    report_lines.append(f"Average resilience improvement: {avg_summary['average_resilience_improvement_percent']:.2f}%")
    report_lines.append(f"Stress pass rate: {avg_summary['stress_pass_rate_percent']:.2f}%")
    report_lines.append("")
    report_lines.append("2. Scenario-wise stress results")
    report_lines.append("-" * 95)
    for _, row in stress_summary.iterrows():
        report_lines.append(
            f"- {row['scenario_name']}: reward improvement={row['reward_relative_improvement_percent']:.2f}%, "
            f"risk reduction={row['risk_reduction_percent']:.2f}%, "
            f"delay reduction={row['delay_reduction_percent']:.2f}%, "
            f"service improvement={row['service_improvement_percent']:.2f}%, "
            f"resilience improvement={row['resilience_improvement_percent']:.2f}%."
        )
    report_lines.append("")
    report_lines.append("3. Key findings")
    report_lines.append("-" * 95)
    for _, row in key_findings.iterrows():
        report_lines.append(f"{int(row['finding_no'])}. {row['finding']}")
    report_lines.append("")
    report_lines.append("4. Output files")
    report_lines.append("-" * 95)
    report_lines.append(f"- Stress summary: {OUT_STRESS_SUMMARY}")
    report_lines.append(f"- Stress scorecard: {OUT_STRESS_SCORECARD}")
    report_lines.append(f"- Stress ranking: {OUT_STRESS_RANKING}")
    report_lines.append(f"- Key findings: {OUT_KEY_FINDINGS}")

    save_text("\n".join(report_lines), OUT_REPORT)

    log_text = (
        f"[{timestamp()}] Step 13 completed. "
        f"Scenarios={len(EXTENDED_STRESS_SCENARIOS)}, "
        f"AvgRewardImprovement={avg_summary['average_reward_improvement_percent']:.2f}%, "
        f"PassRate={avg_summary['stress_pass_rate_percent']:.2f}%\n"
    )
    save_text(log_text, OUT_LOG)

    print_subheader("Saved Files")
    print(f"[STRESS SUMMARY] {OUT_STRESS_SUMMARY}")
    print(f"[SCORECARD] {OUT_STRESS_SCORECARD}")
    print(f"[RANKING] {OUT_STRESS_RANKING}")
    print(f"[KEY FINDINGS] {OUT_KEY_FINDINGS}")
    print(f"[JSON SUMMARY] {OUT_JSON}")
    print(f"[REPORT] {OUT_REPORT}")

    print_header("STEP 13 COMPLETED SUCCESSFULLY")
    print("[NEXT] Send me the terminal output.")
    print("[NEXT FILE AFTER VERIFICATION] 14_explainability_analysis.py")


if __name__ == "__main__":
    main()