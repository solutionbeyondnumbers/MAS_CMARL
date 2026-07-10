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
    timestamp,
)
from resilientgraph_cmarl.m06_digital_twin_rl import (
    ACTION_SPACE,
    SCENARIO_CONFIGS,
    ProcurementDigitalTwinEnv,
    simulate_action_outcome,
)


SIMULATION_DIR = PROJECT_ROOT / "data" / "simulation"
MODELS_DIR = PROJECT_ROOT / "outputs" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_ENV_FILE = SIMULATION_DIR / "08_train_digital_twin_env.joblib"
VALID_ENV_FILE = SIMULATION_DIR / "08_valid_digital_twin_env.joblib"
TEST_ENV_FILE = SIMULATION_DIR / "08_test_digital_twin_env.joblib"

BASELINE_POLICY_RESULTS_FILE = TABLES_DIR / "09_baseline_policy_evaluation.csv"
BASELINE_POLICY_SUMMARY_FILE = TABLES_DIR / "09_baseline_policy_summary.csv"
BASELINE_STRESS_RESULTS_FILE = TABLES_DIR / "09_baseline_stress_scenario_evaluation.csv"
BASELINE_Q_TABLE_FILE = MODELS_DIR / "09_baseline_q_learning_q_table.joblib"
BASELINE_Q_POLICY_FILE = TABLES_DIR / "09_baseline_q_learning_policy_table.csv"
BEST_BASELINE_FILE = TABLES_DIR / "09_best_baseline_policy.csv"

SUMMARY_JSON_FILE = REPORTS_DIR / "09_baseline_rl_training_summary.json"
REPORT_TEXT_FILE = REPORTS_DIR / "09_baseline_rl_training_report.txt"
LOG_FILE = LOGS_DIR / "09_baseline_rl_training.log"


FIXED_POLICY_ACTIONS = {
    "random_policy": None,
    "cost_priority_policy": 0,
    "balanced_policy": 1,
    "resilience_priority_policy": 2,
    "service_priority_policy": 3,
    "risk_avoidance_policy": 4,
}


def load_env_bundle(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Environment file not found: {path}")

    bundle = joblib.load(path)

    required_keys = ["environment", "state_features", "action_space", "scenario_configs"]

    for key in required_keys:
        if key not in bundle:
            raise KeyError(f"Environment bundle missing key: {key}")

    return bundle


def action_name(action_id: int) -> str:
    return ACTION_SPACE[int(action_id)]["action_name"]


def choose_rule_based_action(row: pd.Series) -> int:
    """
    Baseline rule-based procurement policy.
    This is intentionally simple and interpretable.
    """
    base_risk = float(row.get("dt_base_risk", 0.5))
    delay_prob = float(row.get("dt_base_delay_prob", 0.5))
    service_importance = float(row.get("dt_service_importance", 0.5))
    cost_pressure = float(row.get("dt_cost_pressure", 0.5))
    context_signal = float(row.get("dt_context_signal", 0.5))

    combined_risk = (
        0.35 * base_risk
        + 0.25 * delay_prob
        + 0.20 * service_importance
        + 0.20 * context_signal
    )

    if combined_risk >= 0.68:
        return 4

    if delay_prob >= 0.62 or service_importance >= 0.62:
        return 3

    if combined_risk >= 0.48:
        return 2

    if cost_pressure <= 0.30 and combined_risk <= 0.34:
        return 0

    return 1


def discretize_state(state: np.ndarray, bins: int = 5) -> Tuple[int, ...]:
    state = np.asarray(state, dtype=float)
    state = np.clip(state, 0.0, 1.0)

    bucketed = np.floor(state * bins).astype(int)
    bucketed = np.clip(bucketed, 0, bins - 1)

    return tuple(bucketed.tolist())


def train_baseline_q_learning(
    train_env: ProcurementDigitalTwinEnv,
    episodes: int = 25,
    max_steps_per_episode: int = 6000,
    alpha: float = 0.12,
    gamma: float = 0.92,
    epsilon_start: float = 0.35,
    epsilon_end: float = 0.05,
    bins: int = 5,
    random_state: int = RANDOM_STATE,
) -> Dict:
    rng = np.random.default_rng(random_state)

    q_table = {}
    episode_rows = []

    for episode in range(episodes):
        state = train_env.reset(start_index=0)
        done = False
        step = 0
        total_reward = 0.0

        epsilon = epsilon_end + (epsilon_start - epsilon_end) * max(
            0.0,
            (episodes - episode - 1) / max(episodes - 1, 1),
        )

        while not done and step < max_steps_per_episode:
            state_key = discretize_state(state, bins=bins)

            if state_key not in q_table:
                q_table[state_key] = np.zeros(train_env.n_actions, dtype=float)

            if rng.random() < epsilon:
                action_id = int(rng.integers(0, train_env.n_actions))
            else:
                action_id = int(np.argmax(q_table[state_key]))

            next_state, reward, done, info = train_env.step(action_id)
            next_key = discretize_state(next_state, bins=bins)

            if next_key not in q_table:
                q_table[next_key] = np.zeros(train_env.n_actions, dtype=float)

            best_next_q = 0.0 if done else float(np.max(q_table[next_key]))

            old_q = q_table[state_key][action_id]
            q_table[state_key][action_id] = old_q + alpha * (
                reward + gamma * best_next_q - old_q
            )

            state = next_state
            total_reward += reward
            step += 1

        episode_rows.append({
            "episode": episode + 1,
            "steps": step,
            "epsilon": epsilon,
            "total_reward": total_reward,
            "mean_reward": total_reward / max(step, 1),
            "q_states": len(q_table),
        })

        print(
            f"[Q-LEARNING] Episode {episode + 1:02d}/{episodes} | "
            f"steps={step:,} | mean_reward={total_reward / max(step, 1):.4f} | "
            f"epsilon={epsilon:.4f} | states={len(q_table):,}"
        )

    return {
        "q_table": q_table,
        "bins": bins,
        "training_log": pd.DataFrame(episode_rows),
    }


def choose_q_learning_action(row: pd.Series, q_pack: Dict, state_features: List[str]) -> int:
    state = (
        row[state_features]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
        .to_numpy(dtype=float)
    )

    state_key = discretize_state(state, bins=q_pack["bins"])
    q_table = q_pack["q_table"]

    if state_key not in q_table:
        return choose_rule_based_action(row)

    return int(np.argmax(q_table[state_key]))


def evaluate_policy_on_dataframe(
    df: pd.DataFrame,
    state_features: List[str],
    policy_name: str,
    scenario_name: str = "normal",
    q_pack: Dict = None,
    max_rows: int = None,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    eval_df = df.copy().reset_index(drop=True)

    if max_rows is not None and len(eval_df) > max_rows:
        idx = rng.choice(eval_df.index.to_numpy(), size=max_rows, replace=False)
        eval_df = eval_df.loc[idx].copy().reset_index(drop=True)

    rows = []

    scenario_config = SCENARIO_CONFIGS[scenario_name]

    for _, row in eval_df.iterrows():
        if policy_name == "random_policy":
            action_id = int(rng.integers(0, len(ACTION_SPACE)))
        elif policy_name == "rule_based_policy":
            action_id = choose_rule_based_action(row)
        elif policy_name == "q_learning_policy":
            action_id = choose_q_learning_action(row, q_pack, state_features)
        else:
            action_id = FIXED_POLICY_ACTIONS[policy_name]

        outcome = simulate_action_outcome(
            row=row,
            action_id=action_id,
            scenario_config=scenario_config,
        )

        rows.append({
            "scenario_name": scenario_name,
            "policy_name": policy_name,
            "dt_row_id": int(row.get("dt_row_id", -1)),
            "action_id": action_id,
            "action_name": action_name(action_id),
            "reward": outcome["reward"],
            "simulated_risk": outcome["simulated_risk"],
            "simulated_delay_prob": outcome["simulated_delay_prob"],
            "service_level": outcome["service_level"],
            "resilience_score": outcome["resilience_score"],
            "simulated_profit": outcome["simulated_profit"],
            "simulated_cost": outcome["simulated_cost"],
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


def evaluate_all_baseline_policies(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    state_features: List[str],
    q_pack: Dict,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    all_detail = []
    all_summary = []

    policies = list(FIXED_POLICY_ACTIONS.keys()) + [
        "rule_based_policy",
        "q_learning_policy",
    ]

    split_data = {
        "train": train_df,
        "valid": valid_df,
        "test": test_df,
    }

    for split_name, split_df in split_data.items():
        print(f"[EVALUATION SPLIT] {split_name} | rows={len(split_df):,}")

        for policy_name in policies:
            print(f"  [POLICY] {policy_name}")

            details = evaluate_policy_on_dataframe(
                df=split_df,
                state_features=state_features,
                policy_name=policy_name,
                scenario_name="normal",
                q_pack=q_pack,
                max_rows=None,
                random_state=RANDOM_STATE,
            )

            details["data_split"] = split_name

            summary = summarize_policy_results(details, data_split=split_name)

            all_detail.append(details)
            all_summary.append(summary)

    detail_df = pd.concat(all_detail, ignore_index=True)
    summary_df = pd.concat(all_summary, ignore_index=True)

    return detail_df, summary_df


def evaluate_stress_scenarios(
    test_df: pd.DataFrame,
    state_features: List[str],
    q_pack: Dict,
    max_rows_per_scenario: int = 12000,
) -> pd.DataFrame:
    policies = list(FIXED_POLICY_ACTIONS.keys()) + [
        "rule_based_policy",
        "q_learning_policy",
    ]

    all_summary = []

    for scenario_name in SCENARIO_CONFIGS.keys():
        print(f"[STRESS SCENARIO] {scenario_name}")

        for policy_name in policies:
            details = evaluate_policy_on_dataframe(
                df=test_df,
                state_features=state_features,
                policy_name=policy_name,
                scenario_name=scenario_name,
                q_pack=q_pack,
                max_rows=max_rows_per_scenario,
                random_state=RANDOM_STATE,
            )

            summary = summarize_policy_results(details, data_split="test")
            all_summary.append(summary)

    return pd.concat(all_summary, ignore_index=True)


def save_q_policy_table(q_pack: Dict, path: Path):
    rows = []

    for state_key, q_values in q_pack["q_table"].items():
        row = {
            "state_key": str(state_key),
            "best_action_id": int(np.argmax(q_values)),
            "best_action_name": action_name(int(np.argmax(q_values))),
        }

        for action_id, q_value in enumerate(q_values):
            row[f"q_action_{action_id}"] = float(q_value)

        rows.append(row)

    q_df = pd.DataFrame(rows)
    save_csv(q_df, path)

    return q_df


def save_baseline_reward_figure(summary_df: pd.DataFrame):
    test_summary = summary_df[
        (summary_df["data_split"] == "test")
        & (summary_df["scenario_name"] == "normal")
    ].copy()

    test_summary = test_summary.sort_values("mean_reward", ascending=False)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(test_summary["policy_name"], test_summary["mean_reward"])
    ax.set_title("Step 09 baseline policy mean reward on test digital twin")
    ax.set_xlabel("Baseline policy")
    ax.set_ylabel("Mean reward")
    ax.tick_params(axis="x", rotation=35)

    fig.tight_layout()

    path = FIGURES_DIR / "09_baseline_policy_test_reward_comparison.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)

    print(f"[SAVED] {path}")


def save_risk_resilience_figure(summary_df: pd.DataFrame):
    test_summary = summary_df[
        (summary_df["data_split"] == "test")
        & (summary_df["scenario_name"] == "normal")
    ].copy()

    test_summary = test_summary.sort_values("mean_resilience", ascending=False)

    x = np.arange(len(test_summary))

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(x, test_summary["mean_risk"], marker="o", label="Mean risk")
    ax.plot(x, test_summary["mean_resilience"], marker="o", label="Mean resilience")
    ax.set_xticks(x)
    ax.set_xticklabels(test_summary["policy_name"], rotation=35, ha="right")
    ax.set_title("Step 09 baseline policy risk-resilience comparison")
    ax.set_xlabel("Baseline policy")
    ax.set_ylabel("Score")
    ax.legend()

    fig.tight_layout()

    path = FIGURES_DIR / "09_baseline_policy_risk_resilience_comparison.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)

    print(f"[SAVED] {path}")


def main():
    print_header("STEP 09: TRAIN AND EVALUATE BASELINE RL POLICIES")

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

    state_features = train_bundle["state_features"]

    train_df = train_env.data.copy()
    valid_df = valid_env.data.copy()
    test_df = test_env.data.copy()

    print_subheader("Environment Summary")
    print(f"[STATE DIM] {train_env.state_dim}")
    print(f"[ACTION COUNT] {train_env.n_actions}")
    print(f"[TRAIN ROWS] {train_df.shape[0]:,}")
    print(f"[VALID ROWS] {valid_df.shape[0]:,}")
    print(f"[TEST ROWS] {test_df.shape[0]:,}")

    print_subheader("Training Baseline Q-Learning Policy")

    q_pack = train_baseline_q_learning(
        train_env=train_env,
        episodes=25,
        max_steps_per_episode=6000,
        alpha=0.12,
        gamma=0.92,
        epsilon_start=0.35,
        epsilon_end=0.05,
        bins=5,
        random_state=RANDOM_STATE,
    )

    joblib.dump(q_pack, BASELINE_Q_TABLE_FILE)
    print(f"[SAVED] {BASELINE_Q_TABLE_FILE}")

    save_csv(q_pack["training_log"], TABLES_DIR / "09_q_learning_training_log.csv")
    q_policy_df = save_q_policy_table(q_pack, BASELINE_Q_POLICY_FILE)

    print_subheader("Evaluating Baseline Policies on Train/Validation/Test")

    detail_df, summary_df = evaluate_all_baseline_policies(
        train_df=train_df,
        valid_df=valid_df,
        test_df=test_df,
        state_features=state_features,
        q_pack=q_pack,
    )

    save_csv(detail_df, BASELINE_POLICY_RESULTS_FILE)
    save_csv(summary_df, BASELINE_POLICY_SUMMARY_FILE)

    print_subheader("Baseline Policy Summary")
    normal_test_summary = summary_df[
        (summary_df["data_split"] == "test")
        & (summary_df["scenario_name"] == "normal")
    ].sort_values(["mean_reward", "mean_resilience"], ascending=False)
    print(normal_test_summary.to_string(index=False))

    best_baseline = normal_test_summary.head(1).copy()
    save_csv(best_baseline, BEST_BASELINE_FILE)

    print_subheader("Evaluating Baseline Policies under Stress Scenarios")

    stress_summary = evaluate_stress_scenarios(
        test_df=test_df,
        state_features=state_features,
        q_pack=q_pack,
        max_rows_per_scenario=12000,
    )
    save_csv(stress_summary, BASELINE_STRESS_RESULTS_FILE)

    save_baseline_reward_figure(summary_df)
    save_risk_resilience_figure(summary_df)

    summary = {
        "time": timestamp(),
        "state_dim": int(train_env.state_dim),
        "action_count": int(train_env.n_actions),
        "train_rows": int(train_df.shape[0]),
        "valid_rows": int(valid_df.shape[0]),
        "test_rows": int(test_df.shape[0]),
        "baseline_policies": list(FIXED_POLICY_ACTIONS.keys()) + [
            "rule_based_policy",
            "q_learning_policy",
        ],
        "q_learning": {
            "episodes": 25,
            "max_steps_per_episode": 6000,
            "alpha": 0.12,
            "gamma": 0.92,
            "epsilon_start": 0.35,
            "epsilon_end": 0.05,
            "bins": 5,
            "q_states": int(len(q_pack["q_table"])),
        },
        "best_baseline_policy_test": best_baseline.to_dict(orient="records"),
        "purpose": [
            "Evaluate fixed heuristic procurement policies.",
            "Evaluate interpretable rule-based baseline policy.",
            "Train and evaluate a baseline Q-learning policy.",
            "Create benchmark results for Step 10 proposed CMARL.",
        ],
    }

    save_json(summary, SUMMARY_JSON_FILE)

    report_lines = []
    report_lines.append("STEP 09 BASELINE RL POLICY TRAINING REPORT")
    report_lines.append("=" * 90)
    report_lines.append(f"Time: {summary['time']}")
    report_lines.append(f"State dimension: {summary['state_dim']}")
    report_lines.append(f"Action count: {summary['action_count']}")
    report_lines.append(f"Train rows: {summary['train_rows']:,}")
    report_lines.append(f"Validation rows: {summary['valid_rows']:,}")
    report_lines.append(f"Test rows: {summary['test_rows']:,}")
    report_lines.append("")
    report_lines.append("Baseline policies evaluated:")
    for policy in summary["baseline_policies"]:
        report_lines.append(f"- {policy}")
    report_lines.append("")
    report_lines.append("Q-learning configuration:")
    for key, value in summary["q_learning"].items():
        report_lines.append(f"- {key}: {value}")
    report_lines.append("")
    report_lines.append("Normal test-set baseline ranking:")
    for _, row in normal_test_summary.iterrows():
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
    report_lines.append("Next step:")
    report_lines.append("- Step 10 will train the proposed graph-risk-aware CMARL policy.")
    report_lines.append("- Step 10 must improve over the best Step 09 baseline.")

    save_text("\n".join(report_lines), REPORT_TEXT_FILE)

    log_text = (
        f"[{timestamp()}] Step 09 completed. "
        f"BestBaseline={best_baseline.to_dict(orient='records')}, "
        f"QStates={len(q_pack['q_table'])}\n"
    )
    save_text(log_text, LOG_FILE)

    print_subheader("Saved Files")
    print(f"[POLICY DETAILS] {BASELINE_POLICY_RESULTS_FILE}")
    print(f"[POLICY SUMMARY] {BASELINE_POLICY_SUMMARY_FILE}")
    print(f"[STRESS SUMMARY] {BASELINE_STRESS_RESULTS_FILE}")
    print(f"[Q TABLE] {BASELINE_Q_TABLE_FILE}")
    print(f"[Q POLICY TABLE] {BASELINE_Q_POLICY_FILE}")
    print(f"[BEST BASELINE] {BEST_BASELINE_FILE}")
    print(f"[REPORT] {REPORT_TEXT_FILE}")

    print_header("STEP 09 COMPLETED SUCCESSFULLY")
    print("[NEXT] Send me the terminal output.")
    print("[NEXT FILE AFTER VERIFICATION] 10_train_proposed_cmarl_models.py")


if __name__ == "__main__":
    main()