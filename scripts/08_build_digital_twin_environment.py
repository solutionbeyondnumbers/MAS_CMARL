import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from resilientgraph_cmarl.m00_config import (
    PROCESSED_DIR,
    TABLES_DIR,
    REPORTS_DIR,
    LOGS_DIR,
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
from resilientgraph_cmarl.m06_digital_twin_rl import (
    ACTION_SPACE,
    SCENARIO_CONFIGS,
    ProcurementDigitalTwinEnv,
    prepare_digital_twin_dataset,
    build_counterfactual_transition_table,
    summarize_transition_table,
    validate_environment_rollout,
)


SIMULATION_DIR = PROJECT_ROOT / "data" / "simulation"
SIMULATION_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = PROCESSED_DIR / "07_dataco_leakage_safe_rgc_features.csv"

OUTPUT_DT_DATASET = SIMULATION_DIR / "08_digital_twin_dataset.csv"
OUTPUT_TRANSITION_TABLE = SIMULATION_DIR / "08_counterfactual_transition_table_normal.csv"

TRAIN_ENV_FILE = SIMULATION_DIR / "08_train_digital_twin_env.joblib"
VALID_ENV_FILE = SIMULATION_DIR / "08_valid_digital_twin_env.joblib"
TEST_ENV_FILE = SIMULATION_DIR / "08_test_digital_twin_env.joblib"

STATE_FEATURE_FILE = TABLES_DIR / "08_digital_twin_state_features.csv"
ACTION_SPACE_FILE = TABLES_DIR / "08_digital_twin_action_space.csv"
SCENARIO_CONFIG_FILE = TABLES_DIR / "08_digital_twin_scenario_config.csv"
SPLIT_SUMMARY_FILE = TABLES_DIR / "08_digital_twin_split_summary.csv"
TRANSITION_SUMMARY_FILE = TABLES_DIR / "08_counterfactual_transition_summary_normal.csv"
STRESS_SUMMARY_FILE = TABLES_DIR / "08_stress_scenario_transition_summary.csv"
ROLLOUT_VALIDATION_FILE = TABLES_DIR / "08_environment_rollout_validation.csv"

SUMMARY_JSON_FILE = REPORTS_DIR / "08_digital_twin_environment_summary.json"
REPORT_TEXT_FILE = REPORTS_DIR / "08_digital_twin_environment_report.txt"
LOG_FILE = LOGS_DIR / "08_digital_twin_environment.log"


def save_action_space_table():
    rows = []

    for action_id, action_info in ACTION_SPACE.items():
        row = {"action_id": action_id}
        row.update(action_info)
        rows.append(row)

    action_df = pd.DataFrame(rows)
    save_csv(action_df, ACTION_SPACE_FILE)

    return action_df


def save_scenario_config_table():
    rows = []

    for scenario_name, scenario_info in SCENARIO_CONFIGS.items():
        row = {"scenario_key": scenario_name}
        row.update(scenario_info)
        rows.append(row)

    scenario_df = pd.DataFrame(rows)
    save_csv(scenario_df, SCENARIO_CONFIG_FILE)

    return scenario_df


def build_split_summary(dt_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for split_name, split_df in dt_df.groupby("data_split", observed=False):
        rows.append({
            "data_split": split_name,
            "rows": int(split_df.shape[0]),
            "min_order_date": str(pd.to_datetime(split_df["order date (DateOrders)"], errors="coerce").min())
            if "order date (DateOrders)" in split_df.columns else "",
            "max_order_date": str(pd.to_datetime(split_df["order date (DateOrders)"], errors="coerce").max())
            if "order date (DateOrders)" in split_df.columns else "",
            "mean_base_risk": float(split_df["dt_base_risk"].mean()),
            "mean_base_delay_prob": float(split_df["dt_base_delay_prob"].mean()),
            "mean_context_signal": float(split_df["dt_context_signal"].mean()),
            "mean_demand_pressure": float(split_df["dt_demand_pressure"].mean()),
            "mean_cost_pressure": float(split_df["dt_cost_pressure"].mean()),
            "mean_service_importance": float(split_df["dt_service_importance"].mean()),
        })

    return pd.DataFrame(rows)


def save_environment_bundle(
    env: ProcurementDigitalTwinEnv,
    path: Path,
    state_features,
    split_name: str,
):
    bundle = {
        "split_name": split_name,
        "environment": env,
        "state_features": state_features,
        "action_space": ACTION_SPACE,
        "scenario_configs": SCENARIO_CONFIGS,
        "state_dim": env.state_dim,
        "n_actions": env.n_actions,
        "rows": int(env.data.shape[0]),
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)

    print(f"[SAVED] {path}")


def build_stress_scenario_summary(test_df: pd.DataFrame) -> pd.DataFrame:
    stress_summaries = []

    for scenario_name in SCENARIO_CONFIGS.keys():
        transition_df = build_counterfactual_transition_table(
            dt_df=test_df,
            scenario_name=scenario_name,
            max_rows_per_split=8000,
            random_state=RANDOM_STATE,
        )

        scenario_summary = summarize_transition_table(transition_df)
        stress_summaries.append(scenario_summary)

    if not stress_summaries:
        return pd.DataFrame()

    return pd.concat(stress_summaries, ignore_index=True)


def save_reward_figure(transition_summary: pd.DataFrame):
    test_summary = transition_summary[transition_summary["data_split"] == "test"].copy()

    if test_summary.empty:
        test_summary = transition_summary.copy()

    plot_df = test_summary.sort_values("mean_reward", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(plot_df["action_name"], plot_df["mean_reward"])
    ax.set_title("Digital twin mean reward by procurement action")
    ax.set_xlabel("Procurement action")
    ax.set_ylabel("Mean reward")
    ax.tick_params(axis="x", rotation=35)

    fig.tight_layout()

    fig_path = FIGURES_DIR / "08_digital_twin_action_reward_summary.png"
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)

    print(f"[SAVED] {fig_path}")


def main():
    print_header("STEP 08: BUILD DIGITAL TWIN PROCUREMENT ENVIRONMENT")

    print(f"[TIME] {timestamp()}")
    print(f"[PROJECT ROOT] {PROJECT_ROOT}")
    print(f"[INPUT FILE] {INPUT_FILE}")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Step 07 leakage-safe RGC feature file not found: {INPUT_FILE}"
        )

    print_subheader("Loading Leakage-Safe RGC Dataset")
    df = load_csv_flexible(INPUT_FILE)
    print(f"[INPUT SHAPE] {df.shape[0]:,} rows | {df.shape[1]:,} columns")

    print_subheader("Preparing Digital Twin State Dataset")
    dt_df, state_features, scaler_stats = prepare_digital_twin_dataset(df)

    print(f"[DIGITAL TWIN DATASET] {dt_df.shape[0]:,} rows | {dt_df.shape[1]:,} columns")
    print(f"[STATE FEATURE COUNT] {len(state_features)}")

    save_csv(dt_df, OUTPUT_DT_DATASET)

    state_feature_df = pd.DataFrame({
        "state_feature_index": list(range(len(state_features))),
        "state_feature": state_features,
        "description": [
            "Scaled sales pressure",
            "Scaled order-value pressure",
            "Scaled order-quantity pressure",
            "Scaled discount pressure",
            "Scaled scheduled-shipment pressure",
            "Scaled profit stress",
            "Graph/contextual risk signal",
            "Base disruption-risk intensity",
            "Base delay probability",
            "Demand pressure",
            "Cost pressure",
            "Service importance",
        ],
    })
    save_csv(state_feature_df, STATE_FEATURE_FILE)

    action_df = save_action_space_table()
    scenario_df = save_scenario_config_table()

    print_subheader("Creating Train/Validation/Test Digital Twin Environments")

    train_df = dt_df[dt_df["data_split"] == "train"].copy()
    valid_df = dt_df[dt_df["data_split"] == "valid"].copy()
    test_df = dt_df[dt_df["data_split"] == "test"].copy()

    train_env = ProcurementDigitalTwinEnv(
        data=train_df,
        state_features=state_features,
        scenario_name="normal",
        max_steps=len(train_df),
        random_state=RANDOM_STATE,
    )

    valid_env = ProcurementDigitalTwinEnv(
        data=valid_df,
        state_features=state_features,
        scenario_name="normal",
        max_steps=len(valid_df),
        random_state=RANDOM_STATE,
    )

    test_env = ProcurementDigitalTwinEnv(
        data=test_df,
        state_features=state_features,
        scenario_name="normal",
        max_steps=len(test_df),
        random_state=RANDOM_STATE,
    )

    save_environment_bundle(train_env, TRAIN_ENV_FILE, state_features, "train")
    save_environment_bundle(valid_env, VALID_ENV_FILE, state_features, "valid")
    save_environment_bundle(test_env, TEST_ENV_FILE, state_features, "test")

    print_subheader("Saving Split and Environment Summaries")

    split_summary = build_split_summary(dt_df)
    save_csv(split_summary, SPLIT_SUMMARY_FILE)
    print(split_summary.to_string(index=False))

    print_subheader("Building Counterfactual Action Transition Table")

    transition_table = build_counterfactual_transition_table(
        dt_df=dt_df,
        scenario_name="normal",
        max_rows_per_split=15000,
        random_state=RANDOM_STATE,
    )
    save_csv(transition_table, OUTPUT_TRANSITION_TABLE)

    transition_summary = summarize_transition_table(transition_table)
    save_csv(transition_summary, TRANSITION_SUMMARY_FILE)
    print(transition_summary.to_string(index=False))

    save_reward_figure(transition_summary)

    print_subheader("Building Stress Scenario Summary")

    stress_summary = build_stress_scenario_summary(test_df)
    save_csv(stress_summary, STRESS_SUMMARY_FILE)

    print_subheader("Validating Environment Rollouts")

    rollout_rows = []

    for split_name, env in [
        ("train", train_env),
        ("valid", valid_env),
        ("test", test_env),
    ]:
        for policy in ["random", "balanced", "resilience"]:
            result = validate_environment_rollout(
                env=env,
                policy=policy,
                max_steps=500,
            )
            result["data_split"] = split_name
            rollout_rows.append(result)

    rollout_validation = pd.DataFrame(rollout_rows)
    save_csv(rollout_validation, ROLLOUT_VALIDATION_FILE)
    print(rollout_validation.to_string(index=False))

    summary = {
        "time": timestamp(),
        "input_file": str(INPUT_FILE),
        "digital_twin_dataset": str(OUTPUT_DT_DATASET),
        "rows": int(dt_df.shape[0]),
        "columns": int(dt_df.shape[1]),
        "state_feature_count": int(len(state_features)),
        "state_features": state_features,
        "action_count": int(len(ACTION_SPACE)),
        "actions": action_df.to_dict(orient="records"),
        "scenario_count": int(len(SCENARIO_CONFIGS)),
        "scenarios": scenario_df.to_dict(orient="records"),
        "split_rows": split_summary.to_dict(orient="records"),
        "environment_files": {
            "train": str(TRAIN_ENV_FILE),
            "valid": str(VALID_ENV_FILE),
            "test": str(TEST_ENV_FILE),
        },
        "step08_purpose": [
            "Create a procurement digital twin environment.",
            "Define state space, action space, reward function, and stress scenarios.",
            "Save train/validation/test environments for baseline RL and proposed CMARL training.",
            "No RL model is trained in Step 08.",
        ],
    }

    save_json(summary, SUMMARY_JSON_FILE)

    report_lines = []
    report_lines.append("STEP 08 DIGITAL TWIN ENVIRONMENT REPORT")
    report_lines.append("=" * 90)
    report_lines.append(f"Time: {summary['time']}")
    report_lines.append(f"Input file: {INPUT_FILE}")
    report_lines.append(f"Digital twin dataset shape: {dt_df.shape[0]:,} rows x {dt_df.shape[1]:,} columns")
    report_lines.append(f"State feature count: {len(state_features)}")
    report_lines.append(f"Action count: {len(ACTION_SPACE)}")
    report_lines.append(f"Scenario count: {len(SCENARIO_CONFIGS)}")
    report_lines.append("")
    report_lines.append("State features:")
    for feature in state_features:
        report_lines.append(f"- {feature}")
    report_lines.append("")
    report_lines.append("Procurement actions:")
    for _, row in action_df.iterrows():
        report_lines.append(f"- Action {row['action_id']}: {row['action_name']} | {row['description']}")
    report_lines.append("")
    report_lines.append("Stress scenarios:")
    for _, row in scenario_df.iterrows():
        report_lines.append(f"- {row['scenario_key']}: {row['description']}")
    report_lines.append("")
    report_lines.append("Environment files:")
    report_lines.append(f"- Train: {TRAIN_ENV_FILE}")
    report_lines.append(f"- Validation: {VALID_ENV_FILE}")
    report_lines.append(f"- Test: {TEST_ENV_FILE}")
    report_lines.append("")
    report_lines.append("Next step:")
    report_lines.append("- Step 09 will train/evaluate baseline RL and heuristic procurement policies.")
    report_lines.append("- Step 10 will train/evaluate proposed graph-risk-aware CMARL.")

    save_text("\n".join(report_lines), REPORT_TEXT_FILE)

    log_text = (
        f"[{timestamp()}] Step 08 completed. "
        f"Rows={dt_df.shape[0]}, "
        f"StateFeatures={len(state_features)}, "
        f"Actions={len(ACTION_SPACE)}, "
        f"Scenarios={len(SCENARIO_CONFIGS)}\n"
    )
    save_text(log_text, LOG_FILE)

    print_subheader("Saved Files")
    print(f"[DIGITAL TWIN DATASET] {OUTPUT_DT_DATASET}")
    print(f"[TRANSITION TABLE] {OUTPUT_TRANSITION_TABLE}")
    print(f"[TRAIN ENV] {TRAIN_ENV_FILE}")
    print(f"[VALID ENV] {VALID_ENV_FILE}")
    print(f"[TEST ENV] {TEST_ENV_FILE}")
    print(f"[STATE FEATURES] {STATE_FEATURE_FILE}")
    print(f"[ACTION SPACE] {ACTION_SPACE_FILE}")
    print(f"[SCENARIOS] {SCENARIO_CONFIG_FILE}")
    print(f"[REPORT] {REPORT_TEXT_FILE}")

    print_header("STEP 08 COMPLETED SUCCESSFULLY")
    print("[NEXT] Send me the terminal output.")
    print("[NEXT FILE AFTER VERIFICATION] 09_train_baseline_rl_models.py")


if __name__ == "__main__":
    main()