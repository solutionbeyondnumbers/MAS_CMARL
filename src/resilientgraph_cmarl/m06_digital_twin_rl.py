from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


ACTION_SPACE = {
    0: {
        "action_name": "cost_priority_procurement",
        "description": "Select lowest-cost procurement option with higher disruption exposure.",
        "cost_factor": 0.92,
        "risk_factor": 1.18,
        "delay_factor": 1.14,
        "service_bonus": -0.03,
        "resilience_bonus": 0.00,
        "profit_factor": 1.02,
    },
    1: {
        "action_name": "balanced_procurement",
        "description": "Balance cost, risk, delay, and service reliability.",
        "cost_factor": 1.00,
        "risk_factor": 1.00,
        "delay_factor": 1.00,
        "service_bonus": 0.00,
        "resilience_bonus": 0.03,
        "profit_factor": 1.00,
    },
    2: {
        "action_name": "resilience_priority_procurement",
        "description": "Prefer resilient supplier-route configuration with moderate cost increase.",
        "cost_factor": 1.07,
        "risk_factor": 0.78,
        "delay_factor": 0.82,
        "service_bonus": 0.06,
        "resilience_bonus": 0.12,
        "profit_factor": 0.96,
    },
    3: {
        "action_name": "service_priority_procurement",
        "description": "Prefer high-service procurement option to reduce fulfillment and delay loss.",
        "cost_factor": 1.12,
        "risk_factor": 0.88,
        "delay_factor": 0.66,
        "service_bonus": 0.13,
        "resilience_bonus": 0.09,
        "profit_factor": 0.94,
    },
    4: {
        "action_name": "risk_avoidance_procurement",
        "description": "Use strongest risk-avoidance decision under disruption-prone conditions.",
        "cost_factor": 1.16,
        "risk_factor": 0.62,
        "delay_factor": 0.74,
        "service_bonus": 0.07,
        "resilience_bonus": 0.18,
        "profit_factor": 0.92,
    },
}


SCENARIO_CONFIGS = {
    "normal": {
        "scenario_name": "normal",
        "description": "Historical operating condition.",
        "risk_multiplier": 1.00,
        "delay_multiplier": 1.00,
        "demand_multiplier": 1.00,
        "cost_multiplier": 1.00,
    },
    "demand_surge": {
        "scenario_name": "demand_surge",
        "description": "Demand surge increases fulfillment pressure.",
        "risk_multiplier": 1.08,
        "delay_multiplier": 1.05,
        "demand_multiplier": 1.30,
        "cost_multiplier": 1.04,
    },
    "logistics_disruption": {
        "scenario_name": "logistics_disruption",
        "description": "Transport disruption increases delay and risk exposure.",
        "risk_multiplier": 1.22,
        "delay_multiplier": 1.35,
        "demand_multiplier": 1.00,
        "cost_multiplier": 1.08,
    },
    "supplier_shock": {
        "scenario_name": "supplier_shock",
        "description": "Supplier-side shock increases shortage and procurement cost.",
        "risk_multiplier": 1.28,
        "delay_multiplier": 1.16,
        "demand_multiplier": 1.05,
        "cost_multiplier": 1.14,
    },
    "combined_stress": {
        "scenario_name": "combined_stress",
        "description": "Combined demand, supplier, and logistics stress condition.",
        "risk_multiplier": 1.38,
        "delay_multiplier": 1.42,
        "demand_multiplier": 1.25,
        "cost_multiplier": 1.18,
    },
}


def clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def safe_numeric_value(row: pd.Series, col: str, default: float = 0.0) -> float:
    if col not in row.index:
        return float(default)

    value = pd.to_numeric(row.get(col), errors="coerce")

    if pd.isna(value) or np.isinf(value):
        return float(default)

    return float(value)


def safe_numeric_series(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)

    return (
        pd.to_numeric(df[col], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(default)
        .astype(float)
    )


def build_robust_scaler_stats(train_df: pd.DataFrame, columns: List[str]) -> Dict[str, Dict[str, float]]:
    stats = {}

    for col in columns:
        if col not in train_df.columns:
            continue

        s = safe_numeric_series(train_df, col, default=0.0)

        stats[col] = {
            "q01": float(s.quantile(0.01)),
            "q99": float(s.quantile(0.99)),
            "median": float(s.median()),
        }

    return stats


def robust_minmax_scale(series: pd.Series, col_stats: Dict[str, float]) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    s = s.fillna(col_stats["median"])

    low = col_stats["q01"]
    high = col_stats["q99"]

    if high <= low:
        return pd.Series(np.zeros(len(series)), index=series.index, dtype=float)

    return ((s.clip(low, high) - low) / (high - low)).clip(0, 1).fillna(0)


def get_state_feature_list() -> List[str]:
    return [
        "dt_sales_pressure",
        "dt_order_value_pressure",
        "dt_quantity_pressure",
        "dt_discount_pressure",
        "dt_schedule_pressure",
        "dt_profit_stress",
        "dt_context_signal",
        "dt_base_risk",
        "dt_base_delay_prob",
        "dt_demand_pressure",
        "dt_cost_pressure",
        "dt_service_importance",
    ]


def prepare_digital_twin_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], Dict[str, Dict[str, float]]]:
    df = df.copy()

    if "order date (DateOrders)" in df.columns:
        df["order date (DateOrders)"] = pd.to_datetime(df["order date (DateOrders)"], errors="coerce")
        df = df.sort_values("order date (DateOrders)").reset_index(drop=True)

    if "data_split" not in df.columns:
        df["data_split"] = "unknown"

    df["dt_row_id"] = np.arange(len(df))

    train_df = df[df["data_split"] == "train"].copy()
    if train_df.empty:
        train_df = df.copy()

    scaler_columns = [
        "Sales",
        "Order Item Total",
        "Order Item Quantity",
        "Order Item Discount Rate",
        "Days for shipment (scheduled)",
        "Order Profit Per Order",
        "profit_margin",
        "rgc_context_signal",
        "rgc_graph_signal",
        "rgc_proxy_operational_pressure",
    ]

    scaler_stats = build_robust_scaler_stats(train_df, scaler_columns)

    def scaled(col: str, default: float = 0.0) -> pd.Series:
        if col in df.columns and col in scaler_stats:
            return robust_minmax_scale(df[col], scaler_stats[col])
        return pd.Series(default, index=df.index, dtype=float)

    df["dt_sales_pressure"] = scaled("Sales")
    df["dt_order_value_pressure"] = scaled("Order Item Total")
    df["dt_quantity_pressure"] = scaled("Order Item Quantity")
    df["dt_discount_pressure"] = scaled("Order Item Discount Rate")
    df["dt_schedule_pressure"] = scaled("Days for shipment (scheduled)")

    if "profit_margin" in df.columns and "profit_margin" in scaler_stats:
        df["dt_profit_stress"] = (1.0 - robust_minmax_scale(df["profit_margin"], scaler_stats["profit_margin"])).clip(0, 1)
    elif "Order Profit Per Order" in df.columns and "Order Profit Per Order" in scaler_stats:
        df["dt_profit_stress"] = (1.0 - robust_minmax_scale(df["Order Profit Per Order"], scaler_stats["Order Profit Per Order"])).clip(0, 1)
    else:
        df["dt_profit_stress"] = 0.5

    if "rgc_context_signal" in df.columns:
        df["dt_context_signal"] = safe_numeric_series(df, "rgc_context_signal").clip(0, 1)
    elif "rgc_graph_signal" in df.columns:
        df["dt_context_signal"] = safe_numeric_series(df, "rgc_graph_signal").clip(0, 1)
    elif "rgc_proxy_operational_pressure" in df.columns:
        df["dt_context_signal"] = safe_numeric_series(df, "rgc_proxy_operational_pressure").clip(0, 1)
    else:
        df["dt_context_signal"] = 0.5

    if "composite_disruption_risk_score" in df.columns:
        df["dt_base_risk"] = safe_numeric_series(df, "composite_disruption_risk_score").clip(0, 1)
    elif "risk_label" in df.columns:
        df["dt_base_risk"] = (safe_numeric_series(df, "risk_label") / 2.0).clip(0, 1)
    else:
        df["dt_base_risk"] = df["dt_context_signal"].clip(0, 1)

    if "Late_delivery_risk" in df.columns:
        df["dt_base_delay_prob"] = safe_numeric_series(df, "Late_delivery_risk").clip(0, 1)
    elif "delay_risk" in df.columns:
        df["dt_base_delay_prob"] = safe_numeric_series(df, "delay_risk").clip(0, 1)
    else:
        df["dt_base_delay_prob"] = df["dt_base_risk"].clip(0, 1)

    df["dt_demand_pressure"] = (
        0.45 * df["dt_quantity_pressure"]
        + 0.35 * df["dt_sales_pressure"]
        + 0.20 * df["dt_order_value_pressure"]
    ).clip(0, 1)

    df["dt_cost_pressure"] = (
        0.45 * df["dt_discount_pressure"]
        + 0.35 * df["dt_order_value_pressure"]
        + 0.20 * df["dt_profit_stress"]
    ).clip(0, 1)

    df["dt_service_importance"] = (
        0.35 * df["dt_demand_pressure"]
        + 0.30 * df["dt_base_delay_prob"]
        + 0.20 * df["dt_base_risk"]
        + 0.15 * df["dt_context_signal"]
    ).clip(0, 1)

    state_features = get_state_feature_list()

    for col in state_features:
        df[col] = safe_numeric_series(df, col, default=0.0).clip(0, 1)

    return df, state_features, scaler_stats


def simulate_action_outcome(
    row: pd.Series,
    action_id: int,
    scenario_config: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    if action_id not in ACTION_SPACE:
        raise ValueError(f"Invalid action_id={action_id}. Valid actions={list(ACTION_SPACE.keys())}")

    action = ACTION_SPACE[action_id]

    if scenario_config is None:
        scenario_config = SCENARIO_CONFIGS["normal"]

    base_risk = safe_numeric_value(row, "dt_base_risk", 0.5)
    base_delay = safe_numeric_value(row, "dt_base_delay_prob", 0.5)
    demand_pressure = safe_numeric_value(row, "dt_demand_pressure", 0.5)
    cost_pressure = safe_numeric_value(row, "dt_cost_pressure", 0.5)
    service_importance = safe_numeric_value(row, "dt_service_importance", 0.5)

    sales = max(safe_numeric_value(row, "Sales", 0.0), 0.0)
    order_value = max(safe_numeric_value(row, "Order Item Total", sales), 0.0)
    base_profit = safe_numeric_value(row, "Order Profit Per Order", 0.0)

    if order_value <= 0:
        order_value = max(sales, 1.0)

    scenario_risk_multiplier = float(scenario_config.get("risk_multiplier", 1.0))
    scenario_delay_multiplier = float(scenario_config.get("delay_multiplier", 1.0))
    scenario_demand_multiplier = float(scenario_config.get("demand_multiplier", 1.0))
    scenario_cost_multiplier = float(scenario_config.get("cost_multiplier", 1.0))

    simulated_risk = clip01(
        base_risk
        * action["risk_factor"]
        * scenario_risk_multiplier
        * (1.0 + 0.10 * demand_pressure)
    )

    simulated_delay_prob = clip01(
        base_delay
        * action["delay_factor"]
        * scenario_delay_multiplier
        * (1.0 + 0.08 * demand_pressure)
    )

    simulated_cost = (
        order_value
        * action["cost_factor"]
        * scenario_cost_multiplier
        * (1.0 + 0.05 * cost_pressure)
    )

    service_level = clip01(
        1.0
        - 0.42 * simulated_risk
        - 0.36 * simulated_delay_prob
        - 0.10 * max(scenario_demand_multiplier - 1.0, 0.0)
        + action["service_bonus"]
    )

    resilience_score = clip01(
        1.0
        - 0.50 * simulated_risk
        - 0.30 * simulated_delay_prob
        + action["resilience_bonus"]
    )

    fulfilled_value = order_value * service_level * scenario_demand_multiplier

    simulated_profit = (
        base_profit
        * action["profit_factor"]
        * service_level
        - 0.030 * simulated_cost
        + 0.010 * fulfilled_value
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
        "action_id": int(action_id),
        "action_name": action["action_name"],
        "simulated_risk": float(simulated_risk),
        "simulated_delay_prob": float(simulated_delay_prob),
        "simulated_cost": float(simulated_cost),
        "service_level": float(service_level),
        "resilience_score": float(resilience_score),
        "simulated_profit": float(simulated_profit),
        "reward": float(reward),
    }


class ProcurementDigitalTwinEnv:
    def __init__(
        self,
        data: pd.DataFrame,
        state_features: List[str],
        scenario_name: str = "normal",
        max_steps: Optional[int] = None,
        random_state: int = 42,
    ):
        if scenario_name not in SCENARIO_CONFIGS:
            raise ValueError(f"Unknown scenario_name={scenario_name}. Available={list(SCENARIO_CONFIGS.keys())}")

        self.data = data.copy().reset_index(drop=True)
        self.state_features = list(state_features)
        self.scenario_name = scenario_name
        self.scenario_config = SCENARIO_CONFIGS[scenario_name]
        self.max_steps = max_steps if max_steps is not None else len(self.data)
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)
        self.current_step = 0
        self.episode_reward = 0.0

    @property
    def n_actions(self) -> int:
        return len(ACTION_SPACE)

    @property
    def state_dim(self) -> int:
        return len(self.state_features)

    def reset(self, start_index: Optional[int] = None) -> np.ndarray:
        if start_index is None:
            self.current_step = 0
        else:
            self.current_step = int(np.clip(start_index, 0, max(len(self.data) - 1, 0)))

        self.episode_reward = 0.0
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        if len(self.data) == 0:
            return np.zeros(self.state_dim, dtype=np.float32)

        idx = min(self.current_step, len(self.data) - 1)
        state = (
            self.data.loc[idx, self.state_features]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0)
            .to_numpy(dtype=np.float32)
        )

        return state

    def step(self, action_id: int):
        if len(self.data) == 0:
            next_state = np.zeros(self.state_dim, dtype=np.float32)
            return next_state, 0.0, True, {"empty_environment": True}

        idx = min(self.current_step, len(self.data) - 1)
        row = self.data.loc[idx]

        outcome = simulate_action_outcome(
            row=row,
            action_id=int(action_id),
            scenario_config=self.scenario_config,
        )

        reward = float(outcome["reward"])
        self.episode_reward += reward

        self.current_step += 1

        done = (
            self.current_step >= len(self.data)
            or self.current_step >= self.max_steps
        )

        next_state = (
            np.zeros(self.state_dim, dtype=np.float32)
            if done
            else self._get_state()
        )

        info = dict(outcome)
        info["scenario_name"] = self.scenario_name
        info["data_index"] = int(idx)
        info["episode_reward"] = float(self.episode_reward)

        if "dt_row_id" in row.index:
            info["dt_row_id"] = int(row["dt_row_id"])

        return next_state, reward, done, info


def build_counterfactual_transition_table(
    dt_df: pd.DataFrame,
    scenario_name: str = "normal",
    max_rows_per_split: int = 15000,
    random_state: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    scenario_config = SCENARIO_CONFIGS[scenario_name]

    rows = []

    for split_name, split_df in dt_df.groupby("data_split", observed=False):
        split_df = split_df.copy()

        if len(split_df) > max_rows_per_split:
            sample_idx = rng.choice(split_df.index.to_numpy(), size=max_rows_per_split, replace=False)
            split_df = split_df.loc[sample_idx].copy()

        for _, row in split_df.iterrows():
            for action_id in ACTION_SPACE.keys():
                outcome = simulate_action_outcome(
                    row=row,
                    action_id=action_id,
                    scenario_config=scenario_config,
                )

                rows.append({
                    "scenario_name": scenario_name,
                    "data_split": split_name,
                    "dt_row_id": int(row.get("dt_row_id", -1)),
                    "action_id": action_id,
                    "action_name": outcome["action_name"],
                    "reward": outcome["reward"],
                    "simulated_risk": outcome["simulated_risk"],
                    "simulated_delay_prob": outcome["simulated_delay_prob"],
                    "service_level": outcome["service_level"],
                    "resilience_score": outcome["resilience_score"],
                    "simulated_profit": outcome["simulated_profit"],
                    "simulated_cost": outcome["simulated_cost"],
                })

    return pd.DataFrame(rows)


def summarize_transition_table(transition_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        transition_df.groupby(["scenario_name", "data_split", "action_id", "action_name"], observed=False)
        .agg(
            transition_count=("reward", "count"),
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


def validate_environment_rollout(
    env: ProcurementDigitalTwinEnv,
    policy: str = "random",
    max_steps: int = 500,
) -> Dict[str, float]:
    state = env.reset()
    done = False
    steps = 0
    rewards = []
    risks = []
    delays = []
    services = []
    resiliences = []

    while not done and steps < max_steps:
        if policy == "random":
            action_id = int(env.rng.integers(0, env.n_actions))
        elif policy == "balanced":
            action_id = 1
        elif policy == "resilience":
            action_id = 2
        else:
            action_id = 1

        state, reward, done, info = env.step(action_id)

        rewards.append(reward)
        risks.append(info.get("simulated_risk", 0.0))
        delays.append(info.get("simulated_delay_prob", 0.0))
        services.append(info.get("service_level", 0.0))
        resiliences.append(info.get("resilience_score", 0.0))

        steps += 1

    return {
        "scenario_name": env.scenario_name,
        "policy": policy,
        "steps": int(steps),
        "total_reward": float(np.sum(rewards)) if rewards else 0.0,
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "mean_risk": float(np.mean(risks)) if risks else 0.0,
        "mean_delay": float(np.mean(delays)) if delays else 0.0,
        "mean_service": float(np.mean(services)) if services else 0.0,
        "mean_resilience": float(np.mean(resiliences)) if resiliences else 0.0,
    }