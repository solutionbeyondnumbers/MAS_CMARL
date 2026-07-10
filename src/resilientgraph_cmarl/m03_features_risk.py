from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


ORDER_DATE_COL = "order date (DateOrders)"

GROUP_COLS = [
    "Category Name",
    "Order Region",
    "Shipping Mode",
]

NUMERIC_RISK_COLUMNS = [
    "Order Item Quantity",
    "Sales",
    "Order Profit Per Order",
    "shipping_delay_gap",
    "profit_margin",
]


def safe_minmax(series: pd.Series) -> pd.Series:
    series = pd.to_numeric(series, errors="coerce").fillna(0)
    min_val = series.min()
    max_val = series.max()

    if max_val == min_val:
        return pd.Series(np.zeros(len(series)), index=series.index)

    return (series - min_val) / (max_val - min_val)


def safe_clip_minmax(series: pd.Series, lower_q: float = 0.01, upper_q: float = 0.99) -> pd.Series:
    series = pd.to_numeric(series, errors="coerce").fillna(0)
    lower = series.quantile(lower_q)
    upper = series.quantile(upper_q)

    if upper == lower:
        return pd.Series(np.zeros(len(series)), index=series.index)

    clipped = series.clip(lower, upper)
    return safe_minmax(clipped)


def parse_and_sort(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if ORDER_DATE_COL not in df.columns:
        raise KeyError(f"Required date column not found: {ORDER_DATE_COL}")

    df[ORDER_DATE_COL] = pd.to_datetime(df[ORDER_DATE_COL], errors="coerce")
    df = df.dropna(subset=[ORDER_DATE_COL]).copy()

    existing_group_cols = [c for c in GROUP_COLS if c in df.columns]
    sort_cols = existing_group_cols + [ORDER_DATE_COL]

    df = df.sort_values(sort_cols).reset_index(drop=True)

    return df


def grouped_shift(df: pd.DataFrame, group_cols: List[str], value_col: str, lag: int) -> pd.Series:
    if group_cols:
        return df.groupby(group_cols, observed=False)[value_col].transform(lambda s: s.shift(lag))
    return df[value_col].shift(lag)


def grouped_shift_rolling(
    df: pd.DataFrame,
    group_cols: List[str],
    value_col: str,
    window: int,
    stat: str,
    shift_periods: int = 1,
) -> pd.Series:
    if group_cols:
        grouped = df.groupby(group_cols, observed=False)[value_col]

        if stat == "mean":
            return grouped.transform(
                lambda s: s.shift(shift_periods).rolling(window=window, min_periods=1).mean()
            )
        if stat == "std":
            return grouped.transform(
                lambda s: s.shift(shift_periods).rolling(window=window, min_periods=1).std()
            )

    shifted = df[value_col].shift(shift_periods)

    if stat == "mean":
        return shifted.rolling(window=window, min_periods=1).mean()
    if stat == "std":
        return shifted.rolling(window=window, min_periods=1).std()

    raise ValueError(f"Unsupported rolling stat: {stat}")


def add_temporal_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    group_cols = [c for c in GROUP_COLS if c in df.columns]

    if "Order Item Quantity" in df.columns:
        qty_col = "Order Item Quantity"

        df["qty_lag_1"] = grouped_shift(df, group_cols, qty_col, lag=1)
        df["qty_lag_3"] = grouped_shift(df, group_cols, qty_col, lag=3)
        df["qty_lag_7"] = grouped_shift(df, group_cols, qty_col, lag=7)

        df["qty_roll_mean_3"] = grouped_shift_rolling(
            df, group_cols, qty_col, window=3, stat="mean"
        )
        df["qty_roll_mean_7"] = grouped_shift_rolling(
            df, group_cols, qty_col, window=7, stat="mean"
        )
        df["qty_roll_std_7"] = grouped_shift_rolling(
            df, group_cols, qty_col, window=7, stat="std"
        )

    if "shipping_delay_gap" in df.columns:
        delay_col = "shipping_delay_gap"

        df["delay_lag_1"] = grouped_shift(df, group_cols, delay_col, lag=1)
        df["delay_roll_mean_7"] = grouped_shift_rolling(
            df, group_cols, delay_col, window=7, stat="mean"
        )
        df["delay_roll_std_7"] = grouped_shift_rolling(
            df, group_cols, delay_col, window=7, stat="std"
        )

    if "Order Profit Per Order" in df.columns:
        profit_col = "Order Profit Per Order"

        df["profit_lag_1"] = grouped_shift(df, group_cols, profit_col, lag=1)
        df["profit_roll_mean_7"] = grouped_shift_rolling(
            df, group_cols, profit_col, window=7, stat="mean"
        )
        df["profit_roll_std_7"] = grouped_shift_rolling(
            df, group_cols, profit_col, window=7, stat="std"
        )

    lag_cols = [
        "qty_lag_1",
        "qty_lag_3",
        "qty_lag_7",
        "qty_roll_mean_3",
        "qty_roll_mean_7",
        "qty_roll_std_7",
        "delay_lag_1",
        "delay_roll_mean_7",
        "delay_roll_std_7",
        "profit_lag_1",
        "profit_roll_mean_7",
        "profit_roll_std_7",
    ]

    for col in lag_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    return df


def add_historical_shipping_risk(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "Shipping Mode" not in df.columns or "Late_delivery_risk" not in df.columns:
        df["shipping_mode_historical_late_rate"] = 0
        return df

    global_mean = pd.to_numeric(df["Late_delivery_risk"], errors="coerce").fillna(0).mean()

    df["shipping_mode_historical_late_rate"] = (
        df.groupby("Shipping Mode", observed=False)["Late_delivery_risk"]
        .transform(lambda s: pd.to_numeric(s, errors="coerce").fillna(0).shift(1).expanding(min_periods=1).mean())
        .fillna(global_mean)
    )

    return df


def add_anomaly_risk_proxy(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    available_cols = [c for c in NUMERIC_RISK_COLUMNS if c in df.columns]

    if not available_cols:
        df["anomaly_risk"] = 0
        return df

    z_components = []

    for col in available_cols:
        x = pd.to_numeric(df[col], errors="coerce").fillna(0)
        median = x.median()
        mad = np.median(np.abs(x - median))

        if mad == 0:
            robust_z = pd.Series(np.zeros(len(x)), index=x.index)
        else:
            robust_z = np.abs((x - median) / (1.4826 * mad))

        z_components.append(safe_clip_minmax(robust_z))

    anomaly = pd.concat(z_components, axis=1).mean(axis=1)
    df["anomaly_risk"] = anomaly.fillna(0).clip(0, 1)

    return df


def construct_risk_components(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "shipping_delay_gap" in df.columns:
        positive_delay = pd.to_numeric(df["shipping_delay_gap"], errors="coerce").fillna(0).clip(lower=0)
        df["delay_risk"] = safe_clip_minmax(positive_delay)
    else:
        df["delay_risk"] = 0

    if "Late_delivery_risk" in df.columns:
        df["late_delivery_component"] = (
            pd.to_numeric(df["Late_delivery_risk"], errors="coerce")
            .fillna(0)
            .clip(0, 1)
        )
    else:
        df["late_delivery_component"] = 0

    if "qty_roll_std_7" in df.columns:
        df["demand_volatility_risk"] = safe_clip_minmax(df["qty_roll_std_7"])
    else:
        df["demand_volatility_risk"] = 0

    if "qty_roll_mean_7" in df.columns and "Order Item Quantity" in df.columns:
        demand_pressure = (
            pd.to_numeric(df["Order Item Quantity"], errors="coerce").fillna(0)
            - pd.to_numeric(df["qty_roll_mean_7"], errors="coerce").fillna(0)
        ).clip(lower=0)
        df["shortage_exposure_risk"] = safe_clip_minmax(demand_pressure)
    else:
        df["shortage_exposure_risk"] = 0

    if "profit_margin" in df.columns:
        profit_loss = -pd.to_numeric(df["profit_margin"], errors="coerce").fillna(0)
        profit_loss = profit_loss.clip(lower=0)
        df["profit_loss_risk"] = safe_clip_minmax(profit_loss)
    elif "Order Profit Per Order" in df.columns:
        profit_loss = -pd.to_numeric(df["Order Profit Per Order"], errors="coerce").fillna(0)
        profit_loss = profit_loss.clip(lower=0)
        df["profit_loss_risk"] = safe_clip_minmax(profit_loss)
    else:
        df["profit_loss_risk"] = 0

    if "shipping_mode_historical_late_rate" in df.columns:
        df["shipping_risk"] = safe_clip_minmax(df["shipping_mode_historical_late_rate"])
    else:
        df["shipping_risk"] = 0

    if "anomaly_risk" not in df.columns:
        df["anomaly_risk"] = 0

    component_cols = [
        "delay_risk",
        "late_delivery_component",
        "demand_volatility_risk",
        "anomaly_risk",
        "shortage_exposure_risk",
        "profit_loss_risk",
        "shipping_risk",
    ]

    for col in component_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).clip(0, 1)

    return df


def create_composite_risk_score(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()

    weights = {
        "delay_risk": 0.18,
        "late_delivery_component": 0.22,
        "demand_volatility_risk": 0.14,
        "anomaly_risk": 0.14,
        "shortage_exposure_risk": 0.12,
        "profit_loss_risk": 0.10,
        "shipping_risk": 0.10,
    }

    df["composite_disruption_risk_score"] = 0.0

    for col, weight in weights.items():
        if col in df.columns:
            df["composite_disruption_risk_score"] += df[col] * weight

    df["composite_disruption_risk_score"] = df["composite_disruption_risk_score"].clip(0, 1)

    q_low = df["composite_disruption_risk_score"].quantile(0.33)
    q_high = df["composite_disruption_risk_score"].quantile(0.66)

    conditions = [
        df["composite_disruption_risk_score"] <= q_low,
        (df["composite_disruption_risk_score"] > q_low)
        & (df["composite_disruption_risk_score"] <= q_high),
        df["composite_disruption_risk_score"] > q_high,
    ]

    df["risk_label"] = np.select(conditions, [0, 1, 2], default=1).astype(int)
    df["risk_label_name"] = df["risk_label"].map({0: "Low", 1: "Moderate", 2: "High"})

    weight_rows = []

    for col, weight in weights.items():
        weight_rows.append({
            "component": col,
            "weight": weight,
            "component_mean": float(df[col].mean()) if col in df.columns else 0,
            "component_min": float(df[col].min()) if col in df.columns else 0,
            "component_max": float(df[col].max()) if col in df.columns else 0,
        })

    weight_rows.append({
        "component": "low_to_moderate_q33",
        "weight": float(q_low),
        "component_mean": "",
        "component_min": "",
        "component_max": "",
    })

    weight_rows.append({
        "component": "moderate_to_high_q66",
        "weight": float(q_high),
        "component_mean": "",
        "component_min": "",
        "component_max": "",
    })

    risk_config_report = pd.DataFrame(weight_rows)

    return df, risk_config_report


def make_risk_distribution_report(df: pd.DataFrame) -> pd.DataFrame:
    counts = df["risk_label_name"].value_counts().reset_index()
    counts.columns = ["risk_label_name", "count"]
    counts["percent"] = (counts["count"] / len(df) * 100).round(4)

    order = {"Low": 0, "Moderate": 1, "High": 2}
    counts["sort_order"] = counts["risk_label_name"].map(order)
    counts = counts.sort_values("sort_order").drop(columns=["sort_order"])

    return counts


def make_feature_summary_report(df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [
        "qty_lag_1",
        "qty_lag_3",
        "qty_lag_7",
        "qty_roll_mean_3",
        "qty_roll_mean_7",
        "qty_roll_std_7",
        "delay_lag_1",
        "delay_roll_mean_7",
        "delay_roll_std_7",
        "profit_lag_1",
        "profit_roll_mean_7",
        "profit_roll_std_7",
        "shipping_mode_historical_late_rate",
        "delay_risk",
        "late_delivery_component",
        "demand_volatility_risk",
        "anomaly_risk",
        "shortage_exposure_risk",
        "profit_loss_risk",
        "shipping_risk",
        "composite_disruption_risk_score",
        "risk_label",
    ]

    rows = []

    for col in feature_cols:
        if col in df.columns:
            numeric_col = pd.to_numeric(df[col], errors="coerce")
            rows.append({
                "feature": col,
                "missing": int(df[col].isna().sum()),
                "mean": float(numeric_col.mean()),
                "std": float(numeric_col.std()),
                "min": float(numeric_col.min()),
                "max": float(numeric_col.max()),
            })

    return pd.DataFrame(rows)


def build_feature_engineered_dataset(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    engineered = parse_and_sort(df)

    engineered = add_temporal_lag_features(engineered)
    engineered = add_historical_shipping_risk(engineered)
    engineered = add_anomaly_risk_proxy(engineered)
    engineered = construct_risk_components(engineered)
    engineered, risk_config_report = create_composite_risk_score(engineered)

    risk_distribution_report = make_risk_distribution_report(engineered)
    feature_summary_report = make_feature_summary_report(engineered)

    return {
        "engineered_df": engineered,
        "risk_config_report": risk_config_report,
        "risk_distribution_report": risk_distribution_report,
        "feature_summary_report": feature_summary_report,
    }