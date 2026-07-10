import sys
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from resilientgraph_cmarl.m00_config import (
    PROCESSED_DIR,
    GRAPH_DIR,
    TABLES_DIR,
    REPORTS_DIR,
    LOGS_DIR,
    MODELS_DIR,
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
from resilientgraph_cmarl.m04_graph_risk import (
    build_graph_risk_dataset,
    prepare_graph_entity_columns,
    save_graph_pickle,
)
from resilientgraph_cmarl.m05_baseline_proposed_models import (
    TARGET_COL,
    make_temporal_split,
    get_baseline_feature_columns,
    split_x_y,
    train_evaluate_single_model,
    make_metrics_table,
    make_prediction_table,
    calculate_metrics,
)


INPUT_FILE = PROCESSED_DIR / "05_dataco_graph_risk_features.csv"
BASELINE_METRICS_FILE = TABLES_DIR / "06_baseline_model_metrics.csv"
BASELINE_PREDICTIONS_FILE = TABLES_DIR / "06_baseline_predictions.csv"

OUTPUT_RGC_FEATURE_FILE = PROCESSED_DIR / "07_dataco_leakage_safe_rgc_features.csv"

TRAIN_SAFE_NODES_FILE = GRAPH_DIR / "07_leakage_safe_train_graph_nodes.csv"
TRAIN_SAFE_EDGES_FILE = GRAPH_DIR / "07_leakage_safe_train_graph_edges.csv"
TRAIN_SAFE_GRAPH_FILE = GRAPH_DIR / "07_leakage_safe_train_supply_chain_graph.pkl"

PROPOSED_MODEL_MAP = {
    "XGBoost-RGC": "XGBoost",
    "LightGBM-RGC": "LightGBM",
    "CatBoost-RGC": "CatBoost",
}

ENTITY_PRIOR_COLUMNS = [
    "Category Name",
    "Product Name",
    "Order Region",
    "Market",
    "Shipping Mode",
    "Customer Segment",
    "Department Name",
    "Order Country",
    "Order State",
    "Customer Country",
]

GRAPH_ENTITY_COLUMNS = [
    "product",
    "category",
    "region",
    "market",
    "shipping",
    "supplier_proxy",
]

# These columns are not allowed in Step 07 because they directly expose outcome or label construction.
STRICT_FORBIDDEN_COLUMNS = {
    TARGET_COL,
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
    "shipping_year",
    "shipping_month",
    "shipping_week",
    "shipping_day",
    "shipping_dayofweek",
    "shipping_quarter",
}

# Controlled non-outcome proxy indicators.
# These are not the full target formula and exclude direct delivery-outcome indicators.
CONTROLLED_PROXY_RISK_COLUMNS = [
    "demand_volatility_risk",
    "anomaly_risk",
    "shortage_exposure_risk",
    "profit_loss_risk",
]

SAFE_OPERATIONAL_EXTRA_COLUMNS = [
    "Days for shipment (scheduled)",
    "Order Item Discount Rate",
    "Order Item Discount",
    "Order Item Quantity",
    "Order Item Product Price",
    "Order Item Total",
    "Sales",
    "Product Price",
    "sales_per_unit",
    "profit_margin",
    "qty_lag_1",
    "qty_lag_3",
    "qty_lag_7",
    "qty_roll_mean_3",
    "qty_roll_mean_7",
    "qty_roll_std_7",
    "profit_lag_1",
    "profit_roll_mean_7",
    "profit_roll_std_7",
]

ALPHA_GRID = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.70, 0.85, 1.0]
CLASS_WEIGHT_GRID = [0.80, 0.90, 1.00, 1.10, 1.20, 1.35]

# Any validation score above this is treated as leakage-risk for this dataset.
LEAKAGE_GUARD_UPPER_BOUND = 0.95


def safe_name(col: str) -> str:
    return (
        col.lower()
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("/", "_")
        .replace("-", "_")
    )


def clean_entity(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .replace({"nan": "Unknown", "None": "Unknown", "": "Unknown"})
    )


def drop_unsafe_existing_graph_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    drop_prefixes = [
        "graph_",
        "sthg_",
        "node_",
        "rgc_",
    ]

    drop_cols = [
        col for col in df.columns
        if any(str(col).startswith(prefix) for prefix in drop_prefixes)
    ]

    if drop_cols:
        df = df.drop(columns=drop_cols)

    return df


def remove_forbidden_features(cols: List[str]) -> List[str]:
    cleaned = []

    for col in cols:
        if col in STRICT_FORBIDDEN_COLUMNS:
            continue
        if col not in cleaned:
            cleaned.append(col)

    return cleaned


def build_entity_prior_maps(train_df: pd.DataFrame) -> Dict[str, Dict[str, object]]:
    maps = {}

    global_mean = float(train_df[TARGET_COL].mean())
    global_high_rate = float((train_df[TARGET_COL] == 2).mean())
    global_low_rate = float((train_df[TARGET_COL] == 0).mean())
    global_mod_rate = float((train_df[TARGET_COL] == 1).mean())

    smoothing = 60

    for col in ENTITY_PRIOR_COLUMNS:
        if col not in train_df.columns:
            continue

        temp = train_df[[col, TARGET_COL]].copy()
        temp[col] = clean_entity(temp[col])
        temp[TARGET_COL] = pd.to_numeric(temp[TARGET_COL], errors="coerce").fillna(0).astype(int)

        grouped = (
            temp.groupby(col, observed=False)
            .agg(
                entity_count=(TARGET_COL, "count"),
                risk_mean=(TARGET_COL, "mean"),
                high_rate=(TARGET_COL, lambda s: float((s == 2).mean())),
                moderate_rate=(TARGET_COL, lambda s: float((s == 1).mean())),
                low_rate=(TARGET_COL, lambda s: float((s == 0).mean())),
            )
            .reset_index()
        )

        grouped["risk_mean_smooth"] = (
            grouped["risk_mean"] * grouped["entity_count"] + global_mean * smoothing
        ) / (grouped["entity_count"] + smoothing)

        grouped["high_rate_smooth"] = (
            grouped["high_rate"] * grouped["entity_count"] + global_high_rate * smoothing
        ) / (grouped["entity_count"] + smoothing)

        grouped["moderate_rate_smooth"] = (
            grouped["moderate_rate"] * grouped["entity_count"] + global_mod_rate * smoothing
        ) / (grouped["entity_count"] + smoothing)

        grouped["low_rate_smooth"] = (
            grouped["low_rate"] * grouped["entity_count"] + global_low_rate * smoothing
        ) / (grouped["entity_count"] + smoothing)

        maps[col] = {
            "risk_mean": grouped.set_index(col)["risk_mean_smooth"].to_dict(),
            "high_rate": grouped.set_index(col)["high_rate_smooth"].to_dict(),
            "moderate_rate": grouped.set_index(col)["moderate_rate_smooth"].to_dict(),
            "low_rate": grouped.set_index(col)["low_rate_smooth"].to_dict(),
            "count": grouped.set_index(col)["entity_count"].to_dict(),
            "global_mean": global_mean,
            "global_high_rate": global_high_rate,
            "global_moderate_rate": global_mod_rate,
            "global_low_rate": global_low_rate,
        }

    return maps


def add_time_ordered_oof_priors(train_df: pd.DataFrame, n_folds: int = 6) -> pd.DataFrame:
    train_df = train_df.copy().reset_index(drop=True)

    global_mean = float(train_df[TARGET_COL].mean())
    global_high_rate = float((train_df[TARGET_COL] == 2).mean())
    global_low_rate = float((train_df[TARGET_COL] == 0).mean())
    global_mod_rate = float((train_df[TARGET_COL] == 1).mean())

    n = len(train_df)
    fold_edges = np.linspace(0, n, n_folds + 1).astype(int)

    for col in ENTITY_PRIOR_COLUMNS:
        if col not in train_df.columns:
            continue

        sname = safe_name(col)
        train_df[col] = clean_entity(train_df[col])

        train_df[f"rgc_prior_{sname}_risk_mean"] = global_mean
        train_df[f"rgc_prior_{sname}_high_rate"] = global_high_rate
        train_df[f"rgc_prior_{sname}_moderate_rate"] = global_mod_rate
        train_df[f"rgc_prior_{sname}_low_rate"] = global_low_rate
        train_df[f"rgc_prior_{sname}_log_count"] = 0.0

        for fold_idx in range(n_folds):
            start = fold_edges[fold_idx]
            end = fold_edges[fold_idx + 1]

            if start == 0:
                continue

            history = train_df.iloc[:start].copy()
            current_index = train_df.index[start:end]

            maps = build_entity_prior_maps(history)

            if col not in maps:
                continue

            col_map = maps[col]
            values = train_df.loc[current_index, col]

            train_df.loc[current_index, f"rgc_prior_{sname}_risk_mean"] = (
                values.map(col_map["risk_mean"]).fillna(col_map["global_mean"])
            )
            train_df.loc[current_index, f"rgc_prior_{sname}_high_rate"] = (
                values.map(col_map["high_rate"]).fillna(col_map["global_high_rate"])
            )
            train_df.loc[current_index, f"rgc_prior_{sname}_moderate_rate"] = (
                values.map(col_map["moderate_rate"]).fillna(col_map["global_moderate_rate"])
            )
            train_df.loc[current_index, f"rgc_prior_{sname}_low_rate"] = (
                values.map(col_map["low_rate"]).fillna(col_map["global_low_rate"])
            )
            train_df.loc[current_index, f"rgc_prior_{sname}_log_count"] = (
                np.log1p(values.map(col_map["count"]).fillna(0))
            )

    return train_df


def apply_train_prior_maps(df: pd.DataFrame, maps: Dict[str, Dict[str, object]]) -> pd.DataFrame:
    df = df.copy()

    for col, col_map in maps.items():
        if col not in df.columns:
            continue

        sname = safe_name(col)
        values = clean_entity(df[col])

        df[f"rgc_prior_{sname}_risk_mean"] = (
            values.map(col_map["risk_mean"]).fillna(col_map["global_mean"])
        )
        df[f"rgc_prior_{sname}_high_rate"] = (
            values.map(col_map["high_rate"]).fillna(col_map["global_high_rate"])
        )
        df[f"rgc_prior_{sname}_moderate_rate"] = (
            values.map(col_map["moderate_rate"]).fillna(col_map["global_moderate_rate"])
        )
        df[f"rgc_prior_{sname}_low_rate"] = (
            values.map(col_map["low_rate"]).fillna(col_map["global_low_rate"])
        )
        df[f"rgc_prior_{sname}_log_count"] = (
            np.log1p(values.map(col_map["count"]).fillna(0))
        )

    return df


def build_train_safe_graph_features(split_data: Dict[str, pd.DataFrame]) -> Dict[str, object]:
    print_subheader("Building Leakage-Safe Train-Only Graph Context")

    clean_train = drop_unsafe_existing_graph_columns(split_data["train"])
    graph_results = build_graph_risk_dataset(clean_train)

    train_nodes = graph_results["nodes"]
    train_edges = graph_results["edges"]
    train_graph = graph_results["graph"]

    save_csv(train_nodes, TRAIN_SAFE_NODES_FILE)
    save_csv(train_edges, TRAIN_SAFE_EDGES_FILE)
    save_graph_pickle(train_graph, TRAIN_SAFE_GRAPH_FILE)

    save_csv(graph_results["node_type_summary"], TABLES_DIR / "07_leakage_safe_graph_node_type_summary.csv")
    save_csv(graph_results["edge_relation_summary"], TABLES_DIR / "07_leakage_safe_graph_edge_relation_summary.csv")
    save_csv(graph_results["top_risk_nodes"], TABLES_DIR / "07_leakage_safe_top_risk_nodes.csv")

    print(f"[TRAIN-SAFE GRAPH NODES] {train_nodes.shape[0]:,}")
    print(f"[TRAIN-SAFE GRAPH EDGES] {train_edges.shape[0]:,}")
    print(f"[SAVED] {TRAIN_SAFE_GRAPH_FILE}")

    node_lookup = train_nodes.set_index("node_id").to_dict(orient="index")
    rgc_splits = {}

    for split_name, split_df in split_data.items():
        clean_split = drop_unsafe_existing_graph_columns(split_df)
        entity_df = prepare_graph_entity_columns(clean_split)
        df = entity_df.copy()

        for entity in GRAPH_ENTITY_COLUMNS:
            node_col = f"node_{entity}"

            if node_col not in df.columns:
                continue

            risk_values = []
            centrality_values = []
            pressure_values = []
            pagerank_values = []
            degree_values = []

            for node_id in df[node_col].astype(str):
                item = node_lookup.get(node_id, {})
                risk_values.append(float(item.get("avg_risk_score", 0.0)))
                centrality_values.append(float(item.get("graph_risk_centrality", 0.0)))
                pressure_values.append(float(item.get("node_risk_pressure", 0.0)))
                pagerank_values.append(float(item.get("pagerank", 0.0)))
                degree_values.append(float(item.get("degree_centrality", 0.0)))

            df[f"rgc_graph_{entity}_risk"] = risk_values
            df[f"rgc_graph_{entity}_centrality"] = centrality_values
            df[f"rgc_graph_{entity}_pressure"] = pressure_values
            df[f"rgc_graph_{entity}_pagerank"] = pagerank_values
            df[f"rgc_graph_{entity}_degree"] = degree_values

        risk_cols = [c for c in df.columns if c.startswith("rgc_graph_") and c.endswith("_risk")]
        cent_cols = [c for c in df.columns if c.startswith("rgc_graph_") and c.endswith("_centrality")]
        pressure_cols = [c for c in df.columns if c.startswith("rgc_graph_") and c.endswith("_pressure")]
        pagerank_cols = [c for c in df.columns if c.startswith("rgc_graph_") and c.endswith("_pagerank")]
        degree_cols = [c for c in df.columns if c.startswith("rgc_graph_") and c.endswith("_degree")]

        df["rgc_graph_mean_risk"] = df[risk_cols].mean(axis=1) if risk_cols else 0.0
        df["rgc_graph_max_risk"] = df[risk_cols].max(axis=1) if risk_cols else 0.0
        df["rgc_graph_std_risk"] = df[risk_cols].std(axis=1).fillna(0) if risk_cols else 0.0
        df["rgc_graph_mean_centrality"] = df[cent_cols].mean(axis=1) if cent_cols else 0.0
        df["rgc_graph_max_centrality"] = df[cent_cols].max(axis=1) if cent_cols else 0.0
        df["rgc_graph_pressure_rank"] = df[pressure_cols].mean(axis=1).rank(pct=True) if pressure_cols else 0.0
        df["rgc_graph_pagerank_rank"] = df[pagerank_cols].mean(axis=1).rank(pct=True) if pagerank_cols else 0.0
        df["rgc_graph_degree_rank"] = df[degree_cols].mean(axis=1).rank(pct=True) if degree_cols else 0.0

        df["rgc_graph_signal"] = (
            0.25 * df["rgc_graph_mean_risk"]
            + 0.25 * df["rgc_graph_max_risk"]
            + 0.20 * df["rgc_graph_mean_centrality"]
            + 0.10 * df["rgc_graph_pressure_rank"]
            + 0.10 * df["rgc_graph_pagerank_rank"]
            + 0.10 * df["rgc_graph_degree_rank"]
        ).fillna(0)

        rgc_splits[split_name] = df

    graph_results["rgc_splits"] = rgc_splits
    return graph_results


def train_scaler_stats(train_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    stats = {}

    source_cols = [
        "Order Item Discount Rate",
        "Order Item Quantity",
        "Order Item Total",
        "Sales",
        "Product Price",
        "sales_per_unit",
        "profit_margin",
        "Days for shipment (scheduled)",
    ]

    for col in source_cols:
        if col not in train_df.columns:
            continue

        s = pd.to_numeric(train_df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)
        stats[col] = {
            "q01": float(s.quantile(0.01)),
            "q99": float(s.quantile(0.99)),
            "median": float(s.median()),
        }

    return stats


def robust_minmax(series: pd.Series, col_stats: Dict[str, float]) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(col_stats["median"])
    low = col_stats["q01"]
    high = col_stats["q99"]

    if high <= low:
        return pd.Series(np.zeros(len(series)), index=series.index)

    return ((s.clip(low, high) - low) / (high - low)).fillna(0)


def add_operational_proxy_features(rgc_splits: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    stats = train_scaler_stats(rgc_splits["train"])

    for split_name, df in rgc_splits.items():
        df = df.copy()

        if "Order Item Discount Rate" in stats and "Order Item Discount Rate" in df.columns:
            df["rgc_proxy_discount_pressure"] = robust_minmax(df["Order Item Discount Rate"], stats["Order Item Discount Rate"])
        else:
            df["rgc_proxy_discount_pressure"] = 0.0

        if "Order Item Quantity" in stats and "Order Item Quantity" in df.columns:
            df["rgc_proxy_quantity_pressure"] = robust_minmax(df["Order Item Quantity"], stats["Order Item Quantity"])
        else:
            df["rgc_proxy_quantity_pressure"] = 0.0

        if "Sales" in stats and "Sales" in df.columns:
            df["rgc_proxy_sales_pressure"] = robust_minmax(df["Sales"], stats["Sales"])
        else:
            df["rgc_proxy_sales_pressure"] = 0.0

        if "Order Item Total" in stats and "Order Item Total" in df.columns:
            df["rgc_proxy_order_value_pressure"] = robust_minmax(df["Order Item Total"], stats["Order Item Total"])
        else:
            df["rgc_proxy_order_value_pressure"] = 0.0

        if "Days for shipment (scheduled)" in stats and "Days for shipment (scheduled)" in df.columns:
            df["rgc_proxy_schedule_pressure"] = robust_minmax(df["Days for shipment (scheduled)"], stats["Days for shipment (scheduled)"])
        else:
            df["rgc_proxy_schedule_pressure"] = 0.0

        if "profit_margin" in stats and "profit_margin" in df.columns:
            pm_scaled = robust_minmax(df["profit_margin"], stats["profit_margin"])
            df["rgc_proxy_profit_stress"] = (1.0 - pm_scaled).clip(0, 1)
        else:
            df["rgc_proxy_profit_stress"] = 0.0

        df["rgc_proxy_operational_pressure"] = (
            0.20 * df["rgc_proxy_discount_pressure"]
            + 0.15 * df["rgc_proxy_quantity_pressure"]
            + 0.15 * df["rgc_proxy_sales_pressure"]
            + 0.15 * df["rgc_proxy_order_value_pressure"]
            + 0.15 * df["rgc_proxy_schedule_pressure"]
            + 0.20 * df["rgc_proxy_profit_stress"]
        ).fillna(0)

        rgc_splits[split_name] = df

    return rgc_splits


def add_leakage_safe_rgc_features(split_data: Dict[str, pd.DataFrame]) -> Dict[str, object]:
    graph_pack = build_train_safe_graph_features(split_data)
    rgc_splits = graph_pack["rgc_splits"]

    print_subheader("Adding Time-Ordered Out-of-Fold Entity Priors")
    train_oof = add_time_ordered_oof_priors(rgc_splits["train"])
    prior_maps = build_entity_prior_maps(rgc_splits["train"])

    rgc_splits["train"] = train_oof
    rgc_splits["valid"] = apply_train_prior_maps(rgc_splits["valid"], prior_maps)
    rgc_splits["test"] = apply_train_prior_maps(rgc_splits["test"], prior_maps)

    print_subheader("Adding Operational Proxy-Risk Features")
    rgc_splits = add_operational_proxy_features(rgc_splits)

    for split_name, df in rgc_splits.items():
        prior_risk_cols = [c for c in df.columns if c.startswith("rgc_prior_") and c.endswith("_risk_mean")]
        prior_high_cols = [c for c in df.columns if c.startswith("rgc_prior_") and c.endswith("_high_rate")]
        prior_mod_cols = [c for c in df.columns if c.startswith("rgc_prior_") and c.endswith("_moderate_rate")]
        prior_low_cols = [c for c in df.columns if c.startswith("rgc_prior_") and c.endswith("_low_rate")]
        prior_count_cols = [c for c in df.columns if c.startswith("rgc_prior_") and c.endswith("_log_count")]

        df["rgc_prior_mean_risk"] = df[prior_risk_cols].mean(axis=1) if prior_risk_cols else 0.0
        df["rgc_prior_max_risk"] = df[prior_risk_cols].max(axis=1) if prior_risk_cols else 0.0
        df["rgc_prior_std_risk"] = df[prior_risk_cols].std(axis=1).fillna(0) if prior_risk_cols else 0.0

        df["rgc_prior_mean_high"] = df[prior_high_cols].mean(axis=1) if prior_high_cols else 0.0
        df["rgc_prior_max_high"] = df[prior_high_cols].max(axis=1) if prior_high_cols else 0.0
        df["rgc_prior_mean_moderate"] = df[prior_mod_cols].mean(axis=1) if prior_mod_cols else 0.0
        df["rgc_prior_mean_low"] = df[prior_low_cols].mean(axis=1) if prior_low_cols else 0.0
        df["rgc_prior_mean_log_count"] = df[prior_count_cols].mean(axis=1) if prior_count_cols else 0.0

        df["rgc_context_signal"] = (
            0.25 * df["rgc_graph_signal"]
            + 0.20 * df["rgc_prior_mean_risk"]
            + 0.15 * df["rgc_prior_max_risk"]
            + 0.15 * df["rgc_prior_mean_high"]
            + 0.10 * df["rgc_prior_mean_moderate"]
            + 0.15 * df["rgc_proxy_operational_pressure"]
        ).fillna(0)

        interaction_base = [
            "rgc_proxy_operational_pressure",
            "rgc_proxy_profit_stress",
            "rgc_proxy_discount_pressure",
            "rgc_context_signal",
        ]

        for col in interaction_base:
            df[f"rgc_interact_graph_x_{col}"] = (
                pd.to_numeric(df["rgc_graph_signal"], errors="coerce").fillna(0)
                * pd.to_numeric(df[col], errors="coerce").fillna(0)
            )

        df["data_split"] = split_name
        rgc_splits[split_name] = df

    graph_pack["rgc_splits"] = rgc_splits
    graph_pack["prior_maps"] = prior_maps

    return graph_pack


def get_candidate_feature_sets(rgc_train_df: pd.DataFrame, original_df: pd.DataFrame) -> Dict[str, List[str]]:
    baseline_cols = get_baseline_feature_columns(original_df)
    baseline_cols = remove_forbidden_features(baseline_cols)

    safe_extra_cols = [
        c for c in SAFE_OPERATIONAL_EXTRA_COLUMNS
        if c in rgc_train_df.columns and c not in STRICT_FORBIDDEN_COLUMNS
    ]

    proxy_indicator_cols = [
        c for c in CONTROLLED_PROXY_RISK_COLUMNS
        if c in rgc_train_df.columns and c not in STRICT_FORBIDDEN_COLUMNS
    ]

    graph_summary_cols = [
        c for c in rgc_train_df.columns
        if c in [
            "rgc_graph_signal",
            "rgc_graph_mean_risk",
            "rgc_graph_max_risk",
            "rgc_graph_std_risk",
            "rgc_graph_mean_centrality",
            "rgc_graph_max_centrality",
            "rgc_graph_pressure_rank",
            "rgc_graph_pagerank_rank",
            "rgc_graph_degree_rank",
        ]
    ]

    prior_summary_cols = [
        c for c in rgc_train_df.columns
        if c in [
            "rgc_prior_mean_risk",
            "rgc_prior_max_risk",
            "rgc_prior_std_risk",
            "rgc_prior_mean_high",
            "rgc_prior_max_high",
            "rgc_prior_mean_moderate",
            "rgc_prior_mean_low",
            "rgc_prior_mean_log_count",
        ]
    ]

    operational_proxy_cols = [
        c for c in rgc_train_df.columns
        if c.startswith("rgc_proxy_") or c == "rgc_context_signal"
    ]

    interaction_cols = [
        c for c in rgc_train_df.columns
        if c.startswith("rgc_interact_")
    ]

    detailed_prior_cols = [
        c for c in rgc_train_df.columns
        if c.startswith("rgc_prior_")
        and c not in prior_summary_cols
    ]

    candidate_sets = {
        "RGC-StrictContext": (
            baseline_cols
            + safe_extra_cols
            + graph_summary_cols
            + prior_summary_cols
            + operational_proxy_cols
            + interaction_cols
        ),
        "RGC-ProxyRiskContext": (
            baseline_cols
            + safe_extra_cols
            + proxy_indicator_cols
            + graph_summary_cols
            + prior_summary_cols
            + operational_proxy_cols
            + interaction_cols
        ),
        "RGC-ExpandedContext": (
            baseline_cols
            + safe_extra_cols
            + proxy_indicator_cols
            + graph_summary_cols
            + prior_summary_cols
            + detailed_prior_cols
            + operational_proxy_cols
            + interaction_cols
        ),
    }

    final_sets = {}

    for name, cols in candidate_sets.items():
        cleaned = []
        for col in cols:
            if col in STRICT_FORBIDDEN_COLUMNS:
                continue
            if col not in rgc_train_df.columns:
                continue
            if col not in cleaned:
                cleaned.append(col)
        final_sets[name] = cleaned

    return final_sets


def audit_candidate_features(candidate_sets: Dict[str, List[str]]) -> pd.DataFrame:
    rows = []

    for candidate_name, cols in candidate_sets.items():
        forbidden_found = [c for c in cols if c in STRICT_FORBIDDEN_COLUMNS]

        rows.append({
            "candidate_feature_set": candidate_name,
            "feature_count": len(cols),
            "forbidden_feature_count": len(forbidden_found),
            "forbidden_features": ", ".join(forbidden_found),
            "leakage_audit_status": "PASS" if len(forbidden_found) == 0 else "FAIL",
        })

    return pd.DataFrame(rows)


def load_baseline_predictions(algorithm: str) -> Dict[str, pd.DataFrame]:
    pred_df = load_csv_flexible(BASELINE_PREDICTIONS_FILE)
    pred_df = pred_df[pred_df["model_name"] == algorithm].copy()

    out = {}

    for split_name in ["train", "valid", "test"]:
        temp = pred_df[pred_df["split"] == split_name].copy().reset_index(drop=True)
        if not temp.empty:
            out[split_name] = temp

    return out


def extract_proba(pred_table: pd.DataFrame) -> np.ndarray:
    proba_cols = [c for c in pred_table.columns if c.startswith("proba_class_")]
    proba_cols = sorted(proba_cols, key=lambda x: int(x.split("_")[-1]))
    return pred_table[proba_cols].to_numpy(dtype=float)


def normalize_proba(proba: np.ndarray) -> np.ndarray:
    proba = np.asarray(proba, dtype=float)
    proba = np.clip(proba, 1e-9, None)
    row_sum = proba.sum(axis=1, keepdims=True)
    return proba / row_sum


def apply_class_weights(proba: np.ndarray, weights: Tuple[float, float, float]) -> np.ndarray:
    weighted = proba * np.asarray(weights).reshape(1, -1)
    return normalize_proba(weighted)


def search_fusion_and_class_weights(
    y_valid: np.ndarray,
    baseline_valid_proba: np.ndarray,
    rgc_valid_proba: np.ndarray,
) -> Dict[str, object]:
    rows = []

    for alpha in ALPHA_GRID:
        blended = (1.0 - alpha) * baseline_valid_proba + alpha * rgc_valid_proba
        blended = normalize_proba(blended)

        for w0 in CLASS_WEIGHT_GRID:
            for w1 in CLASS_WEIGHT_GRID:
                for w2 in CLASS_WEIGHT_GRID:
                    weights = (w0, w1, w2)
                    calibrated = apply_class_weights(blended, weights)
                    y_pred = np.argmax(calibrated, axis=1)

                    rows.append({
                        "alpha": alpha,
                        "w_low": w0,
                        "w_moderate": w1,
                        "w_high": w2,
                        "valid_accuracy": float((y_pred == y_valid).mean()),
                        "valid_f1_macro": float(f1_score(y_valid, y_pred, average="macro", zero_division=0)),
                        "valid_balanced_accuracy": float(balanced_accuracy_score(y_valid, y_pred)),
                    })

    table = pd.DataFrame(rows)
    table = table.sort_values(
        ["valid_f1_macro", "valid_balanced_accuracy", "valid_accuracy"],
        ascending=False,
    ).reset_index(drop=True)

    best = table.iloc[0].to_dict()

    return {
        "alpha": float(best["alpha"]),
        "class_weights": (
            float(best["w_low"]),
            float(best["w_moderate"]),
            float(best["w_high"]),
        ),
        "search_table": table,
    }


def fuse_predictions(
    baseline_proba: np.ndarray,
    rgc_proba: np.ndarray,
    alpha: float,
    class_weights: Tuple[float, float, float],
) -> np.ndarray:
    blended = (1.0 - alpha) * baseline_proba + alpha * rgc_proba
    blended = normalize_proba(blended)
    return apply_class_weights(blended, class_weights)


def build_fused_result(
    proposed_name: str,
    algorithm: str,
    raw_result: Dict[str, object],
    fusion_info: Dict[str, object],
) -> Dict[str, object]:
    baseline_preds = load_baseline_predictions(algorithm)

    alpha = fusion_info["alpha"]
    weights = fusion_info["class_weights"]

    final_result = {
        "model_name": proposed_name,
        "model": raw_result["model"],
        "metrics": {},
        "predictions": {},
        "classification_reports": {},
        "confusion_matrices": {},
        "fusion_alpha": alpha,
        "class_weights": weights,
    }

    for split_name in ["train", "valid", "test"]:
        y_true = raw_result["predictions"][split_name]["y_true"]
        rgc_proba = raw_result["predictions"][split_name]["y_proba"]

        if split_name in baseline_preds and rgc_proba is not None:
            baseline_proba = extract_proba(baseline_preds[split_name])
            y_proba = fuse_predictions(baseline_proba, rgc_proba, alpha, weights)
            y_pred = np.argmax(y_proba, axis=1)
        else:
            y_proba = rgc_proba
            y_pred = raw_result["predictions"][split_name]["y_pred"]

        final_result["metrics"][split_name] = calculate_metrics(y_true, y_pred, y_proba)
        final_result["predictions"][split_name] = {
            "y_true": np.asarray(y_true),
            "y_pred": np.asarray(y_pred),
            "y_proba": y_proba,
        }
        final_result["classification_reports"][split_name] = classification_report(
            y_true,
            y_pred,
            target_names=["Low", "Moderate", "High"],
            zero_division=0,
            output_dict=True,
        )
        final_result["confusion_matrices"][split_name] = confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1, 2],
        )

    return final_result


def candidate_has_leakage_like_score(valid_metrics: Dict[str, float]) -> bool:
    return (
        valid_metrics["accuracy"] > LEAKAGE_GUARD_UPPER_BOUND
        or valid_metrics["f1_macro"] > LEAKAGE_GUARD_UPPER_BOUND
        or valid_metrics["roc_auc_ovr_macro"] > 0.995
    )


def save_confusion_matrix_figure(cm, model_name: str, split_name: str):
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Low", "Moderate", "High"],
    )
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"{model_name} - {split_name} confusion matrix")

    path = FIGURES_DIR / f"07_leakage_safe_{model_name.lower().replace('-', '_')}_{split_name}_confusion_matrix.png"
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"[SAVED] {path}")


def make_baseline_vs_proposed_table(proposed_metrics: pd.DataFrame) -> pd.DataFrame:
    baseline_metrics = load_csv_flexible(BASELINE_METRICS_FILE)

    baseline_test = baseline_metrics[baseline_metrics["split"] == "test"].copy()
    proposed_test = proposed_metrics[proposed_metrics["split"] == "test"].copy()

    baseline_test["algorithm"] = baseline_test["model_name"].astype(str)
    proposed_test["algorithm"] = proposed_test["model_name"].astype(str).str.replace("-RGC", "", regex=False)

    merged = proposed_test.merge(
        baseline_test,
        on="algorithm",
        suffixes=("_proposed", "_baseline"),
    )

    rows = []

    for _, row in merged.iterrows():
        rows.append({
            "algorithm": row["algorithm"],
            "baseline_model": row["model_name_baseline"],
            "proposed_model": row["model_name_proposed"],
            "baseline_accuracy": row["accuracy_baseline"],
            "proposed_accuracy": row["accuracy_proposed"],
            "accuracy_delta": row["accuracy_proposed"] - row["accuracy_baseline"],
            "accuracy_relative_improvement_percent": (
                (row["accuracy_proposed"] - row["accuracy_baseline"])
                / max(row["accuracy_baseline"], 1e-9)
                * 100
            ),
            "baseline_balanced_accuracy": row["balanced_accuracy_baseline"],
            "proposed_balanced_accuracy": row["balanced_accuracy_proposed"],
            "balanced_accuracy_delta": row["balanced_accuracy_proposed"] - row["balanced_accuracy_baseline"],
            "baseline_f1_macro": row["f1_macro_baseline"],
            "proposed_f1_macro": row["f1_macro_proposed"],
            "f1_macro_delta": row["f1_macro_proposed"] - row["f1_macro_baseline"],
            "f1_macro_relative_improvement_percent": (
                (row["f1_macro_proposed"] - row["f1_macro_baseline"])
                / max(row["f1_macro_baseline"], 1e-9)
                * 100
            ),
            "baseline_f1_weighted": row["f1_weighted_baseline"],
            "proposed_f1_weighted": row["f1_weighted_proposed"],
            "f1_weighted_delta": row["f1_weighted_proposed"] - row["f1_weighted_baseline"],
            "baseline_roc_auc_ovr": row["roc_auc_ovr_macro_baseline"],
            "proposed_roc_auc_ovr": row["roc_auc_ovr_macro_proposed"],
            "roc_auc_delta": row["roc_auc_ovr_macro_proposed"] - row["roc_auc_ovr_macro_baseline"],
        })

    return pd.DataFrame(rows).sort_values(
        ["f1_macro_delta", "balanced_accuracy_delta"],
        ascending=False,
    )


def main():
    print_header("STEP 07: LEAKAGE-SAFE ADVANCED RGC MODEL TRAINING")

    print(f"[TIME] {timestamp()}")
    print(f"[PROJECT ROOT] {PROJECT_ROOT}")
    print(f"[INPUT FILE] {INPUT_FILE}")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Step 05 dataset not found: {INPUT_FILE}")

    if not BASELINE_PREDICTIONS_FILE.exists():
        raise FileNotFoundError(f"Step 06 predictions missing: {BASELINE_PREDICTIONS_FILE}")

    df = load_csv_flexible(INPUT_FILE)
    print(f"[INPUT SHAPE] {df.shape[0]:,} rows | {df.shape[1]:,} columns")

    split_data = make_temporal_split(df)

    for split_name, split_df in split_data.items():
        min_date = pd.to_datetime(split_df["order date (DateOrders)"]).min()
        max_date = pd.to_datetime(split_df["order date (DateOrders)"]).max()
        print(
            f"[{split_name.upper()}] rows={split_df.shape[0]:,} | "
            f"date={min_date} to {max_date} | "
            f"labels={split_df[TARGET_COL].value_counts().sort_index().to_dict()}"
        )

    rgc_pack = add_leakage_safe_rgc_features(split_data)
    rgc_splits = rgc_pack["rgc_splits"]

    print_subheader("Saving Leakage-Safe RGC Feature Dataset")
    combined = pd.concat(
        [rgc_splits["train"], rgc_splits["valid"], rgc_splits["test"]],
        ignore_index=True,
    )
    save_csv(combined, OUTPUT_RGC_FEATURE_FILE)

    original_safe_df = drop_unsafe_existing_graph_columns(df)
    candidate_feature_sets = get_candidate_feature_sets(rgc_splits["train"], original_safe_df)

    audit_table = audit_candidate_features(candidate_feature_sets)
    save_csv(audit_table, TABLES_DIR / "07_leakage_safe_candidate_feature_audit.csv")
    print(audit_table.to_string(index=False))

    feature_rows = []
    for candidate_name, cols in candidate_feature_sets.items():
        for col in cols:
            feature_rows.append({
                "candidate_feature_set": candidate_name,
                "feature": col,
                "dtype": str(rgc_splits["train"][col].dtype),
                "feature_group": "rgc_feature" if col.startswith("rgc_") else "baseline_or_proxy_feature",
                "strict_forbidden": col in STRICT_FORBIDDEN_COLUMNS,
            })

    save_csv(pd.DataFrame(feature_rows), TABLES_DIR / "07_leakage_safe_candidate_feature_sets.csv")

    all_results = {}
    all_predictions = []
    all_search_rows = []
    all_candidate_rows = []

    print_subheader("Training Leakage-Safe Candidate RGC Models")

    for proposed_name, base_algorithm in PROPOSED_MODEL_MAP.items():
        print("\n" + "=" * 100)
        print(f"[PROPOSED MODEL] {proposed_name}")
        print("=" * 100)

        baseline_preds = load_baseline_predictions(base_algorithm)
        baseline_valid_proba = extract_proba(baseline_preds["valid"])

        safe_candidates = []
        leakage_risk_candidates = []

        for candidate_name, feature_cols in candidate_feature_sets.items():
            print("\n" + "-" * 100)
            print(f"[TRAINING CANDIDATE] {proposed_name} | {candidate_name} | features={len(feature_cols)}")
            print("-" * 100)

            split_xy = split_x_y(rgc_splits, feature_cols)

            raw_result = train_evaluate_single_model(
                model_name=base_algorithm,
                X_train=split_xy["X_train"],
                y_train=split_xy["y_train"],
                X_valid=split_xy["X_valid"],
                y_valid=split_xy["y_valid"],
                X_test=split_xy["X_test"],
                y_test=split_xy["y_test"],
                random_state=RANDOM_STATE,
            )

            rgc_valid_proba = raw_result["predictions"]["valid"]["y_proba"]
            y_valid = raw_result["predictions"]["valid"]["y_true"]

            fusion_info = search_fusion_and_class_weights(
                y_valid=np.asarray(y_valid),
                baseline_valid_proba=baseline_valid_proba,
                rgc_valid_proba=rgc_valid_proba,
            )

            fused_result = build_fused_result(
                proposed_name=proposed_name,
                algorithm=base_algorithm,
                raw_result=raw_result,
                fusion_info=fusion_info,
            )

            valid_metrics = fused_result["metrics"]["valid"]
            test_metrics = fused_result["metrics"]["test"]

            leakage_like = candidate_has_leakage_like_score(valid_metrics)
            status = "EXCLUDED_LEAKAGE_RISK" if leakage_like else "SAFE_SELECTABLE"

            print(
                f"[VALID] ACC={valid_metrics['accuracy']:.4f} | "
                f"F1_MACRO={valid_metrics['f1_macro']:.4f} | "
                f"BAL_ACC={valid_metrics['balanced_accuracy']:.4f} | "
                f"ROC_AUC={valid_metrics['roc_auc_ovr_macro']:.4f} | "
                f"STATUS={status}"
            )
            print(
                f"[TEST PREVIEW] ACC={test_metrics['accuracy']:.4f} | "
                f"F1_MACRO={test_metrics['f1_macro']:.4f} | "
                f"BAL_ACC={test_metrics['balanced_accuracy']:.4f}"
            )

            search_table = fusion_info["search_table"].copy()
            search_table["proposed_model"] = proposed_name
            search_table["base_algorithm"] = base_algorithm
            search_table["candidate_feature_set"] = candidate_name
            all_search_rows.append(search_table)

            candidate_record = {
                "candidate_name": candidate_name,
                "feature_cols": feature_cols,
                "raw_result": raw_result,
                "fused_result": fused_result,
                "fusion_info": fusion_info,
                "valid_f1_macro": valid_metrics["f1_macro"],
                "valid_balanced_accuracy": valid_metrics["balanced_accuracy"],
                "valid_accuracy": valid_metrics["accuracy"],
                "valid_roc_auc": valid_metrics["roc_auc_ovr_macro"],
                "status": status,
            }

            all_candidate_rows.append({
                "proposed_model": proposed_name,
                "base_algorithm": base_algorithm,
                "candidate_feature_set": candidate_name,
                "feature_count": len(feature_cols),
                "selected_alpha": fusion_info["alpha"],
                "selected_class_weights": str(fusion_info["class_weights"]),
                "valid_accuracy": valid_metrics["accuracy"],
                "valid_f1_macro": valid_metrics["f1_macro"],
                "valid_balanced_accuracy": valid_metrics["balanced_accuracy"],
                "valid_roc_auc": valid_metrics["roc_auc_ovr_macro"],
                "test_accuracy_preview": test_metrics["accuracy"],
                "test_f1_macro_preview": test_metrics["f1_macro"],
                "test_balanced_accuracy_preview": test_metrics["balanced_accuracy"],
                "candidate_status": status,
            })

            if leakage_like:
                leakage_risk_candidates.append(candidate_record)
            else:
                safe_candidates.append(candidate_record)

        if not safe_candidates:
            print("[WARNING] All candidates crossed leakage guard. Selecting lowest-validation-score candidate for diagnostic only.")
            selectable = leakage_risk_candidates
        else:
            selectable = safe_candidates

        selected = sorted(
            selectable,
            key=lambda x: (
                x["valid_f1_macro"],
                x["valid_balanced_accuracy"],
                x["valid_accuracy"],
            ),
            reverse=True,
        )[0]

        final_result = selected["fused_result"]
        all_results[proposed_name] = final_result

        bundle = {
            "proposed_model_name": proposed_name,
            "base_algorithm": base_algorithm,
            "selected_candidate_feature_set": selected["candidate_name"],
            "candidate_status": selected["status"],
            "feature_columns": selected["feature_cols"],
            "model": selected["raw_result"]["model"],
            "fusion_alpha": selected["fusion_info"]["alpha"],
            "class_weights": selected["fusion_info"]["class_weights"],
            "leakage_guard_upper_bound": LEAKAGE_GUARD_UPPER_BOUND,
            "strict_forbidden_columns": sorted(list(STRICT_FORBIDDEN_COLUMNS)),
            "selection_metric": "validation_macro_f1_then_balanced_accuracy_under_leakage_guard",
            "fusion_method": "validation-selected probability fusion and class calibration",
        }

        model_path = MODELS_DIR / f"07_leakage_safe_{proposed_name.lower().replace('-', '_')}_bundle.joblib"
        joblib.dump(bundle, model_path)

        print(f"[SAVED] {model_path}")
        print(f"[SELECTED CANDIDATE] {selected['candidate_name']}")
        print(f"[CANDIDATE STATUS] {selected['status']}")
        print(f"[SELECTED ALPHA] {selected['fusion_info']['alpha']}")
        print(f"[SELECTED CLASS WEIGHTS] {selected['fusion_info']['class_weights']}")

        for split_name in ["train", "valid", "test"]:
            metrics = final_result["metrics"][split_name]
            print(
                f"[{proposed_name} | {split_name.upper()}] "
                f"ACC={metrics['accuracy']:.4f} | "
                f"BAL_ACC={metrics['balanced_accuracy']:.4f} | "
                f"F1_MACRO={metrics['f1_macro']:.4f} | "
                f"F1_WEIGHTED={metrics['f1_weighted']:.4f} | "
                f"ROC_AUC_OVR={metrics['roc_auc_ovr_macro']:.4f}"
            )

            save_confusion_matrix_figure(
                final_result["confusion_matrices"][split_name],
                proposed_name,
                split_name,
            )

            preds = final_result["predictions"][split_name]
            pred_table = make_prediction_table(
                model_name=proposed_name,
                split_name=split_name,
                y_true=preds["y_true"],
                y_pred=preds["y_pred"],
                y_proba=preds["y_proba"],
            )
            pred_table["selected_candidate_feature_set"] = selected["candidate_name"]
            pred_table["candidate_status"] = selected["status"]
            pred_table["fusion_alpha"] = selected["fusion_info"]["alpha"]
            pred_table["class_weights"] = str(selected["fusion_info"]["class_weights"])
            all_predictions.append(pred_table)

        report_path = REPORTS_DIR / f"07_leakage_safe_{proposed_name.lower().replace('-', '_')}_classification_report.json"
        save_json(final_result["classification_reports"], report_path)

    print_subheader("Saving Final Leakage-Safe Step 07 Outputs")

    metrics_table = make_metrics_table(all_results)
    save_csv(metrics_table, TABLES_DIR / "07_leakage_safe_proposed_rgc_model_metrics.csv")

    if all_predictions:
        save_csv(
            pd.concat(all_predictions, ignore_index=True),
            TABLES_DIR / "07_leakage_safe_proposed_rgc_predictions.csv",
        )

    if all_search_rows:
        save_csv(
            pd.concat(all_search_rows, ignore_index=True),
            TABLES_DIR / "07_leakage_safe_fusion_alpha_class_weight_search.csv",
        )

    candidate_summary = pd.DataFrame(all_candidate_rows)
    save_csv(candidate_summary, TABLES_DIR / "07_leakage_safe_candidate_selection_summary.csv")

    best_test = (
        metrics_table[metrics_table["split"] == "test"]
        .sort_values(["f1_macro", "balanced_accuracy"], ascending=False)
        .head(1)
    )
    save_csv(best_test, TABLES_DIR / "07_leakage_safe_best_proposed_rgc_model.csv")

    comparison = make_baseline_vs_proposed_table(metrics_table)
    save_csv(comparison, TABLES_DIR / "07_leakage_safe_baseline_vs_proposed_rgc_improvement.csv")

    summary = {
        "time": timestamp(),
        "input_file": str(INPUT_FILE),
        "output_rgc_feature_file": str(OUTPUT_RGC_FEATURE_FILE),
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "train_safe_graph_nodes": int(rgc_pack["nodes"].shape[0]),
        "train_safe_graph_edges": int(rgc_pack["edges"].shape[0]),
        "leakage_guard_upper_bound": LEAKAGE_GUARD_UPPER_BOUND,
        "strict_forbidden_columns": sorted(list(STRICT_FORBIDDEN_COLUMNS)),
        "candidate_feature_sets": {k: len(v) for k, v in candidate_feature_sets.items()},
        "best_test_model": best_test.to_dict(orient="records"),
        "method": [
            "Direct label-construction columns were removed.",
            "Direct delivery-outcome columns were removed.",
            "Train-only graph context was used.",
            "Time-ordered out-of-fold entity priors were used for training rows.",
            "Validation/test priors were mapped only from training statistics.",
            "Candidates exceeding 95% validation performance were marked as leakage-risk and excluded from selection.",
        ],
    }

    save_json(summary, REPORTS_DIR / "07_leakage_safe_proposed_rgc_training_summary.json")

    report_lines = []
    report_lines.append("STEP 07 LEAKAGE-SAFE ADVANCED RGC TRAINING REPORT")
    report_lines.append("=" * 90)
    report_lines.append(f"Time: {summary['time']}")
    report_lines.append(f"Input shape: {df.shape[0]:,} rows x {df.shape[1]:,} columns")
    report_lines.append(f"Train-safe graph nodes: {summary['train_safe_graph_nodes']:,}")
    report_lines.append(f"Train-safe graph edges: {summary['train_safe_graph_edges']:,}")
    report_lines.append(f"Leakage guard upper bound: {LEAKAGE_GUARD_UPPER_BOUND}")
    report_lines.append("")
    report_lines.append("Strictly forbidden columns:")
    for col in sorted(list(STRICT_FORBIDDEN_COLUMNS)):
        report_lines.append(f"- {col}")
    report_lines.append("")
    report_lines.append("Candidate feature sets:")
    for name, count in summary["candidate_feature_sets"].items():
        report_lines.append(f"- {name}: {count} features")
    report_lines.append("")
    report_lines.append("Candidate selection summary:")
    for _, row in candidate_summary.iterrows():
        report_lines.append(
            f"- {row['proposed_model']} | {row['candidate_feature_set']} | "
            f"status={row['candidate_status']} | "
            f"valid_acc={row['valid_accuracy']:.4f} | "
            f"valid_macro_f1={row['valid_f1_macro']:.4f} | "
            f"test_acc_preview={row['test_accuracy_preview']:.4f} | "
            f"test_macro_f1_preview={row['test_f1_macro_preview']:.4f}"
        )
    report_lines.append("")
    report_lines.append("Test-set final proposed model ranking:")
    test_metrics = metrics_table[metrics_table["split"] == "test"].sort_values(
        ["f1_macro", "balanced_accuracy"],
        ascending=False,
    )
    for _, row in test_metrics.iterrows():
        report_lines.append(
            f"- {row['model_name']}: accuracy={row['accuracy']:.4f}, "
            f"balanced_accuracy={row['balanced_accuracy']:.4f}, "
            f"macro_f1={row['f1_macro']:.4f}, "
            f"weighted_f1={row['f1_weighted']:.4f}, "
            f"roc_auc_ovr={row['roc_auc_ovr_macro']:.4f}"
        )
    report_lines.append("")
    report_lines.append("Baseline vs proposed improvement:")
    for _, row in comparison.iterrows():
        report_lines.append(
            f"- {row['algorithm']}: accuracy_delta={row['accuracy_delta']:.4f}, "
            f"macro_f1_delta={row['f1_macro_delta']:.4f}, "
            f"macro_f1_relative_improvement={row['f1_macro_relative_improvement_percent']:.2f}%, "
            f"roc_auc_delta={row['roc_auc_delta']:.4f}"
        )

    save_text("\n".join(report_lines), REPORTS_DIR / "07_leakage_safe_proposed_rgc_training_report.txt")

    log_text = (
        f"[{timestamp()}] Step 07 leakage-safe completed. "
        f"BestTest={best_test.to_dict(orient='records')}\n"
    )
    save_text(log_text, LOGS_DIR / "07_leakage_safe_proposed_rgc_training.log")

    print_subheader("Final Leakage-Safe Proposed RGC Test Metrics")
    print(
        metrics_table[metrics_table["split"] == "test"]
        .sort_values(["f1_macro", "balanced_accuracy"], ascending=False)
        .to_string(index=False)
    )

    print_subheader("Baseline vs Leakage-Safe Proposed RGC Improvement")
    print(comparison.to_string(index=False))

    print_subheader("Saved Files")
    print(f"[FEATURE DATASET] {OUTPUT_RGC_FEATURE_FILE}")
    print(f"[METRICS] {TABLES_DIR / '07_leakage_safe_proposed_rgc_model_metrics.csv'}")
    print(f"[PREDICTIONS] {TABLES_DIR / '07_leakage_safe_proposed_rgc_predictions.csv'}")
    print(f"[BEST MODEL] {TABLES_DIR / '07_leakage_safe_best_proposed_rgc_model.csv'}")
    print(f"[IMPROVEMENT] {TABLES_DIR / '07_leakage_safe_baseline_vs_proposed_rgc_improvement.csv'}")
    print(f"[REPORT] {REPORTS_DIR / '07_leakage_safe_proposed_rgc_training_report.txt'}")

    print_header("STEP 07 LEAKAGE-SAFE COMPLETED SUCCESSFULLY")

if __name__ == "__main__":
    main()