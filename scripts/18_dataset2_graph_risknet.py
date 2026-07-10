# scripts/18_dataset2_graph_risknet.py
# ======================================================================================
# STEP 18: DATASET 2 GRAPH CONSTRUCTION + GRAPH-RISKNET FEATURES
# Project: ResilientGraph-CMARL
# Dataset 2: Brazilian E-Commerce Public Dataset by Olist
#
# Purpose:
#   1. Load Step 17 Olist order-item-level risk-labelled dataset.
#   2. Construct a heterogeneous e-commerce fulfilment graph:
#        seller, product, category, customer_state, seller_state, route, payment, status.
#   3. Create node and edge tables with graph-risk metadata.
#   4. Build NetworkX graph and save graph artifact.
#   5. Generate leakage-aware graph-context features:
#        - cumulative prior risk statistics
#        - train-graph degree/PageRank structural features
#        - seller/category/route/product risk context features
#   6. Save graph-enhanced feature dataset for Step 19 ML modelling.
# ======================================================================================

from __future__ import annotations

import json
import pickle
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd


# --------------------------------------------------------------------------------------
# Project paths
# --------------------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET2_PROCESSED_DIR = PROJECT_ROOT / "data" / "dataset2_processed"
DATASET2_GRAPH_DIR = PROJECT_ROOT / "data" / "dataset2_graph"

OUTPUTS2_DIR = PROJECT_ROOT / "outputs_dataset2"
TABLES2_DIR = OUTPUTS2_DIR / "tables"
FIGURES2_DIR = OUTPUTS2_DIR / "figures"
MODELS2_DIR = OUTPUTS2_DIR / "models"
LOGS2_DIR = OUTPUTS2_DIR / "logs"
REPORTS2_DIR = OUTPUTS2_DIR / "reports"

LOG_FILE = LOGS2_DIR / "18_dataset2_graph_risknet.log"

INPUT_FILE = DATASET2_PROCESSED_DIR / "17_olist_features_risk_labels.csv"
OUTPUT_FEATURE_FILE = DATASET2_PROCESSED_DIR / "18_olist_graph_risk_features.csv"

NODE_FILE = DATASET2_GRAPH_DIR / "18_dataset2_graph_nodes.csv"
EDGE_FILE = DATASET2_GRAPH_DIR / "18_dataset2_graph_edges.csv"
GRAPH_PKL_FILE = DATASET2_GRAPH_DIR / "18_dataset2_olist_fulfillment_graph.pkl"
TRAIN_GRAPH_PKL_FILE = DATASET2_GRAPH_DIR / "18_dataset2_train_only_structural_graph.pkl"


ID_COLUMNS_FOR_GRAPH = {
    "seller": "seller_id",
    "product": "product_id",
    "category": "product_category_name_english",
    "customer_state": "customer_state",
    "seller_state": "seller_state",
    "route": "seller_customer_route",
    "payment": "payment_type_dominant",
    "status": "order_status",
}


GRAPH_PRIOR_GROUPS = {
    "seller": "seller_id",
    "product": "product_id",
    "category": "product_category_name_english",
    "customer_state": "customer_state",
    "seller_state": "seller_state",
    "route": "seller_customer_route",
    "seller_category": "seller_category_key",
    "customer_category": "customer_category_key",
}


TARGET_COLS_FOR_PRIORS = {
    "risk": "dataset2_fulfillment_risk_score",
    "late": "late_delivery_flag",
    "low_review": "low_review_flag",
    "freight": "freight_cost_component",
    "delay_component": "delivery_delay_component",
    "service_component": "review_service_component",
}


# --------------------------------------------------------------------------------------
# Logging helpers
# --------------------------------------------------------------------------------------

def ensure_directories() -> None:
    for d in [
        DATASET2_PROCESSED_DIR,
        DATASET2_GRAPH_DIR,
        OUTPUTS2_DIR,
        TABLES2_DIR,
        FIGURES2_DIR,
        MODELS2_DIR,
        LOGS2_DIR,
        REPORTS2_DIR,
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
# Utility functions
# --------------------------------------------------------------------------------------

def normalize_entity_value(value: object, unknown_value: str = "unknown") -> str:
    if pd.isna(value):
        return unknown_value

    text = str(value).strip().lower()

    if text == "" or text == "nan" or text == "none":
        return unknown_value

    return text.replace(" ", "_")


def node_id(node_type: str, raw_value: object) -> str:
    return f"{node_type}::{normalize_entity_value(raw_value, f'unknown_{node_type}')}"


def load_input_dataset() -> pd.DataFrame:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Step 17 output not found: {INPUT_FILE}. "
            f"Run Step 17 first."
        )

    df = pd.read_csv(INPUT_FILE)

    if "order_purchase_timestamp" in df.columns:
        df["order_purchase_timestamp"] = pd.to_datetime(
            df["order_purchase_timestamp"],
            errors="coerce",
        )

    df = df.sort_values(
        ["order_purchase_timestamp", "order_id", "order_item_id"],
        kind="mergesort",
    ).reset_index(drop=True)

    return df


def safe_numeric(series: pd.Series, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default)


def build_node_table(df: pd.DataFrame) -> pd.DataFrame:
    node_frames: List[pd.DataFrame] = []

    for node_type, col in ID_COLUMNS_FOR_GRAPH.items():
        if col not in df.columns:
            continue

        tmp = df[[col]].copy()
        tmp[col] = tmp[col].map(lambda x: normalize_entity_value(x, f"unknown_{node_type}"))
        tmp = tmp.drop_duplicates()

        tmp["node_id"] = tmp[col].map(lambda x: node_id(node_type, x))
        tmp["node_type"] = node_type
        tmp["node_value"] = tmp[col]

        node_frames.append(tmp[["node_id", "node_type", "node_value"]])

    nodes = pd.concat(node_frames, axis=0, ignore_index=True).drop_duplicates("node_id")

    # Descriptive node-level risk statistics from all rows for reporting only.
    stats_frames: List[pd.DataFrame] = []

    for node_type, col in ID_COLUMNS_FOR_GRAPH.items():
        if col not in df.columns:
            continue

        work = df.copy()
        work["_node_id"] = work[col].map(lambda x: node_id(node_type, x))

        stats = (
            work.groupby("_node_id", as_index=False)
            .agg(
                node_record_count=("order_id", "count"),
                node_unique_orders=("order_id", "nunique"),
                node_mean_risk=("dataset2_fulfillment_risk_score", "mean"),
                node_late_rate=("late_delivery_flag", "mean"),
                node_low_review_rate=("low_review_flag", "mean"),
                node_mean_freight=("freight_value", "mean"),
                node_mean_delay_days=("delivery_delay_days", "mean"),
            )
            .rename(columns={"_node_id": "node_id"})
        )

        stats_frames.append(stats)

    node_stats = pd.concat(stats_frames, axis=0, ignore_index=True)

    nodes = nodes.merge(node_stats, on="node_id", how="left")

    numeric_cols = [
        "node_record_count",
        "node_unique_orders",
        "node_mean_risk",
        "node_late_rate",
        "node_low_review_rate",
        "node_mean_freight",
        "node_mean_delay_days",
    ]

    for col in numeric_cols:
        if col in nodes.columns:
            nodes[col] = safe_numeric(nodes[col], 0.0)

    return nodes


def build_edge_table(df: pd.DataFrame) -> pd.DataFrame:
    edge_specs = [
        ("seller_to_product", "seller", "seller_id", "product", "product_id"),
        ("product_to_category", "product", "product_id", "category", "product_category_name_english"),
        ("seller_to_seller_state", "seller", "seller_id", "seller_state", "seller_state"),
        ("seller_to_customer_state", "seller", "seller_id", "customer_state", "customer_state"),
        ("seller_to_route", "seller", "seller_id", "route", "seller_customer_route"),
        ("route_to_seller_state", "route", "seller_customer_route", "seller_state", "seller_state"),
        ("route_to_customer_state", "route", "seller_customer_route", "customer_state", "customer_state"),
        ("category_to_customer_state", "category", "product_category_name_english", "customer_state", "customer_state"),
        ("category_to_payment", "category", "product_category_name_english", "payment", "payment_type_dominant"),
        ("status_to_customer_state", "status", "order_status", "customer_state", "customer_state"),
        ("seller_to_category", "seller", "seller_id", "category", "product_category_name_english"),
        ("product_to_customer_state", "product", "product_id", "customer_state", "customer_state"),
    ]

    edge_frames: List[pd.DataFrame] = []

    for relation, src_type, src_col, dst_type, dst_col in edge_specs:
        if src_col not in df.columns or dst_col not in df.columns:
            continue

        work = df[
            [
                src_col,
                dst_col,
                "order_id",
                "dataset2_fulfillment_risk_score",
                "late_delivery_flag",
                "low_review_flag",
                "freight_value",
                "delivery_delay_days",
            ]
        ].copy()

        work["source"] = work[src_col].map(lambda x: node_id(src_type, x))
        work["target"] = work[dst_col].map(lambda x: node_id(dst_type, x))
        work["relation"] = relation

        edge_agg = (
            work.groupby(["source", "target", "relation"], as_index=False)
            .agg(
                edge_weight=("order_id", "count"),
                unique_orders=("order_id", "nunique"),
                edge_mean_risk=("dataset2_fulfillment_risk_score", "mean"),
                edge_late_rate=("late_delivery_flag", "mean"),
                edge_low_review_rate=("low_review_flag", "mean"),
                edge_mean_freight=("freight_value", "mean"),
                edge_mean_delay_days=("delivery_delay_days", "mean"),
            )
        )

        edge_frames.append(edge_agg)

    edges = pd.concat(edge_frames, axis=0, ignore_index=True)

    return edges


def build_networkx_graph(nodes: pd.DataFrame, edges: pd.DataFrame) -> nx.Graph:
    graph = nx.Graph()

    for row in nodes.itertuples(index=False):
        graph.add_node(
            row.node_id,
            node_type=row.node_type,
            node_value=row.node_value,
            node_record_count=float(row.node_record_count),
            node_mean_risk=float(row.node_mean_risk),
            node_late_rate=float(row.node_late_rate),
        )

    for row in edges.itertuples(index=False):
        graph.add_edge(
            row.source,
            row.target,
            relation=row.relation,
            weight=float(row.edge_weight),
            mean_risk=float(row.edge_mean_risk),
            late_rate=float(row.edge_late_rate),
        )

    return graph


def save_graph_pickle(graph: nx.Graph, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("wb") as f:
        pickle.dump(graph, f)

    log(f"[SAVED] {path}")


def build_train_structural_graph(df_train: pd.DataFrame) -> nx.Graph:
    train_nodes = build_node_table(df_train)
    train_edges = build_edge_table(df_train)
    graph = build_networkx_graph(train_nodes, train_edges)
    return graph


def compute_graph_centrality_features(graph: nx.Graph) -> pd.DataFrame:
    print_section("Computing Train-Graph Structural Centrality Features")

    if graph.number_of_nodes() == 0:
        return pd.DataFrame(columns=["node_id", "rgc_degree", "rgc_weighted_degree", "rgc_pagerank"])

    degree_dict = dict(graph.degree())

    weighted_degree_dict = dict(graph.degree(weight="weight"))

    try:
        pagerank_dict = nx.pagerank(
            graph,
            weight="weight",
            alpha=0.85,
            max_iter=100,
            tol=1e-06,
        )
        log("[OK] PageRank computed successfully.")
    except Exception as exc:
        log(f"[WARNING] PageRank failed: {exc}")
        pagerank_dict = {node: 0.0 for node in graph.nodes()}

    rows = []

    for n in graph.nodes():
        rows.append(
            {
                "node_id": n,
                "rgc_degree": float(degree_dict.get(n, 0.0)),
                "rgc_weighted_degree": float(weighted_degree_dict.get(n, 0.0)),
                "rgc_pagerank": float(pagerank_dict.get(n, 0.0)),
            }
        )

    centrality_df = pd.DataFrame(rows)

    return centrality_df


def add_cumulative_prior_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    print_section("Creating Leakage-Aware Cumulative Graph-Risk Prior Features")

    df = df.copy()
    prior_summary_rows = []

    for group_name, group_col in GRAPH_PRIOR_GROUPS.items():
        if group_col not in df.columns:
            continue

        group_key = df[group_col].astype(str).fillna(f"unknown_{group_name}")

        for target_alias, target_col in TARGET_COLS_FOR_PRIORS.items():
            if target_col not in df.columns:
                continue

            target_values = safe_numeric(df[target_col], 0.0).astype(float)

            cumulative_sum_before = target_values.groupby(group_key).cumsum() - target_values
            cumulative_count_before = df.groupby(group_key).cumcount()

            global_prior = target_values.expanding().mean().shift(1).fillna(target_values.mean())

            prior_col = f"rgc_{group_name}_prior_{target_alias}_mean"
            count_col = f"rgc_{group_name}_prior_count"

            df[prior_col] = (
                cumulative_sum_before / cumulative_count_before.replace(0, np.nan)
            ).fillna(global_prior)

            df[prior_col] = safe_numeric(df[prior_col], float(target_values.mean()))

            if count_col not in df.columns:
                df[count_col] = cumulative_count_before.astype(float)

            prior_summary_rows.append(
                {
                    "group_name": group_name,
                    "group_column": group_col,
                    "target_alias": target_alias,
                    "target_column": target_col,
                    "created_prior_column": prior_col,
                    "created_count_column": count_col,
                    "mean_prior": float(df[prior_col].mean()),
                    "std_prior": float(df[prior_col].std()),
                    "min_prior": float(df[prior_col].min()),
                    "max_prior": float(df[prior_col].max()),
                }
            )

    prior_summary = pd.DataFrame(prior_summary_rows)

    return df, prior_summary


def add_train_graph_structural_features(
    df: pd.DataFrame,
    centrality_df: pd.DataFrame,
) -> pd.DataFrame:
    print_section("Mapping Train-Graph Structural Features to Order-Item Rows")

    df = df.copy()

    centrality_lookup = centrality_df.set_index("node_id").to_dict(orient="index")

    structural_sources = {
        "seller": "seller_id",
        "product": "product_id",
        "category": "product_category_name_english",
        "customer_state": "customer_state",
        "seller_state": "seller_state",
        "route": "seller_customer_route",
        "payment": "payment_type_dominant",
        "status": "order_status",
    }

    created_degree_cols = []
    created_weighted_degree_cols = []
    created_pagerank_cols = []

    for entity_name, col in structural_sources.items():
        if col not in df.columns:
            continue

        entity_node_col = f"rgc_{entity_name}_node_id"
        degree_col = f"rgc_{entity_name}_degree"
        weighted_degree_col = f"rgc_{entity_name}_weighted_degree"
        pagerank_col = f"rgc_{entity_name}_pagerank"

        df[entity_node_col] = df[col].map(lambda x: node_id(entity_name, x))

        df[degree_col] = df[entity_node_col].map(
            lambda x: centrality_lookup.get(x, {}).get("rgc_degree", 0.0)
        )

        df[weighted_degree_col] = df[entity_node_col].map(
            lambda x: centrality_lookup.get(x, {}).get("rgc_weighted_degree", 0.0)
        )

        df[pagerank_col] = df[entity_node_col].map(
            lambda x: centrality_lookup.get(x, {}).get("rgc_pagerank", 0.0)
        )

        created_degree_cols.append(degree_col)
        created_weighted_degree_cols.append(weighted_degree_col)
        created_pagerank_cols.append(pagerank_col)

    for col in created_degree_cols + created_weighted_degree_cols + created_pagerank_cols:
        df[col] = safe_numeric(df[col], 0.0)

    if created_degree_cols:
        df["rgc_graph_degree_sum"] = df[created_degree_cols].sum(axis=1)
        df["rgc_graph_degree_mean"] = df[created_degree_cols].mean(axis=1)
        df["rgc_graph_degree_max"] = df[created_degree_cols].max(axis=1)
    else:
        df["rgc_graph_degree_sum"] = 0.0
        df["rgc_graph_degree_mean"] = 0.0
        df["rgc_graph_degree_max"] = 0.0

    if created_weighted_degree_cols:
        df["rgc_graph_weighted_degree_sum"] = df[created_weighted_degree_cols].sum(axis=1)
        df["rgc_graph_weighted_degree_mean"] = df[created_weighted_degree_cols].mean(axis=1)
        df["rgc_graph_weighted_degree_max"] = df[created_weighted_degree_cols].max(axis=1)
    else:
        df["rgc_graph_weighted_degree_sum"] = 0.0
        df["rgc_graph_weighted_degree_mean"] = 0.0
        df["rgc_graph_weighted_degree_max"] = 0.0

    if created_pagerank_cols:
        df["rgc_graph_pagerank_sum"] = df[created_pagerank_cols].sum(axis=1)
        df["rgc_graph_pagerank_mean"] = df[created_pagerank_cols].mean(axis=1)
        df["rgc_graph_pagerank_max"] = df[created_pagerank_cols].max(axis=1)
    else:
        df["rgc_graph_pagerank_sum"] = 0.0
        df["rgc_graph_pagerank_mean"] = 0.0
        df["rgc_graph_pagerank_max"] = 0.0

    return df


def add_graph_context_scores(df: pd.DataFrame) -> pd.DataFrame:
    print_section("Creating Graph-RiskNet Context Scores")

    df = df.copy()

    required_defaults = {
        "rgc_seller_prior_risk_mean": 0.0,
        "rgc_product_prior_risk_mean": 0.0,
        "rgc_category_prior_risk_mean": 0.0,
        "rgc_customer_state_prior_risk_mean": 0.0,
        "rgc_seller_state_prior_risk_mean": 0.0,
        "rgc_route_prior_risk_mean": 0.0,
        "rgc_seller_category_prior_risk_mean": 0.0,
        "rgc_customer_category_prior_risk_mean": 0.0,
        "rgc_seller_prior_late_mean": 0.0,
        "rgc_route_prior_late_mean": 0.0,
        "rgc_category_prior_low_review_mean": 0.0,
        "rgc_seller_prior_low_review_mean": 0.0,
        "rgc_seller_prior_freight_mean": 0.0,
        "rgc_route_prior_freight_mean": 0.0,
        "rgc_category_prior_delay_component_mean": 0.0,
        "rgc_route_prior_delay_component_mean": 0.0,
        "rgc_graph_degree_mean": 0.0,
        "rgc_graph_weighted_degree_mean": 0.0,
        "rgc_graph_pagerank_mean": 0.0,
    }

    for col, default in required_defaults.items():
        if col not in df.columns:
            df[col] = default
        df[col] = safe_numeric(df[col], default)

    df["rgc_entity_risk_context"] = (
        0.25 * df["rgc_seller_prior_risk_mean"]
        + 0.17 * df["rgc_product_prior_risk_mean"]
        + 0.15 * df["rgc_category_prior_risk_mean"]
        + 0.15 * df["rgc_route_prior_risk_mean"]
        + 0.10 * df["rgc_customer_state_prior_risk_mean"]
        + 0.08 * df["rgc_seller_state_prior_risk_mean"]
        + 0.06 * df["rgc_seller_category_prior_risk_mean"]
        + 0.04 * df["rgc_customer_category_prior_risk_mean"]
    ).clip(0, 1)

    df["rgc_delay_service_context"] = (
        0.30 * df["rgc_seller_prior_late_mean"]
        + 0.25 * df["rgc_route_prior_late_mean"]
        + 0.20 * df["rgc_route_prior_delay_component_mean"]
        + 0.15 * df["rgc_seller_prior_low_review_mean"]
        + 0.10 * df["rgc_category_prior_low_review_mean"]
    ).clip(0, 1)

    df["rgc_freight_route_context"] = (
        0.45 * df["rgc_seller_prior_freight_mean"]
        + 0.35 * df["rgc_route_prior_freight_mean"]
        + 0.20 * safe_numeric(df.get("freight_cost_component", pd.Series(0, index=df.index)), 0.0)
    ).clip(0, 1)

    weighted_degree_scaled = df["rgc_graph_weighted_degree_mean"].rank(pct=True).fillna(0)
    degree_scaled = df["rgc_graph_degree_mean"].rank(pct=True).fillna(0)
    pagerank_scaled = df["rgc_graph_pagerank_mean"].rank(pct=True).fillna(0)

    df["rgc_structural_exposure_context"] = (
        0.45 * weighted_degree_scaled
        + 0.35 * degree_scaled
        + 0.20 * pagerank_scaled
    ).clip(0, 1)

    df["rgc_graph_context_signal"] = (
        0.38 * df["rgc_entity_risk_context"]
        + 0.28 * df["rgc_delay_service_context"]
        + 0.18 * df["rgc_freight_route_context"]
        + 0.16 * df["rgc_structural_exposure_context"]
    ).clip(0, 1)

    df["rgc_graph_resilience_pressure"] = (
        0.35 * safe_numeric(df.get("delivery_delay_component", pd.Series(0, index=df.index)), 0.0)
        + 0.25 * safe_numeric(df.get("review_service_component", pd.Series(0, index=df.index)), 0.0)
        + 0.20 * df["rgc_entity_risk_context"]
        + 0.20 * df["rgc_structural_exposure_context"]
    ).clip(0, 1)

    df["rgc_graph_cost_pressure"] = (
        0.50 * safe_numeric(df.get("freight_cost_component", pd.Series(0, index=df.index)), 0.0)
        + 0.30 * df["rgc_freight_route_context"]
        + 0.20 * safe_numeric(df.get("payment_complexity_component", pd.Series(0, index=df.index)), 0.0)
    ).clip(0, 1)

    return df


def build_feature_summary(df: pd.DataFrame) -> pd.DataFrame:
    rgc_cols = [c for c in df.columns if c.startswith("rgc_")]

    rows = []

    for col in rgc_cols:
        series = pd.to_numeric(df[col], errors="coerce")

        if series.notna().sum() == 0:
            rows.append(
                {
                    "feature": col,
                    "dtype": str(df[col].dtype),
                    "missing": int(df[col].isna().sum()),
                    "mean": None,
                    "std": None,
                    "min": None,
                    "median": None,
                    "max": None,
                }
            )
        else:
            rows.append(
                {
                    "feature": col,
                    "dtype": str(df[col].dtype),
                    "missing": int(df[col].isna().sum()),
                    "mean": float(series.mean()),
                    "std": float(series.std()),
                    "min": float(series.min()),
                    "median": float(series.median()),
                    "max": float(series.max()),
                }
            )

    return pd.DataFrame(rows)


def build_node_type_summary(nodes: pd.DataFrame) -> pd.DataFrame:
    return (
        nodes.groupby("node_type", as_index=False)
        .agg(
            node_count=("node_id", "count"),
            total_records=("node_record_count", "sum"),
            mean_risk=("node_mean_risk", "mean"),
            mean_late_rate=("node_late_rate", "mean"),
            mean_low_review_rate=("node_low_review_rate", "mean"),
        )
        .sort_values("node_count", ascending=False)
    )


def build_edge_relation_summary(edges: pd.DataFrame) -> pd.DataFrame:
    return (
        edges.groupby("relation", as_index=False)
        .agg(
            edge_count=("source", "count"),
            total_weight=("edge_weight", "sum"),
            mean_edge_weight=("edge_weight", "mean"),
            mean_risk=("edge_mean_risk", "mean"),
            mean_late_rate=("edge_late_rate", "mean"),
            mean_low_review_rate=("edge_low_review_rate", "mean"),
        )
        .sort_values("edge_count", ascending=False)
    )


def build_top_entity_reports(nodes: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    reports = {}

    for node_type in ["seller", "product", "category", "customer_state", "seller_state", "route"]:
        sub = nodes[nodes["node_type"] == node_type].copy()

        if sub.empty:
            reports[node_type] = pd.DataFrame()
            continue

        sub = sub.sort_values(
            ["node_mean_risk", "node_record_count"],
            ascending=[False, False],
        ).head(50)

        reports[node_type] = sub

    return reports


def remove_remaining_missing(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []

    for col in df.columns:
        missing_before = int(df[col].isna().sum())

        if missing_before == 0:
            rows.append(
                {
                    "column_name": col,
                    "missing_before": 0,
                    "fill_strategy": "not_required",
                    "missing_after": 0,
                }
            )
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(0)
            strategy = "numeric_zero"
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].fillna(pd.Timestamp("1900-01-01"))
            strategy = "datetime_sentinel"
        else:
            df[col] = df[col].fillna("unknown")
            strategy = "categorical_unknown"

        rows.append(
            {
                "column_name": col,
                "missing_before": missing_before,
                "fill_strategy": strategy,
                "missing_after": int(df[col].isna().sum()),
            }
        )

    return df, pd.DataFrame(rows)


def build_graph_feature_dataset(df: pd.DataFrame, train_graph: nx.Graph) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.copy()

    before_cols = set(df.columns)

    df, prior_summary = add_cumulative_prior_features(df)

    centrality_df = compute_graph_centrality_features(train_graph)
    save_csv(centrality_df, TABLES2_DIR / "18_dataset2_train_graph_node_centrality.csv")

    df = add_train_graph_structural_features(df, centrality_df)

    df = add_graph_context_scores(df)

    df, fill_report = remove_remaining_missing(df)

    new_cols = [c for c in df.columns if c not in before_cols]
    new_feature_report = pd.DataFrame(
        {
            "new_graph_feature": new_cols,
            "dtype": [str(df[c].dtype) for c in new_cols],
            "missing_count": [int(df[c].isna().sum()) for c in new_cols],
            "unique_values": [int(df[c].nunique(dropna=True)) for c in new_cols],
        }
    )

    return df, prior_summary, pd.concat([new_feature_report, fill_report], axis=0, ignore_index=True)


def build_graph_audit_summary(
    df: pd.DataFrame,
    graph_df: pd.DataFrame,
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    graph: nx.Graph,
    train_graph: nx.Graph,
) -> Dict[str, object]:
    rgc_cols = [c for c in graph_df.columns if c.startswith("rgc_")]

    summary = {
        "step": "18_dataset2_graph_risknet",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input_file": str(INPUT_FILE),
        "output_feature_file": str(OUTPUT_FEATURE_FILE),
        "input_rows": int(df.shape[0]),
        "input_columns": int(df.shape[1]),
        "output_rows": int(graph_df.shape[0]),
        "output_columns": int(graph_df.shape[1]),
        "new_rgc_feature_count": int(len(rgc_cols)),
        "full_graph_nodes": int(graph.number_of_nodes()),
        "full_graph_edges": int(graph.number_of_edges()),
        "train_graph_nodes": int(train_graph.number_of_nodes()),
        "train_graph_edges": int(train_graph.number_of_edges()),
        "node_table_rows": int(len(nodes)),
        "edge_table_rows": int(len(edges)),
        "final_missing_values": int(graph_df.isna().sum().sum()),
        "temporal_split_counts": graph_df["temporal_split"].value_counts().to_dict()
        if "temporal_split" in graph_df.columns
        else {},
    }

    return summary


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

def main() -> None:
    ensure_directories()
    reset_log()

    print_header("STEP 18: DATASET 2 GRAPH CONSTRUCTION + GRAPH-RISKNET FEATURES")
    log(f"[TIME] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"[PROJECT ROOT] {PROJECT_ROOT}")
    log(f"[INPUT FILE] {INPUT_FILE}")
    log(f"[GRAPH DIR] {DATASET2_GRAPH_DIR}")
    log(f"[OUTPUT DIR] {OUTPUTS2_DIR}")

    try:
        print_section("Loading Step 17 Dataset")
        df = load_input_dataset()
        log(f"[INPUT SHAPE] {df.shape[0]:,} rows | {df.shape[1]:,} columns")

        required_cols = [
            "seller_id",
            "product_id",
            "product_category_name_english",
            "customer_state",
            "seller_state",
            "seller_customer_route",
            "payment_type_dominant",
            "order_status",
            "dataset2_fulfillment_risk_score",
            "late_delivery_flag",
            "low_review_flag",
            "temporal_split",
        ]

        missing_required = [c for c in required_cols if c not in df.columns]

        if missing_required:
            raise ValueError(f"Missing required columns from Step 17 output: {missing_required}")

        print_section("Building Heterogeneous Fulfilment Graph Tables")
        nodes = build_node_table(df)
        edges = build_edge_table(df)

        save_csv(nodes, NODE_FILE)
        save_csv(edges, EDGE_FILE)

        node_type_summary = build_node_type_summary(nodes)
        edge_relation_summary = build_edge_relation_summary(edges)

        save_csv(node_type_summary, TABLES2_DIR / "18_dataset2_graph_node_type_summary.csv")
        save_csv(edge_relation_summary, TABLES2_DIR / "18_dataset2_graph_edge_relation_summary.csv")

        print_section("Creating Full Dataset 2 NetworkX Graph")
        graph = build_networkx_graph(nodes, edges)

        log(f"[GRAPH NODES] {graph.number_of_nodes():,}")
        log(f"[GRAPH EDGES] {graph.number_of_edges():,}")

        save_graph_pickle(graph, GRAPH_PKL_FILE)

        print_section("Building Train-Only Structural Graph for Leakage-Safe Features")
        train_df = df[df["temporal_split"] == "train"].copy()

        if train_df.empty:
            raise ValueError("No train rows found in temporal_split.")

        train_graph = build_train_structural_graph(train_df)

        log(f"[TRAIN GRAPH NODES] {train_graph.number_of_nodes():,}")
        log(f"[TRAIN GRAPH EDGES] {train_graph.number_of_edges():,}")

        save_graph_pickle(train_graph, TRAIN_GRAPH_PKL_FILE)

        print_section("Generating Graph-RiskNet Feature Dataset")
        graph_df, prior_summary, graph_feature_report = build_graph_feature_dataset(df, train_graph)

        save_csv(graph_df, OUTPUT_FEATURE_FILE)
        save_csv(prior_summary, TABLES2_DIR / "18_dataset2_cumulative_prior_feature_summary.csv")
        save_csv(graph_feature_report, TABLES2_DIR / "18_dataset2_new_graph_feature_report.csv")

        graph_feature_summary = build_feature_summary(graph_df)
        save_csv(graph_feature_summary, TABLES2_DIR / "18_dataset2_graph_feature_summary.csv")

        print_section("Saving Top-Risk Entity Reports")
        top_entity_reports = build_top_entity_reports(nodes)

        for entity_name, report_df in top_entity_reports.items():
            output_path = TABLES2_DIR / f"18_dataset2_top_risk_{entity_name}_nodes.csv"
            save_csv(report_df, output_path)

        split_feature_summary = (
            graph_df.groupby("temporal_split", as_index=False)
            .agg(
                rows=("order_id", "count"),
                mean_risk_score=("dataset2_fulfillment_risk_score", "mean"),
                mean_graph_context_signal=("rgc_graph_context_signal", "mean"),
                mean_entity_risk_context=("rgc_entity_risk_context", "mean"),
                mean_delay_service_context=("rgc_delay_service_context", "mean"),
                mean_structural_exposure_context=("rgc_structural_exposure_context", "mean"),
                mean_graph_resilience_pressure=("rgc_graph_resilience_pressure", "mean"),
                mean_graph_cost_pressure=("rgc_graph_cost_pressure", "mean"),
            )
        )

        save_csv(split_feature_summary, TABLES2_DIR / "18_dataset2_graph_feature_split_summary.csv")

        print_section("Dataset 2 Graph Node Type Summary")
        log(node_type_summary.to_string(index=False))

        print_section("Dataset 2 Graph Edge Relation Summary")
        log(edge_relation_summary.to_string(index=False))

        print_section("Dataset 2 Graph Feature Split Summary")
        log(split_feature_summary.to_string(index=False))

        final_missing = int(graph_df.isna().sum().sum())

        print_section("Final Graph Feature Dataset")
        log(f"[OUTPUT SHAPE] {graph_df.shape[0]:,} rows | {graph_df.shape[1]:,} columns")
        log(f"[NEW RGC FEATURE COUNT] {len([c for c in graph_df.columns if c.startswith('rgc_')]):,}")
        log(f"[FINAL MISSING VALUES] {final_missing:,}")

        audit_summary = build_graph_audit_summary(
            df=df,
            graph_df=graph_df,
            nodes=nodes,
            edges=edges,
            graph=graph,
            train_graph=train_graph,
        )

        save_json(audit_summary, REPORTS2_DIR / "18_dataset2_graph_risknet_summary.json")

        report_lines: List[str] = []
        report_lines.append("STEP 18: DATASET 2 GRAPH CONSTRUCTION + GRAPH-RISKNET REPORT")
        report_lines.append("=" * 100)
        report_lines.append(f"Timestamp: {audit_summary['timestamp']}")
        report_lines.append(f"Input file: {INPUT_FILE}")
        report_lines.append(f"Output feature file: {OUTPUT_FEATURE_FILE}")
        report_lines.append("")
        report_lines.append("Main Graph Summary")
        report_lines.append("-" * 100)
        report_lines.append(f"Input shape: {df.shape[0]:,} rows x {df.shape[1]:,} columns")
        report_lines.append(f"Output shape: {graph_df.shape[0]:,} rows x {graph_df.shape[1]:,} columns")
        report_lines.append(f"New RGC feature count: {audit_summary['new_rgc_feature_count']}")
        report_lines.append(f"Full graph nodes: {graph.number_of_nodes():,}")
        report_lines.append(f"Full graph edges: {graph.number_of_edges():,}")
        report_lines.append(f"Train structural graph nodes: {train_graph.number_of_nodes():,}")
        report_lines.append(f"Train structural graph edges: {train_graph.number_of_edges():,}")
        report_lines.append(f"Final missing values: {final_missing:,}")
        report_lines.append("")
        report_lines.append("Node Type Summary")
        report_lines.append("-" * 100)
        report_lines.append(node_type_summary.to_string(index=False))
        report_lines.append("")
        report_lines.append("Edge Relation Summary")
        report_lines.append("-" * 100)
        report_lines.append(edge_relation_summary.to_string(index=False))
        report_lines.append("")
        report_lines.append("Graph Feature Split Summary")
        report_lines.append("-" * 100)
        report_lines.append(split_feature_summary.to_string(index=False))
        report_lines.append("")
        report_lines.append("Key Created Graph Context Features")
        report_lines.append("-" * 100)
        key_cols = [
            "rgc_entity_risk_context",
            "rgc_delay_service_context",
            "rgc_freight_route_context",
            "rgc_structural_exposure_context",
            "rgc_graph_context_signal",
            "rgc_graph_resilience_pressure",
            "rgc_graph_cost_pressure",
        ]

        for col in key_cols:
            if col in graph_df.columns:
                report_lines.append(
                    f"{col}: mean={graph_df[col].mean():.6f}, "
                    f"std={graph_df[col].std():.6f}, "
                    f"min={graph_df[col].min():.6f}, "
                    f"max={graph_df[col].max():.6f}"
                )

        save_text(
            "\n".join(report_lines),
            REPORTS2_DIR / "18_dataset2_graph_risknet_report.txt",
        )

        print_section("Step 18 Completed")
        log("[DONE] Dataset 2 graph construction and Graph-RiskNet feature generation completed successfully.")
        log(f"[GRAPH FEATURE DATASET SAVED] {OUTPUT_FEATURE_FILE}")
        log(f"[NODE TABLE SAVED] {NODE_FILE}")
        log(f"[EDGE TABLE SAVED] {EDGE_FILE}")
        log(f"[GRAPH PICKLE SAVED] {GRAPH_PKL_FILE}")
        log(f"[TRAIN GRAPH PICKLE SAVED] {TRAIN_GRAPH_PKL_FILE}")
        log(f"[TABLES SAVED] {TABLES2_DIR}")
        log(f"[REPORTS SAVED] {REPORTS2_DIR}")
        log(f"[LOG SAVED] {LOG_FILE}")
        log("")
        log("NEXT STEP:")
        log("py -3.10 -u .\\scripts\\19_dataset2_ml_baseline_vs_rgc.py")

    except Exception as exc:
        print_section("Step 18 Failed")
        log(f"[ERROR] {exc}")
        log(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()