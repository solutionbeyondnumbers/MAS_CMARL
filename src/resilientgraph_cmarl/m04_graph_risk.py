from pathlib import Path
from typing import Dict, List, Tuple
import pickle

import numpy as np
import pandas as pd
import networkx as nx


ENTITY_COLUMN_MAP = {
    "product": "Product Name",
    "category": "Category Name",
    "region": "Order Region",
    "market": "Market",
    "shipping": "Shipping Mode",
    "buyer_agent": "buyer_agent_key",
    "supplier_proxy": "supplier_proxy_key",
}


RISK_SCORE_COL = "composite_disruption_risk_score"
RISK_LABEL_COL = "risk_label"
SALES_COL = "Sales"
PROFIT_COL = "Order Profit Per Order"
DELAY_COL = "shipping_delay_gap"
ORDER_ID_COL = "Order Id"


def clean_entity_value(value) -> str:
    if pd.isna(value):
        return "Unknown"
    value = str(value).strip()
    if value == "" or value.lower() in ["nan", "none", "null"]:
        return "Unknown"
    return value.replace("::", "_").replace("|", "_")


def make_node_id(node_type: str, value) -> str:
    return f"{node_type}::{clean_entity_value(value)}"


def ensure_required_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    fallback_numeric = {
        RISK_SCORE_COL: 0.0,
        RISK_LABEL_COL: 0,
        SALES_COL: 0.0,
        PROFIT_COL: 0.0,
        DELAY_COL: 0.0,
    }

    for col, default in fallback_numeric.items():
        if col not in df.columns:
            df[col] = default
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default)

    if ORDER_ID_COL not in df.columns:
        df[ORDER_ID_COL] = np.arange(len(df))

    return df


def prepare_graph_entity_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = ensure_required_numeric_columns(df)

    required_text_cols = [
        "Product Name",
        "Category Name",
        "Order Region",
        "Market",
        "Shipping Mode",
        "Customer Segment",
    ]

    for col in required_text_cols:
        if col not in df.columns:
            df[col] = "Unknown"
        df[col] = df[col].apply(clean_entity_value)

    df["buyer_agent_key"] = (
        df["Customer Segment"].astype(str)
        + "_"
        + df["Order Region"].astype(str)
    ).apply(clean_entity_value)

    # DataCo does not include true supplier IDs. Supplier proxy is derived from
    # category + market + shipping mode, which approximates procurement-source behaviour.
    df["supplier_proxy_key"] = (
        df["Category Name"].astype(str)
        + "_"
        + df["Market"].astype(str)
        + "_"
        + df["Shipping Mode"].astype(str)
    ).apply(clean_entity_value)

    for node_type, col in ENTITY_COLUMN_MAP.items():
        df[f"node_{node_type}"] = df[col].apply(lambda x: make_node_id(node_type, x))

    return df


def aggregate_edges(
    df: pd.DataFrame,
    source_col: str,
    target_col: str,
    relation: str,
) -> pd.DataFrame:
    edge_df = (
        df.groupby([source_col, target_col], observed=False)
        .agg(
            order_count=(ORDER_ID_COL, "count"),
            avg_risk_score=(RISK_SCORE_COL, "mean"),
            avg_risk_label=(RISK_LABEL_COL, "mean"),
            avg_delay_gap=(DELAY_COL, "mean"),
            total_sales=(SALES_COL, "sum"),
            total_profit=(PROFIT_COL, "sum"),
        )
        .reset_index()
        .rename(columns={source_col: "source", target_col: "target"})
    )

    edge_df["relation"] = relation
    edge_df["edge_weight"] = edge_df["order_count"]
    edge_df["risk_weight"] = edge_df["avg_risk_score"] * np.log1p(edge_df["order_count"])

    return edge_df[
        [
            "source",
            "target",
            "relation",
            "order_count",
            "edge_weight",
            "avg_risk_score",
            "avg_risk_label",
            "avg_delay_gap",
            "total_sales",
            "total_profit",
            "risk_weight",
        ]
    ]


def build_heterogeneous_edges(df: pd.DataFrame) -> pd.DataFrame:
    edge_parts = []

    relations = [
        ("node_product", "node_category", "product_to_category"),
        ("node_category", "node_region", "category_to_region"),
        ("node_category", "node_market", "category_to_market"),
        ("node_category", "node_shipping", "category_to_shipping"),
        ("node_category", "node_supplier_proxy", "category_to_supplier_proxy"),
        ("node_buyer_agent", "node_region", "buyer_agent_to_region"),
        ("node_supplier_proxy", "node_shipping", "supplier_proxy_to_shipping"),
        ("node_supplier_proxy", "node_market", "supplier_proxy_to_market"),
    ]

    for source_col, target_col, relation in relations:
        if source_col in df.columns and target_col in df.columns:
            edge_parts.append(aggregate_edges(df, source_col, target_col, relation))

    if not edge_parts:
        raise RuntimeError("No graph edges could be created. Check entity columns.")

    edges = pd.concat(edge_parts, ignore_index=True)

    edges = (
        edges.groupby(["source", "target", "relation"], observed=False)
        .agg(
            order_count=("order_count", "sum"),
            edge_weight=("edge_weight", "sum"),
            avg_risk_score=("avg_risk_score", "mean"),
            avg_risk_label=("avg_risk_label", "mean"),
            avg_delay_gap=("avg_delay_gap", "mean"),
            total_sales=("total_sales", "sum"),
            total_profit=("total_profit", "sum"),
            risk_weight=("risk_weight", "sum"),
        )
        .reset_index()
    )

    return edges


def build_heterogeneous_nodes(df: pd.DataFrame) -> pd.DataFrame:
    node_frames = []

    for node_type in ENTITY_COLUMN_MAP.keys():
        node_col = f"node_{node_type}"

        if node_col not in df.columns:
            continue

        temp = df[
            [
                node_col,
                ORDER_ID_COL,
                RISK_SCORE_COL,
                RISK_LABEL_COL,
                DELAY_COL,
                SALES_COL,
                PROFIT_COL,
            ]
        ].copy()

        temp = temp.rename(columns={node_col: "node_id"})
        temp["node_type"] = node_type
        node_frames.append(temp)

    if not node_frames:
        raise RuntimeError("No graph nodes could be created. Check entity columns.")

    all_nodes = pd.concat(node_frames, ignore_index=True)

    nodes = (
        all_nodes.groupby(["node_id", "node_type"], observed=False)
        .agg(
            order_count=(ORDER_ID_COL, "count"),
            avg_risk_score=(RISK_SCORE_COL, "mean"),
            avg_risk_label=(RISK_LABEL_COL, "mean"),
            avg_delay_gap=(DELAY_COL, "mean"),
            total_sales=(SALES_COL, "sum"),
            total_profit=(PROFIT_COL, "sum"),
        )
        .reset_index()
    )

    nodes["node_value"] = nodes["node_id"].apply(lambda x: str(x).split("::", 1)[1] if "::" in str(x) else str(x))
    nodes["node_risk_pressure"] = nodes["avg_risk_score"] * np.log1p(nodes["order_count"])

    return nodes[
        [
            "node_id",
            "node_type",
            "node_value",
            "order_count",
            "avg_risk_score",
            "avg_risk_label",
            "avg_delay_gap",
            "total_sales",
            "total_profit",
            "node_risk_pressure",
        ]
    ]


def build_networkx_graph(nodes: pd.DataFrame, edges: pd.DataFrame) -> nx.DiGraph:
    graph = nx.DiGraph()

    for _, row in nodes.iterrows():
        graph.add_node(
            row["node_id"],
            node_type=row["node_type"],
            node_value=row["node_value"],
            order_count=float(row["order_count"]),
            avg_risk_score=float(row["avg_risk_score"]),
            node_risk_pressure=float(row["node_risk_pressure"]),
        )

    for _, row in edges.iterrows():
        graph.add_edge(
            row["source"],
            row["target"],
            relation=row["relation"],
            weight=float(row["edge_weight"]),
            risk_weight=float(row["risk_weight"]),
            avg_risk_score=float(row["avg_risk_score"]),
            order_count=float(row["order_count"]),
        )

    return graph


def compute_graph_centrality_features(graph: nx.DiGraph, nodes: pd.DataFrame) -> pd.DataFrame:
    nodes = nodes.copy()

    if graph.number_of_nodes() == 0:
        nodes["degree_centrality"] = 0
        nodes["pagerank"] = 0
        nodes["weighted_degree"] = 0
        return nodes

    degree_centrality = nx.degree_centrality(graph)

    try:
        pagerank = nx.pagerank(graph, weight="weight", max_iter=200)
    except Exception:
        pagerank = {node: 0 for node in graph.nodes()}

    weighted_degree = {}
    for node in graph.nodes():
        total_weight = 0.0

        for _, _, data in graph.out_edges(node, data=True):
            total_weight += float(data.get("weight", 0.0))

        for _, _, data in graph.in_edges(node, data=True):
            total_weight += float(data.get("weight", 0.0))

        weighted_degree[node] = total_weight

    nodes["degree_centrality"] = nodes["node_id"].map(degree_centrality).fillna(0)
    nodes["pagerank"] = nodes["node_id"].map(pagerank).fillna(0)
    nodes["weighted_degree"] = nodes["node_id"].map(weighted_degree).fillna(0)

    nodes["graph_risk_centrality"] = (
        nodes["avg_risk_score"] * 0.50
        + nodes["degree_centrality"] * 0.25
        + nodes["pagerank"] * 0.25
    )

    return nodes


def add_row_level_graph_features(df: pd.DataFrame, nodes: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    metric_cols = [
        "avg_risk_score",
        "degree_centrality",
        "pagerank",
        "weighted_degree",
        "graph_risk_centrality",
        "node_risk_pressure",
    ]

    lookup = {
        metric: nodes.set_index("node_id")[metric].to_dict()
        for metric in metric_cols
        if metric in nodes.columns
    }

    entity_names = list(ENTITY_COLUMN_MAP.keys())

    for entity in entity_names:
        node_col = f"node_{entity}"
        if node_col not in df.columns:
            continue

        for metric in metric_cols:
            if metric in lookup:
                feature_name = f"graph_{entity}_{metric}"
                df[feature_name] = df[node_col].map(lookup[metric]).fillna(0)

    risk_feature_cols = [
        c for c in df.columns
        if c.startswith("graph_") and c.endswith("avg_risk_score")
    ]

    pagerank_feature_cols = [
        c for c in df.columns
        if c.startswith("graph_") and c.endswith("pagerank")
    ]

    centrality_feature_cols = [
        c for c in df.columns
        if c.startswith("graph_") and c.endswith("graph_risk_centrality")
    ]

    pressure_feature_cols = [
        c for c in df.columns
        if c.startswith("graph_") and c.endswith("node_risk_pressure")
    ]

    df["sthg_risknet_mean_node_risk"] = df[risk_feature_cols].mean(axis=1) if risk_feature_cols else 0
    df["sthg_risknet_mean_pagerank"] = df[pagerank_feature_cols].mean(axis=1) if pagerank_feature_cols else 0
    df["sthg_risknet_mean_risk_centrality"] = df[centrality_feature_cols].mean(axis=1) if centrality_feature_cols else 0
    df["sthg_risknet_mean_risk_pressure"] = df[pressure_feature_cols].mean(axis=1) if pressure_feature_cols else 0

    df["sthg_risknet_graph_risk_signal"] = (
        0.40 * df["sthg_risknet_mean_node_risk"]
        + 0.30 * df["sthg_risknet_mean_risk_centrality"]
        + 0.20 * df["sthg_risknet_mean_risk_pressure"].rank(pct=True)
        + 0.10 * df["sthg_risknet_mean_pagerank"].rank(pct=True)
    )

    df["sthg_risknet_graph_risk_signal"] = df["sthg_risknet_graph_risk_signal"].fillna(0)

    return df


def make_node_type_summary(nodes: pd.DataFrame) -> pd.DataFrame:
    return (
        nodes.groupby("node_type", observed=False)
        .agg(
            node_count=("node_id", "count"),
            avg_risk_score=("avg_risk_score", "mean"),
            avg_order_count=("order_count", "mean"),
            total_orders=("order_count", "sum"),
            avg_graph_risk_centrality=("graph_risk_centrality", "mean"),
        )
        .reset_index()
        .sort_values("node_count", ascending=False)
    )


def make_edge_relation_summary(edges: pd.DataFrame) -> pd.DataFrame:
    return (
        edges.groupby("relation", observed=False)
        .agg(
            edge_count=("relation", "count"),
            total_order_count=("order_count", "sum"),
            avg_risk_score=("avg_risk_score", "mean"),
            avg_delay_gap=("avg_delay_gap", "mean"),
            total_sales=("total_sales", "sum"),
            total_profit=("total_profit", "sum"),
        )
        .reset_index()
        .sort_values("edge_count", ascending=False)
    )


def save_graph_pickle(graph: nx.DiGraph, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(graph, f)


def build_graph_risk_dataset(df: pd.DataFrame) -> Dict[str, object]:
    graph_df = prepare_graph_entity_columns(df)

    edges = build_heterogeneous_edges(graph_df)
    nodes = build_heterogeneous_nodes(graph_df)

    graph = build_networkx_graph(nodes, edges)
    nodes = compute_graph_centrality_features(graph, nodes)

    graph_feature_df = add_row_level_graph_features(graph_df, nodes)

    node_type_summary = make_node_type_summary(nodes)
    edge_relation_summary = make_edge_relation_summary(edges)

    top_risk_nodes = (
        nodes.sort_values("graph_risk_centrality", ascending=False)
        .head(50)
        .reset_index(drop=True)
    )

    summary = {
        "rows": int(graph_feature_df.shape[0]),
        "columns": int(graph_feature_df.shape[1]),
        "node_count": int(nodes.shape[0]),
        "edge_count": int(edges.shape[0]),
        "networkx_nodes": int(graph.number_of_nodes()),
        "networkx_edges": int(graph.number_of_edges()),
        "node_types": node_type_summary.to_dict(orient="records"),
        "edge_relations": edge_relation_summary.to_dict(orient="records"),
        "core_graph_features": [
            "sthg_risknet_mean_node_risk",
            "sthg_risknet_mean_pagerank",
            "sthg_risknet_mean_risk_centrality",
            "sthg_risknet_mean_risk_pressure",
            "sthg_risknet_graph_risk_signal",
        ],
    }

    return {
        "graph_feature_df": graph_feature_df,
        "nodes": nodes,
        "edges": edges,
        "graph": graph,
        "node_type_summary": node_type_summary,
        "edge_relation_summary": edge_relation_summary,
        "top_risk_nodes": top_risk_nodes,
        "summary": summary,
    }