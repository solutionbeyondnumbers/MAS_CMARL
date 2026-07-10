import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from resilientgraph_cmarl.m00_config import (
    PROCESSED_DIR,
    GRAPH_DIR,
    TABLES_DIR,
    REPORTS_DIR,
    LOGS_DIR,
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
    save_graph_pickle,
)


INPUT_FILE = PROCESSED_DIR / "04_dataco_features_risk_labels.csv"
OUTPUT_FEATURE_FILE = PROCESSED_DIR / "05_dataco_graph_risk_features.csv"

NODES_FILE = GRAPH_DIR / "05_graph_nodes.csv"
EDGES_FILE = GRAPH_DIR / "05_graph_edges.csv"
GRAPH_FILE = GRAPH_DIR / "05_supply_chain_graph.pkl"


def main():
    print_header("STEP 05: GRAPH CONSTRUCTION + STHG-RiskNet FEATURE LAYER")

    print(f"[TIME] {timestamp()}")
    print(f"[PROJECT ROOT] {PROJECT_ROOT}")
    print(f"[INPUT FILE] {INPUT_FILE}")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Step 04 output not found: {INPUT_FILE}")

    print_subheader("Loading Step 04 Risk-Labelled Dataset")
    df = load_csv_flexible(INPUT_FILE)
    print(f"[INPUT SHAPE] {df.shape[0]:,} rows | {df.shape[1]:,} columns")

    print_subheader("Building Heterogeneous Supply Chain Graph")
    results = build_graph_risk_dataset(df)

    graph_feature_df = results["graph_feature_df"]
    nodes = results["nodes"]
    edges = results["edges"]
    graph = results["graph"]
    node_type_summary = results["node_type_summary"]
    edge_relation_summary = results["edge_relation_summary"]
    top_risk_nodes = results["top_risk_nodes"]
    summary = results["summary"]

    print(f"[GRAPH FEATURE DATASET] {graph_feature_df.shape[0]:,} rows | {graph_feature_df.shape[1]:,} columns")
    print(f"[NODES] {nodes.shape[0]:,}")
    print(f"[EDGES] {edges.shape[0]:,}")
    print(f"[NETWORKX NODES] {graph.number_of_nodes():,}")
    print(f"[NETWORKX EDGES] {graph.number_of_edges():,}")

    print_subheader("Saving Graph Outputs")

    save_csv(graph_feature_df, OUTPUT_FEATURE_FILE)
    save_csv(nodes, NODES_FILE)
    save_csv(edges, EDGES_FILE)
    save_graph_pickle(graph, GRAPH_FILE)
    print(f"[SAVED] {GRAPH_FILE}")

    save_csv(node_type_summary, TABLES_DIR / "05_graph_node_type_summary.csv")
    save_csv(edge_relation_summary, TABLES_DIR / "05_graph_edge_relation_summary.csv")
    save_csv(top_risk_nodes, TABLES_DIR / "05_top_risk_nodes.csv")

    save_json(summary, REPORTS_DIR / "05_graph_construction_summary.json")

    report_lines = []
    report_lines.append("STEP 05 GRAPH CONSTRUCTION AND STHG-RiskNet FEATURE REPORT")
    report_lines.append("=" * 90)
    report_lines.append(f"Time: {timestamp()}")
    report_lines.append(f"Input file: {INPUT_FILE}")
    report_lines.append(f"Input shape: {df.shape[0]:,} rows x {df.shape[1]:,} columns")
    report_lines.append(f"Graph feature output shape: {graph_feature_df.shape[0]:,} rows x {graph_feature_df.shape[1]:,} columns")
    report_lines.append(f"Graph nodes: {nodes.shape[0]:,}")
    report_lines.append(f"Graph edges: {edges.shape[0]:,}")
    report_lines.append("")
    report_lines.append("Node types:")
    for row in node_type_summary.to_dict(orient="records"):
        report_lines.append(
            f"- {row['node_type']}: {row['node_count']} nodes, "
            f"average risk={row['avg_risk_score']:.4f}"
        )
    report_lines.append("")
    report_lines.append("Edge relations:")
    for row in edge_relation_summary.to_dict(orient="records"):
        report_lines.append(
            f"- {row['relation']}: {row['edge_count']} edges, "
            f"average risk={row['avg_risk_score']:.4f}"
        )
    report_lines.append("")
    report_lines.append("Created STHG-RiskNet graph features:")
    for col in summary["core_graph_features"]:
        report_lines.append(f"- {col}")
    report_lines.append("")
    report_lines.append("Important note:")
    report_lines.append(
        "DataCo does not provide direct supplier identifiers. Therefore, supplier_proxy nodes "
        "are derived using category, market, and shipping-mode patterns. These graph features "
        "will be used by the proposed RGC models in Step 07."
    )

    save_text("\n".join(report_lines), REPORTS_DIR / "05_graph_construction_report.txt")

    log_text = (
        f"[{timestamp()}] Step 05 completed. "
        f"Rows={graph_feature_df.shape[0]}, Columns={graph_feature_df.shape[1]}, "
        f"Nodes={nodes.shape[0]}, Edges={edges.shape[0]}\n"
    )
    save_text(log_text, LOGS_DIR / "05_graph_construction.log")

    print_subheader("Node Type Summary")
    print(node_type_summary.to_string(index=False))

    print_subheader("Edge Relation Summary")
    print(edge_relation_summary.to_string(index=False))

    print_subheader("Top 10 High-Risk Graph Nodes")
    print(
        top_risk_nodes[
            [
                "node_id",
                "node_type",
                "order_count",
                "avg_risk_score",
                "pagerank",
                "degree_centrality",
                "graph_risk_centrality",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    print_subheader("Saved Files")
    print(f"[GRAPH FEATURE DATASET] {OUTPUT_FEATURE_FILE}")
    print(f"[NODES] {NODES_FILE}")
    print(f"[EDGES] {EDGES_FILE}")
    print(f"[GRAPH PKL] {GRAPH_FILE}")

    print_header("STEP 05 COMPLETED SUCCESSFULLY")
    print("[NEXT] Send me the terminal output.")
    print("[NEXT FILE AFTER VERIFICATION] 06_train_baseline_3_ml_models.py")


if __name__ == "__main__":
    main()