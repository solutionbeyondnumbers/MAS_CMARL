from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
GRAPH_DIR = DATA_DIR / "graph"
SIMULATION_DIR = DATA_DIR / "simulation"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
TABLES_DIR = OUTPUT_DIR / "tables"
FIGURES_DIR = OUTPUT_DIR / "figures"
MODELS_DIR = OUTPUT_DIR / "models"
LOGS_DIR = OUTPUT_DIR / "logs"
REPORTS_DIR = OUTPUT_DIR / "reports"
EXPLAINABILITY_DIR = OUTPUT_DIR / "explainability"
STRESS_TESTS_DIR = OUTPUT_DIR / "stress_tests"
DASHBOARD_DIR = OUTPUT_DIR / "dashboard"

RAW_DATA_FILE = RAW_DIR / "DataCoSupplyChainDataset.csv"

RANDOM_STATE = 42

BASELINE_MODELS = [
    "XGBoost",
    "LightGBM",
    "CatBoost",
    "PPO",
    "MAPPO",
]

PROPOSED_MODELS = [
    "XGBoost-RGC",
    "LightGBM-RGC",
    "CatBoost-RGC",
    "STHG-CPPO",
    "STHG-CMAPPO",
]

RISK_LABEL_MAP = {
    0: "Low",
    1: "Moderate",
    2: "High",
}

EXPECTED_DATACO_COLUMNS = {
    "order_id": ["Order Id", "Order ID", "order_id"],
    "order_date": ["order date (DateOrders)", "Order Date", "order_date"],
    "shipping_date": ["shipping date (DateOrders)", "Shipping Date", "shipping_date"],
    "delivery_status": ["Delivery Status", "delivery_status"],
    "late_delivery_risk": ["Late_delivery_risk", "late_delivery_risk", "Late Delivery Risk"],
    "category": ["Category Name", "category_name", "Category"],
    "product": ["Product Name", "Product Card Id", "Product Category Id", "product_name"],
    "market": ["Market", "market"],
    "region": ["Order Region", "Customer Country", "Customer Segment", "order_region"],
    "shipping_mode": ["Shipping Mode", "shipping_mode"],
    "sales": ["Sales", "sales"],
    "profit": ["Order Profit Per Order", "Benefit per order", "profit"],
    "quantity": ["Order Item Quantity", "quantity"],
    "discount": ["Order Item Discount", "discount"],
    "real_shipping_days": ["Days for shipping (real)"],
    "scheduled_shipping_days": ["Days for shipment (scheduled)"],
}

LEAKAGE_KEYWORDS = [
    "late_delivery_risk",
    "delivery_status",
    "shipping_date",
    "days for shipping",
    "days for shipment",
    "order_status",
    "fraud",
    "risk",
]