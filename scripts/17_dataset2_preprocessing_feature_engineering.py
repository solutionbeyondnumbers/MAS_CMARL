# scripts/17_dataset2_preprocessing_feature_engineering.py
# ======================================================================================
# STEP 17: DATASET 2 PREPROCESSING + FEATURE ENGINEERING + RISK LABELS
# Project: ResilientGraph-CMARL
# Dataset 2: Brazilian E-Commerce Public Dataset by Olist
#
# Purpose:
#   1. Load all Olist raw CSV files from data/dataset2_raw/.
#   2. Integrate orders, order items, customers, sellers, products, payments, reviews,
#      category translation, and geolocation.
#   3. Build an order-item-level fulfilment dataset.
#   4. Engineer delivery, service, freight, payment, seller, route, category, product,
#      customer, geospatial, and temporal features.
#   5. Construct composite fulfilment-disruption risk score and Low/Moderate/High labels.
#   6. Create leakage-aware temporal train/validation/test split.
#   7. Save all outputs in separate Dataset 2 folders.
# ======================================================================================

from __future__ import annotations

import json
import math
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------------------
# Project paths
# --------------------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET2_RAW_DIR = PROJECT_ROOT / "data" / "dataset2_raw"
DATASET2_EXTRACTED_DIR = DATASET2_RAW_DIR / "olist_extracted"
DATASET2_PROCESSED_DIR = PROJECT_ROOT / "data" / "dataset2_processed"
DATASET2_GRAPH_DIR = PROJECT_ROOT / "data" / "dataset2_graph"
DATASET2_SIMULATION_DIR = PROJECT_ROOT / "data" / "dataset2_simulation"

OUTPUTS2_DIR = PROJECT_ROOT / "outputs_dataset2"
TABLES2_DIR = OUTPUTS2_DIR / "tables"
FIGURES2_DIR = OUTPUTS2_DIR / "figures"
MODELS2_DIR = OUTPUTS2_DIR / "models"
LOGS2_DIR = OUTPUTS2_DIR / "logs"
REPORTS2_DIR = OUTPUTS2_DIR / "reports"

LOG_FILE = LOGS2_DIR / "17_dataset2_preprocessing_feature_engineering.log"


REQUIRED_OLIST_FILES = {
    "customers": "olist_customers_dataset.csv",
    "geolocation": "olist_geolocation_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "order_payments": "olist_order_payments_dataset.csv",
    "order_reviews": "olist_order_reviews_dataset.csv",
    "orders": "olist_orders_dataset.csv",
    "products": "olist_products_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "category_translation": "product_category_name_translation.csv",
}


DATE_COLUMNS = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
    "shipping_limit_date",
    "review_creation_date_min",
    "review_answer_timestamp_max",
]


RISK_WEIGHTS = {
    "delivery_delay_component": 0.28,
    "order_status_component": 0.15,
    "review_service_component": 0.17,
    "freight_cost_component": 0.10,
    "seller_historical_delay_component": 0.12,
    "route_historical_delay_component": 0.08,
    "category_demand_volatility_component": 0.07,
    "payment_complexity_component": 0.03,
}


# --------------------------------------------------------------------------------------
# Logging helpers
# --------------------------------------------------------------------------------------

def ensure_directories() -> None:
    dirs = [
        DATASET2_RAW_DIR,
        DATASET2_EXTRACTED_DIR,
        DATASET2_PROCESSED_DIR,
        DATASET2_GRAPH_DIR,
        DATASET2_SIMULATION_DIR,
        OUTPUTS2_DIR,
        TABLES2_DIR,
        FIGURES2_DIR,
        MODELS2_DIR,
        LOGS2_DIR,
        REPORTS2_DIR,
    ]

    for d in dirs:
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
# File loading
# --------------------------------------------------------------------------------------

def candidate_search_dirs() -> List[Path]:
    return [
        DATASET2_RAW_DIR,
        DATASET2_EXTRACTED_DIR,
        PROJECT_ROOT / "data" / "raw" / "dataset2_raw",
        PROJECT_ROOT / "data" / "raw",
        PROJECT_ROOT,
    ]


def locate_required_files() -> Dict[str, Path | None]:
    file_map: Dict[str, Path | None] = {}

    for table_name, file_name in REQUIRED_OLIST_FILES.items():
        found_path = None

        for base_dir in candidate_search_dirs():
            candidate = base_dir / file_name
            if candidate.exists():
                found_path = candidate
                break

        if found_path is None:
            matches = list(PROJECT_ROOT.glob(f"**/{file_name}"))
            if matches:
                found_path = matches[0]

        file_map[table_name] = found_path

    return file_map


def read_csv_safe(path: Path) -> pd.DataFrame:
    encodings = ["utf-8", "utf-8-sig", "latin1"]
    last_error = None

    for enc in encodings:
        try:
            df = pd.read_csv(path, encoding=enc)
            log(f"[OK] Loaded {path.name} using encoding: {enc} | shape={df.shape}")
            return df
        except Exception as exc:
            last_error = exc

    raise RuntimeError(f"Could not read {path}. Last error: {last_error}")


def load_raw_tables() -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    file_map = locate_required_files()

    rows = []
    tables: Dict[str, pd.DataFrame] = {}

    for table_name, file_name in REQUIRED_OLIST_FILES.items():
        path = file_map.get(table_name)

        rows.append(
            {
                "table_name": table_name,
                "expected_file": file_name,
                "file_found": path is not None,
                "file_path": str(path) if path else "",
            }
        )

        if path is None:
            raise FileNotFoundError(
                f"Missing required Olist file: {file_name}. "
                f"Put it inside {DATASET2_RAW_DIR}"
            )

        tables[table_name] = read_csv_safe(path)

    inventory = pd.DataFrame(rows)
    return tables, inventory


# --------------------------------------------------------------------------------------
# Utility functions
# --------------------------------------------------------------------------------------

def parse_datetime_safe(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def normalize_text_column(series: pd.Series, unknown_value: str = "unknown") -> pd.Series:
    return (
        series.astype("string")
        .fillna(unknown_value)
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", "_", regex=True)
        .replace("", unknown_value)
    )


def safe_divide(numerator: pd.Series, denominator: pd.Series, default: float = 0.0) -> pd.Series:
    result = numerator / denominator.replace(0, np.nan)
    return result.replace([np.inf, -np.inf], np.nan).fillna(default)


def percentile_component(series: pd.Series) -> pd.Series:
    clean = pd.to_numeric(series, errors="coerce")

    if clean.notna().sum() == 0:
        return pd.Series(np.zeros(len(series)), index=series.index)

    filled = clean.fillna(clean.median())
    ranked = filled.rank(method="average", pct=True)
    return ranked.clip(0, 1)


def clipped_linear_component(
    series: pd.Series,
    lower: float,
    upper: float,
    fill_value: float = 0.0,
) -> pd.Series:
    clean = pd.to_numeric(series, errors="coerce").fillna(fill_value)

    if upper == lower:
        return pd.Series(np.zeros(len(series)), index=series.index)

    scaled = (clean - lower) / (upper - lower)
    return scaled.clip(0, 1)


def haversine_km(
    lat1: pd.Series,
    lon1: pd.Series,
    lat2: pd.Series,
    lon2: pd.Series,
) -> pd.Series:
    lat1_rad = np.radians(pd.to_numeric(lat1, errors="coerce"))
    lon1_rad = np.radians(pd.to_numeric(lon1, errors="coerce"))
    lat2_rad = np.radians(pd.to_numeric(lat2, errors="coerce"))
    lon2_rad = np.radians(pd.to_numeric(lon2, errors="coerce"))

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    return 6371.0 * c


def build_missing_summary(df: pd.DataFrame, stage: str) -> pd.DataFrame:
    rows = []

    for col in df.columns:
        missing = int(df[col].isna().sum())
        rows.append(
            {
                "stage": stage,
                "column_name": col,
                "dtype": str(df[col].dtype),
                "missing_count": missing,
                "missing_percent": round(missing / len(df) * 100, 4) if len(df) else 0,
                "unique_values": int(df[col].nunique(dropna=True)),
            }
        )

    return pd.DataFrame(rows).sort_values(["missing_count", "missing_percent"], ascending=False)


def build_shape_report(stage_rows: List[Dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(stage_rows)


def add_prior_rate(
    df: pd.DataFrame,
    group_col: str,
    target_col: str,
    new_col: str,
    default_value: float,
) -> pd.DataFrame:
    target = pd.to_numeric(df[target_col], errors="coerce").fillna(0).astype(float)

    group_series = df[group_col].astype(str).fillna("unknown")

    cumulative_sum_before = target.groupby(group_series).cumsum() - target
    cumulative_count_before = df.groupby(group_series).cumcount()

    prior_rate = cumulative_sum_before / cumulative_count_before.replace(0, np.nan)
    df[new_col] = prior_rate.fillna(default_value).clip(0, 1)

    return df


def final_fill_all_missing(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []

    for col in df.columns:
        missing_before = int(df[col].isna().sum())

        if missing_before == 0:
            rows.append(
                {
                    "column_name": col,
                    "dtype": str(df[col].dtype),
                    "missing_before": 0,
                    "fill_strategy": "not_required",
                    "missing_after": 0,
                }
            )
            continue

        if pd.api.types.is_datetime64_any_dtype(df[col]):
            fill_value = pd.Timestamp("1900-01-01")
            df[col] = df[col].fillna(fill_value)
            fill_strategy = "sentinel_datetime_1900_01_01_after_missing_flags_created"

        elif pd.api.types.is_numeric_dtype(df[col]):
            median_value = df[col].median()

            if pd.isna(median_value):
                median_value = 0.0

            df[col] = df[col].fillna(median_value)
            fill_strategy = f"numeric_median_or_zero:{median_value}"

        else:
            df[col] = df[col].fillna("unknown")
            fill_strategy = "categorical_unknown"

        missing_after = int(df[col].isna().sum())

        rows.append(
            {
                "column_name": col,
                "dtype": str(df[col].dtype),
                "missing_before": missing_before,
                "fill_strategy": fill_strategy,
                "missing_after": missing_after,
            }
        )

    return df, pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Aggregation functions
# --------------------------------------------------------------------------------------

def aggregate_payments(payments: pd.DataFrame) -> pd.DataFrame:
    payments = payments.copy()

    payments["payment_type"] = normalize_text_column(payments["payment_type"], "unknown_payment")

    payment_agg = (
        payments.groupby("order_id", as_index=False)
        .agg(
            payment_value_sum=("payment_value", "sum"),
            payment_value_mean=("payment_value", "mean"),
            payment_value_max=("payment_value", "max"),
            payment_installments_mean=("payment_installments", "mean"),
            payment_installments_max=("payment_installments", "max"),
            payment_count=("payment_sequential", "count"),
            payment_type_count=("payment_type", "nunique"),
        )
    )

    dominant_payment = (
        payments.groupby("order_id")["payment_type"]
        .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else "unknown_payment")
        .reset_index()
        .rename(columns={"payment_type": "payment_type_dominant"})
    )

    payment_agg = payment_agg.merge(dominant_payment, on="order_id", how="left")

    return payment_agg


def aggregate_reviews(reviews: pd.DataFrame) -> pd.DataFrame:
    reviews = reviews.copy()

    reviews["review_creation_date"] = pd.to_datetime(reviews["review_creation_date"], errors="coerce")
    reviews["review_answer_timestamp"] = pd.to_datetime(reviews["review_answer_timestamp"], errors="coerce")

    reviews["review_comment_title_present"] = reviews["review_comment_title"].notna().astype(int)
    reviews["review_comment_message_present"] = reviews["review_comment_message"].notna().astype(int)

    reviews["review_response_days"] = (
        reviews["review_answer_timestamp"] - reviews["review_creation_date"]
    ).dt.total_seconds() / 86400.0

    review_agg = (
        reviews.groupby("order_id", as_index=False)
        .agg(
            review_score_mean=("review_score", "mean"),
            review_score_min=("review_score", "min"),
            review_score_max=("review_score", "max"),
            review_count=("review_id", "count"),
            review_comment_title_count=("review_comment_title_present", "sum"),
            review_comment_message_count=("review_comment_message_present", "sum"),
            review_response_days_mean=("review_response_days", "mean"),
            review_creation_date_min=("review_creation_date", "min"),
            review_answer_timestamp_max=("review_answer_timestamp", "max"),
        )
    )

    return review_agg


def prepare_products_with_translation(products: pd.DataFrame, translation: pd.DataFrame) -> pd.DataFrame:
    products = products.copy()
    translation = translation.copy()

    if "product_category_name" in products.columns:
        products["product_category_name"] = normalize_text_column(
            products["product_category_name"],
            "unknown_category",
        )

    if "product_category_name" in translation.columns:
        translation["product_category_name"] = normalize_text_column(
            translation["product_category_name"],
            "unknown_category",
        )

    prod = products.merge(translation, on="product_category_name", how="left")

    if "product_category_name_english" not in prod.columns:
        prod["product_category_name_english"] = prod["product_category_name"]

    prod["product_category_name_english"] = (
        prod["product_category_name_english"]
        .astype("string")
        .fillna(prod["product_category_name"])
        .fillna("unknown_category")
    )

    prod["product_category_name_english"] = normalize_text_column(
        prod["product_category_name_english"],
        "unknown_category",
    )

    return prod


def prepare_geolocation_features(
    df: pd.DataFrame,
    geolocation: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    geo = geolocation.copy()

    geo["geolocation_zip_code_prefix"] = pd.to_numeric(
        geo["geolocation_zip_code_prefix"],
        errors="coerce",
    )

    geo_agg = (
        geo.dropna(subset=["geolocation_zip_code_prefix"])
        .groupby("geolocation_zip_code_prefix", as_index=False)
        .agg(
            geo_lat_median=("geolocation_lat", "median"),
            geo_lng_median=("geolocation_lng", "median"),
            geo_zip_records=("geolocation_lat", "size"),
        )
    )

    df["customer_zip_code_prefix"] = pd.to_numeric(
        df["customer_zip_code_prefix"],
        errors="coerce",
    )
    df["seller_zip_code_prefix"] = pd.to_numeric(
        df["seller_zip_code_prefix"],
        errors="coerce",
    )

    customer_geo = geo_agg.rename(
        columns={
            "geolocation_zip_code_prefix": "customer_zip_code_prefix",
            "geo_lat_median": "customer_lat",
            "geo_lng_median": "customer_lng",
            "geo_zip_records": "customer_geo_records",
        }
    )

    seller_geo = geo_agg.rename(
        columns={
            "geolocation_zip_code_prefix": "seller_zip_code_prefix",
            "geo_lat_median": "seller_lat",
            "geo_lng_median": "seller_lng",
            "geo_zip_records": "seller_geo_records",
        }
    )

    before_rows = len(df)

    df = df.merge(customer_geo, on="customer_zip_code_prefix", how="left")
    df = df.merge(seller_geo, on="seller_zip_code_prefix", how="left")

    df["geo_distance_km"] = haversine_km(
        df["seller_lat"],
        df["seller_lng"],
        df["customer_lat"],
        df["customer_lng"],
    )

    df["geo_distance_missing_flag"] = df["geo_distance_km"].isna().astype(int)

    geo_join_report = pd.DataFrame(
        [
            {
                "metric": "rows_before_geolocation_join",
                "value": before_rows,
            },
            {
                "metric": "rows_after_geolocation_join",
                "value": len(df),
            },
            {
                "metric": "customer_geo_missing_rows",
                "value": int(df["customer_lat"].isna().sum()),
            },
            {
                "metric": "seller_geo_missing_rows",
                "value": int(df["seller_lat"].isna().sum()),
            },
            {
                "metric": "distance_missing_rows",
                "value": int(df["geo_distance_km"].isna().sum()),
            },
            {
                "metric": "geolocation_unique_zip_prefixes",
                "value": int(geo_agg["geolocation_zip_code_prefix"].nunique()),
            },
        ]
    )

    return df, geo_join_report


# --------------------------------------------------------------------------------------
# Main preprocessing and feature engineering
# --------------------------------------------------------------------------------------

def build_integrated_dataset(tables: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    shape_rows: List[Dict[str, object]] = []

    customers = tables["customers"].copy()
    geolocation = tables["geolocation"].copy()
    order_items = tables["order_items"].copy()
    payments = tables["order_payments"].copy()
    reviews = tables["order_reviews"].copy()
    orders = tables["orders"].copy()
    products = tables["products"].copy()
    sellers = tables["sellers"].copy()
    translation = tables["category_translation"].copy()

    shape_rows.append({"stage": "raw_customers", "rows": len(customers), "columns": customers.shape[1]})
    shape_rows.append({"stage": "raw_geolocation", "rows": len(geolocation), "columns": geolocation.shape[1]})
    shape_rows.append({"stage": "raw_order_items", "rows": len(order_items), "columns": order_items.shape[1]})
    shape_rows.append({"stage": "raw_payments", "rows": len(payments), "columns": payments.shape[1]})
    shape_rows.append({"stage": "raw_reviews", "rows": len(reviews), "columns": reviews.shape[1]})
    shape_rows.append({"stage": "raw_orders", "rows": len(orders), "columns": orders.shape[1]})
    shape_rows.append({"stage": "raw_products", "rows": len(products), "columns": products.shape[1]})
    shape_rows.append({"stage": "raw_sellers", "rows": len(sellers), "columns": sellers.shape[1]})
    shape_rows.append({"stage": "raw_category_translation", "rows": len(translation), "columns": translation.shape[1]})

    print_section("Preparing Product, Payment, and Review Aggregations")

    products_translated = prepare_products_with_translation(products, translation)
    payment_agg = aggregate_payments(payments)
    review_agg = aggregate_reviews(reviews)

    shape_rows.append({"stage": "products_translated", "rows": len(products_translated), "columns": products_translated.shape[1]})
    shape_rows.append({"stage": "payment_aggregated_by_order", "rows": len(payment_agg), "columns": payment_agg.shape[1]})
    shape_rows.append({"stage": "review_aggregated_by_order", "rows": len(review_agg), "columns": review_agg.shape[1]})

    print_section("Building Order-Item-Level Integrated Dataset")

    df = order_items.merge(products_translated, on="product_id", how="left")
    shape_rows.append({"stage": "order_items_plus_products", "rows": len(df), "columns": df.shape[1]})

    df = df.merge(orders, on="order_id", how="left")
    shape_rows.append({"stage": "plus_orders", "rows": len(df), "columns": df.shape[1]})

    df = df.merge(customers, on="customer_id", how="left")
    shape_rows.append({"stage": "plus_customers", "rows": len(df), "columns": df.shape[1]})

    df = df.merge(sellers, on="seller_id", how="left")
    shape_rows.append({"stage": "plus_sellers", "rows": len(df), "columns": df.shape[1]})

    df = df.merge(payment_agg, on="order_id", how="left")
    shape_rows.append({"stage": "plus_payment_aggregates", "rows": len(df), "columns": df.shape[1]})

    df = df.merge(review_agg, on="order_id", how="left")
    shape_rows.append({"stage": "plus_review_aggregates", "rows": len(df), "columns": df.shape[1]})

    df, geo_join_report = prepare_geolocation_features(df, geolocation)
    shape_rows.append({"stage": "plus_geolocation_distance", "rows": len(df), "columns": df.shape[1]})

    shape_report = build_shape_report(shape_rows)

    save_csv(geo_join_report, TABLES2_DIR / "17_dataset2_geolocation_join_report.csv")

    return df, shape_report


def engineer_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print_section("Running Feature Engineering")

    df = df.copy()

    before_missing = build_missing_summary(df, "before_feature_engineering")

    # ------------------------------------------------------------------
    # Normalize important categoricals
    # ------------------------------------------------------------------

    categorical_columns = [
        "order_status",
        "product_category_name",
        "product_category_name_english",
        "customer_city",
        "customer_state",
        "seller_city",
        "seller_state",
        "payment_type_dominant",
    ]

    for col in categorical_columns:
        if col in df.columns:
            df[col] = normalize_text_column(df[col], f"unknown_{col}")

    # ------------------------------------------------------------------
    # Parse dates
    # ------------------------------------------------------------------

    df = parse_datetime_safe(df, DATE_COLUMNS)

    df["order_purchase_date"] = df["order_purchase_timestamp"].dt.floor("D")
    df["purchase_year"] = df["order_purchase_timestamp"].dt.year
    df["purchase_month"] = df["order_purchase_timestamp"].dt.month
    df["purchase_day"] = df["order_purchase_timestamp"].dt.day
    df["purchase_dayofweek"] = df["order_purchase_timestamp"].dt.dayofweek
    df["purchase_hour"] = df["order_purchase_timestamp"].dt.hour
    df["purchase_quarter"] = df["order_purchase_timestamp"].dt.quarter
    df["purchase_weekofyear"] = df["order_purchase_timestamp"].dt.isocalendar().week.astype(float)
    df["purchase_is_weekend"] = df["purchase_dayofweek"].isin([5, 6]).astype(int)

    # ------------------------------------------------------------------
    # Delivery and fulfilment features
    # ------------------------------------------------------------------

    df["is_delivered_status"] = (df["order_status"] == "delivered").astype(int)
    df["is_canceled_status"] = df["order_status"].isin(["canceled", "unavailable"]).astype(int)

    df["approved_missing_flag"] = df["order_approved_at"].isna().astype(int)
    df["carrier_date_missing_flag"] = df["order_delivered_carrier_date"].isna().astype(int)
    df["customer_delivery_missing_flag"] = df["order_delivered_customer_date"].isna().astype(int)

    df["approval_time_hours"] = (
        df["order_approved_at"] - df["order_purchase_timestamp"]
    ).dt.total_seconds() / 3600.0

    df["carrier_dispatch_days"] = (
        df["order_delivered_carrier_date"] - df["order_approved_at"]
    ).dt.total_seconds() / 86400.0

    df["customer_delivery_days"] = (
        df["order_delivered_customer_date"] - df["order_purchase_timestamp"]
    ).dt.total_seconds() / 86400.0

    df["estimated_delivery_days"] = (
        df["order_estimated_delivery_date"] - df["order_purchase_timestamp"]
    ).dt.total_seconds() / 86400.0

    df["delivery_delay_days"] = (
        df["order_delivered_customer_date"] - df["order_estimated_delivery_date"]
    ).dt.total_seconds() / 86400.0

    df["shipping_limit_from_purchase_days"] = (
        df["shipping_limit_date"] - df["order_purchase_timestamp"]
    ).dt.total_seconds() / 86400.0

    df["shipping_limit_from_approval_days"] = (
        df["shipping_limit_date"] - df["order_approved_at"]
    ).dt.total_seconds() / 86400.0

    df["positive_delivery_delay_days"] = pd.to_numeric(
        df["delivery_delay_days"],
        errors="coerce",
    ).clip(lower=0)

    df["non_delivered_flag"] = (
        (df["is_delivered_status"] == 0) | (df["order_delivered_customer_date"].isna())
    ).astype(int)

    df["late_delivery_flag"] = (
        (df["positive_delivery_delay_days"].fillna(0) > 0) | (df["non_delivered_flag"] == 1)
    ).astype(int)

    df["early_delivery_days"] = (-pd.to_numeric(df["delivery_delay_days"], errors="coerce")).clip(lower=0)

    # ------------------------------------------------------------------
    # Price, freight, payment, review, and product features
    # ------------------------------------------------------------------

    df["item_total_value"] = pd.to_numeric(df["price"], errors="coerce").fillna(0) + pd.to_numeric(
        df["freight_value"],
        errors="coerce",
    ).fillna(0)

    df["freight_ratio"] = safe_divide(
        pd.to_numeric(df["freight_value"], errors="coerce").fillna(0),
        df["item_total_value"],
        default=0,
    )

    df["payment_value_to_item_value_ratio"] = safe_divide(
        pd.to_numeric(df["payment_value_sum"], errors="coerce").fillna(0),
        df["item_total_value"],
        default=0,
    )

    df["review_missing_flag"] = df["review_score_mean"].isna().astype(int)
    df["review_score_filled"] = pd.to_numeric(df["review_score_mean"], errors="coerce").fillna(3.0)
    df["low_review_flag"] = (df["review_score_filled"] <= 2).astype(int)
    df["high_review_flag"] = (df["review_score_filled"] >= 4).astype(int)

    df["product_volume_cm3"] = (
        pd.to_numeric(df["product_length_cm"], errors="coerce").fillna(0)
        * pd.to_numeric(df["product_height_cm"], errors="coerce").fillna(0)
        * pd.to_numeric(df["product_width_cm"], errors="coerce").fillna(0)
    )

    df["product_weight_kg"] = pd.to_numeric(df["product_weight_g"], errors="coerce") / 1000.0

    df["freight_per_kg"] = safe_divide(
        pd.to_numeric(df["freight_value"], errors="coerce").fillna(0),
        df["product_weight_kg"],
        default=0,
    )

    df["freight_per_1000cm3"] = safe_divide(
        pd.to_numeric(df["freight_value"], errors="coerce").fillna(0),
        df["product_volume_cm3"] / 1000.0,
        default=0,
    )

    df["same_seller_customer_state_flag"] = (df["seller_state"] == df["customer_state"]).astype(int)
    df["seller_customer_route"] = df["seller_state"].astype(str) + "_to_" + df["customer_state"].astype(str)
    df["seller_category_key"] = df["seller_id"].astype(str) + "_x_" + df["product_category_name_english"].astype(str)
    df["customer_category_key"] = df["customer_state"].astype(str) + "_x_" + df["product_category_name_english"].astype(str)

    # ------------------------------------------------------------------
    # Order-level counts mapped back to item-level rows
    # ------------------------------------------------------------------

    order_item_count = (
        df.groupby("order_id")["order_item_id"]
        .count()
        .rename("order_item_count")
        .reset_index()
    )

    order_seller_count = (
        df.groupby("order_id")["seller_id"]
        .nunique()
        .rename("order_seller_count")
        .reset_index()
    )

    order_category_count = (
        df.groupby("order_id")["product_category_name_english"]
        .nunique()
        .rename("order_category_count")
        .reset_index()
    )

    df = df.merge(order_item_count, on="order_id", how="left")
    df = df.merge(order_seller_count, on="order_id", how="left")
    df = df.merge(order_category_count, on="order_id", how="left")

    # ------------------------------------------------------------------
    # Sort temporally before historical features
    # ------------------------------------------------------------------

    df = df.sort_values(["order_purchase_timestamp", "order_id", "order_item_id"]).reset_index(drop=True)

    global_late_rate = float(df["late_delivery_flag"].mean())
    global_low_review_rate = float(df["low_review_flag"].mean())

    df = add_prior_rate(
        df,
        group_col="seller_id",
        target_col="late_delivery_flag",
        new_col="seller_historical_late_rate",
        default_value=global_late_rate,
    )

    df = add_prior_rate(
        df,
        group_col="seller_customer_route",
        target_col="late_delivery_flag",
        new_col="route_historical_late_rate",
        default_value=global_late_rate,
    )

    df = add_prior_rate(
        df,
        group_col="product_category_name_english",
        target_col="late_delivery_flag",
        new_col="category_historical_late_rate",
        default_value=global_late_rate,
    )

    df = add_prior_rate(
        df,
        group_col="seller_id",
        target_col="low_review_flag",
        new_col="seller_historical_low_review_rate",
        default_value=global_low_review_rate,
    )

    df["seller_prior_order_item_count"] = df.groupby("seller_id").cumcount()
    df["product_prior_order_item_count"] = df.groupby("product_id").cumcount()
    df["category_prior_order_item_count"] = df.groupby("product_category_name_english").cumcount()
    df["route_prior_order_item_count"] = df.groupby("seller_customer_route").cumcount()

    # ------------------------------------------------------------------
    # Category daily demand and volatility
    # ------------------------------------------------------------------

    daily_category = (
        df.groupby(["product_category_name_english", "order_purchase_date"], as_index=False)
        .agg(
            category_daily_order_items=("order_id", "count"),
            category_daily_unique_orders=("order_id", "nunique"),
            category_daily_unique_sellers=("seller_id", "nunique"),
        )
    )

    daily_category = daily_category.sort_values(
        ["product_category_name_english", "order_purchase_date"]
    ).reset_index(drop=True)

    daily_category["category_demand_roll_mean_7"] = (
        daily_category.groupby("product_category_name_english")["category_daily_order_items"]
        .transform(lambda s: s.shift(1).rolling(window=7, min_periods=2).mean())
    )

    daily_category["category_demand_roll_std_7"] = (
        daily_category.groupby("product_category_name_english")["category_daily_order_items"]
        .transform(lambda s: s.shift(1).rolling(window=7, min_periods=2).std())
    )

    daily_category["category_demand_roll_mean_14"] = (
        daily_category.groupby("product_category_name_english")["category_daily_order_items"]
        .transform(lambda s: s.shift(1).rolling(window=14, min_periods=3).mean())
    )

    daily_category["category_demand_roll_std_14"] = (
        daily_category.groupby("product_category_name_english")["category_daily_order_items"]
        .transform(lambda s: s.shift(1).rolling(window=14, min_periods=3).std())
    )

    df = df.merge(
        daily_category,
        on=["product_category_name_english", "order_purchase_date"],
        how="left",
    )

    for col in [
        "category_daily_order_items",
        "category_daily_unique_orders",
        "category_daily_unique_sellers",
        "category_demand_roll_mean_7",
        "category_demand_roll_std_7",
        "category_demand_roll_mean_14",
        "category_demand_roll_std_14",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # ------------------------------------------------------------------
    # Risk component engineering
    # ------------------------------------------------------------------

    delay_component_base = clipped_linear_component(
        df["positive_delivery_delay_days"],
        lower=0,
        upper=15,
        fill_value=0,
    )

    df["delivery_delay_component"] = np.where(
        df["non_delivered_flag"] == 1,
        1.0,
        delay_component_base,
    )

    df["order_status_component"] = np.select(
        [
            df["order_status"].isin(["canceled", "unavailable"]),
            df["order_status"].isin(["invoiced", "processing", "shipped", "approved", "created"]),
            df["order_status"].eq("delivered"),
        ],
        [
            1.0,
            0.70,
            0.0,
        ],
        default=0.50,
    )

    df["review_service_component"] = ((5.0 - df["review_score_filled"]) / 4.0).clip(0, 1)
    df["review_service_component"] = np.where(
        df["review_missing_flag"] == 1,
        0.40,
        df["review_service_component"],
    )

    freight_value_component = percentile_component(df["freight_value"])
    freight_ratio_component = percentile_component(df["freight_ratio"])
    distance_component = percentile_component(df["geo_distance_km"])

    df["freight_cost_component"] = (
        0.45 * freight_value_component
        + 0.35 * freight_ratio_component
        + 0.20 * distance_component
    ).clip(0, 1)

    df["seller_historical_delay_component"] = pd.to_numeric(
        df["seller_historical_late_rate"],
        errors="coerce",
    ).fillna(global_late_rate).clip(0, 1)

    df["route_historical_delay_component"] = pd.to_numeric(
        df["route_historical_late_rate"],
        errors="coerce",
    ).fillna(global_late_rate).clip(0, 1)

    volatility_7 = percentile_component(df["category_demand_roll_std_7"])
    volatility_14 = percentile_component(df["category_demand_roll_std_14"])
    seller_pressure = percentile_component(df["seller_prior_order_item_count"])

    df["category_demand_volatility_component"] = (
        0.45 * volatility_7
        + 0.35 * volatility_14
        + 0.20 * seller_pressure
    ).clip(0, 1)

    installment_component = clipped_linear_component(
        df["payment_installments_max"],
        lower=1,
        upper=12,
        fill_value=1,
    )

    payment_count_component = clipped_linear_component(
        df["payment_count"],
        lower=1,
        upper=4,
        fill_value=1,
    )

    payment_type_component = clipped_linear_component(
        df["payment_type_count"],
        lower=1,
        upper=3,
        fill_value=1,
    )

    df["payment_complexity_component"] = (
        0.50 * installment_component
        + 0.30 * payment_count_component
        + 0.20 * payment_type_component
    ).clip(0, 1)

    # ------------------------------------------------------------------
    # Composite risk score
    # ------------------------------------------------------------------

    df["dataset2_fulfillment_risk_score"] = 0.0

    for component_name, weight in RISK_WEIGHTS.items():
        df["dataset2_fulfillment_risk_score"] += weight * pd.to_numeric(
            df[component_name],
            errors="coerce",
        ).fillna(0)

    df["dataset2_fulfillment_risk_score"] = df["dataset2_fulfillment_risk_score"].clip(0, 1)

    q33 = float(df["dataset2_fulfillment_risk_score"].quantile(1 / 3))
    q66 = float(df["dataset2_fulfillment_risk_score"].quantile(2 / 3))

    df["risk_label"] = np.select(
        [
            df["dataset2_fulfillment_risk_score"] <= q33,
            df["dataset2_fulfillment_risk_score"] <= q66,
            df["dataset2_fulfillment_risk_score"] > q66,
        ],
        [
            0,
            1,
            2,
        ],
        default=1,
    ).astype(int)

    label_map = {
        0: "Low",
        1: "Moderate",
        2: "High",
    }

    df["risk_label_name"] = df["risk_label"].map(label_map)

    # ------------------------------------------------------------------
    # Temporal split
    # ------------------------------------------------------------------

    df = df.sort_values(["order_purchase_timestamp", "order_id", "order_item_id"]).reset_index(drop=True)

    n = len(df)
    train_end = int(n * 0.70)
    valid_end = int(n * 0.85)

    df["temporal_split"] = "test"
    df.loc[: train_end - 1, "temporal_split"] = "train"
    df.loc[train_end: valid_end - 1, "temporal_split"] = "valid"

    after_missing = build_missing_summary(df, "after_feature_engineering_before_final_fill")

    df, final_fill_report = final_fill_all_missing(df)

    final_missing = build_missing_summary(df, "after_final_fill")

    missing_report = pd.concat(
        [
            before_missing,
            after_missing,
            final_missing,
        ],
        axis=0,
        ignore_index=True,
    )

    return df, missing_report, final_fill_report


def build_feature_dictionary(df: pd.DataFrame) -> pd.DataFrame:
    identity_cols = {
        "order_id",
        "order_item_id",
        "product_id",
        "seller_id",
        "customer_id",
        "customer_unique_id",
    }

    date_cols = set(DATE_COLUMNS + ["order_purchase_date"])

    label_cols = {
        "risk_label",
        "risk_label_name",
        "dataset2_fulfillment_risk_score",
    }

    risk_component_cols = set(RISK_WEIGHTS.keys())

    outcome_cols = {
        "is_delivered_status",
        "is_canceled_status",
        "non_delivered_flag",
        "late_delivery_flag",
        "positive_delivery_delay_days",
        "delivery_delay_days",
        "customer_delivery_days",
        "early_delivery_days",
        "low_review_flag",
        "high_review_flag",
    }

    categorical_cols = set(df.select_dtypes(include=["object", "string"]).columns.tolist())

    rows = []

    for col in df.columns:
        if col in identity_cols:
            role = "identifier"
        elif col in date_cols:
            role = "date_or_time"
        elif col in label_cols:
            role = "target_or_label"
        elif col in risk_component_cols:
            role = "risk_component"
        elif col in outcome_cols:
            role = "outcome_derived"
        elif col in categorical_cols:
            role = "categorical_context"
        elif "historical" in col or "prior" in col:
            role = "historical_context_feature"
        elif "geo" in col or "distance" in col or "lat" in col or "lng" in col:
            role = "geospatial_feature"
        elif "payment" in col:
            role = "payment_feature"
        elif "freight" in col or "price" in col or "value" in col:
            role = "cost_freight_feature"
        elif "product" in col:
            role = "product_feature"
        elif "seller" in col:
            role = "seller_feature"
        elif "customer" in col:
            role = "customer_feature"
        else:
            role = "engineered_feature"

        rows.append(
            {
                "column_name": col,
                "dtype": str(df[col].dtype),
                "role": role,
                "missing_count": int(df[col].isna().sum()),
                "unique_values": int(df[col].nunique(dropna=True)),
            }
        )

    return pd.DataFrame(rows)


def build_risk_component_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for component, weight in RISK_WEIGHTS.items():
        series = pd.to_numeric(df[component], errors="coerce")

        rows.append(
            {
                "component": component,
                "weight": weight,
                "mean": float(series.mean()),
                "std": float(series.std()),
                "min": float(series.min()),
                "p25": float(series.quantile(0.25)),
                "median": float(series.median()),
                "p75": float(series.quantile(0.75)),
                "max": float(series.max()),
            }
        )

    score = pd.to_numeric(df["dataset2_fulfillment_risk_score"], errors="coerce")

    rows.append(
        {
            "component": "dataset2_fulfillment_risk_score",
            "weight": 1.0,
            "mean": float(score.mean()),
            "std": float(score.std()),
            "min": float(score.min()),
            "p25": float(score.quantile(0.25)),
            "median": float(score.median()),
            "p75": float(score.quantile(0.75)),
            "max": float(score.max()),
        }
    )

    return pd.DataFrame(rows)


def build_risk_label_distribution(df: pd.DataFrame) -> pd.DataFrame:
    dist = (
        df.groupby(["risk_label", "risk_label_name"], as_index=False)
        .agg(count=("order_id", "count"))
        .sort_values("risk_label")
    )

    dist["percent"] = (dist["count"] / len(df) * 100).round(4)

    return dist


def build_temporal_split_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for split_name, group in df.groupby("temporal_split", sort=False):
        label_counts = group["risk_label_name"].value_counts().to_dict()

        rows.append(
            {
                "split": split_name,
                "rows": int(len(group)),
                "percent": round(len(group) / len(df) * 100, 4),
                "min_purchase_date": group["order_purchase_timestamp"].min(),
                "max_purchase_date": group["order_purchase_timestamp"].max(),
                "low_count": int(label_counts.get("Low", 0)),
                "moderate_count": int(label_counts.get("Moderate", 0)),
                "high_count": int(label_counts.get("High", 0)),
            }
        )

    order = {"train": 0, "valid": 1, "test": 2}

    return (
        pd.DataFrame(rows)
        .assign(order=lambda x: x["split"].map(order))
        .sort_values("order")
        .drop(columns=["order"])
    )


def build_status_delivery_review_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {
            "metric": "total_order_item_rows",
            "value": int(len(df)),
        },
        {
            "metric": "unique_orders",
            "value": int(df["order_id"].nunique()),
        },
        {
            "metric": "unique_sellers",
            "value": int(df["seller_id"].nunique()),
        },
        {
            "metric": "unique_products",
            "value": int(df["product_id"].nunique()),
        },
        {
            "metric": "unique_customers",
            "value": int(df["customer_id"].nunique()),
        },
        {
            "metric": "delivered_status_rate",
            "value": round(float(df["is_delivered_status"].mean()), 6),
        },
        {
            "metric": "late_or_non_delivered_rate",
            "value": round(float(df["late_delivery_flag"].mean()), 6),
        },
        {
            "metric": "low_review_rate",
            "value": round(float(df["low_review_flag"].mean()), 6),
        },
        {
            "metric": "mean_delivery_delay_days",
            "value": round(float(pd.to_numeric(df["delivery_delay_days"], errors="coerce").mean()), 6),
        },
        {
            "metric": "mean_customer_delivery_days",
            "value": round(float(pd.to_numeric(df["customer_delivery_days"], errors="coerce").mean()), 6),
        },
        {
            "metric": "mean_geo_distance_km",
            "value": round(float(pd.to_numeric(df["geo_distance_km"], errors="coerce").mean()), 6),
        },
        {
            "metric": "mean_risk_score",
            "value": round(float(df["dataset2_fulfillment_risk_score"].mean()), 6),
        },
    ]

    return pd.DataFrame(rows)


def build_risk_threshold_report(df: pd.DataFrame) -> pd.DataFrame:
    score = df["dataset2_fulfillment_risk_score"]

    rows = [
        {
            "item": "low_moderate_threshold_q33",
            "value": float(score.quantile(1 / 3)),
        },
        {
            "item": "moderate_high_threshold_q66",
            "value": float(score.quantile(2 / 3)),
        },
    ]

    for component, weight in RISK_WEIGHTS.items():
        rows.append(
            {
                "item": f"weight_{component}",
                "value": weight,
            }
        )

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

def main() -> None:
    ensure_directories()
    reset_log()

    print_header("STEP 17: DATASET 2 PREPROCESSING + FEATURE ENGINEERING + RISK LABELS")
    log(f"[TIME] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"[PROJECT ROOT] {PROJECT_ROOT}")
    log(f"[DATASET 2 RAW DIR] {DATASET2_RAW_DIR}")
    log(f"[DATASET 2 PROCESSED DIR] {DATASET2_PROCESSED_DIR}")
    log(f"[DATASET 2 OUTPUT DIR] {OUTPUTS2_DIR}")

    try:
        print_section("Loading Raw Olist Tables")
        tables, inventory = load_raw_tables()
        save_csv(inventory, TABLES2_DIR / "17_dataset2_input_file_inventory.csv")

        print_section("Integrating Olist Tables")
        integrated_df, shape_report = build_integrated_dataset(tables)
        save_csv(shape_report, TABLES2_DIR / "17_dataset2_merge_shape_report.csv")

        pre_feature_path = DATASET2_PROCESSED_DIR / "17_olist_integrated_order_item_level_prefeature.csv"
        save_csv(integrated_df, pre_feature_path)

        log(f"[INTEGRATED SHAPE] {integrated_df.shape[0]:,} rows | {integrated_df.shape[1]:,} columns")

        print_section("Feature Engineering and Composite Risk Label Construction")
        final_df, missing_report, final_fill_report = engineer_features(integrated_df)

        final_path = DATASET2_PROCESSED_DIR / "17_olist_features_risk_labels.csv"
        split_path = DATASET2_PROCESSED_DIR / "17_olist_temporal_split_features.csv"

        save_csv(final_df, final_path)
        save_csv(final_df, split_path)

        log(f"[FINAL SHAPE] {final_df.shape[0]:,} rows | {final_df.shape[1]:,} columns")
        log(f"[FINAL MISSING VALUES] {int(final_df.isna().sum().sum()):,}")

        print_section("Saving Reports and Tables")

        feature_dictionary = build_feature_dictionary(final_df)
        risk_component_summary = build_risk_component_summary(final_df)
        risk_label_distribution = build_risk_label_distribution(final_df)
        temporal_split_summary = build_temporal_split_summary(final_df)
        status_delivery_review_summary = build_status_delivery_review_summary(final_df)
        risk_threshold_report = build_risk_threshold_report(final_df)

        save_csv(missing_report, TABLES2_DIR / "17_dataset2_missing_values_all_stages.csv")
        save_csv(final_fill_report, TABLES2_DIR / "17_dataset2_final_missing_fill_report.csv")
        save_csv(feature_dictionary, TABLES2_DIR / "17_dataset2_feature_dictionary.csv")
        save_csv(risk_component_summary, TABLES2_DIR / "17_dataset2_risk_component_summary.csv")
        save_csv(risk_label_distribution, TABLES2_DIR / "17_dataset2_risk_label_distribution.csv")
        save_csv(temporal_split_summary, TABLES2_DIR / "17_dataset2_temporal_split_summary.csv")
        save_csv(status_delivery_review_summary, TABLES2_DIR / "17_dataset2_status_delivery_review_summary.csv")
        save_csv(risk_threshold_report, TABLES2_DIR / "17_dataset2_risk_weights_thresholds.csv")

        key_columns = pd.DataFrame(
            {
                "column_group": [
                    "identifier",
                    "identifier",
                    "identifier",
                    "identifier",
                    "identifier",
                    "time",
                    "target",
                    "target",
                    "score",
                    "split",
                    "graph_key",
                    "graph_key",
                    "graph_key",
                    "graph_key",
                ],
                "column_name": [
                    "order_id",
                    "order_item_id",
                    "product_id",
                    "seller_id",
                    "customer_id",
                    "order_purchase_timestamp",
                    "risk_label",
                    "risk_label_name",
                    "dataset2_fulfillment_risk_score",
                    "temporal_split",
                    "seller_id",
                    "product_id",
                    "product_category_name_english",
                    "seller_customer_route",
                ],
                "description": [
                    "Order identifier",
                    "Order item sequence identifier",
                    "Product identifier",
                    "Seller identifier used as supplier proxy",
                    "Customer identifier",
                    "Temporal split and historical feature reference",
                    "Numerical risk class: 0 Low, 1 Moderate, 2 High",
                    "Text risk class",
                    "Composite fulfilment-disruption risk score",
                    "Leakage-aware chronological split",
                    "Seller graph node",
                    "Product graph node",
                    "Category graph node",
                    "Seller-customer state route graph edge context",
                ],
            }
        )

        save_csv(key_columns, TABLES2_DIR / "17_dataset2_key_columns.csv")

        print_section("Dataset 2 Risk Label Distribution")
        log(risk_label_distribution.to_string(index=False))

        print_section("Dataset 2 Temporal Split Summary")
        log(temporal_split_summary.to_string(index=False))

        print_section("Dataset 2 Risk Component Summary")
        log(risk_component_summary.to_string(index=False))

        print_section("Dataset 2 Status, Delivery, Review Summary")
        log(status_delivery_review_summary.to_string(index=False))

        summary = {
            "step": "17_dataset2_preprocessing_feature_engineering",
            "dataset": "Brazilian E-Commerce Public Dataset by Olist",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "project_root": str(PROJECT_ROOT),
            "input_rows_order_items": int(len(tables["order_items"])),
            "integrated_rows": int(len(integrated_df)),
            "integrated_columns": int(integrated_df.shape[1]),
            "final_rows": int(len(final_df)),
            "final_columns": int(final_df.shape[1]),
            "final_missing_values": int(final_df.isna().sum().sum()),
            "risk_weights": RISK_WEIGHTS,
            "risk_label_distribution": risk_label_distribution.to_dict(orient="records"),
            "temporal_split_summary": temporal_split_summary.astype(str).to_dict(orient="records"),
            "saved_files": {
                "pre_feature_dataset": str(pre_feature_path),
                "final_feature_dataset": str(final_path),
                "temporal_split_dataset": str(split_path),
                "merge_shape_report": str(TABLES2_DIR / "17_dataset2_merge_shape_report.csv"),
                "missing_report": str(TABLES2_DIR / "17_dataset2_missing_values_all_stages.csv"),
                "risk_label_distribution": str(TABLES2_DIR / "17_dataset2_risk_label_distribution.csv"),
                "temporal_split_summary": str(TABLES2_DIR / "17_dataset2_temporal_split_summary.csv"),
            },
        }

        save_json(summary, REPORTS2_DIR / "17_dataset2_preprocessing_feature_engineering_summary.json")

        report_lines: List[str] = []
        report_lines.append("STEP 17: DATASET 2 PREPROCESSING + FEATURE ENGINEERING REPORT")
        report_lines.append("=" * 100)
        report_lines.append("Dataset: Brazilian E-Commerce Public Dataset by Olist")
        report_lines.append(f"Timestamp: {summary['timestamp']}")
        report_lines.append(f"Project root: {PROJECT_ROOT}")
        report_lines.append("")
        report_lines.append("Main Output")
        report_lines.append("-" * 100)
        report_lines.append(f"Integrated dataset shape: {integrated_df.shape[0]:,} rows x {integrated_df.shape[1]:,} columns")
        report_lines.append(f"Final feature dataset shape: {final_df.shape[0]:,} rows x {final_df.shape[1]:,} columns")
        report_lines.append(f"Final missing values: {int(final_df.isna().sum().sum()):,}")
        report_lines.append("")
        report_lines.append("Risk Weights")
        report_lines.append("-" * 100)

        for component, weight in RISK_WEIGHTS.items():
            report_lines.append(f"{component}: {weight}")

        report_lines.append("")
        report_lines.append("Risk Label Distribution")
        report_lines.append("-" * 100)
        report_lines.append(risk_label_distribution.to_string(index=False))
        report_lines.append("")
        report_lines.append("Temporal Split Summary")
        report_lines.append("-" * 100)
        report_lines.append(temporal_split_summary.to_string(index=False))
        report_lines.append("")
        report_lines.append("Risk Component Summary")
        report_lines.append("-" * 100)
        report_lines.append(risk_component_summary.to_string(index=False))
        report_lines.append("")
        report_lines.append("Status, Delivery, and Review Summary")
        report_lines.append("-" * 100)
        report_lines.append(status_delivery_review_summary.to_string(index=False))

        save_text(
            "\n".join(report_lines),
            REPORTS2_DIR / "17_dataset2_preprocessing_feature_engineering_report.txt",
        )

        print_section("Step 17 Completed")
        log("[DONE] Dataset 2 preprocessing, feature engineering, risk labels, and temporal split completed successfully.")
        log(f"[FINAL DATASET SAVED] {final_path}")
        log(f"[TABLES SAVED] {TABLES2_DIR}")
        log(f"[REPORTS SAVED] {REPORTS2_DIR}")
        log(f"[LOG SAVED] {LOG_FILE}")
        log("")
        log("NEXT STEP:")
        log("py -3.10 -u .\\scripts\\18_dataset2_graph_risknet.py")

    except Exception as exc:
        print_section("Step 17 Failed")
        log(f"[ERROR] {exc}")
        log(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()