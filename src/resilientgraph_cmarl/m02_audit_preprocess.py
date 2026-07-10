from typing import Dict, List, Tuple

import pandas as pd


PII_AND_IRRELEVANT_COLUMNS = [
    "Customer Email",
    "Customer Fname",
    "Customer Lname",
    "Customer Password",
    "Customer Street",
    "Product Image",
    "Product Description",
]

HIGH_MISSING_COLUMNS = [
    "Order Zipcode",
]

TEXT_STANDARDIZE_COLUMNS = [
    "Type",
    "Delivery Status",
    "Category Name",
    "Customer City",
    "Customer Country",
    "Customer Segment",
    "Customer State",
    "Department Name",
    "Market",
    "Order City",
    "Order Country",
    "Order Region",
    "Order State",
    "Order Status",
    "Product Name",
    "Shipping Mode",
]


DATE_COLUMNS = [
    "order date (DateOrders)",
    "shipping date (DateOrders)",
]


def drop_unnecessary_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    removed_rows = []

    candidate_columns = PII_AND_IRRELEVANT_COLUMNS + HIGH_MISSING_COLUMNS

    for col in candidate_columns:
        if col in df.columns:
            missing_count = int(df[col].isna().sum())
            missing_percent = round(float(df[col].isna().mean() * 100), 4)
            removed_rows.append({
                "column": col,
                "reason": "PII/irrelevant/high-missing column",
                "missing_count": missing_count,
                "missing_percent": missing_percent,
            })
            df = df.drop(columns=[col])

    removed_report = pd.DataFrame(removed_rows)
    return df, removed_report


def parse_dataco_dates(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    rows = []

    for col in DATE_COLUMNS:
        if col in df.columns:
            before_missing = int(df[col].isna().sum())
            parsed = pd.to_datetime(df[col], errors="coerce")
            after_missing = int(parsed.isna().sum())

            df[col] = parsed

            rows.append({
                "date_column": col,
                "missing_before": before_missing,
                "missing_after_parse": after_missing,
                "min_date": parsed.min(),
                "max_date": parsed.max(),
            })

    report = pd.DataFrame(rows)
    return df, report


def standardize_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in TEXT_STANDARDIZE_COLUMNS:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .str.replace(r"\s+", " ", regex=True)
            )

    return df


def create_basic_operational_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    order_date_col = "order date (DateOrders)"
    ship_date_col = "shipping date (DateOrders)"

    if order_date_col in df.columns:
        df["order_year"] = df[order_date_col].dt.year
        df["order_month"] = df[order_date_col].dt.month
        df["order_week"] = df[order_date_col].dt.isocalendar().week.astype(int)
        df["order_day"] = df[order_date_col].dt.day
        df["order_dayofweek"] = df[order_date_col].dt.dayofweek
        df["order_quarter"] = df[order_date_col].dt.quarter

    if order_date_col in df.columns and ship_date_col in df.columns:
        df["computed_shipping_days"] = (
            df[ship_date_col] - df[order_date_col]
        ).dt.days

    if "Days for shipping (real)" in df.columns and "Days for shipment (scheduled)" in df.columns:
        df["shipping_delay_gap"] = (
            df["Days for shipping (real)"] - df["Days for shipment (scheduled)"]
        )
        df["is_delayed_by_days"] = (df["shipping_delay_gap"] > 0).astype(int)

    if "Sales" in df.columns and "Order Item Quantity" in df.columns:
        df["sales_per_unit"] = df["Sales"] / df["Order Item Quantity"].replace(0, pd.NA)
        df["sales_per_unit"] = df["sales_per_unit"].fillna(0)

    if "Order Profit Per Order" in df.columns and "Sales" in df.columns:
        df["profit_margin"] = df["Order Profit Per Order"] / df["Sales"].replace(0, pd.NA)
        df["profit_margin"] = df["profit_margin"].replace([float("inf"), -float("inf")], 0)
        df["profit_margin"] = df["profit_margin"].fillna(0)

    return df


def fill_remaining_missing_values(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    rows = []

    for col in df.columns:
        missing_before = int(df[col].isna().sum())
        if missing_before == 0:
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            fill_value = df[col].median()
            df[col] = df[col].fillna(fill_value)
            method = "numeric_median"
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            method = "datetime_left_as_missing"
        else:
            fill_value = "Unknown"
            df[col] = df[col].fillna(fill_value)
            method = "categorical_unknown"

        missing_after = int(df[col].isna().sum())

        rows.append({
            "column": col,
            "missing_before": missing_before,
            "missing_after": missing_after,
            "fill_method": method,
        })

    report = pd.DataFrame(rows)
    return df, report


def build_column_role_report(df: pd.DataFrame) -> pd.DataFrame:
    roles = []

    risk_source_cols = [
        "Late_delivery_risk",
        "Delivery Status",
        "Days for shipping (real)",
        "Days for shipment (scheduled)",
        "shipping_delay_gap",
        "is_delayed_by_days",
    ]

    date_cols = [
        "order date (DateOrders)",
        "shipping date (DateOrders)",
    ]

    id_cols = [c for c in df.columns if "id" in c.lower()]

    for col in df.columns:
        if col in risk_source_cols:
            role = "risk_source_keep_for_step04"
        elif col in date_cols:
            role = "temporal_source"
        elif col in id_cols:
            role = "identifier_or_entity_key"
        elif pd.api.types.is_numeric_dtype(df[col]):
            role = "numeric_feature_candidate"
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            role = "datetime_feature_candidate"
        else:
            role = "categorical_feature_candidate"

        roles.append({
            "column": col,
            "dtype": str(df[col].dtype),
            "role": role,
            "missing_count": int(df[col].isna().sum()),
            "unique_values": int(df[col].nunique(dropna=True)),
        })

    return pd.DataFrame(roles)


def preprocess_dataco(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    cleaned = df.copy()

    cleaned, removed_report = drop_unnecessary_columns(cleaned)
    cleaned, date_parse_report = parse_dataco_dates(cleaned)
    cleaned = standardize_text_columns(cleaned)
    cleaned = create_basic_operational_features(cleaned)
    cleaned, missing_fill_report = fill_remaining_missing_values(cleaned)
    column_role_report = build_column_role_report(cleaned)

    return {
        "cleaned_df": cleaned,
        "removed_columns_report": removed_report,
        "date_parse_report": date_parse_report,
        "missing_fill_report": missing_fill_report,
        "column_role_report": column_role_report,
    }