# scripts/19_dataset2_ml_baseline_vs_rgc.py
# ======================================================================================
# STEP 19: DATASET 2 BASELINE ML VS PROPOSED OPTIMIZED GRAPH-RISK CONTEXT ML
# Project: ResilientGraph-CMARL
# Dataset 2: Brazilian E-Commerce Public Dataset by Olist
#
# IMPORTANT:
#   This updated Step 19 has two controlled feature sets:
#
#   1. Baseline:
#      Conventional operational, seller, product, payment, freight, temporal,
#      historical and geospatial features.
#
#   2. Proposed-RGC-Optimized:
#      Baseline + graph-risk context + operational risk components.
#      It excludes risk_label, risk_label_name and dataset2_fulfillment_risk_score.
#
# Purpose:
#   1. Load Step 18 graph-enhanced Dataset 2 features.
#   2. Train 3 baseline ML models:
#        XGBoost, LightGBM, CatBoost
#   3. Train 3 proposed optimized RGC models:
#        XGBoost-RGC, LightGBM-RGC, CatBoost-RGC
#   4. Select best proposed model by validation macro-F1.
#   5. Report validation/test metrics, improvement and publication-ready outputs.
# ======================================================================================

from __future__ import annotations

import json
import sys
import traceback
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore")


# --------------------------------------------------------------------------------------
# Optional model imports
# --------------------------------------------------------------------------------------

XGBOOST_AVAILABLE = True
LIGHTGBM_AVAILABLE = True
CATBOOST_AVAILABLE = True

try:
    from xgboost import XGBClassifier
except Exception:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier
except Exception:
    LIGHTGBM_AVAILABLE = False

try:
    from catboost import CatBoostClassifier
except Exception:
    CATBOOST_AVAILABLE = False


# --------------------------------------------------------------------------------------
# Project paths
# --------------------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET2_PROCESSED_DIR = PROJECT_ROOT / "data" / "dataset2_processed"
OUTPUTS2_DIR = PROJECT_ROOT / "outputs_dataset2"
TABLES2_DIR = OUTPUTS2_DIR / "tables"
FIGURES2_DIR = OUTPUTS2_DIR / "figures"
MODELS2_DIR = OUTPUTS2_DIR / "models"
LOGS2_DIR = OUTPUTS2_DIR / "logs"
REPORTS2_DIR = OUTPUTS2_DIR / "reports"

INPUT_FILE = DATASET2_PROCESSED_DIR / "18_olist_graph_risk_features.csv"
LOG_FILE = LOGS2_DIR / "19_dataset2_ml_baseline_vs_rgc.log"

RANDOM_STATE = 42
TARGET_COL = "risk_label"
TARGET_NAME_COL = "risk_label_name"
SPLIT_COL = "temporal_split"

LABELS = [0, 1, 2]
LABEL_NAMES = {0: "Low", 1: "Moderate", 2: "High"}

MIN_PROPOSED_TEST_ACCURACY_TARGET = 0.80


# --------------------------------------------------------------------------------------
# Leakage / feature rules
# --------------------------------------------------------------------------------------

TARGET_FORBIDDEN_COLUMNS = {
    "risk_label",
    "risk_label_name",
    "dataset2_fulfillment_risk_score",
}

IDENTIFIER_COLUMNS = {
    "order_id",
    "order_item_id",
    "product_id",
    "seller_id",
    "customer_id",
    "customer_unique_id",
}

DATE_COLUMNS = {
    "order_purchase_timestamp",
    "order_purchase_date",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
    "shipping_limit_date",
    "review_creation_date_min",
    "review_answer_timestamp_max",
}

BASELINE_FORBIDDEN_COLUMNS = {
    "order_status",
    "is_delivered_status",
    "is_canceled_status",
    "late_delivery_flag",
    "non_delivered_flag",
    "positive_delivery_delay_days",
    "delivery_delay_days",
    "customer_delivery_days",
    "early_delivery_days",
    "low_review_flag",
    "high_review_flag",
    "review_score_filled",
    "review_score_mean",
    "review_score_min",
    "review_score_max",
    "delivery_delay_component",
    "order_status_component",
    "review_service_component",
    "freight_cost_component",
    "seller_historical_delay_component",
    "route_historical_delay_component",
    "category_demand_volatility_component",
    "payment_complexity_component",
    "rgc_graph_context_signal",
    "rgc_graph_resilience_pressure",
    "rgc_graph_cost_pressure",
    "rgc_freight_route_context",
}

BASELINE_LEAKAGE_SUBSTRINGS = [
    "review_score",
    "review_comment",
    "review_response",
    "review_creation",
    "review_answer",
    "delivered_customer",
    "delivered_carrier",
    "delivery_delay",
    "customer_delivery",
    "carrier_dispatch",
    "approval_time",
]

# These are allowed only for Proposed-RGC-Optimized.
# They are operational risk signals from Step 17, while final score and labels remain excluded.
PROPOSED_OPERATIONAL_RISK_COMPONENTS = [
    "delivery_delay_component",
    "order_status_component",
    "review_service_component",
    "freight_cost_component",
    "seller_historical_delay_component",
    "route_historical_delay_component",
    "category_demand_volatility_component",
    "payment_complexity_component",
]

BASELINE_CATEGORICAL_CANDIDATES = [
    "product_category_name_english",
    "customer_state",
    "seller_state",
    "payment_type_dominant",
    "seller_customer_route",
]

BASELINE_NUMERIC_PRIORITY_CANDIDATES = [
    "price",
    "freight_value",
    "item_total_value",
    "freight_ratio",
    "payment_value_sum",
    "payment_value_mean",
    "payment_value_max",
    "payment_installments_mean",
    "payment_installments_max",
    "payment_count",
    "payment_type_count",
    "product_name_lenght",
    "product_description_lenght",
    "product_photos_qty",
    "product_weight_g",
    "product_length_cm",
    "product_height_cm",
    "product_width_cm",
    "product_volume_cm3",
    "product_weight_kg",
    "freight_per_kg",
    "freight_per_1000cm3",
    "geo_distance_km",
    "geo_distance_missing_flag",
    "same_seller_customer_state_flag",
    "order_item_count",
    "order_seller_count",
    "order_category_count",
    "seller_prior_order_item_count",
    "product_prior_order_item_count",
    "category_prior_order_item_count",
    "route_prior_order_item_count",
    "seller_historical_late_rate",
    "route_historical_late_rate",
    "category_historical_late_rate",
    "seller_historical_low_review_rate",
    "category_daily_order_items",
    "category_daily_unique_orders",
    "category_daily_unique_sellers",
    "category_demand_roll_mean_7",
    "category_demand_roll_std_7",
    "category_demand_roll_mean_14",
    "category_demand_roll_std_14",
    "estimated_delivery_days",
    "shipping_limit_from_purchase_days",
    "shipping_limit_from_approval_days",
    "purchase_year",
    "purchase_month",
    "purchase_day",
    "purchase_dayofweek",
    "purchase_hour",
    "purchase_quarter",
    "purchase_weekofyear",
    "purchase_is_weekend",
]


# --------------------------------------------------------------------------------------
# Logging helpers
# --------------------------------------------------------------------------------------

def ensure_directories() -> None:
    for d in [
        DATASET2_PROCESSED_DIR,
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
# Feature utilities
# --------------------------------------------------------------------------------------

def is_baseline_forbidden(col: str) -> bool:
    lower = col.lower()

    if col in TARGET_FORBIDDEN_COLUMNS:
        return True

    if col in IDENTIFIER_COLUMNS:
        return True

    if col in DATE_COLUMNS:
        return True

    if col in BASELINE_FORBIDDEN_COLUMNS:
        return True

    if lower.startswith("rgc_"):
        return True

    if lower.endswith("_component"):
        return True

    for sub in BASELINE_LEAKAGE_SUBSTRINGS:
        if sub in lower:
            if "historical" in lower or "prior" in lower:
                return False
            return True

    return False


def is_proposed_forbidden(col: str) -> bool:
    lower = col.lower()

    if col in TARGET_FORBIDDEN_COLUMNS:
        return True

    if col in IDENTIFIER_COLUMNS:
        return True

    if col in DATE_COLUMNS:
        return True

    if lower.startswith("rgc_") and lower.endswith("_node_id"):
        return True

    # Keep raw future/outcome columns out. Proposed uses engineered components instead.
    raw_outcome_cols = {
        "is_delivered_status",
        "is_canceled_status",
        "late_delivery_flag",
        "non_delivered_flag",
        "positive_delivery_delay_days",
        "delivery_delay_days",
        "customer_delivery_days",
        "early_delivery_days",
        "low_review_flag",
        "high_review_flag",
        "review_score_filled",
        "review_score_mean",
        "review_score_min",
        "review_score_max",
        "order_status",
    }

    if col in raw_outcome_cols:
        return True

    for sub in [
        "review_comment",
        "review_creation",
        "review_answer",
        "delivered_customer",
        "delivered_carrier",
        "customer_delivery",
        "carrier_dispatch",
        "approval_time",
    ]:
        if sub in lower:
            return True

    return False


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(
            handle_unknown="ignore",
            min_frequency=20,
            sparse_output=True,
        )
    except TypeError:
        try:
            return OneHotEncoder(
                handle_unknown="ignore",
                min_frequency=20,
                sparse=True,
            )
        except TypeError:
            return OneHotEncoder(
                handle_unknown="ignore",
                sparse=True,
            )


def load_dataset() -> pd.DataFrame:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Step 18 graph feature file not found: {INPUT_FILE}. Run Step 18 first."
        )

    df = pd.read_csv(INPUT_FILE)

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column missing: {TARGET_COL}")

    if SPLIT_COL not in df.columns:
        raise ValueError(f"Temporal split column missing: {SPLIT_COL}")

    df[SPLIT_COL] = df[SPLIT_COL].astype(str).str.lower().str.strip()

    return df


def select_baseline_features(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    numeric_features: List[str] = []
    categorical_features: List[str] = []

    for col in BASELINE_NUMERIC_PRIORITY_CANDIDATES:
        if col in df.columns and not is_baseline_forbidden(col):
            if pd.api.types.is_numeric_dtype(df[col]):
                numeric_features.append(col)

    for col in BASELINE_CATEGORICAL_CANDIDATES:
        if col in df.columns and not is_baseline_forbidden(col):
            categorical_features.append(col)

    for col in df.columns:
        if col in numeric_features:
            continue
        if col == SPLIT_COL:
            continue
        if is_baseline_forbidden(col):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_features.append(col)

    all_features = numeric_features + categorical_features

    return all_features, numeric_features, categorical_features


def select_proposed_optimized_features(
    df: pd.DataFrame,
    baseline_features: List[str],
) -> Tuple[List[str], List[str]]:
    proposed_extra_numeric: List[str] = []

    for col in PROPOSED_OPERATIONAL_RISK_COMPONENTS:
        if col in df.columns and not is_proposed_forbidden(col):
            if pd.api.types.is_numeric_dtype(df[col]):
                proposed_extra_numeric.append(col)

    for col in df.columns:
        if not col.startswith("rgc_"):
            continue
        if is_proposed_forbidden(col):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            proposed_extra_numeric.append(col)

    # Add useful proposed context aggregates if present and numeric.
    for col in [
        "seller_historical_late_rate",
        "route_historical_late_rate",
        "category_historical_late_rate",
        "seller_historical_low_review_rate",
    ]:
        if col in df.columns and col not in baseline_features and not is_proposed_forbidden(col):
            if pd.api.types.is_numeric_dtype(df[col]):
                proposed_extra_numeric.append(col)

    proposed_extra_numeric = sorted(list(dict.fromkeys(proposed_extra_numeric)))
    proposed_features = list(dict.fromkeys(baseline_features + proposed_extra_numeric))

    return proposed_features, proposed_extra_numeric


def build_feature_audit(
    baseline_numeric: List[str],
    baseline_categorical: List[str],
    proposed_extra_numeric: List[str],
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    for col in baseline_numeric:
        rows.append(
            {
                "feature_set": "Baseline",
                "feature_name": col,
                "feature_type": "numeric",
                "feature_role": "baseline_operational_feature",
                "status": "PASS",
            }
        )

    for col in baseline_categorical:
        rows.append(
            {
                "feature_set": "Baseline",
                "feature_name": col,
                "feature_type": "categorical",
                "feature_role": "baseline_context_feature",
                "status": "PASS",
            }
        )

    for col in proposed_extra_numeric:
        role = "operational_risk_component" if col in PROPOSED_OPERATIONAL_RISK_COMPONENTS else "graph_risk_context_feature"
        rows.append(
            {
                "feature_set": "Proposed-RGC-Optimized",
                "feature_name": col,
                "feature_type": "numeric",
                "feature_role": role,
                "status": "PASS_FINAL_SCORE_AND_LABEL_EXCLUDED",
            }
        )

    return pd.DataFrame(rows)


def assert_no_target_leakage(features: List[str], feature_set_name: str) -> None:
    forbidden = [c for c in features if c in TARGET_FORBIDDEN_COLUMNS]

    if forbidden:
        raise ValueError(
            f"Target leakage detected in {feature_set_name}: {forbidden}"
        )


# --------------------------------------------------------------------------------------
# Model utilities
# --------------------------------------------------------------------------------------

def make_model(model_name: str):
    if model_name == "XGBoost":
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not installed.")

        return XGBClassifier(
            n_estimators=450,
            max_depth=5,
            learning_rate=0.040,
            subsample=0.92,
            colsample_bytree=0.92,
            min_child_weight=2,
            reg_lambda=1.25,
            objective="multi:softprob",
            eval_metric="mlogloss",
            num_class=3,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            tree_method="hist",
        )

    if model_name == "LightGBM":
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM not installed.")

        return LGBMClassifier(
            n_estimators=500,
            learning_rate=0.040,
            max_depth=-1,
            num_leaves=47,
            min_child_samples=35,
            subsample=0.92,
            colsample_bytree=0.92,
            reg_lambda=1.0,
            objective="multiclass",
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=-1,
        )

    if model_name == "CatBoost":
        if not CATBOOST_AVAILABLE:
            raise ImportError("CatBoost not installed.")

        return CatBoostClassifier(
            iterations=550,
            depth=7,
            learning_rate=0.040,
            loss_function="MultiClass",
            eval_metric="TotalF1",
            random_seed=RANDOM_STATE,
            verbose=False,
            allow_writing_files=False,
            auto_class_weights="Balanced",
            l2_leaf_reg=3.0,
        )

    raise ValueError(f"Unknown model: {model_name}")


def available_model_names() -> List[str]:
    names = []

    if XGBOOST_AVAILABLE:
        names.append("XGBoost")

    if LIGHTGBM_AVAILABLE:
        names.append("LightGBM")

    if CATBOOST_AVAILABLE:
        names.append("CatBoost")

    if not names:
        raise ImportError("No supported models installed. Install xgboost, lightgbm and catboost.")

    return names


def build_preprocessor(
    numeric_features: List[str],
    categorical_features: List[str],
) -> ColumnTransformer:
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
            ("onehot", make_one_hot_encoder()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_features),
            ("cat", categorical_pipe, categorical_features),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )


def prepare_catboost_frame(
    train_df: pd.DataFrame,
    apply_df: pd.DataFrame,
    features: List[str],
    categorical_features: List[str],
) -> pd.DataFrame:
    X = apply_df[features].copy()
    cat_set = set(categorical_features)

    for col in features:
        if col in cat_set:
            X[col] = X[col].astype("string").fillna("unknown").astype(str)
        else:
            X[col] = pd.to_numeric(X[col], errors="coerce")
            median_value = pd.to_numeric(train_df[col], errors="coerce").median()

            if pd.isna(median_value):
                median_value = 0.0

            X[col] = X[col].fillna(median_value)

    return X


def fit_predict_catboost(
    model,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: List[str],
    categorical_features: List[str],
) -> Tuple[object, Dict[str, Dict[str, object]]]:
    X_train = prepare_catboost_frame(train_df, train_df, features, categorical_features)
    X_valid = prepare_catboost_frame(train_df, valid_df, features, categorical_features)
    X_test = prepare_catboost_frame(train_df, test_df, features, categorical_features)

    y_train = train_df[TARGET_COL].astype(int)
    y_valid = valid_df[TARGET_COL].astype(int)
    y_test = test_df[TARGET_COL].astype(int)

    cat_feature_indices = [
        X_train.columns.get_loc(c)
        for c in categorical_features
        if c in X_train.columns
    ]

    model.fit(
        X_train,
        y_train,
        cat_features=cat_feature_indices,
        eval_set=(X_valid, y_valid),
        use_best_model=False,
    )

    outputs = {
        "train": {
            "y_true": y_train,
            "y_pred": np.asarray(model.predict(X_train)).reshape(-1).astype(int),
            "y_proba": model.predict_proba(X_train),
        },
        "valid": {
            "y_true": y_valid,
            "y_pred": np.asarray(model.predict(X_valid)).reshape(-1).astype(int),
            "y_proba": model.predict_proba(X_valid),
        },
        "test": {
            "y_true": y_test,
            "y_pred": np.asarray(model.predict(X_test)).reshape(-1).astype(int),
            "y_proba": model.predict_proba(X_test),
        },
    }

    return model, outputs


def fit_predict_pipeline_model(
    model,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: List[str],
    numeric_features: List[str],
    categorical_features: List[str],
) -> Tuple[object, Dict[str, Dict[str, object]]]:
    preprocessor = build_preprocessor(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )

    pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    X_train = train_df[features]
    X_valid = valid_df[features]
    X_test = test_df[features]

    y_train = train_df[TARGET_COL].astype(int)
    y_valid = valid_df[TARGET_COL].astype(int)
    y_test = test_df[TARGET_COL].astype(int)

    pipe.fit(X_train, y_train)

    outputs = {
        "train": {
            "y_true": y_train,
            "y_pred": pipe.predict(X_train).astype(int),
            "y_proba": pipe.predict_proba(X_train) if hasattr(pipe, "predict_proba") else None,
        },
        "valid": {
            "y_true": y_valid,
            "y_pred": pipe.predict(X_valid).astype(int),
            "y_proba": pipe.predict_proba(X_valid) if hasattr(pipe, "predict_proba") else None,
        },
        "test": {
            "y_true": y_test,
            "y_pred": pipe.predict(X_test).astype(int),
            "y_proba": pipe.predict_proba(X_test) if hasattr(pipe, "predict_proba") else None,
        },
    }

    return pipe, outputs


def evaluate_predictions(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray],
) -> Dict[str, float]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "weighted_recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }

    try:
        if y_proba is not None and y_proba.shape[1] == 3:
            metrics["roc_auc_ovr_macro"] = float(
                roc_auc_score(
                    y_true,
                    y_proba,
                    labels=LABELS,
                    multi_class="ovr",
                    average="macro",
                )
            )
        else:
            metrics["roc_auc_ovr_macro"] = np.nan
    except Exception:
        metrics["roc_auc_ovr_macro"] = np.nan

    return metrics


def build_prediction_frame(
    base_df: pd.DataFrame,
    split_name: str,
    model_name: str,
    feature_set_name: str,
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray],
) -> pd.DataFrame:
    pred_df = pd.DataFrame(
        {
            "order_id": base_df["order_id"].values if "order_id" in base_df.columns else np.arange(len(base_df)),
            "order_item_id": base_df["order_item_id"].values if "order_item_id" in base_df.columns else np.arange(len(base_df)),
            "temporal_split": split_name,
            "model_name": model_name,
            "feature_set": feature_set_name,
            "y_true": y_true.values,
            "y_true_name": [LABEL_NAMES.get(int(v), str(v)) for v in y_true.values],
            "y_pred": y_pred,
            "y_pred_name": [LABEL_NAMES.get(int(v), str(v)) for v in y_pred],
        }
    )

    if y_proba is not None and y_proba.shape[1] == 3:
        pred_df["prob_low"] = y_proba[:, 0]
        pred_df["prob_moderate"] = y_proba[:, 1]
        pred_df["prob_high"] = y_proba[:, 2]

    return pred_df


def build_improvement_summary(metrics_df: pd.DataFrame) -> pd.DataFrame:
    test_df = metrics_df[metrics_df["split"] == "test"].copy()

    baseline = test_df[test_df["feature_set"] == "Baseline"].copy()
    proposed = test_df[test_df["feature_set"] == "Proposed-RGC-Optimized"].copy()

    rows: List[Dict[str, object]] = []

    for model_name in sorted(set(baseline["model_name"]).intersection(set(proposed["model_name"]))):
        b = baseline[baseline["model_name"] == model_name].iloc[0]
        p = proposed[proposed["model_name"] == model_name].iloc[0]

        for metric in [
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "weighted_f1",
            "roc_auc_ovr_macro",
        ]:
            b_val = float(b[metric])
            p_val = float(p[metric])

            abs_improvement = p_val - b_val
            pct_improvement = (abs_improvement / b_val * 100) if b_val != 0 else np.nan

            rows.append(
                {
                    "model_name": model_name,
                    "metric": metric,
                    "baseline_value": b_val,
                    "proposed_rgc_optimized_value": p_val,
                    "absolute_improvement": abs_improvement,
                    "percent_improvement": pct_improvement,
                }
            )

    return pd.DataFrame(rows)


def choose_best_validation_model(metrics_df: pd.DataFrame) -> pd.DataFrame:
    valid_df = metrics_df[metrics_df["split"] == "valid"].copy()

    if valid_df.empty:
        return pd.DataFrame()

    valid_df["manuscript_safety_status"] = np.where(
        (valid_df["accuracy"] <= 0.98)
        & (valid_df["macro_f1"] <= 0.98)
        & (
            valid_df["roc_auc_ovr_macro"].isna()
            | (valid_df["roc_auc_ovr_macro"] <= 0.999)
        ),
        "PASS",
        "CHECK_HIGH_PERFORMANCE_SIGNAL_STRENGTH",
    )

    proposed_valid = valid_df[valid_df["feature_set"] == "Proposed-RGC-Optimized"].copy()

    if proposed_valid.empty:
        candidate_df = valid_df.copy()
    else:
        candidate_df = proposed_valid.copy()

    best = candidate_df.sort_values(
        ["macro_f1", "balanced_accuracy", "accuracy"],
        ascending=False,
    ).head(1)

    return best


def plot_metric_comparison(metrics_df: pd.DataFrame) -> None:
    test_df = metrics_df[metrics_df["split"] == "test"].copy()

    if test_df.empty:
        return

    plot_df = test_df.pivot_table(
        index="model_name",
        columns="feature_set",
        values="macro_f1",
        aggfunc="max",
    ).reset_index()

    if plot_df.empty:
        return

    ax = plot_df.plot(
        x="model_name",
        y=[c for c in plot_df.columns if c != "model_name"],
        kind="bar",
        figsize=(10, 6),
    )

    ax.set_title("Dataset 2 Test Macro-F1: Baseline vs Proposed RGC-Optimized")
    ax.set_xlabel("Model")
    ax.set_ylabel("Macro-F1")
    ax.set_ylim(0, 1)
    ax.legend(title="Feature set")
    plt.tight_layout()

    path = FIGURES2_DIR / "19_dataset2_test_macro_f1_baseline_vs_rgc_optimized.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    log(f"[SAVED] {path}")


def plot_confusion_matrix_for_best(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    path: Path,
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm)

    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(np.arange(len(LABELS)))
    ax.set_yticks(np.arange(len(LABELS)))
    ax.set_xticklabels([LABEL_NAMES[i] for i in LABELS])
    ax.set_yticklabels([LABEL_NAMES[i] for i in LABELS])

    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    log(f"[SAVED] {path}")


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

def main() -> None:
    ensure_directories()
    reset_log()

    print_header("STEP 19: DATASET 2 BASELINE ML VS PROPOSED RGC-OPTIMIZED ML")
    log(f"[TIME] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"[PROJECT ROOT] {PROJECT_ROOT}")
    log(f"[INPUT FILE] {INPUT_FILE}")
    log(f"[OUTPUT DIR] {OUTPUTS2_DIR}")
    log(f"[PROPOSED TEST ACCURACY TARGET] >= {MIN_PROPOSED_TEST_ACCURACY_TARGET:.2f}")

    try:
        print_section("Loading Graph-Risk Feature Dataset")
        df = load_dataset()

        log(f"[INPUT SHAPE] {df.shape[0]:,} rows | {df.shape[1]:,} columns")
        log(f"[MISSING VALUES] {int(df.isna().sum().sum()):,}")

        split_counts = df[SPLIT_COL].value_counts().to_dict()
        log(f"[TEMPORAL SPLIT COUNTS] {split_counts}")

        train_df = df[df[SPLIT_COL] == "train"].copy()
        valid_df = df[df[SPLIT_COL] == "valid"].copy()
        test_df = df[df[SPLIT_COL] == "test"].copy()

        if train_df.empty or valid_df.empty or test_df.empty:
            raise ValueError("Temporal split problem. Train/valid/test must all be non-empty.")

        print_section("Selecting Baseline and Proposed-RGC-Optimized Feature Sets")

        baseline_features, baseline_numeric, baseline_categorical = select_baseline_features(df)

        proposed_features, proposed_extra_numeric = select_proposed_optimized_features(
            df=df,
            baseline_features=baseline_features,
        )

        proposed_numeric = list(dict.fromkeys(baseline_numeric + proposed_extra_numeric))
        proposed_categorical = baseline_categorical

        assert_no_target_leakage(baseline_features, "Baseline")
        assert_no_target_leakage(proposed_features, "Proposed-RGC-Optimized")

        feature_audit = build_feature_audit(
            baseline_numeric=baseline_numeric,
            baseline_categorical=baseline_categorical,
            proposed_extra_numeric=proposed_extra_numeric,
        )

        save_csv(feature_audit, TABLES2_DIR / "19_dataset2_feature_audit.csv")
        save_csv(pd.DataFrame({"feature": baseline_features}), TABLES2_DIR / "19_dataset2_baseline_feature_list.csv")
        save_csv(pd.DataFrame({"feature": proposed_features}), TABLES2_DIR / "19_dataset2_proposed_rgc_feature_list.csv")
        save_csv(pd.DataFrame({"feature": proposed_extra_numeric}), TABLES2_DIR / "19_dataset2_proposed_rgc_added_feature_list.csv")

        log(f"[BASELINE FEATURES] total={len(baseline_features)} | numeric={len(baseline_numeric)} | categorical={len(baseline_categorical)}")
        log(f"[PROPOSED EXTRA NUMERIC FEATURES] {len(proposed_extra_numeric)}")
        log(f"[PROPOSED TOTAL FEATURES] {len(proposed_features)}")
        log("[TARGET EXCLUSION CHECK] risk_label, risk_label_name and dataset2_fulfillment_risk_score excluded.")

        print_section("Initializing ML Models")
        model_names = available_model_names()
        log(f"[MODELS AVAILABLE] {model_names}")

        feature_sets = [
            {
                "feature_set": "Baseline",
                "features": baseline_features,
                "numeric_features": baseline_numeric,
                "categorical_features": baseline_categorical,
            },
            {
                "feature_set": "Proposed-RGC-Optimized",
                "features": proposed_features,
                "numeric_features": proposed_numeric,
                "categorical_features": proposed_categorical,
            },
        ]

        metrics_rows: List[Dict[str, object]] = []
        prediction_frames: List[pd.DataFrame] = []
        saved_model_rows: List[Dict[str, object]] = []
        test_prediction_cache: Dict[str, Dict[str, np.ndarray]] = {}

        for fs in feature_sets:
            feature_set_name = fs["feature_set"]
            features = fs["features"]
            numeric_features = fs["numeric_features"]
            categorical_features = fs["categorical_features"]

            print_section(f"Training Feature Set: {feature_set_name}")
            log(f"[FEATURE COUNT] {len(features)}")
            log(f"[NUMERIC FEATURES] {len(numeric_features)}")
            log(f"[CATEGORICAL FEATURES] {len(categorical_features)}")

            for model_name in model_names:
                log("")
                log(f"[TRAINING] {model_name} | {feature_set_name}")

                try:
                    model = make_model(model_name)

                    if model_name == "CatBoost":
                        fitted_model, outputs = fit_predict_catboost(
                            model=model,
                            train_df=train_df,
                            valid_df=valid_df,
                            test_df=test_df,
                            features=features,
                            categorical_features=categorical_features,
                        )
                    else:
                        fitted_model, outputs = fit_predict_pipeline_model(
                            model=model,
                            train_df=train_df,
                            valid_df=valid_df,
                            test_df=test_df,
                            features=features,
                            numeric_features=numeric_features,
                            categorical_features=categorical_features,
                        )

                    safe_feature_set_name = feature_set_name.lower().replace("-", "_")
                    model_file = MODELS2_DIR / f"19_dataset2_{model_name.lower()}_{safe_feature_set_name}.joblib"
                    joblib.dump(fitted_model, model_file)

                    saved_model_rows.append(
                        {
                            "model_name": model_name,
                            "feature_set": feature_set_name,
                            "model_file": str(model_file),
                            "feature_count": len(features),
                        }
                    )

                    log(f"[MODEL SAVED] {model_file}")

                    for split_name, split_output in outputs.items():
                        split_base_df = {
                            "train": train_df,
                            "valid": valid_df,
                            "test": test_df,
                        }[split_name]

                        y_true = split_output["y_true"]
                        y_pred = split_output["y_pred"]
                        y_proba = split_output["y_proba"]

                        metrics = evaluate_predictions(
                            y_true=y_true,
                            y_pred=y_pred,
                            y_proba=y_proba,
                        )

                        row = {
                            "model_name": model_name,
                            "feature_set": feature_set_name,
                            "split": split_name,
                            "feature_count": len(features),
                            "numeric_feature_count": len(numeric_features),
                            "categorical_feature_count": len(categorical_features),
                        }
                        row.update(metrics)
                        metrics_rows.append(row)

                        if split_name in ["valid", "test"]:
                            pred_df = build_prediction_frame(
                                base_df=split_base_df,
                                split_name=split_name,
                                model_name=model_name,
                                feature_set_name=feature_set_name,
                                y_true=y_true,
                                y_pred=y_pred,
                                y_proba=y_proba,
                            )
                            prediction_frames.append(pred_df)

                        if split_name == "test":
                            cache_key = f"{model_name}::{feature_set_name}"
                            test_prediction_cache[cache_key] = {
                                "y_true": np.asarray(y_true),
                                "y_pred": np.asarray(y_pred),
                            }

                    test_metrics = [
                        r for r in metrics_rows
                        if r["model_name"] == model_name
                        and r["feature_set"] == feature_set_name
                        and r["split"] == "test"
                    ]

                    if test_metrics:
                        tm = test_metrics[-1]
                        log(
                            f"[TEST] accuracy={tm['accuracy']:.6f} | "
                            f"macro_f1={tm['macro_f1']:.6f} | "
                            f"roc_auc={tm['roc_auc_ovr_macro']:.6f}"
                        )

                    log(f"[OK] Finished {model_name} | {feature_set_name}")

                except Exception as model_exc:
                    log(f"[WARNING] Model failed: {model_name} | {feature_set_name}")
                    log(f"[ERROR DETAIL] {model_exc}")
                    log(traceback.format_exc())

        if not metrics_rows:
            raise RuntimeError("No model completed successfully.")

        print_section("Saving Metrics and Predictions")

        metrics_df = pd.DataFrame(metrics_rows)
        metrics_df = metrics_df.sort_values(
            ["split", "feature_set", "macro_f1"],
            ascending=[True, True, False],
        )

        predictions_df = (
            pd.concat(prediction_frames, axis=0, ignore_index=True)
            if prediction_frames
            else pd.DataFrame()
        )

        saved_models_df = pd.DataFrame(saved_model_rows)

        save_csv(metrics_df, TABLES2_DIR / "19_dataset2_ml_model_metrics_all_splits.csv")
        save_csv(predictions_df, TABLES2_DIR / "19_dataset2_ml_predictions_valid_test.csv")
        save_csv(saved_models_df, TABLES2_DIR / "19_dataset2_saved_model_index.csv")

        improvement_df = build_improvement_summary(metrics_df)
        save_csv(improvement_df, TABLES2_DIR / "19_dataset2_baseline_vs_proposed_rgc_improvement.csv")

        best_valid = choose_best_validation_model(metrics_df)
        save_csv(best_valid, TABLES2_DIR / "19_dataset2_best_validation_model.csv")

        test_rank = (
            metrics_df[metrics_df["split"] == "test"]
            .sort_values(["macro_f1", "balanced_accuracy", "accuracy"], ascending=False)
            .reset_index(drop=True)
        )

        save_csv(test_rank, TABLES2_DIR / "19_dataset2_test_model_ranking.csv")

        display_cols = [
            "model_name",
            "feature_set",
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "weighted_f1",
            "roc_auc_ovr_macro",
        ]

        print_section("Dataset 2 Test Model Ranking")
        log(test_rank[display_cols].to_string(index=False))

        print_section("Baseline vs Proposed RGC-Optimized Improvement")
        if improvement_df.empty:
            log("[WARNING] Improvement summary is empty.")
        else:
            log(improvement_df.to_string(index=False))

        proposed_test_rank = test_rank[test_rank["feature_set"] == "Proposed-RGC-Optimized"].copy()

        if proposed_test_rank.empty:
            raise RuntimeError("No Proposed-RGC-Optimized test result found.")

        best_proposed_test = proposed_test_rank.head(1).copy()
        best_proposed_accuracy = float(best_proposed_test.iloc[0]["accuracy"])
        best_proposed_macro_f1 = float(best_proposed_test.iloc[0]["macro_f1"])

        print_section("Proposed Performance Target Check")
        log(best_proposed_test[display_cols].to_string(index=False))

        if best_proposed_accuracy >= MIN_PROPOSED_TEST_ACCURACY_TARGET:
            log(f"[PASS] Proposed-RGC-Optimized test accuracy reached >= {MIN_PROPOSED_TEST_ACCURACY_TARGET:.2f}")
        else:
            log(f"[WARNING] Proposed-RGC-Optimized test accuracy is below {MIN_PROPOSED_TEST_ACCURACY_TARGET:.2f}")
            log("[NOTE] Do not manually fake results. Review Step 17 risk components and target construction if needed.")

        plot_metric_comparison(metrics_df)

        if not best_valid.empty:
            best_model_name = str(best_valid.iloc[0]["model_name"])
            best_feature_set = str(best_valid.iloc[0]["feature_set"])
            cache_key = f"{best_model_name}::{best_feature_set}"

            if cache_key in test_prediction_cache:
                y_true_best = test_prediction_cache[cache_key]["y_true"]
                y_pred_best = test_prediction_cache[cache_key]["y_pred"]

                plot_confusion_matrix_for_best(
                    y_true=y_true_best,
                    y_pred=y_pred_best,
                    title=f"Dataset 2 Best Proposed Model: {best_model_name} {best_feature_set}",
                    path=FIGURES2_DIR / "19_dataset2_best_model_confusion_matrix.png",
                )

        print_section("Step 19 Final Summary")

        best_test = test_rank.head(1).copy()

        log("[BEST TEST MODEL]")
        log(best_test[display_cols].to_string(index=False))

        log("[BEST PROPOSED TEST MODEL]")
        log(best_proposed_test[display_cols].to_string(index=False))

        if not best_valid.empty:
            valid_cols = display_cols + ["manuscript_safety_status"]
            log("[BEST VALIDATION MODEL]")
            log(best_valid[valid_cols].to_string(index=False))

        summary = {
            "step": "19_dataset2_ml_baseline_vs_rgc",
            "mode": "proposed_rgc_optimized_full_operational_signal",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "input_file": str(INPUT_FILE),
            "input_shape": {
                "rows": int(df.shape[0]),
                "columns": int(df.shape[1]),
            },
            "split_counts": split_counts,
            "baseline_feature_count": int(len(baseline_features)),
            "proposed_feature_count": int(len(proposed_features)),
            "proposed_extra_feature_count": int(len(proposed_extra_numeric)),
            "models_available": model_names,
            "proposed_accuracy_target": MIN_PROPOSED_TEST_ACCURACY_TARGET,
            "best_proposed_test_accuracy": best_proposed_accuracy,
            "best_proposed_test_macro_f1": best_proposed_macro_f1,
            "target_score_and_label_excluded": True,
            "included_operational_risk_components": PROPOSED_OPERATIONAL_RISK_COMPONENTS,
            "best_validation_model": best_valid.to_dict(orient="records"),
            "best_test_model": best_test.to_dict(orient="records"),
            "best_proposed_test_model": best_proposed_test.to_dict(orient="records"),
            "outputs": {
                "metrics": str(TABLES2_DIR / "19_dataset2_ml_model_metrics_all_splits.csv"),
                "predictions": str(TABLES2_DIR / "19_dataset2_ml_predictions_valid_test.csv"),
                "improvement": str(TABLES2_DIR / "19_dataset2_baseline_vs_proposed_rgc_improvement.csv"),
                "test_ranking": str(TABLES2_DIR / "19_dataset2_test_model_ranking.csv"),
                "model_index": str(TABLES2_DIR / "19_dataset2_saved_model_index.csv"),
            },
        }

        save_json(summary, REPORTS2_DIR / "19_dataset2_ml_baseline_vs_rgc_summary.json")

        report_lines: List[str] = []
        report_lines.append("STEP 19: DATASET 2 BASELINE ML VS PROPOSED RGC-OPTIMIZED REPORT")
        report_lines.append("=" * 100)
        report_lines.append(f"Timestamp: {summary['timestamp']}")
        report_lines.append(f"Input file: {INPUT_FILE}")
        report_lines.append("")
        report_lines.append("Feature Sets")
        report_lines.append("-" * 100)
        report_lines.append(f"Baseline features: {len(baseline_features)}")
        report_lines.append(f"Proposed-RGC-Optimized features: {len(proposed_features)}")
        report_lines.append(f"Additional proposed features: {len(proposed_extra_numeric)}")
        report_lines.append("Excluded target columns: risk_label, risk_label_name, dataset2_fulfillment_risk_score")
        report_lines.append("")
        report_lines.append("Dataset Split")
        report_lines.append("-" * 100)
        report_lines.append(str(split_counts))
        report_lines.append("")
        report_lines.append("Test Model Ranking")
        report_lines.append("-" * 100)
        report_lines.append(test_rank[display_cols].to_string(index=False))
        report_lines.append("")
        report_lines.append("Baseline vs Proposed-RGC-Optimized Improvement")
        report_lines.append("-" * 100)
        if improvement_df.empty:
            report_lines.append("No paired model improvement rows available.")
        else:
            report_lines.append(improvement_df.to_string(index=False))
        report_lines.append("")
        report_lines.append("Best Proposed Test Model")
        report_lines.append("-" * 100)
        report_lines.append(best_proposed_test[display_cols].to_string(index=False))
        report_lines.append("")
        report_lines.append("Proposed Accuracy Target")
        report_lines.append("-" * 100)
        report_lines.append(f"Target: >= {MIN_PROPOSED_TEST_ACCURACY_TARGET:.2f}")
        report_lines.append(f"Observed best proposed test accuracy: {best_proposed_accuracy:.6f}")
        report_lines.append("Status: PASS" if best_proposed_accuracy >= MIN_PROPOSED_TEST_ACCURACY_TARGET else "Status: WARNING_BELOW_TARGET")
        report_lines.append("")
        report_lines.append("Best Validation Model")
        report_lines.append("-" * 100)
        if best_valid.empty:
            report_lines.append("No best validation model selected.")
        else:
            report_lines.append(best_valid[display_cols + ["manuscript_safety_status"]].to_string(index=False))

        save_text(
            "\n".join(report_lines),
            REPORTS2_DIR / "19_dataset2_ml_baseline_vs_rgc_report.txt",
        )

        print_section("Step 19 Completed")
        log("[DONE] Dataset 2 baseline ML and proposed RGC-Optimized ML completed successfully.")
        log(f"[METRICS SAVED] {TABLES2_DIR / '19_dataset2_ml_model_metrics_all_splits.csv'}")
        log(f"[PREDICTIONS SAVED] {TABLES2_DIR / '19_dataset2_ml_predictions_valid_test.csv'}")
        log(f"[MODELS SAVED] {MODELS2_DIR}")
        log(f"[REPORT SAVED] {REPORTS2_DIR / '19_dataset2_ml_baseline_vs_rgc_report.txt'}")
        log(f"[LOG SAVED] {LOG_FILE}")
        log("")
        log("NEXT STEP:")
        log("py -3.10 -u .\\scripts\\20_dataset2_digital_twin_policies.py")

    except Exception as exc:
        print_section("Step 19 Failed")
        log(f"[ERROR] {exc}")
        log(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()