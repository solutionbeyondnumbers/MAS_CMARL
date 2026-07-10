from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler


TARGET_COL = "risk_label"
DATE_COL = "order date (DateOrders)"


DIRECT_TARGET_AND_LEAKAGE_COLUMNS = [
    "risk_label",
    "risk_label_name",
    "composite_disruption_risk_score",
    "delay_risk",
    "late_delivery_component",
    "demand_volatility_risk",
    "anomaly_risk",
    "shortage_exposure_risk",
    "profit_loss_risk",
    "shipping_risk",
    "shipping_mode_historical_late_rate",
    "Late_delivery_risk",
    "Delivery Status",
    "Order Status",
    "Days for shipping (real)",
    "Days for shipment (scheduled)",
    "shipping date (DateOrders)",
    "computed_shipping_days",
    "shipping_delay_gap",
    "is_delayed_by_days",
]


POST_OUTCOME_PROFIT_COLUMNS = [
    "Benefit per order",
    "Order Profit Per Order",
    "Order Item Profit Ratio",
    "profit_margin",
]


IDENTIFIER_AND_PRIVACY_COLUMNS = [
    "Order Id",
    "Order Item Id",
    "Customer Id",
    "Order Customer Id",
    "Category Id",
    "Department Id",
    "Product Card Id",
    "Product Category Id",
    "Order Item Cardprod Id",
    "Customer Zipcode",
    "Latitude",
    "Longitude",
]


RAW_DATE_COLUMNS = [
    "order date (DateOrders)",
    "shipping date (DateOrders)",
]


GRAPH_PREFIXES = [
    "node_",
    "graph_",
    "sthg_",
]


def import_baseline_model(model_name: str, random_state: int = 42):
    if model_name == "XGBoost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=250,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.90,
            colsample_bytree=0.90,
            objective="multi:softprob",
            eval_metric="mlogloss",
            num_class=3,
            random_state=random_state,
            n_jobs=-1,
            tree_method="hist",
        )

    if model_name == "LightGBM":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            n_estimators=350,
            learning_rate=0.05,
            max_depth=-1,
            num_leaves=64,
            subsample=0.90,
            colsample_bytree=0.90,
            objective="multiclass",
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        )

    if model_name == "CatBoost":
        from catboost import CatBoostClassifier

        return CatBoostClassifier(
            iterations=350,
            depth=8,
            learning_rate=0.05,
            loss_function="MultiClass",
            random_seed=random_state,
            verbose=False,
            allow_writing_files=False,
        )

    raise ValueError(f"Unsupported model name: {model_name}")


def make_temporal_split(
    df: pd.DataFrame,
    date_col: str = DATE_COL,
    train_ratio: float = 0.70,
    valid_ratio: float = 0.15,
) -> Dict[str, pd.DataFrame]:
    df = df.copy()

    if date_col not in df.columns:
        raise KeyError(f"Date column not found for temporal split: {date_col}")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)

    n = len(df)
    train_end = int(n * train_ratio)
    valid_end = int(n * (train_ratio + valid_ratio))

    return {
        "train": df.iloc[:train_end].copy(),
        "valid": df.iloc[train_end:valid_end].copy(),
        "test": df.iloc[valid_end:].copy(),
    }


def should_drop_baseline_column(col: str) -> bool:
    if col in DIRECT_TARGET_AND_LEAKAGE_COLUMNS:
        return True

    if col in POST_OUTCOME_PROFIT_COLUMNS:
        return True

    if col in IDENTIFIER_AND_PRIVACY_COLUMNS:
        return True

    if col in RAW_DATE_COLUMNS:
        return True

    for prefix in GRAPH_PREFIXES:
        if col.startswith(prefix):
            return True

    return False


def get_baseline_feature_columns(df: pd.DataFrame) -> List[str]:
    if TARGET_COL not in df.columns:
        raise KeyError(f"Target column not found: {TARGET_COL}")

    feature_cols = []

    for col in df.columns:
        if should_drop_baseline_column(col):
            continue
        if col == TARGET_COL:
            continue
        feature_cols.append(col)

    return feature_cols


def get_proposed_rgc_feature_columns(df: pd.DataFrame) -> List[str]:
    baseline_cols = get_baseline_feature_columns(df)

    graph_cols = [
        c for c in df.columns
        if c.startswith("sthg_risknet_")
        or (
            c.startswith("graph_")
            and any(
                key in c
                for key in [
                    "avg_risk_score",
                    "degree_centrality",
                    "pagerank",
                    "weighted_degree",
                    "graph_risk_centrality",
                    "node_risk_pressure",
                ]
            )
        )
    ]

    final_cols = list(dict.fromkeys(baseline_cols + graph_cols))
    final_cols = [c for c in final_cols if c in df.columns and c != TARGET_COL]

    return final_cols


def split_x_y(
    split_data: Dict[str, pd.DataFrame],
    feature_cols: List[str],
) -> Dict[str, object]:
    data = {}

    for split_name, split_df in split_data.items():
        X = split_df[feature_cols].copy()
        y = pd.to_numeric(split_df[TARGET_COL], errors="coerce").fillna(0).astype(int)

        data[f"X_{split_name}"] = X
        data[f"y_{split_name}"] = y

    return data


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_cols),
            ("categorical", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_model_pipeline(model_name: str, X_train: pd.DataFrame, random_state: int = 42):
    preprocessor = build_preprocessor(X_train)
    model = import_baseline_model(model_name, random_state=random_state)

    if model_name == "CatBoost":
        return {
            "model_type": "catboost_bundle",
            "model_name": model_name,
            "preprocessor": preprocessor,
            "model": model,
            "feature_columns": X_train.columns.tolist(),
        }

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def fit_model(model_bundle, X_train: pd.DataFrame, y_train: pd.Series):
    if isinstance(model_bundle, dict) and model_bundle.get("model_type") == "catboost_bundle":
        preprocessor = model_bundle["preprocessor"]
        model = model_bundle["model"]

        X_train_transformed = preprocessor.fit_transform(X_train)
        model.fit(X_train_transformed, y_train)

        model_bundle["is_fitted"] = True
        return model_bundle

    model_bundle.fit(X_train, y_train)
    return model_bundle


def predict_with_proba(model_bundle, X: pd.DataFrame):
    if isinstance(model_bundle, dict) and model_bundle.get("model_type") == "catboost_bundle":
        preprocessor = model_bundle["preprocessor"]
        model = model_bundle["model"]

        X_transformed = preprocessor.transform(X)
        y_pred = model.predict(X_transformed)
        y_pred = np.asarray(y_pred).reshape(-1).astype(int)

        y_proba = None
        try:
            y_proba = model.predict_proba(X_transformed)
        except Exception:
            y_proba = None

        return y_pred, y_proba

    y_pred = model_bundle.predict(X)
    y_pred = np.asarray(y_pred).reshape(-1).astype(int)

    y_proba = None
    if hasattr(model_bundle, "predict_proba"):
        try:
            y_proba = model_bundle.predict_proba(X)
        except Exception:
            y_proba = None

    return y_pred, y_proba


def calculate_metrics(y_true, y_pred, y_proba=None) -> Dict[str, float]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }

    if y_proba is not None:
        try:
            metrics["roc_auc_ovr_macro"] = float(
                roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
            )
        except Exception:
            metrics["roc_auc_ovr_macro"] = np.nan
    else:
        metrics["roc_auc_ovr_macro"] = np.nan

    return metrics


def train_evaluate_single_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    random_state: int = 42,
) -> Dict[str, object]:
    model = build_model_pipeline(model_name, X_train, random_state=random_state)
    model = fit_model(model, X_train, y_train)

    results = {
        "model_name": model_name,
        "model": model,
        "metrics": {},
        "predictions": {},
        "classification_reports": {},
        "confusion_matrices": {},
    }

    for split_name, X_split, y_split in [
        ("train", X_train, y_train),
        ("valid", X_valid, y_valid),
        ("test", X_test, y_test),
    ]:
        y_pred, y_proba = predict_with_proba(model, X_split)

        results["metrics"][split_name] = calculate_metrics(y_split, y_pred, y_proba)

        results["predictions"][split_name] = {
            "y_true": np.asarray(y_split),
            "y_pred": np.asarray(y_pred),
            "y_proba": y_proba,
        }

        results["classification_reports"][split_name] = classification_report(
            y_split,
            y_pred,
            target_names=["Low", "Moderate", "High"],
            zero_division=0,
            output_dict=True,
        )

        results["confusion_matrices"][split_name] = confusion_matrix(
            y_split,
            y_pred,
            labels=[0, 1, 2],
        )

    return results


def save_model(model, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def make_metrics_table(all_results: Dict[str, Dict[str, object]]) -> pd.DataFrame:
    rows = []

    for model_name, result in all_results.items():
        for split_name, metrics in result["metrics"].items():
            row = {
                "model_name": model_name,
                "split": split_name,
            }
            row.update(metrics)
            rows.append(row)

    return pd.DataFrame(rows)


def make_prediction_table(
    model_name: str,
    split_name: str,
    y_true,
    y_pred,
    y_proba=None,
) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "model_name": model_name,
            "split": split_name,
            "y_true": y_true,
            "y_pred": y_pred,
        }
    )

    if y_proba is not None:
        for class_idx in range(y_proba.shape[1]):
            df[f"proba_class_{class_idx}"] = y_proba[:, class_idx]

    return df