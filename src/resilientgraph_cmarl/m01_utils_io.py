import json
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def print_header(title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def print_subheader(title: str) -> None:
    print("\n" + "-" * 100)
    print(title)
    print("-" * 100)


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def save_json(data, path: Path) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, default=str)
    print(f"[SAVED] {path}")


def save_text(text: str, path: Path) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"[SAVED] {path}")


def save_csv(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=False)
    print(f"[SAVED] {path}")


def load_csv_flexible(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    encodings = ["utf-8", "latin1", "ISO-8859-1", "cp1252"]
    last_error = None

    for enc in encodings:
        try:
            df = pd.read_csv(path, encoding=enc, low_memory=False)
            print(f"[OK] Loaded CSV using encoding: {enc}")
            return df
        except Exception as e:
            last_error = e

    raise RuntimeError(f"Could not read CSV file. Last error: {last_error}")


def normalize_column_name(col: str) -> str:
    return (
        str(col)
        .strip()
        .replace("\n", " ")
        .replace("\t", " ")
        .replace("  ", " ")
    )


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [normalize_column_name(c) for c in df.columns]
    return df


def get_column_match(columns: List[str], candidates: List[str]) -> Optional[str]:
    col_map = {c.lower().strip(): c for c in columns}

    for candidate in candidates:
        key = candidate.lower().strip()
        if key in col_map:
            return col_map[key]

    for candidate in candidates:
        candidate_lower = candidate.lower().strip()
        for col in columns:
            if candidate_lower in col.lower().strip():
                return col

    return None


def find_expected_columns(columns: List[str], expected_map: Dict[str, List[str]]) -> Dict[str, Optional[str]]:
    found = {}
    for role, candidates in expected_map.items():
        found[role] = get_column_match(columns, candidates)
    return found


def infer_possible_date_columns(df: pd.DataFrame) -> List[str]:
    return [col for col in df.columns if "date" in col.lower() or "time" in col.lower()]


def infer_possible_id_columns(df: pd.DataFrame) -> List[str]:
    return [col for col in df.columns if "id" in col.lower() or col.lower().endswith("_id")]


def infer_possible_leakage_columns(df: pd.DataFrame, leakage_keywords: List[str]) -> List[str]:
    possible = []
    for col in df.columns:
        c = col.lower()
        for kw in leakage_keywords:
            if kw.lower() in c:
                possible.append(col)
                break
    return possible


def dataframe_memory_mb(df: pd.DataFrame) -> float:
    return round(df.memory_usage(deep=True).sum() / (1024 ** 2), 4)


def make_missing_report(df: pd.DataFrame) -> pd.DataFrame:
    report = pd.DataFrame({
        "column": df.columns,
        "missing_count": df.isna().sum().values,
        "missing_percent": (df.isna().mean().values * 100).round(4),
        "dtype": [str(df[c].dtype) for c in df.columns],
        "unique_values": [df[c].nunique(dropna=True) for c in df.columns],
    })
    return report.sort_values("missing_percent", ascending=False)


def make_dtype_report(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        rows.append({
            "column": col,
            "dtype": str(df[col].dtype),
            "unique_values": df[col].nunique(dropna=True),
            "sample_values": ", ".join(map(str, df[col].dropna().astype(str).head(5).tolist())),
        })
    return pd.DataFrame(rows)


def make_categorical_summary(df: pd.DataFrame, max_unique: int = 30, top_n: int = 10) -> pd.DataFrame:
    rows = []
    object_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    for col in object_cols:
        nunique = df[col].nunique(dropna=True)
        if nunique <= max_unique:
            value_counts = df[col].value_counts(dropna=False).head(top_n)
            rows.append({
                "column": col,
                "unique_values": nunique,
                "top_values": " | ".join([f"{idx}: {val}" for idx, val in value_counts.items()]),
            })
        else:
            rows.append({
                "column": col,
                "unique_values": nunique,
                "top_values": "High-cardinality column",
            })

    return pd.DataFrame(rows)


def safe_numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        return pd.DataFrame()
    return numeric_df.describe().T.reset_index().rename(columns={"index": "column"})