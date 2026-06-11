"""Reusable pipeline for the Index Return Prediction portfolio.

Place raw CSV files in data/raw/, then run:
    python src/index_return_pipeline.py --data-dir data/raw --output-dir outputs

This script is intentionally compact and focuses on the clean workflow:
- load stock, company, index, train, and test files
- build classification and regression datasets
- compare key models
- export metrics and July regression predictions when possible
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import f1_score, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.svm import SVC, SVR

SEED = 42


def norm_month(series: pd.Series) -> pd.Series:
    """Normalize month_id variants into YYYYMM strings."""
    clean = series.astype(str).str.replace(r"\D", "", regex=True)
    return clean.apply(lambda x: x if len(x) == 6 else x[:4] + x[-2:].zfill(2))


def load_data(data_dir: Path) -> Dict[str, pd.DataFrame]:
    files = {
        "stock": "stock_data.csv",
        "company": "company_info.csv",
        "index": "index.csv",
        "train_y": "training_targets.csv",
        "test_ids": "testing_targets.csv",
    }
    missing = [name for name in files.values() if not (data_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required files in {data_dir}: {missing}")

    data = {key: pd.read_csv(data_dir / fname) for key, fname in files.items()}
    for df in data.values():
        if "month_id" in df.columns:
            df["month_id_norm"] = norm_month(df["month_id"])
    return data


def build_base_dataset(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    stock = data["stock"]
    company = data["company"]
    targets = data["train_y"]
    return stock.merge(company, on="stock_id", how="left").merge(
        targets[["stock_id", "month_id", "outperform_binary", "excess_return"]],
        on=["stock_id", "month_id"],
        how="inner",
    )


def split_features(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    drop_cols = ["stock_id", "month_id", "month_id_norm", "outperform_binary", "excess_return"]
    X = df.drop(columns=drop_cols, errors="ignore").copy()
    y = df[target].copy()
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    return X, y, numeric_cols, categorical_cols


def make_preprocessor(numeric_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ],
        remainder="drop",
    )


def evaluate_classification(df: pd.DataFrame) -> Dict[str, float]:
    clean = df.dropna(subset=["outperform_binary"]).dropna()
    X, y, numeric_cols, categorical_cols = split_features(clean, "outperform_binary")
    y = y.astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    pre = make_preprocessor(numeric_cols, categorical_cols)
    models = {
        "logistic_regression": LogisticRegression(max_iter=500, random_state=SEED),
        "random_forest": RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1),
        "svm_rbf": SVC(kernel="rbf", random_state=SEED),
        "knn": KNeighborsClassifier(),
        "naive_bayes": GaussianNB(),
    }
    scores = {}
    for name, model in models.items():
        pipe = Pipeline([("preprocess", pre), ("model", model)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        scores[name] = float(f1_score(y_test, pred))
    return scores


def build_lagged_regression_dataset(data: Dict[str, pd.DataFrame], lags=(1, 3)) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str], pd.DataFrame]:
    stock = data["stock"].sort_values(["stock_id", "month_id_norm"]).copy()
    company = data["company"].copy()
    train_y = data["train_y"].copy()

    numeric_candidates = [
        "intramonth_return", "return_1m", "return_3m", "return_6m",
        "intramonth_volatility", "volatility_3m", "volatility_6m",
        "monthly_volume", "avg_volume_3m", "volume_ratio", "price_range_ratio", "trading_days",
    ]
    numeric_base = [c for c in numeric_candidates if c in stock.columns]

    for col in numeric_base:
        for lag in lags:
            stock[f"{col}_lag{lag}"] = stock.groupby("stock_id")[col].shift(lag)

    stock["month_num"] = stock["month_id_norm"].str[-2:].astype(int)
    stock["month_sin"] = np.sin(2 * np.pi * stock["month_num"] / 12)
    stock["month_cos"] = np.cos(2 * np.pi * stock["month_num"] / 12)

    cat_candidates = [
        "sector", "market_cap_category", "profitability_profile", "business_maturity",
        "competitive_position", "revenue_tier", "asset_intensity", "financial_strength",
    ]
    cat_cols = [c for c in cat_candidates if c in company.columns]
    lag_cols = [c for c in stock.columns if any(c.endswith(f"_lag{lag}") for lag in lags)]
    num_cols = lag_cols + ["month_sin", "month_cos"]

    train = stock.merge(company[["stock_id"] + cat_cols], on="stock_id", how="left").merge(
        train_y[["stock_id", "month_id_norm", "excess_return"]],
        on=["stock_id", "month_id_norm"],
        how="inner",
    ).dropna(subset=lag_cols + ["excess_return"])

    X = train[num_cols + cat_cols].copy()
    y = train["excess_return"].copy()
    return X, y, num_cols, cat_cols, stock


def evaluate_regression(data: Dict[str, pd.DataFrame]) -> Tuple[Dict[str, float], Pipeline, pd.DataFrame, List[str], List[str], pd.DataFrame]:
    X, y, num_cols, cat_cols, stock_with_lags = build_lagged_regression_dataset(data)
    pre = make_preprocessor(num_cols, cat_cols)
    models = {
        "linear_regression": LinearRegression(),
        "ridge": Ridge(random_state=SEED),
        "lasso": Lasso(alpha=0.01, max_iter=10000, random_state=SEED),
        "svr": SVR(kernel="rbf"),
        "random_forest": RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1),
        "gradient_boosting": GradientBoostingRegressor(random_state=SEED),
    }
    tscv = TimeSeriesSplit(n_splits=5)
    scores = {}
    best_name, best_score, best_pipe = None, float("inf"), None

    for name, model in models.items():
        pipe = Pipeline([("preprocess", pre), ("model", model)])
        rmses = []
        for train_idx, valid_idx in tscv.split(X):
            pipe.fit(X.iloc[train_idx], y.iloc[train_idx])
            pred = pipe.predict(X.iloc[valid_idx])
            rmses.append(math.sqrt(mean_squared_error(y.iloc[valid_idx], pred)))
        score = float(np.mean(rmses))
        scores[name] = score
        if score < best_score:
            best_name, best_score, best_pipe = name, score, pipe

    best_pipe.fit(X, y)
    return scores, best_pipe, X, num_cols, cat_cols, stock_with_lags


def export_july_predictions(data: Dict[str, pd.DataFrame], model: Pipeline, stock_with_lags: pd.DataFrame, num_cols: List[str], cat_cols: List[str], output_dir: Path) -> None:
    test_ids = data["test_ids"].copy()
    company = data["company"].copy()
    last_train_month = stock_with_lags["month_id_norm"].max()
    latest_features = stock_with_lags[stock_with_lags["month_id_norm"] == last_train_month][["stock_id"] + num_cols]
    july = test_ids[["stock_id", "month_id_norm"]].merge(latest_features, on="stock_id", how="left")
    july = july.merge(company[["stock_id"] + cat_cols], on="stock_id", how="left")
    july = july.dropna(subset=num_cols)
    preds = model.predict(july[num_cols + cat_cols])
    out = july[["stock_id"]].copy()
    out["excess_return"] = preds
    out.to_csv(output_dir / "testing_targets_regression.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = load_data(args.data_dir)
    base = build_base_dataset(data)
    metrics = {
        "classification_f1": evaluate_classification(base),
    }
    reg_scores, best_reg_model, _, num_cols, cat_cols, stock_with_lags = evaluate_regression(data)
    metrics["regression_rmse"] = reg_scores
    export_july_predictions(data, best_reg_model, stock_with_lags, num_cols, cat_cols, args.output_dir)

    with open(args.output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
