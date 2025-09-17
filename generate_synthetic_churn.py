"""Generate synthetic customer churn data for stress-testing the analytics pipeline.

Usage:
    python generate_synthetic_churn.py --rows 1000 --output synthetic_customer_churn.csv

The script bootstraps feature distributions from the historical dataset and
uses a logistic regression model to assign churn labels and probabilities.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "customer_churn_data.csv"
NUMERIC_FEATURES = ["tenure_months", "monthly_spend", "support_tickets", "last_login_days"]
CATEGORICAL_FEATURES = ["plan_type"]
DEFAULT_OUTPUT = BASE_DIR / "synthetic_customer_churn.csv"

def load_source_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [col.strip().lower() for col in df.columns]
    return df


def train_logistic(df: pd.DataFrame) -> Pipeline:
    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X = df[features]
    y = df["churned"].astype(int)

    preprocess = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[("scaler", StandardScaler())]), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )

    model = Pipeline(
        steps=[
            ("prep", preprocess),
            ("clf", LogisticRegression(max_iter=1000, solver="lbfgs")),
        ]
    )

    X_train, X_test, y_train, _ = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    model.fit(X_train, y_train)
    return model


def sample_numeric(series: pd.Series, size: int, noise: float = 0.15) -> np.ndarray:
    values = series.dropna().to_numpy()
    bootstrap = np.random.choice(values, size=size, replace=True)
    std_dev = series.std(ddof=0)
    jitter = np.random.normal(0, std_dev * noise, size=size)
    sampled = np.clip(bootstrap + jitter, values.min(), values.max())
    if series.dtype.kind in {"i", "u"}:
        return np.rint(sampled).astype(int)
    return sampled


def sample_support(series: pd.Series, size: int) -> np.ndarray:
    lam = max(series.mean(), 0.1)
    samples = np.random.poisson(lam, size=size)
    return samples.clip(0, int(series.max()))


def sample_categorical(series: pd.Series, size: int) -> list[str]:
    probs = series.value_counts(normalize=True).sort_index()
    categories = probs.index.to_list()
    weights = probs.values
    return list(np.random.choice(categories, size=size, p=weights))


def synthesize_dataset(df: pd.DataFrame, rows: int, seed: int | None = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    size = rows
    result = pd.DataFrame()

    result["tenure_months"] = sample_numeric(df["tenure_months"], size)
    result["monthly_spend"] = np.round(sample_numeric(df["monthly_spend"], size), 2)
    result["support_tickets"] = sample_support(df["support_tickets"], size)
    result["last_login_days"] = sample_numeric(df["last_login_days"], size)
    result["plan_type"] = sample_categorical(df["plan_type"], size)

    fake = Faker()
    Faker.seed(seed)
    result["customer_id"] = [f"SYN-{rng.integers(1_000_000, 9_999_999)}" for _ in range(size)]

    trained_model = train_logistic(df)
    probabilities = trained_model.predict_proba(result[NUMERIC_FEATURES + CATEGORICAL_FEATURES])[:, 1]
    result["churn_probability"] = np.round(probabilities, 3)
    result["churned"] = (probabilities >= 0.5).astype(int)
    ordered_columns = [
        "customer_id",
        "tenure_months",
        "monthly_spend",
        "support_tickets",
        "last_login_days",
        "plan_type",
        "churn_probability",
        "churned",
    ]
    return result[ordered_columns]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic churn dataset")
    parser.add_argument("--rows", type=int, default=500, help="Number of synthetic rows to generate")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output CSV file path",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DATA_PATH,
        help="Path to the original churn dataset",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.source.exists():
        raise FileNotFoundError(f"Source dataset not found at {args.source}")

    source_df = load_source_data(args.source)
    synthetic_df = synthesize_dataset(source_df, args.rows, seed=args.seed)
    synthetic_df.to_csv(args.output, index=False)
    print(f"Synthetic dataset saved to {args.output.resolve()} ({len(synthetic_df)} rows)")


if __name__ == "__main__":
    main()
