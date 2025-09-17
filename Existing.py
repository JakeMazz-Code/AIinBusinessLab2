"""Customer churn analytics, reporting, and dashboard prep pipeline.

This version automates the full lab deliverable and then some:
  * Calculates core churn metrics, extended correlations, and cohort comparisons.
  * Ranks drivers with correlation, mean deltas, and mutual information.
  * Builds a rule-based heuristic alongside a logistic regression benchmark.
  * Generates visualizations, a markdown report, JSON/CSV outputs, and a Streamlit-ready dataset.
  * Prints a concise console summary so findings are visible when you run it.

Install dependencies with `pip install -r requirements.txt`, then run `python Existing.py`.
Launch the interactive dashboard with `streamlit run dashboard.py` once outputs are generated.
All artifacts are saved under the `reports/` folder.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import matplotlib

# Use a non-interactive backend so charts save correctly in any environment.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

pd.options.display.float_format = "{:.2f}".format
sns.set_theme(style="whitegrid")

DATA_PATH = Path(__file__).with_name("customer_churn_data.csv")
REPORT_DIR = Path(__file__).with_name("reports")
FIGURES_DIR = REPORT_DIR / "figures"
REPORT_PATH = REPORT_DIR / "churn_analysis_report.md"
PREDICTIONS_PATH = REPORT_DIR / "churn_risk_scoring.csv"
SUMMARY_PATH = REPORT_DIR / "metrics_summary.json"

NUMERIC_FEATURES = [
    "tenure_months",
    "monthly_spend",
    "support_tickets",
    "last_login_days",
]
CATEGORICAL_FEATURES = ["plan_type"]
AT_RISK_THRESHOLD = 0.60
PAIRPLOT_SAMPLE = 400
FORECAST_MONTHS = 6


@dataclass
class RuleBasedModel:
    """Simple heuristic model that flags high-risk customers."""

    thresholds: Dict[str, float]
    min_conditions: int = 2

    def _build_signal_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "short_tenure": df["tenure_months"] <= self.thresholds["tenure_months"],
                "low_spend": df["monthly_spend"] <= self.thresholds["monthly_spend"],
                "high_tickets": df["support_tickets"] >= self.thresholds["support_tickets"],
                "inactive_login": df["last_login_days"] >= self.thresholds["last_login_days"],
            }
        )

    def predict(self, df: pd.DataFrame) -> pd.Series:
        signals = self._build_signal_frame(df)
        condition_count = signals.sum(axis=1)
        return (condition_count >= self.min_conditions).astype(int)

    def signal_strength(self, df: pd.DataFrame) -> pd.Series:
        signals = self._build_signal_frame(df)
        return signals.sum(axis=1) / len(signals.columns)

    def describe(self) -> str:
        return (
            "Flag a customer if at least two of the following conditions are true:\n"
            f"  - tenure_months <= {self.thresholds['tenure_months']:.1f}\n"
            f"  - monthly_spend <= ${self.thresholds['monthly_spend']:.2f}\n"
            f"  - support_tickets >= {self.thresholds['support_tickets']:.0f}\n"
            f"  - last_login_days >= {self.thresholds['last_login_days']:.0f}"
        )


def ensure_output_dirs() -> None:
    REPORT_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset(csv_path: Path = DATA_PATH) -> pd.DataFrame:
    """Load and type-cast the churn dataset."""

    df = pd.read_csv(csv_path)
    df.columns = [col.strip().lower() for col in df.columns]
    df["churned"] = pd.to_numeric(df["churned"], errors="coerce").fillna(0).astype(int)

    for feature in NUMERIC_FEATURES:
        df[feature] = pd.to_numeric(df[feature], errors="coerce")

    df = df.dropna(subset=NUMERIC_FEATURES + ["churned"])
    df["plan_type"] = df.get("plan_type", "Unknown").astype(str)
    return df


def compute_basic_metrics(df: pd.DataFrame) -> Dict[str, pd.DataFrame | float]:
    """Return churn rate, cohort comparisons, and plan-level churn."""

    churn_rate = df["churned"].mean()
    cohort_means = df.groupby("churned")[NUMERIC_FEATURES].mean()
    plan_churn = df.groupby("plan_type")["churned"].mean().sort_values(ascending=False)
    return {
        "churn_rate": churn_rate,
        "cohort_means": cohort_means,
        "plan_churn": plan_churn,
    }


def compute_feature_rankings(
    df: pd.DataFrame, cohort_means: pd.DataFrame
) -> Tuple[pd.Series, pd.DataFrame, pd.DataFrame, pd.Series]:
    """Calculate churn correlations and rank features."""

    correlations = (
        df[NUMERIC_FEATURES + ["churned"]]
        .corr(numeric_only=True)["churned"]
        .drop("churned")
    )
    mean_deltas = cohort_means.loc[1] - cohort_means.loc[0]
    ranking = (
        pd.DataFrame(
            {
                "correlation": correlations,
                "mean_delta_vs_retained": mean_deltas,
                "abs_correlation": correlations.abs(),
            }
        )
        .sort_values("abs_correlation", ascending=False)
        .drop(columns="abs_correlation")
    )

    mutual_info_values = mutual_info_classif(
        df[NUMERIC_FEATURES], df["churned"], random_state=42
    )
    mutual_info = pd.Series(
        mutual_info_values, index=NUMERIC_FEATURES, name="mutual_information"
    ).sort_values(ascending=False)
    ranking["mutual_information"] = mutual_info.reindex(ranking.index)

    correlation_matrix = df[NUMERIC_FEATURES + ["churned"]].corr(numeric_only=True)
    return correlations, ranking, correlation_matrix, mutual_info


def derive_rule_thresholds(df: pd.DataFrame) -> Dict[str, float]:
    """Use churned-customer quantiles to set decision thresholds."""

    churned_subset = df[df["churned"] == 1]
    return {
        "tenure_months": churned_subset["tenure_months"].median(),
        "monthly_spend": churned_subset["monthly_spend"].median(),
        "support_tickets": churned_subset["support_tickets"].quantile(0.75),
        "last_login_days": churned_subset["last_login_days"].median(),
    }


def evaluate_model(actual: pd.Series, predictions: pd.Series) -> Dict[str, float]:
    """Compute accuracy and confusion-matrix-derived metrics."""

    actual = actual.astype(int)
    predictions = predictions.astype(int)
    true_positive = ((predictions == 1) & (actual == 1)).sum()
    true_negative = ((predictions == 0) & (actual == 0)).sum()
    false_positive = ((predictions == 1) & (actual == 0)).sum()
    false_negative = ((predictions == 0) & (actual == 1)).sum()

    accuracy = (predictions == actual).mean()
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "true_positive": float(true_positive),
        "true_negative": float(true_negative),
        "false_positive": float(false_positive),
        "false_negative": float(false_negative),
    }


def train_logistic_model(
    df: pd.DataFrame,
) -> Tuple[Pipeline, Dict[str, float], Dict[str, float], str]:
    """Fit a logistic regression benchmark and return evaluation details."""

    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X = df[features]
    y = df["churned"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y,
    )

    preprocess = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[("scaler", StandardScaler())]), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(drop="first", handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    base_pipeline = Pipeline(
        steps=[
            ("prep", preprocess),
            ("clf", LogisticRegression(max_iter=1000, solver="lbfgs")),
        ]
    )

    base_pipeline.fit(X_train, y_train)
    y_pred = base_pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", zero_division=0
    )
    conf_matrix = confusion_matrix(y_test, y_pred)
    report_txt = classification_report(y_test, y_pred, digits=2)

    metric_summary = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }
    confusion_summary = {
        "true_negative": float(conf_matrix[0, 0]),
        "false_positive": float(conf_matrix[0, 1]),
        "false_negative": float(conf_matrix[1, 0]),
        "true_positive": float(conf_matrix[1, 1]),
    }

    final_model = clone(base_pipeline)
    final_model.fit(X, y)

    return final_model, metric_summary, confusion_summary, report_txt


def tidy_feature_name(name: str) -> str:
    name = name.replace("num__", "").replace("cat__", "")
    return name.replace("plan_type_", "plan=")


def extract_logistic_coefficients(model: Pipeline) -> pd.DataFrame:
    feature_names = model.named_steps["prep"].get_feature_names_out()
    coefficients = model.named_steps["clf"].coef_[0]
    odds_ratios = np.exp(coefficients)

    coeff_df = (
        pd.DataFrame(
            {
                "feature": [tidy_feature_name(name) for name in feature_names],
                "coefficient": coefficients,
                "odds_ratio": odds_ratios,
            }
        )
        .assign(abs_coefficient=lambda d: d["coefficient"].abs())
        .sort_values("abs_coefficient", ascending=False)
        .drop(columns="abs_coefficient")
    )
    return coeff_df

def ensure_monotonic_bins(values: Iterable[float]) -> list[float]:
    adjusted: list[float] = []
    previous: float | None = None
    for raw_value in values:
        value = float(raw_value)
        if previous is not None and value <= previous:
            value = previous + 1
        adjusted.append(value)
        previous = value
    return adjusted


def compute_segment_analysis(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    segments: Dict[str, pd.DataFrame] = {}
    working = df.copy()

    tenure_bins = ensure_monotonic_bins([0, 6, 12, 24, 36, working["tenure_months"].max() + 1])
    tenure_labels = ["0-6", "7-12", "13-24", "25-36", "36+"]
    working["tenure_bucket"] = pd.cut(
        working["tenure_months"],
        bins=tenure_bins,
        duplicates="drop",
        labels=tenure_labels,
        include_lowest=True,
        right=False,
    )
    segments["tenure"] = working.groupby("tenure_bucket").agg(
        churn_rate=("churned", "mean"),
        avg_monthly_spend=("monthly_spend", "mean"),
        avg_support_tickets=("support_tickets", "mean"),
        avg_churn_probability=("logistic_probability", "mean"),
        customers=("customer_id", "count"),
    )

    inactivity_bins = ensure_monotonic_bins([-1, 14, 30, 60, working["last_login_days"].max() + 1])
    inactivity_labels = ["0-14 days", "15-30", "31-60", "60+"]
    working["inactivity_bucket"] = pd.cut(
        working["last_login_days"],
        bins=inactivity_bins,
        duplicates="drop",
        labels=inactivity_labels,
        include_lowest=True,
    )
    segments["inactivity"] = working.groupby("inactivity_bucket").agg(
        churn_rate=("churned", "mean"),
        avg_monthly_spend=("monthly_spend", "mean"),
        avg_support_tickets=("support_tickets", "mean"),
        avg_churn_probability=("logistic_probability", "mean"),
        customers=("customer_id", "count"),
    )

    support_bins = ensure_monotonic_bins([-1, 0, 2, 5, working["support_tickets"].max() + 1])
    support_labels = ["0", "1-2", "3-5", "6+"]
    working["support_bucket"] = pd.cut(
        working["support_tickets"],
        bins=support_bins,
        duplicates="drop",
        labels=support_labels,
        include_lowest=True,
    )
    segments["support"] = working.groupby("support_bucket").agg(
        churn_rate=("churned", "mean"),
        avg_monthly_spend=("monthly_spend", "mean"),
        avg_last_login_days=("last_login_days", "mean"),
        avg_churn_probability=("logistic_probability", "mean"),
        customers=("customer_id", "count"),
    )

    segments["plan_profile"] = working.groupby("plan_type").agg(
        churn_rate=("churned", "mean"),
        avg_monthly_spend=("monthly_spend", "mean"),
        avg_support_tickets=("support_tickets", "mean"),
        avg_churn_probability=("logistic_probability", "mean"),
        customers=("customer_id", "count"),
    )

    return segments


def summarize_revenue_at_risk(
    df: pd.DataFrame, logistic_scores: np.ndarray, threshold: float
) -> Dict[str, float]:
    at_risk_mask = logistic_scores >= threshold
    at_risk_count = int(at_risk_mask.sum())
    customer_count = len(df)

    monthly_recurring_revenue = float(df["monthly_spend"].sum())
    threshold_loss_6m = float((df.loc[at_risk_mask, "monthly_spend"] * FORECAST_MONTHS).sum())
    weighted_loss_6m = float((df["monthly_spend"] * logistic_scores * FORECAST_MONTHS).sum())
    expected_monthly_loss = float((df["monthly_spend"] * logistic_scores).sum())
    avg_prob_at_risk = float(logistic_scores[at_risk_mask].mean()) if at_risk_count else 0.0

    return {
        "threshold": threshold,
        "at_risk_count": at_risk_count,
        "at_risk_share": at_risk_count / customer_count if customer_count else 0.0,
        "monthly_recurring_revenue": monthly_recurring_revenue,
        "threshold_loss_6m": threshold_loss_6m,
        "weighted_loss_6m": weighted_loss_6m,
        "expected_monthly_loss": expected_monthly_loss,
        "avg_probability_at_risk": avg_prob_at_risk,
    }


def generate_visualizations(
    df: pd.DataFrame,
    plan_churn: pd.Series,
    ranking: pd.DataFrame,
    correlation_matrix: pd.DataFrame,
    logistic_scores: np.ndarray,
) -> Dict[str, Path]:
    outputs: "OrderedDict[str, Path]" = OrderedDict()
    churn_labels = {0: "Retained", 1: "Churned"}
    df_plot = df.copy()
    df_plot["churn_label"] = df_plot["churned"].map(churn_labels)

    plan_df = plan_churn.reset_index(name="churn_rate")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(
        data=plan_df,
        x="plan_type",
        y="churn_rate",
        hue="plan_type",
        palette="viridis",
        legend=False,
        ax=ax,
    )
    ax.set_ylabel("Churn Rate")
    ax.set_xlabel("Plan Type")
    ax.set_title("Churn Rate by Subscription Plan")
    ax.set_ylim(0, 1)
    for container in ax.containers:
        ax.bar_label(container, fmt="{:.0%}")
    plt.tight_layout()
    path_plan = FIGURES_DIR / "plan_churn_rate.png"
    fig.savefig(path_plan, dpi=300)
    plt.close(fig)
    outputs["plan_churn_rate"] = path_plan

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.kdeplot(
        data=df_plot,
        x="tenure_months",
        hue="churn_label",
        fill=True,
        common_norm=False,
        palette="rocket",
        ax=ax,
    )
    ax.set_title("Tenure Distribution by Churn Outcome")
    ax.set_xlabel("Tenure (months)")
    plt.tight_layout()
    path_tenure = FIGURES_DIR / "tenure_distribution.png"
    fig.savefig(path_tenure, dpi=300)
    plt.close(fig)
    outputs["tenure_distribution"] = path_tenure

    ranking_sorted = ranking.reset_index().rename(columns={"index": "feature"})
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(
        data=ranking_sorted,
        y="feature",
        x="correlation",
        hue="feature",
        palette="crest",
        legend=False,
        ax=ax,
    )
    ax.set_title("Feature Correlation with Churn")
    ax.set_xlabel("Correlation")
    plt.tight_layout()
    path_corr = FIGURES_DIR / "feature_correlations.png"
    fig.savefig(path_corr, dpi=300)
    plt.close(fig)
    outputs["feature_correlations"] = path_corr

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(
        correlation_matrix.loc[NUMERIC_FEATURES, NUMERIC_FEATURES],
        annot=True,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        fmt=".2f",
        ax=ax,
    )
    ax.set_title("Correlation Matrix (Numeric Features)")
    plt.tight_layout()
    path_heatmap = FIGURES_DIR / "correlation_heatmap.png"
    fig.savefig(path_heatmap, dpi=300)
    plt.close(fig)
    outputs["correlation_heatmap"] = path_heatmap

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(
        data=df_plot,
        x="churn_label",
        y="monthly_spend",
        palette="Set2",
        ax=ax,
    )
    ax.set_title("Monthly Spend by Churn Status")
    ax.set_xlabel("")
    ax.set_ylabel("Monthly Spend ($)")
    plt.tight_layout()
    path_spend = FIGURES_DIR / "monthly_spend_by_churn.png"
    fig.savefig(path_spend, dpi=300)
    plt.close(fig)
    outputs["monthly_spend_by_churn"] = path_spend

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.violinplot(
        data=df_plot,
        x="churn_label",
        y="support_tickets",
        palette="Pastel1",
        cut=0,
        inner="quartile",
        ax=ax,
    )
    ax.set_title("Support Tickets by Churn Status")
    ax.set_xlabel("")
    ax.set_ylabel("Support Tickets (last 6 months)")
    plt.tight_layout()
    path_support = FIGURES_DIR / "support_tickets_by_churn.png"
    fig.savefig(path_support, dpi=300)
    plt.close(fig)
    outputs["support_tickets_by_churn"] = path_support

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(logistic_scores, bins=20, kde=True, color="#1f77b4", ax=ax)
    ax.axvline(
        AT_RISK_THRESHOLD,
        color="crimson",
        linestyle="--",
        label=f"High-risk threshold ({AT_RISK_THRESHOLD:.0%})",
    )
    ax.set_title("Churn Probability Distribution (Logistic Model)")
    ax.set_xlabel("Predicted Churn Probability")
    ax.legend()
    plt.tight_layout()
    path_prob = FIGURES_DIR / "logistic_probability_distribution.png"
    fig.savefig(path_prob, dpi=300)
    plt.close(fig)
    outputs["logistic_probability_distribution"] = path_prob

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.scatterplot(
        data=df_plot,
        x="tenure_months",
        y="last_login_days",
        hue="churn_label",
        palette="coolwarm",
        alpha=0.7,
        ax=ax,
    )
    ax.set_title("Tenure vs. Days Since Last Login")
    ax.set_xlabel("Tenure (months)")
    ax.set_ylabel("Days Since Last Login")
    plt.tight_layout()
    path_scatter = FIGURES_DIR / "tenure_vs_last_login_scatter.png"
    fig.savefig(path_scatter, dpi=300)
    plt.close(fig)
    outputs["tenure_vs_last_login_scatter"] = path_scatter

    sample_size = min(PAIRPLOT_SAMPLE, len(df_plot))
    sample_df = df_plot.sample(n=sample_size, random_state=42)
    pair_plot = sns.pairplot(
        sample_df,
        vars=NUMERIC_FEATURES,
        hue="churn_label",
        diag_kind="kde",
        corner=True,
        palette="husl",
    )
    pair_plot.fig.suptitle("Pairwise Numeric Relationships by Churn", y=1.02)
    path_pair = FIGURES_DIR / "numeric_pairplot.png"
    pair_plot.fig.savefig(path_pair, dpi=220)
    plt.close(pair_plot.fig)
    outputs["numeric_pairplot"] = path_pair

    return outputs


def save_predictions(df: pd.DataFrame, output_path: Path) -> None:
    """Persist customer-level risk scores for further analysis."""

    df.to_csv(output_path, index=False, float_format="%.4f")


def format_dataframe_block(df: pd.DataFrame) -> str:
    return "\n".join(["```", df.to_string(), "```"])


def build_report(
    df: pd.DataFrame,
    churn_rate: float,
    cohort_means: pd.DataFrame,
    ranking: pd.DataFrame,
    correlation_matrix: pd.DataFrame,
    mutual_info: pd.Series,
    plan_churn: pd.Series,
    rule_model: RuleBasedModel,
    rule_eval: Dict[str, float],
    logistic_eval: Dict[str, float],
    logistic_confusion: Dict[str, float],
    logistic_report: str,
    logistic_coefficients: pd.DataFrame,
    segment_tables: Dict[str, pd.DataFrame],
    revenue_summary: Dict[str, float],
    chart_paths: Dict[str, Path],
    top_risk: pd.DataFrame,
) -> str:
    timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")

    report_lines = [
        "# Customer Churn Analysis Report",
        f"_Generated on {timestamp}_",
        "",
        "## 1. Business Snapshot",
        f"- Total customers analyzed: {len(df)}",
        f"- Overall churn rate: {churn_rate:.1%} (approximately 1 in {int(round(1 / churn_rate))} customers)",
        f"- Monthly recurring revenue: ${revenue_summary['monthly_recurring_revenue']:,.0f}",
        (
            f"- High-risk customers (probability = {revenue_summary['threshold']:.0%}): "
            f"{revenue_summary['at_risk_count']} ({revenue_summary['at_risk_share']:.1%} of base)"
        ),
        (
            f"- Probability-weighted revenue at risk over {FORECAST_MONTHS} months: "
            f"${revenue_summary['weighted_loss_6m']:,.0f}"
        ),
        "",
        "### Cohort Averages (0 = retained, 1 = churned)",
        format_dataframe_block(cohort_means.round(2)),
        "",
        "## 2. Driver Analysis",
        "- Feature ranking is based on Pearson correlation with churn, mean differences vs. retained customers, and mutual information.",
        format_dataframe_block(ranking.round(3)),
        "",
        "### 2.1 Correlation Matrix",
        format_dataframe_block(correlation_matrix.round(3)),
        "",
        "### 2.2 Mutual Information with Churn",
        format_dataframe_block(mutual_info.to_frame().round(3)),
        "",
        "### 2.3 Logistic Regression Coefficients (odds ratios)",
        format_dataframe_block(logistic_coefficients.round({"coefficient": 3, "odds_ratio": 2})),
        "",
        "## 3. Segment Spotlight",
    ]

    segment_titles = [
        ("tenure", "Tenure bucket churn profile"),
        ("inactivity", "Inactivity (days since last login)"),
        ("support", "Support ticket volume"),
        ("plan_profile", "Plan profile summary"),
    ]

    for idx, (key, title) in enumerate(segment_titles, start=1):
        if key in segment_tables:
            report_lines.extend(
                [
                    f"### 3.{idx} {title}",
                    format_dataframe_block(segment_tables[key].round(3)),
                    "",
                ]
            )

    report_lines.extend(
        [
            "## 4. Prediction Approaches",
            "### Rule-Based Heuristic",
            rule_model.describe(),
            "",
            "Evaluation (entire dataset):",
            f"- Accuracy: {rule_eval['accuracy']:.1%}",
            f"- Precision: {rule_eval['precision']:.1%}",
            f"- Recall: {rule_eval['recall']:.1%}",
            f"- Confusion matrix counts (TP/TN/FP/FN): {int(rule_eval['true_positive'])}/{int(rule_eval['true_negative'])}/{int(rule_eval['false_positive'])}/{int(rule_eval['false_negative'])}",
            "",
            "### Logistic Regression Benchmark",
            "- Trained on a 70/30 train-test split with scaled numeric features and one-hot encoded plan type.",
            f"- Accuracy: {logistic_eval['accuracy']:.1%}",
            f"- Precision: {logistic_eval['precision']:.1%}",
            f"- Recall: {logistic_eval['recall']:.1%}",
            f"- F1 score: {logistic_eval['f1']:.1%}",
            f"- Confusion matrix counts (TP/TN/FP/FN): {int(logistic_confusion['true_positive'])}/{int(logistic_confusion['true_negative'])}/{int(logistic_confusion['false_positive'])}/{int(logistic_confusion['false_negative'])}",
            "",
            "Classification report (test set):",
            "```",
            logistic_report.strip(),
            "```",
            "",
            "## 5. Revenue Impact & High-Risk Cohort",
            (
                f"- Customers flagged as high risk (= {revenue_summary['threshold']:.0%} probability): "
                f"{revenue_summary['at_risk_count']} averaging {revenue_summary['avg_probability_at_risk']:.0%} churn likelihood."
            ),
            (
                f"- Revenue exposure if high-risk customers churn (6-month projection): "
                f"${revenue_summary['threshold_loss_6m']:,.0f}."
            ),
            (
                f"- Probability-weighted loss estimate (6-month projection): "
                f"${revenue_summary['weighted_loss_6m']:,.0f}."
            ),
            "",
            "### Highest-Risk Customers (Top 10 by Logistic Probability)",
            format_dataframe_block(top_risk.round({"logistic_probability": 3})),
            "",
            "## 6. Visual Insights",
        ]
    )

    for label, path in chart_paths.items():
        rel_path = path.relative_to(REPORT_DIR)
        pretty_label = label.replace("_", " ").title()
        report_lines.append(f"![{pretty_label}]({rel_path.as_posix()})")
        report_lines.append("")

    report_lines.extend(
        [
            "## 7. Recommendations & Next Steps",
            "1. Prioritize outreach for short-tenure, low-spend customers who recently filed multiple tickets and have been inactive; they trigger most rule conditions.",
            "2. Launch a re-engagement sequence for customers inactive 30+ days, combining personalized offers with content recommendations.",
            "3. Use the logistic regression scores to focus retention efforts on the top third of customers by churn probability and A/B test targeted offers.",
            "4. Collect additional behavioral signals (content engagement, satisfaction scores) and validate the models on a future cohort or synthetic holdout set before deployment.",
            "",
            "## 8. Deliverables",
            f"- Executive summary: `{REPORT_PATH.name}`",
            f"- Visualizations: `{FIGURES_DIR.name}/`",
            f"- Customer-level scores: `{PREDICTIONS_PATH.name}`",
            f"- Metrics summary (Streamlit ready): `{SUMMARY_PATH.name}`",
        ]
    )

    return "\n".join(report_lines)


def print_section(title: str, lines: Iterable[str]) -> None:
    border = "=" * len(title)
    print(border)
    print(title)
    print(border)
    for line in lines:
        print(line)
    print()


def main() -> None:
    ensure_output_dirs()

    df = load_dataset()
    metrics = compute_basic_metrics(df)
    correlations, ranking, corr_matrix, mutual_info = compute_feature_rankings(
        df, metrics["cohort_means"]
    )

    thresholds = derive_rule_thresholds(df)
    rule_model = RuleBasedModel(thresholds=thresholds)
    rule_predictions = rule_model.predict(df)
    rule_scores = rule_model.signal_strength(df)
    rule_eval = evaluate_model(df["churned"], rule_predictions)

    logistic_model, logistic_eval, logistic_confusion, logistic_report = train_logistic_model(df)
    logistic_scores = logistic_model.predict_proba(
        df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    )[:, 1]
    logistic_flags = (logistic_scores >= AT_RISK_THRESHOLD).astype(int)

    risk_output = df.copy()
    risk_output["rule_flag"] = rule_predictions
    risk_output["rule_signal_strength"] = rule_scores.round(2)
    risk_output["logistic_probability"] = logistic_scores
    risk_output["logistic_flag"] = logistic_flags

    revenue_summary = summarize_revenue_at_risk(df, logistic_scores, AT_RISK_THRESHOLD)
    segment_tables = compute_segment_analysis(risk_output)
    logistic_coefficients = extract_logistic_coefficients(logistic_model)

    risk_output_sorted = risk_output.sort_values(
        "logistic_probability", ascending=False
    )
    save_predictions(risk_output_sorted, PREDICTIONS_PATH)

    top_risk = risk_output_sorted.head(10)[
        [
            "customer_id",
            "logistic_probability",
            "rule_flag",
            "tenure_months",
            "monthly_spend",
            "support_tickets",
            "last_login_days",
            "plan_type",
        ]
    ].copy()

    chart_paths = generate_visualizations(
        df,
        metrics["plan_churn"],
        ranking,
        corr_matrix,
        logistic_scores,
    )

    report_text = build_report(
        df=df,
        churn_rate=metrics["churn_rate"],
        cohort_means=metrics["cohort_means"],
        ranking=ranking,
        correlation_matrix=corr_matrix,
        mutual_info=mutual_info,
        plan_churn=metrics["plan_churn"],
        rule_model=rule_model,
        rule_eval=rule_eval,
        logistic_eval=logistic_eval,
        logistic_confusion=logistic_confusion,
        logistic_report=logistic_report,
        logistic_coefficients=logistic_coefficients,
        segment_tables=segment_tables,
        revenue_summary=revenue_summary,
        chart_paths=chart_paths,
        top_risk=top_risk,
    )
    REPORT_PATH.write_text(report_text, encoding="utf-8")

    summary_payload: Dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "customer_count": len(df),
        "churn_rate": float(metrics["churn_rate"]),
        "monthly_recurring_revenue": revenue_summary["monthly_recurring_revenue"],
        "at_risk_threshold": AT_RISK_THRESHOLD,
        "at_risk_count": revenue_summary["at_risk_count"],
        "at_risk_share": revenue_summary["at_risk_share"],
        "revenue_at_risk_6m": revenue_summary["threshold_loss_6m"],
        "weighted_revenue_at_risk_6m": revenue_summary["weighted_loss_6m"],
        "rule_metrics": rule_eval,
        "logistic_metrics": logistic_eval,
        "segment_highlights": {
            key: table.round(3).reset_index().to_dict(orient="list")
            for key, table in segment_tables.items()
        },
        "figures": {key: path.name for key, path in chart_paths.items()},
        "feature_ranking": ranking.round(3).reset_index().rename(columns={"index": "feature"}).to_dict(orient="list"),
        "logistic_coefficients": logistic_coefficients.round({"coefficient": 3, "odds_ratio": 2}).to_dict(orient="list"),
    }
    SUMMARY_PATH.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    print_section(
        "Run Summary",
        [
            f"Customers analyzed: {len(df)}",
            f"Overall churn rate: {metrics['churn_rate']:.1%}",
            f"Rule-based accuracy: {rule_eval['accuracy']:.1%} (recall {rule_eval['recall']:.1%})",
            f"Logistic benchmark accuracy: {logistic_eval['accuracy']:.1%} (recall {logistic_eval['recall']:.1%})",
            f"High-risk customers >= {AT_RISK_THRESHOLD:.0%}: {revenue_summary['at_risk_count']}",
            f"Report saved to: {REPORT_PATH}",
            f"Risk scores saved to: {PREDICTIONS_PATH}",
            f"Metrics summary saved to: {SUMMARY_PATH}",
        ],
    )

    print_section(
        "Next Steps",
        [
            "Review the markdown report for details and copy-ready narrative.",
            "Launch the interactive dashboard: streamlit run dashboard.py",
        ],
    )


if __name__ == "__main__":
    main()


