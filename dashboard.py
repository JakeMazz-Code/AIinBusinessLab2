"""Streamlit dashboard for the customer churn analysis deliverable.

Run `streamlit run dashboard.py` after executing `python Existing.py`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import altair as alt
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
REPORT_DIR = BASE_DIR / "reports"
PREDICTIONS_PATH = REPORT_DIR / "churn_risk_scoring.csv"
SUMMARY_PATH = REPORT_DIR / "metrics_summary.json"
FIGURES_DIR = REPORT_DIR / "figures"
REPORT_MARKDOWN_PATH = REPORT_DIR / "churn_analysis_report.md"

SYNTHETIC_REPORT_DIR = REPORT_DIR / "synthetic"
SYNTHETIC_PREDICTIONS_PATH = SYNTHETIC_REPORT_DIR / "churn_risk_scoring.csv"
SYNTHETIC_SUMMARY_PATH = SYNTHETIC_REPORT_DIR / "metrics_summary.json"
SYNTHETIC_FIGURES_DIR = SYNTHETIC_REPORT_DIR / "figures"
SYNTHETIC_MARKDOWN_PATH = SYNTHETIC_REPORT_DIR / "churn_analysis_report.md"

PAGE_CONFIG = {
    "page_title": "Customer Churn Command Center",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
}

st.set_page_config(**PAGE_CONFIG)

st.title("Customer Churn Command Center")
st.caption(
    "Generated with `Existing.py`. Rerun the pipeline whenever the source data changes."
)

DATASET_CATALOG = {
    "Historical (actual data)": {
        "slug": "historical",
        "predictions": PREDICTIONS_PATH,
        "summary": SUMMARY_PATH,
        "report": REPORT_MARKDOWN_PATH,
        "figures": FIGURES_DIR,
        "is_synthetic": False,
    }
}

if SYNTHETIC_SUMMARY_PATH.exists() and SYNTHETIC_PREDICTIONS_PATH.exists():
    DATASET_CATALOG["Synthetic scenario (test data)"] = {
        "slug": "synthetic",
        "predictions": SYNTHETIC_PREDICTIONS_PATH,
        "summary": SYNTHETIC_SUMMARY_PATH,
        "report": SYNTHETIC_MARKDOWN_PATH,
        "figures": SYNTHETIC_FIGURES_DIR,
        "is_synthetic": True,
    }

AVAILABLE_DATASETS = {
    name: cfg
    for name, cfg in DATASET_CATALOG.items()
    if cfg["summary"].exists() and cfg["predictions"].exists()
}

if not AVAILABLE_DATASETS:
    st.warning(
        "Outputs not found. Run `python Existing.py` to generate the baseline. Use `python run_synthetic_analysis.py` for the synthetic scenario."
    )
    st.stop()


@st.cache_data(show_spinner=False)
def load_predictions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    numeric_cols = df.select_dtypes(include=["float", "int"]).columns
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def load_summary(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@st.cache_data(show_spinner=False)
def load_markdown_report(path: Path) -> str:
    if not path.exists():
        return "Report markdown not found. Rerun the corresponding pipeline to regenerate it."
    return path.read_text(encoding="utf-8")


def render_metric(label: str, value: str, delta: str | None = None) -> None:
    st.metric(label, value, delta)


def render_chart_gallery(figures: Dict[str, str], base_dir: Path) -> None:
    if not figures:
        st.info("Charts were not exported. Re-run `python Existing.py` to generate them.")
        return
    cols = st.columns(2)
    for idx, (label, filename) in enumerate(figures.items()):
        figure_path = base_dir / filename
        if not figure_path.exists():
            continue
        pretty = label.replace("_", " ").title()
        with cols[idx % 2]:
            st.image(str(figure_path), caption=pretty, use_column_width=True)

def render_probability_histogram(df: pd.DataFrame, threshold: float) -> None:
    if df.empty:
        st.info("No customers match the current filters.")
        return
    chart = (
        alt.Chart(df)
        .transform_bin(
            ["prob_bucket", "prob_bucket_end"],
            "logistic_probability",
            bin=alt.Bin(maxbins=30),
        )
        .mark_bar(color="#4c78a8")
        .encode(
            x=alt.X("prob_bucket:Q", title="Predicted churn probability"),
            x2="prob_bucket_end:Q",
            y=alt.Y("count()", title="Customers"),
            tooltip=[alt.Tooltip("count()", title="Customers")],
        )
    )
    threshold_rule = alt.Chart(pd.DataFrame({"threshold": [threshold]})).mark_rule(color="crimson").encode(
        x="threshold"
    )
    st.altair_chart(chart + threshold_rule, use_container_width=True)
    st.caption(
        "Histogram shows how many customers fall into each probability bucket. The red line is the current threshold."
    )


dataset_names = list(AVAILABLE_DATASETS.keys())
default_index = dataset_names.index("Historical (actual data)") if "Historical (actual data)" in dataset_names else 0
selected_dataset = st.sidebar.selectbox("Dataset to display", dataset_names, index=default_index)
dataset_config = AVAILABLE_DATASETS[selected_dataset]

summary = load_summary(dataset_config["summary"])
predictions = load_predictions(dataset_config["predictions"])
report_markdown = load_markdown_report(dataset_config["report"])
figures_dir = dataset_config["figures"]
dataset_slug = dataset_config["slug"]
is_synthetic = dataset_config["is_synthetic"]
dataset_label = summary.get("dataset_label", selected_dataset)

st.sidebar.markdown(f"**Dataset:** {dataset_label}")
if is_synthetic:
    st.sidebar.warning("Synthetic scenario - generated test data, not actual customers.")
    st.warning("Synthetic scenario in view. Metrics reflect generated data for experimentation and are not real customer outcomes.")

st.markdown(f"**Dataset in view:** {dataset_label}")

st.sidebar.header("Filters")
threshold_default = float(summary.get("at_risk_threshold", 0.6))
prob_threshold = st.sidebar.slider(
    "Churn probability threshold",
    min_value=0.10,
    max_value=0.95,
    value=float(round(threshold_default, 2)),
    step=0.05,
)
st.sidebar.caption(
    "Move the slider to decide how sure the model must be before a customer is flagged."
    " A value of 0.70 means we only flag customers when the model is 70% confident they will churn."
)
plan_filter = st.sidebar.multiselect(
    "Plan types",
    options=sorted(predictions["plan_type"].unique()),
    default=list(sorted(predictions["plan_type"].unique())),
)

filtered = predictions[predictions["plan_type"].isin(plan_filter)].copy()
filtered["is_high_risk"] = filtered["logistic_probability"] >= prob_threshold
high_risk = filtered[filtered["is_high_risk"]]
st.sidebar.metric("Customers above threshold", f"{len(high_risk):,}")
st.sidebar.caption(
    "As you raise the threshold, this count drops because fewer customers meet the higher bar."
)

st.info(
    "**How to read the dashboard:** The slider controls how strict we are about calling someone high-risk."
    " The table and metrics update instantly, so you can test different intervention sizes."
)

render_probability_histogram(filtered, prob_threshold)

with st.expander("Executive summary preview"):
    st.markdown(report_markdown)

st.subheader(f"Business Pulse - {dataset_label}")
col1, col2, col3, col4 = st.columns(4)
with col1:
    render_metric("Customers", f"{int(summary['customer_count']):,}")
with col2:
    render_metric("Churn rate", f"{summary['churn_rate']:.1%}")
with col3:
    render_metric("Monthly revenue", f"${summary['monthly_recurring_revenue']:,.0f}")
with col4:
    render_metric(
        "High-risk customers",
        f"{len(high_risk):,}",
        delta=f"{prob_threshold:.0%} threshold",
    )

col5, col6 = st.columns(2)
with col5:
    render_metric(
        "Revenue at risk (6m, threshold)",
        f"${summary['revenue_at_risk_6m']:,.0f}",
    )
with col6:
    render_metric(
        "Revenue at risk (6m, weighted)",
        f"${summary['weighted_revenue_at_risk_6m']:,.0f}",
    )

st.divider()

st.subheader(f"High-Risk Customer List - {dataset_label}")
st.caption(
    "Use filters to target specific plan types or adjust the probability cut-off to size retention campaigns."
)
st.dataframe(
    high_risk[
        [
            "customer_id",
            "plan_type",
            "logistic_probability",
            "tenure_months",
            "monthly_spend",
            "support_tickets",
            "last_login_days",
            "rule_flag",
        ]
    ].sort_values("logistic_probability", ascending=False),
    use_container_width=True,
)

st.download_button(
    "Download filtered customers",
    data=high_risk.to_csv(index=False).encode("utf-8"),
    file_name=f"high_risk_customers_{dataset_slug}.csv",
    mime="text/csv",
)

st.divider()

segment_data = summary.get("segment_highlights", {})
if segment_data:
    st.subheader("Segment Highlights")
    tabs = st.tabs(["Tenure", "Inactivity", "Support", "Plan Profile"])
    tab_keys = ["tenure", "inactivity", "support", "plan_profile"]
    for tab, key in zip(tabs, tab_keys):
        with tab:
            payload = segment_data.get(key)
            if payload:
                df_segment = pd.DataFrame(payload)
                st.dataframe(df_segment, use_container_width=True)
            else:
                st.info("Segment summary unavailable. Rerun the pipeline to refresh outputs.")
else:
    st.info("Segment summaries were not exported. Rerun the pipeline to refresh outputs.")

st.divider()

st.subheader("Model Diagnostics")
rule_metrics = summary.get("rule_metrics", {})
logistic_metrics = summary.get("logistic_metrics", {})
metrics_col1, metrics_col2 = st.columns(2)
with metrics_col1:
    st.markdown("**Rule-based heuristic**")
    for key, value in rule_metrics.items():
        st.write(f"{key.replace('_', ' ').title()}: {value:.2f}")
with metrics_col2:
    st.markdown("**Logistic regression**")
    for key, value in logistic_metrics.items():
        st.write(f"{key.replace('_', ' ').title()}: {value:.2f}")

feature_ranking_payload = summary.get("feature_ranking")
if feature_ranking_payload:
    st.markdown("**Feature ranking (top drivers)**")
    ranking_df = pd.DataFrame(feature_ranking_payload)
    st.dataframe(ranking_df, use_container_width=True)

coeff_payload = summary.get("logistic_coefficients")
if coeff_payload:
    st.markdown("**Logistic coefficients (odds ratios)**")
    coeff_df = pd.DataFrame(coeff_payload)
    st.dataframe(coeff_df, use_container_width=True)

st.divider()

st.subheader("Visual Gallery")
render_chart_gallery(summary.get("figures", {}), figures_dir)

st.caption(
    "Need to refresh? Rerun `python Existing.py` to regenerate metrics and visuals, then reload this page."
)
