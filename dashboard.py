"""Streamlit dashboard for the customer churn analysis deliverable.

Run `streamlit run dashboard.py` after executing `python Existing.py`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
REPORT_DIR = BASE_DIR / "reports"
PREDICTIONS_PATH = REPORT_DIR / "churn_risk_scoring.csv"
SUMMARY_PATH = REPORT_DIR / "metrics_summary.json"
FIGURES_DIR = REPORT_DIR / "figures"

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

if not PREDICTIONS_PATH.exists() or not SUMMARY_PATH.exists():
    st.warning(
        "Outputs not found. Run `python Existing.py` first to generate the report, risk scores, and summary files."
    )
    st.stop()


@st.cache_data(show_spinner=False)
def load_predictions() -> pd.DataFrame:
    df = pd.read_csv(PREDICTIONS_PATH)
    numeric_cols = df.select_dtypes(include=["float", "int"]).columns
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def load_summary() -> Dict:
    with SUMMARY_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def render_metric(label: str, value: str, delta: str | None = None) -> None:
    st.metric(label, value, delta)


def render_chart_gallery(figures: Dict[str, str]) -> None:
    if not figures:
        st.info("Charts were not exported. Re-run `python Existing.py` to generate them.")
        return
    cols = st.columns(2)
    for idx, (label, filename) in enumerate(figures.items()):
        figure_path = FIGURES_DIR / filename
        if not figure_path.exists():
            continue
        pretty = label.replace("_", " ").title()
        with cols[idx % 2]:
            st.image(str(figure_path), caption=pretty, use_column_width=True)


summary = load_summary()
predictions = load_predictions()

st.sidebar.header("Filters")
threshold_default = float(summary.get("at_risk_threshold", 0.6))
prob_threshold = st.sidebar.slider(
    "Churn probability threshold",
    min_value=0.10,
    max_value=0.95,
    value=float(round(threshold_default, 2)),
    step=0.05,
)
plan_filter = st.sidebar.multiselect(
    "Plan types",
    options=sorted(predictions["plan_type"].unique()),
    default=list(sorted(predictions["plan_type"].unique())),
)

filtered = predictions[predictions["plan_type"].isin(plan_filter)].copy()
filtered["is_high_risk"] = filtered["logistic_probability"] >= prob_threshold
high_risk = filtered[filtered["is_high_risk"]]

st.subheader("Business Pulse")
col1, col2, col3, col4 = st.columns(4)
with col1:
    render_metric(
        "Customers",
        f"{int(summary['customer_count']):,}",
    )
with col2:
    render_metric(
        "Churn rate",
        f"{summary['churn_rate']:.1%}",
    )
with col3:
    render_metric(
        "Monthly revenue",
        f"${summary['monthly_recurring_revenue']:,.0f}",
    )
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

st.subheader("High-Risk Customer List")
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
    file_name="high_risk_customers.csv",
    mime="text/csv",
)

st.divider()

segment_data = summary.get("segment_highlights", {})
if segment_data:
    st.subheader("Segment Highlights")
    tabs = st.tabs([
        "Tenure",
        "Inactivity",
        "Support",
        "Plan Profile",
    ])
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
render_chart_gallery(summary.get("figures", {}))

st.caption(
    "Need to refresh? Rerun `python Existing.py` to regenerate metrics and visuals, then reload this page."
)

