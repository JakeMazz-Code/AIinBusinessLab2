# AI in Business Lab 2

This project analyzes customer churn for a streaming service dataset. It provides:

- Automated descriptive analytics and executive report generation (`Existing.py`).
- Visualizations, HTML reports, and risk scoring outputs stored in the `reports/` directory.
- A Streamlit dashboard (`dashboard.py`) for interactive exploration of high-risk customers and model diagnostics.
- A synthetic data generator (`generate_synthetic_churn.py`) to stress-test the modeling workflow.

## Quick Start

```bash
python -m pip install -r requirements.txt
python Existing.py
streamlit run dashboard.py
```

The pipeline saves refreshed markdown/HTML reports, figures, customer-level scores, and summary JSON under `reports/`.

## Dashboard Tips

- The **churn probability threshold slider** controls how confident the model must be before a customer is flagged (e.g., `0.70` = 70% confidence). The histogram updates instantly to show how many customers sit above the current threshold.
- Expand **Executive summary preview** to read the latest markdown report in-app without leaving the dashboard.
- Use the plan-type filter or download button to export targeted customer lists for retention outreach.

## Sharing Results

`python Existing.py` creates both a markdown report (`reports/churn_analysis_report.md`) and an HTML copy (`reports/churn_analysis_report.html`).

- Send the HTML file directly, or convert the markdown to PDF with Pandoc:

```bash
pandoc reports/churn_analysis_report.md -o churn_analysis_report.pdf
```

## Generate Synthetic Data

Quickly create stress-test datasets that mirror the original schema and run a full analysis without swapping files:

`ash
python run_synthetic_analysis.py --rows 1000
`

That command learns feature distributions from customer_churn_data.csv, writes the synthetic dataset (default synthetic_customer_churn.csv), and saves all outputs under eports/synthetic/. Point Streamlit at that directory to compare findings.

Need more control? Generate only the data first:

`ash
python generate_synthetic_churn.py --rows 1000 --output synthetic_customer_churn.csv
`

Then run the main pipeline manually with python Existing.py after temporarily redirecting Existing.DATA_PATH or swapping the CSV.

