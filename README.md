# AI in Business Lab 2

Customer churn analytics for a subscription service. The toolkit includes:

- `Existing.py`: end-to-end pipeline that loads data, engineers metrics, trains classifiers, and produces reports/charts.
- `dashboard.py`: Streamlit app for interactive exploration of churn drivers, risk scoring, and scenario comparisons.
- `generate_synthetic_churn.py`: synthesizes churn-like cohorts for stress testing.
- `run_synthetic_analysis.py`: one-shot command that generates synthetic data and runs the full pipeline against it.

All outputs are written to `reports/` (historical baseline) and `reports/synthetic/` (synthetic scenario).

## Quick Start

```bash
python -m pip install -r requirements.txt
python Existing.py
streamlit run dashboard.py
```

The pipeline refreshes markdown/HTML reports, figures, customer-level scores, and a JSON summary under `reports/`.

## Synthetic Scenario

Generate a synthetic cohort and run the full analysis in one step:

```bash
python run_synthetic_analysis.py --rows 1000
```

- Learns feature distributions from `customer_churn_data.csv`.
- Saves the synthetic CSV (default `synthetic_customer_churn.csv`).
- Writes reports and metrics to `reports/synthetic/` so the dashboard can pick them up automatically.

Need only the data? Generate it first, then point the pipeline at the new file:

```bash
python generate_synthetic_churn.py --rows 1000 --output synthetic_customer_churn.csv
python Existing.py
```

## Dashboard Tips

- Use the **Dataset to display** selector (sidebar) to toggle between the historical baseline and the synthetic scenario. When synthetic data is active, the app highlights that it is test data.
- Tune the **churn probability threshold** slider to control how confident the model must be before flagging high-risk customers. The histogram shows how many customers fall above the threshold.
- Expand **Executive summary preview** to read the latest markdown report in-app.
- Filter by plan type or download a CSV of high-risk customers for targeted outreach.

## Sharing Results

`python Existing.py` creates both a markdown report (`reports/churn_analysis_report.md`) and an HTML copy (`reports/churn_analysis_report.html`).

Convert the markdown to PDF with Pandoc if needed:

```bash
pandoc reports/churn_analysis_report.md -o churn_analysis_report.pdf
```

## Troubleshooting

- If the dashboard can't find outputs, rerun `python Existing.py` (historical) or `python run_synthetic_analysis.py` (synthetic).
- Keep dependencies in sync with `python -m pip install -r requirements.txt` before running the pipelines.
