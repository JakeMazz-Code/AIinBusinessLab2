# AI in Business Lab 2

This project analyzes customer churn for a streaming service dataset. It provides:

- Automated descriptive analytics and executive report generation (`Existing.py`).
- Visualizations and risk scoring outputs stored in the `reports/` directory.
- A Streamlit dashboard (`dashboard.py`) for interactive exploration of high-risk customers and model diagnostics.

## Quick Start

```bash
python -m pip install -r requirements.txt
python Existing.py
streamlit run dashboard.py
```

The generated markdown report, figures, and risk tables are saved under `reports/`.

## Dashboard Tips

- The **churn probability threshold slider** controls how confident the model must be before a customer is flagged. For example, a value of `0.70` means only customers with a 70% or higher predicted chance of churning are marked as high-risk.
- Use the plan-type filter to drill into specific subscription tiers and see how the high-risk list changes.

## Sharing Results

`python Existing.py` now creates both a markdown report (`reports/churn_analysis_report.md`) and an HTML copy (`reports/churn_analysis_report.html`). Send the HTML file directly or convert the markdown to PDF with a tool like Pandoc:

```bash
pandoc reports/churn_analysis_report.md -o churn_analysis_report.pdf
```

The `reports` folder also includes CSV exports and ready-made figures for slide decks.
