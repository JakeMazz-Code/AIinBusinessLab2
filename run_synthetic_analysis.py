"""Run the churn analysis pipeline against a freshly generated synthetic dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import Existing
import generate_synthetic_churn as synth

DEFAULT_ROWS = 1000
DEFAULT_OUTPUT = Path('synthetic_customer_churn.csv')
DEFAULT_REPORT_DIR = Path('reports') / 'synthetic'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic churn data and run analysis")
    parser.add_argument('--rows', type=int, default=DEFAULT_ROWS, help='Number of synthetic rows to generate')
    parser.add_argument('--source', type=Path, default=synth.DATA_PATH, help='Source dataset used to learn distributions')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT, help='Synthetic CSV output path')
    parser.add_argument('--report-dir', type=Path, default=DEFAULT_REPORT_DIR, help='Destination directory for reports')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    return parser.parse_args()


def configure_existing(data_path: Path, report_dir: Path) -> None:
    Existing.DATA_PATH = data_path
    Existing.REPORT_DIR = report_dir
    Existing.FIGURES_DIR = report_dir / 'figures'
    Existing.REPORT_PATH = report_dir / 'churn_analysis_report.md'
    Existing.REPORT_HTML_PATH = report_dir / 'churn_analysis_report.html'
    Existing.PREDICTIONS_PATH = report_dir / 'churn_risk_scoring.csv'
    Existing.SUMMARY_PATH = report_dir / 'metrics_summary.json'


def main() -> None:
    args = parse_args()
    report_dir = args.report_dir
    report_dir.mkdir(parents=True, exist_ok=True)

    source_df = synth.load_source_data(args.source)
    synthetic_df = synth.synthesize_dataset(source_df, args.rows, seed=args.seed)
    synthetic_df.to_csv(args.output, index=False)

    configure_existing(args.output, report_dir)
    Existing.DATA_LABEL = f"Synthetic cohort ({args.rows} rows)"
    Existing.ensure_output_dirs()
    Existing.main()

    print('\nSynthetic analysis completed:')
    print(f'  Data file: {args.output.resolve()}')
    print(f'  Reports written to: {report_dir.resolve()}')


if __name__ == '__main__':
    main()
