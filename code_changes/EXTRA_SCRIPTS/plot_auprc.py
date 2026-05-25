#!/usr/bin/env python3
"""
Plot AUPRC scores per run for each model in a results folder.
Usage: python plot_auprc.py <results_folder> [results_folder2 ...]
       python plot_auprc.py --title "My Graph" <results_folder>
       python plot_auprc.py  (auto-discovers all RESULTS_* and ALL_RESULTS/ folders)
"""

import argparse
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import statistics

SCRIPT_DIR = Path(__file__).parent.parent
METRICS_FILE = 'confusion_matrix_metrics.txt'
MODELS = ['PERCEPTRON', 'RNN', 'KNN', 'LGBM']
# Colorblind-safe palette (Wong 2011)
COLORS = {'PERCEPTRON': '#0072B2', 'RNN': '#D55E00', 'KNN': '#009E73', 'LGBM': '#E69F00'}
MARKERS = {'PERCEPTRON': 'o', 'RNN': 's', 'KNN': '^', 'LGBM': 'D'}
LABELS = {'PERCEPTRON': 'Perceptron', 'RNN': 'RNN', 'KNN': 'KNN', 'LGBM': 'LightGBM'}
MIN_LABEL_GAP = 0.04  # minimum vertical separation between median labels (in AUPRC units)


def parse_auprc(metrics_path: Path) -> float:
    with open(metrics_path) as f:
        for line in f:
            if line.startswith('AUPRC:') or line.startswith('AUPRC (macro):'):
                return float(line.split(':', 1)[1].strip())
    raise ValueError(f'AUPRC not found in {metrics_path}')


def load_results(results_dir: Path) -> dict:
    """Returns {model: {run: auprc}} for all available runs."""
    data = {}
    for model in MODELS:
        model_dir = results_dir / model
        if not model_dir.exists():
            continue
        runs = {}
        for run_dir in sorted(model_dir.iterdir()):
            if not run_dir.is_dir() or not run_dir.name.startswith('run'):
                continue
            metrics_path = run_dir / METRICS_FILE
            if not metrics_path.exists():
                continue
            run_num = int(run_dir.name.replace('run', ''))
            runs[run_num] = parse_auprc(metrics_path)
        if runs:
            data[model] = runs
    return data


def resolve_label_positions(medians: list[tuple[float, str, str]]) -> list[tuple[float, float, str, str]]:
    """
    Given [(median_val, model, color), ...], return [(median_val, label_y, model, color), ...]
    with label_y adjusted so no two labels are closer than MIN_LABEL_GAP.
    Sorts by median, then nudges upward greedily.
    """
    sorted_items = sorted(medians, key=lambda x: x[0])
    positions = []
    last_y = -999.0

    for median, model, color in sorted_items:
        label_y = median
        if label_y - last_y < MIN_LABEL_GAP:
            label_y = last_y + MIN_LABEL_GAP
        last_y = label_y
        positions.append((median, label_y, model, color))

    return positions


def plot(results_dir: Path, title: str = None, max_runs: int = None):
    data = load_results(results_dir)
    if not data:
        print(f'[SKIP] No results found in {results_dir}')
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.subplots_adjust(right=0.70)

    median_items = []

    for model, runs in data.items():
        xs = sorted(runs.keys())
        if max_runs is not None:
            xs = xs[:max_runs]
        ys = [runs[x] for x in xs]
        color = COLORS.get(model)

        ax.plot(
            xs, ys,
            label=LABELS.get(model, model),
            color=color,
            marker=MARKERS.get(model),
            linewidth=2,
            markersize=7,
        )

        median = statistics.median(ys)
        ax.axhline(
            median,
            color=color,
            linestyle='--',
            linewidth=1.2,
            alpha=0.7,
        )
        median_items.append((median, model, color))

    # Place median labels outside the right edge with overlap resolution
    for median, label_y, model, color in resolve_label_positions(median_items):
        ax.annotate(
            f'Median = {median:.3f}  ({LABELS.get(model, model)})',
            xy=(1.0, label_y),
            xycoords=('axes fraction', 'data'),
            xytext=(8, 0),
            textcoords='offset points',
            color=color,
            fontsize=10,
            va='center',
            clip_on=False,
        )

    ax.set_title(title if title else results_dir.name, fontsize=13)
    ax.set_xlabel('Run', fontsize=12)
    ax.set_ylabel('AUPRC Score', fontsize=12)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.set_ylim(0, 1)
    ax.legend(fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    out_path = results_dir / (f'auprc_per_run_{max_runs}.png' if max_runs else 'auprc_per_run.png')
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f'Saved: {out_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('folders', nargs='*', help='Results folders to plot')
    parser.add_argument('--title', '-t', default=None,
                        help='Custom graph title (only used when plotting a single folder)')
    parser.add_argument('--runs', '-n', type=int, default=None,
                        help='Only use the first N runs (saves as auprc_per_run_N.png)')
    args = parser.parse_args()

    if args.folders:
        folders = [Path(p) for p in args.folders]
    else:
        folders = sorted(SCRIPT_DIR.glob('RESULTS_*')) + sorted((SCRIPT_DIR / 'ALL_RESULTS').glob('*'))

    if not folders:
        print('No results folders found. Pass a folder path as an argument.')
        sys.exit(1)

    if args.title and len(folders) > 1:
        print('[WARN] --title is ignored when plotting multiple folders')

    for folder in folders:
        if not folder.is_dir():
            print(f'[SKIP] Not a directory: {folder}')
            continue
        title = args.title if (args.title and len(folders) == 1) else None
        plot(folder, title=title, max_runs=args.runs)


if __name__ == '__main__':
    main()
