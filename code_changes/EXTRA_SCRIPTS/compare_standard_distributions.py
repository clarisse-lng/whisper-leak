#!/usr/bin/env python3
"""
Compare packet-level distributions between:
  A) DockerSetup_standard_prompts/data/copy-collector-*  (no mitigations, obfuscation OFF)
  B) data/gpt4o-mini_STANDARD_WITH_MITIGATIONS/          (with mitigations, obfuscation ON)
  C) data/gpt4o-mini_MULTITHEME_WITH_MITIGATIONS/        (multitheme, obfuscation OFF)

B and C are randomly sampled to match the number of entries in A.

Usage:
  python compare_standard_distributions.py
  python compare_standard_distributions.py --no-multitheme
  python compare_standard_distributions.py --plots sizes seqs
  python compare_standard_distributions.py --no-multitheme --plots times
"""

import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

SCRIPT_DIR = Path(__file__).parent.parent
DOCKER_BASE = SCRIPT_DIR / 'DockerSetup_standard_prompts' / 'data'
MITIGATIONS_DIR = SCRIPT_DIR / 'data' / 'gpt4o-mini_STANDARD_WITH_MITIGATIONS'
MULTITHEME_DIR = SCRIPT_DIR / 'data' / 'gpt4o-mini_MULTITHEME_WITH_MITIGATIONS'
SEED = 42

COLORS = ['#0072B2', '#D55E00', '#009E73']

PLOT_OPTIONS = {
    'sizes': ('Taille des paquets (octets)',      'Octets',                     True),
    'times': ('Temps inter-paquets (secondes)',   'Secondes',                   True),
    'seqs':  ('Longueur de séquence',             'Paquets par requête',         False),
}

YLABELS = {
    'sizes': 'Nombre de paquets',
    'times': 'Nombre de paires de paquets',
    'seqs':  'Nombre de requêtes',
}


def load_docker(base: Path):
    entries = []
    for collector in sorted(base.glob('copy-collector-*')):
        for jf in sorted(collector.glob('*.json')):
            entries.extend(json.load(open(jf)))
    return entries


def load_folder(folder: Path):
    entries = []
    for jf in sorted(folder.glob('*.json')):
        entries.extend(json.load(open(jf)))
    return entries


def sample_to(entries, n, seed=SEED):
    if len(entries) <= n:
        return entries
    random.seed(seed)
    return random.sample(entries, n)


def extract(entries):
    all_sizes, all_times, seq_lengths = [], [], []
    for e in entries:
        sizes = e.get('data_lengths', [])
        times = e.get('time_diffs', [])
        all_sizes.extend(sizes)
        all_times.extend(times)
        seq_lengths.append(len(sizes))
    return all_sizes, all_times, seq_lengths


def plot_dist(ax, datasets, title, xlabel, ylabel='Count', bins=60, log_scale=False, use_log_ticks=False):
    arrays = [np.array(vals, dtype=float) for vals, _, _ in datasets]

    if log_scale:
        arrays = [a[a > 0] for a in arrays]
        lo = min(a.min() for a in arrays if len(a))
        hi = max(a.max() for a in arrays if len(a))
        bins_edges = np.logspace(np.log10(lo), np.log10(hi), bins)
        ax.set_xscale('log')
    else:
        lo = min(np.percentile(a, 1) for a in arrays)
        hi = max(np.percentile(a, 99) for a in arrays)
        bins_edges = np.linspace(lo, hi, bins)

    for arr, (_, label, color) in zip(arrays, datasets):
        ax.hist(arr, bins=bins_edges, alpha=0.55, color=color, label=label)

    if log_scale and not use_log_ticks:
        ax.xaxis.set_major_locator(ticker.LogLocator(base=10, subs=[1, 2, 5]))
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:g}'))

    formatter = ticker.ScalarFormatter()
    formatter.set_powerlimits((0, 0))
    ax.yaxis.set_major_formatter(formatter)
    ax.figure.canvas.draw()  # needed to resolve the offset text
    offset = ax.yaxis.get_major_formatter().get_offset()
    if offset:
        ax.yaxis.offsetText.set_visible(False)

    ax.set_title(title, fontsize=13, fontweight='bold', pad=6)
    ax.set_xlabel(xlabel, fontsize=12)
    ylabel_full = f'{ylabel} ({offset})' if offset else ylabel
    ax.set_ylabel(ylabel_full, fontsize=12)
    ax.tick_params(labelsize=10)
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(axis='y', linestyle='--', alpha=0.4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-multitheme', action='store_true',
                        help='Skip multitheme dataset (compare only A vs B)')
    parser.add_argument('--plots', nargs='+', choices=list(PLOT_OPTIONS.keys()),
                        default=['sizes', 'times'],
                        help='Which plots to include: sizes, times, seqs (default: all three)')
    args = parser.parse_args()

    selected = args.plots

    print('Loading Docker standard (no mitigations, obfuscation OFF) …')
    docker_entries = load_docker(DOCKER_BASE)
    n = len(docker_entries)
    print(f'  {n} entries')

    print('Loading standard WITH_MITIGATIONS (obfuscation ON) …')
    mit_entries = load_folder(MITIGATIONS_DIR)
    sampled_mit = sample_to(mit_entries, n)
    print(f'  {len(mit_entries)} total → sampled {len(sampled_mit)}')

    print('Extracting features …')
    a_sizes, a_times, a_seqs = extract(docker_entries)
    b_sizes, b_times, b_seqs = extract(sampled_mit)

    label_a = 'Standard — sans obfuscation'
    label_b = 'Standard — avec obfuscation'

    data = {
        'sizes': [a_sizes, b_sizes],
        'times': [a_times, b_times],
        'seqs':  [a_seqs,  b_seqs],
    }
    labels = [label_a, label_b]
    summary = [(label_a, a_sizes, a_times, a_seqs), (label_b, b_sizes, b_times, b_seqs)]

    if not args.no_multitheme:
        print('Loading multitheme (obfuscation OFF) …')
        multi_entries = load_folder(MULTITHEME_DIR)
        sampled_multi = sample_to(multi_entries, n, seed=SEED + 1)
        print(f'  {len(multi_entries)} total → sampled {len(sampled_multi)}')
        c_sizes, c_times, c_seqs = extract(sampled_multi)
        label_c = 'Multithème — sans obfuscation'
        data['sizes'].append(c_sizes)
        data['times'].append(c_times)
        data['seqs'].append(c_seqs)
        labels.append(label_c)
        summary.append((label_c, c_sizes, c_times, c_seqs))

    n_plots = len(selected)
    fig, axes = plt.subplots(1, n_plots, figsize=(5.5 * n_plots, 4))
    if n_plots == 1:
        axes = [axes]

    for ax, key in zip(axes, selected):
        title, xlabel, log_scale = PLOT_OPTIONS[key]
        datasets = [(vals, lbl, COLORS[i]) for i, (vals, lbl) in enumerate(zip(data[key], labels))]
        plot_dist(ax, datasets, title, xlabel, ylabel=YLABELS[key], log_scale=log_scale,
                  use_log_ticks=(key == 'times'))

    plt.tight_layout()

    suffix_parts = []
    if args.no_multitheme:
        suffix_parts.append('two')
    if set(selected) != set(PLOT_OPTIONS.keys()):
        suffix_parts.append('_'.join(selected))
    suffix = ('_' + '_'.join(suffix_parts)) if suffix_parts else ''

    out_path = SCRIPT_DIR / 'EXTRA_SCRIPTS' / f'standard_distribution_comparison{suffix}.png'
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'\nSaved: {out_path}')

    print('\nSummary stats:')
    for label, sizes, times, seqs in summary:
        print(f'\n  {label}')
        print(f'    packet sizes : mean={np.mean(sizes):.1f}  median={np.median(sizes):.1f}  p99={np.percentile(sizes, 99):.1f}')
        print(f'    time diffs   : mean={np.mean(times):.4f}  median={np.median(times):.4f}')
        print(f'    seq lengths  : mean={np.mean(seqs):.1f}  median={np.median(seqs):.1f}  max={max(seqs)}')


if __name__ == '__main__':
    main()
