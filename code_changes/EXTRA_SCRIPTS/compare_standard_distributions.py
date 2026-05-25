#!/usr/bin/env python3
"""
Compare packet-level distributions between:
  A) DockerSetup_standard_prompts/data/copy-collector-*  (no mitigations, obfuscation OFF)
  B) data/gpt4o-mini_STANDARD_WITH_MITIGATIONS/          (with mitigations, obfuscation ON)
  C) data/gpt4o-mini_MULTITHEME_WITH_MITIGATIONS/        (multitheme, obfuscation OFF)

B and C are randomly sampled to match the number of entries in A.
Plots: packet sizes, inter-packet time diffs, sequence lengths.
"""

import json
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).parent.parent
DOCKER_BASE = SCRIPT_DIR / 'DockerSetup_standard_prompts' / 'data'
MITIGATIONS_DIR = SCRIPT_DIR / 'data' / 'gpt4o-mini_STANDARD_WITH_MITIGATIONS'
MULTITHEME_DIR = SCRIPT_DIR / 'data' / 'gpt4o-mini_MULTITHEME_WITH_MITIGATIONS'
SEED = 42

# Colorblind-safe (Wong 2011)
COLORS = ['#0072B2', '#D55E00', '#009E73']


def load_docker(base: Path):
    entries = []
    for collector in sorted(base.glob('copy-collector-*')):
        for jf in sorted(collector.glob('*.json')):
            with open(jf) as f:
                data = json.load(f)
            entries.extend(data)
    return entries


def load_folder(folder: Path):
    entries = []
    for jf in sorted(folder.glob('*.json')):
        with open(jf) as f:
            data = json.load(f)
        entries.extend(data)
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


def plot_dist(ax, datasets, title, xlabel, bins=60, log_scale=False):
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
        ax.hist(arr, bins=bins_edges, alpha=0.5, color=color, label=label, density=True)

    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(axis='y', linestyle='--', alpha=0.4)


def main():
    print('Loading Docker standard (no mitigations, obfuscation OFF) …')
    docker_entries = load_docker(DOCKER_BASE)
    n = len(docker_entries)
    print(f'  {n} entries')

    print('Loading standard WITH_MITIGATIONS (obfuscation ON) …')
    mit_entries = load_folder(MITIGATIONS_DIR)
    sampled_mit = sample_to(mit_entries, n)
    print(f'  {len(mit_entries)} total → sampled {len(sampled_mit)}')

    print('Loading multitheme (obfuscation OFF) …')
    multi_entries = load_folder(MULTITHEME_DIR)
    sampled_multi = sample_to(multi_entries, n, seed=SEED + 1)
    print(f'  {len(multi_entries)} total → sampled {len(sampled_multi)}')

    print('Extracting features …')
    a_sizes, a_times, a_seqs = extract(docker_entries)
    b_sizes, b_times, b_seqs = extract(sampled_mit)
    c_sizes, c_times, c_seqs = extract(sampled_multi)

    label_a = f'Standard — no mitigations (n={n})'
    label_b = f'Standard — with mitigations (n={len(sampled_mit)})'
    label_c = f'Multitheme — no mitigations (n={len(sampled_multi)})'

    datasets_sizes = [(a_sizes, label_a, COLORS[0]),
                      (b_sizes, label_b, COLORS[1]),
                      (c_sizes, label_c, COLORS[2])]
    datasets_times = [(a_times, label_a, COLORS[0]),
                      (b_times, label_b, COLORS[1]),
                      (c_times, label_c, COLORS[2])]
    datasets_seqs  = [(a_seqs,  label_a, COLORS[0]),
                      (b_seqs,  label_b, COLORS[1]),
                      (c_seqs,  label_c, COLORS[2])]

    fig, axes = plt.subplots(1, 3, figsize=(17, 4))
    fig.suptitle('Packet-level distribution comparison', fontsize=13, y=1.01)

    plot_dist(axes[0], datasets_sizes, 'Packet sizes', 'Bytes', log_scale=True)
    plot_dist(axes[1], datasets_times, 'Inter-packet time diffs', 'Seconds', log_scale=True)
    plot_dist(axes[2], datasets_seqs,  'Sequence lengths', 'Packets per conversation')

    out_path = SCRIPT_DIR / 'EXTRA_SCRIPTS' / 'standard_distribution_comparison.png'
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'\nSaved: {out_path}')

    print('\nSummary stats:')
    for label, sizes, times, seqs in [
        (label_a, a_sizes, a_times, a_seqs),
        (label_b, b_sizes, b_times, b_seqs),
        (label_c, c_sizes, c_times, c_seqs),
    ]:
        print(f'\n  {label}')
        print(f'    packet sizes : mean={np.mean(sizes):.1f}  median={np.median(sizes):.1f}  p99={np.percentile(sizes, 99):.1f}')
        print(f'    time diffs   : mean={np.mean(times):.4f}  median={np.median(times):.4f}')
        print(f'    seq lengths  : mean={np.mean(seqs):.1f}  median={np.median(seqs):.1f}  max={max(seqs)}')


if __name__ == '__main__':
    main()
