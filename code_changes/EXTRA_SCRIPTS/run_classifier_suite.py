#!/usr/bin/env python3
"""
Run all classifiers N times on a data folder and save results to ALL_RESULTS/.

Works for any flat data folder (standard, Docker-merged, multitheme, etc.).
For Docker data, run merge_docker_collectors.py first to produce the flat folder.

Usage:
  python run_classifier_suite.py -i data/gpt4o-mini_STANDARD_WITH_MITIGATIONS
  python run_classifier_suite.py -i data/docker_standard_merged -n 10
  python run_classifier_suite.py -i data/multitheme --skip KNN LGBM --run-range 6-10
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.parent
NUM_RUNS = 5
MODELS = ['PERCEPTRON', 'RNN', 'KNN', 'LGBM']


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run classifiers N times on a data folder.'
    )
    parser.add_argument('-i', '--input-folder', required=True,
                        help='Data folder containing JSON files (relative to repo root)')
    parser.add_argument('-c', '--chatbot', default=None,
                        help='Chatbot name — auto-detected from folder contents if omitted')
    parser.add_argument('-p', '--prompts', default='./prompts/standard/prompts.json',
                        help='Prompts JSON file (default: prompts/standard/prompts.json)')
    parser.add_argument('-n', '--runs', type=int, default=NUM_RUNS,
                        help=f'Number of runs per model (default: {NUM_RUNS})')
    parser.add_argument('-r', '--run-range', default=None, metavar='RANGE',
                        help='Run range, e.g. --run-range 6-10 or --run-range 3')
    parser.add_argument('-e', '--epochs', type=int, default=100)
    parser.add_argument('-b', '--batchsize', type=int, default=32)
    parser.add_argument('-l', '--learningrate', type=float, default=0.0001)
    parser.add_argument('-P', '--patience', type=int, default=20)
    parser.add_argument('-t', '--testsize', type=int, default=20)
    parser.add_argument('-v', '--validsize', type=int, default=5)
    parser.add_argument('-ds', '--downsample', type=float, default=1.0)
    parser.add_argument('--skip', nargs='+', metavar='MODEL',
                        help='Models to skip, e.g. --skip KNN LGBM')
    parser.add_argument('--staging-dir', default='results', metavar='DIR',
                        help='Staging directory used during training (default: results). '
                             'Change to run two experiments simultaneously.')
    return parser.parse_args()


def detect_chatbot(folder: Path) -> str:
    # Try PCAP files first: {hash}_{trial}_{ChatbotName}.pcap
    pcap_names = {f.stem.rsplit('_', 1)[-1] for f in folder.glob('*.pcap')}
    if len(pcap_names) == 1:
        return pcap_names.pop()

    # Try JSON files: {ChatbotName}.json or {ChatbotName}_{n}.json (Docker-merged)
    json_stems = [f.stem for f in folder.glob('*.json')]
    # Strip trailing _{digits} suffix
    candidates = set()
    for stem in json_stems:
        parts = stem.rsplit('_', 1)
        if len(parts) == 2 and parts[1].isdigit():
            candidates.add(parts[0])
        else:
            candidates.add(stem)
    if len(candidates) == 1:
        return candidates.pop()

    print(f'[ERROR] Could not auto-detect chatbot name from {folder}')
    print('Please specify with -c.')
    sys.exit(1)


def parse_run_range(run_range: str) -> range:
    try:
        if '-' in run_range:
            start, end = run_range.split('-', 1)
            return range(int(start), int(end) + 1)
        else:
            n = int(run_range)
            return range(n, n + 1)
    except ValueError:
        print(f'[ERROR] Invalid --run-range value "{run_range}". Use e.g. 6-10 or 3.')
        sys.exit(1)


def build_command(model_type, seed, args):
    return [
        sys.executable,
        str(SCRIPT_DIR / 'whisper_leak_train.py'),
        '-c', args.chatbot,
        '-m', model_type,
        '-s', str(seed),
        '-i', args.input_folder,
        '-p', args.prompts,
        '-e', str(args.epochs),
        '-b', str(args.batchsize),
        '-l', str(args.learningrate),
        '-P', str(args.patience),
        '-t', str(args.testsize),
        '-v', str(args.validsize),
        '-ds', str(args.downsample),
        '--results-dir', args.staging_dir,
    ]


def copy_non_csv_results(dst_dir: Path, results_src: Path):
    dst_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for f in results_src.iterdir():
        if f.is_file() and f.suffix != '.csv':
            shutil.copy2(f, dst_dir / f.name)
            copied.append(f.name)
    return copied


def header(msg):
    bar = '=' * 60
    print(f'\n{bar}\n  {msg}\n{bar}')


def main():
    args = parse_args()

    folder = SCRIPT_DIR / args.input_folder
    if not folder.exists():
        print(f'[ERROR] Input folder not found: {folder}')
        sys.exit(1)

    if args.chatbot is None:
        args.chatbot = detect_chatbot(folder)
        print(f'Auto-detected chatbot: {args.chatbot}')

    results_src = SCRIPT_DIR / args.staging_dir
    output_base = SCRIPT_DIR / 'ALL_RESULTS' / folder.name

    skip = {m.upper() for m in (args.skip or [])}
    models = [m for m in MODELS if m not in skip]

    run_range = parse_run_range(args.run_range) if args.run_range else range(1, args.runs + 1)

    header(f'WhisperLeak Experiments  |  chatbot: {args.chatbot}')
    print(f'Data folder      : {folder}')
    print(f'Output directory : {output_base}')
    print(f'Runs             : {args.run_range or f"1-{args.runs}"}')
    print(f'Models           : {", ".join(models)}')
    if skip:
        print(f'Skipping         : {", ".join(skip)}')

    for model_type in models:
        for run in run_range:
            seed = run
            label = f'{model_type}  run {run}  (seed={seed})'
            header(label)

            dst_dir = output_base / model_type / f'run{run}'
            if dst_dir.exists() and any(dst_dir.iterdir()):
                print(f'[WARNING] {dst_dir} already contains results.')
                answer = input('Overwrite? [y/N] ').strip().lower()
                if answer != 'y':
                    print('Skipping.')
                    continue

            if results_src.exists():
                shutil.rmtree(results_src)
            results_src.mkdir()

            cmd = build_command(model_type, seed, args)
            print('Command:', ' '.join(cmd))
            result = subprocess.run(cmd, cwd=str(SCRIPT_DIR))

            if result.returncode != 0:
                print(f'[ERROR] Training failed for {label} (exit code {result.returncode})')
                sys.exit(1)

            copied = copy_non_csv_results(dst_dir, results_src)
            print(f'\nSaved {len(copied)} file(s) to: {dst_dir}')
            for name in sorted(copied):
                print(f'  {name}')

    header('All experiments complete')
    print(f'All {len(models) * len(run_range)} runs completed successfully.')
    print(f'Results in: {output_base}')


if __name__ == '__main__':
    main()
