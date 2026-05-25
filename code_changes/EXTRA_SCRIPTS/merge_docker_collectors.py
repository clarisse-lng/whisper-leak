#!/usr/bin/env python3
"""
Merge Docker collector JSON files into a flat folder for use with run_classifier_suite.py.

Copies {chatbot}.json from each copy-collector-* subdir into a single flat folder,
naming them {chatbot}_1.json, {chatbot}_2.json, etc.

Usage:
  python merge_docker_collectors.py
  python merge_docker_collectors.py -d DockerSetup_conversation_prompts/data -o data/docker_conversations_merged
  python merge_docker_collectors.py -c GPT4oMini -d DockerSetup_standard_prompts/data
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.parent
DEFAULT_DOCKER_DATA = 'DockerSetup_standard_prompts/data'
DEFAULT_OUTPUT = 'data/docker_standard_merged'
DEFAULT_CHATBOT = 'GPT4oMini'


def parse_args():
    parser = argparse.ArgumentParser(
        description='Merge Docker collector JSON files into a flat folder.'
    )
    parser.add_argument('-d', '--docker-data', default=DEFAULT_DOCKER_DATA,
                        help=f'Docker data folder containing copy-collector-* subdirs '
                             f'(default: {DEFAULT_DOCKER_DATA})')
    parser.add_argument('-c', '--chatbot', default=DEFAULT_CHATBOT,
                        help=f'Chatbot name to collect (default: {DEFAULT_CHATBOT})')
    parser.add_argument('-o', '--output', default=DEFAULT_OUTPUT,
                        help=f'Output folder for merged JSON files (default: {DEFAULT_OUTPUT})')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing output folder without prompting')
    return parser.parse_args()


def main():
    args = parse_args()

    docker_base = SCRIPT_DIR / args.docker_data
    output_folder = SCRIPT_DIR / args.output

    if not docker_base.exists():
        print(f'[ERROR] Docker data folder not found: {docker_base}')
        sys.exit(1)

    collector_dirs = sorted(
        d for d in docker_base.iterdir()
        if d.is_dir() and d.name.startswith('copy-collector-')
    )
    if not collector_dirs:
        print(f'[ERROR] No copy-collector-* dirs found in {docker_base}')
        sys.exit(1)

    if output_folder.exists() and any(output_folder.glob('*.json')):
        if not args.overwrite:
            answer = input(f'{output_folder} already has JSON files. Overwrite? [y/N] ').strip().lower()
            if answer != 'y':
                print('Aborted.')
                sys.exit(0)
        for f in output_folder.glob('*.json'):
            f.unlink()

    output_folder.mkdir(parents=True, exist_ok=True)

    total = 0
    for i, cdir in enumerate(collector_dirs, start=1):
        src = cdir / f'{args.chatbot}.json'
        if not src.exists():
            print(f'[WARN] {src} not found, skipping')
            continue
        dst = output_folder / f'{args.chatbot}_{i}.json'
        shutil.copy2(src, dst)
        count = len(json.loads(dst.read_text()))
        total += count
        print(f'  {cdir.name} → {dst.name}  ({count} entries)')

    print(f'\nMerged {total} total entries into {output_folder}')
    print(f'\nNext step:')
    print(f'  python EXTRA_SCRIPTS/run_classifier_suite.py -i {args.output}')


if __name__ == '__main__':
    main()
