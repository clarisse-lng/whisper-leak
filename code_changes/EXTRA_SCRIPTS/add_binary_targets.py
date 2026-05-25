"""
Adds binary target labels to the multitheme dataset.
theme == 'other' → target = 0 (negative)
any other theme  → target = 1 (positive)

Reads from data/gpt4o-mini_MULTITHEME_WITH_MITIGATIONS/
Writes to  data/gpt4o-mini_MULTITHEME_BINARY/
"""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
SRC_DIR = REPO_ROOT / 'data' / 'gpt4o-mini_MULTITHEME_WITH_MITIGATIONS'
DST_DIR = REPO_ROOT / 'data' / 'gpt4o-mini_MULTITHEME_BINARY_WITH_MITIGATIONS'


def main():
    DST_DIR.mkdir(parents=True, exist_ok=True)

    json_files = sorted(SRC_DIR.glob('*.json'))
    if not json_files:
        print(f'No JSON files found in {SRC_DIR}')
        return

    total = 0
    for src_file in json_files:
        with open(src_file) as f:
            entries = json.load(f)

        for entry in entries:
            theme = entry.get('theme', 'other')
            entry['target'] = 0 if theme == 'other' else 1

        dst_file = DST_DIR / src_file.name
        with open(dst_file, 'w') as f:
            json.dump(entries, f)

        total += len(entries)
        print(f'  {src_file.name}: {len(entries)} entries')

    pos = sum(1 for f in DST_DIR.glob('*.json') for e in json.load(open(f)) if e['target'] == 1)
    neg = total - pos
    print(f'\nDone. Total: {total} | Positive: {pos} | Negative: {neg}')
    print(f'Output: {DST_DIR}')


if __name__ == '__main__':
    main()
