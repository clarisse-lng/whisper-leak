"""
Adds multi-class target labels to the multitheme dataset.

Theme → class mapping:
    other              → 0  (negative)
    abortion           → 1
    drug               → 2
    gender             → 3
    genetic_experiment → 4
    immigration        → 5
    mental_health      → 6
    murder             → 7
"""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
SRC_DIR = REPO_ROOT / 'data' / 'gpt4o-mini_MULTITHEME_WITH_MITIGATIONS'
DST_DIR = REPO_ROOT / 'data' / 'gpt4o-mini_MULTITHEME_MULTICLASS_WITH_MITIGATIONS'

THEME_TO_CLASS = {
    'other': 0,
    'abortion': 1,
    'drug': 2,
    'gender': 3,
    'genetic_experiment': 4,
    'immigration': 5,
    'mental_health': 6,
    'murder': 7,
}


def main():
    DST_DIR.mkdir(parents=True, exist_ok=True)

    json_files = sorted(SRC_DIR.glob('*.json'))
    if not json_files:
        print(f'No JSON files found in {SRC_DIR}')
        return

    unknown_themes = set()
    total = 0
    for src_file in json_files:
        with open(src_file) as f:
            entries = json.load(f)

        for entry in entries:
            theme = entry.get('theme', 'other')
            if theme not in THEME_TO_CLASS:
                unknown_themes.add(theme)
                entry['target'] = -1
            else:
                entry['target'] = THEME_TO_CLASS[theme]

        dst_file = DST_DIR / src_file.name
        with open(dst_file, 'w') as f:
            json.dump(entries, f)

        total += len(entries)
        print(f'  {src_file.name}: {len(entries)} entries')

    if unknown_themes:
        print(f'\n[WARNING] Unknown themes encountered: {unknown_themes}')

    class_counts = {label: 0 for label in THEME_TO_CLASS.values()}
    for src_file in DST_DIR.glob('*.json'):
        for e in json.load(open(src_file)):
            label = e.get('target', -1)
            if label in class_counts:
                class_counts[label] += 1

    print(f'\nDone. Total: {total}')
    for theme, cls in THEME_TO_CLASS.items():
        print(f'  Class {cls} ({theme}): {class_counts.get(cls, 0)}')
    print(f'\nOutput: {DST_DIR}')
    print(f'Class mapping: {THEME_TO_CLASS}')


if __name__ == '__main__':
    main()
