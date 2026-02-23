import json
import sys
from collections import Counter
from pathlib import Path
from openai import OpenAI

DEFAULT_DATASET_FILE = "dataset-small.json"
LABELS_FILE = "labels.json"
DEFAULT_OUTPUT_FILE = "label_counts.json"

SYSTEM_PROMPT = """
You are a strict classifier.

Classify the query into exactly ONE label.

If none apply, return: inconclusive

Return ONLY the label name.
"""


def load_labels():
    return json.loads(Path(LABELS_FILE).read_text(encoding="utf-8"))["labels"]


def build_label_text(labels):
    return "\n".join(f"- {k}: {v}" for k, v in labels.items())


def classify_query(client, query, label_text, valid_labels):
    prompt = f"""
Labels:
{label_text}

Query: {query}
Label:
"""

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
    )

    label = response.output_text.strip()

    if label not in valid_labels:
        return "inconclusive"

    return label


def classify_dataset(dataset_path, output_file=DEFAULT_OUTPUT_FILE):
    dataset_path = Path(dataset_path)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    labels = load_labels()

    label_text = build_label_text(labels)
    valid_labels = set(labels.keys())

    prompts = dataset["negative"]["prompts"]

    client = OpenAI()

    results = []

    for i, prompt in enumerate(prompts, start=1):
        label = classify_query(client, prompt, label_text, valid_labels)
        results.append(label)
        print(f"[{i}] {label} → {prompt}")

    counts = Counter(results)

    Path(output_file).write_text(
        json.dumps(dict(counts), indent=2),
        encoding="utf-8"
    )

    print(f"\nSaved counts to {output_file}")


def main():
    # CLI usage:
    # python classify.py dataset.json output.json

    dataset_file = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATASET_FILE
    output_file = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUTPUT_FILE

    classify_dataset(dataset_file, output_file)


if __name__ == "__main__":
    main()