import json
import matplotlib.pyplot as plt
from pathlib import Path

INPUT_FILE = "label_counts.json"
OUTPUT_IMAGE = "label_frequencies.png"


def main():
    counts = json.loads(Path(INPUT_FILE).read_text())

    labels = list(counts.keys())
    values = list(counts.values())

    plt.figure()
    plt.bar(labels, values)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Count")
    plt.title("Label Frequency Distribution")
    plt.tight_layout()

    plt.savefig(OUTPUT_IMAGE, dpi=300)
    print("Saved chart to", OUTPUT_IMAGE)


if __name__ == "__main__":
    main()