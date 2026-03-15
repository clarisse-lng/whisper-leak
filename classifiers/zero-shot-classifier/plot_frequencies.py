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
    plt.barh(labels, values)
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Quantite")
    plt.ylabel("Labels")
    plt.title("Distribution des labels par un classifieur zero shot")
    plt.tight_layout()

    plt.savefig(OUTPUT_IMAGE, dpi=300)
    print("Saved chart to", OUTPUT_IMAGE)


if __name__ == "__main__":
    main()