"""
Split a prompts file into N collector-*.json files for Docker collectors.

Auto-detects format from the input file:

  Standard format  (prompts/standard/prompts.json):
    {"positive": {"prompts": [str, ...]}, "negative": {"prompts": [str, ...]}}
    → collector-N.json: {"positive": {"repeat": R, "prompts": [...]},
                         "negative": {"repeat": 1, "prompts": [...]}}

  Conversations format  (prompts/conversations/conversations.json):
    {"positive": {"sessions": [...]}, "negative_general": {...}, "negative_code": {...}}
    → collector-N.json: {"positive":         {"repeat": R, "sessions": [...]},
                         "negative_general": {"repeat": 1, "sessions": [...]},
                         "negative_code":    {"repeat": 1, "sessions": [...]}}

Usage examples:
  python split_prompts_for_docker.py                        # standard, 5 containers
  python split_prompts_for_docker.py -n 15                  # standard, 15 containers
  python split_prompts_for_docker.py --input prompts/conversations/conversations.json -n 15
"""

import json
import argparse
import random
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
STANDARD_PROMPTS = REPO_ROOT / "prompts" / "standard" / "prompts.json"
CONVERSATIONS_PATH = REPO_ROOT / "prompts" / "conversations" / "conversations.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "DOCKER_copy" / "prompts"


def split_evenly(items, n):
    base, extra = divmod(len(items), n)
    chunks, start = [], 0
    for i in range(n):
        end = start + base + (1 if i < extra else 0)
        chunks.append(items[start:end])
        start = end
    return chunks


def split_standard(data, args):
    rng = random.Random(args.seed)
    pos_all = data["positive"]["prompts"]
    neg_all = data["negative"]["prompts"]

    pos_sample = rng.sample(pos_all, min(args.max_positive, len(pos_all)))
    neg_sample = rng.sample(neg_all, min(args.max_negative, len(neg_all)))

    n = args.num_containers
    print(f"Sampled {len(pos_sample)} positive (repeat={args.repeat}), {len(neg_sample)} negative (repeat=1)")
    print(f"Total API calls: {len(pos_sample) * args.repeat + len(neg_sample)} across {n} containers")

    pos_chunks = split_evenly(pos_sample, n)
    neg_chunks = split_evenly(neg_sample, n)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nWriting {n} collector files to {args.output_dir}")
    for i in range(n):
        out = {
            "positive": {"repeat": args.repeat, "prompts": pos_chunks[i]},
            "negative": {"repeat": 1, "prompts": neg_chunks[i]},
        }
        out_path = args.output_dir / f"collector-{i + 1}.json"
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"  collector-{i + 1}.json: {len(pos_chunks[i])} pos × {args.repeat}, {len(neg_chunks[i])} neg")


def split_conversations(data, args):
    n = args.num_containers
    pos_sessions = data["positive"]["sessions"]
    pos_repeat = data["positive"]["repeat"]
    neg_general = data.get("negative_general", {}).get("sessions", [])
    neg_code = data.get("negative_code", {}).get("sessions", [])

    print(f"Positive: {len(pos_sessions)} sessions (repeat={pos_repeat})")
    print(f"Negative general: {len(neg_general)} sessions")
    print(f"Negative code: {len(neg_code)} sessions")

    pos_chunks = split_evenly(pos_sessions, n)
    neg_general_chunks = split_evenly(neg_general, n)
    neg_code_chunks = split_evenly(neg_code, n)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nWriting {n} collector files to {args.output_dir}")
    for i in range(n):
        out = {
            "positive": {"repeat": pos_repeat, "sessions": pos_chunks[i]},
            "negative_general": {"repeat": 1, "sessions": neg_general_chunks[i]},
            "negative_code": {"repeat": 1, "sessions": neg_code_chunks[i]},
        }
        out_path = args.output_dir / f"collector-{i + 1}.json"
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"  collector-{i + 1}.json: "
              f"{len(pos_chunks[i])} pos × {pos_repeat}, "
              f"{len(neg_general_chunks[i])} neg_general, "
              f"{len(neg_code_chunks[i])} neg_code")


def detect_format(data):
    pos = data.get("positive", {})
    if "prompts" in pos:
        return "standard"
    if "sessions" in pos:
        return "conversations"
    raise ValueError("Cannot detect format: 'positive' key has neither 'prompts' nor 'sessions'.")


def main():
    parser = argparse.ArgumentParser(
        description="Split a prompts file into per-container collector-N.json files for Docker."
    )
    parser.add_argument("--input", "-i", type=Path, default=STANDARD_PROMPTS,
                        help="Input prompts JSON file (default: prompts/standard/prompts.json)")
    parser.add_argument("-n", "--num-containers", type=int, default=5,
                        help="Number of collector containers (default: 5)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Output directory for collector-N.json files")
    # Standard-only options (ignored for conversations format)
    parser.add_argument("-r", "--repeat", type=int, default=5,
                        help="Positive prompt repeat count, standard format only (default: 5)")
    parser.add_argument("--max-positive", type=int, default=50,
                        help="Max positive prompts to sample, standard format only (default: 50)")
    parser.add_argument("--max-negative", type=int, default=500,
                        help="Max negative prompts to sample, standard format only (default: 500)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling, standard format only (default: 42)")
    args = parser.parse_args()

    print(f"Reading {args.input}")
    with open(args.input) as f:
        data = json.load(f)

    fmt = detect_format(data)
    print(f"Detected format: {fmt}")

    if fmt == "standard":
        split_standard(data, args)
    else:
        split_conversations(data, args)

    print("\nDone.")


if __name__ == "__main__":
    main()
