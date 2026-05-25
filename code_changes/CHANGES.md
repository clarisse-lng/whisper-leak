# Changes Summary

## New Collection Script

**`generate_conversation_prompts.py`** — generates `prompts/conversations/conversations.json`. Contains all hardcoded session definitions: 100 positive Python Data Science sessions with per-task follow-up pools, 1,000 adversarial negative_code sessions across 5 themes (JavaScript, C++, SQL, Java, HTML/CSS), and 9,000 Quora-based negative_general sessions. Run once to produce or regenerate the conversations file.

---

## New Classifiers

### `core/classifiers/rnn_classifier.py`
Vanilla RNN classifier operating on (time, size) sequences. Uses packed sequences for efficiency. Supports binary and multiclass output.

### `core/classifiers/perceptron_classifier.py`
MLP classifier operating on flattened feature vectors. Fast to train, used as a baseline. Supports binary and multiclass.

### `core/classifiers/knn_classifier.py`
K-Nearest Neighbours classifier using rank-normalised features. Supports binary and multiclass.

---

## Multiclass Support

All classifiers and the training pipeline now support multiclass classification. The number of classes (`n_classes`) is auto-detected from the data via `df['target'].nunique()`. No argument changes are needed — the same commands work for both binary and multiclass datasets.

**Key changes per file:**

- **`core/classifiers/utils.py`** — `ModelTrainer` takes `n_classes` param. Uses `CrossEntropyLoss` for multiclass, `BCEWithLogitsLoss` for binary. Training and prediction loops branch accordingly.
- **`core/classifiers/base_classifier.py`** — `inference()` returns softmax probabilities + argmax for multiclass; sigmoid + threshold for binary.
- **`core/classifiers/lightgbm_classifier.py`** — switches `objective`, `metric`, and `num_class` based on `n_classes`. `predict_proba` returns full probability matrix for multiclass.
- **`core/classifiers/visualization.py`** — all plots (confusion matrix, ROC, PR curve, dashboard) now handle N-class output. Uses OvR macro averaging for multiclass metrics. Metric file uses `AUPRC (macro):` label for multiclass.
- **`whisper_leak_train.py`** — detects `n_classes`, passes to model and trainer. Test predictions use `argmax` for multiclass. Score columns named `score_class_0`, `score_class_1`, ... for multiclass. Added `--results-dir` argument to control staging folder name (enables parallel runs).

---

## Obfuscation Fix

**`chatbots/gpt_4o_mini.py`** — Added `stream_options={"include_obfuscation": False}` to all API calls. OpenAI's streaming obfuscation is ON by default; without this flag all previously collected data had obfuscation active, making traffic patterns harder to classify and inconsistent across datasets.

**Deleted** `chatbots/gpt_4o_mini_obfuscation.py`, `chatbots/mistral_large.py`, `chatbots/mistral_small.py` — removed unused/obfuscation chatbot files so they cannot be accidentally selected.

**`DockerSetup_standard_prompts/generate-compose.py`** and **`DockerSetup_conversation_prompts/generate-compose.py`** — changed default chatbot from `GPT4oMiniObfuscation` to `GPT4oMini`.

---

## Terminal Output Fix

**`core/utils.py`** — `PrintUtils.start_stage()` now caps line width to actual terminal width using `shutil.get_terminal_size()`. Previously the fixed 120-character line would wrap in narrow terminals, breaking the `\r` override and printing duplicate progress lines.

---

## New Scripts (`EXTRA_SCRIPTS/`)

### `run_classifier_suite.py`
Run all classifiers N times on a data folder, saving results to `ALL_RESULTS/`.
Works for any flat JSON data folder. Auto-detects chatbot name from file contents.

```bash
# Standard or any pre-processed data
python EXTRA_SCRIPTS/run_classifier_suite.py -i data/gpt4o-mini_STANDARD_WITH_MITIGATIONS

# Docker data (after merging), 10 runs, skip slow models
python EXTRA_SCRIPTS/run_classifier_suite.py -i data/docker_standard_merged -n 10 --skip KNN

# Resume from a specific run (e.g. after interruption)
python EXTRA_SCRIPTS/run_classifier_suite.py -i data/multitheme --run-range 6-10

# Run two datasets simultaneously without staging folder conflict
python EXTRA_SCRIPTS/run_classifier_suite.py -i data/dataset_a --staging-dir results_a
python EXTRA_SCRIPTS/run_classifier_suite.py -i data/dataset_b --staging-dir results_b
```

### `merge_docker_collectors.py`
Merges `copy-collector-*/chatbot.json` files from a Docker data folder into a single flat folder, ready for `run_classifier_suite.py`.

```bash
# Standard Docker setup (default)
python EXTRA_SCRIPTS/merge_docker_collectors.py

# Conversation Docker setup
python EXTRA_SCRIPTS/merge_docker_collectors.py \
    -d DockerSetup_conversation_prompts/data \
    -o data/docker_conversations_merged
```

### `plot_auprc.py`
Plot AUPRC per run for each model. Handles binary (`AUPRC:`) and multiclass (`AUPRC (macro):`). Labels are non-overlapping, placed outside the plot area. Colorblind-safe palette, 300 DPI.

```bash
# Auto-discovers all RESULTS_* and ALL_RESULTS/ folders
python EXTRA_SCRIPTS/plot_auprc.py

# Specific folder with custom title
python EXTRA_SCRIPTS/plot_auprc.py ALL_RESULTS/docker_standard_merged --title "Standard — No Mitigations"

# Plot only first 5 runs (saves as auprc_per_run_5.png)
python EXTRA_SCRIPTS/plot_auprc.py ALL_RESULTS/docker_standard_merged --runs 5
```

### `split_prompts_for_docker.py`
Split a prompts file into per-container `collector-N.json` files. Auto-detects standard (flat `prompts` list) vs conversations (nested `sessions`) format.

```bash
# Standard prompts, 5 containers (default)
python EXTRA_SCRIPTS/split_prompts_for_docker.py -n 5

# Conversation prompts, 15 containers
python EXTRA_SCRIPTS/split_prompts_for_docker.py \
    --input prompts/conversations/conversations.json -n 15
```

### `add_binary_targets.py` / `add_multiclass_targets.py`
Assign classification targets to multitheme data.

```bash
python EXTRA_SCRIPTS/add_binary_targets.py -i data/multitheme
python EXTRA_SCRIPTS/add_multiclass_targets.py -i data/multitheme
```

### `compare_standard_distributions.py`
Compare packet-level distributions across three datasets (Docker standard, standard with mitigations, multitheme). Samples B and C to match size of A for a fair comparison. Saves a 300 DPI PNG.

```bash
python EXTRA_SCRIPTS/compare_standard_distributions.py
# Output: EXTRA_SCRIPTS/standard_distribution_comparison.png
```
