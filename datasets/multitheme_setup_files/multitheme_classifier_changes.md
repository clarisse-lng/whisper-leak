# Changelog — Multi-theme Classification Support

## Overview

These changes add support for training on a multi-theme JSON format
(`multitheme_prompts.json`) where prompts are grouped by topic (e.g. `abortion`,
`gender`, `drug`, …) instead of the legacy binary `positive` / `negative` format.
Two classification modes are introduced: **binary** (sensitive vs. neutral) and
**multiclass** (one label per theme).

---

## `whisper_leak_train.py`

### 1. New function — `load_multitheme_data()`

Reads the multi-theme JSON format and builds a labelled DataFrame.

- Iterates over all theme keys in the JSON (`{ theme: { repeat, prompts[] } }`).
- **Binary mode** (`--multitheme_mode binary`): the neutral theme (default `other`)
  gets label `0`, every other theme gets label `1`.
- **Multiclass mode** (`--multitheme_mode multiclass`): themes are sorted
  alphabetically and each receives a unique integer label.
- Calls the existing `load_chatbot_data()` to load raw timing sequences, then
  **overwrites the `target` column** with the new multi-theme labels.
- Drops rows whose prompt is not found in the JSON (with a warning count).
- Adds a `theme` column (human-readable theme name per row).
- Returns `(df, label_map)` where `label_map` is a `{theme: int}` dict.

### 2. New helper — `is_multitheme_json()`

Heuristic that inspects the top-level JSON structure:
returns `True` if values contain a `prompts` key (multi-theme format),
`False` otherwise (legacy binary format). Used for auto-detection at runtime.

### 3. New CLI arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--multitheme_mode` | `binary` \| `multiclass` | `binary` | Labelling strategy |
| `--neutral_theme` | `str` | `other` | Theme key used as negative class in binary mode |

### 4. `parse_arguments()` — validation

No additional validation needed beyond the `choices=` constraint on
`--multitheme_mode`; the neutral theme existence is checked inside
`load_multitheme_data()` with an explicit `ValueError`.

### 5. `main()` — data loading branch

Replaced the single `load_chatbot_data()` call with an auto-detection branch:

```python
if is_multitheme_json(args.prompts):
    df, label_map = load_multitheme_data(...)
    num_classes = len(set(label_map.values()))
else:
    df = load_chatbot_data(...)          # legacy path, unchanged
    num_classes = 2
    label_map = {'negative': 0, 'positive': 1}
```

### 6. `create_model()` — `num_classes` parameter

- Signature extended: `create_model(..., num_classes=2)`.
- `num_classes` is forwarded to every classifier constructor.
- Model file names now include a class-count suffix:
  `cnn_binary_classifier.pth` → `cnn_8class_classifier.pth` for 8 themes.

### 7. `main()` — evaluation & inference

Introduced `is_multiclass = num_classes > 2` flag that drives the following
conditional logic throughout the evaluation block:

**Predictions:**
- Binary: `test_preds = (test_scores > 0.5).astype(int)` (unchanged)
- Multiclass: `test_preds = np.argmax(test_scores_all, axis=1)`

**Plots — skipped in multiclass mode** (binary-only):
- ROC curve
- Precision-Recall curve
- Score distribution
- Model performance dashboard

**Confusion matrix** — in multiclass mode, `labels` and `display_labels` are
passed so that axes show theme names instead of integers.

**Test results CSV** — two extra columns in multiclass mode:
- `theme`: ground-truth theme name
- `predicted_theme`: predicted theme name (via inverted `label_map`)

**Metrics** — in multiclass mode, `calculate_metrics()` (binary-only) is
replaced by `sklearn.metrics.classification_report` + `accuracy_score`,
producing per-theme precision / recall / F1.

---

## `utils.py`

### 1. `ModelTrainer.__init__()` — adaptive loss criterion

```python
# Before
self.criterion = nn.BCEWithLogitsLoss() if isinstance(model, nn.Module) else None

# After
def __init__(self, model, config, device, num_classes=2):
    self.num_classes = num_classes
    if isinstance(model, nn.Module):
        self.criterion = (nn.BCEWithLogitsLoss() if num_classes == 2
                          else nn.CrossEntropyLoss())
```

### 2. `_train_pytorch_model()` — training and validation loops

Both the train phase and the validation phase must handle two tensor shapes:

```python
# Binary (unchanged shape)
X, y = X.to(device), y.to(device).float().unsqueeze(1)   # y: (B, 1)
loss  = criterion(outputs, y)                              # BCEWithLogitsLoss
preds = (torch.sigmoid(outputs) > 0.5).float()

# Multiclass (new)
X, y = X.to(device), y.to(device).long()                  # y: (B,)
loss  = criterion(outputs, y)                              # CrossEntropyLoss
preds = torch.argmax(outputs, dim=1).float()
```

### 3. `ModelTrainer.predict()` — output scores

```python
# Binary
scores = torch.sigmoid(outputs)          # (B, 1) → flatten to (B,)

# Multiclass
scores = torch.softmax(outputs, dim=1)   # (B, num_classes) — kept as 2-D
```

### 4. `load_chatbot_data()` — guard against multi-theme JSON

The function ends with:
```python
df['target'] = df['prompt'].apply(
    lambda x: 1 if x in prompts['positive']['prompts'] else 0
)
```
This crashes when called with a multi-theme JSON that has no `positive` /
`negative` keys. Fix: make the target assignment conditional.

```python
# After
if 'positive' in prompts and 'negative' in prompts:
    df['target'] = df['prompt'].apply(
        lambda x: 1 if x in prompts['positive']['prompts'] else 0
    )
else:
    df['target'] = -1   # sentinel — overwritten by load_multitheme_data()
```

> **Recommended refactor**: extract raw sequence loading into a standalone
> `load_raw_sequences(chatbot, input_folder)` function so that
> `load_multitheme_data()` does not need to call `load_chatbot_data()` at all
> and the coupling is eliminated entirely.

---

## `loader.py`

### `__getitem__()` — target tensor dtype

`BCEWithLogitsLoss` expects `float`, `CrossEntropyLoss` expects `long`.
The cleanest fix is to always return `long` and let `ModelTrainer` cast as needed.

```python
# Before
return (torch.tensor(sample, dtype=torch.float32),
        torch.tensor(target, dtype=torch.float32))

# After
return (torch.tensor(sample, dtype=torch.float32),
        torch.tensor(int(target), dtype=torch.long))
```

In `ModelTrainer._train_pytorch_model()`, binary mode then does:
```python
y = y.float().unsqueeze(1)
```

---

## `base_classifier.py`

### `__init__()` — store `num_classes`

```python
def __init__(self, norm, num_classes=2):
    super().__init__()
    self.norm = norm
    self.num_classes = num_classes
    ...
```

### `inference()` — multiclass output path

**DataFrame input:**
```python
# Before
output = torch.sigmoid(output).cpu().numpy()
all_probs.extend(output.flatten())
...
predictions = (all_probs > 0.5).astype(int)

# After
if self.num_classes == 2:
    output = torch.sigmoid(output).cpu().numpy().flatten()
    all_probs.extend(output)
    ...
    predictions = (np.array(all_probs) > 0.5).astype(int)
else:
    output = torch.softmax(output, dim=1).cpu().numpy()   # (B, num_classes)
    all_probs.append(output)
    ...
    all_probs  = np.vstack(all_probs)
    predictions = np.argmax(all_probs, axis=1)
```

**Tuple input:** same branching, `softmax` replaces `sigmoid` in multiclass mode
and `argmax` replaces threshold `> 0.5`.

### `save()` / `load()` — `num_classes` persistence

`num_classes` must be included in `self.args` (done at the individual classifier
level below) so that it is written to `_norm_params.json` and restored on load
without any change needed in `save()` / `load()` themselves.

---

## `rnn_classifier.py`

The final linear layer is hardcoded to `1`. To support multiclass:

```python
# Current
def __init__(self, norm, hidden_size=64, num_layers=1, dropout_rate=0.3):
    self.args = {
        'hidden_size':  hidden_size,
        'num_layers':   num_layers,
        'dropout_rate': dropout_rate,
    }
    ...
    nn.Linear(32, 1)

# Required
def __init__(self, norm, hidden_size=64, num_layers=1, dropout_rate=0.3, num_classes=2):
    self.args = {
        'hidden_size':  hidden_size,
        'num_layers':   num_layers,
        'dropout_rate': dropout_rate,
        'num_classes':  num_classes,    # persisted via BaseClassifier.save()
    }
    ...
    nn.Linear(32, 1 if num_classes == 2 else num_classes)
```

---

## `lightgbm_classifier.py`

### `__init__()` — objective and metric for multiclass

```python
# Before
default_params = {
    'objective': 'binary',
    'metric':    'binary_logloss',
    ...
}

# After
def __init__(self, norm, num_classes=2, **kwargs):
    self.num_classes = num_classes
    is_multiclass = num_classes > 2
    default_params = {
        'objective':  'multiclass' if is_multiclass else 'binary',
        'metric':     'multi_logloss' if is_multiclass else 'binary_logloss',
        'num_class':  num_classes if is_multiclass else None,
        ...
    }
    if not is_multiclass:
        default_params.pop('num_class')
```

> `predict_proba()` already returns `(N, num_classes)` via sklearn's LightGBM
> wrapper, so no change is needed there. The caller in `whisper_leak_train.py`
> already handles the 2-D output correctly with `argmax`.

---

## `knn_classifier.py`

The public API (`fit`, `predict`, `predict_proba`, `save`, `load`) is unchanged.
One latent bug and one missing import were identified in the existing file.

### Bug — missing `import os` in `load()`

```python
# Current load() references os.path.exists() but os is never imported
norm_params_path = path.replace('.pth', '_norm_params.json')
if not obj.norm and os.path.exists(norm_params_path):   # <- NameError at runtime

# Fix: add at the top of the file
import os
```

### Note — `norm_params.json` saves only the norm dict, not class metadata

`BaseClassifier.load()` expects `_norm_params.json` to contain
`{ normalization_params, class_name, args }`, but `KNNClassifier.save()` writes
only the raw norm dict:

```python
# Current
with open(norm_params_path, 'w') as f:
    json.dump(self.norm, f)           # missing class_name and args keys

# Required for BaseClassifier.load() compatibility
with open(norm_params_path, 'w') as f:
    json.dump({
        'normalization_params': self.norm,
        'class_name': 'KNNClassifier',
        'args': {
            'n_neighbors': self.n_neighbors,
            'metric': self.metric,
        }
    }, f)
```

### Multiclass — no changes needed

`KNeighborsClassifier.predict_proba()` natively returns `(N, num_classes)` for
any number of classes. The caller in `whisper_leak_train.py` already handles
this correctly with `argmax`.

---

## `perceptron_classifier.py` 

### Design

Wraps `sklearn.linear_model.Perceptron` with the same external interface as
`KNNClassifier` and `LightGBMClassifier` (no `BaseClassifier` inheritance —
sklearn model, not a PyTorch `nn.Module`).

| Method | Description |
|---|---|
| `_prepare_features(df)` | Flattens and pads `time_diffs` + `data_lengths` into a `(N, 2*max_len)` array |
| `fit(train_df, val_df)` | Scales with `StandardScaler`, fits perceptron; `val_df` ignored |
| `decision_scores(df)` | Returns sigmoid-scaled `decision_function()` output in `(0, 1)` |
| `inference(input_data)` | Unified API: DataFrame or `(time_diffs, data_lengths)` tuple |
| `save(filepath)` | Saves `.perceptron.joblib` + `_norm_params.json` stub |
| `load(filepath)` | Restores model, scaler, and `max_len` from joblib |

### Changes needed for multiclass support

The perceptron natively supports multiclass (OvA strategy) via sklearn. Two
adaptations are required:

**1. Constructor — `num_classes` parameter:**
```python
# Current
def __init__(self, max_len=700, max_iter=1000):
    self.model = Perceptron(max_iter=max_iter, class_weight='balanced', random_state=42)

# Required
def __init__(self, max_len=700, max_iter=1000, num_classes=2):
    self.num_classes = num_classes
    # Perceptron handles multiclass natively — no change to model instantiation needed
    self.model = Perceptron(max_iter=max_iter, class_weight='balanced', random_state=42)
```

**2. `decision_scores()` — multiclass output:**

`decision_function()` returns `(N,)` for binary and `(N, num_classes)` for
multiclass. The current sigmoid scaling assumes a 1-D output:

```python
# Current (binary only)
def decision_scores(self, df):
    scores = self.model.decision_function(...)
    return 1 / (1 + np.exp(-scores / scores.std() if scores.std() > 0 else -scores))

# Required
def decision_scores(self, df):
    scores = self.model.decision_function(...)
    if scores.ndim == 1:
        # Binary: sigmoid-scale the single decision value
        std = scores.std() if scores.std() > 0 else 1.0
        return 1 / (1 + np.exp(-scores / std))
    else:
        # Multiclass: softmax across class scores
        exp_s = np.exp(scores - scores.max(axis=1, keepdims=True))
        return exp_s / exp_s.sum(axis=1, keepdims=True)   # (N, num_classes)
```

**3. `inference()` — multiclass predictions:**

```python
# Current
scores = self.decision_scores(input_data)
preds = (scores > 0).astype(int)

# Required
scores = self.decision_scores(input_data)
if scores.ndim == 1:
    preds = (scores > 0.5).astype(int)
else:
    preds = np.argmax(scores, axis=1)
```

**4. `save()` / `load()` — persist `num_classes` and fix `_norm_params.json`:**

The current JSON stub written by `save()` has empty `normalization_params` and
`args`, which is fine for the perceptron's own `load()` but breaks
`BaseClassifier.load()` if it ever routes through there. Recommended fix:

```python
json.dump({
    'normalization_params': {},
    'class_name': 'PerceptronClassifier',
    'args': {
        'max_len':     self.max_len,
        'num_classes': self.num_classes,
    }
}, f)
```
