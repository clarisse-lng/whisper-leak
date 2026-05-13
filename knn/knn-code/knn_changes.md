# Perceptron Changes

---

## perceptron_classifier.py

**NEW FILE.** Contains the full `PerceptronClassifier` implementation.
Uses `sklearn.linear_model.Perceptron` and `sklearn.preprocessing.StandardScaler`.
Does not inherit from `BaseClassifier` or `nn.Module` — same pattern as `LightGBMClassifier`.

---

## base_classifier.py

Added import:
```python
from core.classifiers.perceptron_classifier import PerceptronClassifier
```

In `load()` — class name routing block:
```python
elif class_name == "PerceptronClassifier":
    module_name = "core.classifiers.perceptron_classifier"
```

In `load()` — load block:
```python
elif class_name == "PerceptronClassifier":
    classifier = PerceptronClassifier.load(filepath)
```
This bypasses PyTorch's `load_state_dict` since the perceptron uses joblib instead.

---

## utils.py

Added import:
```python
from core.classifiers.perceptron_classifier import PerceptronClassifier
```

In `ModelTrainer.fit()` — added branch alongside the LightGBM branch:
```python
elif isinstance(self.model, PerceptronClassifier):
    self.model.fit(train_df=train_data.df, val_df=val_data.df)
    self.history = {'best_epoch': 0, 'train_losses': [], 'val_losses': [], 'train_accs': [], 'val_accs': []}
```
Passes DataFrames directly instead of using a DataLoader.

In `ModelTrainer.predict()` — added branch alongside the LightGBM branch:
```python
elif isinstance(self.model, PerceptronClassifier):
    scores = self.model.decision_scores(data.df)
    labels = data.df['target'].values
    return scores, labels, None
```
Uses `decision_scores()` instead of `predict_proba()` since `Perceptron` has no probability output.

---

## whisper_leak_train.py

Added import:
```python
from core.classifiers.perceptron_classifier import PerceptronClassifier
```

In `create_model()` — added case alongside LGBM:
```python
elif model_type == 'PERCEPTRON':
    model = PerceptronClassifier(max_len=norm['max_len'])
    model_path = os.path.join(models_dir, 'perceptron_binary_classifier.pth')
```

---

## Run command

```bash
python3 whisper_leak_train.py -c GPT4o -m PERCEPTRON -i data/main/gpt4o -p prompts/standard/prompts.json
```
