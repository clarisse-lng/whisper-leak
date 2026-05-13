# KNN Changes

---

## knn_classifier.py

---

## base_classifier.py

Added import:
```python
from core.classifiers.knn_classifier import KNNClassifier
```

In `load()` — class name routing block:
```python
elif class_name == "KNNClassifier":
    module_name = "core.classifiers.knn_classifier"
```

In `load()` — load block:
```python
elif class_name == "KNNClassifier":
    classifier = KNNClassifier.load(filepath)
```

---

## utils.py

Added import:
```python
from core.classifiers.knnn_classifier import KNNClassifier
```

In `ModelTrainer.fit()` — added branch alongside the LightGBM branch:
```python
elif isinstance(self.model, KNNClassifier):
    self.model.fit(train_df=train_data.df, val_df=val_data.df)
    self.history = {'best_epoch': 0, 'train_losses': [], 'val_losses': [], 'train_accs': [], 'val_accs': []}
```
Passes DataFrames directly instead of using a DataLoader.

In `ModelTrainer.predict()` — added branch alongside the LightGBM branch:
```python
elif isinstance(self.model, KNNClassifier):
    scores = self.model.decision_scores(data.df)
    labels = data.df['target'].values
    return scores, labels, None
```
Uses `decision_scores()` instead of `predict_proba()` since `Perceptron` has no probability output.

---

## whisper_leak_train.py

Added import:
```python
from core.classifiers.knn_classifier import KNNClassifier
```

In `create_model()` — added case alongside LGBM:
```python
elif model_type == "KNN":
    model = KNNClassifier(norm, n_neighbors=5)
    model_path = os.path.join(models_dir, 'knn_binary_classifier.pth')
```

---

## Run command

```bash
python3 whisper_leak_train.py -c GPT4o -m KNN -i data/main/gpt4o -p prompts/standard/prompts.json
```
