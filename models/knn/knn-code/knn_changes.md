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
from core.classifiers.knn_classifier import KNNClassifier
```

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
