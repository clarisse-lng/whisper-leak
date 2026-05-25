# models/knn_classifier.py

import numpy as np
import torch
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import json


class KNNClassifier:

    MODEL_NAME = "knn_binary_classifier"

    def __init__(self, norm, n_neighbors: int = 5, metric: str = "euclidean"):
        self.norm = norm
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.model = KNeighborsClassifier(
            n_neighbors=self.n_neighbors,
            metric=self.metric,
            algorithm='ball_tree',
            n_jobs=1,
        )
        self.scaler = StandardScaler()

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled, y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    # ------------------------------------------------------------------
    # Persistence — .pth via torch.save, matching bert/lstm/cnn pattern
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        torch.save({
            "model": self.model,
            "scaler": self.scaler,
            "normalization_params": self.norm  # ← add this key
        }, path)
        
        # Also keep the JSON file since inference script looks for it separately
        norm_params_path = path.replace('.pth', '_norm_params.json')
        with open(norm_params_path, 'w') as f:
            json.dump(self.norm, f)
        
        print(f"[KNN] Saved to {path}")
        
    @classmethod
    def load(cls, path: str, device=None) -> "KNNClassifier":
        obj = cls(norm={})
        payload = torch.load(path, weights_only=False)
        obj.model = payload["model"]
        obj.scaler = payload["scaler"]
        obj.norm = payload.get("normalization_params", {})
        
        # Also try the JSON file as fallback
        norm_params_path = path.replace('.pth', '_norm_params.json')
        if not obj.norm and os.path.exists(norm_params_path):
            with open(norm_params_path, 'r') as f:
                obj.norm = json.load(f)
        
        return obj
