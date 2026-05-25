import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


class KNNClassifier:

    def __init__(self, norm, n_neighbors: int = 5, metric: str = "euclidean", n_classes: int = 2):
        self.norm = norm
        self.max_len = norm.get('max_len', 700) if norm else 700
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.n_classes = n_classes
        self.scaler = StandardScaler()
        self.model = KNeighborsClassifier(
            n_neighbors=self.n_neighbors,
            metric=self.metric,
            algorithm='ball_tree',
            n_jobs=1,
        )

    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        num_samples = len(df)
        X = np.zeros((num_samples, self.max_len * 2), dtype=np.float32)
        for i in range(num_samples):
            times = df['time_diffs'].iloc[i][:self.max_len]
            lengths = df['data_lengths'].iloc[i][:self.max_len]
            X[i, :len(times)] = times
            X[i, self.max_len:self.max_len + len(lengths)] = lengths
        return X

    def fit(self, train_df: pd.DataFrame, val_df=None, **kwargs) -> None:
        X = self.scaler.fit_transform(self._prepare_features(train_df))
        self.model.fit(X, train_df['target'].values)

    def predict_scores(self, df: pd.DataFrame) -> np.ndarray:
        X = self.scaler.transform(self._prepare_features(df))
        proba = self.model.predict_proba(X)
        if self.n_classes > 2:
            return proba  # [n_samples, n_classes]
        return proba[:, 1]  # [n_samples]

    def inference(self, input_data, device=None):
        if isinstance(input_data, pd.DataFrame):
            scores = self.predict_scores(input_data)
            preds = np.argmax(scores, axis=1) if self.n_classes > 2 else (scores > 0.5).astype(int)
            return scores, preds

        elif isinstance(input_data, tuple) and len(input_data) == 2:
            time_diffs, data_lengths = input_data
            row = pd.DataFrame([{'time_diffs': list(time_diffs), 'data_lengths': list(data_lengths)}])
            scores = self.predict_scores(row)
            if self.n_classes > 2:
                return scores[0], int(np.argmax(scores[0]))
            return scores[0], (1 if scores[0] > 0.5 else 0)

        else:
            raise Exception('Input must be a DataFrame or tuple of (time_diffs, data_lengths)')

    def save(self, filepath: str) -> None:
        joblib.dump((self.model, self.scaler, self.max_len, self.n_neighbors, self.metric, self.n_classes),
                    filepath.replace('.pth', '.knn.joblib'))
        with open(filepath.replace('.pth', '_norm_params.json'), 'w') as f:
            json.dump({'normalization_params': self.norm, 'class_name': 'KNNClassifier', 'args': {'n_classes': self.n_classes}}, f)

    @classmethod
    def load(cls, filepath: str, device=None) -> 'KNNClassifier':
        instance = cls(norm={})
        loaded = joblib.load(filepath.replace('.pth', '.knn.joblib'))
        if len(loaded) == 6:
            instance.model, instance.scaler, instance.max_len, instance.n_neighbors, instance.metric, instance.n_classes = loaded
        else:
            instance.model, instance.scaler, instance.max_len, instance.n_neighbors, instance.metric = loaded
            instance.n_classes = 2
        norm_path = filepath.replace('.pth', '_norm_params.json')
        if os.path.exists(norm_path):
            with open(norm_path) as f:
                instance.norm = json.load(f).get('normalization_params', {})
        return instance
