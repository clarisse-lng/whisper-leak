import numpy as np
import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler


class PerceptronClassifier:

    def __init__(self, max_len=700, max_iter=1000, n_classes=2):
        """
        Initialises the perceptron classifier.

        Args:
            max_len: Maximum sequence length — sequences longer than this are truncated, shorter ones are zero-padded.
            max_iter: Maximum number of passes over the training data.
            n_classes: Number of output classes (2 for binary, >2 for multiclass).
        """
        self.max_len = max_len
        self.n_classes = n_classes
        self.scaler = StandardScaler()
        self.model = Perceptron(max_iter=max_iter, class_weight=None, random_state=42)

    def _prepare_features(self, df):
        """
        Flattens and pads time_diffs and data_lengths into a 2D feature array.

        Each row becomes [time_1...time_N, length_1...length_N] of fixed size 2*max_len.
        Sequences shorter than max_len are zero-padded; longer ones are truncated.

        Args:
            df: DataFrame with 'time_diffs' and 'data_lengths' columns.

        Returns:
            numpy array of shape [n_samples, 2*max_len].
        """
        num_samples = len(df)
        X = np.zeros((num_samples, self.max_len * 2), dtype=np.float32)
        for i in range(num_samples):
            times = df['time_diffs'].iloc[i][:self.max_len]
            lengths = df['data_lengths'].iloc[i][:self.max_len]
            X[i, :len(times)] = times
            X[i, self.max_len:self.max_len + len(lengths)] = lengths
        return X

    def fit(self, train_df, val_df=None, **kwargs):
        """
        Trains the perceptron on the training data.

        Scales features using StandardScaler then fits the perceptron.
        val_df is accepted for interface compatibility but ignored — the perceptron has no validation step.

        Args:
            train_df: DataFrame with 'time_diffs', 'data_lengths', and 'target' columns.
            val_df: Ignored.
        """
        X = self.scaler.fit_transform(self._prepare_features(train_df))
        self.model.fit(X, train_df['target'].values)

    def decision_scores(self, df):
        """
        Returns calibrated scores for each sample in df.

        Binary: sigmoid-scaled 1D scores in (0, 1).
        Multiclass: softmax-scaled [n_samples, n_classes] probability matrix.

        Args:
            df: DataFrame with 'time_diffs' and 'data_lengths' columns.

        Returns:
            numpy array of shape (n_samples,) for binary or (n_samples, n_classes) for multiclass.
        """
        raw = self.model.decision_function(self.scaler.transform(self._prepare_features(df)))
        if self.n_classes > 2:
            # raw is [n_samples, n_classes] (OvR); apply softmax row-wise
            std = raw.std(axis=0, keepdims=True)
            std[std == 0] = 1.0
            scaled = raw / std
            shifted = scaled - scaled.max(axis=1, keepdims=True)
            exp = np.exp(shifted)
            return exp / exp.sum(axis=1, keepdims=True)
        else:
            return 1 / (1 + np.exp(-raw / raw.std() if raw.std() > 0 else -raw))

    def inference(self, input_data, device=None):
        """
        Runs inference on a single sample or a batch.

        Args:
            input_data: Either a DataFrame or a tuple of (time_diffs, data_lengths).
            device: Ignored — included for interface compatibility with PyTorch models.

        Returns:
            For DataFrame: (scores_array, predictions_array)
            For tuple: (score_or_probs, prediction)
        """
        if isinstance(input_data, pd.DataFrame):
            scores = self.decision_scores(input_data)
            preds = np.argmax(scores, axis=1) if self.n_classes > 2 else (scores > 0).astype(int)
            return scores, preds

        elif isinstance(input_data, tuple) and len(input_data) == 2:
            time_diffs, data_lengths = input_data
            row = pd.DataFrame([{'time_diffs': list(time_diffs), 'data_lengths': list(data_lengths)}])
            scores = self.decision_scores(row)
            if self.n_classes > 2:
                return scores[0], int(np.argmax(scores[0]))
            else:
                score = scores[0]
                return score, (1 if score > 0 else 0)
        else:
            raise Exception('Input must be a DataFrame or tuple of (time_diffs, data_lengths)')

    def save(self, filepath):
        """
        Saves the model, scaler, max_len, and n_classes to disk.

        Args:
            filepath: Path ending in .pth — extensions are replaced automatically.
        """
        import joblib, json, os
        joblib.dump((self.model, self.scaler, self.max_len, self.n_classes), filepath.replace('.pth', '.perceptron.joblib'))
        with open(filepath.replace('.pth', '_norm_params.json'), 'w') as f:
            json.dump({'normalization_params': {}, 'class_name': 'PerceptronClassifier', 'args': {'n_classes': self.n_classes}}, f)

    @classmethod
    def load(cls, filepath, device=None):
        """
        Loads a saved PerceptronClassifier from disk.

        Args:
            filepath: Path ending in .pth — extensions are replaced automatically.
            device: Ignored — included for interface compatibility with PyTorch models.

        Returns:
            PerceptronClassifier instance with model, scaler, max_len, and n_classes restored.
        """
        import joblib, json
        instance = cls()
        loaded = joblib.load(filepath.replace('.pth', '.perceptron.joblib'))
        if len(loaded) == 4:
            instance.model, instance.scaler, instance.max_len, instance.n_classes = loaded
        else:
            instance.model, instance.scaler, instance.max_len = loaded
            instance.n_classes = 2
        return instance
