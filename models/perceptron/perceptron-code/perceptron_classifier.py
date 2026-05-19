import numpy as np
import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler


class PerceptronClassifier:

    def __init__(self, max_len=700, max_iter=1000):
        """
        Initialises the perceptron classifier.

        Args:
            max_len: Maximum sequence length — sequences longer than this are truncated, shorter ones are zero-padded.
            max_iter: Maximum number of passes over the training data.
        """
        self.max_len = max_len
        self.scaler = StandardScaler()
        self.model = Perceptron(max_iter=max_iter, class_weight='balanced', random_state=42)

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
        Returns sigmoid-scaled scores for each sample in df.

        Raw decision_function() output is divided by its std to bring it into a sensible range,
        then passed through sigmoid to produce values in (0, 1) comparable to probabilities.

        Args:
            df: DataFrame with 'time_diffs' and 'data_lengths' columns.

        Returns:
            numpy array of scores in (0, 1).
        """
        scores = self.model.decision_function(self.scaler.transform(self._prepare_features(df)))
        return 1 / (1 + np.exp(-scores / scores.std() if scores.std() > 0 else -scores))

    def inference(self, input_data, device=None):
        """
        Runs inference on a single sample or a batch.

        Args:
            input_data: Either a DataFrame or a tuple of (time_diffs, data_lengths).
            device: Ignored — included for interface compatibility with PyTorch models.

        Returns:
            For DataFrame: (scores_array, predictions_array)
            For tuple: (score, prediction) where prediction is 1 if score > 0.5 else 0
        """
        if isinstance(input_data, pd.DataFrame):
            scores = self.decision_scores(input_data)
            preds = (scores > 0).astype(int)
            return scores, preds
        elif isinstance(input_data, tuple) and len(input_data) == 2:
            time_diffs, data_lengths = input_data
            row = pd.DataFrame([{'time_diffs': list(time_diffs), 'data_lengths': list(data_lengths)}])
            score = self.decision_scores(row)[0]
            pred = 1 if score > 0 else 0
            return score, pred
        else:
            raise Exception('Input must be a DataFrame or tuple of (time_diffs, data_lengths)')

    def save(self, filepath):
        """
        Saves the model, scaler, and max_len to disk.

        Saves a .perceptron.joblib file containing (model, scaler, max_len) and a
        _norm_params.json stub so BaseClassifier.load() can identify the class type.

        Args:
            filepath: Path ending in .pth — extensions are replaced automatically.
        """
        import joblib, json, os
        joblib.dump((self.model, self.scaler, self.max_len), filepath.replace('.pth', '.perceptron.joblib'))
        with open(filepath.replace('.pth', '_norm_params.json'), 'w') as f:
            json.dump({'normalization_params': {}, 'class_name': 'PerceptronClassifier', 'args': {}}, f)

    @classmethod
    def load(cls, filepath, device=None):
        """
        Loads a saved PerceptronClassifier from disk.

        Args:
            filepath: Path ending in .pth — extensions are replaced automatically.
            device: Ignored — included for interface compatibility with PyTorch models.

        Returns:
            PerceptronClassifier instance with model, scaler, and max_len restored.
        """
        import joblib, json
        instance = cls()
        instance.model, instance.scaler, instance.max_len = joblib.load(filepath.replace('.pth', '.perceptron.joblib'))
        return instance
