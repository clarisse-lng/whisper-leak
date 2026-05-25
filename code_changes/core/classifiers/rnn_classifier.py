import torch
import torch.nn as nn
from .base_classifier import BaseClassifier


class RNNClassifier(BaseClassifier):
    """
    Vanilla RNN binary classifier operating on (time_z, size_z) sequences.
    Uses the final hidden state as the sequence representation.
    """

    def __init__(self, norm, hidden_size=64, num_layers=1, dropout_rate=0.3, n_classes=2):
        super().__init__(norm)
        self.class_name = self.__class__.__name__
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.n_classes = n_classes

        self.args = {
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'dropout_rate': dropout_rate,
            'n_classes': n_classes,
        }

        self.rnn = nn.RNN(
            input_size=2,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
        )

        output_size = n_classes if n_classes > 2 else 1
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, output_size),
        )

        self._init_weights()

    def _init_weights(self):
        for name, param in self.rnn.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        # x: [batch, 2, max_len] → [batch, max_len, 2]
        x = x.permute(0, 2, 1)

        lengths = (x.sum(dim=2) != 0).sum(dim=1).clamp(min=1).cpu().int()

        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        _, hidden = self.rnn(packed)  # hidden: [num_layers, batch, hidden_size]

        out = hidden[-1]  # [batch, hidden_size]
        return self.fc(out)
