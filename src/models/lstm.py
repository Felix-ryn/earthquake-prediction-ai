import logging
from src.utils import setup_logger

import torch
from torch import nn

logger = setup_logger(__name__)


class LSTMAngleDir(nn.Module):
    def __init__(
        self,
        n_features: int,
        hidden_size: int = 64,
        n_layers: int = 1,
        n_classes: int = 8,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, 64)
        self.dropout = nn.Dropout(p=0.2)
        self.head_angle = nn.Linear(64, 1)
        self.head_dir = nn.Linear(64, n_classes)

    def forward(self, x: torch.Tensor):        
        out, (h_n, c_n) = self.lstm(x)
        h_last = h_n[-1]  
        x = self.dropout(torch.relu(self.fc(h_last)))
        angle = self.head_angle(x).squeeze(-1)
        logits = self.head_dir(x)
        return angle, logits
