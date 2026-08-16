import logging
from src.utils import setup_logger

import torch
from torch import nn

logger = setup_logger(__name__)


class CNNAngleDir(nn.Module):
    def __init__(self, n_features: int, n_classes: int = 8):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels=n_features,
            out_channels=32,
            kernel_size=3,
            padding=1,
        )
        self.conv2 = nn.Conv1d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1,
        )
        self.act = nn.ReLU()
        self.pool = nn.AdaptiveMaxPool1d(output_size=1)
        self.fc = nn.Linear(64, 64)
        self.dropout = nn.Dropout(p=0.2)
        self.head_angle = nn.Linear(64, 1)
        self.head_dir = nn.Linear(64, n_classes)

    def forward(self, x: torch.Tensor):        
        x = x.permute(0, 2, 1)
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.pool(x).squeeze(-1)  
        x = self.dropout(self.act(self.fc(x)))
        angle = self.head_angle(x).squeeze(-1)
        logits = self.head_dir(x) 
        return angle, logits
