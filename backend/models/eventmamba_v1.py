import torch.nn as nn

from .eventmamba_backbone import (
    Attention,
    EventMambaBackbone,
    Linear1Layer,
    Linear2Layer,
)

__all__ = ["Attention", "EventMamba", "Linear1Layer", "Linear2Layer"]


class EventMamba(EventMambaBackbone):
    """Center-point EventMamba model."""

    def __init__(self, num_classes=6):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_list[3], 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
            nn.Sigmoid(),
        )
