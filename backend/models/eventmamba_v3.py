import torch.nn as nn

from .eventmamba_backbone import (
    Attention,
    EventMambaBackbone,
    Linear1Layer,
    Linear2Layer,
)

__all__ = ["Attention", "EventMamba", "Linear1Layer", "Linear2Layer"]


class EventMamba(EventMambaBackbone):
    """Ellipse VSA EventMamba model."""

    def __init__(self, num_classes=1024):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_list[3], 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes),
        )
