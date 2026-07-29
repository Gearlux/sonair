"""Sonair: a standard PyTorch-Lightning image classifier, using marainer for optional tracking."""

from sonair.lightning_classifier import LightningClassifier
from sonair.timm_classifier import TimmModel

__version__ = "0.2.0"

__all__ = [
    "TimmModel",
    "LightningClassifier",
]
