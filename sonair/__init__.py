"""Sonair: image-classification models and adapters for the Marainer training framework."""

from sonair.classification import classification_collate_fn
from sonair.evaluator import LightningClassificationEvaluator
from sonair.timm_classifier import TimmModel
from sonair.trainer import LightningClassificationTrainer

__version__ = "0.1.0"

__all__ = [
    "TimmModel",
    "classification_collate_fn",
    "LightningClassificationTrainer",
    "LightningClassificationEvaluator",
]
