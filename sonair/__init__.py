"""Sonair: image-classification models and adapters for the Marainer training framework."""

from sonair.classification import ClassificationLoss, classification_collate_fn
from sonair.evaluator import LightningClassificationEvaluator
from sonair.hf_dataset import HuggingFaceClassificationDataset
from sonair.timm_classifier import TimmClassifierForPrediction, TimmClassifierModel
from sonair.trainer import LightningClassificationTrainer

__version__ = "0.1.0"

__all__ = [
    "TimmClassifierModel",
    "TimmClassifierForPrediction",
    "ClassificationLoss",
    "classification_collate_fn",
    "HuggingFaceClassificationDataset",
    "LightningClassificationTrainer",
    "LightningClassificationEvaluator",
]
