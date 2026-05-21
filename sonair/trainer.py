"""LightningClassificationTrainer: specialised image-classification trainer.

Subclass of :class:`~marainer.trainer.LightningTrainer` that bakes in all
classification-specific defaults so a minimal YAML config only needs to specify
the dataset, model backbone, and number of classes — no adapter wiring for loss,
collate, or metrics required.

Baked-in defaults (all overridable via YAML ``@configurable`` attributes):

* ``loss_fn`` — :class:`~sonair.classification.ClassificationLoss`
* ``train/val/test_loader`` ``collate_fn`` — :func:`~sonair.classification.classification_collate_fn`
* ``optimizer`` — ``AdamW(lr=1e-4, weight_decay=0.05)``
* Metrics — :class:`~torchmetrics.classification.MulticlassAccuracy` (top-1)
  + :class:`~torchmetrics.classification.MulticlassF1Score` (macro) per split;
  top-5 accuracy added when ``num_classes >= 5``.

Minimal YAML example::

    trainer: !class:sonair.trainer.LightningClassificationTrainer
      num_classes: 37
      model: !class:sonair.timm_classifier.TimmClassifierModel
        model_name: efficientvit_b0
        num_classes: !ref:num_classes
      train_set: !class:sonair.hf_dataset.HuggingFaceClassificationDataset
        repo_id: kitofrank/RFUAV
        split: train
      val_set: !class:sonair.hf_dataset.HuggingFaceClassificationDataset
        repo_id: kitofrank/RFUAV
        split: validation
"""

from typing import Any, Optional

import pytorch_lightning as L
import torch
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score

from confluid import Class, Lazy, configurable
from marainer.trainer import LightningTrainer
from sonair.classification import ClassificationLoss, classification_collate_fn


def _build_classification_metrics(num_classes: int) -> MetricCollection:
    metrics: dict[str, Any] = {
        "accuracy_top1": MulticlassAccuracy(num_classes=num_classes, top_k=1),
        "f1_macro": MulticlassF1Score(num_classes=num_classes, average="macro"),
    }
    if num_classes >= 5:
        metrics["accuracy_top5"] = MulticlassAccuracy(num_classes=num_classes, top_k=5)
    return MetricCollection(metrics)


@configurable
class LightningClassificationTrainer(LightningTrainer):
    """Specialised image-classification trainer with baked-in defaults.

    Args:
        model: Classification model that emits ``[N, num_classes]`` logits.
        train_set: Training dataset or Flux pipeline.
        num_classes: Number of output classes; drives metric construction.
        val_set: Validation dataset or Flux pipeline.
        test_set: Held-out test dataset (triggers ``lightning.test()`` after fit).
        optimizer: Deferred optimizer (default: AdamW lr=1e-4, wd=0.05).
        train_loader: Deferred DataLoader (default includes ``classification_collate_fn``).
        val_loader: Deferred DataLoader (default includes ``classification_collate_fn``).
        test_loader: Deferred DataLoader (default includes ``classification_collate_fn``).
        lightning: PyTorch Lightning Trainer config.
        experiment_name: MLflow experiment name.
        run_name: MLflow run name.
    """

    def __init__(
        self,
        model: Any,
        train_set: Any,
        num_classes: int,
        val_set: Any = None,
        test_set: Any = None,
        optimizer: Lazy[Any] = Class(torch.optim.AdamW, lr=1e-4, weight_decay=0.05),
        train_loader: Lazy[Any] = Class(
            DataLoader, collate_fn=classification_collate_fn
        ),
        val_loader: Lazy[Any] = Class(DataLoader, collate_fn=classification_collate_fn),
        test_loader: Lazy[Any] = Class(
            DataLoader, collate_fn=classification_collate_fn
        ),
        lightning: Any = Class(L.Trainer),
        experiment_name: str = "sonair_classification",
        run_name: str = "train",
    ) -> None:
        super().__init__(
            model=model,
            train_set=train_set,
            val_set=val_set,
            test_set=test_set,
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            lightning=lightning,
            experiment_name=experiment_name,
            run_name=run_name,
        )
        self.loss_fn = ClassificationLoss()
        base = _build_classification_metrics(num_classes)
        self.train_metrics: Optional[MetricCollection] = base.clone(prefix="train/")
        self.val_metrics: Optional[MetricCollection] = base.clone(prefix="val/")
        self.test_metrics: Optional[MetricCollection] = base.clone(prefix="test/")
