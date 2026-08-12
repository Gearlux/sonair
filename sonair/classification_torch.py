"""``ClassificationTorchHooks`` — the torch-family implementations of the task hooks.

The classification answers to the trainer API's typed hooks
(:class:`matrainer.torch.runnable.TorchTrainerRunnable`): the target dtype choice, the
single/multi-label loss + metric defaults, the metric sizing keyword, the prediction
contract, the confusion-matrix reporting, and the torch spelling of the class-weight
injection. ONE module serves every torch-family backend (lightning, the bare loop, fastai,
huggingface); the torch-free task logic lives in :mod:`sonair.classification_base`, and a
concrete backend composes ``(ClassificationTorchHooks, ClassificationBase, <Framework>Runnable)``.

This module imports torch, so a keras/jax/mlx backend must never import it — the same rule as
:mod:`sonair.torch_classification` (the registered loss/metric set, which is a different
module: that one REGISTERS upstream classes, this one implements hooks).
"""

import inspect
from typing import Any, Dict, List, Mapping, Optional, Union, cast

import numpy as np
import torch
import torch.nn as nn
from confluid import Fluid, PartialClass, flow
from confluid.registry import resolve_class
from loggair import get_logger
from matrainer.torch.metrics import MetricsSpec
from matrainer.torch.runnable import TorchTrainerRunnable
from matrainer.tracking.base import Tracker
from matrainer.tracking.render import report_matrices
from recordstream import batch_tensor, classification_output, multi_hot
from recordstream.items import Record
from recordstream.outputs import ClassificationOutput
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score, MultilabelAccuracy, MultilabelF1Score

from sonair.classification_base import ClassificationBase

logger = get_logger(__name__)

__all__ = ["ClassificationTorchHooks"]


class ClassificationTorchHooks(ClassificationBase, TorchTrainerRunnable):
    """The torch half of the classification task, mixed into every torch-family backend."""

    #: The inverse-frequency vector `apply_class_weights` injected, kept for tracking
    #: (initialized to None by the concrete backend's __init__).
    class_weights: Optional[Tensor]

    def _targets(self, batch: Record) -> Tensor:
        """The target tensor this task's loss expects — the ONLY task-shaped choice left.

        Both branches are recordstream calls; what is classification-specific is WHICH one
        and the dtype named for it. Multi-label wants an ``[N, C]`` float multi-hot (what
        ``BCEWithLogitsLoss`` and friends consume), single-label an ``[N]`` int64 class-id
        vector — ``int64`` explicitly, because a dataset yielding int32 label tensors is
        legal and ``CrossEntropyLoss`` rejects it with *"expected target dtype to be Long or
        Byte, but got Int"*.
        """
        if self.multilabel:
            if not self.num_classes:
                raise RuntimeError("multi-label targets need num_classes — call initialize() first.")
            return torch.as_tensor(multi_hot(batch, self.target_key, self.num_classes)).to(self.device)
        return batch_tensor(batch, self.target_key, self.device, dtype=torch.int64)

    def _default_loss(self) -> nn.Module:
        # Multi-label needs per-class independent sigmoids; single-label a softmax over classes.
        return nn.BCEWithLogitsLoss() if self.multilabel else nn.CrossEntropyLoss()

    def _default_metrics(self) -> MetricsSpec:
        if self.multilabel:
            return {
                "accuracy": PartialClass[Metric](MultilabelAccuracy),
                "f1_macro": PartialClass[Metric](MultilabelF1Score, average="macro"),
            }
        return {
            # `accuracy`, not `accuracy_top1`: every other backend's accuracy metric is named
            # by its own engine (`fn.__name__`, a keras metric's `name=`) and none of them can
            # say `_top1`, so a sweep across engines would report one quantity in two columns.
            # `top_k=1` is in the template, where it belongs.
            "accuracy": PartialClass[Metric](MulticlassAccuracy, top_k=1),
            "f1_macro": PartialClass[Metric](MulticlassF1Score, average="macro"),
        }

    def _metric_flow_dims(self) -> Dict[str, Union[int, float]]:
        """torchmetrics sizes multi-label metrics by ``num_labels`` and multi-class ones by
        ``num_classes`` — same number, different keyword."""
        assert self.num_classes is not None
        return {"num_labels": self.num_classes} if self.multilabel else {"num_classes": self.num_classes}

    def _prediction_output(self, preds: Tensor) -> ClassificationOutput[Tensor]:
        return classification_output(preds)

    def _report_nonscalars(self, values: Mapping[str, Any], step: Optional[int], trackers: List[Tracker]) -> None:
        """Confusion MATRICES to the log + every tracker's text (and figure) channel.

        Which computed values ARE confusion matrices is recordstream's rule (square 2-D, by
        SHAPE not by metric name); the fan-out is matrainer's ``report_matrices``.
        """
        report_matrices(values, self.class_names or None, trackers, step=step, plot=self.plot_metrics)

    # ---- class balancing: the STATISTIC is recordstream's, the INJECTION is torch's -------- #
    def accepts_class_weights(self) -> bool:
        """Whether the loss can consume per-class weights, answered for torch.

        ``torch.nn`` losses take a ``weight`` TENSOR in the constructor, so the question is
        whether the loss's signature declares one. Works on a live loss OR a deferred
        :class:`~confluid.fluid.Fluid`, which matters: ``_initialize_fit`` calls this BEFORE
        the loss is built to decide whether walking the whole target stream for weights is
        worth it at all.
        """
        loss_cls = resolve_class(self.loss_fn.target) if isinstance(self.loss_fn, Fluid) else type(self.loss_fn)
        if loss_cls is None:
            return False
        try:
            return "weight" in inspect.signature(loss_cls).parameters
        except (ValueError, TypeError):  # pragma: no cover - C-extension ctor without a signature
            return False

    def apply_class_weights(self, weights: np.ndarray) -> None:
        """Inject per-class ``weights`` into the loss, the torch way.

        The weights become a ``Tensor`` on the loss's ``weight`` argument. A deferred loss is
        FLOWED with it (a ``Fluid`` cannot be mutated in place, so the slot is rebound); a
        live loss gets the attribute set. ``self.class_weights`` keeps the vector for
        tracking.
        """
        tensor = torch.as_tensor(weights)
        if isinstance(self.loss_fn, Fluid):
            self.loss_fn = flow(self.loss_fn, weight=tensor)
        else:
            cast(nn.Module, self.loss_fn).weight = tensor
        self.class_weights = tensor
        logger.info(f"Balanced loss: injected inverse-frequency class weights for {len(tensor)} classes.")
