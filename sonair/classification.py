"""Generic image-classification adapters that plug into Marainer's
:class:`~marainer.trainer.LightningTrainer`.

Classification is the simplest fit for the base Trainer: a classifier takes
``[N, 3, H, W]`` images and emits ``[N, C]`` logits, and the loss / metric
pair both consume ``(logits, target_class_ids)`` directly. No bundled batch
type, no step-output dict, no per-mode forward divergence — the Trainer's
standard ``preds = self(batch.input)`` /
``loss = self.loss_fn(preds, batch.target)`` flow works unchanged.

This module provides:

* :class:`ClassificationLoss` — a :class:`torch.nn.CrossEntropyLoss` wrapper
  exposed for Confluid configuration (so per-class weights, label smoothing,
  and ignore_index can be set from YAML without writing a Python class).
* :func:`classification_collate_fn` — a DataLoader collate that stacks
  per-sample ``input`` images and ``target`` scalar class ids into batched
  tensors and keeps per-sample metadata as a list (the default collate would
  crash on string-valued metadata such as ``pack_id`` or ``iq_file``).

Most torchmetrics classification metrics
(:class:`torchmetrics.classification.MulticlassAccuracy`,
:class:`~torchmetrics.classification.MulticlassF1Score`, ...) accept logits
directly as ``preds`` and emit a scalar by default — no adapter is needed,
they can be wired into ``val_metrics`` straight from YAML.

Typical wiring (YAML)::

    loss_fn: !class:sonair.classification.ClassificationLoss
      label_smoothing: 0.1            # timm's modern-classifier default
      # class_weights: [...]          # optional — per-class re-weighting

    val_metrics: !class:torchmetrics.classification.MulticlassAccuracy
      num_classes: 37

    train_loader: !class:torch.utils.data.DataLoader
      collate_fn: !ref:sonair.classification.classification_collate_fn
"""

from typing import Any, Dict, List, Optional, cast

import torch
import torch.nn as nn
from confluid import configurable
from dataflux.sample import Sample
from logflow import get_logger
from torch import Tensor

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Loss adapter
# ---------------------------------------------------------------------------


@configurable
class ClassificationLoss(nn.Module):
    """:class:`torch.nn.CrossEntropyLoss` wrapped for Confluid configuration.

    The base Trainer calls ``loss_fn(preds, target)`` where ``preds`` are
    raw logits ``[N, C]`` and ``target`` is an int64 class-id tensor ``[N]``.
    PyTorch's CrossEntropyLoss handles softmax + NLL internally, so we
    forward raw logits straight through.

    Args:
        class_weights: Optional per-class loss multipliers, length
            ``num_classes``. Useful when class frequencies are skewed.
        label_smoothing: Smoothing factor in ``[0.0, 1.0)``. ``0.0`` (default)
            is plain cross-entropy; ``0.1`` is timm's standard recipe for
            ViT-family fine-tuning.
        ignore_index: Class id in ``target`` that contributes zero loss.
            Defaults to ``-100`` (the PyTorch default).
        reduction: ``"mean"`` | ``"sum"`` | ``"none"``. Defaults to ``"mean"``
            — averages over the batch and returns a single scalar that
            Lightning's logger can consume directly.
    """

    def __init__(
        self,
        class_weights: Optional[List[float]] = None,
        label_smoothing: float = 0.0,
        ignore_index: int = -100,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        weight: Optional[Tensor] = None
        if class_weights is not None:
            if len(class_weights) == 0:
                raise ValueError("class_weights must contain at least one entry.")
            weight = torch.tensor(class_weights, dtype=torch.float32)
        self.class_weights = list(class_weights) if class_weights is not None else None
        self.label_smoothing = float(label_smoothing)
        self.ignore_index = int(ignore_index)
        self.reduction = reduction
        # Register as a buffer (when present) so .to(device) moves it
        # automatically. When absent, no buffer is registered — CrossEntropyLoss
        # then defaults to unweighted, which is exactly what we want.
        if weight is not None:
            self.register_buffer("_weight_buf", weight, persistent=False)
        self._ce = nn.CrossEntropyLoss(
            weight=weight,
            label_smoothing=self.label_smoothing,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
        )

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        if logits.dim() != 2:
            raise ValueError(f"ClassificationLoss expects logits [N,C], got shape {tuple(logits.shape)}.")
        if target.dim() != 1:
            raise ValueError(
                f"ClassificationLoss expects target [N] int64 (class ids), got shape {tuple(target.shape)}."
            )
        if target.dtype != torch.int64:
            target = target.long()
        # Re-bind weight from buffer in case the module was moved to another
        # device — CrossEntropyLoss captures the weight reference at
        # construction time and won't pick up the buffer's moved state.
        self._ce.weight = getattr(self, "_weight_buf", None)
        return cast(Tensor, self._ce(logits, target))


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------


def classification_collate_fn(batch: List[Sample]) -> Sample:
    """Collate per-sample classification samples into a batched Trainer-ready Sample.

    Each input ``Sample`` is expected to carry:

    * ``sample.input`` — ``[3, H, W]`` float tensor (e.g. produced by
      :class:`waivefront.classification_ops.MetadataLabelToClassificationTargetOp`).
    * ``sample.target`` — int64 scalar tensor (single class id) or 0-D tensor.

    The returned :class:`~dataflux.sample.Sample` has:

    * ``.input`` — ``[N, 3, H, W]`` stacked image tensor.
    * ``.target`` — ``[N]`` stacked int64 class-id tensor.
    * ``.metadata`` — ``{"per_sample": [<per-sample metadata dicts>]}``,
      preserved as a list so string-valued fields (``pack_id``, ``iq_file``,
      ``drone``) survive collation. The default DataLoader collate would
      crash on those.

    All per-sample image tensors must already share the same spatial dims;
    mixed sizes must be resized upstream (e.g. via Waivefront's
    :class:`~waivefront.visualizers.RenderSpectrogramImageOp`).
    """
    if not batch:
        raise ValueError("classification_collate_fn received an empty batch.")
    images: List[Tensor] = []
    targets: List[Tensor] = []
    metas: List[Dict[str, Any]] = []
    for s in batch:
        if not isinstance(s.input, Tensor):
            raise TypeError(
                f"classification_collate_fn expects sample.input to be a torch.Tensor; "
                f"got {type(s.input).__name__}."
            )
        if not isinstance(s.target, Tensor):
            raise TypeError(
                f"classification_collate_fn expects sample.target to be a torch.Tensor; "
                f"got {type(s.target).__name__}."
            )
        images.append(s.input)
        # Coerce to 0-D int64 so torch.stack produces a clean [N] tensor.
        # Tolerate already-int64 0-D tensors (no-op) and other int dtypes.
        t = s.target
        if t.dtype != torch.int64:
            t = t.long()
        if t.dim() != 0:
            raise ValueError(
                f"classification_collate_fn expects sample.target to be a 0-D scalar tensor, "
                f"got shape {tuple(t.shape)}."
            )
        targets.append(t)
        metas.append(s.metadata)

    return Sample(
        input=torch.stack(images),
        target=torch.stack(targets),
        metadata={"per_sample": metas},
    )


__all__ = [
    "ClassificationLoss",
    "classification_collate_fn",
]
