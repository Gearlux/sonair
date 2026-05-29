"""Generic image-classification adapters that plug into Marainer's
:class:`~marainer.trainer.LightningTrainer`.

Classification is the simplest fit for the base Trainer: a classifier takes
``[N, 3, H, W]`` images and emits ``[N, C]`` logits, and the loss / metric
pair both consume ``(logits, target_class_ids)`` directly. No bundled batch
type, no step-output dict, no per-mode forward divergence — the Trainer's
standard ``preds = self(batch.input)`` /
``loss = self.loss_fn(preds, batch.target)`` flow works unchanged.

Loss: there is **no sonair loss wrapper** — :class:`torch.nn.CrossEntropyLoss`
is the classification loss, registered here as a confluid configurable
(``task="classification", role="loss"``) so it's discoverable / pickable via
navigaitor's task surfaces and usable straight from YAML
(``label_smoothing`` / ``ignore_index`` / ``reduction`` all work; per-class
``weight`` needs a tensor and isn't expressible in YAML). It consumes raw
logits ``[N, C]`` + int64 targets ``[N]`` directly.

This module provides:

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

    loss_fn: !class:torch.nn.CrossEntropyLoss
      label_smoothing: 0.1            # timm's modern-classifier default

    val_metrics: !class:torchmetrics.classification.MulticlassAccuracy
      num_classes: 37

    train_loader: !class:torch.utils.data.DataLoader
      collate_fn: !ref:sonair.classification.classification_collate_fn
"""

from typing import Any, Dict, List

import torch
import torch.nn as nn
from confluid import register
from dataflux.sample import Sample
from logflow import get_logger
from torch import Tensor

logger = get_logger(__name__)

# The classification loss IS ``torch.nn.CrossEntropyLoss`` — register it (no
# wrapper class) so it's a pickable ``task="classification", role="loss"`` option
# and surfaces in navigaitor's form-spec / generated configs. ``register`` tags
# the third-party class without modifying its behaviour.
register(nn.CrossEntropyLoss, task="classification", role="loss")


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
    "classification_collate_fn",
]
