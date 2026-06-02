"""Image-classification adapters for Marainer.

Classification is the simplest fit for the base Trainer: a classifier takes
``[N, 3, H, W]`` images and emits ``[N, C]`` logits, and the loss / metric
pair both consume ``(logits, target_class_ids)`` directly. No bundled batch
type, no step-output dict, no per-mode forward divergence — the Trainer's
standard ``preds = self(batch.input)`` /
``loss = self.loss_fn(preds, batch.target)`` flow works unchanged.

Loss: the default classification loss is :class:`torch.nn.CrossEntropyLoss`,
registered here as a confluid configurable (``task="classification",
role="loss"``) so it's discoverable / pickable via navigaitor's task surfaces
and usable straight from YAML (``label_smoothing`` / ``ignore_index`` /
``reduction`` all work; per-class ``weight`` needs a tensor and isn't
expressible in YAML). It consumes raw logits ``[N, C]`` + int64 targets ``[N]``
directly.

The one loss that needs a sonair-side class is :class:`SigmoidFocalLoss`: its
upstream, :func:`torchvision.ops.sigmoid_focal_loss`, is a plain *function*
(not an :class:`torch.nn.Module`), and confluid's ``register`` only tags
*classes*. A thin :class:`torch.nn.Module` shim that stores ``alpha`` / ``gamma``
/ ``reduction`` and delegates to the function in ``forward`` is therefore the
minimal way to make it a pickable, YAML-wirable ``role="loss"`` configurable —
this is the "generic shim" escape hatch named in ``CLAUDE.md``, not a bespoke
loss reimplementation. Like the multi-label family (``BCEWithLogitsLoss`` /
``MultiLabelSoftMarginLoss``) it expects **float** targets the same shape as the
logits (e.g. one-hot ``[N, C]``), not int64 class ids.

Collate: :func:`classification_collate_fn` is the default DataLoader collate the
sonair trainer / evaluator wire in. Unlike the generic
:func:`marainer.collate.collate_fn_with_metadata` (which *requires* tensor
inputs and raises on a PIL image), it **forces each input to an image-like float
tensor** — PIL / ndarray → ``[3, H, W]`` (grayscale expanded to 3 channels, the
``TimmModel`` ``in_channels=3``), scaled to ``[0, 1]`` — then stacks to
``[N, 3, H, W]``, stacks the scalar class ids to ``[N]`` int64, and preserves
per-sample metadata as a list (so string-valued fields such as ``pack_id`` /
``iq_file`` survive collation). **No resize**: every image in a batch must already
share spatial dims — resize (or a model-keyed transform) is a deliberate follow-up.

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
      collate_fn: !ref:marainer.collate.collate_fn_with_metadata
"""

from typing import Any, Dict, List, Literal, cast

import numpy as np
import torch
import torch.nn as nn
from confluid import configurable, register
from dataflux.sample import Sample
from torch import Tensor
from torchvision.ops import sigmoid_focal_loss

#: Closed set of reductions :func:`torchvision.ops.sigmoid_focal_loss` accepts —
#: typed as a ``Literal`` so the value is statically checkable and enumerable by
#: navigaitor's form-spec / FluxStudio pickers straight from the annotation.
FocalLossReduction = Literal["none", "mean", "sum"]

# The classification loss IS ``torch.nn.CrossEntropyLoss`` — register it (no
# wrapper class) so it's a pickable ``task="classification", role="loss"`` option
# and surfaces in navigaitor's form-spec / generated configs. ``register`` tags
# the third-party class without modifying its behaviour.
register(nn.BCELoss, task="classification", role="loss")
register(nn.BCEWithLogitsLoss, task="classification", role="loss")
register(nn.HingeEmbeddingLoss, task="classification", role="loss")
register(nn.CrossEntropyLoss, task="classification", role="loss")
register(nn.MultiLabelSoftMarginLoss, task="classification", role="loss")
register(nn.MultiMarginLoss, task="classification", role="loss")
register(nn.MultiLabelMarginLoss, task="classification", role="loss")
register(nn.SoftMarginLoss, task="classification", role="loss")


@configurable(task="classification", role="loss")
class SigmoidFocalLoss(nn.Module):
    """RetinaNet sigmoid focal loss (`Lin et al. 2017 <https://arxiv.org/abs/1708.02002>`_).

    A thin :class:`torch.nn.Module` wrapper over
    :func:`torchvision.ops.sigmoid_focal_loss` so the upstream *function* becomes
    a confluid-``@configurable`` ``role="loss"`` class — pickable in navigaitor's
    task surfaces and wirable from YAML. The constructor does no functional work
    (it only stores the three knobs), satisfying the zero-arg / lazy-init mandate;
    ``forward`` delegates verbatim.

    Like the multi-label loss family it expects **float** ``targets`` the same
    shape as ``inputs`` (the raw logits ``[N, C]``) — e.g. one-hot — encoding the
    binary label per element. It does *not* accept int64 class ids (use
    :class:`torch.nn.CrossEntropyLoss` for those).

    The default ``reduction="mean"`` differs from torchvision's ``"none"``: a
    ``loss_fn`` must return a scalar for the Trainer's ``loss.backward()`` flow,
    so the unreduced default would be unusable as wired.

    YAML::

        loss_fn: !class:sonair.classification.SigmoidFocalLoss
          alpha: 0.25
          gamma: 2.0

    Args:
        alpha: Weighting factor in ``[0, 1]`` balancing positive vs negative
            examples, or ``-1`` to disable. Default ``0.25``.
        gamma: Exponent of the modulating factor ``(1 - p_t)`` that down-weights
            easy examples. Default ``2.0``.
        reduction: ``"none"`` | ``"mean"`` | ``"sum"`` applied to the per-element
            loss. Default ``"mean"`` (see note above).
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: FocalLossReduction = "mean",
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """Compute the focal loss between ``inputs`` (logits) and float ``targets``."""
        return cast(
            Tensor,
            sigmoid_focal_loss(
                inputs,
                targets,
                alpha=self.alpha,
                gamma=self.gamma,
                reduction=self.reduction,
            ),
        )


def _as_image_tensor(value: Any) -> Tensor:
    """Coerce a single ``Sample.input`` into an image-like ``[3, H, W]`` float tensor in ``[0, 1]``.

    Accepts a PIL image (converted to RGB), a numpy array (``[H,W]`` or ``[H,W,C]``), or a tensor
    (``[H,W]`` / ``[C,H,W]``). A 1-channel (grayscale) result is expanded to 3 channels because the
    sonair ``TimmModel`` is built with ``in_channels=3``. Values above ``1`` are scaled by ``1/255``
    (the uint8 path). No resize — spatial dims are left untouched.
    """
    if isinstance(value, Tensor):
        t = value.float()
    else:
        if hasattr(value, "convert"):  # PIL image → RGB so grayscale/RGBA become 3 channels
            arr = np.asarray(value.convert("RGB"))
        else:
            arr = np.asarray(value)
        t = torch.from_numpy(np.ascontiguousarray(arr)).float()
        if t.dim() == 3:  # HWC → CHW
            t = t.permute(2, 0, 1)
    if t.dim() == 2:  # [H,W] → [1,H,W]
        t = t.unsqueeze(0)
    if t.dim() != 3:
        raise ValueError(f"classification_collate_fn: expected an image-like input, got tensor shape {tuple(t.shape)}.")
    if float(t.max()) > 1.0:  # uint8 / 0–255 range → [0,1]
        t = t / 255.0
    if t.shape[0] == 1:  # grayscale → 3 channels (TimmModel in_channels=3)
        t = t.expand(3, -1, -1)
    return t.contiguous()


def _as_target_tensor(value: Any) -> Tensor:
    """Coerce a single ``Sample.target`` into a 0-D ``int64`` class-id tensor."""
    t = value if isinstance(value, Tensor) else torch.as_tensor(value)
    if t.dtype != torch.int64:
        t = t.long()
    if t.dim() != 0:
        raise ValueError(f"classification_collate_fn: expected a scalar class-id target, got shape {tuple(t.shape)}.")
    return t


def classification_collate_fn(batch: List[Sample]) -> Sample:
    """Collate classification :class:`~dataflux.sample.Sample`\\ s into one batched, Trainer-ready Sample.

    The default DataLoader collate the sonair trainer / evaluator wire in. Per sample it **forces**
    the input to an image-like ``[3, H, W]`` float tensor (PIL / ndarray / tensor → :func:`_as_image_tensor`)
    and the target to a 0-D ``int64`` class id, then stacks:

    * ``.input`` → ``[N, 3, H, W]`` float;
    * ``.target`` → ``[N]`` int64;
    * ``.metadata`` → ``list`` of the per-sample metadata dicts (so string-valued fields survive — the
      default torch collate would crash on them).

    **No resize** — all per-sample images must already share spatial dims (``torch.stack`` requires it);
    resize / a model-keyed transform is a deliberate follow-up.
    """
    if not batch:
        raise ValueError("classification_collate_fn received an empty batch.")
    inputs: List[Tensor] = []
    targets: List[Tensor] = []
    metas: List[Dict[str, Any]] = []
    for s in batch:
        inputs.append(_as_image_tensor(s.input))
        targets.append(_as_target_tensor(s.target))
        metas.append(s.meta)  # per-sample dict (the batched output below carries the list)
    try:
        batched_input = torch.stack(inputs)
    except RuntimeError as exc:  # mismatched spatial dims — resize isn't wired yet
        shapes = sorted({tuple(t.shape) for t in inputs})
        raise ValueError(
            f"classification_collate_fn: cannot stack inputs with differing shapes {shapes} — every image "
            "in a batch must share spatial dims (resize upstream; per-batch resize is not wired yet)."
        ) from exc
    return Sample(input=batched_input, target=torch.stack(targets), metadata=metas)


__all__ = ["SigmoidFocalLoss", "classification_collate_fn"]
