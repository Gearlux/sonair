"""Image-classification adapters for Marainer.

Classification is the simplest fit for the base Trainer: a classifier takes
``[N, 3, H, W]`` images and emits ``[N, C]`` logits, and the loss / metric
pair both consume ``(logits, target_class_ids)`` directly. No bundled batch
type, no step-output dict, no per-mode forward divergence — the Trainer's
standard ``preds = self(batch.input)`` /
``loss = self.loss_fn(preds, batch.target)`` flow works unchanged.

Loss: the default classification loss is :class:`torch.nn.CrossEntropyLoss`
(raw logits ``[N, C]`` + int64 targets ``[N]``). A family of off-the-shelf
``torch.nn`` losses is registered here (``_CLASSIFICATION_LOSSES``) as confluid
configurables (``task="classification", role="loss"``) so they're discoverable /
pickable via navigaitor's task surfaces, usable straight from YAML
(``label_smoothing`` / ``ignore_index`` / ``reduction`` …; per-class ``weight``
needs a tensor and isn't expressible in YAML), AND FluxStudio OBJECT nodes. They
surface in FluxStudio via the ``sonair-classification`` entry point + this
module's ``__all__`` (``scan_module`` filters by ``__module__`` = ``torch``).

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

Metrics: the useful ``torchmetrics.classification`` multiclass metrics
(:class:`~torchmetrics.classification.MulticlassAccuracy`,
:class:`~torchmetrics.classification.MulticlassF1Score`, …) are registered here
(``_CLASSIFICATION_METRICS``, ``role="metric"``) — they accept logits directly as
``preds`` and emit a scalar by default (no adapter), so they're navigaitor
metric-slot options, FluxStudio OBJECT nodes, and wirable into ``val_metrics``
straight from YAML. Each takes ``num_classes`` (dataset-derived at run time).

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
from torch.nn import (
    BCELoss,
    BCEWithLogitsLoss,
    CrossEntropyLoss,
    HingeEmbeddingLoss,
    MultiLabelMarginLoss,
    MultiLabelSoftMarginLoss,
    MultiMarginLoss,
    SoftMarginLoss,
)
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassAUROC,
    MulticlassAveragePrecision,
    MulticlassCalibrationError,
    MulticlassCohenKappa,
    MulticlassConfusionMatrix,
    MulticlassF1Score,
    MulticlassHammingDistance,
    MulticlassJaccardIndex,
    MulticlassMatthewsCorrCoef,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassSpecificity,
)
from torchvision.ops import sigmoid_focal_loss

#: Closed set of reductions :func:`torchvision.ops.sigmoid_focal_loss` accepts —
#: typed as a ``Literal`` so the value is statically checkable and enumerable by
#: navigaitor's form-spec / FluxStudio pickers straight from the annotation.
FocalLossReduction = Literal["none", "mean", "sum"]

# The classification loss IS an off-the-shelf ``torch.nn`` loss — register each (no wrapper
# class) so it's a pickable ``task="classification", role="loss"`` option and surfaces in
# navigaitor's form-spec / generated configs AND as a FluxStudio OBJECT node (``role="loss"`` is in
# the bridge ``_MEMBER_ROLES``). ``register`` only tags the third-party class. They are bound to
# module-level names AND listed in ``__all__`` (below) so FluxStudio's ``__all__``-gated discovery
# pass surfaces them — ``scan_module`` alone would skip a ``torch.nn``-module class.
_CLASSIFICATION_LOSSES = (
    BCELoss,
    BCEWithLogitsLoss,
    CrossEntropyLoss,
    HingeEmbeddingLoss,
    MultiLabelSoftMarginLoss,
    MultiMarginLoss,
    MultiLabelMarginLoss,
    SoftMarginLoss,
)
for _loss_cls in _CLASSIFICATION_LOSSES:
    register(_loss_cls, task="classification", role="loss")

# The useful ``torchmetrics`` multiclass metrics — registered ``task="classification",
# role="metric"`` so they're navigaitor metric-slot options AND FluxStudio OBJECT nodes
# (``role="metric"`` is in ``_MEMBER_ROLES``). All take a ``num_classes`` knob (dataset-derived at
# run time, like the model) plus per-metric options (``top_k`` / ``average`` …). Wire a metric's
# ``class`` output into a trainer's metrics slot (the trainer flows it with ``num_classes``), or set
# ``num_classes`` on the node and take ``instance``. Bound to names + ``__all__`` for discovery.
_CLASSIFICATION_METRICS = (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassAUROC,
    MulticlassAveragePrecision,
    MulticlassSpecificity,
    MulticlassCohenKappa,
    MulticlassMatthewsCorrCoef,
    MulticlassConfusionMatrix,
    MulticlassJaccardIndex,
    MulticlassCalibrationError,
    MulticlassHammingDistance,
)
for _metric_cls in _CLASSIFICATION_METRICS:
    register(_metric_cls, task="classification", role="metric")


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


# Every registered loss / metric is named in ``__all__`` so FluxStudio's ``__all__``-gated discovery
# pass (run on this module — re-exported from ``sonair/__init__.py``) surfaces each as a node. Keep
# this list in sync with ``_CLASSIFICATION_LOSSES`` / ``_CLASSIFICATION_METRICS`` above (a literal,
# not a comprehension, so static analysers can read the exported-symbol set);
# ``tests/test_classification.py::test_all_exports_registered_losses_and_metrics`` pins the sync.
__all__ = [
    "SigmoidFocalLoss",
    "classification_collate_fn",
    # losses (torch.nn) — role="loss"
    "BCELoss",
    "BCEWithLogitsLoss",
    "CrossEntropyLoss",
    "HingeEmbeddingLoss",
    "MultiLabelSoftMarginLoss",
    "MultiMarginLoss",
    "MultiLabelMarginLoss",
    "SoftMarginLoss",
    # metrics (torchmetrics.classification) — role="metric"
    "MulticlassAccuracy",
    "MulticlassF1Score",
    "MulticlassPrecision",
    "MulticlassRecall",
    "MulticlassAUROC",
    "MulticlassAveragePrecision",
    "MulticlassSpecificity",
    "MulticlassCohenKappa",
    "MulticlassMatthewsCorrCoef",
    "MulticlassConfusionMatrix",
    "MulticlassJaccardIndex",
    "MulticlassCalibrationError",
    "MulticlassHammingDistance",
]
