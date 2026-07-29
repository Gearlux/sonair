"""Image-classification adapters for Marainer.

Classification is the simplest fit for the classifier: it takes ``[N, 3, H, W]``
images and emits ``[N, C]`` logits, and the loss / metric pair both consume
``(logits, target_class_ids)`` directly. No step-output dict, no per-mode forward
divergence — the classifier recovers the input tensor + target tensor from the
batched record dict (via ``_batch_input`` / ``_batch_target``)
and its standard ``preds = self(x)`` / ``loss = self.loss_fn(preds, y)`` flow runs.

Loss: the default classification loss is :class:`torch.nn.CrossEntropyLoss`
(raw logits ``[N, C]`` + int64 targets ``[N]``). A family of off-the-shelf
``torch.nn`` losses is registered here (``_CLASSIFICATION_LOSSES``) as confluid
configurables (``task="classification", role="loss", framework="torch"``) so they're
discoverable /
pickable via navigaitor's task surfaces, usable straight from YAML
(``label_smoothing`` / ``ignore_index`` / ``reduction`` …; per-class ``weight``
needs a tensor and isn't expressible in YAML), AND StreamStudio OBJECT nodes. They
surface in StreamStudio via the ``sonair-classification`` entry point + this
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

Collate: the sonair classifier wires the generic :func:`recordstream.collate_records` as
its DataLoader collate — there is no bespoke sonair collate. The config pipeline
tensorizes the input (``ConvertToImageOp`` → ``ToTensorOp``) so at collate time each
record's input key is already a ``[3, H, W]`` :class:`~recordstream.Image`;
``collate_records`` stacks it to ``[N, 3, H, W]`` and gathers the remaining
keys as per-record lists. The classifier's model boundary
(:func:`sonair.lightning_classifier._batch_input` / ``_batch_target``) unwraps the batched input
tensor and stacks the ``Label`` target's per-record list into an ``[N]`` int64 tensor.

Metrics: the useful ``torchmetrics.classification`` multiclass metrics
(:class:`~torchmetrics.classification.MulticlassAccuracy`,
:class:`~torchmetrics.classification.MulticlassF1Score`, …) are registered here
(``_CLASSIFICATION_METRICS``, ``role="metric"``) — they accept logits directly as
``preds`` and emit a scalar by default (no adapter), so they're navigaitor
metric-slot options, StreamStudio OBJECT nodes, and wirable into ``val_metrics``
straight from YAML. Each takes ``num_classes`` (dataset-derived at run time).

Typical wiring (YAML)::

    loss_fn: !class:torch.nn.CrossEntropyLoss
      label_smoothing: 0.1            # timm's modern-classifier default

    val_metrics: !class:torchmetrics.classification.MulticlassAccuracy
      num_classes: 37

    train_loader: !class:torch.utils.data.DataLoader
      collate_fn: !ref:recordstream.collate_records
"""

from typing import Literal, cast

import torch.nn as nn
from confluid import configurable, register
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
#: navigaitor's form-spec / StreamStudio pickers straight from the annotation.
FocalLossReduction = Literal["none", "mean", "sum"]

# The classification loss IS an off-the-shelf ``torch.nn`` loss — register each (no wrapper
# class) so it's a pickable ``task="classification", role="loss"`` option and surfaces in
# navigaitor's form-spec / generated configs AND as a StreamStudio OBJECT node (``role="loss"`` is in
# the bridge ``_MEMBER_ROLES``). ``register`` only tags the third-party class. They are bound to
# module-level names AND listed in ``__all__`` (below) so StreamStudio's ``__all__``-gated discovery
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
    register(_loss_cls, task="classification", role="loss", framework="torch")

# The useful ``torchmetrics`` multiclass metrics — registered ``task="classification",
# role="metric"`` so they're navigaitor metric-slot options AND StreamStudio OBJECT nodes
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
    register(_metric_cls, task="classification", role="metric", framework="torch")


@configurable(task="classification", role="loss", framework="torch")
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


# Every registered loss / metric is named in ``__all__`` so StreamStudio's ``__all__``-gated discovery
# pass (run on this module — re-exported from ``sonair/__init__.py``) surfaces each as a node. Keep
# this list in sync with ``_CLASSIFICATION_LOSSES`` / ``_CLASSIFICATION_METRICS`` above (a literal,
# not a comprehension, so static analysers can read the exported-symbol set);
# ``tests/test_classification.py::test_all_exports_registered_losses_and_metrics`` pins the sync.
__all__ = [
    "SigmoidFocalLoss",
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
