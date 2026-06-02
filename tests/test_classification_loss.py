# mypy: disable-error-code="attr-defined"
"""Unit coverage for :class:`sonair.classification.SigmoidFocalLoss`.

The wrapper exists because confluid's ``register`` only tags *classes*, while
the upstream ``torchvision.ops.sigmoid_focal_loss`` is a plain function — so
these tests pin both the numerical behaviour (it delegates faithfully, honours
every reduction, and back-propagates) and the discovery tags that make it a
pickable ``task="classification", role="loss"`` configurable.
"""

import pytest
import torch
import torch.nn.functional as F
from confluid.registry import get_registry
from torchvision.ops import sigmoid_focal_loss

from sonair.classification import FocalLossReduction, SigmoidFocalLoss


def _logits_and_targets() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    logits = torch.randn(4, 5)
    targets = F.one_hot(torch.randint(0, 5, (4,)), num_classes=5).float()
    return logits, targets


def test_defaults_match_retinanet() -> None:
    loss = SigmoidFocalLoss()
    assert loss.alpha == 0.25
    assert loss.gamma == 2.0
    # Default reduction is "mean" (not torchvision's "none") so the value is a
    # usable scalar loss for ``loss.backward()``.
    assert loss.reduction == "mean"


def test_constructor_does_no_functional_work() -> None:
    """Zero-arg / lazy-init mandate: ``__init__`` only stores the knobs."""
    loss = SigmoidFocalLoss(alpha=0.3, gamma=1.5, reduction="sum")
    assert (loss.alpha, loss.gamma, loss.reduction) == (0.3, 1.5, "sum")


def test_delegates_to_torchvision() -> None:
    logits, targets = _logits_and_targets()
    loss = SigmoidFocalLoss(alpha=0.4, gamma=3.0, reduction="mean")
    expected = sigmoid_focal_loss(logits, targets, alpha=0.4, gamma=3.0, reduction="mean")
    assert torch.allclose(loss(logits, targets), expected)


@pytest.mark.parametrize(
    "reduction, expected_shape",
    [("none", (4, 5)), ("mean", ()), ("sum", ())],
)
def test_reduction_modes(reduction: FocalLossReduction, expected_shape: tuple[int, ...]) -> None:
    logits, targets = _logits_and_targets()
    out = SigmoidFocalLoss(reduction=reduction)(logits, targets)
    assert out.shape == torch.Size(expected_shape)


def test_backward_flows_gradients() -> None:
    logits, targets = _logits_and_targets()
    logits = logits.requires_grad_(True)
    SigmoidFocalLoss()(logits, targets).backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_registered_as_classification_loss() -> None:
    registry = get_registry()
    assert "SigmoidFocalLoss" in registry.list_classes(task="classification", role="loss")
    assert "SigmoidFocalLoss" in registry.list_classes(category="classification_loss")
    assert SigmoidFocalLoss.__confluid_task__ == "classification"
    assert SigmoidFocalLoss.__confluid_role__ == "loss"
    assert SigmoidFocalLoss.__confluid_category__ == "classification_loss"
