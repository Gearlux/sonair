"""Tests for :mod:`sonair.classification` (loss + collate)."""

import math
from typing import List

import pytest
import torch

from dataflux.sample import Sample
from sonair.classification import ClassificationLoss, classification_collate_fn


def _make_logits_and_target(n: int = 4, c: int = 5) -> tuple:
    logits = torch.randn(n, c, requires_grad=True)
    target = torch.randint(0, c, (n,), dtype=torch.int64)
    return logits, target


# --- ClassificationLoss ----------------------------------------------------


def test_classification_loss_returns_finite_scalar_with_grad() -> None:
    loss_fn = ClassificationLoss()
    logits, target = _make_logits_and_target()
    loss = loss_fn(logits, target)
    assert loss.dim() == 0
    assert math.isfinite(float(loss.detach()))
    loss.backward()
    assert logits.grad is not None
    assert logits.grad.abs().sum() > 0


def test_classification_loss_accepts_int32_target_and_promotes_to_long() -> None:
    """Targets coming through some sources may be int32; we promote silently."""
    loss_fn = ClassificationLoss()
    logits = torch.randn(2, 3)
    target = torch.tensor([0, 2], dtype=torch.int32)
    loss = loss_fn(logits, target)
    assert math.isfinite(float(loss))


def test_label_smoothing_changes_loss_value() -> None:
    """Label smoothing >0 produces a strictly different loss on non-uniform logits.

    Note: with *uniform* logits (e.g. zeros), softmax is uniform and CE is
    log(C) regardless of smoothing — the smoothed target still sums to 1
    and every class has the same log-prob. Smoothing only matters once
    predictions are non-uniform, so we use logits that strongly prefer
    the correct class.
    """
    logits = torch.tensor([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0]])
    target = torch.tensor([0, 1], dtype=torch.int64)

    base = float(ClassificationLoss(label_smoothing=0.0)(logits, target))
    smooth = float(ClassificationLoss(label_smoothing=0.1)(logits, target))
    # Smoothed target spreads mass to the wrong classes, where the model
    # assigns near-zero probability — CE goes up.
    assert smooth > base


def test_class_weights_change_loss_value() -> None:
    """Up-weighting one class on an all-that-class batch yields a strictly larger sum-loss."""
    logits = torch.zeros(2, 2)
    target = torch.ones(2, dtype=torch.int64)

    base = ClassificationLoss(reduction="sum")(logits, target)
    weighted = ClassificationLoss(class_weights=[1.0, 5.0], reduction="sum")(
        logits, target
    )
    assert float(weighted) == pytest.approx(5.0 * float(base))


def test_ignore_index_excludes_samples_from_loss() -> None:
    """A target full of ignore_index produces zero (or NaN-tolerant) loss."""
    loss_fn = ClassificationLoss(ignore_index=-1)
    logits = torch.zeros(2, 2)
    target = torch.tensor([-1, -1], dtype=torch.int64)
    loss = loss_fn(logits, target)
    assert torch.isnan(loss) or float(loss) == 0.0


def test_class_weights_empty_list_raises() -> None:
    with pytest.raises(ValueError, match="class_weights"):
        ClassificationLoss(class_weights=[])


def test_classification_loss_rejects_wrong_logit_shape() -> None:
    loss_fn = ClassificationLoss()
    target = torch.zeros(2, dtype=torch.int64)
    bad_logits = torch.randn(2, 3, 4)  # 3D — not [N,C]
    with pytest.raises(ValueError, match=r"\[N,C\]"):
        loss_fn(bad_logits, target)


def test_classification_loss_rejects_wrong_target_shape() -> None:
    loss_fn = ClassificationLoss()
    logits = torch.randn(2, 3)
    bad_target = torch.zeros(2, 3, dtype=torch.int64)  # 2D — not [N]
    with pytest.raises(ValueError, match=r"\[N\]"):
        loss_fn(logits, bad_target)


def test_class_weights_buffer_moves_with_module() -> None:
    """After .to('cpu') the weight buffer is still on cpu and usable."""
    loss_fn = ClassificationLoss(class_weights=[1.0, 2.0]).to("cpu")
    logits = torch.zeros(2, 2)
    target = torch.tensor([0, 1], dtype=torch.int64)
    loss_fn(logits, target)


# --- classification_collate_fn ---------------------------------------------


def _make_sample(value: int = 0, h: int = 8, w: int = 8) -> Sample:
    return Sample(
        input=torch.full((3, h, w), float(value)),
        target=torch.tensor(value, dtype=torch.int64),
        metadata={"pack_id": f"pack{value}", "iq_file": f"f{value}.iq"},
    )


def test_collate_stacks_inputs_and_targets() -> None:
    batch = [_make_sample(0), _make_sample(1), _make_sample(2)]
    result = classification_collate_fn(batch)
    assert result.input.shape == (3, 3, 8, 8)
    assert result.target.shape == (3,)
    assert result.target.dtype == torch.int64
    assert result.target.tolist() == [0, 1, 2]


def test_collate_preserves_string_metadata_as_list() -> None:
    """Strings can't be torch.stack'd — they MUST end up in a per_sample list."""
    batch = [_make_sample(0), _make_sample(1)]
    result = classification_collate_fn(batch)
    metas = result.metadata["per_sample"]
    assert isinstance(metas, list)
    assert len(metas) == 2
    assert metas[0]["pack_id"] == "pack0"
    assert metas[1]["iq_file"] == "f1.iq"


def test_collate_promotes_non_int64_targets() -> None:
    s = Sample(
        input=torch.zeros(3, 4, 4),
        target=torch.tensor(1, dtype=torch.int32),
        metadata={},
    )
    result = classification_collate_fn([s])
    assert result.target.dtype == torch.int64


def test_collate_handles_empty_metadata_dict() -> None:
    s = Sample(
        input=torch.zeros(3, 4, 4),
        target=torch.tensor(0, dtype=torch.int64),
        metadata={},
    )
    result = classification_collate_fn([s])
    assert result.metadata["per_sample"] == [{}]


def test_collate_rejects_empty_batch() -> None:
    with pytest.raises(ValueError, match="empty batch"):
        classification_collate_fn([])


def test_collate_rejects_non_tensor_input() -> None:
    bad = Sample(
        input="not a tensor",  # type: ignore[arg-type]
        target=torch.tensor(0, dtype=torch.int64),
        metadata={},
    )
    with pytest.raises(TypeError, match="sample.input"):
        classification_collate_fn([bad])


def test_collate_rejects_non_tensor_target() -> None:
    bad = Sample(input=torch.zeros(3, 4, 4), target=0, metadata={})  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="sample.target"):
        classification_collate_fn([bad])


def test_collate_rejects_non_scalar_target() -> None:
    """Targets must be 0-D scalars (one class id per sample)."""
    bad = Sample(
        input=torch.zeros(3, 4, 4),
        target=torch.tensor([0, 1], dtype=torch.int64),  # 1-D — not allowed
        metadata={},
    )
    with pytest.raises(ValueError, match="0-D scalar"):
        classification_collate_fn([bad])


# --- end-to-end integration: collate → loss --------------------------------


def test_collate_then_loss_runs_end_to_end() -> None:
    """Sanity check: collate produces shapes the loss accepts."""
    batch: List[Sample] = [_make_sample(0), _make_sample(1)]
    collated = classification_collate_fn(batch)
    n = collated.input.shape[0]
    logits = torch.randn(n, 3)  # 3-class output
    loss = ClassificationLoss()(logits, collated.target)
    assert math.isfinite(float(loss))
