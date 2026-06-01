"""Tests for :mod:`sonair.classification` (collate fn).

The classification loss is ``torch.nn.CrossEntropyLoss`` (registered in
``sonair.classification`` as the ``task="classification", role="loss"``
configurable — no sonair wrapper class), so there's nothing sonair-specific to
test about the loss itself; the collate → loss integration check below confirms
the collate output is shaped the way CrossEntropyLoss expects.
"""

import math
from typing import List

import pytest
import torch
import torch.nn as nn
from dataflux.sample import Sample

from sonair.classification import classification_collate_fn


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
    """Strings can't be torch.stack'd — they MUST end up in a metadata list."""
    batch = [_make_sample(0), _make_sample(1)]
    result = classification_collate_fn(batch)
    metas = result.metadata
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
    assert result.metadata == [{}]


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
    """Sanity check: collate produces shapes torch.nn.CrossEntropyLoss accepts."""
    batch: List[Sample] = [_make_sample(0), _make_sample(1)]
    collated = classification_collate_fn(batch)
    n = collated.input.shape[0]
    logits = torch.randn(n, 3)  # 3-class output
    loss = nn.CrossEntropyLoss()(logits, collated.target)
    assert math.isfinite(float(loss))
