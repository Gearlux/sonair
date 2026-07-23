"""Tests for the classifier's use of the generic :func:`sampleflux.typed_collate`.

Sonair no longer ships a bespoke classification collate: the config pipeline tensorizes the input
(``ConvertToImageOp`` → ``ToTensorOp``) so each ``Sample``'s primary ``input`` is already a
``[3, H, W]`` :class:`~sampleflux.Image` at collate time. ``sampleflux.typed_collate`` stacks it and
the model boundary (``sonair.classifier._batch_input`` / ``_batch_target`` / ``_batch_metadata``)
recovers the batched input tensor, stacks the ``Label`` target's per-sample list into an ``[N]`` int64
tensor, and transposes the aux fields into a per-sample metadata list.
"""

import numpy as np
import pytest
import torch
from sampleflux import Image, Label, Sample, item_data, primary, typed_collate

from sonair.classifier import _batch_input, _batch_metadata, _batch_target


def _chw(size: int = 8) -> np.ndarray:
    """A tensorized ``[3, H, W]`` float image (as the pipeline yields before collate)."""
    return np.random.rand(3, size, size).astype("float32")


def _sample(image: np.ndarray, target: int, **meta: object) -> Sample:
    """A per-sample Sample shaped like the post-pipeline HuggingFaceSource output."""
    fields: dict = {"image": Image(image, layout="CHW"), "class": Label(target)}
    roles: dict = {"class": "target"}
    for key, value in meta.items():
        fields[key] = Label(value)
        roles[key] = "aux"
    return Sample(fields, roles)


def test_typed_collate_stacks_tensorized_inputs() -> None:
    batch = typed_collate([_sample(_chw(), i % 3, i=i) for i in range(4)])
    assert isinstance(batch, Sample)
    x = _batch_input(batch)
    assert x.shape == (4, 3, 8, 8)
    assert x.dtype == torch.float32


def test_target_label_list_stacked_to_int64() -> None:
    batch = typed_collate([_sample(_chw(), i % 3) for i in range(4)])
    # typed_collate leaves the Label target's value as a per-sample list; the boundary stacks it.
    assert isinstance(primary(batch, "target")[1].value, list)
    y = _batch_target(batch)
    assert y.shape == (4,) and y.dtype == torch.int64


def test_metadata_preserved_via_batch_metadata() -> None:
    batch = typed_collate([_sample(_chw(), i % 3, idx=i, name=f"s{i}") for i in range(4)])
    metas = _batch_metadata(batch)
    assert isinstance(metas, list) and len(metas) == 4
    assert metas[2]["name"] == "s2"  # string metadata preserved
    assert metas[2]["idx"] == 2


def test_input_field_is_image_target_is_label() -> None:
    batch = typed_collate([_sample(_chw(), 0) for _ in range(2)])
    _, image_item = primary(batch, "input")
    _, label_item = primary(batch, "target")
    assert isinstance(image_item, Image)
    # Image cannot hold a live tensor — its payload is an ndarray recovered at the model boundary.
    assert isinstance(item_data(image_item), np.ndarray)
    assert isinstance(label_item, Label) and isinstance(label_item.value, list)


def test_batch_target_uses_prestacked_tensor_verbatim() -> None:
    """A hand-built batch whose Label target already holds an ``[N]`` tensor is used as-is."""
    batch = Sample(
        {"image": Image(np.zeros((2, 3, 4, 4), dtype="float32"), layout="CHW"), "class": Label(torch.tensor([5, 7]))},
        roles={"class": "target"},
    )
    y = _batch_target(batch)
    assert y.tolist() == [5, 7]


def test_batch_metadata_reads_legacy_metadata_field() -> None:
    batch = Sample(
        {
            "image": Image(np.zeros((2, 3, 4, 4), dtype="float32"), layout="CHW"),
            "class": Label(torch.tensor([0, 1])),
            "metadata": [{"idx": 0}, {"idx": 1}],
        },
        roles={"class": "target", "metadata": "aux"},
    )
    assert _batch_metadata(batch) == [{"idx": 0}, {"idx": 1}]


def test_typed_collate_rejects_empty_batch() -> None:
    with pytest.raises(ValueError):
        typed_collate([])
