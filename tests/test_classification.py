"""Tests for the classifier's use of the generic :func:`recordstream.collate_records`.

Sonair no longer ships a bespoke classification collate: the config pipeline tensorizes the input
(``ConvertToImage`` → ``ToTensor``) so each record's ``input_key`` value is already a ``[3, H, W]``
image at collate time. ``recordstream.collate_records`` stacks it and the model boundary
(``recordstream.batch_tensor`` / ``batch_metadata`` / ``multi_hot``) recovers the batched
input tensor, stacks the ``Label`` target's per-record list into an ``[N]`` int64 tensor, and
transposes the remaining plain keys into a per-record metadata list.
"""

import numpy as np
import pytest
import torch
from recordstream import Image, Label, Record, batch_metadata, batch_tensor, collate_records, item_data


def _chw(size: int = 8) -> np.ndarray:
    """A tensorized ``[3, H, W]`` float image (as the pipeline yields before collate)."""
    return np.random.rand(3, size, size).astype("float32")


def _record(image: np.ndarray, target: int, **meta: object) -> Record:
    """A per-record dict shaped like the post-pipeline HuggingFaceSource output."""
    record: Record = {"image": Image(image, layout="CHW"), "class": Label(target)}
    record.update(meta)
    return record


def test_collate_records_stacks_tensorized_inputs() -> None:
    batch = collate_records([_record(_chw(), i % 3, i=i) for i in range(4)])
    assert isinstance(batch, dict)
    x = batch_tensor(batch, "image")
    assert x.shape == (4, 3, 8, 8)
    assert x.dtype == torch.float32


def test_target_label_list_stacked_to_int64() -> None:
    batch = collate_records([_record(_chw(), i % 3) for i in range(4)])
    # collate_records leaves the Label target's value as a per-record list; the boundary stacks it.
    assert isinstance(batch["class"].value, list)
    y = batch_tensor(batch, "class", dtype=torch.int64)
    assert y.shape == (4,) and y.dtype == torch.int64


def test_metadata_preserved_via_batch_metadata() -> None:
    batch = collate_records([_record(_chw(), i % 3, idx=i, name=f"s{i}") for i in range(4)])
    metas = batch_metadata(batch, exclude=("image", "class"))
    assert isinstance(metas, list) and len(metas) == 4
    assert metas[2]["name"] == "s2"  # string metadata preserved
    assert metas[2]["idx"] == 2


def test_input_key_is_image_target_is_label() -> None:
    batch = collate_records([_record(_chw(), 0) for _ in range(2)])
    image_item = batch["image"]
    label_item = batch["class"]
    assert isinstance(image_item, Image)
    # Image cannot hold a live tensor — its payload is an ndarray recovered at the model boundary.
    assert isinstance(item_data(image_item), np.ndarray)
    assert isinstance(label_item, Label) and isinstance(label_item.value, list)


def test_live_tensor_inputs_collate_as_list_and_stack_at_boundary() -> None:
    """A live CHW ``torch.Tensor`` plain value (the ``ToTensor`` output) batches as a per-record
    list; ``batch_tensor`` stacks it to ``[N, 3, H, W]`` at the model boundary."""
    batch = collate_records([{"image": torch.from_numpy(_chw()), "class": Label(i)} for i in range(3)])
    x = batch_tensor(batch, "image")
    assert isinstance(x, torch.Tensor)
    assert x.shape == (3, 3, 8, 8) and x.dtype == torch.float32


def test_batch_target_uses_prestacked_tensor_verbatim() -> None:
    """A hand-built batch whose Label target already holds an ``[N]`` tensor is used as-is."""
    batch: Record = {
        "image": Image(np.zeros((2, 3, 4, 4), dtype="float32"), layout="CHW"),
        "class": Label(torch.tensor([5, 7])),
    }
    y = batch_tensor(batch, "class", dtype=torch.int64)
    assert y.tolist() == [5, 7]


def test_batch_metadata_reads_legacy_metadata_field() -> None:
    batch: Record = {
        "image": Image(np.zeros((2, 3, 4, 4), dtype="float32"), layout="CHW"),
        "class": Label(torch.tensor([0, 1])),
        "metadata": [{"idx": 0}, {"idx": 1}],
    }
    assert batch_metadata(batch, exclude=("image", "class")) == [{"idx": 0}, {"idx": 1}]


def test_collate_records_rejects_empty_batch() -> None:
    with pytest.raises(ValueError):
        collate_records([])
