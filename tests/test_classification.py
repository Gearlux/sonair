"""Tests for sonair.classification.classification_collate_fn (image-coercing collate)."""

import numpy as np
import pytest
import torch
from PIL import Image
from sampleflux.sample import Sample

from sonair.classification import classification_collate_fn


def _pil(mode: str, size: int = 8) -> Image.Image:
    arr = (np.random.rand(size, size, 3 if mode == "RGB" else 1) * 255).astype("uint8")
    if mode == "L":
        return Image.fromarray(arr[:, :, 0], mode="L")
    return Image.fromarray(arr, mode="RGB")


def test_grayscale_pil_becomes_3channel_float_batch() -> None:
    batch = [Sample(input=_pil("L"), target=i % 3, metadata={"i": i}) for i in range(4)]
    out = classification_collate_fn(batch)
    assert isinstance(out.input, torch.Tensor)
    assert out.input.shape == (4, 3, 8, 8)  # grayscale expanded to 3 channels
    assert out.input.dtype == torch.float32
    assert 0.0 <= float(out.input.min()) and float(out.input.max()) <= 1.0  # scaled to [0,1]


def test_rgb_pil_batch() -> None:
    batch = [Sample(input=_pil("RGB"), target=1, metadata={}) for _ in range(3)]
    out = classification_collate_fn(batch)
    assert out.input.shape == (3, 3, 8, 8)


def test_tensor_inputs_pass_through() -> None:
    batch = [Sample(input=torch.rand(3, 8, 8), target=i, metadata={}) for i in range(2)]
    out = classification_collate_fn(batch)
    assert out.input.shape == (2, 3, 8, 8)


def test_targets_stacked_int64_and_metadata_list() -> None:
    batch = [Sample(input=_pil("RGB"), target=i % 3, metadata={"idx": i, "name": f"s{i}"}) for i in range(4)]
    out = classification_collate_fn(batch)
    assert out.target.shape == (4,) and out.target.dtype == torch.int64
    assert isinstance(out.metadata, list) and len(out.metadata) == 4
    assert out.metadata[2]["name"] == "s2"  # string metadata preserved


def test_empty_batch_raises() -> None:
    with pytest.raises(ValueError):
        classification_collate_fn([])


def test_mismatched_spatial_dims_raise() -> None:
    # No resize wired — differing image sizes can't be stacked.
    batch = [
        Sample(input=_pil("RGB", size=8), target=0, metadata={}),
        Sample(input=_pil("RGB", size=16), target=1, metadata={}),
    ]
    with pytest.raises(ValueError):
        classification_collate_fn(batch)
