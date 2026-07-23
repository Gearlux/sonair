"""Tests for sonair.classification.classification_collate_fn (typed-bag image-coercing collate)."""

import numpy as np
import pytest
import torch
from sampleflux import Image, Label, Sample, item_data, primary

from sonair.classification import classification_collate_fn
from sonair.classifier import _batch_input, _batch_target


def _sample(input_value: object, target: object, **meta: object) -> Sample:
    """A per-sample Sample shaped like HuggingFaceSource output: Image input + Label target + aux."""
    fields: dict = {
        "image": Image(np.asarray(input_value)) if not isinstance(input_value, torch.Tensor) else input_value
    }
    fields["class"] = Label(target)
    roles: dict = {"class": "target"}
    for key, value in meta.items():
        fields[key] = Label(value)
        roles[key] = "aux"
    return Sample(fields, roles)


def _gray(size: int = 8) -> np.ndarray:
    return (np.random.rand(size, size).astype("float32") * 255).astype("uint8")


def _rgb(size: int = 8) -> np.ndarray:
    return (np.random.rand(size, size, 3) * 255).astype("uint8")


def test_grayscale_becomes_3channel_float_batch() -> None:
    batch = [_sample(_gray(), i % 3, i=i) for i in range(4)]
    out = classification_collate_fn(batch)
    assert isinstance(out, Sample)
    x = _batch_input(out)
    assert x.shape == (4, 3, 8, 8)  # grayscale expanded to 3 channels
    assert x.dtype == torch.float32
    assert 0.0 <= float(x.min()) and float(x.max()) <= 1.0  # scaled to [0,1]


def test_rgb_batch() -> None:
    batch = [_sample(_rgb(), 1) for _ in range(3)]
    out = classification_collate_fn(batch)
    assert _batch_input(out).shape == (3, 3, 8, 8)


def test_raw_tensor_input_field_passes_through() -> None:
    # A field holding a raw CHW tensor (not wrapped in an Image item) is used verbatim.
    batch = [_sample(torch.rand(3, 8, 8), i) for i in range(2)]
    out = classification_collate_fn(batch)
    assert _batch_input(out).shape == (2, 3, 8, 8)


def test_targets_stacked_int64_and_metadata_list() -> None:
    batch = [_sample(_rgb(), i % 3, idx=i, name=f"s{i}") for i in range(4)]
    out = classification_collate_fn(batch)
    y = _batch_target(out)
    assert y.shape == (4,) and y.dtype == torch.int64
    metas = out["metadata"]
    assert isinstance(metas, list) and len(metas) == 4
    assert metas[2]["name"] == "s2"  # string metadata preserved


def test_input_field_is_image_target_is_label() -> None:
    out = classification_collate_fn([_sample(_rgb(), 0) for _ in range(2)])
    _, image_item = primary(out, "input")
    _, label_item = primary(out, "target")
    assert isinstance(image_item, Image) and image_item.layout == "CHW"
    # Image cannot hold a live tensor — its payload is an ndarray recovered at the model boundary.
    assert isinstance(item_data(image_item), np.ndarray)
    assert isinstance(label_item, Label) and isinstance(label_item.value, torch.Tensor)


def test_empty_batch_raises() -> None:
    with pytest.raises(ValueError):
        classification_collate_fn([])


def test_mismatched_spatial_dims_raise() -> None:
    # No resize wired — differing image sizes can't be stacked.
    batch = [_sample(_rgb(size=8), 0), _sample(_rgb(size=16), 1)]
    with pytest.raises(ValueError):
        classification_collate_fn(batch)
