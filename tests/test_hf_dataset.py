"""Tests for :class:`sonair.hf_dataset.HuggingFaceClassificationDataset`.

Uses a mocked dataset to avoid hitting HF Hub during CI runs — the test
focuses on the wrapper's contract (Sample shape, dtype, metadata, label
mapping) rather than network connectivity.
"""

from typing import Any, List
from unittest.mock import patch

import pytest
import torch
from PIL import Image

from sonair.hf_dataset import HuggingFaceClassificationDataset


class _FakeFeature:
    """Stand-in for a HuggingFace ClassLabel feature with .names."""

    def __init__(self, names: List[str]) -> None:
        self.names = list(names)


class _FakeHFDataset:
    """In-memory HuggingFace-style dataset: image + label per item."""

    def __init__(self, items: List[dict], class_names: List[str]) -> None:
        self._items = list(items)
        self.features = {"label": _FakeFeature(class_names)}

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> dict:
        return self._items[idx]


@pytest.fixture
def fake_dataset() -> _FakeHFDataset:
    """Three samples, three classes, varied colors so resize doesn't homogenize."""
    items = [
        {"image": Image.new("RGB", (50, 30), color=(255, 0, 0)), "label": 0},
        {"image": Image.new("RGB", (40, 40), color=(0, 255, 0)), "label": 1},
        {"image": Image.new("RGB", (100, 100), color=(0, 0, 255)), "label": 2},
    ]
    return _FakeHFDataset(items, class_names=["red_drone", "green_drone", "blue_drone"])


def _build(fake_dataset: _FakeHFDataset, **kwargs: Any) -> HuggingFaceClassificationDataset:
    """Bypass HF Hub: patch load_dataset to return our fake dataset."""
    with patch("datasets.load_dataset", return_value=fake_dataset):
        return HuggingFaceClassificationDataset(repo_id="fake/RFUAV", **kwargs)


# --- Length, shape, and label mapping --------------------------------------


def test_len_matches_underlying_dataset(fake_dataset: _FakeHFDataset) -> None:
    ds = _build(fake_dataset)
    assert len(ds) == 3


def test_num_classes_picked_up_from_features(fake_dataset: _FakeHFDataset) -> None:
    ds = _build(fake_dataset)
    assert ds.num_classes == 3


def test_label_names_is_public_attribute(fake_dataset: _FakeHFDataset) -> None:
    """``label_names`` is part of the dataset's public interface — the marainer
    Trainer / Evaluator reads it via duck-typing to persist class_names.json
    and to label confusion-matrix axes."""
    ds = _build(fake_dataset)
    assert ds.label_names == ("red_drone", "green_drone", "blue_drone")


def test_label_names_empty_when_feature_has_no_names() -> None:
    """No ClassLabel feature → ``label_names`` is an empty tuple, not missing."""
    fake = _FakeHFDataset(items=[{"image": Image.new("RGB", (16, 16)), "label": 0}], class_names=[])
    fake.features = {}
    ds = _build(fake)
    assert ds.label_names == ()
    assert ds.num_classes == 0


def test_getitem_returns_correctly_shaped_sample(fake_dataset: _FakeHFDataset) -> None:
    ds = _build(fake_dataset, image_size=224)
    sample = ds[0]
    assert isinstance(sample.input, torch.Tensor)
    assert sample.input.shape == (3, 224, 224)
    assert sample.input.dtype == torch.float32
    assert 0.0 <= float(sample.input.min()) <= float(sample.input.max()) <= 1.0
    assert sample.target.shape == ()
    assert sample.target.dtype == torch.int64
    assert int(sample.target) == 0


def test_metadata_carries_label_name_and_index(fake_dataset: _FakeHFDataset) -> None:
    ds = _build(fake_dataset, split="custom_split")
    sample = ds[1]
    assert sample.metadata["label_idx"] == 1
    assert sample.metadata["label_name"] == "green_drone"
    assert sample.metadata["split"] == "custom_split"
    assert sample.metadata["repo_id"] == "fake/RFUAV"
    assert sample.metadata["index"] == 1


def test_resize_changes_spatial_dims(fake_dataset: _FakeHFDataset) -> None:
    ds = _build(fake_dataset, image_size=64)
    sample = ds[2]
    assert sample.input.shape == (3, 64, 64)


def test_no_resize_when_image_already_correct_size() -> None:
    """If the source image already matches image_size, the resize is a no-op."""
    fake = _FakeHFDataset(
        items=[{"image": Image.new("RGB", (32, 32), color=(128, 128, 128)), "label": 0}],
        class_names=["only_class"],
    )
    ds = _build(fake, image_size=32)
    sample = ds[0]
    assert sample.input.shape == (3, 32, 32)


# --- Custom field names ----------------------------------------------------


def test_custom_image_and_label_field_names() -> None:
    """Some HF datasets use non-default column names."""
    fake = _FakeHFDataset(
        items=[{"img": Image.new("RGB", (16, 16), color=(10, 20, 30)), "y": 5}],
        class_names=[],  # no ClassLabel feature; falls back to str(idx)
    )
    fake.features = {}  # nothing under the default key
    ds = _build(fake, image_field="img", label_field="y")
    sample = ds[0]
    assert int(sample.target) == 5
    # No ClassLabel → label_name falls back to stringified index.
    assert sample.metadata["label_name"] == "5"


# --- Mode coercion ---------------------------------------------------------


def test_grayscale_image_converted_to_rgb() -> None:
    """L-mode (grayscale) images must be converted to RGB before tensorizing."""
    fake = _FakeHFDataset(
        items=[{"image": Image.new("L", (32, 32), color=128), "label": 0}],
        class_names=["c"],
    )
    ds = _build(fake)
    sample = ds[0]
    assert sample.input.shape == (3, 224, 224)


def test_label_idx_out_of_class_names_falls_back_to_str() -> None:
    """If the label is outside the ClassLabel.names range, label_name → str(idx)."""
    fake = _FakeHFDataset(
        items=[{"image": Image.new("RGB", (16, 16)), "label": 99}],
        class_names=["a", "b"],
    )
    ds = _build(fake)
    sample = ds[0]
    assert sample.metadata["label_name"] == "99"
