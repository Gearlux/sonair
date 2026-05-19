"""HuggingFace image-classification dataset bridge for sonair.

Wraps a HuggingFace ``imagefolder``-style dataset (one ``image`` field of
PIL images, one ``label`` field of int class ids) into a
:class:`torch.utils.data.Dataset` whose items are
:class:`dataflux.sample.Sample` records — exactly what
:class:`marainer.trainer.Trainer` and
:func:`sonair.classification.classification_collate_fn` expect.

The first concrete user is ``kitofrank/RFUAV``: a 37-class drone-type
spectrogram corpus where each example is a pre-rendered JPEG. No
IQ→spectrogram→render pipeline needed; sonair just classifies the image.

Typical wiring (YAML)::

    train_set: !class:sonair.hf_dataset.HuggingFaceClassificationDataset
      repo_id: "kitofrank/RFUAV"
      split: train
      image_size: 224

    val_set: !class:sonair.hf_dataset.HuggingFaceClassificationDataset
      repo_id: "kitofrank/RFUAV"
      split: validation
      image_size: 224
"""

from typing import Any, Optional, Tuple

import torch
from confluid import configurable
from dataflux.sample import Sample
from logflow import get_logger
from PIL import Image
from torch.utils.data import Dataset

logger = get_logger(__name__)


@configurable
class HuggingFaceClassificationDataset(Dataset[Sample]):
    """Wrap a HuggingFace imagefolder dataset as a torch Dataset of Samples.

    Each item is a :class:`~dataflux.sample.Sample` carrying:

    * ``input`` — ``[3, image_size, image_size]`` float32 tensor in ``[0, 1]``.
    * ``target`` — 0-D int64 scalar tensor, the class id.
    * ``metadata`` — ``{"label_idx": int, "label_name": str, "split": str,
      "repo_id": str, "index": int}``.

    Public attributes (informal protocol consumed by marainer Trainer /
    Evaluator):

    * ``label_names: Tuple[str, ...]`` — human-readable class names in
      class-id order, sourced from the HF feature's ``names`` list. Empty
      tuple when the dataset has no labeled feature.
    * ``num_classes: int`` — ``len(label_names)``.

    Args:
        repo_id: HuggingFace dataset identifier (e.g. ``"kitofrank/RFUAV"``).
        split: Which split to load (``"train"``, ``"validation"``, ``"test"``,
            etc.). The dataset must expose this split — call
            ``datasets.get_dataset_split_names(repo_id)`` if unsure.
        image_size: Square edge length to resize each image to before
            tensorizing. Defaults to ``224`` (efficientvit_b0 / resnet50 /
            most timm classifiers expect this resolution).
        image_field: Name of the image column in the HF dataset. Defaults
            to ``"image"`` (the imagefolder convention).
        label_field: Name of the integer-label column. Defaults to
            ``"label"`` (the imagefolder convention).
        cache_dir: Optional override for HuggingFace's local cache
            directory. ``None`` (default) uses the standard
            ``~/.cache/huggingface/datasets`` location.

    Why eager dataset materialization (no streaming): random-access
    ``__getitem__`` is required by Lightning's standard map-style
    ``DataLoader`` — streaming yields a one-pass iterable that breaks
    ``shuffle=True`` and ``num_workers > 1``. The HF cache amortizes the
    download across runs, so the up-front cost only hits once.
    """

    def __init__(
        self,
        repo_id: str,
        split: str = "train",
        image_size: int = 224,
        image_field: str = "image",
        label_field: str = "label",
        cache_dir: Optional[str] = None,
    ) -> None:
        from datasets import load_dataset

        self.repo_id = repo_id
        self.split = split
        self.image_size = int(image_size)
        self.image_field = image_field
        self.label_field = label_field
        self._ds: Any = load_dataset(repo_id, split=split, cache_dir=cache_dir)

        # Cache the human-readable label names (when present) for metadata.
        feature = getattr(self._ds, "features", {}).get(label_field) if hasattr(self._ds, "features") else None
        names = getattr(feature, "names", None)
        self.label_names: Tuple[str, ...] = tuple(names) if names else ()
        self.num_classes = len(self.label_names) if self.label_names else 0

        logger.info(
            f"HuggingFaceClassificationDataset: loaded {repo_id}@{split} — "
            f"{len(self._ds)} samples, {self.num_classes} classes"
        )

    def __len__(self) -> int:
        return int(len(self._ds))

    def __getitem__(self, index: int) -> Sample:
        item = self._ds[int(index)]
        img: Image.Image = item[self.image_field]
        if img.mode != "RGB":
            img = img.convert("RGB")
        if img.size != (self.image_size, self.image_size):
            img = img.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)

        # PIL → [3, H, W] float in [0, 1]. We avoid waivefront's
        # `_pil_to_chw_float_tensor` here so sonair stays free of a
        # waivefront dependency (sonair is task-generic; waivefront is
        # signal-specific).
        import numpy as np

        arr = np.array(img, dtype=np.uint8, copy=True)  # writable
        tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous().to(torch.float32) / 255.0

        label_idx = int(item[self.label_field])
        label_name = self.label_names[label_idx] if 0 <= label_idx < len(self.label_names) else str(label_idx)

        return Sample(
            input=tensor,
            target=torch.tensor(label_idx, dtype=torch.int64),
            metadata={
                "label_idx": label_idx,
                "label_name": label_name,
                "split": self.split,
                "repo_id": self.repo_id,
                "index": int(index),
            },
        )


__all__ = ["HuggingFaceClassificationDataset"]
