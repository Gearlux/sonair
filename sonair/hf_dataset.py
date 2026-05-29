"""HuggingFace image-classification dataset bridge for sonair.

Wraps a HuggingFace ``imagefolder``-style dataset (one ``image`` field of
PIL images, one ``label`` field of int class ids) into a
:class:`torch.utils.data.Dataset` whose items are
:class:`dataflux.sample.Sample` records — exactly what
:class:`marainer.trainer.LightningTrainer` and
:func:`sonair.classification.classification_collate_fn` expect.

The first concrete user is ``kitofrank/RFUAV``: a 37-class drone-type
spectrogram corpus where each example is a pre-rendered JPEG. No
IQ→spectrogram→render pipeline needed; sonair just classifies the image.

Class-count derivation (the source of truth for ``num_classes``)
----------------------------------------------------------------
The dataset is the authority on how many classes it contains, so the
class-count lives **here**, not in marainer (which stays task-agnostic) or
navigaitor (which only generates config). Two cooperating pieces:

* :meth:`HuggingFaceClassificationDataset.project` implements dataflux's
  :class:`~dataflux.projection.SupportsProjection` — a *target-only* walk
  reads just the label column and never decodes an image.
* :meth:`HuggingFaceClassificationDataset.num_classes` is a **lazy method**
  (never computed in ``__init__``) that walks every target via that
  projection and returns ``max(class_id) + 1``. :meth:`derived_dimensions`
  exposes it under the generic ``{"num_classes": …}`` contract the trainer
  forwards into a lazily-flowed model / metrics.

So a training config may **omit** ``num_classes`` entirely: when omitted,
navigaitor emits the model + metrics as ``!lazy:`` and marainer derives the
count from this dataset just before training. Provide it explicitly to skip
the walk (or for datasets that don't expose it).

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

from typing import Any, Collection, Dict, Iterator, Optional, Tuple

import numpy as np
import torch
from confluid import configurable
from dataflux.projection import num_classes as _derive_num_classes
from dataflux.sample import Sample
from logflow import get_logger
from PIL import Image
from torch.utils.data import Dataset

logger = get_logger(__name__)


@configurable(task="classification", role="dataset")
class HuggingFaceClassificationDataset(Dataset[Sample]):
    """Wrap a HuggingFace imagefolder dataset as a torch Dataset of Samples.

    Each item is a :class:`~dataflux.sample.Sample` carrying:

    * ``input`` — ``[3, image_size, image_size]`` float32 tensor in ``[0, 1]``.
    * ``target`` — 0-D int64 scalar tensor, the class id.
    * ``metadata`` — ``{"label_idx": int, "label_name": str, "split": str,
      "repo_id": str, "index": int}``.

    Public surface (informal protocols consumed by marainer Trainer /
    Evaluator):

    * ``label_names: Tuple[str, ...]`` — human-readable class names in
      class-id order, sourced from the HF feature's ``names`` list (a cheap
      feature read, cached at construction). Empty tuple when the dataset has
      no labeled feature. Consumed by ``_resolve_label_names`` for
      ``class_names.json``.
    * :meth:`num_classes` — a **lazy method** (NOT an attribute, NOT computed
      in ``__init__``) that walks every target and returns ``max(id) + 1``.
    * :meth:`derived_dimensions` — ``{"num_classes": self.num_classes()}``,
      the generic dimension contract marainer forwards into lazily-flowed
      model / metrics when a config omits ``num_classes``.
    * :meth:`project` — efficient field projection (target-only skips image
      decode), implementing :class:`dataflux.projection.SupportsProjection`.

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

        # Cache the human-readable label names (when present) for metadata /
        # class_names.json. This is a cheap *feature* read, not a class-count
        # walk — ``num_classes`` is derived lazily, on demand, via project().
        feature = getattr(self._ds, "features", {}).get(label_field) if hasattr(self._ds, "features") else None
        names = getattr(feature, "names", None)
        self.label_names: Tuple[str, ...] = tuple(names) if names else ()

        suffix = f", {len(self.label_names)} label names" if self.label_names else ""
        logger.info(f"HuggingFaceClassificationDataset: loaded {repo_id}@{split} — {len(self._ds)} samples{suffix}")

    def __len__(self) -> int:
        return int(len(self._ds))

    # ------------------------------------------------------------------ #
    # Sample construction helpers (shared by __getitem__ and project())
    # ------------------------------------------------------------------ #

    def _image_tensor(self, item: Dict[str, Any]) -> torch.Tensor:
        """PIL image → ``[3, H, W]`` float32 tensor in ``[0, 1]``.

        We avoid waivefront's ``_pil_to_chw_float_tensor`` here so sonair stays
        free of a waivefront dependency (sonair is task-generic; waivefront is
        signal-specific).
        """
        img: Image.Image = item[self.image_field]
        if img.mode != "RGB":
            img = img.convert("RGB")
        if img.size != (self.image_size, self.image_size):
            img = img.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        arr = np.array(img, dtype=np.uint8, copy=True)  # writable
        return torch.from_numpy(arr).permute(2, 0, 1).contiguous().to(torch.float32) / 255.0

    @staticmethod
    def _target_tensor(label_idx: int) -> torch.Tensor:
        return torch.tensor(label_idx, dtype=torch.int64)

    def _metadata(self, label_idx: int, index: int) -> Dict[str, Any]:
        label_name = self.label_names[label_idx] if 0 <= label_idx < len(self.label_names) else str(label_idx)
        return {
            "label_idx": label_idx,
            "label_name": label_name,
            "split": self.split,
            "repo_id": self.repo_id,
            "index": int(index),
        }

    def __getitem__(self, index: int) -> Sample:
        item = self._ds[int(index)]
        label_idx = int(item[self.label_field])
        return Sample(
            input=self._image_tensor(item),
            target=self._target_tensor(label_idx),
            metadata=self._metadata(label_idx, int(index)),
        )

    # ------------------------------------------------------------------ #
    # dataflux SupportsProjection + class-count derivation
    # ------------------------------------------------------------------ #

    def project(self, fields: Collection[str]) -> Iterator[Sample]:
        """Yield Samples carrying only ``fields`` — skipping image decode when possible.

        Implements :class:`dataflux.projection.SupportsProjection`. When the
        ``input`` is not requested (the class-count walk asks for ``target``
        only) we read the label **column** directly, so no image is ever
        decoded — the efficiency win the projection primitive exists for.
        ``fields`` is a subset of ``{"input", "target", "metadata"}``.
        """
        want = frozenset(fields)
        need_input = "input" in want
        need_target = "target" in want
        need_meta = "metadata" in want

        if not need_input:
            # Column read — never materializes/decodes the image feature.
            for index, label in enumerate(self._ds[self.label_field]):
                label_idx = int(label)
                yield Sample(
                    input=None,
                    target=self._target_tensor(label_idx) if need_target else None,
                    metadata=self._metadata(label_idx, index) if need_meta else {},
                )
            return

        for index in range(len(self)):
            item = self._ds[int(index)]
            label_idx = int(item[self.label_field])
            yield Sample(
                input=self._image_tensor(item),
                target=self._target_tensor(label_idx) if need_target else None,
                metadata=self._metadata(label_idx, index) if need_meta else {},
            )

    def num_classes(self) -> int:
        """Number of classes, derived by walking every target (lazy; never on init).

        Always walks the target stream (target-only, so no images are decoded)
        and returns ``max(class_id) + 1`` — robust to a class id absent from
        this split. Computed on each call by design: the cost is paid only when
        a caller (marainer's setup phase) actually needs it.
        """
        return _derive_num_classes(self)

    def derived_dimensions(self) -> Dict[str, Any]:
        """Generic model dimensions this dataset can supply (the trainer forwards these).

        Returns ``{"num_classes": self.num_classes()}``. marainer reads this
        duck-typed and injects the entries into a lazily-flowed model / metrics
        — so the trainer never hardcodes ``num_classes`` while the *semantics*
        stay here in the classification dataset.
        """
        return {"num_classes": self.num_classes()}


__all__ = ["HuggingFaceClassificationDataset"]
