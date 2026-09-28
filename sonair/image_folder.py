"""Classification datasets that live as a directory of images, one folder per class.

The layout every image-classification tool understands::

    root/
      train/            # optional split level
        0/  1234.png
        7/  0042.png
      test/
        ...

**Why this lives in sonair and not in recordstream.** A folder-per-class encodes exactly one
categorical value per image — the folder IS the label. That is a *classification* convention,
not a storage format, and the workspace already puts task-shaped dataset formats in the task
package: detection's YOLO writer lives in the detection package for the same reason.
recordstream's own sinks (``DirectorySink``, ``HDF5Sink``, ``ZarrGroupSink``) are the
task-AGNOSTIC ones — they serialize any record without interpreting what its entries mean.

It deliberately does NOT generalize to detection or segmentation: boxes and masks cannot be
expressed by a directory name. Those get their own writers in their own packages.
"""

from pathlib import Path
from typing import Any, Iterator, List, Optional, Union

from confluid import configurable, output
from loggair import get_logger
from PIL import Image as PILImage
from recordstream import Record, item_data
from recordstream.sources.huggingface import HuggingFaceSource
from recordstream.storage.base import Storage

logger = get_logger(__name__)

#: Record entry carrying the file a record was read from (stamped by the source).
SOURCE_FILE_KEY = "hf_file"


@configurable(category="source")
class ImageFolderSource:
    """Read a folder-per-class image dataset — ``root/[split/]<class>/*.png``.

    A thin, honestly-named front for the reader that already handles this layout: it delegates
    to :class:`~recordstream.sources.huggingface.HuggingFaceSource`, whose loader treats a local
    directory as an *imagefolder*. Delegating rather than re-walking the tree is the point —
    the caching, the class vocabulary read from the folder names, and the per-record provenance
    all come for free and stay in one implementation.

    What this class adds is discoverability: nothing about a node called "HuggingFaceSource"
    suggests it reads a local directory, so nobody looking for a folder reader ever found it.

    Args:
        root: Dataset root directory (a path; a string is accepted and converted). The folders
            under it — or under ``split`` — name the classes.
        split: Sub-directory to read when the dataset has a split level (``train`` / ``test``);
            the loader's own default applies when the tree has no split level.

    Example:
        >>> source = ImageFolderSource(root="~/data/mnist_png", split="test")
        >>> source.class_names                                   # doctest: +SKIP
        ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    """

    def __init__(self, root: Union[str, Path] = "", split: str = "train") -> None:
        # Union, stored as a real Path: a config document or a GUI widget hands a string, while
        # code that builds one directly deserves the type it means. Same shape as the storage
        # sinks in the record engine.
        self.root = Path(root)
        self.split = split
        self._reader: Optional[HuggingFaceSource] = None

    @property
    def reader(self) -> HuggingFaceSource:
        """The delegate, built on first use (the constructor does no work)."""
        if self._reader is None:
            if not str(self.root) or str(self.root) == ".":
                raise ValueError("ImageFolderSource.root is empty — set it to the dataset directory.")
            self._reader = HuggingFaceSource(path=str(self.root.expanduser()), split=self.split)
        return self._reader

    @property
    @output
    def class_names(self) -> List[str]:
        """The classes, in class-id order — read from the DIRECTORY NAMES, nothing else.

        A declared ``@output``, so the node exposes it as a socket you can wire into whatever
        states the graph's vocabulary.

        It deliberately does NOT go through the reader: asking the HF machinery for a LOCAL
        folder's class names triggers a FULL dataset prepare (the builder has no schema until a
        split is built), which meant the graph editor spent minutes copying 70 001 images into
        whatever cache its process happened to have — on every graph save, again per cache
        location — to answer a question ``listdir`` answers in milliseconds. Sorted, which is
        the imagefolder loader's own class-id ordering, so the names line up with the encoded
        labels the reader yields.
        """
        base = self.root.expanduser()
        if self.split and (base / self.split).is_dir():
            base = base / self.split
        if not base.is_dir():
            return []
        return sorted(d.name for d in base.iterdir() if d.is_dir() and not d.name.startswith("."))

    @property
    def dataset_uri(self) -> Optional[str]:
        """Identity, forwarded so a consumer names the dataset the same way it would any source."""
        return self.reader.dataset_uri

    def __len__(self) -> int:
        return len(self.reader)

    def __iter__(self) -> Iterator[Record]:
        return iter(self.reader)

    def __getitem__(self, index: int) -> Record:
        return self.reader[index]

    def project(self, keys: Any) -> Iterator[Record]:
        """Forward the projection protocol — the delegate's efficiency must survive delegation.

        Without this one method the wrapper silently DOWNGRADES its reader: a projection-aware
        consumer duck-types ``project``, finds nothing here, and falls back to full iteration —
        so a label-only filter decoded every image again. Measured through the annotation
        workspace's filter on 60 002 records: ~9.5 s wrapped against 0.26 s on the bare reader,
        the entire regression being this absent forward.
        """
        yield from self.reader.project(keys)


@configurable(category="sink")
class ImageFolderSink(Storage):
    """Write records back out as a folder-per-class image dataset.

    Wired as the ``sink:`` of a :class:`~recordstream.processing.DatasetProcessor`. Each record
    is written to ``root/[split/]<class name>/<original filename>``, where the class name comes
    from the record's own label vocabulary — so a relabelled record lands in its NEW class
    folder, which is what makes this the write half of a correction pass.

    ``delete_original`` makes it a MOVE: the file the record was read from is removed once the
    new one is written. It is guarded, because it destroys data that is not ours:

    * OFF by default;
    * the original is removed only AFTER the new file exists on disk;
    * never when source and destination resolve to the same file — a record whose class did not
      change must not be deleted;
    * a record carrying no source path is written and its (absent) original left alone, rather
      than failing the run.

    Args:
        root: Destination dataset root, created if missing (a path; a string is accepted).
        split: Optional split level written under ``root`` (``""`` = classes directly under it).
        delete_original: Remove the file the record came from once it has been rewritten (MOVE).
        overwrite: Allow replacing a destination file that already exists.
        image_key: Record entry carrying the image.
        target_key: Record entry carrying the label that names the folder.
    """

    def __init__(
        self,
        root: Union[str, Path] = "",
        split: str = "",
        delete_original: bool = False,
        overwrite: bool = False,
        image_key: str = "image",
        target_key: str = "class",
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.delete_original = bool(delete_original)
        self.overwrite = bool(overwrite)
        self.image_key = str(image_key)
        self.target_key = str(target_key)
        self._root: Optional[Path] = None
        self._written = 0
        self._moved = 0

    def open(self) -> "ImageFolderSink":
        if not str(self.root) or str(self.root) == ".":
            raise ValueError("ImageFolderSink.root is empty — set it to the destination directory.")
        self._root = self.root.expanduser()
        if self.split:
            self._root = self._root / self.split
        self._root.mkdir(parents=True, exist_ok=True)
        return self

    def class_folder(self, record: Record) -> str:
        """The folder this record belongs in — its class NAME when the label carries one.

        A ``Label`` knows its own vocabulary, so the folder is named rather than numbered; a
        label without one falls back to the raw value, which still round-trips (the reader turns
        the folder name back into a class).
        """
        label = record.get(self.target_key)
        value = getattr(label, "value", label)
        classes = list(getattr(label, "classes", None) or [])
        if isinstance(value, int) and 0 <= value < len(classes):
            return str(classes[value])
        return str(value)

    def write(self, record: Record) -> None:
        if self._root is None:
            raise RuntimeError("ImageFolderSink.write before open()")
        source_file = self._source_file(record)
        destination = self._destination(record, source_file)
        if destination.exists() and not self.overwrite:
            raise FileExistsError(f"ImageFolderSink: {destination} already exists (set overwrite to replace it)")

        destination.parent.mkdir(parents=True, exist_ok=True)
        PILImage.fromarray(_as_uint8(item_data(record[self.image_key]))).save(destination)
        self._written += 1

        if not self.delete_original or source_file is None:
            return
        if not destination.exists():  # pragma: no cover - defensive: the save above raises first
            return
        if source_file.resolve() == destination.resolve():
            # The class did not change, so the write landed on the original. Deleting it here
            # would remove the record from the dataset entirely.
            logger.debug(f"ImageFolderSink: {source_file} is its own destination; not deleting")
            return
        try:
            source_file.unlink()
            self._moved += 1
        except OSError as exc:  # a read-only source is not this write's failure
            logger.warning(f"ImageFolderSink: wrote {destination} but could not remove {source_file}: {exc}")

    def _source_file(self, record: Record) -> Optional[Path]:
        entry = record.get(SOURCE_FILE_KEY)
        value = getattr(entry, "value", entry)
        return Path(str(value)) if value else None

    def _destination(self, record: Record, source_file: Optional[Path]) -> Path:
        assert self._root is not None
        # Keep the original filename so a record stays recognisable across a relabel; a record
        # with no file of its own gets a stable sequential name.
        name = source_file.name if source_file is not None else f"{self._written:06d}.png"
        return self._root / self.class_folder(record) / name

    def flush(self) -> None:
        """Nothing is buffered — each record is written as it arrives."""

    def close(self) -> None:
        moved = f", moved {self._moved}" if self.delete_original else ""
        logger.info(f"ImageFolderSink: wrote {self._written} image(s) to {self._root}{moved}")


def _as_uint8(array: Any) -> Any:
    """An array PIL can save: uint8, and 2-D or HWC."""
    import numpy as np

    data = np.asarray(array)
    if data.dtype != np.uint8:
        finite = np.nan_to_num(data.astype(np.float64))
        top = float(finite.max()) if finite.size else 0.0
        data = (finite * (255.0 / top) if 0 < top <= 1.0 else finite).clip(0, 255).astype(np.uint8)
    return data


__all__ = ["ImageFolderSink", "ImageFolderSource", "SOURCE_FILE_KEY"]
