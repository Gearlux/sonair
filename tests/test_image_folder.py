"""Folder-per-class datasets: the reader front and the writer that can MOVE files.

These live in sonair, not recordstream, because a folder-per-class encodes exactly one
categorical value per image — the folder IS the label, which is a classification convention
rather than a storage format. recordstream's own sinks serialize any record without
interpreting it; the detection package owns the YOLO writer for the same reason.

``delete_original`` destroys data that is not ours, so most of what is pinned here is when it
must NOT fire.
"""

from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import pytest
from PIL import Image as PILImage
from recordstream import Image, Label

from sonair.image_folder import SOURCE_FILE_KEY, ImageFolderSink, ImageFolderSource


def _image(value: int = 0) -> Image:
    return Image(np.full((4, 4), value, dtype=np.uint8))


def _record(class_index: int, classes: Optional[List[str]], source_file: Optional[Path] = None) -> dict:
    record: dict = {"image": _image(), "class": Label(value=class_index, classes=classes)}
    if source_file is not None:
        record[SOURCE_FILE_KEY] = Label(str(source_file))
    return record


def _on_disk(root: Path) -> List[str]:
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*.png"))


class TestTheFolderNamesTheClass:
    def test_the_class_NAME_is_the_folder_when_the_label_carries_a_vocabulary(self, tmp_path: Path) -> None:
        with ImageFolderSink(root=str(tmp_path)) as sink:
            sink.write(_record(1, ["cat", "dog"]))
        assert _on_disk(tmp_path) == ["dog/000000.png"]

    def test_a_label_without_a_vocabulary_falls_back_to_its_value(self, tmp_path: Path) -> None:
        """Still round-trips: the reader turns the folder name back into a class."""
        with ImageFolderSink(root=str(tmp_path)) as sink:
            sink.write(_record(7, None))
        assert _on_disk(tmp_path) == ["7/000000.png"]

    def test_a_split_level_is_written_when_asked(self, tmp_path: Path) -> None:
        with ImageFolderSink(root=str(tmp_path), split="test") as sink:
            sink.write(_record(0, ["cat", "dog"]))
        assert _on_disk(tmp_path) == ["test/cat/000000.png"]

    def test_the_original_filename_is_kept(self, tmp_path: Path) -> None:
        """A record stays recognisable across a relabel."""
        origin = tmp_path / "in" / "cat" / "1234.png"
        origin.parent.mkdir(parents=True)
        PILImage.fromarray(np.zeros((4, 4), np.uint8)).save(origin)
        out = tmp_path / "out"
        with ImageFolderSink(root=str(out)) as sink:
            sink.write(_record(1, ["cat", "dog"], origin))
        assert _on_disk(out) == ["dog/1234.png"]


class TestDeleteOriginalIsAMove:
    """It removes a file the user gave us. Every guard here exists to not lose their data."""

    def _origin(self, tmp_path: Path, folder: str = "cat", name: str = "1234.png") -> Path:
        origin = tmp_path / "in" / folder / name
        origin.parent.mkdir(parents=True, exist_ok=True)
        PILImage.fromarray(np.zeros((4, 4), np.uint8)).save(origin)
        return origin

    def test_off_by_default_the_original_survives(self, tmp_path: Path) -> None:
        origin = self._origin(tmp_path)
        with ImageFolderSink(root=str(tmp_path / "out")) as sink:
            sink.write(_record(1, ["cat", "dog"], origin))
        assert origin.exists(), "a copy must never delete anything"

    def test_ON_it_moves_the_file(self, tmp_path: Path) -> None:
        origin = self._origin(tmp_path)
        out = tmp_path / "out"
        with ImageFolderSink(root=str(out), delete_original=True) as sink:
            sink.write(_record(1, ["cat", "dog"], origin))
        assert _on_disk(out) == ["dog/1234.png"]
        assert not origin.exists()

    def test_a_record_relabelled_to_its_OWN_folder_is_never_deleted(self, tmp_path: Path) -> None:
        """The destination IS the original — deleting would remove it from the dataset."""
        root = tmp_path / "set"
        origin = root / "cat" / "1234.png"
        origin.parent.mkdir(parents=True)
        PILImage.fromarray(np.zeros((4, 4), np.uint8)).save(origin)
        with ImageFolderSink(root=str(root), delete_original=True, overwrite=True) as sink:
            sink.write(_record(0, ["cat", "dog"], origin))
        assert origin.exists(), "writing onto the original must not then delete it"

    def test_a_record_with_no_source_file_is_written_anyway(self, tmp_path: Path) -> None:
        """Nothing to move; the write must not fail over it."""
        out = tmp_path / "out"
        with ImageFolderSink(root=str(out), delete_original=True) as sink:
            sink.write(_record(1, ["cat", "dog"], None))
        assert _on_disk(out) == ["dog/000000.png"]

    def test_an_unremovable_original_leaves_the_write_standing(self, tmp_path: Path, monkeypatch: Any) -> None:
        """A read-only source is not this write's failure — the new file is already correct."""
        origin = self._origin(tmp_path)
        out = tmp_path / "out"

        def refuse(self: Path) -> None:
            raise OSError("read-only file system")

        monkeypatch.setattr(Path, "unlink", refuse)
        with ImageFolderSink(root=str(out), delete_original=True) as sink:
            sink.write(_record(1, ["cat", "dog"], origin))
        assert _on_disk(out) == ["dog/1234.png"]


class TestRefusals:
    def test_an_existing_destination_is_refused_unless_overwrite(self, tmp_path: Path) -> None:
        """Two records that came from the same filename land on the same destination."""
        origin = tmp_path / "in" / "1234.png"
        origin.parent.mkdir(parents=True)
        PILImage.fromarray(np.zeros((4, 4), np.uint8)).save(origin)
        with ImageFolderSink(root=str(tmp_path / "out")) as sink:
            sink.write(_record(1, ["cat", "dog"], origin))
            with pytest.raises(FileExistsError, match="overwrite"):
                sink.write(_record(1, ["cat", "dog"], origin))

    def test_file_less_records_get_distinct_names_rather_than_colliding(self, tmp_path: Path) -> None:
        """A record with no file of its own is named sequentially, so a batch never self-collides."""
        with ImageFolderSink(root=str(tmp_path)) as sink:
            sink.write(_record(1, ["cat", "dog"]))
            sink.write(_record(1, ["cat", "dog"]))
        assert _on_disk(tmp_path) == ["dog/000000.png", "dog/000001.png"]

    def test_a_sink_without_a_root_says_so(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="root is empty"):
            ImageFolderSink().open()

    def test_a_source_without_a_root_says_so(self) -> None:
        with pytest.raises(ValueError, match="root is empty"):
            ImageFolderSource().reader

    def test_writing_before_open_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="before open"):
            ImageFolderSink(root=str(tmp_path)).write(_record(0, ["cat"]))


class TestTheyAreCanvasNodes:
    def test_the_source_and_sink_carry_the_categories_a_visual_editor_reads(self) -> None:
        from confluid import marks

        assert marks(ImageFolderSource).category == "source"
        assert marks(ImageFolderSink).category == "sink"

    def test_both_build_with_no_arguments(self) -> None:
        """A template PLACES them before anything is configured."""
        assert ImageFolderSource().split == "train" and ImageFolderSink().delete_original is False

    def test_root_is_a_PATH_even_when_given_a_string(self) -> None:
        """A config document or a widget hands a string; code that builds one means a path."""
        assert isinstance(ImageFolderSource(root="/data/x").root, Path)
        assert isinstance(ImageFolderSink(root="/data/x").root, Path)
        assert ImageFolderSink(root=Path("/data/x")).root == Path("/data/x")


class TestTheDelegateKeepsItsReadersEfficiency:
    """The wrapper must forward `project`, or it silently downgrades the reader.

    A projection-aware consumer duck-types the method; absent here, it falls back to full
    iteration and a label-only filter decodes every image again — measured as ~9.5s wrapped
    against 0.26s bare, on 60 002 records.
    """

    def test_project_is_forwarded_to_the_reader(self) -> None:
        class _Reader:
            def __init__(self) -> None:
                self.asked: list = []

            def project(self, keys: Any) -> Any:
                self.asked.append(set(keys))
                yield {"class": 1}

        source = ImageFolderSource(root="/data/x")
        reader = _Reader()
        source._reader = reader  # type: ignore[assignment]
        rows = list(source.project({"class"}))
        assert reader.asked == [{"class"}]
        assert rows == [{"class": 1}]

    def test_the_wrapper_satisfies_the_projection_protocol(self) -> None:
        from recordstream.projection import SupportsProjection

        assert isinstance(ImageFolderSource(root="/data/x"), SupportsProjection)


class TestClassNamesAreTheDirectoryNames:
    """Reading the vocabulary must never build the dataset.

    Asking the HF machinery for a LOCAL folder's names triggers a full prepare (no schema
    until a split is built) — measured as the graph editor spending minutes copying 70 001
    images into its cache on every save, to answer a listdir question. Reported live as a
    2-minute sync timeout with "still nothing".
    """

    def _tree(self, tmp_path: Path, with_split: bool = True) -> Path:
        root = tmp_path / "set"
        base = root / "train" if with_split else root
        for cls in ("7", "0", "3"):
            (base / cls).mkdir(parents=True)
        return root

    def test_the_names_are_the_sorted_folder_names(self, tmp_path: Path) -> None:
        source = ImageFolderSource(root=str(self._tree(tmp_path)), split="train")
        assert source.class_names == ["0", "3", "7"], "sorted = the imagefolder loader's own id order"

    def test_a_tree_without_a_split_level_works_too(self, tmp_path: Path) -> None:
        source = ImageFolderSource(root=str(self._tree(tmp_path, with_split=False)), split="train")
        assert source.class_names == ["0", "3", "7"]

    def test_reading_them_NEVER_builds_the_dataset(self, tmp_path: Path) -> None:
        source = ImageFolderSource(root=str(self._tree(tmp_path)), split="train")
        assert source.class_names == ["0", "3", "7"]
        assert source._reader is None, "the reader (and its dataset prepare) must not have been touched"

    def test_a_missing_root_is_empty_not_an_error(self, tmp_path: Path) -> None:
        assert ImageFolderSource(root=str(tmp_path / "nope")).class_names == []
