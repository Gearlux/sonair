# mypy: disable-error-code="attr-defined"
"""``HuggingFaceClassifier`` — the ``transformers.Trainer`` backend.

This backend hands the whole training program to somebody else, so the claim worth pinning is
about the SEAM rather than the loop: the batch mapping `Trainer` is handed, the knob
translation into `TrainingArguments`, and that everything shared (labels, class names, the
prediction contract, the artifacts) still comes from recordstream and marainer rather than
being reimplemented here.

Skipped when transformers is not installed — it is a selectable backend
(``sonair[huggingface]``), not a core dependency.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
import pytest

pytest.importorskip("transformers", reason="the huggingface backend needs sonair[huggingface]")

import torch  # noqa: E402 - after the skip
from confluid import Fluid, lazy_param_names  # noqa: E402
from recordstream import Stream  # noqa: E402
from transformers.modeling_outputs import ImageClassifierOutput  # noqa: E402

from sonair.huggingface_classifier import HuggingFaceClassifier, _accuracy, _accuracy_multi, _as_metric  # noqa: E402


class _CaptureSink:
    def __init__(self) -> None:
        self.rows: List[Tuple[Any, Any]] = []
        self.closed = False

    def write(self, prediction: Any, metadata: Any) -> None:
        self.rows.append((prediction, metadata))

    def close(self) -> None:
        self.closed = True


class _TinyHF(torch.nn.Module):
    """A stand-in with the transformers surface this backend actually touches.

    Hand-rolled rather than a real Hub checkpoint so the suite needs no network: what is being
    tested is sonair's seam (`pixel_values`/`labels` in, `.logits`/`.loss` out), not that
    transformers can classify.
    """

    def __init__(self, num_classes: int = 3) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.fc = torch.nn.Linear(3 * 8 * 8, num_classes)

        class _Config:
            problem_type = None

        self.config = _Config()

    def forward(self, pixel_values: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Any:
        # The real `ImageClassifierOutput`, not a look-alike: Trainer SUBSCRIPTS the output
        # (`outputs["loss"]`), which a plain object does not support — a hand-rolled stand-in
        # passed the unit tests and died the moment Trainer touched it.
        logits = self.fc(pixel_values.flatten(1))
        if labels is None:
            return ImageClassifierOutput(logits=logits)
        if self.config.problem_type == "multi_label_classification":
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
        else:
            loss = torch.nn.functional.cross_entropy(logits, labels)
        return ImageClassifierOutput(loss=cast(Any, loss), logits=logits)


#: What a test wires when it never touches the model slot. `model` is
#: `Mandatory[Lazy[HuggingFaceNetwork]]` — it must be wired, and it must be the wrapper or a live
#: torch model — so these tests wire the cheapest legal thing rather than `None`, which only
#: type-checked while the slot was `Lazy[Any]`.
_NO_MODEL = torch.nn.Identity()


def _records(n: int, num_classes: int = 3) -> List[dict]:
    """Channels-FIRST records — what a torch preprocessing chain (``ToTensor``) produces."""
    from recordstream.items import Image

    rng = np.random.default_rng(0)
    return [
        {
            "image": Image(torch.as_tensor(rng.random((3, 8, 8)).astype("float32"))),
            "class": i % num_classes,
            "row_id": i,
        }
        for i in range(n)
    ]


def _multilabel_records(n: int, num_classes: int = 3) -> List[dict]:
    from recordstream.items import Image, MultiLabel

    return [
        {
            "image": Image(torch.zeros(3, 8, 8)),
            "class": MultiLabel([i % num_classes, (i + 1) % num_classes]),
        }
        for i in range(n)
    ]


def _stream(records: List[dict]) -> Stream:
    return Stream(source=records)


# ---------------------------------------------------------------------------
# Discovery + construction
# ---------------------------------------------------------------------------


def test_discovery_tags() -> None:
    """A dropped tag silently empties navigaitor's picker rather than failing anything."""
    assert HuggingFaceClassifier.__confluid_task__ == "classification"
    assert HuggingFaceClassifier.__confluid_role__ == "trainer"
    assert HuggingFaceClassifier.__confluid_framework__ == "huggingface"


def test_zero_arg_construction_does_no_work() -> None:
    """The workspace lazy-init mandate: constructing must not build, download or walk."""
    clf = HuggingFaceClassifier(model=_NO_MODEL)

    assert clf.num_classes is None and clf.metrics is None
    assert clf._computed_metrics == {}


def test_the_deferred_slots_are_declared_lazy() -> None:
    """`model`/`train_set`/... are flowed at run time, so they must declare it."""
    assert {"model", "train_set", "test_set"} <= lazy_param_names(HuggingFaceClassifier)


# ---------------------------------------------------------------------------
# The batch mapping — all this backend says about batches
# ---------------------------------------------------------------------------


def test_the_collate_produces_the_two_keys_transformers_takes() -> None:
    """`pixel_values` is what an image model's forward takes; `labels` is what makes it
    return a loss. Those key names are transformers' contract, not ours."""
    clf = HuggingFaceClassifier(model=_NO_MODEL, num_classes=3)
    batch = clf._to_hf_batch(_records(4))

    assert set(batch) == {"pixel_values", "labels"}
    assert batch["pixel_values"].shape == (4, 3, 8, 8)
    assert batch["labels"].shape == (4,)


def test_int64_labels_are_named_explicitly() -> None:
    """Same reason as every other backend: a dataset yielding another integer width is legal,
    and the loss is the thing with the requirement."""
    from recordstream.items import Image

    records = [{"image": Image(torch.zeros(3, 8, 8)), "class": np.int32(2)}]
    assert HuggingFaceClassifier(model=_NO_MODEL)._to_hf_batch(records)["labels"].dtype == torch.int64


def test_a_multilabel_target_becomes_a_FLOAT_multi_hot() -> None:
    """`problem_type="multi_label_classification"` routes to BCEWithLogitsLoss, which compares
    against float targets — an int64 multi-hot would raise inside the model."""
    clf = HuggingFaceClassifier(model=_NO_MODEL, num_classes=3)
    clf.multilabel = True

    labels = clf._to_hf_batch(_multilabel_records(2))["labels"]

    assert labels.dtype == torch.float32
    assert labels.tolist() == [[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]]


def test_a_multilabel_batch_without_num_classes_says_so() -> None:
    clf = HuggingFaceClassifier(model=_NO_MODEL)
    clf.multilabel = True

    with pytest.raises(RuntimeError, match="num_classes"):
        clf._to_hf_batch(_multilabel_records(2))


def test_the_batch_shape_matches_the_lightning_backend() -> None:
    """The reuse claim, made checkable: the image tensor is the reference backend's, verbatim."""
    from recordstream import batch_tensor, collate_records

    records = _records(4)
    batch = HuggingFaceClassifier(model=_NO_MODEL, num_classes=3)._to_hf_batch(records)

    assert torch.equal(batch["pixel_values"], batch_tensor(collate_records(records), "image"))


# ---------------------------------------------------------------------------
# Knob translation + the Trainer wiring
# ---------------------------------------------------------------------------


def test_the_shared_knobs_become_training_arguments(tmp_path: Path) -> None:
    """One flat config drives this backend and every other — the translation is here."""
    clf = HuggingFaceClassifier(
        model=_NO_MODEL, max_epochs=7, batch_size=4, num_workers=2, learning_rate=1e-3, output_dir=str(tmp_path)
    )
    args = clf._training_arguments()

    assert args.num_train_epochs == 7
    assert args.per_device_train_batch_size == 4 and args.per_device_eval_batch_size == 4
    assert args.dataloader_num_workers == 2
    assert args.learning_rate == pytest.approx(1e-3)


def test_unused_column_removal_is_off() -> None:
    """A recordstream source is not a `datasets.Dataset`, so Trainer must not try to drop
    "unused" columns by inspecting the model signature — the collate already produces exactly
    the two keys the model takes."""
    assert HuggingFaceClassifier(model=_NO_MODEL)._training_arguments().remove_unused_columns is False


def test_reporting_is_off_by_default() -> None:
    """Trainer auto-enables every tracker it can import. On a workspace with wandb and mlflow
    installed, an innocent run would silently start reporting to both."""
    args = HuggingFaceClassifier(model=_NO_MODEL)._training_arguments()

    assert args.report_to in ([], ["none"], "none"), args.report_to


# ---------------------------------------------------------------------------
# Task-shaped defaults
# ---------------------------------------------------------------------------


def test_the_metric_default_is_resolved_from_the_data() -> None:
    """A live default in the signature could not be told apart from a user's explicit choice."""
    clf = HuggingFaceClassifier(model=_TinyHF(), train_set=_stream(_records(6)), num_classes=3)
    assert clf.metrics is None

    clf.initialize()
    assert clf.metrics == ["accuracy"]


def test_a_multilabel_run_picks_the_element_wise_metric_and_problem_type() -> None:
    """Both follow from the ONE `multilabel` flag the peek sets."""
    clf = HuggingFaceClassifier(model=_TinyHF(), train_set=_stream(_multilabel_records(6)), num_classes=3)
    clf.initialize()

    assert clf.metrics == ["accuracy_multi"]
    assert clf._network().config.problem_type == "multi_label_classification"


def test_an_explicitly_wired_metric_is_never_replaced() -> None:
    clf = HuggingFaceClassifier(model=_TinyHF(), train_set=_stream(_records(6)), num_classes=3, metrics=["accuracy"])
    clf.multilabel = True
    clf._resolve_task_defaults()

    assert clf.metrics == ["accuracy"]


def test_metrics_resolve_by_name_and_reject_an_unknown_one() -> None:
    """Plain functions over numpy — no class for a `!class:` tag to name, the fastai situation."""
    assert _as_metric("accuracy") is _accuracy
    assert _as_metric("accuracy_multi") is _accuracy_multi
    with pytest.raises(ValueError, match="Unknown metric"):
        _as_metric("nope")


def test_a_callable_metric_is_taken_as_is() -> None:
    def custom(logits: np.ndarray, labels: np.ndarray) -> float:
        return 0.5

    assert _as_metric(custom) is custom


def test_accuracy_thresholds_multilabel_in_LOGIT_space() -> None:
    """0.0, not 0.5 — the head emits raw logits, exactly as the Keras thresholded metrics take
    `thresholds: 0.0`."""
    logits = np.array([[-1.0, 2.0], [3.0, -4.0]])
    labels = np.array([[0.0, 1.0], [1.0, 0.0]])

    assert _accuracy_multi(logits, labels) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# The shared surfaces — reused, not reimplemented
# ---------------------------------------------------------------------------


def test_num_classes_is_derived_from_the_data() -> None:
    clf = HuggingFaceClassifier(model=_TinyHF(), train_set=_stream(_records(9, num_classes=3)))
    clf.initialize()

    assert clf.num_classes == 3


def test_string_targets_are_encoded_through_a_fitted_label_map() -> None:
    """recordstream's LabelMap, not a sonair-side encoder."""
    from recordstream.items import Image

    records = [
        {"image": Image(torch.zeros(3, 8, 8)), "class": name} for name in ["cat", "dog", "cat", "bird", "dog", "bird"]
    ]
    clf = HuggingFaceClassifier(model=_TinyHF(), train_set=_stream(records))
    clf.initialize()

    assert clf.num_classes == 3
    assert clf.class_names == ["bird", "cat", "dog"]


def test_the_eval_path_reads_num_classes_off_the_MODEL() -> None:
    """The trained head is the single source of truth — an eval subset would mis-size metrics."""
    clf = HuggingFaceClassifier(model=_TinyHF(num_classes=7), task="test", test_set=_stream(_records(4)))
    clf.initialize()

    assert clf.num_classes == 7


# ---------------------------------------------------------------------------
# Predictions — the shared contract
# ---------------------------------------------------------------------------


def test_predict_writes_one_entry_per_record_with_its_metadata() -> None:
    sink = _CaptureSink()
    clf = HuggingFaceClassifier(model=_TinyHF(), test_set=_stream(_records(7)), task="predict", batch_size=3)
    clf.predictions_sink = sink  # type: ignore[assignment]
    clf.run()

    assert len(sink.rows) == 7, "a short final batch must not be dropped"
    prediction, metadata = sink.rows[0]
    assert set(prediction) == {"logits", "probs", "class_idx"}, "the contract, not raw logits"
    assert metadata == {"row_id": 0}
    assert pytest.approx(float(prediction["probs"].sum()), abs=1e-5) == 1.0
    assert sink.closed


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------


def test_outputs_raise_before_the_run_and_answer_after() -> None:
    clf = HuggingFaceClassifier(model=_deferred(), train_set=_stream(_records(4)))
    with pytest.raises(RuntimeError, match="still deferred"):
        _ = clf.trained_model


def _deferred() -> Any:
    """A still-deferred model slot, as a config leaves it before `initialize()`."""
    from confluid import LazyClass

    return LazyClass(_TinyHF)


def test_computed_metrics_and_class_names_are_outputs() -> None:
    from confluid import output_specs

    names = {spec["name"] for spec in output_specs(HuggingFaceClassifier)}
    assert {"trained_model", "computed_metrics", "class_names"} <= names


def test_a_deferred_model_is_reported_as_such() -> None:
    clf = HuggingFaceClassifier(model=_deferred())
    assert isinstance(clf.model, Fluid)
    with pytest.raises(RuntimeError, match="still deferred"):
        _ = clf.trained_model


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------


def test_fit_runs_end_to_end_and_reports_metrics(tmp_path: Path) -> None:
    """The whole seam at once: Trainer drives our collate over a recordstream source."""
    clf = HuggingFaceClassifier(
        model=_TinyHF(),
        train_set=_stream(_records(12)),
        val_set=_stream(_records(6)),
        num_classes=3,
        max_epochs=1,
        batch_size=4,
        output_dir=str(tmp_path / "trainer"),
    )
    clf.metrics_sidecar_path = str(tmp_path / "metrics.json")
    clf.run()

    metrics: Dict[str, float] = clf.computed_metrics
    assert any("loss" in k for k in metrics), metrics
    assert (tmp_path / "metrics.json").exists()
    assert all(isinstance(v, float) for v in metrics.values())


def test_evaluate_scores_a_wired_model(tmp_path: Path) -> None:
    clf = HuggingFaceClassifier(
        model=_TinyHF(), test_set=_stream(_records(8)), task="test", batch_size=4, output_dir=str(tmp_path)
    )
    clf.run()

    assert clf.computed_metrics, "evaluate must report something"


@pytest.mark.parametrize("method", ["test", "predict"])
def test_eval_verbs_without_a_test_set_are_clear_errors(method: str) -> None:
    clf = HuggingFaceClassifier(model=_TinyHF(), num_classes=3)
    with pytest.raises(ValueError, match="test_set"):
        getattr(clf, method)()


def test_fit_without_a_train_set_is_a_clear_error() -> None:
    clf = HuggingFaceClassifier(model=_TinyHF(), num_classes=3)
    with pytest.raises(ValueError, match="train_set"):
        clf.fit()


def test_it_registers_no_loss_module() -> None:
    """A transformers model computes its own loss from `config.problem_type`, so there is
    nothing for a config to wire — and therefore no `huggingface_classification.py`."""
    import importlib.util

    from sonair.cli import FRAMEWORKS

    assert FRAMEWORKS["huggingface"].registrations is None
    assert importlib.util.find_spec("sonair.huggingface_classification") is None
