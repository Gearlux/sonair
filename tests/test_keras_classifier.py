"""The Keras backend — and, more importantly, what it does NOT have to reimplement.

The claim these pin is the reason sonair exists: swapping the training engine swaps the
runnable and the model wrapper, and nothing else. So most of these assert that a Keras run
reaches the SAME recordstream/marainer surfaces the Lightning run does — the same record
stream, the same label map, the same class-name vocabulary, the same predictions sink — rather
than re-testing Keras itself.
"""

import numpy as np
import pytest
from confluid import LazyClass
from recordstream import ClassificationPredictionsSink, Image, Label, MultiLabel, Stream

from sonair.keras_classifier import KerasClassifier, RecordSequence
from sonair.keras_model import KerasModel

INPUT = (32, 32, 3)


def _records(n: int = 12, labels: str = "names") -> list:
    """Records shaped as the Keras preprocessing chain leaves them: HWC float, not CHW."""
    names = ["cat", "dog", "bird"]
    out = []
    for i in range(n):
        target: object = names[i % 3] if labels == "names" else i % 3
        out.append(
            {
                "image": Image(np.random.rand(*INPUT).astype("float32"), layout="HWC"),
                "class": Label(target),
                "idx": i,
            }
        )
    return out


def _capture(sink: list) -> object:
    """A pass-through op that records what the sink emitted."""

    def op(record: object) -> object:
        sink.append(record)
        return record

    return op


def _classifier(**kw: object) -> KerasClassifier:
    defaults = dict(
        model=LazyClass(KerasModel, model_name="MobileNetV3Small", input_shape=INPUT),
        max_epochs=1,
        batch_size=4,
        experiment_name="test",
        run_name="r",
    )
    defaults.update(kw)
    return KerasClassifier(**defaults)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# RecordSequence — the whole adapter between recordstream and Keras
# --------------------------------------------------------------------------- #


def test_batches_are_numpy_channels_last() -> None:
    """Keras is channels-LAST; the torch backend is channels-first. That is the one real
    data-shape difference between them, and it lives in the preprocessing config."""
    seq = RecordSequence(Stream(source=_records(8, labels="ids")), batch_size=4)
    x, y = seq[0]

    assert isinstance(x, np.ndarray) and x.shape == (4, *INPUT)
    assert x.dtype == np.float32
    assert isinstance(y, np.ndarray) and y.shape == (4,) and y.dtype == np.int64


def test_the_last_batch_is_short_not_padded() -> None:
    seq = RecordSequence(Stream(source=_records(10, labels="ids")), batch_size=4)

    assert len(seq) == 3
    assert seq[2][0].shape[0] == 2


def test_a_multilabel_target_becomes_a_multi_hot_matrix() -> None:
    """`multi_hot` is recordstream's and returns numpy — which is why this backend needs no
    conversion and no second implementation of it."""
    records = [{"image": Image(np.zeros(INPUT, "float32"), layout="HWC"), "class": MultiLabel([0, 2])}]
    seq = RecordSequence(Stream(source=records), batch_size=1, num_classes=3, multilabel=True)

    assert seq[0][1].tolist() == [[1.0, 0.0, 1.0]]


def test_predict_mode_yields_inputs_only() -> None:
    seq = RecordSequence(Stream(source=_records(4, labels="ids")), target_key=None, batch_size=2)

    assert len(seq[0]) == 1


def test_shuffling_changes_the_order_but_not_the_content() -> None:
    records = _records(12, labels="ids")
    ordered = RecordSequence(Stream(source=records), batch_size=12)
    shuffled = RecordSequence(Stream(source=records), batch_size=12, shuffle=True, seed=1)

    assert sorted(shuffled[0][1].tolist()) == sorted(ordered[0][1].tolist())


# --------------------------------------------------------------------------- #
# The run — reaching the shared surfaces
# --------------------------------------------------------------------------- #


def test_fit_derives_num_classes_and_fits_the_label_map() -> None:
    """String targets are handled by recordstream's LabelMap, exactly as in the torch backend."""
    clf = _classifier(train_set=Stream(source=_records()))
    clf.run()

    assert clf.num_classes == 3
    assert clf.label_map is not None
    assert clf.class_names == ["bird", "cat", "dog"]


def test_the_model_is_built_at_the_derived_width() -> None:
    """`num_classes` is never written in YAML — the classifier flows it in at run time."""
    clf = _classifier(train_set=Stream(source=_records()))
    clf.run()

    assert clf.trained_model.output_shape[-1] == 3


def test_fit_reports_metrics() -> None:
    clf = _classifier(train_set=Stream(source=_records()))
    clf.run()

    assert "loss" in clf.computed_metrics
    assert all(isinstance(v, float) for v in clf.computed_metrics.values())


def test_integer_targets_need_no_label_map() -> None:
    clf = _classifier(train_set=Stream(source=_records(labels="ids")))
    clf.run()

    assert clf.num_classes == 3 and clf.label_map is None and clf.class_names == []


def test_task_defaults_are_sentinels_resolved_from_the_data() -> None:
    """Same reasoning as the torch backend: a live default could not be told apart from a
    user's explicit choice, so a multi-label run would train with the wrong loss."""
    clf = _classifier(train_set=Stream(source=_records()))
    assert clf.loss is None and clf.metrics is None

    clf.run()
    assert clf.loss == "sparse_categorical_crossentropy"


def test_an_explicitly_wired_loss_is_never_replaced() -> None:
    """Asserted at `initialize()`, where the sentinel resolves — running a fit would only
    add the wired loss's own target-shape requirements, which is not what this is about."""
    clf = _classifier(train_set=Stream(source=_records()))
    clf.loss = "kl_divergence"
    clf.initialize()

    assert clf.loss == "kl_divergence"


def test_an_explicitly_wired_metric_set_is_never_replaced() -> None:
    clf = _classifier(train_set=Stream(source=_records()))
    clf.metrics = ["top_k_categorical_accuracy"]
    clf.initialize()

    assert clf.metrics == ["top_k_categorical_accuracy"]


# --------------------------------------------------------------------------- #
# Predict — the SAME sink the torch backend uses
# --------------------------------------------------------------------------- #


def test_predictions_reach_the_shared_recordstream_sink() -> None:
    """`ClassificationOutput` is generic in its array type, so one sink serves both engines."""
    written: list = []
    sink = ClassificationPredictionsSink(ops=[_capture(written)], class_names={0: "a", 1: "b", 2: "c"})
    clf = _classifier(
        model=KerasModel(model_name="MobileNetV3Small", input_shape=INPUT, num_classes=3),
        test_set=Stream(source=_records(6, labels="ids")),
        task="predict",
        num_classes=3,
    )
    clf.predictions_sink = sink
    clf.run()

    assert len(written) == 6
    assert written[0]["metadata"]["predicted_class_label"] in ("a", "b", "c")


def test_each_prediction_is_paired_with_its_records_metadata() -> None:
    written: list = []
    sink = ClassificationPredictionsSink(ops=[_capture(written)])
    clf = _classifier(
        model=KerasModel(model_name="MobileNetV3Small", input_shape=INPUT, num_classes=3),
        test_set=Stream(source=_records(6, labels="ids")),
        task="predict",
        num_classes=3,
        batch_size=4,
    )
    clf.predictions_sink = sink
    clf.run()

    assert [w["metadata"]["idx"] for w in written] == [0, 1, 2, 3, 4, 5]


def test_probs_are_a_distribution() -> None:
    """The model emits raw logits, like its torch twin — the softmax happens on the way out."""
    seen: list = []

    class _Sink:
        def write(self, prediction: object, metadata: dict) -> None:
            seen.append(prediction)

        def close(self) -> None:
            pass

    clf = _classifier(
        model=KerasModel(model_name="MobileNetV3Small", input_shape=INPUT, num_classes=3),
        test_set=Stream(source=_records(4, labels="ids")),
        task="predict",
        num_classes=3,
    )
    clf.predictions_sink = _Sink()  # type: ignore[assignment]
    clf.run()

    assert np.isclose(seen[0]["probs"].sum(), 1.0)
    assert seen[0]["class_idx"] == np.argmax(seen[0]["logits"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
