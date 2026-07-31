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

from sonair._keras import keras
from sonair.keras_classifier import KerasClassifier
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
# The batch SHAPE — all this backend adds to recordstream's PyDataset adapter
# --------------------------------------------------------------------------- #
# `RecordSequence` itself is recordstream's (row order, slicing, reshuffle, collate_records) and
# is pinned there, task-free. What is sonair's is `_to_arrays`, handed to it the way a collate_fn
# is handed to a DataLoader — so these assert the SHAPE a classification batch takes, not batching.


def test_batches_are_numpy_channels_last() -> None:
    """Keras is channels-LAST; the torch backend is channels-first. That is the one real
    data-shape difference between them, and it lives in the preprocessing config."""
    clf = _classifier()
    x, y = clf._sequence(Stream(source=_records(8, labels="ids")))[0]

    assert isinstance(x, np.ndarray) and x.shape == (4, *INPUT)
    assert x.dtype == np.float32
    assert isinstance(y, np.ndarray) and y.shape == (4,) and y.dtype == np.int64


def test_int64_targets_are_named_explicitly_not_inherited_from_the_data() -> None:
    """Same reason the torch backend spells int64: a dataset yielding another integer width is
    legal, and the loss is the thing with the requirement."""
    records = [{"image": Image(np.zeros(INPUT, "float32"), layout="HWC"), "class": Label(np.int32(2))}]
    _, y = _classifier()._sequence(Stream(source=records))[0]

    assert y.dtype == np.int64


def test_a_multilabel_target_becomes_a_multi_hot_matrix() -> None:
    """`multi_hot` is recordstream's and returns numpy — which is why this backend needs no
    conversion and no second implementation of it."""
    records = [{"image": Image(np.zeros(INPUT, "float32"), layout="HWC"), "class": MultiLabel([0, 2])}]
    clf = _classifier(num_classes=3)
    clf.multilabel = True

    assert clf._sequence(Stream(source=records))[0][1].tolist() == [[1.0, 0.0, 1.0]]


def test_a_multilabel_batch_without_num_classes_says_so() -> None:
    """The multi-hot width cannot be guessed from one batch — an all-zero column would be a
    legitimate row, so a short matrix would be silently wrong instead of loudly missing."""
    records = [{"image": Image(np.zeros(INPUT, "float32"), layout="HWC"), "class": MultiLabel([0, 2])}]
    clf = _classifier()
    clf.multilabel = True
    clf.num_classes = None

    with pytest.raises(RuntimeError, match="num_classes"):
        clf._sequence(Stream(source=records))[0]


def test_predict_mode_yields_inputs_only() -> None:
    seq = _classifier()._sequence(Stream(source=_records(4, labels="ids")), with_targets=False)

    assert len(seq[0]) == 1


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


def test_a_live_model_is_left_exactly_as_it_arrived() -> None:
    """Why `_compile` flows the model UNCONDITIONALLY, with no `isinstance` around it.

    Handed a live object, `flow()` returns that same object and silently DROPS the runtime
    kwarg — so a live model keeps the width its own constructor was given, and the call costs
    nothing. The deferred path is pinned by `test_the_model_is_built_at_the_derived_width`.
    """
    live = KerasModel(model_name="MobileNetV3Small", input_shape=INPUT, num_classes=3)
    clf = _classifier(model=live, num_classes=7)
    clf._compile()

    assert clf.model is live
    assert live.num_classes == 3
    assert clf.trained_model.output_shape[-1] == 3


# --------------------------------------------------------------------------- #
# The model slot: `KerasModel` IS a `keras.Model`
# --------------------------------------------------------------------------- #
# The Keras half of the mirroring — `TimmModel` IS an `nn.Module`, so this IS a `keras.Model`.
# What it buys is that the classifier addresses `self.model` with no unwrapping step, which is
# what let `getattr(self.model, "model", self.model)` be deleted from every call site.


def test_keras_model_is_a_keras_model() -> None:
    """The structural claim the classifier's call sites rest on."""
    assert issubclass(KerasModel, keras.Model)


def test_the_model_slot_takes_a_BARE_keras_model_too() -> None:
    """The payoff: a plain `keras.Model` — which has no `.model` attribute at all — is wired in
    the same slot and trains identically, because the classifier never unwraps anything."""
    inputs = keras.Input(shape=INPUT)
    bare = keras.Model(inputs, keras.layers.Dense(3)(keras.layers.GlobalAveragePooling2D()(inputs)))

    clf = _classifier(model=bare, train_set=Stream(source=_records()))
    clf.run()

    assert clf.trained_model is bare
    assert "loss" in clf.computed_metrics


def test_a_never_flowed_model_still_builds_on_first_use() -> None:
    """Why the build lives in `build()` and not in `call()`.

    Keras refuses to let a BUILT layer add sub-layers, so creating the network on the first
    `call()` dies with "You cannot add new elements of state…". `build()` runs before that
    first call, which keeps a directly-constructed instance usable without `flow()`.
    """
    model = KerasModel(model_name="MobileNetV3Small", input_shape=INPUT, num_classes=3)
    assert model._model is None  # nothing built by the constructor (lazy-init mandate)

    logits = model(np.zeros((2, *INPUT), "float32"))

    assert tuple(logits.shape) == (2, 3)
    assert model._model is not None


def test_output_shape_is_the_one_member_that_needs_a_hop() -> None:
    """`output_shape` is Functional-only — a subclassed model raises for it, so it is the single
    member forwarded by hand. Everything else on the Keras surface is inherited."""
    model = KerasModel(model_name="MobileNetV3Small", input_shape=INPUT, num_classes=3)

    assert model.output_shape[-1] == 3
    for inherited in ("compile", "fit", "evaluate", "predict", "save", "save_weights", "export"):
        assert getattr(type(model), inherited) is getattr(keras.Model, inherited), inherited


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
    assert isinstance(clf.loss, keras.losses.SparseCategoricalCrossentropy)
    assert clf.loss.from_logits, "the head emits raw logits; a probability loss trains on garbage"


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


# --------------------------------------------------------------------------- #
# from_logits — the bug that produced a NaN loss on a real MNIST run
# --------------------------------------------------------------------------- #
# The head is a bare Dense with no activation, matching the torch backend, whose
# CrossEntropyLoss takes logits by definition. Keras's STRING losses default to
# `from_logits=False`: handed logits they normalize by the row SUM instead of a softmax,
# then CLIP rather than raise — so the loss stays finite and merely wrong, and the only
# symptom is a model that does not learn. Caught only by watching a real run diverge.


def test_the_head_emits_raw_logits_not_probabilities() -> None:
    """If this ever gains a softmax, the from_logits losses below become wrong instead."""
    model = KerasModel(model_name="MobileNetV3Small", input_shape=INPUT, num_classes=3).model
    out = model.predict(np.random.rand(2, *INPUT).astype("float32"), verbose=0)

    assert not np.allclose(out.sum(axis=-1), 1.0), "output is already normalized — head has an activation"


def test_the_default_loss_reads_its_input_as_logits() -> None:
    clf = _classifier(train_set=Stream(source=_records()))
    clf.initialize()

    assert isinstance(clf.loss, keras.losses.SparseCategoricalCrossentropy)
    assert clf.loss.from_logits is True


def test_a_probability_loss_on_logits_disagrees_with_the_correct_one() -> None:
    """The concrete cost of the string form: a different number, hence different gradients."""
    logits = np.array([[12.0, -8.0, 3.0]], dtype="float32")
    y = np.array([1], dtype="int64")

    # Reduced through `keras.ops.convert_to_numpy`, not `float(...)`, so this stays
    # engine-portable — which matters now that `sonair tensorflow` gives a reason to run the
    # suite with KERAS_BACKEND=tensorflow. The functional form returns a PER-ROW value, and a
    # shape-(1,) `tf.Tensor` refuses `float()`; a bare `np.asarray` then fails the other way, on
    # a torch tensor that lives on MPS. Keras's own converter handles both.
    def _scalar(value: object) -> float:
        return float(keras.ops.convert_to_numpy(value).mean())

    wrong = _scalar(keras.losses.get("sparse_categorical_crossentropy")(y, logits))
    right = _scalar(keras.losses.SparseCategoricalCrossentropy(from_logits=True)(y, logits))

    assert not np.isclose(wrong, right), "if these agree, the from_logits distinction has changed"


def test_the_multilabel_default_is_also_from_logits() -> None:
    records = [
        {"image": Image(np.zeros(INPUT, "float32"), layout="HWC"), "class": MultiLabel([0, 2])} for _ in range(4)
    ]
    clf = _classifier(train_set=Stream(source=records), num_classes=3)
    clf.initialize()

    assert isinstance(clf.loss, keras.losses.BinaryCrossentropy) and clf.loss.from_logits is True
