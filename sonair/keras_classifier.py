"""``KerasClassifier`` — the same classification run, on Keras 3 instead of Lightning.

This is the second backend, and its job is to make one claim checkable: **the config is the
program**. A Keras run and a Lightning run share the dataset, the batch/epoch knobs, the run
identity, the label handling, the class-name artifact and the metrics sidecar. What differs is
the training loop and the model wrapper — which is exactly what the `!scope:framework=` block in
the example config overrides, and nothing else.

What that costs in code is the interesting part, because almost none of it is duplicated:

* **Data** comes from recordstream unchanged — a ``Stream`` of records, the same one Lightning
  consumes — and so does the BATCHING: ``recordstream.keras.RecordSequence`` is the
  ``keras.utils.PyDataset`` a ``DataLoader`` would have been (row order, slicing, reshuffle,
  ``collate_records``). All this backend adds is :meth:`KerasClassifier._to_arrays`, the eight
  lines that say what a classification batch looks like — the twin of the Lightning backend's
  ``_targets``, passed in exactly as a ``collate_fn`` is passed to a ``DataLoader``. It reads
  batches with ``batch_values`` / ``multi_hot``, which return **numpy**: that is why those are
  not torch-typed — a torch return would have forced a second implementation here.
* **Labels** come from recordstream too — ``LabelMap`` fits and encodes, ``class_names`` reads
  the vocabulary back off the encoded stream.
* **Run artifacts** come from marainer — ``save_class_names`` / ``write_metrics_sidecar``.
* **Tracking** is marainer's ``keras_tracking_callback``.

So what is actually written here is: a batch-shaping function, ``compile()`` + ``fit()``, and the
task dispatch. That ratio is the demo.

Keras 3 picks its backend from ``KERAS_BACKEND`` (``torch`` / ``tensorflow`` / ``jax``); this
module never sets it, so ``sonair keras train`` runs on whichever the operator chose.

Minimal flat YAML::

    runnable: !class:sonair.KerasClassifier
      model: !lazy:sonair.KerasModel { model_name: MobileNetV3Small }
    train_set: !class:recordstream.Stream { source: ..., ops: !ref:preprocess }
    batch_size: 32
    max_epochs: 3
"""

from functools import partial
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Union, cast

import numpy as np
from confluid import Fluid, Lazy, Mandatory, configurable, flow, output
from loggair import get_logger
from marainer.artifacts import load_class_names, save_class_names, write_metrics_sidecar
from marainer.run_names import generate_run_name
from marainer.tracking.base import Tracker
from recordstream import (
    ClassificationOutput,
    LabelMap,
    PredictionsSink,
    RecordSource,
    batch_metadata,
    batch_values,
    class_names,
    ensure_record_dataset,
    is_class_id,
    iter_key,
    multi_hot,
)
from recordstream.keras import RecordSequence
from recordstream.runnable import ProgressReporting, entrypoint, run_entrypoint

from sonair._keras import keras, keras_backend

logger = get_logger(__name__)

#: Which entry point :meth:`KerasClassifier.run` dispatches to — the same vocabulary the
#: Lightning backend uses, because the verb is a property of the TASK, not of the engine.
ClassifierTask = Literal["fit", "evaluate", "test", "predict"]

# What Keras accepts in each compile slot: its own class, or the string alias it resolves to.
# Spelled out rather than left as `Any` because `Any` types nothing (workspace mandate) and
# gives `to_pydantic` nothing to build a widget from — navigaitor's form-spec and StreamStudio
# read these to know that `loss` takes a `keras.losses.Loss`, which is the other half of the
# registered set in `sonair.keras_classification` being pickable at all.
#
# It does NOT change how a wired value materializes: a nested `!class:` in a CONSTRUCTOR slot
# stays a deferred `Class` marker regardless of annotation (only the instance form
# `!class:X()` builds eagerly, the same trap the example config's `HuggingFaceSource()`
# documents), so `_compile` flows these at use — exactly as `_callbacks()` already does for
# the `logger` slot.
KerasLoss = Union[str, keras.losses.Loss]
KerasMetric = Union[str, keras.metrics.Metric]
KerasOptimizer = Union[str, keras.optimizers.Optimizer]

__all__ = ["ClassifierTask", "KerasClassifier"]


def _first_target(source: Any, target_key: str) -> Any:
    """The first non-``None`` target in ``source`` — one peek is enough to know the label kind."""
    for target in iter_key(source, target_key):
        if target is not None:
            return target
    return None


@configurable(task="classification", role="trainer", framework="keras")
class KerasClassifier(ProgressReporting):
    """Train, evaluate or predict an image classifier with Keras 3.

    Args:
        model: The classification model — wire ``!lazy:`` so ``initialize()`` can inject the
            dataset-derived ``num_classes``.
        train_set: Training source (``task="fit"``).
        test_set: Evaluation / prediction source.
        num_classes: Class count. Omitted, it is derived from the data or the label map.
        task: Which entry point ``run()`` dispatches to.
        max_epochs: Training epochs.
        batch_size: Rows per batch.
        logger: Keras callbacks (``keras.callbacks.CSVLogger`` …) — the Keras analogue of the
            Lightning backend's ``logger`` slot, so the same flat key drives both.
        experiment_name: Run identity, recorded in artifacts and tracking.
        run_name: Run identity; a random ``adjective_noun`` when omitted.
        input_key: Record key holding the image.
        target_key: Record key holding the label.
        label_map: Pin the name-to-id encoding instead of fitting it from the train targets.
        optimizer: Keras optimizer — a string alias (``"adam"``) or a ``keras.optimizers``
            instance.
        loss: Keras loss — a string alias or a ``keras.losses`` instance (the registered set
            lives in :mod:`sonair.keras_classification`; a logit-emitting head needs
            ``from_logits=True``). ``None`` picks by target shape, like the Lightning
            backend's sentinel.
        metrics: Keras metrics — string aliases or ``keras.metrics`` instances. ``None`` picks
            a sensible default.
    """

    def __init__(
        self,
        model: Mandatory[Lazy[keras.Model]],
        train_set: Optional[Lazy[RecordSource]] = None,
        test_set: Optional[Lazy[RecordSource]] = None,
        num_classes: Optional[int] = None,
        task: ClassifierTask = "fit",
        max_epochs: int = 10,
        batch_size: int = 32,
        logger: Optional[Iterable[Any]] = None,
        experiment_name: str = "sonair_classification",
        run_name: Optional[str] = None,
        input_key: str = "image",
        target_key: str = "class",
        label_map: Optional[LabelMap] = None,
        optimizer: KerasOptimizer = "adam",
        loss: Optional[KerasLoss] = None,
        metrics: Optional[Sequence[KerasMetric]] = None,
    ) -> None:
        super().__init__()
        # `keras.Model`, not `Lazy[keras.Model]`: the DEFERRED-ness is declared once, on the ctor
        # param above, which is what `lazy_param_names` / `to_pydantic` read. Restating it
        # here would only make the attribute a union no method could call through.
        # `keras.Model` and not something looser because this class calls `compile` / `fit` /
        # `export` straight through the slot. The TensorFlow backend, which drives a hand-written
        # loop instead, admits a `tf.Module` too and says so in its own `TensorFlowNetwork`.
        self.model: keras.Model = model
        self.train_set: Optional[Lazy[RecordSource]] = train_set
        self.val_set: Optional[Lazy[RecordSource]] = None
        self.test_set: Optional[Lazy[RecordSource]] = test_set
        self.task: ClassifierTask = task
        self.num_classes = int(num_classes) if num_classes is not None else None
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.callbacks: List[Any] = list(logger or [])
        self.experiment_name = experiment_name
        self.run_name = run_name if run_name is not None else generate_run_name(prefix=_run_prefix(task))
        self.input_key = str(input_key)
        self.target_key = str(target_key)
        self.label_map = label_map
        self.optimizer: KerasOptimizer = optimizer
        # Sentinels, for the same reason the Lightning backend uses them: the right loss and
        # metrics depend on whether the targets are single- or multi-label, which is only known
        # once `initialize()` has peeked the data. A live default here could not be told apart
        # from a user's explicit choice.
        self.loss: Optional[KerasLoss] = loss
        self.metrics: Optional[Sequence[KerasMetric]] = metrics
        self.multilabel: bool = False
        self.predictions_sink: Optional[Lazy[PredictionsSink]] = None
        self.metrics_sidecar_path: Optional[str] = None
        self.trackers: Optional[List[Tracker]] = None
        self._class_names: Optional[List[str]] = None
        self._history: Dict[str, Any] = {}
        self._computed_metrics: Dict[str, float] = {}

    # ---- run dispatch ------------------------------------------------------- #
    def run(self) -> None:
        self.initialize()
        # The `@entrypoint` markers below ARE the dispatch table (recordstream docs/runnable.md).
        run_entrypoint(self, self.task)

    # ---- setup -------------------------------------------------------------- #
    def initialize(self) -> None:
        """Resolve labels and ``num_classes``, then build and compile the Keras model."""
        logger.info(f"Keras backend in effect: {keras_backend()}")
        if self.task == "fit":
            self._initialize_fit()
        else:
            self._initialize_eval()
        self._compile()

    def _initialize_fit(self) -> None:
        self.train_set = ensure_record_dataset(self.train_set)
        if self.val_set is not None:
            self.val_set = ensure_record_dataset(self.val_set)
        if self.test_set is not None:
            self.test_set = ensure_record_dataset(self.test_set)

        first = _first_target(self.train_set, self.target_key)
        self.multilabel = isinstance(first, (list, tuple, set))
        probe = next(iter(first), None) if self.multilabel else first
        targets_are_names = probe is not None and not is_class_id(probe)

        if targets_are_names and self.label_map is None:
            self.label_map = LabelMap.fit(iter_key(self.train_set, self.target_key))
            logger.info(f"Fitted LabelMap with {self.label_map.num_classes} classes from train targets.")
        if targets_are_names and self.label_map is not None:
            self.train_set = self.label_map.encode(self.train_set)
            if self.val_set is not None:
                self.val_set = self.label_map.encode(self.val_set)
            if self.test_set is not None:
                self.test_set = self.label_map.encode(self.test_set)

        if self.num_classes is None:
            if self.label_map is not None:
                self.num_classes = self.label_map.num_classes
            else:
                ids = [i for t in iter_key(self.train_set, self.target_key) for i in LabelMap().to_ids(t)]
                self.num_classes = max(ids) + 1 if ids else None
        if not self.num_classes:
            raise RuntimeError("could not determine num_classes — set it, or wire a labelled dataset.")
        self._class_names = class_names(self.train_set, self.val_set, self.test_set)

    def _initialize_eval(self) -> None:
        self.model = flow(self.model)
        nc = getattr(self.model, "num_classes", None)
        if not nc:
            raise RuntimeError("could not determine num_classes from the model (it must expose `num_classes`).")
        self.num_classes = int(nc)
        if self.test_set is None:
            return
        dataset = ensure_record_dataset(self.test_set)
        first = _first_target(dataset, self.target_key)
        self.multilabel = isinstance(first, (list, tuple, set))
        probe = next(iter(first), None) if self.multilabel else first
        if probe is not None and not is_class_id(probe):
            label_map = self.label_map or load_class_names(getattr(self.model, "checkpoint_path", None))
            if label_map is None:
                raise RuntimeError("eval dataset has string targets but no label_map and no class_names.json found.")
            self.label_map = label_map
            dataset = label_map.encode(dataset)
        self.test_set = dataset
        self._class_names = class_names(dataset)

    def _resolve_task_defaults(self) -> None:
        """Fill the unset (``None``) task-shaped slots. Idempotent; never overwrites.

        ``from_logits=True`` is not optional here. The head is a bare ``Dense`` with no
        activation — raw logits, matching the torch backend, whose ``CrossEntropyLoss`` takes
        logits by definition. Keras's STRING losses default to ``from_logits=False``, i.e.
        they expect probabilities, and handed logits they normalize by the row SUM instead of
        a softmax. Keras clips the result rather than raising, so the loss stays finite and
        merely wrong — the failure shows up as a model that does not learn, which is why this
        is spelled out rather than left to a string.
        """
        if self.loss is None:
            self.loss = (
                keras.losses.BinaryCrossentropy(from_logits=True)
                if self.multilabel
                else keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            )
        if self.metrics is None:
            self.metrics = ["binary_accuracy"] if self.multilabel else ["accuracy"]

    def _compile(self) -> None:
        """Build the model at the resolved width, then compile it.

        The compile slots are flowed at USE, the way ``_callbacks()`` and ``predict()`` flow
        theirs — a nested ``!class:`` in a constructor slot arrives as a deferred ``Class``
        marker, and Keras takes an unbuilt marker for a custom callable and fails deep inside
        ``compile`` rather than saying what is wrong.

        A bare ``flow()`` is the whole guard: it returns a live object, a string alias and a
        list unchanged (``confluid.fluid.flow`` opens with that idempotency check), so an
        ``isinstance(..., Fluid)`` around any of these calls only restates it.

        The model included. It reaches here deferred — ``!lazy:`` is skipped by confluid's
        auto-flow walkers precisely so the dataset-derived width can be injected now — and a
        model that is ALREADY live comes back untouched, keeping the width its own constructor
        was given (the runtime kwarg is dropped, which is what makes the call unconditional).

        Everything after that is addressed to ``self.model`` directly, with no unwrapping step:
        :class:`~sonair.keras_model.KerasModel` IS a ``keras.Model``, so it and a bare
        ``keras.Model`` are the same thing to this call — the reason a
        ``getattr(self.model, "model", self.model)`` used to stand here and no longer does.
        """
        self._resolve_task_defaults()
        self.model = flow(self.model, num_classes=self.num_classes)
        self.model.compile(
            optimizer=flow(self.optimizer),
            loss=flow(self.loss),
            metrics=[flow(m) for m in (self.metrics or [])],
        )

    def _sequence(self, source: Any, shuffle: bool = False, with_targets: bool = True) -> RecordSequence:
        """recordstream's batching adapter, wearing THIS task's batch shape.

        The split is the same one torch draws: ``RecordSequence`` is the engine's DataLoader half
        (row order, slicing, per-epoch reshuffle, ``collate_records``) and knows nothing about
        classification; ``_to_arrays`` is handed to it the way a ``collate_fn`` is handed to a
        ``DataLoader``.
        """
        return RecordSequence(
            source,
            batch_size=self.batch_size,
            shuffle=shuffle,
            transform=partial(self._to_arrays, with_targets=with_targets),
        )

    def _to_arrays(self, batch: Any, with_targets: bool = True) -> Any:
        """One collated record batch → the numpy ``(x, y)`` Keras consumes.

        The Keras twin of the Lightning backend's ``_targets``, and short for the same reason:
        recordstream hands over numpy already — ``batch_values`` unwraps a batched key past its
        item type, ``multi_hot`` renders a multi-label column as an ``[N, C]`` matrix — so nothing
        is converted twice and nothing is reimplemented for this backend.

        Keras is **channels-last**: images arrive ``[N, H, W, C]``. The preprocessing config for a
        Keras run therefore ends after ``ConvertToImage`` rather than adding a CHW ``ToTensor`` —
        the one genuine data-shape difference between the backends, and it lives in the config
        where a reader can see it.
        """
        x = np.asarray(batch_values(batch, self.input_key), dtype="float32")
        if not with_targets:
            return (x,)
        if self.multilabel:
            if not self.num_classes:
                raise RuntimeError("KerasClassifier: a multi-label target needs num_classes.")
            return x, multi_hot(batch, self.target_key, self.num_classes)
        return x, np.asarray(batch_values(batch, self.target_key), dtype="int64")

    def _callbacks(self) -> List[Any]:
        """Wired Keras callbacks plus OPTIONAL marainer tracking (a few lines; removable)."""
        callbacks = [flow(cb) for cb in self.callbacks]
        for callback in callbacks:
            # Keras's file-writing callbacks (`CSVLogger`, `ModelCheckpoint`) open their path
            # without creating its parent, so a perfectly reasonable `./keras_logs/run.csv`
            # dies with FileNotFoundError several minutes into a fit — after the data walk and
            # the model build. Creating the directory here costs nothing and moves that class
            # of failure from "mid-training" to "never".
            target = getattr(callback, "filename", None) or getattr(callback, "filepath", None)
            if isinstance(target, (str, Path)):
                Path(target).expanduser().parent.mkdir(parents=True, exist_ok=True)
        trackers = [flow(t) for t in (self.trackers or [])]
        if trackers:
            from marainer.keras import keras_tracking_callback

            callbacks.append(keras_tracking_callback(trackers, params=self._tracked_params()))
        return callbacks

    def _tracked_params(self) -> Dict[str, Any]:
        return {
            "experiment_name": self.experiment_name,
            "run_name": self.run_name,
            "max_epochs": self.max_epochs,
            "batch_size": self.batch_size,
            "num_classes": self.num_classes,
        }

    # ---- the four entry points ---------------------------------------------- #
    @entrypoint("fit", role="trainer", primary=True)
    def fit(self) -> None:
        logger.info(f"sonair keras training: {self.experiment_name} ({self.run_name})")
        if self.train_set is None:
            raise ValueError("KerasClassifier.fit() requires a 'train_set'.")
        train_seq = self._sequence(self.train_set, shuffle=True)
        val_seq = self._sequence(self.val_set) if self.val_set is not None else None
        history = self.model.fit(
            train_seq, validation_data=val_seq, epochs=self.max_epochs, callbacks=self._callbacks()
        )
        self._history = {k: [float(x) for x in v] for k, v in history.history.items()}
        self._computed_metrics = {k: v[-1] for k, v in self._history.items()}
        logger.info("Training complete.")
        save_class_names(self._class_names, self._artifact_dir(), self._live_trackers())
        write_metrics_sidecar(self.metrics_sidecar_path, self._computed_metrics, self.experiment_name, self.run_name)

    @entrypoint("evaluate", role="evaluator")
    def evaluate(self) -> None:
        self.test()

    @entrypoint("test", role="evaluator", primary=True)
    def test(self) -> None:
        if self.test_set is None:
            raise ValueError("KerasClassifier.test() requires a 'test_set'.")
        results = self.model.evaluate(self._sequence(self.test_set), return_dict=True, callbacks=self._callbacks())
        self._computed_metrics = {k: float(v) for k, v in results.items()}
        logger.info(f"Evaluation complete: {self._computed_metrics}")
        write_metrics_sidecar(self.metrics_sidecar_path, self._computed_metrics, self.experiment_name, self.run_name)

    @entrypoint("predict", role="predictor", primary=True)
    def predict(self) -> None:
        if self.test_set is None:
            raise ValueError("KerasClassifier.predict() requires a 'test_set'.")
        sink = flow(self.predictions_sink)
        sequence = self._sequence(self.test_set, with_targets=False)
        for index, batch in enumerate(sequence.batches()):
            logits = self.model.predict(sequence[index][0], verbose=0)
            if sink is not None:
                self._write_predictions(np.asarray(logits), batch, cast(PredictionsSink, sink))
        if sink is not None and hasattr(sink, "close"):
            sink.close()
        logger.info("Prediction complete.")

    # ---- predictions -------------------------------------------------------- #
    def _write_predictions(self, logits: np.ndarray, batch: Any, sink: PredictionsSink) -> None:
        """One :class:`~recordstream.ClassificationOutput` per record, paired with its metadata.

        The contract is recordstream's and framework-neutral — ``ClassificationOutput`` is
        generic in its array type, so the SAME sink that consumes the torch backend's output
        consumes this numpy one with no adapter.
        """
        probs = _softmax(logits)
        per_record_meta = batch_metadata(batch, exclude=(self.input_key, self.target_key))
        for i in range(len(logits)):
            out: ClassificationOutput[np.ndarray] = {
                "logits": logits[i],
                "probs": probs[i],
                "class_idx": np.asarray(np.argmax(logits[i])),
            }
            meta = per_record_meta[i] if per_record_meta and i < len(per_record_meta) else {}
            sink.write(out, meta)

    # ---- artifacts + tracking ----------------------------------------------- #
    def _artifact_dir(self) -> Optional[str]:
        """Where run artifacts land. Keras has no checkpoint-callback convention to read, so
        this follows the sidecar's directory when one is configured."""
        if self.metrics_sidecar_path:
            return str(Path(self.metrics_sidecar_path).expanduser().parent)
        return None

    def _live_trackers(self) -> List[Tracker]:
        return [flow(t) for t in (self.trackers or [])]

    # ---- @outputs ------------------------------------------------------------ #
    @property
    @output
    def trained_model(self) -> Any:
        """The live trained model (post-run ``@output``). Raises while still deferred.

        Whatever was wired, unwrapped by nobody — which is also what the torch backends return.
        A :class:`~sonair.keras_model.KerasModel` IS a ``keras.Model`` AND carries
        ``num_classes``, which an evaluation run reads straight off its model slot, so this
        output docks into a downstream classifier and still answers ``predict`` /
        ``output_shape`` for a consumer that wants the network.
        """
        if isinstance(self.model, Fluid):
            raise RuntimeError("trained_model is only available after run() — the model is still deferred.")
        return self.model

    @property
    @output
    def computed_metrics(self) -> Dict[str, float]:
        """Final scalar metrics from the last fit/evaluate."""
        return dict(self._computed_metrics)

    @property
    @output
    def class_names(self) -> List[str]:
        """Resolved class names for the run (post-run ``@output``), or ``[]``."""
        if self.label_map is not None:
            return list(self.label_map.class_names)
        return list(self._class_names or [])


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Row-wise softmax. The model emits raw logits (no head activation), like the torch twin."""
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return cast(np.ndarray, exp / exp.sum(axis=-1, keepdims=True))


def _run_prefix(task: str) -> str:
    return {"fit": "train", "predict": "predict"}.get(task, "evaluate")
