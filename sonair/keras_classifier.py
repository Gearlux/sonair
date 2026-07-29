"""``KerasClassifier`` — the same classification run, on Keras 3 instead of Lightning.

This is the second backend, and its job is to make one claim checkable: **the config is the
program**. A Keras run and a Lightning run share the dataset, the batch/epoch knobs, the run
identity, the label handling, the class-name artifact and the metrics sidecar. What differs is
the training loop and the model wrapper — which is exactly what the `!scope:framework=` block in
the example config overrides, and nothing else.

What that costs in code is the interesting part, because almost none of it is duplicated:

* **Data** comes from recordstream unchanged — a ``Stream`` of records, the same one Lightning
  consumes. :class:`RecordSequence` adapts it to ``keras.utils.PyDataset`` in ~20 lines by
  reading batches with ``batch_values`` / ``multi_hot``, which return **numpy**. That is why
  those are not torch-typed: a torch return would have forced a second implementation here.
* **Labels** come from recordstream too — ``LabelMap`` fits and encodes, ``class_names`` reads
  the vocabulary back off the encoded stream.
* **Run artifacts** come from marainer — ``save_class_names`` / ``write_metrics_sidecar``.
* **Tracking** is marainer's ``keras_tracking_callback``.

So what is actually written here is: a ``PyDataset`` adapter, ``compile()`` + ``fit()``, and the
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

from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, cast

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
    collate_records,
    ensure_record_dataset,
    is_class_id,
    iter_key,
    multi_hot,
)
from recordstream.runnable import ProgressReporting, entrypoint, run_entrypoint

from sonair._keras import keras, keras_backend

logger = get_logger(__name__)

#: Which entry point :meth:`KerasClassifier.run` dispatches to — the same vocabulary the
#: Lightning backend uses, because the verb is a property of the TASK, not of the engine.
ClassifierTask = Literal["fit", "evaluate", "test", "predict"]

__all__ = ["ClassifierTask", "KerasClassifier", "RecordSequence"]


def _first_target(source: Any, target_key: str) -> Any:
    """The first non-``None`` target in ``source`` — one peek is enough to know the label kind."""
    for target in iter_key(source, target_key):
        if target is not None:
            return target
    return None


class RecordSequence(keras.utils.PyDataset):
    """A recordstream source as a ``keras.utils.PyDataset`` of numpy batches.

    The whole adapter between the two worlds, and it is short because recordstream hands over
    numpy already: ``batch_values`` unwraps a batched record's key past its item type, and
    ``multi_hot`` renders a multi-label column as an ``[N, C]`` matrix. Neither is torch-typed,
    so nothing is converted twice and nothing is reimplemented for this backend.

    Keras is **channels-last**: images arrive ``[N, H, W, C]``. The preprocessing config for a
    Keras run therefore ends after ``ConvertToImage`` rather than adding a CHW ``ToTensor`` — the
    one genuine data-shape difference between the backends, and it lives in the config where a
    reader can see it.

    Args:
        source: Any map-style recordstream source (a ``Stream`` is one).
        input_key: Record key holding the image.
        target_key: Record key holding the label. ``None`` yields inputs only (predict).
        batch_size: Rows per batch.
        num_classes: Needed only for a multi-label target, to size the multi-hot matrix.
        multilabel: Emit an ``[N, C]`` float multi-hot instead of ``[N]`` integer class ids.
        shuffle: Reshuffle the row order every epoch (training).
        seed: Shuffle seed, so a run is reproducible.
    """

    def __init__(
        self,
        source: Any,
        input_key: str = "image",
        target_key: Optional[str] = "class",
        batch_size: int = 32,
        num_classes: Optional[int] = None,
        multilabel: bool = False,
        shuffle: bool = False,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.source = source
        self.input_key = input_key
        self.target_key = target_key
        self.batch_size = int(batch_size)
        self.num_classes = num_classes
        self.multilabel = multilabel
        self.shuffle = shuffle
        self._order = np.arange(len(source))
        self._rng = np.random.default_rng(seed)
        if shuffle:
            self._rng.shuffle(self._order)

    def __len__(self) -> int:
        """Number of batches — Keras asks once per epoch."""
        return int(np.ceil(len(self._order) / self.batch_size))

    def __getitem__(self, index: int) -> Any:
        rows = self._order[index * self.batch_size : (index + 1) * self.batch_size]
        batch = collate_records([self.source[int(i)] for i in rows])
        x = np.asarray(batch_values(batch, self.input_key), dtype="float32")
        if self.target_key is None:
            return (x,)
        if self.multilabel:
            if not self.num_classes:
                raise RuntimeError("RecordSequence: a multi-label target needs num_classes.")
            return x, multi_hot(batch, self.target_key, self.num_classes)
        return x, np.asarray(batch_values(batch, self.target_key), dtype="int64")

    def on_epoch_end(self) -> None:
        """Reshuffle between epochs when training."""
        if self.shuffle:
            self._rng.shuffle(self._order)

    def records(self) -> Iterable[Any]:
        """The underlying records in batch order — for pairing predictions with metadata."""
        for index in range(len(self)):
            rows = self._order[index * self.batch_size : (index + 1) * self.batch_size]
            yield collate_records([self.source[int(i)] for i in rows])


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
        optimizer: Keras optimizer name or instance.
        loss: Keras loss. ``None`` picks by target shape, like the Lightning backend's sentinel.
        metrics: Keras metric names/instances. ``None`` picks a sensible default.
    """

    def __init__(
        self,
        model: Mandatory[Lazy[Any]],
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
        optimizer: Any = "adam",
        loss: Any = None,
        metrics: Optional[Sequence[Any]] = None,
    ) -> None:
        super().__init__()
        self.model: Lazy[Any] = model
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
        self.optimizer = optimizer
        # Sentinels, for the same reason the Lightning backend uses them: the right loss and
        # metrics depend on whether the targets are single- or multi-label, which is only known
        # once `initialize()` has peeked the data. A live default here could not be told apart
        # from a user's explicit choice.
        self.loss: Any = loss
        self.metrics: Optional[Sequence[Any]] = metrics
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
        """Fill the unset (``None``) task-shaped slots. Idempotent; never overwrites."""
        if self.loss is None:
            self.loss = "binary_crossentropy" if self.multilabel else "sparse_categorical_crossentropy"
        if self.metrics is None:
            self.metrics = ["binary_accuracy"] if self.multilabel else ["accuracy"]

    def _compile(self) -> None:
        """Build the model at the resolved width, then compile it."""
        self._resolve_task_defaults()
        if isinstance(self.model, Fluid) or hasattr(self.model, "solidify"):
            self.model = flow(self.model, num_classes=self.num_classes)
        network = getattr(self.model, "model", self.model)
        network.compile(optimizer=self.optimizer, loss=self.loss, metrics=list(self.metrics or []))
        self._network = network

    def _sequence(self, source: Any, shuffle: bool = False, with_targets: bool = True) -> RecordSequence:
        return RecordSequence(
            source,
            input_key=self.input_key,
            target_key=self.target_key if with_targets else None,
            batch_size=self.batch_size,
            num_classes=self.num_classes,
            multilabel=self.multilabel,
            shuffle=shuffle,
        )

    def _callbacks(self) -> List[Any]:
        """Wired Keras callbacks plus OPTIONAL marainer tracking (a few lines; removable)."""
        callbacks = [flow(cb) if isinstance(cb, Fluid) else cb for cb in self.callbacks]
        for callback in callbacks:
            # Keras's file-writing callbacks (`CSVLogger`, `ModelCheckpoint`) open their path
            # without creating its parent, so a perfectly reasonable `./keras_logs/run.csv`
            # dies with FileNotFoundError several minutes into a fit — after the data walk and
            # the model build. Creating the directory here costs nothing and moves that class
            # of failure from "mid-training" to "never".
            target = getattr(callback, "filename", None) or getattr(callback, "filepath", None)
            if isinstance(target, (str, Path)):
                Path(target).expanduser().parent.mkdir(parents=True, exist_ok=True)
        trackers = [flow(t) if isinstance(t, Fluid) else t for t in (self.trackers or [])]
        if trackers:
            from marainer.tensorflow import keras_tracking_callback

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
        history = self._network.fit(
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
        results = self._network.evaluate(self._sequence(self.test_set), return_dict=True, callbacks=self._callbacks())
        self._computed_metrics = {k: float(v) for k, v in results.items()}
        logger.info(f"Evaluation complete: {self._computed_metrics}")
        write_metrics_sidecar(self.metrics_sidecar_path, self._computed_metrics, self.experiment_name, self.run_name)

    @entrypoint("predict", role="predictor", primary=True)
    def predict(self) -> None:
        if self.test_set is None:
            raise ValueError("KerasClassifier.predict() requires a 'test_set'.")
        sink = flow(self.predictions_sink) if isinstance(self.predictions_sink, Fluid) else self.predictions_sink
        sequence = self._sequence(self.test_set, with_targets=False)
        for index, batch in enumerate(sequence.records()):
            logits = self._network.predict(sequence[index][0], verbose=0)
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
        return [flow(t) if isinstance(t, Fluid) else t for t in (self.trackers or [])]

    # ---- @outputs ------------------------------------------------------------ #
    @property
    @output
    def trained_model(self) -> Any:
        """The live trained Keras model (post-run ``@output``). Raises while still deferred."""
        if isinstance(self.model, Fluid):
            raise RuntimeError("trained_model is only available after run() — the model is still deferred.")
        return getattr(self.model, "model", self.model)

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
