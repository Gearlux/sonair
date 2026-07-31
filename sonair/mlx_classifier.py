"""``MlxClassifier`` — the same classification run, on Apple-silicon unified memory.

The seventh and last backend. Like the JAX and TensorFlow ones it owns its loop, and the three
of them together are the interesting comparison: the same epoch, written against three
different execution models.

What MLX contributes to that comparison is **unified memory and laziness**:

* There is no device transfer. On Apple silicon the CPU and GPU share physical memory, so an
  array is simply usable from either — no ``.to(device)``, no ``pin_memory``, no host/device
  copy in the batch path. It is the one place where sonair's batch shaping does LESS than the
  other backends rather than the same amount.
* **Arrays are lazy.** ``mx.eval`` is what forces computation, and forgetting it does not raise
  — it builds an ever-growing graph until memory runs out. So the loop evaluates once per step,
  deliberately and visibly. Both facts are the reason this backend exists at all rather than
  being "torch with different imports".

Everything else is shared: the record stream, ``LabelMap`` fitting and encoding, ``class_names``,
the predictions sink and contract, ``save_class_names`` / ``write_metrics_sidecar``, and the
``@entrypoint`` dispatch. What is written here is :meth:`MlxClassifier._to_arrays` (the batch
shape — the Keras backend's method, verbatim, for the same numpy reason), the step function and
the epoch loop.

Losses and metrics are **names**, not ``!class:`` tags: ``mlx.nn.losses.*`` are plain functions,
so there is no class for a tag to resolve or for confluid's registry to tag. Same as the fastai
and JAX backends, and therefore no ``mlx_classification.py`` (``docs/architecture.md`` §5).

**Apple silicon only** — the ``mlx`` extra carries the platform marker, and CI on Linux can
never run this. Its tests skip there.

Minimal flat YAML::

    runnable: !class:sonair.MlxClassifier
      model: !lazy:sonair.MlxModel
    train_set: !class:recordstream.Stream { source: ..., ops: !ref:preprocess }
    batch_size: 32
    max_epochs: 3
"""

from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Sequence, Union, cast

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
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

logger = get_logger(__name__)

#: Which entry point :meth:`MlxClassifier.run` dispatches to — the same vocabulary every other
#: backend uses, because the verb is a property of the TASK, not of the engine.
ClassifierTask = Literal["fit", "evaluate", "test", "predict"]

#: What the ``loss`` / ``metrics`` / ``optimizer`` slots accept: a NAME resolved against
#: ``mlx.nn.losses`` / ``mlx.optimizers``, or the callable itself. Spelled out rather than left
#: as ``Any`` so ``to_pydantic`` has a real type to build from.
MlxLoss = Union[str, Callable[..., Any]]
MlxMetric = Union[str, Callable[..., Any]]
MlxOptimizer = Union[str, Callable[..., Any]]

__all__ = ["ClassifierTask", "MlxClassifier"]


def _first_target(source: Any, target_key: str) -> Any:
    """The first non-``None`` target in ``source`` — one peek is enough to know the label kind."""
    for target in iter_key(source, target_key):
        if target is not None:
            return target
    return None


def _run_prefix(task: str) -> str:
    return {"fit": "train", "predict": "predict"}.get(task, "evaluate")


def _accuracy(logits: Any, labels: Any) -> Any:
    """Top-1 accuracy for single-label targets (int32 ids)."""
    return mx.mean(mx.argmax(logits, axis=-1) == labels)


def _accuracy_multi(logits: Any, labels: Any) -> Any:
    """Element-wise accuracy for a multi-label target, thresholded in LOGIT space.

    ``0.0`` rather than ``0.5`` because the head emits raw logits — the same choice every other
    backend's thresholded metric makes.
    """
    return mx.mean((logits > 0.0).astype(labels.dtype) == labels)


#: Metric names this backend resolves. Plain functions, so nothing for confluid to tag.
_METRICS: Dict[str, Callable[..., Any]] = {"accuracy": _accuracy, "accuracy_multi": _accuracy_multi}


@configurable(task="classification", role="trainer", framework="mlx")
class MlxClassifier(ProgressReporting):
    """Train, evaluate or predict an image classifier with MLX on Apple silicon.

    Args:
        model: The classification model — wire ``!lazy:sonair.MlxModel`` so ``initialize()``
            can inject the dataset-derived ``num_classes``.
        train_set: Training source (``task="fit"``).
        val_set: Validation source, scored at the end of each epoch when wired.
        test_set: Evaluation / prediction source.
        num_classes: Class count. Omitted, it is derived from the data or the label map.
        task: Which entry point ``run()`` dispatches to.
        max_epochs: Training epochs.
        batch_size: Rows per batch.
        logger: Callables invoked as ``callback(epoch, logs)`` after each epoch — MLX ships no
            callback protocol, so this is the plainest thing that works, as in the JAX backend.
        experiment_name: Run identity, recorded in artifacts and tracking.
        run_name: Run identity; a random ``adjective_noun`` when omitted.
        input_key: Record key holding the image.
        target_key: Record key holding the label.
        label_map: Pin the name-to-id encoding instead of fitting it from the train targets.
        learning_rate: Peak LR for the optimizer.
        optimizer: An ``mlx.optimizers`` name (``"Adam"``, ``"SGD"``, ``"AdamW"``) or a
            ready-made optimizer instance.
        loss: An ``mlx.nn.losses`` name or a callable over ``(logits, labels)``. ``None`` picks
            by target shape, like every other backend's sentinel.
        metrics: Metric names (``"accuracy"``, ``"accuracy_multi"``) or callables. ``None``
            picks by target shape.
    """

    def __init__(
        self,
        model: Mandatory[Lazy[Any]],
        train_set: Optional[Lazy[RecordSource]] = None,
        val_set: Optional[Lazy[RecordSource]] = None,
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
        learning_rate: float = 1e-3,
        optimizer: MlxOptimizer = "Adam",
        loss: Optional[MlxLoss] = None,
        metrics: Optional[Sequence[MlxMetric]] = None,
    ) -> None:
        super().__init__()
        self.model: Any = model
        self.train_set: Optional[Lazy[RecordSource]] = train_set
        self.val_set: Optional[Lazy[RecordSource]] = val_set
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
        self.learning_rate = learning_rate
        self.optimizer: MlxOptimizer = optimizer
        # Sentinels, for the same reason every other backend uses them: the right loss and
        # metric depend on whether the targets are single- or multi-label, which is only known
        # once `initialize()` has peeked the data.
        self.loss: Optional[MlxLoss] = loss
        self.metrics: Optional[Sequence[MlxMetric]] = metrics
        self.multilabel: bool = False
        self.predictions_sink: Optional[Lazy[PredictionsSink]] = None
        self.metrics_sidecar_path: Optional[str] = None
        self.trackers: Optional[List[Tracker]] = None
        self._class_names: Optional[List[str]] = None
        self._computed_metrics: Dict[str, float] = {}

    # ---- run dispatch ------------------------------------------------------- #
    def run(self) -> None:
        self.initialize()
        # The `@entrypoint` markers below ARE the dispatch table (recordstream docs/runnable.md).
        run_entrypoint(self, self.task)

    # ---- setup -------------------------------------------------------------- #
    def initialize(self) -> None:
        """Resolve labels and ``num_classes``, then build the model at that width."""
        logger.info(f"MLX device: {mx.default_device()} (unified memory — no host/device copies)")
        if self.task == "fit":
            self._initialize_fit()
        else:
            self._initialize_eval()
        self._resolve_task_defaults()
        self.model = flow(self.model, num_classes=self.num_classes)

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
        count = getattr(self.model, "num_classes", None)
        if not count:
            raise RuntimeError("could not determine num_classes from the model (it must expose `num_classes`).")
        self.num_classes = int(count)
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

        Both defaults take LOGITS, which is what the head emits: ``cross_entropy`` in MLX
        accepts logits directly, and ``binary_cross_entropy`` takes a ``with_logits`` flag that
        :func:`_as_loss` binds for the multi-label case.
        """
        if self.loss is None:
            self.loss = "binary_cross_entropy" if self.multilabel else "cross_entropy"
        if self.metrics is None:
            self.metrics = ["accuracy_multi"] if self.multilabel else ["accuracy"]

    # ---- data --------------------------------------------------------------- #
    def _sequence(self, source: Any, shuffle: bool = False, with_targets: bool = True) -> RecordSequence:
        """recordstream's batching adapter, wearing THIS task's batch shape.

        The same engine-neutral adapter the Keras and JAX backends use — it knows nothing about
        classification or about MLX. ``_to_arrays`` is handed to it exactly as a ``collate_fn``
        is handed to a ``DataLoader``.
        """
        return RecordSequence(
            source,
            batch_size=self.batch_size,
            shuffle=shuffle,
            transform=partial(self._to_arrays, with_targets=with_targets),
        )

    def _to_arrays(self, batch: Any, with_targets: bool = True) -> Any:
        """One collated record batch → the numpy ``(x, y)`` this loop consumes.

        The Keras backend's method verbatim, and for the same reason: recordstream hands over
        NUMPY (``batch_values`` / ``multi_hot`` are not torch-typed), and ``mx.array`` takes
        numpy directly. On unified memory there is no transfer to make, so this is the one
        backend where the batch path is genuinely shorter rather than merely different.

        Channels-LAST — MLX's native layout, so the preprocessing chain is the Keras one.
        """
        x = np.asarray(batch_values(batch, self.input_key), dtype="float32")
        if not with_targets:
            return (x,)
        if self.multilabel:
            if not self.num_classes:
                raise RuntimeError("MlxClassifier: a multi-label target needs num_classes.")
            return x, multi_hot(batch, self.target_key, self.num_classes)
        # int32, not int64: MLX indexes with int32 and `cross_entropy` expects it.
        return x, np.asarray(batch_values(batch, self.target_key), dtype="int32")

    # ---- the loop ----------------------------------------------------------- #
    def _loss_and_grad(self) -> Any:
        """The step: a loss over ``(model, x, y)``, differentiated by ``nn.value_and_grad``."""
        loss_fn = _as_loss(self.loss, multilabel=self.multilabel)

        def compute(model: Any, x: Any, y: Any) -> Any:
            return mx.mean(loss_fn(model(x), y))

        return nn.value_and_grad(self.model.model, compute)

    def _metric_values(self, logits: Any, labels: Any) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for metric in self.metrics or []:
            fn = _as_metric(metric)
            out[getattr(fn, "__name__", str(metric)).lstrip("_")] = float(fn(logits, labels))
        return out

    def _evaluate(self, sequence: RecordSequence) -> Dict[str, float]:
        """One pass in EVAL mode, returning ``{"loss": …, **metrics}``."""
        loss_fn = _as_loss(self.loss, multilabel=self.multilabel)
        total, batches = 0.0, 0
        sums: Dict[str, float] = {}
        for index in range(len(sequence)):
            x, y = sequence[index]
            xa, ya = mx.array(x), mx.array(y)
            logits = self.model(xa, training=False)
            loss = mx.mean(loss_fn(logits, ya))
            mx.eval(loss, logits)  # lazy arrays: nothing has been computed until here
            total += float(loss)
            batches += 1
            for key, value in self._metric_values(logits, ya).items():
                sums[key] = sums.get(key, 0.0) + value
        if not batches:
            return {"loss": float("nan")}
        return {"loss": total / batches, **{k: v / batches for k, v in sums.items()}}

    # ---- the four entry points ---------------------------------------------- #
    @entrypoint("fit", role="trainer", primary=True)
    def fit(self) -> None:
        logger.info(f"sonair mlx training: {self.experiment_name} ({self.run_name})")
        if self.train_set is None:
            raise ValueError("MlxClassifier.fit() requires a 'train_set'.")
        network = self.model.model
        optimizer = _as_optimizer(self.optimizer, self.learning_rate)
        step = self._loss_and_grad()
        train_seq = self._sequence(self.train_set, shuffle=True)
        val_seq = self._sequence(self.val_set) if self.val_set is not None else None

        logs: Dict[str, float] = {}
        for epoch in range(self.max_epochs):
            network.train(True)
            total, batches = 0.0, 0
            for index in range(len(train_seq)):
                x, y = train_seq[index]
                loss, grads = step(network, mx.array(x), mx.array(y))
                optimizer.update(network, grads)
                # MLX is LAZY: without this the graph grows every step until memory runs out,
                # and nothing warns — the run just gets slower and then dies. One eval per step,
                # over the parameters AND the optimizer state, is what forces the work.
                mx.eval(network.parameters(), optimizer.state, loss)
                total += float(loss)
                batches += 1
            logs = {"loss": total / batches if batches else float("nan")}
            if val_seq is not None:
                logs.update({f"val_{k}": v for k, v in self._evaluate(val_seq).items()})
            # A `fit()` would have done this; a hand-written loop must. Forgetting it is
            # INVISIBLE — training silently runs on one fixed row order forever.
            train_seq.on_epoch_end()
            for callback in self.callbacks:
                flow(callback)(epoch, logs)
            logger.info(f"epoch {epoch + 1}/{self.max_epochs}: " + ", ".join(f"{k}={v:.4f}" for k, v in logs.items()))

        self._computed_metrics = logs
        logger.info("Training complete.")
        save_class_names(self._class_names, self._artifact_dir(), self._live_trackers())
        write_metrics_sidecar(self.metrics_sidecar_path, self._computed_metrics, self.experiment_name, self.run_name)

    @entrypoint("evaluate", role="evaluator")
    def evaluate(self) -> None:
        self.test()

    @entrypoint("test", role="evaluator", primary=True)
    def test(self) -> None:
        if self.test_set is None:
            raise ValueError("MlxClassifier.test() requires a 'test_set'.")
        self._computed_metrics = self._evaluate(self._sequence(self.test_set))
        logger.info(f"Evaluation complete: {self._computed_metrics}")
        write_metrics_sidecar(self.metrics_sidecar_path, self._computed_metrics, self.experiment_name, self.run_name)

    @entrypoint("predict", role="predictor", primary=True)
    def predict(self) -> None:
        """Stream per-record predictions to the sink.

        Walks ``sequence.batches()`` — the COLLATED RECORD batches — rather than the arrays,
        because the sink needs the metadata that lives beside the image and ``_to_arrays`` has
        already dropped it. One walk, so output and metadata cannot desync.
        """
        if self.test_set is None:
            raise ValueError("MlxClassifier.predict() requires a 'test_set'.")
        sink = flow(self.predictions_sink)
        sequence = self._sequence(self.test_set, with_targets=False)
        for index, batch in enumerate(sequence.batches()):
            logits = self.model(mx.array(sequence[index][0]), training=False)
            mx.eval(logits)
            if sink is not None:
                self._write_predictions(np.asarray(logits), batch, cast(PredictionsSink, sink))
        if sink is not None and hasattr(sink, "close"):
            sink.close()
        logger.info("Prediction complete.")

    # ---- predictions -------------------------------------------------------- #
    def _write_predictions(self, logits: np.ndarray, batch: Any, sink: PredictionsSink) -> None:
        """One :class:`~recordstream.ClassificationOutput` per record, paired with its metadata.

        The contract is recordstream's and framework-neutral — generic in its array type, so the
        SAME sink consumes this numpy output and every other backend's.
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
        """Where run artifacts land — beside the sidecar when one is configured."""
        if self.metrics_sidecar_path:
            return str(Path(self.metrics_sidecar_path).expanduser().parent)
        return None

    def _live_trackers(self) -> List[Tracker]:
        return [flow(t) for t in (self.trackers or [])]

    # ---- @outputs ------------------------------------------------------------ #
    @property
    @output
    def trained_model(self) -> Any:
        """The live trained model (post-run ``@output``). Raises while still deferred."""
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
    """Row-wise softmax. The model emits raw logits (no head activation), like every twin."""
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return cast(np.ndarray, exp / exp.sum(axis=-1, keepdims=True))


def _as_loss(value: Optional[MlxLoss], multilabel: bool = False) -> Callable[..., Any]:
    """An MLX loss from a callable or its NAME in ``mlx.nn.losses``.

    The string form is not sugar: MLX's losses are plain FUNCTIONS, so there is no class for a
    ``!class:`` tag to name or for confluid's registry to tag — the fastai and JAX situation.

    ``binary_cross_entropy`` is bound with ``with_logits=True`` for the multi-label case,
    because the head emits raw logits and the MLX default expects probabilities. That is the
    same trap the Keras backend documents for its string losses, and it fails the same silent
    way: a finite loss that simply does not train.
    """
    built = flow(value)
    if callable(built):
        return cast(Callable[..., Any], built)
    name = str(built)
    loss = getattr(nn.losses, name, None)
    if loss is None:
        raise ValueError(
            f"Unknown MLX loss {name!r}. Names come from `mlx.nn.losses` — e.g. cross_entropy, "
            "binary_cross_entropy, hinge_loss."
        )
    if multilabel and name == "binary_cross_entropy":
        return cast(Callable[..., Any], partial(loss, with_logits=True))
    return cast(Callable[..., Any], loss)


def _as_metric(value: MlxMetric) -> Callable[..., Any]:
    """A metric from a callable or its NAME (``"accuracy"`` / ``"accuracy_multi"``)."""
    built = flow(value)
    if callable(built):
        return cast(Callable[..., Any], built)
    metric = _METRICS.get(str(built))
    if metric is None:
        raise ValueError(f"Unknown metric {built!r}. Known names: {', '.join(sorted(_METRICS))}.")
    return metric


def _as_optimizer(value: MlxOptimizer, learning_rate: float) -> Any:
    """An ``mlx.optimizers`` optimizer from an instance, a class, or its NAME.

    Unlike optax (factories) MLX ships CLASSES, so a name is resolved and INSTANTIATED with the
    learning rate — keeping the shared ``learning_rate`` knob the one place an LR is written.
    """
    built = flow(value)
    if isinstance(built, optim.Optimizer):
        return built
    if isinstance(built, type) and issubclass(built, optim.Optimizer):
        return built(learning_rate=learning_rate)  # type: ignore[call-arg]  # each concrete optimizer takes it
    factory = getattr(optim, str(built), None)
    if factory is None:
        raise ValueError(f"Unknown MLX optimizer {built!r}. Names come from `mlx.optimizers` — e.g. Adam, AdamW, SGD.")
    return factory(learning_rate=learning_rate)
