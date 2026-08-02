"""``JaxClassifier`` — the same classification run, on a jitted ``optax`` loop.

The sixth backend, and the one that shows what "owns the loop" means when the framework hands
you nothing. Lightning, Keras, Trainer and ``Learner`` each bring an epoch loop; JAX brings
function transformations and expects you to write the loop yourself. So this backend is the
closest sibling of :mod:`sonair.tensorflow_classifier` — both replace ``fit()`` with a traced
step function — and the differences between them are exactly the differences between the two
programming models:

* **The step is a PURE FUNCTION of (model, batch).** ``nnx.value_and_grad`` differentiates it
  and ``nnx.Optimizer`` applies the update in place, which is flax's object-oriented face over
  what is underneath a functional ``(params, grads) -> params``.
* **It is jitted with ``nnx.jit``, once**, and reused across epochs. A fresh ``jit`` per epoch
  would retrace the whole graph every time — the same trap the TensorFlow backend documents
  for ``tf.function``.
* **Randomness is explicit.** Dropout needs a key, so the model carries its own ``nnx.Rngs``
  rather than reading a global seed.

Everything else is shared and reused, not rewritten: the record stream, ``LabelMap`` fitting
and encoding, ``class_names``, the predictions sink and contract, ``save_class_names`` /
``write_metrics_sidecar``, and the ``@entrypoint`` dispatch. What is written here is
:meth:`JaxClassifier._to_arrays` (the batch shape — the twin of the Keras backend's method of
the same name), the step functions, and the epoch loop.

**Losses and metrics are NAMES, not ``!class:`` tags, and that is a JAX fact.** ``optax``
losses (``softmax_cross_entropy_with_integer_labels`` and friends) are plain *functions*, so
there is no class for a ``!class:`` tag to resolve or for confluid's registry to tag — exactly
the situation the fastai backend's ``fastai.metrics`` names are in. Hence no
``jax_classification.py`` (``docs/architecture.md`` §5).

Minimal flat YAML::

    runnable: !class:sonair.JaxClassifier
      model: !lazy:sonair.FlaxConvNet
    train_set: !class:recordstream.Stream { source: ..., ops: !ref:preprocess }
    batch_size: 32
    max_epochs: 3
"""

from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Sequence, Union, cast

import jax
import jax.numpy as jnp
import numpy as np
import optax
from confluid import Fluid, Lazy, Mandatory, configurable, flow, output
from flax import nnx
from loggair import get_logger
from marainer.artifacts import load_class_names, save_class_names, write_metrics_sidecar
from marainer.run_names import generate_run_name
from marainer.tracking.base import Tracker, as_tracker_list
from marainer.tracking.provenance import capture_log_file, log_run_context
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

#: Which entry point :meth:`JaxClassifier.run` dispatches to — the same vocabulary every other
#: backend uses, because the verb is a property of the TASK, not of the engine.
ClassifierTask = Literal["fit", "evaluate", "test", "predict"]

#: What the ``loss`` / ``metrics`` / ``optimizer`` slots accept: a NAME resolved against
#: ``optax``, or the callable itself. Spelled out rather than left as ``Any`` so ``to_pydantic``
#: (form specs / MCP schemas / editor widgets) has a real type to build from.
JaxLoss = Union[str, Callable[..., Any]]
JaxMetric = Union[str, Callable[..., Any]]
JaxOptimizer = Union[str, Callable[..., Any]]

__all__ = ["ClassifierTask", "JaxClassifier"]


def _first_target(source: Any, target_key: str) -> Any:
    """The first non-``None`` target in ``source`` — one peek is enough to know the label kind."""
    for target in iter_key(source, target_key):
        if target is not None:
            return target
    return None


def _run_prefix(task: str) -> str:
    return {"fit": "train", "predict": "predict"}.get(task, "evaluate")


def _accuracy(logits: Any, labels: Any) -> Any:
    """Top-1 accuracy for single-label targets (int64 ids)."""
    return jnp.mean(jnp.argmax(logits, axis=-1) == labels)


def _accuracy_multi(logits: Any, labels: Any) -> Any:
    """Element-wise accuracy for a multi-label target, thresholded in LOGIT space.

    ``0.0`` rather than ``0.5`` because the head emits raw logits — the same choice the Keras
    backend's thresholded metrics make with ``thresholds: 0.0``.
    """
    return jnp.mean((logits > 0.0).astype(labels.dtype) == labels)


#: Metric names this backend resolves. Plain functions, so there is nothing for confluid to
#: tag — see the module docstring.
_METRICS: Dict[str, Callable[..., Any]] = {"accuracy": _accuracy, "accuracy_multi": _accuracy_multi}


@configurable(task="classification", role="trainer", framework="jax")
class JaxClassifier(ProgressReporting):
    """Train, evaluate or predict an image classifier with a jitted optax loop.

    Args:
        model: The classification model — wire ``!lazy:sonair.FlaxConvNet`` so ``initialize()``
            can inject the dataset-derived ``num_classes``.
        train_set: Training source (``task="fit"``).
        val_set: Validation source, scored at the end of each epoch when wired.
        test_set: Evaluation / prediction source.
        num_classes: Class count. Omitted, it is derived from the data or the label map.
        task: Which entry point ``run()`` dispatches to.
        max_epochs: Training epochs.
        batch_size: Rows per batch.
        logger: Callables invoked as ``callback(epoch, logs)`` after each epoch — the JAX
            analogue of the other backends' ``logger`` slot. JAX ships no callback protocol,
            so this is deliberately the plainest thing that works.
        experiment_name: Run identity, recorded in artifacts and tracking.
        run_name: Run identity; a random ``adjective_noun`` when omitted.
        input_key: Record key holding the image.
        target_key: Record key holding the label.
        label_map: Pin the name-to-id encoding instead of fitting it from the train targets.
        learning_rate: Peak LR for the optimizer.
        optimizer: An ``optax`` optimizer name (``"adam"``, ``"adamw"``, ``"sgd"``) or a
            callable returning a ``GradientTransformation``.
        loss: An ``optax`` loss name or callable over ``(logits, labels)``. ``None`` picks by
            target shape, like every other backend's sentinel.
        metrics: Metric names (``"accuracy"``, ``"accuracy_multi"``) or callables. ``None``
            picks by target shape.
        jit: Trace the step functions with ``nnx.jit``. On by default — that IS the JAX story.
            Turn it off to read a Python traceback from inside a step.
    """

    def __init__(
        self,
        model: Mandatory[Lazy[nnx.Module]],
        train_set: Optional[Lazy[RecordSource]] = None,
        val_set: Optional[Lazy[RecordSource]] = None,
        test_set: Optional[Lazy[RecordSource]] = None,
        num_classes: Optional[int] = None,
        task: ClassifierTask = "fit",
        max_epochs: int = 10,
        batch_size: int = 32,
        trackers: Optional[Union[Lazy[Tracker], Iterable[Lazy[Tracker]]]] = None,
        experiment_name: str = "sonair_classification",
        run_name: Optional[str] = None,
        input_key: str = "image",
        target_key: str = "class",
        label_map: Optional[LabelMap] = None,
        learning_rate: float = 1e-3,
        optimizer: JaxOptimizer = "adam",
        loss: Optional[JaxLoss] = None,
        metrics: Optional[Sequence[JaxMetric]] = None,
        jit: bool = True,
    ) -> None:
        super().__init__()
        # `nnx.Module`, not `Lazy[nnx.Module]`: the deferred-ness is declared once, on the ctor
        # param, which is what `lazy_param_names` / `to_pydantic` read; the ATTRIBUTE's useful
        # type is what it holds once `initialize()` has flowed it. `nnx.Optimizer(model, …)` and
        # `nnx.value_and_grad` both require a real `nnx.Module`, so that IS the slot's contract.
        self.model: nnx.Module = cast(nnx.Module, model)
        self.train_set: Optional[Lazy[RecordSource]] = train_set
        self.val_set: Optional[Lazy[RecordSource]] = val_set
        self.test_set: Optional[Lazy[RecordSource]] = test_set
        self.task: ClassifierTask = task
        self.num_classes = int(num_classes) if num_classes is not None else None
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        # Internal only: this backend's own tracking callback is appended at run time.
        self.callbacks: List[Any] = []
        self.experiment_name = experiment_name
        self.run_name = run_name if run_name is not None else generate_run_name(prefix=_run_prefix(task))
        self.input_key = str(input_key)
        self.target_key = str(target_key)
        self.label_map = label_map
        self.learning_rate = learning_rate
        self.optimizer: JaxOptimizer = optimizer
        # Sentinels, for the same reason every other backend uses them: the right loss and
        # metric depend on whether the targets are single- or multi-label, which is only known
        # once `initialize()` has peeked the data. A live default here could not be told apart
        # from a user's explicit choice.
        self.loss: Optional[JaxLoss] = loss
        self.metrics: Optional[Sequence[JaxMetric]] = metrics
        self.jit = jit
        self.multilabel: bool = False
        self.predictions_sink: Optional[Lazy[PredictionsSink]] = None
        self.metrics_sidecar_path: Optional[str] = None
        # Stored VERBATIM (lazy-init mandate: the constructor does no work). It may hold a
        # single tracker, a list, or deferred `Fluid` markers; `_live_trackers()` normalizes
        # and flows it at run time.
        self.trackers: Optional[Union[Lazy[Tracker], Iterable[Lazy[Tracker]]]] = trackers
        self._class_names: Optional[List[str]] = None
        self._computed_metrics: Dict[str, float] = {}
        self._optimizer_state: Optional[Any] = None

    # ---- run dispatch ------------------------------------------------------- #
    def run(self) -> None:
        self.initialize()
        # The `@entrypoint` markers below ARE the dispatch table (recordstream docs/runnable.md).
        run_entrypoint(self, self.task)

    # ---- setup -------------------------------------------------------------- #
    def initialize(self) -> None:
        """Resolve labels and ``num_classes``, then build the model at that width."""
        logger.info(f"JAX devices: {jax.devices()}")
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

        The multi-label pair is optax's ``sigmoid_binary_cross_entropy`` — the JAX spelling of
        ``BCEWithLogitsLoss`` — against the float multi-hot ``_to_arrays`` produces. Both
        default losses take LOGITS, which is what the head emits.
        """
        if self.loss is None:
            self.loss = (
                "sigmoid_binary_cross_entropy" if self.multilabel else "softmax_cross_entropy_with_integer_labels"
            )
        if self.metrics is None:
            self.metrics = ["accuracy_multi"] if self.multilabel else ["accuracy"]

    # ---- data --------------------------------------------------------------- #
    def _sequence(self, source: Any, shuffle: bool = False, with_targets: bool = True) -> RecordSequence:
        """recordstream's batching adapter, wearing THIS task's batch shape.

        The same adapter the Keras backend uses — it is engine-neutral (row order, slicing,
        per-epoch reshuffle, ``collate_records``) and knows nothing about classification or
        about JAX. ``_to_arrays`` is handed to it exactly as a ``collate_fn`` is handed to a
        ``DataLoader``.
        """
        from functools import partial

        return RecordSequence(
            source,
            batch_size=self.batch_size,
            shuffle=shuffle,
            transform=partial(self._to_arrays, with_targets=with_targets),
        )

    def _to_arrays(self, batch: Any, with_targets: bool = True) -> Any:
        """One collated record batch → the numpy ``(x, y)`` this loop consumes.

        Identical to the Keras backend's method of the same name, and for the same reason:
        recordstream hands over NUMPY already (``batch_values`` / ``multi_hot`` are not
        torch-typed), and numpy is exactly what ``jnp.asarray`` takes with no copy on CPU. So
        the JAX backend needs no conversion layer at all — that is what those return types buy.

        Channels-LAST, like Keras: the preprocessing chain for a JAX run stops after
        ``ConvertToImage`` rather than adding a CHW ``ToTensor``.
        """
        x = np.asarray(batch_values(batch, self.input_key), dtype="float32")
        if not with_targets:
            return (x,)
        if self.multilabel:
            if not self.num_classes:
                raise RuntimeError("JaxClassifier: a multi-label target needs num_classes.")
            return x, multi_hot(batch, self.target_key, self.num_classes)
        return x, np.asarray(batch_values(batch, self.target_key), dtype="int64")

    # ---- the traced steps --------------------------------------------------- #
    def _loss_fn(self) -> Callable[..., Any]:
        """The resolved loss as a callable over ``(logits, labels)`` returning a scalar."""
        loss = _as_loss(self.loss)

        def compute(logits: Any, labels: Any) -> Any:
            return jnp.mean(loss(logits, labels))

        return compute

    def _make_steps(self) -> Any:
        """Build the train and eval steps once, and jit them once.

        Built once and reused across epochs on purpose: a fresh ``nnx.jit`` each epoch retraces
        the whole graph every time, which is slower than the eager loop it replaced — the same
        trap the TensorFlow backend documents for ``tf.function``.
        """
        loss_fn = self._loss_fn()

        def train_step(model: Any, optimizer: Any, x: Any, y: Any) -> Any:
            def forward(m: Any) -> Any:
                return loss_fn(m(x, training=True), y)

            loss, grads = nnx.value_and_grad(forward)(model)
            optimizer.update(model, grads)
            return loss

        def eval_step(model: Any, x: Any, y: Any) -> Any:
            logits = model(x, training=False)
            return loss_fn(logits, y), logits

        if self.jit:
            return nnx.jit(train_step), nnx.jit(eval_step)
        return train_step, eval_step

    def _metric_values(self, logits: Any, labels: Any) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for metric in self.metrics or []:
            fn = _as_metric(metric)
            out[getattr(fn, "__name__", str(metric)).lstrip("_")] = float(fn(logits, labels))
        return out

    def _run_epoch(self, sequence: RecordSequence, step: Any, model: Any) -> Dict[str, float]:
        """One pass over ``sequence`` in EVAL mode, returning ``{"loss": …, **metrics}``."""
        total, batches = 0.0, 0
        sums: Dict[str, float] = {}
        for index in range(len(sequence)):
            x, y = sequence[index]
            loss, logits = step(model, jnp.asarray(x), jnp.asarray(y))
            total += float(loss)
            batches += 1
            for key, value in self._metric_values(logits, jnp.asarray(y)).items():
                sums[key] = sums.get(key, 0.0) + value
        if not batches:
            return {"loss": float("nan")}
        return {"loss": total / batches, **{k: v / batches for k, v in sums.items()}}

    # ---- the four entry points ---------------------------------------------- #
    @entrypoint("fit", role="trainer", primary=True)
    def fit(self) -> None:
        logger.info(f"sonair jax training: {self.experiment_name} ({self.run_name})")
        if self.train_set is None:
            raise ValueError("JaxClassifier.fit() requires a 'train_set'.")
        # Provenance through marainer's plain FUNCTIONS — `runnable=self` so the
        # hyperparameters and the resolved config come from marainer's ONE derivation
        # rather than a dict this backend transcribes for itself.
        trackers = self._live_trackers()
        manifest = log_run_context(trackers, runnable=self)
        model = self.model
        optimizer = nnx.Optimizer(model, _as_optimizer(self.optimizer, self.learning_rate), wrt=nnx.Param)
        train_step, eval_step = self._make_steps()
        train_seq = self._sequence(self.train_set, shuffle=True)
        val_seq = self._sequence(self.val_set) if self.val_set is not None else None

        logs: Dict[str, float] = {}
        for epoch in range(self.max_epochs):
            total, batches = 0.0, 0
            for index in range(len(train_seq)):
                x, y = train_seq[index]
                total += float(train_step(model, optimizer, jnp.asarray(x), jnp.asarray(y)))
                batches += 1
            logs = {"loss": total / batches if batches else float("nan")}
            if val_seq is not None:
                logs.update({f"val_{k}": v for k, v in self._run_epoch(val_seq, eval_step, model).items()})
            # A `fit()` would have done this for us; a hand-written loop has to. Forgetting it
            # is INVISIBLE — training just runs on one fixed row order forever.
            train_seq.on_epoch_end()
            for callback in self.callbacks:
                flow(callback)(epoch, logs)
            # Trackers are the other per-epoch consumer. A `fit()` forwards its logs to every
            # attached logger; a hand-written loop reaches them the same way it reaches callbacks.
            for tracker in trackers:
                tracker.log_metrics(logs, step=epoch)
            logger.info(f"epoch {epoch + 1}/{self.max_epochs}: " + ", ".join(f"{k}={v:.4f}" for k, v in logs.items()))

        self._computed_metrics = logs
        self._optimizer_state = optimizer
        logger.info("Training complete.")
        save_class_names(self._class_names, self._artifact_dir(), trackers)
        write_metrics_sidecar(self.metrics_sidecar_path, self._computed_metrics, self.experiment_name, self.run_name)
        # Uploads the log file AND stamps the manifest's terminal status. Not optional once a
        # manifest exists: left unfinished it stays `running`, and marainer's atexit hook then
        # calls a successful run `interrupted`.
        capture_log_file(trackers, manifest=manifest, status="success")

    @entrypoint("evaluate", role="evaluator")
    def evaluate(self) -> None:
        self.test()

    @entrypoint("test", role="evaluator", primary=True)
    def test(self) -> None:
        if self.test_set is None:
            raise ValueError("JaxClassifier.test() requires a 'test_set'.")
        _, eval_step = self._make_steps()
        self._computed_metrics = self._run_epoch(self._sequence(self.test_set), eval_step, self.model)
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
            raise ValueError("JaxClassifier.predict() requires a 'test_set'.")
        sink = flow(self.predictions_sink)
        sequence = self._sequence(self.test_set, with_targets=False)
        for index, batch in enumerate(sequence.batches()):
            logits = np.asarray(self.model(jnp.asarray(sequence[index][0]), training=False))
            if sink is not None:
                self._write_predictions(logits, batch, cast(PredictionsSink, sink))
        if sink is not None and hasattr(sink, "close"):
            sink.close()
        logger.info("Prediction complete.")

    # ---- predictions -------------------------------------------------------- #
    def _write_predictions(self, logits: np.ndarray, batch: Any, sink: PredictionsSink) -> None:
        """One :class:`~recordstream.ClassificationOutput` per record, paired with its metadata.

        The contract is recordstream's and framework-neutral — it is generic in its array type,
        so the SAME sink consumes this numpy output, the torch backends' tensors and the Keras
        backend's arrays with no adapter.
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
        """Normalize and flow the wired trackers ONCE, writing the result back to the slot.

        Written back rather than returned fresh each call: ``flow()`` on a LIVE object returns
        that same object (idempotent), but a deferred ``Fluid`` would be REBUILT on every call —
        two MLflow runs for one training run. One attribute, flowed in place, is the same idiom
        this class already uses for ``model`` / ``loss_fn`` / ``predictions_sink``.
        """
        # `flow()` resolves a marker to the live object, which the Lazy-typed slot cannot
        # express — the cast says what the call guarantees.
        self.trackers = [flow(t) for t in as_tracker_list(self.trackers) or []]
        return cast(List[Tracker], self.trackers)

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


def _as_loss(value: Optional[JaxLoss]) -> Callable[..., Any]:
    """An optax loss from a callable or its NAME in ``optax.losses``.

    The string form is not sugar: optax's losses are plain FUNCTIONS, so there is no class for
    a ``!class:`` tag to name or for confluid's registry to tag — the same situation the fastai
    backend's metrics are in.
    """
    built = flow(value)
    if callable(built):
        return cast(Callable[..., Any], built)
    loss = getattr(optax.losses, str(built), None) or getattr(optax, str(built), None)
    if loss is None:
        raise ValueError(
            f"Unknown optax loss {built!r}. Names come from `optax.losses` — e.g. "
            "softmax_cross_entropy_with_integer_labels, sigmoid_binary_cross_entropy."
        )
    return cast(Callable[..., Any], loss)


def _as_metric(value: JaxMetric) -> Callable[..., Any]:
    """A metric from a callable or its NAME (``"accuracy"`` / ``"accuracy_multi"``)."""
    built = flow(value)
    if callable(built):
        return cast(Callable[..., Any], built)
    metric = _METRICS.get(str(built))
    if metric is None:
        raise ValueError(f"Unknown metric {built!r}. Known names: {', '.join(sorted(_METRICS))}.")
    return metric


def _as_optimizer(value: JaxOptimizer, learning_rate: float) -> Any:
    """An ``optax.GradientTransformation`` from a callable or its NAME in ``optax``.

    A NAME is resolved and CALLED with the learning rate, because optax optimizers are
    factories (``optax.adam(1e-3)``) rather than classes — so the ``learning_rate`` knob stays
    the single place the LR is written, shared with every other backend's config.
    """
    built = flow(value)
    if callable(built) and not isinstance(built, str):
        made = built(learning_rate) if not hasattr(built, "update") else built
        return made
    factory = getattr(optax, str(built), None)
    if factory is None:
        raise ValueError(f"Unknown optax optimizer {built!r}. Names come from `optax` — e.g. adam, adamw, sgd.")
    return factory(learning_rate)
