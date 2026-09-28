"""``JaxClassifier`` — the same classification run, on a jitted ``optax`` loop.

Since P6 (2026-08-08) this is ``(ClassificationBase, matrainer.jax.JaxLoopRunnable)``: the
loop machinery — the jit-ONCE step builders (``nnx.value_and_grad`` with the logits as an aux
output), the epoch driver with the per-epoch reshuffle, the plain-function tracking hookup,
the per-epoch callback fan-out, the record-batch predict walk — lives in
:class:`matrainer.jax.loop.JaxLoopRunnable`; the label pipeline (the multilabel peek,
``LabelMap`` fitting/encoding, ``num_classes`` resolution) in
:class:`~sonair.classification_base.ClassificationBase`. What stays HERE is classification's
jax vocabulary:

* **The loss/metric defaults branch on the peeked target shape** —
  ``sigmoid_binary_cross_entropy`` (the JAX spelling of ``BCEWithLogitsLoss``) against the
  float multi-hot for a multi-label run, ``softmax_cross_entropy_with_integer_labels``
  otherwise; ``accuracy_multi`` (thresholded in LOGIT space at 0.0) vs ``accuracy``.
* **``f1_macro`` is accumulated over the split, never averaged per batch** — a per-batch
  average reads three to five points LOW on 10 classes at batch 32
  (``matrainer.metric_functions`` quantifies it), so the state hooks keep one confusion
  matrix per split and reduce once. A multi-label run DROPS a confusion metric with a
  warning rather than approximating one.
* **The class names persist in :meth:`_fit_epilogue`** — beside the sidecar, onto the trackers.

Losses and metrics are NAMES, not ``!class:`` tags, and that is a JAX fact — optax ships
plain functions, so there is no class for a tag to resolve; hence no
``jax_classification.py`` (``docs/architecture.md`` §5).

Minimal flat YAML::

    runnable: !class:sonair.JaxClassifier
      model: !lazy:sonair.FlaxConvNet
    train_set: !class:recordstream.Stream { source: ..., ops: !ref:preprocess }
    batch_size: 32
    max_epochs: 3
"""

from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Sequence, Tuple, Union, cast

import jax.numpy as jnp
import numpy as np
from confluid import Mandatory, Partial, configurable, flow
from flax import nnx
from loggair import get_logger
from matrainer.artifacts import save_class_names
from matrainer.jax.loop import EpochCallback, JaxLoopRunnable, JaxLoss, JaxMetric, JaxOptimizer
from matrainer.metric_functions import accumulate_confusion, macro_f1, zero_confusion
from matrainer.metric_names import MetricNames
from matrainer.metric_slots import merge_metric_specs
from matrainer.tracking.base import Tracker
from recordstream import ClassificationOutput, LabelMap, RecordSource, batch_values, multi_hot

from sonair.classification_base import ClassificationBase

logger = get_logger(__name__)

#: Which entry point :meth:`JaxClassifier.run` dispatches to — the same vocabulary every other
#: backend uses, because the verb is a property of the TASK, not of the engine.
ClassifierTask = Literal["fit", "evaluate", "test", "predict"]

__all__ = ["ClassifierTask", "EpochCallback", "JaxClassifier"]


def _accuracy(logits: Any, labels: Any) -> Any:
    """Top-1 accuracy for single-label targets (int64 ids)."""
    return jnp.mean(jnp.argmax(logits, axis=-1) == labels)


def _accuracy_multi(logits: Any, labels: Any) -> Any:
    """Element-wise accuracy for a multi-label target, thresholded in LOGIT space.

    ``0.0`` rather than ``0.5`` because the head emits raw logits — the same choice the Keras
    backend's thresholded metrics make with ``thresholds: 0.0``.
    """
    return jnp.mean((logits > 0.0).astype(labels.dtype) == labels)


#: Metric names this backend resolves PER BATCH. Plain functions, so there is nothing for
#: confluid to tag — see the module docstring.
_METRICS: Dict[str, Callable[..., Any]] = {"accuracy": _accuracy, "accuracy_multi": _accuracy_multi}

#: Metric names this backend resolves from the split's CONFUSION MATRIX rather than per batch,
#: because averaging them over batches gives a different — and systematically worse — number
#: than computing them once (``matrainer.metric_functions`` quantifies it: three to five points
#: of macro-F1 on 10 classes at batch 32). A name in here is accumulated across the whole pass
#: and reduced at the end; a name in ``_METRICS`` is scored per batch and averaged. Single-label
#: only: a confusion matrix has no meaning for a multi-hot target, and ``_split_confusion``
#: refuses one rather than reporting a number nobody can interpret.
_CONFUSION_METRICS: Dict[str, Callable[[Any], float]] = {"f1_macro": macro_f1}


@configurable(task="classification", role="trainer", framework="jax")
class JaxClassifier(ClassificationBase, JaxLoopRunnable):
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
        trackers: matrainer trackers — one or a list. This run's ONLY output wiring:
            metrics, provenance and the run manifest all go to them.
        callbacks: Callables invoked as ``callback(epoch, logs)`` after each epoch —
            JAX ships no callback protocol, so this is the plainest thing that works.
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
        metrics: Metric names (``"accuracy"``, ``"accuracy_multi"``, ``"f1_macro"``) or
            callables. ``None`` picks by target shape.
        train_metrics / val_metrics / test_metrics: That split's OWN metrics, merged over
            ``metrics``. The training pass scores its metrics from the logits the gradient step
            already computed, so ``train_metrics`` costs one extra reduction per batch and no
            second forward pass.
        metric_names: How recorded metrics are named — see
            :class:`~matrainer.metric_names.MetricNames`. This loop already writes the canonical
            ``<split>/<name>``, so the default translates nothing; wiring ``target:`` is what
            makes it record in another engine's spelling.
        jit: Trace the step functions with ``nnx.jit``. On by default — that IS the JAX story.
            Turn it off to read a Python traceback from inside a step.
    """

    def __init__(
        self,
        model: Optional[Mandatory[Partial[nnx.Module]]] = None,
        train_set: Optional[Partial[RecordSource]] = None,
        val_set: Optional[Partial[RecordSource]] = None,
        test_set: Optional[Partial[RecordSource]] = None,
        num_classes: Optional[int] = None,
        task: ClassifierTask = "fit",
        max_epochs: int = 10,
        batch_size: int = 32,
        trackers: Optional[Union[Partial[Tracker], Iterable[Partial[Tracker]]]] = None,
        callbacks: Optional[List[Partial[EpochCallback]]] = None,
        experiment_name: str = "sonair_classification",
        run_name: Optional[str] = None,
        input_key: str = "image",
        target_key: str = "class",
        label_map: Optional[LabelMap] = None,
        learning_rate: float = 1e-3,
        optimizer: JaxOptimizer = "adam",
        loss: Optional[JaxLoss] = None,
        metrics: Optional[Sequence[JaxMetric]] = None,
        train_metrics: Optional[Sequence[JaxMetric]] = None,
        val_metrics: Optional[Sequence[JaxMetric]] = None,
        test_metrics: Optional[Sequence[JaxMetric]] = None,
        metric_names: Optional[Partial[MetricNames]] = None,
        jit: bool = True,
    ) -> None:
        super().__init__(
            model=model,
            train_set=train_set,
            test_set=test_set,
            task=task,
            max_epochs=max_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            optimizer=optimizer,
            jit=jit,
            loss=loss,
            metrics=metrics,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
            trackers=trackers,
            callbacks=callbacks,
            experiment_name=experiment_name,
            run_name=run_name,
            input_key=input_key,
            target_key=target_key,
            metric_names=metric_names,
        )
        self.val_set = val_set
        self.num_classes = int(num_classes) if num_classes is not None else None
        self.label_map = label_map
        self.multilabel: bool = False
        # Balancing is a loss-side convention the optax functions do not carry — the task
        # base's capability probe answers no.
        self.balance_classes = False
        self._class_names: Optional[List[str]] = None

    # ---- the task hooks ---------------------------------------------------------------------- #
    def _default_jax_loss(self) -> Optional[JaxLoss]:
        """Both defaults take LOGITS, which is what the head emits — the JAX spellings of
        ``BCEWithLogitsLoss`` and ``CrossEntropyLoss``."""
        return "sigmoid_binary_cross_entropy" if self.multilabel else "softmax_cross_entropy_with_integer_labels"

    def _default_jax_metrics(self) -> Optional[List[Any]]:
        return ["accuracy_multi"] if self.multilabel else ["accuracy"]

    def _to_arrays(self, batch: Any, with_targets: bool = True) -> Any:
        """float32 image; ``multi_hot`` for a multi-label target, else int64 ids.

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

    # ---- the metric state: batchwise sums + ONE confusion matrix per split ------------------- #
    def _partition_metrics(self, split: str) -> Any:
        """One split's metrics, split into the per-batch ones and the confusion-matrix ones.

        Partitioned by NAME before resolution, which is why only a named metric can be
        confusion-based: a user-supplied callable is scored per batch, since nothing here can
        know what it would want accumulated.
        """
        batchwise: List[JaxMetric] = []
        confusion: Dict[str, Callable[[Any], float]] = {}
        for metric in merge_metric_specs(self.metrics, getattr(self, f"{split}_metrics")) or []:
            name = flow(metric)
            if isinstance(name, str) and name in _CONFUSION_METRICS:
                confusion[name] = _CONFUSION_METRICS[name]
            else:
                batchwise.append(metric)
        return batchwise, confusion

    def _split_confusion(self, confusion: Dict[str, Callable[[Any], float]]) -> Optional[np.ndarray]:
        """A zeroed accumulator when this split wants one, else ``None``.

        Multi-label runs get ``None`` even when a confusion metric was wired: a row with two
        true labels has no single cell to land in. The metric is DROPPED rather than
        approximated — a number under a name that means something else is worse than a gap.
        """
        if not confusion or self.multilabel:
            if confusion and self.multilabel:
                logger.warning(
                    f"JaxClassifier: {', '.join(sorted(confusion))} needs single-label targets and this run is "
                    f"multi-label — the metric is not reported. Use `accuracy_multi` for a multi-hot target."
                )
            return None
        if not self.num_classes:
            raise RuntimeError("JaxClassifier: a confusion-matrix metric needs num_classes.")
        return zero_confusion(self.num_classes)

    def _new_split_state(self, split: str) -> Any:
        batchwise, confusion = self._partition_metrics(split)
        return {
            "batchwise": batchwise,
            "confusion": confusion,
            "matrix": self._split_confusion(confusion),
            "sums": {},
            "batches": 0,
        }

    def _score_batch(self, state: Any, outputs: Any, arrays: Tuple[Any, ...], split: str) -> None:
        labels = arrays[1]
        state["batches"] += 1
        for metric in state["batchwise"]:
            fn = _as_metric(metric)
            name = getattr(fn, "__name__", str(metric)).lstrip("_")
            state["sums"][name] = state["sums"].get(name, 0.0) + float(fn(outputs, labels))
        if state["matrix"] is not None:
            accumulate_confusion(state["matrix"], np.asarray(outputs), np.asarray(labels))

    def _split_scores(self, state: Any, split: str, step: Optional[int] = None) -> Dict[str, float]:
        if not state["batches"]:
            return {}
        out = {name: value / state["batches"] for name, value in state["sums"].items()}
        if state["matrix"] is not None:
            out.update({name: fn(state["matrix"]) for name, fn in state["confusion"].items()})
        return out

    # ---- artifacts ---------------------------------------------------------------------------- #
    def _fit_epilogue(self) -> None:
        """Persist the run's label vocabulary beside the sidecar and onto the trackers."""
        save_class_names(self._class_names, self._artifact_dir(), self._live_trackers())

    def _artifact_dir(self) -> Optional[str]:
        """Where run artifacts land — beside the sidecar when one is configured."""
        if self.metrics_sidecar_path:
            return str(Path(self.metrics_sidecar_path).expanduser().parent)
        return None

    # ---- predictions -------------------------------------------------------------------------- #
    def _prediction_output(self, preds: Any) -> Any:
        """The batched :class:`~recordstream.ClassificationOutput` — the machinery slices it.

        The contract is recordstream's and framework-neutral — generic in its array type, so
        the SAME sink consumes this numpy output, the torch backends' tensors and the Keras
        backend's arrays with no adapter.
        """
        logits = np.asarray(preds)
        out: ClassificationOutput[np.ndarray] = {
            "logits": logits,
            "probs": _softmax(logits),
            "class_idx": np.argmax(logits, axis=-1),
        }
        return out


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Row-wise softmax. The model emits raw logits (no head activation), like every twin."""
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return cast(np.ndarray, exp / exp.sum(axis=-1, keepdims=True))


def _as_metric(value: JaxMetric) -> Callable[..., Any]:
    """A PER-BATCH metric from a callable or its NAME (``"accuracy"`` / ``"accuracy_multi"``).

    A confusion-matrix name (``"f1_macro"``) never reaches here — ``_partition_metrics`` takes
    those out first — but it is listed in the error, because "unknown metric" for a name the
    backend does support would send the reader looking in the wrong place.
    """
    built = flow(value)
    if callable(built):
        return cast(Callable[..., Any], built)
    metric = _METRICS.get(str(built))
    if metric is None:
        known = sorted({*_METRICS, *_CONFUSION_METRICS})
        raise ValueError(f"Unknown metric {built!r}. Known names: {', '.join(known)}.")
    return metric
