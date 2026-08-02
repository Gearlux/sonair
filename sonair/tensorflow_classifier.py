"""``TensorFlowClassifier`` — the same classification run as a NATIVE TensorFlow training loop.

This backend exists to answer a question the project had left open in ``TASKS.md``: does
``sonair train --framework tensorflow`` need a runnable of its own at all, when ``KERAS_BACKEND=tensorflow
sonair train <cfg> --framework keras`` already executes the identical config on TensorFlow? The answer is
*only for what Keras 3 does not expose*:

* **``tf.data`` instead of a ``PyDataset``.** :meth:`_dataset` wraps recordstream's
  ``RecordSequence`` in ``tf.data.Dataset.from_generator`` and appends
  ``.prefetch(tf.data.AUTOTUNE)``, so the record walk (decode, resize, collate) overlaps the
  device step. ``keras.Model.fit`` prefetches with a worker pool instead; the graph pipeline is
  a TensorFlow primitive with no Keras equivalent.
* **``tf.GradientTape`` under ``@tf.function`` instead of ``fit()``.** The forward/backward pair
  is written out and traced into a graph, which is what makes ``jit_compile=True`` (XLA) a knob
  at all.
* **SavedModel export.** ``export_dir`` writes a ``tf.saved_model`` directory — the TensorFlow
  serving artifact, and the reason a run would pick this backend over the Keras one.

**This class is a SIBLING of :class:`~sonair.keras_classifier.KerasClassifier`, not a subclass
(2026-08-01).** It was one until this change, and the inheritance was measurably load-bearing —
fourteen of the parent's twenty-one methods came through untouched. What retired it is that the
seven it DID override never delegated: `fit`, `test`, `predict`, `_compile` and
`_resolve_task_defaults` replaced the parent's bodies outright, so the base class was carrying
code reuse rather than substitutability. It also NARROWED a precondition —
:meth:`_require_tensorflow_backend` raises where a ``KerasClassifier`` runs fine — which is a
Liskov violation a picker offering "any KerasClassifier" would walk straight into.

The cost is honest and visible: the label pipeline, the batch shape, the prediction contract and
the artifact writers are now DUPLICATED between this module and the Keras one, and they are
duplicated in the Lightning, fastai, jax and mlx backends too. Extracting an engine-neutral base
is deliberately deferred until the detection projects have been through the same treatment, so
the base gets designed against classification AND detection rather than against classification
twice — see ``TASKS.md``. Until then, a change to the label handling has to be made in each
backend, and ``tests/test_tensorflow_classifier.py`` pins that this one still agrees with the
Keras one.

It still declares ``framework="keras"``, and that is unchanged by the un-subclassing: the
framework axis names the API a class can be WIRED TO (``docs/architecture.md`` §2), and in
TensorFlow >= 2.16 ``tf.keras`` IS Keras 3 — the losses, the metrics and ``KerasModel`` are the
same objects here as there. Which is also why there is still no ``tensorflow_classification.py``.

Minimal flat YAML — note it differs from the Keras block by ONE key, the runnable's class::

    runnable: !class:sonair.TensorFlowClassifier
      model: !lazy:sonair.KerasModel { model_name: MobileNetV3Small }
      export_dir: ./exported/mnist
    train_set: !class:recordstream.Stream { source: ..., ops: !ref:preprocess }
    batch_size: 32
    max_epochs: 3

Run it with ``sonair train cfg.yaml --framework tensorflow``, which sets ``KERAS_BACKEND=tensorflow`` before
anything imports keras — see :mod:`sonair._tensorflow` for why that ordering is load-bearing.
"""

from functools import partial
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple, Union, cast

import numpy as np
from confluid import Fluid, Lazy, Mandatory, configurable, flow, output
from loggair import get_logger
from marainer.artifacts import load_class_names, save_class_names, write_metrics_sidecar
from marainer.metric_slots import merge_metric_specs
from marainer.run_names import generate_run_name, run_prefix
from marainer.tracking.base import Tracker, as_tracker_list
from recordstream import (
    ClassificationOutput,
    LabelMap,
    PredictionsSink,
    RecordSource,
    batch_metadata,
    batch_values,
    class_names,
    ensure_record_dataset,
    first_value,
    is_class_id,
    iter_key,
    multi_hot,
)
from recordstream.runnable import ProgressReporting, entrypoint, run_entrypoint

# keras, TensorFlow AND the batching adapter all come from `sonair._tensorflow`, which pins
# KERAS_BACKEND=tensorflow before any of them load. Importing `RecordSequence` from its own
# module (`recordstream.keras`) would defeat that: isort sorts a `recordstream` line into the
# third-party group, above every `sonair` line, and `recordstream.keras` is where the competing
# default lives. Pinned by `tests/test_tensorflow_classifier.py`.
from sonair._tensorflow import RecordSequence, keras, keras_backend, tf

logger = get_logger(__name__)

#: Which entry point :meth:`TensorFlowClassifier.run` dispatches to — the same vocabulary every
#: other backend uses, because the verb is a property of the TASK, not of the engine.
ClassifierTask = Literal["fit", "evaluate", "test", "predict"]

# What each slot accepts: the Keras class, or the string alias it resolves to. Spelled out
# rather than left as `Any` because `Any` types nothing (workspace mandate) and gives
# `to_pydantic` nothing to build a widget from — navigaitor's form-spec and StreamStudio read
# these to know that `loss` takes a `keras.losses.Loss`, which is the other half of the set
# registered in `sonair.keras_classification` being pickable at all.
#
# It does NOT change how a wired value materializes: a nested `!class:` in a CONSTRUCTOR slot
# stays a deferred `Class` marker regardless of annotation (only the instance form
# `!class:X()` builds eagerly), so `_compile` flows these at use.
KerasLoss = Union[str, keras.losses.Loss]
KerasMetric = Union[str, keras.metrics.Metric]
KerasOptimizer = Union[str, keras.optimizers.Optimizer]

#: What this backend's ``model`` slot admits — the ONE place in sonair where two frameworks meet
#: in a single slot. ``keras.Model`` covers ``KerasModel`` and ``KerasConvNet``; ``tf.Module``
#: covers ``TensorFlowModel`` (a restored SavedModel) and ``TensorFlowConvNet`` (raw ``tf.nn``
#: ops), and NEITHER is a subclass of the other: under the TensorFlow backend a ``keras.Model``
#: descends ``Layer -> TFLayer -> KerasAutoTrackable`` (``keras/src/backend/tensorflow/
#: trackable.py``), so the nearest shared ancestor is ``tf.__internal__.tracking.AutoTrackable``
#: — a private API this must not name. A union of the two public types is therefore the honest
#: declaration, and it is spelled out rather than left as ``Any`` so a reader, mypy and
#: ``to_pydantic`` (form specs / MCP schemas / editor widgets) all see the real contract.
#: The two arms are told apart at the points where they differ — ``_export`` branches on
#: ``hasattr(model, "export")``, ``_restored_trainable_variables`` on the declared variable list.
TensorFlowNetwork = Union[keras.Model, tf.Module]

__all__ = ["ClassifierTask", "TensorFlowClassifier", "TensorFlowNetwork"]


@configurable(task="classification", role="trainer", framework="keras")
class TensorFlowClassifier(ProgressReporting):
    """Train, evaluate or predict an image classifier with a native TensorFlow loop.

    Every constructor knob means the same thing as its :class:`~sonair.keras_classifier.
    KerasClassifier` namesake, deliberately: one config drives either backend and only the
    runnable's class changes. The three at the end are what this backend adds.

    Args:
        model: The classification model — wire ``!lazy:sonair.KerasModel`` so ``initialize()``
            can inject the dataset-derived ``num_classes``. ``tf.keras`` IS Keras 3, so the
            Keras backend's wrapper is the wrapper here too — and this backend ALSO takes a
            plain ``tf.Module`` (``TensorFlowModel``, ``TensorFlowConvNet``), which is why the
            slot is typed :data:`TensorFlowNetwork`.
        train_set: Training source (``task="fit"``).
        test_set: Evaluation / prediction source.
        num_classes: Class count. Omitted, it is derived from the data or the label map.
        task: Which entry point ``run()`` dispatches to.
        max_epochs: Training epochs.
        batch_size: Rows per batch.
        trackers: marainer trackers — one or a list. This run's ONLY output wiring:
            metrics, provenance and the run manifest all go to them.
        callbacks: ``keras.callbacks.Callback`` instances, driven by hand through a
            ``keras.callbacks.CallbackList`` — this loop has no ``fit()`` to run them.
        experiment_name: Run identity, recorded in artifacts and tracking.
        run_name: Run identity; a random ``adjective_noun`` when omitted.
        input_key: Record key holding the image.
        target_key: Record key holding the label.
        label_map: Pin the name-to-id encoding instead of fitting it from the train targets.
        optimizer: Keras optimizer — a string alias (``"adam"``) or a ``keras.optimizers``
            instance. Resolved to an INSTANCE before the loop, since there is no ``compile()``
            to do it.
        loss: Keras loss. ``None`` picks by target shape (``from_logits=True`` either way — the
            head is a bare ``Dense``).
        metrics: ``keras.metrics.Metric`` instances. Unlike the Keras backend this cannot take
            the bare string ``"accuracy"`` at face value — see :meth:`_resolve_task_defaults`.
        train_metrics / val_metrics / test_metrics: That split's OWN metrics, merged over
            ``metrics`` — how a metric is added to one split only. This backend narrows properly:
            each traced step closes over its own split's metric objects.
        export_dir: Write a ``tf.saved_model`` directory here after a successful ``fit``.
            ``None`` (the default) exports nothing.
        jit_compile: Trace the train/eval steps with XLA. A real speedup on fixed shapes and a
            real compile cost on the first batch, so it is opt-in.
        prefetch: Append ``.prefetch(tf.data.AUTOTUNE)`` to every ``tf.data`` pipeline, which
            overlaps the record walk with the device step. Turn it off only to measure.
    """

    def __init__(
        self,
        model: Mandatory[Lazy[TensorFlowNetwork]],
        train_set: Optional[Lazy[RecordSource]] = None,
        test_set: Optional[Lazy[RecordSource]] = None,
        num_classes: Optional[int] = None,
        task: ClassifierTask = "fit",
        max_epochs: int = 10,
        batch_size: int = 32,
        trackers: Optional[Union[Lazy[Tracker], Iterable[Lazy[Tracker]]]] = None,
        callbacks: Optional[List[Lazy[keras.callbacks.Callback]]] = None,
        experiment_name: str = "sonair_classification",
        run_name: Optional[str] = None,
        input_key: str = "image",
        target_key: str = "class",
        label_map: Optional[LabelMap] = None,
        optimizer: Lazy[KerasOptimizer] = "adam",
        loss: Optional[Lazy[KerasLoss]] = None,
        metrics: Optional[Sequence[KerasMetric]] = None,
        train_metrics: Optional[Sequence[KerasMetric]] = None,
        val_metrics: Optional[Sequence[KerasMetric]] = None,
        test_metrics: Optional[Sequence[KerasMetric]] = None,
        export_dir: Optional[str] = None,
        jit_compile: bool = False,
        prefetch: bool = True,
    ) -> None:
        super().__init__()
        # `TensorFlowNetwork`, not `Lazy[TensorFlowNetwork]`: the DEFERRED-ness is declared once,
        # on the ctor param above, which is what `lazy_param_names` / `to_pydantic` read.
        # Restating it here would only make the attribute a union no method could call through.
        self.model: TensorFlowNetwork = model
        self.train_set: Optional[Lazy[RecordSource]] = train_set
        self.val_set: Optional[Lazy[RecordSource]] = None
        self.test_set: Optional[Lazy[RecordSource]] = test_set
        self.task: ClassifierTask = task
        self.num_classes = int(num_classes) if num_classes is not None else None
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        # Internal only: this backend's own tracking callback is appended at run time.
        # Stored VERBATIM like `trackers` (lazy-init): entries may be deferred `Fluid`
        # markers, and they are flowed where they are used.
        self.callbacks: Optional[List[Lazy[keras.callbacks.Callback]]] = callbacks
        self.experiment_name = experiment_name
        self.run_name = run_name if run_name is not None else generate_run_name(prefix=run_prefix(task))
        self.input_key = str(input_key)
        self.target_key = str(target_key)
        self.label_map = label_map
        self.optimizer: KerasOptimizer = optimizer
        # Sentinels, for the same reason every other backend uses them: the right loss and
        # metrics depend on whether the targets are single- or multi-label, which is only known
        # once `initialize()` has peeked the data. A live default here could not be told apart
        # from a user's explicit choice.
        self.loss: Optional[KerasLoss] = loss
        self.metrics: Optional[Sequence[KerasMetric]] = metrics
        # Per-split ADDITIONS, merged over `metrics` by the shared rule. This backend can serve
        # all THREE — unlike its Keras sibling, whose `compile()` owns one metric list for the
        # whole run — because the loop here is hand-written and each traced step closes over
        # its own split's metric OBJECTS.
        self.train_metrics: Optional[Sequence[KerasMetric]] = train_metrics
        self.val_metrics: Optional[Sequence[KerasMetric]] = val_metrics
        self.test_metrics: Optional[Sequence[KerasMetric]] = test_metrics
        self.multilabel: bool = False
        self.predictions_sink: Optional[Lazy[PredictionsSink]] = None
        self.metrics_sidecar_path: Optional[str] = None
        # Stored VERBATIM (lazy-init mandate: the constructor does no work). It may hold a
        # single tracker, a list, or deferred `Fluid` markers; `_live_trackers()` normalizes
        # and flows it at run time.
        self.trackers: Optional[Union[Lazy[Tracker], Iterable[Lazy[Tracker]]]] = trackers
        self.export_dir = export_dir
        self.jit_compile = jit_compile
        self.prefetch = prefetch
        self._class_names: Optional[List[str]] = None
        self._history: Dict[str, Any] = {}
        self._computed_metrics: Dict[str, float] = {}
        # The loop's parts, built by `_compile()` — what `keras.Model.compile` would have owned.
        self._optimizer: Any = None
        self._loss_fn: Any = None
        # One list of LIVE metric objects per split: they are stateful, so the splits cannot
        # share instances once their metric sets may differ.
        self._metric_objects: Dict[str, List[Any]] = {}

    # ---- run dispatch ------------------------------------------------------- #
    def run(self) -> None:
        self.initialize()
        # The `@entrypoint` markers below ARE the dispatch table (recordstream docs/runnable.md).
        run_entrypoint(self, self.task)

    # ---- setup -------------------------------------------------------------- #
    def initialize(self) -> None:
        """Refuse to run on the wrong engine, then resolve labels, ``num_classes`` and the loop."""
        self._require_tensorflow_backend()
        if self.task == "fit":
            self._initialize_fit()
        else:
            self._initialize_eval()
        self._compile()

    @staticmethod
    def _require_tensorflow_backend() -> None:
        """Fail loudly when keras loaded some OTHER engine — the one way this backend goes wrong.

        ``tf.GradientTape`` cannot differentiate a torch-backed ``keras.Model``, and the failure
        deep inside the tape says nothing useful. This is not hypothetical: ``recordstream.keras``
        defaults ``KERAS_BACKEND`` to the first backend INSTALLED, in the order torch -> tensorflow
        -> jax, so any machine with both ``sonair[lightning]`` and ``sonair[tensorflow]`` lands on
        torch unless something set the variable first (:mod:`sonair._tensorflow` does; ``sonair
        tensorflow <verb>`` does it earlier still).

        This check is also why the class is no longer a ``KerasClassifier``: it NARROWS a
        precondition its former parent does not have.
        """
        backend = keras_backend()
        if backend != "tensorflow":
            raise RuntimeError(
                f"TensorFlowClassifier needs Keras on the TensorFlow engine, but keras loaded the "
                f"{backend!r} one. Keras reads KERAS_BACKEND ONCE, at import time. Run this as "
                f"`sonair tensorflow <verb> <config>` (which sets it before anything imports keras), "
                f"or export KERAS_BACKEND=tensorflow yourself. To train on {backend!r} instead, use "
                f"the `keras` backend — it runs the identical config on whichever engine is loaded."
            )

    def _initialize_fit(self) -> None:
        """Fit the label encoding and resolve ``num_classes`` from the training data.

        Deliberately identical to the Keras backend's method of the same name (and to the
        Lightning, fastai, jax and mlx ones): none of it is engine-specific, and it is duplicated
        rather than shared until the base can be designed against detection too — see the module
        docstring. `tests/test_tensorflow_classifier.py` asserts the two still agree.
        """
        self.train_set = ensure_record_dataset(self.train_set)
        if self.val_set is not None:
            self.val_set = ensure_record_dataset(self.val_set)
        if self.test_set is not None:
            self.test_set = ensure_record_dataset(self.test_set)

        # `iter_key` unwraps a MultiLabel to its list of values, so a sequence here IS a
        # multi-label target — the item type, not a guess about what a list might mean.
        first = first_value(self.train_set, self.target_key)
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
                # An empty map still maps ids through — that is the `to_ids` contract.
                ids = [i for t in iter_key(self.train_set, self.target_key) for i in LabelMap().to_ids(t)]
                self.num_classes = max(ids) + 1 if ids else None
        if not self.num_classes:
            raise RuntimeError("could not determine num_classes — set it, or wire a labelled dataset.")
        self._class_names = class_names(self.train_set, self.val_set, self.test_set)

    def _initialize_eval(self) -> None:
        """Model-driven setup: the trained head's width is authoritative, not the eval split's.

        An eval dataset may be a subset that never shows every class, so deriving the count from
        it would mis-size the metrics against the head that produced the logits.
        """
        self.model = flow(self.model)
        nc = getattr(self.model, "num_classes", None)
        if not nc:
            raise RuntimeError("could not determine num_classes from the model (it must expose `num_classes`).")
        self.num_classes = int(nc)
        if self.test_set is None:
            return
        dataset = ensure_record_dataset(self.test_set)
        first = first_value(dataset, self.target_key)
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
        """Fill the unset (``None``) task-shaped slots with OBJECTS, not string aliases.

        The Keras backend can default ``metrics`` to ``["accuracy"]`` because ``compile()``
        resolves that alias against the loss and the target shape. A hand-written loop has no
        ``compile()``, and ``keras.metrics.get("accuracy")`` returns the *elementwise-equality*
        function — which, handed int64 ids and ``[N, C]`` logits, reports 0.0 forever. So the
        defaults here are the concrete classes ``compile()`` would have chosen, and
        :func:`_as_metric` maps the alias the same way for anyone who writes it explicitly.

        ``from_logits=True`` is mandatory for the identical reason as on the Keras backend: the
        head is a bare ``Dense``, and a crossentropy told otherwise normalizes by the row SUM,
        then clips — finite, and wrong.
        """
        if self.loss is None:
            self.loss = (
                keras.losses.BinaryCrossentropy(from_logits=True)
                if self.multilabel
                else keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            )
        if self.metrics is None:
            self.metrics = (
                [keras.metrics.BinaryAccuracy(threshold=0.0)]
                if self.multilabel
                else [keras.metrics.SparseCategoricalAccuracy()]
            )

    def _compile(self) -> None:
        """Build the network and the loop's PARTS — deliberately never ``keras.Model.compile``.

        A compiled model owns its own training step, which is precisely what this backend
        replaces. The slots are flowed at USE for the same reason every other backend does it —
        a nested ``!class:`` in a CONSTRUCTOR slot arrives as a deferred marker.

        A bare ``flow()`` is the whole guard: it returns a live object, a string alias and a list
        unchanged, so an ``isinstance(..., Fluid)`` around any of these calls only restates it.
        The model included — it reaches here deferred (``!lazy:`` is skipped by confluid's
        auto-flow walkers precisely so the dataset-derived width can be injected now), and one
        that is ALREADY live comes back untouched.
        """
        self._resolve_task_defaults()
        self.model = flow(self.model, num_classes=self.num_classes)
        self._optimizer = _as_optimizer(self.optimizer)
        self._loss_fn = flow(self.loss)
        self._metric_objects = {
            split: [
                _as_metric(m, multilabel=self.multilabel)
                for m in (merge_metric_specs(self.metrics, getattr(self, f"{split}_metrics")) or [])
            ]
            for split in ("train", "val", "test")
        }

    # ---- batching ------------------------------------------------------------ #
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
        """One collated record batch → the numpy ``(x, y)`` the traced step consumes.

        recordstream hands over numpy already — ``batch_values`` unwraps a batched key past its
        item type, ``multi_hot`` renders a multi-label column as an ``[N, C]`` matrix — so nothing
        is converted twice and nothing is reimplemented for this backend.

        TensorFlow is **channels-last**: images arrive ``[N, H, W, C]``. The preprocessing config
        therefore ends after ``ConvertToImage`` rather than adding a CHW ``ToTensor``.
        """
        x = np.asarray(batch_values(batch, self.input_key), dtype="float32")
        if not with_targets:
            return (x,)
        if self.multilabel:
            if not self.num_classes:
                raise RuntimeError("TensorFlowClassifier: a multi-label target needs num_classes.")
            return x, multi_hot(batch, self.target_key, self.num_classes)
        return x, np.asarray(batch_values(batch, self.target_key), dtype="int64")

    # ---- tf.data ------------------------------------------------------------- #
    def _dataset(self, source: Any, shuffle: bool = False, with_targets: bool = True) -> Tuple[Any, Any]:
        """A ``tf.data.Dataset`` over recordstream batches, plus the sequence that feeds it.

        The sequence comes back too because the caller owns one thing ``fit()`` would have done
        for it: :meth:`RecordSequence.on_epoch_end`, the per-epoch reshuffle. Keras calls that
        itself; a hand-written loop that forgets it trains every epoch on the same row order.

        ``output_signature`` is derived from a real first batch rather than guessed — the batch
        shape is :meth:`_to_arrays`'s, and hard-coding it here would put the same fact twice.
        """
        sequence = self._sequence(source, shuffle=shuffle, with_targets=with_targets)
        dataset = tf.data.Dataset.from_generator(
            lambda: (sequence[i] for i in range(len(sequence))),
            output_signature=_output_signature(sequence[0]),
        )
        return (dataset.prefetch(tf.data.AUTOTUNE) if self.prefetch else dataset), sequence

    # ---- the graph steps ----------------------------------------------------- #
    def _train_step_fn(self) -> Any:
        """The forward/backward pair, traced into a graph.

        Built once and reused across epochs: a fresh ``@tf.function`` each epoch would retrace
        the whole graph every time, which is slower than the eager loop it replaced.
        """
        network, optimizer, loss_fn = self.model, self._optimizer, self._loss_fn
        metrics = self._metric_objects["train"]

        @tf.function(jit_compile=True if self.jit_compile else None)
        def train_step(x: Any, y: Any) -> Any:
            with tf.GradientTape() as tape:
                logits = network(x, training=True)
                loss = loss_fn(y, logits)
            variables = network.trainable_variables
            optimizer.apply_gradients(zip(tape.gradient(loss, variables), variables))
            for metric in metrics:
                metric.update_state(y, logits)
            return loss

        return train_step

    def _eval_step_fn(self, split: str = "val") -> Any:
        """The forward pass plus metric update — no tape, and ``training=False``.

        ``split`` selects which metric OBJECTS the traced step updates, so a metric wired to
        ``val_metrics`` is not also computed on the test pass.
        """
        network, loss_fn = self.model, self._loss_fn
        metrics = self._metric_objects[split]

        @tf.function(jit_compile=True if self.jit_compile else None)
        def eval_step(x: Any, y: Any) -> Any:
            logits = network(x, training=False)
            for metric in metrics:
                metric.update_state(y, logits)
            return loss_fn(y, logits)

        return eval_step

    def _predict_step_fn(self) -> Any:
        """Logits only — the graph half of :meth:`predict`."""
        network = self.model

        @tf.function(jit_compile=True if self.jit_compile else None)
        def predict_step(x: Any) -> Any:
            return network(x, training=False)

        return predict_step

    # ---- metric bookkeeping (what `compile()` would have owned) -------------- #
    def _reset_metrics(self, split: str) -> None:
        for metric in self._metric_objects[split]:
            metric.reset_state()

    def _metric_results(self, split: str) -> Dict[str, float]:
        return {metric.name: float(metric.result()) for metric in self._metric_objects[split]}

    def _run_epoch(self, dataset: Any, step: Any, callbacks: Any = None, split: str = "train") -> Dict[str, float]:
        """One pass over ``dataset``, returning ``{"loss": …, **metrics}``.

        Shared by training and evaluation because the only difference between them is which
        traced step function is handed in — the loss averaging and the metric reset are the same
        bookkeeping either way.
        """
        self._reset_metrics(split)
        total, batches = 0.0, 0
        for index, (x, y) in enumerate(dataset):
            total += float(step(x, y))
            batches += 1
            if callbacks is not None:
                callbacks.on_train_batch_end(index, {"loss": total / batches})
        return {"loss": total / batches if batches else float("nan"), **self._metric_results(split)}

    # ---- callbacks + tracking ------------------------------------------------ #
    def _callbacks(self) -> List[Any]:
        """Wired Keras callbacks plus OPTIONAL marainer tracking (a few lines; removable)."""
        callbacks = [flow(cb) for cb in (self.callbacks or [])]
        for callback in callbacks:
            # Keras's file-writing callbacks (`CSVLogger`, `ModelCheckpoint`) open their path
            # without creating its parent, so a perfectly reasonable `./tf_logs/run.csv` dies
            # with FileNotFoundError several minutes into a fit — after the data walk and the
            # model build. Creating the directory here moves that failure to "never".
            target = getattr(callback, "filename", None) or getattr(callback, "filepath", None)
            if isinstance(target, (str, Path)):
                Path(target).expanduser().parent.mkdir(parents=True, exist_ok=True)
        trackers = self._live_trackers()
        if trackers:
            from marainer.keras import keras_tracking_callback

            # `runnable=self` rather than a params dict: marainer derives the hyperparameters
            # from the flowed object graph AND dumps the resolved config, so no backend
            # transcribes its own configuration into a second, diverging answer.
            callbacks.append(keras_tracking_callback(trackers, runnable=self))
        return callbacks

    # ---- the four entry points ---------------------------------------------- #
    @entrypoint("fit", role="trainer", primary=True)
    def fit(self) -> None:
        """The custom loop: epochs, ``tf.GradientTape`` steps, callbacks, then the SavedModel."""
        logger.info(f"sonair tensorflow training: {self.experiment_name} ({self.run_name})")
        if self.train_set is None:
            raise ValueError("TensorFlowClassifier.fit() requires a 'train_set'.")
        train_data, sequence = self._dataset(self.train_set, shuffle=True)
        val_data = self._dataset(self.val_set)[0] if self.val_set is not None else None

        # Build the optimizer's slot variables OUTSIDE the graph. Keras builds them on the first
        # `apply_gradients`, and doing that inside a `tf.function` creates variables during
        # tracing — legal on the first trace, an error on any retrace (a short final batch is
        # enough to trigger one).
        self._optimizer.build(self.model.trainable_variables)
        train_step, eval_step = self._train_step_fn(), self._eval_step_fn()

        callbacks = keras.callbacks.CallbackList(self._callbacks(), add_history=True, model=self.model)
        callbacks.set_params({"epochs": self.max_epochs, "steps": len(sequence), "verbose": 0})
        callbacks.on_train_begin()

        # BEFORE the loop, because `callbacks.on_train_end` below closes the run out (this backend
        # hands its lifecycle to `keras_tracking_callback`, exactly as the Keras one does) — a
        # tracker write sequenced after it arrives at an ENDED run. MLflow tolerates that;
        # `WandbTracker.finish()` drops the run handle, so the artifact would land in a NEW run.
        # It is also the more honest place: the vocabulary comes from the DATA, known since
        # `initialize()`, so a run that dies mid-loop still records what it was training on.
        save_class_names(self._class_names, self._artifact_dir(), self._live_trackers())

        history: Dict[str, List[float]] = {}
        logs: Dict[str, float] = {}
        for epoch in range(self.max_epochs):
            callbacks.on_epoch_begin(epoch)
            logs = self._run_epoch(train_data, train_step, callbacks)
            if val_data is not None:
                logs.update({f"val_{k}": v for k, v in self._run_epoch(val_data, eval_step, split="val").items()})
            callbacks.on_epoch_end(epoch, logs)
            # Keras's `fit()` calls this between epochs; a hand-written loop has to.
            sequence.on_epoch_end()
            for key, value in logs.items():
                history.setdefault(key, []).append(value)
            logger.info(f"epoch {epoch + 1}/{self.max_epochs}: " + ", ".join(f"{k}={v:.4f}" for k, v in logs.items()))
        callbacks.on_train_end(logs)

        self._history = {k: list(v) for k, v in history.items()}
        self._computed_metrics = {k: v[-1] for k, v in history.items()}
        logger.info("Training complete.")
        # Both write LOCAL files and touch no tracker, so they are safe after the run has ended.
        write_metrics_sidecar(self.metrics_sidecar_path, self._computed_metrics, self.experiment_name, self.run_name)
        self._export()

    @entrypoint("evaluate", role="evaluator")
    def evaluate(self) -> None:
        self.test()

    @entrypoint("test", role="evaluator", primary=True)
    def test(self) -> None:
        """Score the held-out split with the same traced step the training loop validates with."""
        if self.test_set is None:
            raise ValueError("TensorFlowClassifier.test() requires a 'test_set'.")
        dataset, _ = self._dataset(self.test_set)
        self._computed_metrics = self._run_epoch(dataset, self._eval_step_fn("test"), split="test")
        logger.info(f"Evaluation complete: {self._computed_metrics}")
        write_metrics_sidecar(self.metrics_sidecar_path, self._computed_metrics, self.experiment_name, self.run_name)

    @entrypoint("predict", role="predictor", primary=True)
    def predict(self) -> None:
        """Stream per-record predictions to the sink, pairing each batch with its metadata.

        Walks ``sequence.batches()`` — the COLLATED record batches — rather than the ``tf.data``
        pipeline, because the sink needs the metadata that lives beside the image and
        ``_to_arrays`` has already dropped it. One walk, so output and metadata cannot desync.
        """
        if self.test_set is None:
            raise ValueError("TensorFlowClassifier.predict() requires a 'test_set'.")
        sink = flow(self.predictions_sink)
        predict_step = self._predict_step_fn()
        sequence = self._sequence(self.test_set, with_targets=False)
        for batch in sequence.batches():
            x = self._to_arrays(batch, with_targets=False)[0]
            logits = np.asarray(predict_step(tf.convert_to_tensor(x)))
            if sink is not None:
                self._write_predictions(logits, batch, cast(PredictionsSink, sink))
        if sink is not None and hasattr(sink, "close"):
            sink.close()
        logger.info("Prediction complete.")

    # ---- predictions -------------------------------------------------------- #
    def _write_predictions(self, logits: np.ndarray, batch: Any, sink: PredictionsSink) -> None:
        """One :class:`~recordstream.ClassificationOutput` per record, paired with its metadata.

        The contract is recordstream's and framework-neutral — ``ClassificationOutput`` is
        generic in its array type, so the SAME sink that consumes the torch backends' output
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

    # ---- the TensorFlow artifact -------------------------------------------- #
    def _export(self) -> None:
        """Write the SavedModel — the one run artifact that is TensorFlow's and not Keras's.

        ``keras.Model.export`` emits a ``tf.saved_model`` directory (serving signatures, frozen
        graph) and is only implemented on the TensorFlow engine, which is the concrete reason a
        run picks this backend over ``sonair keras``.

        Two writers, because the model slot takes two kinds of thing. A `keras.Model` has
        `export`, which builds the serving signature for you. A raw `tf.Module` — which is what
        `sonair.convnet.tensorflow_convnet.TensorFlowConvNet` is — does NOT, so it goes through
        `tf.saved_model.save` with an explicitly traced signature. Branching on the capability
        rather than on the class keeps anything else callable working too.
        """
        if not self.export_dir:
            return
        path = Path(self.export_dir).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        if hasattr(self.model, "export"):
            self.model.export(str(path))
        else:
            # `tf.saved_model.save` needs a concrete function to serialize, and a bare
            # `tf.Module.__call__` is untraced Python. Tracing it here at the batch shape the
            # run actually used is what makes the export loadable by `TensorFlowModel`.
            spec = tf.TensorSpec(shape=(None, *self._input_shape()), dtype=tf.float32)
            call = tf.function(lambda x: self.model(x, training=False)).get_concrete_function(spec)
            tf.saved_model.save(self.model, str(path), signatures={"serving_default": call})
        logger.info(f"Exported SavedModel to {path}")

    def _input_shape(self) -> Tuple[int, ...]:
        """The per-record input shape, for the export signature.

        Read off the model when it declares one (every sonair model wrapper does), else off a
        real batch — which always works but costs one dataset read.
        """
        declared = getattr(self.model, "input_shape", None)
        if declared:
            return tuple(declared)
        source = self.test_set or self.val_set or self.train_set
        sequence = self._sequence(source, with_targets=False)
        return tuple(np.asarray(sequence[0][0]).shape[1:])

    # ---- artifacts + tracking ----------------------------------------------- #
    def _artifact_dir(self) -> Optional[str]:
        """Where run artifacts land. There is no checkpoint-callback convention to read here, so
        this follows the sidecar's directory when one is configured."""
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
        """The live trained model (post-run ``@output``). Raises while still deferred.

        Whatever was wired, unwrapped by nobody — which is also what the torch backends return.
        A ``KerasModel`` IS a ``keras.Model`` AND carries ``num_classes``, which an evaluation
        run reads straight off its model slot, so this output docks into a downstream classifier.
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
    """Row-wise softmax. The model emits raw logits (no head activation), like every other head."""
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return cast(np.ndarray, exp / exp.sum(axis=-1, keepdims=True))


def _output_signature(batch: Any) -> Tuple[Any, ...]:
    """``tf.TensorSpec``s matching a real batch, with the row count left dynamic.

    ``from_generator`` needs the element spec up front. Deriving it from batch zero keeps the
    batch SHAPE defined in exactly one place — :meth:`TensorFlowClassifier._to_arrays` — instead
    of restating "float32 [N, H, W, C] and int64 [N]" here where it could drift.
    """
    arrays = [np.asarray(item) for item in batch]
    return tuple(tf.TensorSpec(shape=(None, *a.shape[1:]), dtype=tf.as_dtype(a.dtype)) for a in arrays)


def _as_optimizer(value: Any) -> Any:
    """A live ``keras.optimizers.Optimizer`` from an instance, a class, or a string alias.

    The bare-CLASS branch is tested BEFORE ``flow()`` rather than after, which is the one place
    in this backend where the order matters: ``flow()`` INSTANTIATES a bare type on the way past
    — and for a confluid-registered class it does so through ``materialize()``, where sibling
    config keys broadcast into the constructor. Classifying it first keeps a class meaning
    exactly "call it with no arguments", which is what the annotation promises.
    """
    if isinstance(value, type) and issubclass(value, keras.optimizers.Optimizer):
        return value()
    built = flow(value)
    if isinstance(built, keras.optimizers.Optimizer):
        return built
    return keras.optimizers.get(built)


def _as_metric(value: Any, multilabel: bool) -> Any:
    """A live ``keras.metrics.Metric`` from an instance, a class, or a string alias.

    The ``"accuracy"`` / ``"acc"`` alias is special-cased on purpose: it is the one name
    ``keras.Model.compile`` resolves CONTEXTUALLY (by loss and target shape) and
    ``keras.metrics.get`` does not — ``get("accuracy")`` hands back the elementwise-equality
    function, which scores 0.0 against a raw-logits head. Mapping it by ``multilabel`` here is
    the same choice ``compile()`` would have made, so a config that writes the alias behaves as
    its author expects instead of reporting a silent zero.

    The bare-CLASS branch precedes ``flow()`` for the reason :func:`_as_optimizer` spells out.
    """
    if isinstance(value, type) and issubclass(value, keras.metrics.Metric):
        return value()
    built = flow(value)
    if isinstance(built, keras.metrics.Metric):
        return built
    if isinstance(built, str) and built in ("accuracy", "acc"):
        return keras.metrics.BinaryAccuracy(threshold=0.0) if multilabel else keras.metrics.SparseCategoricalAccuracy()
    resolved = keras.metrics.get(built)
    if isinstance(resolved, keras.metrics.Metric):
        return resolved
    return keras.metrics.MeanMetricWrapper(resolved, name=getattr(resolved, "__name__", "metric"))
