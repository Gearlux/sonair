"""``TensorFlowClassifier`` — the same classification run as a NATIVE TensorFlow training loop.

This backend exists to answer a question the project had left open in ``TASKS.md``: does
``sonair tensorflow train`` need a runnable of its own at all, when ``KERAS_BACKEND=tensorflow
sonair keras train <cfg>`` already executes the identical config on TensorFlow? The answer is
*only for what Keras 3 does not expose* — and that is exactly what this class contains and
nothing else:

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

Everything ELSE is inherited from :class:`~sonair.keras_classifier.KerasClassifier`, because in
TensorFlow >= 2.16 ``tf.keras`` **is** Keras 3: the model wrapper (``KerasModel``), the losses
and metrics (``sonair.keras_classification``), the label handling, the batch shape
(``_to_arrays``), the prediction contract and the run artifacts are the same objects, not
look-alikes. That is why this class declares ``framework="keras"`` — the framework axis names
the API a class can be WIRED TO, not the tensor runtime underneath (``docs/architecture.md``
§2) — and why there is no ``tensorflow_classification.py``: its loss set would have been the
Keras one under a second name.

What that leaves is roughly a hundred lines of loop. That ratio is the demo, exactly as it was
for the Keras backend.

Minimal flat YAML — note it differs from the Keras block by ONE key, the runnable's class::

    runnable: !class:sonair.TensorFlowClassifier
      model: !lazy:sonair.KerasModel { model_name: MobileNetV3Small }
      export_dir: ./exported/mnist
    train_set: !class:recordstream.Stream { source: ..., ops: !ref:preprocess }
    batch_size: 32
    max_epochs: 3

Run it with ``sonair tensorflow train cfg.yaml``, which sets ``KERAS_BACKEND=tensorflow`` before
anything imports keras — see :mod:`sonair._tensorflow` for why that ordering is load-bearing.
"""

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, cast

import numpy as np
from confluid import Fluid, Lazy, Mandatory, configurable, flow
from loggair import get_logger
from marainer.artifacts import save_class_names, write_metrics_sidecar
from recordstream import LabelMap, PredictionsSink, RecordSource
from recordstream.runnable import entrypoint

# keras comes from `sonair._tensorflow`, NOT `sonair._keras`: that module pins
# KERAS_BACKEND=tensorflow first, and isort would sort a `sonair._keras` line ABOVE it — loading
# keras on whichever engine happened to be found first. Pinned by
# `tests/test_tensorflow_classifier.py`. This import must also precede `sonair.keras_classifier`
# (which reaches keras too); isort keeps that order, since `_tensorflow` < `keras_classifier`.
from sonair._tensorflow import keras, keras_backend, tf
from sonair.keras_classifier import ClassifierTask, KerasClassifier, KerasLoss, KerasMetric, KerasOptimizer, _built

logger = get_logger(__name__)

__all__ = ["TensorFlowClassifier"]


@configurable(task="classification", role="trainer", framework="keras")
class TensorFlowClassifier(KerasClassifier):
    """Train, evaluate or predict an image classifier with a native TensorFlow loop.

    Every constructor knob of :class:`~sonair.keras_classifier.KerasClassifier` means the same
    thing here and is restated rather than swallowed by ``**kwargs``, because static
    introspection (``to_pydantic`` -> form specs / MCP schemas / editor widgets) reads the
    signature. The three at the end are what this backend adds.

    Args:
        model: The classification model — wire ``!lazy:sonair.KerasModel`` so ``initialize()``
            can inject the dataset-derived ``num_classes``. ``tf.keras`` IS Keras 3, so the
            Keras backend's wrapper is the wrapper here too.
        train_set: Training source (``task="fit"``).
        test_set: Evaluation / prediction source.
        num_classes: Class count. Omitted, it is derived from the data or the label map.
        task: Which entry point ``run()`` dispatches to.
        max_epochs: Training epochs.
        batch_size: Rows per batch.
        logger: ``keras.callbacks.Callback`` instances. They are driven by hand through a
            ``keras.callbacks.CallbackList``, so a ``CSVLogger`` — or marainer's Keras tracking
            callback — behaves exactly as it would under ``fit()``.
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
        export_dir: Write a ``tf.saved_model`` directory here after a successful ``fit``.
            ``None`` (the default) exports nothing.
        jit_compile: Trace the train/eval steps with XLA. A real speedup on fixed shapes and a
            real compile cost on the first batch, so it is opt-in.
        prefetch: Append ``.prefetch(tf.data.AUTOTUNE)`` to every ``tf.data`` pipeline, which
            overlaps the record walk with the device step. Turn it off only to measure.
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
        optimizer: KerasOptimizer = "adam",
        loss: Optional[KerasLoss] = None,
        metrics: Optional[Sequence[KerasMetric]] = None,
        export_dir: Optional[str] = None,
        jit_compile: bool = False,
        prefetch: bool = True,
    ) -> None:
        super().__init__(
            model=model,
            train_set=train_set,
            test_set=test_set,
            num_classes=num_classes,
            task=task,
            max_epochs=max_epochs,
            batch_size=batch_size,
            logger=logger,
            experiment_name=experiment_name,
            run_name=run_name,
            input_key=input_key,
            target_key=target_key,
            label_map=label_map,
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
        )
        self.export_dir = export_dir
        self.jit_compile = jit_compile
        self.prefetch = prefetch
        self._optimizer: Any = None
        self._loss_fn: Any = None
        self._metric_objects: List[Any] = []

    # ---- setup -------------------------------------------------------------- #
    def initialize(self) -> None:
        """Refuse to run on the wrong engine, then set up exactly as the Keras backend does."""
        self._require_tensorflow_backend()
        super().initialize()

    @staticmethod
    def _require_tensorflow_backend() -> None:
        """Fail loudly when keras loaded some OTHER engine — the one way this backend goes wrong.

        ``tf.GradientTape`` cannot differentiate a torch-backed ``keras.Model``, and the failure
        deep inside the tape says nothing useful. This is not hypothetical: ``recordstream.keras``
        defaults ``KERAS_BACKEND`` to the first backend INSTALLED, in the order torch -> tensorflow
        -> jax, so any machine with both ``sonair[lightning]`` and ``sonair[tensorflow]`` lands on
        torch unless something set the variable first (:mod:`sonair._tensorflow` does; ``sonair
        tensorflow <verb>`` does it earlier still).
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

        The name is the parent's (``initialize()`` calls it) but the body is the divergence: a
        compiled model owns its own training step, which is precisely what this backend replaces.
        The slots are flowed at USE through ``_built`` for the same reason the parent does it —
        a nested ``!class:`` in a CONSTRUCTOR slot arrives as a deferred marker.
        """
        self._resolve_task_defaults()
        if isinstance(self.model, Fluid) or hasattr(self.model, "solidify"):
            self.model = flow(self.model, num_classes=self.num_classes)
        self._network = getattr(self.model, "model", self.model)
        self._optimizer = _as_optimizer(self.optimizer)
        self._loss_fn = _built(self.loss)
        self._metric_objects = [_as_metric(m, multilabel=self.multilabel) for m in (self.metrics or [])]

    # ---- tf.data ------------------------------------------------------------- #
    def _dataset(self, source: Any, shuffle: bool = False, with_targets: bool = True) -> Tuple[Any, Any]:
        """A ``tf.data.Dataset`` over recordstream batches, plus the sequence that feeds it.

        The sequence comes back too because the caller owns one thing ``fit()`` would have done
        for it: :meth:`RecordSequence.on_epoch_end`, the per-epoch reshuffle. Keras calls that
        itself; a hand-written loop that forgets it trains every epoch on the same row order.

        ``output_signature`` is derived from a real first batch rather than guessed — the batch
        shape is the classifier's (``_to_arrays``), and hard-coding it here would put the same
        fact in two places.
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
        network, optimizer, loss_fn, metrics = self._network, self._optimizer, self._loss_fn, self._metric_objects

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

    def _eval_step_fn(self) -> Any:
        """The forward pass plus metric update — no tape, and ``training=False``."""
        network, loss_fn, metrics = self._network, self._loss_fn, self._metric_objects

        @tf.function(jit_compile=True if self.jit_compile else None)
        def eval_step(x: Any, y: Any) -> Any:
            logits = network(x, training=False)
            for metric in metrics:
                metric.update_state(y, logits)
            return loss_fn(y, logits)

        return eval_step

    def _predict_step_fn(self) -> Any:
        """Logits only — the graph half of :meth:`predict`."""
        network = self._network

        @tf.function(jit_compile=True if self.jit_compile else None)
        def predict_step(x: Any) -> Any:
            return network(x, training=False)

        return predict_step

    # ---- metric bookkeeping (what `compile()` would have owned) -------------- #
    def _reset_metrics(self) -> None:
        for metric in self._metric_objects:
            metric.reset_state()

    def _metric_results(self) -> Dict[str, float]:
        return {metric.name: float(metric.result()) for metric in self._metric_objects}

    def _run_epoch(self, dataset: Any, step: Any, callbacks: Any = None) -> Dict[str, float]:
        """One pass over ``dataset``, returning ``{"loss": …, **metrics}``.

        Shared by training and evaluation because the only difference between them is which
        traced step function is handed in — the loss averaging and the metric reset are the same
        bookkeeping either way.
        """
        self._reset_metrics()
        total, batches = 0.0, 0
        for index, (x, y) in enumerate(dataset):
            total += float(step(x, y))
            batches += 1
            if callbacks is not None:
                callbacks.on_train_batch_end(index, {"loss": total / batches})
        return {"loss": total / batches if batches else float("nan"), **self._metric_results()}

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
        self._optimizer.build(self._network.trainable_variables)
        train_step, eval_step = self._train_step_fn(), self._eval_step_fn()

        callbacks = keras.callbacks.CallbackList(self._callbacks(), add_history=True, model=self._network)
        callbacks.set_params({"epochs": self.max_epochs, "steps": len(sequence), "verbose": 0})
        callbacks.on_train_begin()

        history: Dict[str, List[float]] = {}
        logs: Dict[str, float] = {}
        for epoch in range(self.max_epochs):
            callbacks.on_epoch_begin(epoch)
            logs = self._run_epoch(train_data, train_step, callbacks)
            if val_data is not None:
                logs.update({f"val_{k}": v for k, v in self._run_epoch(val_data, eval_step).items()})
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
        save_class_names(self._class_names, self._artifact_dir(), self._live_trackers())
        write_metrics_sidecar(self.metrics_sidecar_path, self._computed_metrics, self.experiment_name, self.run_name)
        self._export()

    @entrypoint("test", role="evaluator", primary=True)
    def test(self) -> None:
        """Score the held-out split with the same traced step the training loop validates with."""
        if self.test_set is None:
            raise ValueError("TensorFlowClassifier.test() requires a 'test_set'.")
        dataset, _ = self._dataset(self.test_set)
        self._computed_metrics = self._run_epoch(dataset, self._eval_step_fn())
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
        sink = flow(self.predictions_sink) if isinstance(self.predictions_sink, Fluid) else self.predictions_sink
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

    # ---- the TensorFlow artifact -------------------------------------------- #
    def _export(self) -> None:
        """Write the SavedModel — the one run artifact that is TensorFlow's and not Keras's.

        ``keras.Model.export`` emits a ``tf.saved_model`` directory (serving signatures, frozen
        graph) and is only implemented on the TensorFlow engine, which is the concrete reason a
        run picks this backend over ``sonair keras``.
        """
        if not self.export_dir:
            return
        path = Path(self.export_dir).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        self._network.export(str(path))
        logger.info(f"Exported SavedModel to {path}")


def _output_signature(batch: Any) -> Tuple[Any, ...]:
    """``tf.TensorSpec``s matching a real batch, with the row count left dynamic.

    ``from_generator`` needs the element spec up front. Deriving it from batch zero keeps the
    batch SHAPE defined in exactly one place — ``KerasClassifier._to_arrays`` — instead of
    restating "float32 [N, H, W, C] and int64 [N]" here where it could drift.
    """
    arrays = [np.asarray(item) for item in batch]
    return tuple(tf.TensorSpec(shape=(None, *a.shape[1:]), dtype=tf.as_dtype(a.dtype)) for a in arrays)


def _as_optimizer(value: Any) -> Any:
    """A live ``keras.optimizers.Optimizer`` from an instance, a class, or a string alias."""
    built = _built(value)
    if isinstance(built, keras.optimizers.Optimizer):
        return built
    if isinstance(built, type) and issubclass(built, keras.optimizers.Optimizer):
        return built()
    return keras.optimizers.get(built)


def _as_metric(value: Any, multilabel: bool) -> Any:
    """A live ``keras.metrics.Metric`` from an instance, a class, or a string alias.

    The ``"accuracy"`` / ``"acc"`` alias is special-cased on purpose: it is the one name
    ``keras.Model.compile`` resolves CONTEXTUALLY (by loss and target shape) and
    ``keras.metrics.get`` does not — ``get("accuracy")`` hands back the elementwise-equality
    function, which scores 0.0 against a raw-logits head. Mapping it by ``multilabel`` here is
    the same choice ``compile()`` would have made, so a config that writes the alias behaves as
    its author expects instead of reporting a silent zero.
    """
    built = _built(value)
    if isinstance(built, keras.metrics.Metric):
        return built
    if isinstance(built, type) and issubclass(built, keras.metrics.Metric):
        return built()
    if isinstance(built, str) and built in ("accuracy", "acc"):
        return keras.metrics.BinaryAccuracy(threshold=0.0) if multilabel else keras.metrics.SparseCategoricalAccuracy()
    resolved = keras.metrics.get(built)
    if isinstance(resolved, keras.metrics.Metric):
        return resolved
    return keras.metrics.MeanMetricWrapper(resolved, name=getattr(resolved, "__name__", "metric"))
