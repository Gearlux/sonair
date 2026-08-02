"""``FastaiClassifier`` — the same classification run, driven by a fastai ``Learner``.

The fourth backend, and the one that tests the claim from the other direction. Keras and
TensorFlow swap the whole stack; fastai swaps only the *loop*. It runs on torch, so this backend
shares the reference backend's model wrapper (``TimmModel``), its tensors, its collate and its
``[N, 3, H, W]`` batch shape — literally the same objects, not equivalents. What changes is who
owns the epoch: ``Learner.fit_one_cycle`` instead of ``pytorch_lightning.Trainer.fit``.

So the interesting number here is not how little was written but how little was *replaced*:

* **Data** comes from the same deferred ``train_loader`` / ``val_loader`` / ``test_loader``
  slots the Lightning backend declares — ``LazyClass(DataLoader, …)`` baked in ``__init__``
  with the ergonomic ``batch_size`` / ``num_workers`` knobs, flowed at run time with the
  ``dataset=`` that does not exist yet. fastai takes plain PyTorch loaders directly
  (``DataLoaders(train_dl, valid_dl)``), so the LOADER needs no adaptation at all; the one
  substitution is the ``collate_fn``, where :meth:`FastaiClassifier._to_tuple` stands in for
  ``collate_records`` because fastai unpacks a batch positionally. That method is the twin of
  the Lightning backend's ``_targets`` and the Keras backend's ``_to_arrays``, and it is the
  whole of what this backend says about batches.
* **The model** is ``sonair.TimmModel``, unchanged and un-wrapped. A fastai config's `model:`
  block is byte-identical to a Lightning one.
* **Losses** are ``fastai.losses`` classes, registered ``framework="torch"`` in
  :mod:`sonair.fastai_classification` because that is what they consume — a Lightning run can
  take ``CrossEntropyLossFlat`` just as well (it is a callable over torch tensors wrapping a
  ``torch.nn.CrossEntropyLoss``, not a class of its own).
* **Labels, class names, artifacts, the prediction contract** come from recordstream and
  marainer exactly as for every other backend.

**Metrics are named as STRINGS here, and that is a fastai fact, not a sonair shortcut.** Nearly
every fastai classification metric is a plain *function* (``accuracy``, ``error_rate``,
``top_k_accuracy``) or a factory that returns one (``F1Score()``, ``RocAuc()``), so there is no
class for a ``!class:`` tag to name and nothing for confluid's registry to tag. The ``metrics``
slot therefore also accepts a string, resolved against ``fastai.metrics`` — the same ergonomic
the Keras backend gets from ``"accuracy"``::

    metrics: [accuracy, top_k_accuracy]

Minimal flat YAML — compare it to the top-level (Lightning) block of the shipped example and
note that only the runnable's class differs::

    runnable: !class:sonair.FastaiClassifier
      model: !lazy:sonair.TimmModel { model_name: efficientvit_b0, pretrained: true }
    train_set: !class:recordstream.Stream { source: ..., ops: !ref:preprocess }
    batch_size: 32
    max_epochs: 3
"""

from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Sequence, Tuple, Union, cast

import torch
from confluid import Fluid, Lazy, LazyClass, Mandatory, configurable, flow, output
from fastai import metrics as fastai_metrics
from fastai import optimizer as fastai_optimizer

# Imported for its SIDE EFFECT: `Learner.fit_one_cycle` is a `@patch` this module installs, and
# fastai's `@patch` returns None — so `from fastai.callback.schedule import fit_one_cycle` binds
# None, and without the import `learn.fit_one_cycle` falls through fastcore's attribute
# delegation to the MODEL and raises `'TimmModel' object has no attribute 'fit_one_cycle'`
# (hit for real). Do not "tidy" this away.
from fastai.callback import schedule  # noqa: F401
from fastai.data.core import DataLoaders
from fastai.learner import Learner
from fastai.losses import BCEWithLogitsLossFlat, CrossEntropyLossFlat
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
    batch_tensor,
    class_names,
    classification_output,
    collate_records,
    ensure_record_dataset,
    is_class_id,
    iter_key,
    multi_hot,
)
from recordstream.runnable import ProgressReporting, TorchRunner, entrypoint, run_entrypoint
from torch.utils.data import DataLoader

from sonair.metric_slots import merge_metric_specs

logger = get_logger(__name__)

#: Which entry point :meth:`FastaiClassifier.run` dispatches to — the same vocabulary every
#: other backend uses, because the verb is a property of the TASK, not of the engine.
ClassifierTask = Literal["fit", "evaluate", "test", "predict"]

#: What the ``metrics`` slot accepts: a fastai metric callable, or its NAME in ``fastai.metrics``.
#: Spelled out rather than left as ``Any`` so ``to_pydantic`` (form specs / MCP schemas / editor
#: widgets) has a real type to build from.
FastaiMetric = Union[str, Callable[..., Any]]

#: A fastai optimizer FUNCTION (``fastai.optimizer.Adam`` and friends) or its name.
FastaiOptimizer = Union[str, Callable[..., Any]]

__all__ = ["ClassifierTask", "FastaiClassifier"]


def _first_target(source: Any, target_key: str) -> Any:
    """The first non-``None`` target in ``source`` — one peek is enough to know the label kind."""
    for target in iter_key(source, target_key):
        if target is not None:
            return target
    return None


def _run_prefix(task: str) -> str:
    return {"fit": "train", "predict": "predict"}.get(task, "evaluate")


@configurable(task="classification", role="trainer", framework="torch")
class FastaiClassifier(TorchRunner, ProgressReporting):
    """Train, evaluate or predict an image classifier with a fastai ``Learner``.

    Args:
        model: The classification model — wire ``!lazy:sonair.TimmModel`` so ``initialize()``
            can inject the dataset-derived ``num_classes``. fastai is torch, so this is the
            reference backend's wrapper, unchanged.
        train_set: Training source (``task="fit"``).
        val_set: Validation source. fastai's ``Learner`` always validates, so one is used
            whether or not it is wired — see :meth:`_dataloaders`.
        test_set: Evaluation / prediction source.
        num_classes: Class count. Omitted, it is derived from the data or the label map.
        task: Which entry point ``run()`` dispatches to.
        max_epochs: Training epochs.
        batch_size: Rows per batch.
        num_workers: DataLoader worker processes.
        trackers: marainer trackers — one or a list. This run's ONLY output wiring:
            metrics, provenance and the run manifest all go to them.
        callbacks: ``fastai.callback.core.Callback`` instances (``CSVLogger``,
            ``SaveModelCallback`` …), appended to the ``Learner``'s own set.
        experiment_name: Run identity, recorded in artifacts and tracking.
        run_name: Run identity; a random ``adjective_noun`` when omitted.
        input_key: Record key holding the image.
        target_key: Record key holding the label.
        label_map: Pin the name-to-id encoding instead of fitting it from the train targets.
        learning_rate: Peak LR for :meth:`fit` (``lr_max`` under one-cycle, ``lr`` otherwise).
        one_cycle: Use ``fit_one_cycle`` (fastai's default schedule) rather than a flat
            ``fit``. This is fastai's headline training behaviour, so it is on by default.
        weight_decay: fastai's ``wd``. ``None`` leaves the ``Learner`` default.
        opt_func: A ``fastai.optimizer`` function or its name (``"Adam"``, ``"SGD"``,
            ``"ranger"``, …).
        loss: A fastai / torch loss module. ``None`` picks by target shape, like every other
            backend's sentinel.
        metrics: fastai metrics — callables, or names resolved against ``fastai.metrics``.
            ``None`` picks ``accuracy`` / ``accuracy_multi`` by target shape.
        val_metrics / test_metrics: That split's OWN metrics, merged over ``metrics``. There is
            no ``train_metrics``: a fastai ``Learner`` computes its metrics on the validation
            pass only.
    """

    def __init__(
        self,
        model: Mandatory[Lazy[torch.nn.Module]],
        train_set: Optional[Lazy[RecordSource]] = None,
        val_set: Optional[Lazy[RecordSource]] = None,
        test_set: Optional[Lazy[RecordSource]] = None,
        num_classes: Optional[int] = None,
        task: ClassifierTask = "fit",
        max_epochs: int = 10,
        batch_size: int = 32,
        num_workers: int = 0,
        trackers: Optional[Union[Lazy[Tracker], Iterable[Lazy[Tracker]]]] = None,
        callbacks: Optional[Iterable[Any]] = None,
        experiment_name: str = "sonair_classification",
        run_name: Optional[str] = None,
        input_key: str = "image",
        target_key: str = "class",
        label_map: Optional[LabelMap] = None,
        learning_rate: float = 1e-3,
        one_cycle: bool = True,
        weight_decay: Optional[float] = None,
        opt_func: FastaiOptimizer = "Adam",
        loss: Optional[Any] = None,
        metrics: Optional[Sequence[FastaiMetric]] = None,
        val_metrics: Optional[Sequence[FastaiMetric]] = None,
        test_metrics: Optional[Sequence[FastaiMetric]] = None,
    ) -> None:
        super().__init__()
        # `torch.nn.Module`, not `Lazy[torch.nn.Module]`: the deferred-ness is declared once, on
        # the ctor param, which is what `lazy_param_names` / `to_pydantic` read. A union here
        # would only stop every method calling through the slot (`.eval()`, `.parameters()`).
        # The cast is the whole of it: the ctor accepts a deferred marker, while the ATTRIBUTE's
        # useful type is what it holds once `initialize()` has flowed it.
        self.model: torch.nn.Module = cast(torch.nn.Module, model)
        self.train_set: Optional[Lazy[RecordSource]] = train_set
        self.val_set: Optional[Lazy[RecordSource]] = val_set
        self.test_set: Optional[Lazy[RecordSource]] = test_set
        self.task: ClassifierTask = task
        self.num_classes = int(num_classes) if num_classes is not None else None
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.num_workers = num_workers
        # Internal only: this backend's own tracking callback is appended at run time.
        # Stored VERBATIM like `trackers` (lazy-init): entries may be deferred `Fluid`
        # markers, and they are flowed where they are used.
        self.callbacks: Optional[Iterable[Any]] = callbacks
        self.experiment_name = experiment_name
        self.run_name = run_name if run_name is not None else generate_run_name(prefix=_run_prefix(task))
        self.input_key = str(input_key)
        self.target_key = str(target_key)
        self.label_map = label_map
        self.learning_rate = learning_rate
        self.one_cycle = one_cycle
        self.weight_decay = weight_decay
        self.opt_func: FastaiOptimizer = opt_func
        # Deferred loader stubs, baked here exactly as the Lightning backend bakes them, and
        # for the same payoff: `batch_size`/`num_workers` stay ergonomic flat-config knobs while
        # the LOADER itself remains a post-construction slot a config can replace wholesale
        # (`pin_memory`, `drop_last`, a sampler — none of which was reachable while this backend
        # built its loaders inline). `run()` flows each with the `dataset=` that does not exist
        # yet, which is the whole reason they are deferred rather than built now.
        #
        # The ONE divergence from the Lightning stubs is the collate, and it is this backend's
        # single genuine adaptation: fastai unpacks a batch POSITIONALLY, so `_to_tuple` sits
        # where `collate_records` sits there. `predict()` flows `test_loader` with
        # `collate_fn=collate_records` to get the record dict back — see the note there.
        loader_kw = dict(
            collate_fn=self._to_tuple,
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=num_workers != 0,
        )
        self.train_loader: Lazy[DataLoader[Any]] = LazyClass(DataLoader, shuffle=True, **loader_kw)
        self.val_loader: Lazy[DataLoader[Any]] = LazyClass(DataLoader, shuffle=False, **loader_kw)
        self.test_loader: Lazy[DataLoader[Any]] = LazyClass(DataLoader, shuffle=False, **loader_kw)
        # Sentinels, for the same reason every other backend uses them: the right loss and
        # metrics depend on whether the targets are single- or multi-label, which is only known
        # once `initialize()` has peeked the data. A live default here could not be told apart
        # from a user's explicit choice.
        self.loss: Optional[Any] = loss
        self.metrics: Optional[Sequence[FastaiMetric]] = metrics
        # Per-split ADDITIONS, merged over `metrics` by the shared rule. There is no
        # `train_metrics`, and that is a fastai fact rather than an omission: a `Learner`
        # computes its metrics on the VALIDATION pass only, so a train-only metric would have
        # nowhere to be computed.
        self.val_metrics: Optional[Sequence[FastaiMetric]] = val_metrics
        self.test_metrics: Optional[Sequence[FastaiMetric]] = test_metrics
        self.multilabel: bool = False
        self.predictions_sink: Optional[Lazy[PredictionsSink]] = None
        self.metrics_sidecar_path: Optional[str] = None
        # Stored VERBATIM (lazy-init mandate: the constructor does no work). It may hold a
        # single tracker, a list, or deferred `Fluid` markers; `_live_trackers()` normalizes
        # and flows it at run time.
        self.trackers: Optional[Union[Lazy[Tracker], Iterable[Lazy[Tracker]]]] = trackers
        self._class_names: Optional[List[str]] = None
        self._computed_metrics: Dict[str, float] = {}
        self._learner: Optional[Learner] = None

    @property
    def __needs_autograd__(self) -> bool:  # type: ignore[override]  # dynamic per-task override
        """Only ``fit`` optimizes — an evaluate/predict run is happy under inference mode."""
        return self.task == "fit"

    # ---- run dispatch ------------------------------------------------------- #
    def run(self) -> None:
        self.initialize()
        # The `@entrypoint` markers below ARE the dispatch table (recordstream docs/runnable.md).
        run_entrypoint(self, self.task)

    # ---- setup -------------------------------------------------------------- #
    def initialize(self) -> None:
        """Resolve labels and ``num_classes``, then build the model at that width."""
        if self.task == "fit":
            self._initialize_fit()
        else:
            self._initialize_eval()
        self._resolve_task_defaults()
        self.model = flow(self.model, num_classes=self.num_classes)
        # Checked HERE, at the flow site, rather than behind an accessor at each use: a wrong
        # model should fail during setup with its own name in the message, not several minutes
        # into a fit inside `Learner`. This is a validation, not a guard around `flow`.
        if not isinstance(self.model, torch.nn.Module):
            raise TypeError(f"FastaiClassifier needs a torch nn.Module, got {type(self.model).__name__}.")

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

        The ``*Flat`` losses are fastai's own: they flatten the trailing dimensions before
        delegating to the torch loss underneath, which is what lets one loss serve a classifier,
        a segmenter and a language model. For this task they behave exactly as
        ``nn.CrossEntropyLoss`` / ``nn.BCEWithLogitsLoss`` do — logits in, no head activation —
        so the choice matches the Lightning backend's sentinel pair rather than diverging from it.
        """
        if self.loss is None:
            self.loss = BCEWithLogitsLossFlat() if self.multilabel else CrossEntropyLossFlat()
        if self.metrics is None:
            self.metrics = ["accuracy_multi"] if self.multilabel else ["accuracy"]

    # ---- data --------------------------------------------------------------- #
    def _to_tuple(self, records: Sequence[Any], with_targets: bool = True) -> Tuple[torch.Tensor, ...]:
        """One list of records → the ``(x, y)`` tuple fastai consumes.

        The fastai twin of the Lightning backend's ``_targets`` and the Keras backend's
        ``_to_arrays``, and short for the same reason: recordstream does the work.
        ``collate_records`` stacks the already-tensorized image to ``[N, 3, H, W]``, and the
        target is a two-branch CHOICE, not a shaper — ``multi_hot`` for a multi-label target,
        else int64 ids. int64 is named EXPLICITLY because a dataset yielding int32 labels is
        legal and the cross-entropy family rejects it (*"expected target dtype to be Long or
        Byte, but got Int"*).
        """
        batch = collate_records(list(records))
        x = batch_tensor(batch, self.input_key)
        if not with_targets:
            return (x,)
        if self.multilabel:
            if not self.num_classes:
                raise RuntimeError("FastaiClassifier: a multi-label target needs num_classes.")
            return x, torch.as_tensor(multi_hot(batch, self.target_key, self.num_classes), dtype=torch.float32)
        return x, batch_tensor(batch, self.target_key, dtype=torch.int64)

    def _dataloaders(self) -> DataLoaders:
        """fastai's ``DataLoaders`` — a train loader and a validation loader, always both.

        ``Learner`` validates at the end of every epoch and its ``Recorder`` reports from that
        pass, so unlike ``pytorch_lightning.Trainer`` it has no "no validation" mode. When no
        ``val_set`` is wired the training data is reused, which keeps a run working but makes
        the reported metrics training metrics — worth saying out loud, hence the warning.

        The loaders come from the deferred ``train_loader`` / ``val_loader`` slots, flowed with
        the ``dataset=`` that did not exist at construction — the same two lines the Lightning
        backend's ``fit()`` writes.
        """
        train_ds = ensure_record_dataset(self.train_set)
        train = flow(self.train_loader, dataset=train_ds)
        if self.val_set is not None:
            return DataLoaders(train, flow(self.val_loader, dataset=ensure_record_dataset(self.val_set)))
        logger.warning(
            "No 'val_set' wired: fastai's Learner always validates, so the TRAINING data is being "
            "used for validation and the reported metrics measure fit, not generalization. Wire a "
            "'val_set' to get a real held-out score."
        )
        return DataLoaders(train, flow(self.val_loader, dataset=train_ds))

    # ---- the learner -------------------------------------------------------- #
    def _build_learner(self, dataloaders: DataLoaders, split: str = "val") -> Learner:
        """Assemble the ``Learner``: model, loss, metrics, callbacks, artifact path.

        ``split`` picks which metric slot is merged over the shared ``metrics`` — a `Learner`
        holds ONE list and computes it on the validation pass, so the split is decided by which
        entry point is building it (``fit`` validates, ``test`` scores the held-out split)
        rather than by anything inside the Learner.
        """
        path = self._artifact_dir()
        learner = Learner(
            dataloaders,
            self.model,
            loss_func=flow(self.loss),
            opt_func=_as_opt_func(self.opt_func),
            lr=self.learning_rate,
            metrics=[
                _as_metric(m) for m in (merge_metric_specs(self.metrics, getattr(self, f"{split}_metrics")) or [])
            ],
            cbs=self._callbacks(),
            path=Path(path) if path else None,
            **({"wd": self.weight_decay} if self.weight_decay is not None else {}),
        )
        self._learner = learner
        return learner

    def _callbacks(self) -> List[Any]:
        """The wired fastai callbacks, with any file destination's parent directory created.

        Same defensive step the Keras backend takes, for the same reason: fastai's ``CSVLogger``
        opens ``learn.path/fname`` without creating it, so a perfectly reasonable
        ``./fastai_logs/run.csv`` would die *after* the data walk and the model build.
        """
        callbacks = [flow(cb) for cb in (self.callbacks or [])]
        base = Path(self._artifact_dir() or ".")
        for callback in callbacks:
            fname = getattr(callback, "fname", None)
            if isinstance(fname, (str, Path)):
                (base / fname).expanduser().parent.mkdir(parents=True, exist_ok=True)
        return callbacks

    # ---- the four entry points ---------------------------------------------- #
    @entrypoint("fit", role="trainer", primary=True)
    def fit(self) -> None:
        logger.info(f"sonair fastai training: {self.experiment_name} ({self.run_name})")
        if self.train_set is None:
            raise ValueError("FastaiClassifier.fit() requires a 'train_set'.")
        # Provenance through marainer's plain FUNCTIONS — `runnable=self` so the
        # hyperparameters and the resolved config come from marainer's ONE derivation
        # rather than a dict this backend transcribes for itself.
        trackers = self._live_trackers()
        manifest = log_run_context(trackers, runnable=self)
        learner = self._build_learner(self._dataloaders())
        if self.one_cycle:
            learner.fit_one_cycle(self.max_epochs, lr_max=self.learning_rate)
        else:
            learner.fit(self.max_epochs, lr=self.learning_rate)
        self._computed_metrics = _recorded_metrics(learner)
        _log_recorded_epochs(learner, trackers)
        logger.info(f"Training complete: {self._computed_metrics}")
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
            raise ValueError("FastaiClassifier.test() requires a 'test_set'.")
        loader = flow(self.test_loader, dataset=ensure_record_dataset(self.test_set))
        learner = self._build_learner(DataLoaders(loader, loader), "test")
        values = learner.validate()
        self._computed_metrics = _named_values(learner, values)
        logger.info(f"Evaluation complete: {self._computed_metrics}")
        write_metrics_sidecar(self.metrics_sidecar_path, self._computed_metrics, self.experiment_name, self.run_name)

    @entrypoint("predict", role="predictor", primary=True)
    def predict(self) -> None:
        """Stream per-record predictions to the sink.

        A plain ``torch.no_grad()`` forward rather than ``Learner.get_preds``: ``get_preds``
        returns only the stacked predictions, and the sink contract needs each row paired with
        the metadata from the record it came from. One walk over the record loader keeps the two
        in step by construction.

        Same ``test_loader`` slot as :meth:`test`, flowed with ``collate_fn=collate_records`` —
        a runtime kwarg overrides the stored one, so this swaps the batch shape back to the
        record dict without a second loader. The default collate is ``_to_tuple``, which has
        already dropped the metadata the sink needs; every other knob the operator set on the
        slot (workers, batch size, a sampler) still applies.
        """
        if self.test_set is None:
            raise ValueError("FastaiClassifier.predict() requires a 'test_set'.")
        sink = flow(self.predictions_sink)
        model = self.model.eval()
        device = next(model.parameters()).device
        records = flow(self.test_loader, dataset=ensure_record_dataset(self.test_set), collate_fn=collate_records)
        with torch.no_grad():
            for batch in records:
                logits = model(batch_tensor(batch, self.input_key, device))
                if sink is not None:
                    self._write_predictions(logits, batch, cast(PredictionsSink, sink))
        if sink is not None and hasattr(sink, "close"):
            sink.close()
        logger.info("Prediction complete.")

    # ---- predictions -------------------------------------------------------- #
    def _write_predictions(self, logits: torch.Tensor, batch: Any, sink: PredictionsSink) -> None:
        """One :class:`~recordstream.ClassificationOutput` per record, paired with its metadata.

        ``classification_output`` builds the CONTRACT (logits / probs / class_idx) — a sink reads
        those by name and cannot softmax on the caller's behalf, since only the model knows
        whether its head already did. The contract is generic in its array type, which is why the
        very same sink consumes this torch output and the Keras backend's numpy one.
        """
        batched = classification_output(logits)
        per_record_meta = batch_metadata(batch, exclude=(self.input_key, self.target_key))
        for i in range(len(logits)):
            # Sliced key by key rather than by comprehension: the contract is a TypedDict, so
            # naming the three fields is what keeps it TYPED on the way out (and matches the
            # Keras backend's `_write_predictions` line for line).
            row: ClassificationOutput[torch.Tensor] = {
                "logits": batched["logits"][i],
                "probs": batched["probs"][i],
                "class_idx": batched["class_idx"][i],
            }
            meta = per_record_meta[i] if per_record_meta and i < len(per_record_meta) else {}
            sink.write(row, meta)

    # ---- artifacts + tracking ----------------------------------------------- #
    def _artifact_dir(self) -> Optional[str]:
        """Where run artifacts land. fastai has no checkpoint-callback convention to read, so
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


def _as_metric(value: FastaiMetric) -> Any:
    """A fastai metric from a callable or from its NAME in ``fastai.metrics``.

    The string form is not sugar. ``accuracy``, ``error_rate`` and ``top_k_accuracy`` are plain
    functions and ``F1Score`` / ``RocAuc`` are factories that RETURN one, so none of them is a
    class — there is nothing for a ``!class:`` tag to name or for confluid's registry to tag. A
    name is the only thing a YAML config can write, so this resolves it and says which names
    exist when it cannot.
    """
    built = flow(value)
    if not isinstance(built, str):
        return built
    metric = getattr(fastai_metrics, built, None)
    if metric is None:
        raise ValueError(
            f"Unknown fastai metric {built!r}. Names come from `fastai.metrics` — e.g. accuracy, "
            f"accuracy_multi, error_rate, top_k_accuracy, BalancedAccuracy, MatthewsCorrCoef."
        )
    return metric


def _as_opt_func(value: FastaiOptimizer) -> Any:
    """A fastai optimizer function from a callable or its name (``"Adam"``, ``"SGD"``, …)."""
    built = flow(value)
    if not isinstance(built, str):
        return built
    opt = getattr(fastai_optimizer, built, None)
    if opt is None:
        raise ValueError(
            f"Unknown fastai optimizer {built!r}. Names come from `fastai.optimizer` — e.g. Adam, "
            f"SGD, RMSProp, RAdam, QHAdam, Lamb, ranger."
        )
    return opt


def _metric_names(learner: Learner) -> List[str]:
    """The Recorder's column names, minus its two bookkeeping columns.

    fastai names the columns ``['epoch', 'train_loss', 'valid_loss', <metrics…>, 'time']`` while
    ``recorder.values`` holds only the numeric middle, so the two ends have to come off before
    the names line up with the values.
    """
    names = list(getattr(learner.recorder, "metric_names", []) or [])
    return [name for name in names if name not in ("epoch", "time")]


def _named_values(learner: Learner, values: Sequence[Any]) -> Dict[str, float]:
    """Pair a Recorder row with its column names, tolerating a shorter row.

    ``Learner.validate`` returns one fewer value than a training epoch does (there is no
    ``train_loss``), so the names are matched from the END — where ``valid_loss`` and the metrics
    line up — rather than from the front.
    """
    names = _metric_names(learner)
    if len(names) > len(values):
        names = names[len(names) - len(values) :]
    return {name: float(value) for name, value in zip(names, values)}


def _recorded_metrics(learner: Learner) -> Dict[str, float]:
    """The last epoch's metrics as ``{name: value}``, or ``{}`` if nothing was recorded."""
    values = list(getattr(learner.recorder, "values", []) or [])
    return _named_values(learner, values[-1]) if values else {}


def _log_recorded_epochs(learner: Learner, trackers: Sequence[Tracker]) -> None:
    """Forward EVERY recorded epoch to the trackers, not just the final one.

    fastai's ``Recorder`` already holds the whole history (``recorder.values`` is one row per
    epoch), so replaying it after the fit gives the same curve a per-epoch callback would,
    without inserting a marainer callback into fastai's callback chain — one fewer thing that
    can reorder or swallow. ``_recorded_metrics`` reads the last row of this same list, so the
    two cannot disagree about what the run scored.
    """
    if not trackers:
        return
    for epoch, row in enumerate(getattr(learner.recorder, "values", []) or []):
        metrics = _named_values(learner, row)
        for tracker in trackers:
            tracker.log_metrics(metrics, step=epoch)
