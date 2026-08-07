"""``ClassificationBase`` — the task half of every sonair backend, written once.

The classification-shaped logic that used to exist per backend module: the multilabel peek,
the label-encoding pipeline (fit/pin a :class:`~recordstream.LabelMap` for string targets,
reload it on the eval path), ``num_classes`` resolution, the single announced target walk that
serves label fitting AND class counting AND balance weights, and the ``class_names`` output.

It subclasses the trainer API base (`matrainer.runnable.TrainerRunnable`) and is deliberately
**torch-free**: a keras/jax/mlx backend imports this module too, and the core-is-backend-free
mandate forbids it dragging a framework in. The torch-family hook IMPLEMENTATIONS (target
dtype, default loss/metrics, weight injection) live in :mod:`sonair.classification_torch`; a
concrete backend composes ``(ClassificationTorchHooks, ClassificationBase, <Framework>Runnable)``.
"""

from typing import TYPE_CHECKING, Any, List, Optional

from confluid import Fluid, flow, output
from loggair import get_logger
from matrainer.artifacts import load_class_names
from matrainer.runnable import TrainerRunnable
from recordstream import LabelMap, ensure_record_dataset, first_value, inverse_frequency_weights, is_class_id, iter_key
from tqdm import tqdm

if TYPE_CHECKING:  # annotation-only: this module must stay importable without numpy's consumers
    import numpy as np

logger = get_logger(__name__)

__all__ = ["ClassificationBase"]


class ClassificationBase(TrainerRunnable):
    """The classification task base: label pipeline + ``num_classes`` resolution + outputs.

    The attributes below are STORED by the concrete backend's own ``__init__`` (they are its
    config surface); this base only declares their names/types for the methods that read them.
    """

    log_label = "sonair"

    num_classes: Optional[int]
    label_map: Optional[LabelMap]
    balance_classes: bool
    #: True once ``initialize()`` sees multi-label targets. Drives the loss/metric defaults,
    #: the multi-hot target read, and whether metrics size by ``num_labels`` or ``num_classes``.
    multilabel: bool
    plot_metrics: bool
    _class_names: Optional[List[str]]

    # ---- framework hooks (a family module overrides both together) ------------------------- #
    def accepts_class_weights(self) -> bool:
        """Whether this backend's loss can consume per-class weights — **framework hook**.

        ``False`` is always safe: the run trains unweighted and skips the target walk. The
        torch family answers by inspecting the loss constructor
        (:class:`~sonair.classification_torch.ClassificationTorchHooks`); a Keras backend
        answers a different question entirely (``class_weight`` on ``fit()``).
        """
        return False

    def apply_class_weights(self, weights: "np.ndarray") -> None:
        """Inject per-class ``weights`` into the backend's loss — **framework hook**.

        The WEIGHTS themselves are framework-neutral (a statistic over the labels, from
        :func:`recordstream.inverse_frequency_weights`); only this last step differs per
        backend, which is why the default does nothing.
        """

    # ---- classification specifics ----------------------------------------------------------- #
    def initialize(self) -> None:
        """Resolve the label encoding + ``num_classes``, then flow the model + metrics with it."""
        if self.task == "fit":
            self._initialize_fit()
        else:
            self._initialize_eval()

    def _initialize_fit(self) -> None:
        self.train_set = ensure_record_dataset(self.train_set)
        if self.val_set is not None:
            self.val_set = ensure_record_dataset(self.val_set)
        if self.test_set is not None:
            self.test_set = ensure_record_dataset(self.test_set)

        # `iter_key` unwraps a MultiLabel to its list of values, so a sequence here IS a
        # multi-label target — the item type, not a guess about what a list might mean.
        first = first_value(self.train_set, self.target_key)
        self.multilabel = isinstance(first, (list, tuple, set))

        # Names need a LabelMap; ids do not. `is_class_id` is recordstream's rule (bool
        # excluded, so a boolean flag wired to the target key cannot silently become class 1).
        # For a multi-label target the question is asked of its FIRST member.
        probe = next(iter(first), None) if self.multilabel else first
        targets_are_names = probe is not None and not is_class_id(probe)
        # Resolve the sentinels HERE: the capability probe below reads `loss_fn`, and an
        # unresolved `None` would report "cannot be balanced" and skip the target walk.
        self._resolve_task_defaults()
        nc = self._num_classes_without_data()
        want_weights = self.balance_classes and self.accepts_class_weights()
        need_fit = targets_are_names and self.label_map is None
        raw_targets: Optional[List[Any]] = None
        if need_fit or want_weights or nc is None:
            raw_targets = self._walk_targets(self.train_set)

        if need_fit:
            assert raw_targets is not None
            self.label_map = LabelMap.fit(raw_targets)
            logger.info(f"Fitted LabelMap with {self.label_map.num_classes} classes from train targets.")
        if targets_are_names and self.label_map is not None:
            self.train_set = self.label_map.encode(self.train_set)
            if self.val_set is not None:
                self.val_set = self.label_map.encode(self.val_set)
            if self.test_set is not None:
                self.test_set = self.label_map.encode(self.test_set)

        if nc is None:
            if self.label_map is not None:
                nc = self.label_map.num_classes
            else:
                assert raw_targets is not None
                # An empty map still maps ids through — that is the `to_ids` contract.
                ids = [i for t in raw_targets for i in LabelMap().to_ids(t)]
                nc = max(ids) + 1 if ids else None
        if not nc:
            raise RuntimeError(
                "could not determine num_classes — set it on the trainer/model or wire a labelled dataset."
            )
        self.num_classes = int(nc)

        if self.balance_classes and getattr(self, "class_weights", None) is None:
            self._balance_loss(self.num_classes, raw_targets)

        self.model = flow(self.model, num_classes=self.num_classes)
        self._build_fit_metrics()

    def _initialize_eval(self) -> None:
        # The trained model's head is authoritative for eval — flow it first, read num_classes
        # off it.
        self.model = flow(self.model)
        nc = getattr(self.model, "num_classes", None)
        if not nc:
            raise RuntimeError("could not determine num_classes from the model (it must expose `num_classes`).")
        self.num_classes = int(nc)
        eval_slot = "val_set" if self.task == "evaluate" else "test_set"
        source = getattr(self, eval_slot)
        if source is None:
            return
        dataset = ensure_record_dataset(source)
        first = first_value(dataset, self.target_key)
        self.multilabel = isinstance(first, (list, tuple, set))
        self._resolve_task_defaults()
        probe = next(iter(first), None) if self.multilabel else first
        if probe is not None and not is_class_id(probe):
            label_map = self.label_map or load_class_names(getattr(self.model, "checkpoint_path", None))
            if label_map is None:
                raise RuntimeError("eval dataset has string targets but no label_map and no class_names.json found.")
            self.label_map = label_map
            dataset = label_map.encode(dataset)
        setattr(self, eval_slot, dataset)
        self._build_eval_metrics()

    def _walk_targets(self, source: Any) -> List[Any]:
        """Materialize the target column once, with a progress bar.

        Everything here except the bar is recordstream's: ``iter_key`` projects, so a
        projection-aware source never builds the values this does not ask for, and it
        materializes a deferred source itself. What remains is the PROGRESS — reading 60k
        labels is a visible pause a user needs feedback on — which is a presentation choice
        a data engine should not make.
        """
        total = len(source) if hasattr(source, "__len__") else None
        return list(tqdm(iter_key(source, self.target_key), total=total, desc="sonair: reading targets", unit="record"))

    def _num_classes_without_data(self) -> Optional[int]:
        if not isinstance(self.model, Fluid):
            model_nc = getattr(self.model, "num_classes", None)
            if model_nc:
                if self.num_classes is not None and int(self.num_classes) != int(model_nc):
                    logger.warning(
                        f"Trainer num_classes={self.num_classes} != model num_classes={model_nc}; using model's."
                    )
                return int(model_nc)
        if self.num_classes is not None:
            return int(self.num_classes)
        if self.label_map is not None:
            return self.label_map.num_classes
        if isinstance(self.model, Fluid):
            pinned = self.model.kwargs.get("num_classes")
            if pinned:
                return int(pinned)
        return None

    def _balance_loss(self, num_classes: int, raw_targets: Optional[List[Any]]) -> None:
        """Derive the class weights from the walked targets and hand them to the backend hook.

        `raw_targets` is None when no walk was needed — at this call site that means the loss
        is not weight-capable (`want_weights` gated the walk), so there is nothing to balance.
        Everything about the derivation is recordstream's: ``inverse_frequency_weights``
        flattens each target through the :class:`~recordstream.LabelMap` (names encode, ids
        pass through, a multi-label target counts for every class it names).
        """
        if not raw_targets:
            return
        weights = inverse_frequency_weights(raw_targets, num_classes, self.label_map)
        if weights is None:
            logger.warning("Class balancing requested but no class ids were observed; training unweighted.")
            return
        self.apply_class_weights(weights)

    # ---- @outputs ---------------------------------------------------------------------------- #
    @property
    @output
    def class_names(self) -> List[str]:
        """Resolved class/label names for the run (post-run ``@output``), or ``[]``."""
        if self.label_map is not None:
            return list(self.label_map.class_names)
        return list(self._class_names or [])
