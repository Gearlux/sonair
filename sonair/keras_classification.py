"""The **Keras** classification losses and metrics — registered, never wrapped.

The twin of :mod:`sonair.torch_classification`, and the reason both exist separately: a
loss is engine-bound. ``keras.losses.SparseCategoricalCrossentropy`` cannot be handed to
:class:`~sonair.lightning_classifier.LightningClassifier` and ``torch.nn.CrossEntropyLoss``
cannot be handed to :class:`~sonair.keras_classifier.KerasClassifier`, so every class here
carries ``framework="keras"`` and every class there carries ``framework="torch"`` — the
axis navigaitor's slot picker filters on (``docs/architecture.md`` §2). One module per
engine also means one ``confluid.configurables`` entry point per engine: on a torch-only
install THIS module fails to import and the scanner skips it, which is the correct answer
rather than a degraded one.

Keras is imported through :mod:`sonair._keras` and the classes are reached as
``keras.losses.X`` / ``keras.metrics.X`` **attributes**, never ``from keras.losses import
X`` — a direct keras import sorts above the first-party one and would beat
``recordstream.keras``'s ``KERAS_BACKEND`` default into place (see :mod:`sonair._keras`).

``from_logits=True`` is not optional
------------------------------------
The classifier's head is a bare ``Dense`` with no activation, matching the torch backend.
Every crossentropy loss below defaults to ``from_logits=False``, i.e. it expects
*probabilities*, and handed logits it normalizes by the row SUM instead of a softmax —
then CLIPS rather than raises, so the loss stays finite and merely wrong. Always write
``from_logits: true`` when wiring one::

    loss: !class:keras.losses.SparseCategoricalCrossentropy
      from_logits: true

Picking a metric: target shape first, then the logits knob
----------------------------------------------------------
The first question is not ``from_logits`` — it is what ``KerasClassifier._to_arrays`` hands
the metric as ``y_true``:

* **single-label** (``recordstream.Label``) -> int64 class ids ``[N]``. Only the ``Sparse*``
  metrics read that shape: ``SparseCategoricalAccuracy``, ``SparseTopKCategoricalAccuracy``.
* **multi-label** (``recordstream.MultiLabel``) -> a float multi-hot ``[N, C]``, the same
  shape as the logits. That is what every element-wise metric needs: ``BinaryAccuracy``,
  ``Precision``, ``Recall``, the four TP/TN/FP/FN counters, ``F1Score`` / ``FBetaScore``,
  ``AUC``, and the one-hot ``CategoricalAccuracy`` / ``TopKCategoricalAccuracy``.

Mixing them is not a subtle mistake: ``Precision`` wired beside ``SparseCategoricalAccuracy``
on a 10-class MNIST run dies at the first batch with *"The size of tensor a (32) must match
the size of tensor b (320)"*. This is the same split the torch backend expresses as two
metric FAMILIES (``Multiclass*`` sized by ``num_classes`` vs ``Multilabel*`` by ``num_labels``);
Keras just spells it in the metric name.

Then, for a shape-compatible metric, the logits question (measured, 64x4 random logits vs
their sigmoid):

* **argmax / top-k** — the ``*Accuracy`` family and ``F1Score`` / ``FBetaScore`` with their
  default ``threshold=None`` (which argmaxes): order-preserving, so logits and probabilities
  give the identical number. Nothing to set.
* **thresholded** — ``Precision``, ``Recall``, ``BinaryAccuracy``, ``TruePositives`` and the
  other three counters: the default threshold ``0.5`` is a *probability*. For a logits head
  pass ``thresholds: 0.0`` (``threshold: 0.0`` on ``BinaryAccuracy``) — the exact logit-space
  equivalent, since ``sigmoid(0) == 0.5`` (verified identical to the probability form:
  precision 0.272, 34 true positives either way).
* **``AUC``** — the one metric with the knob: ``from_logits: true``.
* ``F1Score`` / ``FBetaScore`` with an EXPLICIT ``threshold`` reject the logit-space value
  (Keras requires ``0 < threshold <= 1``), so either leave it unset or feed probabilities.

Two upstream families are deliberately NOT registered, because they cannot consume what
this project's head emits and offer no knob to fix it — the same reasoning that keeps a
torch loss out of a Keras picker:

* ``keras.metrics.Accuracy`` compares ``y_true`` to ``y_pred`` elementwise, so it wants
  already-argmaxed class ids; against a logits/one-hot pair it reports ``0.0``.
* ``PrecisionAtRecall`` / ``RecallAtPrecision`` / ``SensitivityAtSpecificity`` /
  ``SpecificityAtSensitivity`` sweep thresholds across ``[0, 1]`` with no ``from_logits``
  and no threshold argument, so on logits they silently report a different number
  (``SensitivityAtSpecificity``: 0.530 on probabilities, 0.515 on the same logits).

Not registering only removes them from the PICKERS; ``!class:keras.metrics.Accuracy``
still resolves from YAML for anyone whose model really does emit class ids.

There is no Keras counterpart to :class:`sonair.torch_classification.SigmoidFocalLoss`:
that shim exists only because ``torchvision.ops.sigmoid_focal_loss`` is a *function* and
confluid's ``register`` tags classes. Keras ships focal loss as a class
(``BinaryFocalCrossentropy`` / ``CategoricalFocalCrossentropy``), so it is registered
directly — nothing to write.

Typical wiring (YAML) — a single-label run::

    keras: !scope:framework=keras
      runnable: !class:sonair.KerasClassifier
        loss: !class:keras.losses.SparseCategoricalCrossentropy
          from_logits: true
        metrics:
          - !class:keras.metrics.SparseCategoricalAccuracy
          - !class:keras.metrics.SparseTopKCategoricalAccuracy { k: 5 }

...and the multi-label counterpart, where the element-wise metrics belong::

        loss: !class:keras.losses.BinaryCrossentropy { from_logits: true }
        metrics:
          - !class:keras.metrics.BinaryAccuracy { threshold: 0.0 }
          - !class:keras.metrics.Precision { thresholds: 0.0 }
          - !class:keras.metrics.AUC { from_logits: true, multi_label: true, num_labels: 10 }

Verified end to end with ``sonair train --framework keras`` (MNIST, the first block above): the loss
descends and both metrics report.
"""

from typing import Any, Literal, Optional

from confluid import configurable, register

from sonair._keras import keras

# The classification loss IS an off-the-shelf ``keras.losses`` class — register each (no wrapper
# class) so it is a pickable ``task="classification", role="loss", framework="keras"`` option in
# navigaitor's task surfaces AND a StreamStudio OBJECT node (``role="loss"`` is in the bridge
# ``_MEMBER_ROLES``). Bound to module-level names AND listed in ``__all__`` (below) because
# ``scan_module`` filters by ``__module__`` — which is ``keras.src.losses.*`` for these — so the
# ``__all__``-gated discovery pass is what surfaces them.
BinaryCrossentropy = keras.losses.BinaryCrossentropy
BinaryFocalCrossentropy = keras.losses.BinaryFocalCrossentropy
CategoricalCrossentropy = keras.losses.CategoricalCrossentropy
CategoricalFocalCrossentropy = keras.losses.CategoricalFocalCrossentropy
CategoricalHinge = keras.losses.CategoricalHinge
Hinge = keras.losses.Hinge
KLDivergence = keras.losses.KLDivergence
SparseCategoricalCrossentropy = keras.losses.SparseCategoricalCrossentropy
SquaredHinge = keras.losses.SquaredHinge

_CLASSIFICATION_LOSSES = (
    # int64 class ids [N] — the shape `KerasClassifier._to_arrays` produces for a single-label
    # target, and the sentinel default `_resolve_task_defaults` picks.
    SparseCategoricalCrossentropy,
    # float [N, C] targets: one-hot (categorical) or multi-hot (binary/multi-label — the
    # sentinel default for a `recordstream.MultiLabel` target).
    CategoricalCrossentropy,
    BinaryCrossentropy,
    # Focal variants — down-weight easy examples. Keras ships them as CLASSES, so unlike the
    # torch side no `nn.Module` shim is needed.
    CategoricalFocalCrossentropy,
    BinaryFocalCrossentropy,
    # Margin family — y_true in {-1, 1} (Keras converts 0 -> -1), raw scores as y_pred.
    Hinge,
    SquaredHinge,
    CategoricalHinge,
    KLDivergence,
)
for _loss_cls in _CLASSIFICATION_LOSSES:
    register(_loss_cls, task="classification", role="loss", framework="keras")

# The useful ``keras.metrics`` classification metrics.
#
# Keras publishes thirteen names under BOTH ``keras.losses`` and ``keras.metrics``, and seven
# of them are losses this module already registers. They are registered as metrics TOO, which
# only became possible with confluid 0.3.0's multi-class names: the registry admits a shared
# name when a discovery tag tells the entries apart, and here ``role`` does
# (``get_class("Hinge", role="metric")``). Until then the second registration silently
# overwrote the first, so these seven had to be left out entirely.
#
# The PYTHON binding still has to be unique, hence the ``*Metric`` suffix on the module-level
# alias — the registry name stays the class's own (``Hinge``), which is what a config writes.
AUC = keras.metrics.AUC
BinaryAccuracy = keras.metrics.BinaryAccuracy
CategoricalAccuracy = keras.metrics.CategoricalAccuracy
F1Score = keras.metrics.F1Score
FBetaScore = keras.metrics.FBetaScore
FalseNegatives = keras.metrics.FalseNegatives
FalsePositives = keras.metrics.FalsePositives
Precision = keras.metrics.Precision
Recall = keras.metrics.Recall
SparseCategoricalAccuracy = keras.metrics.SparseCategoricalAccuracy
SparseTopKCategoricalAccuracy = keras.metrics.SparseTopKCategoricalAccuracy
TopKCategoricalAccuracy = keras.metrics.TopKCategoricalAccuracy
TrueNegatives = keras.metrics.TrueNegatives
TruePositives = keras.metrics.TruePositives
# The seven loss namesakes, as metrics. Tracking a loss you are NOT optimizing is the point
# (train with SparseCategoricalCrossentropy, watch KLDivergence alongside it); Keras reports
# the COMPILED loss automatically, so these earn their place only for the others.
BinaryCrossentropyMetric = keras.metrics.BinaryCrossentropy
CategoricalCrossentropyMetric = keras.metrics.CategoricalCrossentropy
CategoricalHingeMetric = keras.metrics.CategoricalHinge
HingeMetric = keras.metrics.Hinge
KLDivergenceMetric = keras.metrics.KLDivergence
SparseCategoricalCrossentropyMetric = keras.metrics.SparseCategoricalCrossentropy
SquaredHingeMetric = keras.metrics.SquaredHinge

_CLASSIFICATION_METRICS = (
    # argmax-based, so safe on a raw-logits head with nothing to configure.
    SparseCategoricalAccuracy,  # int64 ids [N]
    CategoricalAccuracy,  # one-hot [N, C]
    SparseTopKCategoricalAccuracy,  # int64 ids [N], `k`
    TopKCategoricalAccuracy,  # one-hot [N, C], `k`
    # Threshold-based: correct on logits with `thresholds: 0.0` (`threshold` on BinaryAccuracy),
    # the logit-space equivalent of the probability default 0.5. See the module docstring.
    BinaryAccuracy,
    Precision,
    Recall,
    TruePositives,
    TrueNegatives,
    FalsePositives,
    FalseNegatives,
    # `threshold=None` (the default) argmaxes, which IS logit-safe; an explicit threshold must
    # be a probability in (0, 1] — Keras rejects 0.0 — so leave it unset on a logits head.
    F1Score,
    FBetaScore,
    # The one metric carrying the knob: `from_logits: true`.
    AUC,
    # The loss namesakes. Same class NAME as the loss above, different `role` — which is
    # exactly what lets both be registered (confluid >= 0.3.0). A config picks one with
    # `!class:Hinge@role=metric` or `get_class("Hinge", role="metric")`; a BARE `Hinge`
    # is now an error naming both, rather than silently resolving to whichever won.
    BinaryCrossentropyMetric,
    CategoricalCrossentropyMetric,
    CategoricalHingeMetric,
    HingeMetric,
    KLDivergenceMetric,
    SparseCategoricalCrossentropyMetric,
    SquaredHingeMetric,
)
for _metric_cls in _CLASSIFICATION_METRICS:
    register(_metric_cls, task="classification", role="metric", framework="keras")


#: How ``F1Score`` reduces its per-class scores. Closed, because the set is fixed and short —
#: a picker reads the allowed values straight off the annotation instead of restating them.
F1Average = Literal["macro", "micro", "weighted"]


@configurable(task="classification", role="metric", framework="keras")
class SparseF1Score(keras.metrics.F1Score):
    """``keras.metrics.F1Score`` for INT-ID targets — the one metric shim this module needs.

    Every other metric here is an upstream class registered as-is, which is the rule. This one
    is a subclass because of a shape mismatch that no configuration can close: ``F1Score``
    compares two ``[N, C]`` tensors, while a single-label run in this project produces int64
    class ids ``[N]`` (see ``KerasClassifier._to_arrays``). Wired against sparse targets
    upstream's own class dies on the first batch — the same
    *"size of tensor a (32) must match tensor b (320)"* the module docstring records for
    ``Precision`` — so there is no version of "just register it" that works.

    It one-hots ``y_true`` and delegates; it computes NO F1 arithmetic of its own, which is what
    keeps it a transport shim rather than the reimplementation the loss/metric mandate forbids
    (the precedent is ``sonair.torch_classification.SigmoidFocalLoss``, wrapping a torchvision
    FUNCTION for the same "cannot be registered as it stands" reason).

    ``y_pred`` may be raw logits: the inherited ``threshold=None`` argmaxes, and argmax is
    invariant under softmax — the same property that makes ``SparseCategoricalAccuracy`` safe on
    this project's bare-``Dense`` head. An explicit ``threshold`` must be a probability in
    ``(0, 1]``, so leave it unset on a logits head.

    Args:
        average: How the per-class F1s are reduced. ``macro`` is the unweighted mean — the
            reduction the torch backends' ``MulticlassF1Score(average="macro")`` computes, so
            the two engines' numbers are comparable.
        threshold: Score ``y_pred`` above this value as a positive prediction instead of
            argmaxing. Leave unset for a logits head (see above).
        name: The metric's name in every tracker, and therefore the COLUMN a cross-framework
            comparison lines up on. Defaults to ``f1_<average>``, so the workspace-canonical
            ``f1_macro`` needs nothing written; derived rather than a literal ``"f1_macro"``
            so a run with ``average="micro"`` cannot record it under the macro name.
        dtype: The accumulator dtype, as upstream.

    Typical wiring (YAML), beside the accuracy every backend reports::

        metrics:
          - !class:keras.metrics.SparseCategoricalAccuracy { name: accuracy }
          - !class:sonair.SparseF1Score
    """

    def __init__(
        self,
        average: F1Average = "macro",
        threshold: Optional[float] = None,
        name: Optional[str] = None,
        dtype: Optional[str] = None,
    ) -> None:
        super().__init__(average=average, threshold=threshold, name=name or f"f1_{average}", dtype=dtype)

    def update_state(self, y_true: Any, y_pred: Any, sample_weight: Optional[Any] = None) -> Any:
        """One-hot the int-id targets to ``y_pred``'s width, then accumulate as upstream does.

        The width comes from ``y_pred`` rather than a constructor knob on purpose: the head
        already declares it, and a second declaration is one that can disagree with the model.
        ``keras.ops`` throughout, so this runs on whichever engine ``KERAS_BACKEND`` names.
        """
        ids = keras.ops.cast(keras.ops.reshape(y_true, [-1]), "int32")
        one_hot = keras.ops.one_hot(ids, keras.ops.shape(y_pred)[-1], dtype=y_pred.dtype)
        return super().update_state(one_hot, y_pred, sample_weight)


# Every registered loss / metric is named in ``__all__`` so StreamStudio's ``__all__``-gated
# discovery pass surfaces each as a node. Keep this list in sync with ``_CLASSIFICATION_LOSSES`` /
# ``_CLASSIFICATION_METRICS`` above (a literal, not a comprehension, so static analysers can read
# the exported-symbol set); ``tests/test_keras_classification.py`` pins the sync.
__all__ = [
    # losses (keras.losses) — role="loss"
    "BinaryCrossentropy",
    "BinaryFocalCrossentropy",
    "CategoricalCrossentropy",
    "CategoricalFocalCrossentropy",
    "CategoricalHinge",
    "Hinge",
    "KLDivergence",
    "SparseCategoricalCrossentropy",
    "SquaredHinge",
    # metrics (keras.metrics) — role="metric"
    "AUC",
    "BinaryAccuracy",
    "CategoricalAccuracy",
    "F1Score",
    "FBetaScore",
    "FalseNegatives",
    "FalsePositives",
    "Precision",
    "Recall",
    "SparseCategoricalAccuracy",
    "SparseTopKCategoricalAccuracy",
    "TopKCategoricalAccuracy",
    # the one shim (see SparseF1Score) — sonair's own class, hence @configurable rather than
    # `register`, but exported for the same discovery pass as every registration above.
    "SparseF1Score",
    "TrueNegatives",
    "TruePositives",
    # metrics (keras.metrics) — role="metric", the loss namesakes (see above)
    "BinaryCrossentropyMetric",
    "CategoricalCrossentropyMetric",
    "CategoricalHingeMetric",
    "HingeMetric",
    "KLDivergenceMetric",
    "SparseCategoricalCrossentropyMetric",
    "SquaredHingeMetric",
]
