"""The **fastai** classification losses — registered, never wrapped.

The third member of the ``*_classification.py`` family (after
:mod:`sonair.torch_classification` and :mod:`sonair.keras_classification`), and the one whose
``framework`` tag looks surprising: these are registered ``framework="torch"``, not
``"fastai"``. That is deliberate and it is what the axis means — ``framework`` says what a class
can be WIRED TO, not which package published it (``docs/architecture.md`` §2). Every class here
takes ``(logits, target)`` as torch tensors and returns a torch scalar, so
:class:`~sonair.lightning_classifier.LightningClassifier` can consume ``CrossEntropyLossFlat``
exactly as :class:`~sonair.fastai_classifier.FastaiClassifier` can. Tagging them ``"fastai"``
would have hidden a working option from the picker for no reason.

Two shapes hide behind that one sentence, and only one of them is an ``nn.Module``: the non-flat
``FocalLoss`` / ``LabelSmoothingCrossEntropy`` subclass it, while the ``*Flat`` family are
``fastai.losses.BaseLoss`` objects — plain callables holding the real torch loss in ``.func``
(``CrossEntropyLossFlat().func`` is a ``torch.nn.CrossEntropyLoss``) and forwarding ``.to()``.
Neither shape declares a ``weight`` constructor parameter, so
``LightningClassifier.accepts_class_weights()`` answers *no* for all of them and inverse-frequency
balancing is skipped rather than silently mis-applied.

It is still its OWN module and its own ``confluid.configurables`` entry point, for the reason
every engine's set is: on an install without fastai this module fails to import and the scanner
skips it, so the registrations are absent because the library is — the honest answer, not a
degraded one.

The ``*Flat`` suffix
--------------------
fastai's losses flatten the trailing dimensions before delegating to the torch loss underneath,
which is what lets one class serve a classifier, a segmenter and a language model. For image
classification — logits ``[N, C]``, targets ``[N]`` — a ``*Flat`` loss behaves exactly like the
``torch.nn`` original, so switching between the two backends changes nothing about what is
optimized. They take RAW LOGITS, matching this project's bare-linear head; there is no
``from_logits`` question of the kind the Keras set has.

Metrics are NOT registered here, and cannot be
----------------------------------------------
Nearly every fastai classification metric is a plain *function* (``accuracy``, ``error_rate``,
``top_k_accuracy``, ``accuracy_multi``) or a factory that returns one (``F1Score()``,
``Precision()``, ``RocAuc()``, ``MatthewsCorrCoef()``) — ``skm_to_fastai`` wrappers around
scikit-learn, not classes. confluid's ``register`` tags classes, so there is nothing here to
tag, and a ``!class:`` tag has nothing to name.

So :class:`~sonair.fastai_classifier.FastaiClassifier` takes its metrics by NAME instead,
resolved against ``fastai.metrics``::

    runnable: !class:sonair.FastaiClassifier
      loss: !class:CrossEntropyLossFlat@role=loss,framework=torch()
      metrics: [accuracy, top_k_accuracy]

That is the same shape the Keras backend gets from its string aliases, and it is a property of
fastai's API rather than a shortcut taken here. A metric that *is* a class — anything wrapping
``fastai.metrics.AccumMetric`` — can still be wired with a dotted ``!class:`` path.

Typical wiring (YAML) — a single-label run with label smoothing::

    runnable: !class:sonair.FastaiClassifier
      loss: !class:LabelSmoothingCrossEntropyFlat@role=loss,framework=torch
        eps: 0.1
      metrics: [accuracy]

...and the multi-label counterpart, where the target is a float multi-hot ``[N, C]``::

      loss: !class:BCEWithLogitsLossFlat@role=loss,framework=torch()
      metrics: [accuracy_multi]
"""

from confluid import register
from fastai.losses import (
    BCEWithLogitsLossFlat,
    CrossEntropyLossFlat,
    FocalLoss,
    FocalLossFlat,
    LabelSmoothingCrossEntropy,
    LabelSmoothingCrossEntropyFlat,
)

# The classification loss IS an off-the-shelf ``fastai.losses`` class — register each (no wrapper
# class) so it is a pickable ``task="classification", role="loss", framework="torch"`` option in
# navigaitor's task surfaces AND a StreamStudio OBJECT node (``role="loss"`` is in the bridge
# ``_MEMBER_ROLES``). Listed in ``__all__`` (below) because ``scan_module`` filters by
# ``__module__`` — which is ``fastai.losses`` for these — so the ``__all__``-gated discovery pass
# is what surfaces them.
_CLASSIFICATION_LOSSES = (
    # int64 class ids [N] — the shape `FastaiClassifier._to_tuple` produces for a single-label
    # target, and the sentinel default `_resolve_task_defaults` picks.
    CrossEntropyLossFlat,
    # float [N, C] multi-hot targets — the sentinel default for a `recordstream.MultiLabel`.
    BCEWithLogitsLossFlat,
    # Regularized cross-entropy: spreads `eps` of the probability mass over the other classes, so
    # the model is not pushed toward infinite confidence on a possibly-mislabelled row.
    LabelSmoothingCrossEntropy,
    LabelSmoothingCrossEntropyFlat,
    # Focal variants — down-weight easy examples. The FLAT one is the classification-shaped one;
    # both are classes here, so unlike the torch set no `nn.Module` shim is needed (the torch
    # side needs `SigmoidFocalLoss` only because torchvision ships focal loss as a FUNCTION).
    FocalLoss,
    FocalLossFlat,
)
for _loss_cls in _CLASSIFICATION_LOSSES:
    register(_loss_cls, task="classification", role="loss", framework="torch")


# Every registered loss is named in ``__all__`` so StreamStudio's ``__all__``-gated discovery pass
# surfaces each as a node. Keep this list in sync with ``_CLASSIFICATION_LOSSES`` above (a
# literal, not a comprehension, so static analysers can read the exported-symbol set);
# ``tests/test_fastai_classification.py`` pins the sync.
__all__ = [
    "BCEWithLogitsLossFlat",
    "CrossEntropyLossFlat",
    "FocalLoss",
    "FocalLossFlat",
    "LabelSmoothingCrossEntropy",
    "LabelSmoothingCrossEntropyFlat",
]
