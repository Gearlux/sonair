# mypy: disable-error-code="attr-defined"
"""Discovery-category coverage for sonair ``@configurable`` classes.

These ``category=`` tags drive navigaitor's ``list_configurable_classes(category=...)``
and the visual-editor form-spec picker (``get_node_form_spec``). Pinned here so a
dropped tag fails CI instead of silently emptying a picker.
"""

from confluid.registry import get_registry

import sonair.torch_classification  # noqa: F401 — import for the register(nn.CrossEntropyLoss, ...) side effect
from sonair.timm_model import TimmModel


def test_classification_classes_tagged() -> None:
    assert TimmModel.__confluid_category__ == "classification_model"


def test_categories_enumerable_via_registry() -> None:
    """The picker queries the registry's category index, so it must carry the tags."""
    registry = get_registry()
    assert "TimmModel" in registry.list_classes(category="classification_model")
    # The classification loss is the registered third-party torch.nn.CrossEntropyLoss.
    assert "CrossEntropyLoss" in registry.list_classes(category="classification_loss")
    # SigmoidFocalLoss is sonair's own thin wrapper over torchvision's function.
    assert "SigmoidFocalLoss" in registry.list_classes(category="classification_loss")


def test_task_role_tags() -> None:
    """task/role drive navigaitor's scan-and-generate task surfaces."""
    from sonair.lightning_classifier import LightningClassifier

    assert TimmModel.__confluid_task__ == "classification"
    assert TimmModel.__confluid_role__ == "model"
    assert LightningClassifier.__confluid_task__ == "classification"
    assert LightningClassifier.__confluid_role__ == "trainer"


def test_task_role_enumerable_via_registry() -> None:
    registry = get_registry()
    assert "TimmModel" in registry.list_classes(task="classification", role="model")
    assert "CrossEntropyLoss" in registry.list_classes(task="classification", role="loss")
    assert "LightningClassifier" in registry.list_classes(task="classification", role="trainer")


def test_all_classification_losses_registered() -> None:
    """Every loss in ``_CLASSIFICATION_LOSSES`` (+ SigmoidFocalLoss) is a ``role="loss"`` configurable."""
    from sonair.torch_classification import _CLASSIFICATION_LOSSES

    loss_names = get_registry().list_classes(task="classification", role="loss")
    for cls in _CLASSIFICATION_LOSSES:
        assert cls.__name__ in loss_names, f"{cls.__name__} not registered as a classification loss"
    assert "SigmoidFocalLoss" in loss_names


def test_all_classification_metrics_registered() -> None:
    """Every metric in ``_CLASSIFICATION_METRICS`` is a ``role="metric"`` / ``classification_metric`` configurable."""
    from sonair.torch_classification import _CLASSIFICATION_METRICS

    registry = get_registry()
    metric_names = registry.list_classes(task="classification", role="metric")
    for cls in _CLASSIFICATION_METRICS:
        assert cls.__name__ in metric_names, f"{cls.__name__} not registered as a classification metric"
    # category is derived `classification_metric`, used by navigaitor's metric-slot picker.
    assert "MulticlassAccuracy" in registry.list_classes(category="classification_metric")


def test_all_exports_registered_losses_and_metrics() -> None:
    """``__all__`` must name every registered loss/metric so StreamStudio's ``__all__``-pass surfaces them.

    Pins the literal ``__all__`` in sync with ``_CLASSIFICATION_LOSSES`` / ``_CLASSIFICATION_METRICS``.
    """
    import sonair.torch_classification as c

    for cls in (*c._CLASSIFICATION_LOSSES, *c._CLASSIFICATION_METRICS):
        assert cls.__name__ in c.__all__, f"{cls.__name__} registered but missing from __all__ (won't surface)"


# ---------------------------------------------------------------------------
# framework — every sonair registration claims its engine
# ---------------------------------------------------------------------------
# `task`+`role` say what a class is FOR; `framework` says what it can be WIRED
# TO. Without it a picker asked for "a classification loss" would offer these
# torch losses to a Keras trainer. Each engine registers its OWN set from its own
# module (`torch_classification.py` / `keras_classification.py`), which is what
# makes the two sets disjoint rather than merely tagged.


def test_every_sonair_registration_declares_framework_torch() -> None:
    from sonair.lightning_classifier import LightningClassifier
    from sonair.timm_model import TimmModel
    from sonair.torch_classification import _CLASSIFICATION_LOSSES, _CLASSIFICATION_METRICS, SigmoidFocalLoss

    tagged = (
        *_CLASSIFICATION_LOSSES,
        *_CLASSIFICATION_METRICS,
        SigmoidFocalLoss,
        LightningClassifier,
        TimmModel,
    )
    untagged = [c.__name__ for c in tagged if getattr(c, "__confluid_framework__", None) != "torch"]
    assert not untagged, f"missing framework='torch': {untagged}"


def test_framework_filter_narrows_the_loss_picker() -> None:
    """The query a discovery consumer makes — and what it must NOT return."""
    import confluid

    import sonair.keras_classification  # noqa: F401 — register the keras set too

    reg = confluid.get_registry()
    torch_losses = reg.list_classes(task="classification", role="loss", framework="torch")
    assert "CrossEntropyLoss" in torch_losses
    assert "SigmoidFocalLoss" in torch_losses
    # The point of the axis: each engine's picker offers ONLY its own set. A Keras trainer is
    # never offered a torch loss, and vice versa — the failure this prevents is a config that
    # merges cleanly and dies at run time with a type error far from its cause.
    keras_losses = reg.list_classes(task="classification", role="loss", framework="keras")
    assert not (torch_losses & keras_losses)
    # Seven Keras names are BOTH a loss and a metric, so they publish under their dotted key
    # and the picker's real query is the filtered lookup rather than a name membership test.
    assert reg.get_class("SparseCategoricalCrossentropy", role="loss", framework="keras") is not None
    assert reg.get_class("SparseCategoricalCrossentropy", role="loss", framework="torch") is None
