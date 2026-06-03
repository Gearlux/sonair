# mypy: disable-error-code="attr-defined"
"""Discovery-category coverage for sonair ``@configurable`` classes.

These ``category=`` tags drive navigaitor's ``list_configurable_classes(category=...)``
and the visual-editor form-spec picker (``get_node_form_spec``). Pinned here so a
dropped tag fails CI instead of silently emptying a picker.
"""

from confluid.registry import get_registry

import sonair.classification  # noqa: F401 — import for the register(nn.CrossEntropyLoss, ...) side effect
from sonair.timm_classifier import TimmModel


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
    from sonair.evaluator import LightningClassificationEvaluator
    from sonair.trainer import LightningClassificationTrainer

    assert TimmModel.__confluid_task__ == "classification"
    assert TimmModel.__confluid_role__ == "model"
    assert LightningClassificationTrainer.__confluid_task__ == "classification"
    assert LightningClassificationTrainer.__confluid_role__ == "trainer"
    assert LightningClassificationEvaluator.__confluid_role__ == "evaluator"


def test_task_role_enumerable_via_registry() -> None:
    registry = get_registry()
    assert "TimmModel" in registry.list_classes(task="classification", role="model")
    assert "CrossEntropyLoss" in registry.list_classes(task="classification", role="loss")
    assert "LightningClassificationTrainer" in registry.list_classes(task="classification", role="trainer")


def test_all_classification_losses_registered() -> None:
    """Every loss in ``_CLASSIFICATION_LOSSES`` (+ SigmoidFocalLoss) is a ``role="loss"`` configurable."""
    from sonair.classification import _CLASSIFICATION_LOSSES

    loss_names = get_registry().list_classes(task="classification", role="loss")
    for cls in _CLASSIFICATION_LOSSES:
        assert cls.__name__ in loss_names, f"{cls.__name__} not registered as a classification loss"
    assert "SigmoidFocalLoss" in loss_names


def test_all_classification_metrics_registered() -> None:
    """Every metric in ``_CLASSIFICATION_METRICS`` is a ``role="metric"`` / ``classification_metric`` configurable."""
    from sonair.classification import _CLASSIFICATION_METRICS

    registry = get_registry()
    metric_names = registry.list_classes(task="classification", role="metric")
    for cls in _CLASSIFICATION_METRICS:
        assert cls.__name__ in metric_names, f"{cls.__name__} not registered as a classification metric"
    # category is derived `classification_metric`, used by navigaitor's metric-slot picker.
    assert "MulticlassAccuracy" in registry.list_classes(category="classification_metric")


def test_all_exports_registered_losses_and_metrics() -> None:
    """``__all__`` must name every registered loss/metric so FluxStudio's ``__all__``-pass surfaces them.

    Pins the literal ``__all__`` in sync with ``_CLASSIFICATION_LOSSES`` / ``_CLASSIFICATION_METRICS``.
    """
    import sonair.classification as c

    for cls in (*c._CLASSIFICATION_LOSSES, *c._CLASSIFICATION_METRICS):
        assert cls.__name__ in c.__all__, f"{cls.__name__} registered but missing from __all__ (won't surface)"
