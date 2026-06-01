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
