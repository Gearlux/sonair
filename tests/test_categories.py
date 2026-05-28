# mypy: disable-error-code="attr-defined"
"""Discovery-category coverage for sonair ``@configurable`` classes.

These ``category=`` tags drive navigaitor's ``list_configurable_classes(category=...)``
and the visual-editor form-spec picker (``get_node_form_spec``). Pinned here so a
dropped tag fails CI instead of silently emptying a picker.
"""

from confluid.registry import get_registry

from sonair.classification import ClassificationLoss
from sonair.hf_dataset import HuggingFaceClassificationDataset
from sonair.timm_classifier import TimmClassifierModel


def test_classification_classes_tagged() -> None:
    assert TimmClassifierModel.__confluid_category__ == "classification_model"
    assert ClassificationLoss.__confluid_category__ == "classification_loss"
    assert HuggingFaceClassificationDataset.__confluid_category__ == "classification_dataset"


def test_categories_enumerable_via_registry() -> None:
    """The picker queries the registry's category index, so it must carry the tags."""
    registry = get_registry()
    assert "TimmClassifierModel" in registry.list_classes(category="classification_model")
    assert "ClassificationLoss" in registry.list_classes(category="classification_loss")
    assert "HuggingFaceClassificationDataset" in registry.list_classes(category="classification_dataset")
