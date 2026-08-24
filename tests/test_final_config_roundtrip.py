"""Code + the dumped document must reconstruct the run — derived state stays OUT of the dump.

The tracking manifest dumps the materialized runnable as the run's resolved
configuration. The reconstruction contract is CODE + DOCUMENT: the document
carries the config state; everything the code derives from it — the timm
backbone `solidify()` builds from ``model_name``, the class-weights vector the
balancing pass derives from the dataset — is rebuilt by the same code on
reload, so it must not appear in the document at all. A derived value in the
dump is worse than absent: it either crashes the reload (a placeholder missing
required args) or reloads as a default-constructed lie (an empty tensor where
the weights were).
"""

import torch
import torch.nn as nn
from confluid import dump, load
from recordstream import Label, Stream

from sonair.lightning_classifier import LightningClassifier
from sonair.timm_model import TimmModel


def test_the_built_timm_backbone_stays_out_of_the_dumped_document() -> None:
    """``solidify()`` rebuilds the backbone from ``model_name`` — the document carries the recipe."""
    model = TimmModel(model_name="resnet10t", num_classes=3, pretrained=False)
    model.solidify()
    assert model.classifier is not None, "the fixture must actually build the backbone"

    text = dump(model)
    assert "classifier" not in text
    reloaded = load(text)
    assert reloaded.model_name == "resnet10t"
    assert reloaded.num_classes == 3
    # flow()'s auto-solidify already rebuilt the backbone from the recipe — a FRESH
    # network of the same architecture, not the dumped object smuggled through YAML.
    assert type(reloaded.classifier) is type(model.classifier)
    assert reloaded.classifier is not model.classifier


def test_the_backbone_keeps_its_state_dict_name() -> None:
    """The reason the cache slot is PUBLIC: checkpoint keys must stay ``classifier.*``."""
    model = TimmModel(model_name="resnet10t", num_classes=3, pretrained=False)
    model.solidify()
    keys = list(model.state_dict().keys())
    assert keys, "a built backbone has parameters"
    assert all(k.startswith("classifier.") for k in keys)


class _TinyClassifier(nn.Module):
    def __init__(self, num_classes: int = 3, input_size: int = 4) -> None:
        super().__init__()
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)  # type: ignore[no-any-return]


def test_the_run_documents_tensors_and_metric_collections_carry_their_values() -> None:
    """The user-shaped case: after ``initialize()`` with balancing, the dumped
    runnable used to hold three lies — ``class_weights`` / ``loss_fn.weight`` as
    ``{_target_: torch.Tensor}`` placeholders (reloading as EMPTY tensors) and
    the built metric collections as bare ``{_target_: MetricCollection}``
    (crashing the reload). matrainer's registered dump spellings respell all of
    them faithfully."""
    records = (
        [{"image": torch.randn(4), "class": Label(0)} for _ in range(10)]
        + [{"image": torch.randn(4), "class": Label(1)} for _ in range(3)]
        + [{"image": torch.randn(4), "class": Label(2)} for _ in range(3)]
    )
    trainer = LightningClassifier(
        model=_TinyClassifier(num_classes=3),
        train_set=Stream(source=records),
        num_classes=3,
        balance_classes=True,
    )
    trainer.initialize()
    assert trainer.class_weights is not None

    text = dump(trainer)
    assert "_target_: torch.Tensor" not in text, "no tensor placeholder survives"
    assert "tensor_from_values" in text, "the weights ride as VALUES"
