"""The fastai loss registrations — the set, the tags, and the ``__all__`` sync.

The twin of ``tests/test_keras_classification.py``, minus a metric half: fastai's classification
metrics are plain functions, so there is nothing for confluid's class registry to tag. That
absence is itself pinned here, because it is the reason
:class:`~sonair.fastai_classifier.FastaiClassifier` takes metrics by NAME.
"""

import inspect

import pytest

pytest.importorskip("fastai", reason="the fastai loss registrations need sonair[fastai]")

import torch  # noqa: E402
from confluid.registry import get_registry  # noqa: E402

import sonair.fastai_classification as fastai_classification  # noqa: E402

EXPECTED = {
    "BCEWithLogitsLossFlat",
    "CrossEntropyLossFlat",
    "FocalLoss",
    "FocalLossFlat",
    "LabelSmoothingCrossEntropy",
    "LabelSmoothingCrossEntropyFlat",
}


def test_the_registered_set_is_what_the_module_advertises() -> None:
    assert {cls.__name__ for cls in fastai_classification._CLASSIFICATION_LOSSES} == EXPECTED


def test_all_is_in_sync_with_the_registered_tuple() -> None:
    """``scan_module`` filters by ``__module__`` — ``fastai.losses`` for these — so the
    ``__all__``-gated pass is the ONLY thing that surfaces them to a discovery consumer."""
    assert sorted(fastai_classification.__all__) == sorted(EXPECTED)


def test_all_is_a_literal_list() -> None:
    """A comprehension would keep the runtime behaviour and lose it for static analysers."""
    source = inspect.getsource(fastai_classification)
    assert "__all__ = [\n" in source and "for " not in source.split("__all__ = [")[1].split("]")[0]


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_every_loss_is_registered_as_a_torch_classification_loss(name: str) -> None:
    """``framework="torch"``, not ``"fastai"``: the axis names the API a class can be WIRED TO,
    and these take torch tensors and return a torch scalar — a Lightning ``training_step``
    consumes them exactly as a fastai ``Learner`` does (pinned by the round-trip below)."""
    entry = get_registry().get_class(name, task="classification", role="loss", framework="torch")
    assert entry is not None
    assert entry.__module__.startswith("fastai.")


def test_every_registered_loss_is_zero_arg_constructible_and_callable() -> None:
    """A picker offers these as options, so each has to build without arguments.

    ``isinstance(..., nn.Module)`` is deliberately NOT the check: the ``*Flat`` family are
    ``fastai.losses.BaseLoss`` objects, plain callables that hold the real torch loss in
    ``.func`` (``CrossEntropyLossFlat().func`` is a ``torch.nn.CrossEntropyLoss``). The
    non-flat ``FocalLoss`` / ``LabelSmoothingCrossEntropy`` ARE modules. What every consumer
    actually needs — a Lightning ``training_step`` as much as a fastai ``Learner`` — is
    ``loss(preds, target) -> scalar``, so that is what is pinned.
    """
    for cls in fastai_classification._CLASSIFICATION_LOSSES:
        loss = cls()
        assert callable(loss)
        assert hasattr(loss, "to"), "must follow the model to its device"


def test_the_default_single_label_loss_consumes_logits_and_int64_ids() -> None:
    """No ``from_logits`` question on this side — the ``*Flat`` losses take raw scores, which is
    what the bare-linear head emits."""
    loss = fastai_classification.CrossEntropyLossFlat()
    value = loss(torch.randn(4, 3), torch.tensor([0, 1, 2, 0]))

    assert value.ndim == 0 and torch.isfinite(value)


def test_the_default_multilabel_loss_consumes_a_float_multi_hot() -> None:
    loss = fastai_classification.BCEWithLogitsLossFlat()
    value = loss(torch.randn(4, 3), torch.zeros(4, 3))

    assert value.ndim == 0 and torch.isfinite(value)


def test_fastai_metrics_are_functions_which_is_why_none_are_registered() -> None:
    """Pins the REASON, not just the absence. If fastai ever ships metric classes this fails,
    which is the prompt to register them and drop the string-name path from the docs."""
    from fastai import metrics as fastai_metrics

    for name in ("accuracy", "accuracy_multi", "error_rate", "top_k_accuracy"):
        assert inspect.isfunction(getattr(fastai_metrics, name)), f"{name} became a class — revisit"
    assert not get_registry().list_classes(task="classification", role="metric", framework="fastai")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
