"""Each model wrapper names the weight artifacts it loads (``matrainer.uri``).

The shared invariant across every wrapper here, and the one worth guarding: **asking never
builds**. These properties read stored configuration, so a run's provenance can be recorded from
a wrapper that has not been solidified — and reading it must not download a zoo checkpoint.
"""

from pathlib import Path
from typing import Any

import pytest
from matrainer.uri import model_refs

from sonair import HuggingFaceModel, TimmModel


def test_a_timm_model_names_the_exact_hub_repo_including_its_weight_tag() -> None:
    """`efficientvit_b0` is the shorthand a config writes; the weights fetched are a tagged repo."""
    refs = model_refs(TimmModel(model_name="efficientvit_b0"))
    assert [r.uri for r in refs] == ["hf://models/timm/efficientvit_b0.r224_in1k"]
    assert refs[0].url == "https://huggingface.co/timm/efficientvit_b0.r224_in1k"
    assert refs[0].role == "pretrained"


def test_a_timm_model_reports_the_backbone_and_the_checkpoint_over_it(tmp_path: Path) -> None:
    """Two artifacts were loaded, so two are recorded — the reason the protocol is plural."""
    checkpoint = tmp_path / "epoch=3.ckpt"
    refs = model_refs(TimmModel(model_name="resnet50", checkpoint_path=str(checkpoint)))
    assert [r.role for r in refs] == ["pretrained", "checkpoint"]
    assert refs[0].uri == "hf://models/timm/resnet50.a1_in1k"
    assert refs[1].uri == checkpoint.resolve().as_uri()


def test_a_from_scratch_timm_model_reports_nothing() -> None:
    """Random weights referenced no artifact; the architecture is configuration, recorded there."""
    assert model_refs(TimmModel(model_name="resnet50", pretrained=False)) == []


def test_an_unknown_timm_name_reports_nothing_rather_than_guessing() -> None:
    """`get_pretrained_cfg` answers None for a name it does not know; a run record must not invent."""
    assert model_refs(TimmModel(model_name="not_a_real_architecture")) == []


def test_an_unconfigured_model_reports_nothing() -> None:
    assert model_refs(TimmModel()) == []


def test_asking_a_timm_model_does_not_build_it() -> None:
    """The property must be usable on a wrapper the runnable has not injected `num_classes` into."""
    model = TimmModel(model_name="efficientvit_b0")
    assert model_refs(model)  # non-empty
    assert model.classifier is None  # ...and nothing was created or downloaded


def test_a_hub_model_names_its_repository() -> None:
    refs = model_refs(HuggingFaceModel(model_name="google/vit-base-patch16-224"))
    assert [r.uri for r in refs] == ["hf://models/google/vit-base-patch16-224"]
    assert refs[0].url == "https://huggingface.co/google/vit-base-patch16-224"


def test_an_unpretrained_hub_model_reports_nothing() -> None:
    assert model_refs(HuggingFaceModel(model_name="google/vit-base-patch16-224", pretrained=False)) == []


def test_asking_a_hub_model_does_not_build_it() -> None:
    model = HuggingFaceModel(model_name="google/vit-base-patch16-224")
    assert model_refs(model)
    assert model._model is None


def test_the_convnets_report_nothing_and_need_no_code() -> None:
    """They carry no external weights, so the protocol answers empty without them implementing it."""
    from sonair.convnet.torch_convnet import TorchConvNet

    assert model_refs(TorchConvNet(input_shape=(32, 32, 3), num_classes=2)) == []


@pytest.mark.skipif(pytest.importorskip is None, reason="unreachable")
def test_a_keras_model_names_the_registry_and_entry() -> None:
    """keras.applications weights come from library release assets — a registry, but no page."""
    pytest.importorskip("keras")
    from sonair import KerasModel

    refs = model_refs(KerasModel(model_name="MobileNetV3Small", pretrained=True))
    assert [r.uri for r in refs] == ["keras://applications/MobileNetV3Small?weights=imagenet"]
    assert refs[0].url is None  # no repository page to open; none is invented


def test_an_unpretrained_keras_model_reports_nothing() -> None:
    pytest.importorskip("keras")
    from sonair import KerasModel

    assert model_refs(KerasModel(model_name="MobileNetV3Small", pretrained=False)) == []


def test_a_saved_model_is_an_artifact_brought_in_rather_than_a_checkpoint(tmp_path: Path) -> None:
    """The roles distinguish "from outside the run" from "from a previous run of ours"."""
    pytest.importorskip("tensorflow")
    from sonair import TensorFlowModel

    refs = model_refs(TensorFlowModel(saved_model=str(tmp_path / "exported")))
    assert [r.role for r in refs] == ["pretrained"]
    assert refs[0].uri == (tmp_path / "exported").resolve().as_uri()


def test_a_model_reached_through_the_runnables_slot_is_found() -> None:
    """A consumer asks the runnable; the walk follows `.model` to the wrapper underneath."""

    class Runnable:
        def __init__(self, model: Any) -> None:
            self.model = model

    assert model_refs(Runnable(TimmModel(model_name="resnet50"))) == model_refs(TimmModel(model_name="resnet50"))
