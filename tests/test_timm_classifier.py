"""Tests for :class:`sonair.timm_model.TimmModel`."""

from pathlib import Path

import pytest
import torch

from sonair.timm_model import TimmModel

# efficientvit_b0's native input is 224x224. Use a smaller backbone that
# tolerates arbitrary sizes for fast tests; resnet18 with 64x64 keeps the
# whole suite under a second per case.
TEST_MODEL = "resnet18"
SIZE = 64


def _make_model(num_classes: int = 4, **kwargs: object) -> TimmModel:
    """Build a tiny offline model — pretrained=False avoids any network IO."""
    return TimmModel(
        model_name=TEST_MODEL,
        num_classes=num_classes,
        pretrained=False,
        **kwargs,  # type: ignore[arg-type]
    )


# --- Forward shape contract ------------------------------------------------


def test_forward_returns_per_image_logits_with_correct_shape() -> None:
    model = _make_model(num_classes=5)
    images = torch.rand(2, 3, SIZE, SIZE)
    logits = model(images)
    assert logits.shape == (2, 5)
    assert logits.dtype == torch.float32


def test_forward_handles_single_image_batch_in_eval_mode() -> None:
    """Single-image inference is the typical Evaluator path.

    BatchNorm in resnet cannot normalize a batch of 1 in train mode (variance
    is undefined), so the model must be flipped to ``eval()`` for
    single-image forward — exactly what Lightning's ``test`` / ``predict``
    paths do automatically.
    """
    model = _make_model(num_classes=3).eval()
    logits = model(torch.rand(1, 3, SIZE, SIZE))
    assert logits.shape == (1, 3)


# --- Lazy construction -----------------------------------------------------


def test_zero_arg_construction_is_cheap_and_does_no_build() -> None:
    """``TimmModel()`` must be cheap — no backbone, no network (Lazy-Init mandate)."""
    model = TimmModel()
    assert "classifier" not in model._modules  # backbone not built
    assert list(model.parameters()) == []  # nothing materialised yet


def test_solidify_builds_the_backbone() -> None:
    model = TimmModel(model_name=TEST_MODEL, num_classes=4, pretrained=False)
    assert model.classifier is None
    assert "classifier" not in model._modules
    assert model.solidify() is model.classifier  # returns the cached backbone
    assert "classifier" in model._modules
    assert sum(p.numel() for p in model.parameters()) > 0


def test_missing_model_name_raises_at_build() -> None:
    with pytest.raises(ValueError, match="model_name"):
        TimmModel(num_classes=4, pretrained=False).solidify()


def test_backbone_property_builds_and_caches_without_flow() -> None:
    # A directly-constructed (never-flowed) TimmModel materialises its backbone via the property.
    model = TimmModel(model_name=TEST_MODEL, num_classes=4, pretrained=False)
    assert model.classifier is None
    built = model.backbone  # property access triggers the build
    assert "classifier" in model._modules
    assert built is model.classifier
    # The property, solidify(), and the cache all share ONE backbone (build happens at most once).
    assert model.backbone is built
    assert model.solidify() is built


def test_backbone_property_validates_lazily() -> None:
    # Same lazy validation as solidify() — the property is just the cached accessor over it.
    with pytest.raises(ValueError, match="num_classes"):
        _ = TimmModel(model_name=TEST_MODEL, num_classes=0, pretrained=False).backbone


def test_flowing_a_LIVE_model_builds_it() -> None:
    """A model that never went through a marker is still built by ``flow()``.

    This is the confluid contract every backend relies on rather than a timm detail: a model
    wired ``!class:sonair.TimmModel()`` (eager, parens) or constructed in Python reaches a
    trainer already live, and `flow()` fires `solidify()` on its pass-through so the backbone
    exists before an optimizer reads `parameters()`. Without it the failure lands far away, as
    *"optimizer got an empty parameter list"* — which is exactly how it was found.
    """
    from confluid import flow

    live = TimmModel(model_name=TEST_MODEL, num_classes=4, pretrained=False)
    assert list(live.parameters()) == []  # a bare constructor does no functional work

    assert flow(live) is live, "a live object flows to itself"
    assert sum(p.numel() for p in live.parameters()) > 0, "...and is built on the way through"


def test_flow_solidify_false_leaves_a_live_model_unbuilt() -> None:
    """The cheap-introspection path must stay cheap for a live model too."""
    from confluid import flow

    live = TimmModel(model_name=TEST_MODEL, num_classes=4, pretrained=False)

    assert flow(live, solidify=False) is live
    assert list(live.parameters()) == []


# --- Input validation ------------------------------------------------------


def test_rejects_non_4d_input() -> None:
    model = _make_model()
    with pytest.raises(ValueError, match="4D"):
        model(torch.rand(3, SIZE, SIZE))  # missing batch dim


def test_rejects_wrong_channel_count() -> None:
    model = _make_model()  # default in_channels=3
    with pytest.raises(ValueError, match="in_channels"):
        model(torch.rand(2, 1, SIZE, SIZE))


def test_rejects_zero_num_classes() -> None:
    # num_classes=0 is the "unset" sentinel — cheap to construct, raises at build.
    model = TimmModel(model_name=TEST_MODEL, num_classes=0, pretrained=False)
    with pytest.raises(ValueError, match="num_classes"):
        model.solidify()


# --- Checkpoint round-trip -------------------------------------------------


def test_checkpoint_round_trip_via_init(tmp_path: Path) -> None:
    """Saving a state_dict and reloading via checkpoint_path produces identical weights."""
    src = _make_model(num_classes=4)
    src.solidify()  # build so there's a state_dict to save (src stays the TimmModel)
    ckpt_path = tmp_path / "model.pt"
    torch.save(src.state_dict(), ckpt_path)

    dst = TimmModel(
        model_name=TEST_MODEL,
        num_classes=4,
        pretrained=False,
        checkpoint_path=str(ckpt_path),
    )
    dst.solidify()  # build -> loads the checkpoint
    for (k_src, v_src), (k_dst, v_dst) in zip(src.state_dict().items(), dst.state_dict().items()):
        assert k_src == k_dst
        assert torch.equal(v_src, v_dst)


def test_checkpoint_strips_lightning_model_prefix(tmp_path: Path) -> None:
    """A Lightning-style ``state_dict`` with ``model.`` keys loads cleanly.

    This is the contract the Matrainer Trainer relies on — Lightning wraps the
    inner model as ``self.model = ...``, and on save every key gets a
    ``"model."`` prefix. Our checkpoint loader must strip exactly one such
    prefix and leave keys like ``classifier.stem.0.weight`` matching the
    wrapper layout.
    """
    src = _make_model(num_classes=4)
    src.solidify()
    src_state = src.state_dict()
    lightning_state = {f"model.{k}": v for k, v in src_state.items()}
    ckpt_path = tmp_path / "lightning.ckpt"
    torch.save({"state_dict": lightning_state, "epoch": 1}, ckpt_path)

    dst = TimmModel(
        model_name=TEST_MODEL,
        num_classes=4,
        pretrained=False,
        checkpoint_path=str(ckpt_path),
    )
    dst.solidify()
    for k in src_state:
        assert torch.equal(src_state[k], dst.state_dict()[k])


def test_missing_checkpoint_raises(tmp_path: Path) -> None:
    # Use tmp_path so the path is guaranteed not to exist (and not under
    # a non-traversable parent like `/nonexistent` which can exist on
    # some hosts and surface as PermissionError instead of FileNotFoundError).
    missing = tmp_path / "does_not_exist.ckpt"
    model = TimmModel(
        model_name=TEST_MODEL,
        num_classes=4,
        pretrained=False,
        checkpoint_path=str(missing),
    )
    with pytest.raises(FileNotFoundError):
        model.solidify()  # checkpoint loads at build time, not __init__


# --- Differentiability -----------------------------------------------------


def test_logits_are_differentiable() -> None:
    """Confirm the forward path supports backprop end-to-end.

    Use batch=2 — encoder BatchNorm requires more than 1 value per channel
    in train mode (the realistic training path).
    """
    model = _make_model(num_classes=3)
    images = torch.rand(2, 3, SIZE, SIZE, requires_grad=True)
    logits = model(images)
    loss = logits.sum()
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
