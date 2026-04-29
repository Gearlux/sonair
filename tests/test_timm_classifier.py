"""Tests for :class:`sonair.timm_classifier.TimmClassifierModel`."""

from pathlib import Path

import pytest
import torch

from sonair.timm_classifier import TimmClassifierForPrediction, TimmClassifierModel

# efficientvit_b0's native input is 224x224. Use a smaller backbone that
# tolerates arbitrary sizes for fast tests; resnet18 with 64x64 keeps the
# whole suite under a second per case.
TEST_MODEL = "resnet18"
SIZE = 64


def _make_model(num_classes: int = 4, **kwargs: object) -> TimmClassifierModel:
    """Build a tiny offline model — pretrained=False avoids any network IO."""
    return TimmClassifierModel(
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


def test_forward_supports_non_default_in_channels() -> None:
    model = TimmClassifierModel(
        model_name=TEST_MODEL,
        num_classes=2,
        pretrained=False,
        in_channels=1,
    )
    logits = model(torch.rand(2, 1, SIZE, SIZE))
    assert logits.shape == (2, 2)


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
    with pytest.raises(ValueError, match="num_classes"):
        TimmClassifierModel(model_name=TEST_MODEL, num_classes=0, pretrained=False)


# --- Checkpoint round-trip -------------------------------------------------


def test_checkpoint_round_trip_via_init(tmp_path: Path) -> None:
    """Saving a state_dict and reloading via checkpoint_path produces identical weights."""
    src = _make_model(num_classes=4)
    ckpt_path = tmp_path / "model.pt"
    torch.save(src.state_dict(), ckpt_path)

    dst = TimmClassifierModel(
        model_name=TEST_MODEL,
        num_classes=4,
        pretrained=False,
        checkpoint_path=str(ckpt_path),
    )
    for (k_src, v_src), (k_dst, v_dst) in zip(src.state_dict().items(), dst.state_dict().items()):
        assert k_src == k_dst
        assert torch.equal(v_src, v_dst)


def test_checkpoint_strips_lightning_model_prefix(tmp_path: Path) -> None:
    """A Lightning-style ``state_dict`` with ``model.`` keys loads cleanly.

    This is the contract the Marainer Trainer relies on — Lightning wraps the
    inner model as ``self.model = ...``, and on save every key gets a
    ``"model."`` prefix. Our checkpoint loader must strip exactly one such
    prefix and leave keys like ``classifier.stem.0.weight`` matching the
    wrapper layout.
    """
    src = _make_model(num_classes=4)
    src_state = src.state_dict()
    lightning_state = {f"model.{k}": v for k, v in src_state.items()}
    ckpt_path = tmp_path / "lightning.ckpt"
    torch.save({"state_dict": lightning_state, "epoch": 1}, ckpt_path)

    dst = TimmClassifierModel(
        model_name=TEST_MODEL,
        num_classes=4,
        pretrained=False,
        checkpoint_path=str(ckpt_path),
    )
    for k in src_state:
        assert torch.equal(src_state[k], dst.state_dict()[k])


def test_missing_checkpoint_raises() -> None:
    with pytest.raises(FileNotFoundError):
        TimmClassifierModel(
            model_name=TEST_MODEL,
            num_classes=4,
            pretrained=False,
            checkpoint_path="/nonexistent/path/model.ckpt",
        )


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


# --- TimmClassifierForPrediction -------------------------------------------


def _make_pred_model(num_classes: int = 4) -> TimmClassifierForPrediction:
    return TimmClassifierForPrediction(
        model_name=TEST_MODEL,
        num_classes=num_classes,
        pretrained=False,
    )


def test_prediction_class_emits_one_dict_per_sample() -> None:
    """The prediction wrapper turns a [N, C] batch into a list of N dicts."""
    model = _make_pred_model(num_classes=4).eval()
    out = model(torch.rand(3, 3, SIZE, SIZE))
    assert isinstance(out, list)
    assert len(out) == 3
    for entry in out:
        assert set(entry.keys()) == {"logits", "probs", "class_idx"}
        assert entry["logits"].shape == (1, 4)
        assert entry["probs"].shape == (1, 4)
        # softmax row sums to 1.
        assert float(entry["probs"].sum()) == pytest.approx(1.0, abs=1e-5)


def test_prediction_class_loads_checkpoint_from_base_class(tmp_path: Path) -> None:
    """A checkpoint saved by TimmClassifierModel must load cleanly into TimmClassifierForPrediction.

    Both classes share self.classifier, so the state_dict is structurally
    identical — this is the contract that lets train_*.yaml and predict_*.yaml
    pass the same checkpoint between them.
    """
    src = TimmClassifierModel(model_name=TEST_MODEL, num_classes=4, pretrained=False)
    ckpt_path = tmp_path / "model.pt"
    torch.save(src.state_dict(), ckpt_path)

    dst = TimmClassifierForPrediction(
        model_name=TEST_MODEL,
        num_classes=4,
        pretrained=False,
        checkpoint_path=str(ckpt_path),
    )
    # Weights match → predictions are deterministic given the same input.
    images = torch.rand(2, 3, SIZE, SIZE)
    src.eval()
    dst.eval()
    src_logits = src(images)
    dst_outputs = dst(images)
    for i in range(2):
        assert torch.allclose(src_logits[i : i + 1], dst_outputs[i]["logits"])
