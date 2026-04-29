"""Tests for :func:`sonair.checkpoint.load_lightning_checkpoint`."""

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from sonair.checkpoint import load_lightning_checkpoint


class _Inner(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 2, bias=False)


def _make_lightning_ckpt(path: Path, weights: torch.Tensor) -> None:
    torch.save({"state_dict": {"model.linear.weight": weights}, "epoch": 1}, path)


def _make_plain_state_dict(path: Path, weights: torch.Tensor) -> None:
    torch.save({"linear.weight": weights}, path)


def test_strips_lightning_model_prefix(tmp_path: Path) -> None:
    weights = torch.full((2, 4), 7.0)
    ckpt = tmp_path / "lightning.ckpt"
    _make_lightning_ckpt(ckpt, weights)

    inner = _Inner()
    inner.linear.weight.data.zero_()
    load_lightning_checkpoint(inner, str(ckpt))
    assert torch.equal(inner.linear.weight.data, weights)


def test_accepts_plain_state_dict(tmp_path: Path) -> None:
    weights = torch.full((2, 4), 3.0)
    ckpt = tmp_path / "plain.pt"
    _make_plain_state_dict(ckpt, weights)

    inner = _Inner()
    load_lightning_checkpoint(inner, str(ckpt))
    assert torch.equal(inner.linear.weight.data, weights)


def test_strict_true_raises_on_mismatch(tmp_path: Path) -> None:
    ckpt = tmp_path / "extra.ckpt"
    torch.save(
        {
            "state_dict": {
                "model.linear.weight": torch.zeros(2, 4),
                "model.does_not_exist": torch.zeros(1),
            }
        },
        ckpt,
    )
    with pytest.raises(RuntimeError, match="(?i)unexpected"):
        load_lightning_checkpoint(_Inner(), str(ckpt), strict=True)


def test_strict_false_tolerates_extra_keys(tmp_path: Path) -> None:
    ckpt = tmp_path / "extra.ckpt"
    torch.save(
        {
            "state_dict": {
                "model.linear.weight": torch.full((2, 4), 5.0),
                "model.does_not_exist": torch.zeros(1),
            }
        },
        ckpt,
    )
    inner = _Inner()
    load_lightning_checkpoint(inner, str(ckpt), strict=False)
    assert torch.equal(inner.linear.weight.data, torch.full((2, 4), 5.0))


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_lightning_checkpoint(_Inner(), str(tmp_path / "nope.ckpt"))
