"""Load a model's weights from a PyTorch Lightning ``.ckpt`` file.

Lightning saves the wrapping :class:`pytorch_lightning.LightningModule`'s full
state — including the ``model.`` prefix that comes from
``self.model = model`` inside :class:`marainer.trainer.LightningTrainer` /
:class:`marainer.evaluator.LightningEvaluator`. This helper hides that detail so each
classification-model class can load its own weights directly in ``__init__``,
without dragging Marainer into the picture.

A plain ``torch.save(model.state_dict(), path)`` (no Lightning wrapper) is
also tolerated — the helper looks for a top-level ``state_dict`` key first
and otherwise treats the loaded object as the state dict itself.
"""

from typing import Any, Mapping

import torch
import torch.nn as nn
from logflow import get_logger

logger = get_logger(__name__)

_LIGHTNING_PREFIX = "model."


def load_lightning_checkpoint(model: nn.Module, path: str, strict: bool = False) -> None:
    """Load a Lightning-wrapped ``state_dict`` into ``model`` in place.

    Strips the ``"model."`` prefix Lightning's wrapper adds. Tolerates
    plain ``torch.save(model.state_dict(), ...)`` saves too — if no
    top-level ``state_dict`` key is present, the loaded object is used
    directly.

    Args:
        model: The target ``nn.Module``.
        path: Filesystem path to the ``.ckpt`` (or plain ``.pt`` state-dict).
        strict: Forwarded to :meth:`torch.nn.Module.load_state_dict`.
            Defaults to ``False`` so e.g. a re-trained classifier head with a
            different ``num_classes`` doesn't reject the entire load.

    Raises:
        FileNotFoundError: ``path`` doesn't exist.
        RuntimeError: ``strict=True`` and any keys mismatch.
    """
    logger.info(f"{type(model).__name__}: loading checkpoint from {path}")
    raw: Any = torch.load(path, map_location="cpu", weights_only=False)
    state_dict: Mapping[str, Any]
    if isinstance(raw, Mapping) and "state_dict" in raw:
        state_dict = raw["state_dict"]
    else:
        state_dict = raw

    stripped = {
        (k[len(_LIGHTNING_PREFIX) :] if k.startswith(_LIGHTNING_PREFIX) else k): v for k, v in state_dict.items()
    }
    missing, unexpected = model.load_state_dict(stripped, strict=strict)
    if missing:
        logger.info(f"{type(model).__name__}: checkpoint missing {len(missing)} key(s) (e.g. {missing[:3]})")
    if unexpected:
        logger.info(
            f"{type(model).__name__}: checkpoint had {len(unexpected)} unexpected key(s) (e.g. {unexpected[:3]})"
        )


__all__ = ["load_lightning_checkpoint"]
