"""timm-backed image-classification wrapper for the Marainer Trainer flow.

Wraps :func:`timm.create_model` into a plain :class:`torch.nn.Module` whose
``forward(images) -> logits`` signature plugs straight into the **unchanged**
base :class:`marainer.trainer.LightningTrainer`. Marainer already calls
``preds = self(batch.input)`` and ``loss = self.loss_fn(preds, batch.target)``
— for classification that's all the glue we need. No bundled batch type, no
step-output dict, no per-mode forward divergence.

The wrapper exists for three reasons:

1. **Confluid configurability**: ``@configurable`` exposes every timm
   constructor knob (model name, pretrained flag, in-channels, drop rates)
   for YAML wiring while keeping a sensible default that matches the
   spectrogram-classification baseline (``efficientvit_b0`` + ImageNet).
2. **Checkpoint loading**: ``checkpoint_path`` strips Lightning's ``"model."``
   prefix via :func:`marainer.checkpoint.load_lightning_checkpoint`, so a
   marainer-trained model resumes / evaluates cleanly without forcing the
   caller to know about Lightning's wrapping.
3. **Generic across timm's zoo**: ``model_name`` accepts any of timm's
   ~1000 backbones — efficientvit, resnet, vit, swin, convnext, mobilenetv3,
   etc. — so a single wrapper covers the whole library. timm's per-model
   resolution requirements are enforced inside :func:`timm.create_model`,
   not here.

Typical wiring (YAML) — declare the model ``!lazy:`` so the trainer injects the
dataset-derived ``num_classes`` at flow time (omit it from the config)::

    model: !lazy:sonair.TimmModel
      model_name: efficientvit_b0
      pretrained: true
    loss_fn: !class:torch.nn.CrossEntropyLoss
      label_smoothing: 0.1

(Pin ``num_classes`` explicitly only to override the dataset-derived value.)
"""

from typing import List, Optional, cast

import timm
import torch
import torch.nn as nn
from confluid import configurable
from logflow import get_logger
from marainer.checkpoint import load_lightning_checkpoint
from marainer.outputs import ClassificationOutput, classification_output
from torch import Tensor

logger = get_logger(__name__)


@configurable(task="classification", role="model", lazy=True)
class TimmModel(nn.Module):
    """timm-backed image-classification model wrapped for the Marainer Trainer flow.

    ``forward(images) -> logits`` only — classification models compute loss
    outside the model, so the Trainer's standard ``loss_fn(preds, target)``
    call works without a batch wrapper.

    **Lazy construction (Lazy-Init / Zero-Arg mandate).** ``__init__`` does NO
    functional work — it only stores config. The timm backbone (and any
    pretrained-weight download) is built on first use by :meth:`_ensure_classifier`
    (triggered by :meth:`forward` or the explicit :meth:`realize` hook), so
    ``TimmModel()`` is cheap and does no network I/O. Tagged ``lazy=True`` so a
    config wires it ``!lazy:`` and FluxStudio emits a deferred ``LazyClass`` the
    trainer flows with the derived ``num_classes``. ``model_name`` and
    ``num_classes`` have **no default** (``None``) and are validated lazily when
    the backbone is built — not in ``__init__``.

    The fixed internals ``in_channels=3`` (RGB spectrograms), ``drop_rate=0.0``,
    and non-strict checkpoint load are no longer constructor knobs.

    Args:
        model_name: timm model identifier — ``"efficientvit_b0"``,
            ``"resnet50"``, ``"vit_base_patch16_224"``, ``"swin_tiny_patch4_window7_224"``,
            ``"convnext_tiny"``, ``"mobilenetv3_small_100"``, etc. See
            ``timm.list_models()`` for the full list. **Required to build** —
            ``""`` (the default) means unset; building the backbone with an empty
            ``model_name`` raises. ``efficientvit_b0`` is the spectrogram baseline.
        num_classes: Output class count. timm replaces the original head with a
            fresh linear layer of this size. **Required to build** — ``0`` (the
            default) means unset. In a marainer training run **omit** it in the
            config: the classification trainer derives the count from the dataset
            (:meth:`sonair.hf_dataset.HuggingFaceClassificationDataset.num_classes`,
            which walks the targets) and injects it via ``flow(model, num_classes=…)``.
            Set it explicitly to override the derived value.
        pretrained: Download timm's pretrained weights for ``model_name`` on
            first build. Defaults to ``True``. Pass ``False`` for offline CI or
            from-scratch training.
        checkpoint_path: Optional path to a Lightning ``.ckpt`` (or plain
            torch state-dict) saved by a previous training run, loaded after the
            backbone is built (so checkpoint values win). ``""`` (default) =
            none. The Lightning ``"model."`` prefix is stripped automatically.

    Attributes:
        checkpoint_strict: Whether to enforce strict key matching when loading
            checkpoint. Defaults to ``False`` so incompatible keys (e.g., a
            different ``num_classes``) don't reject the load. Set to ``True``
            to require exact key match.
        drop_path_rate: Drop path rate for stochastic depth. Currently unused
            (no effect on backbone construction). Defaults to ``""``.
        in_channels: Input channel count. Fixed to ``3`` for RGB spectrograms.
            Stored as an attribute for state_dict round-trip compatibility;
            immutable via constructor.
        drop_rate: Dropout rate for the backbone. Fixed to ``0.0``. Stored as
            an attribute for state_dict round-trip compatibility; immutable via
            constructor.
    """

    def __init__(
        self,
        model_name: str = "",
        num_classes: int = 0,
        pretrained: bool = True,
        checkpoint_path: str = "",
    ) -> None:
        super().__init__()
        # Store config ONLY — no functional work (no timm.create_model, no
        # download, no checkpoint load). The backbone is built lazily by
        # _ensure_classifier() on first forward / realize(). model_name and
        # num_classes are validated there, NOT here, so TimmModel() is cheap and
        # zero-arg construction works (Lazy-Init / Zero-Arg mandate).
        self.model_name: str = model_name
        self.num_classes: int = num_classes
        self.pretrained: bool = pretrained
        self.in_channels: int = 3
        self.drop_rate: float = 0.0
        self.drop_path_rate: str = ""
        self.checkpoint_path = checkpoint_path
        self.checkpoint_strict: bool = False

    def _ensure_classifier(self) -> nn.Module:
        """Build the timm backbone on first use (cached), validating lazily.

        Registered under the public submodule name ``classifier`` (NOT a private
        ``_classifier``) so a state_dict saved through marainer's Lightning
        wrapper — which adds a leading ``model.`` prefix that
        :func:`~marainer.checkpoint.load_lightning_checkpoint` strips — round-trips
        to ``classifier.<layer>`` keys, matching this wrapper's layout (and any
        checkpoint written before lazy construction). A property named
        ``classifier`` is deliberately avoided: it would force a ``_classifier``
        backing submodule and rename every state_dict key.
        """
        if "classifier" not in self._modules:
            # Validate the required config lazily (sentinels: "" / 0 mean "unset").
            if not self.model_name:
                raise ValueError("TimmModel.model_name is required to build the backbone (got empty).")
            if self.num_classes < 1:
                raise ValueError(
                    f"TimmModel.num_classes must be >= 1 to build the backbone (got {self.num_classes}) — "
                    "set it explicitly, or let the classification trainer inject it from the dataset."
                )
            self.add_module(
                "classifier",
                timm.create_model(
                    self.model_name,
                    pretrained=self.pretrained,
                    num_classes=self.num_classes,
                    in_chans=self.in_channels,
                    drop_rate=self.drop_rate,
                ),
            )
            if self.checkpoint_path:
                load_lightning_checkpoint(self, self.checkpoint_path, strict=self.checkpoint_strict)
        return cast(nn.Module, self._modules["classifier"])

    def realize(self) -> "TimmModel":
        """Force-build the lazy backbone now, returning self.

        The classification trainer calls this (via ``getattr(model, "realize", …)``)
        right after ``flow(model, num_classes=N)`` so ``self.parameters()`` is
        populated BEFORE Lightning's ``configure_optimizers`` builds the optimizer
        (which would otherwise see an empty parameter set).
        """
        self._ensure_classifier()
        return self

    def forward(self, images: Tensor) -> Tensor:
        """Run the classifier and emit per-image class logits.

        Args:
            images: ``[N, in_channels, H, W]`` float tensor. Spatial dims must
                match whatever the chosen ``model_name`` expects (timm's
                models have per-architecture resolution requirements that
                this wrapper does not double-check — invalid sizes surface as
                an error inside the underlying timm forward).

        Returns:
            ``[N, num_classes]`` raw logits.
        """
        if images.dim() != 4:
            raise ValueError(f"TimmModel expects 4D [N,C,H,W] images, got shape {tuple(images.shape)}.")
        n, c, _, _ = images.shape
        if c != self.in_channels:
            raise ValueError(
                f"TimmModel was built for in_channels={self.in_channels}, " f"got images with {c} channels."
            )

        logits = cast(Tensor, self._ensure_classifier()(images))

        # Per-batch diagnostic: log argmax distribution + logits range.
        # Predicting one class is normal early in training so we don't
        # promote the unanimity case to a warning (unlike the segmentation
        # background-collapse warning) — keep it informational.
        with torch.no_grad():
            pred_classes = logits.argmax(dim=-1)
            unique_pred = sorted({int(v) for v in pred_classes.tolist()})
            logger.trace(
                f"[diag][timm_classifier] batch={n} model={self.model_name!r} "
                f"unique_pred_classes={unique_pred} "
                f"logits_range=[{float(logits.min()):.3f},{float(logits.max()):.3f}]"
            )

        return logits


__all__ = ["TimmModel"]
