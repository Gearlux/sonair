"""timm-backed image-classification wrapper for the Marainer Trainer flow.

Wraps :func:`timm.create_model` into a plain :class:`torch.nn.Module` whose
``forward(images) -> logits`` signature plugs straight into the **unchanged**
base :class:`marainer.trainer.Trainer`. Marainer already calls
``preds = self(batch.input)`` and ``loss = self.loss_fn(preds, batch.target)``
— for classification that's all the glue we need. No bundled batch type, no
step-output dict, no per-mode forward divergence.

The wrapper exists for three reasons:

1. **Confluid configurability**: ``@configurable`` exposes every timm
   constructor knob (model name, pretrained flag, in-channels, drop rates)
   for YAML wiring while keeping a sensible default that matches the
   spectrogram-classification baseline (``efficientvit_b0`` + ImageNet).
2. **Checkpoint loading**: ``checkpoint_path`` strips Lightning's ``"model."``
   prefix via :func:`sonair.checkpoint.load_lightning_checkpoint`, so a
   marainer-trained model resumes / evaluates cleanly without forcing the
   caller to know about Lightning's wrapping.
3. **Generic across timm's zoo**: ``model_name`` accepts any of timm's
   ~1000 backbones — efficientvit, resnet, vit, swin, convnext, mobilenetv3,
   etc. — so a single wrapper covers the whole library. timm's per-model
   resolution requirements are enforced inside :func:`timm.create_model`,
   not here.

Typical wiring (YAML)::

    model: !class:sonair.timm_classifier.TimmClassifierModel
      model_name: efficientvit_b0
      num_classes: 37
      pretrained: true
    loss_fn: !class:sonair.classification.ClassificationLoss
      label_smoothing: 0.1
    val_metrics: !class:torchmetrics.classification.MulticlassAccuracy
      num_classes: 37
"""

from typing import List, Optional, cast

import timm
import torch
import torch.nn as nn
from confluid import configurable
from logflow import get_logger
from marainer.outputs import ClassificationOutput, classification_output
from torch import Tensor

from sonair.checkpoint import load_lightning_checkpoint

logger = get_logger(__name__)


@configurable
class TimmClassifierModel(nn.Module):
    """timm-backed image-classification model wrapped for the Marainer Trainer flow.

    ``forward(images) -> logits`` only — classification models compute loss
    outside the model, so the Trainer's standard ``loss_fn(preds, target)``
    call works without a batch wrapper.

    Args:
        model_name: timm model identifier — ``"efficientvit_b0"``,
            ``"resnet50"``, ``"vit_base_patch16_224"``, ``"swin_tiny_patch4_window7_224"``,
            ``"convnext_tiny"``, ``"mobilenetv3_small_100"``, etc. See
            ``timm.list_models()`` for the full list. Defaults to
            ``"efficientvit_b0"`` — fast modern ViT-CNN hybrid that handles
            spectrograms well at 224×224.
        num_classes: Output class count. timm replaces the original ImageNet
            head with a fresh linear layer of this size whenever
            ``num_classes`` differs from the pretrained checkpoint's class
            count. Defaults to ``1000`` (ImageNet-1k passthrough).
        pretrained: Download timm's pretrained weights for ``model_name`` on
            first use. Defaults to ``True``. Pass ``False`` for offline CI or
            from-scratch training (parallel to deltaid's
            ``encoder_weights="imagenet"``/``None`` knob).
        in_channels: Input channel count. Defaults to ``3`` (RGB spectrogram
            images). timm rebuilds the stem conv when this differs from the
            backbone's native channel count.
        drop_rate: Final-layer dropout probability. ``0.0`` (default) is what
            most timm recipes use; bump to ``0.1``–``0.3`` if overfitting.
        drop_path_rate: Stochastic depth probability. Only honored by
            backbones that support it (most modern ones do); silently ignored
            elsewhere. ``0.0`` (default) is fine for short fine-tuning runs.
        checkpoint_path: Optional path to a Lightning ``.ckpt`` (or plain
            torch state-dict) saved by a previous training run. Loaded after
            the pretrained weights so checkpoint values win. ``"model."``
            prefix from Lightning wrapping is stripped automatically.
        checkpoint_strict: Forwarded to ``load_state_dict(strict=...)``.
            Defaults to ``False`` so e.g. a re-trained classifier head with
            a different ``num_classes`` doesn't reject the whole load.
    """

    def __init__(
        self,
        model_name: str = "efficientvit_b0",
        num_classes: int = 1000,
        pretrained: bool = True,
        in_channels: int = 3,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        checkpoint_path: Optional[str] = None,
        checkpoint_strict: bool = False,
    ) -> None:
        super().__init__()
        if num_classes < 1:
            raise ValueError(f"num_classes must be >= 1, got {num_classes}.")
        self.model_name = model_name
        self.num_classes = int(num_classes)
        self.pretrained = bool(pretrained)
        self.in_channels = int(in_channels)
        # Stored as ``self.classifier`` (not ``self.model``) so a state_dict
        # saved through marainer's Lightning wrapper — which adds a leading
        # ``model.`` prefix — round-trips cleanly. ``load_lightning_checkpoint``
        # strips that single ``model.`` prefix, leaving keys like
        # ``classifier.stem.0.weight`` that match this wrapper's layout. If
        # we used ``self.model``, the stripping would also chew through our
        # own attribute name, producing keys (``stem.0.weight``) that don't
        # match anything.
        # Only forward drop_path_rate when non-default — timm does not filter
        # unknown kwargs and not every backbone exposes it (e.g. efficientvit_b0
        # rejects drop_path_rate at __init__). drop_rate is universal.
        extra_kwargs: dict = {}
        if drop_path_rate > 0.0:
            extra_kwargs["drop_path_rate"] = drop_path_rate
        self.classifier = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=self.num_classes,
            in_chans=self.in_channels,
            drop_rate=drop_rate,
            **extra_kwargs,
        )
        if checkpoint_path:
            load_lightning_checkpoint(self, checkpoint_path, strict=checkpoint_strict)

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
            raise ValueError(f"TimmClassifierModel expects 4D [N,C,H,W] images, got shape {tuple(images.shape)}.")
        n, c, _, _ = images.shape
        if c != self.in_channels:
            raise ValueError(
                f"TimmClassifierModel was built for in_channels={self.in_channels}, " f"got images with {c} channels."
            )

        logits = cast(Tensor, self.classifier(images))

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


@configurable
class TimmClassifierForPrediction(TimmClassifierModel):
    """:class:`TimmClassifierModel` variant that emits per-sample dicts.

    Use this class **only** in :class:`marainer.evaluator.Evaluator`'s
    ``predict_*.yaml`` configs where the ``predictions_sink`` expects
    per-sample :class:`~marainer.outputs.ClassificationOutput` dispatch.
    For ``train_*.yaml`` and ``evaluate_*.yaml`` configs where torchmetrics
    consume raw ``[N, C]`` logits, use the base
    :class:`TimmClassifierModel`.

    Why two classes instead of a flag: the base Trainer's
    ``validation_step`` calls ``loss_fn(preds, target)`` with
    ``preds`` shaped like a tensor. If we toggled per-sample output via
    ``self.training``, validation during training would break (loss can't
    consume a list of dicts). Keeping the two cases as distinct classes
    makes the contract explicit at the YAML level.

    Both classes share the same ``self.classifier`` attribute, so a
    ``.ckpt`` saved through training (which uses
    :class:`TimmClassifierModel`) loads cleanly into this class via the
    same ``checkpoint_path`` argument.
    """

    def forward(self, images: Tensor) -> List[ClassificationOutput]:  # type: ignore[override]
        """Run the classifier and emit one :class:`ClassificationOutput` per sample.

        Args:
            images: ``[N, in_channels, H, W]`` float tensor.

        Returns:
            List of length ``N``, each entry a
            :class:`~marainer.outputs.ClassificationOutput` dict carrying
            ``logits``/``probs`` shaped ``[1, C]`` and ``class_idx`` shaped
            ``[1]`` for that sample.
        """
        logits = super().forward(images)  # [N, C]
        n = logits.shape[0]
        return [classification_output(logits[i : i + 1]) for i in range(n)]


__all__ = ["TimmClassifierModel", "TimmClassifierForPrediction"]
