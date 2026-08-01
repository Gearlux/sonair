"""``TorchConvNet`` — the shared conv spec in ``torch.nn``.

The torch entry in the cross-framework comparison :mod:`sonair.convnet.spec` describes, and the
opposite kind of model from the zoo wrapper it sits beside: :class:`sonair.timm_model.TimmModel`
reaches ~1000 pretrained backbones, this one wraps nothing at all. For real work prefer the zoo
wrapper — a small net trained from scratch loses to a pretrained backbone every time — but five
zoo models would have compared five different networks instead of five frameworks.

It serves BOTH torch backends. ``LightningClassifier`` and ``FastaiClassifier`` differ in the
training loop, not in what a model is, so each takes any ``torch.nn.Module`` and this is one —
the same reason they already share ``TimmModel``.

**This is the one channels-FIRST implementation, and the ``input_shape`` knob still reads
``(H, W, C)``.** The other four consume NHWC because keras, tf, flax and mlx all default to it;
torch is NCHW. Keeping the constructor knob in ``(H, W, C)`` means one config line —
``input_shape: [224, 224, 3]`` — drives every implementation, and only ``C`` affects the
parameter count anyway (the network is fully convolutional up to the global pool). What changes
is the TENSOR: ``forward`` takes ``[N, C, H, W]``, which is exactly what the torch preprocessing
chain already produces, since it ends in ``ToTensor`` transposing HWC to CHW. So the torch
config blocks need no new op — they are the ones that were already correct.

It IS a ``torch.nn.Module``, the same decision as ``TimmModel`` being one and its three siblings
being a ``keras.Model`` / ``nnx.Module`` / ``mlx.nn.Module``: the classifier drives whatever is
wired into its ``model`` slot without asking what it got.
"""

from typing import List, Optional, Tuple, cast

import torch
from confluid import configurable
from loggair import get_logger
from torch import Tensor, nn

from sonair.convnet.spec import CONV_WIDTHS, INPUT_SHAPE, KERNEL_SIZE, POOL_SIZE

logger = get_logger(__name__)

__all__ = ["TorchConvNet"]


def _build(num_classes: int, in_channels: int, widths: Tuple[int, ...], dropout: float) -> nn.Module:
    """The spec as a ``Sequential``: conv-BN-ReLU-pool blocks → GAP → dropout → dense logits.

    Written out rather than borrowed from ``torchvision`` — the point of the spec is that all
    five implementations are the same network, which only holds if each one says so in its own
    framework's primitives.
    """
    layers: List[nn.Module] = []
    channels = in_channels
    for width in widths:
        layers += [
            nn.Conv2d(channels, width, KERNEL_SIZE, padding="same"),
            nn.BatchNorm2d(width),
            nn.ReLU(),
            nn.MaxPool2d(POOL_SIZE),
        ]
        channels = width
    # `AdaptiveAvgPool2d(1)` + `Flatten` IS the global average pool the other four spell as a
    # mean over the spatial axes — stated in torch's vocabulary so a reader recognises it.
    layers += [nn.AdaptiveAvgPool2d(1), nn.Flatten()]
    if dropout:
        layers.append(nn.Dropout(dropout))
    # Raw logits, no activation — like every other head in this project, so the same
    # `CrossEntropyLoss` (which applies its own log-softmax) applies.
    layers.append(nn.Linear(channels, num_classes))
    return nn.Sequential(*layers)


@configurable(task="classification", role="model", lazy=True, framework="torch")
class TorchConvNet(nn.Module):
    """The shared conv classifier (:mod:`sonair.convnet.spec`), implemented in ``torch.nn``.

    Args:
        num_classes: Width of the classification head. Sentinel ``0`` is invalid but permitted
            so the class is zero-arg constructible; a classifier flows the dataset-derived
            count in at run time, so it is not written in YAML.
        input_shape: ``(H, W, C)`` — the SAME channels-last knob the other four take, so one
            config line drives any of them. The tensor ``forward`` consumes is ``[N, C, H, W]``;
            only ``C`` is read here.
        widths: Channels per conv block. Changing it breaks comparability with the other
            implementations — see :mod:`sonair.convnet.spec`.
        dropout: Dropout before the head; ``0.0`` disables it.
        seed: Seed for the parameter initializers, so a config reproduces its weights.
    """

    def __init__(
        self,
        num_classes: int = 0,
        input_shape: Tuple[int, int, int] = INPUT_SHAPE,
        widths: Tuple[int, ...] = CONV_WIDTHS,
        dropout: float = 0.0,
        seed: int = 0,
    ) -> None:
        # `nn.Module.__init__` first — a torch module refuses submodule assignment before it.
        super().__init__()
        # Lazy: store only. No parameter allocation — the workspace zero-arg mandate.
        self.num_classes = num_classes
        self.input_shape = tuple(input_shape)
        self.widths = tuple(widths)
        self.dropout = dropout
        self.seed = seed
        # Public submodule name, NOT `_net`, for the reason `TimmModel` uses `classifier`: a
        # state_dict saved through marainer's Lightning wrapper round-trips as `net.*`, and a
        # leading underscore reads as "private" for something checkpoints address by name.
        self.net: Optional[nn.Module] = None

    def solidify(self) -> nn.Module:
        """Build (once) and return the live network — confluid's ``flow()``-time hook.

        Required configuration is validated HERE rather than in ``__init__``, which is what
        keeps zero-arg construction working. ``flow()`` calls this, so ``self.parameters()`` is
        populated before ``configure_optimizers`` reads it.
        """
        if self.net is None:
            if not self.num_classes:
                raise ValueError(
                    "TorchConvNet.num_classes is not set — a classifier flows the dataset-derived "
                    "count in at run time, or set it explicitly."
                )
            torch.manual_seed(self.seed)
            self.net = _build(
                num_classes=self.num_classes,
                in_channels=int(self.input_shape[-1]),
                widths=self.widths,
                dropout=self.dropout,
            )
            logger.info(f"Built the shared conv spec in torch.nn ({self.num_classes} classes, widths={self.widths}).")
        return cast(nn.Module, self.net)

    @property
    def backbone(self) -> nn.Module:
        """The live network, built on first access — the ergonomic twin of ``TimmModel.backbone``.

        ``flow()`` never touches property getters (confluid's lazy-init rule 3), which is why
        :meth:`solidify` exists as a method as well; this is what makes a directly constructed,
        never-flowed instance usable.
        """
        return self.solidify()

    def forward(self, images: Tensor) -> Tensor:
        """Per-image class logits from an ``[N, C, H, W]`` batch — channels FIRST, torch's layout."""
        return cast(Tensor, self.backbone(images))

    @property
    def parameter_count(self) -> int:
        """Trainable parameters — the number :mod:`sonair.convnet.spec` predicts.

        BatchNorm's running mean/variance are torch BUFFERS rather than parameters, so they are
        excluded by construction here — unlike the MLX implementation, which has to filter them
        out by name because MLX keeps them in the same tree as the weights.
        """
        return int(sum(p.numel() for p in self.backbone.parameters() if p.requires_grad))

    def __repr__(self) -> str:
        """Short and honest before the build — the default would print an empty module."""
        if self.net is None:
            return f"TorchConvNet(num_classes={self.num_classes}, widths={self.widths}, unbuilt)"
        return cast(str, super().__repr__())
