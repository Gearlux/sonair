"""The ONE conv-classifier architecture, written once so five frameworks can implement it.

Every other model in sonair wraps somebody's zoo — timm, ``keras.applications``, the Hub. This
is the opposite: a small network specified here and implemented natively in torch, Keras,
TensorFlow, Flax and MLX. It exists because the zoos do not line up. There is no Flax
efficientvit and no MLX one, so a cross-framework comparison built on zoos compares five
different networks and tells you nothing about the frameworks. Implementing ONE architecture
five times makes the comparison mean something.

**The architecture**::

    input (H, W, C)
      → for w in widths:  Conv 3x3 (w, padding=SAME) → BatchNorm → ReLU → MaxPool 2x2
      → GlobalAveragePool
      → Dropout(dropout)          # 0.0 disables
      → Dense(num_classes)        # raw logits, NO activation

The head emits raw logits like every other model in this project, so the same
``from_logits=True`` losses and logit-space metric thresholds apply.

**Layout is per-implementation, and only the tensor differs.** Keras, TensorFlow, Flax and MLX
all default to NHWC, so those four are channels-LAST and consume the Keras-style ``preprocess``
chain unchanged; torch is NCHW, so :class:`~sonair.convnet.torch_convnet.TorchConvNet` takes
``[N, C, H, W]`` and consumes the torch chain, which already ends in a ``ToTensor`` that
transposes HWC to CHW. The ``input_shape`` KNOB stays ``(H, W, C)`` in all five so one config
line drives any of them, and only ``C`` reaches the parameter count.

**The constants below are the single source of truth.** Each implementation imports them for
its constructor defaults, so "same architecture" is true by construction for the knobs rather
than by five copies agreeing. The LAYER ORDER is prose — five different APIs cannot share
code — and is held together instead by a test that asserts all five report the same trainable
parameter count. That count is what makes "same module" checkable:

    2*3*3*C_in*w + 2w  per conv block (weights + bias + BN scale + BN bias), then 128*C + C

so the defaults below give ``93,696 + 129 * num_classes`` trainable parameters.

Implementations, one per sibling module: :class:`sonair.convnet.torch_convnet.TorchConvNet`
(shared by the lightning and fastai backends — fastai IS torch),
:class:`sonair.convnet.keras_convnet.KerasConvNet`,
:class:`sonair.convnet.tensorflow_convnet.TensorFlowConvNet`,
:class:`sonair.convnet.flax_convnet.FlaxConvNet` and
:class:`sonair.convnet.mlx_convnet.MlxConvNet`.
"""

from typing import Tuple

__all__ = ["CONV_WIDTHS", "KERNEL_SIZE", "POOL_SIZE", "INPUT_SHAPE", "trainable_parameter_count"]

#: Channels per conv block. Three blocks, doubling — small enough to train in seconds on CPU,
#: deep enough that BatchNorm and pooling actually matter to the comparison.
CONV_WIDTHS: Tuple[int, ...] = (32, 64, 128)

#: Square conv kernel, ``padding="same"`` so only the pooling changes the spatial size.
KERNEL_SIZE: int = 3

#: Square max-pool window and stride.
POOL_SIZE: int = 2

#: Default input, matching the other backends' ``(H, W, C)`` — channels LAST.
INPUT_SHAPE: Tuple[int, int, int] = (224, 224, 3)


def trainable_parameter_count(
    num_classes: int,
    widths: Tuple[int, ...] = CONV_WIDTHS,
    in_channels: int = 3,
    kernel_size: int = KERNEL_SIZE,
) -> int:
    """The parameter count the spec implies — what every implementation must report.

    Derived from the architecture rather than measured from one framework, so it is a claim
    the four implementations are each checked against instead of one of them being the
    reference. Counts TRAINABLE parameters only: BatchNorm's running mean/variance are state,
    not parameters, and the frameworks disagree about whether to include them.
    """
    total = 0
    channels = in_channels
    for width in widths:
        total += kernel_size * kernel_size * channels * width + width  # conv weights + bias
        total += 2 * width  # BatchNorm scale + bias
        channels = width
    return total + channels * num_classes + num_classes  # dense head
