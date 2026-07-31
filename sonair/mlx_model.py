"""``MlxModel`` — the shared conv spec in ``mlx.nn``, the MLX backend's model.

Like the JAX one this wraps no zoo, and for a comparable reason: the only real MLX image zoo,
``mlx-image``, hard-pins ``huggingface_hub==0.24.0``, ``albumentations==1.4.1`` and
``Pillow==10.4.0``. This workspace runs 1.26 / 2.0.8 / 12.2.0, and that hub downgrade is
precisely the recurring skew ``aisland doctor`` checks for — it would break
``import pytorch_lightning`` for every other backend. A shared venv cannot have it, so the MLX
backend implements :mod:`sonair.convnet_spec` natively instead.

MLX is **Apple-silicon only**, which is a property of the framework rather than a choice here:
the extra already carries ``sys_platform == 'darwin' and platform_machine == 'arm64'``, and
this module is only importable where that holds.

Two MLX facts shape the code:

* **Channels-last is native.** ``mlx.nn.Conv2d`` takes ``NHWC`` directly, so this consumes the
  same preprocessing chain as the Keras, TensorFlow and JAX runs with no transpose.
* **Arrays are lazy.** Nothing computes until ``mx.eval`` (or a value is read). That matters in
  the training loop rather than here — see :mod:`sonair.mlx_classifier` — but it is why this
  module never inspects a result to "check" it.
"""

from typing import Any, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
from confluid import configurable
from loggair import get_logger

from sonair.convnet_spec import CONV_WIDTHS, INPUT_SHAPE, KERNEL_SIZE, POOL_SIZE

logger = get_logger(__name__)

__all__ = ["MlxModel"]


class _ConvNet(nn.Module):
    """The spec: conv-BN-ReLU-pool blocks → global average pool → dropout → dense logits."""

    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        widths: Tuple[int, ...],
        dropout: float,
    ) -> None:
        super().__init__()
        channels = in_channels
        convs, norms = [], []
        for width in widths:
            convs.append(nn.Conv2d(channels, width, kernel_size=KERNEL_SIZE, padding=KERNEL_SIZE // 2))
            norms.append(nn.BatchNorm(width))
            channels = width
        self.convs = convs
        self.norms = norms
        self.drop = nn.Dropout(p=dropout)
        self.head = nn.Linear(channels, num_classes)

    def __call__(self, x: Any) -> Any:
        for conv, norm in zip(self.convs, self.norms):
            x = norm(conv(x))
            x = nn.relu(x)
            # MLX ships no max-pool over NHWC in every version, and reshaping to windows is
            # both portable and exactly what pooling is: fold each 2x2 window into an axis and
            # take the max over it.
            batch, height, width_, channels = x.shape
            x = x.reshape(batch, height // POOL_SIZE, POOL_SIZE, width_ // POOL_SIZE, POOL_SIZE, channels)
            x = x.max(axis=(2, 4))
        x = x.mean(axis=(1, 2))  # global average pool over H, W — channels are LAST
        x = self.drop(x)
        # Raw logits, no activation — like every other head in this project.
        return self.head(x)


@configurable(task="classification", role="model", lazy=True, framework="mlx")
class MlxModel:
    """The shared conv classifier (:mod:`sonair.convnet_spec`), implemented in ``mlx.nn``.

    Args:
        num_classes: Width of the classification head. Sentinel ``0`` is invalid but permitted
            so the class is zero-arg constructible — a classifier injects the dataset-derived
            count via ``flow(model, num_classes=N)`` at run time.
        input_shape: ``(H, W, C)`` — channels LAST, which is MLX's native layout.
        widths: Channels per conv block. Changing it breaks comparability with the other
            implementations — see :mod:`sonair.convnet_spec`.
        dropout: Dropout before the head; ``0.0`` disables it.
        seed: Seed for parameter init, so a config reproduces its weights.
    """

    def __init__(
        self,
        num_classes: int = 0,
        input_shape: Tuple[int, int, int] = INPUT_SHAPE,
        widths: Tuple[int, ...] = CONV_WIDTHS,
        dropout: float = 0.0,
        seed: int = 0,
    ) -> None:
        # Lazy: store only. No parameter allocation — the workspace zero-arg mandate.
        self.num_classes = num_classes
        self.input_shape = tuple(input_shape)
        self.widths = tuple(widths)
        self.dropout = dropout
        self.seed = seed
        self._model: Optional[Any] = None

    def solidify(self) -> Any:
        """Build (once) and return the live ``nn.Module`` — confluid's ``flow()``-time hook."""
        if self._model is not None:
            return self._model
        if not self.num_classes:
            raise ValueError(
                "MlxModel.num_classes is not set — a classifier flows the dataset-derived "
                "count in at run time, or set it explicitly."
            )
        mx.random.seed(self.seed)
        self._model = _ConvNet(
            num_classes=self.num_classes,
            in_channels=int(self.input_shape[-1]),
            widths=self.widths,
            dropout=self.dropout,
        )
        logger.info(
            f"Built the shared conv spec in mlx.nn ({self.num_classes} classes, "
            f"widths={self.widths}, device={mx.default_device()})."
        )
        return self._model

    @property
    def model(self) -> Any:
        """The live ``nn.Module``, built on first access.

        The ergonomic accessor for a directly-constructed instance that was never flowed —
        ``flow()`` does not touch property getters, which is why :meth:`solidify` exists too.
        """
        return self.solidify()

    def __call__(self, x: Any, training: bool = False) -> Any:
        """Per-image class logits.

        ``training`` drives dropout and batch-norm through MLX's own module mode rather than a
        per-call flag, which is why it is applied here instead of threaded down.
        """
        model = self.model
        model.train(training)
        return model(x)

    @property
    def parameter_count(self) -> int:
        """Trainable parameters — the number :mod:`sonair.convnet_spec` predicts.

        Counted off the leaves of the parameter tree. MLX keeps batch-norm's running mean and
        variance in that same tree, so they are excluded by NAME — the other three frameworks
        hold them as non-trainable state and the spec counts neither.
        """
        from mlx.utils import tree_flatten

        leaves: Any = tree_flatten(self.model.parameters())
        return int(
            sum(
                value.size
                for path, value in leaves
                if not path.endswith("running_mean") and not path.endswith("running_var")
            )
        )
