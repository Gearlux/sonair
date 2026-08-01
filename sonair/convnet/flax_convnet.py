"""``FlaxConvNet`` — the shared conv spec in ``flax.nnx``, the JAX backend's model.

Unlike every other model wrapper in sonair this wraps no zoo, because there is no Flax one to
wrap: ``transformers`` dropped Flax entirely in v5 (no ``FlaxAutoModelForImageClassification``)
and ``flaxmodels`` last released in 2023. So the JAX backend implements
:mod:`sonair.convnet.spec` natively — see that module for why one architecture written four
times is worth more here than four different zoo models.

It IS an ``nnx.Module``, the same decision as ``KerasModel`` being a ``keras.Model`` and
``TimmModel`` an ``nn.Module``: the classifier drives whatever is wired into its ``model`` slot
without asking what it got, and ``nnx.state`` / ``nnx.Optimizer`` reach the parameters straight
through the wrapper.

The lazy build has a Flax-specific constraint worth knowing before you touch it. An attribute
first assigned ``None`` in ``__init__`` becomes STATIC in nnx's pytree, and assigning a Module
to it later raises *"Cannot assign data value … to static attribute"*. Wrapping the value in
``nnx.data`` is the fix flax itself names, and it is what keeps zero-arg construction (the
workspace lazy-init mandate) compatible with building at ``flow()`` time. Measured, not
guessed — the unwrapped version raises on the first call.
"""

from typing import Any, Optional, Tuple

from confluid import configurable
from flax import nnx
from loggair import get_logger

from sonair.convnet.spec import CONV_WIDTHS, INPUT_SHAPE, KERNEL_SIZE, POOL_SIZE

logger = get_logger(__name__)

__all__ = ["FlaxConvNet"]


class _ConvNet(nnx.Module):
    """The spec itself: conv-BN-ReLU-pool blocks → global average pool → dropout → dense."""

    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        widths: Tuple[int, ...],
        dropout: float,
        seed: int,
    ) -> None:
        rngs = nnx.Rngs(seed)
        channels = in_channels
        self.convs: Any = nnx.List([])
        self.norms: Any = nnx.List([])
        for width in widths:
            self.convs.append(nnx.Conv(channels, width, kernel_size=(KERNEL_SIZE, KERNEL_SIZE), rngs=rngs))
            self.norms.append(nnx.BatchNorm(width, rngs=rngs))
            channels = width
        self.drop = nnx.Dropout(rate=dropout, rngs=rngs)
        self.head = nnx.Linear(channels, num_classes, rngs=rngs)

    def __call__(self, x: Any, training: bool = False) -> Any:
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x)
            # `use_running_average` is the INVERSE of "training": in eval we consume the running
            # statistics rather than the batch's, exactly as keras and torch do.
            x = norm(x, use_running_average=not training)
            x = nnx.relu(x)
            x = nnx.max_pool(x, window_shape=(POOL_SIZE, POOL_SIZE), strides=(POOL_SIZE, POOL_SIZE))
        x = x.mean(axis=(1, 2))  # global average pool over H, W — channels are LAST
        x = self.drop(x, deterministic=not training)
        return self.head(x)


@configurable(task="classification", role="model", lazy=True, framework="jax")
class FlaxConvNet(nnx.Module):
    """The shared conv classifier, implemented in ``flax.nnx``.

    Args:
        num_classes: Width of the classification head. Sentinel ``0`` is invalid but permitted
            so the class is zero-arg constructible — a classifier injects the dataset-derived
            count via ``flow(model, num_classes=N)`` at run time, so it is not written in YAML.
        input_shape: ``(H, W, C)`` — channels LAST, like Keras and unlike the torch backends.
            Only ``C`` affects the parameter count; ``H``/``W`` are documentation here, since
            the network is fully convolutional up to the global pool.
        widths: Channels per conv block. Changing this changes the architecture and therefore
            breaks comparability with the other implementations — see :mod:`sonair.convnet.spec`.
        dropout: Dropout before the head; ``0.0`` disables it.
        seed: PRNG seed for parameter init. JAX makes randomness explicit, so this is a real
            knob rather than a global ``set_seed`` — the same config gives the same weights.
    """

    def __init__(
        self,
        num_classes: int = 0,
        input_shape: Tuple[int, int, int] = INPUT_SHAPE,
        widths: Tuple[int, ...] = CONV_WIDTHS,
        dropout: float = 0.0,
        seed: int = 0,
    ) -> None:
        # Lazy: store only. No parameter allocation — see the module docstring.
        self.num_classes = num_classes
        self.input_shape = tuple(input_shape)
        self.widths = tuple(widths)
        self.dropout = dropout
        self.seed = seed
        self._model: Optional[Any] = None

    def solidify(self) -> Any:
        """Build (once) and return the live inner ``nnx.Module``.

        This is confluid's ``flow()``-time hook as well as the cache: flowing this class with
        ``num_classes=N`` allocates a head of that width immediately. Required configuration is
        validated HERE rather than in ``__init__``, which is what keeps zero-arg construction
        working.
        """
        if self._model is not None:
            return self._model
        if not self.num_classes:
            raise ValueError(
                "FlaxConvNet.num_classes is not set — a classifier flows the dataset-derived "
                "count in at run time, or set it explicitly."
            )
        # `nnx.data` is required, not decorative — see the module docstring.
        self._model = nnx.data(
            _ConvNet(
                num_classes=self.num_classes,
                in_channels=int(self.input_shape[-1]),
                widths=self.widths,
                dropout=self.dropout,
                seed=self.seed,
            )
        )
        logger.info(f"Built the shared conv spec in flax.nnx ({self.num_classes} classes, widths={self.widths}).")
        return self._model

    @property
    def model(self) -> Any:
        """The live inner ``nnx.Module``, built on first access.

        The ergonomic accessor for a directly-constructed instance that was never flowed —
        ``flow()`` does not touch property getters, which is why :meth:`solidify` exists as a
        method as well.
        """
        return self.solidify()

    def __call__(self, x: Any, training: bool = False) -> Any:
        """Per-image class logits — the twin of ``TimmModel.forward``."""
        return self.model(x, training=training)
