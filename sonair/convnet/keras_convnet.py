"""``KerasConvNet`` — the shared conv spec in ``keras.layers``.

This is the Keras entry in the four-way comparison :mod:`sonair.convnet.spec` describes, and it
is the opposite kind of model from :mod:`sonair.keras_model`: that one wraps a zoo, this one
wraps nothing. For real work prefer the zoo wrapper — a small net trained from scratch loses to
a pretrained backbone every time — but four zoo models would have compared four different
networks instead of four frameworks, which is why this exists.

Being written in ``keras.layers`` it runs on whichever engine Keras is bound to, so both the
``keras`` and ``tensorflow`` backends can wire it. That is also why it is not called
``TensorFlowConvNet``: it is not TensorFlow-specific. The genuinely TF-native implementation,
in raw ``tf.nn`` ops, is its sibling :mod:`sonair.convnet.tensorflow_convnet`.

It IS a ``keras.Model``, the same decision as ``KerasModel`` being one — see that module for why
the base class rather than delegation, and for the ``build()`` constraint the lazy build has to
respect.
"""

from typing import Any, Optional, Tuple

from confluid import configurable
from loggair import get_logger

from sonair._keras import keras, keras_backend
from sonair.convnet.spec import CONV_WIDTHS, INPUT_SHAPE, KERNEL_SIZE, POOL_SIZE

logger = get_logger(__name__)

__all__ = ["KerasConvNet"]


@configurable(task="classification", role="model", lazy=True, framework="keras")
class KerasConvNet(keras.Model):
    """The shared conv classifier (:mod:`sonair.convnet.spec`), implemented in ``keras.layers``.

    Args:
        num_classes: Width of the classification head. Sentinel ``0`` is invalid but permitted
            so the class is zero-arg constructible; a classifier flows the dataset-derived
            count in at run time.
        input_shape: ``(H, W, C)`` — channels LAST.
        widths: Channels per conv block. Changing it breaks comparability with the other
            implementations — see :mod:`sonair.convnet.spec`.
        dropout: Dropout before the head; ``0.0`` disables it.
        seed: Seed applied before the build, so a config reproduces its weights — the same knob
            its four sibling implementations carry, and the reason this engine was the one whose
            sweep rows moved between identical runs.
    """

    def __init__(
        self,
        num_classes: int = 0,
        input_shape: Tuple[int, int, int] = INPUT_SHAPE,
        widths: Tuple[int, ...] = CONV_WIDTHS,
        dropout: float = 0.0,
        seed: int = 0,
    ) -> None:
        # Partial: store only, and `super().__init__()` first — a `keras.Layer` refuses attribute
        # assignment before it. Same contract as `KerasModel`.
        super().__init__()
        self.num_classes = num_classes
        self.input_shape = tuple(input_shape)
        self.widths = tuple(widths)
        self.dropout = dropout
        self.seed = int(seed)
        self._model: Optional[Any] = None

    def solidify(self) -> Any:
        """Build (once) and return the live ``keras.Model`` — confluid's ``flow()``-time hook."""
        if self._model is not None:
            return self._model
        if not self.num_classes:
            raise ValueError(
                "KerasConvNet.num_classes is not set — a classifier flows the dataset-derived "
                "count in at run time, or set it explicitly."
            )
        # Seeded like every sibling implementation, and for the same reason: an unseeded run is
        # not comparable to itself, so a refactor cannot be checked by re-running it. Keras owns
        # no per-layer seed argument the way raw TF does, so this is the global call its own
        # docs prescribe — it covers the initializers of every layer built below.
        keras.utils.set_random_seed(self.seed)
        inputs = keras.Input(shape=self.input_shape)
        x = inputs
        for width in self.widths:
            x = keras.layers.Conv2D(width, KERNEL_SIZE, padding="same")(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.ReLU()(x)
            x = keras.layers.MaxPooling2D(POOL_SIZE)(x)
        x = keras.layers.GlobalAveragePooling2D()(x)
        if self.dropout:
            x = keras.layers.Dropout(self.dropout)(x)
        # Raw logits, no activation — like every other head in this project, so the same
        # `from_logits=True` losses and logit-space metric thresholds apply.
        outputs = keras.layers.Dense(self.num_classes, name="predictions")(x)
        self._model = keras.Model(inputs, outputs, name="sonair_convnet")
        logger.info(
            f"Built the shared conv spec in keras ({keras_backend()} backend, "
            f"{self.num_classes} classes, widths={self.widths})."
        )
        return self._model

    @property
    def model(self) -> Any:
        """The live ``keras.Model``, built on first access."""
        return self.solidify()

    def build(self, input_shape: Any = None) -> None:
        """Keras's build hook — the ONE place a subclassed model may create state.

        Same constraint `KerasModel.build` documents: a network created on the first ``call()``
        dies with *"You cannot add new elements of state … to a layer that is already built"*.
        """
        self.solidify()
        super().build(input_shape)

    def call(self, x: Any, training: bool = False) -> Any:
        """Per-image class logits."""
        return self.model(x, training=training)

    @property
    def output_shape(self) -> Any:
        """Functional-only on a subclassed model, so it is forwarded — see `KerasModel`."""
        return self.model.output_shape
