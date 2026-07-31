"""The Keras models: a ``keras.applications`` zoo wrapper, and the shared conv spec.

``KerasModel`` — a thin ``keras.applications`` wrapper, the Keras twin of ``TimmModel``.

The point of this class is how LITTLE it does. It mirrors ``sonair.timm_model.TimmModel``
exactly — same constructor shape, same lazy build, same ``num_classes`` injection at run time —
so a reader comparing the two sees that only the library call differs. That comparison IS the
demo: swapping the training engine swaps the model wrapper and nothing else in the config.

The mirroring goes down to the base class: ``TimmModel`` IS an ``nn.Module``, so this IS a
``keras.Model``. That is what makes ``compile`` / ``fit`` / ``evaluate`` / ``predict`` /
``export`` / ``save`` / ``trainable_variables`` arrive for free, and it is why a classifier can
address whatever is wired into its ``model`` slot — this class or a bare ``keras.Model`` —
without ever asking which it got. (The alternative, forwarding each of those by hand, was
tried and thrown away: it is a interface re-declared in the wrapper instead of inherited.)

Like its torch twin it is LAZY: ``__init__`` only stores values (the workspace zero-arg /
lazy-init mandate), and the network is built on first use by :meth:`solidify` — which is also
confluid's ``flow()``-time hook, so a classifier that flows this with a dataset-derived
``num_classes`` gets a head of the right width without ever writing the count in YAML.

Keras 3 picks its backend from ``KERAS_BACKEND`` at import time; sonair imports Keras through
:mod:`sonair._keras`, which defaults it to ``torch`` (the backend this package guarantees) only
when the operator has not chosen. See that module for why.
"""

from typing import Any, Optional, Tuple

from confluid import configurable
from loggair import get_logger

from sonair._keras import keras, keras_backend
from sonair.convnet_spec import CONV_WIDTHS, INPUT_SHAPE, KERNEL_SIZE, POOL_SIZE

logger = get_logger(__name__)

__all__ = ["KerasConvNet", "KerasModel"]


@configurable(task="classification", role="model", lazy=True, framework="keras")
class KerasModel(keras.Model):
    """A ``keras.applications`` backbone with a fresh classification head.

    Args:
        model_name: A ``keras.applications`` constructor name, e.g. ``"MobileNetV3Small"`` or
            ``"EfficientNetV2B0"``. Sentinel ``""`` is invalid but permitted so the class is
            zero-arg constructible; it is rejected at build time with a clear error.
        num_classes: Width of the classification head. Sentinel ``0`` likewise — a classifier
            injects the dataset-derived count via ``flow(model, num_classes=N)`` at run time,
            so it is not written in YAML.
        pretrained: Load ImageNet weights for the backbone. ``False`` (the default) keeps a
            first run offline and reproducible; the head is always fresh either way.
        input_shape: ``(H, W, C)`` — Keras is CHANNELS-LAST, unlike the torch backend's
            ``(C, H, W)``. That difference is real and is why the two backends' preprocessing
            configs differ by their last op.
        pooling: How the backbone's feature map is reduced before the head — ``"avg"`` or
            ``"max"``.
        dropout: Dropout applied before the head; ``0.0`` disables it.
    """

    def __init__(
        self,
        model_name: str = "",
        num_classes: int = 0,
        pretrained: bool = False,
        input_shape: Tuple[int, int, int] = (224, 224, 3),
        pooling: str = "avg",
        dropout: float = 0.0,
    ) -> None:
        # Lazy: store only. No network build — see the module docstring. `super().__init__()`
        # comes first because a `keras.Layer` refuses attribute assignment before it.
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.input_shape = tuple(input_shape)
        self.pooling = pooling
        self.dropout = dropout
        self._model: Optional[Any] = None

    def solidify(self) -> Any:
        """Build (once) and return the live ``keras.Model``.

        This is confluid's ``flow()``-time hook as well as the cache: flowing this class with
        ``num_classes=N`` builds a head of that width immediately, so the caller never has to
        remember a separate build step. Required configuration is validated HERE rather than in
        ``__init__``, which is what keeps zero-arg construction working.
        """
        if self._model is not None:
            return self._model
        if not self.model_name:
            raise ValueError("KerasModel.model_name is not set — name a keras.applications model.")
        if not self.num_classes:
            raise ValueError(
                "KerasModel.num_classes is not set — a classifier flows the dataset-derived "
                "count in at run time, or set it explicitly."
            )

        factory = getattr(keras.applications, self.model_name, None)
        if factory is None:
            available = ", ".join(sorted(n for n in dir(keras.applications) if n[:1].isupper())[:8])
            raise ValueError(f"Unknown keras.applications model {self.model_name!r} (e.g. {available}, …).")

        backbone = factory(
            input_shape=self.input_shape,
            include_top=False,
            weights="imagenet" if self.pretrained else None,
            pooling=self.pooling,
        )
        head = keras.layers.Dropout(self.dropout)(backbone.output) if self.dropout else backbone.output
        outputs = keras.layers.Dense(self.num_classes, name="predictions")(head)
        self._model = keras.Model(backbone.input, outputs, name=f"{self.model_name}_classifier")
        logger.info(
            f"Built {self.model_name} ({keras_backend()} backend, {self.num_classes} classes, "
            f"pretrained={self.pretrained})."
        )
        return self._model

    @property
    def model(self) -> Any:
        """The live ``keras.Model``, built on first access.

        The ergonomic accessor for a directly-constructed instance that was never flowed —
        ``flow()`` does not touch property getters, which is why :meth:`solidify` exists as a
        method as well.
        """
        return self.solidify()

    def build(self, input_shape: Any = None) -> None:
        """Keras's own build hook — the ONE place a subclassed model may create state.

        Not decoration: Keras forbids a layer from adding sub-layers once it is built, so a
        network created on the first ``call()`` dies with *"You cannot add new elements of state
        … to a layer that is already built"* (hit for real). ``build()`` runs BEFORE that first
        call, which is what keeps "never flowed, just call it" working — the ergonomics
        :meth:`model` promises. The normal path never reaches here: ``flow()`` fires
        :meth:`solidify` long before, and the ``self._model is not None`` cache makes the second
        trigger free.
        """
        self.solidify()
        super().build(input_shape)

    def call(self, x: Any, training: bool = False) -> Any:
        """Per-image class logits — the twin of ``TimmModel.forward``."""
        return self.model(x, training=training)

    @property
    def output_shape(self) -> Any:
        """The network's output shape, so a consumer can read the head width back off it.

        The one member of the ``keras.Model`` surface that does NOT come by inheritance:
        ``output_shape`` is Functional-only, and a subclassed model raises ``AttributeError``
        for it (measured). The inner network IS functional, so the answer is one hop away.
        """
        return self.model.output_shape


@configurable(task="classification", role="model", lazy=True, framework="keras")
class KerasConvNet(keras.Model):
    """The shared conv classifier (:mod:`sonair.convnet_spec`), implemented in ``keras.layers``.

    The zoo wrapper above is what you want for real work — pretrained weights beat a small net
    from scratch every time. This exists for the comparison: the SAME architecture is
    implemented in Flax, MLX and raw TensorFlow, and four zoo models would have compared four
    different networks instead of four frameworks.

    Being written in ``keras.layers`` it runs on whichever engine Keras is bound to, so both
    the ``keras`` and ``tensorflow`` backends wire it. That is also why it is not called
    ``TensorFlowModel``: it is not TensorFlow-specific. The genuinely TF-native pair lives in
    :mod:`sonair.tensorflow_model`.

    Args:
        num_classes: Width of the classification head. Sentinel ``0`` is invalid but permitted
            so the class is zero-arg constructible; a classifier flows the dataset-derived
            count in at run time.
        input_shape: ``(H, W, C)`` — channels LAST.
        widths: Channels per conv block. Changing it breaks comparability with the other
            implementations — see :mod:`sonair.convnet_spec`.
        dropout: Dropout before the head; ``0.0`` disables it.
    """

    def __init__(
        self,
        num_classes: int = 0,
        input_shape: Tuple[int, int, int] = INPUT_SHAPE,
        widths: Tuple[int, ...] = CONV_WIDTHS,
        dropout: float = 0.0,
    ) -> None:
        # Lazy: store only, and `super().__init__()` first — a `keras.Layer` refuses attribute
        # assignment before it. Same contract as `KerasModel` above.
        super().__init__()
        self.num_classes = num_classes
        self.input_shape = tuple(input_shape)
        self.widths = tuple(widths)
        self.dropout = dropout
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
