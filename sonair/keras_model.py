"""``KerasModel`` — a thin ``keras.applications`` wrapper, the Keras twin of ``TimmModel``.

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

This module holds the ZOO WRAPPER only. The other kind of Keras model — the shared conv spec of
:mod:`sonair.convnet.spec`, which wraps nothing — is :class:`sonair.convnet.keras_convnet.KerasConvNet`.
"""

from typing import Any, List, Optional, Tuple
from urllib.parse import quote, urlencode

from confluid import configurable
from loggair import get_logger
from matrainer.uri import ModelRef

from sonair._keras import keras, keras_backend

logger = get_logger(__name__)

#: Scheme naming the ``keras.applications`` zoo in a model reference. Its weights come from the
#: library's own release assets rather than a public repository, so there is a registry to name
#: but no URL to open — see :attr:`KerasModel.model_refs`.
KERAS_URI_PREFIX = "keras://applications/"

__all__ = ["KERAS_URI_PREFIX", "KerasModel"]


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
        seed: Seed applied before the build — the head is ALWAYS freshly initialized (and with
            ``pretrained=False`` the whole backbone is), so an unseeded run is not comparable
            to itself. The same knob every other model here carries; Keras owns no per-layer
            seed argument, so this is the global ``set_random_seed`` its own docs prescribe.
    """

    def __init__(
        self,
        model_name: str = "",
        num_classes: int = 0,
        pretrained: bool = False,
        input_shape: Tuple[int, int, int] = (224, 224, 3),
        pooling: str = "avg",
        dropout: float = 0.0,
        seed: int = 0,
    ) -> None:
        # Partial: store only. No network build — see the module docstring. `super().__init__()`
        # comes first because a `keras.Layer` refuses attribute assignment before it.
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.input_shape = tuple(input_shape)
        self.pooling = pooling
        self.dropout = dropout
        self.seed = int(seed)
        self._model: Optional[Any] = None

    @property
    def model_refs(self) -> List[ModelRef]:
        """The weight artifacts this model loads — see :mod:`matrainer.uri`.

        ``keras.applications`` weights have no public repository URL to point at — the library
        downloads them from its own release assets — so the reference names the registry and the
        entry within it, and reports no browsable link rather than inventing one.
        """
        if not (self.pretrained and self.model_name):
            return []
        return [ModelRef(uri=f"{KERAS_URI_PREFIX}{quote(self.model_name)}?{urlencode({'weights': 'imagenet'})}")]

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

        # Seeded like every sibling model, and for the same reason: the head below is always a
        # fresh random init (the backbone too when pretrained=False), so an unseeded run is not
        # comparable to itself. Keras owns no per-layer seed argument, so this is the global
        # call its own docs prescribe — it covers every layer built below.
        keras.utils.set_random_seed(self.seed)
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
