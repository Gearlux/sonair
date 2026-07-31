"""The TensorFlow-native models: a SavedModel zoo wrapper, and the shared conv spec in raw TF.

``docs/architecture.md`` §5 originally recorded that the TensorFlow backend contributes no
model module, on the grounds that a ``TensorFlowModel`` would be ``KerasModel`` with the class
name changed — ``tf.keras.applications`` *is* ``keras.applications``. That reasoning is sound
but narrower than it looked: it rules out a duplicate **zoo wrapper over keras.applications**,
which is not the only TensorFlow model source. Two things here duplicate nothing:

* :class:`TensorFlowModel` loads a **SavedModel** with ``tf.saved_model.load`` — the format
  Kaggle Models serves (where TF Hub redirected) and what any ``model.export()`` produces. That
  is a genuinely different zoo from ``keras.applications``, reachable with no new dependency.
  ``tensorflow-hub`` itself is deliberately NOT used: it last released 2024-01-30.
* :class:`TensorFlowConvNet` implements :mod:`sonair.convnet_spec` in raw ``tf.nn`` ops on a
  ``tf.Module``. It wraps nothing at all, and it is what makes the cross-framework comparison
  include a real TensorFlow implementation rather than Keras-on-TF standing in for one.

Both are tagged ``framework="keras"`` rather than ``"tensorflow"``, which looks wrong and is
deliberate: the framework axis names the API a class is **wired to** (§2), and
``TensorFlowClassifier`` — the only trainer that can consume these — declares ``"keras"``
itself. A ``"tensorflow"`` tag would be truer to what the code imports and would hide both
classes from the one slot picker that should offer them.

**Neither is a `keras.Model`**, which is the one place this diverges from the rest of sonair's
model wrappers. A ``tf.Module`` gives ``trainable_variables`` and ``__call__`` — everything the
traced loop needs — but not ``export``, so ``TensorFlowClassifier._export`` branches on that.
"""

from typing import Any, List, Optional, Tuple

from confluid import configurable
from loggair import get_logger

# `sonair._tensorflow` pins KERAS_BACKEND=tensorflow BEFORE importing keras; importing `tf`
# through it keeps that ordering in one place. See `sonair/tensorflow_classifier.py`.
from sonair._tensorflow import tf
from sonair.convnet_spec import CONV_WIDTHS, INPUT_SHAPE, KERNEL_SIZE, POOL_SIZE

logger = get_logger(__name__)

#: The signature `tf.saved_model` gives an export that names none — TensorFlow's own default.
_SERVING_DEFAULT = "serving_default"

__all__ = ["TensorFlowConvNet", "TensorFlowModel"]


class _ConvBlock(tf.Module):
    """One conv → batch-norm → ReLU → max-pool block, in raw ``tf.nn`` ops.

    BatchNorm is written out rather than borrowed because borrowing it would mean
    ``keras.layers.BatchNormalization``, and then this would be a Keras model wearing a
    TensorFlow name. The moving statistics are ``trainable=False`` variables so they are STATE
    rather than parameters — which is what lets the parameter count match the other three
    implementations, since the frameworks disagree about whether to count them.
    """

    def __init__(self, in_channels: int, width: int, seed: int, momentum: float = 0.99) -> None:
        super().__init__(name=f"block_{in_channels}_{width}")
        initializer = tf.initializers.GlorotUniform(seed=seed)
        self.kernel = tf.Variable(initializer(shape=(KERNEL_SIZE, KERNEL_SIZE, in_channels, width)), name="kernel")
        self.bias = tf.Variable(tf.zeros((width,)), name="bias")
        self.gamma = tf.Variable(tf.ones((width,)), name="gamma")
        self.beta = tf.Variable(tf.zeros((width,)), name="beta")
        self.moving_mean = tf.Variable(tf.zeros((width,)), trainable=False, name="moving_mean")
        self.moving_var = tf.Variable(tf.ones((width,)), trainable=False, name="moving_var")
        self.momentum = momentum

    def __call__(self, x: Any, training: bool = False) -> Any:
        x = tf.nn.conv2d(x, self.kernel, strides=1, padding="SAME") + self.bias
        if training:
            # Batch statistics while training, and fold them into the moving estimates the
            # eval path will use — the update Keras/torch do for you inside their layer.
            mean, variance = tf.nn.moments(x, axes=[0, 1, 2])
            self.moving_mean.assign(self.moving_mean * self.momentum + mean * (1.0 - self.momentum))
            self.moving_var.assign(self.moving_var * self.momentum + variance * (1.0 - self.momentum))
        else:
            mean, variance = self.moving_mean, self.moving_var
        x = tf.nn.batch_normalization(x, mean, variance, self.beta, self.gamma, variance_epsilon=1e-3)
        x = tf.nn.relu(x)
        return tf.nn.max_pool2d(x, ksize=POOL_SIZE, strides=POOL_SIZE, padding="VALID")


class _ConvNet(tf.Module):
    """The spec: conv blocks → global average pool → dropout → dense logits."""

    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        widths: Tuple[int, ...],
        dropout: float,
        seed: int,
    ) -> None:
        super().__init__(name="sonair_convnet")
        self.blocks: List[_ConvBlock] = []
        channels = in_channels
        for index, width in enumerate(widths):
            self.blocks.append(_ConvBlock(channels, width, seed=seed + index))
            channels = width
        head_init = tf.initializers.GlorotUniform(seed=seed + len(widths))
        self.head_kernel = tf.Variable(head_init(shape=(channels, num_classes)), name="head_kernel")
        self.head_bias = tf.Variable(tf.zeros((num_classes,)), name="head_bias")
        self.dropout = dropout

    def __call__(self, x: Any, training: bool = False) -> Any:
        for block in self.blocks:
            x = block(x, training=training)
        x = tf.reduce_mean(x, axis=[1, 2])  # global average pool — channels are LAST
        if self.dropout and training:
            x = tf.nn.dropout(x, rate=self.dropout)
        # Raw logits, no activation — like every other head in this project.
        return tf.matmul(x, self.head_kernel) + self.head_bias


@configurable(task="classification", role="model", lazy=True, framework="keras")
class TensorFlowConvNet(tf.Module):
    """The shared conv classifier (:mod:`sonair.convnet_spec`), in raw TensorFlow ops.

    The TensorFlow entry in the four-way comparison. For real work prefer ``KerasModel`` (a
    pretrained ``keras.applications`` backbone) — a small net trained from scratch loses to
    transfer learning every time. This exists so the comparison is between FRAMEWORKS rather
    than between four different networks.

    Args:
        num_classes: Width of the classification head. Sentinel ``0`` is invalid but permitted
            so the class is zero-arg constructible; a classifier flows the dataset-derived
            count in at run time.
        input_shape: ``(H, W, C)`` — channels LAST, TensorFlow's native layout.
        widths: Channels per conv block. Changing it breaks comparability with the other
            implementations — see :mod:`sonair.convnet_spec`.
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
        # `tf.Module.__init__` first — it sets up the name scope the variables register in.
        super().__init__(name="sonair_tf_convnet")
        # Lazy: store only. No variables allocated until `solidify()`.
        self.num_classes = num_classes
        self.input_shape = tuple(input_shape)
        self.widths = tuple(widths)
        self.dropout = dropout
        self.seed = seed
        self._model: Optional[_ConvNet] = None

    def solidify(self) -> Any:
        """Build (once) and return the live ``tf.Module`` — confluid's ``flow()``-time hook."""
        if self._model is not None:
            return self._model
        if not self.num_classes:
            raise ValueError(
                "TensorFlowConvNet.num_classes is not set — a classifier flows the "
                "dataset-derived count in at run time, or set it explicitly."
            )
        self._model = _ConvNet(
            num_classes=self.num_classes,
            in_channels=int(self.input_shape[-1]),
            widths=self.widths,
            dropout=self.dropout,
            seed=self.seed,
        )
        logger.info(f"Built the shared conv spec in raw TensorFlow ({self.num_classes} classes, widths={self.widths}).")
        return self._model

    @property
    def model(self) -> Any:
        """The live inner ``tf.Module``, built on first access."""
        return self.solidify()

    def __call__(self, x: Any, training: bool = False) -> Any:
        """Per-image class logits — what the traced training step calls."""
        return self.model(x, training=training)

    @property
    def trainable_variables(self) -> Any:
        """The variables the optimizer updates. Building on access, so the classifier can call
        ``optimizer.build(model.trainable_variables)`` before the first step."""
        return self.model.trainable_variables

    @property
    def parameter_count(self) -> int:
        """Trainable parameters — the number :mod:`sonair.convnet_spec` predicts."""
        return int(sum(int(tf.size(v)) for v in self.model.trainable_variables))


@configurable(task="classification", role="model", lazy=True, framework="keras")
class TensorFlowModel(tf.Module):
    """A SavedModel loaded with ``tf.saved_model.load`` — TensorFlow's own zoo format.

    This is the TF-native counterpart to ``KerasModel``'s ``keras.applications`` and
    ``HuggingFaceModel``'s Hub: point it at any exported SavedModel — one this project produced
    with ``sonair tensorflow train``, or one downloaded from Kaggle Models, which is where TF
    Hub redirected. Core TensorFlow, no extra dependency.

    Args:
        saved_model: Path to a SavedModel DIRECTORY (the one containing ``saved_model.pb``).
            Sentinel ``""`` is invalid but permitted so the class is zero-arg constructible.
        num_classes: The head width. Unlike the other wrappers this cannot be derived — a
            SavedModel is an opaque graph that does not declare its class count — and an
            evaluation run reads it straight off the model, so wire it when you evaluate.
        trainable: Whether the loaded variables take gradients. A SavedModel exported for
            inference may carry none at all; see :meth:`solidify`.
        tags: The MetaGraph tags to load. TensorFlow's default is ``serve``.
    """

    def __init__(
        self,
        saved_model: str = "",
        num_classes: int = 0,
        trainable: bool = True,
        tags: Optional[Tuple[str, ...]] = None,
    ) -> None:
        super().__init__(name="sonair_saved_model")
        # Lazy: store only. No disk read — the load happens in `solidify()`.
        self.saved_model = saved_model
        self.num_classes = num_classes
        self.trainable = trainable
        self.tags = tags
        self._model: Optional[Any] = None

    def solidify(self) -> Any:
        """Load (once) and return the live SavedModel object.

        Fine-tuning a SavedModel only works if it was exported WITH its variables: an
        inference-only export carries none, and training it would silently apply empty
        gradients forever. That is checked here, at build time, rather than discovered as a
        loss that never moves.
        """
        if self._model is not None:
            return self._model
        if not self.saved_model:
            raise ValueError(
                "TensorFlowModel.saved_model is not set — give the path to a SavedModel "
                "directory (the one containing saved_model.pb)."
            )
        from pathlib import Path

        path = Path(self.saved_model).expanduser()
        if not path.exists():
            raise ValueError(f"TensorFlowModel.saved_model does not exist: {path}")
        self._model = tf.saved_model.load(str(path), tags=list(self.tags) if self.tags else None)
        variables = _restored_trainable_variables(self._model)
        if self.trainable and not variables:
            raise ValueError(
                f"The SavedModel at {path} carries no trainable variables, so it cannot be "
                "fine-tuned — it was exported for inference. Load it with `trainable: false` "
                "to run evaluation/prediction only."
            )
        logger.info(f"Loaded SavedModel from {path} ({len(variables)} trainable variables).")
        return self._model

    @property
    def model(self) -> Any:
        """The live SavedModel object, loaded on first access."""
        return self.solidify()

    def __call__(self, x: Any, training: bool = False) -> Any:
        """Forward through the loaded graph.

        Three shapes reach here, and the differences are all about how the artifact was made:

        * **A Keras export** restores a CALLABLE object — call it directly.
        * **A ``tf.saved_model.save`` of a plain ``tf.Module``** restores a ``_UserObject`` that
          is NOT callable (measured: ``'_UserObject' object is not callable``). Its graph lives
          in ``signatures["serving_default"]``, which takes the input as a keyword and returns a
          DICT of outputs.
        * Either may or may not accept ``training=``, depending on how it was traced, so that is
          passed only when the signature declares it rather than assumed.
        """
        import inspect

        fn = self.model
        if not callable(fn):
            signature = fn.signatures.get(_SERVING_DEFAULT)
            if signature is None:
                raise ValueError(
                    f"The SavedModel at {self.saved_model} is not callable and has no "
                    f"{_SERVING_DEFAULT!r} signature; available: {list(fn.signatures)}."
                )
            outputs = signature(tf.convert_to_tensor(x))
            # A serving signature returns a dict keyed by output name. There is exactly one
            # output for a classifier, and its name is the exporter's choice, so take the value
            # rather than guessing the key.
            return next(iter(outputs.values())) if isinstance(outputs, dict) else outputs
        try:
            names = set(inspect.signature(fn).parameters)
        except (TypeError, ValueError):  # pragma: no cover - a concrete function without one
            names = set()
        if "training" in names:
            return fn(x, training=training)
        return fn(x)

    @property
    def trainable_variables(self) -> Any:
        """The loaded graph's trainable variables, for the optimizer."""
        return _restored_trainable_variables(self.model) if self.trainable else ()

    @property
    def parameter_count(self) -> int:
        """Trainable parameters in the loaded graph."""
        return int(sum(int(tf.size(v)) for v in self.trainable_variables))


def _restored_trainable_variables(loaded: Any) -> Tuple[Any, ...]:
    """Every trainable variable in a loaded SavedModel, however it was exported.

    Two export shapes reach here and they differ in a way that is easy to get wrong:

    * **A Keras export** (``keras.Model.export``) restores an object that HAS ``variables`` /
      ``trainable_variables`` — Keras writes those lists into the artifact deliberately.
    * **A ``tf.saved_model.save`` of a plain ``tf.Module``** does not. ``load`` restores tracked
      DATA, not the Python class, and ``tf.Module.variables`` is a computed property of the
      class — so it is simply gone, and the variables sit in the restored child objects
      (``_model.blocks[0].kernel`` …). Measured: a Keras export reports its variables, the
      module export reports none while its children hold all twenty.

    So: take the list when the artifact provides one, otherwise walk the restored trackable
    tree. The walk uses ``_trackable_children()``, which is private API — accepted knowingly,
    because the alternative is declaring every sonair-exported SavedModel un-fine-tunable. It
    is guarded so a TensorFlow change degrades this to "no variables found" (a clear error at
    build time) rather than an exception.
    """
    declared = getattr(loaded, "trainable_variables", None)
    if declared:
        return tuple(declared)

    found: List[Any] = []
    seen: set = set()

    def walk(node: Any, depth: int = 0) -> None:
        if depth > 12 or id(node) in seen:  # depth cap: a cycle would otherwise not terminate
            return
        seen.add(id(node))
        if isinstance(node, tf.Variable):
            if getattr(node, "trainable", False):
                found.append(node)
            return
        try:
            children = node._trackable_children().values()
        except (AttributeError, TypeError):  # pragma: no cover - a leaf, or a TF API change
            return
        for child in children:
            walk(child, depth + 1)

    walk(loaded)
    return tuple(found)
