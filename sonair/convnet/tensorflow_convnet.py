"""``TensorFlowConvNet`` — the shared conv spec in raw ``tf.nn`` ops.

The TensorFlow entry in the four-way comparison :mod:`sonair.convnet.spec` describes, and the
counterpart to the zoo wrapper next door in :mod:`sonair.tensorflow_model`: that one loads
somebody's SavedModel, this one wraps nothing at all. It is what makes the cross-framework
comparison include a real TensorFlow implementation rather than Keras-on-TF standing in for one
— :class:`sonair.convnet.keras_convnet.KerasConvNet` is already the Keras entry, and it runs on
whichever engine Keras is bound to.

Tagged ``framework="keras"`` rather than ``"tensorflow"``, which looks wrong and is deliberate:
the framework axis names the API a class is **wired to** (``docs/architecture.md`` §2), and
``TensorFlowClassifier`` — the only trainer that can consume this — declares ``"keras"`` itself.
A ``"tensorflow"`` tag would be truer to what the code imports and would hide the class from the
one slot picker that should offer it.

**It is a `tf.Module`, not a `keras.Model`**, which is the one place this diverges from the rest
of sonair's models. A ``tf.Module`` gives ``trainable_variables`` and ``__call__`` — everything
the traced loop needs — but not ``export``, so ``TensorFlowClassifier._export`` branches on that.
"""

from typing import Any, List, Optional, Tuple

from confluid import configurable
from loggair import get_logger

# `sonair._tensorflow` pins KERAS_BACKEND=tensorflow BEFORE importing keras; importing `tf`
# through it keeps that ordering in one place. See `sonair/tensorflow_classifier.py`.
from sonair._tensorflow import tf
from sonair.convnet.spec import CONV_WIDTHS, INPUT_SHAPE, KERNEL_SIZE, POOL_SIZE

logger = get_logger(__name__)

__all__ = ["TensorFlowConvNet"]


def _glorot_uniform(shape: Tuple[int, ...], seed: int) -> Any:
    """Glorot-uniform weights from ``tf.random``, written out rather than taken from an initializer.

    **`tf.initializers` IS Keras**, not TensorFlow: since TF 2.16 `tf.keras` is Keras 3, and
    `tf.initializers.GlorotUniform` resolves to `keras.initializers.GlorotUniform`
    (`is`-identical, measured). Called in a process where Keras is bound to the TORCH backend —
    which is this workspace's default, since `recordstream.keras` picks the first engine installed
    and torch wins — it returns a **torch tensor**, and `tf.Variable` then dies converting an MPS
    tensor to numpy (*"can't convert mps:0 device type tensor to numpy"*). So a module claiming to
    be raw `tf.nn` cannot reach for `tf.initializers` at all.

    The bug is invisible until both engines share ONE interpreter, so a file that passes alone
    fails after any Keras test has run. It was found in the segmentation project's twin of this
    module and fixed here in the same change; the five lines are duplicated rather than shared
    because the two projects must not depend on each other.

    ``stateless_uniform`` rather than ``tf.random.uniform``: the seed is the whole argument for a
    reproducible model, and the stateful op reads a GLOBAL seed another library can move.

    The fans are computed exactly as ``keras.src.initializers.random_initializers.compute_fans``
    does, so both kernel shapes this model uses are scaled correctly — a Dense head ``(in, out)``
    has no receptive field, a conv kernel ``(kh, kw, in, out)`` does.
    """
    receptive_field = 1
    for dim in shape[:-2]:
        receptive_field *= int(dim)
    if len(shape) == 2:  # a Dense kernel `(in, out)` — no receptive field at all
        fan_in, fan_out = int(shape[0]), int(shape[1])
    else:  # a conv kernel `(..., in, out)`
        fan_in, fan_out = int(shape[-2]) * receptive_field, int(shape[-1]) * receptive_field
    limit = (6.0 / float(fan_in + fan_out)) ** 0.5
    return tf.random.stateless_uniform(shape, seed=(seed, 0), minval=-limit, maxval=limit, dtype=tf.float32)


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
        kernel = _glorot_uniform((KERNEL_SIZE, KERNEL_SIZE, in_channels, width), seed=seed)
        self.kernel = tf.Variable(kernel, name="kernel")
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
        head_kernel = _glorot_uniform((channels, num_classes), seed=seed + len(widths))
        self.head_kernel = tf.Variable(head_kernel, name="head_kernel")
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
    """The shared conv classifier (:mod:`sonair.convnet.spec`), in raw TensorFlow ops.

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
        """Trainable parameters — the number :mod:`sonair.convnet.spec` predicts."""
        return int(sum(int(tf.size(v)) for v in self.model.trainable_variables))
