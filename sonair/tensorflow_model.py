"""``TensorFlowModel`` — a SavedModel loaded through ``tf.saved_model.load``, TF's own zoo.

``docs/architecture.md`` §5 originally recorded that the TensorFlow backend contributes no
model module, on the grounds that a ``TensorFlowModel`` would be ``KerasModel`` with the class
name changed — ``tf.keras.applications`` *is* ``keras.applications``. That reasoning is sound
but narrower than it looked: it rules out a duplicate **zoo wrapper over keras.applications**,
which is not the only TensorFlow model source. This class duplicates nothing — it loads a
**SavedModel**, the format Kaggle Models serves (where TF Hub redirected) and what any
``model.export()`` produces. A genuinely different zoo, reachable with no new dependency;
``tensorflow-hub`` itself is deliberately NOT used, as it last released 2024-01-30.

It is tagged ``framework="keras"`` rather than ``"tensorflow"``, which looks wrong and is
deliberate: the framework axis names the API a class is **wired to** (§2), and
``TensorFlowClassifier`` — the only trainer that can consume this — declares ``"keras"``
itself. A ``"tensorflow"`` tag would be truer to what the code imports and would hide the class
from the one slot picker that should offer it.

**It is not a `keras.Model`**, which is the one place this diverges from the rest of sonair's
model wrappers. A ``tf.Module`` gives ``trainable_variables`` and ``__call__`` — everything the
traced loop needs — but not ``export``, so ``TensorFlowClassifier._export`` branches on that.

This module holds the ZOO WRAPPER only. The other TensorFlow-native model — the shared conv
spec of :mod:`sonair.convnet.spec` in raw ``tf.nn`` ops, wrapping nothing — is
:class:`sonair.convnet.tensorflow_convnet.TensorFlowConvNet`.
"""

from typing import Any, List, Optional, Tuple

from confluid import configurable
from loggair import get_logger
from matrainer.uri import ModelRef, file_model_ref

# `sonair._tensorflow` pins KERAS_BACKEND=tensorflow BEFORE importing keras; importing `tf`
# through it keeps that ordering in one place. See `sonair/tensorflow_classifier.py`.
from sonair._tensorflow import tf

logger = get_logger(__name__)

#: The signature `tf.saved_model` gives an export that names none — TensorFlow's own default.
_SERVING_DEFAULT = "serving_default"

__all__ = ["TensorFlowModel"]


@configurable(task="classification", role="model", lazy=True, framework="keras")
class TensorFlowModel(tf.Module):
    """A SavedModel loaded with ``tf.saved_model.load`` — TensorFlow's own zoo format.

    This is the TF-native counterpart to ``KerasModel``'s ``keras.applications`` and
    ``HuggingFaceModel``'s Hub: point it at any exported SavedModel — one this project produced
    with ``sonair train --framework tensorflow``, or one downloaded from Kaggle Models, which is where TF
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

    There is deliberately NO ``seed`` knob, unlike every other model here: the build is pure
    deserialization — every weight comes from the artifact, no head is attached, and nothing
    random is drawn — so a seed would seed nothing and read as a promise the class cannot keep.
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

    @property
    def model_refs(self) -> List[ModelRef]:
        """The weight artifacts this model loads — see :mod:`matrainer.uri`.

        The SavedModel directory IS the model, and its role is ``pretrained`` rather than
        ``checkpoint``: the distinction the roles draw is whether the weights came from outside
        the run, and a SavedModel wired into a config did — whoever exported it, this run did not.
        """
        return [file_model_ref(self.saved_model, role="pretrained")] if self.saved_model else []

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
