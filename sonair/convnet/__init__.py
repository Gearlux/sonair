"""The ONE conv-classifier architecture, and its four native implementations.

Every other model in sonair wraps somebody's zoo — timm, ``keras.applications``, the Hub, a
SavedModel. This package is the opposite: :mod:`sonair.convnet.spec` writes a small network
down once, and four sibling modules each implement THAT network in their own framework's idiom,
wrapping nothing::

    spec.py                 the architecture, and the parameter count it implies
    torch_convnet.py        TorchConvNet        torch.nn      (lightning AND fastai)
    keras_convnet.py        KerasConvNet        keras.layers
    tensorflow_convnet.py   TensorFlowConvNet   raw tf.nn ops on a tf.Module
    flax_convnet.py         FlaxConvNet         flax.nnx
    mlx_convnet.py          MlxConvNet          mlx.nn

It exists because the zoos do not line up — there is no Flax efficientvit and no MLX one — so a
cross-framework comparison built on zoos compares four different networks and tells you nothing
about the frameworks.

**The directory carries the grouping; each filename carries its class.** Being in here is what
says "implements the spec", which is why the zoo wrappers stay at the top level, one
``<backend>_model.py`` per backend. Within the package the naming rule is the project's usual
one — a module is named for the class it exports — and since every class here is a ``*ConvNet``,
every module is a ``*_convnet.py``. ``spec.py`` is the exception because it exports no class.

Each of the five **IS its framework's own model type** — ``torch.nn.Module``, ``keras.Model``,
``tf.Module``, ``nnx.Module``, ``mlx.nn.Module`` — the same decision as ``TimmModel`` being an
``nn.Module``. That is what lets each classifier drive whatever is wired into its ``model`` slot
without asking what it got, and it is why no backend reaches through a wrapper to find the
network.

**Only ``torch_convnet`` is channels-FIRST**, because torch is NCHW while the other four
default to NHWC. The ``input_shape`` KNOB stays ``(H, W, C)`` in all five so one config line
drives any of them; what differs is the tensor each ``forward`` consumes, and each backend's
preprocessing chain already produces the right one.

**This module imports NOTHING**, deliberately. ``sonair/__init__.py`` resolves every
backend-owned name lazily so a Keras-only install never imports torch (or jax, or mlx); a
package ``__init__`` that re-exported the four classes would import all four frameworks the
moment anything under here is touched — including :mod:`sonair.convnet.spec`, which is pure
Python and must stay reachable everywhere. Import the implementations by their own module path,
or as ``sonair.KerasConvNet`` / ``sonair.FlaxConvNet`` / … through the lazy top-level names.
"""

__all__: list = []
