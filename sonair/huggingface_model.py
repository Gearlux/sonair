"""``HuggingFaceModel`` — an ``AutoModelForImageClassification`` behind the workspace's
model contract.

The Hub is the zoo here, so this is the same shape as ``TimmModel`` and ``KerasModel`` with a
different `create_model` underneath: ``__init__`` stores only, the network is built on first
use by :meth:`solidify` (confluid's ``flow()``-time hook), and required configuration is
validated at build time so zero-arg construction keeps working.

Two things are specific to the Hub and are the reason this class exists at all rather than a
config wiring ``transformers`` directly:

* **``num_classes`` arrives at run time.** A pretrained checkpoint carries its own head — 1000
  ImageNet classes for a ViT — so pointing it at a 10-class dataset is a shape mismatch. The
  build passes ``num_labels`` plus ``ignore_mismatched_sizes=True``, which is what tells
  transformers to DISCARD the pretrained head and start a fresh one at the requested width.
  Without it the load raises; with it, fine-tuning works the way the Hub intends.
* **The model computes its own loss.** ``AutoModelForImageClassification`` returns an object
  with ``.loss`` and ``.logits`` when handed ``labels=``, which is why this backend registers
  no loss set — see :mod:`sonair.huggingface_classifier`.

``model_name`` is a Hub id (``"google/vit-base-patch16-224"``), not a local path — the same
string you would pass to ``from_pretrained``.
"""

from typing import Any, List, Optional

from confluid import configurable
from loggair import get_logger
from marainer.uri import ModelRef, hf_model_ref

logger = get_logger(__name__)

__all__ = ["HuggingFaceModel"]


@configurable(task="classification", role="model", lazy=True, framework="huggingface")
class HuggingFaceModel:
    """A Hub image-classification checkpoint with a head sized to your dataset.

    Args:
        model_name: A Hub model id, e.g. ``"google/vit-base-patch16-224"`` or
            ``"microsoft/resnet-18"``. Sentinel ``""`` is invalid but permitted so the class
            is zero-arg constructible; it is rejected at build time with a clear error.
        num_classes: Width of the classification head. Sentinel ``0`` likewise — a classifier
            injects the dataset-derived count via ``flow(model, num_classes=N)`` at run time,
            so it is not written in YAML.
        pretrained: Load the checkpoint's weights. ``False`` builds the same architecture from
            its config with fresh weights, which keeps a first run offline and reproducible.
        trust_remote_code: Allow a Hub repo to execute its own modelling code. Off by default
            because it runs code from the internet; turn it on only for a repo you trust.
    """

    def __init__(
        self,
        model_name: str = "",
        num_classes: int = 0,
        pretrained: bool = True,
        trust_remote_code: bool = False,
    ) -> None:
        # Lazy: store only. No Hub call, no download — see the module docstring.
        self.model_name = model_name
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.trust_remote_code = trust_remote_code
        self._model: Optional[Any] = None

    @property
    def model_refs(self) -> List[ModelRef]:
        """The weight artifacts this model loads — see :mod:`marainer.uri`.

        ``model_name`` IS a Hub repository id, so the reference is the id itself; nothing has to
        be resolved and nothing is fetched to answer. ``pretrained=False`` builds the same
        architecture with fresh weights and therefore loads no artifact at all.
        """
        return [hf_model_ref(self.model_name)] if self.pretrained and self.model_name else []

    def solidify(self) -> Any:
        """Build (once) and return the live ``transformers`` model.

        This is confluid's ``flow()``-time hook as well as the cache: flowing this class with
        ``num_classes=N`` gives a head of that width immediately, so the caller never has to
        remember a separate build step.
        """
        if self._model is not None:
            return self._model
        if not self.model_name:
            raise ValueError(
                "HuggingFaceModel.model_name is not set — name a Hub model id, e.g. 'google/vit-base-patch16-224'."
            )
        if not self.num_classes:
            raise ValueError(
                "HuggingFaceModel.num_classes is not set — a classifier flows the "
                "dataset-derived count in at run time, or set it explicitly."
            )

        from transformers import AutoConfig, AutoModelForImageClassification

        kwargs: dict = {
            "num_labels": self.num_classes,
            "trust_remote_code": self.trust_remote_code,
        }
        if self.pretrained:
            # The checkpoint's own head is almost never the right width, so it is discarded
            # and rebuilt. Without this flag transformers raises on the size mismatch.
            self._model = AutoModelForImageClassification.from_pretrained(
                self.model_name, ignore_mismatched_sizes=True, **kwargs
            )
        else:
            config = AutoConfig.from_pretrained(self.model_name, **kwargs)
            self._model = AutoModelForImageClassification.from_config(config, trust_remote_code=self.trust_remote_code)
        logger.info(f"Built {self.model_name} ({self.num_classes} classes, pretrained={self.pretrained}).")
        return self._model

    @property
    def model(self) -> Any:
        """The live ``transformers`` model, built on first access.

        The ergonomic accessor for a directly-constructed instance that was never flowed —
        ``flow()`` does not touch property getters, which is why :meth:`solidify` exists as a
        method as well.
        """
        return self.solidify()

    def __call__(self, pixel_values: Any, **kwargs: Any) -> Any:
        """Forward to the wrapped model, building it if this is the first use."""
        return self.model(pixel_values=pixel_values, **kwargs)
