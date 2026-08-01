"""`TorchConvNet` — the shared conv spec in `torch.nn`.

The torch entry in the five-way comparison, and the one that differs in LAYOUT: torch is NCHW
while the other four are NHWC. That is the interesting thing to pin here, because the
constructor knob deliberately does NOT follow the tensor — `input_shape` stays `(H, W, C)` in
all five so one config line drives any of them, and getting that backwards would build a network
with the wrong number of input channels while still running.

Cross-implementation agreement lives in `tests/test_convnet_spec.py`; this file is the torch
implementation's own behaviour.
"""

import pytest

torch = pytest.importorskip("torch", reason="needs sonair[lightning]")

from sonair.convnet.spec import CONV_WIDTHS, trainable_parameter_count  # noqa: E402
from sonair.convnet.torch_convnet import TorchConvNet  # noqa: E402

NUM_CLASSES = 10
INPUT = (16, 16, 3)


def test_it_matches_the_shared_spec() -> None:
    model = TorchConvNet(num_classes=NUM_CLASSES, input_shape=INPUT)
    model.solidify()

    assert model.parameter_count == trainable_parameter_count(NUM_CLASSES)


def test_it_is_a_torch_module() -> None:
    """The same decision as `TimmModel` being one, so `LightningClassifier` and
    `FastaiClassifier` take it in their `model` slot with no adapter."""
    assert issubclass(TorchConvNet, torch.nn.Module)


def test_zero_arg_construction_allocates_nothing() -> None:
    model = TorchConvNet()

    assert model.net is None
    assert not list(model.parameters())
    with pytest.raises(ValueError, match="num_classes"):
        model.solidify()


def test_it_consumes_a_channels_FIRST_tensor() -> None:
    """The layout difference, pinned.

    `input_shape` is `(H, W, C)` — the same knob the four NHWC implementations take — but the
    TENSOR is `[N, C, H, W]`, which is what the torch preprocessing chain already produces
    (it ends in a `ToTensor` that transposes HWC to CHW). Reading the knob as NCHW would build
    `Conv2d(16, ...)` instead of `Conv2d(3, ...)`: still runnable, silently a different network.
    """
    model = TorchConvNet(num_classes=NUM_CLASSES, input_shape=INPUT)

    out = model(torch.ones(2, 3, 16, 16))

    assert out.shape == (2, NUM_CLASSES)
    first_conv = next(m for m in model.modules() if isinstance(m, torch.nn.Conv2d))
    assert first_conv.in_channels == 3, "in_channels comes from input_shape[-1], not input_shape[0]"
    assert first_conv.out_channels == CONV_WIDTHS[0]


def test_the_head_emits_raw_logits() -> None:
    """No activation on the head — the same contract every other model here honours, so
    `CrossEntropyLoss` (which applies its own log-softmax) is the correct loss."""
    out = TorchConvNet(num_classes=4, input_shape=INPUT)(torch.ones(2, 3, 16, 16))

    assert not torch.allclose(out.sum(dim=-1), torch.ones(2))
    last = list(TorchConvNet(num_classes=4, input_shape=INPUT).backbone.children())[-1]
    assert isinstance(last, torch.nn.Linear), "the head is the last layer — nothing follows it"


def test_batchnorm_running_stats_are_buffers_not_parameters() -> None:
    """Why this implementation needs no name filter, unlike the MLX one.

    Torch keeps the running statistics as BUFFERS, so `parameters()` already excludes them and
    the count matches the spec by construction. They are still in the state_dict, which is what
    makes a checkpoint round-trip carry them.
    """
    model = TorchConvNet(num_classes=NUM_CLASSES, input_shape=INPUT)
    model.solidify()

    names = list(model.state_dict())
    assert any(n.endswith("running_mean") for n in names), "they ARE saved…"
    assert not any("running_mean" in n for n, _ in model.named_parameters()), "…but are not parameters"


def test_the_network_is_a_public_submodule_so_checkpoints_address_it() -> None:
    """`net`, not `_net` — the same reason `TimmModel` uses `classifier`: a state_dict saved
    through marainer's Lightning wrapper round-trips to `net.*` keys."""
    model = TorchConvNet(num_classes=NUM_CLASSES, input_shape=INPUT)
    model.solidify()

    assert all(key.startswith("net.") for key in model.state_dict())


def test_it_builds_at_most_once() -> None:
    model = TorchConvNet(num_classes=NUM_CLASSES, input_shape=INPUT)

    built = model.solidify()

    assert model.solidify() is built
    assert model.backbone is built


def test_the_seed_reproduces_the_weights() -> None:
    """A config that names a seed gets the same network twice — the same knob the TensorFlow,
    Flax and MLX implementations take."""
    first = TorchConvNet(num_classes=NUM_CLASSES, input_shape=INPUT, seed=7)
    second = TorchConvNet(num_classes=NUM_CLASSES, input_shape=INPUT, seed=7)
    other = TorchConvNet(num_classes=NUM_CLASSES, input_shape=INPUT, seed=8)

    a = next(iter(first.backbone.parameters()))
    b = next(iter(second.backbone.parameters()))
    c = next(iter(other.backbone.parameters()))

    assert torch.allclose(a, b)
    assert not torch.allclose(a, c)


def test_dropout_is_only_present_when_asked_for() -> None:
    """`0.0` disables it — the spec's wording, and a `Dropout(0.0)` layer would still show up
    in the module list and mislead a reader comparing implementations."""
    without = TorchConvNet(num_classes=NUM_CLASSES, input_shape=INPUT, dropout=0.0)
    with_drop = TorchConvNet(num_classes=NUM_CLASSES, input_shape=INPUT, dropout=0.25)

    assert not any(isinstance(m, torch.nn.Dropout) for m in without.backbone.modules())
    assert any(isinstance(m, torch.nn.Dropout) for m in with_drop.backbone.modules())


def test_it_trains() -> None:
    """A few steps must actually move the loss.

    The cheapest guard against a network that builds, runs, reports the right parameter count
    and does not learn — which is exactly what a detached head or a mis-wired optimizer looks
    like from every other test in this file.
    """
    model = TorchConvNet(num_classes=4, input_shape=INPUT, seed=0)
    model.solidify()  # REQUIRED before reading `parameters()` — see the test below
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = torch.nn.CrossEntropyLoss()
    x = torch.randn(8, 3, 16, 16)
    y = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])

    model.train()
    losses = []
    for _ in range(15):
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
        losses.append(loss.detach().item())

    assert losses[-1] < losses[0], f"loss did not descend: {losses[0]:.4f} -> {losses[-1]:.4f}"


def test_parameters_are_empty_until_the_build_and_that_is_the_TimmModel_contract() -> None:
    """The lazy-build hazard, stated rather than hidden.

    `parameters()` is torch's own method walking `_modules`, and it cannot trigger a build —
    so on a never-flowed instance it returns NOTHING and `torch.optim.Adam(model.parameters())`
    dies with "optimizer got an empty parameter list", a message that says nothing about why.

    This is not a defect here, it is `TimmModel`'s contract (`solidify()` is confluid's
    `flow()`-time hook precisely because `flow` never touches property getters), and the
    production path is safe: `LightningClassifier.initialize()` / `FastaiClassifier.initialize()`
    flow the model — which calls `solidify()` — before `configure_optimizers` reads it. A
    directly constructed instance must call `solidify()` or touch `backbone` first.
    """
    model = TorchConvNet(num_classes=NUM_CLASSES, input_shape=INPUT)

    assert not list(model.parameters())
    with pytest.raises(ValueError, match="empty parameter list"):
        torch.optim.Adam(model.parameters())

    model.backbone  # the ergonomic trigger; `solidify()` is the explicit one

    assert list(model.parameters())
    torch.optim.Adam(model.parameters())  # no longer raises


def test_the_configurable_tags_are_declared() -> None:
    """`framework="torch"` is what keeps it out of a Keras trainer's model picker, and what
    lets BOTH torch backends (lightning, fastai) be offered it."""
    # `getattr`, not attribute access: `@configurable` stamps these at run time, so a statically
    # typed class (which this is, unlike its `tf.Module` sibling) has no declared attribute for
    # mypy to find. Same form `tests/test_categories.py` uses for the framework tag.
    assert getattr(TorchConvNet, "__confluid_task__", None) == "classification"
    assert getattr(TorchConvNet, "__confluid_role__", None) == "model"
    assert getattr(TorchConvNet, "__confluid_framework__", None) == "torch"


def test_it_repr_s_honestly_before_the_build() -> None:
    """torch's default `__repr__` on an unbuilt module prints an empty body, which reads as
    "this network has no layers" rather than "this network has not been built yet"."""
    assert "unbuilt" in repr(TorchConvNet(num_classes=NUM_CLASSES))
    assert "unbuilt" not in repr(TorchConvNet(num_classes=NUM_CLASSES).backbone)
