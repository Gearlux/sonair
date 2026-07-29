"""The predict path: what the model emits and what the sink receives.

This surface had NO coverage, which is how a real defect survived: ``predict_step`` returned
the raw ``[N, C]`` logits tensor while ``recordstream.predictions.ClassificationPredictionsSink.write``
reads ``prediction.get("probs")`` — so a wired sink raised
``AttributeError: 'Tensor' object has no attribute 'get'`` on the first batch, and the
list-wrapping in ``_write_predictions`` would have written ONE entry for an N-record batch
anyway. Both halves are pinned here.
"""

from typing import Any, Dict, List

import pytest
import torch
import torch.nn as nn
from recordstream import Image, Label, collate_records

from sonair.lightning_classifier import LightningClassifier, _per_record_predictions


class _TinyClassifier(nn.Module):
    def __init__(self, num_classes: int = 3, input_size: int = 4) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)  # type: ignore[no-any-return]


class _RecordingSink:
    """Stands in for a real predictions sink: records the (prediction, metadata) pairs."""

    def __init__(self) -> None:
        self.writes: List[Any] = []

    def write(self, prediction: Any, metadata: Dict[str, Any]) -> None:
        self.writes.append((prediction, metadata))

    def close(self) -> None:
        pass


def _batch(n: int = 4, num_classes: int = 3) -> Any:
    records = [
        {"image": Image(torch.randn(4).numpy(), layout="CHW"), "class": Label(i % num_classes), "idx": i}
        for i in range(n)
    ]
    return collate_records(records)


def _predictor(num_classes: int = 3) -> LightningClassifier:
    clf = LightningClassifier(model=_TinyClassifier(num_classes=num_classes), task="predict")
    clf.num_classes = num_classes
    return clf


# --------------------------------------------------------------------------- #
# predict_step emits the CONTRACT, not raw logits
# --------------------------------------------------------------------------- #


def test_predict_step_returns_a_classification_output() -> None:
    out = _predictor().predict_step(_batch(4), 0)

    assert set(out) == {"logits", "probs", "class_idx"}
    assert out["logits"].shape == (4, 3)
    assert out["probs"].shape == (4, 3)
    assert out["class_idx"].shape == (4,)


def test_probs_are_a_distribution_and_class_idx_its_argmax() -> None:
    """The sink ranks by `probs` — raw logits would rank the same but report bogus confidences."""
    out = _predictor().predict_step(_batch(4), 0)

    assert torch.allclose(out["probs"].sum(dim=-1), torch.ones(4), atol=1e-6)
    assert torch.equal(out["class_idx"], out["logits"].argmax(dim=-1))
    assert out["class_idx"].dtype is torch.int64


# --------------------------------------------------------------------------- #
# the sink receives ONE prediction per record, paired with that record's metadata
# --------------------------------------------------------------------------- #


def test_sink_receives_one_write_per_record() -> None:
    clf = _predictor()
    clf.predictions_sink = _RecordingSink()
    clf.predict_step(_batch(4), 0)

    assert len(clf.predictions_sink.writes) == 4, "a batched output must be sliced, not written whole"


def test_each_write_carries_that_records_metadata() -> None:
    clf = _predictor()
    clf.predictions_sink = _RecordingSink()
    clf.predict_step(_batch(3), 0)

    assert [m["idx"] for _, m in clf.predictions_sink.writes] == [0, 1, 2]


def test_each_write_is_an_unbatched_per_record_mapping() -> None:
    """What the sink actually indexes: `probs` must be `[C]`, not `[N, C]`."""
    clf = _predictor(num_classes=3)
    clf.predictions_sink = _RecordingSink()
    clf.predict_step(_batch(4), 0)

    prediction, _ = clf.predictions_sink.writes[0]
    assert prediction["probs"].shape == (3,)
    assert prediction["class_idx"].ndim == 0


def test_the_real_sink_consumes_the_output() -> None:
    """End to end against `recordstream.predictions` — the exact pairing that used to raise."""
    from recordstream import ClassificationPredictionsSink

    written: List[Any] = []

    def _capture(record: Any) -> Any:
        written.append(record)
        return record

    clf = _predictor(num_classes=3)
    clf.predictions_sink = ClassificationPredictionsSink(ops=[_capture], class_names={0: "a", 1: "b", 2: "c"})
    clf.predict_step(_batch(4), 0)

    assert len(written) == 4
    meta = written[0]["metadata"]
    assert meta["predicted_class_label"] in ("a", "b", "c")
    assert 0.0 <= meta["predicted_confidence"] <= 1.0


# --------------------------------------------------------------------------- #
# _per_record_predictions — the shapes a model can hand back
# --------------------------------------------------------------------------- #


def test_a_batched_mapping_is_sliced_row_wise() -> None:
    sliced = _per_record_predictions({"probs": torch.zeros(3, 2), "class_idx": torch.zeros(3)})
    assert len(sliced) == 3
    assert sliced[0]["probs"].shape == (2,)


def test_a_list_is_already_per_record() -> None:
    assert _per_record_predictions([{"a": 1}, {"a": 2}]) == [{"a": 1}, {"a": 2}]


def test_a_predictions_wrapper_is_unwrapped() -> None:
    """A detector's `{"predictions": [...]}` convention."""
    assert _per_record_predictions({"predictions": [{"boxes": 1}, {"boxes": 2}]}) == [{"boxes": 1}, {"boxes": 2}]


def test_a_ragged_mapping_falls_back_to_one_prediction() -> None:
    """Disagreeing lengths mean it is not a batch — pass it through rather than mis-slicing."""
    ragged = {"probs": torch.zeros(3, 2), "extra": torch.zeros(5)}
    assert _per_record_predictions(ragged) == [ragged]


def test_a_bare_value_is_one_prediction() -> None:
    tensor = torch.zeros(3)
    assert _per_record_predictions(tensor) == [tensor]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
