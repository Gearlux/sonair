# sonair — backlog

Open work for this project. Cross-cutting / multi-project initiatives live in the
workspace root `TASKS.md`. Completed items are not archived here — git history is the record.

- [ ] Drive classification from local annotation corpus + IQ→spec→render pipeline (parity with deltaid). @medium
- [ ] CI: Jenkinsfile + GitHub Actions via `aisland jenkins scaffold --project sonair`. @medium
- [ ] ClassificationMetricAdapter for evaluating per-sample dict outputs (so a single model class works in evaluate + predict). @low
  - [ ] **Image resize for the flat sonair config** @feature — `classification_collate_fn` requires same-size images per batch (no resize). Add resize (a `HuggingFaceSource.image_size` and/or a model-keyed transform) so arbitrary HF image datasets (e.g. mixed sizes) train; pairs with proper ImageNet normalization for pretrained backbones.
  - [ ] **Multi-label string classification (sonair)** @feature — string *sequence* targets currently raise `NotImplementedError` in the classification trainer/evaluator. Support needs a multi-hot encoder (sklearn `MultiLabelBinarizer` analogue → a `sampleflux` op), a BCE-style loss, and collate changes (float `[N, C]` targets). `sampleflux.LabelMap` already gives the label↔id vocabulary to build on.
