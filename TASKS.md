# sonair — backlog

Open work for this project. Cross-cutting / multi-project initiatives live in the
workspace root `TASKS.md`. Completed items are not archived here — git history is the record.

- [ ] Implement the `keras` backend: `sonair/keras_classifier.py` + replace the `keras` template block in `sonair/cli.py`. Keras 3 is the most instructive second backend (one API over TF / JAX / PyTorch), so it doubles as the multi-backend story. See `docs/frameworks.md`. @medium @framework
- [ ] Implement the `tensorflow` backend: `sonair/tensorflow_classifier.py` + replace the `tensorflow` template block in `sonair/cli.py`. @low @framework
- [ ] **StreamStudio: framework-aware object sockets.** confluid 0.3.0 ships the `framework` axis and navigaitor filters slot options by it, but StreamStudio still lets any `RECORDSTREAM_OBJECT:model` dock into any model slot regardless of engine. Qualifying the socket type (`…:model:torch`) is a BREAKING workflow-format change — ComfyUI enforces socket compatibility by string equality and persists the strings in saved `workflow.json` — so it needs its own decision + a migration story. Until then a canvas can still wire a torch loss into a Keras trainer. See `docs/architecture.md` §2. @medium @architecture
- [ ] Implement the `jax` / `huggingface` / `mlx` backends (templates registered; a sub-app each, modelled on `lightning`). @low @framework
- [ ] Consider a NON-deep-learning backend (`sklearn`, or XGBoost/LightGBM) — no epochs, no autograd, so it stress-tests the claim that the `runnable:` + `task:` shape is engine-agnostic. @low @framework
- [ ] Demonstrate the LOGGING axis as explicitly as the framework axis — example configs wiring TensorBoard / MLflow / W&B / CSV through `logger:` + marainer `trackers:`, so the demo covers both dimensions it claims. @medium @demo
- [ ] Drive classification from local annotation corpus + IQ→spec→render pipeline (parity with deltaid). @medium
- [ ] CI: Jenkinsfile + GitHub Actions via `aisland jenkins scaffold --project sonair`. @medium
- [ ] ClassificationMetricAdapter for evaluating per-record dict outputs (so a single model class works in evaluate + predict). @low
  - [ ] **Image resize for the flat sonair config** @feature — `classification_collate_fn` requires same-size images per batch (no resize). Add resize (a `HuggingFaceSource.image_size` and/or a model-keyed transform) so arbitrary HF image datasets (e.g. mixed sizes) train; pairs with proper ImageNet normalization for pretrained backbones.
  - [ ] **Multi-label predictions sink** @feature — `marainer.sinks.ClassificationPredictionsSink` writes top-k entries per record, which is the wrong shape for multi-label inference (the natural output is *every* class over a threshold, count unknown). Needs a threshold knob on the sink and a decision about `marainer.outputs.ClassificationOutput`, which currently models one `predicted_class`. Multi-label TRAINING/EVAL landed 2026-07-29; only the predict-side sink is open.
