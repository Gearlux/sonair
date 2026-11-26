# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the versioning is
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- **sonair requires the newest workspace releases** — confluid 0.5.0 (was `>=0.3.1`), loggair 0.3.0 (was
  `>=0.2.0`), recordstream 0.1.0a4 (was `>=0.1.0a2`) and matrainer 0.1.0a3 (was `>=0.1.0a1`), in the core and in
  every extra; the suite is tested against them.
- **The licence is now the PolyForm Noncommercial License 1.0.0** (`LICENSE`, `pyproject.toml`): free for personal, academic and other noncommercial use; commercial use needs a separate licence from the author. It was MIT; the versions released under MIT keep that licence.

## [0.1.0a1] — 2026-09-28

First public pre-release. The surface it ships:

### Backends
- **Eight training engines over one config**, each a runnable class behind its own extra:
  `LightningClassifier` (`[lightning]`, the reference), `TorchClassifier` (`[torch]`, a
  hand-written loop), `KerasClassifier` (`[keras]`), `TensorFlowClassifier` (`[tensorflow]`,
  `tf.data` + `GradientTape`), `FastaiClassifier` (`[fastai]`), `HuggingFaceClassifier`
  (`[huggingface]`, `transformers.Trainer`), `JaxClassifier` (`[jax]`) and `MlxClassifier`
  (`[mlx]`, Apple silicon). Each runnable dispatches on `task`: `fit` / `evaluate` / `test` /
  `predict`.
- **Backend-free core:** `import sonair` loads no ML framework. Touching a backend that is not
  installed raises an `ImportError` naming the extra to install.

### Models
- `TimmModel` — any timm backbone by `model_name` (default `efficientvit_b0`).
- `KerasModel` (`keras.applications`) and `HuggingFaceModel` (any Hub image-classification
  checkpoint).
- The `sonair.convnet` package: one small ConvNet spec with five native implementations
  (`TorchConvNet`, `KerasConvNet`, `TensorFlowConvNet`, `FlaxConvNet`, `MlxConvNet`) whose
  parameter counts are checked against the spec, for comparing engines on one network.

### Data, losses and metrics
- `ImageFolderSource` / `ImageFolderSink` for the folder-per-class layout.
- Class-name string labels are encoded to ids automatically, and the same name↔id map is reloaded
  for evaluate and predict; inverse-frequency class weights balance the loss by default;
  multi-label classification.
- Upstream losses and metrics registered per engine (torch, Keras, fastai) with no wrappers;
  per-split metric choices, macro-F1 on every backend, and confusion matrices.

### Command line
- `sonair train | evaluate | predict <config.yaml> [--framework <name>]` and `sonair frameworks`;
  the same config also runs under `recordstream run`.
- Optional experiment tracking through matrainer (CSV, MLflow, TensorBoard, …) and a run record
  describing the model that was trained.
