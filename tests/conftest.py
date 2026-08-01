"""Session-wide setup for the sonair suite.

One thing only, and it is a native-runtime ordering fix rather than a testing convenience.

**pyarrow must not be imported before TensorFlow (macOS, 2026-08-01).** If pyarrow's native
thread pool already exists when TensorFlow initialises, the first eager op inside TensorFlow's
parallel-for (`tf.vectorized_map` -> `pfor` -> `quick_execute`) NEVER RETURNS. Not slow: hung,
with no error, no timeout and no output. Import TensorFlow first and the identical libraries,
threads and call all work.

How it surfaced: `KERAS_BACKEND=tensorflow pytest` stopped dead at
`tests/test_keras_classification.py::test_auc_is_the_metric_that_carries_from_logits`. That test
is the tripwire, not the cause — `keras.metrics.AUC(multi_label=True)` is the only metric in the
suite whose `update_state` reaches `vectorized_map`. The file passes in 3.5 s on its own.

**pyarrow arrives without ever being asked for.** In sonair the chain is
`torchmetrics -> transformers -> sklearn -> pandas -> pyarrow`: torchmetrics eagerly imports its
TEXT metrics, whose BERTScore imports transformers, whose speculative-decoding candidate
generator imports `sklearn.metrics.roc_curve`. recordstream also uses pyarrow directly. So
importing almost anything first-party pulls it in, and `tests/test_categories.py`
(`import sonair.torch_classification`) is enough on its own.

Measured, and each of these was a WRONG theory discarded on evidence:

* It is not sklearn, though sklearn triggers it (it imports pandas, which imports pyarrow).
  Bare `import pyarrow` reproduces it exactly; so does `import pandas`.
* It is not duplicate OpenMP runtimes. Both import orders load an IDENTICAL set of images
  (checked via `_dyld_get_image_name`), TensorFlow ships no OpenMP runtime at all, and loading
  sklearn's `libomp.dylib` directly with `ctypes.CDLL` is harmless.
* It is not torch, scipy, joblib, threadpoolctl, torchvision or matplotlib — all fine.
* No environment knob avoids it: `OMP_NUM_THREADS=1`, `KMP_DUPLICATE_LIB_OK=TRUE`,
  `TF_NUM_INTEROP_THREADS=1`, `TF_NUM_INTRAOP_THREADS=1`, `ARROW_IO_THREADS=1` and
  `pa.set_cpu_count()` all still hang. The OpenMP knobs never applied: the two extra native
  threads `import pyarrow` spawns (1 -> 3) are Arrow's, not OpenMP's.

Import ORDER is the only variable that changes the outcome, and it changes it completely:
TensorFlow first works every time, pyarrow first hangs every time. What is NOT established is
*why* TensorFlow's eager executor deadlocks against a pre-existing Arrow thread pool; that needs
TensorFlow/Arrow internals. The condition is exhaustively reproducible, the mechanism is not
explained.

So when the session is going to run on the TensorFlow engine, TensorFlow is imported HERE.
conftest is imported before any test module, which is the only hook early enough to win the race
— and it wins regardless of which of the several paths would otherwise drag pyarrow in first.
Deliberately conditional: importing TensorFlow into a torch-engine run would cost several
seconds and bind nothing useful, and on a machine without the `tensorflow` extra it is absent.
"""

import os


def _tensorflow_engine_selected() -> bool:
    """Is this session going to bind Keras to TensorFlow?

    Read from the environment rather than from `recordstream.keras.keras_backend()`, because
    calling that would import keras — and keras imports its engine, which is the very thing
    whose ordering is at stake.
    """
    return os.environ.get("KERAS_BACKEND") == "tensorflow"


if _tensorflow_engine_selected():  # pragma: no cover - ordering fix, see the module docstring
    try:
        import tensorflow  # noqa: F401
    except ImportError:
        # No `sonair[tensorflow]` here; the TF-engine tests will skip themselves anyway.
        pass
