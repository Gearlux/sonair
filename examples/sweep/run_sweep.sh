#!/usr/bin/env bash
# Train the SAME MNIST config on every backend this machine has, twice per backend — once on a
# model-zoo backbone and once on the shared conv spec — and report the lot from one tracker.
#
#   examples/sweep/run_sweep.sh                     # every installed backend, into MLflow
#   examples/sweep/run_sweep.sh --tracker csv       # ...into the always-available CSV backend
#   examples/sweep/run_sweep.sh --only jax,mlx      # just those two
#   examples/sweep/run_sweep.sh --dry-run           # print the matrix and the commands, run nothing
#
# ---------------------------------------------------------------------------
# What it is for
# ---------------------------------------------------------------------------
# The project's claim is that the config is the program: the same file trains under eight
# engines and only a `!scope:framework=` block changes. A sweep is that claim, executed —
# every run below is the SAME `examples/sweep/all_backends.yaml` with two flags changed, and
# the tracker at the end is where they become comparable.
#
# ---------------------------------------------------------------------------
# Three things it does that are not obvious
# ---------------------------------------------------------------------------
#  1. INSTALLED, not declared. The CLI advertises all eight backends whether or not their
#     framework is importable, so the matrix is built by PROBING (`find_spec`) rather than by
#     reading `sonair frameworks`. A machine with three extras runs three backends and says so,
#     instead of failing six times.
#  2. ONE PROCESS PER RUN, which is not a stylistic choice: Keras 3 binds its compute engine at
#     import time, so `keras` and `tensorflow` cannot both run in one interpreter (the CLI's
#     `_pin_keras_backend` sets `KERAS_BACKEND` from argv before anything imports keras). A
#     subprocess per run is what makes the sweep possible at all.
#  3. FAILURE IS ISOLATED. One backend dying must not abort the other twelve — the point of a
#     sweep is the comparison, and a missing row is itself a result. Each failure is recorded
#     and printed in the summary with its log file.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"   # …/sonair
WORKSPACE_DIR="$(cd "${PROJECT_DIR}/.." && pwd)"
CONFIG="examples/sweep/all_backends.yaml"
# The report is matrainer's, not this project's: naming an experiment and reading a sweep back are
# queries against a TRACKER, and three task projects carried byte-identical copies of them before the
# lift. What stays here is the matrix and the per-engine quirks — the halves that are genuinely this
# project's. matrainer ships no binary, so it is a module main.
REPORT=(python -m matrainer.sweep)

TRACKER="mlflow"
ONLY=""
DRY_RUN=0
EPOCHS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tracker)  TRACKER="$2"; shift 2 ;;
    --only)     ONLY="$2"; shift 2 ;;
    --epochs)   EPOCHS="$2"; shift 2 ;;
    --dry-run)  DRY_RUN=1; shift ;;
    -h|--help)  sed -n '2,12p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *)          echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

# `project.bashrc` activates .venv AND exports the workspace .env — the mandated incantation,
# because the configs read `$DATA_ROOT`-style variables through confluid substitution.
#
# `set +u` around it on purpose: an activation script is a user profile, written against an
# interactive shell where referencing an unset variable is normal. Under `nounset` the first
# such reference aborts this script INSIDE the source, before a single line of its own output —
# which is exactly how it failed the first time it was run.
set +u
# shellcheck disable=SC1091
source "${WORKSPACE_DIR}/project.bashrc" >/dev/null 2>&1
set -u
cd "${PROJECT_DIR}" || exit 1

# ---------------------------------------------------------------------------
# The matrix: which backends are here, and what each can be given as a model
# ---------------------------------------------------------------------------
# Two gaps, both structural rather than missing work (see `all_backends.yaml`): jax and mlx have
# no zoo to wrap, and huggingface has no convnet because a `transformers.Trainer` drives models
# that compute their own loss.
#
#   backend : import probe : variants
BACKENDS=(
  "lightning:pytorch_lightning:zoo convnet"
  "torch:torch:zoo convnet"
  "fastai:fastai:zoo convnet"
  "keras:keras:zoo convnet"
  "tensorflow:tensorflow:zoo convnet"
  "huggingface:transformers:zoo"
  "jax:flax:convnet"
  "mlx:mlx:convnet"
)

installed() { python -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('$1') else 1)" 2>/dev/null; }

MATRIX=()
SKIPPED=()
for entry in "${BACKENDS[@]}"; do
  IFS=":" read -r backend probe variants <<< "${entry}"
  if [[ -n "${ONLY}" && ",${ONLY}," != *",${backend},"* ]]; then continue; fi
  if ! installed "${probe}"; then
    SKIPPED+=("${backend} (no ${probe})")
    continue
  fi
  for variant in ${variants}; do MATRIX+=("${backend}:${variant}"); done
done

if [[ ${#MATRIX[@]} -eq 0 ]]; then
  echo "nothing to run: no selected backend has its framework installed." >&2
  [[ ${#SKIPPED[@]} -gt 0 ]] && printf '  skipped: %s\n' "${SKIPPED[@]}" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# The tracker has to be up before the first run, not after the first failure
# ---------------------------------------------------------------------------
if [[ "${TRACKER}" == "mlflow" && ${DRY_RUN} -eq 0 ]]; then
  if ! aisland mlflow status 2>/dev/null | grep -qi "running"; then
    echo "==> starting MLflow (aisland mlflow start)"
    aisland mlflow start || { echo "could not start MLflow" >&2; exit 1; }
  fi
  echo "==> MLflow: $(aisland mlflow info 2>/dev/null | grep -i 'url' | head -1 | tr -s ' ')"
fi

# `Sun_Aug_2_2026_3` — the date, then which run of the sweep this is. The counter comes from the
# TRACKER (see sweep_report.py), so a wiped tracker restarts at 1 rather than counting a history
# that is no longer there.
PREFIX="$(date '+%a_%b_%-d_%Y')"
if ! EXPERIMENT="$("${REPORT[@]}" next-experiment "${TRACKER}" "${PREFIX}")" || [[ -z "${EXPERIMENT}" ]]; then
  # Checked rather than assumed: the first run of this script hit an MLflow whose SQLite schema
  # had never been created, so every API call 500'd. Unguarded, the empty name flowed into
  # `--experiment_name ''` and thirteen runs would have trained into an unnamed experiment.
  echo "could not name the experiment: the ${TRACKER} tracker did not answer." >&2
  [[ "${TRACKER}" == "mlflow" ]] && echo "  check: aisland mlflow status && aisland mlflow restart" >&2
  exit 1
fi

echo
echo "experiment : ${EXPERIMENT}"
echo "tracker    : ${TRACKER}"
echo "config     : ${CONFIG}"
echo "runs       : ${#MATRIX[@]}  (${MATRIX[*]})"
[[ ${#SKIPPED[@]} -gt 0 ]] && printf 'skipped    : %s\n' "${SKIPPED[*]}"
echo

LOG_DIR="sweep_logs/${EXPERIMENT}"
mkdir -p "${LOG_DIR}"
PASSED=(); FAILED=()

for pair in "${MATRIX[@]}"; do
  IFS=":" read -r backend variant <<< "${pair}"
  run_name="${backend}_${variant}"
  log="${LOG_DIR}/${run_name}.log"

  cmd=(sonair train "${CONFIG}"
       --framework "${backend}"
       --tracker "${TRACKER}"
       --experiment_name "${EXPERIMENT}"
       --run_name "${run_name}")
  [[ "${variant}" == "convnet" ]] && cmd+=(--model convnet)
  [[ -n "${EPOCHS}" ]] && cmd+=(--max_epochs "${EPOCHS}")

  if [[ ${DRY_RUN} -eq 1 ]]; then
    printf '%s\n' "${cmd[*]}"
    continue
  fi

  printf '==> %-24s ' "${run_name}"
  start=${SECONDS}
  if "${cmd[@]}" >"${log}" 2>&1; then
    printf 'ok   (%ds)\n' "$((SECONDS - start))"
    PASSED+=("${run_name}")
  else
    printf 'FAIL (%ds) — %s\n' "$((SECONDS - start))" "${log}"
    # The last non-empty line is almost always the exception; enough to triage without opening it.
    tail -n 20 "${log}" | grep -vE '^\s*$' | tail -n 1 | sed 's/^/      /'
    FAILED+=("${run_name}")
  fi
done

[[ ${DRY_RUN} -eq 1 ]] && exit 0

echo
echo "============================================================"
echo " ${EXPERIMENT} — ${#PASSED[@]} ok, ${#FAILED[@]} failed"
echo "============================================================"
[[ ${#FAILED[@]} -gt 0 ]] && printf '  failed: %s\n  logs:   %s/\n' "${FAILED[*]}" "${LOG_DIR}"
echo
"${REPORT[@]}" report "${TRACKER}" "${EXPERIMENT}"

# A failed run is a failed sweep for CI purposes, but only AFTER the report — the comparison is
# the deliverable and it is still worth reading when one backend is broken.
[[ ${#FAILED[@]} -eq 0 ]]
