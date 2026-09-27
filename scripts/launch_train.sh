#!/usr/bin/env bash
# Launch a training run, single process or distributed.
#
#   scripts/launch_train.sh main                    # one process on one device
#   scripts/launch_train.sh main 4                  # four ranks via torchrun
#   scripts/launch_train.sh ablation_no_appearance 1 --stage type
#
# The script exports PYTHONHASHSEED so a rerun reproduces the same data order.
set -euo pipefail

EXPERIMENT="${1:-main}"
RANKS="${2:-1}"
shift || true
shift || true
EXTRA=("$@")

export PYTHONHASHSEED="${PYTHONHASHSEED:-20260101}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TOKENIZERS_PARALLELISM=false

export PYTHONPATH="${PYTHONPATH:-}:$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/src"

if [[ "${RANKS}" -le 1 ]]; then
  exec python -m conductance_ctv.cli.train "${EXPERIMENT}" --out runs "${EXTRA[@]}"
fi

exec torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="${RANKS}" \
  -m conductance_ctv.cli.train "${EXPERIMENT}" --out runs "${EXTRA[@]}"
