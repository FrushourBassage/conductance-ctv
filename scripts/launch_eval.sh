#!/usr/bin/env bash
# Launch an evaluation run and write its per-table report.
#
#   scripts/launch_eval.sh main
#   scripts/launch_eval.sh _smoke --steps 4 --out reports/smoke
#   scripts/launch_eval.sh supplementary_transport --out reports/transport
#
# Reports are plain text; the JSON beside them records the configuration, the
# cohort shape and the wall-clock of every stage.
set -euo pipefail

EXPERIMENT="${1:-main}"
shift || true

export PYTHONHASHSEED="${PYTHONHASHSEED:-20260101}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

export PYTHONPATH="${PYTHONPATH:-}:$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/src"

if [[ $# -eq 0 ]]; then
  set -- --out reports
fi

exec python -m conductance_ctv.cli.evaluate "${EXPERIMENT}" "$@"
