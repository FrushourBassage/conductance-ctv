#!/usr/bin/env bash
# Run the full quality gate and then the verification pass.
#
# The verification pass is last on purpose: the integrity manifest digests the
# tree it finds, so anything written after it would make the manifest stale.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONHASHSEED=20260101

echo "== ruff =="
python -m ruff check .

echo "== black =="
python -m black --check .

echo "== mypy =="
python -m mypy --strict src/conductance_ctv

echo "== pytest =="
python -m pytest -q

echo "== verification pass =="
python -m conductance_ctv.cli.verify

echo "== done =="
