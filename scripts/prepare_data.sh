#!/usr/bin/env bash
# Stage the public pretraining corpora and build the derived cohort assembly.
#
# The clinical cohort of the manuscript is not distributed, so the only material
# this script fetches is the public pretraining data. Both corpora are large and
# the transfer rate to them is throttled from some networks, so the downloads are
# explicit and resumable rather than implicit in a training command.
#
#   scripts/prepare_data.sh check      # report what is already staged
#   scripts/prepare_data.sh public     # fetch both corpora into data/public
#   scripts/prepare_data.sh derived    # assemble the generated cohort
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-${ROOT}/data}"
PUBLIC="${DATA_ROOT}/public"

TOTALSEGMENTATOR_URL="https://zenodo.org/records/10047292/files/Totalsegmentator_dataset_v201.zip"
ABDOMENATLAS_URL="https://github.com/MrGiovanni/AbdomenAtlas"
TOTALSEGMENTATOR_SHA_URL="https://zenodo.org/api/records/10047292"

usage() {
  sed -n '2,12p' "${BASH_SOURCE[0]}"
  exit 1
}

check() {
  echo "data root: ${DATA_ROOT}"
  for path in "${PUBLIC}/totalsegmentator_v201" "${PUBLIC}/abdomenatlas_1_0"; do
    if [[ -e "${path}" ]]; then
      echo "  present : ${path}"
    else
      echo "  missing : ${path}"
    fi
  done
  echo
  echo "TotalSegmentator release 2.0.1 is distributed under CC BY 4.0;"
  echo "AbdomenAtlas is distributed under the licence stated at its repository."
  echo "See NOTICE for the full provenance of both."
}

fetch_public() {
  mkdir -p "${PUBLIC}"
  if [[ ! -f "${PUBLIC}/Totalsegmentator_dataset_v201.zip" ]]; then
    echo "fetching the TotalSegmentator dataset archive (about 23.6 GB)"
    curl -L --retry 5 --retry-delay 10 --continue-at - \
      -o "${PUBLIC}/Totalsegmentator_dataset_v201.zip" \
      "${TOTALSEGMENTATOR_URL}"
  fi
  echo "record metadata:"
  curl -s "${TOTALSEGMENTATOR_SHA_URL}" | python -c \
    "import json,sys; d=json.load(sys.stdin); print('  version', d['metadata']['version']); print('  files', [(f['key'], round(f['size']/1e9, 2)) for f in d['files']])"
  mkdir -p "${PUBLIC}/totalsegmentator_v201"
  unzip -q -n "${PUBLIC}/Totalsegmentator_dataset_v201.zip" -d "${PUBLIC}/totalsegmentator_v201"
  echo "AbdomenAtlas is fetched through its own project channels:"
  echo "  ${ABDOMENATLAS_URL}"
  mkdir -p "${PUBLIC}/abdomenatlas_1_0"
}

prepare_derived() {
  export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
  mkdir -p "${DATA_ROOT}/derived"
  python -m conductance_ctv.cli.hydraulics --shape septum --area 1800 --channels 3 --step 0.5 --json \
    > "${DATA_ROOT}/derived/septum_reference.json"
  python -m conductance_ctv.cli.architectures --json > "${DATA_ROOT}/derived/architecture_inventory.json"
  echo "derived descriptors written to ${DATA_ROOT}/derived"
}

case "${1:-}" in
  check) check ;;
  public) fetch_public ;;
  derived) prepare_derived ;;
  *) usage ;;
esac
