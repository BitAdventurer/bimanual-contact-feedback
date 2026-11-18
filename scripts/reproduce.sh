#!/usr/bin/env bash
# Run from a checkout or extracted ZIP with the documented CPU dependencies.
set -euo pipefail
if [[ $# -gt 2 ]] || [[ ! ${2:-2} =~ ^[1-9][0-9]*$ ]]; then
  echo "Usage: bash scripts/reproduce.sh [new-output-directory] [workers]" >&2
  exit 2
fi
destination=${1:-runs/full}
workers=${2:-2}
if [[ $destination != /* ]]; then
  destination="$PWD/$destination"
fi
repo_dir=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$(dirname "$destination")"
mkdir "$destination"
cd "$repo_dir"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python analysis/reproduce.py --output "$destination/alignment.json"
python mechanism/verify.py | tee "$destination/contact-cut.json"
python -m mechanism.replay --workers "$workers" | tee "$destination/native-contact-cut.json"
python simulation/replay.py test-grid --workers "$workers" \
  --output "$destination/native-grid.npz" --report "$destination/native-grid.json"
python training/run.py fit --out "$destination/contact" --workers "$workers" --controls
python training/run.py evaluate --run "$destination/contact" \
  --native-test "$destination/native-grid.npz" --json "$destination/native-test.json"
echo "Results: $destination/native-test.json"
