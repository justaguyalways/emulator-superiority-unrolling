#!/bin/bash
# usage: ./run_grid.sh <grid> [max_runs]   -- runs a grid in recycled processes (frees JAX compile caches)
cd "$(dirname "$0")"
G=$1; K=${2:-3}
while true; do
  .venv/bin/python src/run_sweep.py --grid "$G" --max-runs "$K"
  rc=$?
  [ $rc -eq 3 ] || exit $rc
done
