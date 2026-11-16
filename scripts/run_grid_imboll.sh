#!/bin/bash
# W-ImbOLL grid search (Supplementary Table S5), text-only model, five seeds per point, development-set selection.
# Loss in the standard form w(y)^beta * |y - c|^alpha (2026-09-10). Writes runs/grm_grid_a<alpha>_b<beta>/T_seed*.
# Usage: [PYTHON=python] [GPUS=0,1] [PER_GPU=6] scripts/run_grid_imboll.sh      (from any directory)
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
R="$PY -u src/run.py --txt bge --per_gpu ${PER_GPU:-6} --gpus ${GPUS:-0,1}"
for a in 1.0 1.5 2.0; do for b in 0.5 1.0; do
  $R --cond grm_grid_a${a}_b${b} --mods T --common "--alpha $a --beta $b"
done; done
echo GRID_DONE
