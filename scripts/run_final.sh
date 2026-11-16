#!/bin/bash
# Train the final model (all modality rows, five seeds) into runs/grm_full and build the tables.
cd "$(dirname "$0")"
PY=${PYTHON:-python}
$PY -u run.py --txt bge --per_gpu ${PER_GPU:-6} --gpus ${GPUS:-0,1} --cond grm_full --mods T,A,V,TA,TV,TAV
$PY collect.py grm_full
echo FINAL_DONE
