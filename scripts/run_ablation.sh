#!/bin/bash
# Leave-one-out ablation of the final model, five seeds -> runs/grm_{full,noseg,nocross,nogrm,nomtl}.
# Rows: full | no segmentation (needs cache_unseg/ built from the original transcripts) | no cross-modal attention |
#       no graded-response head (eight item heads) | no multi-task learning (eight single-item models per seed)
cd "$(dirname "$0")"
PY=${PYTHON:-python}; P=${PREFIX:-grm}
R="$PY -u run.py --txt bge --per_gpu ${PER_GPU:-6} --gpus ${GPUS:-0,1}"
$R --cond ${P}_full    --mods T,A,V,TA,TV,TAV
$R --cond ${P}_nocross --mods TA,TV,TAV --init_cond ${P}_full --flags=--no_cross_modal
$R --cond ${P}_nogrm   --mods T,A,V,TA,TV,TAV --common "--head items"
$R --cond ${P}_nomtl   --mods T,A,V,TA,TV,TAV --single_items
[ -d cache_unseg ] && $R --cond ${P}_noseg --mods T,A,V,TA,TV,TAV --cache cache_unseg
$PY collect.py ${P}_full ${P}_noseg ${P}_nocross ${P}_nogrm ${P}_nomtl
echo ABLATION_DONE
