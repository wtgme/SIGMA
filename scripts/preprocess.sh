#!/bin/bash
# Preprocessing, from the E-DAIC data folder to the per-utterance feature caches (README steps 0-2). Writes under data/.
#   0. speaker diarization -> participant-only transcripts      data/Patient_data/       (needs pyannote.audio and HF_TOKEN)
#   1. transcript-window repair                                  data/Patient_data_fixed/, data/EDAIC_transcripts_fixed/
#   2. per-utterance feature caches                              data/cache/, data/cache_unseg/ (no-segmentation ablation)
# Usage: HF_TOKEN=... [PYTHON=python] [EDAIC=data/E-DAIC/data] scripts/preprocess.sh      (from any directory)
# Steps whose outputs already exist are cheap to rerun: diarization and feature files that are present are not recomputed.
set -e
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}; EDAIC=${EDAIC:-data/E-DAIC/data}
[ -d "$EDAIC" ] || { echo "E-DAIC data folder not found: $EDAIC (see data/README.md)"; exit 1; }
$PY preprocessing/diarize_transcripts.py --edaic "$EDAIC" --out data/Patient_data
$PY preprocessing/fix_transcripts.py --src data/Patient_data --out data/Patient_data_fixed
$PY preprocessing/fix_transcripts.py --src "$EDAIC" --out data/EDAIC_transcripts_fixed
$PY preprocessing/build_cache.py --feats bge,egemaps23,openface49 --t_path data/Patient_data_fixed --edaic "$EDAIC" --out data/cache
$PY preprocessing/build_cache.py --feats bge,egemaps23,openface49 --t_path data/EDAIC_transcripts_fixed --edaic "$EDAIC" --out data/cache_unseg
echo PREPROCESS_DONE
