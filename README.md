# SIGMA: speaker-isolated, symptom-structured estimation of depression severity from clinical interviews

Code for SIGMA (Speaker-Isolated Graded-response Multimodal Architecture), which estimates PHQ-8 depression severity from
E-DAIC interviews at the item level. This repository holds the model, training, evaluation and preprocessing code only, starting from the original E-DAIC
data folder. E-DAIC data, derived feature caches, labels and trained checkpoints are not included (see *Data*).

## Pipeline

| stage | what it does | where |
|---|---|---|
| Participant-speech isolation | speaker diarization with [pyannote-audio](https://github.com/pyannote/pyannote-audio) (`speaker-diarization-community-1`) labels the turns of each session; the speaker with the greatest total speaking time is taken as the participant, and the transcript rows are regrouped by the speaker whose turns they fall in (`--rule overlap`: largest overlap; `--rule any`: any overlap with the participant); the participant's rows, with their original timestamps, form the participant-only transcript | `diarize_transcripts.py` |
| Utterance features | one feature vector per participant utterance: text `bge-large-en-v1.5` CLS embedding, L2-normalised (1024-d); audio: mean of the 23 eGeMAPS descriptors over the utterance window; video: mean of 49 OpenFace pose/gaze/AU features over the window, L2-normalised. Video needs no speaker filtering (only the participant is on camera) but is pooled over the same participant windows | `build_cache.py` |
| Encoders | per modality a BiLSTM (50 units per direction) with 4-head self-attention over the utterance sequence; sequences are padded to the longest interview (203 utterances) and packed, so the recurrent layer never reads padding; masked mean pooling | `mtl.py` |
| Cross-modal fusion | cross-attention over time between every ordered pair of modalities (two streams for a bimodal model, six for the trimodal one), each stream mean-pooled over the real utterances of its query modality, concatenated and projected to 200-d. A unimodal model uses the pooled self-attention output directly | `mtl.py` |
| Graded-response head | the fused feature maps to a latent severity trait θ (two factors); item *i* has a discrimination vector a_i and three ordered thresholds b_i1 < b_i2 < b_i3, with P(y_i ≥ k) = σ(a_i·θ − b_ik) | `mtl.py` |
| Loss and training | Weighted Imbalanced Ordinal Log-Loss per item, w(y)^β · \|y − c\|^α with class weights (N/N_c)^β from the training split (β = 0.5, α = 1), learned task weights; AdamW 5e-4, weight decay 1e-3, batch 10 unimodal / 8 multimodal, 50 epochs, gradient clip 1.0, item parameters at 50× learning rate; checkpoint = lowest validation loss. Multimodal models load the unimodal trunks (text frozen) | `mtl.py`, `run.py` |
| Decoding | expected value: PHQ-8 total = sum of the expected item scores (headline); argmax per item summed for comparison | `mtl.py`, `predict.py` |

## Installation

Python 3.10, PyTorch 2.7 (CUDA optional: every run fits on CPU in a few minutes with `OMP_NUM_THREADS=2`). Feature extraction
additionally needs `transformers`, `soundfile` and `librosa`; diarization needs `pyannote.audio` and a Hugging Face token for
which the conditions of the `pyannote/speaker-diarization-community-1` model have been accepted.

```bash
pip install -r requirements.txt
```

## Data

E-DAIC (AVEC 2019) is distributed by USC ICT under a licence agreement and is not included. The code expects:

- `E-DAIC/data/<pid>_P/<pid>_AUDIO.wav`, `E-DAIC/data/<pid>_P/<pid>_Transcript.csv` (columns `Start_Time, End_Time, Text,
  Confidence`) and `E-DAIC/data/<pid>_P/features/` with the distributed `<pid>_OpenSMILE2.3.0_egemaps.csv` and
  `<pid>_OpenFace2.1.0_Pose_gaze_AUs.csv` files (path given by `--edaic`).
- `labels/detailed_lables.csv` with columns `Participant, PHQ8_1_NoInterest, ..., PHQ8_8_Psychomotor` (the eight item scores),
  and `labels/{train,dev,test}_split.csv` with a `Participant_ID` column (the official partition lists). Note that participants
  485 and 486 are swapped in the `Detailed_PHQ8_Labels.csv` shipped with E-DAIC; the item file used here has them corrected.
- Participant-only transcripts `Patient_data/<pid>_P/<pid>_Transcript.csv` (columns `Start_Time, End_Time, Text, Confidence,
  Speaker`) are produced from the two files above by `diarize_transcripts.py` (step 0 below), which also saves the raw speaker
  turns of each session as `<pid>_diarization.csv`. The original E-DAIC transcripts are used unchanged for the no-segmentation
  ablation. The participant transcripts used in the paper were produced by an earlier implementation of the same procedure;
  the assignment of transcript rows to speakers may differ for individual rows.

## Usage

```bash
# 0. speaker diarization -> participant-only transcripts (needs pyannote.audio and a Hugging Face token)
HF_TOKEN=... python diarize_transcripts.py --edaic E-DAIC/data --out Patient_data

# 1. repair the transcript windows (83 of the 275 original transcripts carry the session start time in their last row)
python fix_transcripts.py --src Patient_data --out Patient_data_fixed
python fix_transcripts.py --src E-DAIC/data --out EDAIC_transcripts_fixed             # original transcripts, for the ablation

# 2. per-utterance feature caches -> cache/<feature>/<pid>.npy and cache/n_sentences.json
python build_cache.py --feats bge,egemaps23,openface49 --t_path Patient_data_fixed --edaic E-DAIC/data --out cache
python build_cache.py --feats bge,egemaps23,openface49 --t_path EDAIC_transcripts_fixed --edaic E-DAIC/data --out cache_unseg

# 3. train the final model (T, A, V, T+A, T+V, T+A+V; five seeds) -> runs/grm_full, then the result summary
PYTHON=python GPUS=0,1 PER_GPU=6 ./run_final.sh

# one run by hand
python mtl.py --mods T   --seed 42 --out runs/grm_full/T_seed42 --train
python mtl.py --mods TAV --seed 42 --out runs/grm_full/TAV_seed42 --train --init_dir runs/grm_full --freeze_txt

# 4. leave-one-out ablation (no segmentation, no cross-modal attention, eight item heads, single-item models)
./run_ablation.sh

# 5. W-ImbOLL grid search on the development set (text-only model)
./run_grid_imboll.sh

# inference with a trained run (item and total predictions, expected and argmax, and the latent trait)
python predict.py --ckpt runs/grm_full/TAV_seed42 --split test --out predictions.csv
python predict.py --ckpt runs/grm_full/T_seed42 --pids 300,301
```

`mtl.py --help` lists every switch: the hyperparameters, the ablation switches used by `run_ablation.sh`
(`--no_cross_modal`, `--head items`, `--single_item`, `--cache cache_unseg`) and `--cv_fold` for k-fold cross-validation over
all participants. Each run writes `best_loss.pt`, `metrics.json` (arguments, per-item metrics on dev and test, item parameters,
attention-entropy and modality-permutation diagnostics), `preds_loss.npz` (probabilities and labels) and `log.txt`.

## Contents

- `mtl.py` model, training, evaluation (total and per-item CCC / RMSE / MAE, diagnostics)
- `run.py` scheduler: unimodal runs first, then the multimodal runs initialised from them, several runs per GPU
- `predict.py` inference from a run folder
- `collect.py` result summary: per-seed and mean ± s.d. metrics of every run under `runs/<cond>/`, plus five-seed ensembles
- `diarize_transcripts.py`, `fix_transcripts.py`, `build_cache.py` preprocessing: diarization, transcript repair, feature caches
- `run_final.sh`, `run_ablation.sh`, `run_grid_imboll.sh` the experiments of the paper
- `requirements.txt` package versions used

## Acknowledgement

The per-utterance feature windows and the unimodal trunks follow the QuestMF reference implementation
(Ubiquitous Knowledge Processing Lab, TU Darmstadt, Apache License 2.0), on which this code was originally built.
