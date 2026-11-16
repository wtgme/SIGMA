# SIGMA: speaker-isolated, symptom-structured multimodal estimation of depression severity from clinical interviews

Code for SIGMA (Speaker-Isolated Graded-response Multimodal Architecture), which estimates PHQ-8 depression severity from
E-DAIC interviews at the item level. This repository holds the model, training, evaluation and preprocessing code only, starting from the original E-DAIC
data folder. E-DAIC data, derived feature caches, labels and trained checkpoints are not included (see *Data*).

## Repository structure

```
SIGMA/
├── preprocessing/               # E-DAIC session folders -> per-utterance feature caches
│   ├── diarize_transcripts.py   #   0. speaker diarization -> participant-only transcripts
│   ├── fix_transcripts.py       #   1. repair of the transcript windows
│   └── build_cache.py           #   2. text / audio / video features per utterance
├── src/                         # model, training and evaluation
│   ├── mtl.py                   #   SIGMA model; trains and evaluates one run
│   ├── run.py                   #   scheduler: unimodal runs, then the multimodal runs initialised from them, several per GPU
│   ├── collect.py               #   result tables: per-seed, mean ± s.d. and five-seed ensembles of runs/<cond>/
│   └── predict.py               #   inference from a trained run
├── scripts/                     # one entry point per stage of the paper
│   ├── preprocess.sh            #   preprocessing steps 0-2 for the whole dataset
│   ├── run_final.sh             #   final model, all modality combinations, five seeds
│   ├── run_ablation.sh          #   leave-one-out ablation
│   └── run_grid_imboll.sh       #   W-ImbOLL hyperparameter grid
├── data/                        # not tracked: E-DAIC, labels and derived data (layout in data/README.md)
├── runs/                        # not tracked: created by training, one folder per condition and run
└── requirements.txt             # package versions used for the paper
```

Run the Python commands below from the repository root; the shell scripts can be started from any directory.

## Pipeline

| stage | what it does | where |
|---|---|---|
| Participant-speech isolation | speaker diarization with [pyannote-audio](https://github.com/pyannote/pyannote-audio) (`speaker-diarization-community-1`) labels the turns of each session; the speaker with the greatest total speaking time is taken as the participant, and the transcript rows are regrouped by the speaker whose turns they fall in (`--rule overlap`: largest overlap; `--rule any`: any overlap with the participant); the participant's rows, with their original timestamps, form the participant-only transcript | `preprocessing/diarize_transcripts.py` |
| Utterance features | one feature vector per participant utterance: text `bge-large-en-v1.5` CLS embedding, L2-normalised (1024-d); audio: mean of the 23 eGeMAPS descriptors over the utterance window; video: mean of 49 OpenFace pose/gaze/AU features over the window, L2-normalised. Video needs no speaker filtering (only the participant is on camera) but is pooled over the same participant windows | `preprocessing/build_cache.py` |
| Encoders | per modality a BiLSTM (50 units per direction) with 4-head self-attention over the utterance sequence; sequences are padded to the longest interview (203 utterances) and packed, so the recurrent layer never reads padding; masked mean pooling | `src/mtl.py` |
| Cross-modal fusion | cross-attention over time between every ordered pair of modalities (two streams for a bimodal model, six for the trimodal one), each stream mean-pooled over the real utterances of its query modality, concatenated and projected to 200-d. A unimodal model uses the pooled self-attention output directly | `src/mtl.py` |
| Graded-response head | the fused feature maps to a latent severity trait θ (two factors); item *i* has a discrimination vector a_i and three ordered thresholds b_i1 < b_i2 < b_i3, with P(y_i ≥ k) = σ(a_i·θ − b_ik) | `src/mtl.py` |
| Loss and training | Weighted Imbalanced Ordinal Log-Loss per item, w(y)^β · \|y − c\|^α with class weights (N/N_c)^β from the training split (β = 0.5, α = 1), learned task weights; AdamW 5e-4, weight decay 1e-3, batch 10 unimodal / 8 multimodal, 50 epochs, gradient clip 1.0, item parameters at 50× learning rate; checkpoint = lowest validation loss. Multimodal models load the unimodal trunks (text frozen) | `src/mtl.py`, `src/run.py` |
| Decoding | expected value: PHQ-8 total = sum of the expected item scores (headline); argmax per item summed for comparison | `src/mtl.py`, `src/predict.py` |

## Installation

Python 3.10, PyTorch 2.7 (CUDA optional: every run fits on CPU in a few minutes with `OMP_NUM_THREADS=2`). Feature extraction
additionally needs `transformers`, `soundfile` and `librosa`; diarization needs `pyannote.audio` and a Hugging Face token for
which the conditions of the `pyannote/speaker-diarization-community-1` model have been accepted.

```bash
git clone https://github.com/wtgme/SIGMA.git && cd SIGMA
conda create -n sigma python=3.10 && conda activate sigma     # or any Python 3.10 environment
pip install -r requirements.txt
```

## Data

E-DAIC (AVEC 2019) is distributed by USC ICT under a licence agreement and is not included, nor are the derived feature
caches, labels or trained checkpoints. Put the E-DAIC session folders and the label files under `data/` (copy or symlink);
[`data/README.md`](data/README.md) gives the full layout:

- `data/E-DAIC/data/<pid>_P/` with `<pid>_AUDIO.wav`, `<pid>_Transcript.csv` (columns `Start_Time, End_Time, Text,
  Confidence`) and the distributed `features/<pid>_OpenSMILE2.3.0_egemaps.csv` and `features/<pid>_OpenFace2.1.0_Pose_gaze_AUs.csv`.
- `data/E-DAIC/labels/`, as distributed: `detailed_lables.csv` (sic; columns `Participant, PHQ8_1_NoInterest, ...,
  PHQ8_8_Psychomotor`, the eight item scores of all 275 participants) and `{train,dev,test}_split.csv` (the official
  partition, column `Participant_ID`). The `Detailed_PHQ8_Labels.csv` in the same folder is not used: it has participants
  485 and 486 swapped.

The simplest setup is a single symlink: `ln -s /path/to/E-DAIC data/E-DAIC`.

Participant-only transcripts (`data/Patient_data/<pid>_P/<pid>_Transcript.csv`, columns `Start_Time, End_Time, Text,
Confidence, Speaker`) are produced from the E-DAIC files by `preprocessing/diarize_transcripts.py`, which also saves the raw
speaker turns of each session as `<pid>_diarization.csv`. The original E-DAIC transcripts are used unchanged for the
no-segmentation ablation. The participant transcripts used in the paper were produced by an earlier implementation of the
same procedure; the assignment of transcript rows to speakers may differ for individual rows.

## Quick start

```bash
# 1. preprocessing: diarization, transcript repair, feature caches -> data/   (needs pyannote.audio and a Hugging Face token)
HF_TOKEN=... scripts/preprocess.sh

# 2. train the final model (T, A, V, T+A, T+V, T+A+V; five seeds) -> runs/grm_full/, then print the result tables
GPUS=0,1 PER_GPU=6 scripts/run_final.sh

# 3. PHQ-8 item and total predictions with a trained run
python src/predict.py --ckpt runs/grm_full/TAV_seed42 --split test --out predictions.csv
```

## Usage

### Preprocessing (the steps of `scripts/preprocess.sh`)

```bash
# 0. speaker diarization -> participant-only transcripts (needs pyannote.audio and a Hugging Face token)
HF_TOKEN=... python preprocessing/diarize_transcripts.py --edaic data/E-DAIC/data --out data/Patient_data

# 1. repair the transcript windows (83 of the 275 original transcripts carry the session start time in their last row)
python preprocessing/fix_transcripts.py --src data/Patient_data --out data/Patient_data_fixed
python preprocessing/fix_transcripts.py --src data/E-DAIC/data --out data/EDAIC_transcripts_fixed   # original transcripts, for the ablation

# 2. per-utterance feature caches -> data/cache/<feature>/<pid>.npy and data/cache/n_sentences.json
python preprocessing/build_cache.py --feats bge,egemaps23,openface49 --t_path data/Patient_data_fixed --edaic data/E-DAIC/data --out data/cache
python preprocessing/build_cache.py --feats bge,egemaps23,openface49 --t_path data/EDAIC_transcripts_fixed --edaic data/E-DAIC/data --out data/cache_unseg
```

### Experiments of the paper

```bash
PYTHON=python GPUS=0,1 PER_GPU=6 scripts/run_final.sh   # final model -> runs/grm_full
scripts/run_ablation.sh                                 # leave-one-out ablation (no segmentation, no cross-modal attention, eight
                                                        #   item heads, single-item models) -> runs/grm_{full,noseg,nocross,nogrm,nomtl}
scripts/run_grid_imboll.sh                              # W-ImbOLL grid search on the development set (text-only model)
python src/collect.py grm_full grm_nocross              # result tables of any conditions -> runs/{summary,per_seed,ensembles}.csv
```

`PYTHON`, `GPUS` and `PER_GPU` choose the interpreter, the GPUs and the number of concurrent runs per GPU. Finished runs
(those with a `metrics.json`) are skipped, so an interrupted script can be restarted.

### Single runs and inference

```bash
python src/mtl.py --mods T   --seed 42 --out runs/grm_full/T_seed42 --train
python src/mtl.py --mods TAV --seed 42 --out runs/grm_full/TAV_seed42 --train --init_dir runs/grm_full --freeze_txt   # needs T, A, V of seed 42

# item and total predictions (expected and argmax) and the latent trait
python src/predict.py --ckpt runs/grm_full/TAV_seed42 --split test --out predictions.csv
python src/predict.py --ckpt runs/grm_full/T_seed42 --pids 300,301
```

`python src/mtl.py --help` lists every switch: the hyperparameters, the ablation switches used by `scripts/run_ablation.sh`
(`--no_cross_modal`, `--head items`, `--single_item`, `--cache data/cache_unseg`), the data paths (`--cache`, `--labels`) and `--cv_fold` for k-fold cross-validation
over all participants. Each run writes `best_loss.pt`, `metrics.json` (arguments, per-item metrics on dev and test, item
parameters, attention-entropy and modality-permutation diagnostics), `preds_loss.npz` (probabilities and labels) and `log.txt`.

## Acknowledgement

The per-utterance feature windows and the unimodal trunks follow the QuestMF reference implementation
(Ubiquitous Knowledge Processing Lab, TU Darmstadt, Apache License 2.0), on which this code was originally built.
