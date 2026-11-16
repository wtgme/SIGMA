# data/

Everything in this folder except this file is ignored by git. E-DAIC is distributed by USC ICT under a licence agreement
and is not included; place (or symlink) it here. All scripts default to the paths below.

```
data/
├── E-DAIC/                                  # you provide: the E-DAIC release as distributed (ln -s /path/to/E-DAIC data/E-DAIC)
│   ├── data/<pid>_P/
│   │   ├── <pid>_AUDIO.wav
│   │   ├── <pid>_Transcript.csv             # Start_Time, End_Time, Text, Confidence
│   │   └── features/
│   │       ├── <pid>_OpenSMILE2.3.0_egemaps.csv
│   │       └── <pid>_OpenFace2.1.0_Pose_gaze_AUs.csv
│   └── labels/
│       ├── detailed_lables.csv              # Participant, PHQ8_1_NoInterest, ..., PHQ8_8_Psychomotor (all 275 participants)
│       ├── train_split.csv                  # Participant_ID (official AVEC 2019 partition)
│       ├── dev_split.csv
│       └── test_split.csv
│
│   # produced by scripts/preprocess.sh
├── Patient_data/<pid>_P/                    # participant-only transcripts + <pid>_diarization.csv (speaker turns)
├── Patient_data_fixed/<pid>_P/              # the same with repaired transcript windows
├── EDAIC_transcripts_fixed/<pid>_P/         # original transcripts, repaired (no-segmentation ablation)
├── cache/<feature>/<pid>.npy                # per-utterance features: bge, egemaps23, openface49; n_sentences.json
└── cache_unseg/<feature>/<pid>.npy          # the same from the original transcripts
```

Notes

- The labels are used exactly as distributed with E-DAIC. `detailed_lables.csv` (sic) holds the eight PHQ-8 item scores
  of every participant. `Detailed_PHQ8_Labels.csv`, also in `labels/`, is not used: it has participants 485 and 486 swapped
  (their item scores disagree with `PHQ_Score` in the split files).
- If your data lives elsewhere, symlink it (`ln -s /path/to/E-DAIC data/E-DAIC`) or pass the paths explicitly
  (`--edaic`, `--t_path`, `--out`, `--cache`, `--labels`; see each script's `--help`).
