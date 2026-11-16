"""Repair the transcript windows. In 83 of the 275 E-DAIC transcripts the LAST row carries the
session's first start time instead of its own, so its window spans the whole interview; the
diarized Patient_data transcripts inherit this in the 73 sessions where that last utterance is the
participant's. The true start is not recorded anywhere, so it is estimated from the utterance
length: start = max(previous end, end - max(1 s, 0.45 s x words)).
Writes <out>/<id>_P/<id>_Transcript.csv; nothing under <src> is modified.
Usage: python preprocessing/fix_transcripts.py [--src data/Patient_data] [--out data/Patient_data_fixed]   (also used on the original transcripts for the no-segmentation ablation)"""
import argparse, glob, os
import numpy as np, pandas as pd
DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')   # <repo>/data, the default for every path

ap = argparse.ArgumentParser(); ap.add_argument('--src', default=os.path.join(DATA, 'Patient_data')); ap.add_argument('--out', default=os.path.join(DATA, 'Patient_data_fixed')); args = ap.parse_args()
n_fix = n_rows = 0
for p in sorted(glob.glob(f'{args.src}/*_P')):
    pid = os.path.basename(p)[:-2]
    tr = pd.read_csv(f'{p}/{pid}_Transcript.csv'); st, en = tr.Start_Time.values.copy(), tr.End_Time.values
    os.makedirs(f'{args.out}/{pid}_P', exist_ok=True)
    for i in np.where((en - st) > 60)[0]:
        words = len(str(tr.Text[i]).split()); est = en[i] - max(1.0, 0.45 * words)
        st[i] = max(en[i - 1] if i > 0 else 0.0, est); n_fix += 1
    tr['Start_Time'] = np.round(st, 2)
    tr.to_csv(f'{args.out}/{pid}_P/{pid}_Transcript.csv', index=False); n_rows += len(tr)
print(f'fixed {n_fix} rows across {n_rows} sentences in', len(glob.glob(f'{args.out}/*_P')), 'participants')
