"""Participant-speech isolation: speaker diarization of the E-DAIC session audio with pyannote-audio, and reduction of each
transcript to the participant's turns.

For every session folder <edaic>/<pid>_P/ holding <pid>_AUDIO.wav and <pid>_Transcript.csv (columns Start_Time, End_Time,
Text, Confidence):
  1. the pyannote pipeline (default pyannote/speaker-diarization-community-1) labels the speech of the whole recording with
     speaker turns; the turns are saved to <out>/<pid>_P/<pid>_diarization.csv (start, end, speaker) so the step can be inspected
     or rerun without the model;
  2. the speaker with the greatest total speaking time is taken as the participant (the interviewer, the virtual agent Ellie,
     mainly asks short questions);
  3. the transcript rows are regrouped by speaker: with --rule overlap (default) a row goes to the speaker whose turns
     overlap it the most, with --rule any a row is kept whenever any participant turn overlaps it; the participant's rows are
     written to <out>/<pid>_P/<pid>_Transcript.csv with the columns Start_Time, End_Time, Text, Confidence, Speaker
     (Speaker = "patient"), the format read by fix_transcripts.py and build_cache.py. Rows that overlap no diarized speech
     are dropped.

Usage: python diarize_transcripts.py --edaic E-DAIC/data --out Patient_data [--model pyannote/speaker-diarization-community-1]
       [--token HF_TOKEN] [--pids 300,301] [--device cuda] [--rule overlap|any]
The pyannote pipeline needs pyannote.audio (https://github.com/pyannote/pyannote-audio) and a Hugging Face token for which the
model's conditions have been accepted (pass --token or set HF_TOKEN). If <pid>_diarization.csv already exists the model is
not run for that session.
"""
import argparse, glob, os
import numpy as np, pandas as pd

def diarize(wav_path, pipeline):
    """-> DataFrame(start, end, speaker) of the pipeline's speaker turns."""
    out = pipeline(wav_path)
    ann = getattr(out, 'speaker_diarization', out)          # pyannote.audio >= 4 returns a DiarizeOutput, 3.x an Annotation
    rows = [(seg.start, seg.end, spk) for seg, _, spk in ann.itertracks(yield_label=True)]
    return pd.DataFrame(rows, columns=['start', 'end', 'speaker'])

def participant_speaker(turns):
    """speaker with the greatest total speaking time"""
    dur = (turns.end - turns.start).groupby(turns.speaker).sum(); return dur.idxmax()

def assign_rows(transcript, turns):
    """Overlap (seconds) of every transcript row with every speaker -> the speaker with the largest overlap per row,
    NaN where the row overlaps no turn. Returns a Series aligned with the transcript."""
    st, en = transcript.Start_Time.values.astype(float), transcript.End_Time.values.astype(float)
    speakers = sorted(turns.speaker.unique()); ov = np.zeros((len(transcript), len(speakers)))
    for j, spk in enumerate(speakers):
        t = turns[turns.speaker == spk]
        for a, b in zip(t.start.values, t.end.values):
            ov[:, j] += np.clip(np.minimum(en, b) - np.maximum(st, a), 0, None)
    best = ov.argmax(1); best = np.where(ov.max(1) > 0, best, -1)
    return pd.Series([speakers[k] if k >= 0 else np.nan for k in best], index=transcript.index)

def participant_transcript(transcript, turns, rule='overlap'):
    spk = participant_speaker(turns)
    if rule == 'any':
        t = turns[turns.speaker == spk]; st, en = transcript.Start_Time.values.astype(float), transcript.End_Time.values.astype(float)
        mask = np.zeros(len(transcript), bool)
        for a, b in zip(t.start.values, t.end.values): mask |= (np.minimum(en, b) - np.maximum(st, a)) > 0
        keep = transcript[mask].copy()
    else: keep = transcript[assign_rows(transcript, turns) == spk].copy()
    keep['Speaker'] = 'patient'
    return keep[['Start_Time', 'End_Time', 'Text', 'Confidence', 'Speaker']], spk

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--edaic', default='E-DAIC/data', help='folder holding <pid>_P/<pid>_AUDIO.wav and <pid>_P/<pid>_Transcript.csv')
    ap.add_argument('--out', default='Patient_data'); ap.add_argument('--model', default='pyannote/speaker-diarization-community-1')
    ap.add_argument('--token', default=os.environ.get('HF_TOKEN', '')); ap.add_argument('--pids', default='', help='comma-separated subset')
    ap.add_argument('--device', default='cuda'); ap.add_argument('--rule', default='overlap', choices=['overlap', 'any'], help='row assignment: largest overlap, or any overlap with the participant')
    args = ap.parse_args()
    pids = [p for p in args.pids.split(',') if p] or sorted(os.path.basename(d)[:-2] for d in glob.glob(os.path.join(args.edaic, '*_P')))
    pipeline = None
    for k, pid in enumerate(pids):
        src, dst = os.path.join(args.edaic, f'{pid}_P'), os.path.join(args.out, f'{pid}_P'); os.makedirs(dst, exist_ok=True)
        turns_path = os.path.join(dst, f'{pid}_diarization.csv')
        if os.path.exists(turns_path): turns = pd.read_csv(turns_path)
        else:
            if pipeline is None:
                import torch
                from pyannote.audio import Pipeline
                try: pipeline = Pipeline.from_pretrained(args.model, token=args.token or None)
                except TypeError: pipeline = Pipeline.from_pretrained(args.model, use_auth_token=args.token or None)   # pyannote.audio 3.x
                if args.device == 'cuda' and torch.cuda.is_available(): pipeline.to(torch.device('cuda'))
            turns = diarize(os.path.join(src, f'{pid}_AUDIO.wav'), pipeline); turns.to_csv(turns_path, index=False)
        tr = pd.read_csv(os.path.join(src, f'{pid}_Transcript.csv'))
        kept, spk = participant_transcript(tr, turns, args.rule); kept.to_csv(os.path.join(dst, f'{pid}_Transcript.csv'), index=False)
        print(f'{pid}: {len(turns.speaker.unique())} speakers, participant = {spk}, {len(kept)}/{len(tr)} transcript rows kept', flush=True)
    print('DIARIZATION_DONE', len(pids), 'sessions')
