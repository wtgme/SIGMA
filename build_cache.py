"""Per-sentence feature cache for the final model. One .npy per participant and feature set,
shape (n_sentences, dim), sentences from the repaired diarized transcripts (Patient_data_fixed).

Text   : distilroberta (as in the original code), mpnet, bge, e5   -> sentence embeddings, L2-normalised
Audio  : egemaps23 (as in the original code), mfcc78 (mean+std of 39 MFCC/deltas), wavlm768 (WavLM-base-plus, mean of last layer)
Video  : openface49 (as in the original code), resnet2048, densenet1920 (E-DAIC CNN features, mean per sentence)
Windows follow the original code: frame index = int(time x fps), fps = n_frames / audio duration for video,
100 Hz for openSMILE; invalid windows give a zero vector.
Usage: python build_cache.py --feats bge,egemaps23,openface49 [--t_path Patient_data_fixed/] [--edaic E-DAIC/data] [--out cache] [--limit N]
"""
import argparse, glob, json, os, time
import numpy as np, pandas as pd, torch, torch.nn.functional as F, soundfile as sf

ap = argparse.ArgumentParser()
ap.add_argument('--feats', required=True)
ap.add_argument('--t_path', default='Patient_data_fixed/')
ap.add_argument('--out', default='cache'); ap.add_argument('--edaic', default='E-DAIC/data', help='E-DAIC data folder holding <pid>_P/<pid>_AUDIO.wav and <pid>_P/features/')
ap.add_argument('--limit', type=int, default=0)
args = ap.parse_args()
FEATS = args.feats.split(',')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TEXT_MODELS = {'distilroberta': ('sentence-transformers/all-distilroberta-v1', 'mean', ''),
               'mpnet': ('sentence-transformers/all-mpnet-base-v2', 'mean', ''),
               'bge': ('BAAI/bge-large-en-v1.5', 'cls', ''),
               'e5': ('intfloat/e5-large-v2', 'mean', 'query: ')}
OPENFACE_COLS = ['pose_Tx', 'pose_Ty', 'pose_Tz', 'pose_Rx', 'pose_Ry', 'pose_Rz', 'gaze_0_x', 'gaze_0_y', 'gaze_0_z', 'gaze_1_x', 'gaze_1_y', 'gaze_1_z',
                 'gaze_angle_x', 'gaze_angle_y', 'AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r', 'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r',
                 'AU17_r', 'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r', 'AU45_r', 'AU01_c', 'AU02_c', 'AU04_c', 'AU05_c', 'AU06_c', 'AU07_c', 'AU09_c', 'AU10_c',
                 'AU12_c', 'AU14_c', 'AU15_c', 'AU17_c', 'AU20_c', 'AU23_c', 'AU25_c', 'AU26_c', 'AU28_c', 'AU45_c']

def windows_mean(frames, starts, ends, fps, max_frames=None, std=False):
    n, d = frames.shape; max_frames = n if max_frames is None else min(max_frames, n)
    out = np.zeros((len(starts), d * (2 if std else 1)), dtype=np.float32)
    for i, (s, e) in enumerate(zip(starts, ends)):
        a, b = int(s * fps), int(e * fps)
        if a < 0 or b > max_frames or a >= b: continue
        seg = frames[a:min(b, max_frames)]
        if len(seg) == 0: continue
        out[i, :d] = seg.mean(0)
        if std: out[i, d:] = seg.std(0)
    return out

def text_encoder(name):
    from transformers import AutoTokenizer, AutoModel
    hf, pool, prefix = TEXT_MODELS[name]
    tok, mdl = AutoTokenizer.from_pretrained(hf), AutoModel.from_pretrained(hf).to(device).eval()
    def enc(sents):
        outs = []
        for i in range(0, len(sents), 64):
            batch = tok([prefix + s for s in sents[i:i + 64]], padding=True, truncation=True, max_length=256, return_tensors='pt').to(device)
            with torch.no_grad(): h = mdl(**batch).last_hidden_state
            if pool == 'cls': v = h[:, 0]
            else:
                m = batch['attention_mask'].unsqueeze(-1).float(); v = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
            outs.append(F.normalize(v, p=2, dim=1).float().cpu())
        return torch.cat(outs).numpy()
    return enc

def wavlm_encoder():
    from transformers import AutoFeatureExtractor, WavLMModel
    fe = AutoFeatureExtractor.from_pretrained('microsoft/wavlm-base-plus'); mdl = WavLMModel.from_pretrained('microsoft/wavlm-base-plus').to(device).eval().half()
    def enc(wav, sr, starts, ends):
        out = np.zeros((len(starts), 768), dtype=np.float32)
        for i, (s, e) in enumerate(zip(starts, ends)):
            a, b = int(s * sr), int(min(e, len(wav) / sr) * sr)
            if b - a < int(0.3 * sr): continue
            seg = wav[a:b]; chunks = [seg[j:j + 20 * sr] for j in range(0, len(seg), 20 * sr)]; hs = []
            for c in chunks:
                if len(c) < int(0.3 * sr): continue
                x = fe(c, sampling_rate=sr, return_tensors='pt').input_values.to(device).half()
                with torch.no_grad(): hs.append(mdl(x).last_hidden_state.float().mean(1)[0].cpu().numpy())
            if hs: out[i] = np.mean(hs, 0)
        return out
    return enc

pids = sorted(os.path.basename(p)[:-2] for p in glob.glob(os.path.join(args.t_path, '*_P')))
if args.limit: pids = pids[:args.limit]
for f in FEATS: os.makedirs(os.path.join(args.out, f), exist_ok=True)
enc_txt = {f: text_encoder(f) for f in FEATS if f in TEXT_MODELS}
enc_wav = wavlm_encoder() if 'wavlm768' in FEATS else None
meta = {}
t0 = time.time()
for k, pid in enumerate(pids):
    tr = pd.read_csv(os.path.join(args.t_path, f'{pid}_P', f'{pid}_Transcript.csv'))
    sents = [str(t) for t in tr.Text.tolist()]; st, en = tr.Start_Time.values.astype(float), tr.End_Time.values.astype(float)
    dur = sf.info(f'{args.edaic}/{pid}_P/{pid}_AUDIO.wav').duration
    meta[pid] = len(sents)
    for f in FEATS:
        path = os.path.join(args.out, f, f'{pid}.npy')
        if os.path.exists(path): continue
        if f in TEXT_MODELS: arr = enc_txt[f](sents)
        elif f == 'egemaps23':
            df = pd.read_csv(f'{args.edaic}/{pid}_P/features/{pid}_OpenSMILE2.3.0_egemaps.csv', sep=';')
            arr = windows_mean(df.iloc[:, 2:].values.astype(np.float32), st, en, 100, max_frames=int(dur * 100))
        elif f == 'mfcc78':
            df = pd.read_csv(f'{args.edaic}/{pid}_P/features/{pid}_OpenSMILE2.3.0_mfcc.csv', sep=';')
            arr = windows_mean(df.iloc[:, 2:].values.astype(np.float32), st, en, 100, max_frames=int(dur * 100), std=True)
        elif f == 'wavlm768':
            wav, sr = sf.read(f'{args.edaic}/{pid}_P/{pid}_AUDIO.wav', dtype='float32')
            if wav.ndim > 1: wav = wav.mean(1)
            if sr != 16000:
                import librosa; wav = librosa.resample(wav, orig_sr=sr, target_sr=16000); sr = 16000
            arr = enc_wav(wav, sr, st, en)
        elif f == 'openface49':
            df = pd.read_csv(f'{args.edaic}/{pid}_P/features/{pid}_OpenFace2.1.0_Pose_gaze_AUs.csv'); df.columns = [c.strip() for c in df.columns]
            cols = [c for c in OPENFACE_COLS if c in df.columns]; fr = df[cols].values.astype(np.float32)
            arr = windows_mean(fr, st, en, len(fr) / dur if dur > 0 else 30)
        elif f == 'resnet2048':
            import scipy.io as sio
            fr = sio.loadmat(f'{args.edaic}/{pid}_P/features/{pid}_CNN_ResNet.mat')['feature'].astype(np.float32); arr = windows_mean(fr, st, en, len(fr) / dur)
        elif f == 'densenet1920':
            fr = pd.read_csv(f'{args.edaic}/{pid}_P/features/{pid}_densenet201.csv').iloc[:, 2:].values.astype(np.float32); arr = windows_mean(fr, st, en, len(fr) / dur)
        else: raise ValueError(f)
        np.save(path, arr.astype(np.float32))
    if k % 25 == 0: print(f'{k + 1}/{len(pids)} participants, {(time.time() - t0) / 60:.1f} min', flush=True)
json.dump(meta, open(os.path.join(args.out, 'n_sentences.json'), 'w'))
print('CACHE_DONE', FEATS, f'{(time.time() - t0) / 60:.1f} min', flush=True)
