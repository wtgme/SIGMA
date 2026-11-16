"""Inference with a trained CTMTL v3 checkpoint.

python predict.py --ckpt checkpoints/TAV_seed42 --pids 300,301,302 [--cache cache] [--out predictions.csv]
python predict.py --ckpt checkpoints/T_seed42 --split test                 # all participants of a label split

Reads the run's metrics.json for the model configuration, loads best_loss.pt, and writes one row per participant:
expected score per PHQ-8 item, the expected total (headline decoding), the argmax total, and the latent trait theta.
Features must exist in the cache for every participant (see build_cache.py).
"""
import argparse, json, os, sys, runpy
import numpy as np, pandas as pd, torch, torch.nn.functional as F
ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', required=True, help='run folder holding metrics.json and best_loss.pt')
ap.add_argument('--pids', default='', help='comma-separated participant ids'); ap.add_argument('--split', default='', choices=['', 'train', 'dev', 'test'])
ap.add_argument('--cache', default=''); ap.add_argument('--out', default='predictions.csv')
a = ap.parse_args()
HERE = os.path.dirname(os.path.abspath(__file__)); cfg = json.load(open(os.path.join(a.ckpt, 'metrics.json')))['args']
# rebuild the model definition with the run's own switches (mtl.py is a script; execute it up to its training section)
flags = ['--mods', cfg['mods'], '--out', a.ckpt, '--cache', a.cache or cfg['cache'], '--labels', cfg['labels']]
for k in ['txt_feat', 'aud_feat', 'vid_feat', 'enc_hidden', 'dropout', 'max_len', 'head_rep', 'norm_vid', 'head', 'grm_factors', 'single_item']:
    if k in cfg: flags += [f'--{k}', str(cfg[k])]
if cfg.get('no_cross_modal'): flags.append('--no_cross_modal')
src = open(os.path.join(HERE, 'mtl.py')).read().split('# ---------------------------------------------------------------- init')[0]
sys.argv = ['mtl.py'] + flags; g = {'__file__': os.path.join(HERE, 'mtl.py')}; exec(compile(src, 'mtl.py', 'exec'), g)
model = g['Model']().to(g['device']); model.load_state_dict(torch.load(os.path.join(a.ckpt, 'best_loss.pt'), map_location=g['device'])); model.eval()
pids = [int(p) for p in a.pids.split(',') if p] if a.pids else g['splits'][a.split or 'test']
ds = g['Feats'](pids); ITEMS = g['ITEMS']; rows = []
with torch.no_grad():
    for i, pid in enumerate(pids):
        batch = [t.unsqueeze(0) for t in ds.items[i]]; seqs, masks, y = g['unpack'](batch)
        P = torch.stack([F.softmax(l, 1) for l in model(seqs, masks)], 1)[0].cpu().numpy()      # (8, 4)
        ev = (P * np.arange(4)).sum(-1); am = P.argmax(-1)
        row = {'participant': pid, 'phq8_total_expected': round(float(ev.sum()), 3), 'phq8_total_argmax': int(am.sum())}
        for k, it in enumerate(ITEMS): row[f'{it}_expected'] = round(float(ev[k]), 3); row[f'{it}_argmax'] = int(am[k])
        if hasattr(model, '_theta'): row.update({f'theta_{j + 1}': round(float(v), 3) for j, v in enumerate(model._theta[0].cpu().numpy())})
        if pid in g['lab'].index: row['phq8_total_label'] = int(g['lab'].loc[pid, ITEMS].sum())
        rows.append(row)
df = pd.DataFrame(rows); df.to_csv(a.out, index=False); print(df.head(10).to_string()); print(f'wrote {a.out} ({len(df)} participants)')
