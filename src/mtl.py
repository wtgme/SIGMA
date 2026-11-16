"""SIGMA: model, training and evaluation. Relative --cache / --labels paths are resolved against the repository root:
data/cache/<feature>/<pid>.npy, data/E-DAIC/labels/. Run from the repository root.
One script for every modality subset: unimodal runs give the pretrained trunks, multimodal runs load them.

python src/mtl.py --mods TAV --seed 42 --out runs/<cond>/TAV_seed42 --train [--init_dir runs/<cond>] [--freeze_txt] [flags]

Model: participant-only sentence features, padded to the longest interview and packed (--max_len); per-modality BiLSTM +
self-attention trunks with masked mean pooling; pairwise cross-attention fusion over all ordered modality pairs for the multimodal runs; a graded-response head
(--head grm, --grm_factors, --grm_lr_mult): the fused feature maps to a latent severity trait, each PHQ-8 item has a learned
discrimination and three ordered thresholds, and the total score is the sum of the expected item scores. Loss: W-ImbOLL per
item with class weights and learned task weights; checkpoint: lowest validation loss. metrics.json holds per-item metrics for
dev and test, the selected epoch, the item parameters and attention / modality-shuffle diagnostics.

Ablation switches (scripts/run_ablation.sh): --no_cross_modal (keys and values from the query modality: same depth and parameters, no
cross-modal information flow), --head items (eight independent item heads instead of the graded-response head), --single_item i
(one item per model), --cache data/cache_unseg (original transcripts). --cv_fold k runs k-fold cross-validation over all participants.
"""
import argparse, json, os, random, time
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # the repository root: data/cache/, data/E-DAIC/labels/, runs/
ap = argparse.ArgumentParser()
ap.add_argument('--mods', required=True, help='T, A, V, TA, TV, AV or TAV')
ap.add_argument('--txt_feat', default='bge'); ap.add_argument('--aud_feat', default='egemaps23'); ap.add_argument('--vid_feat', default='openface49')
ap.add_argument('--cache', default='data/cache'); ap.add_argument('--labels', default='data/E-DAIC/labels', help='folder with detailed_lables.csv and {train,dev,test}_split.csv (the E-DAIC labels/ folder)')
ap.add_argument('--seed', type=int, default=42); ap.add_argument('--out', required=True); ap.add_argument('--train', action='store_true')
ap.add_argument('--alpha', type=float, default=1.0, help='ordinal-distance exponent of W-ImbOLL'); ap.add_argument('--beta', type=float, default=0.5, help='class-weight exponent of W-ImbOLL')
ap.add_argument('--enc_hidden', type=int, default=50, help='BiLSTM units per direction'); ap.add_argument('--dropout', type=float, default=0.5, help='attention dropout')
ap.add_argument('--epochs', type=int, default=50); ap.add_argument('--lr', type=float, default=5e-4); ap.add_argument('--wd', type=float, default=1e-3)
ap.add_argument('--batch', type=int, default=None, help='default 8 for multimodal, 10 for unimodal runs')
ap.add_argument('--init_dir', default='', help='folder with unimodal runs (<mod>_seed<s>/ or, for --single_item i, <mod>_item<i>_seed<s>/) to initialise the encoders')
ap.add_argument('--freeze_txt', action='store_true', help='freeze the text trunk after initialisation')
ap.add_argument('--no_cross_modal', action='store_true', help='ablation: keys/values from the query modality')
ap.add_argument('--head', default='grm', choices=['grm', 'items'], help='grm = graded-response head (default); items = eight independent item heads (ablation)')
ap.add_argument('--grm_factors', type=int, default=2, help='latent severity factors'); ap.add_argument('--grm_lr_mult', type=float, default=50.0, help='learning-rate multiplier for the item parameters (no weight decay on them)')
ap.add_argument('--single_item', type=int, default=0, help='1-8: single-task model for one PHQ-8 item (ablation); 0 = all items')
ap.add_argument('--max_len', type=int, default=203, help='padded sequence length; must cover the longest interview (a warning is printed otherwise)')
ap.add_argument('--head_rep', type=int, default=120, help='unimodal head: the pooled feature is repeated this many times before the flatten layer')
ap.add_argument('--norm_vid', type=int, default=1, help='L2-normalise each sentence video vector')
ap.add_argument('--cv_fold', type=int, default=-1, help='>=0: k-fold cross-validation over all participants; this fold = test, next fold = dev')
ap.add_argument('--cv_nfold', type=int, default=5); ap.add_argument('--cv_seed', type=int, default=0)
ap.add_argument('--rules', default='loss', help='checkpoint rules to save and evaluate: loss, itemccc, devrmse, devccc')
args = ap.parse_args()
MODS = [{'T': 'txt', 'A': 'aud', 'V': 'vid'}[c] for c in args.mods]; UNI = len(MODS) == 1
if args.batch is None: args.batch = 10 if UNI else 8
for a in ('cache', 'labels'):
    if not os.path.isabs(getattr(args, a)): setattr(args, a, os.path.join(ROOT, getattr(args, a)))
FEAT = {'txt': args.txt_feat, 'aud': args.aud_feat, 'vid': args.vid_feat}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPS = 1e-12; T = args.max_len; REP = args.head_rep; RULES = args.rules.split(','); FOCUS = args.single_item - 1
ITEMS = ['PHQ8_1_NoInterest', 'PHQ8_2_Depressed', 'PHQ8_3_Sleep', 'PHQ8_4_Tired', 'PHQ8_5_Appetite', 'PHQ8_6_Failure', 'PHQ8_7_Concentration', 'PHQ8_8_Psychomotor']

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
set_seed(args.seed); os.makedirs(args.out, exist_ok=True)

# ---------------------------------------------------------------- data
lab = pd.read_csv(os.path.join(args.labels, 'detailed_lables.csv')).set_index('Participant')
splits = {s: pd.read_csv(os.path.join(args.labels, f'{s}_split.csv')).Participant_ID.astype(int).tolist() for s in ['train', 'dev', 'test']}
if args.cv_fold >= 0:
    allp = sorted(set(splits['train']) | set(splits['dev']) | set(splits['test'])); perm = np.random.RandomState(args.cv_seed).permutation(len(allp))
    folds = [[allp[i] for i in perm[k::args.cv_nfold]] for k in range(args.cv_nfold)]
    k = args.cv_fold; splits = {'test': folds[k], 'dev': folds[(k + 1) % args.cv_nfold], 'train': [p for j, f in enumerate(folds) if j not in (k, (k + 1) % args.cv_nfold) for p in f]}
    json.dump(splits, open(os.path.join(args.out, 'cv_splits.json'), 'w'))
TRUNC = []
class Feats(Dataset):
    def __init__(self, pids):
        self.items = []
        for pid in pids:
            entry = []
            for m in MODS:
                x = np.load(os.path.join(args.cache, FEAT[m], f'{pid}.npy'))
                if len(x) > T: TRUNC.append(pid); x = x[:T]
                if m == 'vid' and args.norm_vid: x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
                n = len(x); pad = np.zeros((T - n, x.shape[1]), dtype=np.float32); mask = np.ones(T, dtype=bool); mask[:n] = False
                entry += [torch.from_numpy(np.concatenate([x, pad]).astype(np.float32)), torch.from_numpy(mask)]
            entry.append(torch.tensor(lab.loc[pid, ITEMS].values.astype(np.float32))); self.items.append(entry)
    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]
DIMS = {m: int(np.load(os.path.join(args.cache, FEAT[m], f'{splits["train"][0]}.npy')).shape[1]) for m in MODS}
data = {s: Feats(splits[s]) for s in ['train', 'dev', 'test']}
print(f'# mods {MODS} feats {[FEAT[m] for m in MODS]} dims {DIMS} | train/dev/test {[len(data[s]) for s in data]} | max_len {T}' + (f' | WARNING {len(set(TRUNC))} participants truncated' if TRUNC else ''), flush=True)

def class_weights(item):   # (n_total / n_class) ** beta over the training split
    y = lab.loc[splits['train'], item].values.astype(int); n = len(y)
    return torch.tensor([(n / max(1, (y == c).sum())) for c in range(4)], dtype=torch.float32) ** args.beta
CW = [class_weights(it).to(device) for it in ITEMS]

def imboll(logits, w, labels):   # Weighted Imbalanced Ordinal Log-Loss: w(y)^beta * |y - c|^alpha (w already carries beta; see class_weights)
    p = F.softmax(logits, dim=1); y = labels.long()
    dist = w[y].unsqueeze(1) * (y.unsqueeze(1) - torch.arange(4, device=logits.device)).abs().float() ** args.alpha   # until 2026-09-10: (w |y-c|)^alpha, identical at alpha = 1
    return (-torch.log(1 - p + EPS) * dist).sum(1).mean()

# ---------------------------------------------------------------- model
H2 = 2 * args.enc_hidden
def run_lstm(lstm, x, mask):   # packed: the BiLSTM never reads the padding
    lengths = (~mask).sum(1).cpu(); packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
    return nn.utils.rnn.pad_packed_sequence(lstm(packed)[0], batch_first=True, total_length=x.size(1))[0]
def entropy_ratio(aw, mask):   # mean attention entropy / log(n real keys): 1.0 = uniform over the participant's sentences
    n = (~mask).sum(1).float().clamp(min=2); ent = -(aw.clamp_min(1e-12) * aw.clamp_min(1e-12).log()).sum(-1).mean(1); return (ent / n.log()).mean().item()

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        for m in MODS:
            setattr(self, f'{m}_lstm', nn.LSTM(DIMS[m], args.enc_hidden, batch_first=True, bidirectional=True))
            setattr(self, f'{m}_attention', nn.MultiheadAttention(H2, 4, batch_first=True, dropout=args.dropout))
        if len(MODS) > 1:
            for q in MODS:
                for k in MODS:
                    if q != k: setattr(self, f'cross_{q}_{k}', nn.MultiheadAttention(H2, 4, batch_first=True, dropout=args.dropout))
        self.streams = self._streams(); self.fusion_proj = nn.Identity() if UNI else nn.Linear(len(self.streams) * H2, 200)
        self.collect = False; self.stats = {}
    def _streams(self):   # (query, key/value) pairs whose pooled outputs are concatenated
        if len(MODS) == 1: return [(MODS[0], None)]
        if len(MODS) == 2: a, b = MODS; return [(a, b), (b, a)]
        return [(q, k) for q in MODS for k in MODS if q != k]   # all six ordered pairs: every modality serves as keys/values for the other two
    def _att(self, name, q, kv, mask):
        out, aw = getattr(self, name)(q, kv, kv, key_padding_mask=mask)
        if self.collect: self.stats.setdefault(name, []).append(entropy_ratio(aw, mask))
        return out
    def _pool(self, x, mask):   # masked mean over the real sentences
        keep = (~mask).float().unsqueeze(-1); return (x * keep).sum(1) / keep.sum(1).clamp(min=1.0)
    def forward(self, seqs, masks):
        enc = {}
        for m in MODS:
            h = run_lstm(getattr(self, f'{m}_lstm'), seqs[m], masks[m]); enc[m] = self._att(f'{m}_attention', h, h, masks[m])
        outs = []
        for q, k in self.streams:
            if k is None: outs.append(enc[q]); continue
            src, smask = (enc[q], masks[q]) if args.no_cross_modal else (enc[k], masks[k])
            outs.append(self._att(f'cross_{q}_{k}', enc[q], src, smask))
        return self.fusion_proj(torch.cat([self._pool(o, masks[q]) for o, (q, k) in zip(outs, self.streams)], -1))

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder(); d = H2 if UNI else 200
        self.task_weights = nn.Parameter(torch.tensor([1.0, 1.2, 1.0, 1.0, 1.0, 1.2, 1.0, 1.0])); self.stats = {}
        n_heads, n_out = (1, args.grm_factors) if args.head == 'grm' else (8, 4)
        if UNI:   # unimodal head: the pooled feature is repeated REP times and flattened into a 256-wide layer; class weights are learnable (softplus)
            p = 0.2 if MODS[0] == 'vid' else 0.5
            self.heads = nn.ModuleList([nn.Sequential(nn.Flatten(), nn.Dropout(p), nn.Linear(REP * d, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, n_out)) for _ in range(n_heads)])
            self.class_weights = nn.Parameter(torch.stack([class_weights(it) for it in ITEMS]))
        else:
            self.heads = nn.ModuleList([nn.Sequential(nn.Linear(200, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, n_out)) for _ in range(n_heads)])
        if args.head == 'grm':   # graded-response model: per item a discrimination vector and three ordered thresholds
            K = args.grm_factors; self.grm_a = nn.Parameter(torch.full((8, K), 0.5413))   # softplus(0.5413) = 1
            ytr = torch.tensor(lab.loc[splits['train'], ITEMS].values, dtype=torch.float32)
            Pge = torch.stack([(ytr >= k).float().mean(0) for k in (1, 2, 3)], 1).clamp(0.02, 0.98)   # thresholds from the training marginals
            b = -torch.log(Pge / (1 - Pge)); inv_sp = lambda x: torch.log(torch.expm1(x.clamp_min(1e-3)))
            self.grm_b0 = nn.Parameter(b[:, 0].clone()); self.grm_db = nn.Parameter(inv_sp(b[:, 1:] - b[:, :-1]))
    def _head(self, i, x): return self.heads[i](x.unsqueeze(1).expand(-1, REP, -1) if UNI else x)
    def grm_params(self):
        a = F.softplus(self.grm_a) if args.grm_factors == 1 else torch.cat([F.softplus(self.grm_a[:, :1]), self.grm_a[:, 1:]], 1)
        b = torch.cumsum(torch.cat([self.grm_b0.unsqueeze(1), F.softplus(self.grm_db)], 1), 1); return a, b
    def _grm(self, f):   # -> eight log-probability vectors (B, 4); P(y_i >= k) = sigmoid(a_i . theta - b_ik)
        theta = self._head(0, f); a, b = self.grm_params(); z = theta @ a.T
        Pge = torch.sigmoid(z.unsqueeze(-1) - b.unsqueeze(0))
        p = torch.cat([1 - Pge[..., :1], Pge[..., :-1] - Pge[..., 1:], Pge[..., -1:]], -1).clamp_min(1e-6)
        self._theta = theta.detach(); return [torch.log(p[:, i]) for i in range(8)]
    def forward(self, seqs, masks):
        f = self.encoder(seqs, masks)
        if args.head == 'grm': return self._grm(f)
        if args.single_item:
            out = self._head(FOCUS, f); z = torch.zeros_like(out); return [out if j == FOCUS else z for j in range(8)]
        return [self._head(i, f) for i in range(8)]

def unpack(batch):
    seqs = {m: batch[2 * i].to(device) for i, m in enumerate(MODS)}; masks = {m: batch[2 * i + 1].to(device) for i, m in enumerate(MODS)}
    return seqs, masks, batch[-1].to(device)
def task_w(model): w = torch.abs(model.task_weights); return w / (w.sum() + EPS)
def total_loss(model, logits, y):
    if FOCUS >= 0: return imboll(logits[FOCUS], CW[FOCUS], y[:, FOCUS])
    w = task_w(model); return sum(w[i] * imboll(logits[i], F.softplus(model.class_weights[i]) if UNI else CW[i], y[:, i]) for i in range(8))

def ccc(p, l):
    p, l = np.asarray(p, float), np.asarray(l, float); return 2 * np.cov(p, l, bias=True)[0, 1] / (p.var() + l.var() + (p.mean() - l.mean()) ** 2)
def metrics(p, l): return dict(CCC=float(ccc(p, l)) if np.std(p) > 0 else 0.0, RMSE=float(np.sqrt(np.mean((p - l) ** 2))), MAE=float(np.mean(np.abs(p - l))))
@torch.no_grad()
def predict(model, ds):
    model.eval(); P, Y, L = [], [], 0.0
    for batch in DataLoader(ds, batch_size=args.batch):
        seqs, masks, y = unpack(batch); logits = model(seqs, masks); L += total_loss(model, logits, y).item() * y.size(0)
        P.append(torch.stack([F.softmax(l, 1) for l in logits], 1).cpu()); Y.append(y.cpu())
    P, Y = torch.cat(P).numpy(), torch.cat(Y).numpy(); return P, Y, L / len(ds)
def evaluate(P, Y):
    """Total-score and per-item metrics for argmax and expected-value decoding. Single-item runs report their item only."""
    out = {}
    for dec, item in [('argmax', P.argmax(-1).astype(float)), ('ev', (P * np.arange(4)).sum(-1))]:
        per_item = [metrics(item[:, i], Y[:, i]) for i in range(8)]
        if FOCUS >= 0: out[dec] = {'item': per_item[FOCUS]}
        else: out[dec] = dict(metrics(item.sum(1), Y.sum(1)), items=per_item, item_CCC=float(np.mean([m['CCC'] for m in per_item])))
    return out

# ---------------------------------------------------------------- init
model = Model().to(device)
if args.init_dir:
    for m in MODS:
        M_ = {'txt': 'T', 'aud': 'A', 'vid': 'V'}[m]; p = os.path.join(args.init_dir, f'{M_}_item{FOCUS + 1}_seed{args.seed}' if FOCUS >= 0 else f'{M_}_seed{args.seed}', 'best_loss.pt')
        if FOCUS >= 0 and not os.path.exists(p): p = os.path.join(args.init_dir, f'{M_}_seed{args.seed}', 'best_loss.pt')
        sd = torch.load(p, map_location='cpu')
        for part in ('lstm', 'attention'):
            getattr(model.encoder, f'{m}_{part}').load_state_dict({k.split('.', 2)[2]: v for k, v in sd.items() if k.startswith(f'encoder.{m}_{part}.')})
        print(f'# initialised {m} encoder from {p}', flush=True)
    if args.freeze_txt and 'txt' in MODS:
        for mm in (model.encoder.txt_lstm, model.encoder.txt_attention):
            for p_ in mm.parameters(): p_.requires_grad = False
        print('# text encoder frozen', flush=True)

# ---------------------------------------------------------------- train
if args.train:
    loss_side = {'task_weights', 'class_weights'}; grm = lambda n: n.startswith('grm_')
    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    groups = [{'params': [p for n, p in params if n not in loss_side and not grm(n)], 'weight_decay': args.wd},
              {'params': [p for n, p in params if grm(n)], 'weight_decay': 0.0, 'lr': args.lr * args.grm_lr_mult},
              {'params': [p for n, p in params if n in loss_side], 'weight_decay': 0.0}]
    opt = torch.optim.AdamW([g for g in groups if g['params']], lr=args.lr, eps=1e-8)
    best = {r: (1e9 if r in ('loss', 'devrmse') else -1e9) for r in RULES}; best_ep = {}; t0 = time.time()
    for ep in range(args.epochs):
        model.train(); tl = 0
        for batch in DataLoader(data['train'], batch_size=args.batch, shuffle=True):
            seqs, masks, y = unpack(batch); model.zero_grad(); loss = total_loss(model, model(seqs, masks), y); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); tl += loss.item() * y.size(0)
        P, Y, vl = predict(model, data['dev']); ev_ = evaluate(P, Y)
        if FOCUS >= 0: cur = {'loss': vl, 'itemccc': ev_['argmax']['item']['CCC'], 'devrmse': ev_['argmax']['item']['RMSE'], 'devccc': ev_['argmax']['item']['CCC']}
        else: cur = {'loss': vl, 'itemccc': ev_['argmax']['item_CCC'], 'devrmse': ev_['ev']['RMSE'], 'devccc': ev_['ev']['CCC']}
        for r in RULES:
            better = cur[r] < best[r] if r in ('loss', 'devrmse') else cur[r] > best[r]
            if better: best[r] = cur[r]; best_ep[r] = ep + 1; torch.save(model.state_dict(), os.path.join(args.out, f'best_{r}.pt'))
        extra = f' | task w {np.round(task_w(model).detach().cpu().numpy(), 3).tolist()}' if FOCUS < 0 and ((ep + 1) % 10 == 0 or ep + 1 == args.epochs) else ''
        print(f'epoch {ep + 1:3d} | train {tl / len(data["train"]):.4f} | dev loss {vl:.4f} | dev CCC {cur["devccc"]:.4f} RMSE {cur["devrmse"]:.4f} | item CCC {cur["itemccc"]:.4f} | {(time.time() - t0) / 60:.1f} min{extra}', flush=True)
    json.dump(best_ep, open(os.path.join(args.out, 'best_epoch.json'), 'w'))

# ---------------------------------------------------------------- evaluate every saved checkpoint rule
@torch.no_grad()
def diagnostics(model):
    """Attention entropy per module and modality-shuffle sensitivity on the test set."""
    model.eval(); items = data['test'].items; batch = [torch.stack([it[j] for it in items]) for j in range(len(items[0]))]; seqs, masks, y = unpack(batch)
    model.encoder.collect, model.encoder.stats = True, {}
    def total(s, m): return torch.stack([F.softmax(l, 1) for l in model(s, m)], 1).argmax(-1).sum(1).cpu().numpy()
    base = total(seqs, masks); d = {k: float(np.mean(v)) for k, v in model.encoder.stats.items()}; model.encoder.collect = False
    if len(MODS) > 1:
        perm = torch.tensor(np.random.RandomState(0).permutation(len(items)))
        for m in MODS:
            s2, m2 = dict(seqs), dict(masks); s2[m], m2[m] = seqs[m][perm], masks[m][perm]; d[f'shuffle_{m}_changed'] = float((total(s2, m2) != base).mean())
    return d
results = {'args': vars(args), 'dims': DIMS, 'truncated': sorted(set(TRUNC))}
if os.path.exists(os.path.join(args.out, 'best_epoch.json')): results['best_epoch'] = json.load(open(os.path.join(args.out, 'best_epoch.json')))
for r in RULES:
    path = os.path.join(args.out, f'best_{r}.pt')
    if not os.path.exists(path): continue
    model.load_state_dict(torch.load(path, map_location=device))
    Pd, Yd, _ = predict(model, data['dev']); Pt, Yt, _ = predict(model, data['test'])
    results[r] = {'dev': evaluate(Pd, Yd), 'test': evaluate(Pt, Yt), 'diag': diagnostics(model)}
    if FOCUS < 0:
        results[r]['task_weights'] = task_w(model).detach().cpu().numpy().round(4).tolist()
        if UNI: results[r]['class_weights'] = F.softplus(model.class_weights).detach().cpu().numpy().round(3).tolist()
    if args.head == 'grm':
        a, b = model.grm_params(); results[r]['grm'] = {'discrimination': a.detach().cpu().numpy().round(3).tolist(), 'thresholds': b.detach().cpu().numpy().round(3).tolist()}
    np.savez(os.path.join(args.out, f'preds_{r}.npz'), dev_probs=Pd, dev_labels=Yd, test_probs=Pt, test_labels=Yt)
    t = results[r]['test']
    if FOCUS >= 0: print(f"[{r:8s}] item {FOCUS + 1} test argmax CCC {t['argmax']['item']['CCC']:.4f} RMSE {t['argmax']['item']['RMSE']:.4f} | ev CCC {t['ev']['item']['CCC']:.4f}", flush=True)
    else: print(f"[{r:8s}] test ev CCC {t['ev']['CCC']:.4f} RMSE {t['ev']['RMSE']:.4f} MAE {t['ev']['MAE']:.4f} item CCC {t['ev']['item_CCC']:.4f} | argmax CCC {t['argmax']['CCC']:.4f} RMSE {t['argmax']['RMSE']:.4f} MAE {t['argmax']['MAE']:.4f} | diag {json.dumps({k: round(v, 3) for k, v in results[r]['diag'].items()})}", flush=True)
json.dump(results, open(os.path.join(args.out, 'metrics.json'), 'w'), indent=1)
print('DONE', flush=True)
