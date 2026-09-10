"""Collect runs/<cond>/<mods>_seed<s>/metrics.json into per-seed and mean +- s.d. tables, plus
five-seed ensembles from the saved probabilities. Usage: python collect.py [cond ...]"""
import glob, json, os, sys
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.abspath(__file__)); RUNS = os.path.join(ROOT, 'runs')
conds = sys.argv[1:] or sorted(os.listdir(RUNS))
def ccc(p, l): p, l = np.asarray(p, float), np.asarray(l, float); return 2 * np.cov(p, l, bias=True)[0, 1] / (p.var() + l.var() + (p.mean() - l.mean()) ** 2)
rows, ens = [], []
def questmf_rows(cond):
    """Combine the eight item runs of a QuestMF condition into total-score predictions per (mods, seed)."""
    out_rows, out_ens = [], []
    keys = sorted({(os.path.basename(d).split('_item')[0], int(os.path.basename(d).split('_seed')[1])) for d in glob.glob(os.path.join(RUNS, cond, '*_item*_seed*'))})
    for rule in ['loss', 'itemccc']:
        per_mods = {}
        for mods, seed in keys:
            files = [os.path.join(RUNS, cond, f'{mods}_item{i}_seed{seed}', f'preds_{rule}.npz') for i in range(1, 9)]
            if not all(os.path.exists(f) for f in files): continue
            Z = [np.load(f) for f in files]; Yt = Z[0]['test_labels'].sum(1); Yd = Z[0]['dev_labels'].sum(1)
            am = sum(z['test_probs'][:, i].argmax(-1) for i, z in enumerate(Z)).astype(float); ev = sum((z['test_probs'][:, i] * np.arange(4)).sum(-1) for i, z in enumerate(Z))
            for dec, p in [('argmax', am), ('ev', ev)]:
                out_rows.append(dict(cond=cond, mods=mods, seed=seed, rule=rule, decode=dec, CCC=ccc(p, Yt), RMSE=float(np.sqrt(np.mean((p - Yt) ** 2))), MAE=float(np.mean(np.abs(p - Yt)))))
                per_mods.setdefault((mods, dec), []).append(p); per_mods.setdefault((mods, 'Y'), Yt)
        for (mods, dec), ps in per_mods.items():
            if dec == 'Y' or len(ps) < 5: continue
            p = np.mean(ps, 0); Yt = per_mods[(mods, 'Y')]
            out_ens.append(dict(cond=cond, mods=mods, rule=rule, decode=dec, n=len(ps), CCC=ccc(p, Yt), RMSE=float(np.sqrt(np.mean((p - Yt) ** 2))), MAE=float(np.mean(np.abs(p - Yt)))))
    return out_rows, out_ens

for cond in conds:
    if glob.glob(os.path.join(RUNS, cond, '*_item*_seed*')):
        r_, e_ = questmf_rows(cond); rows += r_; ens += e_; continue
    for d in sorted(glob.glob(os.path.join(RUNS, cond, '*_seed*'))):
        mf = os.path.join(d, 'metrics.json')
        if not os.path.exists(mf): continue
        mods, seed = os.path.basename(d).split('_seed'); r = json.load(open(mf))
        for rule in ['loss', 'itemccc', 'devrmse', 'devccc']:
            if rule not in r or 'CCC' not in r[rule]['test']['argmax']: continue
            for dec in ['argmax', 'ev']:
                t = r[rule]['test'][dec]; rows.append(dict(cond=cond, mods=mods, seed=int(seed), rule=rule, decode=dec, **{k: t[k] for k in ['CCC', 'RMSE', 'MAE']}))
    for mods in sorted({os.path.basename(d).split('_seed')[0] for d in glob.glob(os.path.join(RUNS, cond, '*_seed*'))}):
        for rule in ['loss', 'devrmse']:
            files = sorted(glob.glob(os.path.join(RUNS, cond, f'{mods}_seed*', f'preds_{rule}.npz')))
            if len(files) < 5: continue
            Z = [np.load(f) for f in files]; Y = Z[0]['test_labels'].sum(1)
            am = np.mean([z['test_probs'].argmax(-1).sum(1) for z in Z], 0); ev = np.mean([(z['test_probs'] * np.arange(4)).sum(-1).sum(1) for z in Z], 0)
            for dec, p in [('argmax', am), ('ev', ev)]:
                ens.append(dict(cond=cond, mods=mods, rule=rule, decode=dec, n=len(files), CCC=ccc(p, Y), RMSE=float(np.sqrt(np.mean((p - Y) ** 2))), MAE=float(np.mean(np.abs(p - Y)))))
df = pd.DataFrame(rows)
if len(df):
    g = df.groupby(['cond', 'mods', 'rule', 'decode']); s = g.agg(n=('seed', 'size'), CCC=('CCC', 'mean'), CCC_sd=('CCC', 'std'), RMSE=('RMSE', 'mean'), RMSE_sd=('RMSE', 'std'), MAE=('MAE', 'mean'), MAE_sd=('MAE', 'std')).reset_index()
    s.to_csv(os.path.join(RUNS, 'summary.csv'), index=False); df.to_csv(os.path.join(RUNS, 'per_seed.csv'), index=False)
    print('| cond | mods | rule | decode | n | CCC | RMSE | MAE |\n|---|---|---|---|---|---|---|---|')
    for _, r in s.iterrows(): print(f"| {r.cond} | {r.mods} | {r.rule} | {r.decode} | {int(r.n)} | {r.CCC:.3f} ± {r.CCC_sd:.3f} | {r.RMSE:.3f} ± {r.RMSE_sd:.3f} | {r.MAE:.3f} ± {r.MAE_sd:.3f} |")
if ens:
    e = pd.DataFrame(ens); e.to_csv(os.path.join(RUNS, 'ensembles.csv'), index=False)
    print('\n| cond | mods | rule | decode | ensemble CCC | RMSE | MAE |\n|---|---|---|---|---|---|---|')
    for _, r in e.iterrows(): print(f"| {r.cond} | {r.mods} | {r.rule} | {r.decode} | {r.CCC:.3f} | {r.RMSE:.3f} | {r.MAE:.3f} |")
