"""Scheduler for src/mtl.py (writes runs/<cond>/ under the repository root). A condition = feature set + flags. For each seed it trains the
unimodal runs (T, A, V as needed), then the multimodal runs initialised from them (text frozen).
Finished runs (metrics.json present) are skipped.

python src/run.py --cond NAME --txt mpnet --aud wavlm768 --vid resnet2048 [--mods TA,TV,TAV] [--flags "--aug_crop 0.7"] [--seeds 42,43,44,45,46]
"""
import argparse, os, subprocess, threading, time
import sys; ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); PY = sys.executable
MTL = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mtl.py')
ap = argparse.ArgumentParser()
ap.add_argument('--cond', required=True); ap.add_argument('--txt', default='bge'); ap.add_argument('--aud', default='egemaps23'); ap.add_argument('--vid', default='openface49')
ap.add_argument('--mods', default='T,A,V,TA,TV,TAV'); ap.add_argument('--flags', default=''); ap.add_argument('--uni_flags', default='')
ap.add_argument('--seeds', default='42,43,44,45,46'); ap.add_argument('--per_gpu', type=int, default=6); ap.add_argument('--gpus', default='0,1'); ap.add_argument('--dry_run', action='store_true')
ap.add_argument('--no_freeze', action='store_true')
ap.add_argument('--single_items', action='store_true', help='train 8 single-item models per (mods, seed) instead of one multi-task model')
ap.add_argument('--common', default='', help='flags appended to every mtl.py command (unimodal and multimodal)'); ap.add_argument('--cache', default='data/cache'); ap.add_argument('--init_cond', default='', help='take the unimodal trunks from this condition instead of the current one')
args = ap.parse_args()
OUT = os.path.join(ROOT, 'runs', args.cond); os.makedirs(OUT, exist_ok=True)
SEEDS = [int(s) for s in args.seeds.split(',')]; MODS = args.mods.split(','); GPUS = [int(g) for g in args.gpus.split(',')]
FEATS = ['--txt_feat', args.txt, '--aud_feat', args.aud, '--vid_feat', args.vid, '--cache', args.cache] + args.common.split()
INIT = os.path.join(ROOT, 'runs', args.init_cond) if args.init_cond else OUT

def log(m): print(f'[{time.strftime("%H:%M:%S")}] {m}', flush=True)
jobs = []
if True:
    for seed in SEEDS:
        for m in MODS:
            items = range(1, 9) if args.single_items else [0]
            for it in items:
                tag = f'{m}_item{it}_seed{seed}' if it else f'{m}_seed{seed}'; out = os.path.join(OUT, tag)
                cmd = [PY, '-u', MTL, '--mods', m, '--seed', str(seed), '--out', out, '--train'] + FEATS + (['--single_item', str(it)] if it else [])
                deps = []
                if len(m) > 1:
                    cmd += ['--init_dir', INIT] + ([] if args.no_freeze else ['--freeze_txt']) + args.flags.split()
                    deps = [] if args.init_cond else [f'{c}_item{it}_seed{seed}' if it else f'{c}_seed{seed}' for c in m]
                else:
                    cmd += args.uni_flags.split() + (args.flags.split() if it else [])
                jobs.append(dict(name=tag, cmd=cmd, deps=deps, out=out))
    have = {j['name'] for j in jobs}
    for j in list(jobs):
        for d in j['deps']:
            if d not in have:
                m, s_ = d.split('_seed'); out = os.path.join(OUT, d); m, it = (m.split('_item') + [''])[:2]
                jobs.insert(0, dict(name=d, cmd=[PY, '-u', MTL, '--mods', m, '--seed', s_, '--out', out, '--train'] + FEATS + (['--single_item', it] if it else []) + args.uni_flags.split() + (args.flags.split() if it else []), deps=[], out=out)); have.add(d)
def finished(name): return os.path.exists(os.path.join(OUT, name, 'metrics.json'))
if args.dry_run:
    for j in jobs: print(j['name'] + ':', ' '.join(j['cmd'][2:]))
    raise SystemExit
pending = [j for j in jobs if not finished(j['name'])]; log(f'{args.cond}: {len(jobs)} jobs, {len(pending)} to run')
lock = threading.Lock(); state = {'running': 0, 'failed': set()}
def pick():
    with lock:
        for j in list(pending):
            if any(d in state['failed'] for d in j['deps']): pending.remove(j); state['failed'].add(j['name']); log(f"{j['name']}: skipped (dependency failed)"); continue
            if all(finished(d) for d in j['deps']): pending.remove(j); state['running'] += 1; return j
        return None
def worker(gpu):
    while True:
        j = pick()
        if j is None:
            with lock: quiet = (not pending) or state['running'] == 0
            if not quiet: time.sleep(15); continue
            j = pick()
            if j is None: return
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2', TOKENIZERS_PARALLELISM='false'); t0 = time.time(); os.makedirs(j['out'], exist_ok=True)
        with open(os.path.join(j['out'], 'log.txt'), 'w') as f: rc = subprocess.call(j['cmd'], cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, env=env)
        ok = rc == 0 and finished(j['name'])
        with lock:
            state['running'] -= 1
            if not ok: state['failed'].add(j['name'])
        log(f"[gpu{gpu}] {j['name']}: {'ok' if ok else 'FAILED rc=' + str(rc)} ({(time.time() - t0) / 60:.1f} min)")
threads = [threading.Thread(target=worker, args=(g,)) for g in GPUS for _ in range(args.per_gpu)]
for t in threads: t.start()
for t in threads: t.join()
if pending: log(f"NOT STARTED (unmet dependencies): {[j['name'] for j in pending]}")
log(f"ALL DONE {args.cond}: {len(state['failed'])} failed" + (f": {sorted(state['failed'])}" if state['failed'] else '') + (f", {len(pending)} not started" if pending else ''))
