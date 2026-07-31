from __future__ import annotations
import argparse
import glob
import json
import time
import warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings('ignore')
import lib_fullcal as fl
from arbcal import config as CFG

def load_losses(globpat):
    SL = {}
    for f in sorted(glob.glob(globpat)):
        c = np.load(f, allow_pickle=True)
        for row, t in enumerate(np.asarray(c['t_idx'], int)):
            SL[int(t)] = np.asarray(c['SL'][row], float)
    return SL

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--panel', required=True)
    ap.add_argument('--gauss-glob', required=True)
    ap.add_argument('--t-globs', required=True)
    ap.add_argument('--canonical', required=True, help='fullcal_canonical.json (selected gammas)')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    t0 = time.time()
    days, S, F, DISC, C = fl.load_panel(args.panel)
    manifest = fl.load_manifest()
    masks = fl.panel_masks(days, manifest)
    D = len(days)
    L = fl.realized_losses(days, S, F, DISC, C, masks['vt'])
    vsrun = fl.load_vs_run()
    mu, sd = vsrun.rolling_loc_scale(L)
    gsel = {float(a): g for a, g in json.loads(Path(args.canonical).read_text())['meta']['gamma_selected'].items()}
    laws = {'gauss': load_losses(args.gauss_glob)}
    for nu, g in json.loads(args.t_globs).items():
        laws[f't{nu}'] = load_losses(g)
    print('[fullcal_tsel] laws:', {k: len(v) for k, v in laws.items()}, flush=True)
    dev = masks['dev']
    results = {'mode': 'select', 'levels': {}}
    for alpha in CFG.ALPHAS:
        lvl = {}
        for law, SL in laws.items():
            r = fl.run_rank_method(alpha, D, L, mu, sd, SL, 10000, gamma=gsel[alpha])
            m = r['defined'] & dev
            n = int(m.sum())
            res = r['reserve'][m]
            lvl[law] = {'n': n, 'fwe': float(np.nanmean(r['exceed'][m])) if n else np.nan, 'n_exceed': int(np.nansum(r['exceed'][m])), 'availability': float(np.mean(r['avail'][m])) if n else np.nan, 'mean_finite_reserve': float(np.mean(res[np.isfinite(res)])) if np.any(np.isfinite(res)) else np.nan}
        results['levels'][f'{alpha:g}'] = lvl
    cands = list(laws.keys())

    def fwe_ok(law, levels):
        for alpha in levels:
            r = results['levels'][f'{alpha:g}'][law]
            se = np.sqrt(alpha * (1 - alpha) / max(r['n'], 1))
            if abs(r['fwe'] - alpha) > 2 * se:
                return False
        return True
    pool = [c for c in cands if fwe_ok(c, CFG.ALPHAS)]
    rule = 'all levels'
    if not pool:
        pool = [c for c in cands if fwe_ok(c, (0.1, 0.05))]
        rule = '90/95 fallback'
    mean_avail = {c: float(np.mean([results['levels'][f'{a:g}'][c]['availability'] for a in CFG.ALPHAS])) for c in cands}

    def nu_of(c):
        return np.inf if c == 'gauss' else float(c[1:])
    pool.sort(key=lambda c: (-mean_avail[c], nu_of(c)))
    selected = pool[0] if pool else None
    results['selection'] = {'criterion': f'max mean availability s.t. dev FWE within 2 SE ({rule}); ties -> smaller nu', 'pool': pool, 'selected': selected, 'mean_availability': mean_avail, 'gamma_used': {f'{a:g}': gsel[a] for a in CFG.ALPHAS}}
    Path(args.out).write_text(json.dumps(results, indent=1, default=float))
    print(f'[fullcal_tsel] selected: {selected} (pool {pool}, {rule}) ({time.time() - t0:.0f}s)')
if __name__ == '__main__':
    main()
