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
from arbcal import repricing as rp
from arbcal.aci import ExactACI, upper_rank, rank_threshold
from arbcal.scenarios import shrunk_cov
M = 10000

def fixed_or_rank(alpha, D, L, mu, sd, SL_by_t, fixed, gamma):
    a = None if fixed else ExactACI(alpha, gamma)
    defined = np.zeros(D, bool)
    exceed = np.full(D, np.nan)
    for t in range(D):
        if t not in SL_by_t or not np.all(np.isfinite(mu[t])) or (not np.all(np.isfinite(L[t]))):
            continue
        SL = SL_by_t[t]
        if fixed:
            Bs = np.array([rank_threshold(SL[:, j], 1.0 - alpha) for j in range(fl.K_BOOKS)])
            defined[t] = True
            exceed[t] = float(np.any(L[t] > Bs))
        else:
            u = np.array([upper_rank(SL[:, j], L[t, j]) for j in range(fl.K_BOOKS)])
            s = float(np.max(u))
            beta = a.threshold()
            if not a.in_warmup:
                if beta == np.inf or beta >= 1.0:
                    Bs = np.full(fl.K_BOOKS, np.inf)
                elif beta == -np.inf:
                    Bs = np.full(fl.K_BOOKS, -np.inf)
                else:
                    Bs = np.array([rank_threshold(SL[:, j], beta) for j in range(fl.K_BOOKS)])
                defined[t] = True
                exceed[t] = float(np.any(L[t] > Bs))
            a.update(s, beta)
    return (defined, exceed)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--panel', required=True)
    ap.add_argument('--losses-glob', required=True)
    ap.add_argument('--canonical', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--law', choices=['gauss', 't'], default='gauss', help='law of the losses chunks; the raw redraw must match it (scen.py draw order: normals, then chi-square mixture)')
    ap.add_argument('--nu', type=float, default=3.0)
    args = ap.parse_args()
    t0 = time.time()
    vsrun = fl.load_vs_run()
    days, S, F, DISC, C = fl.load_panel(args.panel)
    manifest = fl.load_manifest()
    masks = fl.panel_masks(days, manifest)
    vt = masks['vt']
    D = len(days)
    gsel = {float(a): g for a, g in json.loads(Path(args.canonical).read_text())['meta']['gamma_selected'].items()}
    SLp = {}
    seed_base = None
    for f in sorted(glob.glob(args.losses_glob)):
        c = np.load(f, allow_pickle=True)
        sb = int(c['seed_base'])
        assert seed_base in (None, sb)
        seed_base = sb
        for row, t in enumerate(np.asarray(c['t_idx'], int)):
            SLp[int(t)] = np.asarray(c['SL'][row], float)
    L = fl.realized_losses(days, S, F, DISC, C, vt)
    mu, sd = fl.load_vs_run().rolling_loc_scale(L)
    spotret_abs, atmchg = fl.lagged_covariates(S, C, vt)
    SLr = {}
    for i, t in enumerate(sorted(SLp)):
        innov, Yv_full = fl.gap_innovations(S, C, vt, t)
        E = innov[-CFG.LW_WINDOW:]
        K = shrunk_cov(E)
        mu_d = np.concatenate([[0.0], Yv_full[-1]])
        rng = np.random.default_rng(seed_base + t)
        Lc = np.linalg.cholesky(K)
        eps = rng.standard_normal((M, len(mu_d))) @ Lc.T
        if args.law == 't':
            W = rng.chisquare(args.nu, size=M)
            eps = eps * np.sqrt((args.nu - 2.0) / W)[:, None]
        draws = mu_d[None, :] + eps
        R = draws[:, 0]
        surf_raw = draws[:, 1:].reshape(M, CFG.N_X, CFG.N_TAU)
        books = rp.build_books(S[t - 1], F[t - 1], DISC[t - 1], C[t - 1])
        St = S[t - 1] * np.exp(R)
        Fm, Dm = rp.scenario_forward_discount(St, S[t - 1], F[t - 1], DISC[t - 1])
        sl = np.empty((M, fl.K_BOOKS))
        for j, name in enumerate(fl.BOOKS):
            sl[:, j] = rp.book_loss_bps(books[name], St, Fm, Dm, surf_raw)
        SLr[t] = sl
        if (i + 1) % 100 == 0:
            print(f'[fullcal_abl10k] raw {i + 1}/{len(SLp)} ({time.time() - t0:.0f}s)', flush=True)
    results = {'arms': {}, 'meta': {'M': M, 'gap_aware': True, 'law': args.law, 'nu': args.nu if args.law == 't' else None, 'gamma_selected': {f'{a:g}': gsel[a] for a in gsel}}}
    for alpha in (0.1, 0.05):
        out_sc, _ = fl.run_scenario_methods(alpha, days, L, mu, sd, None, None, {}, spotret_abs, atmchg, gamma=gsel[alpha], methods=('direct', 'bonf', 'cqr'))
        d_rank, _ = fixed_or_rank(alpha, D, L, mu, sd, SLp, False, gsel[alpha])
        common = d_rank & out_sc['direct']['defined'] & out_sc['cqr']['defined'] & out_sc['bonf']['defined']
        arms = {}
        for key, SL_arm, fixed in (('graw_fixed', SLr, True), ('gproj_fixed', SLp, True), ('g_rank', SLr, False), ('gproj_rank', SLp, False)):
            dfn, exc = fixed_or_rank(alpha, D, L, mu, sd, SL_arm, fixed, gsel[alpha])
            arms[key] = {}
            for period in ('primary', 'y2023'):
                m = common & dfn & masks[period]
                arms[key][period] = {'n': int(m.sum()), 'fwe': float(np.nanmean(exc[m])) if m.any() else None, 'n_exceed': int(np.nansum(exc[m]))}
        results['arms'][f'{alpha:g}'] = arms
        print(f'[fullcal_abl10k] alpha={alpha} done ({time.time() - t0:.0f}s)', flush=True)
    Path(args.out).write_text(json.dumps(results, indent=1, default=float))
    print(f'[fullcal_abl10k] wrote {args.out} ({time.time() - t0:.0f}s)')
if __name__ == '__main__':
    main()
