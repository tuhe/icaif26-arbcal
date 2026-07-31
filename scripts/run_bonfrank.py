from __future__ import annotations
import argparse
import glob
import json
import time
from pathlib import Path
import numpy as np
import lib_fullcal as fl
from arbcal import repricing as rp
from arbcal.aci import ExactACI, upper_rank, rank_threshold
BOOKS = fl.BOOKS
K_BOOKS = fl.K_BOOKS
NS = fl.NAMESPACE
ALPHAS = (0.1, 0.05, 0.01)
G_GRID = np.geomspace(1.0 / K_BOOKS, 1.0, 40)

def run_bonfrank(alpha, gamma, D, L, mu, sd, SL_by_i, scen_t):
    acis = [ExactACI(alpha / K_BOOKS, gamma / K_BOOKS) for _ in range(K_BOOKS)]
    arms = {m: dict(defined=np.zeros(D, bool), exceed=np.full(D, np.nan), reserve=np.full(D, np.nan), avail=np.zeros(D, bool)) for m in ('bonfrank', 'bonfrank_fixed')}
    for t in sorted(scen_t):
        if not (np.all(np.isfinite(L[t])) and np.all(np.isfinite(mu[t])) and np.all(np.isfinite(sd[t]))):
            continue
        SL = SL_by_i[scen_t[t]]
        u = np.array([upper_rank(SL[:, j], L[t, j]) for j in range(K_BOOKS)])
        betas = np.array([a.threshold() for a in acis])
        if not any((a.in_warmup for a in acis)):
            Bs = np.array([np.inf if betas[j] >= 1.0 or betas[j] == np.inf else -np.inf if betas[j] == -np.inf else rank_threshold(SL[:, j], betas[j]) for j in range(K_BOOKS)])
            r = arms['bonfrank']
            r['defined'][t] = True
            r['exceed'][t] = float(np.any(L[t] > Bs))
            r['avail'][t] = bool(np.all(np.isfinite(Bs)))
            rn = (Bs - mu[t]) / sd[t]
            r['reserve'][t] = float(np.mean(rn)) if np.all(np.isfinite(rn)) else np.inf
        for j, a in enumerate(acis):
            a.update(float(u[j]), betas[j])
        Bf = np.array([rank_threshold(SL[:, j], 1.0 - alpha / K_BOOKS) for j in range(K_BOOKS)])
        r = arms['bonfrank_fixed']
        r['defined'][t] = True
        r['exceed'][t] = float(np.any(L[t] > Bf))
        r['avail'][t] = bool(np.all(np.isfinite(Bf)))
        rn = (Bf - mu[t]) / sd[t]
        r['reserve'][t] = float(np.mean(rn)) if np.all(np.isfinite(rn)) else np.inf
    diags = [a.diagnostics() for a in acis]
    return (arms, diags)

def run_static_levels(alpha, D, L, mu, sd, SL_by_i, scen_t, dev_mask):
    nG = len(G_GRID)
    defined = np.zeros(D, bool)
    exceed = np.full((nG, D), np.nan)
    reserve = np.full((nG, D), np.nan)
    for t in sorted(scen_t):
        if not (np.all(np.isfinite(L[t])) and np.all(np.isfinite(mu[t])) and np.all(np.isfinite(sd[t]))):
            continue
        SL = SL_by_i[scen_t[t]]
        defined[t] = True
        for gi, g in enumerate(G_GRID):
            q = 1.0 - g * alpha
            Bs = np.array([rank_threshold(SL[:, j], q) for j in range(K_BOOKS)])
            exceed[gi, t] = float(np.any(L[t] > Bs))
            rn = (Bs - mu[t]) / sd[t]
            reserve[gi, t] = float(np.mean(rn)) if np.all(np.isfinite(rn)) else np.inf
    dm = defined & dev_mask
    dev_fwe = np.array([float(np.nanmean(exceed[gi, dm])) for gi in range(nG)])
    ok = np.where(dev_fwe <= alpha)[0]
    gi_sel = int(ok.max()) if len(ok) else int(np.argmin(dev_fwe))
    sel = dict(defined=defined, exceed=exceed[gi_sel], reserve=reserve[gi_sel], avail=defined & np.isfinite(reserve[gi_sel]))
    curve = dict(g_grid=G_GRID.copy(), dev_fwe=dev_fwe, g_selected=float(G_GRID[gi_sel]), gi_selected=gi_sel, exceed_all=exceed, reserve_all=reserve)
    return (sel, curve)

def selftest():
    rng = np.random.default_rng(0)
    D, M = (2200, 1000)
    L = rng.normal(size=(D, K_BOOKS))
    mu = np.zeros((D, K_BOOKS))
    sd = np.ones((D, K_BOOKS))
    SL_by_i = [rng.normal(size=(M, K_BOOKS)) for _ in range(D)]
    scen_t = {t: t for t in range(D)}
    for alpha in (0.1, 0.05):
        arms, diags = run_bonfrank(alpha, alpha / 10, D, L, mu, sd, SL_by_i, scen_t)
        for m, r in arms.items():
            d = r['defined']
            fwe = float(np.nanmean(r['exceed'][d]))
            n = int(d.sum())
            se = np.sqrt(alpha * (1 - alpha) / n)
            print(f"  alpha={alpha:g} {m:15s} n={n} FWE={fwe:.4f} (target <= {alpha:g}; union bound {1 - (1 - alpha / 6) ** 6:.4f}; 2SE={2 * se:.4f}) avail={float(np.mean(r['avail'][d])):.3f}")
            assert fwe <= alpha + 3 * se, (m, alpha, fwe)
        bounds = [dd['n_alpha_low'] + dd['n_alpha_high'] for dd in diags]
        print(f'    per-book ACI boundary events: {bounds}')
        dev_mask = np.zeros(D, bool)
        dev_mask[:D // 2] = True
        sel, curve = run_static_levels(alpha, D, L, mu, sd, SL_by_i, scen_t, dev_mask)
        d = sel['defined'] & ~dev_mask
        fwe = float(np.nanmean(sel['exceed'][d]))
        se = np.sqrt(alpha * (1 - alpha) / d.sum())
        print(f"  alpha={alpha:g} inflrank[indep] g*={curve['g_selected']:.3f} out-of-dev FWE={fwe:.4f} (target <= {alpha:g}, 2SE={2 * se:.4f})")
        assert curve['g_selected'] <= 0.25, curve['g_selected']
        assert fwe <= alpha + 3 * se, (alpha, fwe)
        f_re = rng.normal(size=(D, 1))
        Lc = 0.95 * f_re + 0.05 * rng.normal(size=(D, K_BOOKS))
        fs = rng.normal(size=(D, M, 1))
        SLc = list(0.95 * fs + 0.05 * rng.normal(size=(D, M, K_BOOKS)))
        selc, curvec = run_static_levels(alpha, D, Lc, mu, sd, SLc, scen_t, dev_mask)
        dc = selc['defined'] & ~dev_mask
        fwec = float(np.nanmean(selc['exceed'][dc]))
        print(f"  alpha={alpha:g} inflrank[comono] g*={curvec['g_selected']:.3f} out-of-dev FWE={fwec:.4f}")
        assert curvec['g_selected'] > 0.5, curvec['g_selected']
        assert fwec <= alpha + 3 * se, (alpha, fwec)
    print('SELFTEST PASS')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--panel')
    ap.add_argument('--scen-glob')
    ap.add_argument('--outdir')
    ap.add_argument('--selftest', action='store_true')
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    if not (args.panel and args.scen_glob and args.outdir):
        raise SystemExit('--panel/--scen-glob/--outdir required (or --selftest)')
    t0 = time.time()
    outdir = Path(args.outdir)
    vsrun = fl.load_vs_run()
    days, S, F, DISC, C = fl.load_panel(args.panel)
    manifest = fl.load_manifest()
    masks = fl.panel_masks(days, manifest)
    vt = masks['vt']
    D = len(days)
    L = fl.realized_losses(days, S, F, DISC, C, vt)
    mu, sd = vsrun.rolling_loc_scale(L)
    gamma_sel = json.load(open(NS / 'fullcal_canonical.json'))['meta']['gamma_selected']
    scen_files = sorted(glob.glob(args.scen_glob))
    if not scen_files:
        raise SystemExit(f'no scenario files match {args.scen_glob}')
    scen_t = {}
    SLp, SLr = ([], [])
    for f in scen_files:
        c = np.load(f, allow_pickle=True)
        for row, t in enumerate(np.asarray(c['t_idx'], int)):
            if t <= 0 or t >= D or str(c['dates'][row]) != days[t]:
                raise SystemExit(f'chunk/panel mismatch at row {row} of {f}')
            if str(c['cutoff_dates'][row]) != days[t - 1]:
                raise SystemExit(f'leakage stamp mismatch at {days[t]}')
            if not vt[t]:
                raise SystemExit(f'chunk contains invalid-transition date {days[t]}')
            books = rp.build_books(S[t - 1], F[t - 1], DISC[t - 1], C[t - 1])
            R = np.asarray(c['R'][row], float)
            St = S[t - 1] * np.exp(R)
            Fm, Dm = rp.scenario_forward_discount(St, S[t - 1], F[t - 1], DISC[t - 1])
            for target, surf in ((SLp, np.asarray(c['surf'][row], float)), (SLr, np.asarray(c['surf_raw'][row], float))):
                sl = np.empty((len(R), K_BOOKS))
                for j, name in enumerate(BOOKS):
                    sl[:, j] = rp.book_loss_bps(books[name], St, Fm, Dm, surf)
                target.append(sl)
            scen_t[t] = len(SLp) - 1
        print(f'[bonfrank] {Path(f).name}: cum {len(SLp)} scen dates ({time.time() - t0:.0f}s)', flush=True)
    results = {'meta': {'panel': str(args.panel), 'scen_glob': args.scen_glob, 'gamma_selected': gamma_sel, 'aci_per_book': 'ExactACI(alpha/6, gamma/6)', 'inflrank': 'static per-book rank level 1 - g*alpha; g = largest grid value with dev (2018-2020) FWE <= alpha, frozen, then evaluated out-of-dev', 'n_scen_dates': len(scen_t)}, 'alphas': {}}
    store = {}
    years = np.array([d[:4] for d in days])
    postdev = masks['primary'] & ~masks['dev']

    def year_fwe(defined, exceed):
        return {y: {'fwe': float(np.nanmean(exceed[defined & (years == y)])), 'n': int((defined & (years == y)).sum())} for y in sorted(set(years[defined])) if (defined & (years == y)).any()}
    for alpha in ALPHAS:
        gamma = float(gamma_sel[f'{alpha:g}'])
        blk = {}
        for lib, SL_by_i in (('proj', SLp), ('raw', SLr)):
            arms, diags = run_bonfrank(alpha, gamma, D, L, mu, sd, SL_by_i, scen_t)
            sel, curve = run_static_levels(alpha, D, L, mu, sd, SL_by_i, scen_t, masks['dev'])
            arms['inflrank'] = sel
            for m, r in arms.items():
                key = f'{m}_{lib}'
                blk[key] = {'periods': fl.summarize_periods(vsrun, masks, r['defined'], r['exceed'], r['reserve']), 'availability': {p: float(np.mean(r['avail'][r['defined'] & masks[p]])) for p in ('primary', 'y2023') if (r['defined'] & masks[p]).any()}, 'fwe_by_year': year_fwe(r['defined'], r['exceed'])}
                if m == 'bonfrank':
                    blk[key]['aci_boundary'] = [fl.boundary_report(dd) for dd in diags]
                store[f'a{alpha:g}_{key}'] = {k: v for k, v in r.items() if k != 'aci'}
            blk[f'inflrank_{lib}']['g_selected'] = curve['g_selected']
            blk[f'inflrank_{lib}']['transfer_curve'] = {'g_grid': [float(g) for g in curve['g_grid']], 'dev_fwe': [float(v) for v in curve['dev_fwe']], 'postdev_fwe': [float(np.nanmean(curve['exceed_all'][gi, sel['defined'] & postdev])) for gi in range(len(G_GRID))], 'y2023_fwe': [float(np.nanmean(curve['exceed_all'][gi, sel['defined'] & masks['y2023']])) for gi in range(len(G_GRID))]}
        results['alphas'][f'{alpha:g}'] = blk
        print(f"[bonfrank] alpha={alpha:g} done (g* proj={results['alphas'][f'{alpha:g}']['inflrank_proj']['g_selected']:.3f}) ({time.time() - t0:.0f}s)", flush=True)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / 'bonfrank.json').write_text(json.dumps(results, indent=1, default=float))
    np.savez_compressed(outdir / 'bonfrank_arrays.npz', dates=np.array(days), **{f'{ak}_{f}': v for ak, r in store.items() for f, v in r.items()})
    print(f'[bonfrank] wrote {outdir}/bonfrank.json — rerun scripts/q2q3_scen_native.py to fold the arm into the Task-A table')
if __name__ == '__main__':
    main()
