from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy.stats import norm
import lib_fullcal as fl
REPO = Path(__file__).resolve().parents[1]
NS = REPO / 'results'
OUT = REPO / 'results/q2q3_soft_baselines'
ALPHAS = (0.1, 0.05, 0.01)
W = 250
EWMA_LAM = 0.94
K = fl.K_BOOKS
LABELS = {'gauss_bonf': 'Gaussian parametric VaR + Bonferroni', 'hs_bonf': 'historical simulation VaR + Bonferroni', 'fhs_bonf': 'filtered HS (EWMA) VaR + Bonferroni', 'hs_max': 'historical simulation on max-loss (scalar)', 'gproj_fixed': 'nominal scenario quantiles', 'rank10k': 'ArbCal-Rank (M=10k)', 'direct': 'direct max-score ACI', 'bonf': 'direct-loss Bonferroni ACI'}

def run_soft_baselines(alpha, D, L, mu, sd):
    q_book = 1.0 - alpha / K
    q_max = 1.0 - alpha
    zg = norm.ppf(q_book)
    arms = {m: dict(defined=np.zeros(D, bool), exceed=np.full(D, np.nan), reserve=np.full(D, np.nan), B_bps=np.full((D, K), np.nan)) for m in ('gauss_bonf', 'hs_bonf', 'fhs_bonf', 'hs_max')}
    zhist = []
    ehist = []
    smax_hist = []
    ewma_var = None
    for t in range(D):
        if not (np.all(np.isfinite(L[t])) and np.all(np.isfinite(mu[t])) and np.all(np.isfinite(sd[t]))):
            continue
        z = (L[t] - mu[t]) / sd[t]
        s_max = float(np.max(z))
        n = len(zhist)
        if ewma_var is None:
            sig_pred = None
        else:
            sig_pred = np.sqrt(ewma_var)
        if n >= W:
            Zw = np.asarray(zhist[-W:], float)
            Ew = np.asarray(ehist[-W:], float)
            Sw = np.asarray(smax_hist[-W:], float)
            lims = {'gauss_bonf': Zw.mean(axis=0) + zg * Zw.std(axis=0, ddof=1), 'hs_bonf': np.quantile(Zw, q_book, axis=0), 'fhs_bonf': sig_pred * np.quantile(Ew, q_book, axis=0)}
            for m, Bz in lims.items():
                r = arms[m]
                r['defined'][t] = True
                r['exceed'][t] = float(np.any(z > Bz))
                r['reserve'][t] = float(np.mean(Bz))
                r['B_bps'][t] = mu[t] + sd[t] * Bz
            th = float(np.quantile(Sw, q_max))
            r = arms['hs_max']
            r['defined'][t] = True
            r['exceed'][t] = float(s_max > th)
            r['reserve'][t] = th
        if ewma_var is None:
            ewma_var = z ** 2
            sig_pred = np.sqrt(ewma_var)
        e = z / sig_pred
        ewma_var = EWMA_LAM * ewma_var + (1.0 - EWMA_LAM) * z ** 2
        zhist.append(z)
        ehist.append(e)
        smax_hist.append(s_max)
    return arms

def severity_bps(L, B_bps, mask):
    ex = np.max(L - B_bps, axis=1)
    breach = mask & np.isfinite(ex) & (ex > 0)
    v = ex[breach]
    return dict(n_breach=int(breach.sum()), median_bps=float(np.median(v)) if len(v) else None, mean_bps=float(np.mean(v)) if len(v) else None, max_bps=float(np.max(v)) if len(v) else None)

def main():
    vsrun = fl.load_vs_run()
    z = np.load(NS / 'fullcal_arrays.npz', allow_pickle=True)
    zb = np.load(NS / 'fullcal_canonical_b_arrays.npz', allow_pickle=True)
    days = [str(d) for d in z['dates']]
    assert [str(d) for d in zb['dates']] == days
    masks = fl.panel_masks(days, fl.load_manifest())
    years = np.array([d[:4] for d in days])
    L = np.asarray(z['L'], float)
    mu = np.asarray(z['mu'], float)
    sd = np.asarray(z['sd'], float)
    D = len(days)
    volproxy = np.where(np.all(np.isfinite(sd), axis=1), np.nanmean(sd, axis=1), np.nan)
    out = {'meta': {'inputs': ['fullcal_arrays.npz', 'fullcal_canonical_b_arrays.npz'], 'window': W, 'ewma_lambda': EWMA_LAM, 'quantile_convention': 'linear-interpolation empirical quantile (practitioner convention) — deliberate: at alpha/6 on a 250-day window a finite-sample-exact order statistic does not exist', 'reserve_units': 'per-book arms: mean over books of the standardized limit; scalar arms (hs_max, direct): max-score threshold, z-units', 'severity_note': 'nominal scenario arms (gproj_fixed/graw_fixed) have no committed per-book limits; severity only for rank10k + constructed arms'}, 'alphas': {}}
    for alpha in ALPHAS:
        soft = run_soft_baselines(alpha, D, L, mu, sd)
        p = f'a{alpha:g}'
        arms = {m: (soft[m]['defined'], soft[m]['exceed'], soft[m]['reserve']) for m in soft}
        for m in ('gproj_fixed', 'direct', 'bonf'):
            arms[m] = (np.asarray(z[f'{p}_{m}_defined'], bool), np.asarray(z[f'{p}_{m}_exceed'], float), np.asarray(z[f'{p}_{m}_reserve'], float))
        arms['rank10k'] = (np.asarray(zb[f'{p}_rank_defined'], bool), np.asarray(zb[f'{p}_rank_exceed'], float), np.asarray(zb[f'{p}_rank_reserve'], float))
        common = np.ones(D, bool)
        for d, _, _ in arms.values():
            common &= d
        pm_all = common & masks['primary'] & np.isfinite(volproxy)
        edges = np.nanquantile(volproxy[pm_all], [1 / 3, 2 / 3])
        terc = np.full(D, -1)
        terc[pm_all & (volproxy <= edges[0])] = 0
        terc[pm_all & (volproxy > edges[0]) & (volproxy <= edges[1])] = 1
        terc[pm_all & (volproxy > edges[1])] = 2
        blk = {'n_common': {per: int((common & masks[per]).sum()) for per in ('primary', 'y2023')}, 'vol_tercile_edges_bps': [float(e) for e in edges]}
        for m, (d, e, r) in arms.items():
            mm = {}
            for per in ('primary', 'y2023'):
                pmask = common & masks[per]
                nn = int(pmask.sum())
                fin = pmask & np.isfinite(r)
                fwe = float(np.nanmean(e[pmask])) if nn else None
                se = float(np.sqrt(alpha * (1 - alpha) / nn)) if nn else None
                row = {'n': nn, 'fwe': fwe, 'n_exceed': int(np.nansum(e[pmask])) if nn else None, 'finite_avail': float(fin.sum() / nn) if nn else None, 'mean_reserve_finite': float(np.mean(r[fin])) if fin.any() else None, 'gate_2se': None if fwe is None else bool(fwe <= alpha + 2 * se)}
                for ref in ('rank10k', 'gproj_fixed'):
                    if m == ref:
                        continue
                    _, _, rr = arms[ref]
                    both = pmask & np.isfinite(r) & np.isfinite(rr)
                    dd = (r - rr)[both]
                    row[f'delta_res_vs_{ref}'] = vsrun.block_boot(dd) if len(dd) else None
                mm[per] = row
            mm['fwe_by_year'] = {}
            for y in sorted(set(years[common & masks['primary']])):
                ym = common & (years == y)
                if ym.any():
                    mm['fwe_by_year'][y] = {'fwe': float(np.nanmean(e[ym])), 'n_exceed': int(np.nansum(e[ym])), 'n': int(ym.sum())}
            mm['fwe_by_vol_tercile'] = {}
            for k_, name in enumerate(('low', 'mid', 'high')):
                tm = terc == k_
                if tm.any():
                    mm['fwe_by_vol_tercile'][name] = {'fwe': float(np.nanmean(e[tm])), 'n_exceed': int(np.nansum(e[tm])), 'n': int(tm.sum())}
            blk[m] = mm
        sev = {}
        B_rank = np.asarray(zb[f'{p}_rank_B_bps'], float)
        for per in ('primary', 'y2023'):
            pmask = common & masks[per]
            s = {'rank10k': severity_bps(L, B_rank, pmask)}
            for m in ('gauss_bonf', 'hs_bonf', 'fhs_bonf'):
                s[m] = severity_bps(L, soft[m]['B_bps'], pmask)
            sev[per] = s
        book_breach = (L > B_rank) & common[:, None] & masks['primary'][:, None]
        blk['severity_bps'] = sev
        blk['rank10k_breach_by_book_primary'] = {fl.BOOKS[j]: int(book_breach[:, j].sum()) for j in range(K)}
        out['alphas'][f'{alpha:g}'] = blk
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'summary.json').write_text(json.dumps(out, indent=1, default=float))
    ORDER = ('gproj_fixed', 'gauss_bonf', 'hs_bonf', 'fhs_bonf', 'hs_max', 'rank10k')
    DISP = {'gproj_fixed': 'nominal scenario quantiles', 'gauss_bonf': 'Gaussian VaR ($\\alpha/6$ Bonferroni)', 'hs_bonf': 'historical-sim.\\ VaR ($\\alpha/6$ Bonferroni)', 'fhs_bonf': 'filtered HS (EWMA) ($\\alpha/6$ Bonferroni)', 'hs_max': 'historical-sim.\\ max-loss', 'rank10k': '\\textbf{ArbCal-Rank} ($M{=}10$k)'}
    for suffix, per in (('2018_2022', 'primary'), ('2023', 'y2023')):
        rows = []
        for alpha in ALPHAS:
            blk = out['alphas'][f'{alpha:g}']
            lev = f'{100 * (1 - alpha):g}\\%'
            for i, m in enumerate(ORDER):
                s = blk[m][per]
                dag = '' if s['gate_2se'] else '^{\\dagger}'
                rows.append(f"{(lev if i == 0 else '')} & {DISP[m]} & ${s['fwe']:.3f}{dag}$ (${s['n_exceed']}$) & ${s['finite_avail']:.3f}$ & ${s['mean_reserve_finite']:.2f}$\\\\")
            rows.append('\\midrule' if alpha != ALPHAS[-1] else '')
        tab = '\n'.join(['% AUTO-GENERATED by scripts/run_scalar_baselines.py -- candidate, NOT wired into main.tex.', f'% Classic risk-desk baselines vs ArbCal-Rank ({per}); W={W}, EWMA lambda={EWMA_LAM}; reserve units per meta.', '% \\dagger marks FWE above target by more than 2 binomial SE.', '\\begin{tabular}{@{}llccc@{}}', '\\toprule', 'level & method & FWE (count) & finite avail.\\ & mean reserve\\\\', '\\midrule', *[r_ for r_ in rows if r_], '\\bottomrule', '\\end{tabular}']) + '\n'
        (OUT / f'tab_q2q3_soft_baselines_{suffix}.tex').write_text(tab)
    for alpha in ALPHAS:
        blk = out['alphas'][f'{alpha:g}']
        print(f"\n=== alpha={alpha:g} (common: primary {blk['n_common']['primary']}, 2023 {blk['n_common']['y2023']}; vol tercile edges {blk['vol_tercile_edges_bps']})")
        for m in ORDER + ('direct', 'bonf'):
            s = blk[m]
            for per in ('primary', 'y2023'):
                r = s[per]
                print(f"  {m:11s} {per:7s} FWE {r['fwe']:.3f} ({r['n_exceed']:3d}/{r['n']}) avail {r['finite_avail']:.3f} meanRes {r['mean_reserve_finite'] or float('nan'):6.2f}")
            yb = s['fwe_by_year']
            print('              by-year FWE: ' + '  '.join((f"{y}:{yb[y]['fwe']:.3f}" for y in sorted(yb))))
            tb = s['fwe_by_vol_tercile']
            print('              by-vol FWE:  ' + '  '.join((f"{k}:{tb[k]['fwe']:.3f}" for k in ('low', 'mid', 'high') if k in tb)))
        print(f"  rank10k breach-by-book (primary): {blk['rank10k_breach_by_book_primary']}")
        for per in ('primary', 'y2023'):
            print(f'  severity ({per}): ' + '  '.join((f"{m}: n={v['n_breach']} med={(v['median_bps'] if v['median_bps'] is not None else float('nan')):.0f}bp" for m, v in blk['severity_bps'][per].items())))
    print(f'\nwrote {OUT}/summary.json + candidate tables')
if __name__ == '__main__':
    main()
