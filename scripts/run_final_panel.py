from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import numpy as np
import lib_fullcal as fl
import run_limits_m as rm
from run_bonfrank import run_bonfrank
from run_scalar_baselines import run_soft_baselines
from arbcal.aci import rank_threshold, upper_rank
BOOKS = fl.BOOKS
K = fl.K_BOOKS
NS = fl.NAMESPACE
ALPHAS = (0.1, 0.05, 0.01)
ARMS_ORDER = ('nominal', 'rank_aci', 'rank_roll', 'bonfrank_aci', 'bonfrank_static', 'fhs', 'direct')
DISP = {'nominal': 'nominal scenario quantiles', 'rank_aci': '\\textbf{ArbCal-Rank} (max-rank, exact ACI)', 'rank_roll': 'max-rank, fixed-level rolling', 'bonfrank_aci': 'per-book rank, exact ACI ($\\alpha/6$)', 'bonfrank_static': 'per-book rank, static ($1{-}\\alpha/6$)', 'fhs': 'filtered HS (EWMA) + Bonferroni', 'direct': 'direct max-score ACI'}

def run_static_perbook(alpha, D, L, mu, sd, SL_by_t, M, level_rule):
    q = level_rule(alpha)
    r = dict(defined=np.zeros(D, bool), exceed=np.full(D, np.nan), reserve=np.full(D, np.nan))
    for t in sorted(SL_by_t):
        if not (np.all(np.isfinite(L[t])) and np.all(np.isfinite(mu[t])) and np.all(np.isfinite(sd[t]))):
            continue
        SL = SL_by_t[t][:M]
        Bs = np.array([rank_threshold(SL[:, j], q) for j in range(K)])
        r['defined'][t] = True
        r['exceed'][t] = float(np.any(L[t] > Bs))
        rn = (Bs - mu[t]) / sd[t]
        r['reserve'][t] = float(np.mean(rn)) if np.all(np.isfinite(rn)) else np.inf
    return r

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--panel', required=True)
    ap.add_argument('--losses-glob', required=True)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--tag', choices=['gauss', 't3'], default='gauss', help='scenario law of the losses chunks; controls output names + which committed cross-check anchor is used')
    args = ap.parse_args()
    t0 = time.time()
    outdir = Path(args.outdir)
    vsrun = fl.load_vs_run()
    days, S, F, DISC, C = fl.load_panel(args.panel)
    masks = fl.panel_masks(days, fl.load_manifest())
    vt = masks['vt']
    D = len(days)
    L = fl.realized_losses(days, S, F, DISC, C, vt)
    mu, sd = vsrun.rolling_loc_scale(L)
    SL_by_t = rm.load_losses(args.losses_glob, days, vt)
    M_have = next(iter(SL_by_t.values())).shape[0]
    print(f'[final_panel] {len(SL_by_t)} scen dates, M={M_have} ({time.time() - t0:.0f}s)')
    scen_t = {t: t for t in SL_by_t}
    gamma_sel = json.load(open(NS / 'fullcal_canonical.json'))['meta']['gamma_selected']
    zb = np.load(NS / 'fullcal_canonical_b_arrays.npz', allow_pickle=True)
    assert [str(d) for d in zb['dates']] == days
    out = {'meta': {'losses_glob': args.losses_glob, 'M': int(M_have), 'gamma_selected': gamma_sel, 'rank_roll': 'ExactACI(alpha, gamma=0): same window/warmup/boundary conventions, alpha_t pinned at alpha', 'inference': 'moving-block bootstrap BLOCK_LEN=10 N_BOOT=2000 (frozen spec) on exceed series and paired diffs; binomial 2SE gate reported alongside', 'no_cqr': 'CQR excluded by design to maximize the common date set'}, 'alphas': {}}
    for alpha in ALPHAS:
        gamma = float(gamma_sel[f'{alpha:g}'])
        p = f'a{alpha:g}'
        arms = {}
        r = fl.run_rank_method(alpha, D, L, mu, sd, SL_by_t, M_have, gamma=gamma)
        if args.tag == 'gauss':
            ref_e = np.asarray(zb[f'{p}_rank_exceed'], float)
            n_dd = int(np.sum(np.asarray(zb[f'{p}_rank_defined'], bool) != r['defined']))
            n_ed = int(np.sum(np.nan_to_num(ref_e, nan=-9) != np.nan_to_num(r['exceed'], nan=-9)))
            xchk = {'kind': 'per-date vs canonical_b arrays', 'defined_diffs': n_dd, 'exceed_diffs': n_ed}
        else:
            cb = json.load(open(NS / 'fullcal_canonical_b.json'))
            cmask = r['defined'].copy()
            for m_ in ('rank', 'direct', 'bonf', 'cqr'):
                cmask &= np.asarray(zb[f'{p}_{m_}_defined'], bool)
            xchk = {'kind': 'summary vs canonical_b.json t3 arm'}
            for per in ('primary', 'y2023'):
                pm = cmask & masks[per]
                ref = cb['alphas'][f'{alpha:g}']['methods'][per]['t3']
                xchk[per] = {'n': [int(pm.sum()), ref['n_dates']], 'n_exceed': [int(np.nansum(r['exceed'][pm])), ref['n_exceed']], 'avail': [round(float(np.mean(np.isfinite(r['reserve'][pm]))), 4), round(ref['quote_availability'], 4)]}
        print(f'[final_panel] a={alpha:g} rank_aci cross-check: {xchk}')
        arms['rank_aci'] = r
        arms['rank_roll'] = fl.run_rank_method(alpha, D, L, mu, sd, SL_by_t, M_have, gamma=0.0)
        for M in (M_have, 1000):
            SL_M = [SL_by_t[t][:M] for t in sorted(SL_by_t)]
            scen_i = {t: i for i, t in enumerate(sorted(SL_by_t))}
            ba, _ = run_bonfrank(alpha, gamma, D, L, mu, sd, SL_M, scen_i)
            tag = '' if M == M_have else '_m1k'
            arms[f'bonfrank_aci{tag}'] = ba['bonfrank']
            arms[f'bonfrank_static{tag}'] = ba['bonfrank_fixed']
        arms['nominal'] = run_static_perbook(alpha, D, L, mu, sd, SL_by_t, M_have, lambda a: 1.0 - a)
        soft = run_soft_baselines(alpha, D, L, mu, sd)
        arms['fhs'] = soft['fhs_bonf']
        za = np.load(NS / 'fullcal_arrays.npz', allow_pickle=True)
        arms['direct'] = {'defined': np.asarray(za[f'{p}_direct_defined'], bool), 'exceed': np.asarray(za[f'{p}_direct_exceed'], float), 'reserve': np.asarray(za[f'{p}_direct_reserve'], float)}
        common = np.ones(D, bool)
        for m in ARMS_ORDER:
            common &= arms[m]['defined']
        blk = {'n_common': {per: int((common & masks[per]).sum()) for per in ('primary', 'y2023')}, 'rank_aci_crosscheck': xchk}
        for m, r in arms.items():
            e, res = (r['exceed'], r['reserve'])
            mm = {}
            for per in ('primary', 'y2023'):
                pm = common & masks[per]
                n = int(pm.sum())
                fin = pm & np.isfinite(res)
                fwe_ci = vsrun.block_boot(e[pm])
                se = float(np.sqrt(alpha * (1 - alpha) / n)) if n else None
                row = {'n': n, 'fwe': fwe_ci['mean'], 'fwe_lo': fwe_ci['lo'], 'fwe_hi': fwe_ci['hi'], 'n_exceed': int(np.nansum(e[pm])), 'gate_2se': bool(fwe_ci['mean'] <= alpha + 2 * se), 'gate_boot': bool(fwe_ci['lo'] <= alpha), 'finite_avail': float(fin.sum() / n) if n else None, 'mean_reserve_finite': float(np.mean(res[fin])) if fin.any() else None}
                for ref in ('nominal', 'rank_aci'):
                    if m == ref:
                        continue
                    er = arms[ref]['exceed']
                    row[f'dfwe_vs_{ref}'] = vsrun.block_boot((e - er)[pm])
                    rr = arms[ref]['reserve']
                    both = pm & np.isfinite(res) & np.isfinite(rr)
                    row[f'dres_vs_{ref}'] = vsrun.block_boot((res - rr)[both]) if both.any() else None
                mm[per] = row
            blk[m] = mm
        out['alphas'][f'{alpha:g}'] = blk
        print(f'[final_panel] alpha={alpha:g} done ({time.time() - t0:.0f}s)', flush=True)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f'final_panel_{args.tag}.json').write_text(json.dumps(out, indent=1, default=float))
    BLOCKS = ((('nominal', 'rank_roll', 'rank_aci', 'bonfrank_aci'), 'nominal'), (('direct', 'fhs'), 'rank_aci'))
    for suffix, per in (('2018_2022', 'primary'), ('2023', 'y2023')):
        rows = []
        for alpha in ALPHAS:
            blk = out['alphas'][f'{alpha:g}']
            lev = f'{100 * (1 - alpha):g}\\%'
            first = True
            for bi, (block, ref) in enumerate(BLOCKS):
                for m in block:
                    s = blk[m][per]
                    dag = '' if s['gate_2se'] else '^{\\dagger}'
                    if m == ref:
                        dstr = 'reference'
                    else:
                        dv = s.get(f'dres_vs_{ref}')
                        dstr = f"${dv['mean']:+.2f}$ $[{dv['lo']:+.2f},{dv['hi']:+.2f}]$ vs " + ('nom.' if ref == 'nominal' else 'rank') if dv and dv['n'] else '---'
                    rows.append(f"{(lev if first else '')} & {DISP[m]} & ${s['fwe']:.3f}{dag}$ $[{s['fwe_lo']:.3f},{s['fwe_hi']:.3f}]$ {{\\scriptsize(${s['n_exceed']}$)}} & ${s['finite_avail']:.3f}$ & {dstr}\\\\")
                    first = False
                if bi == 0:
                    rows.append('\\cmidrule(lr){2-5}')
            rows.append('\\midrule' if alpha != ALPHAS[-1] else '')
        tab = '\n'.join(['% AUTO-GENERATED by scripts/q2q3_final_panel.py -- candidate, NOT wired into main.tex.', f'% law={args.tag}, M={M_have}, one common date set (no CQR).', '% FWE interval: moving-block bootstrap (BLOCK_LEN=10, N_BOOT=2000) — main uncertainty procedure; \\dagger = the predeclared mechanical screen (2 binomial SE) fails.', '% Block 1 = scenario-derived methods (reserve delta vs nominal); block 2 = scalar limit methods (delta vs ArbCal-Rank ACI).', '\\begin{tabular}{@{}llccc@{}}', '\\toprule', 'level & method & FWE [95\\% CI] {\\scriptsize(count)} & finite avail.\\ & reserve $\\Delta$ [95\\% CI]\\\\', '\\midrule', *[x for x in rows if x], '\\bottomrule', '\\end{tabular}']) + '\n'
        (outdir / f'tab_q2q3_final_{args.tag}_{suffix}.tex').write_text(tab)
    for alpha in ALPHAS:
        blk = out['alphas'][f'{alpha:g}']
        print(f"\n=== alpha={alpha:g} (common: primary {blk['n_common']['primary']}, 2023 {blk['n_common']['y2023']})")
        for m in ARMS_ORDER + ('bonfrank_aci_m1k', 'bonfrank_static_m1k'):
            for per in ('primary', 'y2023'):
                s = blk[m][per]
                print(f"  {m:18s} {per:7s} FWE {s['fwe']:.3f} [{s['fwe_lo']:.3f},{s['fwe_hi']:.3f}] ({s['n_exceed']:3d}/{s['n']}) avail {s['finite_avail']:.3f} res {s['mean_reserve_finite'] or float('nan'):6.2f}")
    print(f'\nwrote {outdir}/final_panel.json + candidate tables ({time.time() - t0:.0f}s)')
if __name__ == '__main__':
    main()
