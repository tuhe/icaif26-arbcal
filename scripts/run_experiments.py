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
from arbcal.aci import ExactACI, rank_threshold
from arbcal.polyhedron import polyhedron_rows, PolyProjector, ball_admits_violation, box_admits_violation, FAMILIES
BOOKS = fl.BOOKS
K_BOOKS = fl.K_BOOKS

def select_gamma(gs, alpha):
    cand = []
    for key, r in gs.items():
        d = r['aci']
        boundary = d['n_alpha_low'] + d['n_alpha_high'] > 0
        n = max(r['dev']['n_dates'], 1)
        se = np.sqrt(alpha * (1 - alpha) / n)
        fwe_ok = abs(r['dev']['fwe'] - alpha) <= 2 * se
        cand.append((key, r['gamma'], boundary, fwe_ok, r['dev']['mean_reserve']))
    pool = [c for c in cand if not c[2] and c[3]]
    if not pool:
        pool = [c for c in cand if not c[2]] or cand
    pool.sort(key=lambda c: (c[4], c[1]))
    return (pool[0][0], pool[0][1], {'candidates': {c[0]: {'gamma': c[1], 'boundary_event': bool(c[2]), 'dev_fwe_within_2se': bool(c[3]), 'dev_mean_reserve': c[4]} for c in cand}})

def experiment_a(days, Yv, G, h, fam, masks, vsrun):
    vt, vt2 = (masks['vt'], masks['vt2'])
    D, dim = Yv.shape
    proj = PolyProjector(G, h, family=fam)
    alpha_a = 0.1
    dates = np.asarray(days)
    fc = {'rw': np.full((D, dim), np.nan), 'ar1': np.full((D, dim), np.nan)}
    for t in range(1, D):
        if vt[t]:
            fc['rw'][t] = Yv[t - 1]
        w0 = max(0, t - CFG.ACI_WINDOW)
        Yw = Yv[w0:t]
        pair_ok = vt[w0 + 1:t]
        if len(Yw) >= CFG.WARMUP and vt[t]:
            ybar = Yw.mean(0)
            A0 = (Yw[1:] - ybar)[pair_ok]
            A1 = (Yw[:-1] - ybar)[pair_ok]
            num = np.sum(A0 * A1, 0)
            den = np.sum(A1 ** 2, 0) + 1e-12
            phi = np.clip(num / den, -1.0, 1.0)
            fc['ar1'][t] = ybar + phi * (Yv[t - 1] - ybar)
    out = {}
    per_date = {}
    for fname, Shat in fc.items():
        ok = np.all(np.isfinite(Shat), axis=1)
        proj_c = np.full((D, dim), np.nan)
        pdist = np.full(D, np.nan)
        for t in np.where(ok)[0]:
            zt, feas = proj.project(Shat[t])
            if zt is not None:
                proj_c[t] = zt
                pdist[t] = float(np.linalg.norm(Shat[t] - zt))
        s_raw = np.linalg.norm(Yv - Shat, axis=1)
        s_prj = np.linalg.norm(Yv - proj_c, axis=1)
        resid = np.abs(Yv - Shat)
        aci_ball, aci_box, aci_raw = (ExactACI(alpha_a), ExactACI(alpha_a), ExactACI(alpha_a))
        rec = dict(t=[], cov_prj=[], cov_box=[], cov_raw=[], q_prj=[], q_raw=[], lam=[], wshape=[], ball_viol=[], box_viol=[], raw_center_infeas=[], pdist=[])
        for t in range(D):
            if not (ok[t] and np.isfinite(s_prj[t])):
                continue
            w0 = max(0, t - CFG.ACI_WINDOW)
            past = resid[w0:t][np.all(np.isfinite(resid[w0:t]), axis=1)]
            wshape = np.quantile(past, 0.9, axis=0) + 1e-08 if len(past) >= CFG.WARMUP else None
            q_p = aci_ball.threshold()
            q_r = aci_raw.threshold()
            lam = aci_box.threshold()
            if not (aci_ball.in_warmup or wshape is None):
                rec['t'].append(t)
                rec['cov_prj'].append(float(s_prj[t] <= q_p))
                rec['cov_raw'].append(float(s_raw[t] <= q_r))
                smx = float(np.max(np.abs(Yv[t] - Shat[t]) / wshape))
                rec['cov_box'].append(float(smx <= lam))
                rec['q_prj'].append(q_p if np.isfinite(q_p) else np.nan)
                rec['q_raw'].append(q_r if np.isfinite(q_r) else np.nan)
                rec['lam'].append(lam)
                rec['wshape'].append(wshape.copy())
                rec['ball_viol'].append([ball_admits_violation(Shat[t], q_r, G, h, fam)[f] for f in FAMILIES] if np.isfinite(q_r) else None)
                rec['box_viol'].append([box_admits_violation(Shat[t], lam * wshape, G, h, fam)[f] for f in FAMILIES] if np.isfinite(lam) else None)
                rec['raw_center_infeas'].append(float(pdist[t] > 1e-07))
                rec['pdist'].append(float(pdist[t]))
            aci_ball.update(s_prj[t], q_p)
            aci_raw.update(s_raw[t], q_r)
            if wshape is not None:
                smx = float(np.max(np.abs(Yv[t] - Shat[t]) / wshape))
                aci_box.update(smx, lam)
        ts = np.asarray(rec['t'], int)
        blk = {'n_scored_total': int(len(ts)), 'first_scored_date': days[ts[0]] if len(ts) else None, 'aci_boundary': {k: fl.boundary_report(a.diagnostics()) for k, a in (('ball', aci_ball), ('box', aci_box), ('raw', aci_raw))}}
        for period in ('dev', 'primary', 'y2023'):
            pm = masks[period][ts]
            n = int(pm.sum())
            if not n:
                blk[period] = {'n_scored': 0}
                continue
            sel = lambda k: np.asarray(rec[k], float)[pm]
            bv = [rec['ball_viol'][i] for i in range(len(ts)) if pm[i]]
            xv = [rec['box_viol'][i] for i in range(len(ts)) if pm[i]]
            bvv = np.asarray([v for v in bv if v is not None], float)
            xvv = np.asarray([v for v in xv if v is not None], float)
            blk[period] = {'n_scored': n, 'coverage_projected_set': float(np.mean(sel('cov_prj'))), 'coverage_raw_ball': float(np.mean(sel('cov_raw'))), 'coverage_joint_box': float(np.mean(sel('cov_box'))), 'mean_radius_projected': float(np.nanmean(sel('q_prj'))), 'mean_radius_raw': float(np.nanmean(sel('q_raw'))), 'raw_ball_admits_violation_frac': {f: float(np.mean(bvv[:, i])) if len(bvv) else None for i, f in enumerate(FAMILIES)}, 'joint_box_admits_violation_frac': {f: float(np.mean(xvv[:, i])) if len(xvv) else None for i, f in enumerate(FAMILIES)}, 'n_box_finite_threshold': int(sum((v is not None for v in xv))), 'forecast_out_of_set_frac': float(np.mean(sel('raw_center_infeas')))}
        out[fname] = blk
        per_date[fname] = {'t': ts, 'cov_prj': np.asarray(rec['cov_prj']), 'cov_raw': np.asarray(rec['cov_raw']), 'cov_box': np.asarray(rec['cov_box']), 'q_prj': np.asarray(rec['q_prj']), 'q_raw': np.asarray(rec['q_raw']), 'lam': np.asarray(rec['lam']), 'wshape': np.asarray(rec['wshape']), 'Shat': Shat[ts], 'proj_c': proj_c[ts], 'pdist': pdist[ts]}
    return (out, per_date)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--panel', required=True)
    ap.add_argument('--scen-glob', required=True)
    ap.add_argument('--outdir', required=True)
    args = ap.parse_args()
    t0 = time.time()
    outdir = Path(args.outdir)
    vsrun = fl.load_vs_run()
    days, S, F, DISC, C = fl.load_panel(args.panel)
    manifest = fl.load_manifest()
    masks = fl.panel_masks(days, manifest)
    vt = masks['vt']
    D = len(days)
    dates_np = np.array(days)
    Yv = C.reshape(D, -1)
    L = fl.realized_losses(days, S, F, DISC, C, vt)
    mu, sd = vsrun.rolling_loc_scale(L)
    spotret_abs, atmchg = fl.lagged_covariates(S, C, vt)
    scen_files = sorted(glob.glob(args.scen_glob))
    if not scen_files:
        raise SystemExit(f'no scenario files match {args.scen_glob}')
    scen_t = {}
    SLp, SLr = ([], [])
    viol_raw, viol_proj, nprojfail = ([], [], [])
    books_cache = {}
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
            books_cache[t] = books
            R = np.asarray(c['R'][row], float)
            St = S[t - 1] * np.exp(R)
            Fm, Dm = rp.scenario_forward_discount(St, S[t - 1], F[t - 1], DISC[t - 1])
            for target, surf in ((SLp, np.asarray(c['surf'][row], float)), (SLr, np.asarray(c['surf_raw'][row], float))):
                sl = np.empty((len(R), K_BOOKS))
                for j, name in enumerate(BOOKS):
                    sl[:, j] = rp.book_loss_bps(books[name], St, Fm, Dm, surf)
                target.append(sl)
            scen_t[t] = len(SLp) - 1
            viol_raw.append(np.asarray(c['raw_viol_frac'][row], float))
            viol_proj.append(np.asarray(c['proj_viol_frac'][row], float))
            nprojfail.append(int(c['n_proj_fail'][row]))
        print(f'[fullcal_run] {Path(f).name}: cum {len(SLp)} scen dates ({time.time() - t0:.0f}s)', flush=True)
    SLp = np.stack(SLp)
    SLr = np.stack(SLr)
    viol_raw = np.stack(viol_raw)
    viol_proj = np.stack(viol_proj)
    from arbcal.aci import rank_threshold as rth
    t_star = sorted(scen_t)[len(scen_t) // 2]
    SL_star = SLp[scen_t[t_star]]
    B_before = [rth(SL_star[:, j], 0.9) for j in range(K_BOOKS)]
    C_pert = C.copy()
    C_pert[t_star] += 0.01
    S_pert = S.copy()
    S_pert[t_star] *= 1.05
    books_p = rp.build_books(S_pert[t_star - 1], F[t_star - 1], DISC[t_star - 1], C_pert[t_star - 1])
    same_books = all((books_p[n].legs == books_cache[t_star][n].legs and books_p[n].v0 == books_cache[t_star][n].v0 for n in BOOKS))
    leakage_ok = bool(same_books and np.allclose(B_before, [rth(SL_star[:, j], 0.9) for j in range(K_BOOKS)]))
    gamma_study, gamma_selected = ({}, {})
    for alpha in CFG.ALPHAS:
        gs = {}
        for kdiv in CFG.GAMMA_SENSITIVITY:
            g = alpha / kdiv
            out, diags = fl.run_scenario_methods(alpha, days, L, mu, sd, SLp, SLr, scen_t, spotret_abs, atmchg, gamma=g, methods=('gproj_rank',))
            r = out['gproj_rank']
            m = r['defined'] & masks['dev']
            fin = m & np.isfinite(r['reserve'])
            gs[f'alpha/{kdiv:g}'] = {'gamma': g, 'dev': vsrun.summarize_series(m, r['exceed'], r['reserve']), 'reserve_sd_dev': float(np.nanstd(r['reserve'][fin])) if fin.any() else float('nan'), 'aci': diags['gproj_rank']}
        key, gsel, audit = select_gamma(gs, alpha)
        gamma_study[f'{alpha:g}'] = {'study': gs, 'selected_rule': key, 'selected_gamma': gsel, 'selection_audit': audit}
        gamma_selected[alpha] = gsel
        print(f'[fullcal_run] gamma selected alpha={alpha}: {key} (gamma={gsel:g})', flush=True)
    results = {'alphas': {}, 'meta': {}, 'gamma_study_dev_only': gamma_study}
    per_date_store = {}
    for alpha in CFG.ALPHAS:
        out, diags = fl.run_scenario_methods(alpha, days, L, mu, sd, SLp, SLr, scen_t, spotret_abs, atmchg, gamma=gamma_selected[alpha])
        common = np.ones(D, bool)
        for m in ('direct', 'bonf', 'cqr', 'g_rank', 'gproj_rank'):
            common &= out[m]['defined']
        blk = {}
        for m, r in out.items():
            blk[m] = {'own': vsrun.summarize_series(r['defined'], r['exceed'], r['reserve']), 'periods_own': fl.summarize_periods(vsrun, masks, r['defined'], r['exceed'], r['reserve']), 'periods_common': fl.summarize_periods(vsrun, masks, common, r['exceed'], r['reserve'])}
            if m != 'direct':
                for period in ('primary', 'y2023'):
                    dd = r['reserve'] - out['direct']['reserve']
                    dd = np.where(common & masks[period], dd, np.nan)
                    dd = dd[np.isfinite(dd)]
                    blk[m][f'delta_vs_direct_{period}'] = vsrun.block_boot(dd) if len(dd) else None
        results['alphas'][f'{alpha:g}'] = {'methods': blk, 'aci_diagnostics': diags, 'aci_boundary': {m: fl.boundary_report(d) for m, d in diags.items()}, 'n_common_primary': int((common & masks['primary']).sum()), 'n_common_y2023': int((common & masks['y2023']).sum()), 'gamma': gamma_selected[alpha]}
        per_date_store[f'a{alpha:g}'] = {m: dict(defined=out[m]['defined'], exceed=out[m]['exceed'], reserve=out[m]['reserve']) for m in out}
        print(f'[fullcal_run] alpha={alpha} done ({time.time() - t0:.0f}s)', flush=True)
    G, h, fam = polyhedron_rows()
    exp_a, exp_a_dates = experiment_a(days, Yv, G, h, fam, masks, vsrun)
    results['experiment_a'] = exp_a
    results['meta'] = {'panel': str(args.panel), 'scen_glob': args.scen_glob, 'n_panel_days': D, 'n_scen_dates': len(scen_t), 'n_valid_transitions': int(vt.sum()), 'first_date': days[0], 'last_date': days[-1], 'books': BOOKS, 'book_tau': CFG.BOOK_TAU, 'gamma_selected': {f'{a:g}': g for a, g in gamma_selected.items()}, 'aci': 'exact-unclipped+boundary', 'loss_units': 'bps of lagged spot S_{t-1}', 'dev_end': CFG.DEV_END, 'periods': {'dev': [fl.PRIMARY_START, fl.DEV_END], 'primary': [fl.PRIMARY_START, fl.Y2023_START], 'y2023': [fl.Y2023_START, '2024-01-01']}, 'leakage_check_pass': leakage_ok, 'scenario_violations': {'raw_frac_mean': {f: float(np.mean(viol_raw[:, i])) for i, f in enumerate(FAMILIES)}, 'proj_frac_mean': {f: float(np.mean(viol_proj[:, i])) for i, f in enumerate(FAMILIES)}, 'n_proj_fail_total': int(np.sum(nprojfail))}, 'gap_aware': True, 'elapsed_s': round(time.time() - t0, 1)}
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / 'fullcal_canonical.json').write_text(json.dumps(results, indent=1, default=float))
    np.savez_compressed(outdir / 'fullcal_arrays.npz', dates=dates_np, L=L, mu=mu, sd=sd, vt=vt, scen_dates=np.array([days[t] for t in sorted(scen_t)]), **{f'expa_{fn}_{k}': v for fn, dd in exp_a_dates.items() for k, v in dd.items()}, **{f'{ak}_{m}_{f}': v for ak, ms in per_date_store.items() for m, fields in ms.items() for f, v in fields.items()})
    print(f'[fullcal_run] wrote {outdir}/fullcal_canonical.json ({time.time() - t0:.0f}s)')
if __name__ == '__main__':
    main()
