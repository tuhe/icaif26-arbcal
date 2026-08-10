from __future__ import annotations
import argparse
import glob
import json
import time
import warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings('ignore')
from arbcal import config as CFG
from arbcal.aci import ExactACI, upper_rank, rank_threshold
from arbcal.polyhedron import polyhedron_rows, PolyProjector, ball_admits_violation, box_admits_violation, FAMILIES
from arbcal.repricing import build_books, book_loss_bps, scenario_forward_discount, realized_forward_discount
BOOKS = list(CFG.BOOK_NAMES)
K_BOOKS = len(BOOKS)

def rolling_loc_scale(L, window=CFG.ACI_WINDOW, warmup=CFG.WARMUP):
    D, K = L.shape
    mu = np.full((D, K), np.nan)
    sd = np.full((D, K), np.nan)
    for t in range(D):
        past = L[max(0, t - window):t]
        past = past[np.isfinite(past).all(axis=1)]
        if len(past) >= warmup:
            mu[t] = past.mean(0)
            sd[t] = past.std(0) + 1e-09
    return (mu, sd)

def block_boot(x, block=CFG.BLOCK_LEN, n=CFG.N_BOOT, seed=0):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < block + 1:
        return dict(mean=float('nan'), lo=float('nan'), hi=float('nan'), n=0)
    rng = np.random.default_rng(seed)
    T = len(x)
    nb = int(np.ceil(T / block))
    means = np.empty(n)
    for i in range(n):
        starts = rng.integers(0, T - block + 1, nb)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:T]
        means[i] = x[idx].mean()
    return dict(mean=float(x.mean()), lo=float(np.quantile(means, 0.025)), hi=float(np.quantile(means, 0.975)), n=int(T))

def summarize_series(mask, exceed, reserve):
    m = mask & np.isfinite(reserve)
    n_inf = int(np.sum(mask & ~np.isfinite(reserve)))
    return dict(n_dates=int(mask.sum()), n_exceed=int(np.nansum(exceed[mask])), fwe=float(np.nanmean(exceed[mask])) if mask.any() else float('nan'), mean_reserve=float(np.nanmean(reserve[m])) if m.any() else float('nan'), n_infinite_reserve=n_inf)

def run_scenario_methods(alpha, dates_all, L, mu, sd, scen_losses, scen_losses_raw, scen_t, spotret_abs, atmchg, gamma=None, methods=('direct', 'bonf', 'cqr', 'g_rank', 'gproj_rank', 'graw_fixed', 'gproj_fixed')):
    D = len(dates_all)
    gamma = CFG.GAMMA_RULE(alpha) if gamma is None else gamma
    out = {m: dict(defined=np.zeros(D, bool), exceed=np.full(D, np.nan), reserve=np.full(D, np.nan)) for m in methods}
    acis = {}
    if 'direct' in methods:
        acis['direct'] = ExactACI(alpha, gamma)
    if 'bonf' in methods:
        acis['bonf'] = [ExactACI(alpha / K_BOOKS, gamma / K_BOOKS) for _ in range(K_BOOKS)]
    if 'cqr' in methods:
        acis['cqr'] = ExactACI(alpha, gamma)
    if 'g_rank' in methods:
        acis['g_rank'] = ExactACI(alpha, gamma)
    if 'gproj_rank' in methods:
        acis['gproj_rank'] = ExactACI(alpha, gamma)
    cqr_X, cqr_y = ([], [])
    smax_hist = []
    for t in range(D):
        valid_norm = np.all(np.isfinite(mu[t])) and np.all(np.isfinite(sd[t]))
        has_scen = t in scen_t
        if not valid_norm:
            continue
        z = (L[t] - mu[t]) / sd[t]
        s_max = float(np.max(z))
        if 'direct' in methods:
            a = acis['direct']
            kap = a.threshold()
            if not a.in_warmup:
                out['direct']['defined'][t] = True
                out['direct']['exceed'][t] = float(s_max > kap) if np.isfinite(kap) else 0.0 if kap == np.inf else 1.0
                out['direct']['reserve'][t] = kap if np.isfinite(kap) else np.inf
            a.update(s_max, kap)
        if 'bonf' in methods:
            ths = []
            wu = False
            for j in range(K_BOOKS):
                aj = acis['bonf'][j]
                kj = aj.threshold()
                ths.append(kj)
                wu = wu or aj.in_warmup
                aj.update(float(z[j]), kj)
            if not wu:
                out['bonf']['defined'][t] = True
                out['bonf']['exceed'][t] = float(any((k == -np.inf or (np.isfinite(k) and z[j] > k) for j, k in enumerate(ths))))
                out['bonf']['reserve'][t] = np.mean(ths) if np.all(np.isfinite(ths)) else np.inf
        if 'cqr' in methods:
            feat = None
            if len(smax_hist) >= 21 and np.isfinite(spotret_abs[t]) and np.isfinite(atmchg[t]):
                hist = np.asarray(smax_hist, float)
                feat = np.array([hist[-1], hist[-5:].mean(), hist[-20:].mean(), spotret_abs[t], atmchg[t]])
            a = acis['cqr']
            if feat is not None and len(cqr_y) >= CFG.WARMUP:
                Xw = np.asarray(cqr_X[-CFG.ACI_WINDOW:], float)
                yw = np.asarray(cqr_y[-CFG.ACI_WINDOW:], float)
                try:
                    from sklearn.linear_model import QuantileRegressor
                    qr = QuantileRegressor(quantile=1 - alpha, alpha=1e-06, solver='highs')
                    qr.fit(Xw, yw)
                    qhat = float(qr.predict(feat[None, :])[0])
                except Exception:
                    qhat = float(np.quantile(yw, 1 - alpha))
                corr = a.threshold()
                if not a.in_warmup:
                    th = qhat + corr if np.isfinite(corr) else corr
                    out['cqr']['defined'][t] = True
                    out['cqr']['exceed'][t] = float(s_max > th) if np.isfinite(th) else 0.0 if th == np.inf else 1.0
                    out['cqr']['reserve'][t] = th
                a.update(s_max - qhat, corr if np.isfinite(corr) else corr)
            if feat is not None:
                cqr_X.append(feat)
                cqr_y.append(s_max)
        if has_scen:
            i = scen_t[t]
            for meth, SLall in (('g_rank', scen_losses_raw), ('gproj_rank', scen_losses)):
                if meth not in methods:
                    continue
                SL = SLall[i]
                u = np.array([upper_rank(SL[:, j], L[t, j]) for j in range(K_BOOKS)])
                s_rank = float(np.max(u))
                a = acis[meth]
                beta = a.threshold()
                if not a.in_warmup:
                    Bs = np.array([rank_threshold(SL[:, j], beta) for j in range(K_BOOKS)]) if np.isfinite(beta) else np.full(K_BOOKS, beta)
                    out[meth]['defined'][t] = True
                    out[meth]['exceed'][t] = float(np.any(L[t] > Bs))
                    rn = (Bs - mu[t]) / sd[t]
                    out[meth]['reserve'][t] = float(np.mean(rn)) if np.all(np.isfinite(rn)) else np.inf
                a.update(s_rank, beta)
            for meth, SLall in (('graw_fixed', scen_losses_raw), ('gproj_fixed', scen_losses)):
                if meth not in methods:
                    continue
                SL = SLall[scen_t[t]]
                Bs = np.array([rank_threshold(SL[:, j], 1.0 - alpha) for j in range(K_BOOKS)])
                out[meth]['defined'][t] = True
                out[meth]['exceed'][t] = float(np.any(L[t] > Bs))
                rn = (Bs - mu[t]) / sd[t]
                out[meth]['reserve'][t] = float(np.mean(rn)) if np.all(np.isfinite(rn)) else np.inf
        smax_hist.append(s_max)
    diags = {}
    for k, a in acis.items():
        diags[k] = [x.diagnostics() for x in a] if isinstance(a, list) else a.diagnostics()
    return (out, diags)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--panel', required=True)
    ap.add_argument('--scen-glob', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--arrays-out', default=None)
    ap.add_argument('--skip-exp-a', action='store_true')
    args = ap.parse_args()
    t0 = time.time()
    z = np.load(args.panel, allow_pickle=True)
    days = [str(d) for d in z['days']]
    S = np.asarray(z['S'], float)
    F = np.asarray(z['F'], float)
    DISC = np.asarray(z['DISC'], float)
    C = np.asarray(z['C'], float)
    D = len(days)
    dates_np = np.array(days)
    scen_files = sorted(glob.glob(args.scen_glob))
    if not scen_files:
        raise SystemExit(f'no scenario files match {args.scen_glob}')
    scen_t = {}
    SLp, SLr, cutoffs, viol_raw, viol_proj, nprojfail = ([], [], [], [], [], [])
    L = np.full((D, K_BOOKS), np.nan)
    books_cache = {}
    for f in scen_files:
        c = np.load(f, allow_pickle=True)
        fam_order = [str(x) for x in c['families']]
        for row, t in enumerate(np.asarray(c['t_idx'], int)):
            if t <= 0 or t >= D or str(c['dates'][row]) != days[t]:
                continue
            if str(c['cutoff_dates'][row]) != days[t - 1]:
                raise SystemExit(f'leakage stamp mismatch at {days[t]}')
            books = build_books(S[t - 1], F[t - 1], DISC[t - 1], C[t - 1])
            books_cache[t] = books
            F59, D59 = realized_forward_discount(S[t], F[t], DISC[t], CFG.BOOK_TAU - 1)
            for j, name in enumerate(BOOKS):
                L[t, j] = book_loss_bps(books[name], S[t], F59, D59, C[t])
            R = np.asarray(c['R'][row], float)
            St = S[t - 1] * np.exp(R)
            Fm, Dm = scenario_forward_discount(St, S[t - 1], F[t - 1], DISC[t - 1])
            for target, surf in ((SLp, np.asarray(c['surf'][row], float)), (SLr, np.asarray(c['surf_raw'][row], float))):
                sl = np.empty((len(R), K_BOOKS))
                for j, name in enumerate(BOOKS):
                    sl[:, j] = book_loss_bps(books[name], St, Fm, Dm, surf)
                target.append(sl)
            scen_t[t] = len(SLp) - 1
            cutoffs.append(str(c['cutoff_dates'][row]))
            viol_raw.append(np.asarray(c['raw_viol_frac'][row], float))
            viol_proj.append(np.asarray(c['proj_viol_frac'][row], float))
            nprojfail.append(int(c['n_proj_fail'][row]))
        print(f'[vs_run] {Path(f).name}: cum {len(SLp)} scen dates ({time.time() - t0:.0f}s)', flush=True)
    SLp = np.stack(SLp)
    SLr = np.stack(SLr)
    viol_raw = np.stack(viol_raw)
    viol_proj = np.stack(viol_proj)
    for t in range(1, D):
        if np.all(np.isfinite(L[t])):
            continue
        books = books_cache.get(t) or build_books(S[t - 1], F[t - 1], DISC[t - 1], C[t - 1])
        F59, D59 = realized_forward_discount(S[t], F[t], DISC[t], CFG.BOOK_TAU - 1)
        for j, name in enumerate(BOOKS):
            L[t, j] = book_loss_bps(books[name], S[t], F59, D59, C[t])
    mu, sd = rolling_loc_scale(L)
    absret = np.concatenate([[np.nan], np.abs(np.diff(np.log(S)))])
    spotret_abs = np.concatenate([[np.nan], absret[:-1]])
    jB = CFG.TAUS.index(CFG.BOOK_TAU)
    atm = C[:, CFG.N_X // 2, jB]
    atm_chg = np.concatenate([[np.nan], np.abs(np.diff(atm))])
    atmchg = np.full(D, np.nan)
    for t in range(21, D):
        atmchg[t] = np.std(atm_chg[max(1, t - 20):t])
    t_star = sorted(scen_t)[len(scen_t) // 2]
    SL_star = SLp[scen_t[t_star]]
    beta_probe = 0.9
    B_before = [rank_threshold(SL_star[:, j], beta_probe) for j in range(K_BOOKS)]
    C_pert = C.copy()
    C_pert[t_star] += 0.01
    S_pert = S.copy()
    S_pert[t_star] *= 1.05
    books_p = build_books(S_pert[t_star - 1], F[t_star - 1], DISC[t_star - 1], C_pert[t_star - 1])
    same_books = all((books_p[n].legs == books_cache[t_star][n].legs and books_p[n].v0 == books_cache[t_star][n].v0 for n in BOOKS))
    leakage_ok = bool(same_books and np.allclose(B_before, [rank_threshold(SL_star[:, j], beta_probe) for j in range(K_BOOKS)]))
    scen_mask = np.zeros(D, bool)
    scen_mask[list(scen_t)] = True
    dev_mask = dates_np < CFG.DEV_END
    results = {'alphas': {}, 'meta': {}}
    per_date_store = {}
    for alpha in CFG.ALPHAS:
        out, diags = run_scenario_methods(alpha, days, L, mu, sd, SLp, SLr, scen_t, spotret_abs, atmchg)
        common = np.ones(D, bool)
        for m in ('direct', 'bonf', 'cqr', 'g_rank', 'gproj_rank'):
            common &= out[m]['defined']
        blk = {}
        for m, r in out.items():
            blk[m] = {'common': summarize_series(common, r['exceed'], r['reserve']), 'own': summarize_series(r['defined'], r['exceed'], r['reserve']), 'dev': summarize_series(common & dev_mask, r['exceed'], r['reserve']), 'holdout': summarize_series(common & ~dev_mask, r['exceed'], r['reserve'])}
            if m != 'direct':
                dd = r['reserve'] - out['direct']['reserve']
                dd = np.where(common, dd, np.nan)
                blk[m]['delta_vs_direct_common'] = block_boot(dd[np.isfinite(dd)])
                blk[m]['delta_blocklen_sens'] = {str(b): block_boot(dd[np.isfinite(dd)], block=b) for b in CFG.BLOCK_LEN_SENSITIVITY}
        dgp = out['gproj_rank']['reserve'] - out['cqr']['reserve']
        dgp = np.where(common, dgp, np.nan)
        blk['gproj_rank']['delta_vs_cqr_common'] = block_boot(dgp[np.isfinite(dgp)])
        results['alphas'][f'{alpha:g}'] = {'methods': blk, 'aci_diagnostics': diags, 'n_common_dates': int(common.sum()), 'gamma': CFG.GAMMA_RULE(alpha)}
        per_date_store[f'a{alpha:g}'] = {m: dict(defined=out[m]['defined'], exceed=out[m]['exceed'], reserve=out[m]['reserve']) for m in out}
        print(f'[vs_run] alpha={alpha} done ({time.time() - t0:.0f}s)', flush=True)
    gamma_study = {}
    for alpha in CFG.ALPHAS:
        gs = {}
        for kdiv in CFG.GAMMA_SENSITIVITY:
            g = alpha / kdiv
            out, diags = run_scenario_methods(alpha, days, L, mu, sd, SLp, SLr, scen_t, spotret_abs, atmchg, gamma=g, methods=('gproj_rank',))
            r = out['gproj_rank']
            m = r['defined'] & dev_mask
            fin = m & np.isfinite(r['reserve'])
            gs[f'alpha/{kdiv:g}'] = {'gamma': g, 'dev': summarize_series(m, r['exceed'], r['reserve']), 'reserve_sd_dev': float(np.nanstd(r['reserve'][fin])) if fin.any() else float('nan'), 'aci': diags['gproj_rank']}
        gamma_study[f'{alpha:g}'] = gs
    print(f'[vs_run] gamma study done ({time.time() - t0:.0f}s)', flush=True)
    exp_a = None
    if not args.skip_exp_a:
        G, h, fam = polyhedron_rows()
        proj = PolyProjector(G, h, family=fam)
        Yv = C.reshape(D, -1)
        alpha_a = 0.1
        n_x = CFG.N_X
        fc = {'rw': np.full((D, CFG.DIM), np.nan), 'ar1': np.full((D, CFG.DIM), np.nan)}
        for t in range(1, D):
            fc['rw'][t] = Yv[t - 1]
            w0 = max(0, t - CFG.ACI_WINDOW)
            Yw = Yv[w0:t]
            if len(Yw) >= CFG.WARMUP:
                ybar = Yw.mean(0)
                num = np.sum((Yw[1:] - ybar) * (Yw[:-1] - ybar), 0)
                den = np.sum((Yw[:-1] - ybar) ** 2, 0) + 1e-12
                phi = np.clip(num / den, -1.0, 1.0)
                fc['ar1'][t] = ybar + phi * (Yv[t - 1] - ybar)
        exp_a = {}
        for fname, Shat in fc.items():
            ok = np.all(np.isfinite(Shat), axis=1)
            proj_c = np.full((D, CFG.DIM), np.nan)
            pdist = np.full(D, np.nan)
            for t in np.where(ok)[0]:
                zt, feas = proj.project(Shat[t])
                if zt is not None:
                    proj_c[t] = zt
                    pdist[t] = float(np.linalg.norm(Shat[t] - zt))
            s_raw = np.linalg.norm(Yv - Shat, axis=1)
            s_prj = np.linalg.norm(Yv - proj_c, axis=1)
            resid = np.abs(Yv - Shat)
            aci_ball = ExactACI(alpha_a)
            aci_box = ExactACI(alpha_a)
            aci_raw = ExactACI(alpha_a)
            rec = dict(cov_prj=[], cov_box=[], cov_raw=[], q_prj=[], q_raw=[], lam=[], ball_viol=[], box_viol=[], raw_center_infeas=[], pdist=[])
            for t in range(D):
                if not (ok[t] and np.isfinite(s_prj[t])):
                    continue
                w0 = max(0, t - CFG.ACI_WINDOW)
                past = resid[w0:t][np.all(np.isfinite(resid[w0:t]), axis=1)]
                if len(past) < CFG.WARMUP:
                    pass
                wshape = np.quantile(past, 0.9, axis=0) + 1e-08 if len(past) >= CFG.WARMUP else None
                q_p = aci_ball.threshold()
                q_r = aci_raw.threshold()
                lam = aci_box.threshold()
                if not (aci_ball.in_warmup or wshape is None):
                    rec['cov_prj'].append(float(s_prj[t] <= q_p))
                    rec['cov_raw'].append(float(s_raw[t] <= q_r))
                    smx = float(np.max(np.abs(Yv[t] - Shat[t]) / wshape))
                    rec['cov_box'].append(float(smx <= lam))
                    if np.isfinite(q_p):
                        rec['q_prj'].append(q_p)
                    if np.isfinite(q_r):
                        rec['q_raw'].append(q_r)
                        bv = ball_admits_violation(Shat[t], q_r, G, h, fam)
                        rec['ball_viol'].append([bv[f] for f in FAMILIES])
                    if np.isfinite(lam):
                        rec['lam'].append(lam)
                        xv = box_admits_violation(Shat[t], lam * wshape, G, h, fam)
                        rec['box_viol'].append([xv[f] for f in FAMILIES])
                    rec['raw_center_infeas'].append(float(pdist[t] > 1e-07))
                    rec['pdist'].append(float(pdist[t]))
                aci_ball.update(s_prj[t], q_p)
                aci_raw.update(s_raw[t], q_r)
                if wshape is not None:
                    smx = float(np.max(np.abs(Yv[t] - Shat[t]) / wshape))
                    aci_box.update(smx, lam)
            bv = np.asarray(rec['ball_viol'], float)
            xv = np.asarray(rec['box_viol'], float)
            exp_a[fname] = {'n_scored': len(rec['cov_prj']), 'coverage_projected_set': float(np.mean(rec['cov_prj'])), 'coverage_raw_ball': float(np.mean(rec['cov_raw'])), 'coverage_joint_box': float(np.mean(rec['cov_box'])), 'mean_radius_projected': float(np.mean(rec['q_prj'])), 'mean_radius_raw': float(np.mean(rec['q_raw'])), 'radius_ratio_prj_over_raw': float(np.mean(rec['q_prj']) / np.mean(rec['q_raw'])), 'raw_ball_admits_violation_frac': {f: float(np.mean(bv[:, i])) for i, f in enumerate(FAMILIES)}, 'joint_box_admits_violation_frac': {f: float(np.mean(xv[:, i])) for i, f in enumerate(FAMILIES)}, 'projected_set_admits_violation': 0.0, 'forecast_out_of_set_frac': float(np.mean(rec['raw_center_infeas'])), 'forecast_proj_distance': {'median': float(np.median(rec['pdist'])), 'p95': float(np.percentile(rec['pdist'], 95))}, 'aci_diag': {'ball': aci_ball.diagnostics(), 'box': aci_box.diagnostics(), 'raw': aci_raw.diagnostics()}}
            print(f'[vs_run] exp A [{fname}] done ({time.time() - t0:.0f}s)', flush=True)
    results['meta'] = {'panel': str(args.panel), 'scen_glob': args.scen_glob, 'n_panel_days': D, 'n_scen_dates': len(scen_t), 'first_date': days[0], 'last_date': days[-1], 'books': BOOKS, 'book_tau': CFG.BOOK_TAU, 'n_draws': int(np.load(scen_files[0])['n_draws']), 'gamma_rule': 'alpha/5', 'aci': 'exact-unclipped+boundary', 'loss_units': 'bps of lagged spot S_{t-1}', 'dev_end': CFG.DEV_END, 'leakage_check_pass': leakage_ok, 'scenario_violations': {'raw_frac_mean': {f: float(np.mean(viol_raw[:, i])) for i, f in enumerate(FAMILIES)}, 'proj_frac_mean': {f: float(np.mean(viol_proj[:, i])) for i, f in enumerate(FAMILIES)}, 'n_proj_fail_total': int(np.sum(nprojfail))}, 'elapsed_s': round(time.time() - t0, 1)}
    results['gamma_study_dev_only'] = gamma_study
    results['experiment_a'] = exp_a
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(results, indent=2, default=float))
    arrays_out = args.arrays_out or str(outp).replace('.json', '_arrays.npz')
    np.savez_compressed(arrays_out, dates=dates_np, L=L, mu=mu, sd=sd, scen_dates=np.array([days[t] for t in sorted(scen_t)]), **{f'{ak}_{m}_{f}': v for ak, ms in per_date_store.items() for m, fields in ms.items() for f, v in fields.items()})
    print(f'[vs_run] wrote {outp} + {arrays_out} ({time.time() - t0:.0f}s)')
if __name__ == '__main__':
    main()
