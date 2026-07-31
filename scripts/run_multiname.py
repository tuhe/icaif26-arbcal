from __future__ import annotations
import argparse
import json
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')
from arbcal import config as CFG
from arbcal.aci import ExactACI
from arbcal.panel import fit_day, _ridge_matrix
from arbcal.polyhedron import polyhedron_rows, PolyProjector, ball_admits_violation, box_admits_violation, FAMILIES
WINDOW, WARMUP = (104, 40)
ALPHA = 0.1

def day_frame(g, spot, date):
    g = g.copy()
    g['STRIKE'] = g['strike_price'].astype(float) / 1000.0
    g['dte'] = (pd.to_datetime(g['exdate']) - pd.to_datetime(date)).dt.days
    piv = g.pivot_table(index=['exdate', 'STRIKE', 'dte'], columns='cp_flag', values=['best_bid', 'best_offer'], aggfunc='first')
    piv.columns = [f'{a}_{b}' for a, b in piv.columns]
    piv = piv.reset_index()
    out = pd.DataFrame({'EXPIRE_DATE': piv['exdate'].astype(str), 'DTE': piv['dte'].astype(float), 'STRIKE': piv['STRIKE'].astype(float), 'C_BID': piv.get('best_bid_C', np.nan), 'C_ASK': piv.get('best_offer_C', np.nan), 'P_BID': piv.get('best_bid_P', np.nan), 'P_ASK': piv.get('best_offer_P', np.nan)})
    out['UNDERLYING_LAST'] = float(spot)
    return out

def run_name(name, panel, G, h, fam, projector):
    C = np.stack(panel['C'])
    D = len(C)
    Yv = C.reshape(D, -1)
    resid = np.abs(Yv[1:] - Yv[:-1])
    aci_ball = ExactACI(ALPHA, window=WINDOW, warmup=WARMUP)
    aci_box = ExactACI(ALPHA, window=WINDOW, warmup=WARMUP)
    aci_prj = ExactACI(ALPHA, window=WINDOW, warmup=WARMUP)
    cov_ball, cov_box, cov_prj = ([], [], [])
    ball_bad, box_bad = ([], [])
    for t in range(1, D):
        Shat = Yv[t - 1]
        s_raw = float(np.linalg.norm(Yv[t] - Shat))
        w0 = max(0, t - 1 - WINDOW)
        past = resid[w0:t - 1]
        wshape = np.quantile(past, 0.9, axis=0) + 1e-08 if len(past) >= WARMUP else None
        q_r = aci_ball.threshold()
        q_p = aci_prj.threshold()
        lam = aci_box.threshold()
        if wshape is not None and (not aci_ball.in_warmup):
            smx = float(np.max(np.abs(Yv[t] - Shat) / wshape))
            cov_ball.append(float(s_raw <= q_r))
            cov_prj.append(float(s_raw <= q_p))
            cov_box.append(float(smx <= lam))
            if np.isfinite(q_r):
                bv = ball_admits_violation(Shat, q_r, G, h, fam)
                ball_bad.append(float(any(bv.values())))
            if np.isfinite(lam):
                xv = box_admits_violation(Shat, lam * wshape, G, h, fam)
                box_bad.append(float(any(xv.values())))
        aci_ball.update(s_raw, q_r)
        aci_prj.update(s_raw, q_p)
        if wshape is not None:
            smx = float(np.max(np.abs(Yv[t] - Shat) / wshape))
            aci_box.update(smx, lam)
    return {'n_days': D, 'n_scored': len(cov_prj), 'coverage_ball': float(np.mean(cov_ball)) if cov_ball else None, 'coverage_box': float(np.mean(cov_box)) if cov_box else None, 'coverage_set': float(np.mean(cov_prj)) if cov_prj else None, 'ball_admits_frac': float(np.mean(ball_bad)) if ball_bad else None, 'box_admits_frac': float(np.mean(box_bad)) if box_bad else None, 'rmse_unc_median': float(np.median(panel['rmse_unc'])), 'rmse_con_median': float(np.median(panel['rmse_con'])), 'clean_dist_median': float(np.median(panel['clean']))}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ivydb-dir', default='data/raw/ivydb_multiname')
    ap.add_argument('--out', required=True)
    ap.add_argument('--expiry-mode', default='independent', choices=('independent', 'joint'), help='expiry->grid-maturity assignment rule (fix_surface_fit_outcome.md); canonical = independent')
    args = ap.parse_args()
    t0 = time.time()
    opt = pd.read_parquet(Path(args.ivydb_dir) / 'options.parquet')
    smap = pd.read_parquet(Path(args.ivydb_dir) / 'secid_map.parquet')
    px = pd.read_parquet(Path(args.ivydb_dir) / 'secpx.parquet')
    print('[mn] secid_map cols:', list(smap.columns), 'secpx cols:', list(px.columns), flush=True)
    name_col = next((c for c in smap.columns if smap[c].dtype == object))
    px_val = [c for c in px.columns if px[c].dtype != object and c not in ('secid',)][-1]
    px['date'] = px['date'].astype(str).str[:10]
    opt['date'] = opt['date'].astype(str).str[:10]
    G, h, fam = polyhedron_rows()
    ridge = _ridge_matrix()
    projector = PolyProjector(G, h, family=fam)
    results = {}
    for _, row in smap.iterrows():
        sid, name = (int(row['secid']), str(row[name_col]))
        o = opt[opt['secid'] == sid]
        if not len(o):
            continue
        pxn = px[px['secid'] == sid] if 'secid' in px.columns else px
        pxmap = dict(zip(pxn['date'], pxn[px_val].astype(float)))
        panel = {'C': [], 'rmse_unc': [], 'rmse_con': [], 'clean': []}
        n_seen = 0
        for date, g in o.groupby('date'):
            spot = pxmap.get(date)
            if spot is None or not np.isfinite(spot) or spot <= 0:
                continue
            n_seen += 1
            try:
                fit = fit_day(day_frame(g, spot, date), date, G, h, fam, ridge, min_quotes=5, expiry_mode=args.expiry_mode)
            except Exception:
                fit = None
            if fit is not None:
                panel['C'].append(fit.C)
                panel['rmse_unc'].append(fit.rmse_unc)
                panel['rmse_con'].append(fit.rmse_con)
                panel['clean'].append(fit.clean_dist_l2)
        if len(panel['C']) < WARMUP + 20:
            results[name] = {'n_days': len(panel['C']), 'n_seen': n_seen, 'skipped': 'insufficient retained dates'}
            print(f"[mn] {name}: kept {len(panel['C'])}/{n_seen} -> skipped", flush=True)
            continue
        r = run_name(name, panel, G, h, fam, projector)
        r['n_seen'] = n_seen
        results[name] = r
        print(f"[mn] {name}: kept {r['n_days']}/{n_seen}, scored {r['n_scored']}, cov set {r['coverage_set']:.3f} ({time.time() - t0:.0f}s)", flush=True)
    ok = {k: v for k, v in results.items() if 'skipped' not in v}
    summary = {'n_names': len(ok), 'mean_coverage': {k: float(np.mean([v[f'coverage_{k}'] for v in ok.values()])) for k in ('ball', 'box', 'set')}, 'coverage_set_range': [float(min((v['coverage_set'] for v in ok.values()))), float(max((v['coverage_set'] for v in ok.values())))], 'mean_admits': {'ball': float(np.mean([v['ball_admits_frac'] for v in ok.values()])), 'box': float(np.mean([v['box_admits_frac'] for v in ok.values()])), 'set': 0.0}, 'min_ball_admits': float(min((v['ball_admits_frac'] for v in ok.values()))), 'rmse_increase_median': float(np.median([(v['rmse_con_median'] - v['rmse_unc_median']) / max(v['rmse_unc_median'], 1e-12) for v in ok.values()])), 'window': WINDOW, 'warmup': WARMUP, 'alpha': ALPHA, 'elapsed_s': round(time.time() - t0, 1)}
    Path(args.out).write_text(json.dumps({'summary': summary, 'per_name': results}, indent=2))
    print(json.dumps(summary, indent=2))
if __name__ == '__main__':
    main()
