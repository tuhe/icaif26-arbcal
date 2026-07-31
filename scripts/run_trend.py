from __future__ import annotations
import argparse
import json
import time
import warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings('ignore')
import lib_fullcal as fl
from arbcal import config as CFG
from arbcal.polyhedron import polyhedron_rows, PolyProjector, ball_admits_violation
ALPHA = 0.1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--panel', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    t0 = time.time()
    days, S, F, DISC, C = fl.load_panel(args.panel)
    manifest = fl.load_manifest()
    masks = fl.panel_masks(days, manifest)
    vt2 = masks['vt2']
    D = len(days)
    Yv = C.reshape(D, -1)
    G, h, fam = polyhedron_rows()
    proj = PolyProjector(G, h, family=fam)
    rec = dict(t=[], outside=[], pdist=[], s_raw=[], s_prj=[], rmse_raw=[], rmse_prj=[])
    for t in range(2, D):
        if not vt2[t]:
            continue
        shat = 2.0 * Yv[t - 1] - Yv[t - 2]
        zt, ok = proj.project(shat)
        if zt is None:
            continue
        pd_ = float(np.linalg.norm(shat - zt))
        rec['t'].append(t)
        rec['outside'].append(float(pd_ > 1e-07))
        rec['pdist'].append(pd_)
        rec['s_raw'].append(float(np.linalg.norm(Yv[t] - shat)))
        rec['s_prj'].append(float(np.linalg.norm(Yv[t] - zt)))
        rec['rmse_raw'].append(float(np.sqrt(np.mean((Yv[t] - shat) ** 2))))
        rec['rmse_prj'].append(float(np.sqrt(np.mean((Yv[t] - zt) ** 2))))
    n = len(rec['t'])
    ts = np.asarray(rec['t'], int)
    s_raw = np.asarray(rec['s_raw'])
    s_prj = np.asarray(rec['s_prj'])
    pdist = np.asarray(rec['pdist'])
    outside = np.asarray(rec['outside'])
    W, WU = (CFG.ACI_WINDOW, CFG.WARMUP)
    scored_i, cov_raw, cov_prj, q_raws, q_prjs, ball_bad = ([], [], [], [], [], [])
    for i in range(n):
        if i < WU:
            continue
        w0 = max(0, i - W)
        lvl = min(1.0, (1 - ALPHA) * (1 + 1.0 / (i - w0)))
        qr = float(np.quantile(s_raw[w0:i], lvl, method='higher'))
        qp = float(np.quantile(s_prj[w0:i], lvl, method='higher'))
        scored_i.append(i)
        cov_raw.append(float(s_raw[i] <= qr))
        cov_prj.append(float(s_prj[i] <= qp))
        q_raws.append(qr)
        q_prjs.append(qp)
        t = rec['t'][i]
        shat = 2.0 * Yv[t - 1] - Yv[t - 2]
        bv = ball_admits_violation(shat, qr, G, h, fam)
        ball_bad.append(float(any(bv.values())))
    scored_t = ts[np.asarray(scored_i, int)]
    out = {'forecast': '2*Y_{t-1} - Y_{t-2} (lambda=1, predeclared)', 'gap_aware': True, 'n_dates': n, 'n_scored_total': len(cov_raw), 'frac_outside_A': float(outside.mean()), 'proj_distance': {'median': float(np.median(pdist[outside > 0])) if outside.any() else 0.0, 'p95': float(np.percentile(pdist[outside > 0], 95)) if outside.any() else 0.0}, 'common_level': 1 - ALPHA, 'periods': {}, 'elapsed_s': None}
    cov_raw = np.asarray(cov_raw)
    cov_prj = np.asarray(cov_prj)
    q_raws = np.asarray(q_raws)
    q_prjs = np.asarray(q_prjs)
    ball_bad = np.asarray(ball_bad)
    rmse_raw = np.asarray(rec['rmse_raw'])[np.asarray(scored_i, int)]
    rmse_prj = np.asarray(rec['rmse_prj'])[np.asarray(scored_i, int)]
    for period in ('dev', 'primary', 'y2023'):
        pm = masks[period][scored_t]
        if not pm.any():
            out['periods'][period] = {'n_scored': 0}
            continue
        out['periods'][period] = {'n_scored': int(pm.sum()), 'coverage_raw_ball': float(np.mean(cov_raw[pm])), 'coverage_projected_set': float(np.mean(cov_prj[pm])), 'mean_q_raw': float(np.mean(q_raws[pm])), 'mean_q_proj': float(np.mean(q_prjs[pm])), 'radius_ratio_proj_over_raw': float(np.mean(q_prjs[pm]) / np.mean(q_raws[pm])), 'raw_ball_admits_violator_frac': float(np.mean(ball_bad[pm])), 'projected_set_admits_violator': 0.0, 'rmse_raw_median': float(np.median(rmse_raw[pm])), 'rmse_proj_median': float(np.median(rmse_prj[pm]))}
    out['elapsed_s'] = round(time.time() - t0, 1)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ('n_dates', 'n_scored_total', 'frac_outside_A', 'periods')}, indent=1))
if __name__ == '__main__':
    main()
