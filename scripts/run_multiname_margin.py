from __future__ import annotations
import argparse
import importlib.util
import json
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')
from arbcal.aci import ExactACI
from arbcal.panel import fit_day, _ridge_matrix
from arbcal.polyhedron import polyhedron_rows
_spec = importlib.util.spec_from_file_location('vs_multiname', Path(__file__).parent / 'run_multiname.py')
_mn = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mn)
day_frame, WINDOW, WARMUP, ALPHA = (_mn.day_frame, _mn.WINDOW, _mn.WARMUP, _mn.ALPHA)

def run_name_margin(panel, Gd, h, gl1, gn2, absG):
    C = np.stack(panel['C'])
    D = len(C)
    Yv = C.reshape(D, -1)
    resid = np.abs(Yv[1:] - Yv[:-1])
    aci_ball = ExactACI(ALPHA, window=WINDOW, warmup=WARMUP)
    aci_box = ExactACI(ALPHA, window=WINDOW, warmup=WARMUP)
    m_ball, m_box = ([], [])
    for t in range(1, D):
        Shat = Yv[t - 1]
        s_raw = float(np.linalg.norm(Yv[t] - Shat))
        w0 = max(0, t - 1 - WINDOW)
        past = resid[w0:t - 1]
        wshape = np.quantile(past, 0.9, axis=0) + 1e-08 if len(past) >= WARMUP else None
        q_r = aci_ball.threshold()
        lam = aci_box.threshold()
        if wshape is not None and (not aci_ball.in_warmup):
            gS = Gd @ Shat
            if np.isfinite(q_r):
                raw = np.maximum(h - (gS - q_r * gn2), 0.0)
                m_ball.append(float(np.max(10000.0 * raw / gl1)))
            if np.isfinite(lam):
                slack = gS - absG @ (lam * wshape)
                raw = np.maximum(h - slack, 0.0)
                m_box.append(float(np.max(10000.0 * raw / gl1)))
        aci_ball.update(s_raw, q_r)
        if wshape is not None:
            smx = float(np.max(np.abs(Yv[t] - Shat) / wshape))
            aci_box.update(smx, lam)
    return (m_ball, m_box)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ivydb-dir', default='data/raw/ivydb_multiname')
    ap.add_argument('--out', required=True)
    ap.add_argument('--expiry-mode', default='independent')
    args = ap.parse_args()
    t0 = time.time()
    opt = pd.read_parquet(Path(args.ivydb_dir) / 'options.parquet')
    smap = pd.read_parquet(Path(args.ivydb_dir) / 'secid_map.parquet')
    px = pd.read_parquet(Path(args.ivydb_dir) / 'secpx.parquet')
    name_col = next((c for c in smap.columns if smap[c].dtype == object))
    px_val = [c for c in px.columns if px[c].dtype != object and c not in ('secid',)][-1]
    px['date'] = px['date'].astype(str).str[:10]
    opt['date'] = opt['date'].astype(str).str[:10]
    G, h, fam = polyhedron_rows()
    Gd = G.toarray()
    gl1 = np.abs(Gd).sum(axis=1)
    gn2 = np.sqrt((Gd ** 2).sum(axis=1))
    absG = np.abs(Gd)
    ridge = _ridge_matrix()
    per_name = {}
    for _, row in smap.iterrows():
        sid, name = (int(row['secid']), str(row[name_col]))
        o = opt[opt['secid'] == sid]
        if not len(o):
            continue
        pxn = px[px['secid'] == sid] if 'secid' in px.columns else px
        pxmap = dict(zip(pxn['date'], pxn[px_val].astype(float)))
        panel = {'C': []}
        for date, g in o.groupby('date'):
            spot = pxmap.get(date)
            if spot is None or not np.isfinite(spot) or spot <= 0:
                continue
            try:
                fit = fit_day(day_frame(g, spot, date), date, G, h, fam, ridge, min_quotes=5, expiry_mode=args.expiry_mode)
            except Exception:
                fit = None
            if fit is not None:
                panel['C'].append(fit.C)
        if len(panel['C']) < WARMUP + 20:
            print(f"[mnm] {name}: skipped ({len(panel['C'])} dates)", flush=True)
            continue
        m_ball, m_box = run_name_margin(panel, Gd, h, gl1, gn2, absG)
        per_name[name] = {'n_scored_ball': len(m_ball), 'n_scored_box': len(m_box), 'median_margin_ball': float(np.median(m_ball)) if m_ball else None, 'median_margin_box': float(np.median(m_box)) if m_box else None}
        print(f"[mnm] {name}: median ball {per_name[name]['median_margin_ball']:.1f} bp, box {per_name[name]['median_margin_box']:.1f} bp ({time.time() - t0:.0f}s)", flush=True)
    summary = {'n_names': len(per_name), 'mean_of_median_margin_ball': float(np.mean([v['median_margin_ball'] for v in per_name.values()])), 'mean_of_median_margin_box': float(np.mean([v['median_margin_box'] for v in per_name.values()])), 'margin_set': 0.0, 'units': 'normalized_bp', 'window': WINDOW, 'warmup': WARMUP, 'alpha': ALPHA, 'elapsed_s': round(time.time() - t0, 1)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({'summary': summary, 'per_name': per_name}, indent=2))
    print(json.dumps(summary, indent=2))
if __name__ == '__main__':
    main()
