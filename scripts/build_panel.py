from __future__ import annotations
import argparse
import glob
import json
import os
import time
import warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings('ignore')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--diag', default=None)
    ap.add_argument('--months-glob', default='*.txt', help='restrict to a subset of monthly files (smoke tests)')
    args = ap.parse_args()
    from arbcal import config as CFG
    from arbcal.panel import build_panel, save_panel, fit_day, _ridge_matrix
    from arbcal.polyhedron import polyhedron_rows, FAMILIES
    from arbcal.optionsdx import read_month
    t0 = time.time()
    G, h, family = polyhedron_rows()
    ridge = _ridge_matrix()
    fits = []
    files = sorted(glob.glob(os.path.join(args.data_dir, args.months_glob)))
    if not files:
        raise SystemExit(f'no files match {args.months_glob} in {args.data_dir}')
    n_days_seen = 0
    for mfile in files:
        df = read_month(mfile)
        for date in sorted(df['QUOTE_DATE'].astype(str).str.strip().unique()):
            n_days_seen += 1
            day = df[df['QUOTE_DATE'].astype(str).str.strip() == date]
            try:
                out = fit_day(day, date, G, h, family, ridge)
            except Exception as e:
                print(f'[panel] {date} FAILED: {type(e).__name__}: {e}')
                out = None
            if out is not None:
                fits.append(out)
        print(f'[panel] {os.path.basename(mfile)} -> cum {len(fits)}/{n_days_seen} days kept ({time.time() - t0:.0f}s)')
    if not fits:
        raise SystemExit('no days retained')
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    save_panel(fits, outp)
    fb = np.stack([f.parity_fallback for f in fits])
    uv = np.array([[f.unc_viol[k] for k in FAMILIES] for f in fits])
    diag = {'n_days_seen': n_days_seen, 'n_days_kept': len(fits), 'first_date': fits[0].date, 'last_date': fits[-1].date, 'parity_fallback_frac': {'any_interp': float(np.mean(np.any(fb == 1, axis=1))), 'any_carry_free': float(np.mean(np.any(fb == 2, axis=1))), 'per_tau_interp': np.mean(fb == 1, axis=0).tolist(), 'per_tau_carry_free': np.mean(fb == 2, axis=0).tolist()}, 'unconstrained_violation_day_frac': {k: float(np.mean(uv[:, i] > 0)) for i, k in enumerate(FAMILIES)}, 'cleaning_distance_l2': {'median': float(np.median([f.clean_dist_l2 for f in fits])), 'p95': float(np.percentile([f.clean_dist_l2 for f in fits], 95)), 'max': float(np.max([f.clean_dist_l2 for f in fits]))}, 'cleaning_distance_inf': {'median': float(np.median([f.clean_dist_inf for f in fits])), 'p95': float(np.percentile([f.clean_dist_inf for f in fits], 95))}, 'weighted_rmse': {'unconstrained_median': float(np.median([f.rmse_unc for f in fits])), 'constrained_median': float(np.median([f.rmse_con for f in fits]))}, 'within_bid_ask_frac': {'unconstrained_mean': float(np.mean([f.within_ba_unc for f in fits])), 'constrained_mean': float(np.mean([f.within_ba_con for f in fits]))}, 'active_constraints': {'median': float(np.median([f.n_active for f in fits])), 'p95': float(np.percentile([f.n_active for f in fits], 95))}, 'n_quotes_median': float(np.median([f.n_quotes for f in fits])), 'config': {'x_grid': CFG.X_GRID.tolist(), 'taus': list(CFG.TAUS), 'feas_tol': CFG.FEAS_TOL, 'fit_ridge': CFG.FIT_RIDGE}, 'elapsed_s': round(time.time() - t0, 1)}
    if args.diag:
        Path(args.diag).parent.mkdir(parents=True, exist_ok=True)
        Path(args.diag).write_text(json.dumps(diag, indent=2))
    print(json.dumps(diag, indent=2))
    print(f'[panel] wrote {outp} ({len(fits)} days, {time.time() - t0:.0f}s)')
if __name__ == '__main__':
    main()
