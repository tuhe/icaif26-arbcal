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
M_GRID = (1000, 5000, 10000)
M_SELECTED = 10000

def load_losses(globpat, days, vt):
    SL = {}
    for f in sorted(glob.glob(globpat)):
        c = np.load(f, allow_pickle=True)
        assert str(c['extrapolation']) == 'extend'
        for row, t in enumerate(np.asarray(c['t_idx'], int)):
            if str(c['cutoff_dates'][row]) != days[t - 1]:
                raise SystemExit(f'leakage stamp mismatch at {days[t]}')
            if not vt[t]:
                raise SystemExit(f'invalid-transition date in chunk: {days[t]}')
            SL[int(t)] = np.asarray(c['SL'][row], float)
    return SL

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--panel', required=True)
    ap.add_argument('--losses-glob', required=True)
    ap.add_argument('--t-glob', default=None, help='selected Student-t law chunks (full span)')
    ap.add_argument('--t-law-name', default='t5')
    ap.add_argument('--canonical', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    t0 = time.time()
    vsrun = fl.load_vs_run()
    days, S, F, DISC, C = fl.load_panel(args.panel)
    manifest = fl.load_manifest()
    masks = fl.panel_masks(days, manifest)
    vt = masks['vt']
    D = len(days)
    dates_np = np.array(days)
    gsel = {float(a): g for a, g in json.loads(Path(args.canonical).read_text())['meta']['gamma_selected'].items()}
    SL_by_t = load_losses(args.losses_glob, days, vt)
    M_have = next(iter(SL_by_t.values())).shape[0]
    assert M_have >= max(M_GRID)
    SL5_by_t = load_losses(args.t_glob, days, vt) if args.t_glob else {}
    print(f'[fullcal_run_m] {len(SL_by_t)} scen dates (M={M_have}), {len(SL5_by_t)} t-law dates ({time.time() - t0:.0f}s)', flush=True)
    L = fl.realized_losses(days, S, F, DISC, C, vt)
    mu, sd = vsrun.rolling_loc_scale(L)
    spotret_abs, atmchg = fl.lagged_covariates(S, C, vt)

    def seg_summary(r, segmask, direct=None):
        m = r['defined'] & segmask
        n = int(m.sum())
        o = {'n_dates': n, 'fwe': float(np.nanmean(r['exceed'][m])) if n else None, 'n_exceed': int(np.nansum(r['exceed'][m])), 'quote_availability': float(np.mean(r['avail'][m])) if 'avail' in r and n else None}
        res = r['reserve']
        fin = m & np.isfinite(res)
        o['mean_finite_reserve'] = float(np.mean(res[fin])) if fin.any() else None
        o['n_finite'] = int(fin.sum())
        if direct is not None:
            both = m & np.isfinite(res) & np.isfinite(direct['reserve'])
            o['n_paired_finite'] = int(both.sum())
            if both.any():
                o['paired_mean_reserve_method'] = float(np.mean(res[both]))
                o['paired_mean_reserve_direct'] = float(np.mean(direct['reserve'][both]))
                dd = (res - direct['reserve'])[both]
                o['paired_finite_delta_vs_direct'] = vsrun.block_boot(dd)
                o['delta_blocklen_sens'] = {str(b): vsrun.block_boot(dd, block=b) for b in CFG.BLOCK_LEN_SENSITIVITY}
        return o
    results = {'meta': {'M_selected': str(M_SELECTED), 'm_grid': M_GRID, 'gamma_selected': {f'{a:g}': gsel[a] for a in CFG.ALPHAS}, 't_law': args.t_law_name, 'n_scen_dates': len(SL_by_t)}, 'alphas': {}}
    per_date = {}
    for alpha in CFG.ALPHAS:
        out_sc, diag_sc = fl.run_scenario_methods(alpha, days, L, mu, sd, None, None, {}, spotret_abs, atmchg, gamma=gsel[alpha], methods=('direct', 'bonf', 'cqr'))
        mm = {}
        rM = {}
        for M in M_GRID:
            r = fl.run_rank_method(alpha, D, L, mu, sd, SL_by_t, M, gamma=gsel[alpha])
            rM[M] = r
        r_sel = rM[M_SELECTED]
        common = r_sel['defined'] & out_sc['direct']['defined'] & out_sc['cqr']['defined'] & out_sc['bonf']['defined']
        r5 = None
        if SL5_by_t:
            r5 = fl.run_rank_method(alpha, D, L, mu, sd, SL5_by_t, M_SELECTED, gamma=gsel[alpha])
        blk = {'n_common_primary': int((common & masks['primary']).sum()), 'n_common_y2023': int((common & masks['y2023']).sum()), 'methods': {}, 'm_grid': {}, 'aci_boundary': {}}
        for period in ('primary', 'y2023'):
            seg = common & masks[period]
            pm = {}
            pm['rank'] = seg_summary(r_sel, seg, direct=out_sc['direct'])
            for mname in ('direct', 'bonf', 'cqr'):
                rr = dict(out_sc[mname])
                rr.setdefault('avail', np.isfinite(out_sc[mname]['reserve']))
                pm[mname] = seg_summary(rr, seg, direct=out_sc['direct'] if mname != 'direct' else None)
            if r5 is not None:
                c5 = seg & r5['defined']
                pm[args.t_law_name] = seg_summary(r5, c5, direct=out_sc['direct'])
            blk['methods'][period] = pm
        for M in M_GRID:
            blk['m_grid'][str(M)] = {period: seg_summary(rM[M], rM[M]['defined'] & common & masks[period]) for period in ('primary', 'y2023')}
        blk['aci_boundary']['rank'] = fl.boundary_report(r_sel['aci'])
        if r5 is not None:
            blk['aci_boundary'][args.t_law_name] = fl.boundary_report(r5['aci'])
        for mname, d in diag_sc.items():
            blk['aci_boundary'][mname] = fl.boundary_report(d)
        results['alphas'][f'{alpha:g}'] = blk
        per_date[f'a{alpha:g}'] = {'rank': {k: rM[M_SELECTED][k] for k in ('defined', 'exceed', 'reserve', 'avail', 'B_bps')}, **{mname: dict(defined=out_sc[mname]['defined'], exceed=out_sc[mname]['exceed'], reserve=out_sc[mname]['reserve']) for mname in ('direct', 'bonf', 'cqr')}, 'common': common}
        print(f'[fullcal_run_m] alpha={alpha} done ({time.time() - t0:.0f}s)', flush=True)
    outp = Path(args.out)
    outp.write_text(json.dumps(results, indent=1, default=float))
    np.savez_compressed(str(outp).replace('.json', '_arrays.npz'), dates=dates_np, vt=vt, **{f'{ak}_{m}_{f}': v for ak, ms in per_date.items() for m, fields in ms.items() for f, v in (fields.items() if isinstance(fields, dict) else [])})
    print(f'[fullcal_run_m] wrote {outp} ({time.time() - t0:.0f}s)')
if __name__ == '__main__':
    main()
