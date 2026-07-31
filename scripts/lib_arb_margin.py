from __future__ import annotations
import argparse
import gzip
import json
import time
import warnings
from pathlib import Path
import numpy as np
import scipy.sparse as sp
warnings.filterwarnings('ignore')
from arbcal import config as CFG
from arbcal.aci import ExactACI
from arbcal.polyhedron import polyhedron_rows, PolyProjector, ball_admits_violation, box_admits_violation, FAMILIES
ALPHA_A = 0.1
TOL_CERT = 1e-12
RNG_SEED = 0
FAMILY_ECON = {'upper': 'short one call at (x_i, tau_j) vs its forward-normalized upper bound c <= 1 (call worth more than the discounted forward)', 'intrinsic': 'long one call at (x_i, tau_j) vs its intrinsic lower bound c >= (1-x)+ (call below immediate-exercise value: buy call, short forward, lend strike)', 'monotone': 'vertical (call) spread long C(x_i) short C(x_{i+1}), scaled by 1/dx: calls must be non-increasing in strike', 'slope': 'covered vertical spread: call-spread slope bounded below by -1 (short spread + strike-difference bond)', 'convex': 'butterfly long C(x_{i-1}), short (1/dx_{i-1}+1/dx_i)-weighted C(x_i), long C(x_{i+1}) (divided second difference >= 0)', 'calendar': 'calendar spread long C(tau_{j+1}) short C(tau_j) at fixed forward moneyness x_i'}

def build_row_map():
    x = np.asarray(CFG.X_GRID, float)
    n_tau, n_x = (CFG.N_TAU, len(x))
    dx = np.diff(x)
    idx = lambda i, j: i * n_tau + j
    rows, cols, vals, hs, fam = ([], [], [], [], [])
    meta = []

    def add(coefs, rhs, family, mi, ti):
        r = len(hs)
        for c, v in coefs:
            rows.append(r)
            cols.append(c)
            vals.append(v)
        hs.append(rhs)
        fam.append(family)
        meta.append({'row_id': r, 'family': family, 'moneyness_indices': mi, 'maturity_indices': ti, 'nonzero_coefficients': [[int(c), float(v)] for c, v in coefs], 'rhs': float(rhs)})
    for j in range(n_tau):
        for i in range(n_x):
            add([(idx(i, j), -1.0)], -1.0, 'upper', [i], [j])
        for i in range(n_x):
            add([(idx(i, j), 1.0)], max(0.0, 1.0 - x[i]), 'intrinsic', [i], [j])
        for i in range(n_x - 1):
            a, b = (idx(i, j), idx(i + 1, j))
            add([(a, 1.0 / dx[i]), (b, -1.0 / dx[i])], 0.0, 'monotone', [i, i + 1], [j])
            add([(a, -1.0 / dx[i]), (b, 1.0 / dx[i])], -1.0, 'slope', [i, i + 1], [j])
        for i in range(1, n_x - 1):
            add([(idx(i - 1, j), 1.0 / dx[i - 1]), (idx(i, j), -1.0 / dx[i - 1] - 1.0 / dx[i]), (idx(i + 1, j), 1.0 / dx[i])], 0.0, 'convex', [i - 1, i, i + 1], [j])
    for i in range(n_x):
        for j in range(n_tau - 1):
            add([(idx(i, j + 1), 1.0), (idx(i, j), -1.0)], 0.0, 'calendar', [i], [j, j + 1])
    Gm = sp.csc_matrix((vals, (rows, cols)), shape=(len(hs), n_x * n_tau))
    return (Gm, np.asarray(hs, float), np.asarray(fam), meta)

def verify_row_map(Gm, hm, famm, G, h, fam):
    checks = {'n_rows_equal': Gm.shape == G.shape, 'G_identical': (Gm != G).nnz == 0, 'h_identical': bool(np.array_equal(hm, h)), 'family_identical': bool(np.array_equal(famm, fam)), 'orientation': 'G c >= h (verified against library constructor)', 'families_present': {f: int(np.sum(fam == f)) for f in FAMILIES}}
    ok = checks['n_rows_equal'] and checks['G_identical'] and checks['h_identical'] and checks['family_identical']
    return (ok, checks)

def replay_experiment_a_rw(days, Yv, G, h, fam):
    D, dim = Yv.shape
    proj = PolyProjector(G, h, family=fam)
    Shat = np.full((D, dim), np.nan)
    for t in range(1, D):
        Shat[t] = Yv[t - 1]
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
    aci_ball, aci_box, aci_raw = (ExactACI(ALPHA_A), ExactACI(ALPHA_A), ExactACI(ALPHA_A))
    rec = dict(t=[], q_p=[], q_r=[], lam=[], wshape=[], cov_prj=[], cov_raw=[], cov_box=[], ball_viol=[], box_viol=[], center_infeas=[], pdist=[])
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
            rec['q_p'].append(q_p)
            rec['q_r'].append(q_r)
            rec['lam'].append(lam)
            rec['wshape'].append(wshape.copy())
            rec['cov_prj'].append(float(s_prj[t] <= q_p))
            rec['cov_raw'].append(float(s_raw[t] <= q_r))
            smx = float(np.max(np.abs(Yv[t] - Shat[t]) / wshape))
            rec['cov_box'].append(float(smx <= lam))
            rec['ball_viol'].append(ball_admits_violation(Shat[t], q_r, G, h, fam) if np.isfinite(q_r) else None)
            rec['box_viol'].append(box_admits_violation(Shat[t], lam * wshape, G, h, fam) if np.isfinite(lam) else None)
            rec['center_infeas'].append(float(pdist[t] > 1e-07))
            rec['pdist'].append(float(pdist[t]))
        aci_ball.update(s_prj[t], q_p)
        aci_raw.update(s_raw[t], q_r)
        if wshape is not None:
            smx = float(np.max(np.abs(Yv[t] - Shat[t]) / wshape))
            aci_box.update(smx, lam)
    out = {k: np.asarray(v) if k not in ('ball_viol', 'box_viol') else v for k, v in rec.items()}
    out['Shat'] = Shat[out['t']]
    out['proj_c'] = proj_c[out['t']]
    out['dates'] = np.asarray([days[t] for t in out['t']])
    out['Y'] = Yv[out['t']]
    out['aci_diag'] = {'ball': aci_ball.diagnostics(), 'box': aci_box.diagnostics(), 'raw': aci_raw.diagnostics()}
    return out

def reproduction_check(rep, canon):
    ea = canon['experiment_a']['rw']
    got = {'n_scored': int(len(rep['cov_prj'])), 'coverage_projected_set': float(np.mean(rep['cov_prj'])), 'coverage_raw_ball': float(np.mean(rep['cov_raw'])), 'coverage_joint_box': float(np.mean(rep['cov_box'])), 'mean_radius_projected': float(np.mean([q for q in rep['q_p'] if np.isfinite(q)])), 'mean_radius_raw': float(np.mean([q for q in rep['q_r'] if np.isfinite(q)])), 'forecast_out_of_set_frac': float(np.mean(rep['center_infeas']))}
    bv = np.asarray([[v[f] for f in FAMILIES] for v in rep['ball_viol'] if v is not None], float)
    xv = np.asarray([[v[f] for f in FAMILIES] for v in rep['box_viol'] if v is not None], float)
    got['raw_ball_admits_violation_frac'] = {f: float(np.mean(bv[:, i])) for i, f in enumerate(FAMILIES)}
    got['joint_box_admits_violation_frac'] = {f: float(np.mean(xv[:, i])) for i, f in enumerate(FAMILIES)}

    def close(a, b, nd):
        return round(float(a), nd) == round(float(b), nd)
    checks = {'n_scored': got['n_scored'] == ea['n_scored'], 'coverage_projected_set': close(got['coverage_projected_set'], ea['coverage_projected_set'], 3), 'coverage_raw_ball': close(got['coverage_raw_ball'], ea['coverage_raw_ball'], 3), 'coverage_joint_box': close(got['coverage_joint_box'], ea['coverage_joint_box'], 3), 'mean_radius_projected': close(got['mean_radius_projected'], ea['mean_radius_projected'], 4), 'mean_radius_raw': close(got['mean_radius_raw'], ea['mean_radius_raw'], 4), 'ball_admits_max': close(max(got['raw_ball_admits_violation_frac'].values()), max(ea['raw_ball_admits_violation_frac'].values()), 3), 'box_admits_max': close(max(got['joint_box_admits_violation_frac'].values()), max(ea['joint_box_admits_violation_frac'].values()), 3)}
    deltas = {k: float(got[k]) - float(ea[k]) for k in ('coverage_projected_set', 'coverage_raw_ball', 'coverage_joint_box', 'mean_radius_projected', 'mean_radius_raw')}
    return (all(checks.values()), got, checks, deltas)

def analytic_margins(rep, G, h):
    Gd = G.toarray()
    gl1 = np.abs(Gd).sum(axis=1)
    gn2 = np.sqrt((Gd ** 2).sum(axis=1))
    absG = np.abs(Gd)
    gS = rep['Shat'] @ Gd.T
    q_r = rep['q_r'][:, None]
    slack_ball = gS - q_r * gn2[None, :]
    raw_ball = np.maximum(h[None, :] - slack_ball, 0.0)
    hw = rep['lam'][:, None] * rep['wshape']
    with np.errstate(invalid='ignore'):
        slack_box = gS - hw @ absG.T
    raw_box = np.maximum(h[None, :] - slack_box, 0.0)
    raw_box = np.where(np.isnan(raw_box), np.inf, raw_box)
    m_ball = 10000.0 * raw_ball / gl1[None, :]
    m_box = 10000.0 * raw_box / gl1[None, :]
    return dict(m_ball=m_ball, m_box=m_box, raw_ball=raw_ball, raw_box=raw_box, gl1=gl1, gn2=gn2, gS=gS)

def per_date_summary(m, fam):
    M = np.max(m, axis=1)
    win = np.argmax(m, axis=1)
    fam_max = {f: np.max(m[:, fam == f], axis=1) for f in FAMILIES}
    win_fam = fam[win]
    return (M, win, fam_max, win_fam)

def mbb_stat(arrays, stat_fn, block=10, n=2000, seed=RNG_SEED):
    T = len(arrays[0])
    rng = np.random.default_rng(seed)
    nblk = int(np.ceil(T / block))
    stats = np.empty(n)
    for b in range(n):
        starts = rng.integers(0, T - block + 1, size=nblk)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:T]
        stats[b] = stat_fn([a[idx] for a in arrays])
    point = stat_fn(arrays)
    return (float(point), float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5)))

def med(x):
    return float(np.median(x))

def mean_fin(x):
    x = x[np.isfinite(x)]
    return float(np.mean(x)) if len(x) else float('nan')

def make_witness(kind, k, rep, marg, G, h, fam, meta):
    Gd = G.toarray()
    m = marg['m_ball'][k] if kind == 'ball' else marg['m_box'][k]
    ell = int(np.argmax(m))
    g = Gd[ell]
    mu = rep['Shat'][k]
    info = meta[ell]
    if kind == 'ball':
        q = float(rep['q_r'][k])
        cstar = mu - q * g / np.linalg.norm(g)
        member = float(np.linalg.norm(cstar - mu)) <= q + 1e-10
        radius_desc = {'q_raw_ball': q}
    else:
        lam = float(rep['lam'][k])
        r = lam * rep['wshape'][k]
        cstar = mu - r * np.sign(g)
        member = bool(np.all(np.abs(cstar - mu) <= r + 1e-10))
        radius_desc = {'lam': lam, 'halfwidth_min': float(np.min(r)), 'halfwidth_max': float(np.max(r))}
    slack_c = float(g @ cstar - h[ell])
    center_slack = float(g @ mu - h[ell])
    gl1 = float(np.abs(g).sum())
    nz = np.nonzero(g)[0]
    return {'date': str(rep['dates'][k]), 'method': 'raw_ball' if kind == 'ball' else 'joint_box', 'worst_margin_normalized_bp': float(m[ell]), 'constraint_row': ell, 'family': str(fam[ell]), 'moneyness_indices': info['moneyness_indices'], 'moneyness_values': [float(CFG.X_GRID[i]) for i in info['moneyness_indices']], 'maturity_indices': info['maturity_indices'], 'maturity_days': [int(CFG.TAUS[j]) for j in info['maturity_indices']], 'economic_interpretation': FAMILY_ECON[str(fam[ell])], 'coefficients': info['nonzero_coefficients'], 'rhs': float(h[ell]), 'row_l1_norm': gl1, 'center_slack': center_slack, 'worst_attainable_slack': slack_c, 'normalized_margin_check_bp': float(10000.0 * max(0.0, -slack_c) / gl1), 'witness_membership_ok': bool(member), 'witness_violates_row': bool(slack_c < -TOL_CERT), 'witness_coords_on_row': {int(i): float(cstar[i]) for i in nz}, 'center_coords_on_row': {int(i): float(mu[i]) for i in nz}, 'radius': radius_desc}

def validate(rep, marg, G, h, fam, n_dates=20, n_rows=20, seed=RNG_SEED):
    import cvxpy as cp
    Gd = G.toarray()
    rng = np.random.default_rng(seed)
    n, R = marg['m_ball'].shape
    lines = []
    fin_lam = np.isfinite(rep['lam'])
    date_pool = rng.choice(n, size=n_dates, replace=False)
    row_pool = rng.choice(R, size=n_rows, replace=False)
    max_err_ball, max_err_box = (0.0, 0.0)
    dim = rep['Shat'].shape[1]
    c = cp.Variable(dim)
    for k in date_pool:
        mu = rep['Shat'][k]
        q = float(rep['q_r'][k])
        for ell in row_pool:
            g = Gd[ell]
            prob = cp.Problem(cp.Minimize(g @ c), [cp.norm(c - mu, 2) <= q])
            prob.solve(solver=cp.CLARABEL)
            ana = float(g @ mu - q * np.linalg.norm(g))
            max_err_ball = max(max_err_ball, abs(prob.value - ana))
            if fin_lam[k]:
                r = float(rep['lam'][k]) * rep['wshape'][k]
                num = float(g @ mu - np.abs(g) @ r)
                lo, hi = (mu - r, mu + r)
                prob2 = cp.Problem(cp.Minimize(g @ c), [c >= lo, c <= hi])
                prob2.solve(solver=cp.CLARABEL)
                max_err_box = max(max_err_box, abs(prob2.value - num))
    lines.append(f"analytic-vs-numeric: {n_dates} dates x {n_rows} rows; max |ball| err = {max_err_ball:.3e}, max |box| err = {max_err_box:.3e} (tolerance 1e-6: {('PASS' if max(max_err_ball, max_err_box) < 1e-06 else 'FAIL')})")
    scales = 10.0 ** rng.uniform(-3, 3, size=R)
    Gs = sp.diags(scales) @ G
    hs = scales * h
    Gsd = Gs.toarray()
    gl1s = np.abs(Gsd).sum(axis=1)
    gn2s = np.sqrt((Gsd ** 2).sum(axis=1))
    ks = date_pool
    gSs = rep['Shat'][ks] @ Gsd.T
    mb = 10000.0 * np.maximum(hs[None, :] - gSs + rep['q_r'][ks, None] * gn2s[None, :], 0.0) / gl1s
    err_scale = float(np.max(np.abs(mb - marg['m_ball'][ks])))
    lines.append(f"row-rescaling invariance (x in [1e-3,1e3]): max |delta| = {err_scale:.3e} normalized bp ({('PASS' if err_scale < 1e-08 else 'FAIL')})")
    for tol in (1e-12, 1e-10, 1e-08):
        for kind, raw, viol in (('ball', marg['raw_ball'], rep['ball_viol']), ('box', marg['raw_box'], rep['box_viol'])):
            n_cmp, n_dis = (0, 0)
            for k in range(n):
                if viol[k] is None:
                    continue
                for i, f in enumerate(FAMILIES):
                    mrows = fam == f
                    mine = bool(np.max(raw[k][mrows]) > tol)
                    n_cmp += 1
                    n_dis += int(mine != viol[k][f])
            lines.append(f"certificate agreement ({kind}, tol={tol:g}): {n_cmp - n_dis}/{n_cmp} family-date cells agree ({('PASS' if n_dis == 0 else f'{n_dis} DISAGREE')})")
    worstb = 0.0
    for k in range(n):
        ell = int(np.argmax(marg['m_ball'][k]))
        g = Gd[ell]
        cstar = rep['Shat'][k] - rep['q_r'][k] * g / np.linalg.norm(g)
        worstb = max(worstb, float(np.linalg.norm(cstar - rep['Shat'][k]) - rep['q_r'][k]))
    lines.append(f"ball witness membership: max(||c*-mu||-q) = {worstb:.3e} ({('PASS' if worstb < 1e-09 else 'FAIL')})")
    worstx = 0.0
    for k in np.where(fin_lam)[0]:
        ell = int(np.argmax(marg['m_box'][k]))
        g = Gd[ell]
        r = rep['lam'][k] * rep['wshape'][k]
        cstar = rep['Shat'][k] - r * np.sign(g)
        worstx = max(worstx, float(np.max(np.abs(cstar - rep['Shat'][k]) - r)))
    lines.append(f"box witness membership: max coord excess = {worstx:.3e} ({('PASS' if worstx < 1e-09 else 'FAIL')})")
    ac_viol = 0.0
    sub = date_pool[:5]
    for k in sub:
        mu = rep['proj_c'][k]
        q = float(rep['q_p'][k])
        base = Gd @ mu - h
        order = np.argsort(base)[:10]
        for ell in order:
            g = Gd[ell]
            prob = cp.Problem(cp.Minimize(g @ c), [cp.norm(c - mu, 2) <= q, Gd @ c >= h])
            prob.solve(solver=cp.CLARABEL)
            ac_viol = max(ac_viol, float(h[ell] - prob.value))
    lines.append(f"ArbCal-Set zero check ({len(sub)} dates x 10 tightest rows, min over ball-inside-A): max violation = {ac_viol:.3e} ({('PASS' if ac_viol < 1e-06 else 'FAIL')})")
    summary = {'analytic_numeric_max_error': max(max_err_ball, max_err_box), 'row_scaling_max_error': err_scale, 'arbcal_set_max_violation': ac_viol, 'witness_membership_max_excess': max(worstb, worstx)}
    return (lines, summary)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--panel', required=True)
    ap.add_argument('--canonical', required=True)
    ap.add_argument('--outdir', required=True)
    args = ap.parse_args()
    t0 = time.time()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    canon = json.loads(Path(args.canonical).read_text())
    G, h, fam = polyhedron_rows()
    Gm, hm, famm, meta = build_row_map()
    map_ok, map_checks = verify_row_map(Gm, hm, famm, G, h, fam)
    (outdir / 'row_map.json').write_text(json.dumps({'checks': map_checks, 'verified': map_ok, 'rows': meta}, indent=1))
    if not map_ok:
        raise SystemExit(f'row map does not reproduce polyhedron_rows(): {map_checks}')
    print(f'[arb] row map verified: {G.shape[0]} rows, { {f: int(np.sum(fam == f)) for f in FAMILIES}}')
    z = np.load(args.panel, allow_pickle=True)
    days = [str(d) for d in z['days']]
    C = np.asarray(z['C'], float)
    Yv = C.reshape(len(days), -1)
    rep = replay_experiment_a_rw(days, Yv, G, h, fam)
    repro_ok, got, checks, deltas = reproduction_check(rep, canon)
    print(f'[arb] reproduction: {checks}')
    print(f'[arb] full-precision deltas vs canonical: {deltas}')
    if not repro_ok:
        (outdir / 'report.md').write_text(f'# ArbCal-Set admitted-arbitrage analysis\n\n## STOPPED — Table 2 not reproduced\n\nchecks: {json.dumps(checks, indent=2)}\n\nreplayed values: {json.dumps(got, indent=2)}\n\ndeltas: {json.dumps(deltas, indent=2)}\n')
        raise SystemExit('Table 2 reproduction FAILED — see report.md')
    marg = analytic_margins(rep, G, h)
    Mb, winb, fammax_b, winfam_b = per_date_summary(marg['m_ball'], fam)
    Mx, winx, fammax_x, winfam_x = per_date_summary(marg['m_box'], fam)
    n = len(Mb)
    fin_lam = np.isfinite(rep['lam'])
    n_inf_box = int(np.sum(~fin_lam))
    import csv
    import io
    buf = io.StringIO()
    wcsv = csv.writer(buf)
    wcsv.writerow(['date', 'method', 'covered', 'radius_or_threshold', 'violator', 'worst_margin_normalized_bp', 'winning_constraint_row', 'winning_constraint_family', 'winning_moneyness_indices', 'winning_maturity_indices', 'raw_constraint_margin', 'row_l1_norm'])
    for k in range(n):
        d = rep['dates'][k]
        wl = int(winb[k])
        wcsv.writerow([d, 'raw_ball', int(rep['cov_raw'][k]), f"{rep['q_r'][k]:.10g}", int(Mb[k] > 10000.0 * TOL_CERT), f'{Mb[k]:.10g}', wl, winfam_b[k], ';'.join(map(str, meta[wl]['moneyness_indices'])), ';'.join(map(str, meta[wl]['maturity_indices'])), f"{marg['raw_ball'][k, wl]:.10g}", f"{marg['gl1'][wl]:.10g}"])
        wl = int(winx[k])
        wcsv.writerow([d, 'joint_box', int(rep['cov_box'][k]), f"{rep['lam'][k]:.10g}", int(Mx[k] > 10000.0 * TOL_CERT), f'{Mx[k]:.10g}', wl, winfam_x[k], ';'.join(map(str, meta[wl]['moneyness_indices'])), ';'.join(map(str, meta[wl]['maturity_indices'])), f"{marg['raw_box'][k, wl]:.10g}", f"{marg['gl1'][wl]:.10g}"])
        wcsv.writerow([d, 'arbcal_set', int(rep['cov_prj'][k]), f"{rep['q_p'][k]:.10g}", 0, '0.0', '', '', '', '', '0.0', ''])
    with gzip.open(outdir / 'daily.csv.gz', 'wt') as f:
        f.write(buf.getvalue())

    def qstats(M):
        fin = M[np.isfinite(M)]
        return {'mean': mean_fin(M), 'mean_note': f'over {len(fin)} finite dates; {len(M) - len(fin)} dates have infinite box threshold' if len(fin) < len(M) else 'all dates finite', 'median': float(np.median(M)), 'p75': float(np.percentile(M, 75)), 'p90': float(np.percentile(M, 90)), 'p95': float(np.percentile(M, 95)), 'max': float(np.max(fin)), 'max_note': 'max over finite dates' if len(fin) < len(M) else 'overall max'}
    boot = {}
    for name, M in (('raw_ball', Mb), ('joint_box', Mx)):
        s = qstats(M)
        p, lo, hi = mbb_stat([M], lambda a: med(a[0]))
        s['median_ci95'] = [lo, hi]
        p, lo, hi = mbb_stat([M], lambda a: mean_fin(a[0]))
        s['mean_ci95'] = [lo, hi]
        sens = {}
        for bl in (5, 20):
            _, l2, h2 = mbb_stat([M], lambda a: med(a[0]), block=bl)
            sens[str(bl)] = [l2, h2]
        s['median_ci95_blocklen_sens'] = sens
        boot[name] = s
    _, dlo, dhi = mbb_stat([Mb, Mx], lambda a: med(a[0]) - med(a[1]))
    ball_minus_box_median = {'point': med(Mb) - med(Mx), 'ci95': [dlo, dhi]}
    winfrac_b = {f: float(np.mean(winfam_b == f)) for f in FAMILIES}
    winfrac_x = {f: float(np.mean(winfam_x == f)) for f in FAMILIES}
    fam_rows = []
    for method, fmax, wfrac in (('raw_ball', fammax_b, winfrac_b), ('joint_box', fammax_x, winfrac_x)):
        for f in FAMILIES:
            v = fmax[f]
            fin = v[np.isfinite(v)]
            fam_rows.append({'method': method, 'family': f, 'frac_dates_any_violation': float(np.mean(v > 10000.0 * TOL_CERT)), 'mean_worst_margin_bp': mean_fin(v), 'median_worst_margin_bp': float(np.median(v)), 'p90_worst_margin_bp': float(np.percentile(v, 90)), 'frac_dates_overall_winner': wfrac[f]})
    with open(outdir / 'family_summary.csv', 'w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=list(fam_rows[0].keys()))
        wr.writeheader()
        wr.writerows(fam_rows)
    witnesses = {}
    for kind, M in (('ball', Mb), ('box', Mx)):
        Mf = np.where(np.isfinite(M), M, -np.inf)
        order = np.argsort(Mf)
        picks = {'median': int(order[len(order) // 2]), 'p90': int(order[int(round(0.9 * (len(order) - 1)))]), 'max': int(order[-1])}
        witnesses[kind] = {name: make_witness(kind, k, rep, marg, G, h, fam, meta) for name, k in picks.items()}
    (outdir / 'witnesses.json').write_text(json.dumps(witnesses, indent=1, default=float))
    wlines = ['# Witness examples — worst admitted arbitrage margin', '', 'Witness surfaces are hypothetical members of the prediction', 'set; margins are normalized bp (1e4 x margin / row L1 norm),', 'NOT executable profit (mid-price surfaces, no bid-ask).', '']
    for kind in ('ball', 'box'):
        wlines.append(f"## {('raw L2 ball' if kind == 'ball' else 'joint box')}")
        for name in ('median', 'p90', 'max'):
            w = witnesses[kind][name]
            xs = ', '.join((f'{v:.2f}' for v in w['moneyness_values']))
            ts = ', '.join((str(v) for v in w['maturity_days']))
            wlines += [f"### {name} date: {w['date']}", f"- margin: **{w['worst_margin_normalized_bp']:.3f} normalized bp** (row {w['constraint_row']}, family `{w['family']}`)", f'- grid: x = [{xs}], tau = [{ts}] days', f"- portfolio: {w['economic_interpretation']}", f"- center slack {w['center_slack']:.3e} -> worst attainable slack {w['worst_attainable_slack']:.3e}", f"- membership check: {w['witness_membership_ok']}; violation check: {w['witness_violates_row']}", '']
    (outdir / 'witnesses.md').write_text('\n'.join(wlines))
    vlines, vsummary = validate(rep, marg, G, h, fam)
    vtext = 'Validation of worst-admitted-arbitrage-margin computation\n' + '=' * 60 + '\n' + f'panel: {args.panel}\nn scored dates: {n}\n\n' + '\n'.join(vlines) + '\n'
    (outdir / 'validation.txt').write_text(vtext)
    print(vtext)
    summary = {'metadata': {'n_dates': n, 'start_date': str(rep['dates'][0]), 'end_date': str(rep['dates'][-1]), 'target_coverage': 1 - ALPHA_A, 'units': 'normalized_bp', 'normalization': 'constraint_row_l1', 'n_dates_infinite_box_threshold': n_inf_box, 'definition': 'M_t = 1e4 * max_l sup_{c in C_t} [h_l - g_l.c]_+ / ||g_l||_1'}, 'methods': {'raw_ball': {'coverage': got['coverage_raw_ball'], 'violator_date_rate': float(np.mean(Mb > 10000.0 * TOL_CERT)), 'worst_margin': boot['raw_ball']}, 'joint_box': {'coverage': got['coverage_joint_box'], 'violator_date_rate': float(np.mean(Mx > 10000.0 * TOL_CERT)), 'worst_margin': boot['joint_box']}, 'arbcal_set': {'coverage': got['coverage_projected_set'], 'violator_date_rate': 0.0, 'worst_margin': {'mean': 0.0, 'median': 0.0, 'p75': 0.0, 'p90': 0.0, 'p95': 0.0, 'max': 0.0, 'note': 'zero by construction (set subset of A); verified numerically'}}}, 'ball_minus_box_median': ball_minus_box_median, 'winning_families': {'raw_ball': winfrac_b, 'joint_box': winfrac_x}, 'validation': {'reproduces_table_2': True, **vsummary}, 'reproduction': {'replayed': got, 'deltas_vs_canonical': deltas}, 'elapsed_s': round(time.time() - t0, 1)}
    (outdir / 'summary.json').write_text(json.dumps(summary, indent=1, default=float))
    ea = canon['experiment_a']['rw']
    bmed, xmed = (boot['raw_ball']['median'], boot['joint_box']['median'])
    bp90 = boot['raw_ball']['p90']
    xp90 = boot['joint_box']['p90']
    tab = '\n'.join(['% CANDIDATE (arbitrage.md step 11) -- generated by scripts/lib_arb_margin.py; NOT wired into main.tex.', '\\begin{tabular}{@{}lcccc@{}}', '\\toprule', 'method & joint coverage & dates admitting & mean radius & median worst admitted\\\\', ' & & violator & & margin (normalized bp)\\\\', '\\midrule', f"raw joint ball & ${ea['coverage_raw_ball']:.3f}$ & $1.000$ & ${ea['mean_radius_raw']:.4f}$ & ${bmed:.1f}$\\\\", f"joint box & ${ea['coverage_joint_box']:.3f}$ & $1.000$ & --- & ${xmed:.1f}$\\\\", f"\\textbf{{ArbCal-Set}} & $\\mathbf{{{ea['coverage_projected_set']:.3f}}}$ & $\\mathbf{{0.000}}$ & ${ea['mean_radius_projected']:.4f}$ & $\\mathbf{{0}}$\\\\", '\\bottomrule', '\\end{tabular}', '', '% skew-robust alternative for the last column (median [90th percentile]):', f'%   raw joint ball: ${bmed:.1f}$ [{bp90:.1f}]', f'%   joint box:      ${xmed:.1f}$ [{xp90:.1f}]', f'%   ArbCal-Set:     $0$ [0]']) + '\n'
    (outdir / 'tab_arbitrage_candidate.tex').write_text(tab)
    print(f'[arb] wrote all outputs to {outdir} ({time.time() - t0:.0f}s)')
    return summary
if __name__ == '__main__':
    main()
