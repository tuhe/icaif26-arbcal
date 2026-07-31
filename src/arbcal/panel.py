from __future__ import annotations
import glob
import os
from dataclasses import dataclass, field
import numpy as np
import scipy.sparse as sp
from arbcal.optionsdx import read_month
from . import config as CFG
from .polyhedron import polyhedron_rows, violations, FAMILIES

def parity_forward_discount(K, c_mid, p_mid, w):
    K = np.asarray(K, float)
    y = np.asarray(c_mid, float) - np.asarray(p_mid, float)
    w = np.asarray(w, float)
    m = np.isfinite(K) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    K, y, w = (K[m], y[m], w[m])
    if len(K) < 5:
        return (np.nan, np.nan, False)

    def wfit(K, y, w):
        W = w / w.mean()
        X = np.column_stack([K, np.ones_like(K)])
        sol, *_ = np.linalg.lstsq(X * np.sqrt(W)[:, None], y * np.sqrt(W), rcond=None)
        return sol
    slope, icpt = wfit(K, y, w)
    resid = y - (slope * K + icpt)
    mad = np.median(np.abs(resid - np.median(resid))) + 1e-12
    keep = np.abs(resid - np.median(resid)) <= 5.0 * 1.4826 * mad
    if keep.sum() >= 5 and keep.sum() < len(K):
        slope, icpt = wfit(K[keep], y[keep], w[keep])
    D = float(-slope)
    F = float(icpt / D) if D > 0 else np.nan
    return (F, D, bool(np.isfinite(F) and D > 0))

def _interp_row(x, grid):
    x = float(np.clip(x, grid[0], grid[-1]))
    j = int(np.clip(np.searchsorted(grid, x) - 1, 0, len(grid) - 2))
    hgt = grid[j + 1] - grid[j]
    return ((j, (grid[j + 1] - x) / hgt), (j + 1, (x - grid[j]) / hgt))

def _ridge_matrix():
    n_x, n_tau = (CFG.N_X, CFG.N_TAU)
    rows, cols, vals = ([], [], [])
    r = 0
    for j in range(n_tau):
        for i in range(1, n_x - 1):
            for di, v in ((-1, 1.0), (0, -2.0), (1, 1.0)):
                rows.append(r)
                cols.append((i + di) * n_tau + j)
                vals.append(v)
            r += 1
    return sp.csr_matrix((vals, (rows, cols)), shape=(r, CFG.DIM))

@dataclass
class DayFit:
    date: str
    spot: float
    F: np.ndarray
    D: np.ndarray
    C: np.ndarray
    parity_fallback: np.ndarray
    n_quotes: int
    unc_viol: dict
    clean_dist_l2: float
    clean_dist_inf: float
    rmse_unc: float
    rmse_con: float
    within_ba_unc: float
    within_ba_con: float
    n_active: int
    extra: dict = field(default_factory=dict)

def pick_expiries(groups, mode: str='independent'):
    if mode == 'independent':
        picked = {}
        for jt, tau in enumerate(CFG.TAUS):
            cand = [(abs(d - tau), e) for e, (d, _) in groups.items() if abs(d - tau) <= CFG.MAX_DTE_RATIO * tau]
            if not cand:
                return None
            picked[jt] = min(cand)[1]
        return picked
    if mode != 'joint':
        raise ValueError(f'unknown expiry mode {mode!r}')
    exps = sorted(groups.items(), key=lambda kv: kv[1][0])
    dtes = [d for _, (d, _) in exps]
    m, J = (len(exps), len(CFG.TAUS))
    INF = float('inf')
    cost = [[INF] * m for _ in range(J)]
    back = [[-1] * m for _ in range(J)]
    for j, tau in enumerate(CFG.TAUS):
        for i, d in enumerate(dtes):
            if abs(d - tau) > CFG.MAX_DTE_RATIO * tau:
                continue
            rel = abs(d - tau) / tau
            if j == 0:
                cost[0][i] = rel
            else:
                best, bi = (INF, -1)
                for i2 in range(i):
                    if cost[j - 1][i2] < best:
                        best, bi = (cost[j - 1][i2], i2)
                if bi >= 0:
                    cost[j][i] = best + rel
                    back[j][i] = bi
    finite = [i for i in range(m) if cost[J - 1][i] < INF]
    if not finite:
        return None
    i = min(finite, key=lambda k: cost[J - 1][k])
    picked = {}
    for j in range(J - 1, -1, -1):
        picked[j] = exps[i][0]
        i = back[j][i]
    return picked

def fit_day(day_df, date: str, G, h, family, ridge, *, max_rel_spread=0.5, min_quotes=None, expiry_mode: str='independent') -> DayFit | None:
    min_quotes = CFG.MIN_QUOTES if min_quotes is None else int(min_quotes)
    spot = float(day_df['UNDERLYING_LAST'].iloc[0])
    if not np.isfinite(spot) or spot <= 0:
        return None
    quotes = []
    Fs = np.full(CFG.N_TAU, np.nan)
    Ds = np.full(CFG.N_TAU, np.nan)
    fb = np.zeros(CFG.N_TAU, int)
    groups = {}
    for exp, g in day_df.groupby('EXPIRE_DATE'):
        dte = float(g['DTE'].iloc[0])
        if np.isfinite(dte) and dte >= 5:
            groups[exp] = (dte, g)
    if not groups:
        return None
    picked = pick_expiries(groups, expiry_mode)
    if picked is None:
        return None
    for jt, tau in enumerate(CFG.TAUS):
        _, g = groups[picked[jt]]
        cb, ca = (g['C_BID'].to_numpy(float), g['C_ASK'].to_numpy(float))
        pb, pa = (g['P_BID'].to_numpy(float), g['P_ASK'].to_numpy(float))
        K = g['STRIKE'].to_numpy(float)
        okc = np.isfinite(cb) & np.isfinite(ca) & (cb > 0) & (ca > cb)
        okp = np.isfinite(pb) & np.isfinite(pa) & (pb > 0) & (pa > pb)
        both = okc & okp & (np.abs(K / spot - 1.0) < 0.1)
        if both.sum() >= 5:
            cm, pm = (0.5 * (cb + ca), 0.5 * (pb + pa))
            hs = 0.5 * (ca - cb + (pa - pb))
            F, D, ok = parity_forward_discount(K[both], cm[both], pm[both], 1.0 / np.maximum(hs[both], 0.0001))
            if ok and CFG.PARITY_D_BOUNDS[0] <= D <= CFG.PARITY_D_BOUNDS[1] and (CFG.PARITY_BASIS_BOUNDS[0] <= F / spot <= CFG.PARITY_BASIS_BOUNDS[1]):
                Fs[jt], Ds[jt] = (F, D)
    tarr = np.asarray(CFG.TAUS, float)
    good = np.isfinite(Fs) & np.isfinite(Ds)
    if good.sum() == 0:
        Fs[:], Ds[:] = (spot, 1.0)
        fb[:] = 2
    elif good.sum() < CFG.N_TAU:
        lb = np.log(Fs[good] / spot)
        ld = np.log(Ds[good])
        for jt in np.where(~good)[0]:
            Fs[jt] = spot * np.exp(np.interp(tarr[jt], tarr[good], lb))
            Ds[jt] = np.exp(np.interp(tarr[jt], tarr[good], ld))
            fb[jt] = 1
    for jt, tau in enumerate(CFG.TAUS):
        _, g = groups[picked[jt]]
        cb, ca = (g['C_BID'].to_numpy(float), g['C_ASK'].to_numpy(float))
        K = g['STRIKE'].to_numpy(float)
        ok = np.isfinite(cb) & np.isfinite(ca) & (cb > 0) & (ca > cb)
        mid = 0.5 * (cb + ca)
        hs = 0.5 * (ca - cb)
        ok &= hs / np.maximum(mid, 1e-06) < max_rel_spread
        x = K / Fs[jt]
        ok &= (x >= CFG.X_GRID[0]) & (x <= CFG.X_GRID[-1])
        if ok.sum() < min_quotes:
            return None
        den = Ds[jt] * Fs[jt]
        for xx, mm, ss in zip(x[ok], mid[ok] / den, hs[ok] / den):
            quotes.append((jt, xx, mm, ss))
    nq = len(quotes)
    rows, cols, vals = ([], [], [])
    y = np.empty(nq)
    hs_n = np.empty(nq)
    for r, (jt, xx, mm, ss) in enumerate(quotes):
        for i, v in _interp_row(xx, CFG.X_GRID):
            rows.append(r)
            cols.append(i * CFG.N_TAU + jt)
            vals.append(v)
        y[r] = mm
        hs_n[r] = ss
    A = sp.csr_matrix((vals, (rows, cols)), shape=(nq, CFG.DIM))
    w = 1.0 / np.maximum(hs_n, 1e-05)
    w = w / w.mean()
    Aw = sp.diags(w) @ A
    yw = w * y
    lam = CFG.FIT_RIDGE * float(np.mean(w ** 2))
    R = ridge
    lhs = (Aw.T @ Aw + lam * (R.T @ R)).toarray()
    rhs = np.asarray(Aw.T @ yw).ravel()
    try:
        c_unc = np.linalg.solve(lhs + 1e-12 * np.eye(CFG.DIM), rhs)
    except np.linalg.LinAlgError:
        return None
    v_unc = violations(c_unc, G, h, family, tol=CFG.FEAS_TOL)
    import osqp
    P = sp.csc_matrix(2.0 * (Aw.T @ Aw + lam * (R.T @ R)))
    qv = -2.0 * rhs
    prob = osqp.OSQP()
    prob.setup(P=P, q=qv, A=G.tocsc(), l=h, u=np.full(G.shape[0], np.inf), eps_abs=1e-09, eps_rel=1e-09, max_iter=200000, polishing=True, polish_refine_iter=20, verbose=False)
    res = prob.solve()
    if res.info.status_val not in (1, 2) or res.x is None:
        return None
    c_con = np.asarray(res.x, float)
    resid = np.asarray(G @ c_con) - h
    if np.min(resid) < -CFG.FEAS_TOL:
        return None

    def rmse(c):
        return float(np.sqrt(np.mean((w * (np.asarray(A @ c).ravel() - y)) ** 2)))

    def within_ba(c):
        e = np.abs(np.asarray(A @ c).ravel() - y)
        return float(np.mean(e <= hs_n + 1e-12))
    return DayFit(date=date, spot=spot, F=Fs, D=Ds, C=c_con.reshape(CFG.N_X, CFG.N_TAU), parity_fallback=fb, n_quotes=nq, unc_viol=dict(v_unc.by_family), clean_dist_l2=float(np.linalg.norm(c_con - c_unc)), clean_dist_inf=float(np.max(np.abs(c_con - c_unc))), rmse_unc=rmse(c_unc), rmse_con=rmse(c_con), within_ba_unc=within_ba(c_unc), within_ba_con=within_ba(c_con), n_active=int(np.sum(np.abs(resid) <= 10 * CFG.FEAS_TOL)), extra={'picked_dte': [float(groups[picked[jt]][0]) for jt in range(CFG.N_TAU)], 'expiry_mode': expiry_mode})

def build_panel(data_dir, *, verbose=True, expiry_mode='independent'):
    G, h, family = polyhedron_rows()
    ridge = _ridge_matrix()
    fits = []
    for mfile in sorted(glob.glob(os.path.join(str(data_dir), '*.txt'))):
        df = read_month(mfile)
        for date in sorted(df['QUOTE_DATE'].astype(str).str.strip().unique()):
            day = df[df['QUOTE_DATE'].astype(str).str.strip() == date]
            try:
                out = fit_day(day, date, G, h, family, ridge, expiry_mode=expiry_mode)
            except Exception as e:
                if verbose:
                    print(f'[panel] {date} FAILED: {type(e).__name__}: {e}')
                out = None
            if out is not None:
                fits.append(out)
        if verbose:
            print(f'[panel] {os.path.basename(mfile)} -> cum {len(fits)} days')
    return fits

def save_panel(fits, path):
    np.savez_compressed(path, days=np.array([f.date for f in fits]), S=np.array([f.spot for f in fits]), F=np.stack([f.F for f in fits]), DISC=np.stack([f.D for f in fits]), C=np.stack([f.C for f in fits]).astype(np.float64), parity_fallback=np.stack([f.parity_fallback for f in fits]), n_quotes=np.array([f.n_quotes for f in fits]), unc_viol=np.array([[f.unc_viol[k] for k in FAMILIES] for f in fits]), unc_viol_families=np.array(FAMILIES), clean_dist_l2=np.array([f.clean_dist_l2 for f in fits]), clean_dist_inf=np.array([f.clean_dist_inf for f in fits]), rmse_unc=np.array([f.rmse_unc for f in fits]), rmse_con=np.array([f.rmse_con for f in fits]), within_ba_unc=np.array([f.within_ba_unc for f in fits]), within_ba_con=np.array([f.within_ba_con for f in fits]), n_active=np.array([f.n_active for f in fits]), x_grid=CFG.X_GRID, taus=np.array(CFG.TAUS))

def load_panel(path):
    z = np.load(path, allow_pickle=True)
    return z
