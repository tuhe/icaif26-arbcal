from __future__ import annotations
import csv
import importlib.util
from pathlib import Path
import numpy as np
from arbcal import config as CFG
from arbcal import repricing as rp
from arbcal.aci import ExactACI, upper_rank, rank_threshold
from arbcal.polyhedron import FAMILIES
from arbcal.scenarios import shrunk_cov
BOOKS = list(CFG.BOOK_NAMES)
K_BOOKS = len(BOOKS)
REPO = Path(__file__).resolve().parents[1]
NAMESPACE = REPO / 'results'
PRIMARY_START = '2018-01-01'
Y2023_START = '2023-01-01'
DEV_END = CFG.DEV_END

def load_vs_run():
    spec = importlib.util.spec_from_file_location('vs_run', Path(__file__).parent / 'vs_run.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def load_manifest(path=None):
    path = Path(path or NAMESPACE / 'date_manifest.csv')
    rows = list(csv.DictReader(open(path)))
    return {r['date']: r for r in rows}

def panel_masks(days, manifest):
    D = len(days)
    vt = np.zeros(D, bool)
    for t in range(1, D):
        m = manifest.get(days[t])
        if m is None:
            raise SystemExit(f'panel day {days[t]} missing from manifest')
        if not int(m['retained']):
            raise SystemExit(f'panel day {days[t]} not retained in manifest')
        vt[t] = bool(int(m['valid_one_session_transition'])) and m['previous_expected_trading_date'] == days[t - 1]
    vt2 = vt.copy()
    vt2[1:] &= vt[:-1]
    dates = np.asarray(days)
    masks = {'vt': vt, 'vt2': vt2, 'prehistory': dates < PRIMARY_START, 'dev': (dates >= PRIMARY_START) & (dates < DEV_END), 'primary': (dates >= PRIMARY_START) & (dates < Y2023_START), 'y2023': dates >= Y2023_START}
    return masks

def load_panel(path):
    z = np.load(path, allow_pickle=True)
    days = [str(d) for d in z['days']]
    S = np.asarray(z['S'], float)
    F = np.asarray(z['F'], float)
    DISC = np.asarray(z['DISC'], float)
    C = np.asarray(z['C'], float)
    return (days, S, F, DISC, C)

def gap_innovations(S, C, vt, t):
    Yv = C[:t].reshape(t, -1)
    innov = np.column_stack([np.diff(np.log(S[:t])), np.diff(Yv, axis=0)])
    return (innov[vt[1:t]], Yv)

def realized_losses(days, S, F, DISC, C, vt):
    D = len(days)
    L = np.full((D, K_BOOKS), np.nan)
    for t in range(1, D):
        if not vt[t]:
            continue
        books = rp.build_books(S[t - 1], F[t - 1], DISC[t - 1], C[t - 1])
        F59, D59 = rp.realized_forward_discount(S[t], F[t], DISC[t], CFG.BOOK_TAU - 1)
        for j, name in enumerate(BOOKS):
            L[t, j] = rp.book_loss_bps(books[name], S[t], F59, D59, C[t])
    return L

def lagged_covariates(S, C, vt):
    D = len(S)
    absret = np.full(D, np.nan)
    for t in range(1, D):
        if vt[t]:
            absret[t] = abs(np.log(S[t] / S[t - 1]))
    spotret_abs = np.concatenate([[np.nan], absret[:-1]])
    jB = CFG.TAUS.index(CFG.BOOK_TAU)
    atm = C[:, CFG.N_X // 2, jB]
    atm_chg = np.full(D, np.nan)
    for t in range(1, D):
        if vt[t]:
            atm_chg[t] = abs(atm[t] - atm[t - 1])
    atmchg = np.full(D, np.nan)
    for t in range(21, D):
        w = atm_chg[max(1, t - 20):t]
        atmchg[t] = np.std(w)
    return (spotret_abs, atmchg)

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
        if not np.all(np.isfinite(L[t])):
            continue
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

def run_rank_method(alpha, D, L, mu, sd, SL_by_t, M, gamma=None):
    gamma = CFG.GAMMA_RULE(alpha) if gamma is None else gamma
    a = ExactACI(alpha, gamma)
    rec = dict(defined=np.zeros(D, bool), exceed=np.full(D, np.nan), reserve=np.full(D, np.nan), avail=np.zeros(D, bool), beta_sat=np.zeros(D, bool), maxrank_sat=np.zeros(D, bool), booksat=np.zeros((D, K_BOOKS), bool), B_bps=np.full((D, K_BOOKS), np.nan), book_exceed=np.zeros((D, K_BOOKS), bool))
    for t in range(D):
        if t not in SL_by_t or not np.all(np.isfinite(mu[t])) or (not np.all(np.isfinite(L[t]))):
            continue
        SL = SL_by_t[t][:M]
        u = np.array([upper_rank(SL[:, j], L[t, j]) for j in range(K_BOOKS)])
        s = float(np.max(u))
        beta = a.threshold()
        if not a.in_warmup:
            if beta == np.inf or beta >= 1.0:
                Bs = np.full(K_BOOKS, np.inf)
            elif beta == -np.inf:
                Bs = np.full(K_BOOKS, -np.inf)
            else:
                Bs = np.array([rank_threshold(SL[:, j], beta) for j in range(K_BOOKS)])
            rec['defined'][t] = True
            rec['exceed'][t] = float(np.any(L[t] > Bs))
            rec['avail'][t] = bool(np.all(np.isfinite(Bs)))
            rec['beta_sat'][t] = bool(beta >= 1.0)
            rec['maxrank_sat'][t] = bool(s >= 1.0)
            rec['booksat'][t] = u >= 1.0
            rec['B_bps'][t] = Bs
            rec['book_exceed'][t] = L[t] > Bs
            rn = (Bs - mu[t]) / sd[t]
            rec['reserve'][t] = float(np.mean(rn)) if np.all(np.isfinite(rn)) else np.inf
        a.update(s, beta)
    rec['aci'] = a.diagnostics()
    return rec

def summarize_periods(vsrun, masks, defined, exceed, reserve):
    out = {}
    for period in ('dev', 'primary', 'y2023'):
        m = defined & masks[period]
        out[period] = vsrun.summarize_series(m, exceed, reserve)
        idx = np.where(m)[0]
        out[period]['first_scored'] = None if not len(idx) else int(idx[0])
        out[period]['last_scored'] = None if not len(idx) else int(idx[-1])
    return out

def boundary_report(diag):
    keys = ('n_scored', 'n_alpha_low', 'n_alpha_high', 'n_inf_threshold', 'n_empty_threshold')
    if isinstance(diag, list):
        return {f'book{j}': {k: d.get(k) for k in keys} for j, d in enumerate(diag)}
    return {k: diag.get(k) for k in keys}
