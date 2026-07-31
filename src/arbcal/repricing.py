from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from . import config as CFG
_TAUS = np.asarray(CFG.TAUS, float)

def _tau_weights(tau_days: float):
    t = float(np.clip(tau_days, _TAUS[0], _TAUS[-1]))
    j = int(np.clip(np.searchsorted(_TAUS, t) - 1, 0, CFG.N_TAU - 2))
    a = (t - _TAUS[j]) / (_TAUS[j + 1] - _TAUS[j])
    return (j, j + 1, float(a))
EXTRAPOLATION = 'extend'

def interp_call_norm(Csurf, x, tau_days: float):
    g = CFG.X_GRID
    jl, jh, a = _tau_weights(tau_days)
    xr = np.atleast_1d(np.asarray(x, float))
    xq = np.clip(xr, g[0], g[-1])
    i = np.clip(np.searchsorted(g, xq) - 1, 0, CFG.N_X - 2)
    wr = (xq - g[i]) / (g[i + 1] - g[i])
    col = (1.0 - a) * np.asarray(Csurf)[..., jl] + a * np.asarray(Csurf)[..., jh]
    if col.ndim == 1:
        v = (1.0 - wr) * col[i] + wr * col[i + 1]
        if EXTRAPOLATION == 'extend':
            sL = np.clip((col[1] - col[0]) / (g[1] - g[0]), -1.0, 0.0)
            sR = np.clip((col[-1] - col[-2]) / (g[-1] - g[-2]), -1.0, 0.0)
            lo = xr < g[0]
            hi = xr > g[-1]
            v = np.where(lo, col[0] + sL * (xr - g[0]), v)
            v = np.where(hi, col[-1] + sR * (xr - g[-1]), v)
            v = np.clip(np.maximum(v, np.maximum(1.0 - xr, 0.0)), 0.0, 1.0)
    else:
        m = np.arange(col.shape[0])
        if xq.shape[0] == 1:
            i = np.repeat(i, col.shape[0])
            wr = np.repeat(wr, col.shape[0])
            xr = np.repeat(xr, col.shape[0])
        v = (1.0 - wr) * col[m, i] + wr * col[m, i + 1]
        if EXTRAPOLATION == 'extend':
            sL = np.clip((col[:, 1] - col[:, 0]) / (g[1] - g[0]), -1.0, 0.0)
            sR = np.clip((col[:, -1] - col[:, -2]) / (g[-1] - g[-2]), -1.0, 0.0)
            lo = xr < g[0]
            hi = xr > g[-1]
            v = np.where(lo, col[:, 0] + sL * (xr - g[0]), v)
            v = np.where(hi, col[:, -1] + sR * (xr - g[-1]), v)
            v = np.clip(np.maximum(v, np.maximum(1.0 - xr, 0.0)), 0.0, 1.0)
    return v if v.size > 1 else float(v[0])

def out_of_grid_stats(K_legs, F_t):
    g = CFG.X_GRID
    xs = (np.asarray(K_legs, float)[:, None] / np.atleast_1d(np.asarray(F_t, float))[None, :]).ravel()
    lo = xs < g[0]
    hi = xs > g[-1]
    dist = np.maximum(np.where(lo, g[0] - xs, 0.0), np.where(hi, xs - g[-1], 0.0))
    return dict(frac_below=float(np.mean(lo)), frac_above=float(np.mean(hi)), max_dist=float(np.max(dist)) if len(dist) else 0.0)

def call_price(Csurf, K, S_unused, F, D, tau_days):
    x = np.asarray(K, float) / np.asarray(F, float)
    return np.asarray(D, float) * np.asarray(F, float) * interp_call_norm(Csurf, x, tau_days)

def put_price(Csurf, K, S_unused, F, D, tau_days):
    C = call_price(Csurf, K, None, F, D, tau_days)
    return C - np.asarray(D, float) * (np.asarray(F, float) - np.asarray(K, float))

@dataclass(frozen=True)
class Book:
    name: str
    legs: tuple
    hedge_shares: float
    v0: float
    s0: float

def _fwd_call_delta_curve(c_col, D):
    g = CFG.X_GRID
    return D * (c_col - g * np.gradient(c_col, g))

def _strike_for_fwd_delta(c_col, D, target):
    cd = _fwd_call_delta_curve(c_col, D)
    return float(np.interp(target, cd[::-1], CFG.X_GRID[::-1]))

def build_books(S0, F_row, D_row, C_surf):
    jB = CFG.TAUS.index(CFG.BOOK_TAU)
    F0, D0 = (float(F_row[jB]), float(D_row[jB]))
    col = np.asarray(C_surf, float)[:, jB]
    x25p = _strike_for_fwd_delta(col, D0, D0 - 0.25)
    x25c = _strike_for_fwd_delta(col, D0, 0.25)
    tau0 = float(CFG.BOOK_TAU)

    def px(kind, K):
        f = call_price if kind == 'c' else put_price
        return float(f(np.asarray(C_surf, float), K, None, F0, D0, tau0))
    specs = {'short_straddle': ([('c', F0, -1.0), ('p', F0, -1.0)], 0.0), 'short_25d_put': ([('p', x25p * F0, -1.0)], 0.0), 'dh_short_call': ([('c', F0, -1.0)], None), 'risk_reversal': ([('c', x25c * F0, +1.0), ('p', x25p * F0, -1.0)], 0.0), 'short_strangle': ([('c', x25c * F0, -1.0), ('p', x25p * F0, -1.0)], 0.0), 'long_butterfly': ([('c', 0.97 * F0, +1.0), ('c', 1.0 * F0, -2.0), ('c', 1.03 * F0, +1.0)], 0.0)}
    atm_fwd_delta = float(np.interp(1.0, CFG.X_GRID, _fwd_call_delta_curve(col, D0)))
    specs['dh_short_call'] = (specs['dh_short_call'][0], +atm_fwd_delta * F0 / float(S0))
    books = {}
    for name, (legs, hedge) in specs.items():
        v0 = sum((q * px(k, K) for k, K, q in legs)) + hedge * float(S0)
        books[name] = Book(name=name, legs=tuple(((k, float(K), float(q)) for k, K, q in legs)), hedge_shares=float(hedge), v0=float(v0), s0=float(S0))
    return books

def scenario_forward_discount(S_t, S0, F0_row, D0_row):
    jB = CFG.TAUS.index(CFG.BOOK_TAU)
    tau = float(CFG.BOOK_TAU)
    F = np.asarray(S_t, float) * float(F0_row[jB]) / float(S0)
    D = float(D0_row[jB]) ** ((tau - 1.0) / tau)
    return (F, D)

def realized_forward_discount(S_t, F_row_t, D_row_t, tau_days):
    lb = np.log(np.asarray(F_row_t, float) / float(S_t))
    ld = np.log(np.asarray(D_row_t, float))
    F = float(S_t) * float(np.exp(np.interp(tau_days, _TAUS, lb)))
    D = float(np.exp(np.interp(tau_days, _TAUS, ld)))
    return (F, D)

def book_value(book: Book, S_t, F_t, D_t, Csurf_t, tau_days):
    S_t = np.asarray(S_t, float)
    v = book.hedge_shares * S_t
    for kind, K, q in book.legs:
        f = call_price if kind == 'c' else put_price
        v = v + q * f(Csurf_t, K, None, F_t, D_t, tau_days)
    return v

def book_loss_bps(book: Book, S_t, F_t, D_t, Csurf_t, tau_days=float(CFG.BOOK_TAU - 1)):
    v_t = book_value(book, S_t, F_t, D_t, Csurf_t, tau_days)
    return (book.v0 - v_t) / book.s0 * 10000.0
