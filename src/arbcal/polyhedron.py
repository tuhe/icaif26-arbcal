from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import scipy.sparse as sp
from . import config as CFG
FAMILIES = ('upper', 'intrinsic', 'monotone', 'slope', 'convex', 'calendar')

def polyhedron_rows(x_grid=None, n_tau=None):
    x = np.asarray(CFG.X_GRID if x_grid is None else x_grid, float)
    n_tau = CFG.N_TAU if n_tau is None else int(n_tau)
    n_x = len(x)
    dx = np.diff(x)
    idx = lambda i, j: i * n_tau + j
    rows, cols, vals, hs, fam = ([], [], [], [], [])

    def add(coefs, rhs, family):
        r = len(hs)
        for c, v in coefs:
            rows.append(r)
            cols.append(c)
            vals.append(v)
        hs.append(rhs)
        fam.append(family)
    for j in range(n_tau):
        for i in range(n_x):
            add([(idx(i, j), -1.0)], -1.0, 'upper')
        for i in range(n_x):
            add([(idx(i, j), 1.0)], max(0.0, 1.0 - x[i]), 'intrinsic')
        for i in range(n_x - 1):
            a, b = (idx(i, j), idx(i + 1, j))
            add([(a, 1.0 / dx[i]), (b, -1.0 / dx[i])], 0.0, 'monotone')
            add([(a, -1.0 / dx[i]), (b, 1.0 / dx[i])], -1.0, 'slope')
        for i in range(1, n_x - 1):
            add([(idx(i - 1, j), 1.0 / dx[i - 1]), (idx(i, j), -1.0 / dx[i - 1] - 1.0 / dx[i]), (idx(i + 1, j), 1.0 / dx[i])], 0.0, 'convex')
    for i in range(n_x):
        for j in range(n_tau - 1):
            add([(idx(i, j + 1), 1.0), (idx(i, j), -1.0)], 0.0, 'calendar')
    G = sp.csc_matrix((vals, (rows, cols)), shape=(len(hs), n_x * n_tau))
    return (G, np.asarray(hs, float), np.asarray(fam))

@dataclass
class ViolationReport:
    by_family: dict
    worst_by_family: dict
    total: int

    @property
    def any(self):
        return self.total > 0

def violations(c_vec, G, h, family, tol=None) -> ViolationReport:
    tol = CFG.FEAS_TOL if tol is None else float(tol)
    r = np.asarray(G @ np.asarray(c_vec, float).ravel()) - h
    bad = r < -tol
    by, worst = ({}, {})
    for f in FAMILIES:
        m = family == f
        by[f] = int(np.sum(bad & m))
        worst[f] = float(np.min(r[m])) if np.any(m) else 0.0
    return ViolationReport(by, worst, int(np.sum(bad)))

class PolyProjector:

    def __init__(self, G, h, family=None, eps=1e-09, max_iter=200000):
        import osqp
        self.G, self.h = (G, np.asarray(h, float))
        self.family = family
        n = G.shape[1]
        self._prob = osqp.OSQP()
        self._prob.setup(P=sp.identity(n, format='csc'), q=np.zeros(n), A=G.tocsc(), l=self.h, u=np.full(G.shape[0], np.inf), eps_abs=eps, eps_rel=eps, max_iter=max_iter, polishing=True, polish_refine_iter=20, adaptive_rho=True, verbose=False)
        self.n = n

    def __call__(self, v):
        v = np.asarray(v, float).ravel()
        self._prob.update(q=-v)
        res = self._prob.solve()
        ok = res.info.status_val in (1, 2)
        z = np.asarray(res.x, float) if ok and res.x is not None else None
        if z is None or not np.all(np.isfinite(z)):
            return None
        return z

    def project(self, v, family=None, tol=None):
        z = self(v)
        if z is None:
            return (None, False)
        tol = CFG.FEAS_TOL if tol is None else tol
        r = np.asarray(self.G @ z) - self.h
        return (z, bool(np.min(r) >= -tol))

def dist_to_set(v, projector: PolyProjector):
    z, ok = projector.project(v)
    if z is None:
        return (np.nan, None)
    return (float(np.linalg.norm(np.asarray(v, float).ravel() - z)), z)

def ball_admits_violation(center, radius, G, h, family):
    p = np.asarray(center, float).ravel()
    gp = np.asarray(G @ p)
    gn = np.sqrt(np.asarray(G.multiply(G).sum(axis=1)).ravel())
    bad = gp - float(radius) * gn < h - 1e-12
    return {f: bool(np.any(bad[family == f])) for f in FAMILIES}

def box_admits_violation(center, halfwidth, G, h, family):
    p = np.asarray(center, float).ravel()
    w = np.asarray(halfwidth, float).ravel()
    gp = np.asarray(G @ p)
    gabs = np.asarray(abs(G) @ w)
    bad = gp - gabs < h - 1e-12
    return {f: bool(np.any(bad[family == f])) for f in FAMILIES}
