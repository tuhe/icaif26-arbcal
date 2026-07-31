from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from . import config as CFG
from .polyhedron import violations, FAMILIES

class LaggedView:

    def __init__(self, days, S, F, DISC, C, t: int):
        if not 0 < t <= len(days):
            raise ValueError(f't={t} outside panel')
        self.t = int(t)
        self.days = np.asarray(days)[:t]
        self.S = np.asarray(S, float)[:t]
        self.F = np.asarray(F, float)[:t]
        self.DISC = np.asarray(DISC, float)[:t]
        self.C = np.asarray(C, float)[:t]
        self.cutoff_date = str(self.days[-1])

def shrunk_cov(E):
    try:
        from sklearn.covariance import ledoit_wolf
        K, _ = ledoit_wolf(E)
    except Exception:
        S = np.cov(E, rowvar=False)
        d = S.shape[0]
        K = 0.85 * S + 0.15 * (np.trace(S) / d) * np.eye(d)
    K = 0.5 * (K + K.T)
    K.flat[::K.shape[0] + 1] += 1e-10
    return K

@dataclass
class ScenarioSet:
    date: str
    cutoff_date: str
    R: np.ndarray
    surf: np.ndarray
    surf_raw: np.ndarray
    n_proj_fail: int
    raw_viol_frac: dict
    proj_viol_frac: dict

def generate(lagged: LaggedView, projector, eval_date: str, *, n_draws=CFG.N_DRAWS, seed: int=0) -> ScenarioSet | None:
    C, S = (lagged.C, lagged.S)
    Yv = C.reshape(len(C), -1)
    innov = np.column_stack([np.diff(np.log(S)), np.diff(Yv, axis=0)])
    E = innov[-CFG.LW_WINDOW:]
    if len(E) < CFG.MIN_INNOV:
        return None
    K = shrunk_cov(E)
    mu = np.concatenate([[0.0], Yv[-1]])
    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(K)
    draws = mu[None, :] + rng.standard_normal((n_draws, len(mu))) @ L.T
    R = draws[:, 0]
    Zraw = draws[:, 1:]
    G, h, fam = (projector.G, projector.h, projector.family)
    surf = np.empty_like(Zraw)
    n_fail = 0
    raw_bad = {f: 0 for f in FAMILIES}
    proj_bad = {f: 0 for f in FAMILIES}
    for m in range(n_draws):
        vr = violations(Zraw[m], G, h, fam, tol=CFG.FEAS_TOL)
        for f in FAMILIES:
            raw_bad[f] += int(vr.by_family[f] > 0)
        z, ok = projector.project(Zraw[m])
        if z is None:
            n_fail += 1
            z = Zraw[m]
        elif not ok:
            vp = violations(z, G, h, fam, tol=CFG.FEAS_TOL)
            for f in FAMILIES:
                proj_bad[f] += int(vp.by_family[f] > 0)
        surf[m] = z
    return ScenarioSet(date=str(eval_date), cutoff_date=lagged.cutoff_date, R=R, surf=surf.reshape(n_draws, CFG.N_X, CFG.N_TAU), surf_raw=Zraw.reshape(n_draws, CFG.N_X, CFG.N_TAU), n_proj_fail=n_fail, raw_viol_frac={f: raw_bad[f] / n_draws for f in FAMILIES}, proj_viol_frac={f: proj_bad[f] / n_draws for f in FAMILIES})
