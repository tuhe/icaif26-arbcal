from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from . import config as CFG

@dataclass
class ExactACI:
    alpha: float
    gamma: float | None = None
    window: int = CFG.ACI_WINDOW
    warmup: int = CFG.WARMUP
    alpha_t: float = field(init=False)
    _past: list = field(init=False, default_factory=list)
    n_low: int = field(init=False, default=0)
    n_high: int = field(init=False, default=0)
    n_inf: int = field(init=False, default=0)
    n_ninf: int = field(init=False, default=0)
    alpha_min: float = field(init=False, default=np.inf)
    alpha_max: float = field(init=False, default=-np.inf)
    n_scored: int = field(init=False, default=0)

    def __post_init__(self):
        if self.gamma is None:
            self.gamma = CFG.GAMMA_RULE(self.alpha)
        self.alpha_t = float(self.alpha)

    @property
    def in_warmup(self) -> bool:
        return len(self._past) < self.warmup

    def threshold(self) -> float:
        if self.in_warmup:
            lvl = 1.0 - self.alpha
        elif self.alpha_t <= 0.0:
            self.n_low += 1
            self.n_inf += 1
            return np.inf
        elif self.alpha_t >= 1.0:
            self.n_high += 1
            self.n_ninf += 1
            return -np.inf
        else:
            lvl = 1.0 - self.alpha_t
        past = np.asarray(self._past[-self.window:], float)
        if len(past) == 0:
            return np.inf
        return float(np.quantile(past, lvl, method='higher'))

    def update(self, score: float, threshold: float):
        err = 1.0 if threshold == -np.inf or (threshold != np.inf and score > threshold) else 0.0
        scored = not self.in_warmup
        self._past.append(float(score))
        if len(self._past) > 4 * self.window:
            self._past = self._past[-2 * self.window:]
        if scored:
            self.alpha_t = self.alpha_t + self.gamma * (self.alpha - err)
            self.alpha_min = min(self.alpha_min, self.alpha_t)
            self.alpha_max = max(self.alpha_max, self.alpha_t)
            self.n_scored += 1
        return err if scored else None

    def diagnostics(self) -> dict:
        return dict(alpha=self.alpha, gamma=self.gamma, n_scored=self.n_scored, n_alpha_low=self.n_low, n_alpha_high=self.n_high, n_inf_threshold=self.n_inf, n_empty_threshold=self.n_ninf, alpha_min=None if not np.isfinite(self.alpha_min) else float(self.alpha_min), alpha_max=None if not np.isfinite(self.alpha_max) else float(self.alpha_max))

def upper_rank(scen_losses, realized) -> float:
    s = np.asarray(scen_losses, float)
    return (1.0 + float(np.sum(s <= float(realized)))) / (len(s) + 1.0)

def rank_threshold(scen_losses, beta: float) -> float:
    if beta >= 1.0 or beta == np.inf:
        return np.inf
    s = np.sort(np.asarray(scen_losses, float))
    M = len(s)
    if beta == -np.inf or beta < 1.0 / (M + 1.0):
        return -np.inf
    cnt = np.searchsorted(s, s, side='right')
    Fh = (1.0 + cnt) / (M + 1.0)
    above = Fh > beta
    return float(s[np.argmax(above)]) if np.any(above) else np.inf
