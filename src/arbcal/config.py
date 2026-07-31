from __future__ import annotations
import numpy as np
X_GRID = np.round(np.arange(0.9, 1.1001, 0.01), 4)
TAUS = (14, 30, 60, 90, 180)
N_X = len(X_GRID)
N_TAU = len(TAUS)
DIM = N_X * N_TAU
MAX_DTE_RATIO = 0.4
MIN_QUOTES = 8
FIT_RIDGE = 0.0001
FEAS_TOL = 1e-08
PARITY_D_BOUNDS = (0.9, 1.02)
PARITY_BASIS_BOUNDS = (0.9, 1.1)
BOOK_TAU = 60
BOOK_NAMES = ('short_straddle', 'short_25d_put', 'dh_short_call', 'risk_reversal', 'short_strangle', 'long_butterfly')
LW_WINDOW = 250
MIN_INNOV = 60
N_DRAWS = 1000
SEED_BASE = 7000
ACI_WINDOW = 250
WARMUP = 60
ALPHAS = (0.1, 0.05, 0.01)

def GAMMA_RULE(alpha: float) -> float:
    return alpha / 5.0
GAMMA_SENSITIVITY = (10.0, 5.0, 2.0)
DEV_END = '2021-01-01'
BLOCK_LEN = 10
N_BOOT = 2000
BLOCK_LEN_SENSITIVITY = (5, 10, 20)
