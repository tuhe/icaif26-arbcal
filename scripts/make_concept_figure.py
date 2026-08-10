from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
REPO = Path(__file__).resolve().parents[1]
OUT = REPO / 'figures/fig_concept_two_interfaces.pdf'
CONE = '#dfe9f5'
CONE_EDGE = '#5B9BD5'
RED = '#C0392B'
GREY = '#4a4a4a'
XMAX, YMAX = (4.4, 3.2)
POLY = np.array([[0.55, 1.3], [1.55, 0.62], [3.55, 1.15], [4.25, 2.05], [3.3, 2.95], [1.35, 2.8]])

def _edges():
    return [(POLY[i], POLY[(i + 1) % len(POLY)]) for i in range(len(POLY))]

def wedge_mask(x, y):
    inside = np.ones(np.shape(x), bool)
    for a, b in _edges():
        inside &= (b[0] - a[0]) * (y - a[1]) - (b[1] - a[1]) * (x - a[0]) >= 0
    return inside

def draw_cone(ax):
    ax.fill(POLY[:, 0], POLY[:, 1], color=CONE, zorder=0)
    ax.plot(np.r_[POLY[:, 0], POLY[0, 0]], np.r_[POLY[:, 1], POLY[0, 1]], c=CONE_EDGE, lw=1.2)
    ax.text(3.15, 2.98, '$\\mathcal{A}=\\{c:Gc\\geq h\\}$', fontsize=10, color=CONE_EDGE)

def proj_poly(z):
    zz = np.atleast_2d(np.asarray(z, float))
    out = zz.copy()
    for k, p in enumerate(zz):
        if bool(wedge_mask(p[0], p[1])):
            continue
        best, bd = (None, np.inf)
        for a, b in _edges():
            ab = b - a
            t = np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0.0, 1.0)
            q = a + t * ab
            d = np.linalg.norm(p - q)
            if d < bd:
                best, bd = (q, d)
        out[k] = best
    return out
proj_lower = proj_poly

def left_panel(ax):
    draw_cone(ax)
    shat = np.array([3.35, 0.82])
    p = proj_lower(shat)[0]
    y = np.array([1.85, 1.95])
    q_raw, q = (0.8, 0.62)
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(shat[0] + q_raw * np.cos(th), shat[1] + q_raw * np.sin(th), ls='--', c=GREY, lw=1.0)
    xx, yy = np.meshgrid(np.linspace(0, XMAX, 500), np.linspace(0, YMAX, 400))
    inside = ((xx - p[0]) ** 2 + (yy - p[1]) ** 2 <= q ** 2) & wedge_mask(xx, yy)
    ax.contourf(xx, yy, inside.astype(float), levels=[0.5, 1.5], colors=[RED], alpha=0.22, zorder=1)
    ax.plot(p[0] + q * np.cos(th), p[1] + q * np.sin(th), c=RED, lw=1.1)
    ax.annotate('', xy=p, xytext=shat, arrowprops=dict(arrowstyle='->', color=GREY, lw=1.2))
    ax.scatter(*shat, c=GREY, s=22, zorder=5)
    ax.scatter(*p, c=RED, s=26, zorder=5)
    ax.scatter(*y, c='k', marker='*', s=70, zorder=5)
    ax.text(shat[0] + 0.08, shat[1] - 0.22, '$\\hat Y_t\\notin\\mathcal{A}$', fontsize=9, color=GREY)
    ax.text(p[0] + 0.34, p[1] - 0.12, '$P_t=\\Pi_{\\mathcal{A}}(\\hat Y_t)$', fontsize=9, color=RED)
    ax.text(y[0] - 0.05, y[1] + 0.12, '$Y_t$', fontsize=10)
    ax.text(shat[0] - q_raw - 0.1, shat[1] + q_raw * 0.7, 'raw ball,\n$q_{\\mathrm{raw}}$', fontsize=8.2, color=GREY, ha='right')
    ax.text(p[0] - 0.95, p[1] + q + 0.14, '$\\mathcal{C}_t=B(P_t,q)\\cap\\mathcal{A}$,  $q\\leq q_{\\mathrm{raw}}$', fontsize=8.4, color=RED, ha='left')
    ax.set_title('(a) ArbCal-Set: coverage + admissibility', fontsize=9.5)

def right_panel(ax):
    draw_cone(ax)
    rng = np.random.default_rng(7)
    Z = rng.multivariate_normal([2.9, 1.3], [[0.16, 0.04], [0.04, 0.11]], 52)
    Z = Z[(Z[:, 0] > 0.3) & (Z[:, 1] > 0.05) & (Z[:, 0] < XMAX - 0.1)]
    out = ~wedge_mask(Z[:, 0], Z[:, 1])
    P = proj_poly(Z)
    ax.scatter(*Z[out].T, s=8, c=GREY, alpha=0.55, lw=0)
    ax.scatter(*P.T, s=8, c=RED, alpha=0.75, lw=0, zorder=4)
    for zi, pi in zip(Z[out][:14], P[out][:14]):
        ax.annotate('', xy=pi, xytext=zi, arrowprops=dict(arrowstyle='->', color=GREY, lw=0.5, alpha=0.6))
    ax.text(2.45, 0.46, '1 draw joint scenario:\n' + '$(R_t,Z_t)\\sim Q_t(\\,\\cdot\\mid\\mathcal{F}_{t-1})$', fontsize=8.2, color=GREY, va='top')
    ax.text(0.42, 1.52, '2 project surface', fontsize=8.6, color=RED)
    ins = ax.inset_axes([0.02, 0.62, 0.42, 0.34])
    losses = rng.gamma(2.2, 1.0, 4000) - 1.2
    ins.hist(losses, bins=42, color=RED, alpha=0.35, density=True)
    q_raw = np.quantile(losses, 0.9)
    q_cal = np.quantile(losses, 0.965)
    ins.axvline(q_raw, ls='--', c=GREY, lw=1)
    ins.axvline(q_cal, c=RED, lw=1.2)
    ins.text(q_raw - 0.12, ins.get_ylim()[1] * 0.96, 'raw $1{-}\\alpha$', fontsize=6.8, color=GREY, ha='right', va='top', rotation=90)
    ins.text(q_cal + 0.12, ins.get_ylim()[1] * 0.96, '4 calibrate\nmax-rank', fontsize=6.8, color=RED, ha='left', va='top', rotation=90, linespacing=1.1)
    ins.set_xticks([])
    ins.set_yticks([])
    ins.set_xlabel('3 reprice fixed books', fontsize=6.8, labelpad=1)
    ax.set_title('(b) ArbCal-Rank: admissible scenarios + calibrated limits', fontsize=9.5)

def main():
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 2.75))
    for ax in axes:
        ax.set_xlim(0, XMAX)
        ax.set_ylim(0, YMAX)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    left_panel(axes[0])
    right_panel(axes[1])
    fig.tight_layout(w_pad=2.5)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches='tight')
    fig.savefig(OUT.with_suffix('.png'), dpi=200, bbox_inches='tight')
    print(f'[fig] wrote {OUT}')
if __name__ == '__main__':
    main()
