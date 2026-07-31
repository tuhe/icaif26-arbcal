from __future__ import annotations
import argparse
import glob
import json
import warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import lib_fullcal as fl
from arbcal.aci import ExactACI, upper_rank, rank_threshold
ALPHA = 0.05
TRAIL = 250
C_ACI = '#1f5fa8'
C_ROLL = '#c2701e'

def rank_stream(alpha, D, L, mu, sd, SL_by_t, gamma):
    a = ExactACI(alpha, gamma)
    idx, errs, alphas = ([], [], [])
    for t in range(D):
        if t not in SL_by_t or not np.all(np.isfinite(mu[t])) or (not np.all(np.isfinite(L[t]))):
            continue
        SL = SL_by_t[t]
        u = np.array([upper_rank(SL[:, j], L[t, j]) for j in range(fl.K_BOOKS)])
        s = float(np.max(u))
        beta = a.threshold()
        if not a.in_warmup:
            if beta == np.inf or beta >= 1.0:
                Bs = np.full(fl.K_BOOKS, np.inf)
            elif beta == -np.inf:
                Bs = np.full(fl.K_BOOKS, -np.inf)
            else:
                Bs = np.array([rank_threshold(SL[:, j], beta) for j in range(fl.K_BOOKS)])
            idx.append(t)
            errs.append(float(np.any(L[t] > Bs)))
            alphas.append(a.alpha_t)
        a.update(s, beta)
    return (np.asarray(idx, int), np.asarray(errs), np.asarray(alphas))

def trailing(e, w=TRAIL):
    out = np.full(len(e), np.nan)
    c = np.concatenate([[0.0], np.cumsum(e)])
    out[w - 1:] = (c[w:] - c[:-w]) / w
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--panel', required=True)
    ap.add_argument('--losses-glob', required=True)
    ap.add_argument('--canonical', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    days, S, F, DISC, C = fl.load_panel(args.panel)
    masks = fl.panel_masks(days, fl.load_manifest())
    vt = masks['vt']
    D = len(days)
    gsel = {float(a_): g for a_, g in json.loads(Path(args.canonical).read_text())['meta']['gamma_selected'].items()}
    SL_by_t = {}
    for f in sorted(glob.glob(args.losses_glob)):
        c = np.load(f, allow_pickle=True)
        for row, t in enumerate(np.asarray(c['t_idx'], int)):
            SL_by_t[int(t)] = np.asarray(c['SL'][row], float)
    L = fl.realized_losses(days, S, F, DISC, C, vt)
    mu, sd = fl.load_vs_run().rolling_loc_scale(L)
    i_aci, e_aci, al_aci = rank_stream(ALPHA, D, L, mu, sd, SL_by_t, gsel[ALPHA])
    i_roll, e_roll, _ = rank_stream(ALPHA, D, L, mu, sd, SL_by_t, 0.0)
    assert np.array_equal(i_aci, i_roll)
    tr_aci = trailing(e_aci)
    tr_roll = trailing(e_roll)
    dts = np.array(days, dtype='datetime64[D]')[i_aci]
    show = (dts >= np.datetime64('2018-01-01')) & np.isfinite(tr_aci)
    x = dts[show].astype('datetime64[ms]').astype(object)
    fig, (axA, axB) = plt.subplots(2, 1, figsize=(3.6, 2.6), sharex=True, height_ratios=[1.15, 1.0])
    episodes = [('2020-02-15', '2020-04-15'), ('2022-01-01', '2022-07-01')]
    for ax in (axA, axB):
        for a_, b_ in episodes:
            ax.axvspan(np.datetime64(a_).astype('datetime64[ms]').item(), np.datetime64(b_).astype('datetime64[ms]').item(), color='0.88', zorder=0)
        ax.axvline(np.datetime64('2023-01-01').astype('datetime64[ms]').item(), color='0.35', lw=0.7, ls='-.', zorder=1)
    axA.plot(x, tr_roll[show], lw=1.0, color=C_ROLL, label='rolling max-rank')
    axA.plot(x, tr_aci[show], lw=1.1, color=C_ACI, label='ArbCal-Rank (ACI)')
    axA.axhline(ALPHA, color='k', lw=0.7, ls=':')
    axA.set_ylabel(f'trail. FWE ({TRAIL}d)')
    axA.legend(frameon=False, fontsize=7, loc='upper left')
    axA.text(np.datetime64('2023-07-01').astype('datetime64[ms]').item(), axA.get_ylim()[1], '2023\nvalidation', fontsize=6, ha='center', va='top', color='0.35')
    axB.plot(x, al_aci[show], lw=0.9, color=C_ACI)
    axB.axhline(ALPHA, color='k', lw=0.7, ls=':')
    axB.set_ylabel('adaptive level $\\alpha_t$')
    axB.xaxis.set_major_locator(mdates.YearLocator())
    axB.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    for ax in (axA, axB):
        ax.tick_params(labelsize=7)
        ax.yaxis.label.set_size(8)
    fig.tight_layout(h_pad=0.4)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches='tight')
    print(f"[fig_online_t3_95] wrote {args.out}: n={show.sum()} plotted, trailing FWE 2023 mean aci={np.nanmean(tr_aci[dts >= np.datetime64('2023-01-01')]):.3f} roll={np.nanmean(tr_roll[dts >= np.datetime64('2023-01-01')]):.3f}")
if __name__ == '__main__':
    main()
