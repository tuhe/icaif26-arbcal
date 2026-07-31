from __future__ import annotations
import argparse
import time
import warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings('ignore')
import lib_fullcal as fl
from arbcal import config as CFG
from arbcal import repricing as rp
from arbcal.polyhedron import polyhedron_rows, PolyProjector, violations, FAMILIES
from arbcal.scenarios import shrunk_cov

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--panel', required=True)
    ap.add_argument('--manifest', default=None)
    ap.add_argument('--phase', choices=['a', 'b'], required=True)
    ap.add_argument('--t-start', type=int, default=None)
    ap.add_argument('--t-end', type=int, default=None)
    ap.add_argument('--n-draws', type=int, default=None)
    ap.add_argument('--law', choices=['gauss', 't'], default='gauss')
    ap.add_argument('--nu', type=float, default=5.0)
    ap.add_argument('--seed-base', type=int, default=None)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    t0 = time.time()
    assert rp.EXTRAPOLATION == 'extend'
    seed_base = CFG.SEED_BASE if args.seed_base is None else args.seed_base
    n_draws = args.n_draws or (CFG.N_DRAWS if args.phase == 'a' else 10000)
    days, S, F, DISC, C = fl.load_panel(args.panel)
    manifest = fl.load_manifest(args.manifest)
    masks = fl.panel_masks(days, manifest)
    vt = masks['vt']
    D = len(days)
    t_start = max(args.t_start if args.t_start is not None else CFG.MIN_INNOV + 1, 1)
    t_end = min(args.t_end if args.t_end is not None else D, D)
    G, h, fam = polyhedron_rows()
    projector = PolyProjector(G, h, family=fam)
    t_idx, dates, cutoff = ([], [], [])
    nfail = []
    Rs, surfs, surfs_raw = ([], [], [])
    raw_vf = {f: [] for f in FAMILIES}
    proj_vf = {f: [] for f in FAMILIES}
    SL, proj_bad_any = ([], [])
    n_skipped_transition = 0
    for t in range(t_start, t_end):
        if not vt[t]:
            n_skipped_transition += 1
            continue
        innov, Yv_full = fl.gap_innovations(S, C, vt, t)
        E = innov[-CFG.LW_WINDOW:]
        if len(E) < CFG.MIN_INNOV:
            continue
        K = shrunk_cov(E)
        mu = np.concatenate([[0.0], Yv_full[-1]])
        rng = np.random.default_rng(seed_base + t)
        Lc = np.linalg.cholesky(K)
        eps = rng.standard_normal((n_draws, len(mu))) @ Lc.T
        if args.phase == 'b' and args.law == 't':
            W = rng.chisquare(args.nu, size=n_draws)
            eps = eps * np.sqrt((args.nu - 2.0) / W)[:, None]
        draws = mu[None, :] + eps
        R = draws[:, 0]
        Zraw = draws[:, 1:]
        surf = np.empty_like(Zraw)
        fails = 0
        if args.phase == 'a':
            raw_bad = {f: 0 for f in FAMILIES}
            proj_bad = {f: 0 for f in FAMILIES}
            for m in range(n_draws):
                vr = violations(Zraw[m], G, h, fam, tol=CFG.FEAS_TOL)
                for f in FAMILIES:
                    raw_bad[f] += int(vr.by_family[f] > 0)
                zx, ok = projector.project(Zraw[m])
                if zx is None:
                    fails += 1
                    zx = Zraw[m]
                elif not ok:
                    vp = violations(zx, G, h, fam, tol=CFG.FEAS_TOL)
                    for f in FAMILIES:
                        proj_bad[f] += int(vp.by_family[f] > 0)
                surf[m] = zx
            for f in FAMILIES:
                raw_vf[f].append(raw_bad[f] / n_draws)
                proj_vf[f].append(proj_bad[f] / n_draws)
            Rs.append(R.astype(np.float32))
            surfs.append(surf.reshape(n_draws, CFG.N_X, CFG.N_TAU).astype(np.float32))
            surfs_raw.append(Zraw.reshape(n_draws, CFG.N_X, CFG.N_TAU).astype(np.float32))
        else:
            bad = 0
            for m in range(n_draws):
                zx, ok = projector.project(Zraw[m])
                if zx is None:
                    fails += 1
                    zx = Zraw[m]
                elif not ok:
                    bad += 1
                surf[m] = zx
            surf3 = surf.reshape(n_draws, CFG.N_X, CFG.N_TAU)
            books = rp.build_books(S[t - 1], F[t - 1], DISC[t - 1], C[t - 1])
            St = S[t - 1] * np.exp(R)
            Fm, Dm = rp.scenario_forward_discount(St, S[t - 1], F[t - 1], DISC[t - 1])
            sl = np.empty((n_draws, fl.K_BOOKS), np.float32)
            for j, name in enumerate(fl.BOOKS):
                sl[:, j] = rp.book_loss_bps(books[name], St, Fm, Dm, surf3)
            SL.append(sl)
            proj_bad_any.append(bad)
        t_idx.append(t)
        dates.append(days[t])
        cutoff.append(days[t - 1])
        nfail.append(fails)
        if len(t_idx) % 25 == 0:
            el = time.time() - t0
            print(f'[fullcal_scen:{args.phase}] {len(t_idx)} dates ({el:.0f}s, {el / len(t_idx):.2f}s/date)', flush=True)
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    common = dict(t_idx=np.array(t_idx), dates=np.array(dates), cutoff_dates=np.array(cutoff), n_proj_fail=np.array(nfail), families=np.array(FAMILIES), n_draws=n_draws, seed_base=seed_base, lw_window=CFG.LW_WINDOW, extrapolation='extend', n_skipped_invalid_transition=n_skipped_transition, panel=str(args.panel), gap_aware=True)
    if args.phase == 'a':
        np.savez_compressed(outp, **common, R=np.stack(Rs) if Rs else np.zeros((0, n_draws), np.float32), surf=np.stack(surfs) if surfs else np.zeros((0, n_draws, CFG.N_X, CFG.N_TAU), np.float32), surf_raw=np.stack(surfs_raw) if surfs_raw else np.zeros((0, n_draws, CFG.N_X, CFG.N_TAU), np.float32), raw_viol_frac=np.array([[raw_vf[f][i] for f in FAMILIES] for i in range(len(t_idx))]), proj_viol_frac=np.array([[proj_vf[f][i] for f in FAMILIES] for i in range(len(t_idx))]))
    else:
        np.savez_compressed(outp, **common, SL=np.stack(SL) if SL else np.zeros((0, n_draws, fl.K_BOOKS), np.float32), n_proj_bad=np.array(proj_bad_any), books=np.array(fl.BOOKS), law=args.law, nu=args.nu if args.law == 't' else np.inf)
    print(f'[fullcal_scen:{args.phase}] wrote {outp} n_dates={len(t_idx)} skipped_invalid={n_skipped_transition} ({time.time() - t0:.0f}s)', flush=True)
if __name__ == '__main__':
    main()
