from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import lib_fullcal as fl

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--panel', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    z = np.load(args.panel, allow_pickle=True)
    days = [str(d) for d in z['days']]
    manifest = fl.load_manifest()
    keep = np.array([bool(int(manifest[d]['retained'])) if d in manifest else False for d in days])
    dropped = [d for d, k in zip(days, keep) if not k]
    kept = [d for d, k in zip(days, keep) if k]
    print(f'[sessions] panel {len(days)} days -> {len(kept)} retained; dropped {len(dropped)}: {dropped}')
    ref_days = sorted((d for d, m in manifest.items() if int(m['retained'])))
    if kept != ref_days:
        extra = sorted(set(kept) - set(ref_days))
        miss = sorted(set(ref_days) - set(kept))
        raise SystemExit(f'ANCHOR MISMATCH vs committed date manifest: +{extra[:5]} -{miss[:5]} (n={len(kept)} vs {len(ref_days)})')
    print(f'[sessions] anchor PASS: day list identical to the committed date manifest ({len(kept)} retained sessions)')
    out = {k: z[k][keep] if z[k].shape[:1] == (len(days),) else z[k] for k in z.files}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **out)
    print(f'[sessions] wrote {args.out}')
if __name__ == '__main__':
    main()
