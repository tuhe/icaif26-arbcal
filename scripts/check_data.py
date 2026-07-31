import argparse
import hashlib
import json
from pathlib import Path

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw', default='data/raw')
    ap.add_argument('--manifest', default='manifests/optionsdx_manifest.json')
    ap.add_argument('--fast', action='store_true')
    args = ap.parse_args()
    man = json.loads(Path(args.manifest).read_text())
    checks = man['checksums']
    raw = Path(args.raw)
    ok = missing = bad = 0
    for month, meta in sorted(checks.items()):
        cands = list(raw.rglob(meta['file']))
        if not cands:
            print(f'MISSING  {month}  {meta["file"]}')
            missing += 1
            continue
        p = cands[0]
        if p.stat().st_size != meta['bytes']:
            print(f'SIZE     {month}  {p} ({p.stat().st_size} != {meta["bytes"]})')
            bad += 1
            continue
        if not args.fast and sha256(p) != meta['sha256']:
            print(f'SHA256   {month}  {p}')
            bad += 1
            continue
        ok += 1
    print(f'checked {len(checks)} months: {ok} ok, {missing} missing, {bad} mismatched')
    print(f'expected sessions {man["n_expected_sessions"]}, retained in paper {man["n_retained"]}')
    raise SystemExit(0 if missing == bad == 0 else 1)

if __name__ == '__main__':
    main()
