from __future__ import annotations
import argparse
import csv
import glob
import hashlib
import json
import os
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
START, END = ('2017-01-01', '2023-12-31')

def month_of(date: str) -> str:
    return date[:4] + date[5:7]

def archive_month_map(archive_dir):
    out = {}
    for a in sorted(glob.glob(os.path.join(archive_dir, 'spx_eod_*.7z'))):
        txt = subprocess.run(['7z', 'l', a], capture_output=True, text=True)
        for line in txt.stdout.splitlines():
            line = line.strip()
            if line.endswith('.txt') and 'spx_eod_' in line:
                name = line.split()[-1]
                ym = name.replace('spx_eod_', '').replace('.txt', '')
                out.setdefault(ym, os.path.basename(a))
    return out

def dates_in_month_file(path):
    cnt = Counter()
    with open(path) as f:
        next(f)
        for line in f:
            parts = line.split(',', 3)
            if len(parts) >= 3:
                cnt[parts[2].strip()] += 1
    return cnt

def sha256_file(path, chunk=1 << 20):
    hsh = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            hsh.update(b)
    return hsh.hexdigest()

def classify_drop(date, data_dir):
    from arbcal import config as CFG
    from arbcal.panel import fit_day, _ridge_matrix, pick_expiries
    from arbcal.polyhedron import polyhedron_rows
    from arbcal.optionsdx import read_month
    ym = month_of(date)
    df = read_month(os.path.join(data_dir, f'spx_eod_{ym}.txt'))
    day = df[df['QUOTE_DATE'].astype(str).str.strip() == date]
    if not len(day):
        return 'missing_from_archive'
    spot = float(day['UNDERLYING_LAST'].iloc[0])
    if not np.isfinite(spot) or spot <= 0:
        return 'bad_spot'
    groups = {}
    for exp, g in day.groupby('EXPIRE_DATE'):
        dte = float(g['DTE'].iloc[0])
        if np.isfinite(dte) and dte >= 5:
            groups[exp] = (dte, g)
    if not groups:
        return 'no_expiries_dte_ge_5'
    if pick_expiries(groups, 'independent') is None:
        return 'no_feasible_maturity_assignment'
    G, h, family = polyhedron_rows()
    out = fit_day(day, date, G, h, family, _ridge_matrix())
    if out is None:
        return 'cleaning_failed_quotes_or_fit'
    return 'UNEXPECTED_fit_succeeds_on_rerun'

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--archive-dir', required=True)
    ap.add_argument('--panel', required=True)
    ap.add_argument('--outdir', required=True)
    args = ap.parse_args()
    t0 = time.time()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    import exchange_calendars as xc
    cal = xc.get_calendar('XNYS')
    sessions = [str(s.date()) for s in cal.sessions_in_range(START, END)]
    arch_map = archive_month_map(args.archive_dir)
    month_files = sorted(glob.glob(os.path.join(args.data_dir, 'spx_eod_*.txt')))
    avail = {}
    per_month_rows = {}
    checksums = {}
    for mf in month_files:
        ym = Path(mf).name.replace('spx_eod_', '').replace('.txt', '')
        cnt = dates_in_month_file(mf)
        per_month_rows[ym] = {'n_dates': len(cnt), 'n_rows': int(sum(cnt.values()))}
        checksums[ym] = {'file': str(Path(mf).resolve().name), 'sha256': sha256_file(os.path.realpath(mf)), 'bytes': os.path.getsize(os.path.realpath(mf))}
        for d, n in cnt.items():
            avail[d] = (ym, int(n))
    z = np.load(args.panel, allow_pickle=True)
    retained = set((str(d) for d in z['days']))
    drop_reason = {}
    for d in sessions:
        if d in avail and d not in retained:
            drop_reason[d] = classify_drop(d, args.data_dir)
            print(f'[manifest] dropped {d}: {drop_reason[d]}', flush=True)
    rows = []
    prev = None
    prev_valid = None
    for d in sessions:
        ym = month_of(d)
        is_avail = d in avail
        is_ret = d in retained
        reason = '' if is_ret else drop_reason.get(d, 'missing_from_archive')
        valid_trans = bool(is_ret and prev is not None and (prev_valid == prev))
        rows.append({'date': d, 'archive': arch_map.get(ym, ''), 'source_file': f'spx_eod_{ym}.txt' if is_avail else '', 'n_raw_rows': avail[d][1] if is_avail else 0, 'available': int(is_avail), 'cleaning_passed': int(is_ret), 'drop_reason': reason, 'retained': int(is_ret), 'previous_expected_trading_date': prev or '', 'valid_one_session_transition': int(valid_trans)})
        if is_ret:
            prev_valid = d
        prev = d
    with open(outdir / 'date_manifest.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    n_avail = sum((r['available'] for r in rows))
    n_ret = sum((r['retained'] for r in rows))
    n_validt = sum((r['valid_one_session_transition'] for r in rows))
    by_year = defaultdict(lambda: [0, 0, 0])
    for r in rows:
        y = r['date'][:4]
        by_year[y][0] += 1
        by_year[y][1] += r['retained']
        by_year[y][2] += r['valid_one_session_transition']
    audit = {'calendar': 'XNYS', 'start': START, 'end': END, 'n_expected_sessions': len(rows), 'n_available': n_avail, 'n_retained': n_ret, 'n_valid_transitions': n_validt, 'n_months': len(month_files), 'months_expected': 84, 'missing_expected_sessions': [r['date'] for r in rows if not r['available']], 'dropped_by_cleaning': {d: drop_reason[d] for d in sorted(drop_reason)}, 'per_year': {y: {'expected': v[0], 'retained': v[1], 'valid_transitions': v[2]} for y, v in sorted(by_year.items())}, 'per_month_rows': per_month_rows, 'checksums': checksums, 'elapsed_s': round(time.time() - t0, 1)}
    (outdir / 'manifest_audit.json').write_text(json.dumps(audit, indent=1))
    print(f'[manifest] {len(rows)} expected sessions, {n_avail} available, {n_ret} retained, {n_validt} valid transitions ({time.time() - t0:.0f}s)')
if __name__ == '__main__':
    main()
