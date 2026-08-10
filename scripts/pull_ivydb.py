from __future__ import annotations
import argparse
import os
from pathlib import Path
DEFAULT_TICKERS = ['SPX', 'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'NVDA', 'META', 'TSLA', 'JPM', 'BAC', 'XOM', 'WMT', 'JNJ', 'PG', 'KO', 'DIS']

def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--start', type=int, default=2019)
    p.add_argument('--end', type=int, default=2022)
    p.add_argument('--tickers', nargs='+', default=DEFAULT_TICKERS)
    p.add_argument('--lib', default='optionm_all')
    p.add_argument('--sample', default='wednesdays', choices=['wednesdays', 'monthend', 'all'])
    p.add_argument('--dte-max', type=int, default=400)
    p.add_argument('--username', default=os.environ.get('WRDS_USERNAME'))
    p.add_argument('--out', default='data/raw/ivydb_multiname')
    a = p.parse_args(argv)
    import pandas as pd
    import wrds
    os.environ.setdefault('PGUSER', a.username)
    db = wrds.Connection(wrds_username=a.username)
    dowf = {'wednesdays': 'and extract(dow from date) = 3', 'monthend': "and date = (date_trunc('month', date)+interval '1 month - 1 day')::date", 'all': ''}[a.sample]
    tick_list = ','.join((f"'{t}'" for t in a.tickers))
    nm = db.raw_sql(f'select secid, ticker, effect_date from {a.lib}.secnmd where ticker in ({tick_list})')
    nm = nm.sort_values('effect_date').groupby('ticker').tail(1)
    sec_map = {r.ticker: int(r.secid) for r in nm.itertuples()}
    secids = sorted(set(sec_map.values()))
    print('resolved secids:', sec_map)
    if not secids:
        print('[FAIL] no secids resolved')
        db.close()
        return 1
    sec_in = ','.join((str(s) for s in secids))
    OUT = Path(a.out)
    OUT.mkdir(parents=True, exist_ok=True)
    parts = []
    for yr in range(a.start, a.end + 1):
        tbl = f'{a.lib}.opprcd{yr}'
        try:
            q = f'select secid, date, exdate, cp_flag, strike_price, best_bid, best_offer, impl_volatility, volume, open_interest from {tbl} where secid in ({sec_in}) and (exdate - date) <= {a.dte_max} and best_bid > 0 and best_offer >= best_bid {dowf}'
            df = db.raw_sql(q)
            print(f"  {tbl}: {len(df):>9} rows, {df['secid'].nunique()} secids")
            parts.append(df)
        except Exception as e:
            print(f'  {tbl}: SKIP ({type(e).__name__}: {e})')
    if not parts:
        print('[FAIL] no option rows')
        db.close()
        return 1
    opt = pd.concat(parts, ignore_index=True)
    opt.to_parquet(OUT / 'options.parquet')
    print(f"[wrote] options.parquet  ({len(opt)} rows, {opt['secid'].nunique()} names, {opt['date'].nunique()} dates)")
    sec = db.raw_sql(f'select secid, date, close from {a.lib}.secprd where secid in ({sec_in}) and extract(year from date) between {a.start} and {a.end}')
    sec.to_parquet(OUT / 'secpx.parquet')
    print(f'[wrote] secpx.parquet    ({len(sec)} rows)')
    pd.DataFrame([{'ticker': t, 'secid': s} for t, s in sec_map.items()]).to_parquet(OUT / 'secid_map.parquet')
    db.close()
    print(f'\n[done] multi-name IvyDB panel at {OUT}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
