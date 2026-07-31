from __future__ import annotations
import glob
import os
from pathlib import Path
import numpy as np
import pandas as pd
from .option_data import clean_chain, SmileGrid

def _norm_cols(cols) -> list[str]:
    return [str(c).strip().strip('[]').strip() for c in cols]
_NEED = ['QUOTE_DATE', 'UNDERLYING_LAST', 'EXPIRE_DATE', 'DTE', 'STRIKE', 'C_BID', 'C_ASK', 'C_IV', 'P_BID', 'P_ASK', 'P_IV']

def read_month(path: str | Path, *, usecols=None) -> pd.DataFrame:
    uc = None
    if usecols is not None:
        want = set(usecols)
        uc = lambda c: _norm_cols([c])[0] in want
    df = pd.read_csv(path, dtype=str, skipinitialspace=True, encoding='utf-8-sig', usecols=uc)
    df.columns = _norm_cols(df.columns)
    obj = df.select_dtypes(include='object').columns
    df[obj] = df[obj].apply(lambda s: s.str.strip())
    num = ['UNDERLYING_LAST', 'DTE', 'STRIKE', 'C_BID', 'C_ASK', 'C_IV', 'P_BID', 'P_ASK', 'P_IV']
    for c in num:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df

def available_dates(data_dir: str | Path) -> list[str]:
    dates: set[str] = set()
    for f in sorted(glob.glob(os.path.join(str(data_dir), '*.txt'))):
        d = pd.read_csv(f, usecols=lambda c: _norm_cols([c])[0] == 'QUOTE_DATE', dtype=str, skipinitialspace=True)
        dates.update(d.iloc[:, 0].astype(str).str.strip().unique())
    return sorted(dates)

def _smiles_from_day(day: pd.DataFrame, asof: str, *, ticker: str, min_dte: float, max_dte: float, min_strikes: int, max_rel_spread: float, moneyness) -> list[SmileGrid]:
    spot = float(day['UNDERLYING_LAST'].iloc[0])
    smiles = []
    for exp, g in day.groupby('EXPIRE_DATE'):
        dte = float(g['DTE'].iloc[0])
        if not min_dte <= dte <= max_dte:
            continue
        calls = pd.DataFrame({'strike': g['STRIKE'], 'bid': g['C_BID'], 'ask': g['C_ASK'], 'impliedVolatility': g['C_IV']})
        puts = pd.DataFrame({'strike': g['STRIKE'], 'bid': g['P_BID'], 'ask': g['P_ASK'], 'impliedVolatility': g['P_IV']})
        try:
            sg = clean_chain(calls, puts, spot=spot, asof=pd.Timestamp(asof), expiry=pd.Timestamp(exp), ticker=ticker, max_rel_spread=max_rel_spread, moneyness=moneyness)
        except Exception:
            continue
        if len(sg.strike) >= min_strikes and sg.T > 0.001:
            smiles.append(sg)
    smiles.sort(key=lambda s: s.T)
    return smiles

def load_spx_surface(asof: str, data_dir: str | Path, *, ticker: str='^SPX', min_dte: float=7.0, max_dte: float=365.0, min_strikes: int=6, max_rel_spread: float=0.5, moneyness=(0.85, 1.15), _month_df: pd.DataFrame | None=None) -> list[SmileGrid]:
    if _month_df is None:
        ym = pd.Timestamp(asof).strftime('%Y%m')
        matches = glob.glob(os.path.join(str(data_dir), f'*{ym}*.txt'))
        if not matches:
            raise FileNotFoundError(f'no optionsDX file for {ym} in {data_dir}')
        _month_df = read_month(matches[0])
    day = _month_df[_month_df['QUOTE_DATE'].astype(str).str.strip() == str(asof)]
    if day.empty:
        raise ValueError(f'no rows for QUOTE_DATE={asof}')
    return _smiles_from_day(day, asof, ticker=ticker, min_dte=min_dte, max_dte=max_dte, min_strikes=min_strikes, max_rel_spread=max_rel_spread, moneyness=moneyness)

def iter_month_surfaces(path: str | Path, **kw):
    df = read_month(path)
    for asof in sorted(df['QUOTE_DATE'].astype(str).str.strip().unique()):
        yield (asof, load_spx_surface(asof, None, _month_df=df, **kw))
