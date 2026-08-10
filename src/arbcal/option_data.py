from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass
class SmileGrid:
    ticker: str
    asof: pd.Timestamp
    expiry: pd.Timestamp
    T: float
    spot: float
    forward: float
    discount: float
    strike: np.ndarray
    mid: np.ndarray
    half_spread: np.ndarray
    log_moneyness: np.ndarray

def load_yfinance_chain(ticker: str, expiry: str | None=None):
    import yfinance as yf
    tk = yf.Ticker(ticker)
    expiries = tk.options
    if not expiries:
        raise RuntimeError(f'no option expiries returned for {ticker}')
    expiry = expiry or expiries[len(expiries) // 3]
    chain = tk.option_chain(expiry)
    hist = tk.history(period='1d')
    spot = float(hist['Close'].iloc[-1]) if len(hist) else float('nan')
    asof = pd.Timestamp.utcnow().normalize().tz_localize(None)
    return (chain.calls, chain.puts, spot, asof, pd.Timestamp(expiry))

def _forward_discount(calls: pd.DataFrame, puts: pd.DataFrame):
    merged = pd.merge(calls[['strike', '_mid']], puts[['strike', '_mid']], on='strike', suffixes=('_c', '_p'))
    merged = merged.dropna()
    if len(merged) < 3:
        return (np.nan, np.nan)
    K = merged['strike'].to_numpy()
    cp = (merged['_mid_c'] - merged['_mid_p']).to_numpy()
    slope, icpt = np.polyfit(K, cp, 1)
    discount = float(-slope)
    forward = float(icpt / discount) if discount > 0 else np.nan
    return (forward, discount)

def clean_chain(calls: pd.DataFrame, puts: pd.DataFrame, *, spot: float, asof: pd.Timestamp, expiry: pd.Timestamp, ticker: str, max_rel_spread: float=0.5, moneyness=(0.75, 1.25)) -> SmileGrid:

    def _prep(df):
        df = df.copy()
        df['_mid'] = 0.5 * (df['bid'] + df['ask'])
        ok = (df['bid'] > 0) & (df['ask'] > df['bid']) & (df['_mid'] > 0.001)
        rel = (df['ask'] - df['bid']) / df['_mid'].clip(lower=1e-06)
        ok &= rel < max_rel_spread
        if 'impliedVolatility' in df:
            ok &= df['impliedVolatility'].fillna(0) > 0
        return df[ok]
    calls_c, puts_c = (_prep(calls), _prep(puts))
    forward, discount = _forward_discount(calls_c, puts_c)
    if not np.isfinite(forward):
        forward, discount = (spot, 1.0)
    T = max((expiry - asof).days, 1) / 365.0
    lo, hi = moneyness
    sel = calls_c[(calls_c['strike'] >= lo * forward) & (calls_c['strike'] <= hi * forward)].sort_values('strike')
    K = sel['strike'].to_numpy()
    mid = sel['_mid'].to_numpy()
    half = 0.5 * (sel['ask'] - sel['bid']).to_numpy()
    return SmileGrid(ticker=ticker, asof=asof, expiry=expiry, T=T, spot=spot, forward=forward, discount=discount, strike=K, mid=mid, half_spread=half, log_moneyness=np.log(K / forward))

def fetch_smile(ticker: str='SPY', expiry: str | None=None, **clean_kw) -> SmileGrid:
    calls, puts, spot, asof, exp = load_yfinance_chain(ticker, expiry)
    return clean_chain(calls, puts, spot=spot, asof=asof, expiry=exp, ticker=ticker, **clean_kw)
