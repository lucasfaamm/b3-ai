import math
import numpy as np
import pandas as pd


def first_present(d, aliases, default=np.nan):
    if not isinstance(d, dict):
        return default
    for key in aliases:
        if key in d and d[key] not in (None, ""):
            try:
                return float(d[key])
            except (TypeError, ValueError):
                return d[key]
    return default


def safe_div(a, b):
    try:
        if b is None or pd.isna(b) or float(b) == 0:
            return np.nan
        return float(a) / float(b)
    except Exception:
        return np.nan


def annualized_vol(returns):
    s = pd.Series(returns, dtype="float64").dropna()
    if len(s) < 20:
        return np.nan
    return float(s.std(ddof=1) * math.sqrt(252))


def max_drawdown(prices):
    s = pd.Series(prices, dtype="float64").dropna()
    if len(s) < 20:
        return np.nan
    return float((s / s.cummax() - 1.0).min())


def pct_return(prices, days):
    s = pd.Series(prices, dtype="float64").dropna()
    if len(s) <= days:
        return np.nan
    return float(s.iloc[-1] / s.iloc[-days - 1] - 1.0)
