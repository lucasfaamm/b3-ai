
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
import requests
import vectorbt as vbt
import yfinance as yf

OUT = Path("results"); OUT.mkdir(exist_ok=True)
SUM = Path("results/vectorbt_summary.json")
FUND = Path("data/fundamentals.csv")

if not SUM.exists() or not FUND.exists():
    raise SystemExit("Faltam vectorbt_summary.json ou fundamentals.csv")

base = json.loads(SUM.read_text(encoding="utf-8"))
params = base["best_parameters"]
split = pd.Timestamp(base["train_test_split"])

fund = pd.read_csv(FUND)
tickers = fund["ticker"].dropna().astype(str).str.upper().drop_duplicates().tolist()[:120]
symbols = [f"{t}.SA" for t in tickers] + ["^BVSP"]

raw = yf.download(
    symbols, period="5y", interval="1d", auto_adjust=True,
    progress=False, threads=True, group_by="ticker"
)

def close(sym):
    try:
        d = raw[sym]
        c = d["Close"]
        if isinstance(c, pd.DataFrame):
            c = c.iloc[:,0]
        c = pd.to_numeric(c, errors="coerce")
        c.index = pd.to_datetime(c.index).tz_localize(None)
        return c.sort_index()
    except Exception:
        return pd.Series(dtype=float)

panel = pd.DataFrame({t:close(f"{t}.SA") for t in tickers}).sort_index()
panel = panel.dropna(axis=1, thresh=max(400, int(len(panel)*.55))).ffill(limit=3)
ibov = close("^BVSP").reindex(panel.index).ffill()

fast = int(params["fast_ma"]); slow = int(params["slow_ma"]); mom = int(params["momentum_days"])
fast_ma = panel.rolling(fast, min_periods=fast).mean()
slow_ma = panel.rolling(slow, min_periods=slow).mean()
momentum = panel / panel.shift(mom) - 1

entries = ((fast_ma > slow_ma) & (momentum > 0) & (panel > slow_ma)).fillna(False)
exits = ((fast_ma < slow_ma) | (momentum < 0)).fillna(False)

pf = vbt.Portfolio.from_signals(
    panel, entries, exits,
    init_cash=100000, fees=.0005, slippage=.0005, freq="1D"
)
r = pf.returns()
if isinstance(r, pd.Series):
    r = r.to_frame()
strategy = r.mean(axis=1, skipna=True).fillna(0.0)
strategy = strategy[strategy.index >= split]

ibov_ret = ibov.pct_change().fillna(0)
ibov_ret = ibov_ret[ibov_ret.index >= split].reindex(strategy.index).fillna(0)

def cdi_daily(start,end):
    try:
        url="https://api.bcb.gov.br/dados/serie/bcdata.sgs.12/dados"
        p={"formato":"json","dataInicial":start.strftime("%d/%m/%Y"),"dataFinal":end.strftime("%d/%m/%Y")}
        rr=requests.get(url,params=p,timeout=20);rr.raise_for_status()
        d=pd.DataFrame(rr.json())
        d["date"]=pd.to_datetime(d["data"],dayfirst=True,errors="coerce")
        d["rate"]=pd.to_numeric(d["valor"].astype(str).str.replace(",",".",regex=False),errors="coerce")
        s=d.dropna(subset=["date","rate"]).set_index("date")["rate"]/100.0
        # Série 12 é taxa diária percentual.
        return s.reindex(strategy.index).fillna(0)
    except Exception as e:
        print(f"[WARN] CDI indisponível: {e}")
        return pd.Series(np.nan,index=strategy.index)

cdi_ret=cdi_daily(strategy.index.min(),strategy.index.max())

def metrics(s):
    s=pd.Series(s,dtype=float).dropna()
    if len(s)<30:return {}
    wealth=(1+s).cumprod()
    years=len(s)/252
    cagr=float(wealth.iloc[-1]**(1/years)-1) if years>0 else np.nan
    vol=float(s.std(ddof=1)*np.sqrt(252))
    sharpe=float(s.mean()*252/vol) if vol>0 else np.nan
    dd=float((wealth/wealth.cummax()-1).min())
    return {
        "days":int(len(s)),"total_return":float(wealth.iloc[-1]-1),
        "cagr":cagr,"volatility":vol,"sharpe_rf0":sharpe,"max_drawdown":dd
    }

ms=metrics(strategy); mi=metrics(ibov_ret); mc=metrics(cdi_ret.dropna()) if cdi_ret.notna().any() else {}

accepted = (
    bool(ms) and bool(mi)
    and ms["cagr"] > mi["cagr"]
    and ms["sharpe_rf0"] > mi["sharpe_rf0"]
)
if mc:
    accepted = accepted and ms["cagr"] > mc["cagr"]

out={
    "period_start":str(strategy.index.min().date()),
    "period_end":str(strategy.index.max().date()),
    "best_parameters":params,
    "strategy":ms,"ibov":mi,"cdi":mc,
    "beats_ibov_cagr": bool(ms and mi and ms["cagr"]>mi["cagr"]),
    "beats_ibov_sharpe": bool(ms and mi and ms["sharpe_rf0"]>mi["sharpe_rf0"]),
    "beats_cdi_cagr": (bool(ms and mc and ms["cagr"]>mc["cagr"]) if mc else None),
    "accepted_as_confirmation_engine":bool(accepted),
    "rule":"Accepted only if OOS CAGR and Sharpe beat IBOV, and CAGR beats CDI when CDI is available."
}
(OUT/"vectorbt_benchmark.json").write_text(json.dumps(out,ensure_ascii=False,indent=2,default=str),encoding="utf-8")
print(json.dumps(out,ensure_ascii=False,indent=2,default=str))
