
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
import requests
import yfinance as yf

PIT=Path("data/cvm_pit_fundamentals.csv.gz")
OUT=Path("results"); OUT.mkdir(exist_ok=True)
if not PIT.exists():
    raise SystemExit("Falta data/cvm_pit_fundamentals.csv.gz")

pit=pd.read_csv(PIT,parse_dates=["reference_date","received_date"])
tickers=pit["ticker"].dropna().astype(str).str.upper().drop_duplicates().tolist()

raw=yf.download(
    [f"{t}.SA" for t in tickers]+["^BVSP"],
    period="7y",interval="1d",auto_adjust=True,progress=False,threads=True,group_by="ticker"
)

def close(sym):
    try:
        x=raw[sym]["Close"]
        if isinstance(x,pd.DataFrame):x=x.iloc[:,0]
        x=pd.to_numeric(x,errors="coerce");x.index=pd.to_datetime(x.index).tz_localize(None)
        return x.sort_index()
    except Exception:return pd.Series(dtype=float)

prices={t:close(f"{t}.SA") for t in tickers}
ibov=close("^BVSP")
all_dates=ibov.index
rebalance=pd.Series(all_dates,index=all_dates).groupby(all_dates.to_period("M")).max().values
rebalance=pd.DatetimeIndex(rebalance)
rebalance=rebalance[rebalance>=pd.Timestamp("2021-01-01")]

def px(t,dt):
    s=prices.get(t,pd.Series(dtype=float))
    s=s[s.index<=dt]
    return float(s.iloc[-1]) if len(s) else np.nan

def history(t,dt,n):
    s=prices.get(t,pd.Series(dtype=float));s=s[s.index<=dt].dropna()
    return s.tail(n)

def percentile(s,positive=True):
    r=s.rank(pct=True,method="average")*100
    return r if positive else 100-r

def score_components(x):
    out=x.copy()
    # winsorize cross-section before ranks
    numeric=[
        "roe","operating_margin","profit_margin","debt_to_equity",
        "revenue_growth_yoy","earnings_growth_yoy",
        "earnings_yield","book_to_price","ret_6m","ret_12m","volatility","max_drawdown"
    ]
    for c in numeric:
        if c in out:
            lo=out[c].quantile(.02);hi=out[c].quantile(.98)
            out[c]=out[c].clip(lo,hi)

    out["quality_score"]=pd.concat([
        percentile(out["roe"]),percentile(out["operating_margin"]),
        percentile(out["profit_margin"]),percentile(out["debt_to_equity"],False)
    ],axis=1).mean(axis=1)
    out["growth_score"]=pd.concat([
        percentile(out["revenue_growth_yoy"]),percentile(out["earnings_growth_yoy"])
    ],axis=1).mean(axis=1)
    out["valuation_score"]=pd.concat([
        percentile(out["earnings_yield"]),percentile(out["book_to_price"])
    ],axis=1).mean(axis=1)
    out["momentum_score"]=pd.concat([
        percentile(out["ret_6m"]),percentile(out["ret_12m"])
    ],axis=1).mean(axis=1)
    out["risk_score"]=pd.concat([
        percentile(out["volatility"],False),percentile(out["max_drawdown"])
    ],axis=1).mean(axis=1)
    return out

frames=[]
for dt in rebalance:
    available=pit[pit["received_date"]<=dt].sort_values(["ticker","reference_date","received_date"])
    snap=available.groupby("ticker",as_index=False).tail(1).copy()
    if len(snap)<20:continue
    rows=[]
    for _,r in snap.iterrows():
        t=r["ticker"];p=px(t,dt)
        h=history(t,dt,260)
        if pd.isna(p) or len(h)<200:continue
        shares=r.get("shares_outstanding",np.nan)
        market_cap=p*shares if not pd.isna(shares) and shares>0 else np.nan
        ni=r.get("net_income_annualized",np.nan);rev=r.get("revenue_annualized",np.nan)
        op=r.get("operating_income_annualized",np.nan);eq=r.get("equity",np.nan)
        debt=r.get("debt",np.nan)
        ret6=float(h.iloc[-1]/h.iloc[-127]-1) if len(h)>=127 else np.nan
        ret12=float(h.iloc[-1]/h.iloc[-253]-1) if len(h)>=253 else np.nan
        daily=h.pct_change().dropna()
        vol=float(daily.tail(63).std()*np.sqrt(252)) if len(daily)>=63 else np.nan
        dd=float((h/h.cummax()-1).tail(252).min())
        rows.append({
            "date":dt,"ticker":t,"sector":r.get("sector","Unknown"),"price":p,
            "roe":ni/eq if eq and not pd.isna(eq) else np.nan,
            "operating_margin":op/rev if rev and not pd.isna(rev) else np.nan,
            "profit_margin":ni/rev if rev and not pd.isna(rev) else np.nan,
            "debt_to_equity":debt/eq if eq and not pd.isna(eq) else np.nan,
            "revenue_growth_yoy":r.get("revenue_growth_yoy",np.nan),
            "earnings_growth_yoy":r.get("earnings_growth_yoy",np.nan),
            "earnings_yield":ni/market_cap if market_cap and not pd.isna(market_cap) else np.nan,
            "book_to_price":eq/market_cap if market_cap and not pd.isna(market_cap) else np.nan,
            "ret_6m":ret6,"ret_12m":ret12,"volatility":vol,"max_drawdown":dd,
        })
    x=pd.DataFrame(rows)
    if len(x)<20:continue
    x=score_components(x)
    frames.append(x)

panel=pd.concat(frames,ignore_index=True)
dates=pd.Index(sorted(panel["date"].unique()))

# Forward one-month returns
next_map={dates[i]:dates[i+1] for i in range(len(dates)-1)}
def fwd(t,dt):
    if dt not in next_map:return np.nan
    n=next_map[dt]
    p0=px(t,dt);p1=px(t,n)
    return p1/p0-1 if p0 and not pd.isna(p0) and not pd.isna(p1) else np.nan
panel["fwd_return"]=[fwd(t,d) for t,d in zip(panel["ticker"],panel["date"])]

ibov_prices=ibov
def ibret(dt):
    if dt not in next_map:return np.nan
    n=next_map[dt]
    a=ibov_prices[ibov_prices.index<=dt];b=ibov_prices[ibov_prices.index<=n]
    return float(b.iloc[-1]/a.iloc[-1]-1) if len(a) and len(b) else np.nan
ibmap={d:ibret(d) for d in dates}
panel["ibov_return"]=panel["date"].map(ibmap)
panel["alpha"]=panel["fwd_return"]-panel["ibov_return"]

components=["quality_score","growth_score","valuation_score","momentum_score","risk_score"]

# deterministic candidate weights: current + archetypes + Dirichlet
rng=np.random.default_rng(42)
candidates=[
    np.array([.25,.20,.20,.20,.15]),
    np.array([.35,.15,.25,.15,.10]),
    np.array([.20,.15,.35,.20,.10]),
    np.array([.20,.15,.15,.35,.15]),
    np.array([.20,.20,.20,.20,.20]),
]
candidates += list(rng.dirichlet(np.ones(5)*2.0,350))

n=len(dates)
train_dates=dates[:int(n*.55)]
valid_dates=dates[int(n*.55):int(n*.75)]
test_dates=dates[int(n*.75):]

def evaluate(weights,use_dates):
    rows=[]
    for dt in use_dates:
        g=panel[panel["date"]==dt].dropna(subset=["fwd_return"]).copy()
        if len(g)<15:continue
        g["score"]=sum(g[c]*w for c,w in zip(components,weights))
        top=g.sort_values("score",ascending=False).head(min(10,len(g)))
        rows.append({
            "date":dt,"ret":float(top["fwd_return"].mean()),
            "ibov":float(top["ibov_return"].mean()),
            "alpha":float(top["alpha"].mean())
        })
    z=pd.DataFrame(rows)
    if len(z)<6:return None
    a=z["alpha"]
    ann_ret=(1+z["ret"]).prod()**(12/len(z))-1
    ann_ib=(1+z["ibov"]).prod()**(12/len(z))-1
    sharpe=float(a.mean()/a.std(ddof=1)*np.sqrt(12)) if a.std(ddof=1)>0 else -999
    dd=float(((1+z["ret"]).cumprod()/((1+z["ret"]).cumprod().cummax())-1).min())
    return {"months":len(z),"avg_alpha":float(a.mean()),"alpha_hit_rate":float((a>0).mean()),
            "alpha_sharpe":sharpe,"cagr":float(ann_ret),"ibov_cagr":float(ann_ib),"max_drawdown":dd}

train_scores=[]
for i,w in enumerate(candidates):
    m=evaluate(w,train_dates)
    if m:
        objective=m["alpha_sharpe"]+.5*m["avg_alpha"]*12
        train_scores.append((objective,i,m))
train_scores.sort(reverse=True,key=lambda x:x[0])

# Only top 25 train candidates compete in validation.
valid_scores=[]
for _,i,tm in train_scores[:25]:
    vm=evaluate(candidates[i],valid_dates)
    if vm:
        objective=vm["alpha_sharpe"]+.5*vm["avg_alpha"]*12
        valid_scores.append((objective,i,tm,vm))
valid_scores.sort(reverse=True,key=lambda x:x[0])

if not valid_scores:
    raise SystemExit("Calibração sem candidato válido")

_,best_i,train_m,valid_m=valid_scores[0]
best=np.array(candidates[best_i])
test_m=evaluate(best,test_dates)

accepted=bool(
    test_m
    and test_m["avg_alpha"]>0
    and test_m["alpha_hit_rate"]>=.50
    and test_m["cagr"]>test_m["ibov_cagr"]
    and test_m["alpha_sharpe"]>0
)

weights={k:float(v) for k,v in zip(["quality","growth","valuation","momentum","risk"],best)}
summary={
    "weights":weights,"accepted":accepted,
    "train":train_m,"validation":valid_m,"test":test_m,
    "train_period":[str(pd.Timestamp(train_dates.min()).date()),str(pd.Timestamp(train_dates.max()).date())],
    "validation_period":[str(pd.Timestamp(valid_dates.min()).date()),str(pd.Timestamp(valid_dates.max()).date())],
    "test_period":[str(pd.Timestamp(test_dates.min()).date()),str(pd.Timestamp(test_dates.max()).date())],
    "method":"350 deterministic Dirichlet candidates + predefined profiles; select on train then validation; final test untouched.",
    "limitations":["Point-in-time filing availability is controlled by CVM received_date.",
                   "Current listed ticker universe still causes survivorship bias.",
                   "Historical market cap uses reported total shares times the queried share-class price, an approximation for multi-class issuers."]
}
(OUT/"calibrated_weights.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2,default=str),encoding="utf-8")
panel.to_csv(OUT/"fundamental_pit_backtest_panel.csv.gz",index=False,compression="gzip")
print(json.dumps(summary,ensure_ascii=False,indent=2,default=str))
