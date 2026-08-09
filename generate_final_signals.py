
from __future__ import annotations
from pathlib import Path
import json, math
import numpy as np
import pandas as pd
import yfinance as yf

OUT=Path("results"); OUT.mkdir(exist_ok=True)
rank=pd.read_csv("results/ranking.csv")

default={"quality":.25,"growth":.20,"valuation":.20,"momentum":.20,"risk":.15}
cw=default.copy(); calibration_ok=False
p=Path("results/calibrated_weights.json")
if p.exists():
    x=json.loads(p.read_text(encoding="utf-8"))
    if x.get("accepted"):
        cw=x["weights"]; calibration_ok=True

# Base current score using calibrated component weights only if OOS accepted.
rank["calibrated_score"]=0.0
for k,w in cw.items():
    c=f"{k}_score"
    if c in rank:
        rank["calibrated_score"] += pd.to_numeric(rank[c],errors="coerce").fillna(0)*float(w)

# Technical engine benchmark validity
vb_ok=False;vb_params=None
p=Path("results/vectorbt_benchmark.json")
if p.exists():
    v=json.loads(p.read_text(encoding="utf-8"))
    vb_ok=bool(v.get("accepted_as_confirmation_engine",False))
    vb_params=v.get("best_parameters")

# Current technical confirmation
tickers=rank["ticker"].dropna().astype(str).str.upper().tolist()
raw=yf.download(
    [f"{t}.SA" for t in tickers]+["^BVSP","^VIX","BRL=X"],
    period="2y",interval="1d",auto_adjust=True,progress=False,threads=True,group_by="ticker"
)
def close(sym):
    try:
        x=raw[sym]["Close"]
        if isinstance(x,pd.DataFrame):x=x.iloc[:,0]
        return pd.to_numeric(x,errors="coerce").dropna()
    except Exception:return pd.Series(dtype=float)

technical={}
for t in tickers:
    s=close(f"{t}.SA")
    if len(s)<253:
        technical[t]=False;continue
    if vb_params:
        f=int(vb_params["fast_ma"]);sl=int(vb_params["slow_ma"]);m=int(vb_params["momentum_days"])
    else:
        f,sl,m=20,200,252
    ok=(len(s)>=max(sl,m)+1 and s.tail(f).mean()>s.tail(sl).mean() and s.iloc[-1]/s.iloc[-m-1]-1>0 and s.iloc[-1]>s.tail(sl).mean())
    technical[t]=bool(ok)
rank["technical_confirm"]=rank["ticker"].map(technical).fillna(False)

# Market regime
ib=close("^BVSP");vix=close("^VIX");usd=close("BRL=X")
regime_score=50
if len(ib)>=200:regime_score += 20 if ib.iloc[-1]>ib.tail(200).mean() else -20
if len(vix):regime_score += 10 if vix.iloc[-1]<20 else (-15 if vix.iloc[-1]>30 else 0)
if len(usd)>=64:
    r=usd.iloc[-1]/usd.iloc[-64]-1
    regime_score += -10 if r>.08 else (5 if r<-.05 else 0)
regime_score=float(np.clip(regime_score,0,100))
regime="RISK_ON" if regime_score>=65 else ("RISK_OFF" if regime_score<=40 else "NEUTRAL")

# Lightweight news risk only for top 30 calibrated candidates.
NEG=["fraud","fraude","investigation","investigação","default","bankruptcy","recuperação judicial",
     "downgrade","loss warning","guidance cut","multa","corruption","corrupção"]
rank["news_risk"]=False
rank["news_headlines"]=""
for idx in rank.sort_values("calibrated_score",ascending=False).head(30).index:
    q=str(rank.at[idx,"company_name"]) if "company_name" in rank.columns else str(rank.at[idx,"ticker"])
    try:
        sr=yf.Search(q,news_count=8)
        titles=[]
        for item in getattr(sr,"news",[]) or []:
            c=item.get("content") if isinstance(item,dict) else None
            title=(c.get("title") if isinstance(c,dict) else item.get("title","")) if isinstance(item,dict) else ""
            if title:titles.append(title)
        txt=" | ".join(titles).lower()
        rank.at[idx,"news_risk"]=any(k in txt for k in NEG)
        rank.at[idx,"news_headlines"]=" || ".join(titles[:4])
    except Exception:
        pass

# Gates
def num(r,c,default=0):
    try:
        v=float(r.get(c,default));return v if np.isfinite(v) else default
    except:return default

def decide(r):
    gates={
        "calibration":calibration_ok,
        "data":str(r.get("data_quality","OK"))=="OK",
        "base_score":num(r,"calibrated_score")>=72,
        "quality":num(r,"quality_score")>=65,
        "valuation":num(r,"valuation_score")>=55,
        "momentum":num(r,"momentum_score")>=55,
        "risk":num(r,"risk_score")>=45,
        "technical": bool(r.get("technical_confirm",False)) if vb_ok else True,
        "market":regime!="RISK_OFF",
        "news":not bool(r.get("news_risk",False)),
    }
    passed=sum(gates.values()); total=len(gates)
    confidence=.70*num(r,"calibrated_score")+.20*(100*passed/total)+.10*regime_score

    # Strong means validated calibration must exist; no "strong" from unvalidated fixed weights.
    if calibration_ok and passed>=9 and confidence>=80 and num(r,"calibrated_score")>=78:
        signal="COMPRA_FORTE"
    elif calibration_ok and passed>=8 and confidence>=73:
        signal="AVALIAR_COMPRA"
    elif num(r,"calibrated_score")>=68:
        signal="WATCHLIST"
    else:
        signal="AGUARDAR"

    return pd.Series({
        "final_confidence_score":float(np.clip(confidence,0,100)),
        "final_signal":signal,
        "gates_passed":passed,"gates_total":total,
        "failed_gates":",".join([k for k,v in gates.items() if not v]),
        "market_regime":regime,"market_regime_score":regime_score,
        "calibration_validated":calibration_ok,
        "vectorbt_validated":vb_ok,
    })

extra=rank.apply(decide,axis=1)
rank=pd.concat([rank,extra],axis=1)
rank=rank.sort_values("final_confidence_score",ascending=False)

rank.to_csv(OUT/"final_signals.csv",index=False)
strong=rank[rank["final_signal"].isin(["COMPRA_FORTE","AVALIAR_COMPRA"])]
strong.to_csv(OUT/"final_opportunities.csv",index=False)

status={
    "market_regime":regime,"market_regime_score":regime_score,
    "calibration_validated":calibration_ok,"vectorbt_validated":vb_ok,
    "strong_buys":int((rank["final_signal"]=="COMPRA_FORTE").sum()),
    "evaluate_buys":int((rank["final_signal"]=="AVALIAR_COMPRA").sum()),
    "watchlist":int((rank["final_signal"]=="WATCHLIST").sum()),
    "important":"final_confidence_score is an evidence score, not a probability of profit."
}
(OUT/"final_signal_status.json").write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding="utf-8")
print(rank[["ticker","calibrated_score","final_confidence_score","final_signal","failed_gates"]].head(20).to_string(index=False))
