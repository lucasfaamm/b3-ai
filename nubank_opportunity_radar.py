
from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import json
import os
import time
import math
import re

import numpy as np
import pandas as pd
import requests
import yfinance as yf

DATA=Path("data"); OUT=Path("results")
DATA.mkdir(exist_ok=True); OUT.mkdir(exist_ok=True)

TOKEN=os.getenv("BRAPI_TOKEN","").strip()
HEAD={"Authorization":f"Bearer {TOKEN}"} if TOKEN else {}

FIXED_INCOME_WORDS=re.compile(
    r"\b(TESOURO|IPCA|IMA[- ]?B|DI\b|RENDA FIXA|FIXED INCOME|BOND|TREASURY|"
    r"DEBENTURE|DEBÊNTURE|CREDIT|CRÉDITO|LFT|LTN|NTN|CDI)\b", re.I
)

CLASS_LABEL={
    "stock":"AÇÃO B3",
    "unit":"AÇÃO/UNIT B3",
    "bdr":"AÇÃO GLOBAL / BDR",
    "etf":"ETF",
    "fii":"FII",
    "fi-infra":"FI-INFRA",
    "fi-agro":"FIAGRO",
    "fip":"FIP",
    "fidc":"FIDC",
}

BENCHMARKS={
    "AÇÃO B3":"BOVA11.SA",
    "AÇÃO/UNIT B3":"BOVA11.SA",
    "AÇÃO GLOBAL / BDR":"IVVB11.SA",
    "ETF":"BOVA11.SA",
    "FII":"XFIX11.SA",
    "FI-INFRA":"BOVA11.SA",
    "FIAGRO":"BOVA11.SA",
    "FIP":"BOVA11.SA",
    "FIDC":"BOVA11.SA",
}

def req_json(url, params=None, tries=4, timeout=45):
    last=None
    for i in range(tries):
        try:
            r=requests.get(url,headers=HEAD,params=params,timeout=timeout)
            if r.status_code==200:
                return r.json()
            last=RuntimeError(f"HTTP {r.status_code}: {url}")
            if r.status_code not in {408,429,500,502,503,504}:
                break
        except Exception as e:
            last=e
        time.sleep(min(2**i,8))
    raise last

def brapi_universe():
    rows=[]
    page=1
    while True:
        j=req_json(
            "https://brapi.dev/api/quote/list",
            params={"limit":"250","page":str(page)}
        )
        batch=j.get("stocks") or []
        rows.extend(batch)
        if not j.get("hasNextPage") or not batch:
            break
        page+=1
        if page>30:
            break

    x=pd.DataFrame(rows)
    if x.empty:
        raise RuntimeError("brapi universe returned zero assets.")

    x=x.rename(columns={
        "stock":"ticker","name":"name","market_cap":"market_cap",
        "subType":"subtype","type":"type"
    })
    for c in ["ticker","name","sector","type","subtype"]:
        if c not in x.columns:x[c]=""
    for c in ["close","volume","market_cap"]:
        x[c]=pd.to_numeric(x.get(c),errors="coerce")

    x["ticker"]=x["ticker"].astype(str).str.upper().str.strip()
    x["name"]=x["name"].astype(str)
    x["subtype"]=x["subtype"].astype(str).str.lower().str.strip()
    x["asset_class"]=x["subtype"].map(CLASS_LABEL)

    allowed=x["subtype"].isin(CLASS_LABEL)
    x=x[allowed].copy()

    # User explicitly excluded fixed income.
    fixed_name=(x["name"].fillna("")+" "+x["ticker"].fillna("")).str.contains(FIXED_INCOME_WORDS,na=False)
    x=x[~fixed_name].copy()

    x["turnover_proxy"]=x["close"].fillna(0)*x["volume"].fillna(0)
    x=x[x["close"].fillna(0)>0].drop_duplicates("ticker")
    x.to_csv(DATA/"nubank_b3_universe.csv",index=False)
    return x

def select_investable(u):
    frames=[]
    limits={
        "AÇÃO GLOBAL / BDR":180,
        "ETF":140,
        "FII":250,
        "FI-INFRA":80,
        "FIAGRO":100,
        "FIP":40,
        "FIDC":40,
    }
    for cls,n in limits.items():
        g=u[u.asset_class.eq(cls)].copy()
        if g.empty:continue
        # A liquidity screen prevents unusable securities from dominating the radar.
        g=g.sort_values(["turnover_proxy","volume"],ascending=False).head(n)
        frames.append(g)
    return pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()

def yf_download(symbols,period="2y"):
    collected={}
    for start in range(0,len(symbols),70):
        chunk=symbols[start:start+70]
        try:
            raw=yf.download(
                chunk,period=period,interval="1d",auto_adjust=True,
                actions=True,progress=False,threads=4,group_by="ticker"
            )
        except Exception as e:
            print("[WARN] yfinance chunk",start,e,flush=True)
            continue
        for sym in chunk:
            try:
                if isinstance(raw.columns,pd.MultiIndex):
                    if sym in raw.columns.get_level_values(0):
                        d=raw[sym].copy()
                    elif sym in raw.columns.get_level_values(1):
                        d=raw.xs(sym,axis=1,level=1).copy()
                    else:
                        continue
                else:
                    d=raw.copy()
                d.index=pd.to_datetime(d.index).tz_localize(None)
                collected[sym]=d
            except Exception:
                pass
    return collected

def price_features(sel):
    symbols=[f"{t}.SA" for t in sel.ticker.astype(str)]
    benches=sorted(set(BENCHMARKS.values()))
    data=yf_download(symbols+benches,period="2y")

    bench_returns={}
    for b in benches:
        d=data.get(b,pd.DataFrame())
        if d.empty or "Close" not in d:continue
        c=pd.to_numeric(d["Close"],errors="coerce").dropna()
        if len(c)>=127:
            bench_returns[b]=float(c.iloc[-1]/c.iloc[-127]-1)

    rows=[]
    for r in sel.to_dict("records"):
        sym=f"{r['ticker']}.SA"
        d=data.get(sym,pd.DataFrame())
        if d.empty or "Close" not in d:continue
        c=pd.to_numeric(d["Close"],errors="coerce").dropna()
        if len(c)<150:continue
        vol=pd.to_numeric(d.get("Volume"),errors="coerce").reindex(c.index)
        div=pd.to_numeric(d.get("Dividends"),errors="coerce").reindex(c.index).fillna(0) if "Dividends" in d else pd.Series(0,index=c.index)

        price=float(c.iloc[-1])
        ret21=float(c.iloc[-1]/c.iloc[-22]-1) if len(c)>=22 else np.nan
        ret63=float(c.iloc[-1]/c.iloc[-64]-1) if len(c)>=64 else np.nan
        ret126=float(c.iloc[-1]/c.iloc[-127]-1) if len(c)>=127 else np.nan
        ret252=float(c.iloc[-1]/c.iloc[-253]-1) if len(c)>=253 else np.nan
        daily=c.pct_change().dropna()
        vola=float(daily.tail(63).std(ddof=1)*np.sqrt(252)) if len(daily)>=63 else np.nan
        dd=float((c.tail(252)/c.tail(252).cummax()-1).min())
        ma50=float(c.tail(50).mean()); ma200=float(c.tail(200).mean()) if len(c)>=200 else np.nan
        adv=float((c*vol).tail(63).mean()) if vol.notna().sum()>=40 else r.get("turnover_proxy",np.nan)

        div12=float(div.tail(252).sum()) if len(div) else 0.0
        dy=div12/price if price>0 else np.nan
        div_months=int((div.tail(252)>0).groupby(div.tail(252).index.to_period("M")).any().sum()) if len(div) else 0

        bench=BENCHMARKS.get(r["asset_class"],"BOVA11.SA")
        rel=ret126-bench_returns.get(bench,np.nan) if not pd.isna(ret126) else np.nan
        trend=bool(not pd.isna(ma200) and price>ma200 and ma50>ma200 and (pd.isna(ret126) or ret126>0))

        rows.append({
            **r,"price_now":price,"ret_21d":ret21,"ret_63d":ret63,
            "ret_126d":ret126,"ret_252d":ret252,"volatility":vola,
            "max_drawdown":dd,"ma50":ma50,"ma200":ma200,
            "avg_daily_value":adv,"dividend_yield_12m":dy,
            "dividend_months_12m":div_months,"relative_strength_6m":rel,
            "trend_confirm":trend,
        })
    return pd.DataFrame(rows)

def pct(s,positive=True):
    x=pd.to_numeric(s,errors="coerce")
    r=x.rank(pct=True,method="average")*100
    return r if positive else 100-r

def deep_bdr_fundamentals(df):
    # Market score works for every BDR; fundamentals are an extra layer for the
    # most liquid BDRs when the user's free brapi plan exposes these modules.
    out=[]
    deep=df[df.asset_class.eq("AÇÃO GLOBAL / BDR")].sort_values("avg_daily_value",ascending=False).head(80)
    for i,r in enumerate(deep.to_dict("records"),1):
        try:
            j=req_json(
                f"https://brapi.dev/api/quote/{r['ticker']}",
                params={"modules":"defaultKeyStatistics,financialData,summaryProfile"}
            )
            item=(j.get("results") or [{}])[0]
            ks=item.get("defaultKeyStatistics") or {}
            fd=item.get("financialData") or {}
            pf=item.get("summaryProfile") or {}
            out.append({
                "ticker":r["ticker"],
                "profit_margin":ks.get("profitMargins",fd.get("profitMargins")),
                "roe":fd.get("returnOnEquity"),
                "revenue_growth":fd.get("revenueGrowth"),
                "earnings_growth":fd.get("earningsGrowth"),
                "debt_to_equity":fd.get("debtToEquity"),
                "free_cash_flow":fd.get("freeCashflow"),
                "market_cap_fund":item.get("marketCap"),
                "forward_pe":ks.get("forwardPE"),
                "price_to_book":ks.get("priceToBook"),
                "recommendation_key":fd.get("recommendationKey"),
                "analyst_count":fd.get("numberOfAnalystOpinions"),
                "global_sector":pf.get("sector"),
                "global_industry":pf.get("industry"),
                "fundamental_coverage":True,
            })
        except Exception:
            out.append({"ticker":r["ticker"],"fundamental_coverage":False})
        if i%20==0:print(f"[BDR fundamentals] {i}/{len(deep)}",flush=True)
        time.sleep(.08)
    return pd.DataFrame(out)

def market_regime():
    raw=yf.download(["BOVA11.SA","IVVB11.SA","^VIX","BRL=X"],period="1y",
                    auto_adjust=True,progress=False,threads=4,group_by="ticker")
    def close(sym):
        try:
            d=raw[sym] if isinstance(raw.columns,pd.MultiIndex) and sym in raw.columns.get_level_values(0) else raw
            s=d["Close"];s=s.iloc[:,0] if isinstance(s,pd.DataFrame) else s
            return pd.to_numeric(s,errors="coerce").dropna()
        except:return pd.Series(dtype=float)
    b=close("BOVA11.SA");iv=close("IVVB11.SA");v=close("^VIX");usd=close("BRL=X")
    br=50.0;glob=50.0
    if len(b)>=200:br+=25 if b.iloc[-1]>b.tail(200).mean() else -25
    if len(iv)>=200:glob+=25 if iv.iloc[-1]>iv.tail(200).mean() else -25
    if len(v):
        if v.iloc[-1]<20:glob+=10
        elif v.iloc[-1]>30:glob-=15
    if len(usd)>=64:
        u=float(usd.iloc[-1]/usd.iloc[-64]-1)
        if u>.10:glob-=5
    br=float(np.clip(br,0,100));glob=float(np.clip(glob,0,100))
    label=lambda x:"RISK_ON" if x>=65 else ("RISK_OFF" if x<=40 else "NEUTRAL")
    return {"BRASIL":label(br),"BRASIL_SCORE":br,"GLOBAL":label(glob),"GLOBAL_SCORE":glob}

def score_assets(x,bdrfund):
    if x.empty:return x
    x=x.merge(bdrfund,on="ticker",how="left") if not bdrfund.empty else x.copy()

    # Common cross-sectional market dimensions.
    x["momentum_score"]=pd.concat([
        pct(x.ret_63d),pct(x.ret_126d),pct(x.relative_strength_6m)
    ],axis=1).mean(axis=1)
    x["risk_score"]=pd.concat([
        pct(x.volatility,False),pct(x.max_drawdown)
    ],axis=1).mean(axis=1)
    x["liquidity_score"]=pct(np.log1p(pd.to_numeric(x.avg_daily_value,errors="coerce")))
    x["trend_score"]=np.where(x.trend_confirm,100,35)

    scored=[]
    for cls,g in x.groupby("asset_class",dropna=False):
        g=g.copy()

        if cls=="FII":
            g["income_score"]=pd.concat([
                pct(g.dividend_yield_12m),
                pct(g.dividend_months_12m)
            ],axis=1).mean(axis=1)
            g["opportunity_score"]=(
                .30*g.income_score+.22*g.momentum_score+.20*g.risk_score+
                .12*g.liquidity_score+.16*g.trend_score
            )

        elif cls in {"FI-INFRA","FIAGRO","FIP","FIDC"}:
            g["income_score"]=pd.concat([
                pct(g.dividend_yield_12m),pct(g.dividend_months_12m)
            ],axis=1).mean(axis=1)
            g["opportunity_score"]=(
                .25*g.income_score+.25*g.momentum_score+.20*g.risk_score+
                .15*g.liquidity_score+.15*g.trend_score
            )

        elif cls=="ETF":
            g["opportunity_score"]=(
                .35*g.momentum_score+.27*g.risk_score+
                .18*g.liquidity_score+.20*g.trend_score
            )

        elif cls=="AÇÃO GLOBAL / BDR":
            # If fundamentals exist, use them; otherwise remain market-based and
            # apply a small confidence haircut.
            for c in ["roe","profit_margin","revenue_growth","earnings_growth","debt_to_equity","forward_pe","price_to_book"]:
                if c not in g:g[c]=np.nan

            g["quality_score"]=pd.concat([
                pct(g.roe),pct(g.profit_margin),pct(g.debt_to_equity,False)
            ],axis=1).mean(axis=1)
            g["growth_score"]=pd.concat([
                pct(g.revenue_growth),pct(g.earnings_growth)
            ],axis=1).mean(axis=1)
            g["valuation_score"]=pd.concat([
                pct(g.forward_pe,False),pct(g.price_to_book,False)
            ],axis=1).mean(axis=1)

            fund_score=.30*g.quality_score+.25*g.growth_score+.20*g.valuation_score+.15*g.momentum_score+.10*g.risk_score
            market_only=.42*g.momentum_score+.28*g.risk_score+.15*g.liquidity_score+.15*g.trend_score
            coverage=g.get("fundamental_coverage",False).fillna(False).astype(bool)
            g["opportunity_score"]=np.where(coverage,fund_score,market_only*.94)
            g["fundamental_coverage"]=coverage

        else:
            g["opportunity_score"]=(
                .35*g.momentum_score+.25*g.risk_score+
                .20*g.liquidity_score+.20*g.trend_score
            )

        scored.append(g)

    x=pd.concat(scored,ignore_index=True)
    x["class_percentile"]=x.groupby("asset_class")["opportunity_score"].rank(pct=True)*100

    reg=market_regime()
    x["regime"]=np.where(
        x.asset_class.eq("AÇÃO GLOBAL / BDR"),
        reg["GLOBAL"],reg["BRASIL"]
    )
    x["regime_score"]=np.where(
        x.asset_class.eq("AÇÃO GLOBAL / BDR"),
        reg["GLOBAL_SCORE"],reg["BRASIL_SCORE"]
    )

    # Universal score is normalized within asset class, avoiding invalid raw
    # comparisons such as FII dividend metrics versus an ETF's volatility metrics.
    x["global_opportunity_score"]=(
        .70*x.class_percentile+.20*x.opportunity_score+.10*x.regime_score
    )

    # Volatility-based attention band, not a fair-value target.
    mvol=x.volatility/np.sqrt(12)
    x["entry_watch_low"]=x.price_now*(1-.30*mvol.clip(lower=0,upper=1))
    x["entry_watch_high"]=x.price_now

    def decision(r):
        s=float(r.global_opportunity_score)
        trend=bool(r.trend_confirm)
        regime=str(r.regime)
        if s>=83 and trend and regime!="RISK_OFF":
            return "AVALIAR_COMPRA"
        if s>=72:
            return "WATCHLIST"
        if s<42:
            return "EVITAR_AGORA"
        return "AGUARDAR"

    x["signal"]=x.apply(decision,axis=1)

    def strengths(r):
        vals=[]
        for c,label in [
            ("momentum_score","momentum"),("risk_score","risco controlado"),
            ("liquidity_score","liquidez"),("quality_score","qualidade"),
            ("growth_score","crescimento"),("valuation_score","valuation")
        ]:
            try:
                if float(r.get(c,np.nan))>=75:vals.append(label)
            except:pass
        if bool(r.get("trend_confirm",False)):vals.append("tendência positiva")
        return ", ".join(vals[:4]) if vals else "sem vantagem clara"

    def risks(r):
        vals=[]
        for c,label in [
            ("momentum_score","momentum fraco"),("risk_score","risco elevado"),
            ("liquidity_score","liquidez baixa"),("quality_score","qualidade fraca"),
            ("growth_score","crescimento fraco"),("valuation_score","valuation caro")
        ]:
            try:
                if float(r.get(c,np.nan))<35:vals.append(label)
            except:pass
        if not bool(r.get("trend_confirm",False)):vals.append("tendência não confirmada")
        if str(r.get("regime"))=="RISK_OFF":vals.append("regime desfavorável")
        return ", ".join(vals[:4]) if vals else "nenhum alerta quantitativo principal"

    x["strengths"]=x.apply(strengths,axis=1)
    x["risks"]=x.apply(risks,axis=1)
    return x.sort_values("global_opportunity_score",ascending=False),reg

def integrate_b3_stocks(other,reg):
    p=OUT/"final_signals.csv"
    if not p.exists():return other
    b=pd.read_csv(p)
    if b.empty:return other

    price_col="price_now" if "price_now" in b else ("price" if "price" in b else None)
    if price_col is None:return other

    z=pd.DataFrame({
        "ticker":b["ticker"].astype(str),
        "name":b["company_name"] if "company_name" in b else b["ticker"],
        "asset_class":"AÇÃO B3",
        "price_now":pd.to_numeric(b[price_col],errors="coerce"),
        "opportunity_score":pd.to_numeric(b.get("validated_score",b.get("score")),errors="coerce"),
        "global_opportunity_score":pd.to_numeric(b.get("final_confidence_score",b.get("validated_score")),errors="coerce"),
        "signal":b.get("final_signal","AGUARDAR"),
        "entry_watch_low":pd.to_numeric(b.get("entry_watch_low"),errors="coerce"),
        "entry_watch_high":pd.to_numeric(b.get("entry_watch_high"),errors="coerce"),
        "regime":b.get("market_regime",reg["BRASIL"]),
        "strengths":"",
        "risks":b.get("failed_gates",""),
        "trend_confirm":b.get("trend_confirm",False),
    })
    # Re-normalize stock score to make the global table comparable by class rank.
    z["class_percentile"]=z["global_opportunity_score"].rank(pct=True)*100
    z["global_opportunity_score"]=.70*z["class_percentile"]+.30*z["global_opportunity_score"]
    return pd.concat([other,z],ignore_index=True,sort=False)

def portfolio_actions(ranking):
    p=Path("portfolio.csv")
    if not p.exists() or not p.stat().st_size:
        return pd.DataFrame()
    pf=pd.read_csv(p)
    if "ticker" not in pf:return pd.DataFrame()
    pf["ticker"]=pf["ticker"].astype(str).str.upper()
    keep=ranking.sort_values("global_opportunity_score",ascending=False).drop_duplicates("ticker")
    cols=[c for c in ["ticker","asset_class","price_now","global_opportunity_score","signal","risks","strengths"] if c in keep]
    x=pf.merge(keep[cols],on="ticker",how="left")
    q=pd.to_numeric(x.get("quantity"),errors="coerce")
    pr=pd.to_numeric(x.get("price_now"),errors="coerce")
    x["current_value"]=q*pr if q is not None else np.nan
    total=x["current_value"].sum(skipna=True)
    x["portfolio_weight"]=x["current_value"]/total if total and total>0 else np.nan

    def action(r):
        s=pd.to_numeric(pd.Series([r.get("global_opportunity_score")]),errors="coerce").iloc[0]
        w=pd.to_numeric(pd.Series([r.get("portfolio_weight")]),errors="coerce").iloc[0]
        if pd.isna(s):return "SEM_DADOS"
        if s>=82 and (pd.isna(w) or w<.15):return "AUMENTAR_GRADUALMENTE"
        if s>=62:return "MANTER"
        if s<45:return "REAVALIAR_REDUZIR"
        return "NAO_AUMENTAR"
    x["portfolio_action"]=x.apply(action,axis=1)
    return x

def validate_forward(ranking):
    logp=DATA/"nubank_forward_validation.csv"
    today=pd.Timestamp.now(tz="UTC").normalize().tz_localize(None)
    month=today.to_period("M").strftime("%Y-%m")

    candidates=[]
    for cls,g in ranking.groupby("asset_class"):
        g=g.sort_values("global_opportunity_score",ascending=False).head(10)
        for _,r in g.iterrows():
            candidates.append({
                "signal_month":month,"signal_date":today.date().isoformat(),
                "ticker":r.ticker,"asset_class":cls,
                "score":r.global_opportunity_score
            })
    new=pd.DataFrame(candidates)
    log=pd.read_csv(logp) if logp.exists() and logp.stat().st_size else pd.DataFrame()
    log=pd.concat([log,new],ignore_index=True).drop_duplicates(["signal_month","ticker"],keep="first")

    for c in ["entry_date","entry_price","target_date","target_price","return_20d","benchmark_return_20d","alpha_20d"]:
        if c not in log:log[c]=np.nan

    tickers=log.ticker.dropna().astype(str).unique().tolist()
    syms=[f"{t}.SA" for t in tickers]+sorted(set(BENCHMARKS.values()))
    d=yf_download(syms,period="3y")

    def close(sym):
        f=d.get(sym,pd.DataFrame())
        if f.empty or "Close" not in f:return pd.Series(dtype=float)
        s=pd.to_numeric(f["Close"],errors="coerce").dropna()
        return s

    for i,r in log.iterrows():
        s=close(f"{r.ticker}.SA")
        if s.empty:continue
        dt=pd.Timestamp(r.signal_date)
        pos=int(s.index.searchsorted(dt))
        if pos>=len(s):continue
        ed=s.index[pos];ep=float(s.iloc[pos])
        log.at[i,"entry_date"]=ed.date().isoformat()
        log.at[i,"entry_price"]=ep
        if pos+20>=len(s):continue
        td=s.index[pos+20];tp=float(s.iloc[pos+20])
        ret=tp/ep-1
        log.at[i,"target_date"]=td.date().isoformat()
        log.at[i,"target_price"]=tp
        log.at[i,"return_20d"]=ret

        bench=BENCHMARKS.get(str(r.asset_class),"BOVA11.SA")
        b=close(bench)
        if not b.empty:
            a=int(b.index.searchsorted(ed));bb=int(b.index.searchsorted(td))
            if a<len(b) and bb<len(b):
                br=float(b.iloc[bb]/b.iloc[a]-1)
                log.at[i,"benchmark_return_20d"]=br
                log.at[i,"alpha_20d"]=ret-br

    log.to_csv(logp,index=False)

    ev=log.dropna(subset=["alpha_20d"])
    if len(ev):
        summary=ev.groupby("asset_class").agg(
            evaluated=("signal_month","nunique"),
            avg_return_20d=("return_20d","mean"),
            avg_alpha_20d=("alpha_20d","mean"),
            positive_alpha_rate=("alpha_20d",lambda s:float((s>0).mean()))
        ).reset_index()
    else:
        summary=pd.DataFrame(columns=["asset_class","evaluated","avg_return_20d","avg_alpha_20d","positive_alpha_rate"])
    summary.to_csv(OUT/"nubank_forward_validation_summary.csv",index=False)

def main():
    u=brapi_universe()
    sel=select_investable(u)
    market=price_features(sel)
    bdr=deep_bdr_fundamentals(market)
    ranked,reg=score_assets(market,bdr)
    ranked=integrate_b3_stocks(ranked,reg)
    ranked=ranked.sort_values("global_opportunity_score",ascending=False)

    ranked.to_csv(OUT/"nubank_global_ranking.csv",index=False)
    ranked[ranked.signal.astype(str).str.contains("AVALIAR_COMPRA",na=False)].to_csv(
        OUT/"nubank_opportunities.csv",index=False
    )

    pf=portfolio_actions(ranked)
    if not pf.empty:pf.to_csv(OUT/"nubank_portfolio_actions.csv",index=False)

    validate_forward(ranked)

    status={
        "updated_at":datetime.now(timezone.utc).isoformat(),
        "universe_total":int(len(u)),
        "ranked_total":int(len(ranked)),
        "by_class":ranked.asset_class.value_counts().to_dict(),
        "brasil_regime":reg["BRASIL"],"global_regime":reg["GLOBAL"],
        "fixed_income_excluded":True,
        "bdr_deep_fundamentals_requested":80,
        "note":"Opportunity scores are evidence/ranking scores, not probabilities of profit."
    }
    (OUT/"nubank_radar_status.json").write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding="utf-8")
    print(ranked[["ticker","asset_class","price_now","global_opportunity_score","signal","strengths","risks"]].head(40).to_string(index=False))

if __name__=="__main__":
    main()
