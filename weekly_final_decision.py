
from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import json
import math
import re

import numpy as np
import pandas as pd
import yfinance as yf

DATA=Path("data"); OUT=Path("results")
DATA.mkdir(exist_ok=True); OUT.mkdir(exist_ok=True)

RANK=OUT/"nubank_global_ranking.csv"
B3=OUT/"final_signals.csv"
VALID=OUT/"nubank_forward_validation_summary.csv"
PORT=Path("portfolio.csv")
HIST=DATA/"weekly_final_decision_history.csv"

NEGATIVE_NEWS=re.compile(
    r"\b(fraud|fraude|investigation|investigação|probe|bankrupt|bankruptcy|"
    r"fal[eê]ncia|default|calote|restatement|accounting issue|lawsuit|processo|"
    r"sanction|san[cç][aã]o|delist|deslist|recall|hack|breach|vazamento|"
    r"downgrade|rebaixamento|dividend cut|corte de dividendo|insolvency|"
    r"insolv[eê]ncia|corruption|corrup[cç][aã]o|guidance cut|profit warning)\b", re.I
)

CLASS_CAP={
    "AÇÃO B3":0.08, "AÇÃO/UNIT B3":0.08, "AÇÃO GLOBAL / BDR":0.08,
    "ETF":0.18, "FII":0.10, "FI-INFRA":0.08, "FIAGRO":0.08,
    "FIP":0.06, "FIDC":0.06
}

VOL_LIMIT={
    "AÇÃO B3":0.55, "AÇÃO/UNIT B3":0.55, "AÇÃO GLOBAL / BDR":0.60,
    "ETF":0.40, "FII":0.35, "FI-INFRA":0.35, "FIAGRO":0.40,
    "FIP":0.45, "FIDC":0.40
}

DD_LIMIT={
    "AÇÃO B3":-0.35, "AÇÃO/UNIT B3":-0.35, "AÇÃO GLOBAL / BDR":-0.38,
    "ETF":-0.28, "FII":-0.25, "FI-INFRA":-0.25, "FIAGRO":-0.28,
    "FIP":-0.30, "FIDC":-0.28
}

def n(v):
    return pd.to_numeric(pd.Series([v]), errors="coerce").iloc[0]

def truth(v):
    if isinstance(v,bool): return v
    return str(v).strip().lower() in {"true","1","sim","yes"}

def ys(ticker):
    return f"{ticker}.SA"

def download_one(ticker):
    try:
        d=yf.download(ys(ticker),period="2y",interval="1d",
                      auto_adjust=True,progress=False,threads=False)
        if d.empty:return pd.DataFrame()
        if isinstance(d.columns,pd.MultiIndex):
            d.columns=d.columns.get_level_values(0)
        d.index=pd.to_datetime(d.index).tz_localize(None)
        return d
    except Exception:
        return pd.DataFrame()

def technical_plan(ticker, current):
    d=download_one(ticker)
    if d.empty or "Close" not in d or len(d)<150:return {}

    for c in ["Open","High","Low","Close","Volume"]:
        if c in d:d[c]=pd.to_numeric(d[c],errors="coerce")
    d=d.dropna(subset=["High","Low","Close"])
    c=d["Close"]; h=d["High"]; l=d["Low"]

    p=float(c.iloc[-1]) if pd.isna(current) else float(current)
    prev=c.shift(1)
    tr=pd.concat([(h-l).abs(),(h-prev).abs(),(l-prev).abs()],axis=1).max(axis=1)
    atr=float(tr.tail(14).mean())

    ma20=float(c.tail(20).mean())
    ma50=float(c.tail(50).mean())
    ma200=float(c.tail(200).mean()) if len(c)>=200 else np.nan
    high52=float(h.tail(252).max())
    low52=float(l.tail(252).min())
    daily=c.pct_change().dropna()
    vol20=float(daily.tail(20).std(ddof=1)*np.sqrt(252))
    vol126=float(daily.tail(126).std(ddof=1)*np.sqrt(252))
    ret1=float(c.iloc[-1]/c.iloc[-2]-1) if len(c)>=2 else np.nan
    ret21=float(c.iloc[-1]/c.iloc[-22]-1) if len(c)>=22 else np.nan
    ret63=float(c.iloc[-1]/c.iloc[-64]-1) if len(c)>=64 else np.nan

    buy_low=max(0.01, min(p,ma20)-0.45*atr)
    buy_high=max(buy_low, min(p+0.15*atr, ma20+0.75*atr))
    no_chase=max(buy_high,ma20+1.35*atr)

    supports=[ma50-1.15*atr, l.tail(20).min()-0.30*atr]
    if not pd.isna(ma200):supports.append(ma200-0.45*atr)
    valid_support=[float(x) for x in supports if not pd.isna(x) and x<p]
    invalid=max(0.01,max(valid_support,default=p-2.5*atr))
    if invalid>=p:invalid=max(0.01,p-2.5*atr)

    entry_mid=(buy_low+buy_high)/2
    risk=max(entry_mid-invalid,0.01)
    risk_pct=risk/entry_mid

    return {
        "atr14":atr,"ma20_live":ma20,"ma50_live":ma50,"ma200_live":ma200,
        "high_52w":high52,"low_52w":low52,"ret_1d":ret1,"ret_21d_live":ret21,
        "ret_63d_live":ret63,"vol20":vol20,"vol126":vol126,
        "buy_zone_low":buy_low,"buy_zone_high":buy_high,
        "do_not_chase_above":no_chase,"invalidation_price":invalid,
        "risk_to_invalidation_pct":risk_pct,
        "review_price_2R":entry_mid+2*risk,
        "review_price_3R":entry_mid+3*risk,
        "price_in_buy_zone":bool(p>=buy_low and p<=buy_high),
        "price_above_buy_zone":bool(p>buy_high and p<=no_chase),
        "price_overextended":bool(p>no_chase),
        "volatility_spike":bool(vol20>1.5*vol126) if not pd.isna(vol126) and vol126>0 else False,
        "gap_risk":bool(abs(ret1)>0.08) if not pd.isna(ret1) else False,
    }

def news_check(ticker):
    try:
        items=yf.Ticker(ys(ticker)).news or []
        texts=[]
        for item in items[:15]:
            title=item.get("title") or ""
            summary=item.get("summary") or ""
            content=item.get("content") or {}
            if isinstance(content,dict):
                title=title or content.get("title") or ""
                summary=summary or content.get("summary") or content.get("description") or ""
            txt=f"{title} {summary}".strip()
            if txt:texts.append(txt)
        if not texts:return "UNAVAILABLE",""
        bad=[x for x in texts if NEGATIVE_NEWS.search(x)]
        return ("NEGATIVE_VETO" if bad else "CLEAR"),(" | ".join(bad[:2])[:500] if bad else "")
    except Exception:
        return "UNAVAILABLE",""

def load_b3():
    if not B3.exists():return {}
    b=pd.read_csv(B3)
    b["ticker"]=b["ticker"].astype(str).str.upper()
    return {str(r.ticker):r for _,r in b.drop_duplicates("ticker").iterrows()}

def validation():
    if not VALID.exists():return {}
    try:
        v=pd.read_csv(VALID)
        return {str(r.asset_class):r for _,r in v.iterrows()}
    except:return {}

def forward_evidence(cls,vmap):
    r=vmap.get(cls)
    if r is None:return "INSUFFICIENT",True
    months=n(getattr(r,"evaluated",np.nan))
    alpha=n(getattr(r,"avg_alpha_20d",np.nan))
    hit=n(getattr(r,"positive_alpha_rate",np.nan))
    if pd.isna(months) or months<6:return "INSUFFICIENT",True
    if (not pd.isna(alpha) and alpha<0) or (not pd.isna(hit) and hit<0.50):
        return "NEGATIVE",False
    if months>=12 and (pd.isna(alpha) or alpha<=0.002):
        return "WEAK",True
    return "POSITIVE",True

def portfolio():
    if not PORT.exists() or not PORT.stat().st_size:return pd.DataFrame(),np.nan
    try:
        p=pd.read_csv(PORT)
        if "ticker" not in p:return pd.DataFrame(),np.nan
        p["ticker"]=p["ticker"].astype(str).str.upper()
        if "current_value" not in p.columns:
            if "quantity" in p and "price_now" in p:
                p["current_value"]=pd.to_numeric(p.quantity,errors="coerce")*pd.to_numeric(p.price_now,errors="coerce")
        total=pd.to_numeric(p.get("current_value"),errors="coerce").sum(skipna=True)
        return p,total
    except:return pd.DataFrame(),np.nan

def weight(ticker,p,total):
    if p.empty or pd.isna(total) or total<=0:return 0.0
    z=p[p.ticker.eq(ticker)]
    if z.empty:return 0.0
    return float(pd.to_numeric(z.get("current_value"),errors="coerce").sum(skipna=True)/total)

def hard_review(row,b3row,vmap,pf,total):
    cls=str(row.get("asset_class",""))
    failures=[]
    cautions=[]

    score=n(row.get("global_opportunity_score"))
    class_pct=n(row.get("class_percentile"))
    risk=n(row.get("risk_score"))
    mom=n(row.get("momentum_score"))
    liq=n(row.get("liquidity_score"))
    vol=n(row.get("volatility"))
    dd=n(row.get("max_drawdown"))
    regime=str(row.get("regime",""))
    trend=truth(row.get("trend_confirm"))
    risk_text=str(row.get("risks","")).lower()

    if pd.isna(score) or score<88:failures.append("score abaixo de 88")
    if not pd.isna(class_pct) and class_pct<88:failures.append("fora do top 12% da classe")
    if regime=="RISK_OFF":failures.append("regime de mercado RISK_OFF")
    if not trend:failures.append("tendência não confirmada")
    if "risco elevado" in risk_text:failures.append("radar marcou risco elevado")
    if not pd.isna(risk) and risk<60:failures.append("risk score abaixo de 60")
    if not pd.isna(mom) and mom<65:failures.append("momentum abaixo de 65")
    if not pd.isna(liq) and liq<50:failures.append("liquidez abaixo de 50")
    if not pd.isna(vol) and vol>VOL_LIMIT.get(cls,.55):failures.append("volatilidade excessiva")
    if not pd.isna(dd) and dd<DD_LIMIT.get(cls,-.35):failures.append("drawdown excessivo")

    if cls=="AÇÃO GLOBAL / BDR":
        if not truth(row.get("fundamental_coverage")):
            failures.append("fundamentos globais insuficientes")
        for col,limit,label in [
            ("quality_score",62,"qualidade insuficiente"),
            ("growth_score",50,"crescimento insuficiente"),
            ("valuation_score",38,"valuation pouco atraente"),
        ]:
            x=n(row.get(col))
            if not pd.isna(x) and x<limit:failures.append(label)

    if cls=="FII":
        inc=n(row.get("income_score"))
        if not pd.isna(inc) and inc<60:failures.append("renda/distribuições insuficientes")

    if cls=="ETF" and not pd.isna(mom) and mom<70:
        failures.append("ETF sem momentum suficiente")

    if cls=="AÇÃO B3" and b3row is not None:
        vs=n(b3row.get("validated_score"))
        fc=n(b3row.get("final_confidence_score"))
        fs=str(b3row.get("final_signal",""))
        fg=str(b3row.get("failed_gates",""))
        if not pd.isna(vs) and vs<75:failures.append("score fundamental B3 abaixo de 75")
        if not pd.isna(fc) and fc<75:failures.append("confiança B3 abaixo de 75")
        if fs not in {"AVALIAR_COMPRA","COMPRA_FORTE"}:failures.append("modelo B3 não liberou compra")
        if fg and fg.lower() not in {"nan","none",""}:failures.append("modelo B3 possui gates falhando")

    fe,forward_ok=forward_evidence(cls,vmap)
    if not forward_ok:failures.append("validação futura da classe negativa")
    elif fe=="INSUFFICIENT":cautions.append("validação futura ainda curta")
    elif fe=="WEAK":cautions.append("vantagem futura da classe ainda fraca")

    w=weight(str(row.get("ticker")),pf,total)
    cap=CLASS_CAP.get(cls,.08)
    if w>=cap:failures.append("posição já atingiu limite de concentração")

    return failures,cautions,fe,w,cap

def historical_consistency(ticker):
    if not HIST.exists() or not HIST.stat().st_size:return 0,0
    try:
        h=pd.read_csv(HIST)
        z=h[h.ticker.astype(str).eq(str(ticker))].sort_values("week").tail(4)
        good=z["decision"].astype(str).isin(["COMPRARIA_AGORA","COMPRARIA_SOMENTE_ATE"]).sum()
        return int(good),int(len(z))
    except:return 0,0

def explain(row,failures,cautions,plan):
    positives=[]
    for col,label in [
        ("quality_score","qualidade forte"),("growth_score","crescimento forte"),
        ("valuation_score","valuation favorável"),("momentum_score","momentum forte"),
        ("risk_score","risco relativo controlado"),("liquidity_score","boa liquidez"),
        ("income_score","renda/distribuições fortes")
    ]:
        v=n(row.get(col))
        if not pd.isna(v) and v>=75:positives.append(label)
    if truth(row.get("trend_confirm")):positives.append("tendência confirmada")
    if str(row.get("regime"))=="RISK_ON":positives.append("regime favorável")
    if not positives:positives=["nenhuma vantagem quantitativa dominante"]
    neg=failures+cautions
    if plan.get("price_overextended"):neg.append("preço esticado")
    if plan.get("volatility_spike"):neg.append("volatilidade recente disparou")
    if plan.get("gap_risk"):neg.append("gap diário anormal")
    return ", ".join(positives[:5]), ", ".join(dict.fromkeys(neg)) if neg else "sem veto principal"

def main():
    if not RANK.exists():raise SystemExit("Run nubank_opportunity_radar.py first.")

    rank=pd.read_csv(RANK)
    rank["ticker"]=rank["ticker"].astype(str).str.upper()
    b3map=load_b3()
    vmap=validation()
    pf,total=portfolio()

    # Reanalyse only the cream of the ranking, but across all asset classes.
    pool=[]
    for cls,g in rank.groupby("asset_class",dropna=False):
        pool.append(g.sort_values("global_opportunity_score",ascending=False).head(20))
    shortlist=pd.concat(pool,ignore_index=True)
    shortlist=shortlist.sort_values("global_opportunity_score",ascending=False).drop_duplicates("ticker").head(100)

    interim=[]
    for _,row in shortlist.iterrows():
        d=row.to_dict()
        ticker=d["ticker"]
        failures,cautions,fe,w,cap=hard_review(d,b3map.get(ticker),vmap,pf,total)
        plan=technical_plan(ticker,n(d.get("price_now")))

        if not plan:
            failures.append("dados técnicos insuficientes")
        else:
            if plan.get("volatility_spike"):failures.append("spike de volatilidade")
            if plan.get("gap_risk"):failures.append("gap diário excessivo")
            if plan.get("price_overextended"):cautions.append("preço acima da faixa ideal")

        base_pass=(len(failures)==0)
        interim.append({
            **d,**plan,
            "hard_failures":"; ".join(dict.fromkeys(failures)),
            "cautions":"; ".join(dict.fromkeys(cautions)),
            "forward_evidence":fe,
            "current_portfolio_weight":w,
            "max_position_pct":cap,
            "base_pass":base_pass
        })

    # News is the last check, and only for candidates that survived quantitative review.
    news_map={}
    news_candidates=sorted(
        [x for x in interim if x["base_pass"]],
        key=lambda x:x.get("global_opportunity_score",0),
        reverse=True
    )[:30]
    for x in news_candidates:
        news_map[x["ticker"]]=news_check(x["ticker"])

    rows=[]
    for x in interim:
        ticker=x["ticker"]
        news_status,news_reason=news_map.get(ticker,("NOT_RUN",""))
        x["news_check"]=news_status
        x["news_veto_reason"]=news_reason

        failures=[z.strip() for z in str(x.get("hard_failures","")).split(";") if z.strip()]
        cautions=[z.strip() for z in str(x.get("cautions","")).split(";") if z.strip()]
        if x["base_pass"]:
            if news_status=="NEGATIVE_VETO":
                failures.append("notícia negativa relevante")
            elif news_status=="UNAVAILABLE":
                cautions.append("notícias indisponíveis para confirmação automática")

        positives,negatives=explain(x,failures,cautions,x)
        x["why_buy"]=positives
        x["why_not"]=negatives

        price=n(x.get("price_now"))
        buy_low=n(x.get("buy_zone_low")); buy_high=n(x.get("buy_zone_high"))
        no_chase=n(x.get("do_not_chase_above"))

        if failures:
            # Strong asset with only one market/price blocker is worth waiting for.
            score=n(x.get("global_opportunity_score"))
            if not pd.isna(score) and score>=78:
                decision="ESPERARIA"
            else:
                decision="NAO_COMPRARIA"
        elif news_status!="CLEAR":
            decision="ESPERARIA"
        elif not pd.isna(price) and not pd.isna(buy_low) and not pd.isna(buy_high) and buy_low<=price<=buy_high:
            decision="COMPRARIA_AGORA"
        elif not pd.isna(price) and not pd.isna(buy_high) and not pd.isna(no_chase) and buy_high<price<=no_chase:
            decision="COMPRARIA_SOMENTE_ATE"
        else:
            decision="ESPERARIA"

        rp=n(x.get("risk_to_invalidation_pct"))
        cap=n(x.get("max_position_pct"))
        if not pd.isna(rp) and rp>0:
            target=float(min(cap,0.0075/rp))
        else:
            target=float(cap) if not pd.isna(cap) else np.nan

        # Initial tranche is deliberately smaller than the total target.
        initial=float(min(target/2,0.04)) if not pd.isna(target) else np.nan
        prev_good,prev_obs=historical_consistency(ticker)

        x["final_decision"]=decision
        x["max_buy_price"]=buy_high
        x["suggested_total_position_pct"]=target
        x["suggested_initial_position_pct"]=initial
        x["previous_positive_weeks"]=prev_good
        x["previous_weeks_observed"]=prev_obs

        if decision=="COMPRARIA_AGORA":
            x["plain_language"]=(
                f"Eu compraria agora apenas dentro de {buy_low:.2f}–{buy_high:.2f}. "
                f"Não perseguiria acima de {no_chase:.2f}. Reavaliaria a tese abaixo de {x['invalidation_price']:.2f}."
            )
        elif decision=="COMPRARIA_SOMENTE_ATE":
            x["plain_language"]=(
                f"Eu compraria somente até {buy_high:.2f}; acima disso esperaria recuo. "
                f"Não perseguiria acima de {no_chase:.2f}."
            )
        elif decision=="ESPERARIA":
            x["plain_language"]="Eu não colocaria dinheiro agora; manteria no radar até os vetos/entrada melhorarem."
        else:
            x["plain_language"]="Eu não compraria com as evidências atuais."

        rows.append(x)

    f=pd.DataFrame(rows)

    order={
        "COMPRARIA_AGORA":0,
        "COMPRARIA_SOMENTE_ATE":1,
        "ESPERARIA":2,
        "NAO_COMPRARIA":3
    }
    f["_ord"]=f.final_decision.map(order).fillna(9)
    f=f.sort_values(["_ord","global_opportunity_score"],ascending=[True,False]).drop(columns="_ord")

    f.to_csv(OUT/"weekly_final_decisions.csv",index=False)
    f[f.final_decision.eq("COMPRARIA_AGORA")].to_csv(OUT/"buy_now.csv",index=False)
    f[f.final_decision.eq("COMPRARIA_SOMENTE_ATE")].to_csv(OUT/"buy_only_up_to.csv",index=False)

    week=pd.Timestamp.now(tz="UTC").to_period("W").start_time.date().isoformat()
    h=f[["ticker","asset_class","global_opportunity_score","final_decision"]].copy()
    h["week"]=week
    old=pd.read_csv(HIST) if HIST.exists() and HIST.stat().st_size else pd.DataFrame()
    h=pd.concat([old,h],ignore_index=True).drop_duplicates(["week","ticker"],keep="last")
    h.to_csv(HIST,index=False)

    status={
        "updated_at":datetime.now(timezone.utc).isoformat(),
        "reanalyzed":int(len(f)),
        "buy_now":int((f.final_decision=="COMPRARIA_AGORA").sum()),
        "buy_only_up_to":int((f.final_decision=="COMPRARIA_SOMENTE_ATE").sum()),
        "wait":int((f.final_decision=="ESPERARIA").sum()),
        "do_not_buy":int((f.final_decision=="NAO_COMPRARIA").sum()),
        "decision_rule":"Strict weekly multi-factor review + class-specific checks + price plan + portfolio concentration + news veto.",
        "risk_budget":"Suggested total position targets <=0.75% portfolio risk to quantitative invalidation, subject to class cap.",
        "important":"Final decision is a systematic evidence-based opinion, never a guarantee of profit."
    }
    (OUT/"weekly_decision_status.json").write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding="utf-8")

    cols=[c for c in [
        "ticker","name","asset_class","price_now","global_opportunity_score",
        "final_decision","buy_zone_low","buy_zone_high","do_not_chase_above",
        "invalidation_price","suggested_initial_position_pct","suggested_total_position_pct",
        "why_buy","why_not","plain_language"
    ] if c in f]
    print(f[cols].head(50).to_string(index=False))

if __name__=="__main__":
    main()
