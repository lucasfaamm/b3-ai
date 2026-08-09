
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
HIST=DATA/"definitive_signal_history.csv"
PORT=Path("portfolio.csv")
VALID=OUT/"nubank_forward_validation_summary.csv"

NEGATIVE_NEWS=re.compile(
    r"\b(fraud|fraude|investigation|investigação|probe|bankrupt|bankruptcy|"
    r"fal[eê]ncia|default|calote|restatement|reformula[cç][aã]o|accounting issue|"
    r"contabilidade|lawsuit|processo|sanction|san[cç][aã]o|delist|deslist|"
    r"recall|hack|breach|vazamento|downgrade|rebaixamento|dividend cut|"
    r"corte de dividendo|insolvency|insolv[eê]ncia|corruption|corrup[cç][aã]o)\b", re.I
)

CAPS={
    "AÇÃO B3":0.08,
    "AÇÃO/UNIT B3":0.08,
    "AÇÃO GLOBAL / BDR":0.08,
    "ETF":0.18,
    "FII":0.10,
    "FI-INFRA":0.08,
    "FIAGRO":0.08,
    "FIP":0.06,
    "FIDC":0.06,
}
VOL_MAX={
    "AÇÃO B3":0.55,"AÇÃO/UNIT B3":0.55,"AÇÃO GLOBAL / BDR":0.60,
    "ETF":0.40,"FII":0.35,"FI-INFRA":0.35,"FIAGRO":0.40,"FIP":0.45,"FIDC":0.40
}
DD_MIN={
    "AÇÃO B3":-0.35,"AÇÃO/UNIT B3":-0.35,"AÇÃO GLOBAL / BDR":-0.38,
    "ETF":-0.28,"FII":-0.25,"FI-INFRA":-0.25,"FIAGRO":-0.28,"FIP":-0.30,"FIDC":-0.28
}

def num(v):
    return pd.to_numeric(pd.Series([v]),errors="coerce").iloc[0]

def boo(v):
    if isinstance(v,bool): return v
    return str(v).strip().lower() in {"true","1","yes","sim"}

def yf_symbol(ticker):
    return f"{ticker}.SA"

def price_plan(ticker, current):
    sym=yf_symbol(ticker)
    try:
        d=yf.download(sym,period="1y",interval="1d",auto_adjust=True,progress=False,threads=False)
        if d.empty:return {}
        if isinstance(d.columns,pd.MultiIndex):
            d.columns=d.columns.get_level_values(0)
        for c in ["High","Low","Close"]:
            d[c]=pd.to_numeric(d[c],errors="coerce")
        d=d.dropna(subset=["High","Low","Close"])
        if len(d)<80:return {}

        c=d["Close"];h=d["High"];l=d["Low"]
        prev=c.shift(1)
        tr=pd.concat([(h-l).abs(),(h-prev).abs(),(l-prev).abs()],axis=1).max(axis=1)
        atr=float(tr.tail(14).mean())
        p=float(c.iloc[-1]) if pd.isna(current) else float(current)
        ma20=float(c.tail(20).mean());ma50=float(c.tail(50).mean())
        ma200=float(c.tail(200).mean()) if len(c)>=200 else np.nan
        low20=float(l.tail(20).min())
        ret1=float(c.iloc[-1]/c.iloc[-2]-1) if len(c)>=2 else np.nan
        vol20=float(c.pct_change().tail(20).std(ddof=1)*np.sqrt(252))
        vol126=float(c.pct_change().tail(126).std(ddof=1)*np.sqrt(252))

        buy_low=max(0.01,p-0.65*atr)
        buy_high=p+0.10*atr
        no_chase=p+0.50*atr

        support_candidates=[low20-0.35*atr, ma50-1.25*atr]
        if not pd.isna(ma200):support_candidates.append(ma200-0.50*atr)
        invalid=max(0.01,max([x for x in support_candidates if not pd.isna(x) and x<p], default=p-2.5*atr))
        if invalid>=p: invalid=max(0.01,p-2.5*atr)

        entry_mid=(buy_low+buy_high)/2
        risk=max(entry_mid-invalid,0.01)
        risk_pct=risk/entry_mid
        review_2r=entry_mid+2*risk
        review_3r=entry_mid+3*risk

        return {
            "atr14":atr,"ma20_live":ma20,"ma50_live":ma50,"ma200_live":ma200,
            "ret_1d":ret1,"vol20":vol20,"vol126":vol126,
            "buy_zone_low":buy_low,"buy_zone_high":buy_high,
            "do_not_chase_above":no_chase,"invalidation_price":invalid,
            "risk_to_invalidation_pct":risk_pct,
            "review_price_2R":review_2r,"review_price_3R":review_3r,
            "volatility_spike":bool(not pd.isna(vol20) and not pd.isna(vol126) and vol20>1.5*vol126),
            "overextended":bool(p>ma50+1.75*atr),
            "gap_risk":bool(not pd.isna(ret1) and abs(ret1)>0.08),
        }
    except Exception:
        return {}

def news_check(ticker):
    try:
        news=yf.Ticker(yf_symbol(ticker)).news or []
        texts=[]
        for n in news[:12]:
            title=n.get("title") or ""
            summary=n.get("summary") or ""
            content=n.get("content") or {}
            if isinstance(content,dict):
                title=title or content.get("title") or ""
                summary=summary or content.get("summary") or content.get("description") or ""
            txt=f"{title} {summary}".strip()
            if txt:texts.append(txt)
        bad=[t for t in texts if NEGATIVE_NEWS.search(t)]
        return ("NEGATIVE_VETO" if bad else "CLEAR"), (" | ".join(bad[:2])[:500] if bad else "")
    except Exception:
        return "UNAVAILABLE",""

def load_b3_detail():
    p=OUT/"final_signals.csv"
    if not p.exists():return pd.DataFrame()
    b=pd.read_csv(p)
    b["ticker"]=b["ticker"].astype(str).str.upper()
    keep=[c for c in [
        "ticker","validated_score","final_confidence_score","final_signal",
        "quality_score","growth_score","valuation_score","momentum_score",
        "risk_score","trend_confirm","market_regime","failed_gates"
    ] if c in b]
    return b[keep].drop_duplicates("ticker")

def validation_map():
    if not VALID.exists():return {}
    try:
        v=pd.read_csv(VALID)
        return {str(r.asset_class):r for _,r in v.iterrows()}
    except:return {}

def class_forward_gate(cls,vm):
    r=vm.get(cls)
    if r is None:return "INSUFFICIENT",True
    n=num(getattr(r,"evaluated",np.nan))
    alpha=num(getattr(r,"avg_alpha_20d",np.nan))
    hit=num(getattr(r,"positive_alpha_rate",np.nan))
    if pd.isna(n) or n<6:return "INSUFFICIENT",True
    if (not pd.isna(alpha) and alpha<0) or (not pd.isna(hit) and hit<0.50):
        return "NEGATIVE_BLOCK",False
    return "POSITIVE",True

def portfolio_state():
    if not PORT.exists() or not PORT.stat().st_size:return pd.DataFrame(),np.nan
    try:
        p=pd.read_csv(PORT)
        if "ticker" not in p:return pd.DataFrame(),np.nan
        p["ticker"]=p["ticker"].astype(str).str.upper()
        if "current_value" not in p:
            q=pd.to_numeric(p.get("quantity"),errors="coerce")
            price=pd.to_numeric(p.get("price_now"),errors="coerce")
            if q is not None and price is not None:p["current_value"]=q*price
        total=pd.to_numeric(p.get("current_value"),errors="coerce").sum(skipna=True)
        return p,total
    except:return pd.DataFrame(),np.nan

def current_weight(ticker,p,total):
    if p.empty or pd.isna(total) or total<=0:return 0.0
    z=p[p.ticker.eq(ticker)]
    if z.empty:return 0.0
    return float(pd.to_numeric(z.get("current_value"),errors="coerce").sum(skipna=True)/total)

def base_gates(r,b3row=None):
    cls=str(r.get("asset_class",""))
    reasons=[]
    hard=True

    score=num(r.get("global_opportunity_score"))
    risk=num(r.get("risk_score"))
    momentum=num(r.get("momentum_score"))
    liq=num(r.get("liquidity_score"))
    dd=num(r.get("max_drawdown"))
    vol=num(r.get("volatility"))
    cp=num(r.get("class_percentile"))
    trend=boo(r.get("trend_confirm"))
    regime=str(r.get("regime",""))
    risks=str(r.get("risks","")).lower()

    if pd.isna(score) or score<88: reasons.append("score<88"); hard=False
    if not pd.isna(cp) and cp<0.88*100: reasons.append("fora_top12_classe"); hard=False
    if regime=="RISK_OFF":reasons.append("regime_RISK_OFF");hard=False
    if not trend:reasons.append("tendencia_nao_confirmada");hard=False
    if "risco elevado" in risks:reasons.append("risco_elevado");hard=False
    if not pd.isna(risk) and risk<58:reasons.append("risk_score<58");hard=False
    if not pd.isna(momentum) and momentum<62:reasons.append("momentum<62");hard=False
    if not pd.isna(liq) and liq<50:reasons.append("liquidez<50");hard=False
    if not pd.isna(vol) and vol>VOL_MAX.get(cls,.55):reasons.append("volatilidade_alta");hard=False
    if not pd.isna(dd) and dd<DD_MIN.get(cls,-.35):reasons.append("drawdown_excessivo");hard=False

    # Class-specific evidence.
    if cls=="AÇÃO GLOBAL / BDR":
        cov=boo(r.get("fundamental_coverage"))
        if not cov:reasons.append("fundamentos_globais_insuficientes");hard=False
        for c,limit,label in [
            ("quality_score",60,"qualidade<60"),
            ("growth_score",50,"crescimento<50"),
            ("valuation_score",35,"valuation<35"),
        ]:
            v=num(r.get(c))
            if not pd.isna(v) and v<limit:reasons.append(label);hard=False

    if cls=="FII":
        inc=num(r.get("income_score"))
        if not pd.isna(inc) and inc<60:reasons.append("income_score<60");hard=False

    if cls=="ETF":
        if not pd.isna(momentum) and momentum<68:reasons.append("ETF_momentum<68");hard=False

    if cls=="AÇÃO B3" and b3row is not None:
        vs=num(b3row.get("validated_score"))
        fc=num(b3row.get("final_confidence_score"))
        fs=str(b3row.get("final_signal",""))
        fg=str(b3row.get("failed_gates",""))
        if not pd.isna(vs) and vs<75:reasons.append("B3_validated_score<75");hard=False
        if not pd.isna(fc) and fc<75:reasons.append("B3_confidence<75");hard=False
        if fs not in {"AVALIAR_COMPRA","COMPRA_FORTE"}:reasons.append("B3_sem_sinal_compra");hard=False
        if fg and fg.lower() not in {"nan","none",""}:reasons.append("B3_failed_gates");hard=False

    return hard,reasons

def append_history(rows):
    today=datetime.now(timezone.utc).date().isoformat()
    hrows=[{
        "date":today,"ticker":r["ticker"],"asset_class":r["asset_class"],
        "preliminary_pass":bool(r["preliminary_pass"]),
        "score":r["global_opportunity_score"]
    } for r in rows]
    new=pd.DataFrame(hrows)
    old=pd.read_csv(HIST) if HIST.exists() and HIST.stat().st_size else pd.DataFrame()
    h=pd.concat([old,new],ignore_index=True)
    h=h.drop_duplicates(["date","ticker"],keep="last")
    h.to_csv(HIST,index=False)
    return h

def persistence(ticker,h):
    z=h[h.ticker.astype(str).eq(str(ticker))].copy()
    if z.empty:return 0,0
    z=z.sort_values("date").drop_duplicates("date",keep="last").tail(3)
    passes=int(z["preliminary_pass"].astype(str).str.lower().isin(["true","1"]).sum())
    return passes,len(z)

def main():
    if not RANK.exists():raise SystemExit("Run nubank_opportunity_radar.py first.")
    rank=pd.read_csv(RANK)
    rank["ticker"]=rank["ticker"].astype(str).str.upper()

    b3=load_b3_detail()
    b3map={r.ticker:r for _,r in b3.iterrows()} if not b3.empty else {}
    vm=validation_map()
    portfolio,total=portfolio_state()

    # Deep re-analysis only for the best part of the universe.
    shortlist=rank.sort_values("global_opportunity_score",ascending=False).head(80).copy()
    prelim=[]

    for _,r in shortlist.iterrows():
        d=r.to_dict()
        b3row=b3map.get(d["ticker"])
        ok,reasons=base_gates(d,b3row)

        plan=price_plan(d["ticker"],num(d.get("price_now")))
        if not plan:
            reasons.append("sem_plano_preco");ok=False
        else:
            if plan.get("volatility_spike"):reasons.append("spike_volatilidade");ok=False
            if plan.get("overextended"):reasons.append("preco_estendido");ok=False
            if plan.get("gap_risk"):reasons.append("gap_1d_excessivo");ok=False

        fwd_label,fwd_ok=class_forward_gate(str(d.get("asset_class")),vm)
        if not fwd_ok:
            reasons.append("validacao_forward_negativa")
            ok=False

        w=current_weight(d["ticker"],portfolio,total)
        cap=CAPS.get(str(d.get("asset_class")),0.08)
        if w>=cap:
            reasons.append("limite_concentracao_atingido");ok=False

        out={**d,**plan}
        out["forward_evidence"]=fwd_label
        out["current_portfolio_weight"]=w
        out["max_position_pct"]=cap
        out["preliminary_pass"]=bool(ok)
        out["preliminary_failures"]=",".join(reasons)
        prelim.append(out)

    hist=append_history(prelim)

    # News veto only after numeric/portfolio gates pass, to minimize noisy requests.
    candidates=[x for x in prelim if x["preliminary_pass"]]
    candidates=sorted(candidates,key=lambda z:z["global_opportunity_score"],reverse=True)[:25]
    news_map={}
    for x in candidates:
        news_map[x["ticker"]]=news_check(x["ticker"])

    final=[]
    for x in prelim:
        passes,obs=persistence(x["ticker"],hist)
        x["persistence_passes_last3"]=passes
        x["persistence_observations"]=obs
        x["news_check"],x["news_veto_reason"]=news_map.get(x["ticker"],("NOT_RUN",""))
        x["confirmed"]=False

        # The asset must first pass AVALIAR_COMPRA-quality filters in at least
        # two of the last three distinct daily observations.
        if not x["preliminary_pass"]:
            stage="NAO_COMPRAR"
        elif obs<2 or passes<2:
            stage="PRE_COMPRA"
        elif x["news_check"]=="NEGATIVE_VETO":
            stage="BLOQUEADO_NOTICIA"
        else:
            stage="COMPRA_CONFIRMADA"
            x["confirmed"]=True

        # Risk-budget sizing: target no more than ~0.75% portfolio loss at the
        # quantitative invalidation point, subject to class cap.
        rp=num(x.get("risk_to_invalidation_pct"))
        cap=num(x.get("max_position_pct"))
        if not pd.isna(rp) and rp>0:
            x["suggested_position_pct"]=float(min(cap,0.0075/rp))
        else:
            x["suggested_position_pct"]=float(cap) if not pd.isna(cap) else np.nan

        x["decision_stage"]=stage
        final.append(x)

    f=pd.DataFrame(final).sort_values(
        ["confirmed","global_opportunity_score"],ascending=[False,False]
    )

    # Existing holdings: systematic exit/reduce layer.
    def exit_rule(r):
        score=num(r.get("global_opportunity_score"))
        trend=boo(r.get("trend_confirm"))
        regime=str(r.get("regime",""))
        p=num(r.get("price_now"));ma200=num(r.get("ma200"))
        if not pd.isna(score) and score<38:return "SAIR_REAVALIAR"
        if regime=="RISK_OFF" and (not trend):return "REDUZIR_REAVALIAR"
        if not pd.isna(p) and not pd.isna(ma200) and p<ma200 and (pd.isna(score) or score<55):
            return "REDUZIR_REAVALIAR"
        if not pd.isna(score) and score<55:return "NAO_AUMENTAR"
        return "MANTER"

    f["holding_action"]=f.apply(exit_rule,axis=1)

    f.to_csv(OUT/"definitive_buy_analysis.csv",index=False)
    f[f.confirmed].to_csv(OUT/"confirmed_buys.csv",index=False)

    status={
        "updated_at":datetime.now(timezone.utc).isoformat(),
        "shortlist_reanalyzed":int(len(f)),
        "pre_buy":int((f.decision_stage=="PRE_COMPRA").sum()),
        "confirmed_buys":int(f.confirmed.sum()),
        "blocked_news":int((f.decision_stage=="BLOQUEADO_NOTICIA").sum()),
        "rule":"COMPRA_CONFIRMADA requires hard quantitative gates + price-plan checks + concentration check + at least 2 passes in last 3 daily observations + no negative-news veto.",
        "risk_budget":"Suggested size aims at <=0.75% portfolio risk to quantitative invalidation, capped by asset class.",
        "important":"COMPRA_CONFIRMADA is a high-evidence screen, not a guarantee of profit or a promise against loss."
    }
    (OUT/"definitive_buy_status.json").write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding="utf-8")

    cols=[c for c in [
        "ticker","name","asset_class","price_now","global_opportunity_score",
        "decision_stage","buy_zone_low","buy_zone_high","do_not_chase_above",
        "invalidation_price","review_price_2R","review_price_3R",
        "suggested_position_pct","persistence_passes_last3","news_check",
        "preliminary_failures","holding_action"
    ] if c in f]
    print(f[cols].head(40).to_string(index=False))

if __name__=="__main__":
    main()
