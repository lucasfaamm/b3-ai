from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
import yfinance as yf

OUT=Path('results'); OUT.mkdir(exist_ok=True); rp=OUT/'ranking.csv'
if not rp.exists(): raise SystemExit('Falta results/ranking.csv')
rank=pd.read_csv(rp)
W={'quality':.25,'growth':.20,'valuation':.20,'momentum':.20,'risk':.15}
rank['validated_score']=0.0
for k,w in W.items():
    c=f'{k}_score'
    if c in rank.columns: rank['validated_score']+=pd.to_numeric(rank[c],errors='coerce').fillna(0)*w
fwd={}; p=OUT/'forward_validation_summary.json'
if p.exists():
    try: fwd=json.loads(p.read_text(encoding='utf-8'))
    except Exception: pass
strong=bool(fwd.get('strong_buy_unlock',False))

tickers=rank['ticker'].dropna().astype(str).str.upper().drop_duplicates().tolist()
raw=yf.download([f'{t}.SA' for t in tickers]+['^BVSP','^VIX','BRL=X'],period='2y',interval='1d',auto_adjust=True,progress=False,threads=True,group_by='ticker')
def close(sym):
    try:
        x=raw[sym]['Close']; x=x.iloc[:,0] if isinstance(x,pd.DataFrame) else x
        x=pd.to_numeric(x,errors='coerce').dropna(); x.index=pd.to_datetime(x.index).tz_localize(None); return x.sort_index()
    except Exception: return pd.Series(dtype=float)
ib=close('^BVSP'); vx=close('^VIX'); usd=close('BRL=X')
def feat(t):
    s=close(f'{t}.SA')
    if len(s)<220:return {'price_now':np.nan,'trend_confirm':False,'relative_strength_6m':np.nan,'entry_watch_low':np.nan,'entry_watch_high':np.nan}
    p=float(s.iloc[-1]); ma50=float(s.tail(50).mean()); ma200=float(s.tail(200).mean()); r6=float(s.iloc[-1]/s.iloc[-127]-1) if len(s)>=127 else np.nan
    bi=ib.reindex(s.index).ffill().dropna(); br=float(bi.iloc[-1]/bi.iloc[-127]-1) if len(bi)>=127 else np.nan; rs=r6-br if not pd.isna(r6) and not pd.isna(br) else np.nan
    d=s.pct_change().dropna(); vol=float(d.tail(20).std(ddof=1)*np.sqrt(252)) if len(d)>=20 else np.nan; mv=vol/np.sqrt(12) if not pd.isna(vol) else np.nan
    return {'price_now':p,'trend_confirm':bool(p>ma200 and ma50>ma200 and not pd.isna(r6) and r6>0 and not pd.isna(rs) and rs>0),'relative_strength_6m':rs,'entry_watch_low':p*(1-.35*mv) if not pd.isna(mv) else np.nan,'entry_watch_high':p}
fm={t:feat(t) for t in tickers}
for c in ['price_now','trend_confirm','relative_strength_6m','entry_watch_low','entry_watch_high']: rank[c]=rank['ticker'].map({t:v[c] for t,v in fm.items()})
reg=50.0
if len(ib)>=200: reg+=20 if ib.iloc[-1]>ib.tail(200).mean() else -20
if len(vx): reg+=10 if vx.iloc[-1]<20 else (-15 if vx.iloc[-1]>30 else 0)
if len(usd)>=64:
    u=float(usd.iloc[-1]/usd.iloc[-64]-1); reg+=-10 if u>.08 else (5 if u<-.05 else 0)
reg=float(np.clip(reg,0,100)); regime='RISK_ON' if reg>=65 else ('RISK_OFF' if reg<=40 else 'NEUTRAL')
rank['news_risk']=False
NEG=['fraud','fraude','investigation','investigação','default','bankruptcy','recuperação judicial','downgrade','guidance cut','multa','corruption','corrupção']
for idx in rank.sort_values('validated_score',ascending=False).head(30).index:
    q=str(rank.at[idx,'company_name']) if 'company_name' in rank.columns else str(rank.at[idx,'ticker'])
    try:
        sr=yf.Search(q,news_count=8); titles=[]
        for item in getattr(sr,'news',[]) or []:
            if isinstance(item,dict):
                ct=item.get('content'); title=str(ct.get('title') or '') if isinstance(ct,dict) else str(item.get('title') or '')
                if title: titles.append(title)
        rank.at[idx,'news_risk']=any(k in ' | '.join(titles).lower() for k in NEG)
    except Exception: pass

def num(r,c,d=0):
    try:
        v=float(r.get(c,d)); return v if np.isfinite(v) else d
    except Exception:return d
def dec(r):
    g={'data':str(r.get('data_quality','OK'))=='OK','score':num(r,'validated_score')>=72,'quality':num(r,'quality_score')>=60,'valuation':num(r,'valuation_score')>=50,'momentum':num(r,'momentum_score')>=50,'risk':num(r,'risk_score')>=40,'trend':bool(r.get('trend_confirm',False)),'market':regime!='RISK_OFF','news':not bool(r.get('news_risk',False))}
    passed=sum(g.values()); conf=float(np.clip(.75*num(r,'validated_score')+.15*(100*passed/len(g))+.10*reg,0,100))
    sig='COMPRA_FORTE' if strong and passed>=8 and num(r,'validated_score')>=80 and conf>=82 else ('AVALIAR_COMPRA' if passed>=7 and num(r,'validated_score')>=74 and regime!='RISK_OFF' else ('WATCHLIST' if num(r,'validated_score')>=68 else 'AGUARDAR'))
    return pd.Series({'final_confidence_score':conf,'final_signal':sig,'gates_passed':passed,'gates_total':len(g),'failed_gates':','.join(k for k,v in g.items() if not v),'market_regime':regime,'market_regime_score':reg,'strong_buy_authorized':strong})
rank=pd.concat([rank,rank.apply(dec,axis=1)],axis=1).sort_values('final_confidence_score',ascending=False)
rank.to_csv(OUT/'final_signals.csv',index=False); rank[rank.final_signal.isin(['COMPRA_FORTE','AVALIAR_COMPRA'])].to_csv(OUT/'final_opportunities.csv',index=False)
status={'market_regime':regime,'market_regime_score':reg,'strong_buy_authorized':strong,'strong_buys':int((rank.final_signal=='COMPRA_FORTE').sum()),'evaluate_buys':int((rank.final_signal=='AVALIAR_COMPRA').sum()),'watchlist':int((rank.final_signal=='WATCHLIST').sum()),'deployed_weights':W,'vectorbt_weight':0,'qlib_weight':0,'forward_months_evaluated':fwd.get('new_forward_months_evaluated',0),'important':'Evidence score, not probability of profit. No automatic orders.'}
(OUT/'final_signal_status.json').write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf-8'); print(rank[['ticker','validated_score','final_confidence_score','final_signal','failed_gates']].head(25).to_string(index=False))
