from pathlib import Path
import json
import numpy as np
import pandas as pd
import yfinance as yf

OUT=Path('results'); OUT.mkdir(exist_ok=True)
rank=pd.read_csv(OUT/'ranking.csv')
default={'quality':.25,'growth':.20,'valuation':.20,'momentum':.20,'risk':.15}; weights=default.copy(); calibration=False
p=OUT/'calibrated_weights.json'
if p.exists():
    c=json.loads(p.read_text())
    if c.get('accepted'): weights=c['weights']; calibration=True
rank['validated_score']=0.0
for k,w in weights.items():
    col=f'{k}_score'
    if col in rank: rank['validated_score']+=pd.to_numeric(rank[col],errors='coerce').fillna(0)*float(w)
vectorbt_ok=False; vb_params=None
p=OUT/'vectorbt_benchmark.json'
if p.exists():
    v=json.loads(p.read_text()); vectorbt_ok=bool(v.get('accepted_as_confirmation_engine',False)); vb_params=v.get('best_parameters')
tickers=rank['ticker'].dropna().astype(str).str.upper().drop_duplicates().tolist(); raw=yf.download([f'{t}.SA' for t in tickers]+['^BVSP','^VIX','BRL=X'],period='2y',interval='1d',auto_adjust=True,progress=False,threads=True,group_by='ticker')
def close(sym):
    try:
        x=raw[sym]['Close']; x=x.iloc[:,0] if isinstance(x,pd.DataFrame) else x; x=pd.to_numeric(x,errors='coerce').dropna(); x.index=pd.to_datetime(x.index).tz_localize(None); return x.sort_index()
    except Exception:return pd.Series(dtype=float)
ib=close('^BVSP'); vix=close('^VIX'); usd=close('BRL=X')
features={}; tech={}
for t in tickers:
    s=close(f'{t}.SA')
    if len(s)<220:
        features[t]=(np.nan,False,np.nan,np.nan,np.nan); tech[t]=False if vectorbt_ok else True; continue
    price=float(s.iloc[-1]); ma200=float(s.tail(200).mean()); ma50=float(s.tail(50).mean()); r6=float(s.iloc[-1]/s.iloc[-127]-1) if len(s)>=127 else np.nan; ibs=ib.reindex(s.index).ffill().dropna(); ib6=float(ibs.iloc[-1]/ibs.iloc[-127]-1) if len(ibs)>=127 else np.nan; rs=r6-ib6 if not pd.isna(r6) and not pd.isna(ib6) else np.nan; daily=s.pct_change().dropna(); vol20=float(daily.tail(20).std(ddof=1)*np.sqrt(252)) if len(daily)>=20 else np.nan; low=price*(1-.35*(vol20/np.sqrt(12))) if not pd.isna(vol20) else np.nan; trend=bool(price>ma200 and ma50>ma200 and r6>0 and not pd.isna(rs) and rs>0)
    features[t]=(price,trend,rs,low,price)
    if vectorbt_ok and vb_params:
        f,sl,m=int(vb_params['fast_ma']),int(vb_params['slow_ma']),int(vb_params['momentum_days']); tech[t]=bool(len(s)>=max(sl,m)+1 and s.tail(f).mean()>s.tail(sl).mean() and s.iloc[-1]>s.tail(sl).mean() and s.iloc[-1]/s.iloc[-m-1]-1>0)
    else: tech[t]=True
rank['price_now']=rank['ticker'].map({t:x[0] for t,x in features.items()}); rank['trend_confirm']=rank['ticker'].map({t:x[1] for t,x in features.items()}).fillna(False); rank['relative_strength_6m_final']=rank['ticker'].map({t:x[2] for t,x in features.items()}); rank['entry_watch_low']=rank['ticker'].map({t:x[3] for t,x in features.items()}); rank['entry_watch_high']=rank['ticker'].map({t:x[4] for t,x in features.items()}); rank['vectorbt_current_confirm']=rank['ticker'].map(tech).fillna(False)
regime=50.0
if len(ib)>=200: regime+=20 if ib.iloc[-1]>ib.tail(200).mean() else -20
if len(vix): regime+=10 if vix.iloc[-1]<20 else (-15 if vix.iloc[-1]>30 else 0)
if len(usd)>=64:
    r=usd.iloc[-1]/usd.iloc[-64]-1; regime+=-10 if r>.08 else (5 if r<-.05 else 0)
regime=float(np.clip(regime,0,100)); market='RISK_ON' if regime>=65 else ('RISK_OFF' if regime<=40 else 'NEUTRAL')
NEG=['fraud','fraude','investigation','investigação','default','bankruptcy','recuperação judicial','downgrade','loss warning','guidance cut','multa','corruption','corrupção']; rank['news_risk']=False; rank['news_headlines']=''
for idx in rank.sort_values('validated_score',ascending=False).head(30).index:
    q=str(rank.at[idx,'company_name']) if 'company_name' in rank.columns else str(rank.at[idx,'ticker'])
    try:
        sr=yf.Search(q,news_count=8); titles=[]
        for item in getattr(sr,'news',[]) or []:
            if not isinstance(item,dict):continue
            c=item.get('content'); title=str(c.get('title') or '') if isinstance(c,dict) else str(item.get('title') or '')
            if title:titles.append(title)
        txt=' | '.join(titles).lower(); rank.at[idx,'news_risk']=any(k in txt for k in NEG); rank.at[idx,'news_headlines']=' || '.join(titles[:4])
    except Exception:pass

def num(r,c,d=0):
    try:
        v=float(r.get(c,d)); return v if np.isfinite(v) else d
    except:return d

def decide(r):
    gates={'validated_calibration':calibration,'data_quality':str(r.get('data_quality','OK'))=='OK','validated_score':num(r,'validated_score')>=74,'quality':num(r,'quality_score')>=65,'valuation':num(r,'valuation_score')>=55,'momentum':num(r,'momentum_score')>=55,'risk':num(r,'risk_score')>=45,'trend_relative_strength':bool(r.get('trend_confirm',False)),'vectorbt':bool(r.get('vectorbt_current_confirm',True)),'market':market!='RISK_OFF','news':not bool(r.get('news_risk',False))}
    passed=sum(gates.values()); total=len(gates); conf=float(np.clip(.70*num(r,'validated_score')+.20*(100*passed/total)+.10*regime,0,100))
    if calibration and passed>=10 and num(r,'validated_score')>=80 and conf>=82: sig='COMPRA_FORTE'
    elif calibration and passed>=9 and num(r,'validated_score')>=74 and conf>=75: sig='AVALIAR_COMPRA'
    elif num(r,'validated_score')>=68: sig='WATCHLIST'
    else:sig='AGUARDAR'
    return pd.Series({'final_confidence_score':conf,'final_signal':sig,'gates_passed':passed,'gates_total':total,'failed_gates':','.join(k for k,v in gates.items() if not v),'market_regime':market,'market_regime_score':regime,'calibration_validated':calibration,'vectorbt_validated':vectorbt_ok})
rank=pd.concat([rank,rank.apply(decide,axis=1)],axis=1).sort_values('final_confidence_score',ascending=False); rank.to_csv(OUT/'final_signals.csv',index=False); rank[rank['final_signal'].isin(['COMPRA_FORTE','AVALIAR_COMPRA'])].to_csv(OUT/'final_opportunities.csv',index=False)
status={'market_regime':market,'market_regime_score':regime,'calibration_validated':calibration,'vectorbt_validated':vectorbt_ok,'strong_buys':int((rank['final_signal']=='COMPRA_FORTE').sum()),'evaluate_buys':int((rank['final_signal']=='AVALIAR_COMPRA').sum()),'watchlist':int((rank['final_signal']=='WATCHLIST').sum()),'important':'final_confidence_score is an evidence score, not a probability of profit.'}
(OUT/'final_signal_status.json').write_text(json.dumps(status,ensure_ascii=False,indent=2)); print(rank[['ticker','validated_score','final_confidence_score','final_signal','failed_gates']].head(25).to_string(index=False))
