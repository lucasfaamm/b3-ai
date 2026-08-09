from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone
import json
import numpy as np
import pandas as pd
import requests
import yfinance as yf

OUT=Path('results'); DATA=Path('data'); OUT.mkdir(exist_ok=True); DATA.mkdir(exist_ok=True)
SIGNALS=OUT/'final_signals.csv'; LOG=DATA/'forward_validation_log.csv'; H=20; TOP_N=10
if not SIGNALS.exists(): raise SystemExit('Falta results/final_signals.csv')
signals=pd.read_csv(SIGNALS)
if 'validated_score' not in signals.columns: raise SystemExit('final_signals.csv sem validated_score')

today=pd.Timestamp.now(tz='UTC').normalize().tz_localize(None)
month=today.to_period('M').strftime('%Y-%m')
top=signals.sort_values('validated_score',ascending=False).head(TOP_N).copy()
new=top[['ticker','validated_score']].copy(); new['signal_month']=month; new['signal_date']=today.date().isoformat()
log=pd.read_csv(LOG) if LOG.exists() and LOG.stat().st_size else pd.DataFrame()
log=pd.concat([log,new],ignore_index=True).drop_duplicates(['ticker','signal_month'],keep='first')
for c in ['entry_date','entry_price','target_date','target_price','return_20d','ibov_20d','alpha_vs_ibov_20d','cdi_20d','excess_vs_cdi_20d']:
    if c not in log.columns: log[c]=np.nan

tickers=log['ticker'].dropna().astype(str).str.upper().unique().tolist()
raw=yf.download([f'{t}.SA' for t in tickers]+['^BVSP'],period='3y',interval='1d',auto_adjust=True,progress=False,threads=True,group_by='ticker')
def close(sym):
    try:
        x=raw[sym]['Close']; x=x.iloc[:,0] if isinstance(x,pd.DataFrame) else x
        x=pd.to_numeric(x,errors='coerce').dropna(); x.index=pd.to_datetime(x.index).tz_localize(None); return x.sort_index()
    except Exception: return pd.Series(dtype=float)
ibov=close('^BVSP')

def get_cdi(start,end):
    try:
        url='https://api.bcb.gov.br/dados/serie/bcdata.sgs.12/dados'
        p={'formato':'json','dataInicial':start.strftime('%d/%m/%Y'),'dataFinal':end.strftime('%d/%m/%Y')}
        r=requests.get(url,params=p,timeout=20); r.raise_for_status(); d=pd.DataFrame(r.json())
        d['date']=pd.to_datetime(d['data'],dayfirst=True,errors='coerce')
        d['ret']=pd.to_numeric(d['valor'].astype(str).str.replace(',','.',regex=False),errors='coerce')/100.0
        return d.dropna(subset=['date','ret']).set_index('date')['ret'].sort_index()
    except Exception as e:
        print('[WARN] CDI',e); return pd.Series(dtype=float)
start=pd.to_datetime(log['signal_date'],errors='coerce').min(); cdi=get_cdi(start-pd.Timedelta(days=10),today+pd.Timedelta(days=10)) if not pd.isna(start) else pd.Series(dtype=float)
def cdi_between(a,b):
    if cdi.empty: return np.nan
    z=cdi[(cdi.index>a)&(cdi.index<=b)]; return float((1+z).prod()-1) if len(z) else 0.0

for i,r in log.iterrows():
    s=close(f"{str(r['ticker']).upper()}.SA")
    if s.empty: continue
    sig=pd.Timestamp(r['signal_date']); pos=int(s.index.searchsorted(sig))
    if pos>=len(s): continue
    entry_date=s.index[pos]; log.at[i,'entry_date']=entry_date.date().isoformat(); log.at[i,'entry_price']=float(s.iloc[pos])
    if pos+H>=len(s): continue
    target_date=s.index[pos+H]; ret=float(s.iloc[pos+H]/s.iloc[pos]-1)
    log.at[i,'target_date']=target_date.date().isoformat(); log.at[i,'target_price']=float(s.iloc[pos+H]); log.at[i,'return_20d']=ret
    if len(ibov):
        a=int(ibov.index.searchsorted(entry_date)); b=int(ibov.index.searchsorted(target_date))
        if a<len(ibov) and b<len(ibov):
            ib=float(ibov.iloc[b]/ibov.iloc[a]-1); log.at[i,'ibov_20d']=ib; log.at[i,'alpha_vs_ibov_20d']=ret-ib
    cd=cdi_between(entry_date,target_date)
    if not pd.isna(cd): log.at[i,'cdi_20d']=cd; log.at[i,'excess_vs_cdi_20d']=ret-cd
log.to_csv(LOG,index=False)

e=log.dropna(subset=['return_20d']).copy()
if len(e):
    m=e.groupby('signal_month').agg(n=('ticker','size'),avg_return_20d=('return_20d','mean'),avg_ibov_20d=('ibov_20d','mean'),avg_alpha_vs_ibov_20d=('alpha_vs_ibov_20d','mean'),avg_cdi_20d=('cdi_20d','mean'),avg_excess_vs_cdi_20d=('excess_vs_cdi_20d','mean')).reset_index()
else:
    m=pd.DataFrame(columns=['signal_month','n','avg_return_20d','avg_ibov_20d','avg_alpha_vs_ibov_20d','avg_cdi_20d','avg_excess_vs_cdi_20d'])
m.to_csv(OUT/'forward_validation_portfolio.csv',index=False)

def boot(s,block=3,n_boot=10000,seed=123):
    a=pd.Series(s,dtype=float).dropna().to_numpy()
    if len(a)<8: return {'n':int(len(a)),'mean':float(a.mean()) if len(a) else None,'ci95_low':None,'ci95_high':None}
    rng=np.random.default_rng(seed); n=len(a); means=np.empty(n_boot)
    for k in range(n_boot):
        sample=[]
        while len(sample)<n:
            st=int(rng.integers(0,n))
            for j in range(block):
                sample.append(a[(st+j)%n])
                if len(sample)>=n: break
        means[k]=np.mean(sample[:n])
    return {'n':int(n),'mean':float(a.mean()),'ci95_low':float(np.quantile(means,.025)),'ci95_high':float(np.quantile(means,.975))}

a=m['avg_alpha_vs_ibov_20d'].dropna(); c=m['avg_excess_vs_cdi_20d'].dropna(); abi=boot(a); cbi=boot(c)
ah=float((a>0).mean()) if len(a) else None; ch=float((c>0).mean()) if len(c) else None
unlock=bool(len(m)>=24 and len(a)>=24 and len(c)>=24 and a.mean()>0 and c.mean()>0 and ah>=.5 and ch>=.5 and abi['ci95_low'] is not None and abi['ci95_low']>0 and cbi['ci95_low'] is not None and cbi['ci95_low']>0)
summary={'frozen_rule':'Top10 fixed weights 25/20/20/20/15; Qlib=0; VectorBT=0','new_forward_months_evaluated':int(len(m)),'avg_alpha_vs_ibov_20d':float(a.mean()) if len(a) else None,'positive_alpha_rate_vs_ibov':ah,'avg_excess_vs_cdi_20d':float(c.mean()) if len(c) else None,'positive_excess_rate_vs_cdi':ch,'bootstrap_alpha_vs_ibov':abi,'bootstrap_excess_vs_cdi':cbi,'strong_buy_unlock':unlock,'unlock_rule':'24 NEW monthly observations plus positive mean/hit-rate and 95% bootstrap lower bound >0 versus both IBOV and CDI','updated_at':datetime.now(timezone.utc).isoformat()}
(OUT/'forward_validation_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(summary,ensure_ascii=False,indent=2))
