from pathlib import Path
import json
import numpy as np
import pandas as pd
import requests
import vectorbt as vbt
import yfinance as yf

OUT=Path('results'); OUT.mkdir(exist_ok=True)
summary=json.loads((OUT/'vectorbt_summary.json').read_text())
params=summary['best_parameters']; split=pd.Timestamp(summary['train_test_split'])
fund=pd.read_csv('data/fundamentals.csv')
tickers=fund['ticker'].dropna().astype(str).str.upper().drop_duplicates().tolist()[:120]
raw=yf.download([f'{t}.SA' for t in tickers]+['^BVSP'],period='5y',interval='1d',auto_adjust=True,progress=False,threads=True,group_by='ticker')

def close(sym):
    try:
        x=raw[sym]['Close']
        if isinstance(x,pd.DataFrame): x=x.iloc[:,0]
        x=pd.to_numeric(x,errors='coerce'); x.index=pd.to_datetime(x.index).tz_localize(None)
        return x.sort_index()
    except Exception: return pd.Series(dtype=float)

panel=pd.DataFrame({t:close(f'{t}.SA') for t in tickers}).sort_index()
panel=panel.dropna(axis=1,thresh=max(400,int(len(panel)*.55))).ffill(limit=3)
ibov=close('^BVSP').reindex(panel.index).ffill()
fast,slow,mom=int(params['fast_ma']),int(params['slow_ma']),int(params['momentum_days'])
fast_ma=panel.rolling(fast,min_periods=fast).mean(); slow_ma=panel.rolling(slow,min_periods=slow).mean(); momentum=panel/panel.shift(mom)-1
entries=((fast_ma>slow_ma)&(momentum>0)&(panel>slow_ma)).fillna(False)
exits=((fast_ma<slow_ma)|(momentum<0)).fillna(False)
pf=vbt.Portfolio.from_signals(panel,entries,exits,init_cash=100000,fees=.0005,slippage=.0005,freq='1D')
r=pf.returns(); r=r.to_frame() if isinstance(r,pd.Series) else r
strategy=r.mean(axis=1,skipna=True).fillna(0); strategy=strategy[strategy.index>=split]
ibret=ibov.pct_change().fillna(0).reindex(strategy.index).fillna(0)

def cdi(start,end):
    try:
        url='https://api.bcb.gov.br/dados/serie/bcdata.sgs.12/dados'; p={'formato':'json','dataInicial':start.strftime('%d/%m/%Y'),'dataFinal':end.strftime('%d/%m/%Y')}
        rr=requests.get(url,params=p,timeout=20); rr.raise_for_status(); d=pd.DataFrame(rr.json())
        d['date']=pd.to_datetime(d['data'],dayfirst=True,errors='coerce'); d['ret']=pd.to_numeric(d['valor'].astype(str).str.replace(',','.',regex=False),errors='coerce')/100
        return d.dropna(subset=['date','ret']).set_index('date')['ret'].sort_index()
    except Exception as e:
        print('[WARN] CDI:',e); return pd.Series(dtype=float)

cd=cdi(strategy.index.min(),strategy.index.max()); cdret=cd.reindex(strategy.index).fillna(0) if len(cd) else pd.Series(dtype=float)

def metrics(s):
    s=pd.Series(s,dtype=float).dropna()
    if len(s)<30:return {}
    w=(1+s).cumprod(); years=len(s)/252; cagr=float(w.iloc[-1]**(1/years)-1); vol=float(s.std(ddof=1)*np.sqrt(252)); sh=float(s.mean()*252/vol) if vol>0 else np.nan; dd=float((w/w.cummax()-1).min())
    return {'days':int(len(s)),'total_return':float(w.iloc[-1]-1),'cagr':cagr,'volatility':vol,'sharpe_rf0':sh,'max_drawdown':dd}

ms,mi,mc=metrics(strategy),metrics(ibret),metrics(cdret) if len(cdret) else {}
accepted=bool(ms and mi and ms['cagr']>mi['cagr'] and ms['sharpe_rf0']>mi['sharpe_rf0'] and (not mc or ms['cagr']>mc['cagr']))
out={'period_start':str(strategy.index.min().date()),'period_end':str(strategy.index.max().date()),'best_parameters':params,'strategy':ms,'ibov':mi,'cdi':mc,'beats_ibov_cagr':bool(ms and mi and ms['cagr']>mi['cagr']),'beats_ibov_sharpe':bool(ms and mi and ms['sharpe_rf0']>mi['sharpe_rf0']),'beats_cdi_cagr':bool(ms and mc and ms['cagr']>mc['cagr']) if mc else None,'accepted_as_confirmation_engine':accepted,'rule':'Accepted only if OOS CAGR and Sharpe beat IBOV and CAGR beats CDI when available.'}
(OUT/'vectorbt_benchmark.json').write_text(json.dumps(out,ensure_ascii=False,indent=2,default=str))
print(json.dumps(out,ensure_ascii=False,indent=2,default=str))
