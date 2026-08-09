from pathlib import Path
import json
import numpy as np
import pandas as pd
import requests
import yfinance as yf

PIT=Path('data/cvm_pit_fundamentals.csv.gz'); OUT=Path('results'); OUT.mkdir(exist_ok=True)
if not PIT.exists(): raise SystemExit('Falta data/cvm_pit_fundamentals.csv.gz')
pit=pd.read_csv(PIT,parse_dates=['reference_date','received_date'])
pit=pit[pit['reference_date'].dt.month.isin([3,6,9,12])].copy()
pit['ticker']=pit['ticker'].astype(str).str.upper(); pit['cnpj']=pit['cnpj'].astype(str); pit['version']=pd.to_numeric(pit.get('version',0),errors='coerce').fillna(0).astype(int)
tickers=pit['ticker'].dropna().drop_duplicates().tolist()
raw=yf.download([f'{t}.SA' for t in tickers]+['^BVSP'],period='7y',interval='1d',auto_adjust=False,progress=False,threads=True,group_by='ticker')

def field(sym,name):
    try:
        x=raw[sym][name]
        if isinstance(x,pd.DataFrame): x=x.iloc[:,0]
        x=pd.to_numeric(x,errors='coerce'); x.index=pd.to_datetime(x.index).tz_localize(None); return x.sort_index()
    except Exception:return pd.Series(dtype=float)

raw_close={}; adj={}; vol={}
for t in tickers:
    sym=f'{t}.SA'; raw_close[t]=field(sym,'Close'); a=field(sym,'Adj Close'); adj[t]=a if len(a) else raw_close[t]; vol[t]=field(sym,'Volume')
ibov=field('^BVSP','Adj Close'); ibov=ibov if len(ibov) else field('^BVSP','Close')
if len(ibov)<800: raise SystemExit('Histórico IBOV insuficiente')
monthly=pd.Series(ibov.index,index=ibov.index).groupby(ibov.index.to_period('M')).max(); dates=pd.DatetimeIndex(monthly.values); dates=dates[dates>=pd.Timestamp('2021-01-01')]

def last(s,dt):
    x=s[s.index<=dt].dropna(); return float(x.iloc[-1]) if len(x) else np.nan

def hist(s,dt,n): return s[s.index<=dt].dropna().tail(n)
def qid(dt):
    p=pd.Timestamp(dt).to_period('Q-DEC'); return int(p.year*4+p.quarter)

def company_snapshot(g,dt):
    a=g[g['received_date']<=dt].sort_values(['reference_date','received_date','version']).drop_duplicates('reference_date',keep='last')
    if a.empty:return None
    refs={qid(r['reference_date']):r for _,r in a.iterrows()}; latest_q=max(refs)
    qflows={}
    for q in sorted(refs):
        r=refs[q]; quarter=pd.Timestamp(r['reference_date']).quarter; vals={}
        for key,col in [('revenue','revenue_ytd_or_annual'),('op','operating_income_ytd_or_annual'),('ni','net_income_ytd_or_annual')]:
            cur=pd.to_numeric(pd.Series([r.get(col,np.nan)]),errors='coerce').iloc[0]
            if pd.isna(cur): vals[key]=np.nan
            elif quarter==1: vals[key]=float(cur)
            else:
                pr=refs.get(q-1); prev=pd.to_numeric(pd.Series([pr.get(col,np.nan) if pr is not None else np.nan]),errors='coerce').iloc[0]
                vals[key]=float(cur-prev) if not pd.isna(prev) else np.nan
        qflows[q]=vals
    def ttm(end,key):
        vals=[qflows.get(x,{}).get(key,np.nan) for x in range(end-3,end+1)]
        return float(sum(vals)) if all(not pd.isna(v) for v in vals) else np.nan
    def latest_nonnull(col):
        if col not in a.columns:return np.nan
        z=pd.to_numeric(a[col],errors='coerce').dropna(); return float(z.iloc[-1]) if len(z) else np.nan
    return {'revenue_ttm':ttm(latest_q,'revenue'),'op_ttm':ttm(latest_q,'op'),'ni_ttm':ttm(latest_q,'ni'),'revenue_prev':ttm(latest_q-4,'revenue'),'ni_prev':ttm(latest_q-4,'ni'),'equity':latest_nonnull('equity'),'debt':latest_nonnull('debt'),'shares':latest_nonnull('shares_outstanding'),'latest_ref':pd.Timestamp(refs[latest_q]['reference_date']),'latest_recv':pd.Timestamp(refs[latest_q]['received_date'])}

groups={(str(c),str(t)):g.copy() for (c,t),g in pit.groupby(['cnpj','ticker'])}

def div(a,b): return float(a/b) if not pd.isna(a) and not pd.isna(b) and b!=0 else np.nan
def growth(a,b): return float((a-b)/abs(b)) if not pd.isna(a) and not pd.isna(b) and b!=0 else np.nan

def get_cdi():
    try:
        url='https://api.bcb.gov.br/dados/serie/bcdata.sgs.12/dados'; p={'formato':'json','dataInicial':dates.min().strftime('%d/%m/%Y'),'dataFinal':dates.max().strftime('%d/%m/%Y')}
        r=requests.get(url,params=p,timeout=20); r.raise_for_status(); d=pd.DataFrame(r.json()); d['date']=pd.to_datetime(d['data'],dayfirst=True,errors='coerce'); d['ret']=pd.to_numeric(d['valor'].astype(str).str.replace(',','.',regex=False),errors='coerce')/100
        s=d.dropna(subset=['date','ret']).set_index('date')['ret'].sort_index(); return (1+s).cumprod()
    except Exception as e: print('[WARN] CDI',e); return pd.Series(dtype=float)
cf=get_cdi()
def cdiret(a,b):
    if cf.empty:return np.nan
    x=cf[cf.index<=a]; y=cf[cf.index<=b]
    return float(y.iloc[-1]/x.iloc[-1]-1) if len(x) and len(y) else np.nan

frames=[]
for i,dt in enumerate(dates[:-1]):
    nxt=dates[i+1]; rows=[]
    for (cnpj,t),g in groups.items():
        f=company_snapshot(g,dt)
        if f is None:continue
        p=last(raw_close.get(t,pd.Series(dtype=float)),dt); pa=last(adj.get(t,pd.Series(dtype=float)),dt); pn=last(adj.get(t,pd.Series(dtype=float)),nxt)
        h=hist(adj.get(t,pd.Series(dtype=float)),dt,270); vh=hist(vol.get(t,pd.Series(dtype=float)),dt,63)
        if pd.isna(p) or pd.isna(pa) or len(h)<200:continue
        av=np.nan
        if len(vh)>=40:
            pp=raw_close[t].reindex(vh.index); av=float((pp*vh).dropna().mean())
        mcap=p*f['shares'] if not pd.isna(f['shares']) and f['shares']>0 else np.nan
        daily=h.pct_change().dropna(); r6=float(h.iloc[-1]/h.iloc[-127]-1) if len(h)>=127 else np.nan; r12=float(h.iloc[-1]/h.iloc[-253]-1) if len(h)>=253 else np.nan; vv=float(daily.tail(63).std(ddof=1)*np.sqrt(252)) if len(daily)>=63 else np.nan; dd=float((h/h.cummax()-1).tail(252).min())
        ib0=last(ibov,dt); ib1=last(ibov,nxt)
        rows.append({'date':dt,'next_date':nxt,'ticker':t,'cnpj':cnpj,'sector':g['sector'].dropna().iloc[-1] if 'sector' in g and g['sector'].notna().any() else 'Unknown','price':p,'avg_daily_volume_brl':av,'roe':div(f['ni_ttm'],f['equity']),'operating_margin':div(f['op_ttm'],f['revenue_ttm']),'profit_margin':div(f['ni_ttm'],f['revenue_ttm']),'debt_to_equity':div(f['debt'],f['equity']),'revenue_growth':growth(f['revenue_ttm'],f['revenue_prev']),'earnings_growth':growth(f['ni_ttm'],f['ni_prev']),'earnings_yield':div(f['ni_ttm'],mcap),'book_to_price':div(f['equity'],mcap),'ret_6m':r6,'ret_12m':r12,'volatility':vv,'max_drawdown':dd,'fwd_return':pn/pa-1 if not pd.isna(pn) and pa>0 else np.nan,'ibov_return':ib1/ib0-1 if ib0>0 else np.nan,'cdi_return':cdiret(dt,nxt),'latest_reference_date':f['latest_ref'],'latest_received_date':f['latest_recv']})
    x=pd.DataFrame(rows)
    if x.empty:continue
    x=x[x['price'].ge(2)&x['avg_daily_volume_brl'].fillna(0).ge(1_000_000)].copy()
    x=x.sort_values(['cnpj','avg_daily_volume_brl'],ascending=[True,False]).drop_duplicates('cnpj')
    if len(x)>=20:frames.append(x); print(dt.date(),len(x),flush=True)
panel=pd.concat(frames,ignore_index=True)
if panel.empty or panel['date'].nunique()<24: raise SystemExit('Painel PIT insuficiente')

def rank_metric(df,col,pos=True):
    v=pd.to_numeric(df[col],errors='coerce'); global_r=v.rank(pct=True)*100; global_r=global_r if pos else 100-global_r; out=pd.Series(index=df.index,dtype=float)
    for _,idx in df.groupby(df['sector'].fillna('Unknown')).groups.items():
        idx=list(idx); local=v.loc[idx]
        if local.notna().sum()>=5:
            r=local.rank(pct=True)*100; r=r if pos else 100-r; out.loc[idx]=r
        else: out.loc[idx]=global_r.loc[idx]
    return out

scored=[]
for dt,g in panel.groupby('date'):
    x=g.copy()
    for c in ['roe','operating_margin','profit_margin','debt_to_equity','revenue_growth','earnings_growth','earnings_yield','book_to_price','ret_6m','ret_12m','volatility','max_drawdown']:
        s=pd.to_numeric(x[c],errors='coerce')
        if s.notna().sum()>=10: x[c]=s.clip(s.quantile(.02),s.quantile(.98))
    x['quality_score']=pd.concat([rank_metric(x,'roe'),rank_metric(x,'operating_margin'),rank_metric(x,'profit_margin'),rank_metric(x,'debt_to_equity',False)],axis=1).mean(axis=1)
    x['growth_score']=pd.concat([rank_metric(x,'revenue_growth'),rank_metric(x,'earnings_growth')],axis=1).mean(axis=1)
    x['valuation_score']=pd.concat([rank_metric(x,'earnings_yield'),rank_metric(x,'book_to_price')],axis=1).mean(axis=1)
    x['momentum_score']=pd.concat([rank_metric(x,'ret_6m'),rank_metric(x,'ret_12m')],axis=1).mean(axis=1)
    x['risk_score']=pd.concat([rank_metric(x,'volatility',False),rank_metric(x,'max_drawdown')],axis=1).mean(axis=1)
    x['coverage']=x[['quality_score','growth_score','valuation_score','momentum_score','risk_score']].notna().mean(axis=1); scored.append(x[x['coverage']>=.8])
panel=pd.concat(scored,ignore_index=True); dts=pd.Index(sorted(panel['date'].unique())); n=len(dts)
train=dts[:int(n*.55)]; valid=dts[int(n*.55):int(n*.75)]; test=dts[int(n*.75):]
cols=['quality_score','growth_score','valuation_score','momentum_score','risk_score']; default=np.array([.25,.20,.20,.20,.15]); rng=np.random.default_rng(42)
cands=[default,np.array([.35,.15,.25,.15,.10]),np.array([.20,.15,.35,.20,.10]),np.array([.20,.15,.15,.35,.15]),np.array([.20,.20,.20,.20,.20])]+list(rng.dirichlet(np.ones(5)*2,500))

def bt(w,use_dates):
    rows=[]; prev={}
    for dt in use_dates:
        g=panel[panel['date']==dt].dropna(subset=['fwd_return']).copy()
        if len(g)<15:continue
        g['score']=sum(pd.to_numeric(g[c],errors='coerce').fillna(0)*float(x) for c,x in zip(cols,w)); top=g.sort_values('score',ascending=False).head(min(10,len(g)))
        cur={t:1/len(top) for t in top['ticker']}; keys=set(prev)|set(cur); overlap=sum(min(prev.get(k,0),cur.get(k,0)) for k in keys); turnover=1-overlap if prev else 1; cost=.002*turnover
        gross=float(top['fwd_return'].mean()); net=gross-cost; ib=float(top['ibov_return'].mean()); cs=pd.to_numeric(top['cdi_return'],errors='coerce'); cd=float(cs.mean()) if cs.notna().any() else np.nan; bench=max(ib,cd) if not pd.isna(cd) else ib
        rows.append({'date':dt,'net_return':net,'ibov_return':ib,'cdi_return':cd,'benchmark_return':bench,'excess_return':net-bench,'turnover':turnover}); prev=cur
    return pd.DataFrame(rows)

def perf(z):
    if z is None or len(z)<6:return None
    r=z['net_return']; b=z['benchmark_return']; ib=z['ibov_return']; w=(1+r).cumprod(); bw=(1+b).cumprod(); iw=(1+ib).cumprod(); yrs=len(z)/12
    cagr=float(w.iloc[-1]**(1/yrs)-1); bc=float(bw.iloc[-1]**(1/yrs)-1); ic=float(iw.iloc[-1]**(1/yrs)-1); cds=z['cdi_return'].dropna(); cdc=None
    if len(cds)>=max(3,int(len(z)*.5)): cdc=float((1+cds).prod()**(12/len(cds))-1)
    e=z['excess_return']; es=float(e.mean()/e.std(ddof=1)*np.sqrt(12)) if e.std(ddof=1)>0 else np.nan; dd=float((w/w.cummax()-1).min()); idd=float((iw/iw.cummax()-1).min())
    return {'months':int(len(z)),'cagr':cagr,'benchmark_cagr_max_ibov_cdi':bc,'ibov_cagr':ic,'cdi_cagr':cdc,'avg_monthly_excess':float(e.mean()),'positive_excess_rate':float((e>0).mean()),'excess_sharpe':es,'max_drawdown':dd,'ibov_max_drawdown':idd,'avg_turnover':float(z['turnover'].mean())}

def obj(m):
    if not m:return -1e9
    s=m['excess_sharpe']; s=-10 if pd.isna(s) else s
    return s+2*m['avg_monthly_excess']+.2*m['positive_excess_rate']+.1*m['max_drawdown']
tr=[]
for i,w in enumerate(cands):
    m=perf(bt(w,train))
    if m:tr.append((obj(m),i,m))
tr.sort(reverse=True,key=lambda x:x[0]); va=[]
for _,i,tm in tr[:40]:
    m=perf(bt(cands[i],valid))
    if m:va.append((obj(m),i,tm,m))
va.sort(reverse=True,key=lambda x:x[0])
if not va:raise SystemExit('Sem candidato válido')
_,bi,tm,vm=va[0]; best=np.array(cands[bi]); tz=bt(best,test); testm=perf(tz); defaultm=perf(bt(default,test))
accepted=bool(testm and testm['cagr']>testm['ibov_cagr'] and (testm['cdi_cagr'] is None or testm['cagr']>testm['cdi_cagr']) and testm['avg_monthly_excess']>0 and testm['positive_excess_rate']>=.50 and testm['excess_sharpe']>0 and testm['max_drawdown']>=testm['ibov_max_drawdown'])
weights={k:float(v) for k,v in zip(['quality','growth','valuation','momentum','risk'],best)}
summary={'accepted':accepted,'weights':weights,'train':tm,'validation':vm,'untouched_test':testm,'default_weights_test':defaultm,'periods':{'train':[str(pd.Timestamp(train.min()).date()),str(pd.Timestamp(train.max()).date())],'validation':[str(pd.Timestamp(valid.min()).date()),str(pd.Timestamp(valid.max()).date())],'untouched_test':[str(pd.Timestamp(test.min()).date()),str(pd.Timestamp(test.max()).date())]},'fundamental_method':'CVM PIT by received_date; ITR YTD flows de-accumulated into calendar quarters; TTM=sum of four consecutive quarter flows.','portfolio_method':'Monthly equal-weight Top10; one most-liquid share class per CNPJ; price>=R$2; avg traded value>=R$1m/day; 20 bps turnover cost.','acceptance_rule':'Untouched test must beat IBOV CAGR, beat CDI CAGR when available, have positive avg excess, >=50% positive excess months, positive excess Sharpe, and max drawdown no worse than IBOV.','limitations':['Current observed issuer universe still causes survivorship bias.','Historical market cap approximates total reported shares times the most-liquid share-class price.','Non-calendar fiscal-year companies are excluded.','Yahoo Finance is free/non-official.']}
(OUT/'calibrated_weights.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,default=str)); panel.to_csv(OUT/'fundamental_pit_backtest_panel.csv.gz',index=False,compression='gzip'); tz.to_csv(OUT/'fundamental_pit_test_monthly.csv',index=False)
print(json.dumps(summary,ensure_ascii=False,indent=2,default=str))
