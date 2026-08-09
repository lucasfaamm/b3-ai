
from __future__ import annotations

from pathlib import Path
import json, math
import numpy as np
import pandas as pd
import requests
import yfinance as yf
import qlib.contrib.model.gbdt as qlib_gbdt

FUND = Path("data/fundamentals.csv")
OUT = Path("results"); OUT.mkdir(exist_ok=True)

MAX_TICKERS = 110
PERIOD = "7y"
STEP = 5
HORIZONS = [20, 60]
ROUND_TRIP_COST = 0.002
MIN_EXTRA_ALPHA = {20: 0.005, 60: 0.010}
TOP_FRACTIONS = [0.05, 0.10, 0.15, 0.20, 0.25]

if not FUND.exists():
    raise SystemExit("Falta data/fundamentals.csv")

fund = pd.read_csv(FUND)
fund["ticker"] = fund["ticker"].astype(str).str.upper()
fund["sector"] = fund.get("sector", "Unknown").fillna("Unknown").astype(str)

tickers = fund["ticker"].drop_duplicates().tolist()[:MAX_TICKERS]
sector_map = fund.drop_duplicates("ticker").set_index("ticker")["sector"].to_dict()

stock_symbols = [f"{t}.SA" for t in tickers]
market_symbols = ["^BVSP","^GSPC","^VIX","BRL=X","EWZ","CL=F","GC=F"]

raw = yf.download(
    stock_symbols + market_symbols,
    period=PERIOD,
    interval="1d",
    auto_adjust=True,
    progress=False,
    threads=True,
    group_by="ticker",
)

def ser(symbol, field="Close"):
    try:
        x = raw[symbol][field]
        if isinstance(x, pd.DataFrame):
            x = x.iloc[:, 0]
        x = pd.to_numeric(x, errors="coerce")
        x.index = pd.to_datetime(x.index).tz_localize(None)
        return x.sort_index()
    except Exception:
        return pd.Series(dtype=float)

def rsi14(c):
    d = c.diff()
    gain = d.clip(lower=0).rolling(14).mean()
    loss = (-d.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100/(1+rs)

ibov = ser("^BVSP").dropna()
if len(ibov) < 1000:
    raise SystemExit("Histórico do Ibovespa insuficiente")

market = {
    "spx": ser("^GSPC").reindex(ibov.index).ffill(),
    "vix": ser("^VIX").reindex(ibov.index).ffill(),
    "usd": ser("BRL=X").reindex(ibov.index).ffill(),
    "ewz": ser("EWZ").reindex(ibov.index).ffill(),
    "oil": ser("CL=F").reindex(ibov.index).ffill(),
    "gold": ser("GC=F").reindex(ibov.index).ffill(),
}

def download_cdi():
    try:
        url = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.12/dados"
        params = {
            "formato":"json",
            "dataInicial":ibov.index.min().strftime("%d/%m/%Y"),
            "dataFinal":ibov.index.max().strftime("%d/%m/%Y"),
        }
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        d = pd.DataFrame(r.json())
        d["date"] = pd.to_datetime(d["data"], dayfirst=True, errors="coerce")
        d["rate"] = pd.to_numeric(
            d["valor"].astype(str).str.replace(",", ".", regex=False),
            errors="coerce",
        )
        return d.dropna(subset=["date","rate"]).set_index("date")["rate"].sort_index()
    except Exception as e:
        print(f"[WARN] CDI indisponível: {e}", flush=True)
        return pd.Series(dtype=float)

cdi = download_cdi()
cdi_factor = (1 + cdi/100.0).cumprod() if not cdi.empty else pd.Series(dtype=float)

def cdi_forward(dt, target_dt):
    if cdi_factor.empty or pd.isna(target_dt):
        return np.nan
    a = cdi_factor[cdi_factor.index <= dt]
    b = cdi_factor[cdi_factor.index <= target_dt]
    if a.empty or b.empty:
        return np.nan
    return float(b.iloc[-1] / a.iloc[-1] - 1.0)

close_panel = pd.DataFrame({t:ser(f"{t}.SA") for t in tickers}).reindex(ibov.index)
vol_panel = pd.DataFrame({t:ser(f"{t}.SA","Volume") for t in tickers}).reindex(ibov.index)

good = close_panel.notna().sum()
tickers = good[good >= 700].index.tolist()
close_panel = close_panel[tickers]
vol_panel = vol_panel[tickers]

if len(tickers) < 30:
    raise SystemExit("Poucas ações com histórico suficiente")

breadth_ma200 = (close_panel > close_panel.rolling(200).mean()).mean(axis=1)
breadth_pos63 = (close_panel.pct_change(63) > 0).mean(axis=1)

parts = []
for i,t in enumerate(tickers,1):
    c = close_panel[t].dropna()
    v = vol_panel[t].reindex(c.index)
    df = pd.DataFrame(index=c.index)

    for n in [5,21,63,126,252]:
        df[f"ret_{n}"] = c.pct_change(n)

    df["vol_20"] = c.pct_change().rolling(20).std()*np.sqrt(252)
    df["vol_63"] = c.pct_change().rolling(63).std()*np.sqrt(252)

    for n in [20,50,200]:
        df[f"price_ma_{n}"] = c/c.rolling(n).mean()-1

    df["rsi_14"] = rsi14(c)/100
    df["drawdown_63"] = c/c.rolling(63).max()-1
    df["volume_ratio_20_60"] = (
        v.rolling(20).mean()/v.rolling(60).mean().replace(0,np.nan)
    )

    ib = ibov.reindex(df.index).ffill()
    for n in [21,63,126,252]:
        df[f"ibov_ret_{n}"] = ib.pct_change(n)
        if f"ret_{n}" in df.columns:
            df[f"rel_ret_{n}"] = df[f"ret_{n}"] - df[f"ibov_ret_{n}"]

    df["ibov_vol_20"] = ib.pct_change().rolling(20).std()*np.sqrt(252)
    df["ibov_ma200"] = ib/ib.rolling(200).mean()-1

    for name,s in market.items():
        a = s.reindex(df.index).ffill()
        df[f"{name}_ret_21"] = a.pct_change(21)
        df[f"{name}_ret_63"] = a.pct_change(63)

    df["vix_level"] = market["vix"].reindex(df.index).ffill()/100
    df["breadth_ma200"] = breadth_ma200.reindex(df.index)
    df["breadth_pos63"] = breadth_pos63.reindex(df.index)

    for h in HORIZONS:
        future_stock = c.shift(-h)/c - 1
        future_ibov = ib.shift(-h)/ib - 1
        future_date = pd.Series(df.index,index=df.index).shift(-h)

        df[f"future_stock_{h}"] = future_stock
        df[f"future_ibov_{h}"] = future_ibov
        df[f"alpha_{h}"] = future_stock - future_ibov
        df[f"future_date_{h}"] = future_date.values

    df["instrument"] = t
    df["sector"] = sector_map.get(t,"Unknown")
    df["datetime"] = df.index
    parts.append(df.reset_index(drop=True))
    print(f"[{i}/{len(tickers)}] {t}", flush=True)

panel = pd.concat(parts,ignore_index=True).replace([np.inf,-np.inf],np.nan)

base_features = [
    "ret_5","ret_21","ret_63","ret_126","ret_252",
    "vol_20","vol_63","price_ma_20","price_ma_50","price_ma_200",
    "rsi_14","drawdown_63","volume_ratio_20_60",
    "ibov_ret_21","ibov_ret_63","ibov_ret_126","ibov_ret_252",
    "rel_ret_21","rel_ret_63","rel_ret_126","rel_ret_252",
    "ibov_vol_20","ibov_ma200",
    "spx_ret_21","spx_ret_63","vix_ret_21","vix_ret_63","vix_level",
    "usd_ret_21","usd_ret_63","ewz_ret_21","ewz_ret_63",
    "oil_ret_21","oil_ret_63","gold_ret_21","gold_ret_63",
    "breadth_ma200","breadth_pos63",
]

sector_dummies = pd.get_dummies(panel["sector"].fillna("Unknown"), prefix="sector", dtype=float)
panel = pd.concat([panel,sector_dummies],axis=1)
features = base_features + list(sector_dummies.columns)

panel["feature_coverage"] = panel[features].notna().mean(axis=1)
panel = panel[panel["feature_coverage"] >= .82].copy()

all_dates = pd.Index(sorted(panel["datetime"].dropna().unique()))
sample_dates = set(all_dates[::STEP])
sampled = panel[panel["datetime"].isin(sample_dates)].copy()

# O label continua sendo binário e economicamente alinhado.
for h in HORIZONS:
    labels=[]; cdis=[]
    for _,r in sampled.iterrows():
        dt=pd.Timestamp(r["datetime"])
        target=pd.Timestamp(r[f"future_date_{h}"]) if not pd.isna(r[f"future_date_{h}"]) else pd.NaT
        cdi_ret=cdi_forward(dt,target); cdis.append(cdi_ret)
        stock=r[f"future_stock_{h}"]; ib=r[f"future_ibov_{h}"]
        if pd.isna(stock) or pd.isna(ib):
            labels.append(np.nan); continue
        hurdle=ib+ROUND_TRIP_COST+MIN_EXTRA_ALPHA[h]
        if not pd.isna(cdi_ret):
            hurdle=max(hurdle,cdi_ret+ROUND_TRIP_COST)
        labels.append(int(stock>hurdle))
    sampled[f"cdi_return_{h}"]=cdis
    sampled[f"label_{h}"]=labels

def preprocess(train,others):
    med=train[features].median(numeric_only=True)
    lo=train[features].quantile(.005); hi=train[features].quantile(.995)
    def one(x):
        y=x.copy()
        y.loc[:,features]=y[features].fillna(med)
        y.loc[:,features]=y[features].clip(lower=lo,upper=hi,axis=1)
        return y
    return one(train),[one(x) for x in others]

def to_qlib(df,label_col=None):
    idx=pd.MultiIndex.from_arrays(
        [pd.to_datetime(df["datetime"]).values,df["instrument"].astype(str).values],
        names=["datetime","instrument"]
    )
    xf=df[features].copy(); xf.index=idx
    if label_col is None: return xf
    yf_=df[[label_col]].rename(columns={label_col:"label"}); yf_.index=idx
    return pd.concat({"feature":xf,"label":yf_},axis=1).sort_index()

class MemoryDataset:
    def __init__(self,train,valid,test):
        self.frames={"train":train,"valid":valid,"test":test}
        self.segments={"train":"train","valid":"valid","test":"test"}
    def prepare(self,segment,col_set=None,data_key=None):
        if isinstance(segment,list): return [self.prepare(s,col_set,data_key) for s in segment]
        df=self.frames[segment]
        if col_set is None:return df
        if col_set=="feature":return df["feature"] if isinstance(df.columns,pd.MultiIndex) else df
        if col_set=="label":return df["label"]
        return df

qlib_gbdt.R.log_metrics=lambda **kwargs:None

def make_model():
    return qlib_gbdt.LGBModel(
        loss="binary",early_stopping_rounds=50,num_boost_round=600,
        learning_rate=.03,num_leaves=31,max_depth=6,
        feature_fraction=.85,bagging_fraction=.85,bagging_freq=1,
        lambda_l1=.2,lambda_l2=1.0,num_threads=4,seed=42
    )

def sector_capped(g,k):
    g=g.sort_values("model_score",ascending=False)
    cap=max(1,math.ceil(k/4)); chosen=[]; counts={}
    for idx,row in g.iterrows():
        s=str(row["sector"])
        if counts.get(s,0)<cap:
            chosen.append(idx);counts[s]=counts.get(s,0)+1
        if len(chosen)>=k:break
    if len(chosen)<k:
        for idx in g.index:
            if idx not in chosen:chosen.append(idx)
            if len(chosen)>=k:break
    return g.loc[chosen[:k]]

def evaluate_fraction(df,pred,h,fraction):
    x=df.copy();x["model_score"]=np.asarray(pred,float)
    rows=[]
    dates=pd.Index(sorted(x["datetime"].unique()))
    dates=dates[::max(1,math.ceil(h/STEP))]
    for dt in dates:
        g=x[x["datetime"]==dt].copy()
        if len(g)<20:continue
        k=max(3,math.ceil(len(g)*fraction))
        pick=sector_capped(g,k)
        stock=float(pick[f"future_stock_{h}"].mean())
        ib=float(pick[f"future_ibov_{h}"].mean())
        cdi_vals=pd.to_numeric(pick[f"cdi_return_{h}"],errors="coerce")
        cdi_ret=float(cdi_vals.mean()) if cdi_vals.notna().any() else np.nan
        rows.append(dict(
            date=str(pd.Timestamp(dt).date()),horizon=h,top_fraction=fraction,
            n=int(len(pick)),precision=float(pick[f"label_{h}"].mean()),
            stock_return=stock,net_alpha_vs_ibov=stock-ib-ROUND_TRIP_COST,
            net_excess_vs_cdi=(stock-cdi_ret-ROUND_TRIP_COST if not pd.isna(cdi_ret) else np.nan),
            avg_model_score=float(pick["model_score"].mean())
        ))
    return pd.DataFrame(rows)

fold_rows=[]; oos_rows=[]; selected_fractions={}

for h in HORIZONS:
    label=f"label_{h}"
    data=sampled.dropna(subset=[label,f"future_stock_{h}",f"future_ibov_{h}"]).copy()
    data=data.sort_values(["datetime","instrument"]).reset_index(drop=True)
    dates=pd.Index(sorted(data["datetime"].unique()))
    n=len(dates);embargo=max(1,math.ceil(h/STEP))
    fractions=[]

    for fold_id,(a,b) in enumerate([(0.55,.70),(.70,.85),(.85,1.0)],1):
        ts=int(n*a);te=min(n,int(n*b));vlen=max(24,int(n*.10))
        vs=ts-vlen;train_end=vs-embargo;ve=ts-embargo
        if train_end<60 or ve<=vs:continue

        tr=data[data["datetime"].isin(dates[:train_end])].copy()
        va=data[data["datetime"].isin(dates[vs:ve])].copy()
        tef=data[data["datetime"].isin(dates[ts:te])].copy()
        tr,processed=preprocess(tr,[va,tef]);va,tef=processed

        ds=MemoryDataset(to_qlib(tr,label),to_qlib(va,label),to_qlib(tef,label))
        m=make_model();m.fit(ds,verbose_eval=0)

        pv=m.predict(ds,"valid")
        candidates=[]
        base=float(va[label].mean())

        for frac in TOP_FRACTIONS:
            ev=evaluate_fraction(va,pv,h,frac)
            if ev.empty:continue
            precision=float(ev["precision"].mean())
            alpha=float(ev["net_alpha_vs_ibov"].mean())
            cdi_excess=float(ev["net_excess_vs_cdi"].dropna().mean()) if ev["net_excess_vs_cdi"].notna().any() else 0
            score=(precision-base)+max(alpha,0)*4+max(cdi_excess,0)*2
            candidates.append((score,precision,alpha,frac))

        candidates.sort(reverse=True)
        chosen=candidates[0][3] if candidates else .10
        fractions.append(chosen)

        pt=m.predict(ds,"test")
        ev=evaluate_fraction(tef,pt,h,chosen)
        if not ev.empty:
            ev["fold"]=fold_id;oos_rows.append(ev)

        fold_rows.append(dict(
            horizon=h,fold=fold_id,chosen_fraction=chosen,
            test_base_rate=float(tef[label].mean()),
            test_precision=float(ev["precision"].mean()) if not ev.empty else np.nan,
            test_precision_lift=(float(ev["precision"].mean())-float(tef[label].mean()) if not ev.empty else np.nan),
            test_net_alpha_vs_ibov=float(ev["net_alpha_vs_ibov"].mean()) if not ev.empty else np.nan,
            test_net_excess_vs_cdi=float(ev["net_excess_vs_cdi"].dropna().mean()) if (not ev.empty and ev["net_excess_vs_cdi"].notna().any()) else np.nan,
            test_periods=int(len(ev)),
            test_start=str(pd.Timestamp(dates[ts]).date()),
            test_end=str(pd.Timestamp(dates[te-1]).date()),
        ))
        print(f"[H={h}] fold={fold_id} top={chosen:.0%}",flush=True)

    selected_fractions[h]=float(pd.Series(fractions).median()) if fractions else .10

fold_df=pd.DataFrame(fold_rows)
oos_df=pd.concat(oos_rows,ignore_index=True) if oos_rows else pd.DataFrame()
if fold_df.empty or oos_df.empty:raise SystemExit("Sem resultados OOS")

results={};accepted=[]
for h in HORIZONS:
    f=fold_df[fold_df["horizon"]==h]
    q=oos_df[oos_df["horizon"]==h]
    if f.empty or q.empty:continue
    precision=float(q["precision"].mean())
    base=float(f["test_base_rate"].mean())
    alpha=float(q["net_alpha_vs_ibov"].mean())
    cdi_excess=float(q["net_excess_vs_cdi"].dropna().mean()) if q["net_excess_vs_cdi"].notna().any() else None
    stable=int(((f["test_precision_lift"]>0)&(f["test_net_alpha_vs_ibov"]>0)).sum())
    periods=int(len(q))
    ok=(periods>=20 and precision>=.55 and precision>=base+.06 and alpha>0
        and stable>=2 and (cdi_excess is None or cdi_excess>0))
    if ok:accepted.append(h)
    results[str(h)]=dict(
        periods=periods,base_rate=base,precision=precision,
        precision_lift=precision-base,avg_net_alpha_vs_ibov=alpha,
        avg_net_excess_vs_cdi=cdi_excess,stable_positive_folds=stable,
        chosen_current_top_fraction=selected_fractions[h],accepted=bool(ok),
        acceptance_rule=">=20 OOS periods; precision>=55%; lift>=6pp; positive net alpha; >=2/3 positive folds; positive CDI excess when available."
    )

# Current ranking; no calibrated-probability claim.
live=panel.sort_values("datetime").groupby("instrument",as_index=False).tail(1).copy()
current_frames=[]
for h in HORIZONS:
    label=f"label_{h}"
    data=sampled.dropna(subset=[label]).copy().sort_values(["datetime","instrument"])
    dates=pd.Index(sorted(data["datetime"].unique()))
    embargo=max(1,math.ceil(h/STEP));vlen=max(30,int(len(dates)*.12))
    vs=len(dates)-vlen;train_end=max(1,vs-embargo)
    tr=data[data["datetime"].isin(dates[:train_end])].copy()
    va=data[data["datetime"].isin(dates[vs:])].copy()
    tr,processed=preprocess(tr,[va,live]);va,livep=processed
    ds=MemoryDataset(to_qlib(tr,label),to_qlib(va,label),to_qlib(livep,None))
    m=make_model();m.fit(ds,verbose_eval=0)
    pred=np.asarray(m.predict(ds,"test"),float)
    cur=livep[["instrument","sector","datetime"]].copy()
    cur[f"raw_score_{h}d"]=pred
    cur[f"rank_score_{h}d"]=pd.Series(pred,index=cur.index).rank(pct=True,method="average")*100
    frac=selected_fractions[h]
    cutoff=100*(1-frac)
    cur[f"confirm_{h}d"]=cur[f"rank_score_{h}d"]>=cutoff
    current_frames.append(cur)

current=current_frames[0]
for nxt in current_frames[1:]:
    current=current.merge(nxt.drop(columns=["sector","datetime"]),on="instrument",how="outer")

accepted_rank_cols=[f"rank_score_{h}d" for h in accepted if f"rank_score_{h}d" in current.columns]
all_rank_cols=[c for c in current.columns if c.startswith("rank_score_")]
current["qlib_v4_ensemble_rank"]=current[accepted_rank_cols if accepted_rank_cols else all_rank_cols].mean(axis=1)

if accepted:
    confirms=[f"confirm_{h}d" for h in accepted if f"confirm_{h}d" in current.columns]
    current["qlib_v4_confirm"]=current[confirms].any(axis=1)
else:
    current["qlib_v4_confirm"]=False

current["qlib_v4_status"]=np.where(current["qlib_v4_confirm"],"CONFIRMA","NAO_CONFIRMA")
current=current.rename(columns={"instrument":"ticker"})

fold_df.to_csv(OUT/"qlib_v4_folds.csv",index=False)
oos_df.to_csv(OUT/"qlib_v4_oos.csv",index=False)
current.to_csv(OUT/"qlib_v4_current_scores.csv",index=False)

summary=dict(
    engine="Qlib LightGBM binary model used as cross-sectional rank signal",
    why_v4="V3 used absolute score thresholds even though LightGBM outputs were not calibrated probabilities. V4 validates top cross-sectional fractions instead.",
    horizons=HORIZONS,top_fractions_tested=TOP_FRACTIONS,
    validation="3 expanding purged walk-forward folds; top fraction selected only on validation",
    horizon_results=results,accepted_horizons=accepted,
    ml_v4_status=("ACCEPTED" if accepted else "REJECTED"),
    current_scored_stocks=int(len(current)),
    current_confirms=int(current["qlib_v4_confirm"].sum()),
    limitations=[
        "Universe still starts from currently listed stocks: survivorship bias remains.",
        "Historical model lacks point-in-time fundamentals.",
        "Yahoo Finance is free/non-official.",
        "Model raw scores are treated only as ranks, not probabilities."
    ]
)
(OUT/"qlib_v4_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2,default=str),encoding="utf-8")
print(json.dumps(summary,ensure_ascii=False,indent=2,default=str))
