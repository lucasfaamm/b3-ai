
from pathlib import Path
import json, math
import numpy as np
import pandas as pd
import requests
import yfinance as yf
import lightgbm as lgb

FUND = Path("data/fundamentals.csv")
OUT = Path("results"); OUT.mkdir(exist_ok=True)
MAX_TICKERS = 110
PERIOD = "7y"
STEP = 5
HORIZONS = [5, 20, 60]
COST = 0.002

CFG = {
    "conservative": dict(
        n_estimators=500, learning_rate=0.03, num_leaves=15, max_depth=4,
        min_child_samples=40, subsample=0.85, colsample_bytree=0.85,
        reg_alpha=0.5, reg_lambda=2.0
    ),
    "balanced": dict(
        n_estimators=600, learning_rate=0.03, num_leaves=31, max_depth=6,
        min_child_samples=30, subsample=0.85, colsample_bytree=0.90,
        reg_alpha=0.2, reg_lambda=1.0
    ),
}

if not FUND.exists():
    raise SystemExit("Falta data/fundamentals.csv")

fund = pd.read_csv(FUND)
fund["ticker"] = fund["ticker"].astype(str).str.upper()
fund["sector"] = fund.get("sector", "Unknown").fillna("Unknown").astype(str)
tickers = fund["ticker"].dropna().drop_duplicates().tolist()[:MAX_TICKERS]
sector_map = fund.drop_duplicates("ticker").set_index("ticker")["sector"].to_dict()

stock_symbols = [f"{t}.SA" for t in tickers]
market_symbols = ["^BVSP","^GSPC","^VIX","BRL=X","EWZ","CL=F","GC=F"]
raw = yf.download(
    stock_symbols + market_symbols, period=PERIOD, interval="1d",
    auto_adjust=True, progress=False, threads=True, group_by="ticker"
)

def ser(symbol, field="Close"):
    try:
        x = raw[symbol][field]
        if isinstance(x, pd.DataFrame):
            x = x.iloc[:,0]
        x = pd.to_numeric(x, errors="coerce")
        x.index = pd.to_datetime(x.index).tz_localize(None)
        return x.sort_index()
    except Exception:
        return pd.Series(dtype=float)

def rsi14(c):
    d = c.diff()
    g = d.clip(lower=0).rolling(14).mean()
    l = (-d.clip(upper=0)).rolling(14).mean()
    rs = g / l.replace(0, np.nan)
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
        r = requests.get(url, params=params, timeout=20); r.raise_for_status()
        d = pd.DataFrame(r.json())
        d["date"] = pd.to_datetime(d["data"], dayfirst=True, errors="coerce")
        d["rate"] = pd.to_numeric(d["valor"].astype(str).str.replace(",", ".", regex=False), errors="coerce")
        return d.dropna(subset=["date","rate"]).set_index("date")["rate"].sort_index()
    except Exception as e:
        print(f"[WARN] CDI indisponível: {e}", flush=True)
        return pd.Series(dtype=float)

cdi = download_cdi()

def cdi_fwd(dt, target):
    if cdi.empty or pd.isna(target):
        return np.nan
    x = cdi[(cdi.index > dt) & (cdi.index <= target)]
    if x.empty:
        return np.nan
    return float(np.prod(1 + x.values/100.0)-1)

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
    df["close"] = c
    for n in [5,21,63,126,252]:
        df[f"ret_{n}"] = c.pct_change(n)
    df["vol_20"] = c.pct_change().rolling(20).std()*np.sqrt(252)
    df["vol_63"] = c.pct_change().rolling(63).std()*np.sqrt(252)
    for n in [20,50,200]:
        df[f"price_ma_{n}"] = c/c.rolling(n).mean()-1
    df["rsi_14"] = rsi14(c)/100
    df["drawdown_63"] = c/c.rolling(63).max()-1
    df["volume_ratio_20_60"] = v.rolling(20).mean()/v.rolling(60).mean().replace(0,np.nan)

    ib = ibov.reindex(df.index).ffill()
    for n in [21,63,126,252]:
        df[f"ibov_ret_{n}"] = ib.pct_change(n)
        if f"ret_{n}" in df:
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
        df[f"future_stock_{h}"] = c.shift(-h)/c - 1
        df[f"future_ibov_{h}"] = ib.shift(-h)/ib - 1
        df[f"alpha_{h}"] = df[f"future_stock_{h}"] - df[f"future_ibov_{h}"]
        df[f"future_date_{h}"] = pd.Series(df.index, index=df.index).shift(-h).values

    df["instrument"] = t
    df["sector"] = sector_map.get(t,"Unknown")
    df["datetime"] = df.index
    parts.append(df.reset_index(drop=True))
    print(f"[{i}/{len(tickers)}] {t}", flush=True)

panel = pd.concat(parts, ignore_index=True).replace([np.inf,-np.inf],np.nan)

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
    "breadth_ma200","breadth_pos63"
]
dummies = pd.get_dummies(panel["sector"].fillna("Unknown"), prefix="sector", dtype=float)
panel = pd.concat([panel,dummies],axis=1)
features = base_features + list(dummies.columns)
panel["feature_coverage"] = panel[features].notna().mean(axis=1)
panel = panel[panel["feature_coverage"] >= 0.82].copy()

all_dates = pd.Index(sorted(panel["datetime"].dropna().unique()))
sample_dates = set(all_dates[::STEP])
sampled = panel[panel["datetime"].isin(sample_dates)].copy()

def relevance(x, alpha_col):
    x = x.copy()
    pct = x.groupby("datetime")[alpha_col].rank(pct=True, method="average")
    x["relevance"] = np.floor(pct*5).clip(0,4).astype(int)
    return x

def preprocess(train, others):
    med = train[features].median(numeric_only=True)
    lo = train[features].quantile(.005); hi = train[features].quantile(.995)
    def one(x):
        y=x.copy()
        y.loc[:,features]=y[features].fillna(med)
        y.loc[:,features]=y[features].clip(lower=lo,upper=hi,axis=1)
        return y
    return one(train), [one(x) for x in others], med, lo, hi

def groups(x):
    return x.groupby("datetime",sort=False).size().tolist()

def model(params):
    return lgb.LGBMRanker(
        objective="lambdarank", metric="ndcg", ndcg_eval_at=[5,10,20],
        random_state=42, n_jobs=4, verbosity=-1, **params
    )

def blend(g, col):
    x=g.copy()
    x["_g"]=x[col].rank(pct=True,method="average")
    x["_s"]=x.groupby("sector")[col].rank(pct=True,method="average")
    x["_blend"]=.70*x["_g"]+.30*x["_s"]
    return x

def pick(g,col,k):
    g=g.sort_values(col,ascending=False)
    cap=max(1,math.ceil(k/4)); chosen=[]; counts={}
    for idx,row in g.iterrows():
        s=str(row["sector"])
        if counts.get(s,0)<cap:
            chosen.append(idx); counts[s]=counts.get(s,0)+1
        if len(chosen)>=k: break
    if len(chosen)<k:
        for idx in g.index:
            if idx not in chosen: chosen.append(idx)
            if len(chosen)>=k: break
    return g.loc[chosen[:k]]

def eval_pred(df,pred,h,source):
    x=df.copy(); x["prediction"]=np.asarray(pred,float)
    rows=[]
    dates=pd.Index(sorted(x["datetime"].unique()))
    dates=dates[::max(1,math.ceil(h/STEP))]
    for dt in dates:
        g=x[x["datetime"]==dt].copy()
        if len(g)<20: continue
        g=blend(g,"prediction")
        fdates=pd.to_datetime(g[f"future_date_{h}"],errors="coerce").dropna()
        target=fdates.iloc[0] if len(fdates) else pd.NaT
        cdi_ret=cdi_fwd(pd.Timestamp(dt),target)
        for label,k in [("top5",5),("top10",10),("top20",20),("top_decile",max(5,math.ceil(len(g)*.10)))]:
            p=pick(g,"_blend",k)
            stock=float(p[f"future_stock_{h}"].mean())
            ib=float(p[f"future_ibov_{h}"].mean())
            alpha=float(p[f"alpha_{h}"].mean())
            rows.append(dict(
                date=str(pd.Timestamp(dt).date()), horizon=h, source=source,
                portfolio=label, n=len(p), stock_return=stock, ibov_return=ib,
                gross_alpha_vs_ibov=alpha, net_alpha_vs_ibov=alpha-COST,
                cdi_return=cdi_ret,
                net_excess_vs_cdi=(stock-COST-cdi_ret if not pd.isna(cdi_ret) else np.nan)
            ))
        m=g.copy(); m["_mom"]=m["ret_126"]
        p=pick(m,"_mom",10)
        stock=float(p[f"future_stock_{h}"].mean())
        ib=float(p[f"future_ibov_{h}"].mean())
        alpha=float(p[f"alpha_{h}"].mean())
        rows.append(dict(
            date=str(pd.Timestamp(dt).date()), horizon=h, source="momentum_126_baseline",
            portfolio="top10", n=len(p), stock_return=stock, ibov_return=ib,
            gross_alpha_vs_ibov=alpha, net_alpha_vs_ibov=alpha-COST,
            cdi_return=cdi_ret,
            net_excess_vs_cdi=(stock-COST-cdi_ret if not pd.isna(cdi_ret) else np.nan)
        ))
    return pd.DataFrame(rows)

def ic_series(df,pred):
    x=df[["datetime","instrument"]].copy()
    x["prediction"]=np.asarray(pred,float)
    x["label"]=df["continuous_alpha"].values
    return x.groupby("datetime",group_keys=False).apply(
        lambda g:g["prediction"].corr(g["label"],method="spearman") if len(g)>=15 else np.nan,
        include_groups=False
    ).dropna()

fold_rows=[]; portfolio_rows=[]; gain_rows=[]; chosen_cfgs={}

for h in HORIZONS:
    labeled=sampled.dropna(subset=[f"alpha_{h}",f"future_stock_{h}",f"future_ibov_{h}"]).copy()
    labeled["continuous_alpha"]=labeled[f"alpha_{h}"]
    labeled=relevance(labeled,f"alpha_{h}").sort_values(["datetime","instrument"]).reset_index(drop=True)
    dates=pd.Index(sorted(labeled["datetime"].unique()))
    n=len(dates); embargo=max(1,math.ceil(h/STEP)); selected=[]
    if n<180: continue

    for fold_id,(a,b) in enumerate([(0.55,.70),(.70,.85),(.85,1.0)],1):
        ts=int(n*a); te=min(n,int(n*b)); vlen=max(24,int(n*.10))
        vs=ts-vlen; train_end=vs-embargo; ve=ts-embargo
        if train_end<60 or ve<=vs: continue
        trd=dates[:train_end]; vad=dates[vs:ve]; ted=dates[ts:te]
        tr=labeled[labeled["datetime"].isin(trd)].copy()
        va=labeled[labeled["datetime"].isin(vad)].copy()
        tef=labeled[labeled["datetime"].isin(ted)].copy()
        tr,processed,_,_,_=preprocess(tr,[va,tef]); va,tef=processed

        cand=[]
        for name,params in CFG.items():
            m=model(params)
            m.fit(
                tr[features],tr["relevance"],group=groups(tr),
                eval_set=[(va[features],va["relevance"])],
                eval_group=[groups(va)],
                callbacks=[lgb.early_stopping(60,verbose=False)]
            )
            pv=m.predict(va[features])
            ev=eval_pred(va,pv,h,f"validation_{name}")
            v10=ev[(ev["source"]==f"validation_{name}")&(ev["portfolio"]=="top10")]
            cand.append((float(v10["net_alpha_vs_ibov"].mean()) if len(v10) else -999,
                         float((v10["net_alpha_vs_ibov"]>0).mean()) if len(v10) else 0,
                         name,m))
        cand.sort(key=lambda z:(z[0],z[1]),reverse=True)
        val_alpha,val_hit,name,m=cand[0]; selected.append(name)
        pt=m.predict(tef[features]); ic=ic_series(tef,pt)
        portfolio_rows.append(eval_pred(tef,pt,h,"ranker_v2"))
        fold_rows.append(dict(
            horizon=h,fold=fold_id,selected_config=name,
            train_start=str(pd.Timestamp(trd.min()).date()),
            train_end=str(pd.Timestamp(trd.max()).date()),
            valid_start=str(pd.Timestamp(vad.min()).date()),
            valid_end=str(pd.Timestamp(vad.max()).date()),
            test_start=str(pd.Timestamp(ted.min()).date()),
            test_end=str(pd.Timestamp(ted.max()).date()),
            embargo_sample_periods=embargo,
            validation_top10_net_alpha=val_alpha,
            validation_top10_hit_rate=val_hit,
            test_ic_mean=float(ic.mean()) if len(ic) else np.nan,
            test_ic_positive_rate=float((ic>0).mean()) if len(ic) else np.nan
        ))
        for feat,gain in zip(features,m.booster_.feature_importance(importance_type="gain")):
            gain_rows.append(dict(horizon=h,fold=fold_id,feature=feat,gain=float(gain)))
        print(f"[H={h}] fold={fold_id} cfg={name} IC={float(ic.mean()) if len(ic) else np.nan:.4f}",flush=True)
    chosen_cfgs[h]=selected

fold_df=pd.DataFrame(fold_rows)
port_df=pd.concat(portfolio_rows,ignore_index=True) if portfolio_rows else pd.DataFrame()
gain_df=pd.DataFrame(gain_rows)
if fold_df.empty or port_df.empty:
    raise SystemExit("Nenhum fold concluído")

results={}; accepted=[]
for h in HORIZONS:
    f=fold_df[fold_df["horizon"]==h]
    p=port_df[(port_df["horizon"]==h)&(port_df["source"]=="ranker_v2")]
    base=port_df[(port_df["horizon"]==h)&(port_df["source"]=="momentum_126_baseline")&(port_df["portfolio"]=="top10")]
    if f.empty or p.empty: continue
    tops={}
    for label in ["top5","top10","top20","top_decile"]:
        q=p[p["portfolio"]==label]
        tops[label]=dict(
            periods=int(len(q)),
            avg_stock_return=float(q["stock_return"].mean()) if len(q) else None,
            avg_ibov_return=float(q["ibov_return"].mean()) if len(q) else None,
            avg_net_alpha_vs_ibov=float(q["net_alpha_vs_ibov"].mean()) if len(q) else None,
            positive_net_alpha_rate=float((q["net_alpha_vs_ibov"]>0).mean()) if len(q) else None,
            avg_net_excess_vs_cdi=float(q["net_excess_vs_cdi"].dropna().mean()) if q["net_excess_vs_cdi"].notna().any() else None
        )
    t10=tops["top10"]; b_alpha=float(base["net_alpha_vs_ibov"].mean()) if len(base) else None
    icm=float(f["test_ic_mean"].mean()); icp=float(f["test_ic_positive_rate"].mean())
    ok=(icm>.03 and t10["avg_net_alpha_vs_ibov"] is not None and t10["avg_net_alpha_vs_ibov"]>.002
        and t10["positive_net_alpha_rate"] is not None and t10["positive_net_alpha_rate"]>=.55
        and (b_alpha is None or t10["avg_net_alpha_vs_ibov"]>b_alpha))
    if ok: accepted.append(h)
    results[str(h)]=dict(
        folds=int(len(f)),mean_test_ic=icm,mean_test_ic_positive_rate=icp,
        top_portfolios=tops,momentum_baseline_top10_net_alpha=b_alpha,accepted=bool(ok),
        acceptance_rule="IC>0.03; Top10 net alpha>0.20%; hit-rate>=55%; beats 126d momentum baseline."
    )

live=panel.sort_values("datetime").groupby("instrument",as_index=False).tail(1).copy()
current_frames=[]
for h in HORIZONS:
    lab=sampled.dropna(subset=[f"alpha_{h}"]).copy()
    lab["continuous_alpha"]=lab[f"alpha_{h}"]
    lab=relevance(lab,f"alpha_{h}").sort_values(["datetime","instrument"]).reset_index(drop=True)
    dates=pd.Index(sorted(lab["datetime"].unique()))
    if len(dates)<120: continue
    names=chosen_cfgs.get(h,[])
    cname=pd.Series(names).mode().iloc[0] if names else "conservative"
    embargo=max(1,math.ceil(h/STEP)); vlen=max(30,int(len(dates)*.12))
    vs=len(dates)-vlen; tr_end=max(1,vs-embargo)
    tr=lab[lab["datetime"].isin(dates[:tr_end])].copy()
    va=lab[lab["datetime"].isin(dates[vs:])].copy()
    tr,processed,_,_,_=preprocess(tr,[va,live]); va,livep=processed
    m=model(CFG[cname])
    m.fit(
        tr[features],tr["relevance"],group=groups(tr),
        eval_set=[(va[features],va["relevance"])],eval_group=[groups(va)],
        callbacks=[lgb.early_stopping(60,verbose=False)]
    )
    pred=m.predict(livep[features])
    cur=livep[["instrument","sector","datetime"]].copy()
    cur[f"prediction_{h}d"]=pred
    cur[f"score_{h}d"]=pd.Series(pred,index=cur.index).rank(pct=True,method="average")*100
    current_frames.append(cur)

current=current_frames[0]
for nxt in current_frames[1:]:
    current=current.merge(nxt.drop(columns=["sector","datetime"]),on="instrument",how="outer")
accepted_cols=[f"score_{h}d" for h in accepted if f"score_{h}d" in current.columns]
all_cols=[c for c in current.columns if c.startswith("score_")]
current["qlib_v2_ensemble_score"]=current[accepted_cols if accepted_cols else all_cols].mean(axis=1)
current["qlib_v2_confirm"]=(current["qlib_v2_ensemble_score"]>=80)&bool(accepted)
current["qlib_v2_status"]=np.where(current["qlib_v2_confirm"],"CONFIRMA","NAO_CONFIRMA")
current=current.rename(columns={"instrument":"ticker"})

fold_df.to_csv(OUT/"qlib_v2_folds.csv",index=False)
port_df.to_csv(OUT/"qlib_v2_topk_oos.csv",index=False)
if not gain_df.empty:
    gain_df.groupby(["horizon","feature"],as_index=False)["gain"].mean().sort_values(
        ["horizon","gain"],ascending=[True,False]
    ).to_csv(OUT/"qlib_v2_feature_importance.csv",index=False)
current.to_csv(OUT/"qlib_v2_current_scores.csv",index=False)

summary=dict(
    engine="Qlib research pipeline + native LightGBM LGBMRanker",
    objective="lambdarank",
    universe_with_history=int(len(tickers)),
    sample_frequency="approximately weekly",
    horizons_trading_days=HORIZONS,
    round_trip_cost_assumption=COST,
    validation_design="3 expanding purged walk-forward folds per horizon; embargo scales with horizon; hyperparameters selected only on validation.",
    sector_control="70% global rank + 30% within-sector rank; Top-K sector concentration cap.",
    benchmarks=["Ibovespa","CDI when BCB API is available","simple 126-day momentum Top10"],
    horizon_results=results,
    accepted_horizons=accepted,
    ml_v2_status=("ACCEPTED" if accepted else "REJECTED"),
    current_scored_stocks=int(len(current)),
    current_ml_confirms=int(current["qlib_v2_confirm"].sum()),
    limitations=[
        "Universe starts from currently listed companies, so survivorship bias remains.",
        "Current sector classifications are reused historically.",
        "Historical ML does not yet include point-in-time fundamentals.",
        "Yahoo Finance is free/non-official and may contain gaps.",
        "ML scores are relative ranks, not probabilities of profit."
    ]
)
(OUT/"qlib_v2_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2,default=str),encoding="utf-8")
print(json.dumps(summary,ensure_ascii=False,indent=2,default=str))
