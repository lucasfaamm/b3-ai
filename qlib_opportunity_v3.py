
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
THRESHOLDS = [0.55, 0.60, 0.65, 0.70]

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
            x = x.iloc[:,0]
        x = pd.to_numeric(x, errors="coerce")
        x.index = pd.to_datetime(x.index).tz_localize(None)
        return x.sort_index()
    except Exception:
        return pd.Series(dtype=float)

def rsi14(c):
    d = c.diff()
    gain = d.clip(lower=0).rolling(14).mean()
    loss = (-d.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0,np.nan)
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

# Constrói índice acumulado de CDI para cálculo vetorizado de forward return.
if not cdi.empty:
    cdi_factor = (1 + cdi/100.0).cumprod()
else:
    cdi_factor = pd.Series(dtype=float)

def cdi_forward(dt, target_dt):
    if cdi_factor.empty or pd.isna(target_dt):
        return np.nan
    before = cdi_factor[cdi_factor.index <= dt]
    after = cdi_factor[cdi_factor.index <= target_dt]
    if before.empty or after.empty:
        return np.nan
    a = float(before.iloc[-1]); b = float(after.iloc[-1])
    return b/a - 1.0 if a else np.nan

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

sector_dummies = pd.get_dummies(
    panel["sector"].fillna("Unknown"),
    prefix="sector",
    dtype=float,
)
panel = pd.concat([panel,sector_dummies],axis=1)
features = base_features + list(sector_dummies.columns)

panel["feature_coverage"] = panel[features].notna().mean(axis=1)
panel = panel[panel["feature_coverage"] >= .82].copy()

all_dates = pd.Index(sorted(panel["datetime"].dropna().unique()))
sample_dates = set(all_dates[::STEP])
sampled = panel[panel["datetime"].isin(sample_dates)].copy()

# Label: vencer IBOV + custos + margem mínima; quando CDI existe,
# também exige vencer CDI + custos.
for h in HORIZONS:
    labels = []
    cdi_returns = []
    for _,r in sampled.iterrows():
        dt = pd.Timestamp(r["datetime"])
        target = pd.Timestamp(r[f"future_date_{h}"]) if not pd.isna(r[f"future_date_{h}"]) else pd.NaT
        cdi_ret = cdi_forward(dt,target)
        cdi_returns.append(cdi_ret)

        stock = r[f"future_stock_{h}"]
        ib = r[f"future_ibov_{h}"]
        if pd.isna(stock) or pd.isna(ib):
            labels.append(np.nan)
            continue

        hurdle = ib + ROUND_TRIP_COST + MIN_EXTRA_ALPHA[h]
        if not pd.isna(cdi_ret):
            hurdle = max(hurdle, cdi_ret + ROUND_TRIP_COST)

        labels.append(int(stock > hurdle))

    sampled[f"cdi_return_{h}"] = cdi_returns
    sampled[f"label_{h}"] = labels


def preprocess(train, others):
    med = train[features].median(numeric_only=True)
    lo = train[features].quantile(.005)
    hi = train[features].quantile(.995)

    def one(x):
        y=x.copy()
        y.loc[:,features] = y[features].fillna(med)
        y.loc[:,features] = y[features].clip(lower=lo,upper=hi,axis=1)
        return y

    return one(train), [one(x) for x in others], med, lo, hi


def to_qlib(df,label_col=None):
    idx = pd.MultiIndex.from_arrays(
        [pd.to_datetime(df["datetime"]).values, df["instrument"].astype(str).values],
        names=["datetime","instrument"],
    )
    xf = df[features].copy(); xf.index = idx
    if label_col is None:
        return xf
    yf_ = df[[label_col]].rename(columns={label_col:"label"}); yf_.index=idx
    return pd.concat({"feature":xf,"label":yf_},axis=1).sort_index()


class MemoryDataset:
    def __init__(self,train,valid,test):
        self.frames={"train":train,"valid":valid,"test":test}
        self.segments={"train":"train","valid":"valid","test":"test"}
    def prepare(self,segment,col_set=None,data_key=None):
        if isinstance(segment,list):
            return [self.prepare(s,col_set,data_key) for s in segment]
        df=self.frames[segment]
        if col_set is None:
            return df
        if col_set=="feature":
            return df["feature"] if isinstance(df.columns,pd.MultiIndex) else df
        if col_set=="label":
            return df["label"]
        return df


qlib_gbdt.R.log_metrics = lambda **kwargs: None

def make_model():
    return qlib_gbdt.LGBModel(
        loss="binary",
        early_stopping_rounds=50,
        num_boost_round=600,
        learning_rate=.03,
        num_leaves=31,
        max_depth=6,
        feature_fraction=.85,
        bagging_fraction=.85,
        bagging_freq=1,
        lambda_l1=.2,
        lambda_l2=1.0,
        num_threads=4,
        seed=42,
    )


def sector_cap(g,k=10):
    g=g.sort_values("probability",ascending=False)
    cap=max(1,math.ceil(k/4))
    chosen=[]; counts={}
    for idx,row in g.iterrows():
        s=str(row["sector"])
        if counts.get(s,0)<cap:
            chosen.append(idx); counts[s]=counts.get(s,0)+1
        if len(chosen)>=k:
            break
    return g.loc[chosen]


def evaluate(df,pred,h,threshold):
    x=df.copy()
    x["probability"]=np.asarray(pred,float)
    rows=[]

    dates=pd.Index(sorted(x["datetime"].unique()))
    dates=dates[::max(1,math.ceil(h/STEP))]

    for dt in dates:
        g=x[x["datetime"]==dt].copy()
        if len(g)<20:
            continue

        eligible=g[g["probability"]>=threshold].copy()
        if eligible.empty:
            continue

        pick=sector_cap(eligible,10)
        if len(pick)<2:
            continue

        stock=float(pick[f"future_stock_{h}"].mean())
        ib=float(pick[f"future_ibov_{h}"].mean())
        cdi_vals=pd.to_numeric(pick[f"cdi_return_{h}"],errors="coerce")
        cdi_ret=float(cdi_vals.mean()) if cdi_vals.notna().any() else np.nan
        precision=float(pick[f"label_{h}"].mean())

        rows.append(dict(
            date=str(pd.Timestamp(dt).date()),
            horizon=h,
            threshold=threshold,
            n=int(len(pick)),
            precision=precision,
            stock_return=stock,
            net_alpha_vs_ibov=stock-ib-ROUND_TRIP_COST,
            net_excess_vs_cdi=(stock-cdi_ret-ROUND_TRIP_COST if not pd.isna(cdi_ret) else np.nan),
            avg_probability=float(pick["probability"].mean()),
        ))

    return pd.DataFrame(rows)


fold_rows=[]
signal_rows=[]
accepted_horizons=[]
chosen_thresholds={}

for h in HORIZONS:
    label_col=f"label_{h}"
    data=sampled.dropna(subset=[label_col,f"future_stock_{h}",f"future_ibov_{h}"]).copy()
    data=data.sort_values(["datetime","instrument"]).reset_index(drop=True)
    dates=pd.Index(sorted(data["datetime"].unique()))
    n=len(dates)
    embargo=max(1,math.ceil(h/STEP))

    if n<180:
        continue

    horizon_fold_tests=[]
    fold_thresholds=[]

    for fold_id,(a,b) in enumerate([(0.55,.70),(.70,.85),(.85,1.0)],1):
        ts=int(n*a); te=min(n,int(n*b))
        vlen=max(24,int(n*.10))
        vs=ts-vlen
        train_end=vs-embargo
        ve=ts-embargo

        if train_end<60 or ve<=vs:
            continue

        trd=dates[:train_end]
        vad=dates[vs:ve]
        ted=dates[ts:te]

        tr=data[data["datetime"].isin(trd)].copy()
        va=data[data["datetime"].isin(vad)].copy()
        tef=data[data["datetime"].isin(ted)].copy()

        tr,processed,_,_,_=preprocess(tr,[va,tef])
        va,tef=processed

        ds=MemoryDataset(
            to_qlib(tr,label_col),
            to_qlib(va,label_col),
            to_qlib(tef,label_col),
        )
        m=make_model()
        m.fit(ds,verbose_eval=0)

        pv=m.predict(ds,"valid")

        # Threshold escolhido exclusivamente na validação.
        candidates=[]
        for thr in THRESHOLDS:
            ev=evaluate(va,pv,h,thr)
            if ev.empty or ev["n"].sum()<15:
                continue
            candidates.append((
                float(ev["precision"].mean()),
                float(ev["net_alpha_vs_ibov"].mean()),
                int(ev["n"].sum()),
                thr,
            ))

        if not candidates:
            chosen_thr=.65
        else:
            candidates.sort(key=lambda z:(z[0],z[1],z[2]),reverse=True)
            chosen_thr=candidates[0][3]

        fold_thresholds.append(chosen_thr)

        pt=m.predict(ds,"test")
        ev_test=evaluate(tef,pt,h,chosen_thr)

        if not ev_test.empty:
            ev_test["fold"]=fold_id
            horizon_fold_tests.append(ev_test)
            signal_rows.append(ev_test)

        base_rate=float(tef[label_col].mean())
        test_precision=float(ev_test["precision"].mean()) if not ev_test.empty else np.nan
        test_alpha=float(ev_test["net_alpha_vs_ibov"].mean()) if not ev_test.empty else np.nan
        test_cdi=float(ev_test["net_excess_vs_cdi"].dropna().mean()) if (not ev_test.empty and ev_test["net_excess_vs_cdi"].notna().any()) else np.nan
        total_signals=int(ev_test["n"].sum()) if not ev_test.empty else 0

        fold_rows.append(dict(
            horizon=h,
            fold=fold_id,
            threshold=chosen_thr,
            base_rate=base_rate,
            precision=test_precision,
            precision_lift=(test_precision-base_rate if not pd.isna(test_precision) else np.nan),
            avg_net_alpha_vs_ibov=test_alpha,
            avg_net_excess_vs_cdi=test_cdi,
            signals=total_signals,
            test_start=str(pd.Timestamp(ted.min()).date()),
            test_end=str(pd.Timestamp(ted.max()).date()),
        ))

        print(
            f"[H={h}] fold={fold_id} thr={chosen_thr} "
            f"precision={test_precision:.3f} alpha={test_alpha:.4f}",
            flush=True,
        )

    chosen_thresholds[h]=float(pd.Series(fold_thresholds).median()) if fold_thresholds else .65

fold_df=pd.DataFrame(fold_rows)
sig_df=pd.concat(signal_rows,ignore_index=True) if signal_rows else pd.DataFrame()

summary_h={}
for h in HORIZONS:
    f=fold_df[fold_df["horizon"]==h]
    s=sig_df[sig_df["horizon"]==h] if not sig_df.empty else pd.DataFrame()

    if f.empty:
        continue

    precision=float(f["precision"].mean())
    base=float(f["base_rate"].mean())
    alpha=float(f["avg_net_alpha_vs_ibov"].mean())
    cdi_excess=float(f["avg_net_excess_vs_cdi"].dropna().mean()) if f["avg_net_excess_vs_cdi"].notna().any() else None
    signals=int(f["signals"].sum())
    stable_folds=int(((f["precision_lift"]>0)&(f["avg_net_alpha_vs_ibov"]>0)).sum())

    accepted=(
        signals>=25
        and precision>=.58
        and precision>=base+.08
        and alpha>0
        and stable_folds>=2
        and (cdi_excess is None or cdi_excess>0)
    )

    if accepted:
        accepted_horizons.append(h)

    summary_h[str(h)]=dict(
        folds=int(len(f)),
        signals=signals,
        base_rate=base,
        precision=precision,
        precision_lift=precision-base,
        avg_net_alpha_vs_ibov=alpha,
        avg_net_excess_vs_cdi=cdi_excess,
        stable_positive_folds=stable_folds,
        chosen_current_threshold=chosen_thresholds.get(h,.65),
        accepted=bool(accepted),
        acceptance_rule=">=25 OOS signals; precision>=58%; lift>=8pp; positive net alpha; >=2/3 positive folds; positive CDI excess when available."
    )

# Modelo atual para cada horizonte
live=panel.sort_values("datetime").groupby("instrument",as_index=False).tail(1).copy()
current_frames=[]

for h in HORIZONS:
    label_col=f"label_{h}"
    data=sampled.dropna(subset=[label_col]).copy().sort_values(["datetime","instrument"])
    dates=pd.Index(sorted(data["datetime"].unique()))
    if len(dates)<120:
        continue

    embargo=max(1,math.ceil(h/STEP))
    vlen=max(30,int(len(dates)*.12))
    vs=len(dates)-vlen
    train_end=max(1,vs-embargo)

    tr=data[data["datetime"].isin(dates[:train_end])].copy()
    va=data[data["datetime"].isin(dates[vs:])].copy()
    tr,processed,_,_,_=preprocess(tr,[va,live])
    va,livep=processed

    ds=MemoryDataset(
        to_qlib(tr,label_col),
        to_qlib(va,label_col),
        to_qlib(livep,None),
    )
    m=make_model()
    m.fit(ds,verbose_eval=0)
    pred=m.predict(ds,"test")

    cur=livep[["instrument","sector","datetime"]].copy()
    cur[f"prob_{h}d"]=np.asarray(pred,float)
    cur[f"threshold_{h}d"]=chosen_thresholds.get(h,.65)
    cur[f"confirm_{h}d"]=cur[f"prob_{h}d"]>=cur[f"threshold_{h}d"]
    current_frames.append(cur)

current=current_frames[0]
for nxt in current_frames[1:]:
    current=current.merge(
        nxt.drop(columns=["sector","datetime"]),
        on="instrument",
        how="outer",
    )

accepted_prob_cols=[f"prob_{h}d" for h in accepted_horizons if f"prob_{h}d" in current.columns]
if accepted_prob_cols:
    current["opportunity_probability_score"]=current[accepted_prob_cols].mean(axis=1)*100
    confirm_cols=[f"confirm_{h}d" for h in accepted_horizons if f"confirm_{h}d" in current.columns]
    current["opportunity_confirm"]=current[confirm_cols].any(axis=1)
else:
    all_prob=[c for c in current.columns if c.startswith("prob_")]
    current["opportunity_probability_score"]=current[all_prob].mean(axis=1)*100
    current["opportunity_confirm"]=False

current["opportunity_status"]=np.where(
    current["opportunity_confirm"],
    "CONFIRMA_OPORTUNIDADE",
    "NAO_CONFIRMA",
)
current=current.rename(columns={"instrument":"ticker"})

fold_df.to_csv(OUT/"qlib_v3_classifier_folds.csv",index=False)
if not sig_df.empty:
    sig_df.to_csv(OUT/"qlib_v3_classifier_oos_signals.csv",index=False)
current.to_csv(OUT/"qlib_v3_current_scores.csv",index=False)

summary=dict(
    engine="Microsoft Qlib LGBModel / LightGBM binary opportunity classifier",
    target="future stock beats Ibovespa + costs + minimum margin; also beats CDI + costs when CDI is available",
    horizons=HORIZONS,
    minimum_extra_alpha=MIN_EXTRA_ALPHA,
    round_trip_cost=ROUND_TRIP_COST,
    validation="3 expanding purged walk-forward folds; embargo by horizon; probability threshold selected only on validation",
    horizon_results=summary_h,
    accepted_horizons=accepted_horizons,
    ml_v3_status=("ACCEPTED" if accepted_horizons else "REJECTED"),
    current_scored_stocks=int(len(current)),
    current_confirms=int(current["opportunity_confirm"].sum()),
    limitations=[
        "Universe begins with currently listed companies, so survivorship bias remains.",
        "Historical model still lacks point-in-time accounting fundamentals.",
        "Yahoo Finance is free/non-official and can have gaps.",
        "Model probabilities are LightGBM scores and are not guaranteed/calibrated probabilities of profit."
    ]
)

(OUT/"qlib_v3_summary.json").write_text(
    json.dumps(summary,ensure_ascii=False,indent=2,default=str),
    encoding="utf-8",
)

print(json.dumps(summary,ensure_ascii=False,indent=2,default=str))
