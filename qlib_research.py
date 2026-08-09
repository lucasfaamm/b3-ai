
from __future__ import annotations

from pathlib import Path
import json
import math

import numpy as np
import pandas as pd
import yfinance as yf


# Qlib is intentionally isolated in this workflow.
import qlib.contrib.model.gbdt as qlib_gbdt


FUND = Path("data/fundamentals.csv")
OUT = Path("results")
OUT.mkdir(exist_ok=True)

HORIZON = 20
MAX_TICKERS = 120
PERIOD = "7y"

if not FUND.exists():
    raise SystemExit("Falta data/fundamentals.csv.")

fund = pd.read_csv(FUND)
tickers = (
    fund["ticker"]
    .dropna()
    .astype(str)
    .str.upper()
    .drop_duplicates()
    .tolist()[:MAX_TICKERS]
)

symbols = [f"{t}.SA" for t in tickers] + ["^BVSP"]

raw = yf.download(
    symbols,
    period=PERIOD,
    interval="1d",
    auto_adjust=True,
    progress=False,
    threads=True,
    group_by="ticker",
)


def field(symbol, name):
    try:
        d = raw[symbol]
        x = d[name]
        if isinstance(x, pd.DataFrame):
            x = x.iloc[:, 0]
        x = pd.to_numeric(x, errors="coerce")
        x.index = pd.to_datetime(x.index)
        return x.sort_index()
    except Exception:
        return pd.Series(dtype=float)


ibov = field("^BVSP", "Close").dropna()
if len(ibov) < 500:
    raise SystemExit("Histórico do Ibovespa insuficiente.")


def rsi14(close):
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def build_one(ticker):
    sym = f"{ticker}.SA"
    close = field(sym, "Close")
    volume = field(sym, "Volume")

    if len(close) < 700:
        return pd.DataFrame()

    df = pd.DataFrame(index=close.index)
    df["close"] = close
    df["volume"] = volume.reindex(df.index)

    for n in [5, 21, 63, 126, 252]:
        df[f"ret_{n}"] = close.pct_change(n)

    df["vol_20"] = close.pct_change().rolling(20).std() * np.sqrt(252)
    df["vol_63"] = close.pct_change().rolling(63).std() * np.sqrt(252)

    for n in [20, 50, 200]:
        ma = close.rolling(n).mean()
        df[f"price_ma_{n}"] = close / ma - 1.0

    df["rsi_14"] = rsi14(close) / 100.0
    df["drawdown_63"] = close / close.rolling(63).max() - 1.0

    v20 = volume.rolling(20).mean()
    v60 = volume.rolling(60).mean()
    df["volume_ratio_20_60"] = v20 / v60.replace(0, np.nan)

    ib = ibov.reindex(df.index).ffill()
    for n in [21, 63, 126]:
        ib_ret = ib.pct_change(n)
        df[f"ibov_ret_{n}"] = ib_ret
        df[f"rel_ret_{n}"] = df[f"ret_{n}"] - ib_ret

    df["ibov_vol_20"] = ib.pct_change().rolling(20).std() * np.sqrt(252)

    future_stock = close.shift(-HORIZON) / close - 1.0
    future_ibov = ib.shift(-HORIZON) / ib - 1.0

    df["future_stock_20d"] = future_stock
    df["future_ibov_20d"] = future_ibov
    df["label"] = future_stock - future_ibov

    df["instrument"] = ticker
    df["datetime"] = df.index
    return df.reset_index(drop=True)


parts = []
for i, t in enumerate(tickers, 1):
    x = build_one(t)
    if not x.empty:
        parts.append(x)
        print(f"[{i}/{len(tickers)}] OK {t}", flush=True)

panel = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
if panel.empty:
    raise SystemExit("Painel ML vazio.")

feature_cols = [
    "ret_5", "ret_21", "ret_63", "ret_126", "ret_252",
    "vol_20", "vol_63",
    "price_ma_20", "price_ma_50", "price_ma_200",
    "rsi_14", "drawdown_63", "volume_ratio_20_60",
    "ibov_ret_21", "ibov_ret_63", "ibov_ret_126",
    "rel_ret_21", "rel_ret_63", "rel_ret_126",
    "ibov_vol_20",
]

panel = panel.sort_values(["datetime", "instrument"])
panel = panel.replace([np.inf, -np.inf], np.nan)

# Linhas com informação técnica suficiente.
panel["feature_coverage"] = panel[feature_cols].notna().mean(axis=1)
panel = panel[panel["feature_coverage"] >= 0.80].copy()

# Amostra semanal para reduzir a dependência causada por labels de 20 dias sobrepostos.
all_dates = pd.Index(sorted(panel["datetime"].dropna().unique()))
sample_dates = set(all_dates[::5])
sampled = panel[panel["datetime"].isin(sample_dates)].copy()

labeled = sampled.dropna(subset=["label"]).copy()
if len(labeled) < 3000:
    raise SystemExit(f"Poucos exemplos rotulados: {len(labeled)}")

dates = pd.Index(sorted(labeled["datetime"].unique()))
n = len(dates)
train_end = dates[int(n * 0.60)]
valid_end = dates[int(n * 0.80)]

train = labeled[labeled["datetime"] < train_end].copy()
valid = labeled[
    (labeled["datetime"] >= train_end) &
    (labeled["datetime"] < valid_end)
].copy()
test = labeled[labeled["datetime"] >= valid_end].copy()

# Imputação aprendida somente no treino.
medians = train[feature_cols].median(numeric_only=True)

for x in [train, valid, test]:
    x.loc[:, feature_cols] = x[feature_cols].fillna(medians)

# Winsoriza usando limites do treino apenas.
lo = train[feature_cols].quantile(0.005)
hi = train[feature_cols].quantile(0.995)

for x in [train, valid, test]:
    x.loc[:, feature_cols] = x[feature_cols].clip(lower=lo, upper=hi, axis=1)


def to_qlib_frame(df, include_label=True):
    idx = pd.MultiIndex.from_arrays(
        [
            pd.to_datetime(df["datetime"]).values,
            df["instrument"].astype(str).values,
        ],
        names=["datetime", "instrument"],
    )

    xf = df[feature_cols].copy()
    xf.index = idx

    if not include_label:
        return xf

    yf_ = df[["label"]].copy()
    yf_.index = idx

    return pd.concat(
        {
            "feature": xf,
            "label": yf_,
        },
        axis=1,
    ).sort_index()


class InMemoryQlibDataset:
    """
    Duck-typed DatasetH interface required by qlib.contrib.model.gbdt.LGBModel.
    Keeps our B3 data in memory, avoiding Qlib's CN/US provider datasets.
    """
    def __init__(self, train_df, valid_df, test_df):
        self.frames = {
            "train": train_df,
            "valid": valid_df,
            "test": test_df,
        }
        self.segments = {
            "train": "train",
            "valid": "valid",
            "test": "test",
        }

    def prepare(self, segment, col_set=None, data_key=None):
        if isinstance(segment, list):
            return [self.prepare(s, col_set, data_key) for s in segment]

        df = self.frames[segment]

        if col_set is None:
            return df

        if col_set == "feature":
            if isinstance(df.columns, pd.MultiIndex):
                return df["feature"]
            return df

        if col_set == "label":
            return df["label"]

        if isinstance(col_set, (list, tuple)):
            return df.loc[:, list(col_set)]

        return df


train_q = to_qlib_frame(train, True)
valid_q = to_qlib_frame(valid, True)
test_q = to_qlib_frame(test, True)

dataset = InMemoryQlibDataset(train_q, valid_q, test_q)

# Qlib's LGBModel logs metrics through its workflow recorder.
# For this isolated GitHub research job, disable only that side effect.
qlib_gbdt.R.log_metrics = lambda **kwargs: None

model = qlib_gbdt.LGBModel(
    loss="mse",
    early_stopping_rounds=50,
    num_boost_round=600,
    learning_rate=0.03,
    num_leaves=31,
    max_depth=6,
    feature_fraction=0.85,
    bagging_fraction=0.85,
    bagging_freq=1,
    lambda_l1=0.2,
    lambda_l2=1.0,
    num_threads=4,
    seed=42,
)

evals_result = {}
model.fit(
    dataset,
    verbose_eval=0,
    evals_result=evals_result,
)

pred = model.predict(dataset, "test")

truth = test_q["label"]["label"].reindex(pred.index)

res = pd.DataFrame({
    "prediction": pred,
    "label": truth,
}).dropna()

res = res.reset_index()

# IC diário / cross-sectional.
daily_ic = (
    res.groupby("datetime", group_keys=False)
    .apply(
        lambda g: g["prediction"].corr(g["label"], method="spearman")
        if len(g) >= 8 else np.nan,
        include_groups=False,
    )
    .dropna()
)

# Avaliação top-10 em datas aproximadamente não sobrepostas.
test_dates = pd.Index(sorted(res["datetime"].unique()))
eval_dates = test_dates[::4]  # amostra semanal -> ~20 pregões entre avaliações

portfolio_rows = []
for dt in eval_dates:
    g = res[res["datetime"] == dt].sort_values("prediction", ascending=False)
    if len(g) < 10:
        continue

    top = g.head(10)
    mean_alpha = float(top["label"].mean())

    raw = test[
        (test["datetime"] == dt) &
        (test["instrument"].isin(top["instrument"]))
    ]

    mean_stock = float(raw["future_stock_20d"].mean()) if len(raw) else np.nan
    mean_ibov = float(raw["future_ibov_20d"].mean()) if len(raw) else np.nan

    portfolio_rows.append({
        "date": str(pd.Timestamp(dt).date()),
        "top10_alpha_20d": mean_alpha,
        "top10_stock_return_20d": mean_stock,
        "ibov_return_20d": mean_ibov,
    })

portfolio = pd.DataFrame(portfolio_rows)

# Feature importance.
booster = model.model
importance = pd.DataFrame({
    "feature": feature_cols,
    "gain": booster.feature_importance(importance_type="gain"),
    "split": booster.feature_importance(importance_type="split"),
}).sort_values("gain", ascending=False)

# Modelo final: usa todo o histórico rotulado, mantém uma janela recente como validação.
all_labeled = sampled.dropna(subset=["label"]).copy()
all_labeled.loc[:, feature_cols] = all_labeled[feature_cols].fillna(medians)
all_labeled.loc[:, feature_cols] = all_labeled[feature_cols].clip(lower=lo, upper=hi, axis=1)

all_dates2 = pd.Index(sorted(all_labeled["datetime"].unique()))
cut = all_dates2[int(len(all_dates2) * 0.88)]

final_train = all_labeled[all_labeled["datetime"] < cut].copy()
final_valid = all_labeled[all_labeled["datetime"] >= cut].copy()

# Última observação disponível de cada ação para score atual.
live = (
    panel.sort_values("datetime")
    .groupby("instrument", as_index=False)
    .tail(1)
    .copy()
)
live.loc[:, feature_cols] = live[feature_cols].fillna(medians)
live.loc[:, feature_cols] = live[feature_cols].clip(lower=lo, upper=hi, axis=1)

final_ds = InMemoryQlibDataset(
    to_qlib_frame(final_train, True),
    to_qlib_frame(final_valid, True),
    to_qlib_frame(live, False),
)

final_model = qlib_gbdt.LGBModel(
    loss="mse",
    early_stopping_rounds=50,
    num_boost_round=600,
    learning_rate=0.03,
    num_leaves=31,
    max_depth=6,
    feature_fraction=0.85,
    bagging_fraction=0.85,
    bagging_freq=1,
    lambda_l1=0.2,
    lambda_l2=1.0,
    num_threads=4,
    seed=42,
)

final_model.fit(final_ds, verbose_eval=0)
live_pred = final_model.predict(final_ds, "test")

live_out = live_pred.rename("qlib_prediction").reset_index()
live_out["qlib_score"] = (
    live_out["qlib_prediction"].rank(pct=True, method="average") * 100.0
)
live_out["qlib_signal"] = pd.cut(
    live_out["qlib_score"],
    bins=[-np.inf, 25, 60, 75, 90, np.inf],
    labels=[
        "NEGATIVE",
        "NEUTRAL",
        "POSITIVE",
        "STRONG_POSITIVE",
        "TOP_SIGNAL",
    ],
)

# Salva tudo.
res.to_csv(OUT / "qlib_test_predictions.csv", index=False)
portfolio.to_csv(OUT / "qlib_top10_oos.csv", index=False)
importance.to_csv(OUT / "qlib_feature_importance.csv", index=False)
live_out.to_csv(OUT / "qlib_current_scores.csv", index=False)

summary = {
    "engine": "Microsoft Qlib LGBModel / LightGBM",
    "prediction_target": "20-trading-day stock return minus Ibovespa return",
    "universe_requested": len(tickers),
    "training_examples": int(len(train)),
    "validation_examples": int(len(valid)),
    "test_examples": int(len(test)),
    "train_end": str(pd.Timestamp(train_end).date()),
    "validation_end": str(pd.Timestamp(valid_end).date()),
    "test_daily_ic_mean": float(daily_ic.mean()) if len(daily_ic) else None,
    "test_daily_ic_median": float(daily_ic.median()) if len(daily_ic) else None,
    "test_daily_ic_positive_rate": float((daily_ic > 0).mean()) if len(daily_ic) else None,
    "oos_top10_periods": int(len(portfolio)),
    "oos_top10_avg_alpha_20d": (
        float(portfolio["top10_alpha_20d"].mean())
        if len(portfolio) else None
    ),
    "oos_top10_positive_alpha_rate": (
        float((portfolio["top10_alpha_20d"] > 0).mean())
        if len(portfolio) else None
    ),
    "oos_top10_avg_stock_return_20d": (
        float(portfolio["top10_stock_return_20d"].mean())
        if len(portfolio) else None
    ),
    "oos_ibov_avg_return_20d": (
        float(portfolio["ibov_return_20d"].mean())
        if len(portfolio) else None
    ),
    "top_features": importance.head(10).to_dict("records"),
    "current_scored_stocks": int(len(live_out)),
    "limitations": [
        "ML histórico usa apenas preço, volume e contexto do Ibovespa para evitar look-ahead de fundamentos atuais.",
        "Universo parte de empresas atuais; ainda existe survivorship bias.",
        "Yahoo Finance é fonte gratuita/não oficial e pode conter falhas.",
        "O Qlib score é ranking relativo do modelo, não probabilidade de lucro.",
        "Fundamentos atuais serão combinados ao ML somente na camada de decisão atual, não no treino histórico.",
    ],
}

(OUT / "qlib_summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2, default=str),
    encoding="utf-8",
)

print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
