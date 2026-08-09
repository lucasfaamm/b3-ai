
from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
import vectorbt as vbt
import yfinance as yf


FUND = Path("data/fundamentals.csv")
OUTDIR = Path("results")
OUTDIR.mkdir(exist_ok=True)

if not FUND.exists():
    raise SystemExit("Falta data/fundamentals.csv")

fund = pd.read_csv(FUND)

tickers = (
    fund["ticker"]
    .dropna()
    .astype(str)
    .str.upper()
    .drop_duplicates()
    .tolist()[:120]
)

symbols = [f"{t}.SA" for t in tickers]

raw = yf.download(
    symbols,
    period="5y",
    interval="1d",
    auto_adjust=True,
    progress=False,
    threads=True,
    group_by="ticker",
)

def get_close(symbol):
    try:
        d = raw[symbol] if len(symbols) > 1 else raw
        c = d["Close"]
        if isinstance(c, pd.DataFrame):
            c = c.iloc[:, 0]
        c = pd.to_numeric(c, errors="coerce")
        return c
    except Exception:
        return pd.Series(dtype=float)

close = pd.DataFrame({
    t: get_close(f"{t}.SA")
    for t in tickers
}).sort_index()

close = close.dropna(
    axis=1,
    thresh=max(400, int(len(close) * 0.55))
)
close = close.ffill(limit=3)

if close.shape[1] < 15:
    raise SystemExit(
        f"Poucos ativos com histórico suficiente: {close.shape[1]}"
    )

split_idx = int(len(close) * 0.65)
split_date = close.index[split_idx]


def metrics(ret):
    s = pd.Series(ret, dtype=float).dropna()

    if len(s) < 30:
        return {
            "days": int(len(s)),
            "cagr": np.nan,
            "volatility": np.nan,
            "sharpe_rf0": np.nan,
            "max_drawdown": np.nan,
            "total_return": np.nan,
        }

    wealth = (1 + s).cumprod()
    years = len(s) / 252.0

    cagr = (
        wealth.iloc[-1] ** (1 / years) - 1
        if years > 0 else np.nan
    )

    vol = s.std(ddof=1) * np.sqrt(252)

    sharpe = (
        s.mean() * 252 / vol
        if vol and np.isfinite(vol)
        else np.nan
    )

    dd = (
        wealth / wealth.cummax() - 1
    ).min()

    return {
        "days": int(len(s)),
        "cagr": float(cagr),
        "volatility": float(vol),
        "sharpe_rf0": (
            float(sharpe)
            if np.isfinite(sharpe)
            else None
        ),
        "max_drawdown": float(dd),
        "total_return": float(
            wealth.iloc[-1] - 1
        ),
    }


def run_strategy(fast, slow, mom_days):

    # Usa pandas para gerar sinais mantendo exatamente
    # as mesmas colunas/tickers. O VectorBT continua
    # responsável pela simulação da carteira.
    fast_ma = close.rolling(
        window=fast,
        min_periods=fast,
    ).mean()

    slow_ma = close.rolling(
        window=slow,
        min_periods=slow,
    ).mean()

    momentum = (
        close / close.shift(mom_days) - 1
    )

    entries = (
        (fast_ma > slow_ma)
        & (momentum > 0)
        & (close > slow_ma)
    ).fillna(False)

    exits = (
        (fast_ma < slow_ma)
        | (momentum < 0)
    ).fillna(False)

    # Um portfólio por ativo/coluna.
    pf = vbt.Portfolio.from_signals(
        close=close,
        entries=entries,
        exits=exits,
        init_cash=100000,
        fees=0.0005,
        slippage=0.0005,
        freq="1D",
    )

    r = pf.returns()

    if isinstance(r, pd.Series):
        r = r.to_frame()

    # Carteira sintética de pesos iguais entre ativos.
    portfolio_ret = (
        r.mean(axis=1, skipna=True)
        .fillna(0.0)
    )

    train = portfolio_ret.loc[
        portfolio_ret.index < split_date
    ]

    test = portfolio_ret.loc[
        portfolio_ret.index >= split_date
    ]

    return metrics(train), metrics(test)


grid = []

for fast in [20, 50]:
    for slow in [100, 150, 200]:

        if fast >= slow:
            continue

        for mom in [63, 126, 252]:

            try:
                tr, te = run_strategy(
                    fast,
                    slow,
                    mom,
                )

                grid.append({
                    "fast_ma": fast,
                    "slow_ma": slow,
                    "momentum_days": mom,

                    "train_sharpe": tr["sharpe_rf0"],
                    "train_cagr": tr["cagr"],
                    "train_max_drawdown": tr["max_drawdown"],

                    "test_sharpe": te["sharpe_rf0"],
                    "test_cagr": te["cagr"],
                    "test_max_drawdown": te["max_drawdown"],
                    "test_total_return": te["total_return"],
                })

                print(
                    f"OK fast={fast} "
                    f"slow={slow} "
                    f"mom={mom}",
                    flush=True,
                )

            except Exception as e:

                print(
                    f"ERRO fast={fast} "
                    f"slow={slow} "
                    f"mom={mom}: {e}",
                    flush=True,
                )


grid_df = pd.DataFrame(grid)

if grid_df.empty:
    raise SystemExit(
        "Nenhuma estratégia VectorBT executou."
    )

grid_df = grid_df.sort_values(
    ["train_sharpe", "train_cagr"],
    ascending=False,
    na_position="last",
)

grid_df.to_csv(
    OUTDIR / "vectorbt_parameter_sweep.csv",
    index=False,
)

best = grid_df.iloc[0].to_dict()

summary = {
    "engine": "vectorbt",
    "universe_size": int(close.shape[1]),
    "history_start": str(
        close.index.min().date()
    ),
    "history_end": str(
        close.index.max().date()
    ),
    "train_test_split": str(
        pd.Timestamp(split_date).date()
    ),
    "selection_rule": (
        "Parâmetros escolhidos somente pelo "
        "Sharpe no treino; métricas seguintes "
        "são do período fora da amostra."
    ),
    "best_parameters": {
        "fast_ma": int(best["fast_ma"]),
        "slow_ma": int(best["slow_ma"]),
        "momentum_days": int(
            best["momentum_days"]
        ),
    },
    "out_of_sample": {
        "cagr": best["test_cagr"],
        "sharpe_rf0": best["test_sharpe"],
        "max_drawdown": best[
            "test_max_drawdown"
        ],
        "total_return": best[
            "test_total_return"
        ],
    },
    "limitations": [
        "Valida somente camada técnica/preço.",
        "Universo usa empresas atuais: existe survivorship bias.",
        "Não substitui backtest fundamentalista point-in-time.",
        "Custos simulados: 0,05% taxa + 0,05% slippage por operação.",
    ],
}

(
    OUTDIR / "vectorbt_summary.json"
).write_text(
    json.dumps(
        summary,
        ensure_ascii=False,
        indent=2,
        default=str,
    ),
    encoding="utf-8",
)

print(
    json.dumps(
        summary,
        ensure_ascii=False,
        indent=2,
        default=str,
    )
)
