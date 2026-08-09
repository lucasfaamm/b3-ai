
from __future__ import annotations

from pathlib import Path
import json
import math

import numpy as np
import pandas as pd


OUT = Path("results")
OUT.mkdir(exist_ok=True)

PANEL = OUT / "fundamental_pit_backtest_panel.csv.gz"
CAL = OUT / "calibrated_weights.json"

if not PANEL.exists() or not CAL.exists():
    raise SystemExit(
        "Faltam results/fundamental_pit_backtest_panel.csv.gz "
        "ou results/calibrated_weights.json"
    )

panel = pd.read_csv(PANEL, parse_dates=["date", "next_date"])
cal = json.loads(CAL.read_text(encoding="utf-8"))

DEFAULT_WEIGHTS = {
    "quality": 0.25,
    "growth": 0.20,
    "valuation": 0.20,
    "momentum": 0.20,
    "risk": 0.15,
}

OPTIMIZED_WEIGHTS = {
    k: float(v)
    for k, v in cal.get("weights", DEFAULT_WEIGHTS).items()
}

test_start = pd.Timestamp(cal["periods"]["untouched_test"][0])
test_end = pd.Timestamp(cal["periods"]["untouched_test"][1])

components = [
    "quality",
    "growth",
    "valuation",
    "momentum",
    "risk",
]

needed = [
    "date", "ticker", "fwd_return", "ibov_return", "cdi_return",
    "quality_score", "growth_score", "valuation_score",
    "momentum_score", "risk_score",
]

missing = [c for c in needed if c not in panel.columns]
if missing:
    raise SystemExit(f"Colunas ausentes no painel: {missing}")

test = panel[
    (panel["date"] >= test_start)
    & (panel["date"] <= test_end)
].copy()

if test["date"].nunique() < 12:
    raise SystemExit("Poucos meses no período final para avaliação.")

def build_monthly(weights):
    rows = []
    previous = {}

    for dt, g in test.groupby("date", sort=True):
        g = g.dropna(subset=["fwd_return"]).copy()

        if len(g) < 15:
            continue

        score = np.zeros(len(g), dtype=float)

        for name in components:
            score += (
                pd.to_numeric(
                    g[f"{name}_score"],
                    errors="coerce",
                )
                .fillna(0.0)
                .values
                * float(weights[name])
            )

        g["model_score"] = score

        top = (
            g.sort_values(
                "model_score",
                ascending=False,
            )
            .head(min(10, len(g)))
            .copy()
        )

        current = {
            t: 1.0 / len(top)
            for t in top["ticker"]
        }

        if previous:
            all_names = set(previous) | set(current)
            overlap = sum(
                min(
                    previous.get(name, 0.0),
                    current.get(name, 0.0),
                )
                for name in all_names
            )
            turnover = 1.0 - overlap
        else:
            turnover = 1.0

        trading_cost = 0.002 * turnover

        gross = float(top["fwd_return"].mean())
        net = gross - trading_cost
        ibov = float(top["ibov_return"].mean())

        cdi_values = pd.to_numeric(
            top["cdi_return"],
            errors="coerce",
        )

        cdi = (
            float(cdi_values.mean())
            if cdi_values.notna().any()
            else np.nan
        )

        rows.append({
            "date": pd.Timestamp(dt),
            "gross_return": gross,
            "net_return": net,
            "ibov_return": ibov,
            "cdi_return": cdi,
            "alpha_vs_ibov": net - ibov,
            "excess_vs_cdi": (
                net - cdi
                if not pd.isna(cdi)
                else np.nan
            ),
            "turnover": turnover,
            "n": int(len(top)),
            "tickers": ",".join(top["ticker"].astype(str).tolist()),
        })

        previous = current

    return pd.DataFrame(rows)


def cagr(s, periods_per_year=12):
    s = pd.Series(s, dtype=float).dropna()
    if len(s) == 0:
        return np.nan
    wealth = float((1 + s).prod())
    years = len(s) / periods_per_year
    return wealth ** (1 / years) - 1 if years > 0 else np.nan


def max_drawdown(s):
    s = pd.Series(s, dtype=float).dropna()
    if len(s) == 0:
        return np.nan
    wealth = (1 + s).cumprod()
    return float((wealth / wealth.cummax() - 1).min())


def sortino(s, target=0.0, periods_per_year=12):
    s = pd.Series(s, dtype=float).dropna()
    if len(s) < 3:
        return np.nan
    excess = s - target
    downside = excess[excess < 0]
    if len(downside) < 2:
        return np.nan
    downside_dev = float(
        np.sqrt((downside ** 2).mean())
    )
    if downside_dev == 0:
        return np.nan
    return float(
        excess.mean()
        / downside_dev
        * np.sqrt(periods_per_year)
    )


def block_bootstrap_mean_ci(series, block=3, n_boot=10000, seed=42):
    x = pd.Series(series, dtype=float).dropna().to_numpy()

    if len(x) < 8:
        return {
            "mean": float(np.mean(x)) if len(x) else None,
            "ci95_low": None,
            "ci95_high": None,
            "probability_mean_gt_zero": None,
            "n": int(len(x)),
        }

    rng = np.random.default_rng(seed)
    n = len(x)
    means = np.empty(n_boot, dtype=float)

    for b in range(n_boot):
        sample = []

        while len(sample) < n:
            start = int(rng.integers(0, n))
            for j in range(block):
                sample.append(
                    x[(start + j) % n]
                )
                if len(sample) >= n:
                    break

        means[b] = np.mean(sample[:n])

    return {
        "mean": float(np.mean(x)),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
        "probability_mean_gt_zero": float((means > 0).mean()),
        "n": int(n),
        "bootstrap": "circular block bootstrap, block=3 months, 10000 resamples",
    }


def summarize(monthly):
    r = monthly["net_return"]
    ib = monthly["ibov_return"]
    cdi = monthly["cdi_return"]

    alpha = monthly["alpha_vs_ibov"]
    excess_cdi = monthly["excess_vs_cdi"].dropna()

    strategy_cagr = float(cagr(r))
    ibov_cagr = float(cagr(ib))
    cdi_cagr = (
        float(cagr(cdi.dropna()))
        if cdi.notna().any()
        else None
    )

    strategy_dd = float(max_drawdown(r))
    ibov_dd = float(max_drawdown(ib))

    calmar = (
        float(strategy_cagr / abs(strategy_dd))
        if strategy_dd < 0
        else None
    )

    return {
        "months": int(len(monthly)),
        "strategy_cagr": strategy_cagr,
        "ibov_cagr": ibov_cagr,
        "cdi_cagr": cdi_cagr,
        "annualized_alpha_vs_ibov_approx": float(alpha.mean() * 12),
        "annualized_excess_vs_cdi_approx": (
            float(excess_cdi.mean() * 12)
            if len(excess_cdi)
            else None
        ),
        "monthly_alpha_vs_ibov_mean": float(alpha.mean()),
        "monthly_alpha_vs_ibov_hit_rate": float((alpha > 0).mean()),
        "monthly_excess_vs_cdi_mean": (
            float(excess_cdi.mean())
            if len(excess_cdi)
            else None
        ),
        "monthly_excess_vs_cdi_hit_rate": (
            float((excess_cdi > 0).mean())
            if len(excess_cdi)
            else None
        ),
        "strategy_sortino_rf0": sortino(r),
        "strategy_max_drawdown": strategy_dd,
        "ibov_max_drawdown": ibov_dd,
        "calmar": calmar,
        "avg_turnover": float(monthly["turnover"].mean()),
        "bootstrap_alpha_vs_ibov": block_bootstrap_mean_ci(alpha),
        "bootstrap_excess_vs_cdi": (
            block_bootstrap_mean_ci(excess_cdi)
            if len(excess_cdi)
            else None
        ),
    }


default_monthly = build_monthly(DEFAULT_WEIGHTS)
optimized_monthly = build_monthly(OPTIMIZED_WEIGHTS)

default_summary = summarize(default_monthly)
optimized_summary = summarize(optimized_monthly)

# Paired comparison is descriptive only. It is NOT used to select a model
# because this test period has already been inspected.
paired = (
    default_monthly[
        ["date", "net_return"]
    ]
    .rename(
        columns={
            "net_return": "default_return",
        }
    )
    .merge(
        optimized_monthly[
            ["date", "net_return"]
        ].rename(
            columns={
                "net_return": "optimized_return",
            }
        ),
        on="date",
        how="inner",
    )
)

paired["default_minus_optimized"] = (
    paired["default_return"]
    - paired["optimized_return"]
)

paired_bootstrap = block_bootstrap_mean_ci(
    paired["default_minus_optimized"]
)

# Conservative evidence gate.
# This does NOT enable a model yet because:
# 1) the period has already been inspected;
# 2) only 17 months are available in the latest test.
def evidence_gate(summary):
    alpha_ci = summary["bootstrap_alpha_vs_ibov"]
    cdi_ci = summary["bootstrap_excess_vs_cdi"]

    enough_months = summary["months"] >= 24

    ibov_ok = (
        summary["strategy_cagr"] > summary["ibov_cagr"]
        and summary["monthly_alpha_vs_ibov_hit_rate"] >= 0.50
        and alpha_ci["ci95_low"] is not None
        and alpha_ci["ci95_low"] > 0
    )

    if summary["cdi_cagr"] is None:
        cdi_ok = True
    else:
        cdi_ok = (
            summary["strategy_cagr"] > summary["cdi_cagr"]
            and summary["monthly_excess_vs_cdi_hit_rate"] >= 0.50
            and cdi_ci is not None
            and cdi_ci["ci95_low"] is not None
            and cdi_ci["ci95_low"] > 0
        )

    drawdown_ok = (
        summary["strategy_max_drawdown"]
        >= summary["ibov_max_drawdown"] * 1.25
    )

    return {
        "enough_months_24": bool(enough_months),
        "ibov_evidence": bool(ibov_ok),
        "cdi_evidence": bool(cdi_ok),
        "drawdown_evidence": bool(drawdown_ok),
        "passes_all": bool(
            enough_months
            and ibov_ok
            and cdi_ok
            and drawdown_ok
        ),
    }


default_gate = evidence_gate(default_summary)
optimized_gate = evidence_gate(optimized_summary)

result = {
    "test_period": {
        "start": str(test_start.date()),
        "end": str(test_end.date()),
        "important": (
            "This period has already been inspected. "
            "It is now treated as an evidence period, not a fresh untouched test "
            "for any new model-selection decisions."
        ),
    },
    "benchmark_fix": (
        "IBOV and CDI are evaluated separately. "
        "The invalid month-by-month max(IBOV, CDI) benchmark has been removed."
    ),
    "default_pre_specified_weights": {
        "weights": DEFAULT_WEIGHTS,
        "metrics": default_summary,
        "evidence_gate": default_gate,
    },
    "optimized_weights_fixed_from_prior_train_validation": {
        "weights": OPTIMIZED_WEIGHTS,
        "metrics": optimized_summary,
        "evidence_gate": optimized_gate,
    },
    "default_vs_optimized_descriptive_only": {
        "bootstrap_default_minus_optimized": paired_bootstrap,
        "important": (
            "Do not choose between the two using this already-seen period."
        ),
    },
    "strong_buy_authorized_now": False,
    "reason_strong_buy_blocked": (
        "No fixed model has >=24 months of fresh forward evidence with "
        "95% bootstrap lower bounds above zero versus both IBOV and CDI."
    ),
    "next_validation_rule": (
        "Freeze the deployed rule now and accumulate new monthly forward outcomes. "
        "Only future observations after this freeze can unlock COMPRA_FORTE."
    ),
}

(OUT / "fixed_model_evidence.json").write_text(
    json.dumps(
        result,
        ensure_ascii=False,
        indent=2,
        default=str,
    ),
    encoding="utf-8",
)

default_monthly.to_csv(
    OUT / "default_weights_evidence_monthly.csv",
    index=False,
)

optimized_monthly.to_csv(
    OUT / "optimized_weights_evidence_monthly.csv",
    index=False,
)

print(
    json.dumps(
        result,
        ensure_ascii=False,
        indent=2,
        default=str,
    )
)
