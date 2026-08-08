import numpy as np
import pandas as pd

GENERAL_POSITIVE = {
    "quality": ["roe", "operating_margin", "profit_margin", "fcf_margin"],
    "growth": ["revenue_growth", "earnings_growth"],
    "valuation": ["fcf_yield"],
    "momentum": ["ret_3m", "ret_6m", "ret_12m", "price_to_ma50", "price_to_ma200"],
    "risk": [],
}
GENERAL_NEGATIVE = {
    "quality": ["debt_to_ebitda"],
    "growth": [],
    "valuation": ["pe", "pb", "ev_ebitda"],
    "momentum": [],
    "risk": ["volatility", "beta"],
}

# Bancos/seguradoras não devem ser tratados como indústria comum:
# dívida e EV/EBITDA têm interpretação distinta. Nesta primeira versão,
# priorizamos ROE, lucro/crescimento, P/L e P/VP.
FIN_POSITIVE = {
    "quality": ["roe", "profit_margin"],
    "growth": ["revenue_growth", "earnings_growth"],
    "valuation": [],
    "momentum": ["ret_3m", "ret_6m", "ret_12m", "price_to_ma50", "price_to_ma200"],
    "risk": [],
}
FIN_NEGATIVE = {
    "quality": [],
    "growth": [],
    "valuation": ["pe", "pb"],
    "momentum": [],
    "risk": ["volatility", "beta"],
}


def _is_financial(row):
    txt = f"{row.get('sector','')} {row.get('industry','')}".lower()
    keys = ["finance", "financial", "banco", "bank", "segur", "insurance", "credito", "credit"]
    return any(k in txt for k in keys)


def _percentile(s, higher_is_better=True):
    s = pd.to_numeric(s, errors="coerce")
    r = s.rank(pct=True, method="average") * 100.0
    return r if higher_is_better else 100.0 - r


def _group_percentile(df, field, positive=True, min_sector_size=5):
    values = pd.to_numeric(df[field], errors="coerce")
    sector = df["sector"].fillna("Unknown")
    out = pd.Series(index=df.index, dtype=float)
    global_rank = _percentile(values, positive)
    for _, idx in df.groupby(sector).groups.items():
        idx = list(idx)
        valid = values.loc[idx].notna().sum()
        if valid >= min_sector_size:
            out.loc[idx] = _percentile(values.loc[idx], positive)
        else:
            out.loc[idx] = global_rank.loc[idx]
    return out


def _score_subset(out, mask, pos_map, neg_map, min_sector_size):
    sub = out.loc[mask].copy()
    if sub.empty:
        return out
    for component in ["quality", "growth", "valuation", "momentum", "risk"]:
        cols = []
        for f in pos_map[component]:
            c = f"__{component}_{f}"
            sub[c] = _group_percentile(sub, f, True, min_sector_size)
            cols.append(c)
        for f in neg_map[component]:
            c = f"__{component}_{f}"
            sub[c] = _group_percentile(sub, f, False, min_sector_size)
            cols.append(c)
        if component == "risk":
            c = "__risk_max_drawdown"
            sub[c] = _group_percentile(sub, "max_drawdown", True, min_sector_size)
            cols.append(c)
        sub[f"{component}_score"] = sub[cols].mean(axis=1, skipna=True) if cols else np.nan
        sub[f"{component}_coverage"] = sub[cols].notna().mean(axis=1) if cols else 0.0
    for c in sub.columns:
        if c not in out.columns:
            out[c] = np.nan
    out.loc[sub.index, sub.columns] = sub
    return out


def score_dataframe(df, weights, min_coverage=0.65, min_sector_size=5):
    out = df.copy()
    out["model_group"] = out.apply(lambda r: "financial" if _is_financial(r) else "general", axis=1)
    out = _score_subset(out, out["model_group"].eq("general"), GENERAL_POSITIVE, GENERAL_NEGATIVE, min_sector_size)
    out = _score_subset(out, out["model_group"].eq("financial"), FIN_POSITIVE, FIN_NEGATIVE, min_sector_size)

    weighted_cols = []
    for component, w in weights.items():
        c = f"__weighted_{component}"
        out[c] = out[f"{component}_score"] * float(w)
        weighted_cols.append(c)
    out["score"] = out[weighted_cols].sum(axis=1, min_count=1)
    coverage_cols = [f"{x}_coverage" for x in weights]
    out["data_coverage"] = out[coverage_cols].mean(axis=1)
    out.loc[out["data_coverage"] < min_coverage, "score"] = np.minimum(
        out.loc[out["data_coverage"] < min_coverage, "score"], 69.99
    )
    return out
