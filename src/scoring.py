
import numpy as np
import pandas as pd

GENERAL_POSITIVE = {
    "quality": ["roe", "operating_margin", "profit_margin"],
    "growth": ["revenue_growth", "earnings_growth"],
    "valuation": ["earnings_yield", "book_to_price", "sales_yield"],
    "momentum": ["ret_3m", "ret_6m", "ret_12m", "price_to_ma50", "price_to_ma200"],
    "risk": [],
}
GENERAL_NEGATIVE = {
    "quality": ["debt_to_equity"],
    "growth": [],
    "valuation": ["ev_to_ebit"],
    "momentum": [],
    "risk": ["volatility"],
}

FIN_POSITIVE = {
    "quality": ["roe"],
    "growth": ["earnings_growth"],
    "valuation": ["earnings_yield", "book_to_price"],
    "momentum": ["ret_3m", "ret_6m", "ret_12m", "price_to_ma50", "price_to_ma200"],
    "risk": [],
}
FIN_NEGATIVE = {
    "quality": [],
    "growth": [],
    "valuation": [],
    "momentum": [],
    "risk": ["volatility"],
}


def _is_financial(row):
    txt = f"{row.get('sector','')} {row.get('industry','')}".lower()
    keys = [
        "finance", "financial", "banco", "bank",
        "segur", "insurance", "credito", "credit"
    ]
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
            if f not in sub.columns:
                continue
            c = f"__{component}_{f}"
            sub[c] = _group_percentile(sub, f, True, min_sector_size)
            cols.append(c)

        for f in neg_map[component]:
            if f not in sub.columns:
                continue
            c = f"__{component}_{f}"
            sub[c] = _group_percentile(sub, f, False, min_sector_size)
            cols.append(c)

        if component == "risk" and "max_drawdown" in sub.columns:
            c = "__risk_max_drawdown"
            sub[c] = _group_percentile(sub, "max_drawdown", True, min_sector_size)
            cols.append(c)

        sub[f"{component}_score"] = (
            sub[cols].mean(axis=1, skipna=True)
            if cols else np.nan
        )
        sub[f"{component}_coverage"] = (
            sub[cols].notna().mean(axis=1)
            if cols else 0.0
        )

    for c in sub.columns:
        if c not in out.columns:
            out[c] = np.nan

    out.loc[sub.index, sub.columns] = sub
    return out


def score_dataframe(df, weights, min_coverage=0.65, min_sector_size=5):
    out = df.copy()

    out["model_group"] = out.apply(
        lambda r: "financial" if _is_financial(r) else "general",
        axis=1,
    )

    out = _score_subset(
        out,
        out["model_group"].eq("general"),
        GENERAL_POSITIVE,
        GENERAL_NEGATIVE,
        min_sector_size,
    )

    out = _score_subset(
        out,
        out["model_group"].eq("financial"),
        FIN_POSITIVE,
        FIN_NEGATIVE,
        min_sector_size,
    )

    weighted_cols = []

    for component, w in weights.items():
        c = f"__weighted_{component}"
        out[c] = out[f"{component}_score"] * float(w)
        weighted_cols.append(c)

    numerator = out[weighted_cols].sum(axis=1, min_count=1)
    available_weight = pd.Series(0.0, index=out.index)

    for component, w in weights.items():
        available_weight += (
            out[f"{component}_score"].notna().astype(float) * float(w)
        )

    out["score"] = numerator / available_weight.replace(0, np.nan)

    coverage_cols = [f"{x}_coverage" for x in weights]
    out["data_coverage"] = out[coverage_cols].mean(axis=1)

    # Além da cobertura média, exige pilares mínimos.
    out["has_quality"] = out["quality_score"].notna()
    out["has_valuation"] = out["valuation_score"].notna()
    out["has_momentum"] = out["momentum_score"].notna()
    out["n_components"] = sum(
        out[f"{c}_score"].notna().astype(int)
        for c in ["quality", "growth", "valuation", "momentum", "risk"]
    )

    weak_data = (
        (out["data_coverage"] < min_coverage)
        | (out["n_components"] < 3)
        | (~out["has_quality"])
        | (~out["has_valuation"])
        | (~out["has_momentum"])
    )

    # Ativo com dados insuficientes nunca vira compra forte.
    out.loc[weak_data, "score"] = np.minimum(
        out.loc[weak_data, "score"],
        69.99,
    )

    out["data_quality"] = np.where(
        weak_data,
        "INSUFFICIENT",
        "OK",
    )

    return out
