import numpy as np
import pandas as pd
from .utils import first_present, safe_div, annualized_vol, max_drawdown, pct_return


def build_fundamental_row(symbol, profile, stats, fin):
    market_cap = first_present(stats, ["marketCap"])
    sector = profile.get("sector") or profile.get("sectorDisp") or profile.get("sectorName") or "Unknown"
    industry = profile.get("industry") or profile.get("industryDisp") or profile.get("industryName") or "Unknown"

    revenue = first_present(fin, ["totalRevenue", "revenue"])
    net_income = first_present(stats, ["netIncomeToCommon"])
    ebitda = first_present(fin, ["ebitda"])
    fcf = first_present(fin, ["freeCashflow", "freeCashFlow"])
    total_debt = first_present(fin, ["totalDebt"])
    cash = first_present(fin, ["totalCash", "cash"])

    return {
        "ticker": symbol,
        "sector": sector,
        "industry": industry,
        "market_cap": market_cap,
        "trailing_eps": first_present(stats, ["trailingEps", "earningsPerShare"]),
        "book_value_per_share": first_present(stats, ["bookValue"]),
        "enterprise_value": first_present(stats, ["enterpriseValue"]),
        "ebitda": ebitda,
        "fcf": fcf,
        "revenue": revenue,
        "roe": first_present(fin, ["returnOnEquity", "roe"]),
        "operating_margin": first_present(fin, ["operatingMargins", "operatingMargin"]),
        "profit_margin": first_present(fin, ["profitMargins", "profitMargin"]),
        "revenue_growth": first_present(fin, ["revenueGrowthAnnual", "revenueGrowth"]),
        "earnings_growth": first_present(fin, ["earningsGrowthAnnual", "earningsGrowth", "earningsQuarterlyGrowth"]),
        "total_debt": total_debt,
        "total_cash": cash,
        "beta": first_present(stats, ["beta"]),
        "raw_pe": first_present(stats, ["trailingPE"]),
        "raw_pb": first_present(stats, ["priceToBook"]),
        "raw_ev_ebitda": first_present(stats, ["enterpriseToEbitda"]),
    }


def add_price_features(fund, price_df):
    x = dict(fund)
    h = price_df.sort_values("date").dropna(subset=["close"]).copy()
    closes = h["close"]
    rets = closes.pct_change()
    price = float(closes.iloc[-1]) if len(closes) else np.nan
    avg_shares = float(h["volume"].dropna().tail(60).mean()) if "volume" in h and h["volume"].notna().any() else np.nan

    eps = x.get("trailing_eps", np.nan)
    bvps = x.get("book_value_per_share", np.nan)
    market_cap = x.get("market_cap", np.nan)
    ebitda = x.get("ebitda", np.nan)
    ev = x.get("enterprise_value", np.nan)
    fcf = x.get("fcf", np.nan)
    revenue = x.get("revenue", np.nan)
    debt = x.get("total_debt", np.nan)
    cash = x.get("total_cash", np.nan)

    # Quando possível, atualiza P/L e P/VP com o preço mais recente.
    pe = safe_div(price, eps) if not pd.isna(eps) and eps > 0 else x.get("raw_pe", np.nan)
    pb = safe_div(price, bvps) if not pd.isna(bvps) and bvps > 0 else x.get("raw_pb", np.nan)
    ev_ebitda = safe_div(ev, ebitda) if not pd.isna(ev) and not pd.isna(ebitda) and ebitda > 0 else x.get("raw_ev_ebitda", np.nan)
    fcf_yield = safe_div(fcf, market_cap)
    fcf_margin = safe_div(fcf, revenue)
    net_debt = (debt - cash) if not pd.isna(debt) and not pd.isna(cash) else np.nan
    debt_to_ebitda = safe_div(net_debt, ebitda) if not pd.isna(ebitda) and ebitda > 0 else np.nan
    ma50 = float(closes.tail(50).mean()) if len(closes) >= 50 else np.nan
    ma200 = float(closes.tail(200).mean()) if len(closes) >= 200 else np.nan

    x.update({
        "price": price,
        "avg_daily_volume_brl": avg_shares * price if not pd.isna(avg_shares) else np.nan,
        "pe": pe,
        "pb": pb,
        "ev_ebitda": ev_ebitda,
        "fcf_yield": fcf_yield,
        "fcf_margin": fcf_margin,
        "debt_to_ebitda": debt_to_ebitda,
        "ret_3m": pct_return(closes, 63),
        "ret_6m": pct_return(closes, 126),
        "ret_12m": pct_return(closes, 252),
        "price_to_ma50": safe_div(price, ma50),
        "price_to_ma200": safe_div(price, ma200),
        "volatility": annualized_vol(rets),
        "max_drawdown": max_drawdown(closes.tail(252)),
        "history_days": int(len(h)),
        "last_price_date": str(h["date"].iloc[-1]) if len(h) else "",
    })
    return x
