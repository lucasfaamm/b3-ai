
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import json
import math
import time

import numpy as np
import pandas as pd
import yfinance as yf
import yaml

from src.brapi_client import BrapiClient


def fnum(x):
    try:
        if x is None:
            return np.nan
        x = float(x)
        return x if math.isfinite(x) else np.nan
    except Exception:
        return np.nan


def safe_div(a, b):
    a = fnum(a)
    b = fnum(b)
    if pd.isna(a) or pd.isna(b) or b == 0:
        return np.nan
    return a / b


def first(info, *keys):
    for k in keys:
        if k in info and info[k] not in (None, ""):
            v = fnum(info[k])
            if not pd.isna(v):
                return v
    return np.nan


def fetch_yahoo(row):
    symbol = row["ticker"]
    ys = f"{symbol}.SA"

    last_error = None

    for attempt in range(3):
        try:
            t = yf.Ticker(ys)
            info = t.get_info() or {}

            market_cap = first(info, "marketCap")
            if pd.isna(market_cap):
                market_cap = fnum(row.get("market_cap"))

            revenue = first(info, "totalRevenue")
            operating_income = first(info, "operatingIncome", "ebitda")
            net_income = first(info, "netIncomeToCommon", "netIncome")
            total_cash = first(info, "totalCash")
            total_debt = first(info, "totalDebt")
            enterprise_value = first(info, "enterpriseValue")
            ebitda = first(info, "ebitda")
            fcf = first(info, "freeCashflow")

            trailing_pe = first(info, "trailingPE")
            price_to_book = first(info, "priceToBook")
            ev_to_ebitda = first(info, "enterpriseToEbitda")

            roe = first(info, "returnOnEquity")
            operating_margin = first(info, "operatingMargins")
            profit_margin = first(info, "profitMargins")
            revenue_growth = first(info, "revenueGrowth")
            earnings_growth = first(info, "earningsGrowth", "earningsQuarterlyGrowth")

            trailing_eps = first(info, "trailingEps")
            book_value_per_share = first(info, "bookValue")
            shares = first(info, "sharesOutstanding")

            equity = np.nan
            if not pd.isna(book_value_per_share) and not pd.isna(shares):
                equity = book_value_per_share * shares
            elif not pd.isna(market_cap) and not pd.isna(price_to_book) and price_to_book > 0:
                equity = market_cap / price_to_book

            if pd.isna(roe):
                roe = safe_div(net_income, equity)

            if pd.isna(operating_margin):
                operating_margin = safe_div(operating_income, revenue)

            if pd.isna(profit_margin):
                profit_margin = safe_div(net_income, revenue)

            debt_to_equity = safe_div(total_debt, equity)

            earnings_yield = (
                safe_div(1.0, trailing_pe)
                if not pd.isna(trailing_pe) and trailing_pe > 0
                else safe_div(net_income, market_cap)
            )

            book_to_price = (
                safe_div(1.0, price_to_book)
                if not pd.isna(price_to_book) and price_to_book > 0
                else safe_div(equity, market_cap)
            )

            sales_yield = safe_div(revenue, market_cap)

            ev_to_ebit = np.nan
            if not pd.isna(enterprise_value) and not pd.isna(operating_income) and operating_income > 0:
                ev_to_ebit = enterprise_value / operating_income

            sector = (
                info.get("sector")
                or row.get("sector_catalog")
                or "Unknown"
            )

            industry = (
                info.get("industry")
                or row.get("subsector_catalog")
                or "Unknown"
            )

            # Exige algum conteúdo fundamental de verdade.
            fundamental_values = [
                market_cap, revenue, net_income, roe,
                trailing_pe, price_to_book,
            ]
            if sum(not pd.isna(v) for v in fundamental_values) < 3:
                raise RuntimeError("dados fundamentais insuficientes no Yahoo")

            return {
                "ticker": symbol,
                "cnpj": "",
                "company_name": info.get("longName") or row.get("name") or symbol,
                "sector": sector,
                "industry": industry,
                "market_cap": market_cap,

                "revenue": revenue,
                "operating_income": operating_income,
                "net_income": net_income,
                "equity": equity,
                "total_cash": total_cash,
                "total_debt": total_debt,

                "roe": roe,
                "operating_margin": operating_margin,
                "profit_margin": profit_margin,
                "debt_to_equity": debt_to_equity,

                "revenue_growth": revenue_growth,
                "earnings_growth": earnings_growth,

                "earnings_yield": earnings_yield,
                "book_to_price": book_to_price,
                "sales_yield": sales_yield,
                "ev_to_ebit": ev_to_ebit,

                "raw_pe": trailing_pe,
                "raw_pb": price_to_book,
                "raw_ev_ebitda": ev_to_ebitda,

                "enterprise_value": enterprise_value,
                "ebitda": ebitda,
                "fcf": fcf,
                "beta": first(info, "beta"),
                "trailing_eps": trailing_eps,
                "book_value_per_share": book_value_per_share,

                "fundamental_source": "Yahoo Finance",
            }

        except Exception as e:
            last_error = e
            time.sleep(2 ** attempt)

    raise RuntimeError(str(last_error))


with open("config.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

configured_limit = int(cfg["runtime"].get("max_fundamental_tickers", 300))
# Mantemos a primeira versão estável em até 200 ações mais líquidas.
# Isso reduz rate-limit do Yahoo e ainda cobre o universo líquido relevante.
limit = min(configured_limit, 200)
workers = min(int(cfg["runtime"].get("fundamental_workers", 4)), 4)

client = BrapiClient(
    pause=float(cfg["runtime"].get("request_pause_seconds", 0.10)),
    max_retries=4,
)

items = client.tickers(limit=2000)

universe = []

for item in items:
    if not isinstance(item, dict):
        continue

    symbol = str(item.get("symbol", "")).upper().strip()

    if not symbol:
        continue

    if not item.get("isActive", True):
        continue

    if str(item.get("assetType", "")).lower() != "stock":
        continue

    subtype = str(item.get("subType", "")).lower()
    if subtype not in ("", "stock", "unit"):
        continue

    q = item.get("quote") or {}

    universe.append({
        "ticker": symbol,
        "name": item.get("name") or item.get("longName") or symbol,
        "sector_catalog": item.get("sector") or "Unknown",
        "subsector_catalog": item.get("subsector") or "Unknown",
        "market_cap": fnum(q.get("marketCap")),
        "catalog_volume": fnum(q.get("volume")),
    })

universe = universe[:limit]

if not universe:
    raise SystemExit("Não foi possível obter o universo B3 pela brapi.")

print(f"[INFO] universo Yahoo: {len(universe)} ações", flush=True)

rows = []
errors = []

with ThreadPoolExecutor(max_workers=workers) as ex:
    futures = {
        ex.submit(fetch_yahoo, row): row["ticker"]
        for row in universe
    }

    for i, fut in enumerate(as_completed(futures), start=1):
        ticker = futures[fut]

        try:
            rows.append(fut.result())
            print(f"[{i}/{len(universe)}] OK {ticker}", flush=True)
        except Exception as e:
            errors.append({
                "ticker": ticker,
                "error": str(e),
            })
            print(f"[{i}/{len(universe)}] ERRO {ticker}: {e}", flush=True)

fund = pd.DataFrame(rows)

if not fund.empty:
    fund = (
        fund
        .sort_values("ticker")
        .drop_duplicates("ticker", keep="last")
    )

Path("data").mkdir(exist_ok=True)
Path("results").mkdir(exist_ok=True)

# Só substitui a base oficial do projeto se a coleta nova for suficientemente boa.
minimum_ok = max(50, int(len(universe) * 0.45))

if len(fund) < minimum_ok:
    pd.DataFrame(errors).to_csv(
        "results/fundamentals_yahoo_errors.csv",
        index=False,
    )

    raise SystemExit(
        f"Yahoo retornou poucos fundamentos válidos "
        f"({len(fund)}/{len(universe)}). "
        "A base anterior foi preservada."
    )

tmp = Path("data/fundamentals.new.csv")
fund.to_csv(tmp, index=False)
tmp.replace("data/fundamentals.csv")

pd.DataFrame(errors).to_csv(
    "results/fundamentals_yahoo_errors.csv",
    index=False,
)

status = {
    "universe_requested": len(universe),
    "fundamentals_ok": len(fund),
    "errors": len(errors),
    "fundamental_source": "Yahoo Finance",
    "universe_source": "brapi public tickers",
    "cvm_status": "not_used_in_github_runner",
}

Path("results/fundamentals_status.json").write_text(
    json.dumps(status, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print(
    f"[OK] fundamentos Yahoo: {len(fund)}/{len(universe)}",
    flush=True,
)
