from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import pandas as pd
import yaml

from src.brapi_client import BrapiClient
from src.features import build_fundamental_row

with open("config.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

pause = float(cfg["runtime"].get("request_pause_seconds", 0.15))
workers = int(cfg["runtime"].get("fundamental_workers", 4))
limit = int(cfg["runtime"].get("max_fundamental_tickers", 300))

universe_client = BrapiClient(pause=pause)
items = universe_client.tickers(limit=2000)


def get_symbol(item):
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        return item.get("symbol") or item.get("ticker") or item.get("stock")
    return None


def is_stock(item):
    if isinstance(item, str):
        return True

    txt = " ".join(
        str(item.get(k, ""))
        for k in ("type", "assetType", "kind", "category", "subType")
    ).lower()

    return not any(
        x in txt
        for x in ("fii", "etf", "bdr", "fund", "option", "future", "index")
    )


symbols = []

for item in items:
    s = get_symbol(item)

    if (
        s
        and is_stock(item)
        and (not isinstance(item, dict) or item.get("isActive", True))
        and s.upper() not in symbols
    ):
        symbols.append(s.upper())

symbols = symbols[:limit]

if not symbols:
    raise SystemExit(
        "Nenhum ticker retornado pela brapi. Verifique BRAPI_TOKEN."
    )


def fetch_one(symbol):
    client = BrapiClient(
        pause=pause,
        max_retries=5
    )

    profile = client.profile(symbol)
    stats = client.statistics(symbol)
    fin = client.financial_data(symbol)

    return build_fundamental_row(
        symbol,
        profile,
        stats,
        fin
    )


rows = []
errors = []
completed = 0

Path("data").mkdir(exist_ok=True)
Path("results").mkdir(exist_ok=True)

with ThreadPoolExecutor(max_workers=workers) as executor:

    futures = {
        executor.submit(fetch_one, symbol): symbol
        for symbol in symbols
    }

    for future in as_completed(futures):

        symbol = futures[future]
        completed += 1

        try:

            rows.append(future.result())

            print(
                f"[{completed}/{len(symbols)}] OK {symbol}",
                flush=True
            )

        except Exception as e:

            errors.append({
                "ticker": symbol,
                "error": str(e)
            })

            print(
                f"[{completed}/{len(symbols)}] ERRO {symbol}: {e}",
                flush=True
            )

        if completed % 25 == 0:

            pd.DataFrame(rows).to_csv(
                "data/fundamentals.csv",
                index=False
            )

            pd.DataFrame(errors).to_csv(
                "results/fundamentals_errors.csv",
                index=False
            )


fund = pd.DataFrame(rows)

if not fund.empty and "ticker" in fund.columns:

    fund = (
        fund
        .sort_values("ticker")
        .drop_duplicates(
            "ticker",
            keep="last"
        )
    )


fund.to_csv(
    "data/fundamentals.csv",
    index=False
)

pd.DataFrame(errors).to_csv(
    "results/fundamentals_errors.csv",
    index=False
)


status = {
    "requested": len(symbols),
    "ok": len(fund),
    "errors": len(errors),
    "workers": workers
}


Path(
    "results/fundamentals_status.json"
).write_text(
    json.dumps(
        status,
        ensure_ascii=False,
        indent=2
    ),
    encoding="utf-8"
)


print(
    f"Fundamentos salvos: {len(fund)} de {len(symbols)}",
    flush=True
)


minimum_ok = max(
    20,
    int(len(symbols) * 0.60)
)

if len(fund) < minimum_ok:

    raise SystemExit(
        f"Poucos fundamentos válidos "
        f"({len(fund)}/{len(symbols)}). "
        "Não é seguro gerar ranking."
    )
