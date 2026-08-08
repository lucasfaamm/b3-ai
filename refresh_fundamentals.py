from pathlib import Path
import json
import pandas as pd
import yaml
from src.brapi_client import BrapiClient
from src.features import build_fundamental_row

with open("config.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

client = BrapiClient(cfg["runtime"]["request_pause_seconds"])
items = client.tickers(limit=2000)

def get_symbol(item):
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        return item.get("symbol") or item.get("ticker") or item.get("stock")

def is_stock(item):
    if isinstance(item, str):
        return True
    txt = " ".join(str(item.get(k, "")) for k in ("type", "assetType", "kind", "category", "subType")).lower()
    if any(x in txt for x in ("fii", "etf", "bdr", "fund", "option", "future", "index")):
        return False
    return True

symbols = []
for item in items:
    s = get_symbol(item)
    if s and is_stock(item) and (not isinstance(item, dict) or item.get("isActive", True)) and s not in symbols:
        symbols.append(s.upper())

limit = int(cfg["runtime"].get("max_fundamental_tickers", 500))
symbols = symbols[:limit]
if not symbols:
    raise SystemExit("Nenhum ticker retornado pela brapi. Verifique BRAPI_TOKEN.")

rows, errors = [], []
for i, s in enumerate(symbols, 1):
    try:
        profile = client.profile(s)
        stats = client.statistics(s)
        fin = client.financial_data(s)
        rows.append(build_fundamental_row(s, profile, stats, fin))
        print(f"[{i}/{len(symbols)}] OK {s}")
    except Exception as e:
        errors.append({"ticker": s, "error": str(e)})
        print(f"[{i}/{len(symbols)}] ERRO {s}: {e}")

Path("data").mkdir(exist_ok=True)
Path("results").mkdir(exist_ok=True)
pd.DataFrame(rows).to_csv("data/fundamentals.csv", index=False)
pd.DataFrame(errors).to_csv("results/fundamentals_errors.csv", index=False)
with open("results/fundamentals_status.json", "w", encoding="utf-8") as f:
    json.dump({"requested": len(symbols), "ok": len(rows), "errors": len(errors)}, f, indent=2)
print(f"Fundamentos salvos: {len(rows)}")
