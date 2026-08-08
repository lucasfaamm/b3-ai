from pathlib import Path
from datetime import datetime, timezone
import json
import numpy as np
import pandas as pd
import yaml
from src.price_provider import download_prices
from src.features import add_price_features
from src.scoring import score_dataframe

with open("config.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

fund_path = Path("data/fundamentals.csv")
if not fund_path.exists():
    raise SystemExit("Falta data/fundamentals.csv. Rode refresh_fundamentals.py primeiro.")
fund = pd.read_csv(fund_path)
symbols = fund["ticker"].dropna().astype(str).str.upper().tolist()
prices = download_prices(symbols, cfg["runtime"]["price_period"], cfg["runtime"]["price_batch_size"])
rows = []
for _, r in fund.iterrows():
    s = str(r["ticker"]).upper()
    if s in prices:
        rows.append(add_price_features(r.to_dict(), prices[s]))

features = pd.DataFrame(rows)
if features.empty:
    raise SystemExit("Não foi possível obter histórico de preços.")

u = cfg["universe"]
inv = features[
    (features["price"].fillna(0) >= u["min_price_brl"]) &
    (features["market_cap"].fillna(0) >= u["min_market_cap_brl"]) &
    (features["avg_daily_volume_brl"].fillna(0) >= u["min_avg_daily_volume_brl"]) &
    (features["history_days"].fillna(0) >= u["min_history_days"])
].copy()

scored = score_dataframe(
    inv,
    cfg["score"]["weights"],
    cfg["score"]["min_data_coverage"],
    cfg["score"].get("min_sector_size", 5),
).sort_values("score", ascending=False)

s = cfg["signals"]
def label(x):
    if x >= s["exceptional"]: return "OPORTUNIDADE_EXCEPCIONAL"
    if x >= s["strong"]: return "FORTE_CANDIDATA"
    if x >= s["watch"]: return "WATCHLIST"
    if x >= s["neutral"]: return "NEUTRO"
    return "NAO_PRIORIZAR"
scored["signal"] = scored["score"].map(label)

# Faixa estatística de alerta, NÃO valuation/preço-alvo.
monthly_vol = scored["volatility"].clip(lower=0, upper=1.5) / np.sqrt(12)
scored["entry_watch_low"] = scored["price"] * (1 - 0.35 * monthly_vol)
scored["entry_watch_high"] = scored["price"]
scored["max_position_pct"] = s["max_position_pct_default"]

portfolio_path = Path("portfolio.csv")
held = set()
if portfolio_path.exists():
    try:
        p = pd.read_csv(portfolio_path, comment="#")
        held = set(p.get("ticker", pd.Series(dtype=str)).dropna().astype(str).str.upper())
    except Exception:
        held = set()
scored["in_portfolio"] = scored["ticker"].isin(held)

def action(row):
    score = float(row["score"])
    held_now = bool(row["in_portfolio"])
    if held_now:
        if score >= 85: return "MANTER_OU_AVALIAR_COMPRAR_MAIS"
        if score >= 70: return "MANTER"
        if score >= 60: return "REAVALIAR"
        return "REDUZIR_OU_SAIR_APOS_REVISAO"
    if score >= 85: return "AVALIAR_COMPRA"
    if score >= 80: return "AGUARDAR_FAIXA_DE_ENTRADA"
    return "AGUARDAR"
scored["action"] = scored.apply(action, axis=1)

Path("results").mkdir(exist_ok=True)
features.to_csv("results/latest_features.csv", index=False)
scored.to_csv("results/ranking.csv", index=False)
scored[scored["score"] >= s["strong"]].to_csv("results/opportunities.csv", index=False)

status = {
    "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    "fundamentals": int(len(fund)),
    "price_series_ok": int(len(features)),
    "investible": int(len(scored)),
    "strong_or_better": int((scored["score"] >= s["strong"]).sum()),
}
Path("results/status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
print(scored[["ticker","score","signal","action","price","data_coverage"]].head(20).to_string(index=False))
