import json
from pathlib import Path
import pandas as pd
import yaml
from src.brapi_client import BrapiClient
from src.telegram_notify import send_telegram

with open("config.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
rank_path = Path("results/ranking.csv")
if not rank_path.exists():
    raise SystemExit("Sem ranking diário.")
rank = pd.read_csv(rank_path).sort_values("score", ascending=False)
rank["ticker"] = rank["ticker"].astype(str).str.upper()

held = set()
pp = Path("portfolio.csv")
if pp.exists():
    try:
        p = pd.read_csv(pp, comment="#")
        held = set(p.get("ticker", pd.Series(dtype=str)).dropna().astype(str).str.upper())
    except Exception:
        pass

top_n = int(cfg["runtime"].get("intraday_top_n", 20))
watch = list(rank.head(top_n)["ticker"]) + list(held)
watch = list(dict.fromkeys(watch))
client = BrapiClient(cfg["runtime"]["request_pause_seconds"])

state_path = Path("results/alert_state.json")
try:
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
except Exception:
    state = {}
new_state = dict(state)
messages = []

for ticker in watch:
    try:
        q = client.quote(ticker)
        price = q.get("regularMarketPrice") or q.get("price")
        if price is None:
            continue
        price = float(price)
        rr = rank.loc[rank["ticker"] == ticker]
        if rr.empty:
            continue
        r = rr.iloc[0]
        score = float(r.get("score", 0))
        low, high = r.get("entry_watch_low"), r.get("entry_watch_high")
        event = None
        if ticker in held and score < 60:
            event = f"HELD_LOW_SCORE_{int(score//5)*5}"
            text = (f"⚠️ {ticker} — revisar posição\nPreço: R$ {price:.2f}\n"
                    f"Score: {score:.1f}/100\nAção do modelo: {r.get('action','REAVALIAR')}\n"
                    "Não vender automaticamente; revisar tese, risco e impostos.")
        elif pd.notna(low) and pd.notna(high) and score >= 80 and float(low) <= price <= float(high):
            event = f"ENTRY_{round(float(low),2)}_{round(float(high),2)}"
            text = (f"🟢 {ticker} — entrou na faixa de atenção\nPreço: R$ {price:.2f}\n"
                    f"Score: {score:.1f}/100\nFaixa: R$ {float(low):.2f}–{float(high):.2f}\n"
                    f"Ação: {r.get('action','AVALIAR')}\nRevisar antes de executar.")
        if event and state.get(ticker) != event:
            messages.append(text)
            new_state[ticker] = event
        elif not event and ticker in new_state:
            new_state.pop(ticker, None)
    except Exception as e:
        print(f"[WARN] {ticker}: {e}")

if messages:
    send_telegram("\n\n".join(messages[:8]))
    state_path.write_text(json.dumps(new_state, ensure_ascii=False, indent=2), encoding="utf-8")
else:
    print("Nenhum novo alerta relevante.")
