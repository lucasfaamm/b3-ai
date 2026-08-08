from pathlib import Path
import pandas as pd
from src.telegram_notify import send_telegram

p = Path("results/ranking.csv")
if not p.exists():
    raise SystemExit("Sem ranking.")
df = pd.read_csv(p).sort_values("score", ascending=False)
top = df.head(8)
lines = ["📊 B3 AI — resumo pós-fechamento", ""]
for _, r in top.iterrows():
    lines.append(f"{r['ticker']}: {float(r['score']):.1f}/100 | {r.get('action','')} | R$ {float(r.get('price',0)):.2f}")
send_telegram("\n".join(lines))
