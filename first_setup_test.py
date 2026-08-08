import os
import sys
from src.brapi_client import BrapiClient

if not os.getenv("BRAPI_TOKEN", "").strip():
    raise SystemExit("ERRO: BRAPI_TOKEN não foi encontrado nos GitHub Secrets.")

client = BrapiClient(pause=0.05)
items = client.tickers(limit=5)
if not items:
    raise SystemExit("ERRO: a brapi não retornou tickers. Confira o token.")

q = client.quote("PETR4")
price = q.get("regularMarketPrice") or q.get("price")
if price is None:
    raise SystemExit("ERRO: teste de cotação PETR4 não retornou preço.")

print("OK: BRAPI_TOKEN reconhecido.")
print(f"OK: endpoint de tickers respondeu ({len(items)} itens no teste).")
print(f"OK: PETR4 retornou preço: R$ {float(price):.2f}")
print("SETUP INICIAL APROVADO.")
