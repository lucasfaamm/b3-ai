import os
import requests
from dotenv import load_dotenv
load_dotenv()
token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
if not token:
    raise SystemExit("Defina TELEGRAM_BOT_TOKEN no .env primeiro.")
r = requests.get(f"https://api.telegram.org/bot{token}/getUpdates", timeout=30)
r.raise_for_status()
data = r.json()
ids = []
for x in data.get("result", []):
    msg = x.get("message") or x.get("channel_post") or {}
    chat = msg.get("chat") or {}
    if chat.get("id") is not None:
        ids.append((chat.get("id"), chat.get("first_name") or chat.get("title") or ""))
print("Chats encontrados:", ids or "nenhum. Envie /start ao bot e rode novamente.")
