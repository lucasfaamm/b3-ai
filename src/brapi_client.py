import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()
BASE = "https://brapi.dev/api/v2"


class BrapiClient:
    def __init__(self, pause=0.15, max_retries=4):
        self.token = os.getenv("BRAPI_TOKEN", "").strip()
        self.pause = pause
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "b3-ai-personal-radar/1.0"})
        if self.token and self.token != "SEU_TOKEN_AQUI":
            self.session.headers.update({"Authorization": f"Bearer {self.token}"})

    def _get(self, path, params=None):
        url = BASE + path
        last = None
        for attempt in range(self.max_retries):
            try:
                r = self.session.get(url, params=params or {}, timeout=45)
                if r.status_code == 429 or 500 <= r.status_code < 600:
                    wait = min(20, 2 ** attempt)
                    print(f"[WARN] HTTP {r.status_code}; nova tentativa em {wait}s")
                    time.sleep(wait)
                    last = RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
                    continue
                r.raise_for_status()
                time.sleep(self.pause)
                return r.json()
            except requests.RequestException as e:
                last = e
                if attempt == self.max_retries - 1:
                    break
                time.sleep(min(20, 2 ** attempt))
        raise last or RuntimeError("Falha desconhecida na brapi")

    @staticmethod
    def _data(payload):
        results = payload.get("results", []) if isinstance(payload, dict) else []
        if not results:
            return {}
        return results[0].get("data", {}) or {}

    def tickers(self, limit=2000):
        payload = self._get(
            "/tickers",
            {
                "type": "stock",
                "sortBy": "volume",
                "sortOrder": "desc",
                "limit": int(limit),
                "page": 1,
            },
        )
        if isinstance(payload, dict):
            for key in ("results", "stocks", "tickers", "data"):
                v = payload.get(key)
                if isinstance(v, list):
                    return v
        return []

    def quote(self, symbol):
        return self._data(self._get("/stocks/quote", {"symbols": symbol}))

    def profile(self, symbol):
        return self._data(self._get("/stocks/profile", {"symbols": symbol}))

    def statistics(self, symbol):
        return self._data(self._get("/stocks/statistics", {"symbols": symbol, "mode": "current"}))

    def financial_data(self, symbol):
        return self._data(self._get("/stocks/financial-data", {"symbols": symbol, "mode": "current"}))
