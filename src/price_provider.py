import time
import pandas as pd
import yfinance as yf


def _chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i+n]


def download_prices(symbols, period="1y", batch_size=40):
    """Baixa preços ajustados via Yahoo Finance/yfinance.
    Retorna dict ticker B3 -> DataFrame(date, close, volume).
    Esta fonte é prática e gratuita, mas não é fonte oficial da B3.
    """
    out = {}
    symbols = [str(s).upper().strip() for s in symbols if str(s).strip()]
    for batch in _chunks(symbols, batch_size):
        yahoo = [f"{s}.SA" for s in batch]
        try:
            data = yf.download(
                yahoo,
                period=period,
                interval="1d",
                auto_adjust=True,
                group_by="ticker",
                threads=True,
                progress=False,
            )
        except Exception as e:
            print(f"[WARN] yfinance batch: {e}")
            continue

        for s, ys in zip(batch, yahoo):
            try:
                if len(batch) == 1:
                    d = data.copy()
                else:
                    d = data[ys].copy()
                if d.empty or "Close" not in d.columns:
                    continue
                df = pd.DataFrame({
                    "date": pd.to_datetime(d.index, utc=True),
                    "close": pd.to_numeric(d["Close"], errors="coerce"),
                    "volume": pd.to_numeric(d.get("Volume"), errors="coerce"),
                }).dropna(subset=["close"])
                if not df.empty:
                    out[s] = df.reset_index(drop=True)
            except Exception:
                continue
        time.sleep(0.3)
    return out
