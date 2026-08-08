from pathlib import Path
import sys

required = [
    "config.yaml",
    "requirements.txt",
    "streamlit_app.py",
    "portfolio.csv",
    "refresh_fundamentals.py",
    "refresh_ranking.py",
    "intraday_watch.py",
    "first_setup_test.py",
    ".github/workflows/first_setup_test.yml",
    ".github/workflows/weekly_fundamentals.yml",
    ".github/workflows/daily.yml",
    ".github/workflows/intraday.yml",
    "src/brapi_client.py",
    "src/features.py",
    "src/scoring.py",
    "src/price_provider.py",
]
missing = [x for x in required if not Path(x).exists()]
if missing:
    print("FALTANDO:", *missing, sep="\n- ")
    sys.exit(1)
print("OK: estrutura essencial presente.")
