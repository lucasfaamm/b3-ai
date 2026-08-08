# B3 AI Radar — Cloud Ready MVP

Painel privado + ranking quantitativo + carteira + Telegram + automações GitHub.

## Rodar localmente
```bash
python -m venv .venv
# Windows: .\.venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python health_check.py
python refresh_fundamentals.py
python refresh_ranking.py
streamlit run streamlit_app.py
```

## Nuvem
Leia `CLOUD_SETUP_PTBR.md`.

## Estado do projeto
**Pronto operacionalmente para MVP/monitoramento.**
Ainda **não validado como estratégia de investimento** até completar backtest point-in-time, corporate actions, survivorship bias, custos/impostos e teste fora da amostra.

## PRIMEIRO TESTE NO GITHUB

Depois do upload e de criar o secret `BRAPI_TOKEN`:

1. Abra **Actions**.
2. Rode **00 - First setup test**.
3. Se ficar verde, rode **Weekly fundamentals refresh**.
4. Depois confira `results/ranking.csv`.

O Telegram é opcional nesta etapa.
