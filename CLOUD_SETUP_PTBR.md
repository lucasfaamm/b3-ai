# B3 AI — implantação curta

## 1) GitHub
Crie um repositório **privado** `b3-ai` e envie TODO o conteúdo desta pasta, inclusive `.github/` e `.gitignore`.

Secrets em `Settings > Secrets and variables > Actions`:
- `BRAPI_TOKEN`
- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID`

## 2) Telegram
- Converse com `@BotFather` -> `/newbot`.
- Copie o token.
- Envie `/start` para seu bot.
- Localmente, coloque o token em `.env` e rode `python telegram_get_chat_id.py` para descobrir o chat id.

## 3) Primeiro processamento
No GitHub: `Actions > Weekly fundamentals refresh > Run workflow`.
Quando terminar, confira se apareceram:
- `data/fundamentals.csv`
- `results/ranking.csv`
- `results/status.json`

Depois rode manualmente `Daily B3 ranking` e `Intraday B3 watch` uma vez para validar.

## 4) Streamlit
Entre no Streamlit Community Cloud com GitHub, selecione o repo privado e use `streamlit_app.py` como entrypoint.
Salve a URL do app no celular.

## 5) Sua carteira
Edite `portfolio.csv` no GitHub:
`ticker,quantity,avg_price_brl,broker,notes`

Exemplo:
`ITUB4,10,35.20,Nubank,`

## Rotina automática
- Domingo: fundamentos.
- Dias úteis após fechamento: ranking completo e resumo Telegram.
- Dias úteis 10h–18h: carteira + top oportunidades, de hora em hora.

## Limitação importante
O plano gratuito da brapi tem limite mensal e histórico curto. Por isso o projeto usa brapi para fundamentos/cotação intraday e yfinance para histórico ajustado de preços. O backtest fundamentalista point-in-time com CVM oficial é a próxima etapa de validação.
