
from pathlib import Path
import json
import pandas as pd
import streamlit as st

st.set_page_config(page_title="B3 AI Radar", page_icon="📈", layout="wide")

st.title("📈 B3 AI Radar")
st.caption(
    "Radar quantitativo experimental. "
    "Sinais precisam de validação/backtest; não há garantia de retorno."
)

RANKING = Path("results/ranking.csv")
PORTFOLIO = Path("portfolio.csv")
STATUS = Path("results/status.json")


def read_csv(path):
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, comment="#")
    except Exception:
        return pd.DataFrame()


ranking = read_csv(RANKING)
portfolio = read_csv(PORTFOLIO)

if STATUS.exists():
    try:
        status = json.loads(STATUS.read_text(encoding="utf-8"))
        st.caption(
            f"Última atualização: {status.get('updated_at','?')} | "
            f"Ativos investíveis: {status.get('investible', len(ranking))}"
        )
    except Exception:
        pass


tabs = st.tabs([
    "🎯 Oportunidades",
    "💼 Minha carteira",
    "📊 Ranking",
    "ℹ️ Regras",
])


with tabs[0]:

    st.subheader("Melhores candidatas atuais")

    if ranking.empty:
        st.info("Ranking ainda não foi gerado.")
    else:
        x = ranking.copy()

        if "data_quality" in x.columns:
            x = x[x["data_quality"] == "OK"]

        x = x.sort_values("score", ascending=False)

        top = x.head(15)

        cols = [
            "ticker", "company_name", "score", "action", "price",
            "quality_score", "growth_score", "valuation_score",
            "momentum_score", "risk_score", "data_coverage"
        ]
        cols = [c for c in cols if c in top.columns]

        st.dataframe(
            top[cols],
            use_container_width=True,
            hide_index=True,
        )

        strong = x[x["score"] >= 80]

        st.markdown("### Oportunidades confirmadas pelo corte atual")

        if strong.empty:
            st.info(
                "Nenhum ativo com score ≥ 80 e qualidade de dados suficiente. "
                "Isso não é erro: o modelo não está forçando uma compra."
            )
        else:
            st.dataframe(
                strong[cols].head(30),
                use_container_width=True,
                hide_index=True,
            )


with tabs[1]:

    st.subheader("Minha carteira")

    if portfolio.empty:
        st.info(
            "A carteira ainda não foi preenchida. "
            "Depois vamos importar posições, quantidade e preço médio."
        )
    else:
        st.dataframe(
            portfolio,
            use_container_width=True,
            hide_index=True,
        )

        if not ranking.empty and "ticker" in portfolio.columns:
            keep = [
                c for c in [
                    "ticker", "score", "action", "price",
                    "quality_score", "growth_score",
                    "valuation_score", "momentum_score",
                    "risk_score", "data_quality"
                ]
                if c in ranking.columns
            ]

            merged = portfolio.merge(
                ranking[keep],
                on="ticker",
                how="left",
            )

            if {"quantity", "price"}.issubset(merged.columns):
                merged["current_value_brl"] = (
                    pd.to_numeric(merged["quantity"], errors="coerce")
                    * pd.to_numeric(merged["price"], errors="coerce")
                )

            st.markdown("### Diagnóstico das posições")

            st.dataframe(
                merged,
                use_container_width=True,
                hide_index=True,
            )


with tabs[2]:

    st.subheader("Ranking B3")

    if ranking.empty:
        st.info("Ranking ainda não gerado.")
    else:
        x = ranking.sort_values("score", ascending=False).copy()

        important = [
            "ticker",
            "company_name",
            "sector",
            "industry",
            "score",
            "action",
            "price",
            "quality_score",
            "growth_score",
            "valuation_score",
            "momentum_score",
            "risk_score",
            "data_coverage",
            "data_quality",
            "market_cap",
        ]
        important = [c for c in important if c in x.columns]

        st.dataframe(
            x[important].head(200),
            use_container_width=True,
            hide_index=True,
        )

        with st.expander("Ver dados fundamentais brutos"):
            st.dataframe(
                x.head(200),
                use_container_width=True,
                hide_index=True,
            )


with tabs[3]:

    st.markdown("""
### Interpretação provisória

- **85–100:** oportunidade excepcional para revisão
- **80–84,99:** forte candidata
- **70–79,99:** watchlist
- **60–69,99:** neutro
- **<60:** não priorizar

### Proteção de qualidade

Um ativo não pode receber sinal forte se estiver faltando:
- qualidade;
- valuation;
- momentum;
- cobertura mínima dos dados.

### Importante

Os cortes e pesos ainda serão calibrados por backtest walk-forward.
O sistema não executa ordens automaticamente.
""")
