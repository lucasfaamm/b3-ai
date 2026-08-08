from pathlib import Path
import json
import pandas as pd
import streamlit as st

st.set_page_config(page_title="B3 AI Radar", page_icon="📈", layout="wide")
st.title("📈 B3 AI Radar")
st.caption("Radar quantitativo experimental. Sinais precisam de validação/backtest; não há garantia de retorno.")

rank_path = Path("results/ranking.csv")
port_path = Path("portfolio.csv")
status_path = Path("results/status.json")

def read_csv(path):
    try:
        return pd.read_csv(path, comment="#") if path.exists() and path.stat().st_size else pd.DataFrame()
    except Exception:
        return pd.DataFrame()

rank = read_csv(rank_path)
port = read_csv(port_path)
if status_path.exists():
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
        st.caption(f"Última atualização: {status.get('updated_at_utc','')} | Ativos investíveis: {status.get('investible','-')}")
    except Exception:
        pass

op, pf, rk, about = st.tabs(["🎯 Oportunidades", "💼 Minha carteira", "📊 Ranking", "ℹ️ Regras"])

with op:
    if rank.empty:
        st.info("Ranking ainda não gerado.")
    else:
        x = rank[rank["score"] >= 80].copy()
        cols = [c for c in ["ticker","score","action","price","entry_watch_low","entry_watch_high","quality_score","growth_score","valuation_score","momentum_score","risk_score","data_coverage","in_portfolio"] if c in x.columns]
        st.dataframe(x[cols].head(40), use_container_width=True, hide_index=True)

with pf:
    if port.empty:
        st.info("Preencha portfolio.csv com suas posições.")
    else:
        port["ticker"] = port["ticker"].astype(str).str.upper()
        if rank.empty:
            st.dataframe(port, use_container_width=True, hide_index=True)
        else:
            cols = [c for c in ["ticker","score","action","price","signal"] if c in rank.columns]
            m = port.merge(rank[cols], on="ticker", how="left")
            if "quantity" in m.columns and "price" in m.columns:
                m["current_value_brl"] = pd.to_numeric(m["quantity"], errors="coerce") * pd.to_numeric(m["price"], errors="coerce")
            if "avg_price_brl" in m.columns and "price" in m.columns:
                avg = pd.to_numeric(m["avg_price_brl"], errors="coerce")
                cur = pd.to_numeric(m["price"], errors="coerce")
                m["return_pct"] = (cur / avg - 1) * 100
            st.dataframe(m, use_container_width=True, hide_index=True)

with rk:
    if not rank.empty:
        st.dataframe(rank.head(300), use_container_width=True, hide_index=True)

with about:
    st.markdown("""
**Score inicial:** Quality 25% + Growth 20% + Valuation 20% + Momentum 20% + Risk 15%.

**Ações do painel:**
- `AVALIAR_COMPRA`: score muito alto; revisar antes de executar.
- `AGUARDAR_FAIXA_DE_ENTRADA`: candidata forte, mas sem sinal para comprar a qualquer preço.
- `MANTER`: posição existente ainda saudável no modelo.
- `REAVALIAR`: score deteriorou.
- `REDUZIR_OU_SAIR_APOS_REVISAO`: alerta de risco, nunca venda automática.

A faixa de entrada atual é **estatística**, não um preço-alvo fundamentalista. O preço-alvo de valuation e os limites definitivos virão após o backtest point-in-time.
""")
