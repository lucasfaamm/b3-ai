
from pathlib import Path
import json
import pandas as pd
import streamlit as st

st.set_page_config(page_title="B3 AI Radar",page_icon="📈",layout="wide")
st.title("📈 B3 AI Radar")
st.caption("Sinais quantitativos validados quando possível. Nenhum score é garantia ou probabilidade de lucro.")

def csv(p):
    p=Path(p)
    if not p.exists() or not p.stat().st_size:return pd.DataFrame()
    try:return pd.read_csv(p,comment="#")
    except:return pd.DataFrame()

final=csv("results/final_signals.csv")
portfolio=csv("portfolio.csv")
status={}
p=Path("results/final_signal_status.json")
if p.exists():
    try:status=json.loads(p.read_text(encoding="utf-8"))
    except:pass

c1,c2,c3,c4=st.columns(4)
c1.metric("Regime",status.get("market_regime","?"))
c2.metric("Score de mercado",status.get("market_regime_score","?"))
c3.metric("Compras fortes",status.get("strong_buys","?"))
c4.metric("Calibração OOS","SIM" if status.get("calibration_validated") else "NÃO")

tabs=st.tabs(["🚨 Sinais finais","🎯 Ranking","💼 Minha carteira","🧪 Validação","ℹ️ Regras"])

with tabs[0]:
    st.subheader("Oportunidades")
    if final.empty:
        st.info("Execute o workflow Final Research + Signals.")
    else:
        x=final[final["final_signal"].isin(["COMPRA_FORTE","AVALIAR_COMPRA","WATCHLIST"])]
        cols=[c for c in ["ticker","company_name","price","calibrated_score","final_confidence_score",
                          "final_signal","quality_score","growth_score","valuation_score","momentum_score",
                          "risk_score","technical_confirm","market_regime","failed_gates"] if c in x.columns]
        st.dataframe(x[cols].head(30),use_container_width=True,hide_index=True)

with tabs[1]:
    if not final.empty:
        cols=[c for c in ["ticker","company_name","sector","price","calibrated_score",
                          "final_confidence_score","final_signal","gates_passed","gates_total",
                          "quality_score","growth_score","valuation_score","momentum_score","risk_score"] if c in final.columns]
        st.dataframe(final[cols].head(250),use_container_width=True,hide_index=True)

with tabs[2]:
    if portfolio.empty:
        st.info("Preencha portfolio.csv.")
    elif final.empty:
        st.dataframe(portfolio,use_container_width=True,hide_index=True)
    else:
        keep=[c for c in ["ticker","price","calibrated_score","final_confidence_score","final_signal","failed_gates"] if c in final.columns]
        x=portfolio.merge(final[keep],on="ticker",how="left")
        if {"quantity","price"}.issubset(x.columns):
            x["current_value_brl"]=pd.to_numeric(x["quantity"],errors="coerce")*pd.to_numeric(x["price"],errors="coerce")
        st.dataframe(x,use_container_width=True,hide_index=True)

with tabs[3]:
    for title,path in [
        ("VectorBT vs IBOV/CDI","results/vectorbt_benchmark.json"),
        ("Pesos calibrados / backtest fundamental PIT","results/calibrated_weights.json"),
        ("CVM PIT","results/cvm_pit_status.json"),
    ]:
        st.markdown(f"### {title}")
        p=Path(path)
        if p.exists():
            st.json(json.loads(p.read_text(encoding="utf-8")))
        else:
            st.info("Ainda não executado.")

with tabs[4]:
    st.markdown("""
### COMPRA_FORTE
Só pode aparecer se a calibração fundamental **fora da amostra** tiver sido aceita.
Além disso, exige convergência de qualidade, valuation, momentum, risco, mercado,
dados e ausência de alerta de notícia. O VectorBT só vira gate obrigatório se
também vencer seus benchmarks fora da amostra.

### Importante
`final_confidence_score` é um **score de evidências**, não probabilidade de lucro.
Se a calibração histórica falhar, o sistema pode mostrar watchlist, mas não
produz `COMPRA_FORTE`.
""")
