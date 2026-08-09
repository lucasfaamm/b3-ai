
from pathlib import Path
import json
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Nubank Global Opportunity Radar",page_icon="🟣",layout="wide")
st.title("🟣 Nubank Global Opportunity Radar")
st.caption("Ações B3 + ações globais via BDR + ETFs + FIIs + fundos listados. Renda fixa excluída. Score não é probabilidade de lucro.")

def csv(p):
    p=Path(p)
    if not p.exists() or not p.stat().st_size:return pd.DataFrame()
    try:return pd.read_csv(p,comment="#")
    except:return pd.DataFrame()

def js(p):
    p=Path(p)
    if not p.exists():return {}
    try:return json.loads(p.read_text(encoding="utf-8"))
    except:return {}

r=csv("results/nubank_global_ranking.csv")
pf=csv("results/nubank_portfolio_actions.csv")
val=csv("results/nubank_forward_validation_summary.csv")
s=js("results/nubank_radar_status.json")

a,b,c,d=st.columns(4)
a.metric("Regime Brasil",s.get("brasil_regime","?"))
b.metric("Regime Global",s.get("global_regime","?"))
c.metric("Ativos avaliados",s.get("ranked_total","?"))
d.metric("Renda fixa","EXCLUÍDA")

tabs=st.tabs([
    "🚨 Melhores oportunidades","🇧🇷 Ações B3","🌎 Ações globais / BDR",
    "📊 ETFs","🏢 FIIs","💼 Outros fundos","👛 Minha carteira",
    "🧪 Validação","ℹ️ Como interpretar"
])

def show_class(cls,n=100):
    if r.empty:return st.info("Execute NUBANK GLOBAL RADAR.")
    x=r[r.asset_class.eq(cls)].copy()
    cols=[c for c in [
        "ticker","name","asset_class","price_now","entry_watch_low","entry_watch_high",
        "global_opportunity_score","opportunity_score","signal","strengths","risks",
        "momentum_score","risk_score","liquidity_score","income_score",
        "quality_score","growth_score","valuation_score","fundamental_coverage","regime"
    ] if c in x.columns]
    st.dataframe(x[cols].head(n),use_container_width=True,hide_index=True)

with tabs[0]:
    if r.empty:st.info("Execute NUBANK GLOBAL RADAR.")
    else:
        x=r[r.signal.isin(["AVALIAR_COMPRA","COMPRA_FORTE","AVALIAR_COMPRA_US"])].copy()
        if x.empty:x=r.head(40)
        cols=[c for c in ["ticker","name","asset_class","price_now","entry_watch_low","entry_watch_high","global_opportunity_score","signal","strengths","risks","regime"] if c in x]
        st.dataframe(x[cols].head(50),use_container_width=True,hide_index=True)

with tabs[1]:show_class("AÇÃO B3",140)
with tabs[2]:show_class("AÇÃO GLOBAL / BDR",140)
with tabs[3]:show_class("ETF",140)
with tabs[4]:show_class("FII",180)

with tabs[5]:
    if r.empty:st.info("Sem dados.")
    else:
        x=r[r.asset_class.isin(["FI-INFRA","FIAGRO","FIP","FIDC"])]
        st.dataframe(x.head(150),use_container_width=True,hide_index=True)
        st.caption("Fundos de prateleira que não são negociados na B3 exigem a lista/CNPJ disponibilizada no seu app. O radar já cobre automaticamente os fundos listados negociáveis pela conta de investimentos.")

with tabs[6]:
    if pf.empty:st.info("Quando portfolio.csv estiver preenchido, esta aba recomenda MANTER, AUMENTAR, NÃO AUMENTAR ou REAVALIAR.")
    else:st.dataframe(pf,use_container_width=True,hide_index=True)

with tabs[7]:
    st.caption("O radar registra mensalmente as 10 melhores de cada classe e mede o retorno 20 pregões depois contra um benchmark da própria classe.")
    st.dataframe(val,use_container_width=True,hide_index=True)

with tabs[8]:
    st.markdown("""
### O que o score significa
Cada classe é avaliada com métricas próprias. FIIs não são julgados como ações, e ETFs não são julgados como empresas.

**Ações B3:** usa o modelo fundamental/point-in-time que já construímos.

**BDRs:** combina momentum, risco, liquidez e tendência; nas BDRs mais líquidas tenta acrescentar fundamentos globais como ROE, margens, crescimento, dívida e valuation.

**ETFs:** momentum, força relativa, volatilidade, drawdown, liquidez e tendência.

**FIIs:** renda/distribuições, consistência, momentum, risco, liquidez e tendência.

**FI-Infra / FIAGRO / FIP / FIDC:** renda, mercado, risco, liquidez e tendência.

### Sinais
- **AVALIAR_COMPRA**: merece due diligence/entrada gradual; não é ordem automática.
- **WATCHLIST**: interessante, mas ainda não convergiu.
- **AGUARDAR**: sem vantagem suficiente.
- **EVITAR_AGORA**: score relativo muito fraco.

A faixa `entry_watch_low/high` é estatística e baseada em volatilidade; não é preço justo.

O sistema não executa ordens automaticamente.
""")
