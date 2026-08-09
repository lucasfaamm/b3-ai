from pathlib import Path
import json
import pandas as pd
import streamlit as st
st.set_page_config(page_title='B3 AI Radar',page_icon='📈',layout='wide')
st.title('📈 B3 AI Radar')
st.caption('Modelo final congelado: qualidade 25%, crescimento 20%, valuation 20%, momentum 20%, risco 15%. Qlib e VectorBT têm peso 0.')
def csv(p):
    p=Path(p)
    if not p.exists() or not p.stat().st_size:return pd.DataFrame()
    try:return pd.read_csv(p,comment='#')
    except:return pd.DataFrame()
def js(p):
    p=Path(p)
    if not p.exists():return {}
    try:return json.loads(p.read_text(encoding='utf-8'))
    except:return {}
final=csv('results/final_signals.csv'); portfolio=csv('portfolio.csv'); status=js('results/final_signal_status.json'); evidence=js('results/fixed_model_evidence.json'); forward=js('results/forward_validation_summary.json')
a,b,c,d=st.columns(4); a.metric('Regime',status.get('market_regime','?')); b.metric('Score mercado',status.get('market_regime_score','?')); c.metric('Compra forte',status.get('strong_buys','?')); d.metric('Meses forward',status.get('forward_months_evaluated',0))
tabs=st.tabs(['🚨 Oportunidades','🎯 Ranking','💼 Minha carteira','🧪 Validação','ℹ️ Regras'])
with tabs[0]:
    if final.empty: st.info('Rode B3 AI FINAL uma vez no GitHub Actions.')
    else:
        x=final[final.final_signal.isin(['COMPRA_FORTE','AVALIAR_COMPRA','WATCHLIST'])]
        cols=[c for c in ['ticker','company_name','price_now','entry_watch_low','entry_watch_high','validated_score','final_confidence_score','final_signal','quality_score','growth_score','valuation_score','momentum_score','risk_score','trend_confirm','market_regime','failed_gates'] if c in x.columns]
        st.dataframe(x[cols].head(40),use_container_width=True,hide_index=True)
with tabs[1]:
    if not final.empty:
        cols=[c for c in ['ticker','company_name','sector','price_now','validated_score','final_confidence_score','final_signal','gates_passed','gates_total'] if c in final.columns]
        st.dataframe(final[cols].head(250),use_container_width=True,hide_index=True)
with tabs[2]:
    if portfolio.empty: st.info('Quando quiser, preencha portfolio.csv. O sistema já está pronto para integrar sua carteira.')
    elif final.empty: st.dataframe(portfolio,use_container_width=True,hide_index=True)
    else:
        keep=[c for c in ['ticker','price_now','validated_score','final_confidence_score','final_signal','failed_gates'] if c in final.columns]; x=portfolio.merge(final[keep],on='ticker',how='left')
        if {'quantity','price_now'}.issubset(x.columns): x['current_value_brl']=pd.to_numeric(x.quantity,errors='coerce')*pd.to_numeric(x.price_now,errors='coerce')
        st.dataframe(x,use_container_width=True,hide_index=True)
with tabs[3]:
    st.markdown('### Evidência histórica corrigida'); st.json(evidence if evidence else {'status':'ainda não executado'})
    st.markdown('### Validação futura'); st.json(forward if forward else {'status':'ainda sem meses avaliados'})
with tabs[4]:
    st.write('COMPRA_FORTE só desbloqueia após 24 observações mensais NOVAS com evidência positiva contra IBOV e CDI. AVALIAR_COMPRA pode aparecer antes quando os filtros convergem. Em RISK_OFF, novas avaliações de compra ficam bloqueadas. Não há execução automática de ordens.')
