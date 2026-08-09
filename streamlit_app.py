from pathlib import Path
import json
import pandas as pd
import streamlit as st
st.set_page_config(page_title='B3 AI Radar',page_icon='📈',layout='wide'); st.title('📈 B3 AI Radar'); st.caption('COMPRA_FORTE só pode existir após validação fundamental point-in-time fora da amostra. Nenhum score é garantia ou probabilidade de lucro.')
def csv(p):
    p=Path(p)
    if not p.exists() or not p.stat().st_size:return pd.DataFrame()
    try:return pd.read_csv(p,comment='#')
    except:return pd.DataFrame()
final=csv('results/final_signals.csv'); portfolio=csv('portfolio.csv'); status={}; p=Path('results/final_signal_status.json')
if p.exists():
    try:status=json.loads(p.read_text())
    except:pass
c1,c2,c3,c4=st.columns(4); c1.metric('Regime',status.get('market_regime','?')); c2.metric('Mercado',status.get('market_regime_score','?')); c3.metric('Compras fortes',status.get('strong_buys','?')); c4.metric('PIT OOS validado','SIM' if status.get('calibration_validated') else 'NÃO')
tabs=st.tabs(['🚨 Sinais finais','🎯 Ranking','💼 Minha carteira','🧪 Validação','ℹ️ Regras'])
with tabs[0]:
    st.subheader('Oportunidades atuais')
    if final.empty:st.info('Execute 02 - Benchmark calibrate final signals.')
    else:
        x=final[final['final_signal'].isin(['COMPRA_FORTE','AVALIAR_COMPRA','WATCHLIST'])]; cols=[c for c in ['ticker','company_name','price_now','validated_score','final_confidence_score','final_signal','entry_watch_low','entry_watch_high','quality_score','growth_score','valuation_score','momentum_score','risk_score','trend_confirm','vectorbt_current_confirm','market_regime','failed_gates'] if c in x.columns]; st.dataframe(x[cols].head(40),use_container_width=True,hide_index=True)
with tabs[1]:
    if not final.empty:
        cols=[c for c in ['ticker','company_name','sector','price_now','validated_score','final_confidence_score','final_signal','gates_passed','gates_total','quality_score','growth_score','valuation_score','momentum_score','risk_score'] if c in final.columns]; st.dataframe(final[cols].head(250),use_container_width=True,hide_index=True)
with tabs[2]:
    if portfolio.empty:st.info('Preencha portfolio.csv.')
    elif final.empty:st.dataframe(portfolio,use_container_width=True,hide_index=True)
    else:
        keep=[c for c in ['ticker','price_now','validated_score','final_confidence_score','final_signal','failed_gates'] if c in final.columns]; x=portfolio.merge(final[keep],on='ticker',how='left');
        if {'quantity','price_now'}.issubset(x.columns):x['current_value_brl']=pd.to_numeric(x['quantity'],errors='coerce')*pd.to_numeric(x['price_now'],errors='coerce')
        st.dataframe(x,use_container_width=True,hide_index=True)
with tabs[3]:
    for title,path in [('VectorBT vs IBOV/CDI','results/vectorbt_benchmark.json'),('Backtest PIT + pesos calibrados','results/calibrated_weights.json'),('Base CVM PIT','results/cvm_pit_status.json')]:
        st.markdown(f'### {title}'); p=Path(path); st.json(json.loads(p.read_text())) if p.exists() else st.info('Ainda não executado.')
with tabs[4]:
    st.markdown('''### COMPRA_FORTE\nSó existe se o teste final point-in-time for aceito. Depois ainda exige qualidade, valuation, momentum, risco, tendência/força relativa, mercado, notícia e VectorBT apenas se o próprio VectorBT vencer benchmarks.\n\n`final_confidence_score` é score de evidências, não probabilidade de lucro. `entry_watch_low/high` é faixa estatística de atenção, não preço-alvo fundamental.''')
