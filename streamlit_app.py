
from pathlib import Path
import json
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Nubank Global Radar",
    page_icon="🟣",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.block-container {padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1500px;}
[data-testid="stMetric"] {
    background: linear-gradient(145deg, rgba(130,65,255,.16), rgba(255,255,255,.03));
    border: 1px solid rgba(160,120,255,.18);
    padding: 16px 18px;
    border-radius: 18px;
}
.hero {
    padding: 25px 28px; border-radius: 24px;
    background: linear-gradient(135deg, rgba(104,45,190,.30), rgba(31,24,58,.28));
    border: 1px solid rgba(175,135,255,.22);
    margin-bottom: 18px;
}
.hero h1 {margin:0 0 5px 0; font-size:2.25rem;}
.hero p {opacity:.78; margin:0;}
.buy-card {
    border:1px solid rgba(120,220,160,.30);
    background:linear-gradient(145deg,rgba(25,120,70,.16),rgba(255,255,255,.02));
    border-radius:20px; padding:20px; margin-bottom:12px;
}
.wait-card {
    border:1px solid rgba(240,185,60,.28);
    background:rgba(240,185,60,.06);
    border-radius:18px; padding:18px; margin-bottom:10px;
}
.muted {opacity:.68; font-size:.92rem;}
.big-score {font-size:2.0rem; font-weight:750;}
.pill-green {display:inline-block;padding:5px 10px;border-radius:999px;background:rgba(28,190,100,.17);border:1px solid rgba(28,190,100,.35);}
.pill-yellow {display:inline-block;padding:5px 10px;border-radius:999px;background:rgba(240,180,40,.15);border:1px solid rgba(240,180,40,.35);}
.pill-red {display:inline-block;padding:5px 10px;border-radius:999px;background:rgba(235,75,75,.14);border:1px solid rgba(235,75,75,.30);}
div[data-testid="stDataFrame"] {border-radius:16px; overflow:hidden;}
</style>
""",unsafe_allow_html=True)

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

rank=csv("results/nubank_global_ranking.csv")
dec=csv("results/weekly_final_decisions.csv")
pf=csv("results/nubank_portfolio_actions.csv")
val=csv("results/nubank_forward_validation_summary.csv")
radar_status=js("results/nubank_radar_status.json")
status=js("results/weekly_decision_status.json")

st.markdown("""
<div class="hero">
  <h1>🟣 Nubank Global Opportunity Radar</h1>
  <p>Oportunidades compráveis na sua conta: ações B3, ações globais via BDR, ETFs, FIIs e fundos listados. Renda fixa excluída.</p>
</div>
""",unsafe_allow_html=True)

m1,m2,m3,m4,m5=st.columns(5)
m1.metric("Regime Brasil",radar_status.get("brasil_regime","?"))
m2.metric("Regime Global",radar_status.get("global_regime","?"))
m3.metric("Ativos avaliados",radar_status.get("ranked_total","?"))
m4.metric("🟢 Compraria agora",status.get("buy_now",0))
m5.metric("🟡 Só até preço",status.get("buy_only_up_to",0))

with st.sidebar:
    st.header("Filtros")
    classes=sorted(dec.asset_class.dropna().unique().tolist()) if not dec.empty and "asset_class" in dec else []
    selected=st.multiselect("Classes",classes,default=classes)
    decisions=["COMPRARIA_AGORA","COMPRARIA_SOMENTE_ATE","ESPERARIA","NAO_COMPRARIA"]
    selected_dec=st.multiselect("Decisão",decisions,default=decisions)
    min_score=st.slider("Score mínimo",0,100,60)
    query=st.text_input("Ticker / nome")
    st.divider()
    st.caption("Atualização completa automática: semanal. Validação de desempenho: mensal.")

view=dec.copy()
if not view.empty:
    if selected:view=view[view.asset_class.isin(selected)]
    if selected_dec:view=view[view.final_decision.isin(selected_dec)]
    view=view[pd.to_numeric(view.global_opportunity_score,errors="coerce").fillna(0)>=min_score]
    if query:
        q=query.lower()
        view=view[
            view.ticker.astype(str).str.lower().str.contains(q,na=False)
            | view.get("name",view.ticker).astype(str).str.lower().str.contains(q,na=False)
        ]

tabs=st.tabs([
    "🏆 Decisão final","🟢 Compraria agora","🟡 Esperaria preço",
    "🌎 Ranking completo","👛 Minha carteira","🧪 Validação","📖 Regras"
])

with tabs[0]:
    st.subheader("A nata da semana")
    st.caption("Esta é a última camada: o radar já filtrou o universo e cada ativo foi reanalisado antes da decisão.")

    buys=view[view.final_decision.eq("COMPRARIA_AGORA")].head(5) if not view.empty else pd.DataFrame()
    if buys.empty:
        st.info("Nenhuma compra passou por todos os filtros esta semana. Isso é um resultado válido: não comprar também é uma decisão.")
    else:
        for _,r in buys.iterrows():
            score=float(r.get("global_opportunity_score",0))
            st.markdown(f"""
<div class="buy-card">
  <div class="pill-green">COMPRARIA AGORA</div>
  <h2 style="margin:.55rem 0 .1rem 0">{r.get('ticker','')} · {r.get('name','')}</h2>
  <div class="muted">{r.get('asset_class','')}</div>
  <div class="big-score">{score:.1f}/100</div>
  <b>Preço atual:</b> {r.get('price_now',float('nan')):.2f}
  &nbsp;&nbsp; <b>Faixa:</b> {r.get('buy_zone_low',float('nan')):.2f} – {r.get('buy_zone_high',float('nan')):.2f}<br>
  <b>Não perseguir acima:</b> {r.get('do_not_chase_above',float('nan')):.2f}
  &nbsp;&nbsp; <b>Invalidação:</b> {r.get('invalidation_price',float('nan')):.2f}<br>
  <b>Posição inicial:</b> {100*r.get('suggested_initial_position_pct',0):.1f}%
  &nbsp;&nbsp; <b>Teto total:</b> {100*r.get('suggested_total_position_pct',0):.1f}%<br><br>
  <b>A favor:</b> {r.get('why_buy','')}<br>
  <b>Riscos:</b> {r.get('why_not','')}<br>
  <span class="muted">{r.get('plain_language','')}</span>
</div>
""",unsafe_allow_html=True)

    if not view.empty:
        st.markdown("#### Resumo das melhores decisões")
        cols=[c for c in [
            "ticker","name","asset_class","price_now","global_opportunity_score","final_decision",
            "buy_zone_low","buy_zone_high","do_not_chase_above","invalidation_price",
            "suggested_initial_position_pct","why_buy","why_not"
        ] if c in view]
        st.dataframe(view[cols].head(40),use_container_width=True,hide_index=True)

with tabs[1]:
    x=view[view.final_decision.eq("COMPRARIA_AGORA")] if not view.empty else view
    if x.empty:st.success("Nenhuma compra confirmada agora. O sistema preferiu ficar de fora.")
    else:
        cols=[c for c in [
            "ticker","name","asset_class","price_now","global_opportunity_score",
            "buy_zone_low","buy_zone_high","do_not_chase_above","invalidation_price",
            "review_price_2R","review_price_3R","suggested_initial_position_pct",
            "suggested_total_position_pct","why_buy","why_not"
        ] if c in x]
        st.dataframe(x[cols],use_container_width=True,hide_index=True)

with tabs[2]:
    x=view[view.final_decision.isin(["COMPRARIA_SOMENTE_ATE","ESPERARIA"])] if not view.empty else view
    if x.empty:st.info("Nenhum ativo aguardando preço/condição.")
    else:
        cols=[c for c in [
            "ticker","name","asset_class","price_now","global_opportunity_score","final_decision",
            "max_buy_price","do_not_chase_above","why_buy","why_not","plain_language"
        ] if c in x]
        st.dataframe(x[cols],use_container_width=True,hide_index=True)

with tabs[3]:
    if rank.empty:st.info("Ranking indisponível.")
    else:
        byclass=rank.groupby("asset_class",dropna=False)["ticker"].count().sort_values(ascending=False)
        st.markdown("#### Cobertura por classe")
        st.bar_chart(byclass)
        cols=[c for c in [
            "ticker","name","asset_class","price_now","global_opportunity_score",
            "signal","strengths","risks","regime"
        ] if c in rank]
        st.dataframe(rank[cols].head(250),use_container_width=True,hide_index=True)

with tabs[4]:
    if pf.empty:
        st.info("Quando portfolio.csv estiver preenchido, o radar também passa a controlar concentração e recomendar MANTER / AUMENTAR / NÃO AUMENTAR / REAVALIAR.")
    else:
        st.dataframe(pf,use_container_width=True,hide_index=True)

with tabs[5]:
    st.markdown("### O radar está acertando?")
    st.caption("As melhores de cada classe são registradas e comparadas com seu benchmark 20 pregões depois.")
    if val.empty:st.info("Ainda não há observações futuras avaliadas.")
    else:st.dataframe(val,use_container_width=True,hide_index=True)
    st.json(status if status else {"status":"aguardando primeira análise semanal"})

with tabs[6]:
    st.markdown("""
### Como a decisão é tomada

O sistema **não compra porque o score chegou a 90**.

Para `COMPRARIA_AGORA`, o ativo precisa passar simultaneamente por:

- score global muito alto e posição entre os melhores da própria classe;
- tendência e momentum;
- risco, volatilidade, drawdown e liquidez;
- regime de mercado;
- regras próprias da classe;
- fundamentos adicionais para BDRs e modelo fundamental separado para ações B3;
- validação futura da classe, quando já houver histórico suficiente;
- concentração da sua própria carteira;
- preço atual dentro da faixa de entrada;
- ausência de alerta relevante nas notícias verificadas.

### Decisões

🟢 **COMPRARIA_AGORA** — passou em tudo e está na faixa quantitativa de entrada.

🟡 **COMPRARIA_SOMENTE_ATE** — eu gosto do ativo, mas só compraria até o preço máximo mostrado.

🟠 **ESPERARIA** — ativo pode ser bom, mas preço/risco/evidência ainda não justificam colocar dinheiro agora.

🔴 **NAO_COMPRARIA** — evidências atuais insuficientes.

### Plano de risco

`invalidation_price` é o ponto em que a estrutura usada para justificar a entrada precisa ser reavaliada.  
`suggested_initial_position_pct` é a primeira parcela.  
`suggested_total_position_pct` é o teto calculado usando orçamento aproximado de 0,75% de risco da carteira até a invalidação, limitado por classe.

Não existe investimento de renda variável sem possibilidade de perda. O objetivo é ser seletivo, controlar o tamanho dos erros e não perseguir preços.
""")
