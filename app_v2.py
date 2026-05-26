"""
Risk Lab — Value at Risk v7.0
Trabalho Final | Modelagem Aplicada ao Mercado Financeiro
v7: SEM sidebar — todos os controles no corpo da página (expander).
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.stats import norm, skew, kurtosis
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Risk Lab — VaR v7",
    page_icon="📉",
    layout="wide",
)

PRIMARY="#22d3ee"; SUCCESS="#34d399"; AMBER="#fbbf24"
DANGER="#f87171";  VIOLET="#a78bfa";  BG="#0b1220"
CARD="#111a2e";    BORDER="#1f2a44";  TEXT="#e5e7eb"; MUTED="#94a3b8"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

#MainMenu,footer,.stDeployButton,div[data-testid="stToolbar"],
section[data-testid="stSidebar"] {{ display:none !important; visibility:hidden !important; }}

html,body {{ background:{BG} !important; color:{TEXT} !important; }}
.stApp {{
    background:{BG} !important; font-family:'Inter',sans-serif !important;
    background-image:
        radial-gradient(ellipse 80% 50% at 0% 0%,rgba(34,211,238,.06),transparent 60%),
        radial-gradient(ellipse 60% 40% at 100% 100%,rgba(167,139,250,.05),transparent 60%) !important;
}}
.main .block-container {{ padding:1.5rem 2.5rem 4rem; max-width:1400px; }}

/* inputs */
.stTextInput input,.stNumberInput input,.stDateInput input {{
    background:{BG} !important; border:1px solid {BORDER} !important;
    color:{TEXT} !important; border-radius:8px !important;
    font-family:'JetBrains Mono',monospace !important; font-size:.85rem !important;
}}
[data-baseweb="select"]>div {{
    background:{BG} !important; border:1px solid {BORDER} !important;
    color:{TEXT} !important; border-radius:8px !important;
}}
[data-baseweb="select"] span,[data-baseweb="select"] div {{ color:{TEXT} !important; }}
[data-baseweb="popover"] {{ background:{CARD} !important; border:1px solid {BORDER} !important; }}
[data-baseweb="menu"]    {{ background:{CARD} !important; }}
[data-baseweb="option"]  {{ background:{CARD} !important; color:{TEXT} !important; }}

/* labels */
label,.stTextInput label,.stNumberInput label,.stDateInput label,
.stSelectbox label,.stSlider label,.stMultiSelect label,.stRadio label {{
    color:{MUTED} !important; font-size:.7rem !important;
    text-transform:uppercase !important; letter-spacing:.1em !important; font-weight:600 !important;
}}
.stMultiSelect [data-baseweb="tag"] {{
    background:rgba(34,211,238,.15) !important; border:1px solid rgba(34,211,238,.3) !important;
    color:{PRIMARY} !important; border-radius:4px !important;
}}

/* expander */
.streamlit-expanderHeader {{
    background:{CARD} !important; border:1px solid {BORDER} !important;
    border-radius:10px !important; color:{TEXT} !important;
    font-weight:700 !important; font-size:.9rem !important;
}}
.streamlit-expanderContent {{
    background:{CARD} !important; border:1px solid {BORDER} !important;
    border-top:none !important; border-radius:0 0 10px 10px !important;
    padding:.75rem 1rem !important;
}}

/* botão calcular */
.stButton>button {{
    background:linear-gradient(135deg,{PRIMARY} 0%,#0ea5e9 100%) !important;
    color:#0b1220 !important; border:none !important; border-radius:10px !important;
    font-weight:700 !important; padding:.75rem 2rem !important;
    font-size:1rem !important; letter-spacing:.02em !important;
    box-shadow:0 8px 30px -8px rgba(34,211,238,.6) !important;
    width:100% !important;
}}

/* tabs */
.stTabs [data-baseweb="tab-list"] {{ gap:0; border-bottom:1px solid {BORDER}; background:transparent !important; }}
.stTabs [data-baseweb="tab"] {{
    background:transparent !important; color:{MUTED} !important;
    border:none !important; padding:.8rem 1.1rem !important;
    font-weight:600 !important; font-size:.82rem !important;
}}
.stTabs [aria-selected="true"] {{ color:{PRIMARY} !important; border-bottom:2px solid {PRIMARY} !important; }}
.stTabs [data-baseweb="tab-panel"] {{ background:transparent !important; padding-top:1rem !important; }}

/* cards */
.kpi-card {{
    background:linear-gradient(180deg,{CARD},#0d1626);
    border:1px solid {BORDER}; border-radius:12px; padding:1.25rem; height:100%;
}}
.kpi-label {{ color:{MUTED}; font-size:.7rem; text-transform:uppercase; letter-spacing:.1em; font-weight:600; }}
.kpi-value {{ font-size:1.8rem; font-weight:700; font-family:'JetBrains Mono',monospace; margin:.4rem 0 .2rem; }}
.kpi-sub   {{ color:{MUTED}; font-size:.75rem; }}
.var-card {{
    background:linear-gradient(180deg,{CARD},#0d1626);
    border:1px solid {BORDER}; border-top:3px solid var(--acc); border-radius:12px; padding:1.5rem;
}}
.var-value {{ font-family:'JetBrains Mono',monospace; font-size:2rem; font-weight:700; color:var(--acc); margin:.5rem 0; }}
.section-title {{
    color:{TEXT}; font-size:1.1rem; font-weight:700;
    margin:2rem 0 .8rem; padding-bottom:.6rem; border-bottom:1px solid {BORDER};
}}
.section-sub {{ color:{MUTED}; font-size:.85rem; font-weight:400; margin-left:.5rem; }}
.info-box {{
    background:rgba(34,211,238,.06); border:1px solid rgba(34,211,238,.2);
    border-left:4px solid {PRIMARY}; border-radius:0 8px 8px 0;
    padding:.9rem 1.1rem; margin:.75rem 0; font-size:.82rem; color:{TEXT}; line-height:1.65;
}}
.warn-box {{
    background:rgba(251,191,36,.06); border:1px solid rgba(251,191,36,.2);
    border-left:4px solid {AMBER}; border-radius:0 8px 8px 0;
    padding:.9rem 1.1rem; margin:.75rem 0; font-size:.82rem; color:{TEXT}; line-height:1.65;
}}
::-webkit-scrollbar {{ width:6px; height:6px; }}
::-webkit-scrollbar-track {{ background:{BG}; }}
::-webkit-scrollbar-thumb {{ background:{BORDER}; border-radius:3px; }}
</style>
""", unsafe_allow_html=True)

# ── helpers ──────────────────────────────────────────────
def kpi(label, value, sub="", color=PRIMARY):
    return (f'<div class="kpi-card"><div class="kpi-label">{label}</div>'
            f'<div class="kpi-value" style="color:{color}">{value}</div>'
            f'<div class="kpi-sub">{sub}</div></div>')

def var_card(label, value, pct, color, desc):
    return (f'<div class="var-card" style="--acc:{color}"><div class="kpi-label">{label}</div>'
            f'<div class="var-value">{value}</div><div class="kpi-sub">{pct} do portfólio</div>'
            f'<p style="color:{MUTED};font-size:.78rem;margin-top:.8rem;padding-top:.8rem;'
            f'border-top:1px solid {BORDER}">{desc}</p></div>')

def section(title, sub=""):
    return (f'<div class="section-title">{title}'
            f'<span class="section-sub">{sub}</span></div>')

def badge(text, color=PRIMARY):
    return (f'<span style="background:{color}18;color:{color};border:1px solid {color}30;'
            f'border-radius:4px;padding:.15rem .5rem;font-size:.7rem;font-weight:700;font-family:monospace">{text}</span>')

def info(text): return f'<div class="info-box">💡 {text}</div>'
def warn(text): return f'<div class="warn-box">⚠️ {text}</div>'

def lbl(text):
    st.markdown(f'<div style="color:{PRIMARY};font-size:.65rem;font-weight:700;'
                f'text-transform:uppercase;letter-spacing:.1em;margin:.8rem 0 .25rem">{text}</div>',
                unsafe_allow_html=True)

# ── catálogos ─────────────────────────────────────────────
ACOES_BR = {
    "PETR4.SA":"Petrobras PN","PETR3.SA":"Petrobras ON","VALE3.SA":"Vale ON",
    "ITUB4.SA":"Itaú Unibanco PN","BBDC4.SA":"Bradesco PN","BBAS3.SA":"Banco do Brasil ON",
    "SANB11.SA":"Santander UNT","B3SA3.SA":"B3 ON","MGLU3.SA":"Magazine Luiza ON",
    "WEGE3.SA":"WEG ON","RENT3.SA":"Localiza ON","LREN3.SA":"Lojas Renner ON",
    "ABEV3.SA":"Ambev ON","GGBR4.SA":"Gerdau PN","SUZB3.SA":"Suzano ON",
    "RADL3.SA":"Raia Drogasil ON","TOTS3.SA":"Totvs ON","EMBR3.SA":"Embraer ON",
    "ELET3.SA":"Eletrobras ON","CMIG4.SA":"Cemig PN","SBSP3.SA":"Sabesp ON",
    "BBSE3.SA":"BB Seguridade ON","PRIO3.SA":"PetroRio ON","BPAC11.SA":"BTG Pactual UNT",
    "AZUL4.SA":"Azul PN","CSAN3.SA":"Cosan ON","VIVT3.SA":"Vivo ON",
    "HYPE3.SA":"Hypera ON","KLBN11.SA":"Klabin UNT","JBSS3.SA":"JBS ON",
    "CYRE3.SA":"Cyrela ON","MRVE3.SA":"MRV ON","MULT3.SA":"Multiplan ON",
}
ACOES_US = {
    "AAPL":"Apple","MSFT":"Microsoft","GOOGL":"Alphabet","AMZN":"Amazon",
    "NVDA":"NVIDIA","META":"Meta Platforms","TSLA":"Tesla","JPM":"JPMorgan Chase",
    "BAC":"Bank of America","GS":"Goldman Sachs","XOM":"ExxonMobil",
    "JNJ":"Johnson & Johnson","UNH":"UnitedHealth","KO":"Coca-Cola","WMT":"Walmart",
    "SPY":"S&P 500 ETF","QQQ":"Nasdaq 100 ETF","GLD":"Gold ETF",
    "BTC-USD":"Bitcoin","ETH-USD":"Ethereum","V":"Visa","MA":"Mastercard",
    "NFLX":"Netflix","AMD":"AMD","BA":"Boeing",
}
ALL_ACOES = {**ACOES_BR, **ACOES_US}

STRESS_EVENTS = {
    "🔴 COVID-19 — Crash Março 2020":         {"start":"2020-02-17","end":"2020-03-23","desc":"Pandemia global. Maior queda em 30 anos em 5 semanas.","cor":DANGER,"categoria":"Pandemia","sp500":-34.0},
    "📉 Crise Financeira Global 2008":         {"start":"2008-09-01","end":"2009-03-09","desc":"Colapso Lehman Brothers. Maior recessão desde 1929.","cor":DANGER,"categoria":"Crise Financeira","sp500":-56.8},
    "⚡ Flash Crash — Maio 2010":              {"start":"2010-05-06","end":"2010-05-10","desc":"Dow Jones caiu 1 000 pontos em minutos.","cor":AMBER,"categoria":"Mercado","sp500":-9.2},
    "🇬🇧 Brexit — Jun 2016":                  {"start":"2016-06-23","end":"2016-07-06","desc":"Reino Unido vota sair da UE.","cor":AMBER,"categoria":"Geopolítico","sp500":-5.3},
    "📊 Crise Q4 2018 — Alta de Juros Fed":   {"start":"2018-10-03","end":"2018-12-24","desc":"Pior dezembro nos EUA desde 1931.","cor":AMBER,"categoria":"Monetário","sp500":-19.8},
    "🦠 Variante Delta — Jul 2021":            {"start":"2021-07-19","end":"2021-08-05","desc":"Temor de novos lockdowns.","cor":"#fb923c","categoria":"Pandemia","sp500":-4.2},
    "🏦 Evergrande — Set 2021":               {"start":"2021-09-13","end":"2021-09-30","desc":"Maior incorporadora chinesa à beira da falência.","cor":DANGER,"categoria":"Crédito","sp500":-5.2},
    "🚀 Ciclo de Juros Fed — 2022":           {"start":"2022-01-03","end":"2022-10-13","desc":"Fed vai de 0% a 4,5% a.a.","cor":DANGER,"categoria":"Monetário","sp500":-27.5},
    "💥 Colapso FTX — Nov 2022":              {"start":"2022-11-07","end":"2022-11-20","desc":"Exchange FTX declara falência.","cor":VIOLET,"categoria":"Cripto","sp500":-4.1},
    "🏦 SVB — Mar 2023":                      {"start":"2023-03-08","end":"2023-03-24","desc":"Silicon Valley Bank quebra em 48h.","cor":DANGER,"categoria":"Crise Financeira","sp500":-6.8},
    "🇧🇷 8 de Janeiro 2023":                  {"start":"2023-01-06","end":"2023-01-13","desc":"Invasão do Congresso e STF.","cor":SUCCESS,"categoria":"Político","sp500":-0.5},
    "⚔️ Iran–Israel — Abr 2024":              {"start":"2024-04-13","end":"2024-04-22","desc":"Irã lança 300+ drones contra Israel.","cor":AMBER,"categoria":"Geopolítico","sp500":-3.1},
    "📉 Carry Trade — Ago 2024":              {"start":"2024-08-01","end":"2024-08-09","desc":"Nikkei -12% num dia.","cor":DANGER,"categoria":"Mercado","sp500":-8.5},
    "🇺🇸 Tarifaço Trump — Abr 2025":          {"start":"2025-04-02","end":"2025-04-09","desc":"EUA impõem tarifas de 10–145%.","cor":DANGER,"categoria":"Geopolítico","sp500":-12.0},
}
EMOJI_CAT={"Pandemia":"🦠","Crise Financeira":"🏦","Mercado":"⚡","Geopolítico":"⚔️",
           "Monetário":"🏛️","Cripto":"💎","Político":"🗳️","Crédito":"💳"}

# ── funções financeiras ───────────────────────────────────
def black_scholes(S,K,T,r,sigma,tipo="call"):
    if T<=0: return max(S-K,0) if tipo=="call" else max(K-S,0)
    if sigma<=0: return max(S-K*np.exp(-r*T),0) if tipo=="call" else max(K*np.exp(-r*T)-S,0)
    d1=(np.log(S/K)+(r+.5*sigma**2)*T)/(sigma*np.sqrt(T)); d2=d1-sigma*np.sqrt(T)
    if tipo=="call": return S*norm.cdf(d1)-K*np.exp(-r*T)*norm.cdf(d2)
    return K*np.exp(-r*T)*norm.cdf(-d2)-S*norm.cdf(-d1)

def todas_gregas(S,K,T,r,sigma,tipo="call"):
    if T<=0 or sigma<=0: return 0.,0.,0.,0.,0.
    d1=(np.log(S/K)+(r+.5*sigma**2)*T)/(sigma*np.sqrt(T)); d2=d1-sigma*np.sqrt(T)
    delta=norm.cdf(d1) if tipo=="call" else norm.cdf(d1)-1
    gamma=norm.pdf(d1)/(S*sigma*np.sqrt(T))
    vega=S*norm.pdf(d1)*np.sqrt(T)
    if tipo=="call":
        theta=(-S*norm.pdf(d1)*sigma/(2*np.sqrt(T))-r*K*np.exp(-r*T)*norm.cdf(d2))/252
        rho=K*T*np.exp(-r*T)*norm.cdf(d2)/100
    else:
        theta=(-S*norm.pdf(d1)*sigma/(2*np.sqrt(T))+r*K*np.exp(-r*T)*norm.cdf(-d2))/252
        rho=-K*T*np.exp(-r*T)*norm.cdf(-d2)/100
    return float(delta),float(gamma),float(vega),float(theta),float(rho)

@st.cache_data(ttl=600,show_spinner=False)
def baixar(tickers_tuple,ini,fim=None):
    tickers=list(tickers_tuple)
    try:
        kw=dict(start=ini,auto_adjust=True,progress=False,threads=False)
        if fim: kw["end"]=fim
        df=yf.download(tickers,**kw)
        p=df["Close"] if "Close" in df.columns else df
        if isinstance(p,pd.Series): p=p.to_frame(tickers[0])
        p=p.dropna(how="all")
        if not p.empty: return p,None
    except: pass
    frames={}
    for t in tickers:
        try:
            kw2=dict(start=ini,auto_adjust=True)
            if fim: kw2["end"]=fim
            h=yf.Ticker(t).history(**kw2)
            if not h.empty: frames[t]=h["Close"]
        except: pass
    if frames:
        df2=pd.DataFrame(frames).dropna(how="all")
        if not df2.empty: return df2,None
    return None,"Falha ao baixar dados."

def chart_rc():
    plt.rcParams.update({
        "figure.facecolor":CARD,"axes.facecolor":CARD,"axes.edgecolor":BORDER,
        "axes.labelcolor":MUTED,"axes.titlecolor":TEXT,"text.color":TEXT,
        "xtick.color":MUTED,"ytick.color":MUTED,"grid.color":BORDER,"grid.alpha":.5,
        "axes.spines.top":False,"axes.spines.right":False,
        "legend.facecolor":CARD,"legend.edgecolor":BORDER,"legend.labelcolor":TEXT,
        "font.family":"monospace","figure.dpi":110,
    })

# ════════════════════════════════════════════════════════
# HEADER
# ════════════════════════════════════════════════════════
st.markdown(f"""
<div style="display:flex;align-items:center;gap:1rem;padding:1.25rem 0 1.5rem;
            border-bottom:1px solid {BORDER};margin-bottom:1.5rem">
  <div style="width:52px;height:52px;border-radius:14px;flex-shrink:0;
              background:linear-gradient(135deg,rgba(34,211,238,.2),rgba(167,139,250,.15));
              border:1px solid {PRIMARY}40;display:flex;align-items:center;
              justify-content:center;font-size:1.7rem">⚡</div>
  <div>
    <p style="color:{MUTED};font-size:.7rem;text-transform:uppercase;
              letter-spacing:.12em;margin:0;font-weight:600">
      Modelagem Aplicada ao Mercado Financeiro</p>
    <h1 style="margin:.15rem 0 0;font-size:1.6rem">
      Risk Lab <span style="color:{MUTED};font-weight:400">— Value at Risk v7.0</span></h1>
  </div>
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════
# PAINEL DE CONFIGURAÇÃO — expander sempre visível
# ════════════════════════════════════════════════════════
with st.expander("⚙️  Configurar Carteira e Parâmetros", expanded=True):

    # ── linha 1: mercado + ativos ──
    col_mkt, col_ativos = st.columns([1, 3])

    with col_mkt:
        lbl("▸ Mercado")
        mercado = st.radio("mercado_radio", ["🇧🇷 Brasil","🇺🇸 EUA/Global","✏️ Manual"],
                           label_visibility="collapsed")

    with col_ativos:
        lbl("▸ Ativos da carteira")
        if mercado == "✏️ Manual":
            raw = st.text_input("Tickers (separados por vírgula)",
                                "PETR4.SA, VALE3.SA, ITUB4.SA",
                                label_visibility="collapsed")
            tickers_sel = [t.strip().upper() for t in raw.split(",") if t.strip()]
        else:
            cat = ACOES_BR if "Brasil" in mercado else ACOES_US
            opt_map = {f"{tk} — {nm}": tk for tk,nm in cat.items()}
            defs_br = ["PETR4.SA — Petrobras PN","VALE3.SA — Vale ON","ITUB4.SA — Itaú Unibanco PN"]
            defs_us = ["AAPL — Apple","MSFT — Microsoft","NVDA — NVIDIA"]
            defs    = [d for d in (defs_br if "Brasil" in mercado else defs_us) if d in opt_map]
            sel     = st.multiselect("Ativos (máx 8)", list(opt_map.keys()),
                                     default=defs, max_selections=8,
                                     label_visibility="collapsed")
            tickers_sel = [opt_map[d] for d in sel]

    st.markdown("<hr style='border:none;border-top:1px solid "+BORDER+";margin:.75rem 0'>",
                unsafe_allow_html=True)

    # ── linha 2: qtd / pesos / data ──
    col_q, col_p, col_d = st.columns(3)
    with col_q:
        lbl("▸ Quantidades (mesma ordem)")
        qty_str = st.text_input("qtd", ", ".join(["1000"]*len(tickers_sel)),
                                label_visibility="collapsed")
    with col_p:
        lbl("▸ Pesos % — vazio = por valor de mercado")
        pesos_str = st.text_input("pesos", "",
                                  help="Ex: 30, 30, 25, 15",
                                  label_visibility="collapsed")
    with col_d:
        lbl("▸ Data início")
        data_ini = st.date_input("data_ini", pd.to_datetime("2022-01-01"),
                                 label_visibility="collapsed")

    st.markdown("<hr style='border:none;border-top:1px solid "+BORDER+";margin:.75rem 0'>",
                unsafe_allow_html=True)

    # ── linha 3: parâmetros VaR + opção ──
    col_v1,col_v2,col_o1,col_o2,col_o3,col_o4,col_o5 = st.columns([1,1,1,1,1,1,1])
    with col_v1:
        lbl("Confiança")
        nivel = st.selectbox("nivel", [0.90,0.95,0.975,0.99], index=1,
                             format_func=lambda x:f"{x*100:.1f}%",
                             label_visibility="collapsed")
    with col_v2:
        lbl("Horizonte (dias)")
        horizonte = st.number_input("horiz", 1, 30, 1, label_visibility="collapsed")
    with col_o1:
        lbl("Ativo-objeto opção")
        opt_ativo = st.selectbox("opt_ativo", tickers_sel if tickers_sel else ["PETR4.SA"],
                                 label_visibility="collapsed")
    with col_o2:
        lbl("Tipo opção")
        opt_tipo = st.selectbox("opt_tipo", ["call","put"], label_visibility="collapsed")
    with col_o3:
        lbl("Qtd. opções")
        opt_qty = st.number_input("opt_qty", 0, value=1000, step=100, label_visibility="collapsed")
    with col_o4:
        lbl("Strike (K)")
        strike = st.number_input("strike", 1.0, value=40.0, step=0.5, label_visibility="collapsed")
    with col_o5:
        lbl("Taxa livre risco a.a.")
        rf = st.number_input("rf", 0.0, 1.0, 0.105, 0.005, "%.3f", label_visibility="collapsed")

    col_te, col_btn = st.columns([1,2])
    with col_te:
        lbl("Vencimento (anos)")
        T_exp = st.number_input("T_exp", 0.01, 5.0, 0.25, 0.05, "%.2f", label_visibility="collapsed")
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        calcular = st.button("▶  CALCULAR VaR", use_container_width=True)

# ── tickers selecionados: mostrar badges ──
if tickers_sel:
    st.markdown(
        '<div style="display:flex;gap:.4rem;flex-wrap:wrap;margin:.5rem 0 1rem">'
        + "".join([badge(t) for t in tickers_sel])
        + '</div>', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════
# ESTADO INICIAL
# ════════════════════════════════════════════════════════
if not calcular:
    st.markdown(f"""
    <div style="text-align:center;padding:3.5rem 2rem;background:{CARD};
                border:1px solid {BORDER};border-radius:16px;margin-top:.5rem">
      <div style="font-size:2.8rem">📉</div>
      <h2 style="margin:.75rem 0 .5rem">Configure a carteira acima e clique em Calcular VaR</h2>
      <p style="color:{MUTED};margin:0">
        Expanda o painel <b style="color:{PRIMARY}">⚙️ Configurar Carteira</b> se estiver fechado,
        escolha os ativos e pressione o botão azul.
      </p>
    </div>""", unsafe_allow_html=True)
    st.stop()

if not tickers_sel:
    st.error("Selecione ao menos um ativo no painel de configuração.")
    st.stop()

# ════════════════════════════════════════════════════════
# DADOS
# ════════════════════════════════════════════════════════
with st.spinner("Conectando ao mercado…"):
    precos, erro = baixar(tuple(tickers_sel), str(data_ini))

if erro or precos is None:
    st.error(f"Erro ao baixar dados: {erro}"); st.stop()

tickers = [t for t in tickers_sel if t in precos.columns]
if not tickers:
    st.error("Nenhum ticker retornou dados. Verifique os símbolos."); st.stop()

try:   qtds=[int(q.strip()) for q in qty_str.split(",")]
except: qtds=[1000]*len(tickers)
while len(qtds)<len(tickers): qtds.append(1000)
quantidades=dict(zip(tickers,qtds))

precos=precos[tickers].dropna()
retornos=precos.pct_change().dropna()
ultimos=precos.iloc[-1]

# pesos
try:
    pw=[float(x.strip()) for x in pesos_str.split(",") if x.strip()]
    if len(pw)==len(tickers) and abs(sum(pw)-100)<1:
        pesos_carteira=np.array(pw)/100; pesos_modo="customizados"
    else: raise ValueError
except:
    vals=np.array([quantidades[t]*float(ultimos[t]) for t in tickers])
    pesos_carteira=vals/vals.sum(); pesos_modo="por valor de mercado"

# ════════════════════════════════════════════════════════
# CÁLCULOS
# ════════════════════════════════════════════════════════
v_acoes   = sum(quantidades[t]*float(ultimos[t]) for t in tickers)
S0        = float(ultimos[opt_ativo]) if opt_ativo in tickers else float(ultimos.iloc[0])
vol_anual = float(retornos[opt_ativo].std()*np.sqrt(252)) if opt_ativo in retornos.columns else 0.3
preco_op  = black_scholes(S0,strike,T_exp,rf,vol_anual,opt_tipo)
v_op      = opt_qty*preco_op; v_total=v_acoes+v_op

ret_cart  = retornos[tickers].dot(pesos_carteira)
mu_c,sig_c= float(ret_cart.mean()),float(ret_cart.std())
pct       = 1-nivel; z_var=norm.ppf(1-nivel)

cov_mat     = retornos[tickers].cov()
sig_cov     = float(np.sqrt(pesos_carteira @ cov_mat.values @ pesos_carteira))
var_param_cov = -(mu_c*horizonte+z_var*sig_cov*np.sqrt(horizonte))*v_acoes
var_hist    = -float(np.percentile(ret_cart,pct*100))*v_acoes

cenarios_pnl=[]
for i in range(len(retornos)):
    ch=retornos[tickers].iloc[i]; np_=ultimos*(1+ch)
    nv=sum(quantidades[t]*float(np_[t]) for t in tickers)
    S_c=float(np_[opt_ativo]) if opt_ativo in tickers else S0
    no=black_scholes(S_c,strike,max(T_exp-horizonte/252,0),rf,vol_anual,opt_tipo)
    cenarios_pnl.append((nv+opt_qty*no)-v_total)
cenarios_pnl=np.array(cenarios_pnl)
var_full  = -float(np.percentile(cenarios_pnl,pct*100))
es_hist   = -float(ret_cart[ret_cart<=np.percentile(ret_cart,pct*100)].mean())*v_acoes
delta_v,gamma_v,vega_v,theta_v,rho_v = todas_gregas(S0,strike,T_exp,rf,vol_anual,opt_tipo)
corr_mat  = retornos[tickers].corr()
chart_rc()

# ════════════════════════════════════════════════════════
# ABAS
# ════════════════════════════════════════════════════════
tab1,tab2,tab3,tab4,tab5,tab6,tab7 = st.tabs([
    "  📊 Resumo  ","  📐 Covariância  ","  📈 Gráficos  ",
    "  🎯 Gregas & Opção  ","  🔢 Janelas & ES  ",
    "  🌡️ Stress Test  ","  📋 Versões  ",
])

# ── TAB 1 RESUMO ─────────────────────────────────────────
with tab1:
    st.markdown(section("Composição",f"{len(tickers)} ativos · pesos {pesos_modo} · {opt_tipo.upper()} {opt_ativo}"),unsafe_allow_html=True)
    c1,c2,c3,c4=st.columns(4)
    c1.markdown(kpi("Ações",f"R$ {v_acoes:,.0f}",f"{len(tickers)} ativos",PRIMARY),unsafe_allow_html=True)
    c2.markdown(kpi("Opções",f"R$ {v_op:,.0f}",f"{opt_qty:,} {opt_tipo}s · K={strike}",SUCCESS),unsafe_allow_html=True)
    c3.markdown(kpi("Total",f"R$ {v_total:,.0f}","valor de mercado",AMBER),unsafe_allow_html=True)
    c4.markdown(kpi("Vol. diária",f"{sig_c*100:.2f}%",f"anual {sig_c*np.sqrt(252)*100:.1f}%",VIOLET),unsafe_allow_html=True)

    st.markdown(section("Value at Risk — 3 Métodos",f"IC {nivel*100:.1f}% · h={horizonte}d"),unsafe_allow_html=True)
    st.markdown(info("<b>Paramétrico:</b> dist. normal, simples, subestima caudas. "
                     "<b>Histórico:</b> percentil empírico, sem hipótese de normalidade. "
                     "<b>Full Valuation:</b> reprecifica Black-Scholes, capta não-linearidade."),unsafe_allow_html=True)
    v1,v2,v3=st.columns(3)
    v1.markdown(var_card("Paramétrico (w′Σw)",f"R$ {var_param_cov:,.0f}",f"{var_param_cov/v_total*100:.2f}%",PRIMARY,"Dist. Normal · w′Σw"),unsafe_allow_html=True)
    v2.markdown(var_card("Histórico",f"R$ {var_hist:,.0f}",f"{var_hist/v_total*100:.2f}%",SUCCESS,"Percentil empírico"),unsafe_allow_html=True)
    v3.markdown(var_card("Full Valuation",f"R$ {var_full:,.0f}",f"{var_full/v_total*100:.2f}%",AMBER,"Reprecificação BS"),unsafe_allow_html=True)

    st.markdown(section("Comparativo dos Métodos"),unsafe_allow_html=True)
    df_comp=pd.DataFrame({
        "Método":["VaR Paramétrico (w′Σw)","VaR Histórico","VaR Full Valuation"],
        "VaR (R$)":[f"R$ {var_param_cov:,.0f}",f"R$ {var_hist:,.0f}",f"R$ {var_full:,.0f}"],
        "% portfólio":[f"{var_param_cov/v_total*100:.2f}%",f"{var_hist/v_total*100:.2f}%",f"{var_full/v_total*100:.2f}%"],
        "Hipótese":["Normal · linear","Empírica","BS · não-linear"],
        "Opção?":["Não","Não","✅ Sim"],
    })
    st.dataframe(df_comp,use_container_width=True,hide_index=True)
    fig_bar,ax_bar=plt.subplots(figsize=(8,3.5))
    bars=ax_bar.bar(["Param.(w′Σw)","Histórico","Full Val."],[var_param_cov,var_hist,var_full],
                    color=[PRIMARY,SUCCESS,AMBER],alpha=.85,width=.5)
    for b in bars:
        h=b.get_height()
        ax_bar.text(b.get_x()+b.get_width()/2,h+h*.01,f"R$ {h:,.0f}",ha="center",va="bottom",fontsize=9,color=TEXT)
    ax_bar.set_ylabel("VaR (R$)"); ax_bar.set_title("Comparativo VaR — 3 Métodos"); ax_bar.grid(axis="y")
    st.pyplot(fig_bar); plt.close(fig_bar)

    st.markdown(section("Posições Individuais"),unsafe_allow_html=True)
    rows_pos=[]
    for i,t in enumerate(tickers):
        p0=float(ultimos[t]); rt=retornos[t]; val=quantidades[t]*p0
        vi=-(rt.mean()+norm.ppf(1-nivel)*rt.std())*val
        rows_pos.append({"Ticker":t,"Nome":ALL_ACOES.get(t,t),"Qtd.":f"{quantidades[t]:,}",
            "Preço":f"R$ {p0:.2f}","Valor":f"R$ {val:,.0f}","Peso":f"{pesos_carteira[i]*100:.1f}%",
            "Vol. diária":f"{rt.std()*100:.2f}%","VaR indiv.":f"R$ {vi:,.0f}"})
    st.dataframe(pd.DataFrame(rows_pos),use_container_width=True,hide_index=True)

    st.markdown(section("Interpretação Didática"),unsafe_allow_html=True)
    st.markdown(info(
        f"<b>VaR de R$ {var_hist:,.0f} com {nivel*100:.0f}% de confiança:</b> "
        f"em condições normais, a perda diária da carteira não ultrapassa esse valor em "
        f"{nivel*100:.0f}% dos dias. Há {pct*100:.0f}% de chance de ser superado."
    ),unsafe_allow_html=True)
    if var_full>var_hist*1.05:
        st.markdown(warn(
            f"Full Valuation (R$ {var_full:,.0f}) > Histórico (R$ {var_hist:,.0f}): "
            "a opção adiciona não-linearidade não capturada pelo VaR linear."
        ),unsafe_allow_html=True)

# ── TAB 2 COVARIÂNCIA ────────────────────────────────────
with tab2:
    st.markdown(section("Matriz de Covariância e Correlação","σ_p = √(w′Σw)"),unsafe_allow_html=True)
    st.markdown(info("A volatilidade da carteira multi-ativo é <b>σ_p = √(w′ Σ w)</b>, "
                     "capturando a correlação entre os ativos."),unsafe_allow_html=True)
    col_cov,col_corr=st.columns(2)
    with col_cov:
        st.markdown(f'<div style="color:{MUTED};font-size:.72rem;font-weight:700;text-transform:uppercase;margin-bottom:.4rem">Covariância (×10⁻⁴)</div>',unsafe_allow_html=True)
        st.dataframe((cov_mat*10000).round(4),use_container_width=True)
    with col_corr:
        st.markdown(f'<div style="color:{MUTED};font-size:.72rem;font-weight:700;text-transform:uppercase;margin-bottom:.4rem">Correlação</div>',unsafe_allow_html=True)
        st.dataframe(corr_mat.round(4),use_container_width=True)
    if len(tickers)>1:
        fig_hm,ax_hm=plt.subplots(figsize=(max(5,len(tickers)*1.2),max(4,len(tickers)*1.0)))
        ca=corr_mat.values
        im=ax_hm.imshow(ca,cmap="RdYlGn",vmin=-1,vmax=1)
        ax_hm.set_xticks(range(len(tickers))); ax_hm.set_xticklabels(tickers,rotation=45,ha="right")
        ax_hm.set_yticks(range(len(tickers))); ax_hm.set_yticklabels(tickers)
        for i in range(len(tickers)):
            for j in range(len(tickers)):
                ax_hm.text(j,i,f"{ca[i,j]:.2f}",ha="center",va="center",fontsize=9,
                           color="black" if abs(ca[i,j])<.6 else "white")
        plt.colorbar(im,ax=ax_hm); ax_hm.set_title("Heatmap de Correlação")
        fig_hm.tight_layout(); st.pyplot(fig_hm); plt.close(fig_hm)
    mcvar_rows=[]
    for i,t in enumerate(tickers):
        contrib=pesos_carteira[i]*sum(pesos_carteira[j]*float(cov_mat.loc[t,tickers[j]]) for j in range(len(tickers)))/sig_cov
        mcvar_rows.append({"Ativo":t,"Peso":f"{pesos_carteira[i]*100:.1f}%",
                           "Vol. diária":f"{retornos[t].std()*100:.3f}%","Contrib. marginal":f"{contrib*100:.1f}%"})
    st.markdown(section("Contribuição Marginal ao VaR"),unsafe_allow_html=True)
    st.dataframe(pd.DataFrame(mcvar_rows),use_container_width=True,hide_index=True)
    st.markdown(info(f"σ_p via w′Σw = <b>{sig_cov*100:.3f}%/dia</b> ({sig_cov*np.sqrt(252)*100:.2f}%/ano) · "
                     f"VaR Paramétrico = <b>R$ {var_param_cov:,.0f}</b>"),unsafe_allow_html=True)

# ── TAB 3 GRÁFICOS ───────────────────────────────────────
with tab3:
    st.markdown(section("Distribuição Histórica dos Retornos"),unsafe_allow_html=True)
    fig1,ax1=plt.subplots(figsize=(12,4))
    ax1.hist(ret_cart,bins=60,color=PRIMARY,alpha=.45,label="Retornos")
    tail=ret_cart[ret_cart<=np.percentile(ret_cart,pct*100)]
    ax1.hist(tail,bins=40,color=DANGER,alpha=.85,label=f"Cauda {pct*100:.0f}%")
    ax1.axvline(np.percentile(ret_cart,pct*100),color=SUCCESS,ls="--",lw=1.8,label=f"VaR Hist. {nivel*100:.0f}%")
    ax1.axvline(-(z_var*sig_cov),color=VIOLET,ls=":",lw=1.6,label="VaR Param.(cov)")
    ax1.set_xlabel("Retorno diário"); ax1.set_ylabel("Frequência")
    ax1.set_title("Distribuição Histórica — Cauda em Vermelho"); ax1.legend()
    st.pyplot(fig1); plt.close(fig1)

    st.markdown(section("Preços Normalizados — Base 100"),unsafe_allow_html=True)
    fig2,ax2=plt.subplots(figsize=(12,4))
    cores=[PRIMARY,SUCCESS,AMBER,VIOLET,DANGER,"#fb923c","#f472b6","#38bdf8"]
    for i,t in enumerate(tickers):
        s=precos[t]/precos[t].iloc[0]*100
        ax2.plot(s.index,s.values,lw=1.8,color=cores[i%len(cores)],label=t)
    ax2.axhline(100,color=BORDER,lw=.8,ls="--"); ax2.set_ylabel("Índice (base 100)"); ax2.legend()
    st.pyplot(fig2); plt.close(fig2)

    st.markdown(section("P&L Full Valuation"),unsafe_allow_html=True)
    fig3,ax3=plt.subplots(figsize=(12,4))
    ax3.hist(cenarios_pnl,bins=60,color=AMBER,alpha=.45,label="P&L cenários")
    tail_pnl=cenarios_pnl[cenarios_pnl<=np.percentile(cenarios_pnl,pct*100)]
    ax3.hist(tail_pnl,bins=40,color=DANGER,alpha=.85,label=f"Cauda {pct*100:.0f}%")
    ax3.axvline(-var_full,color=AMBER,ls="--",lw=1.8,label=f"VaR Full R$ {var_full:,.0f}")
    ax3.set_xlabel("P&L (R$)"); ax3.set_ylabel("Frequência")
    ax3.set_title("P&L Full Valuation — ações + opção BS"); ax3.legend()
    st.pyplot(fig3); plt.close(fig3)

    st.markdown(section("VaR Rolling — janela 63 pregões"),unsafe_allow_html=True)
    rv=[-np.percentile(ret_cart.iloc[i-63:i],pct*100)*v_acoes for i in range(63,len(ret_cart))]
    fig4,ax4=plt.subplots(figsize=(12,3.5))
    ax4.plot(ret_cart.index[63:],rv,color=PRIMARY,lw=1.5)
    ax4.axhline(var_hist,color=DANGER,ls="--",lw=1,label=f"VaR hist. total R$ {var_hist:,.0f}")
    ax4.set_ylabel("VaR (R$)"); ax4.set_title("VaR Histórico Rolling (janela 63d)"); ax4.legend()
    st.pyplot(fig4); plt.close(fig4)

# ── TAB 4 GREGAS ─────────────────────────────────────────
with tab4:
    st.markdown(section("5 Gregas da Opção",f"{opt_tipo.upper()} {opt_ativo} · K={strike} · T={T_exp}a · σ={vol_anual*100:.1f}%"),unsafe_allow_html=True)
    st.markdown(info("<b>Delta:</b> var. preço opção por R$1 no ativo. "
                     "<b>Gamma:</b> var. do Delta (convexidade). "
                     "<b>Vega:</b> sens. à vol (+1 p.p.). "
                     "<b>Theta:</b> decaimento por dia. "
                     "<b>Rho:</b> sens. à taxa (+1 p.p.)."),unsafe_allow_html=True)
    g1,g2,g3,g4,g5=st.columns(5)
    g1.markdown(kpi("Delta Δ",f"{delta_v:.4f}","exposição direcional",PRIMARY),unsafe_allow_html=True)
    g2.markdown(kpi("Gamma Γ",f"{gamma_v:.6f}","convexidade",SUCCESS),unsafe_allow_html=True)
    g3.markdown(kpi("Vega ν",f"{vega_v:.4f}","sens. à volatilidade",AMBER),unsafe_allow_html=True)
    g4.markdown(kpi("Theta Θ",f"{theta_v:.4f}","decaimento/dia",VIOLET),unsafe_allow_html=True)
    g5.markdown(kpi("Rho ρ",f"{rho_v:.4f}","sens. à taxa",MUTED),unsafe_allow_html=True)

    st.markdown(section("Call vs Put — Mesmo Strike",f"K={strike} · T={T_exp}a"),unsafe_allow_html=True)
    pc=black_scholes(S0,strike,T_exp,rf,vol_anual,"call")
    pp=black_scholes(S0,strike,T_exp,rf,vol_anual,"put")
    dc,gc,vc2,tc2,rc2=todas_gregas(S0,strike,T_exp,rf,vol_anual,"call")
    dp,gp,vp2,tp2,rp2=todas_gregas(S0,strike,T_exp,rf,vol_anual,"put")
    st.dataframe(pd.DataFrame({
        "Métrica":["Preço BS","Delta Δ","Gamma Γ","Vega ν","Theta Θ","Rho ρ"],
        "Call":[f"{pc:.4f}",f"{dc:.4f}",f"{gc:.6f}",f"{vc2:.4f}",f"{tc2:.4f}",f"{rc2:.4f}"],
        "Put": [f"{pp:.4f}",f"{dp:.4f}",f"{gp:.6f}",f"{vp2:.4f}",f"{tp2:.4f}",f"{rp2:.4f}"],
    }),use_container_width=True,hide_index=True)

    fig_cv,axes=plt.subplots(1,2,figsize=(12,4))
    ps=np.linspace(S0*.65,S0*1.35,200)
    axes[0].plot(ps,[black_scholes(s,strike,T_exp,rf,vol_anual,"call") for s in ps],color=PRIMARY,lw=2,label="Call")
    axes[0].plot(ps,[black_scholes(s,strike,T_exp,rf,vol_anual,"put")  for s in ps],color=AMBER,  lw=2,label="Put")
    axes[0].axvline(S0,color=SUCCESS,ls=":",lw=1.2,label=f"S0={S0:.1f}")
    axes[0].axvline(strike,color=DANGER,ls="--",alpha=.6,label=f"K={strike}")
    axes[0].legend(); axes[0].set_title("Preço da Opção × Preço do Ativo")
    axes[1].plot(ps,[todas_gregas(s,strike,T_exp,rf,vol_anual,"call")[0] for s in ps],color=PRIMARY,lw=2,label="Δ Call")
    axes[1].plot(ps,[todas_gregas(s,strike,T_exp,rf,vol_anual,"put")[0]  for s in ps],color=AMBER,  lw=2,label="Δ Put")
    axes[1].axhline(0,color=BORDER,lw=.8); axes[1].axvline(strike,color=DANGER,ls="--",alpha=.6)
    axes[1].legend(); axes[1].set_title("Delta × Preço do Ativo")
    st.pyplot(fig_cv); plt.close(fig_cv)

    st.markdown(section("Análise de Sensibilidade — Não-Linearidade da Opção","Seção 11 do notebook"),unsafe_allow_html=True)
    fig_s,ax_s=plt.subplots(figsize=(12,4))
    ps2=np.linspace(S0*.7,S0*1.3,150)
    ax_s.plot(ps2,[black_scholes(s,strike,T_exp,rf,vol_anual,"call") for s in ps2],color=PRIMARY,lw=2,label="Call")
    ax_s.plot(ps2,[black_scholes(s,strike,T_exp,rf,vol_anual,"put")  for s in ps2],color=AMBER,  lw=2,label="Put")
    ax_s.axvline(S0,color=SUCCESS,ls=":",lw=1.2,label=f"S0={S0:.1f}")
    ax_s.axvline(strike,color=DANGER,ls="--",alpha=.6,label=f"K={strike}")
    ax_s.set_xlabel("Preço do ativo"); ax_s.set_ylabel("Preço da opção")
    ax_s.set_title("Sensibilidade do Preço da Opção ao Preço do Ativo — comportamento não-linear")
    ax_s.legend(); st.pyplot(fig_s); plt.close(fig_s)

# ── TAB 5 JANELAS & ES ───────────────────────────────────
with tab5:
    st.markdown(section("Expected Shortfall — CVaR","Perda média além do VaR"),unsafe_allow_html=True)
    st.markdown(info("CVaR responde: quanto se perde, <b>em média</b>, nos piores cenários além do VaR?"),unsafe_allow_html=True)
    e1,e2,e3=st.columns(3)
    e1.markdown(kpi("ES Histórico",f"R$ {es_hist:,.0f}",f"média cauda {pct*100:.0f}%",DANGER),unsafe_allow_html=True)
    e2.markdown(kpi("Razão ES/VaR",f"{es_hist/max(var_hist,1):.2f}×","cauda pesada",VIOLET),unsafe_allow_html=True)
    e3.markdown(kpi("Extra além VaR",f"R$ {es_hist-var_hist:,.0f}","perda adicional",AMBER),unsafe_allow_html=True)

    st.markdown(section("Efeito da Janela Histórica — Exercício 4","Janela crítica em gestão de risco"),unsafe_allow_html=True)
    janelas_def=[("Desde 2020","2020-01-01"),("Desde 2022","2022-01-01"),("Desde 2023","2023-01-01"),
                 ("Últimos 252d",None),("Últimos 63d","63")]
    rows_j=[]
    for nome,ini in janelas_def:
        sub=(retornos.tail(63) if ini=="63" else retornos.tail(252) if ini is None
             else retornos[retornos.index>=pd.to_datetime(ini)])
        if len(sub)<30: continue
        rp=sub[tickers].dot(pesos_carteira)
        vp=-(rp.mean()+norm.ppf(1-nivel)*rp.std())*v_acoes
        vh=-np.percentile(rp,(1-nivel)*100)*v_acoes
        es_=-rp[rp<=np.percentile(rp,(1-nivel)*100)].mean()*v_acoes
        rows_j.append({"Janela":nome,"N obs.":len(sub),"VaR Param.":f"R$ {vp:,.0f}",
                       "VaR Hist.":f"R$ {vh:,.0f}","CVaR":f"R$ {es_:,.0f}",
                       "Vol diária":f"{rp.std()*100:.2f}%","Assimetria":f"{skew(rp):.2f}","Curtose":f"{kurtosis(rp):.2f}"})
    st.dataframe(pd.DataFrame(rows_j),use_container_width=True,hide_index=True)

    st.markdown(section("Sensibilidade ao Nível de Confiança — Exercício 2"),unsafe_allow_html=True)
    rows_ni=[]
    for ni in [0.90,0.95,0.975,0.99]:
        p_=1-ni
        rows_ni.append({"IC":f"{ni*100:.1f}%",
            "VaR Param.":f"R$ {-(mu_c*horizonte+norm.ppf(1-ni)*sig_cov*np.sqrt(horizonte))*v_acoes:,.0f}",
            "VaR Hist.":f"R$ {-np.percentile(ret_cart,p_*100)*v_acoes:,.0f}",
            "CVaR":f"R$ {-ret_cart[ret_cart<=np.percentile(ret_cart,p_*100)].mean()*v_acoes:,.0f}"})
    st.dataframe(pd.DataFrame(rows_ni),use_container_width=True,hide_index=True)

# ── TAB 6 STRESS TEST ────────────────────────────────────
with tab6:
    st.markdown(section("🌡️ Stress Test","VaR + stress test = gestão de risco completa"),unsafe_allow_html=True)
    st.markdown(info("O stress test simula cenários extremos que podem nunca ter ocorrido na janela histórica. "
                     "Use <b>junto com</b> o VaR."),unsafe_allow_html=True)

    st_a,st_b=st.tabs(["  🔨 Choque Manual (Ex.8)  ","  📰 Marcos Históricos  "])

    with st_a:
        st.markdown(section("Choque Manual — Exercício 8","Simule choques como no enunciado"),unsafe_allow_html=True)
        c_s1,c_s2=st.columns(2)
        with c_s1: choque_global=st.slider("Choque global em TODOS os ativos (%)",-50,50,-5)
        with c_s2: st.markdown(f'<div style="color:{MUTED};font-size:.7rem;font-weight:600;text-transform:uppercase;margin-bottom:.4rem">Choques individuais por ativo (%)</div>',unsafe_allow_html=True)
        choques_ind={}
        cols_ch=st.columns(min(len(tickers),4))
        for i,t in enumerate(tickers):
            choques_ind[t]=cols_ch[i%4].number_input(t,value=-10 if i==0 else 0,min_value=-100,max_value=100,step=5)

        def calc_stress(choques_dict):
            vv=sum(quantidades[t]*float(ultimos[t])*(1+choques_dict[t]/100) for t in tickers)
            S_c=float(ultimos[opt_ativo])*(1+choques_dict.get(opt_ativo,0)/100) if opt_ativo in tickers else S0
            vv+=opt_qty*black_scholes(S_c,strike,max(T_exp-1/252,0),rf,vol_anual,opt_tipo)
            return vv-v_total

        pnl_A=calc_stress({t:choque_global for t in tickers})
        pnl_B=calc_stress(choques_ind)
        c_a,c_b,c_c=st.columns(3)
        c_a.markdown(kpi("Portfólio Atual",f"R$ {v_total:,.0f}","antes do choque",PRIMARY),unsafe_allow_html=True)
        c_b.markdown(kpi(f"Cenário A ({choque_global:+}% global)",f"R$ {pnl_A:+,.0f}",f"{pnl_A/v_total*100:+.2f}%",SUCCESS if pnl_A>=0 else DANGER),unsafe_allow_html=True)
        c_c.markdown(kpi("Cenário B (individual)",f"R$ {pnl_B:+,.0f}",f"{pnl_B/v_total*100:+.2f}%",SUCCESS if pnl_B>=0 else DANGER),unsafe_allow_html=True)
        for lbl_,pnl_v in [("A",pnl_A),("B",pnl_B)]:
            if pnl_v<0 and abs(pnl_v)>var_hist:
                st.markdown(warn(f"Cenário {lbl_}: R$ {abs(pnl_v):,.0f} <b>excede o VaR Histórico</b> (R$ {var_hist:,.0f}). Stress test complementar ao VaR."),unsafe_allow_html=True)
            elif pnl_v<0:
                st.markdown(info(f"Cenário {lbl_}: R$ {abs(pnl_v):,.0f} dentro do VaR Histórico (R$ {var_hist:,.0f})."),unsafe_allow_html=True)

        rows_stress=[]
        for t in tickers:
            p0=float(ultimos[t]); pA=p0*(1+choque_global/100); pB=p0*(1+choques_ind[t]/100)
            rows_stress.append({"Ativo":t,"Preço Atual":f"R$ {p0:.2f}",
                f"Preço A({choque_global:+}%)":f"R$ {pA:.2f}",f"P&L A":f"R$ {quantidades[t]*(pA-p0):+,.0f}",
                f"Preço B({choques_ind[t]:+}%)":f"R$ {pB:.2f}",f"P&L B":f"R$ {quantidades[t]*(pB-p0):+,.0f}"})
        st.dataframe(pd.DataFrame(rows_stress),use_container_width=True,hide_index=True)

    with st_b:
        st.markdown(section("Marcos Históricos","Impacto real nos seus ativos"),unsafe_allow_html=True)
        cat_all=sorted(set(v["categoria"] for v in STRESS_EVENTS.values()))
        c_f1,c_f2=st.columns([1,2])
        with c_f1: cat_sel=st.multiselect("Categoria",cat_all,default=cat_all)
        ev_filt={k:v for k,v in STRESS_EVENTS.items() if v["categoria"] in cat_sel}
        with c_f2: ev_sel=st.selectbox("Evento",list(ev_filt.keys()))
        ev=ev_filt[ev_sel]
        st.markdown(f"""
        <div style="background:{CARD};border:1px solid {BORDER};border-left:5px solid {ev['cor']};
                    border-radius:0 12px 12px 0;padding:1rem 1.3rem;margin:.75rem 0">
          <div style="display:flex;align-items:flex-start;gap:1.5rem;flex-wrap:wrap">
            <div style="flex:1"><div style="font-size:1rem;font-weight:700">{ev_sel}</div>
              <div style="color:{MUTED};font-size:.82rem;margin-top:.3rem">{ev['desc']}</div>
              <div style="margin-top:.4rem">
                <span style="background:{ev['cor']}18;color:{ev['cor']};border:1px solid {ev['cor']}30;
                             border-radius:4px;padding:.1rem .45rem;font-size:.65rem;font-weight:700">
                  {EMOJI_CAT.get(ev['categoria'],'📌')} {ev['categoria']}
                </span>
                <span style="color:{MUTED};font-size:.7rem;margin-left:.5rem">{ev['start']} → {ev['end']}</span>
              </div>
            </div>
            <div style="text-align:right">
              <div style="color:{MUTED};font-size:.62rem;text-transform:uppercase;font-weight:600">S&P 500</div>
              <div style="font-size:1.9rem;font-weight:800;font-family:monospace;
                          color:{DANGER if ev['sp500']<0 else SUCCESS}">{ev['sp500']:+.1f}%</div>
            </div>
          </div>
        </div>""",unsafe_allow_html=True)

        with st.spinner(f"Buscando dados {ev['start']} → {ev['end']}…"):
            ini_ev=(pd.to_datetime(ev["start"])-pd.Timedelta(days=10)).strftime("%Y-%m-%d")
            p_ev,e_ev=baixar(tuple(tickers),ini_ev,ev["end"])

        if e_ev or p_ev is None or p_ev.empty:
            st.warning("Dados indisponíveis para este período.")
        else:
            tks_ev=[t for t in tickers if t in p_ev.columns]
            pp=p_ev[tks_ev].dropna(how="all")
            pp=pp[pp.index>=pd.to_datetime(ev["start"])]
            if len(pp)<2:
                st.warning("Dados insuficientes.")
            else:
                ini_p=pp.iloc[0]; fim_p=pp.iloc[-1]
                cols_ev=st.columns(len(tks_ev))
                for i,t in enumerate(tks_ev):
                    r=(float(fim_p[t])-float(ini_p[t]))/float(ini_p[t])*100
                    pl=quantidades.get(t,1000)*(float(fim_p[t])-float(ini_p[t]))
                    cols_ev[i].markdown(kpi(ALL_ACOES.get(t,t),f"{r:+.1f}%",f"P&L R$ {pl:+,.0f}",SUCCESS if r>=0 else DANGER),unsafe_allow_html=True)
                v_i=sum(quantidades.get(t,1000)*float(ini_p[t]) for t in tks_ev)
                v_f=sum(quantidades.get(t,1000)*float(fim_p[t]) for t in tks_ev)
                ret_p=(v_f-v_i)/v_i*100; pnl_p=v_f-v_i; cobriu=abs(pnl_p)<=var_hist
                st.markdown(f"""
                <div style="display:flex;gap:1rem;margin:1rem 0;flex-wrap:wrap">
                  <div style="background:{CARD};border:1px solid {BORDER};border-radius:10px;padding:.9rem 1.3rem;flex:1;min-width:130px">
                    <div style="color:{MUTED};font-size:.62rem;font-weight:600;text-transform:uppercase">Início</div>
                    <div style="font-size:1.3rem;font-weight:700;font-family:monospace">R$ {v_i:,.0f}</div>
                  </div>
                  <div style="background:{CARD};border:1px solid {BORDER};border-radius:10px;padding:.9rem 1.3rem;flex:1;min-width:130px">
                    <div style="color:{MUTED};font-size:.62rem;font-weight:600;text-transform:uppercase">Fim</div>
                    <div style="font-size:1.3rem;font-weight:700;font-family:monospace">R$ {v_f:,.0f}</div>
                  </div>
                  <div style="background:{CARD};border:1px solid {'#15803d' if ret_p>=0 else '#7f1d1d'};border-radius:10px;padding:.9rem 1.3rem;flex:1;min-width:130px">
                    <div style="color:{MUTED};font-size:.62rem;font-weight:600;text-transform:uppercase">Variação</div>
                    <div style="font-size:1.3rem;font-weight:700;font-family:monospace;color:{SUCCESS if ret_p>=0 else DANGER}">{ret_p:+.2f}%</div>
                    <div style="color:{MUTED};font-size:.7rem">R$ {pnl_p:+,.0f}</div>
                  </div>
                  <div style="background:{CARD};border:1px solid {'#15803d' if cobriu else '#7f1d1d'};border-radius:10px;padding:.9rem 1.3rem;flex:1;min-width:130px">
                    <div style="color:{MUTED};font-size:.62rem;font-weight:600;text-transform:uppercase">VaR Cobriu?</div>
                    <div style="font-size:1.1rem;font-weight:700;color:{SUCCESS if cobriu else DANGER}">{'✅ Sim' if cobriu else '❌ Excedeu'}</div>
                    <div style="color:{MUTED};font-size:.7rem">VaR hist. R$ {var_hist:,.0f}</div>
                  </div>
                </div>""",unsafe_allow_html=True)
                fig_ev,ax_ev=plt.subplots(figsize=(12,4))
                for i,t in enumerate(tks_ev):
                    s=pp[t].dropna()
                    if len(s)>0: ax_ev.plot(s.index,s/s.iloc[0]*100,lw=2,color=cores[i%len(cores)],label=t)
                ax_ev.axhline(100,color=BORDER,lw=.8,ls="--",label="Base 100")
                ax_ev.set_ylabel("Índice (base 100)"); ax_ev.legend()
                ax_ev.set_title(f"Stress Test: {ev_sel}"); st.pyplot(fig_ev); plt.close(fig_ev)

        st.markdown(f'<div style="margin-top:1.5rem">{section("Todos os Marcos")}</div>',unsafe_allow_html=True)
        st.dataframe(pd.DataFrame([{"Evento":n,"Período":f"{i['start']}→{i['end']}",
            "Categoria":f"{EMOJI_CAT.get(i['categoria'],'📌')} {i['categoria']}",
            "S&P 500":f"{i['sp500']:+.1f}%"} for n,i in STRESS_EVENTS.items()]),
            use_container_width=True,hide_index=True)

# ── TAB 7 VERSÕES ────────────────────────────────────────
with tab7:
    st.markdown(section("📋 Histórico de Versões","Evolução do Risk Lab"),unsafe_allow_html=True)
    versoes=[
        {"version":"v7.0","date":"2025","title":"Sem Sidebar — Controles no corpo da página","cor":PRIMARY,"changes":[
            ("🐛 Fix","Sidebar removida — causa raiz do problema em todos os ambientes",DANGER),
            ("✨ Novo","Painel de configuração via st.expander no corpo principal",SUCCESS),
            ("✨ Novo","Todos os controles acessíveis sem sidebar",SUCCESS),
        ]},
        {"version":"v6.0","date":"2025","title":"Auditoria completa ao notebook","cor":VIOLET,"changes":[
            ("✨ Novo","Matriz de covariância w′Σw + heatmap + contribuição marginal",SUCCESS),
            ("✨ Novo","Theta Θ e Rho ρ — todas as 5 gregas",SUCCESS),
            ("✨ Novo","Análise de sensibilidade e perfil de Gamma",SUCCESS),
            ("✨ Novo","Stress Test manual com choques por ativo (Ex.8)",SUCCESS),
            ("✨ Novo","Pesos customizáveis (Ex.2) · Tabela comparativa 3 métodos (Seção 10)",SUCCESS),
        ]},
        {"version":"v5.x","date":"2025","title":"Stress Test Histórico & Seletor de Ativos","cor":AMBER,"changes":[
            ("✨ Novo","14 marcos históricos globais",SUCCESS),
            ("✨ Novo","60+ ações BR, EUA, ETFs, cripto",SUCCESS),
        ]},
        {"version":"v2.0","date":"2024","title":"Black-Scholes & CVaR","cor":SUCCESS,"changes":[
            ("✨ Novo","BS · Gregas · ES · Full Valuation",SUCCESS),
        ]},
        {"version":"v1.0","date":"2024","title":"Versão Inicial","cor":MUTED,"changes":[
            ("✨ Novo","VaR Param. e Hist. · Yahoo Finance",SUCCESS),
        ]},
    ]
    for v in versoes:
        st.markdown(f"""
        <div style="margin-bottom:1.25rem;padding:1.1rem;background:{CARD};
                    border:1px solid {BORDER};border-left:4px solid {v['cor']};border-radius:0 10px 10px 0">
          <div style="display:flex;align-items:center;gap:.75rem;margin-bottom:.6rem">
            <span style="background:{v['cor']}18;color:{v['cor']};border:1px solid {v['cor']}40;
                         border-radius:6px;padding:.15rem .6rem;font-size:.75rem;font-weight:700;font-family:monospace">{v['version']}</span>
            <span style="color:{TEXT};font-size:.9rem;font-weight:700">{v['title']}</span>
            <span style="color:{MUTED};font-size:.7rem;margin-left:auto">{v['date']}</span>
          </div>""",unsafe_allow_html=True)
        for tipo,desc,cor in v["changes"]:
            st.markdown(f"""
            <div style="display:flex;align-items:flex-start;gap:.6rem;padding:.25rem 0;border-bottom:1px solid {BORDER}40">
              <span style="background:{cor}18;color:{cor};border:1px solid {cor}30;border-radius:4px;
                           padding:.1rem .4rem;font-size:.62rem;font-weight:700;white-space:nowrap">{tipo}</span>
              <span style="color:{TEXT};font-size:.8rem">{desc}</span>
            </div>""",unsafe_allow_html=True)
        st.markdown("</div>",unsafe_allow_html=True)

    st.markdown(section("✅ Cobertura dos Requisitos"),unsafe_allow_html=True)
    reqs=[
        ("✅","Teoria 1","O que é VaR — definição e interpretação","Resumo"),
        ("✅","Teoria 2","Limitações · stress test complementar","Stress Test"),
        ("✅","Teoria 3","VaR Param. — w′Σw","Resumo + Covariância"),
        ("✅","Teoria 4","VaR Histórico — percentil empírico","Resumo + Janelas"),
        ("✅","Teoria 5","VaR Full Valuation — reprecificação BS","Resumo + Gráficos"),
        ("✅","Teoria 6-8","Call/Put · Black-Scholes · 5 Gregas","Gregas & Opção"),
        ("✅","Ex. 1","VaR Paramétrico ações BR","Resumo"),
        ("✅","Ex. 2","Pesos customizáveis (30/30/25/15%)","Painel config. → campo Pesos %"),
        ("✅","Ex. 3","Comparativo Param. vs Histórico","Resumo"),
        ("✅","Ex. 4","Efeito da janela histórica","Janelas & ES"),
        ("✅","Ex. 5","Call europeia + gregas","Gregas & Opção"),
        ("✅","Ex. 6","Ações vs Full Valuation","Resumo"),
        ("✅","Ex. 7","Call vs Put","Gregas & Opção"),
        ("✅","Ex. 8","Stress test choques -5% e -10%","Stress Test → Choque Manual"),
        ("⭐","Bônus","14 marcos históricos reais","Stress Test → Marcos"),
        ("⭐","Bônus","Histórico de versões","Esta aba"),
    ]
    st.dataframe(pd.DataFrame(reqs,columns=["Status","Ref.","Requisito","Onde no app"]),
                 use_container_width=True,hide_index=True)
