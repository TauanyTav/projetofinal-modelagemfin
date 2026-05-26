"""
Risk Lab — Value at Risk Calculator v6.0
Trabalho Final | Modelagem Aplicada ao Mercado Financeiro
Auditado contra o notebook oficial — todos os requisitos cobertos.
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

# ── PAGE CONFIG ───────────────────────────────────────────
st.set_page_config(
    page_title="Risk Lab — VaR v6",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

PRIMARY = "#22d3ee"; SUCCESS = "#34d399"; AMBER = "#fbbf24"
DANGER  = "#f87171"; VIOLET  = "#a78bfa"; BG    = "#0b1220"
CARD    = "#111a2e"; BORDER  = "#1f2a44"; TEXT  = "#e5e7eb"
MUTED   = "#94a3b8"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

#MainMenu, footer, .stDeployButton, div[data-testid="stToolbar"] {{ display:none !important; }}

html, body {{ background:{BG} !important; color:{TEXT} !important; }}
.stApp {{
    background:{BG} !important; font-family:'Inter',sans-serif !important;
    background-image:
        radial-gradient(ellipse 80% 50% at 0% 0%,rgba(34,211,238,.05),transparent 60%),
        radial-gradient(ellipse 60% 40% at 100% 100%,rgba(167,139,250,.05),transparent 60%) !important;
}}
.main .block-container {{ padding:1.5rem 2rem 4rem; max-width:1600px; }}

section[data-testid="stSidebar"] {{
    background:{CARD} !important; border-right:1px solid {BORDER} !important;
    display:flex !important; visibility:visible !important; min-width:290px !important;
}}
section[data-testid="stSidebar"] > div {{
    display:flex !important; visibility:visible !important; width:100% !important;
}}
button[data-testid="collapsedControl"] {{
    display:flex !important; visibility:visible !important;
    background:{CARD} !important; color:{PRIMARY} !important; border:1px solid {BORDER} !important;
}}

.stTextInput input, .stNumberInput input, .stDateInput input {{
    background:{BG} !important; border:1px solid {BORDER} !important;
    color:{TEXT} !important; border-radius:8px !important;
    font-family:'JetBrains Mono',monospace !important; font-size:.85rem !important;
}}
[data-baseweb="select"] > div {{
    background:{BG} !important; border:1px solid {BORDER} !important;
    color:{TEXT} !important; border-radius:8px !important;
}}
[data-baseweb="select"] span, [data-baseweb="select"] div {{ color:{TEXT} !important; }}

label, .stTextInput label, .stNumberInput label, .stDateInput label,
.stSelectbox label, .stSlider label, .stMultiSelect label, .stRadio label {{
    color:{MUTED} !important; font-size:.7rem !important;
    text-transform:uppercase !important; letter-spacing:.1em !important; font-weight:600 !important;
}}
.stMultiSelect [data-baseweb="tag"] {{
    background:rgba(34,211,238,.15) !important; border:1px solid rgba(34,211,238,.3) !important;
    color:{PRIMARY} !important; border-radius:4px !important;
}}
[data-baseweb="popover"] {{ background:{CARD} !important; border:1px solid {BORDER} !important; }}
[data-baseweb="menu"] {{ background:{CARD} !important; }}
[data-baseweb="option"] {{ background:{CARD} !important; color:{TEXT} !important; }}

.stButton > button {{
    background:linear-gradient(135deg,{PRIMARY} 0%,#0ea5e9 100%) !important;
    color:#0b1220 !important; border:none !important; border-radius:10px !important;
    font-weight:700 !important; padding:.7rem 1.5rem !important; width:100% !important;
    box-shadow:0 8px 30px -8px rgba(34,211,238,.5) !important; font-size:.9rem !important;
}}
.stTabs [data-baseweb="tab-list"] {{ gap:0; border-bottom:1px solid {BORDER}; background:transparent !important; }}
.stTabs [data-baseweb="tab"] {{
    background:transparent !important; color:{MUTED} !important;
    border:none !important; padding:.8rem 1.1rem !important; font-weight:600 !important; font-size:.82rem !important;
}}
.stTabs [aria-selected="true"] {{ color:{PRIMARY} !important; border-bottom:2px solid {PRIMARY} !important; }}
.stTabs [data-baseweb="tab-panel"] {{ background:transparent !important; padding-top:1rem !important; }}

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

# ── HELPERS ──────────────────────────────────────────────
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
            f'border-radius:4px;padding:.15rem .5rem;font-size:.7rem;font-weight:700;'
            f'font-family:monospace">{text}</span>')

def info(text):  return f'<div class="info-box">💡 {text}</div>'
def warn(text):  return f'<div class="warn-box">⚠️ {text}</div>'

# ── CATÁLOGO ─────────────────────────────────────────────
ACOES_BR = {
    "PETR4.SA":"Petrobras PN","PETR3.SA":"Petrobras ON","VALE3.SA":"Vale ON",
    "ITUB4.SA":"Itaú Unibanco PN","BBDC4.SA":"Bradesco PN","BBAS3.SA":"Banco do Brasil ON",
    "SANB11.SA":"Santander UNT","B3SA3.SA":"B3 ON","MGLU3.SA":"Magazine Luiza ON",
    "WEGE3.SA":"WEG ON","RENT3.SA":"Localiza ON","LREN3.SA":"Lojas Renner ON",
    "ABEV3.SA":"Ambev ON","GGBR4.SA":"Gerdau PN","SUZB3.SA":"Suzano ON",
    "RADL3.SA":"Raia Drogasil ON","TOTS3.SA":"Totvs ON","EMBR3.SA":"Embraer ON",
    "ELET3.SA":"Eletrobras ON","CMIG4.SA":"Cemig PN","SBSP3.SA":"Sabesp ON",
    "BBSE3.SA":"BB Seguridade ON","PRIO3.SA":"PetroRio ON","BPAC11.SA":"BTG Pactual UNT",
    "RDOR3.SA":"Rede D'Or ON","HAPV3.SA":"Hapvida ON","AZUL4.SA":"Azul PN",
    "CSAN3.SA":"Cosan ON","VIVT3.SA":"Vivo ON","HYPE3.SA":"Hypera ON",
    "KLBN11.SA":"Klabin UNT","JBSS3.SA":"JBS ON","CYRE3.SA":"Cyrela ON",
    "MRVE3.SA":"MRV ON","MULT3.SA":"Multiplan ON",
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

# ── STRESS EVENTS ────────────────────────────────────────
STRESS_EVENTS = {
    "🔴 COVID-19 — Crash Março 2020":          {"start":"2020-02-17","end":"2020-03-23","desc":"Pandemia global. Maior queda em 30 anos em 5 semanas.","cor":DANGER,"categoria":"Pandemia","sp500":-34.0},
    "📉 Crise Financeira Global 2008":          {"start":"2008-09-01","end":"2009-03-09","desc":"Colapso Lehman Brothers / subprime. Maior recessão desde 1929.","cor":DANGER,"categoria":"Crise Financeira","sp500":-56.8},
    "⚡ Flash Crash — Maio 2010":               {"start":"2010-05-06","end":"2010-05-10","desc":"Dow Jones caiu 1 000 pontos em minutos por ordens algorítmicas.","cor":AMBER,"categoria":"Mercado","sp500":-9.2},
    "🇬🇧 Brexit — Referendo Jun 2016":          {"start":"2016-06-23","end":"2016-07-06","desc":"Reino Unido vota sair da UE. Libra ao menor nível em 30 anos.","cor":AMBER,"categoria":"Geopolítico","sp500":-5.3},
    "📊 Alta de Juros Fed — Q4 2018":           {"start":"2018-10-03","end":"2018-12-24","desc":"Fed sobe juros agressivamente. Pior dezembro desde 1931.","cor":AMBER,"categoria":"Monetário","sp500":-19.8},
    "🦠 Variante Delta — Jul 2021":             {"start":"2021-07-19","end":"2021-08-05","desc":"Nova variante COVID-19. Temor de lockdowns.","cor":"#fb923c","categoria":"Pandemia","sp500":-4.2},
    "🏦 Colapso Evergrande — Set 2021":         {"start":"2021-09-13","end":"2021-09-30","desc":"Maior incorporadora da China à beira da falência.","cor":DANGER,"categoria":"Crédito","sp500":-5.2},
    "🚀 Ciclo de Juros Fed — 2022":             {"start":"2022-01-03","end":"2022-10-13","desc":"Fed vai de 0% a 4,5% a.a. Pior ano para bonds em 40 anos.","cor":DANGER,"categoria":"Monetário","sp500":-27.5},
    "💥 Colapso FTX — Nov 2022":                {"start":"2022-11-07","end":"2022-11-20","desc":"Exchange FTX declara falência. Bitcoin -25% em dias.","cor":VIOLET,"categoria":"Cripto","sp500":-4.1},
    "🏦 SVB / Bancos Regionais — Mar 2023":     {"start":"2023-03-08","end":"2023-03-24","desc":"Silicon Valley Bank e Signature Bank quebram em 48 h.","cor":DANGER,"categoria":"Crise Financeira","sp500":-6.8},
    "🇧🇷 8 de Janeiro 2023 — Brasil":           {"start":"2023-01-06","end":"2023-01-13","desc":"Invasão do Congresso e STF. Ibovespa cai, real se deprecia.","cor":SUCCESS,"categoria":"Político","sp500":-0.5},
    "⚔️ Ataque Iran–Israel — Abr 2024":         {"start":"2024-04-13","end":"2024-04-22","desc":"Irã lança 300+ drones e mísseis contra Israel.","cor":AMBER,"categoria":"Geopolítico","sp500":-3.1},
    "📉 Crash Agosto 2024 — Carry Trade":       {"start":"2024-08-01","end":"2024-08-09","desc":"Banco do Japão sobe juros. Nikkei -12% num dia.","cor":DANGER,"categoria":"Mercado","sp500":-8.5},
    "🇺🇸 Tarifaço Trump — Abr 2025":            {"start":"2025-04-02","end":"2025-04-09","desc":"EUA impõem tarifas de 10–145%. Pânico global.","cor":DANGER,"categoria":"Geopolítico","sp500":-12.0},
}

EMOJI_CAT = {"Pandemia":"🦠","Crise Financeira":"🏦","Mercado":"⚡","Geopolítico":"⚔️",
             "Monetário":"🏛️","Cripto":"💎","Político":"🗳️","Crédito":"💳"}

# ── FUNÇÕES FINANCEIRAS ───────────────────────────────────
def black_scholes(S, K, T, r, sigma, tipo="call"):
    """Preço de opção europeia — Black-Scholes (seção 2 do notebook)."""
    if T <= 0: return max(S-K,0) if tipo=="call" else max(K-S,0)
    if sigma <= 0: return max(S-K*np.exp(-r*T),0) if tipo=="call" else max(K*np.exp(-r*T)-S,0)
    d1 = (np.log(S/K)+(r+.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    if tipo=="call": return S*norm.cdf(d1)-K*np.exp(-r*T)*norm.cdf(d2)
    return K*np.exp(-r*T)*norm.cdf(-d2)-S*norm.cdf(-d1)

def todas_gregas(S, K, T, r, sigma, tipo="call"):
    """Delta, Gamma, Vega, Theta, Rho — seções 3 e 8 do notebook."""
    if T<=0 or sigma<=0: return 0.,0.,0.,0.,0.
    d1 = (np.log(S/K)+(r+.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2 = d1-sigma*np.sqrt(T)
    delta = norm.cdf(d1) if tipo=="call" else norm.cdf(d1)-1
    gamma = norm.pdf(d1)/(S*sigma*np.sqrt(T))
    vega  = S*norm.pdf(d1)*np.sqrt(T)
    if tipo=="call":
        theta = (-S*norm.pdf(d1)*sigma/(2*np.sqrt(T)) - r*K*np.exp(-r*T)*norm.cdf(d2))/252
        rho   = K*T*np.exp(-r*T)*norm.cdf(d2)/100
    else:
        theta = (-S*norm.pdf(d1)*sigma/(2*np.sqrt(T)) + r*K*np.exp(-r*T)*norm.cdf(-d2))/252
        rho   = -K*T*np.exp(-r*T)*norm.cdf(-d2)/100
    return float(delta), float(gamma), float(vega), float(theta), float(rho)

@st.cache_data(ttl=600, show_spinner=False)
def baixar(tickers_tuple, ini, fim=None):
    tickers = list(tickers_tuple)
    try:
        kw = dict(start=ini, auto_adjust=True, progress=False, threads=False)
        if fim: kw["end"] = fim
        df   = yf.download(tickers, **kw)
        p    = df["Close"] if "Close" in df.columns else df
        if isinstance(p, pd.Series): p = p.to_frame(tickers[0])
        p = p.dropna(how="all")
        if not p.empty: return p, None
    except: pass
    frames={}
    for t in tickers:
        try:
            kw2=dict(start=ini, auto_adjust=True)
            if fim: kw2["end"]=fim
            h=yf.Ticker(t).history(**kw2)
            if not h.empty: frames[t]=h["Close"]
        except: pass
    if frames:
        df2=pd.DataFrame(frames).dropna(how="all")
        if not df2.empty: return df2, None
    return None, "Falha ao baixar dados."

def chart_rc():
    plt.rcParams.update({
        "figure.facecolor":CARD,"axes.facecolor":CARD,"axes.edgecolor":BORDER,
        "axes.labelcolor":MUTED,"axes.titlecolor":TEXT,"text.color":TEXT,
        "xtick.color":MUTED,"ytick.color":MUTED,"grid.color":BORDER,"grid.alpha":.5,
        "axes.spines.top":False,"axes.spines.right":False,
        "legend.facecolor":CARD,"legend.edgecolor":BORDER,"legend.labelcolor":TEXT,
        "font.family":"monospace","figure.dpi":110,
    })

# ── SIDEBAR ───────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <h2 style="color:{PRIMARY};font-weight:800;margin:0;font-size:1.25rem">⚡ Risk Lab</h2>
    <p style="color:{MUTED};font-size:.76rem;margin:.15rem 0 .75rem">
        VaR · Black-Scholes · Stress Test · v6.0</p>
    <hr style="border:none;border-top:1px solid {BORDER};margin:0 0 1rem">
    """, unsafe_allow_html=True)

    # ── MERCADO ──
    st.markdown(f'<p style="color:{PRIMARY};font-size:.65rem;font-weight:700;text-transform:uppercase;letter-spacing:.1em;margin-bottom:.3rem">▸ Mercado</p>', unsafe_allow_html=True)
    mercado = st.radio("mercado", ["🇧🇷 Brasil","🇺🇸 EUA/Global","✏️ Manual"],
                       horizontal=False, label_visibility="collapsed")

    # ── ATIVOS ──
    st.markdown(f'<p style="color:{PRIMARY};font-size:.65rem;font-weight:700;text-transform:uppercase;letter-spacing:.1em;margin-top:.9rem;margin-bottom:.3rem">▸ Carteira</p>', unsafe_allow_html=True)

    if mercado == "✏️ Manual":
        raw = st.text_input("Tickers (vírgula)", "PETR4.SA, VALE3.SA, ITUB4.SA")
        tickers_sel = [t.strip().upper() for t in raw.split(",") if t.strip()]
        # pesos customizáveis (Ex.2 do notebook)
        pesos_str = st.text_input("Pesos % (vírgula) — vazio = igual", "",
                                   help="Ex: 30, 30, 25, 15  |  Deixe em branco para pesos iguais")
    else:
        cat = ACOES_BR if "Brasil" in mercado else ACOES_US
        opt_map = {f"{tk} — {nm}": tk for tk,nm in cat.items()}
        defs_br = ["PETR4.SA — Petrobras PN","VALE3.SA — Vale ON","ITUB4.SA — Itaú Unibanco PN"]
        defs_us = ["AAPL — Apple","MSFT — Microsoft","NVDA — NVIDIA"]
        defs    = [d for d in (defs_br if "Brasil" in mercado else defs_us) if d in opt_map]
        sel     = st.multiselect("Ativos (máx. 8)", list(opt_map.keys()),
                                  default=defs, max_selections=8, label_visibility="collapsed")
        tickers_sel = [opt_map[d] for d in sel]
        pesos_str   = st.text_input("Pesos % (vírgula) — vazio = igual", "",
                                     help="Ex: 30, 30, 25, 15  |  Deixe em branco para pesos iguais")

    qty_str  = st.text_input("Quantidades (mesma ordem)", ", ".join(["1000"]*len(tickers_sel)))
    data_ini = st.date_input("Data início", pd.to_datetime("2022-01-01"))

    # ── VaR ──
    st.markdown(f'<p style="color:{PRIMARY};font-size:.65rem;font-weight:700;text-transform:uppercase;letter-spacing:.1em;margin-top:.9rem;margin-bottom:.3rem">▸ Parâmetros VaR</p>', unsafe_allow_html=True)
    nivel     = st.selectbox("Nível de confiança", [0.90,0.95,0.975,0.99], index=1,
                              format_func=lambda x:f"{x*100:.1f}%")
    horizonte = st.number_input("Horizonte (dias úteis)", 1, 30, 1)

    # ── OPÇÃO ──
    st.markdown(f'<p style="color:{PRIMARY};font-size:.65rem;font-weight:700;text-transform:uppercase;letter-spacing:.1em;margin-top:.9rem;margin-bottom:.3rem">▸ Opção Black-Scholes</p>', unsafe_allow_html=True)
    opt_ativo = st.selectbox("Ativo-objeto da opção", tickers_sel if tickers_sel else ["PETR4.SA"])
    opt_tipo  = st.selectbox("Tipo", ["call","put"])
    opt_qty   = st.number_input("Qtd. de opções", 0, value=1000, step=100)
    strike    = st.number_input("Strike (K)", 1.0, value=40.0, step=0.5)
    rf        = st.number_input("Taxa livre de risco a.a.", 0.0, 1.0, 0.105, 0.005, "%.3f")
    T_exp     = st.number_input("Vencimento (anos)", 0.01, 5.0, 0.25, 0.05, "%.2f")

    st.markdown("<br>", unsafe_allow_html=True)
    calcular = st.button("▶  CALCULAR VaR", use_container_width=True)

# ── HEADER ───────────────────────────────────────────────
st.markdown(f"""
<div style="display:flex;align-items:center;gap:1rem;padding:1.5rem 0 2rem;
            border-bottom:1px solid {BORDER};margin-bottom:1.5rem">
  <div style="width:52px;height:52px;border-radius:14px;flex-shrink:0;
              background:linear-gradient(135deg,rgba(34,211,238,.2),rgba(167,139,250,.15));
              border:1px solid {PRIMARY}40;display:flex;align-items:center;
              justify-content:center;font-size:1.7rem">⚡</div>
  <div>
    <p style="color:{MUTED};font-size:.72rem;text-transform:uppercase;
              letter-spacing:.12em;margin:0;font-weight:600">
      Modelagem Aplicada ao Mercado Financeiro</p>
    <h1 style="margin:.2rem 0 0;font-size:1.7rem">
      Risk Lab <span style="color:{MUTED};font-weight:400">— Value at Risk v6.0</span></h1>
  </div>
  <div style="margin-left:auto;display:flex;gap:.4rem;flex-wrap:wrap;justify-content:flex-end">
    {''.join([badge(t) for t in tickers_sel[:6]])}
  </div>
</div>
""", unsafe_allow_html=True)

if not calcular:
    st.markdown(f"""
    <div style="text-align:center;padding:4rem 2rem;background:{CARD};
                border:1px solid {BORDER};border-radius:16px">
      <div style="font-size:3rem">📉</div>
      <h2 style="margin-top:1rem">Configure a carteira para começar</h2>
      <p style="color:{MUTED};margin-bottom:1.5rem">
        Selecione os ativos na barra lateral e clique em
        <b style="color:{PRIMARY}">▶ Calcular VaR</b>.
      </p>
      <div style="display:flex;gap:.5rem;justify-content:center;flex-wrap:wrap">
        {''.join([badge(t) for t in list(ACOES_BR.keys())[:10]])}
      </div>
    </div>""", unsafe_allow_html=True)
    st.stop()

if not tickers_sel:
    st.error("Selecione ao menos um ativo na barra lateral.")
    st.stop()

# ── DADOS ────────────────────────────────────────────────
with st.spinner("Conectando ao mercado…"):
    precos, erro = baixar(tuple(tickers_sel), str(data_ini))

if erro or precos is None:
    st.error(f"Erro ao baixar dados: {erro}")
    st.stop()

tickers = [t for t in tickers_sel if t in precos.columns]
if not tickers:
    st.error("Nenhum ticker retornou dados. Verifique os símbolos.")
    st.stop()

try:   qtds = [int(q.strip()) for q in qty_str.split(",")]
except: qtds = [1000]*len(tickers)
while len(qtds) < len(tickers): qtds.append(1000)
quantidades = dict(zip(tickers, qtds))

precos   = precos[tickers].dropna()
retornos = precos.pct_change().dropna()
ultimos  = precos.iloc[-1]

# Pesos (customizáveis — Ex.2 do notebook)
try:
    pw = [float(x.strip()) for x in pesos_str.split(",") if x.strip()]
    if len(pw) == len(tickers) and abs(sum(pw)-100) < 1:
        pesos_carteira = np.array(pw)/100
        pesos_modo = "customizados"
    else:
        raise ValueError
except:
    vals = np.array([quantidades[t]*float(ultimos[t]) for t in tickers])
    pesos_carteira = vals/vals.sum()
    pesos_modo = "por valor de mercado"

# ── CÁLCULOS PRINCIPAIS ───────────────────────────────────
v_acoes  = sum(quantidades[t]*float(ultimos[t]) for t in tickers)
S0       = float(ultimos[opt_ativo]) if opt_ativo in tickers else float(ultimos.iloc[0])
vol_anual= float(retornos[opt_ativo].std()*np.sqrt(252)) if opt_ativo in retornos.columns else 0.3
preco_op = black_scholes(S0, strike, T_exp, rf, vol_anual, opt_tipo)
v_op     = opt_qty*preco_op
v_total  = v_acoes+v_op

# Retorno da carteira (ponderado)
ret_cart = retornos[tickers].dot(pesos_carteira)
mu_c, sig_c = float(ret_cart.mean()), float(ret_cart.std())
pct          = 1-nivel

# VaR PARAMÉTRICO via volatilidade da carteira — fórmula: -(mu*h + z*sig*sqrt(h))*V
z_var        = norm.ppf(1-nivel)
var_param    = -(mu_c*horizonte + z_var*sig_c*np.sqrt(horizonte))*v_acoes

# VaR PARAMÉTRICO via matriz de covariância (w' Σ w) — seção 3 e Ex.1 do notebook
cov_mat      = retornos[tickers].cov()
sig_cov      = float(np.sqrt(pesos_carteira @ cov_mat.values @ pesos_carteira))
var_param_cov= -(mu_c*horizonte + z_var*sig_cov*np.sqrt(horizonte))*v_acoes

# VaR HISTÓRICO
var_hist     = -float(np.percentile(ret_cart, pct*100))*v_acoes

# VaR FULL VALUATION (seção 9 do notebook) — reprecifica Black-Scholes em cada cenário
cenarios_pnl = []
for i in range(len(retornos)):
    choque         = retornos[tickers].iloc[i]
    novos_precos   = ultimos*(1+choque)
    novo_v_acoes   = sum(quantidades[t]*float(novos_precos[t]) for t in tickers)
    S_cen          = float(novos_precos[opt_ativo]) if opt_ativo in tickers else S0
    T_cen          = max(T_exp-horizonte/252, 0)
    novo_op        = black_scholes(S_cen, strike, T_cen, rf, vol_anual, opt_tipo)
    cenarios_pnl.append((novo_v_acoes+opt_qty*novo_op)-v_total)
cenarios_pnl = np.array(cenarios_pnl)
var_full     = -float(np.percentile(cenarios_pnl, pct*100))

# Expected Shortfall (CVaR)
es_hist = -float(ret_cart[ret_cart <= np.percentile(ret_cart, pct*100)].mean())*v_acoes

# Gregas completas (5 gregas — seção 8 do notebook)
delta_v, gamma_v, vega_v, theta_v, rho_v = todas_gregas(S0, strike, T_exp, rf, vol_anual, opt_tipo)

chart_rc()

# ── ABAS ─────────────────────────────────────────────────
tabs = st.tabs([
    "  📊 Resumo  ",
    "  📐 Covariância  ",
    "  📈 Gráficos  ",
    "  🎯 Gregas & Opção  ",
    "  🔢 Janelas & ES  ",
    "  🌡️ Stress Test  ",
    "  📋 Versões  ",
])
tab1,tab2,tab3,tab4,tab5,tab6,tab7 = tabs

# ════════════════════════════════════════════════
# TAB 1 — RESUMO
# ════════════════════════════════════════════════
with tab1:
    # Composição
    st.markdown(section("Composição da Carteira",
        f"{len(tickers)} ativos · pesos {pesos_modo} · {opt_tipo.upper()} {opt_ativo}"),
        unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    c1.markdown(kpi("Ações",        f"R$ {v_acoes:,.0f}",  f"{len(tickers)} ativos",           PRIMARY), unsafe_allow_html=True)
    c2.markdown(kpi("Opções",       f"R$ {v_op:,.0f}",     f"{opt_qty:,} {opt_tipo}s · K={strike}", SUCCESS), unsafe_allow_html=True)
    c3.markdown(kpi("Total",        f"R$ {v_total:,.0f}",  "valor de mercado",                 AMBER),   unsafe_allow_html=True)
    c4.markdown(kpi("Vol. diária",  f"{sig_c*100:.2f}%",   f"anual {sig_c*np.sqrt(252)*100:.1f}%", VIOLET), unsafe_allow_html=True)

    # VaR — três métodos (seção 10 do notebook)
    st.markdown(section("Value at Risk — Comparativo dos 3 Métodos",
        f"IC {nivel*100:.1f}% · horizonte {horizonte}d"), unsafe_allow_html=True)
    st.markdown(info(
        "<b>VaR Paramétrico:</b> assume distribuição normal dos retornos. "
        "Simples e rápido, mas pode subestimar caudas gordas. "
        "<b>VaR Histórico:</b> usa percentil empírico — não exige normalidade, depende da janela. "
        "<b>Full Valuation:</b> reprecifica Black-Scholes em cada cenário — capta não-linearidade das opções."
    ), unsafe_allow_html=True)
    v1,v2,v3 = st.columns(3)
    v1.markdown(var_card("Paramétrico (cov)",   f"R$ {var_param_cov:,.0f}", f"{var_param_cov/v_total*100:.2f}%", PRIMARY, "w′Σw · dist. Normal"),      unsafe_allow_html=True)
    v2.markdown(var_card("Histórico",           f"R$ {var_hist:,.0f}",     f"{var_hist/v_total*100:.2f}%",     SUCCESS, "Percentil empírico histórico"), unsafe_allow_html=True)
    v3.markdown(var_card("Full Valuation",      f"R$ {var_full:,.0f}",     f"{var_full/v_total*100:.2f}%",     AMBER,   "Reprecificação Black-Scholes"),  unsafe_allow_html=True)

    # Comparativo em tabela + gráfico de barras (seção 10 do notebook)
    st.markdown(section("Tabela Comparativa dos Métodos"), unsafe_allow_html=True)
    df_comp = pd.DataFrame({
        "Método":         ["VaR Paramétrico (cov.)","VaR Histórico","VaR Full Valuation"],
        "VaR (R$)":       [f"R$ {var_param_cov:,.0f}", f"R$ {var_hist:,.0f}", f"R$ {var_full:,.0f}"],
        "VaR (% portf.)": [f"{var_param_cov/v_total*100:.2f}%",f"{var_hist/v_total*100:.2f}%",f"{var_full/v_total*100:.2f}%"],
        "Hipótese":       ["Normal · linear","Empírica · histórica","BS · não-linear"],
        "Inclui opção?":  ["Não (linear)","Não (linear)","✅ Sim"],
    })
    st.dataframe(df_comp, use_container_width=True, hide_index=True)

    fig_bar, ax_bar = plt.subplots(figsize=(8,3.5))
    labels = ["Param. (cov)","Histórico","Full Val."]
    vals   = [var_param_cov, var_hist, var_full]
    colors = [PRIMARY, SUCCESS, AMBER]
    bars   = ax_bar.bar(labels, vals, color=colors, alpha=.85, width=.5)
    for b in bars:
        h=b.get_height()
        ax_bar.text(b.get_x()+b.get_width()/2, h+h*.01, f"R$ {h:,.0f}",
                    ha="center", va="bottom", fontsize=9, color=TEXT)
    ax_bar.set_ylabel("VaR (R$)"); ax_bar.set_title("Comparativo VaR — 3 Métodos")
    ax_bar.grid(axis="y")
    st.pyplot(fig_bar); plt.close(fig_bar)

    # Tabela de posições
    st.markdown(section("Posições Individuais"), unsafe_allow_html=True)
    rows_pos=[]
    for i,t in enumerate(tickers):
        p0=float(ultimos[t]); rt=retornos[t]; val=quantidades[t]*p0
        vi=-(rt.mean()+norm.ppf(1-nivel)*rt.std())*val
        rows_pos.append({
            "Ticker":t,"Nome":ALL_ACOES.get(t,t),
            "Qtd.":f"{quantidades[t]:,}","Preço":f"R$ {p0:.2f}",
            "Valor":f"R$ {val:,.0f}","Peso":f"{pesos_carteira[i]*100:.1f}%",
            "Vol. diária":f"{rt.std()*100:.2f}%","VaR indiv.":f"R$ {vi:,.0f}",
        })
    st.dataframe(pd.DataFrame(rows_pos), use_container_width=True, hide_index=True)

    # Interpretação (seção 12 do notebook)
    st.markdown(section("Interpretação Didática"), unsafe_allow_html=True)
    st.markdown(info(
        f"<b>VaR de R$ {var_hist:,.0f} com {nivel*100:.0f}% de confiança significa:</b> "
        f"em condições normais de mercado, espera-se que a perda diária da carteira "
        f"<b>não ultrapasse R$ {var_hist:,.0f} em {nivel*100:.0f}% dos dias</b>. "
        f"Ou seja, existe {pct*100:.0f}% de probabilidade de a perda ser maior que esse valor."
    ), unsafe_allow_html=True)
    if var_full > var_hist*1.05:
        st.markdown(warn(
            f"O Full Valuation (R$ {var_full:,.0f}) é maior que o Histórico (R$ {var_hist:,.0f}) — "
            "a opção introduz não-linearidade que o VaR Histórico de ações não captura."
        ), unsafe_allow_html=True)

# ════════════════════════════════════════════════
# TAB 2 — MATRIZ DE COVARIÂNCIA (Ex.1, Teoria 3)
# ════════════════════════════════════════════════
with tab2:
    st.markdown(section("Matriz de Covariância e Correlação",
        "σ_p = √(w′Σw) — fórmula do VaR Paramétrico multi-ativo"), unsafe_allow_html=True)

    st.markdown(info(
        "O VaR Paramétrico para carteiras multi-ativo usa a matriz de covariância para capturar "
        "a correlação entre os ativos. A volatilidade da carteira é: "
        "<b>σ_p = √(w′ Σ w)</b>, onde w são os pesos e Σ é a matriz de covariância."
    ), unsafe_allow_html=True)

    col_cov, col_corr = st.columns(2)

    with col_cov:
        st.markdown(f'<div style="color:{MUTED};font-size:.72rem;font-weight:700;text-transform:uppercase;margin-bottom:.5rem">Matriz de Covariância (diária)</div>', unsafe_allow_html=True)
        cov_pct = (cov_mat*10000).round(4)
        st.dataframe(cov_pct, use_container_width=True)
        st.markdown(f'<div style="color:{MUTED};font-size:.7rem;margin-top:.3rem">Valores × 10⁻⁴</div>', unsafe_allow_html=True)

    with col_corr:
        st.markdown(f'<div style="color:{MUTED};font-size:.72rem;font-weight:700;text-transform:uppercase;margin-bottom:.5rem">Matriz de Correlação</div>', unsafe_allow_html=True)
        corr_mat = retornos[tickers].corr().round(4)
        st.dataframe(corr_mat, use_container_width=True)

    # Heatmap correlação
    if len(tickers) > 1:
        fig_hm, ax_hm = plt.subplots(figsize=(max(5,len(tickers)*1.2), max(4,len(tickers)*1.0)))
        corr_arr = corr_mat.values
        im = ax_hm.imshow(corr_arr, cmap="RdYlGn", vmin=-1, vmax=1)
        ax_hm.set_xticks(range(len(tickers))); ax_hm.set_xticklabels(tickers, rotation=45, ha="right")
        ax_hm.set_yticks(range(len(tickers))); ax_hm.set_yticklabels(tickers)
        for i in range(len(tickers)):
            for j in range(len(tickers)):
                ax_hm.text(j,i,f"{corr_arr[i,j]:.2f}",ha="center",va="center",fontsize=9,
                           color="black" if abs(corr_arr[i,j])<.6 else "white")
        plt.colorbar(im, ax=ax_hm)
        ax_hm.set_title("Heatmap de Correlação")
        fig_hm.tight_layout()
        st.pyplot(fig_hm); plt.close(fig_hm)

    # VaR decomposição por ativo
    st.markdown(section("Decomposição do Risco (Contribuição Marginal ao VaR)"), unsafe_allow_html=True)
    mcvar_rows=[]
    for i,t in enumerate(tickers):
        contrib = pesos_carteira[i]*sum(pesos_carteira[j]*float(cov_mat.loc[t,tickers[j]])
                                        for j in range(len(tickers)))/sig_cov
        mcvar_rows.append({"Ativo":t,"Peso":f"{pesos_carteira[i]*100:.1f}%",
                            "Vol. diária":f"{retornos[t].std()*100:.3f}%",
                            "Contrib. marginal":f"{contrib*100:.1f}%"})
    st.dataframe(pd.DataFrame(mcvar_rows), use_container_width=True, hide_index=True)

    st.markdown(info(
        f"<b>Volatilidade da carteira via w′Σw:</b> {sig_cov*100:.3f}% ao dia "
        f"({sig_cov*np.sqrt(252)*100:.2f}% ao ano) · "
        f"<b>VaR Paramétrico (covariância):</b> R$ {var_param_cov:,.0f}"
    ), unsafe_allow_html=True)

# ════════════════════════════════════════════════
# TAB 3 — GRÁFICOS
# ════════════════════════════════════════════════
with tab3:
    st.markdown(section("Distribuição Histórica dos Retornos da Carteira"), unsafe_allow_html=True)
    fig1,ax1=plt.subplots(figsize=(12,4))
    ax1.hist(ret_cart, bins=60, color=PRIMARY, alpha=.45, label="Retornos")
    tail=ret_cart[ret_cart<=np.percentile(ret_cart,pct*100)]
    ax1.hist(tail, bins=40, color=DANGER, alpha=.85, label=f"Cauda {pct*100:.0f}%")
    ax1.axvline(np.percentile(ret_cart,pct*100), color=SUCCESS, ls="--", lw=1.8, label=f"VaR Hist. {nivel*100:.0f}%")
    ax1.axvline(-(z_var*sig_cov), color=VIOLET, ls=":", lw=1.6, label="VaR Param. (cov)")
    ax1.set_xlabel("Retorno diário"); ax1.set_ylabel("Frequência")
    ax1.set_title("Distribuição Histórica dos Retornos — Cauda de Perda em Vermelho")
    ax1.legend()
    st.pyplot(fig1); plt.close(fig1)

    st.markdown(section("Preços Normalizados — Base 100"), unsafe_allow_html=True)
    fig2,ax2=plt.subplots(figsize=(12,4))
    cores=[PRIMARY,SUCCESS,AMBER,VIOLET,DANGER,"#fb923c","#f472b6","#38bdf8"]
    for i,t in enumerate(tickers):
        s=precos[t]/precos[t].iloc[0]*100
        ax2.plot(s.index, s.values, lw=1.8, color=cores[i%len(cores)], label=t)
    ax2.axhline(100,color=BORDER,lw=.8,ls="--")
    ax2.set_ylabel("Índice (base 100)"); ax2.legend()
    st.pyplot(fig2); plt.close(fig2)

    st.markdown(section("Distribuição de P&L — Full Valuation"), unsafe_allow_html=True)
    fig3,ax3=plt.subplots(figsize=(12,4))
    ax3.hist(cenarios_pnl, bins=60, color=AMBER, alpha=.45, label="P&L cenários")
    tail_pnl=cenarios_pnl[cenarios_pnl<=np.percentile(cenarios_pnl,pct*100)]
    ax3.hist(tail_pnl, bins=40, color=DANGER, alpha=.85, label=f"Cauda {pct*100:.0f}%")
    ax3.axvline(-var_full, color=AMBER, ls="--", lw=1.8, label=f"VaR Full R$ {var_full:,.0f}")
    ax3.set_xlabel("P&L da carteira (R$)"); ax3.set_ylabel("Frequência")
    ax3.set_title("Distribuição de P&L — Full Valuation (ações + opção BS)")
    ax3.legend()
    st.pyplot(fig3); plt.close(fig3)

    # VaR Rolling
    st.markdown(section("VaR Rolling (janela de 63 pregões)"), unsafe_allow_html=True)
    janela_roll = 63
    rolling_var=[]
    for i in range(janela_roll,len(ret_cart)):
        w=ret_cart.iloc[i-janela_roll:i]
        rolling_var.append(-np.percentile(w,pct*100)*v_acoes)
    fig4,ax4=plt.subplots(figsize=(12,3.5))
    ax4.plot(ret_cart.index[janela_roll:], rolling_var, color=PRIMARY, lw=1.5)
    ax4.axhline(var_hist, color=DANGER, ls="--", lw=1, label=f"VaR hist. total R$ {var_hist:,.0f}")
    ax4.set_ylabel("VaR (R$)"); ax4.set_title(f"VaR Histórico Rolling (janela {janela_roll}d)")
    ax4.legend()
    st.pyplot(fig4); plt.close(fig4)

# ════════════════════════════════════════════════
# TAB 4 — GREGAS & ANÁLISE DE SENSIBILIDADE
# ════════════════════════════════════════════════
with tab4:
    st.markdown(section("Gregas da Opção",
        f"{opt_tipo.upper()} {opt_ativo} · K={strike} · T={T_exp}a · σ={vol_anual*100:.1f}%"),
        unsafe_allow_html=True)

    st.markdown(info(
        "<b>Delta:</b> variação do preço da opção por R$1 no ativo. "
        "<b>Gamma:</b> variação do Delta (convexidade). "
        "<b>Vega:</b> sensibilidade à volatilidade (+1 p.p.). "
        "<b>Theta:</b> decaimento temporal (por dia). "
        "<b>Rho:</b> sensibilidade à taxa de juros (+1 p.p.)."
    ), unsafe_allow_html=True)

    g1,g2,g3,g4,g5 = st.columns(5)
    g1.markdown(kpi("Delta Δ",  f"{delta_v:.4f}", "exposição direcional",          PRIMARY), unsafe_allow_html=True)
    g2.markdown(kpi("Gamma Γ",  f"{gamma_v:.6f}", "convexidade / curvatura",       SUCCESS), unsafe_allow_html=True)
    g3.markdown(kpi("Vega ν",   f"{vega_v:.4f}",  "sens. à volatilidade (+1 p.p.)", AMBER),  unsafe_allow_html=True)
    g4.markdown(kpi("Theta Θ",  f"{theta_v:.4f}", "decaimento por dia (R$)",        VIOLET), unsafe_allow_html=True)
    g5.markdown(kpi("Rho ρ",    f"{rho_v:.4f}",   "sens. à taxa (+1 p.p.)",         MUTED),  unsafe_allow_html=True)

    # Call vs Put (Ex.7 do notebook)
    st.markdown(section("Call vs Put — Mesmo Strike",
        f"K={strike} · T={T_exp}a · {opt_ativo}"), unsafe_allow_html=True)
    pc=black_scholes(S0,strike,T_exp,rf,vol_anual,"call")
    pp=black_scholes(S0,strike,T_exp,rf,vol_anual,"put")
    dc,gc,vc2,tc2,rc2=todas_gregas(S0,strike,T_exp,rf,vol_anual,"call")
    dp,gp,vp2,tp2,rp2=todas_gregas(S0,strike,T_exp,rf,vol_anual,"put")
    df_cv=pd.DataFrame({
        "Métrica":["Preço BS","Delta Δ","Gamma Γ","Vega ν","Theta Θ","Rho ρ"],
        "Call":[f"{pc:.4f}",f"{dc:.4f}",f"{gc:.6f}",f"{vc2:.4f}",f"{tc2:.4f}",f"{rc2:.4f}"],
        "Put": [f"{pp:.4f}",f"{dp:.4f}",f"{gp:.6f}",f"{vp2:.4f}",f"{tp2:.4f}",f"{rp2:.4f}"],
    })
    st.dataframe(df_cv, use_container_width=True, hide_index=True)

    # Gráfico preço + delta
    fig_cv,axes=plt.subplots(1,2,figsize=(12,4))
    ps=np.linspace(S0*.65,S0*1.35,200)
    axes[0].plot(ps,[black_scholes(s,strike,T_exp,rf,vol_anual,"call") for s in ps],color=PRIMARY,lw=2,label="Call")
    axes[0].plot(ps,[black_scholes(s,strike,T_exp,rf,vol_anual,"put")  for s in ps],color=AMBER,  lw=2,label="Put")
    axes[0].axvline(S0,color=SUCCESS,ls=":",lw=1.2,label=f"S0={S0:.1f}")
    axes[0].axvline(strike,color=DANGER,ls="--",alpha=.6,label=f"K={strike}")
    axes[0].legend(); axes[0].set_title("Preço da Opção × Preço do Ativo")
    axes[0].set_xlabel("Preço do ativo"); axes[0].set_ylabel("Preço da opção")

    axes[1].plot(ps,[todas_gregas(s,strike,T_exp,rf,vol_anual,"call")[0] for s in ps],color=PRIMARY,lw=2,label="Δ Call")
    axes[1].plot(ps,[todas_gregas(s,strike,T_exp,rf,vol_anual,"put")[0]  for s in ps],color=AMBER,  lw=2,label="Δ Put")
    axes[1].axhline(0,color=BORDER,lw=.8)
    axes[1].axvline(strike,color=DANGER,ls="--",alpha=.6)
    axes[1].legend(); axes[1].set_title("Delta × Preço do Ativo")
    axes[1].set_xlabel("Preço do ativo"); axes[1].set_ylabel("Delta")
    st.pyplot(fig_cv); plt.close(fig_cv)

    # Análise de sensibilidade (seção 11 do notebook)
    st.markdown(section("Análise de Sensibilidade do Preço da Opção",
        "Seção 11 do notebook — não-linearidade das opções"), unsafe_allow_html=True)
    st.markdown(info(
        "Diferente das ações (payoff linear), a opção tem comportamento convexo. "
        "O gráfico abaixo mostra como o preço da opção varia com o preço do ativo-objeto."
    ), unsafe_allow_html=True)

    fig_sens, axes2 = plt.subplots(1,2,figsize=(12,4))
    ps2=np.linspace(S0*.7,S0*1.3,150)
    call_prices=[black_scholes(s,strike,T_exp,rf,vol_anual,"call") for s in ps2]
    put_prices =[black_scholes(s,strike,T_exp,rf,vol_anual,"put")  for s in ps2]
    axes2[0].plot(ps2,call_prices,color=PRIMARY,lw=2,label="Call")
    axes2[0].plot(ps2,put_prices, color=AMBER,  lw=2,label="Put")
    axes2[0].axvline(S0,    color=SUCCESS,ls=":",lw=1.2,label=f"S0 atual = {S0:.1f}")
    axes2[0].axvline(strike,color=DANGER, ls="--",alpha=.6,label=f"Strike = {strike}")
    axes2[0].fill_between(ps2,call_prices,alpha=.08,color=PRIMARY)
    axes2[0].set_xlabel("Preço do ativo objeto"); axes2[0].set_ylabel("Preço da opção")
    axes2[0].set_title("Sensibilidade do Preço (não-linearidade)")
    axes2[0].legend()

    # Gamma profile
    gammas=[todas_gregas(s,strike,T_exp,rf,vol_anual,opt_tipo)[1] for s in ps2]
    axes2[1].plot(ps2,gammas,color=VIOLET,lw=2)
    axes2[1].axvline(strike,color=DANGER,ls="--",alpha=.6,label=f"Strike = {strike}")
    axes2[1].set_xlabel("Preço do ativo objeto"); axes2[1].set_ylabel("Gamma")
    axes2[1].set_title("Perfil de Gamma (convexidade máxima no ATM)")
    axes2[1].legend()
    st.pyplot(fig_sens); plt.close(fig_sens)

# ════════════════════════════════════════════════
# TAB 5 — JANELAS HISTÓRICAS & ES (Ex.2,3,4)
# ════════════════════════════════════════════════
with tab5:
    st.markdown(section("Expected Shortfall — CVaR",
        "Perda média condicional além do VaR"), unsafe_allow_html=True)
    st.markdown(info(
        "O Expected Shortfall (ES / CVaR) responde à limitação do VaR: "
        "<b>quanto se perde, em média, nos piores cenários além do VaR?</b> "
        "É a média dos retornos abaixo do percentil de confiança."
    ), unsafe_allow_html=True)
    e1,e2,e3 = st.columns(3)
    e1.markdown(kpi("ES Histórico",  f"R$ {es_hist:,.0f}",         f"média cauda {pct*100:.0f}%",      DANGER), unsafe_allow_html=True)
    e2.markdown(kpi("Razão ES/VaR",  f"{es_hist/max(var_hist,1):.2f}×", "captura de cauda pesada",    VIOLET), unsafe_allow_html=True)
    e3.markdown(kpi("Diferença",     f"R$ {es_hist-var_hist:,.0f}", "perda extra além do VaR",         AMBER),  unsafe_allow_html=True)

    # Efeito da janela histórica (Ex.4 do notebook)
    st.markdown(section("Efeito da Janela Histórica — Exercício 4",
        "Como a janela escolhida altera o VaR"), unsafe_allow_html=True)
    st.markdown(info(
        "Se a janela não contiver períodos de crise, o VaR parecerá baixo. "
        "Por isso a escolha da janela é crítica em gestão de risco."
    ), unsafe_allow_html=True)

    janelas_def = [
        ("Desde 2020","2020-01-01"),("Desde 2022","2022-01-01"),
        ("Desde 2023","2023-01-01"),("Últimos 252 pregões",None),("Últimos 63 pregões","63"),
    ]
    rows_j=[]
    for nome,ini in janelas_def:
        if ini=="63":   sub=retornos.tail(63)
        elif ini is None: sub=retornos.tail(252)
        else: sub=retornos[retornos.index>=pd.to_datetime(ini)]
        if len(sub)<30: continue
        rp  = sub[tickers].dot(pesos_carteira)
        vp  = -(rp.mean()+norm.ppf(1-nivel)*rp.std())*v_acoes
        vh  = -np.percentile(rp,(1-nivel)*100)*v_acoes
        es_ = -rp[rp<=np.percentile(rp,(1-nivel)*100)].mean()*v_acoes
        sk_ = skew(rp); ku_ = kurtosis(rp)
        rows_j.append({
            "Janela":nome,"N obs.":len(sub),
            "VaR Param.":f"R$ {vp:,.0f}","VaR Hist.":f"R$ {vh:,.0f}",
            "CVaR":f"R$ {es_:,.0f}","Vol diária":f"{rp.std()*100:.2f}%",
            "Assimetria":f"{sk_:.2f}","Curtose":f"{ku_:.2f}",
        })
    st.dataframe(pd.DataFrame(rows_j), use_container_width=True, hide_index=True)

    # Comparativo níveis de confiança (Ex.2 do notebook)
    st.markdown(section("Sensibilidade ao Nível de Confiança — Exercício 2",
        "Como 95% vs 99% altera o capital de risco"), unsafe_allow_html=True)
    rows_ni=[]
    for ni in [0.90,0.95,0.975,0.99]:
        p_=1-ni
        vp_=-(mu_c*horizonte+norm.ppf(1-ni)*sig_cov*np.sqrt(horizonte))*v_acoes
        vh_=-np.percentile(ret_cart,p_*100)*v_acoes
        es__=-ret_cart[ret_cart<=np.percentile(ret_cart,p_*100)].mean()*v_acoes
        rows_ni.append({"IC":f"{ni*100:.1f}%","VaR Param.":f"R$ {vp_:,.0f}",
                         "VaR Hist.":f"R$ {vh_:,.0f}","CVaR":f"R$ {es__:,.0f}"})
    st.dataframe(pd.DataFrame(rows_ni), use_container_width=True, hide_index=True)

# ════════════════════════════════════════════════
# TAB 6 — STRESS TEST
# ════════════════════════════════════════════════
with tab6:
    st.markdown(section("🌡️ Stress Test",
        "VaR não substitui stress test — use ambos (Teoria 2 e Ex.8 do notebook)"),
        unsafe_allow_html=True)
    st.markdown(info(
        "O stress test simula cenários extremos que podem nunca ter ocorrido na janela histórica. "
        "Bancos e fundos usam VaR <b>junto com</b> stress test, expected shortfall e análise de cenários."
    ), unsafe_allow_html=True)

    stress_tab_a, stress_tab_b = st.tabs(["  🔨 Choque Manual (Ex.8)  ","  📰 Marcos Históricos  "])

    # ── Stress Manual (Exercício 8 do notebook) ──
    with stress_tab_a:
        st.markdown(section("Choque Manual — Exercício 8",
            "Simule choques de -5% e -10% como no enunciado"), unsafe_allow_html=True)

        col_s1, col_s2 = st.columns(2)
        with col_s1:
            choque_global = st.slider("Choque global em TODOS os ativos (%)", -50, 50, -5)
        with col_s2:
            st.markdown(f'<div style="color:{MUTED};font-size:.7rem;font-weight:600;text-transform:uppercase;letter-spacing:.08em;margin-bottom:.5rem">Choques individuais (%)</div>', unsafe_allow_html=True)

        choques_ind = {}
        cols_ch = st.columns(min(len(tickers),4))
        for i,t in enumerate(tickers):
            choques_ind[t] = cols_ch[i%4].number_input(f"{t}", value=-10 if i==0 else 0,
                                                         min_value=-100, max_value=100, step=5)

        # Cenário A: choque global
        val_ini_s = v_total
        val_chA = 0
        for t in tickers:
            p_novo = float(ultimos[t])*(1+choque_global/100)
            val_chA += quantidades[t]*p_novo
        S_chA = float(ultimos[opt_ativo])*(1+choque_global/100) if opt_ativo in tickers else S0
        val_chA += opt_qty*black_scholes(S_chA, strike, max(T_exp-1/252,0), rf, vol_anual, opt_tipo)
        pnl_A = val_chA - val_ini_s
        ret_A = pnl_A/val_ini_s*100

        # Cenário B: choques individuais
        val_chB = 0
        for t in tickers:
            p_novo = float(ultimos[t])*(1+choques_ind[t]/100)
            val_chB += quantidades[t]*p_novo
        S_chB = float(ultimos[opt_ativo])*(1+choques_ind.get(opt_ativo,0)/100) if opt_ativo in tickers else S0
        val_chB += opt_qty*black_scholes(S_chB, strike, max(T_exp-1/252,0), rf, vol_anual, opt_tipo)
        pnl_B = val_chB - val_ini_s
        ret_B = pnl_B/val_ini_s*100

        c_a,c_b,c_c = st.columns(3)
        c_a.markdown(kpi("Portfólio Atual",  f"R$ {val_ini_s:,.0f}", "antes do choque", PRIMARY), unsafe_allow_html=True)
        cor_a = SUCCESS if pnl_A>=0 else DANGER
        c_b.markdown(kpi(f"Cenário A ({choque_global:+}% global)",
                         f"R$ {pnl_A:+,.0f}", f"{ret_A:+.2f}%", cor_a), unsafe_allow_html=True)
        cor_b = SUCCESS if pnl_B>=0 else DANGER
        c_c.markdown(kpi("Cenário B (choques individuais)",
                         f"R$ {pnl_B:+,.0f}", f"{ret_B:+.2f}%", cor_b), unsafe_allow_html=True)

        # Comparar com VaR
        for label, pnl_v, val_v in [("A",pnl_A,var_hist),("B",pnl_B,var_hist)]:
            if pnl_v < 0 and abs(pnl_v) > val_v:
                st.markdown(warn(
                    f"Cenário {label}: perda de R$ {abs(pnl_v):,.0f} "
                    f"<b>excede o VaR Histórico</b> de R$ {val_v:,.0f}. "
                    "Isso demonstra por que stress test é complementar ao VaR."
                ), unsafe_allow_html=True)
            elif pnl_v < 0:
                st.markdown(info(
                    f"Cenário {label}: perda de R$ {abs(pnl_v):,.0f} "
                    f"está <b>dentro do VaR Histórico</b> de R$ {val_v:,.0f}."
                ), unsafe_allow_html=True)

        # Tabela de impacto por ativo
        st.markdown(f'<div style="margin-top:1rem">{section("Impacto por Ativo nos Dois Cenários")}</div>', unsafe_allow_html=True)
        rows_stress=[]
        for t in tickers:
            p0=float(ultimos[t])
            pA=p0*(1+choque_global/100); pB=p0*(1+choques_ind[t]/100)
            rows_stress.append({
                "Ativo":t,"Preço Atual":f"R$ {p0:.2f}",
                f"Preço Cen.A ({choque_global:+}%)":f"R$ {pA:.2f}",
                f"P&L Cen.A":f"R$ {quantidades[t]*(pA-p0):+,.0f}",
                f"Preço Cen.B ({choques_ind[t]:+}%)":f"R$ {pB:.2f}",
                f"P&L Cen.B":f"R$ {quantidades[t]*(pB-p0):+,.0f}",
            })
        st.dataframe(pd.DataFrame(rows_stress), use_container_width=True, hide_index=True)

    # ── Marcos Históricos ──
    with stress_tab_b:
        st.markdown(section("Marcos Históricos Globais",
            "Selecione um evento e veja o impacto real nos seus ativos"), unsafe_allow_html=True)

        cat_all  = sorted(set(v["categoria"] for v in STRESS_EVENTS.values()))
        col_f1,col_f2 = st.columns([1,2])
        with col_f1:
            cat_sel = st.multiselect("Categoria", cat_all, default=cat_all)
        ev_filt = {k:v for k,v in STRESS_EVENTS.items() if v["categoria"] in cat_sel}
        with col_f2:
            ev_sel = st.selectbox("Evento", list(ev_filt.keys()))
        ev = ev_filt[ev_sel]

        st.markdown(f"""
        <div style="background:{CARD};border:1px solid {BORDER};
                    border-left:5px solid {ev['cor']};border-radius:0 12px 12px 0;
                    padding:1.1rem 1.4rem;margin:.75rem 0">
          <div style="display:flex;align-items:flex-start;gap:1.5rem;flex-wrap:wrap">
            <div style="flex:1;min-width:200px">
              <div style="font-size:1rem;font-weight:700;color:{TEXT}">{ev_sel}</div>
              <div style="color:{MUTED};font-size:.82rem;margin-top:.35rem;line-height:1.5">{ev['desc']}</div>
              <div style="margin-top:.5rem">
                <span style="background:{ev['cor']}18;color:{ev['cor']};border:1px solid {ev['cor']}30;
                             border-radius:4px;padding:.1rem .5rem;font-size:.68rem;font-weight:700">
                    {EMOJI_CAT.get(ev['categoria'],'📌')} {ev['categoria']}
                </span>
                <span style="color:{MUTED};font-size:.72rem;margin-left:.6rem">
                    {ev['start']} → {ev['end']}
                </span>
              </div>
            </div>
            <div style="text-align:right;min-width:120px">
              <div style="color:{MUTED};font-size:.62rem;text-transform:uppercase;font-weight:600">S&P 500 no período</div>
              <div style="font-size:2rem;font-weight:800;font-family:monospace;
                          color:{DANGER if ev['sp500']<0 else SUCCESS}">{ev['sp500']:+.1f}%</div>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)

        with st.spinner(f"Buscando dados {ev['start']} → {ev['end']}…"):
            ini_ev=(pd.to_datetime(ev["start"])-pd.Timedelta(days=10)).strftime("%Y-%m-%d")
            precos_ev, erro_ev = baixar(tuple(tickers), ini_ev, ev["end"])

        if erro_ev or precos_ev is None or precos_ev.empty:
            st.warning("⚠️ Dados indisponíveis para este período.")
        else:
            tks_ev=[t for t in tickers if t in precos_ev.columns]
            preco_per=precos_ev[tks_ev].dropna(how="all")
            preco_per=preco_per[preco_per.index>=pd.to_datetime(ev["start"])]
            if len(preco_per)<2:
                st.warning("Dados insuficientes para o período.")
            else:
                ini_p=preco_per.iloc[0]; fim_p=preco_per.iloc[-1]
                st.markdown(section("Impacto por Ativo",f"{len(preco_per)} pregões"), unsafe_allow_html=True)
                cols_ev=st.columns(len(tks_ev))
                for i,t in enumerate(tks_ev):
                    r=(float(fim_p[t])-float(ini_p[t]))/float(ini_p[t])*100
                    pl=quantidades.get(t,1000)*(float(fim_p[t])-float(ini_p[t]))
                    cols_ev[i].markdown(kpi(ALL_ACOES.get(t,t),f"{r:+.1f}%",
                                            f"P&L R$ {pl:+,.0f}",SUCCESS if r>=0 else DANGER),
                                        unsafe_allow_html=True)
                v_ini=sum(quantidades.get(t,1000)*float(ini_p[t]) for t in tks_ev)
                v_fim=sum(quantidades.get(t,1000)*float(fim_p[t]) for t in tks_ev)
                ret_p=(v_fim-v_ini)/v_ini*100; pnl_p=v_fim-v_ini
                cobriu=abs(pnl_p)<=var_hist
                st.markdown(f"""
                <div style="display:flex;gap:1rem;margin:1rem 0;flex-wrap:wrap">
                  <div style="background:{CARD};border:1px solid {BORDER};border-radius:10px;padding:1rem 1.4rem;flex:1;min-width:145px">
                    <div style="color:{MUTED};font-size:.62rem;font-weight:600;text-transform:uppercase">Início</div>
                    <div style="font-size:1.35rem;font-weight:700;font-family:monospace">R$ {v_ini:,.0f}</div>
                  </div>
                  <div style="background:{CARD};border:1px solid {BORDER};border-radius:10px;padding:1rem 1.4rem;flex:1;min-width:145px">
                    <div style="color:{MUTED};font-size:.62rem;font-weight:600;text-transform:uppercase">Fim</div>
                    <div style="font-size:1.35rem;font-weight:700;font-family:monospace">R$ {v_fim:,.0f}</div>
                  </div>
                  <div style="background:{CARD};border:1px solid {'#15803d' if ret_p>=0 else '#7f1d1d'};border-radius:10px;padding:1rem 1.4rem;flex:1;min-width:145px">
                    <div style="color:{MUTED};font-size:.62rem;font-weight:600;text-transform:uppercase">Variação</div>
                    <div style="font-size:1.35rem;font-weight:700;font-family:monospace;color:{SUCCESS if ret_p>=0 else DANGER}">{ret_p:+.2f}%</div>
                    <div style="color:{MUTED};font-size:.72rem">R$ {pnl_p:+,.0f}</div>
                  </div>
                  <div style="background:{CARD};border:1px solid {'#15803d' if cobriu else '#7f1d1d'};border-radius:10px;padding:1rem 1.4rem;flex:1;min-width:145px">
                    <div style="color:{MUTED};font-size:.62rem;font-weight:600;text-transform:uppercase">VaR Cobriu?</div>
                    <div style="font-size:1.1rem;font-weight:700;color:{SUCCESS if cobriu else DANGER}">
                      {'✅ Sim' if cobriu else '❌ Excedeu VaR'}
                    </div>
                    <div style="color:{MUTED};font-size:.7rem">VaR hist. R$ {var_hist:,.0f}</div>
                  </div>
                </div>""", unsafe_allow_html=True)

                fig_ev,ax_ev=plt.subplots(figsize=(12,4))
                cores=[PRIMARY,SUCCESS,AMBER,VIOLET,DANGER,"#fb923c","#f472b6","#38bdf8"]
                for i,t in enumerate(tks_ev):
                    s=preco_per[t].dropna()
                    if len(s)>0: ax_ev.plot(s.index,s/s.iloc[0]*100,lw=2,color=cores[i%len(cores)],label=t)
                ax_ev.axhline(100,color=BORDER,lw=.8,ls="--",label="Base 100")
                ax_ev.set_ylabel("Índice (base 100)"); ax_ev.legend()
                ax_ev.set_title(f"Stress Test: {ev_sel}")
                st.pyplot(fig_ev); plt.close(fig_ev)

        # Tabela geral
        st.markdown(f'<div style="margin-top:2rem">{section("Todos os Marcos Históricos")}</div>', unsafe_allow_html=True)
        df_ev=pd.DataFrame([{"Evento":n,"Período":f"{i['start']} → {i['end']}",
            "Categoria":f"{EMOJI_CAT.get(i['categoria'],'📌')} {i['categoria']}",
            "S&P 500":f"{i['sp500']:+.1f}%"}
            for n,i in STRESS_EVENTS.items()])
        st.dataframe(df_ev, use_container_width=True, hide_index=True)

# ════════════════════════════════════════════════
# TAB 7 — HISTÓRICO DE VERSÕES
# ════════════════════════════════════════════════
with tab7:
    st.markdown(section("📋 Histórico de Versões", "Evolução do Risk Lab"), unsafe_allow_html=True)

    versoes=[
        {"version":"v6.0","date":"2025","title":"Auditoria Completa — Alinhamento ao Notebook","cor":PRIMARY,"changes":[
            ("✨ Novo","Matriz de covariância w′Σw — aba dedicada com heatmap de correlação",SUCCESS),
            ("✨ Novo","Theta Θ e Rho ρ adicionados — agora todas as 5 gregas do notebook",SUCCESS),
            ("✨ Novo","Análise de sensibilidade da opção (seção 11 do notebook) — gráfico de não-linearidade",SUCCESS),
            ("✨ Novo","Perfil de Gamma (convexidade máxima no ATM)",SUCCESS),
            ("✨ Novo","Stress Test manual: choques globais e individuais por ativo (Ex.8)",SUCCESS),
            ("✨ Novo","Pesos customizáveis por ativo — suporte ao Ex.2 (30/30/25/15%)",SUCCESS),
            ("✨ Novo","Tabela comparativa dos 3 métodos com gráfico de barras (seção 10)",SUCCESS),
            ("✨ Novo","Interpretação didática automática de cada resultado (seção 12)",SUCCESS),
            ("✨ Novo","Sensibilidade ao nível de confiança — tabela 90/95/97.5/99%",SUCCESS),
            ("✨ Novo","VaR Rolling 63 dias no gráfico de série temporal",SUCCESS),
            ("🔧 Melhoria","Contribuição marginal ao VaR por ativo (decomposição de risco)",AMBER),
            ("🔧 Melhoria","Caixas info/warn com interpretações didáticas em todas as abas",AMBER),
        ]},
        {"version":"v5.1","date":"2025","title":"Sidebar Fix","cor":VIOLET,"changes":[
            ("🐛 Fix","Remoção de [class*='css'] que ocultava sidebar",DANGER),
            ("🐛 Fix","initial_sidebar_state='expanded' forçando abertura",DANGER),
        ]},
        {"version":"v5.0","date":"2025","title":"Stress Test Histórico & Seletor de Ativos","cor":AMBER,"changes":[
            ("✨ Novo","14 marcos históricos globais com impacto real nos ativos",SUCCESS),
            ("✨ Novo","Seletor visual: 60+ ações BR, EUA, ETFs e cripto",SUCCESS),
        ]},
        {"version":"v2.0","date":"2024","title":"Black-Scholes & CVaR","cor":SUCCESS,"changes":[
            ("✨ Novo","Precificação Black-Scholes · gregas Delta Gamma Vega",SUCCESS),
            ("✨ Novo","Expected Shortfall (CVaR) · Full Valuation não-linear",SUCCESS),
        ]},
        {"version":"v1.0","date":"2024","title":"Versão Inicial","cor":MUTED,"changes":[
            ("✨ Novo","VaR Paramétrico e Histórico · Yahoo Finance · gráfico distribuição",SUCCESS),
        ]},
    ]

    for v in versoes:
        st.markdown(f"""
        <div style="margin-bottom:1.5rem;padding:1.25rem;background:{CARD};
                    border:1px solid {BORDER};border-left:4px solid {v['cor']};border-radius:0 10px 10px 0">
          <div style="display:flex;align-items:center;gap:.75rem;margin-bottom:.75rem">
            <span style="background:{v['cor']}18;color:{v['cor']};border:1px solid {v['cor']}40;
                         border-radius:6px;padding:.2rem .7rem;font-size:.78rem;font-weight:700;font-family:monospace">{v['version']}</span>
            <span style="color:{TEXT};font-size:.95rem;font-weight:700">{v['title']}</span>
            <span style="color:{MUTED};font-size:.72rem;margin-left:auto">{v['date']}</span>
          </div>
        """, unsafe_allow_html=True)
        for tipo,desc,cor in v["changes"]:
            st.markdown(f"""
            <div style="display:flex;align-items:flex-start;gap:.7rem;padding:.3rem 0;border-bottom:1px solid {BORDER}40">
              <span style="background:{cor}18;color:{cor};border:1px solid {cor}30;border-radius:4px;
                           padding:.1rem .45rem;font-size:.65rem;font-weight:700;white-space:nowrap">{tipo}</span>
              <span style="color:{TEXT};font-size:.82rem">{desc}</span>
            </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(section("📋 Cobertura dos Requisitos do Notebook"), unsafe_allow_html=True)
    reqs=[
        ("✅","Teoria 1","O que é VaR — definição e interpretação","Resumo (interpretação didática automática)"),
        ("✅","Teoria 2","Limitações do VaR — stress test complementar","Stress Test · caixas de aviso"),
        ("✅","Teoria 3","VaR Paramétrico — fórmula w′Σw","Resumo + aba Covariância"),
        ("✅","Teoria 4","VaR Histórico — percentil empírico","Resumo + Janelas"),
        ("✅","Teoria 5","VaR Full Valuation — reprecificação BS","Resumo + Gráficos"),
        ("✅","Teoria 6","Opções financeiras — Call e Put","Gregas & Opção"),
        ("✅","Teoria 7","Modelo Black-Scholes","Gregas & Opção"),
        ("✅","Teoria 8","5 Gregas — Delta Gamma Vega Theta Rho","Gregas & Opção"),
        ("✅","Exercício 1","VaR Paramétrico ações BR","Resumo"),
        ("✅","Exercício 2","Pesos customizáveis 30/30/25/15%","Sidebar (campo Pesos %)"),
        ("✅","Exercício 3","Comparação Paramétrico vs Histórico","Resumo — tabela + gráfico barras"),
        ("✅","Exercício 4","Efeito da janela histórica","Janelas & ES"),
        ("✅","Exercício 5","Call europeia — vol. anualizada + gregas","Gregas & Opção"),
        ("✅","Exercício 6","VaR Hist. ações vs Full Valuation","Resumo"),
        ("✅","Exercício 7","Call vs Put — gregas comparadas","Gregas & Opção"),
        ("✅","Exercício 8","Stress test choques -5% e -10%","Stress Test → Choque Manual"),
        ("⭐","Bônus","Stress Test com 14 marcos históricos reais","Stress Test → Marcos Históricos"),
        ("⭐","Bônus","Histórico de versões e roadmap","Esta aba"),
    ]
    df_reqs=pd.DataFrame(reqs,columns=["Status","Ref.","Requisito","Onde no app"])
    st.dataframe(df_reqs, use_container_width=True, hide_index=True)
