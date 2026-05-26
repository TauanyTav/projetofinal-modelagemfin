"""
Risk Lab — Value at Risk Calculator v5.0
Trabalho Final | Modelagem Aplicada ao Mercado Financeiro
Novidades: Seletor de ações, Stress Test histórico, Histórico de versões, UI premium.
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.stats import norm
import warnings
warnings.filterwarnings("ignore")

# ===================== PAGE CONFIG =====================
st.set_page_config(page_title="Risk Lab — VaR v5", page_icon="📉", layout="wide")

PRIMARY = "#22d3ee"
SUCCESS = "#34d399"
AMBER   = "#fbbf24"
DANGER  = "#f87171"
VIOLET  = "#a78bfa"
BG      = "#0b1220"
CARD    = "#111a2e"
BORDER  = "#1f2a44"
TEXT    = "#e5e7eb"
MUTED   = "#94a3b8"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

#MainMenu, footer, header, .stDeployButton, div[data-testid="stToolbar"] {{ display: none !important; visibility: hidden !important; }}

html, body, .stApp, [class*="css"] {{
    background: {BG} !important;
    color: {TEXT} !important;
    font-family: 'Inter', sans-serif !important;
}}
.stApp {{
    background-image:
        radial-gradient(ellipse 80% 50% at 0% 0%, rgba(34,211,238,0.05), transparent 60%),
        radial-gradient(ellipse 60% 40% at 100% 100%, rgba(167,139,250,0.05), transparent 60%) !important;
}}
.main .block-container {{ padding: 1.5rem 2rem 4rem; max-width: 1600px; }}
section[data-testid="stSidebar"] {{ background: {CARD} !important; border-right: 1px solid {BORDER} !important; }}

h1, h2, h3, h4 {{ color: {TEXT}; font-weight: 700; letter-spacing: -0.02em; }}

.stTextInput input, .stNumberInput input, .stDateInput input,
[data-baseweb="select"] > div {{
    background: {BG} !important; border: 1px solid {BORDER} !important;
    color: {TEXT} !important; border-radius: 8px !important;
    font-family: 'JetBrains Mono', monospace !important; font-size: 0.85rem !important;
}}
.stTextInput label, .stNumberInput label, .stDateInput label,
.stSelectbox label, .stSlider label, .stMultiSelect label {{
    color: {MUTED} !important; font-size: 0.7rem !important;
    text-transform: uppercase !important; letter-spacing: 0.1em !important; font-weight: 600 !important;
}}
.stMultiSelect [data-baseweb="tag"] {{
    background: rgba(34,211,238,0.15) !important;
    border: 1px solid rgba(34,211,238,0.3) !important;
    color: {PRIMARY} !important;
    border-radius: 4px !important;
}}
.stButton > button {{
    background: linear-gradient(135deg, {PRIMARY} 0%, #0ea5e9 100%) !important;
    color: #0b1220 !important; border: none !important; border-radius: 10px !important;
    font-weight: 700 !important; padding: 0.7rem 1.5rem !important; width: 100% !important;
    box-shadow: 0 8px 30px -8px rgba(34,211,238,0.5) !important;
    transition: transform 0.15s !important;
}}
.stButton > button:hover {{ transform: translateY(-1px); }}

.stTabs [data-baseweb="tab-list"] {{ gap: 0; border-bottom: 1px solid {BORDER}; }}
.stTabs [data-baseweb="tab"] {{
    background: transparent !important; color: {MUTED} !important;
    border: none !important; padding: 0.8rem 1.2rem !important;
    font-weight: 600 !important; font-size: 0.85rem !important;
}}
.stTabs [aria-selected="true"] {{ color: {PRIMARY} !important; border-bottom: 2px solid {PRIMARY} !important; }}

.kpi-card {{
    background: linear-gradient(180deg, {CARD}, #0d1626);
    border: 1px solid {BORDER}; border-radius: 12px; padding: 1.25rem; height: 100%;
}}
.kpi-label {{ color: {MUTED}; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.1em; font-weight: 600; }}
.kpi-value {{ color: {TEXT}; font-size: 1.8rem; font-weight: 700; font-family: 'JetBrains Mono', monospace; margin: 0.4rem 0 0.2rem; }}
.kpi-sub {{ color: {MUTED}; font-size: 0.75rem; }}
.var-card {{
    background: linear-gradient(180deg, {CARD}, #0d1626);
    border: 1px solid {BORDER}; border-top: 3px solid var(--accent); border-radius: 12px; padding: 1.5rem;
}}
.var-value {{ font-family: 'JetBrains Mono', monospace; font-size: 2rem; font-weight: 700; color: var(--accent); margin: 0.5rem 0; }}
.section-title {{
    color: {TEXT}; font-size: 1.1rem; font-weight: 700;
    margin: 2rem 0 0.8rem; padding-bottom: 0.6rem; border-bottom: 1px solid {BORDER};
}}
.section-sub {{ color: {MUTED}; font-size: 0.85rem; font-weight: 400; margin-left: 0.5rem; }}

.stress-card {{
    background: linear-gradient(180deg, {CARD}, #0d1626);
    border: 1px solid {BORDER}; border-left: 4px solid var(--accent);
    border-radius: 0 12px 12px 0; padding: 1rem 1.25rem; margin-bottom: 0.5rem;
}}
.version-badge {{
    display: inline-block;
    background: rgba(34,211,238,0.12); border: 1px solid rgba(34,211,238,0.3);
    color: {PRIMARY}; border-radius: 6px; padding: 0.15rem 0.6rem;
    font-size: 0.72rem; font-weight: 700; font-family: 'JetBrains Mono', monospace;
}}
.version-row {{
    padding: 1rem 0; border-bottom: 1px solid {BORDER};
}}
::-webkit-scrollbar {{ width: 6px; height: 6px; }}
::-webkit-scrollbar-track {{ background: {BG}; }}
::-webkit-scrollbar-thumb {{ background: {BORDER}; border-radius: 3px; }}
</style>
""", unsafe_allow_html=True)

# ===================== HELPERS =====================
def kpi(label, value, sub="", color=PRIMARY):
    return f'<div class="kpi-card"><div class="kpi-label">{label}</div><div class="kpi-value" style="color:{color}">{value}</div><div class="kpi-sub">{sub}</div></div>'

def var_card(label, value, pct, color, desc):
    return f'<div class="var-card" style="--accent:{color}"><div class="kpi-label">{label}</div><div class="var-value">{value}</div><div class="kpi-sub">{pct} do portfólio</div><p style="color:{MUTED}; font-size:0.78rem; margin-top:0.8rem; padding-top:0.8rem; border-top:1px solid {BORDER}">{desc}</p></div>'

def section(title, sub=""):
    return f'<div class="section-title">{title}<span class="section-sub">{sub}</span></div>'

def badge(text, color=PRIMARY):
    return f'<span style="background:{color}18;color:{color};border:1px solid {color}30;border-radius:4px;padding:0.15rem 0.5rem;font-size:0.7rem;font-weight:700;font-family:\'JetBrains Mono\',monospace">{text}</span>'

# ===================== CATÁLOGO DE AÇÕES =====================
ACOES_BR = {
    "PETR4.SA": "Petrobras PN",
    "PETR3.SA": "Petrobras ON",
    "VALE3.SA": "Vale ON",
    "ITUB4.SA": "Itaú Unibanco PN",
    "BBDC4.SA": "Bradesco PN",
    "BBAS3.SA": "Banco do Brasil ON",
    "B3SA3.SA": "B3 ON",
    "MGLU3.SA": "Magazine Luiza ON",
    "WEGE3.SA": "WEG ON",
    "RENT3.SA": "Localiza ON",
    "LREN3.SA": "Lojas Renner ON",
    "ABEV3.SA": "Ambev ON",
    "GGBR4.SA": "Gerdau PN",
    "USIM5.SA": "Usiminas PNA",
    "SUZB3.SA": "Suzano ON",
    "RADL3.SA": "Raia Drogasil ON",
    "TOTS3.SA": "Totvs ON",
    "EMBR3.SA": "Embraer ON",
    "CPLE6.SA": "Copel PNB",
    "ELET3.SA": "Eletrobras ON",
    "ELET6.SA": "Eletrobras PNB",
    "CMIG4.SA": "Cemig PN",
    "SBSP3.SA": "Sabesp ON",
    "BBSE3.SA": "BB Seguridade ON",
    "PRIO3.SA": "PetroRio ON",
    "BPAC11.SA": "BTG Pactual UNT",
    "RDOR3.SA": "Rede D'Or ON",
    "HAPV3.SA": "Hapvida ON",
    "AZUL4.SA": "Azul PN",
    "GOLL4.SA": "Gol PN",
    "CSAN3.SA": "Cosan ON",
    "VIVT3.SA": "Vivo/Telefônica ON",
    "HYPE3.SA": "Hypera ON",
    "KLBN11.SA": "Klabin UNT",
    "NTCO3.SA": "Natura ON",
    "JBSS3.SA": "JBS ON",
    "MRFG3.SA": "Marfrig ON",
    "BEEF3.SA": "Minerva ON",
    "CYRE3.SA": "Cyrela ON",
    "MRVE3.SA": "MRV ON",
    "EZTC3.SA": "EZTEC ON",
    "MULT3.SA": "Multiplan ON",
}
ACOES_US = {
    "AAPL":  "Apple",
    "MSFT":  "Microsoft",
    "GOOGL": "Alphabet",
    "AMZN":  "Amazon",
    "NVDA":  "NVIDIA",
    "META":  "Meta Platforms",
    "TSLA":  "Tesla",
    "JPM":   "JPMorgan Chase",
    "BAC":   "Bank of America",
    "GS":    "Goldman Sachs",
    "XOM":   "ExxonMobil",
    "CVX":   "Chevron",
    "JNJ":   "Johnson & Johnson",
    "UNH":   "UnitedHealth",
    "PFE":   "Pfizer",
    "KO":    "Coca-Cola",
    "PEP":   "PepsiCo",
    "WMT":   "Walmart",
    "SPY":   "S&P 500 ETF",
    "QQQ":   "Nasdaq 100 ETF",
    "GLD":   "Gold ETF",
    "TLT":   "Treasury Bond ETF",
    "BTC-USD":  "Bitcoin",
    "ETH-USD":  "Ethereum",
    "BRK-B": "Berkshire Hathaway B",
    "V":     "Visa",
    "MA":    "Mastercard",
    "DIS":   "Disney",
    "NFLX":  "Netflix",
    "AMD":   "AMD",
    "INTC":  "Intel",
    "BA":    "Boeing",
    "CAT":   "Caterpillar",
    "IBM":   "IBM",
}
ALL_ACOES = {**ACOES_BR, **ACOES_US}
OPCOES_DISPLAY = {f"{ticker} — {nome}": ticker for ticker, nome in ALL_ACOES.items()}

# ===================== STRESS TEST — MARCOS HISTÓRICOS =====================
STRESS_EVENTS = {
    "🔴 COVID-19 — Crash de Março 2020": {
        "start": "2020-02-17",
        "end":   "2020-03-23",
        "desc":  "Pandemia global declara lockdowns. Maior queda em 30 anos em apenas 5 semanas.",
        "cor":   DANGER,
        "categoria": "Pandemia",
        "queda_sp500": -34.0,
    },
    "📉 Crise Financeira Global 2008": {
        "start": "2008-09-01",
        "end":   "2009-03-09",
        "desc":  "Colapso do Lehman Brothers, crise do subprime. Maior recessão desde 1929.",
        "cor":   DANGER,
        "categoria": "Crise Financeira",
        "queda_sp500": -56.8,
    },
    "⚡ Flash Crash — Maio 2010": {
        "start": "2010-05-06",
        "end":   "2010-05-10",
        "desc":  "Dow Jones caiu 1000 pontos em minutos por ordens algorítmicas em cascata.",
        "cor":   AMBER,
        "categoria": "Mercado",
        "queda_sp500": -9.2,
    },
    "🇬🇧 Brexit — Referendo Junho 2016": {
        "start": "2016-06-23",
        "end":   "2016-07-06",
        "desc":  "Reino Unido vota sair da UE. Libra esterlina despencou ao menor nível em 30 anos.",
        "cor":   AMBER,
        "categoria": "Geopolítico",
        "queda_sp500": -5.3,
    },
    "🇺🇸 Eleição Trump — Nov 2016": {
        "start": "2016-11-07",
        "end":   "2016-11-14",
        "desc":  "Surpresa eleitoral provoca volatilidade intensa. Futuros caíram 5% na madrugada.",
        "cor":   VIOLET,
        "categoria": "Político",
        "queda_sp500": -2.1,
    },
    "📊 Crise de Liquidez 2018 — Q4": {
        "start": "2018-10-03",
        "end":   "2018-12-24",
        "desc":  "Fed sobe juros agressivamente. Mercado tem pior dezembro desde 1931.",
        "cor":   AMBER,
        "categoria": "Monetário",
        "queda_sp500": -19.8,
    },
    "🦠 Variante Delta — Jul 2021": {
        "start": "2021-07-19",
        "end":   "2021-08-05",
        "desc":  "Nova variante do COVID-19 gera temor de lockdowns. Recuperação rápida.",
        "cor":   "#fb923c",
        "categoria": "Pandemia",
        "queda_sp500": -4.2,
    },
    "🏦 Colapso Evergrande — Set 2021": {
        "start": "2021-09-13",
        "end":   "2021-09-30",
        "desc":  "Maior incorporadora da China à beira da falência. Contágio nos mercados globais.",
        "cor":   DANGER,
        "categoria": "Crédito",
        "queda_sp500": -5.2,
    },
    "🚀 Alta de Juros do Fed — 2022": {
        "start": "2022-01-03",
        "end":   "2022-10-13",
        "desc":  "Fed eleva juros de 0% para 4,5% ao ano. Pior ano para bonds em 40 anos.",
        "cor":   DANGER,
        "categoria": "Monetário",
        "queda_sp500": -27.5,
    },
    "💥 Colapso FTX — Nov 2022": {
        "start": "2022-11-07",
        "end":   "2022-11-20",
        "desc":  "Exchange de criptomoedas FTX declara falência. Bitcoin caiu 25% em dias.",
        "cor":   VIOLET,
        "categoria": "Cripto",
        "queda_sp500": -4.1,
    },
    "🏦 Crise dos Bancos Regionais EUA — Mar 2023": {
        "start": "2023-03-08",
        "end":   "2023-03-24",
        "desc":  "Silicon Valley Bank e Signature Bank quebram em 48h. Crise de confiança bancária.",
        "cor":   DANGER,
        "categoria": "Crise Financeira",
        "queda_sp500": -6.8,
    },
    "🇧🇷 8 de Janeiro 2023 — Brasil": {
        "start": "2023-01-06",
        "end":   "2023-01-13",
        "desc":  "Invasão do Congresso e STF em Brasília. Ibovespa cai e real se deprecia.",
        "cor":   "#22c55e",
        "categoria": "Político",
        "queda_sp500": -0.5,
    },
    "⚔️ Ataque Iran a Israel — Abr 2024": {
        "start": "2024-04-13",
        "end":   "2024-04-22",
        "desc":  "Irã lança mais de 300 drones e mísseis contra Israel em ataque histórico direto.",
        "cor":   AMBER,
        "categoria": "Geopolítico",
        "queda_sp500": -3.1,
    },
    "📉 Crash Agosto 2024 — Carry Trade": {
        "start": "2024-08-01",
        "end":   "2024-08-09",
        "desc":  "Banco do Japão sobe juros, desmanche do carry trade do iene. Nikkei caiu 12% em um dia.",
        "cor":   DANGER,
        "categoria": "Mercado",
        "queda_sp500": -8.5,
    },
    "🇺🇸 Tarifaço Trump — Abr 2025": {
        "start": "2025-04-02",
        "end":   "2025-04-09",
        "desc":  "Trump anuncia tarifas de 10-145% sobre importações. Pânico global nos mercados.",
        "cor":   DANGER,
        "categoria": "Geopolítico",
        "queda_sp500": -12.0,
    },
}

# ===================== FINANÇAS =====================
def bs(S, K, T, r, sigma, tipo="call"):
    if T <= 0: return max(S - K, 0) if tipo == "call" else max(K - S, 0)
    if sigma <= 0:
        return max(S - K*np.exp(-r*T), 0) if tipo == "call" else max(K*np.exp(-r*T) - S, 0)
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return (S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)) if tipo == "call" \
           else (K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1))

def greeks(S, K, T, r, sigma, tipo="call"):
    if T <= 0 or sigma <= 0: return 0.0, 0.0, 0.0
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    delta = norm.cdf(d1) if tipo == "call" else norm.cdf(d1) - 1
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega  = S * norm.pdf(d1) * np.sqrt(T)
    return float(delta), float(gamma), float(vega)

@st.cache_data(ttl=600, show_spinner=False)
def baixar(tickers_list, ini, fim=None):
    try:
        kwargs = dict(start=ini, auto_adjust=True, progress=False, threads=False)
        if fim: kwargs["end"] = fim
        df = yf.download(tickers_list, **kwargs)
        prices = df["Close"] if "Close" in df.columns else df
        if isinstance(prices, pd.Series): prices = prices.to_frame(tickers_list[0])
        prices = prices.dropna(how="all")
        if not prices.empty: return prices, None
    except Exception: pass
    frames = {}
    for t in tickers_list:
        try:
            kwargs2 = dict(start=ini, auto_adjust=True)
            if fim: kwargs2["end"] = fim
            h = yf.Ticker(t).history(**kwargs2)
            if not h.empty: frames[t] = h["Close"]
        except: pass
    if frames:
        df2 = pd.DataFrame(frames).dropna(how="all")
        if not df2.empty: return df2, None
    return None, "Falha ao baixar dados."

def chart_style():
    plt.rcParams.update({
        "figure.facecolor": CARD, "axes.facecolor": CARD,
        "axes.edgecolor": BORDER, "axes.labelcolor": MUTED,
        "axes.titlecolor": TEXT, "text.color": TEXT,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "grid.color": BORDER, "grid.alpha": 0.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.facecolor": CARD, "legend.edgecolor": BORDER, "legend.labelcolor": TEXT,
        "font.family": "monospace", "figure.dpi": 110,
    })

# ===================== SIDEBAR =====================
st.sidebar.markdown(f"""
<h2 style="color:{PRIMARY}; font-weight:700; margin-bottom:0">⚡ Risk Lab</h2>
<p style="color:{MUTED}; font-size:0.8rem; margin-top:0">VaR · Black-Scholes · Stress Test</p>
<hr style="border-color:{BORDER}">
""", unsafe_allow_html=True)

# --- CARTEIRA ---
st.sidebar.markdown(f'<p style="color:{PRIMARY}; font-size:0.7rem; font-weight:700; text-transform:uppercase; letter-spacing:0.1em">▸ Carteira</p>', unsafe_allow_html=True)

mercado_sel = st.sidebar.radio("Mercado", ["🇧🇷 Brasil", "🇺🇸 EUA / Global", "✏️ Manual"], horizontal=True, label_visibility="collapsed")

if mercado_sel == "🇧🇷 Brasil":
    catalogo = ACOES_BR
elif mercado_sel == "🇺🇸 EUA / Global":
    catalogo = ACOES_US
else:
    catalogo = ALL_ACOES

st.sidebar.markdown(f'<div style="color:{MUTED};font-size:0.68rem;font-weight:600;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.3rem">Mercado: {mercado_sel}</div>', unsafe_allow_html=True)

if mercado_sel == "✏️ Manual":
    tickers_str_manual = st.sidebar.text_input("Tickers (separados por vírgula)", "PETR4.SA, VALE3.SA")
    tickers_selecionados = [t.strip().upper() for t in tickers_str_manual.split(",") if t.strip()]
    qty_str = st.sidebar.text_input("Quantidades (mesma ordem)", "1000, 800")
else:
    opcoes_lista = list(OPCOES_DISPLAY.keys())
    opcoes_filtradas = [o for o in opcoes_lista if list(OPCOES_DISPLAY.values())[list(OPCOES_DISPLAY.keys()).index(o)] in catalogo]

    default_br  = ["PETR4.SA — Petrobras PN", "VALE3.SA — Vale ON", "ITUB4.SA — Itaú Unibanco PN"]
    default_us  = ["AAPL — Apple", "MSFT — Microsoft", "NVDA — NVIDIA"]
    default_all = default_br if mercado_sel == "🇧🇷 Brasil" else default_us

    selecao_display = st.sidebar.multiselect(
        "Selecione os ativos",
        options=opcoes_filtradas,
        default=[d for d in default_all if d in opcoes_filtradas],
        max_selections=8,
    )
    tickers_selecionados = [OPCOES_DISPLAY[d] for d in selecao_display]

    qty_default = ", ".join(["1000"] * len(tickers_selecionados))
    qty_str = st.sidebar.text_input("Quantidades (mesma ordem)", qty_default)

data_ini = st.sidebar.date_input("Data início", pd.to_datetime("2022-01-01"))

# --- VaR ---
st.sidebar.markdown(f'<p style="color:{PRIMARY}; font-size:0.7rem; font-weight:700; text-transform:uppercase; letter-spacing:0.1em; margin-top:1rem">▸ VaR</p>', unsafe_allow_html=True)
nivel   = st.sidebar.selectbox("Confiança", [0.90, 0.95, 0.975, 0.99], 1, format_func=lambda x: f"{x*100:.1f}%")
horizonte = st.sidebar.number_input("Horizonte (dias)", 1, 30, 1)
janela    = st.sidebar.number_input("Janela rolling", 30, 252, 63)

# --- OPÇÃO ---
st.sidebar.markdown(f'<p style="color:{PRIMARY}; font-size:0.7rem; font-weight:700; text-transform:uppercase; letter-spacing:0.1em; margin-top:1rem">▸ Opção (Black-Scholes)</p>', unsafe_allow_html=True)
opt_ativo = st.sidebar.selectbox("Ativo-base opção", tickers_selecionados if tickers_selecionados else ["PETR4.SA"])
opt_tipo  = st.sidebar.selectbox("Tipo", ["call", "put"])
opt_qty   = st.sidebar.number_input("Quantidade", 0, value=1000, step=100)
strike    = st.sidebar.number_input("Strike (K)", 1.0, value=40.0, step=0.5)
rf        = st.sidebar.number_input("Taxa livre risco a.a.", 0.0, 1.0, 0.105, 0.005, "%.3f")
T_exp     = st.sidebar.number_input("Vencimento (anos)", 0.01, 5.0, 0.25, 0.05, "%.2f")

calcular = st.sidebar.button("▶  CALCULAR VaR")

# ===================== HEADER =====================
st.markdown(f"""
<div style="display:flex; align-items:center; gap:1rem; padding:1.5rem 0 2rem; border-bottom:1px solid {BORDER}; margin-bottom:1.5rem">
  <div style="width:56px; height:56px; border-radius:14px; background:linear-gradient(135deg, rgba(34,211,238,0.2), rgba(167,139,250,0.15)); border:1px solid {PRIMARY}40; display:flex; align-items:center; justify-content:center; font-size:1.8rem">⚡</div>
  <div>
    <p style="color:{MUTED}; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.12em; margin:0; font-weight:600">Modelagem Aplicada ao Mercado Financeiro</p>
    <h1 style="margin:0.2rem 0 0; font-size:1.8rem">Risk Lab <span style="color:{MUTED}; font-weight:400">— Value at Risk v5.0</span></h1>
  </div>
  <div style="margin-left:auto; text-align:right">
    {''.join([f'<span style="background:{PRIMARY}18;color:{PRIMARY};border:1px solid {PRIMARY}30;border-radius:4px;padding:0.2rem 0.6rem;font-size:0.7rem;font-weight:700;font-family:monospace;margin-left:0.3rem">{t}</span>' for t in tickers_selecionados[:5]])}
  </div>
</div>
""", unsafe_allow_html=True)

if not calcular:
    st.markdown(f"""
    <div style="text-align:center; padding:4rem 2rem; background:{CARD}; border:1px solid {BORDER}; border-radius:16px">
      <div style="font-size:3rem">📉</div>
      <h2 style="margin-top:1rem">Configure a carteira para começar</h2>
      <p style="color:{MUTED}">Selecione os ativos na barra lateral e clique em <b style="color:{PRIMARY}">Calcular VaR</b>.</p>
      <div style="margin-top:1.5rem; display:flex; gap:0.5rem; justify-content:center; flex-wrap:wrap">
        {''.join([badge(t, PRIMARY) for t in list(ACOES_BR.keys())[:8]])}
      </div>
    </div>""", unsafe_allow_html=True)
    st.stop()

# ===================== DADOS =====================
if not tickers_selecionados:
    st.error("Selecione ao menos um ativo na barra lateral.")
    st.stop()

with st.spinner("Conectando ao mercado…"):
    precos, erro = baixar(tickers_selecionados, str(data_ini))

if erro or precos is None:
    st.error(erro or "Sem dados.")
    st.stop()

tickers = [t for t in tickers_selecionados if t in precos.columns]
if not tickers:
    st.error("Nenhum dos tickers retornou dados. Verifique os símbolos.")
    st.stop()

try:
    qtds = [int(q.strip()) for q in qty_str.split(",")]
except ValueError:
    qtds = [1000] * len(tickers)
while len(qtds) < len(tickers): qtds.append(1000)
quantidades = dict(zip(tickers, qtds))

precos   = precos[tickers].dropna()
retornos = precos.pct_change().dropna()

ultimos  = precos.iloc[-1]
v_acoes  = sum(quantidades[t] * float(ultimos[t]) for t in tickers)
S0       = float(ultimos[opt_ativo]) if opt_ativo in tickers else float(ultimos.iloc[0])
sig_an   = float(retornos[opt_ativo].std() * np.sqrt(252)) if opt_ativo in retornos.columns else 0.3
preco_op = bs(S0, strike, T_exp, rf, sig_an, opt_tipo)
v_op     = opt_qty * preco_op
v_total  = v_acoes + v_op

pesos     = np.array([quantidades[t] * float(ultimos[t]) / v_acoes for t in tickers])
ret_cart  = retornos[tickers].dot(pesos)
mu, sig   = float(ret_cart.mean()), float(ret_cart.std())
z         = norm.ppf(1 - nivel)
pct       = 1 - nivel
var_param = -(mu * horizonte + z * sig * np.sqrt(horizonte)) * v_acoes
var_hist  = -float(np.percentile(ret_cart, pct * 100)) * v_acoes

pnl = []
for i in range(len(retornos)):
    ch = retornos[tickers].iloc[i]
    np_ = ultimos * (1 + ch)
    nv  = sum(quantidades[t] * float(np_[t]) for t in tickers)
    Tc  = max(T_exp - horizonte / 252, 0)
    no  = bs(float(np_[opt_ativo]) if opt_ativo in tickers else S0, strike, Tc, rf, sig_an, opt_tipo)
    pnl.append((nv + opt_qty * no) - v_total)
pnl = np.array(pnl)
var_full = -float(np.percentile(pnl, pct * 100))

es_param = -float(ret_cart[ret_cart <= np.percentile(ret_cart, pct * 100)].mean()) * v_acoes
es_hist  = es_param

delta_v, gamma_v, vega_v = greeks(S0, strike, T_exp, rf, sig_an, opt_tipo)

# ===================== TABS =====================
chart_style()
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "  Resumo  ",
    "  Gráficos  ",
    "  Janelas & ES  ",
    "  Call vs Put  ",
    "  🌡️ Stress Test  ",
    "  📋 Histórico de Versões  ",
])

# ========== TAB 1 — RESUMO ==========
with tab1:
    st.markdown(section("Composição", f"{len(tickers)} ativos · {opt_tipo.upper()} {opt_ativo}"), unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(kpi("Ações",      f"R$ {v_acoes:,.0f}",  f"{len(tickers)} ativos",     PRIMARY), unsafe_allow_html=True)
    c2.markdown(kpi("Opções",     f"R$ {v_op:,.0f}",     f"{opt_qty:,} {opt_tipo}s",    SUCCESS), unsafe_allow_html=True)
    c3.markdown(kpi("Total",      f"R$ {v_total:,.0f}",  "valor mercado",               AMBER),   unsafe_allow_html=True)
    c4.markdown(kpi("Vol. diária",f"{sig*100:.2f}%",     f"anual {sig*np.sqrt(252)*100:.1f}%", VIOLET), unsafe_allow_html=True)

    st.markdown(section("VaR", f"Conf. {nivel*100:.1f}% · h={horizonte}d"), unsafe_allow_html=True)
    v1, v2, v3 = st.columns(3)
    v1.markdown(var_card("Paramétrico",    f"R$ {var_param:,.0f}", f"{var_param/v_total*100:.2f}%", PRIMARY, "Normal · linear"),                      unsafe_allow_html=True)
    v2.markdown(var_card("Histórico",      f"R$ {var_hist:,.0f}",  f"{var_hist/v_total*100:.2f}%",  SUCCESS, "Empírico"),                              unsafe_allow_html=True)
    v3.markdown(var_card("Full Valuation", f"R$ {var_full:,.0f}",  f"{var_full/v_total*100:.2f}%",  AMBER,   "Black-Scholes · não linear"),             unsafe_allow_html=True)

    st.markdown(section("Gregas BS"), unsafe_allow_html=True)
    g1, g2, g3 = st.columns(3)
    g1.markdown(kpi("Delta Δ", f"{delta_v:.4f}", "exposição direcional", PRIMARY), unsafe_allow_html=True)
    g2.markdown(kpi("Gamma Γ", f"{gamma_v:.6f}", "convexidade",          SUCCESS), unsafe_allow_html=True)
    g3.markdown(kpi("Vega ν",  f"{vega_v:.4f}",  "sens. à vol",         AMBER),   unsafe_allow_html=True)

    st.markdown(section("Posições"), unsafe_allow_html=True)
    rows_pos = []
    for t in tickers:
        preco_atual = float(ultimos[t])
        ret_t = retornos[t]
        valor = quantidades[t] * preco_atual
        vol_d = ret_t.std() * 100
        var_i = -(ret_t.mean() + norm.ppf(1-nivel) * ret_t.std()) * valor
        rows_pos.append({
            "Ticker": t,
            "Nome": ALL_ACOES.get(t, t),
            "Qtd.": f"{quantidades[t]:,}",
            "Preço atual": f"R$ {preco_atual:.2f}",
            "Valor (R$)": f"R$ {valor:,.0f}",
            "Peso": f"{valor/v_acoes*100:.1f}%",
            "Vol. diária": f"{vol_d:.2f}%",
            "VaR individual": f"R$ {var_i:,.0f}",
        })
    st.dataframe(pd.DataFrame(rows_pos), use_container_width=True, hide_index=True)

# ========== TAB 2 — GRÁFICOS ==========
with tab2:
    st.markdown(section("Distribuição de retornos"), unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.hist(ret_cart, bins=60, color=PRIMARY, alpha=0.5)
    ax.hist(ret_cart[ret_cart <= np.percentile(ret_cart, pct*100)], bins=60, color=DANGER, alpha=0.85)
    ax.axvline(np.percentile(ret_cart, pct*100), color=SUCCESS, ls="--", lw=1.5, label=f"VaR {nivel*100:.0f}%")
    ax.legend(); ax.set_xlabel("Retorno"); ax.set_ylabel("Frequência")
    st.pyplot(fig); plt.close(fig)

    st.markdown(section("Preços normalizados (base 100)"), unsafe_allow_html=True)
    fig2, ax2 = plt.subplots(figsize=(12, 4))
    cores_plot = [PRIMARY, SUCCESS, AMBER, VIOLET, DANGER, "#fb923c", "#f472b6"]
    for i, t in enumerate(tickers):
        serie = precos[t] / precos[t].iloc[0] * 100
        ax2.plot(serie.index, serie.values, lw=1.5, color=cores_plot[i % len(cores_plot)], label=t)
    ax2.axhline(100, color=BORDER, lw=0.8, ls="--")
    ax2.legend(); ax2.set_ylabel("Índice (base 100)")
    st.pyplot(fig2); plt.close(fig2)

    st.markdown(section("P&L Full Valuation"), unsafe_allow_html=True)
    fig3, ax3 = plt.subplots(figsize=(12, 4))
    ax3.hist(pnl, bins=60, color=AMBER, alpha=0.5)
    ax3.hist(pnl[pnl <= np.percentile(pnl, pct*100)], bins=60, color=DANGER, alpha=0.85)
    ax3.axvline(-var_full, color=AMBER, ls="--", lw=1.5, label=f"VaR Full R$ {var_full:,.0f}")
    ax3.legend(); ax3.set_xlabel("P&L (R$)"); ax3.set_ylabel("Frequência")
    st.pyplot(fig3); plt.close(fig3)

# ========== TAB 3 — JANELAS & ES ==========
with tab3:
    st.markdown(section("Expected Shortfall (CVaR)", "Perda média condicional à perda > VaR"), unsafe_allow_html=True)
    e1, e2 = st.columns(2)
    e1.markdown(kpi("ES Histórico",  f"R$ {es_hist:,.0f}",          f"vs VaR {var_hist/es_hist*100:.0f}%", DANGER), unsafe_allow_html=True)
    e2.markdown(kpi("Razão ES/VaR",  f"{es_hist/var_hist:.2f}x",    "captura cauda",                        VIOLET), unsafe_allow_html=True)

    st.markdown(section("VaR por janela histórica"), unsafe_allow_html=True)
    janelas = [("Desde 2020", "2020-01-01"), ("Desde 2022", "2022-01-01"), ("Desde 2023", "2023-01-01"), ("Últimos 252d", None), ("Últimos 63d", "63")]
    rows_j = []
    for nome, ini in janelas:
        if ini == "63":
            sub = retornos.tail(63)
        elif ini:
            sub = retornos[retornos.index >= pd.to_datetime(ini)]
        else:
            sub = retornos.tail(252)
        if len(sub) < 30: continue
        rp  = sub[tickers].dot(pesos)
        vp  = -(rp.mean() + norm.ppf(1-nivel) * rp.std()) * v_acoes
        vh  = -np.percentile(rp, (1-nivel)*100) * v_acoes
        es_ = -rp[rp <= np.percentile(rp, (1-nivel)*100)].mean() * v_acoes
        rows_j.append({"Janela": nome, "Obs": len(sub), "VaR Param.": f"R$ {vp:,.0f}",
                        "VaR Histórico": f"R$ {vh:,.0f}", "CVaR": f"R$ {es_:,.0f}", "Vol diária": f"{rp.std()*100:.2f}%"})
    st.dataframe(pd.DataFrame(rows_j), use_container_width=True, hide_index=True)

# ========== TAB 4 — CALL VS PUT ==========
with tab4:
    st.markdown(section("Call vs Put", f"Strike K={strike} · T={T_exp}a"), unsafe_allow_html=True)
    pc = bs(S0, strike, T_exp, rf, sig_an, "call")
    pp = bs(S0, strike, T_exp, rf, sig_an, "put")
    dc, gc, vc2 = greeks(S0, strike, T_exp, rf, sig_an, "call")
    dp, gp, vp2 = greeks(S0, strike, T_exp, rf, sig_an, "put")
    df_cmp = pd.DataFrame({
        "Métrica": ["Preço BS", "Delta Δ", "Gamma Γ", "Vega ν"],
        "Call": [f"{pc:.4f}", f"{dc:.4f}", f"{gc:.6f}", f"{vc2:.4f}"],
        "Put":  [f"{pp:.4f}", f"{dp:.4f}", f"{gp:.6f}", f"{vp2:.4f}"],
    })
    st.dataframe(df_cmp, use_container_width=True, hide_index=True)

    fig_cp, axes = plt.subplots(1, 2, figsize=(12, 4))
    ps = np.linspace(S0*0.65, S0*1.35, 200)
    axes[0].plot(ps, [bs(s, strike, T_exp, rf, sig_an, "call") for s in ps], color=PRIMARY, lw=2, label="Call")
    axes[0].plot(ps, [bs(s, strike, T_exp, rf, sig_an, "put")  for s in ps], color=AMBER,   lw=2, label="Put")
    axes[0].axvline(strike, color=DANGER, ls="--", alpha=0.6); axes[0].legend(); axes[0].set_title("Preço")
    axes[1].plot(ps, [greeks(s, strike, T_exp, rf, sig_an, "call")[0] for s in ps], color=PRIMARY, lw=2, label="Δ Call")
    axes[1].plot(ps, [greeks(s, strike, T_exp, rf, sig_an, "put")[0]  for s in ps], color=AMBER,   lw=2, label="Δ Put")
    axes[1].axhline(0, color=BORDER, lw=0.8); axes[1].legend(); axes[1].set_title("Delta")
    st.pyplot(fig_cp); plt.close(fig_cp)

# ========== TAB 5 — STRESS TEST ==========
with tab5:
    st.markdown(section("🌡️ Stress Test — Marcos Históricos", "Selecione um evento para ver o impacto real nos seus ativos"), unsafe_allow_html=True)

    # Filtro por categoria
    categorias = sorted(set(v["categoria"] for v in STRESS_EVENTS.values()))
    cat_sel = st.multiselect("Filtrar por categoria", categorias, default=categorias,
                              label_visibility="visible")

    eventos_filtrados = {k: v for k, v in STRESS_EVENTS.items() if v["categoria"] in cat_sel}
    evento_sel = st.selectbox(
        "Escolha o evento histórico",
        list(eventos_filtrados.keys()),
        label_visibility="visible"
    )

    ev = eventos_filtrados[evento_sel]

    # Card do evento
    st.markdown(f"""
    <div style="background:{CARD};border:1px solid {BORDER};border-left:5px solid {ev['cor']};
                border-radius:0 12px 12px 0;padding:1.25rem 1.5rem;margin:1rem 0">
        <div style="display:flex;align-items:center;gap:1rem;flex-wrap:wrap">
            <div>
                <div style="font-size:1.05rem;font-weight:700;color:{TEXT}">{evento_sel}</div>
                <div style="color:{MUTED};font-size:0.82rem;margin-top:0.3rem">{ev['desc']}</div>
            </div>
            <div style="margin-left:auto;text-align:right;min-width:140px">
                <div style="color:{MUTED};font-size:0.68rem;text-transform:uppercase;font-weight:600">S&P 500 no período</div>
                <div style="font-size:1.6rem;font-weight:800;font-family:monospace;color:{DANGER if ev['queda_sp500'] < 0 else SUCCESS}">
                    {ev['queda_sp500']:+.1f}%
                </div>
                <div style="color:{MUTED};font-size:0.72rem">{ev['start']} → {ev['end']}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Baixar dados do período
    with st.spinner(f"Buscando dados do período {ev['start']} → {ev['end']}…"):
        # Pega dados do evento + 5 dias antes para contexto
        start_dt = (pd.to_datetime(ev["start"]) - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
        precos_ev, erro_ev = baixar(tickers, start_dt, ev["end"])

    if erro_ev or precos_ev is None or precos_ev.empty:
        st.warning(f"Não foi possível obter dados para este período. Os ativos selecionados podem não existir nessa data.")
    else:
        tickers_ev = [t for t in tickers if t in precos_ev.columns]
        if not tickers_ev:
            st.warning("Nenhum ativo disponível neste período histórico.")
        else:
            precos_ev = precos_ev[tickers_ev].dropna(how="all")
            # Filtra para o período do evento
            ev_start = pd.to_datetime(ev["start"])
            ev_end   = pd.to_datetime(ev["end"])
            precos_periodo = precos_ev[precos_ev.index >= ev_start]

            if len(precos_periodo) < 2:
                st.warning("Dados insuficientes para o período selecionado.")
            else:
                primeiro = precos_periodo.iloc[0]
                ultimo_ev = precos_periodo.iloc[-1]

                # KPIs de impacto
                st.markdown(f'<div style="margin-top:1.5rem">{section("Impacto nos Ativos Selecionados", f"{len(precos_periodo)} pregões")}</div>', unsafe_allow_html=True)

                cols_st = st.columns(len(tickers_ev))
                for i, t in enumerate(tickers_ev):
                    p0  = float(primeiro[t])
                    p1  = float(ultimo_ev[t])
                    ret = (p1 - p0) / p0 * 100
                    pnl_ev = quantidades.get(t, 1000) * (p1 - p0)
                    cor_ret = SUCCESS if ret >= 0 else DANGER
                    cols_st[i].markdown(kpi(
                        ALL_ACOES.get(t, t),
                        f"{ret:+.1f}%",
                        f"P&L: R$ {pnl_ev:+,.0f}",
                        cor_ret
                    ), unsafe_allow_html=True)

                # Impacto no portfólio
                val_ini = sum(quantidades.get(t, 1000) * float(primeiro[t]) for t in tickers_ev)
                val_fim = sum(quantidades.get(t, 1000) * float(ultimo_ev[t]) for t in tickers_ev)
                ret_port = (val_fim - val_ini) / val_ini * 100
                pnl_port = val_fim - val_ini

                st.markdown(f"""
                <div style="display:flex;gap:1rem;margin:1rem 0;flex-wrap:wrap">
                    <div style="background:{CARD};border:1px solid {BORDER};border-radius:10px;padding:1rem 1.5rem;flex:1;min-width:180px">
                        <div style="color:{MUTED};font-size:0.68rem;font-weight:600;text-transform:uppercase">Portfólio — Início</div>
                        <div style="font-size:1.4rem;font-weight:700;font-family:monospace;color:{TEXT}">R$ {val_ini:,.0f}</div>
                    </div>
                    <div style="background:{CARD};border:1px solid {BORDER};border-radius:10px;padding:1rem 1.5rem;flex:1;min-width:180px">
                        <div style="color:{MUTED};font-size:0.68rem;font-weight:600;text-transform:uppercase">Portfólio — Fim</div>
                        <div style="font-size:1.4rem;font-weight:700;font-family:monospace;color:{TEXT}">R$ {val_fim:,.0f}</div>
                    </div>
                    <div style="background:{CARD};border:1px solid {'#15803d' if ret_port >= 0 else '#7f1d1d'};border-radius:10px;padding:1rem 1.5rem;flex:1;min-width:180px">
                        <div style="color:{MUTED};font-size:0.68rem;font-weight:600;text-transform:uppercase">Variação Total</div>
                        <div style="font-size:1.4rem;font-weight:700;font-family:monospace;color:{SUCCESS if ret_port >= 0 else DANGER}">{ret_port:+.2f}%</div>
                        <div style="color:{MUTED};font-size:0.75rem">R$ {pnl_port:+,.0f}</div>
                    </div>
                    <div style="background:{CARD};border:1px solid {BORDER};border-radius:10px;padding:1rem 1.5rem;flex:1;min-width:180px">
                        <div style="color:{MUTED};font-size:0.68rem;font-weight:600;text-transform:uppercase">VaR cobriu?</div>
                        <div style="font-size:1.1rem;font-weight:700;color:{SUCCESS if abs(pnl_port) <= var_hist else DANGER}">
                            {'✅ Sim' if abs(pnl_port) <= var_hist else '❌ Excedeu VaR'}
                        </div>
                        <div style="color:{MUTED};font-size:0.72rem">VaR hist.: R$ {var_hist:,.0f}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Gráfico de evolução no período
                st.markdown(section("Evolução dos Preços no Período (base 100)"), unsafe_allow_html=True)
                fig_st, ax_st = plt.subplots(figsize=(12, 4))
                cores_plot = [PRIMARY, SUCCESS, AMBER, VIOLET, DANGER, "#fb923c", "#f472b6", "#38bdf8"]
                for i, t in enumerate(tickers_ev):
                    s = precos_periodo[t].dropna()
                    if len(s) > 0:
                        ax_st.plot(s.index, s / s.iloc[0] * 100,
                                   lw=2, color=cores_plot[i % len(cores_plot)], label=t)
                ax_st.axhline(100, color=BORDER, lw=0.8, ls="--", label="Base 100")
                ax_st.fill_between(precos_periodo.index,
                                   ax_st.get_ylim()[0] if ax_st.get_ylim()[0] > 0 else 50, 100,
                                   alpha=0.04, color=DANGER)
                ax_st.set_ylabel("Índice (base 100)"); ax_st.legend()
                ax_st.set_title(f"Stress Test: {evento_sel}")
                st.pyplot(fig_st); plt.close(fig_st)

    # Tabela resumo de todos os eventos
    st.markdown(f'<div style="margin-top:2rem">{section("📊 Todos os Marcos Históricos")}</div>', unsafe_allow_html=True)
    rows_ev = []
    for nome, info in STRESS_EVENTS.items():
        emoji_cat = {"Pandemia": "🦠", "Crise Financeira": "🏦", "Mercado": "⚡",
                     "Geopolítico": "⚔️", "Monetário": "🏛️", "Cripto": "💎",
                     "Político": "🗳️", "Crédito": "💳"}.get(info["categoria"], "📌")
        rows_ev.append({
            "Evento": nome,
            "Período": f"{info['start']} → {info['end']}",
            "Categoria": f"{emoji_cat} {info['categoria']}",
            "S&P 500": f"{info['queda_sp500']:+.1f}%",
        })
    df_ev = pd.DataFrame(rows_ev)
    st.dataframe(df_ev, use_container_width=True, hide_index=True)

# ========== TAB 6 — HISTÓRICO DE VERSÕES ==========
with tab6:
    st.markdown(section("📋 Histórico de Versões", "Evolução do Risk Lab"), unsafe_allow_html=True)

    versoes = [
        {
            "version": "v5.0",
            "date": "2025",
            "title": "Stress Test & Seletor de Ativos",
            "cor": PRIMARY,
            "changes": [
                ("✨ Novo", "Aba de Stress Test com 15 marcos históricos globais e brasileiros", SUCCESS),
                ("✨ Novo", "Seletor visual de ativos: +75 ações BR, EUA, ETFs e criptomoedas", SUCCESS),
                ("✨ Novo", "Aba de Histórico de Versões (esta tela)", SUCCESS),
                ("✨ Novo", "Impacto real do portfólio em cada crise histórica com gráfico", SUCCESS),
                ("✨ Novo", "Comparativo 'VaR cobriu?' durante stress events", SUCCESS),
                ("🔧 Melhoria", "Tabela de posições individuais com vol e VaR por ativo", AMBER),
                ("🔧 Melhoria", "Gráfico de preços normalizados (base 100)", AMBER),
                ("🔧 Melhoria", "Filtro de categorias no Stress Test", AMBER),
            ]
        },
        {
            "version": "v4.0",
            "date": "2025",
            "title": "Três Métodos + Monte Carlo",
            "cor": VIOLET,
            "changes": [
                ("✨ Novo", "Método Monte Carlo com 10.000 simulações", SUCCESS),
                ("✨ Novo", "VaR de horizonte multi-dia com raiz do tempo", SUCCESS),
                ("✨ Novo", "Dashboard com 6 KPIs no topo", SUCCESS),
                ("✨ Novo", "Fallback automático com série sintética (SPX-like)", SUCCESS),
                ("🔧 Melhoria", "Cache de dados Yahoo Finance com TTL de 1h", AMBER),
                ("🔧 Melhoria", "Alertas automáticos de assimetria e curtose", AMBER),
            ]
        },
        {
            "version": "v3.4",
            "date": "2024",
            "title": "Enterprise Edition",
            "cor": AMBER,
            "changes": [
                ("✨ Novo", "UI Enterprise com tema dark premium", SUCCESS),
                ("✨ Novo", "Carregamento imediato sem travas", SUCCESS),
                ("🔧 Melhoria", "CSS avançado com variáveis de tema", AMBER),
                ("🔧 Melhoria", "Fontes Plus Jakarta Sans + JetBrains Mono", AMBER),
                ("🐛 Correção", "Remoção de CSS duplicado causando conflitos", DANGER),
            ]
        },
        {
            "version": "v2.0",
            "date": "2024",
            "title": "Black-Scholes & Expected Shortfall",
            "cor": SUCCESS,
            "changes": [
                ("✨ Novo", "Precificação de opções Black-Scholes (Call e Put)", SUCCESS),
                ("✨ Novo", "Gregas: Delta, Gamma, Vega", SUCCESS),
                ("✨ Novo", "Expected Shortfall (CVaR) — perda média além do VaR", SUCCESS),
                ("✨ Novo", "Full Valuation com reavaliação não-linear do portfólio", SUCCESS),
                ("✨ Novo", "Comparativo Call vs Put com gráficos de payoff", SUCCESS),
                ("✨ Novo", "Análise por janelas históricas (2020, 2022, 2023, rolling)", SUCCESS),
                ("🔧 Melhoria", "CSS limpo substituindo versão com múltiplos imports conflitantes", AMBER),
                ("🔧 Melhoria", "Download de múltiplos tickers com fallback por Ticker individual", AMBER),
            ]
        },
        {
            "version": "v1.0",
            "date": "2024",
            "title": "Versão Inicial",
            "cor": MUTED,
            "changes": [
                ("✨ Novo", "VaR Paramétrico com distribuição normal", SUCCESS),
                ("✨ Novo", "VaR Histórico (simulação histórica)", SUCCESS),
                ("✨ Novo", "Sidebar com configuração de carteira e parâmetros", SUCCESS),
                ("✨ Novo", "Gráfico de distribuição de retornos", SUCCESS),
                ("✨ Novo", "Integração com Yahoo Finance via yfinance", SUCCESS),
            ]
        },
    ]

    for v in versoes:
        st.markdown(f"""
        <div style="margin-bottom:2rem">
            <div style="display:flex;align-items:center;gap:1rem;margin-bottom:0.75rem">
                <span style="background:{v['cor']}18;color:{v['cor']};border:1px solid {v['cor']}40;
                             border-radius:6px;padding:0.2rem 0.7rem;font-size:0.8rem;font-weight:700;
                             font-family:monospace">{v['version']}</span>
                <span style="color:{TEXT};font-size:1rem;font-weight:700">{v['title']}</span>
                <span style="color:{MUTED};font-size:0.75rem;margin-left:auto">{v['date']}</span>
            </div>
            <div style="border-left:2px solid {v['cor']}40;padding-left:1rem">
        """, unsafe_allow_html=True)

        for tipo, desc, cor in v["changes"]:
            st.markdown(f"""
            <div style="display:flex;align-items:flex-start;gap:0.75rem;padding:0.4rem 0;
                        border-bottom:1px solid {BORDER}40">
                <span style="background:{cor}18;color:{cor};border:1px solid {cor}30;
                             border-radius:4px;padding:0.1rem 0.5rem;font-size:0.68rem;
                             font-weight:700;white-space:nowrap;margin-top:0.05rem">{tipo}</span>
                <span style="color:{TEXT};font-size:0.83rem">{desc}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div></div>", unsafe_allow_html=True)

    # Roadmap
    st.markdown(section("🚀 Roadmap — Próximas Versões"), unsafe_allow_html=True)
    roadmap = [
        ("v6.0", "Backtesting de VaR — Kupiec test e análise de violações", VIOLET),
        ("v6.0", "Correlação entre ativos e matriz de covariância", VIOLET),
        ("v7.0", "VaR Condicional (CVaR/ES) por simulação de Monte Carlo completa", PRIMARY),
        ("v7.0", "Integração com API B3 e EODHD para dados em tempo real", PRIMARY),
        ("v8.0", "Relatório PDF exportável com todos os gráficos e métricas", AMBER),
        ("v8.0", "Modelo GARCH para volatilidade estocástica", AMBER),
    ]
    for ver, desc, cor in roadmap:
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:0.75rem;padding:0.5rem 0;border-bottom:1px solid {BORDER}40">
            <span style="background:{cor}15;color:{cor};border:1px solid {cor}30;
                         border-radius:4px;padding:0.1rem 0.5rem;font-size:0.68rem;font-weight:700;
                         font-family:monospace;white-space:nowrap">{ver}</span>
            <span style="color:{MUTED};font-size:0.83rem">{desc}</span>
        </div>
        """, unsafe_allow_html=True)
