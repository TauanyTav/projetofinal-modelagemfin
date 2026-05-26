"""
Calculadora de Value at Risk (VaR) — v2
Trabalho Final | Modelagem Aplicada ao Mercado Financeiro
Melhorias: CSS limpo, Expected Shortfall, comparação de janelas, call vs put.
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

# ===================== PAGE / THEME =====================
st.set_page_config(page_title="Risk Lab — VaR", page_icon="📉", layout="wide")

PRIMARY = "#22d3ee"
SUCCESS = "#34d399"
AMBER = "#fbbf24"
DANGER = "#f87171"
VIOLET = "#a78bfa"
BG = "#0b1220"
CARD = "#111a2e"
BORDER = "#1f2a44"
TEXT = "#e5e7eb"
MUTED = "#94a3b8"

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
.stTextInput label, .stNumberInput label, .stDateInput label, .stSelectbox label, .stSlider label {{
    color: {MUTED} !important; font-size: 0.7rem !important;
    text-transform: uppercase !important; letter-spacing: 0.1em !important; font-weight: 600 !important;
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

::-webkit-scrollbar {{ width: 6px; height: 6px; }}
::-webkit-scrollbar-track {{ background: {BG}; }}
::-webkit-scrollbar-thumb {{ background: {BORDER}; border-radius: 3px; }}
</style>
""", unsafe_allow_html=True)

def kpi(label, value, sub="", color=PRIMARY):
    return f'<div class="kpi-card"><div class="kpi-label">{label}</div><div class="kpi-value" style="color:{color}">{value}</div><div class="kpi-sub">{sub}</div></div>'

def var_card(label, value, pct, color, desc):
    return f'<div class="var-card" style="--accent:{color}"><div class="kpi-label">{label}</div><div class="var-value">{value}</div><div class="kpi-sub">{pct} do portfólio</div><p style="color:{MUTED}; font-size:0.78rem; margin-top:0.8rem; padding-top:0.8rem; border-top:1px solid {BORDER}">{desc}</p></div>'

def section(title, sub=""):
    return f'<div class="section-title">{title}<span class="section-sub">{sub}</span></div>'

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
    vega = S * norm.pdf(d1) * np.sqrt(T)
    return float(delta), float(gamma), float(vega)

@st.cache_data(ttl=600, show_spinner=False)
def baixar(tickers_str, ini):
    tickers = [t.strip() for t in tickers_str.split(",") if t.strip()]
    try:
        df = yf.download(tickers, start=ini, auto_adjust=True, progress=False, threads=False)
        if isinstance(df.columns, pd.MultiIndex):
            prices = df["Close"]
        else:
            prices = df
        if isinstance(prices, pd.Series): prices = prices.to_frame(tickers[0])
        prices = prices.dropna(how="all")
        if not prices.empty: return prices, None
    except Exception as e:
        pass
        
    frames = {}
    for t in tickers:
        try:
            h = yf.Ticker(t).history(start=ini, auto_adjust=True)
            if not h.empty: frames[t] = h["Close"]
        except: 
            pass
    if frames:
        df = pd.DataFrame(frames).dropna(how="all")
        if not df.empty: return df, None
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
st.sidebar.markdown(f'<h2 style="color:{PRIMARY}; font-weight:700; margin-bottom:0">⚡ Risk Lab</h2><p style="color:{MUTED}; font-size:0.8rem; margin-top:0">VaR · Black-Scholes · Stress</p><hr style="border-color:{BORDER}">', unsafe_allow_html=True)

st.sidebar.markdown(f'<p style="color:{PRIMARY}; font-size:0.7rem; font-weight:700; text-transform:uppercase; letter-spacing:0.1em">▸ Carteira</p>', unsafe_allow_html=True)
tickers_str = st.sidebar.text_input(
    "Tickers",
    "PETR4.SA,VALE3.SA,ITUB4.SA"
)
qty_str = st.sidebar.text_input("Quantidades", "1000, 800, 1200")
data_ini = st.sidebar.date_input("Data início", pd.to_datetime("2022-01-01"))

st.sidebar.markdown(f'<p style="color:{PRIMARY}; font-size:0.7rem; font-weight:700; text-transform:uppercase; letter-spacing:0.1em; margin-top:1rem">▸ VaR</p>', unsafe_allow_html=True)
nivel = st.sidebar.selectbox("Confiança", [0.90, 0.95, 0.975, 0.99], 1, format_func=lambda x: f"{x*100:.1f}%")
horizonte = st.sidebar.number_input("Horizonte (dias)", 1, 30, 1)
janela = st.sidebar.number_input("Janela rolling", 30, 252, 63)

st.sidebar.markdown(f'<p style="color:{PRIMARY}; font-size:0.7rem; font-weight:700; text-transform:uppercase; letter-spacing:0.1em; margin-top:1rem">▸ Opção</p>', unsafe_allow_html=True)
tickers_lista = [t.strip() for t in tickers_str.split(",") if t.strip()]
opt_ativo = st.sidebar.selectbox("Ativo", tickers_lista)
opt_tipo = st.sidebar.selectbox("Tipo", ["call", "put"])
opt_qty = st.sidebar.number_input("Quantidade", 0, value=1000, step=100)
strike = st.sidebar.number_input("Strike (K)", 1.0, value=40.0, step=0.5)
rf = st.sidebar.number_input("Taxa livre risco a.a.", 0.0, 1.0, 0.105, 0.005, "%.3f")
T_exp = st.sidebar.number_input("Vencimento (anos)", 0.01, 5.0, 0.25, 0.05, "%.2f")

calcular = st.sidebar.button("▶ CALCULAR VaR")

# ===================== HEADER =====================
st.markdown(f"""
<div style="display:flex; align-items:center; gap:1rem; padding:1.5rem 0 2rem; border-bottom:1px solid {BORDER}; margin-bottom:1.5rem">
  <div style="width:56px; height:56px; border-radius:14px; background:linear-gradient(135deg, rgba(34,211,238,0.2), rgba(167,139,250,0.15)); border:1px solid {PRIMARY}40; display:flex; align-items:center; justify-content:center; font-size:1.8rem">⚡</div>
  <div>
    <p style="color:{MUTED}; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.12em; margin:0; font-weight:600">Modelagem Aplicada ao Mercado Financeiro</p>
    <h1 style="margin:0.2rem 0 0; font-size:1.8rem">Risk Lab <span style="color:{MUTED}; font-weight:400">— Value at Risk</span></h1>
  </div>
</div>
""", unsafe_allow_html=True)

if not calcular:
    st.markdown(f"""
    <div style="text-align:center; padding:4rem 2rem; background:{CARD}; border:1px solid {BORDER}; border-radius:16px">
      <div style="font-size:3rem">📉</div>
      <h2 style="margin-top:1rem">Configure a carteira para começar</h2>
      <p style="color:{MUTED}">Defina ativos, quantidades e opção → clique em Calcular VaR.</p>
    </div>""", unsafe_allow_html=True)
    st.stop()

# ===================== CÁLCULO =====================
with st.spinner("Conectando ao mercado…"):
    precos, erro = baixar(tickers_str, str(data_ini))
if erro or precos is None:
    st.error(erro or "Sem dados.")
    st.stop()

tickers = [t for t in tickers_lista if t in precos.columns]
try:
    qtds = [int(q.strip()) for q in qty_str.split(",")]
except ValueError:
    qtds = [1000] * len(tickers)
quantidades = dict(zip(tickers, qtds))
precos = precos[tickers].dropna(how="all")
precos = precos.dropna(axis=1, how="all")

if precos.empty:
    st.error("""
    Não foi possível obter dados válidos dos ativos selecionados.
    Possíveis causas: Ticker incorreto, indisponibilidade do Yahoo Finance ou período sem dados.
    """)
    st.stop()

retornos = precos.pct_change().dropna()

if len(precos) == 0:
    st.error("Sem dados suficientes para cálculo.")
    st.stop()

ultimos = precos.iloc[-1]
v_acoes = sum(quantidades[t] * float(ultimos[t]) for t in tickers)
S0 = float(ultimos[opt_ativo]) if opt_ativo in tickers else float(ultimos.iloc[0])
sig_an = float(retornos[opt_ativo].std() * np.sqrt(252)) if opt_ativo in retornos.columns else 0.3
preco_op = bs(S0, strike, T_exp, rf, sig_an, opt_tipo)
v_op = opt_qty * preco_op
v_total = v_acoes + v_op

pesos = np.array([quantidades[t] * float(ultimos[t]) / v_acoes for t in tickers])
ret_cart = retornos[tickers].dot(pesos)
mu, sig = float(ret_cart.mean()), float(ret_cart.std())

# Estatísticas de cauda bicaudais/unidirecionais ajustadas para perda positiva
z = norm.ppf(nivel)
pct = 1 - nivel
var_param = (z * sig * np.sqrt(horizonte) - mu * horizonte) * v_acoes
var_hist = -float(np.percentile(ret_cart, pct * 100)) * v_acoes

# Full Valuation
pnl = []
for i in range(len(retornos)):
    ch = retornos[tickers].iloc[i]
    np_ = ultimos * (1 + ch)
    nv = sum(quantidades[t] * float(np_[t]) for t in tickers)
    Tc = max(T_exp - horizonte / 252, 0)
    no = bs(float(np_[opt_ativo]) if opt_ativo in tickers else S0, strike, Tc, rf, sig_an, opt_tipo)
    pnl.append((nv + opt_qty * no) - v_total)
pnl = np.array(pnl)
var_full = -float(np.percentile(pnl, pct * 100))

# Expected Shortfall (Correção da Fórmula Analítica Paramétrica)
es_param = (sig * np.sqrt(horizonte) * (norm.pdf(z) / (1 - nivel)) - mu * horizonte) * v_acoes
es_hist = -float(ret_cart[ret_cart <= np.percentile(ret_cart, pct * 100)].mean()) * v_acoes

delta_v, gamma_v, vega_v = greeks(S0, strike, T_exp, rf, sig_an, opt_tipo)

# ===================== TABS =====================
chart_style()
tab1, tab2, tab3, tab4 = st.tabs(["  Resumo  ", "  Gráficos  ", "  Janelas & ES  ", "  Call vs Put  "])

with tab1:
    st.markdown(section("Composição", f"{len(tickers)} ativos · {opt_tipo.upper()} {opt_ativo}"), unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(kpi("Ações", f"R$ {v_acoes:,.0f}", f"{len(tickers)} ativos", PRIMARY), unsafe_allow_html=True)
    c2.markdown(kpi("Opções", f"R$ {v_op:,.0f}", f"{opt_qty:,} {opt_tipo}s", SUCCESS), unsafe_allow_html=True)
    c3.markdown(kpi("Total", f"R$ {v_total:,.0f}", "valor mercado", AMBER), unsafe_allow_html=True)
    c4.markdown(kpi("Vol. diária", f"{sig*100:.2f}%", f"anual {sig*np.sqrt(252)*100:.1f}%", VIOLET), unsafe_allow_html=True)

    st.markdown(section("VaR", f"Conf. {nivel*100:.1f}% · h={horizonte}d"), unsafe_allow_html=True)
    v1, v2, v3 = st.columns(3)
    v1.markdown(var_card("Paramétrico", f"R$ {var_param:,.0f}", f"{var_param/v_total*100:.2f}%", PRIMARY, "Normal · linear"), unsafe_allow_html=True)
    v2.markdown(var_card("Histórico", f"R$ {var_hist:,.0f}", f"{var_hist/v_total*100:.2f}%", SUCCESS, "Empírico"), unsafe_allow_html=True)
    v3.markdown(var_card("Full Valuation", f"R$ {var_full:,.0f}", f"{var_full/v_total*100:.2f}%", AMBER, "Black-Scholes · não linear"), unsafe_allow_html=True)

    st.markdown(section("Gregas BS"), unsafe_allow_html=True)
    g1, g2, g3 = st.columns(3)
    g1.markdown(kpi("Delta Δ", f"{delta_v:.4f}", "exposição direcional", PRIMARY), unsafe_allow_html=True)
    g2.markdown(kpi("Gamma Γ", f"{gamma_v:.6f}", "convexidade", SUCCESS), unsafe_allow_html=True)
    g3.markdown(kpi("Vega ν", f"{vega_v:.4f}", "sens. à vol", AMBER), unsafe_allow_html=True)

with tab2:
    st.markdown(section("Distribuição de retornos"), unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.hist(ret_cart, bins=60, color=PRIMARY, alpha=0.5)
    ax.hist(ret_cart[ret_cart <= np.percentile(ret_cart, pct*100)], bins=60, color=DANGER, alpha=0.85)
    ax.axvline(np.percentile(ret_cart, pct*100), color=SUCCESS, ls="--", lw=1.5, label=f"VaR {nivel*100:.0f}%")
    ax.legend(); ax.set_xlabel("Retorno"); ax.set_ylabel("Frequência")
    st.pyplot(fig); plt.close(fig)

    st.markdown(section("P&L Full Valuation"), unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.hist(pnl, bins=60, color=AMBER, alpha=0.5)
    ax.hist(pnl[pnl <= np.percentile(pnl, pct*100)], bins=60, color=DANGER, alpha=0.85)
    ax.axvline(-var_full, color=AMBER, ls="--", lw=1.5, label=f"VaR Full R$ {var_full:,.0f}")
    ax.legend(); ax.set_xlabel("P&L (R$)"); ax.set_ylabel("Frequência")
    st.pyplot(fig); plt.close(fig)

with tab3:
    st.markdown(section("Expected Shortfall (CVaR)", "Perda média condicional à perda > VaR"), unsafe_allow_html=True)
    e1, e2 = st.columns(2)
    e1.markdown(kpi("ES Histórico", f"R$ {es_hist:,.0f}", f"vs VaR {var_hist/es_hist*100:.0f}%", DANGER), unsafe_allow_html=True)
    e2.markdown(kpi("Razão ES/VaR", f"{es_hist/var_hist:.2f}x", "captura cauda", VIOLET), unsafe_allow_html=True)

    st.markdown(section("VaR por janela histórica (Ex.4 do enunciado)"), unsafe_allow_html=True)
    janelas = [("Desde 2020", "2020-01-01"), ("Desde 2023", "2023-01-01"), ("Últimos 252d", None)]
    rows = []
    for nome, ini in janelas:
        if ini:
            sub = retornos.loc[ini:] if ini in retornos.index.astype(str) else retornos[retornos.index >= pd.to_datetime(ini)]
        else:
            sub = retornos.tail(252)
        if len(sub) < 30: continue
        rp = sub[tickers].dot(pesos)
        vp = (norm.ppf(nivel) * rp.std() - rp.mean()) * v_acoes
        vh = -np.percentile(rp, (1-nivel)*100) * v_acoes
        rows.append({"Janela": nome, "Obs": len(sub), "VaR Paramétrico": f"R$ {vp:,.0f}", "VaR Histórico": f"R$ {vh:,.0f}", "Vol diária": f"{rp.std()*100:.2f}%"})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

with tab4:
    st.markdown(section("Call vs Put (Ex.7 do enunciado)", f"Mesmo strike K={strike} · T={T_exp}a"), unsafe_allow_html=True)
    pc = bs(S0, strike, T_exp, rf, sig_an, "call")
    pp = bs(S0, strike, T_exp, rf, sig_an, "put")
    dc, gc, vc = greeks(S0, strike, T_exp, rf, sig_an, "call")
    dp, gp, vp = greeks(S0, strike, T_exp, rf, sig_an, "put")
    df_cmp = pd.DataFrame({
        "Métrica": ["Preço BS", "Delta Δ", "Gamma Γ", "Vega ν"],
        "Call": [f"{pc:.4f}", f"{dc:.4f}", f"{gc:.6f}", f"{vc:.4f}"],
        "Put":  [f"{pp:.4f}", f"{dp:.4f}", f"{gp:.6f}", f"{vp:.4f}"],
    })
    st.dataframe(df_cmp, use_container_width=True, hide_index=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ps = np.linspace(S0*0.65, S0*1.35, 200)
    axes[0].plot(ps, [bs(s, strike, T_exp, rf, sig_an, "call") for s in ps], color=PRIMARY, lw=2, label="Call")
    axes[0].plot(ps, [bs(s, strike, T_exp, rf, sig_an, "put") for s in ps], color=AMBER, lw=2, label="Put")
    axes[0].axvline(strike, color=DANGER, ls="--", alpha=0.6); axes[0].legend(); axes[0].set_title("Preço")
    axes[1].plot(ps, [greeks(s, strike, T_exp, rf, sig_an, "call")[0] for s in ps], color=PRIMARY, lw=2, label="Δ Call")
    axes[1].plot(ps, [greeks(s, strike, T_exp, rf, sig_an, "put")[0] for s in ps], color=AMBER, lw=2, label="Δ Put")
    axes[1].axhline(0, color=BORDER, lw=0.8); axes[1].legend(); axes[1].set_title("Delta")
    st.pyplot(fig); plt.close(fig)
