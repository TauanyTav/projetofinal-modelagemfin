"""
Calculadora de Value at Risk (VaR)
Trabalho Final — Modelagem Aplicada ao Mercado Financeiro
Design: Dark Financial Dashboard
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

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="VaR Dashboard",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CSS CUSTOMIZADO
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600;700&display=swap');

:root {
    --bg-primary:    #080c14;
    --bg-secondary:  #0d1420;
    --bg-card:       #111827;
    --border:        #1e2d40;
    --accent-blue:   #1d6fa4;
    --accent-cyan:   #0ea5e9;
    --accent-green:  #10b981;
    --accent-red:    #ef4444;
    --accent-amber:  #f59e0b;
    --accent-purple: #a78bfa;
    --text-primary:  #e2e8f0;
    --text-secondary:#7e95b0;
    --text-dim:      #3d5470;
    --mono: 'IBM Plex Mono', monospace;
    --sans: 'IBM Plex Sans', sans-serif;
}

html, body, [class*="css"] {
    font-family: var(--sans) !important;
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
}

#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
.stDecoration { display: none; }
div[data-testid="stToolbar"] { display: none; }

.stApp {
    background: var(--bg-primary) !important;
    background-image:
        radial-gradient(ellipse at 10% 0%, rgba(14,165,233,0.04) 0%, transparent 50%),
        radial-gradient(ellipse at 90% 100%, rgba(16,185,129,0.03) 0%, transparent 50%) !important;
}

.main .block-container {
    padding: 1.5rem 2rem 3rem !important;
    max-width: 1600px !important;
}

section[data-testid="stSidebar"] {
    background: var(--bg-secondary) !important;
    border-right: 1px solid var(--border) !important;
}

.stTextInput > div > div > input,
.stNumberInput > div > div > input {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
    border-radius: 6px !important;
    font-family: var(--mono) !important;
    font-size: 0.82rem !important;
}
.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus {
    border-color: var(--accent-cyan) !important;
    box-shadow: 0 0 0 2px rgba(14,165,233,0.15) !important;
}

[data-baseweb="select"] > div {
    background: var(--bg-card) !important;
    border-color: var(--border) !important;
    color: var(--text-primary) !important;
}
[data-baseweb="popover"] { background: var(--bg-card) !important; }

.stDateInput > div > div > input {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
    font-family: var(--mono) !important;
    border-radius: 6px !important;
}

.stTextInput label, .stNumberInput label,
.stSelectbox label, .stDateInput label {
    color: var(--text-secondary) !important;
    font-size: 0.72rem !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}

.stButton > button {
    background: linear-gradient(135deg, #0ea5e9 0%, #1d6fa4 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: var(--sans) !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.05em !important;
    padding: 0.6rem 1.5rem !important;
    width: 100% !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 20px rgba(14,165,233,0.25) !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 28px rgba(14,165,233,0.4) !important;
}

.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid var(--border) !important;
    gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--text-dim) !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    font-family: var(--sans) !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    padding: 0.75rem 1.2rem !important;
    transition: all 0.2s !important;
}
.stTabs [aria-selected="true"] {
    color: var(--accent-cyan) !important;
    border-bottom: 2px solid var(--accent-cyan) !important;
    background: transparent !important;
}
.stTabs [data-baseweb="tab-panel"] { padding-top: 1.5rem !important; }

.stDataFrame {
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    overflow: hidden !important;
}

.streamlit-expanderHeader {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    color: var(--text-primary) !important;
    font-size: 0.85rem !important;
}
.streamlit-expanderContent {
    background: var(--bg-secondary) !important;
    border: 1px solid var(--border) !important;
    border-top: none !important;
    border-radius: 0 0 6px 6px !important;
}

[data-testid="stMetric"] { display: none !important; }

::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent-blue); }
</style>
""", unsafe_allow_html=True)

# ============================================================
# COMPONENTES HTML
# ============================================================

def kpi_card(label, value, subtitle="", color="#0ea5e9", icon="◈"):
    return f"""
    <div style="background:#111827;border:1px solid #1e2d40;border-top:2px solid {color};
                border-radius:8px;padding:1.1rem 1.2rem;height:100%;position:relative;overflow:hidden;">
        <div style="position:absolute;top:0;right:0;width:60px;height:60px;
                    background:radial-gradient(circle at top right,{color}18,transparent 70%);"></div>
        <div style="font-family:'IBM Plex Mono',monospace;font-size:0.6rem;color:#7e95b0;
                    letter-spacing:0.15em;text-transform:uppercase;margin-bottom:0.5rem;">{icon} {label}</div>
        <div style="font-family:'IBM Plex Mono',monospace;font-size:1.3rem;font-weight:600;
                    color:{color};line-height:1.1;margin-bottom:0.25rem;">{value}</div>
        <div style="font-family:'IBM Plex Sans',sans-serif;font-size:0.68rem;color:#3d5470;">{subtitle}</div>
    </div>"""

def var_card(label, value, pct, color, description):
    return f"""
    <div style="background:#111827;border:1px solid #1e2d40;border-left:3px solid {color};
                border-radius:8px;padding:1.2rem 1.4rem;height:100%;">
        <div style="font-family:'IBM Plex Mono',monospace;font-size:0.58rem;color:#7e95b0;
                    letter-spacing:0.18em;text-transform:uppercase;margin-bottom:0.6rem;">{label}</div>
        <div style="font-family:'IBM Plex Mono',monospace;font-size:1.55rem;font-weight:600;
                    color:{color};margin-bottom:0.3rem;letter-spacing:-0.02em;">{value}</div>
        <span style="background:{color}22;color:{color};font-family:'IBM Plex Mono',monospace;
                     font-size:0.68rem;padding:1px 7px;border-radius:3px;">{pct} do portfólio</span>
        <div style="margin-top:0.6rem;font-size:0.7rem;color:#3d5470;
                    font-family:'IBM Plex Sans',sans-serif;line-height:1.4;">{description}</div>
    </div>"""

def section_title(text, sub=""):
    sub_html = f'<div style="font-size:0.7rem;color:#3d5470;font-family:IBM Plex Mono,monospace;margin-top:3px;">{sub}</div>' if sub else ""
    return f"""
    <div style="margin:1.8rem 0 0.9rem 0;">
        <div style="font-family:'IBM Plex Sans',sans-serif;font-size:0.75rem;font-weight:600;
                    color:#7e95b0;text-transform:uppercase;letter-spacing:0.14em;
                    padding-bottom:0.5rem;border-bottom:1px solid #1e2d40;">{text}</div>
        {sub_html}
    </div>"""

def sidebar_section(text):
    st.sidebar.markdown(f"""
    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.58rem;color:#0ea5e9;
                letter-spacing:0.2em;text-transform:uppercase;padding:1rem 0 0.4rem 0;
                border-top:1px solid #1e2d40;margin-top:0.5rem;">{text}</div>""",
    unsafe_allow_html=True)

def alert_box(msg, color="#ef4444"):
    st.markdown(f"""
    <div style="background:{color}18;border:1px solid {color}55;border-left:3px solid {color};
                border-radius:8px;padding:1rem 1.2rem;margin:0.5rem 0;
                font-family:'IBM Plex Mono',monospace;font-size:0.8rem;color:{color};">
        ⚠ {msg}
    </div>""", unsafe_allow_html=True)

# ============================================================
# MATPLOTLIB THEME
# ============================================================

def apply_chart_style():
    plt.rcParams.update({
        "figure.facecolor":  "#111827",
        "axes.facecolor":    "#111827",
        "axes.edgecolor":    "#1e2d40",
        "axes.labelcolor":   "#7e95b0",
        "axes.titlecolor":   "#e2e8f0",
        "axes.titlesize":    11,
        "axes.labelsize":    9,
        "axes.titleweight":  "600",
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":         True,
        "grid.color":        "#1e2d40",
        "grid.linewidth":    0.6,
        "grid.alpha":        0.8,
        "xtick.color":       "#3d5470",
        "ytick.color":       "#3d5470",
        "xtick.labelsize":   8,
        "ytick.labelsize":   8,
        "legend.facecolor":  "#0d1420",
        "legend.edgecolor":  "#1e2d40",
        "legend.fontsize":   8,
        "legend.labelcolor": "#7e95b0",
        "text.color":        "#e2e8f0",
        "font.family":       "monospace",
        "figure.dpi":        120,
    })

CYAN   = "#0ea5e9"
GREEN  = "#10b981"
AMBER  = "#f59e0b"
RED    = "#ef4444"
PURPLE = "#a78bfa"
DIM    = "#1e2d40"
PALETTE = [CYAN, GREEN, AMBER, PURPLE, RED, "#fb923c", "#e879f9"]

# ============================================================
# FUNÇÕES BLACK-SCHOLES
# ============================================================

def black_scholes(S, K, T, r, sigma, tipo="call"):
    if T <= 0:
        return float(max(S - K, 0)) if tipo == "call" else float(max(K - S, 0))
    if sigma <= 0:
        return (float(max(S - K * np.exp(-r * T), 0)) if tipo == "call"
                else float(max(K * np.exp(-r * T) - S, 0)))
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if tipo == "call":
        return float(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
    return float(K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))

def delta_bs(S, K, T, r, sigma, tipo="call"):
    if T <= 0 or sigma <= 0: return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return float(norm.cdf(d1)) if tipo == "call" else float(norm.cdf(d1) - 1)

def gamma_bs(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0: return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return float(norm.pdf(d1) / (S * sigma * np.sqrt(T)))

def vega_bs(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0: return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return float(S * norm.pdf(d1) * np.sqrt(T))

# ============================================================
# DOWNLOAD COM RETRY
# ============================================================

@st.cache_data(ttl=600, show_spinner=False)
def baixar_dados(tickers_str, data_inicio_str):
    """Cache dos dados — evita rebaixar a cada interação."""
    tickers = [t.strip() for t in tickers_str.split(",") if t.strip()]
    erros = []

    # Tentativa 1: download em bloco
    try:
        df = yf.download(
            tickers,
            start=data_inicio_str,
            auto_adjust=True,
            progress=False,
            threads=False
        )
        if not df.empty:
            if "Close" in df.columns:
                prices = df["Close"]
            else:
                prices = df
            if isinstance(prices, pd.Series):
                prices = prices.to_frame(tickers[0])
            prices = prices.dropna(how="all")
            if not prices.empty:
                return prices, None
    except Exception as e:
        erros.append(str(e))

    # Tentativa 2: um ticker por vez
    frames = {}
    for t in tickers:
        try:
            tk = yf.Ticker(t)
            hist = tk.history(start=data_inicio_str, auto_adjust=True)
            if not hist.empty and "Close" in hist.columns:
                frames[t] = hist["Close"]
        except Exception as e:
            erros.append(f"{t}: {e}")

    if frames:
        df = pd.DataFrame(frames).dropna(how="all")
        if not df.empty:
            return df, None

    return None, f"Falha ao baixar dados. Detalhes: {'; '.join(erros)}"

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown("""
<div style="padding:0.8rem 0 0.6rem 0;border-bottom:1px solid #1e2d40;margin-bottom:0.5rem;">
    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.6rem;color:#0ea5e9;
                letter-spacing:0.22em;text-transform:uppercase;">Risk Engine</div>
    <div style="font-family:'IBM Plex Sans',sans-serif;font-size:1.1rem;font-weight:700;
                color:#e2e8f0;margin-top:2px;">VaR Calculator</div>
</div>""", unsafe_allow_html=True)

sidebar_section("▸ Carteira de Ações")
tickers_input = st.sidebar.text_input("Tickers", value="PETR4.SA, VALE3.SA, ITUB4.SA",
                                       help="Separados por vírgula")
tickers = [t.strip() for t in tickers_input.split(",") if t.strip()]

quantidades_input = st.sidebar.text_input("Quantidades", value="1000, 800, 1200")
try:
    qtd_lista = [int(q.strip()) for q in quantidades_input.split(",")]
    if len(qtd_lista) != len(tickers):
        st.sidebar.error("Nº de quantidades ≠ nº de tickers")
        qtd_lista = [1000] * len(tickers)
except ValueError:
    st.sidebar.error("Use apenas inteiros.")
    qtd_lista = [1000] * len(tickers)
quantidades_acoes = dict(zip(tickers, qtd_lista))

sidebar_section("▸ Período & Parâmetros")
data_inicio     = st.sidebar.date_input("Data de início", value=pd.to_datetime("2022-01-01"))
nivel_confianca = st.sidebar.selectbox("Nível de confiança",
                    [0.90, 0.95, 0.975, 0.99], index=1,
                    format_func=lambda x: f"{x*100:.1f}%")
horizonte_dias  = st.sidebar.number_input("Horizonte (dias)", min_value=1, max_value=30, value=1)
janela_rolling  = st.sidebar.number_input("Janela rolling VaR (dias)", min_value=30, max_value=252, value=63,
                    help="Usada na aba Histórico para calcular o VaR ao longo do tempo")

sidebar_section("▸ Opção Europeia")
ativo_opcao       = st.sidebar.selectbox("Ativo objeto", options=tickers)
tipo_opcao        = st.sidebar.selectbox("Tipo", ["call", "put"])
quantidade_opcoes = st.sidebar.number_input("Quantidade", min_value=0, value=1000, step=100)
strike            = st.sidebar.number_input("Strike (K)", min_value=1.0, value=40.0, step=0.5)
taxa_lr           = st.sidebar.number_input("Taxa livre de risco (a.a.)", min_value=0.0, max_value=1.0,
                                             value=0.105, step=0.005, format="%.3f")
vencimento_anos   = st.sidebar.number_input("Vencimento (anos)", min_value=0.01, max_value=5.0,
                                             value=0.25, step=0.05, format="%.2f")

st.sidebar.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)
calcular = st.sidebar.button("▶  CALCULAR VaR")

st.sidebar.markdown("""
<div style="margin-top:2rem;padding-top:1rem;border-top:1px solid #1e2d40;
            font-family:'IBM Plex Mono',monospace;font-size:0.58rem;color:#3d5470;line-height:1.8;">
    Métodos<br>├ VaR Paramétrico<br>├ VaR Histórico<br>└ VaR Full Valuation<br><br>
    Extras<br>├ Rolling VaR (Histórico)<br>└ Stress Test Interativo<br><br>
    Modelo<br>└ Black-Scholes (1973)
</div>""", unsafe_allow_html=True)

# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div style="display:flex;align-items:flex-end;justify-content:space-between;
            padding:1.5rem 0 1.2rem 0;border-bottom:1px solid #1e2d40;margin-bottom:1.5rem;">
    <div>
        <div style="font-family:'IBM Plex Mono',monospace;font-size:0.65rem;color:#0ea5e9;
                    letter-spacing:0.25em;text-transform:uppercase;margin-bottom:0.4rem;">
            Modelagem Aplicada ao Mercado Financeiro</div>
        <h1 style="font-family:'IBM Plex Sans',sans-serif;font-size:1.9rem;font-weight:700;
                   color:#e2e8f0;margin:0;letter-spacing:-0.02em;line-height:1.1;">
            Value at Risk <span style="color:#0ea5e9;">Dashboard</span></h1>
    </div>
    <div style="text-align:right;">
        <div style="font-family:'IBM Plex Mono',monospace;font-size:0.65rem;color:#3d5470;letter-spacing:0.1em;">
            SISTEMA DE GESTÃO DE RISCO</div>
        <div style="font-family:'IBM Plex Mono',monospace;font-size:0.7rem;color:#7e95b0;margin-top:2px;">
            Black-Scholes · Full Valuation · Stress Test</div>
    </div>
</div>""", unsafe_allow_html=True)

# ============================================================
# ESTADO INICIAL
# ============================================================

if not calcular:
    st.markdown("""
    <div style="background:#111827;border:1px solid #1e2d40;border-radius:10px;
                padding:3rem 2.5rem;text-align:center;margin-top:1rem;">
        <div style="font-size:2.5rem;margin-bottom:1rem;">📉</div>
        <div style="font-family:'IBM Plex Sans',sans-serif;font-size:1.1rem;font-weight:600;
                    color:#e2e8f0;margin-bottom:0.5rem;">Configure a carteira e calcule o VaR</div>
        <div style="font-family:'IBM Plex Mono',monospace;font-size:0.75rem;color:#3d5470;line-height:1.8;">
            Selecione os ativos → defina quantidades → parametrize a opção → clique em Calcular VaR
        </div>
        <div style="display:flex;justify-content:center;gap:2rem;margin-top:2rem;flex-wrap:wrap;">
            <div style="text-align:center;">
                <div style="font-family:'IBM Plex Mono',monospace;font-size:0.6rem;color:#0ea5e9;letter-spacing:0.1em;">MÉTODO 01</div>
                <div style="font-size:0.85rem;color:#7e95b0;margin-top:4px;">VaR Paramétrico</div>
            </div>
            <div style="color:#1e2d40;font-size:1.2rem;">·</div>
            <div style="text-align:center;">
                <div style="font-family:'IBM Plex Mono',monospace;font-size:0.6rem;color:#10b981;letter-spacing:0.1em;">MÉTODO 02</div>
                <div style="font-size:0.85rem;color:#7e95b0;margin-top:4px;">VaR Histórico</div>
            </div>
            <div style="color:#1e2d40;font-size:1.2rem;">·</div>
            <div style="text-align:center;">
                <div style="font-family:'IBM Plex Mono',monospace;font-size:0.6rem;color:#f59e0b;letter-spacing:0.1em;">MÉTODO 03</div>
                <div style="font-size:0.85rem;color:#7e95b0;margin-top:4px;">Full Valuation</div>
            </div>
            <div style="color:#1e2d40;font-size:1.2rem;">·</div>
            <div style="text-align:center;">
                <div style="font-family:'IBM Plex Mono',monospace;font-size:0.6rem;color:#a78bfa;letter-spacing:0.1em;">EXTRA 01</div>
                <div style="font-size:0.85rem;color:#7e95b0;margin-top:4px;">Rolling VaR</div>
            </div>
            <div style="color:#1e2d40;font-size:1.2rem;">·</div>
            <div style="text-align:center;">
                <div style="font-family:'IBM Plex Mono',monospace;font-size:0.6rem;color:#ef4444;letter-spacing:0.1em;">EXTRA 02</div>
                <div style="font-size:0.85rem;color:#7e95b0;margin-top:4px;">Stress Test</div>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)
    st.stop()

# ============================================================
# DOWNLOAD DE DADOS
# ============================================================

with st.spinner("Conectando ao mercado..."):
    precos, erro = baixar_dados(tickers_input, str(data_inicio))

if erro or precos is None:
    alert_box(f"Não foi possível baixar os dados. Possíveis causas: ticker inválido, sem conexão, ou mercado indisponível. Detalhes: {erro}")
    st.markdown("""
    <div style="background:#111827;border:1px solid #1e2d40;border-radius:8px;
                padding:1.2rem 1.5rem;margin-top:1rem;font-family:'IBM Plex Mono',monospace;
                font-size:0.75rem;color:#7e95b0;line-height:2;">
        Dicas:<br>
        · Ações brasileiras usam sufixo .SA — ex: PETR4.SA, VALE3.SA<br>
        · Ações americanas sem sufixo — ex: AAPL, MSFT<br>
        · Verifique se os tickers existem no Yahoo Finance<br>
        · Tente um período de início mais recente (ex: 2023-01-01)
    </div>""", unsafe_allow_html=True)
    st.stop()

# Garantir que todos os tickers estão nas colunas
tickers_disponiveis = [t for t in tickers if t in precos.columns]
if not tickers_disponiveis:
    alert_box("Nenhum dos tickers informados retornou dados. Verifique os símbolos.")
    st.stop()

tickers          = tickers_disponiveis
quantidades_acoes = {t: quantidades_acoes[t] for t in tickers}
precos           = precos[tickers].dropna()
retornos         = precos.pct_change().dropna()

# ============================================================
# CÁLCULOS PRINCIPAIS
# ============================================================

ultimos_precos = precos.iloc[-1]
valor_acoes    = sum(quantidades_acoes[t] * float(ultimos_precos[t]) for t in tickers)

S0        = float(ultimos_precos[ativo_opcao]) if ativo_opcao in ultimos_precos else float(ultimos_precos.iloc[0])
vol_anual = float(retornos[ativo_opcao].std() * np.sqrt(252)) if ativo_opcao in retornos.columns else 0.3

preco_op_hoje    = black_scholes(S0, strike, vencimento_anos, taxa_lr, vol_anual, tipo_opcao)
valor_opcoes     = quantidade_opcoes * preco_op_hoje
valor_total      = valor_acoes + valor_opcoes

delta_op = delta_bs(S0, strike, vencimento_anos, taxa_lr, vol_anual, tipo_opcao)
gamma_op = gamma_bs(S0, strike, vencimento_anos, taxa_lr, vol_anual)
vega_op  = vega_bs(S0, strike, vencimento_anos, taxa_lr, vol_anual)

pesos            = np.array([quantidades_acoes[t] * float(ultimos_precos[t]) / valor_acoes for t in tickers])
retorno_cart     = retornos[tickers].dot(pesos)
media_cart       = float(retorno_cart.mean())
vol_cart         = float(retorno_cart.std())
percentil        = 1 - nivel_confianca
z                = norm.ppf(1 - nivel_confianca)

var_param = -(media_cart * horizonte_dias + z * vol_cart * np.sqrt(horizonte_dias)) * valor_acoes
var_hist  = -float(np.percentile(retorno_cart, percentil * 100)) * valor_acoes

cenarios_pnl = []
for i in range(len(retornos)):
    choque = retornos[tickers].iloc[i]
    np_   = ultimos_precos * (1 + choque)
    nva   = sum(quantidades_acoes[t] * float(np_[t]) for t in tickers)
    T_c   = max(vencimento_anos - horizonte_dias / 252, 0)
    nop   = black_scholes(float(np_[ativo_opcao]) if ativo_opcao in np_ else S0,
                          strike, T_c, taxa_lr, vol_anual, tipo_opcao)
    cenarios_pnl.append((nva + quantidade_opcoes * nop) - valor_total)

cenarios_pnl = np.array(cenarios_pnl)
var_full     = -float(np.percentile(cenarios_pnl, percentil * 100))

# ============================================================
# ROLLING VaR HISTÓRICO
# ============================================================

def calc_rolling_var(retornos_serie, janela, nivel):
    pct = 1 - nivel
    rv = retornos_serie.rolling(janela).apply(
        lambda x: -np.percentile(x, pct * 100), raw=True
    )
    return rv

rolling_var = calc_rolling_var(retorno_cart, int(janela_rolling), nivel_confianca)
rolling_vol = retorno_cart.rolling(int(janela_rolling)).std() * np.sqrt(252)

# Datas de pior VaR
top_piores = (rolling_var * valor_acoes).nlargest(5)

# ============================================================
# TABS
# ============================================================

apply_chart_style()

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "  Resumo  ",
    "  Gráficos  ",
    "  Gregas  ",
    "  Histórico  ",
    "  Stress Test  ",
    "  Teoria  "
])

# ══════════════════════════════════════════════
# TAB 1 — RESUMO
# ══════════════════════════════════════════════
with tab1:
    st.markdown(section_title("Composição da Carteira",
        f"Posição em {len(tickers)} ativo(s) · Opção {tipo_opcao.upper()} sobre {ativo_opcao}"),
        unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(kpi_card("Ações",  f"R$ {valor_acoes:,.0f}", f"{len(tickers)} ativos", CYAN,   "◈"), unsafe_allow_html=True)
    with c2: st.markdown(kpi_card("Opções", f"R$ {valor_opcoes:,.0f}", f"{quantidade_opcoes:,} {tipo_opcao}s", GREEN, "◈"), unsafe_allow_html=True)
    with c3: st.markdown(kpi_card("Total",  f"R$ {valor_total:,.0f}", "valor de mercado", AMBER, "◈"), unsafe_allow_html=True)
    with c4: st.markdown(kpi_card("Vol. Diária", f"{vol_cart*100:.2f}%", f"anual: {vol_cart*np.sqrt(252)*100:.1f}%", PURPLE, "◈"), unsafe_allow_html=True)

    st.markdown(section_title("Posições em Ações"), unsafe_allow_html=True)
    dados_pos = []
    for t in tickers:
        p = float(ultimos_precos[t]); q = quantidades_acoes[t]; v = q * p
        ytd = (float(precos[t].iloc[-1]) / float(precos[t].iloc[0]) - 1) * 100
        dados_pos.append({"Ticker": t, "Último Preço": f"R$ {p:.2f}",
                          "Quantidade": f"{q:,}", "Valor (R$)": f"{v:,.2f}",
                          "Peso": f"{v/valor_acoes*100:.1f}%",
                          "Retorno no período": f"{ytd:+.1f}%"})
    st.dataframe(pd.DataFrame(dados_pos), use_container_width=True, hide_index=True)

    st.markdown(section_title("Resultados de VaR",
        f"Confiança: {nivel_confianca*100:.1f}%  ·  Horizonte: {horizonte_dias} dia(s)  ·  Obs: {len(retorno_cart)}"),
        unsafe_allow_html=True)

    v1, v2, v3 = st.columns(3)
    with v1: st.markdown(var_card("VaR Paramétrico · Ações",      f"R$ {var_param:,.0f}", f"{var_param/valor_total*100:.2f}%", CYAN,  "Normal · Ações lineares"), unsafe_allow_html=True)
    with v2: st.markdown(var_card("VaR Histórico · Ações",         f"R$ {var_hist:,.0f}",  f"{var_hist/valor_total*100:.2f}%",  GREEN, "Empírico · Sem hipótese normal"), unsafe_allow_html=True)
    with v3: st.markdown(var_card("VaR Full Valuation · Ações+Op", f"R$ {var_full:,.0f}",  f"{var_full/valor_total*100:.2f}%",  AMBER, "Black-Scholes · Captura não linearidade"), unsafe_allow_html=True)

    st.markdown(section_title("Tabela Comparativa"), unsafe_allow_html=True)
    df_comp = pd.DataFrame({
        "Método":         ["Paramétrico — Ações", "Histórico — Ações", "Full Valuation — Ações+Opções"],
        "VaR (R$)":       [f"R$ {var_param:,.2f}", f"R$ {var_hist:,.2f}", f"R$ {var_full:,.2f}"],
        "% Portfólio":    [f"{var_param/valor_total*100:.3f}%", f"{var_hist/valor_total*100:.3f}%", f"{var_full/valor_total*100:.3f}%"],
        "Hipótese":       ["Normalidade", "Empírica", "Full repricing"],
        "Escopo":         ["Ações", "Ações", "Ações + Opções"],
    })
    st.dataframe(df_comp, use_container_width=True, hide_index=True)

    st.markdown(section_title("Parâmetros da Opção — Black-Scholes"), unsafe_allow_html=True)
    oc1, oc2, oc3, oc4, oc5 = st.columns(5)
    for col, lbl, val in [
        (oc1, "Ativo", ativo_opcao), (oc2, "Tipo", tipo_opcao.upper()),
        (oc3, "Preço BS", f"R$ {preco_op_hoje:.4f}"),
        (oc4, "Vol. Anual", f"{vol_anual*100:.2f}%"),
        (oc5, "Valor Total", f"R$ {valor_opcoes:,.2f}")]:
        with col: st.markdown(kpi_card(lbl, val, "", PURPLE, ""), unsafe_allow_html=True)

# ══════════════════════════════════════════════
# TAB 2 — GRÁFICOS
# ══════════════════════════════════════════════
with tab2:
    st.markdown(section_title("Distribuição dos Retornos · Carteira de Ações"), unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(12, 3.8))
    ax.hist(retorno_cart, bins=60, color=CYAN, alpha=0.45, edgecolor="none")
    ax.hist(retorno_cart[retorno_cart <= np.percentile(retorno_cart, percentil*100)],
            bins=60, color=RED, alpha=0.75, edgecolor="none", label="Zona de perda")
    vl = np.percentile(retorno_cart, percentil*100)
    ax.axvline(vl, color=GREEN, lw=1.5, ls="--", label=f"VaR Histórico = {vl*100:.2f}%")
    ax.set_xlabel("Retorno Diário"); ax.set_ylabel("Frequência")
    ax.set_title("Distribuição Histórica de Retornos"); ax.legend()
    fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    st.markdown(section_title("P&L Distribution · Full Valuation"), unsafe_allow_html=True)
    fig2, ax2 = plt.subplots(figsize=(12, 3.8))
    ax2.hist(cenarios_pnl, bins=60, color=AMBER, alpha=0.45, edgecolor="none")
    ax2.hist(cenarios_pnl[cenarios_pnl <= np.percentile(cenarios_pnl, percentil*100)],
             bins=60, color=RED, alpha=0.75, edgecolor="none", label="Cauda de perda")
    vl2 = np.percentile(cenarios_pnl, percentil*100)
    ax2.axvline(vl2, color=AMBER, lw=1.5, ls="--", label=f"VaR Full Val. = R$ {-vl2:,.0f}")
    ax2.set_xlabel("P&L (R$)"); ax2.set_ylabel("Frequência")
    ax2.set_title("Distribuição de P&L — Full Valuation"); ax2.legend()
    fig2.tight_layout(); st.pyplot(fig2); plt.close(fig2)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(section_title("Comparação — Métodos de VaR"), unsafe_allow_html=True)
        fig3, ax3 = plt.subplots(figsize=(6, 3.5))
        metodos = ["Paramétrico\n(Ações)", "Histórico\n(Ações)", "Full Valuation\n(A+O)"]
        valores = [var_param, var_hist, var_full]
        bars = ax3.bar(metodos, valores, color=[CYAN, GREEN, AMBER], width=0.5, edgecolor="none", alpha=0.85)
        for b, v in zip(bars, valores):
            ax3.text(b.get_x() + b.get_width()/2, b.get_height() + max(valores)*0.01,
                     f"R$ {v:,.0f}", ha="center", va="bottom", fontsize=7.5, color="#e2e8f0", fontweight="600")
        ax3.set_ylabel("VaR (R$)"); ax3.set_title("Comparativo por Método")
        fig3.tight_layout(); st.pyplot(fig3); plt.close(fig3)

    with col_b:
        st.markdown(section_title("Retorno Acumulado"), unsafe_allow_html=True)
        fig4, ax4 = plt.subplots(figsize=(6, 3.5))
        for i, t in enumerate(tickers):
            ax4.plot(precos[t] / precos[t].iloc[0], color=PALETTE[i % len(PALETTE)], lw=1.4, label=t, alpha=0.9)
        ax4.axhline(1, color=DIM, lw=0.8, ls="--")
        ax4.set_ylabel("Retorno Acumulado (base 1)"); ax4.set_title("Performance Relativa"); ax4.legend()
        fig4.tight_layout(); st.pyplot(fig4); plt.close(fig4)

    st.markdown(section_title(f"Sensibilidade · Preço da {tipo_opcao.upper()} e Delta"), unsafe_allow_html=True)
    fig5, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    ps = np.linspace(S0 * 0.65, S0 * 1.35, 300)
    ops  = [black_scholes(s, strike, vencimento_anos, taxa_lr, vol_anual, tipo_opcao) for s in ps]
    dels = [delta_bs(s, strike, vencimento_anos, taxa_lr, vol_anual, tipo_opcao) for s in ps]
    axes[0].plot(ps, ops, color=CYAN, lw=2)
    axes[0].axvline(strike, color=RED, lw=1, ls="--", alpha=0.7, label=f"K={strike}")
    axes[0].axvline(S0, color=AMBER, lw=1, ls="--", alpha=0.7, label=f"S₀={S0:.2f}")
    axes[0].fill_between(ps, ops, alpha=0.1, color=CYAN)
    axes[0].set_xlabel("Preço do Ativo (R$)"); axes[0].set_ylabel("Preço da Opção (R$)")
    axes[0].set_title(f"Preço da {tipo_opcao.upper()}"); axes[0].legend()
    axes[1].plot(ps, dels, color=GREEN, lw=2)
    axes[1].axvline(strike, color=RED, lw=1, ls="--", alpha=0.7, label=f"K={strike}")
    axes[1].axvline(S0, color=AMBER, lw=1, ls="--", alpha=0.7, label=f"S₀={S0:.2f}")
    axes[1].axhline(0.5, color=DIM, lw=0.8, ls=":")
    axes[1].fill_between(ps, dels, alpha=0.1, color=GREEN)
    axes[1].set_xlabel("Preço do Ativo (R$)"); axes[1].set_ylabel("Delta")
    axes[1].set_title(f"Delta da {tipo_opcao.upper()}"); axes[1].legend()
    fig5.tight_layout(pad=2); st.pyplot(fig5); plt.close(fig5)

# ══════════════════════════════════════════════
# TAB 3 — GREGAS
# ══════════════════════════════════════════════
with tab3:
    st.markdown(section_title(
        f"Gregas da Opção — {tipo_opcao.upper()} {ativo_opcao}",
        f"S₀={S0:.2f}  K={strike}  T={vencimento_anos}a  r={taxa_lr*100:.1f}%  σ={vol_anual*100:.1f}%"),
        unsafe_allow_html=True)

    g1, g2, g3 = st.columns(3)
    with g1: st.markdown(kpi_card("Delta (Δ)", f"{delta_op:.4f}", "Exposição direcional", CYAN,   "Δ"), unsafe_allow_html=True)
    with g2: st.markdown(kpi_card("Gamma (Γ)", f"{gamma_op:.6f}", "Convexidade do Delta", GREEN,  "Γ"), unsafe_allow_html=True)
    with g3: st.markdown(kpi_card("Vega (ν)",  f"{vega_op:.4f}",  "Sensibilidade à vol.",  AMBER,  "ν"), unsafe_allow_html=True)

    st.markdown("""<div style="background:#0d1420;border:1px solid #1e2d40;border-radius:8px;
        padding:1rem 1.5rem;margin:1rem 0;font-family:'IBM Plex Mono',monospace;
        font-size:0.72rem;color:#7e95b0;line-height:2;">
        <span style="color:#0ea5e9;">Δ</span> Para cada R$1 de variação no ativo → preço da opção muda ~Δ reais&nbsp;&nbsp;·&nbsp;&nbsp;
        <span style="color:#10b981;">Γ</span> Para cada R$1 de variação no ativo → Delta muda Γ&nbsp;&nbsp;·&nbsp;&nbsp;
        <span style="color:#f59e0b;">ν</span> Para cada +1% de volatilidade → preço da opção muda Vega reais
    </div>""", unsafe_allow_html=True)

    st.markdown(section_title("Tabela de Sensibilidade — Variação ±20% no Preço do Ativo"), unsafe_allow_html=True)
    rows = []
    for s in np.linspace(S0 * 0.80, S0 * 1.20, 11):
        d = delta_bs(s, strike, vencimento_anos, taxa_lr, vol_anual, tipo_opcao)
        g = gamma_bs(s, strike, vencimento_anos, taxa_lr, vol_anual)
        p = black_scholes(s, strike, vencimento_anos, taxa_lr, vol_anual, tipo_opcao)
        rows.append({"Preço Ativo (R$)": f"{s:.2f}", "Δ Preço (%)": f"{(s-S0)/S0*100:+.1f}%",
                     "Preço Opção (R$)": f"{p:.4f}", "Delta (Δ)": f"{d:.4f}",
                     "Gamma (Γ)": f"{g:.6f}", "Valor Posição (R$)": f"{quantidade_opcoes*p:,.2f}"})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════
# TAB 4 — HISTÓRICO (ROLLING VaR)
# ══════════════════════════════════════════════
with tab4:
    st.markdown(section_title("Evolução Histórica do VaR",
        f"VaR Histórico Rolling · Janela: {janela_rolling} dias · Confiança: {nivel_confianca*100:.0f}%"),
        unsafe_allow_html=True)

    # KPIs do rolling
    rv_valid = rolling_var.dropna()
    h1, h2, h3, h4 = st.columns(4)
    with h1: st.markdown(kpi_card("VaR Atual",   f"{float(rv_valid.iloc[-1])*100:.2f}%", "último pregão", CYAN,   "◈"), unsafe_allow_html=True)
    with h2: st.markdown(kpi_card("VaR Máximo",  f"{float(rv_valid.max())*100:.2f}%",    "pior janela",   RED,    "▲"), unsafe_allow_html=True)
    with h3: st.markdown(kpi_card("VaR Mínimo",  f"{float(rv_valid.min())*100:.2f}%",    "melhor janela", GREEN,  "▼"), unsafe_allow_html=True)
    with h4: st.markdown(kpi_card("VaR Médio",   f"{float(rv_valid.mean())*100:.2f}%",   "média histórica", AMBER, "≈"), unsafe_allow_html=True)

    # Gráfico rolling VaR + volatilidade
    st.markdown(section_title("Rolling VaR Histórico vs. Volatilidade Realizada"), unsafe_allow_html=True)
    fig_r, ax_r = plt.subplots(2, 1, figsize=(12, 6), sharex=True,
                                gridspec_kw={"height_ratios": [2, 1], "hspace": 0.06})

    ax_r[0].fill_between(rolling_var.index, rolling_var * 100, alpha=0.25, color=RED)
    ax_r[0].plot(rolling_var.index, rolling_var * 100, color=RED, lw=1.4,
                 label=f"Rolling VaR {nivel_confianca*100:.0f}% ({janela_rolling}d)")
    ax_r[0].axhline(float(rv_valid.mean()) * 100, color=AMBER, lw=0.9, ls="--", alpha=0.7, label="Média")
    ax_r[0].set_ylabel("VaR (%)")
    ax_r[0].set_title(f"Evolução do VaR Histórico Rolling — Janela {janela_rolling} dias")
    ax_r[0].legend(loc="upper left")

    ax_r[1].fill_between(rolling_vol.index, rolling_vol * 100, alpha=0.25, color=CYAN)
    ax_r[1].plot(rolling_vol.index, rolling_vol * 100, color=CYAN, lw=1.2,
                 label="Volatilidade Anualizada (%)")
    ax_r[1].set_ylabel("Vol. Anual (%)")
    ax_r[1].legend(loc="upper left")

    fig_r.tight_layout(); st.pyplot(fig_r); plt.close(fig_r)

    # Retorno realizado com zonas de breach
    st.markdown(section_title("Retorno Diário vs. Limite de VaR — Identificação de Breaches"), unsafe_allow_html=True)
    fig_b, ax_b = plt.subplots(figsize=(12, 3.8))
    ret_plot   = retorno_cart[rolling_var.dropna().index]
    var_limite = -rolling_var[rolling_var.dropna().index]
    breaches   = ret_plot[ret_plot < var_limite]

    ax_b.bar(ret_plot.index, ret_plot * 100, color=CYAN, alpha=0.4, width=1, label="Retorno diário")
    ax_b.bar(breaches.index, breaches * 100, color=RED,  alpha=0.85, width=1, label=f"Breach VaR ({len(breaches)} dias)")
    ax_b.plot(var_limite.index, var_limite * 100, color=AMBER, lw=1.2, ls="--", label="Limite VaR")
    ax_b.set_ylabel("Retorno (%)")
    ax_b.set_title(f"Breaches do VaR — {len(breaches)} ocorrências em {len(ret_plot)} dias ({len(breaches)/len(ret_plot)*100:.1f}%)")
    ax_b.legend()
    fig_b.tight_layout(); st.pyplot(fig_b); plt.close(fig_b)

    # Tabela de piores 5 janelas
    st.markdown(section_title("Top 5 — Piores Janelas de VaR"), unsafe_allow_html=True)
    top5 = (rolling_var * valor_acoes).nlargest(5).reset_index()
    top5.columns = ["Data", "VaR Janela (R$)"]
    top5["VaR (%)"]       = (rolling_var * 100).nlargest(5).values
    top5["Data"]          = top5["Data"].dt.strftime("%Y-%m-%d")
    top5["VaR Janela (R$)"] = top5["VaR Janela (R$)"].apply(lambda x: f"R$ {x:,.2f}")
    top5["VaR (%)"]       = top5["VaR (%)"].apply(lambda x: f"{x:.2f}%")
    st.dataframe(top5, use_container_width=True, hide_index=True)

    # Gráfico de preços
    st.markdown(section_title("Preços Históricos dos Ativos"), unsafe_allow_html=True)
    fig_p, ax_p = plt.subplots(figsize=(12, 3.5))
    for i, t in enumerate(tickers):
        ax_p.plot(precos[t].index, precos[t].values, color=PALETTE[i % len(PALETTE)], lw=1.3, label=t)
    ax_p.set_ylabel("Preço (R$)"); ax_p.set_title("Evolução de Preços"); ax_p.legend()
    fig_p.tight_layout(); st.pyplot(fig_p); plt.close(fig_p)

# ══════════════════════════════════════════════
# TAB 5 — STRESS TEST
# ══════════════════════════════════════════════
with tab5:
    st.markdown(section_title("Stress Test — Simulação de Choques de Mercado",
        "Aplica choques extremos à carteira e calcula a perda potencial"), unsafe_allow_html=True)

    st.markdown("""
    <div style="background:#0d1420;border:1px solid #1e2d40;border-left:3px solid #ef4444;
                border-radius:8px;padding:1rem 1.4rem;margin-bottom:1.2rem;
                font-family:'IBM Plex Mono',monospace;font-size:0.72rem;color:#7e95b0;line-height:1.7;">
        O Stress Test complementa o VaR avaliando cenários extremos que podem não estar na janela histórica.<br>
        O VaR responde <span style="color:#e2e8f0;">"perda máxima em condições normais"</span> — o Stress Test responde
        <span style="color:#ef4444;">"perda em condições extremas"</span>.
    </div>""", unsafe_allow_html=True)

    # Cenários predefinidos
    cenarios_pre = {
        "Crise 2008 (Lehman Brothers)":       {"choque_acoes": -0.45, "choque_vol": 2.5,  "descricao": "Pior crise financeira desde 1929. S&P -57%."},
        "COVID-19 (mar/2020)":                {"choque_acoes": -0.35, "choque_vol": 3.0,  "descricao": "Queda mais rápida da história. Ibovespa -46% em 5 semanas."},
        "Crise Brasil 2002 (Lula medo)":      {"choque_acoes": -0.40, "choque_vol": 2.0,  "descricao": "Risco-país disparou. Dólar +50%. Ibovespa -45%."},
        "Flash Crash (mai/2010)":             {"choque_acoes": -0.10, "choque_vol": 1.5,  "descricao": "Queda intraday de 10% no Dow Jones em minutos."},
        "Choque de Juros EUA (2022)":         {"choque_acoes": -0.20, "choque_vol": 1.3,  "descricao": "Fed elevou juros mais rápido em 40 anos. Nasdaq -33%."},
        "Cenário Personalizado":              {"choque_acoes":  0.0,  "choque_vol": 1.0,  "descricao": "Configure o choque manualmente abaixo."},
    }

    sc1, sc2 = st.columns([2, 1])
    with sc1:
        cenario_sel = st.selectbox("Selecione o cenário de stress",
                                    options=list(cenarios_pre.keys()), index=0)
    with sc2:
        st.markdown("<div style='height:1.85rem'></div>", unsafe_allow_html=True)

    cfg = cenarios_pre[cenario_sel]

    # Se personalizado, mostrar sliders
    if cenario_sel == "Cenário Personalizado":
        sa1, sa2 = st.columns(2)
        with sa1:
            choque_pct = st.slider("Choque nas ações (%)", min_value=-80, max_value=30,
                                    value=int(cfg["choque_acoes"]*100), step=1) / 100
        with sa2:
            choque_vol_mult = st.slider("Multiplicador de volatilidade", min_value=0.5,
                                         max_value=5.0, value=cfg["choque_vol"], step=0.1)
    else:
        choque_pct      = cfg["choque_acoes"]
        choque_vol_mult = cfg["choque_vol"]

    # Descrição do cenário
    st.markdown(f"""
    <div style="background:#111827;border:1px solid #1e2d40;border-radius:6px;
                padding:0.8rem 1.2rem;margin:0.5rem 0 1rem 0;display:flex;gap:2rem;flex-wrap:wrap;">
        <div>
            <span style="font-family:'IBM Plex Mono',monospace;font-size:0.6rem;color:#7e95b0;
                         letter-spacing:0.1em;text-transform:uppercase;">Choque nas Ações</span>
            <div style="font-family:'IBM Plex Mono',monospace;font-size:1.2rem;font-weight:600;
                        color:#ef4444;margin-top:2px;">{choque_pct*100:+.1f}%</div>
        </div>
        <div>
            <span style="font-family:'IBM Plex Mono',monospace;font-size:0.6rem;color:#7e95b0;
                         letter-spacing:0.1em;text-transform:uppercase;">Mult. Volatilidade</span>
            <div style="font-family:'IBM Plex Mono',monospace;font-size:1.2rem;font-weight:600;
                        color:#f59e0b;margin-top:2px;">×{choque_vol_mult:.1f}</div>
        </div>
        <div style="flex:1;min-width:200px;">
            <span style="font-family:'IBM Plex Mono',monospace;font-size:0.6rem;color:#7e95b0;
                         letter-spacing:0.1em;text-transform:uppercase;">Contexto Histórico</span>
            <div style="font-size:0.78rem;color:#7e95b0;margin-top:4px;
                        font-family:'IBM Plex Sans',sans-serif;">{cfg['descricao']}</div>
        </div>
    </div>""", unsafe_allow_html=True)

    # ── Cálculo do Stress Test ──
    vol_stress = vol_anual * choque_vol_mult

    novo_valor_acoes_stress = sum(
        quantidades_acoes[t] * float(ultimos_precos[t]) * (1 + choque_pct)
        for t in tickers
    )

    S_stress    = S0 * (1 + choque_pct)
    preco_op_stress = black_scholes(S_stress, strike, vencimento_anos, taxa_lr, vol_stress, tipo_opcao)
    novo_valor_opcoes_stress = quantidade_opcoes * preco_op_stress
    novo_valor_total_stress  = novo_valor_acoes_stress + novo_valor_opcoes_stress

    perda_acoes   = valor_acoes - novo_valor_acoes_stress
    perda_opcoes  = valor_opcoes - novo_valor_opcoes_stress
    perda_total   = valor_total - novo_valor_total_stress
    perda_pct     = perda_total / valor_total * 100

    # Comparação VaR vs Stress
    multiplo_param = perda_total / var_param if var_param > 0 else 0
    multiplo_hist  = perda_total / var_hist  if var_hist  > 0 else 0
    multiplo_full  = perda_total / var_full  if var_full  > 0 else 0

    # KPIs
    st.markdown(section_title("Resultado do Stress Test"), unsafe_allow_html=True)
    sk1, sk2, sk3, sk4 = st.columns(4)
    with sk1: st.markdown(kpi_card("Perda Total",    f"R$ {perda_total:,.0f}",    f"{perda_pct:.1f}% do portfólio", RED,    "▼"), unsafe_allow_html=True)
    with sk2: st.markdown(kpi_card("Perda nas Ações", f"R$ {perda_acoes:,.0f}",   f"{perda_acoes/valor_acoes*100:.1f}% das ações", AMBER, "▼"), unsafe_allow_html=True)
    with sk3: st.markdown(kpi_card("Perda nas Opções",f"R$ {perda_opcoes:,.0f}",  f"{perda_opcoes/valor_opcoes*100:.1f}% das opções" if valor_opcoes > 0 else "—", PURPLE, "▼"), unsafe_allow_html=True)
    with sk4: st.markdown(kpi_card("Novo Valor Total",f"R$ {novo_valor_total_stress:,.0f}", "após o choque", CYAN, "◈"), unsafe_allow_html=True)

    # Comparação Stress vs VaR
    st.markdown(section_title("Stress vs. VaR — Contexto da Perda"), unsafe_allow_html=True)
    st.markdown(f"""
    <div style="background:#111827;border:1px solid #1e2d40;border-radius:8px;
                padding:1.2rem 1.5rem;font-family:'IBM Plex Mono',monospace;font-size:0.8rem;
                color:#7e95b0;line-height:2.2;">
        A perda no stress é <span style="color:#ef4444;font-size:1rem;font-weight:600;">
        {multiplo_param:.1f}×</span> o VaR Paramétrico&nbsp;&nbsp;·&nbsp;&nbsp;
        <span style="color:#ef4444;font-size:1rem;font-weight:600;">{multiplo_hist:.1f}×</span>
        o VaR Histórico&nbsp;&nbsp;·&nbsp;&nbsp;
        <span style="color:#ef4444;font-size:1rem;font-weight:600;">{multiplo_full:.1f}×</span>
        o Full Valuation
        <br>
        <span style="color:#3d5470;font-size:0.68rem;">
        Isso ilustra por que o VaR não é suficiente sozinho — cenários extremos requerem stress testing.</span>
    </div>""", unsafe_allow_html=True)

    # Gráfico: waterfall de perda + comparação
    col_w1, col_w2 = st.columns(2)

    with col_w1:
        st.markdown(section_title("Decomposição da Perda"), unsafe_allow_html=True)
        fig_st1, ax_st1 = plt.subplots(figsize=(6, 4))
        labels = ["Portfólio\nAtual", "Perda\nAções", "Perda\nOpções", "Portfólio\nApós Choque"]
        valores_w = [valor_total, -perda_acoes, -perda_opcoes, novo_valor_total_stress]
        cores_w   = [CYAN, RED, AMBER, GREEN if novo_valor_total_stress > 0 else RED]
        bars = ax_st1.bar(labels, [valor_total, perda_acoes, perda_opcoes, novo_valor_total_stress],
                           color=cores_w, width=0.5, edgecolor="none", alpha=0.85)
        for b, v in zip(bars, [valor_total, perda_acoes, perda_opcoes, novo_valor_total_stress]):
            ax_st1.text(b.get_x() + b.get_width()/2, b.get_height() + valor_total*0.005,
                        f"R$\n{v:,.0f}", ha="center", va="bottom", fontsize=7, color="#e2e8f0")
        ax_st1.set_ylabel("R$")
        ax_st1.set_title(f"Impacto: {cenario_sel[:30]}...")
        fig_st1.tight_layout(); st.pyplot(fig_st1); plt.close(fig_st1)

    with col_w2:
        st.markdown(section_title("Stress vs. VaR"), unsafe_allow_html=True)
        fig_st2, ax_st2 = plt.subplots(figsize=(6, 4))
        nomes = ["VaR\nParam.", "VaR\nHist.", "Full\nVal.", "Stress\nTest"]
        vals  = [var_param, var_hist, var_full, perda_total]
        cors  = [CYAN, GREEN, AMBER, RED]
        brs   = ax_st2.bar(nomes, vals, color=cors, width=0.5, edgecolor="none", alpha=0.85)
        for b, v in zip(brs, vals):
            ax_st2.text(b.get_x() + b.get_width()/2, b.get_height() + max(vals)*0.01,
                        f"R$ {v:,.0f}", ha="center", va="bottom", fontsize=7.5, color="#e2e8f0", fontweight="600")
        ax_st2.set_ylabel("Perda (R$)")
        ax_st2.set_title("VaR vs. Perda no Stress Test")
        fig_st2.tight_layout(); st.pyplot(fig_st2); plt.close(fig_st2)

    # Curva de perda por nível de choque
    st.markdown(section_title("Curva de Perda por Intensidade do Choque"), unsafe_allow_html=True)
    fig_c, ax_c = plt.subplots(figsize=(12, 3.8))
    choques_range = np.linspace(-0.60, 0.20, 200)
    perdas_range  = []
    for ch in choques_range:
        nva = sum(quantidades_acoes[t] * float(ultimos_precos[t]) * (1 + ch) for t in tickers)
        nop = black_scholes(S0 * (1 + ch), strike, vencimento_anos, taxa_lr,
                            vol_anual * choque_vol_mult, tipo_opcao)
        perdas_range.append(valor_total - (nva + quantidade_opcoes * nop))

    ax_c.fill_between(choques_range * 100, perdas_range, alpha=0.2, color=RED,
                       where=np.array(perdas_range) > 0)
    ax_c.fill_between(choques_range * 100, perdas_range, alpha=0.2, color=GREEN,
                       where=np.array(perdas_range) <= 0)
    ax_c.plot(choques_range * 100, perdas_range, color="#e2e8f0", lw=1.8)
    ax_c.axhline(var_param, color=CYAN,  lw=1, ls="--", alpha=0.7, label=f"VaR Param R$ {var_param:,.0f}")
    ax_c.axhline(var_hist,  color=GREEN, lw=1, ls="--", alpha=0.7, label=f"VaR Hist.  R$ {var_hist:,.0f}")
    ax_c.axhline(var_full,  color=AMBER, lw=1, ls="--", alpha=0.7, label=f"Full Val.  R$ {var_full:,.0f}")
    ax_c.axvline(choque_pct * 100, color=RED, lw=1.2, ls=":", label=f"Cenário atual: {choque_pct*100:.1f}%")
    ax_c.axhline(0, color=DIM, lw=0.8)
    ax_c.set_xlabel("Choque nas Ações (%)")
    ax_c.set_ylabel("Perda da Carteira (R$)")
    ax_c.set_title("Perfil de Perda — Ações + Opções vs. Magnitude do Choque")
    ax_c.legend(fontsize=7.5)
    fig_c.tight_layout(); st.pyplot(fig_c); plt.close(fig_c)

# ══════════════════════════════════════════════
# TAB 6 — TEORIA
# ══════════════════════════════════════════════
with tab6:
    st.markdown(section_title("Fundamentos Teóricos"), unsafe_allow_html=True)
    teorias = [
        ("📐  VaR Paramétrico", CYAN, """
Assume que os retornos seguem **distribuição normal**.

**Fórmula:**
```
VaR = Z × σ × V × √h
```
onde `Z` = quantil normal, `σ` = volatilidade, `V` = valor, `h` = horizonte.

**Vantagens:** simples, rápido, fácil de comunicar.  
**Limitações:** assume normalidade; não captura caudas gordas; inadequado para opções.
        """),
        ("📊  VaR Histórico & Rolling VaR", GREEN, """
Usa diretamente os **retornos históricos observados** — sem hipótese distribucional.

O **Rolling VaR** recalcula o VaR histórico a cada dia usando uma janela móvel de N dias.
Isso permite observar como o risco da carteira evolui ao longo do tempo.

**Breach:** ocorre quando o retorno realizado ultrapassa o limite do VaR.
Reguladores esperam ~5% de breaches para VaR 95%.

**Limitações:** depende da janela; assume que o passado representa o futuro.
        """),
        ("⚡  VaR Full Valuation", AMBER, """
**Reprecifica toda a carteira** (incluindo opções via Black-Scholes) em cada cenário histórico.

Captura a convexidade que o VaR Paramétrico ignora.

**Vantagens:** mais preciso para derivativos; capta não linearidade.
**Limitações:** computacionalmente mais custoso.
        """),
        ("🔥  Stress Test", RED, """
O Stress Test aplica choques extremos para avaliar perdas em cenários severos.

**Diferença crucial em relação ao VaR:**
- VaR → perda máxima em condições *normais* de mercado
- Stress Test → perda em condições *extremas*

**Por que usar?**
O VaR histórico não captura crises que não ocorreram na janela histórica usada.
O Stress Test com cenários históricos (2008, COVID) complementa essa limitação.

Reguladores como o Banco Central exigem stress testing regular para instituições financeiras.
        """),
        ("⚖️  Black-Scholes (1973)", PURPLE, """
**Call europeia:**
```
C = S·N(d₁) − K·e^{−rT}·N(d₂)
```
**Put europeia:**
```
P = K·e^{−rT}·N(−d₂) − S·N(−d₁)
```
```
d₁ = [ln(S/K) + (r + σ²/2)·T] / (σ·√T)
d₂ = d₁ − σ·√T
```
Hipóteses: sem arbitragem, volatilidade constante, taxa constante, retornos lognormais.
        """),
        ("⚠️  Limitações do VaR", "#7e95b0", """
1. **Não informa a magnitude da perda além do VaR** (cauda cega)
2. **Depende da janela histórica**
3. **Caudas gordas** — mercados têm assimetria e excesso de curtose
4. **Não substitui stress test**
5. **Correlações instáveis** — em crises, correlações se aproximam de 1

Na prática: VaR + Expected Shortfall + Stress Test + limites de perda.
        """),
    ]
    for titulo, cor, texto in teorias:
        with st.expander(titulo):
            st.markdown(f'<div style="border-left:3px solid {cor};padding-left:1rem;margin-bottom:0.5rem;"></div>', unsafe_allow_html=True)
            st.markdown(texto)

# ============================================================
# RODAPÉ
# ============================================================
st.markdown("""
<div style="margin-top:3rem;padding-top:1.2rem;border-top:1px solid #1e2d40;
            display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:0.5rem;">
    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.62rem;color:#3d5470;">
        VaR Dashboard · Modelagem Aplicada ao Mercado Financeiro
    </div>
    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.62rem;color:#3d5470;">
        Black-Scholes · Paramétrico · Histórico · Full Valuation · Rolling VaR · Stress Test
    </div>
</div>""", unsafe_allow_html=True)
