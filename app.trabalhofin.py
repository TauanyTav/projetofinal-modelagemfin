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
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import yfinance as yf
from scipy.stats import norm

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
# CSS CUSTOMIZADO — DARK FINANCIAL THEME
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600;700&display=swap');

/* ── ROOT VARIABLES ── */
:root {
    --bg-primary:    #080c14;
    --bg-secondary:  #0d1420;
    --bg-card:       #111827;
    --bg-card-hover: #162032;
    --border:        #1e2d40;
    --border-glow:   #1e4976;
    --accent-blue:   #1d6fa4;
    --accent-cyan:   #0ea5e9;
    --accent-green:  #10b981;
    --accent-red:    #ef4444;
    --accent-amber:  #f59e0b;
    --text-primary:  #e2e8f0;
    --text-secondary:#7e95b0;
    --text-dim:      #3d5470;
    --mono:          'IBM Plex Mono', monospace;
    --sans:          'IBM Plex Sans', sans-serif;
}

/* ── GLOBAL RESET ── */
html, body, [class*="css"] {
    font-family: var(--sans) !important;
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
}

/* ── HIDE STREAMLIT CHROME ── */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
.stDecoration { display: none; }
div[data-testid="stToolbar"] { display: none; }

/* ── APP BACKGROUND ── */
.stApp {
    background: var(--bg-primary) !important;
    background-image:
        radial-gradient(ellipse at 10% 0%, rgba(14,165,233,0.04) 0%, transparent 50%),
        radial-gradient(ellipse at 90% 100%, rgba(16,185,129,0.03) 0%, transparent 50%) !important;
}

/* ── MAIN CONTENT AREA ── */
.main .block-container {
    padding: 1.5rem 2rem 3rem !important;
    max-width: 1600px !important;
}

/* ── SIDEBAR ── */
section[data-testid="stSidebar"] {
    background: var(--bg-secondary) !important;
    border-right: 1px solid var(--border) !important;
    padding-top: 0 !important;
}
section[data-testid="stSidebar"] > div {
    padding-top: 1rem !important;
}
section[data-testid="stSidebar"] .stMarkdown p {
    color: var(--text-secondary) !important;
    font-size: 0.72rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.12em !important;
    font-weight: 500 !important;
}

/* ── SIDEBAR INPUTS ── */
.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stSelectbox > div > div {
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
.stSelectbox > div > div {
    background: var(--bg-card) !important;
}
[data-baseweb="select"] > div {
    background: var(--bg-card) !important;
    border-color: var(--border) !important;
    color: var(--text-primary) !important;
}
[data-baseweb="popover"] {
    background: var(--bg-card) !important;
}

/* ── DATE INPUT ── */
.stDateInput > div > div > input {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
    font-family: var(--mono) !important;
    border-radius: 6px !important;
}

/* ── LABELS ── */
.stTextInput label, .stNumberInput label,
.stSelectbox label, .stDateInput label,
.stSlider label {
    color: var(--text-secondary) !important;
    font-size: 0.72rem !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}

/* ── BUTTON ── */
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

/* ── TABS ── */
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
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    padding: 0.75rem 1.5rem !important;
    transition: all 0.2s !important;
}
.stTabs [aria-selected="true"] {
    color: var(--accent-cyan) !important;
    border-bottom: 2px solid var(--accent-cyan) !important;
    background: transparent !important;
}
.stTabs [data-baseweb="tab-panel"] {
    padding-top: 1.5rem !important;
}

/* ── DATAFRAME ── */
.stDataFrame {
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    overflow: hidden !important;
}
iframe[title="st_dataframe"] {
    border-radius: 8px !important;
}

/* ── EXPANDER ── */
.streamlit-expanderHeader {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    color: var(--text-primary) !important;
    font-family: var(--sans) !important;
    font-size: 0.85rem !important;
}
.streamlit-expanderContent {
    background: var(--bg-secondary) !important;
    border: 1px solid var(--border) !important;
    border-top: none !important;
    border-radius: 0 0 6px 6px !important;
}

/* ── SPINNER ── */
.stSpinner > div {
    border-top-color: var(--accent-cyan) !important;
}

/* ── SCROLLBAR ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent-blue); }

/* ── METRIC OVERRIDE (hide default) ── */
[data-testid="stMetric"] { display: none !important; }

</style>
""", unsafe_allow_html=True)

# ============================================================
# COMPONENTES HTML CUSTOMIZADOS
# ============================================================

def render_header():
    st.markdown("""
    <div style="
        display: flex;
        align-items: flex-end;
        justify-content: space-between;
        padding: 1.5rem 0 1.2rem 0;
        border-bottom: 1px solid #1e2d40;
        margin-bottom: 1.5rem;
    ">
        <div>
            <div style="
                font-family: 'IBM Plex Mono', monospace;
                font-size: 0.65rem;
                color: #0ea5e9;
                letter-spacing: 0.25em;
                text-transform: uppercase;
                margin-bottom: 0.4rem;
            ">Modelagem Aplicada ao Mercado Financeiro</div>
            <h1 style="
                font-family: 'IBM Plex Sans', sans-serif;
                font-size: 1.9rem;
                font-weight: 700;
                color: #e2e8f0;
                margin: 0;
                letter-spacing: -0.02em;
                line-height: 1.1;
            ">Value at Risk <span style="color:#0ea5e9;">Dashboard</span></h1>
        </div>
        <div style="text-align:right;">
            <div style="
                font-family: 'IBM Plex Mono', monospace;
                font-size: 0.65rem;
                color: #3d5470;
                letter-spacing: 0.1em;
            ">SISTEMA DE GESTÃO DE RISCO</div>
            <div style="
                font-family: 'IBM Plex Mono', monospace;
                font-size: 0.7rem;
                color: #7e95b0;
                margin-top: 2px;
            ">Black-Scholes · Full Valuation · VaR</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def kpi_card(label, value, subtitle="", color="#0ea5e9", icon=""):
    return f"""
    <div style="
        background: #111827;
        border: 1px solid #1e2d40;
        border-top: 2px solid {color};
        border-radius: 8px;
        padding: 1.1rem 1.2rem;
        height: 100%;
        position: relative;
        overflow: hidden;
    ">
        <div style="
            position: absolute; top: 0; right: 0;
            width: 60px; height: 60px;
            background: radial-gradient(circle at top right, {color}18, transparent 70%);
        "></div>
        <div style="
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.6rem;
            color: #7e95b0;
            letter-spacing: 0.15em;
            text-transform: uppercase;
            margin-bottom: 0.5rem;
        ">{icon} {label}</div>
        <div style="
            font-family: 'IBM Plex Mono', monospace;
            font-size: 1.35rem;
            font-weight: 600;
            color: {color};
            line-height: 1.1;
            margin-bottom: 0.25rem;
        ">{value}</div>
        <div style="
            font-family: 'IBM Plex Sans', sans-serif;
            font-size: 0.68rem;
            color: #3d5470;
        ">{subtitle}</div>
    </div>
    """


def var_card(label, value, pct, color, description):
    return f"""
    <div style="
        background: #111827;
        border: 1px solid #1e2d40;
        border-left: 3px solid {color};
        border-radius: 8px;
        padding: 1.2rem 1.4rem;
        height: 100%;
    ">
        <div style="
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.58rem;
            color: #7e95b0;
            letter-spacing: 0.18em;
            text-transform: uppercase;
            margin-bottom: 0.6rem;
        ">{label}</div>
        <div style="
            font-family: 'IBM Plex Mono', monospace;
            font-size: 1.6rem;
            font-weight: 600;
            color: {color};
            margin-bottom: 0.2rem;
            letter-spacing: -0.02em;
        ">{value}</div>
        <div style="
            display: flex;
            align-items: center;
            gap: 0.5rem;
            margin-bottom: 0.6rem;
        ">
            <span style="
                background: {color}22;
                color: {color};
                font-family: 'IBM Plex Mono', monospace;
                font-size: 0.68rem;
                padding: 1px 7px;
                border-radius: 3px;
            ">{pct} do portfólio</span>
        </div>
        <div style="
            font-size: 0.7rem;
            color: #3d5470;
            font-family: 'IBM Plex Sans', sans-serif;
            line-height: 1.4;
        ">{description}</div>
    </div>
    """


def section_title(text, sub=""):
    sub_html = f'<div style="font-size:0.72rem;color:#3d5470;font-family:IBM Plex Mono,monospace;margin-top:2px;">{sub}</div>' if sub else ""
    return f"""
    <div style="margin: 1.8rem 0 0.9rem 0;">
        <div style="
            font-family: 'IBM Plex Sans', sans-serif;
            font-size: 0.78rem;
            font-weight: 600;
            color: #7e95b0;
            text-transform: uppercase;
            letter-spacing: 0.14em;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid #1e2d40;
        ">{text}</div>
        {sub_html}
    </div>
    """


def sidebar_section(text):
    st.sidebar.markdown(f"""
    <div style="
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.58rem;
        color: #0ea5e9;
        letter-spacing: 0.2em;
        text-transform: uppercase;
        padding: 1rem 0 0.4rem 0;
        border-top: 1px solid #1e2d40;
        margin-top: 0.5rem;
    ">{text}</div>
    """, unsafe_allow_html=True)


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
        "figure.dpi":        130,
    })


# ============================================================
# FUNÇÕES BLACK-SCHOLES
# ============================================================

def black_scholes(S, K, T, r, sigma, tipo="call"):
    if T <= 0:
        return max(S - K, 0) if tipo == "call" else max(K - S, 0)
    if sigma <= 0:
        return (max(S - K * np.exp(-r * T), 0) if tipo == "call"
                else max(K * np.exp(-r * T) - S, 0))
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if tipo == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def delta_bs(S, K, T, r, sigma, tipo="call"):
    if T <= 0 or sigma <= 0:
        return 0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1) if tipo == "call" else norm.cdf(d1) - 1


def gamma_bs(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return 0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))


def vega_bs(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return 0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T)


# ============================================================
# SIDEBAR
# ============================================================

# Logo / título sidebar
st.sidebar.markdown("""
<div style="padding: 0.8rem 0 0.6rem 0; border-bottom: 1px solid #1e2d40; margin-bottom: 0.5rem;">
    <div style="font-family:'IBM Plex Mono',monospace; font-size:0.6rem; color:#0ea5e9;
                letter-spacing:0.22em; text-transform:uppercase;">Risk Engine</div>
    <div style="font-family:'IBM Plex Sans',sans-serif; font-size:1.1rem; font-weight:700;
                color:#e2e8f0; margin-top:2px;">VaR Calculator</div>
</div>
""", unsafe_allow_html=True)

sidebar_section("▸ Carteira de Ações")

tickers_input = st.sidebar.text_input("Tickers", value="PETR4.SA, VALE3.SA, ITUB4.SA",
                                       help="Separados por vírgula. Ex: PETR4.SA, VALE3.SA")
tickers = [t.strip() for t in tickers_input.split(",") if t.strip()]

quantidades_input = st.sidebar.text_input("Quantidades", value="1000, 800, 1200",
                                           help="Mesma ordem dos tickers, separadas por vírgula")
try:
    quantidades_lista = [int(q.strip()) for q in quantidades_input.split(",")]
    if len(quantidades_lista) != len(tickers):
        st.sidebar.error("Nº de quantidades ≠ nº de tickers")
        quantidades_lista = [1000] * len(tickers)
except ValueError:
    st.sidebar.error("Use apenas números inteiros.")
    quantidades_lista = [1000] * len(tickers)

quantidades_acoes = dict(zip(tickers, quantidades_lista))

sidebar_section("▸ Período & Parâmetros")

data_inicio = st.sidebar.date_input("Data de início", value=pd.to_datetime("2022-01-01"))

nivel_confianca = st.sidebar.selectbox(
    "Nível de confiança",
    options=[0.90, 0.95, 0.975, 0.99],
    index=1,
    format_func=lambda x: f"{x*100:.1f}%"
)

horizonte_dias = st.sidebar.number_input("Horizonte (dias)", min_value=1, max_value=30, value=1)

sidebar_section("▸ Opção Europeia")

ativo_opcao      = st.sidebar.selectbox("Ativo objeto", options=tickers)
tipo_opcao       = st.sidebar.selectbox("Tipo", options=["call", "put"])
quantidade_opcoes = st.sidebar.number_input("Quantidade", min_value=0, value=1000, step=100)
strike           = st.sidebar.number_input("Strike (K)", min_value=1.0, value=40.0, step=0.5)
taxa_livre_risco = st.sidebar.number_input("Taxa livre de risco (a.a.)", min_value=0.0,
                                            max_value=1.0, value=0.105, step=0.005, format="%.3f")
vencimento_anos  = st.sidebar.number_input("Vencimento (anos)", min_value=0.01,
                                            max_value=5.0, value=0.25, step=0.05, format="%.2f")

st.sidebar.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
calcular = st.sidebar.button("▶  CALCULAR VaR")

st.sidebar.markdown("""
<div style="margin-top:2rem; padding-top:1rem; border-top:1px solid #1e2d40;
            font-family:'IBM Plex Mono',monospace; font-size:0.58rem; color:#3d5470;
            line-height:1.8;">
    Métodos<br>
    ├ VaR Paramétrico<br>
    ├ VaR Histórico<br>
    └ VaR Full Valuation<br><br>
    Modelo de opções<br>
    └ Black-Scholes (1973)
</div>
""", unsafe_allow_html=True)

# ============================================================
# HEADER PRINCIPAL
# ============================================================

render_header()

# ============================================================
# ESTADO INICIAL
# ============================================================

if not calcular:
    st.markdown("""
    <div style="
        background: #111827;
        border: 1px solid #1e2d40;
        border-radius: 10px;
        padding: 3rem 2.5rem;
        text-align: center;
        margin-top: 1rem;
    ">
        <div style="font-size:2.5rem; margin-bottom:1rem;">📉</div>
        <div style="
            font-family:'IBM Plex Sans',sans-serif;
            font-size:1.1rem; font-weight:600;
            color:#e2e8f0; margin-bottom:0.5rem;
        ">Configure a carteira e calcule o VaR</div>
        <div style="
            font-family:'IBM Plex Mono',monospace;
            font-size:0.75rem; color:#3d5470; line-height:1.8;
        ">
            Selecione os ativos → defina quantidades → parametrize a opção → clique em Calcular VaR
        </div>
        <div style="
            display:flex; justify-content:center; gap:2rem;
            margin-top:2rem; flex-wrap:wrap;
        ">
            <div style="text-align:center;">
                <div style="font-family:'IBM Plex Mono',monospace; font-size:0.6rem;
                            color:#0ea5e9; letter-spacing:0.1em;">MÉTODO 01</div>
                <div style="font-size:0.85rem; color:#7e95b0; margin-top:4px;">VaR Paramétrico</div>
            </div>
            <div style="color:#1e2d40; font-size:1.2rem;">·</div>
            <div style="text-align:center;">
                <div style="font-family:'IBM Plex Mono',monospace; font-size:0.6rem;
                            color:#10b981; letter-spacing:0.1em;">MÉTODO 02</div>
                <div style="font-size:0.85rem; color:#7e95b0; margin-top:4px;">VaR Histórico</div>
            </div>
            <div style="color:#1e2d40; font-size:1.2rem;">·</div>
            <div style="text-align:center;">
                <div style="font-family:'IBM Plex Mono',monospace; font-size:0.6rem;
                            color:#f59e0b; letter-spacing:0.1em;">MÉTODO 03</div>
                <div style="font-size:0.85rem; color:#7e95b0; margin-top:4px;">Full Valuation</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ============================================================
# DOWNLOAD DE DADOS
# ============================================================

with st.spinner("Conectando ao mercado..."):
    try:
        precos = yf.download(
            tickers, start=str(data_inicio),
            auto_adjust=True, progress=False
        )["Close"]
        if isinstance(precos, pd.Series):
            precos = precos.to_frame(tickers[0])
        precos = precos.dropna()
        if precos.empty:
            st.error("Nenhum dado encontrado. Verifique os tickers.")
            st.stop()
        retornos = precos.pct_change().dropna()
    except Exception as e:
        st.error(f"Erro ao baixar dados: {e}")
        st.stop()

# ============================================================
# CÁLCULOS
# ============================================================

ultimos_precos = precos.iloc[-1]
valor_acoes    = sum(quantidades_acoes[t] * ultimos_precos[t] for t in tickers)

S0        = ultimos_precos[ativo_opcao]
vol_anual = retornos[ativo_opcao].std() * np.sqrt(252)

preco_opcao_hoje = black_scholes(S0, strike, vencimento_anos, taxa_livre_risco, vol_anual, tipo_opcao)
valor_opcoes     = quantidade_opcoes * preco_opcao_hoje
valor_total      = valor_acoes + valor_opcoes

delta_op = delta_bs(S0, strike, vencimento_anos, taxa_livre_risco, vol_anual, tipo_opcao)
gamma_op = gamma_bs(S0, strike, vencimento_anos, taxa_livre_risco, vol_anual)
vega_op  = vega_bs(S0, strike, vencimento_anos, taxa_livre_risco, vol_anual)

pesos            = np.array([quantidades_acoes[t] * ultimos_precos[t] / valor_acoes for t in tickers])
retorno_carteira = retornos.dot(pesos)
media_cart       = retorno_carteira.mean()
vol_cart         = retorno_carteira.std()
percentil        = 1 - nivel_confianca
z                = norm.ppf(1 - nivel_confianca)

var_param   = -(media_cart * horizonte_dias + z * vol_cart * np.sqrt(horizonte_dias)) * valor_acoes
var_hist    = -np.percentile(retorno_carteira, percentil * 100) * valor_acoes

cenarios_pnl = []
for i in range(len(retornos)):
    choque         = retornos.iloc[i]
    novos_precos   = ultimos_precos * (1 + choque)
    novo_val_acoes = sum(quantidades_acoes[t] * novos_precos[t] for t in tickers)
    T_cen          = max(vencimento_anos - horizonte_dias / 252, 0)
    novo_op        = black_scholes(novos_precos[ativo_opcao], strike, T_cen,
                                   taxa_livre_risco, vol_anual, tipo_opcao)
    cenarios_pnl.append((novo_val_acoes + quantidade_opcoes * novo_op) - valor_total)

cenarios_pnl   = np.array(cenarios_pnl)
var_full        = -np.percentile(cenarios_pnl, percentil * 100)

# ============================================================
# TABS
# ============================================================

apply_chart_style()

tab1, tab2, tab3, tab4 = st.tabs([
    "  Resumo  ",
    "  Gráficos  ",
    "  Gregas  ",
    "  Teoria  "
])

# ──────────────────────────────────────────────
# TAB 1 — RESUMO
# ──────────────────────────────────────────────
with tab1:

    # KPIs principais
    st.markdown(section_title("Composição da Carteira", f"Posição em {len(tickers)} ativo(s) + opções {tipo_opcao.upper()}"), unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(kpi_card("Ações", f"R$ {valor_acoes:,.0f}",
                              f"{len(tickers)} ativos", "#0ea5e9", "◈"), unsafe_allow_html=True)
    with c2:
        st.markdown(kpi_card("Opções", f"R$ {valor_opcoes:,.0f}",
                              f"{quantidade_opcoes:,} {tipo_opcao}s", "#10b981", "◈"), unsafe_allow_html=True)
    with c3:
        st.markdown(kpi_card("Total", f"R$ {valor_total:,.0f}",
                              "valor de mercado", "#f59e0b", "◈"), unsafe_allow_html=True)
    with c4:
        st.markdown(kpi_card("Vol. Diária", f"{vol_cart*100:.2f}%",
                              f"anual: {vol_cart*np.sqrt(252)*100:.1f}%", "#a78bfa", "◈"), unsafe_allow_html=True)

    # Tabela posições
    st.markdown(section_title("Posições em Ações"), unsafe_allow_html=True)

    dados_pos = []
    for t in tickers:
        preco = ultimos_precos[t]
        qtd   = quantidades_acoes[t]
        val   = qtd * preco
        peso  = val / valor_acoes
        dados_pos.append({
            "Ticker":         t,
            "Último Preço":   f"R$ {preco:.2f}",
            "Quantidade":     f"{qtd:,}",
            "Valor (R$)":     f"{val:,.2f}",
            "Peso":           f"{peso*100:.1f}%",
            "Retorno YTD":    f"{((precos[t].iloc[-1]/precos[t].iloc[0])-1)*100:.1f}%"
        })

    st.dataframe(
        pd.DataFrame(dados_pos),
        use_container_width=True,
        hide_index=True
    )

    # VaR Cards
    st.markdown(section_title("Resultados de VaR",
        f"Nível de confiança: {nivel_confianca*100:.1f}%  ·  Horizonte: {horizonte_dias} dia(s)"), unsafe_allow_html=True)

    v1, v2, v3 = st.columns(3)
    with v1:
        st.markdown(var_card(
            "VaR Paramétrico · Ações",
            f"R$ {var_param:,.0f}",
            f"{var_param/valor_total*100:.2f}%",
            "#0ea5e9",
            "Distribuição normal · Rápido · Carteiras lineares"
        ), unsafe_allow_html=True)
    with v2:
        st.markdown(var_card(
            "VaR Histórico · Ações",
            f"R$ {var_hist:,.0f}",
            f"{var_hist/valor_total*100:.2f}%",
            "#10b981",
            "Distribuição empírica · Sem hipótese de normalidade"
        ), unsafe_allow_html=True)
    with v3:
        st.markdown(var_card(
            "VaR Full Valuation · Ações + Opções",
            f"R$ {var_full:,.0f}",
            f"{var_full/valor_total*100:.2f}%",
            "#f59e0b",
            "Reprecificação Black-Scholes · Captura não linearidade"
        ), unsafe_allow_html=True)

    # Tabela comparativa
    st.markdown(section_title("Tabela Comparativa"), unsafe_allow_html=True)

    df_comp = pd.DataFrame({
        "Método":        ["VaR Paramétrico — Ações", "VaR Histórico — Ações", "VaR Full Valuation — Ações + Opções"],
        "VaR (R$)":      [f"R$ {var_param:,.2f}", f"R$ {var_hist:,.2f}", f"R$ {var_full:,.2f}"],
        "% do Portfólio":[f"{var_param/valor_total*100:.3f}%", f"{var_hist/valor_total*100:.3f}%", f"{var_full/valor_total*100:.3f}%"],
        "Hipótese":      ["Normalidade", "Empírica", "Full repricing"],
        "Escopo":        ["Ações", "Ações", "Ações + Opções"],
    })
    st.dataframe(df_comp, use_container_width=True, hide_index=True)

    # Opção info
    st.markdown(section_title("Parâmetros da Opção — Black-Scholes"), unsafe_allow_html=True)
    oc1, oc2, oc3, oc4, oc5 = st.columns(5)
    for col, label, val in [
        (oc1, "Ativo Objeto", ativo_opcao),
        (oc2, "Tipo", tipo_opcao.upper()),
        (oc3, "Preço BS", f"R$ {preco_opcao_hoje:.4f}"),
        (oc4, "Vol. Impl.", f"{vol_anual*100:.2f}% a.a."),
        (oc5, "Valor Total", f"R$ {valor_opcoes:,.2f}"),
    ]:
        with col:
            st.markdown(kpi_card(label, val, "", "#a78bfa", ""), unsafe_allow_html=True)

# ──────────────────────────────────────────────
# TAB 2 — GRÁFICOS
# ──────────────────────────────────────────────
with tab2:

    CYAN   = "#0ea5e9"
    GREEN  = "#10b981"
    AMBER  = "#f59e0b"
    RED    = "#ef4444"
    PURPLE = "#a78bfa"
    DIM    = "#1e2d40"

    # ── Retornos históricos ──
    st.markdown(section_title("Distribuição dos Retornos · Carteira de Ações"), unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(12, 3.8))
    n, bins, patches = ax.hist(retorno_carteira, bins=60, color=CYAN, alpha=0.55, edgecolor="none")
    ax.hist(retorno_carteira[retorno_carteira <= np.percentile(retorno_carteira, percentil*100)],
            bins=60, color=RED, alpha=0.75, edgecolor="none", label="Zona de perda")
    vl = np.percentile(retorno_carteira, percentil*100)
    ax.axvline(vl, color=GREEN, linewidth=1.5, linestyle="--",
               label=f"VaR Histórico {nivel_confianca*100:.0f}% = {vl*100:.2f}%")
    ax.set_xlabel("Retorno Diário", labelpad=8)
    ax.set_ylabel("Frequência")
    ax.set_title("Distribuição Histórica de Retornos")
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # ── P&L Full Valuation ──
    st.markdown(section_title("P&L Distribution · Full Valuation (Ações + Opções)"), unsafe_allow_html=True)

    fig2, ax2 = plt.subplots(figsize=(12, 3.8))
    ax2.hist(cenarios_pnl, bins=60, color=AMBER, alpha=0.5, edgecolor="none")
    ax2.hist(cenarios_pnl[cenarios_pnl <= np.percentile(cenarios_pnl, percentil*100)],
             bins=60, color=RED, alpha=0.75, edgecolor="none", label="Cauda de perda")
    vl2 = np.percentile(cenarios_pnl, percentil*100)
    ax2.axvline(vl2, color=AMBER, linewidth=1.5, linestyle="--",
                label=f"VaR Full Val. = R$ {-vl2:,.0f}")
    ax2.set_xlabel("P&L (R$)", labelpad=8)
    ax2.set_ylabel("Frequência")
    ax2.set_title("Distribuição de P&L — Full Valuation com Opções")
    ax2.legend()
    fig2.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

    # ── Comparação VaR ──
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown(section_title("Comparação — Métodos de VaR"), unsafe_allow_html=True)
        fig3, ax3 = plt.subplots(figsize=(6, 3.5))
        metodos = ["Paramétrico\n(Ações)", "Histórico\n(Ações)", "Full Valuation\n(A+O)"]
        valores  = [var_param, var_hist, var_full]
        cores    = [CYAN, GREEN, AMBER]
        bars     = ax3.bar(metodos, valores, color=cores, width=0.5,
                           edgecolor="none", alpha=0.85)
        for bar, v in zip(bars, valores):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(valores)*0.01,
                     f"R$ {v:,.0f}", ha="center", va="bottom",
                     fontsize=7.5, color="#e2e8f0", fontweight="600")
        ax3.set_ylabel("VaR (R$)")
        ax3.set_title("Comparativo de VaR por Método")
        fig3.tight_layout()
        st.pyplot(fig3)
        plt.close(fig3)

    with col_b:
        st.markdown(section_title("Retorno Acumulado dos Ativos"), unsafe_allow_html=True)
        fig4, ax4 = plt.subplots(figsize=(6, 3.5))
        palette = [CYAN, GREEN, AMBER, PURPLE, RED]
        for i, t in enumerate(tickers):
            norm_prices = precos[t] / precos[t].iloc[0]
            ax4.plot(norm_prices.index, norm_prices.values,
                     color=palette[i % len(palette)], linewidth=1.4, label=t, alpha=0.9)
        ax4.axhline(1, color=DIM, linewidth=0.8, linestyle="--")
        ax4.set_ylabel("Retorno Acumulado (base 1)")
        ax4.set_title("Performance Relativa dos Ativos")
        ax4.legend(loc="upper left")
        fig4.tight_layout()
        st.pyplot(fig4)
        plt.close(fig4)

    # ── Sensibilidade da opção ──
    st.markdown(section_title(f"Sensibilidade · Preço da {tipo_opcao.upper()} vs. Preço do Ativo"), unsafe_allow_html=True)

    fig5, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    precos_sim = np.linspace(S0 * 0.65, S0 * 1.35, 300)
    op_sim     = [black_scholes(s, strike, vencimento_anos, taxa_livre_risco, vol_anual, tipo_opcao) for s in precos_sim]
    delt_sim   = [delta_bs(s, strike, vencimento_anos, taxa_livre_risco, vol_anual, tipo_opcao) for s in precos_sim]

    axes[0].plot(precos_sim, op_sim, color=CYAN, linewidth=2)
    axes[0].axvline(strike, color=RED,   linewidth=1, linestyle="--", alpha=0.7, label=f"Strike K={strike}")
    axes[0].axvline(S0,     color=AMBER, linewidth=1, linestyle="--", alpha=0.7, label=f"S₀={S0:.2f}")
    axes[0].fill_between(precos_sim, op_sim, alpha=0.1, color=CYAN)
    axes[0].set_xlabel("Preço do Ativo (R$)")
    axes[0].set_ylabel("Preço da Opção (R$)")
    axes[0].set_title(f"Preço da {tipo_opcao.upper()} — Black-Scholes")
    axes[0].legend()

    axes[1].plot(precos_sim, delt_sim, color=GREEN, linewidth=2)
    axes[1].axvline(strike, color=RED,   linewidth=1, linestyle="--", alpha=0.7, label=f"Strike K={strike}")
    axes[1].axvline(S0,     color=AMBER, linewidth=1, linestyle="--", alpha=0.7, label=f"S₀={S0:.2f}")
    axes[1].axhline(0.5, color=DIM, linewidth=0.8, linestyle=":")
    axes[1].fill_between(precos_sim, delt_sim, alpha=0.1, color=GREEN)
    axes[1].set_xlabel("Preço do Ativo (R$)")
    axes[1].set_ylabel("Delta")
    axes[1].set_title(f"Delta da {tipo_opcao.upper()}")
    axes[1].legend()

    fig5.tight_layout(pad=2)
    st.pyplot(fig5)
    plt.close(fig5)

# ──────────────────────────────────────────────
# TAB 3 — GREGAS
# ──────────────────────────────────────────────
with tab3:

    st.markdown(section_title(
        f"Gregas da Opção — {tipo_opcao.upper()} {ativo_opcao}",
        f"S₀={S0:.2f}  K={strike}  T={vencimento_anos}a  r={taxa_livre_risco*100:.1f}%  σ={vol_anual*100:.1f}%"
    ), unsafe_allow_html=True)

    g1, g2, g3 = st.columns(3)
    with g1:
        st.markdown(kpi_card("Delta (Δ)", f"{delta_op:.4f}",
                              "Exposição direcional ao ativo", "#0ea5e9", "Δ"), unsafe_allow_html=True)
    with g2:
        st.markdown(kpi_card("Gamma (Γ)", f"{gamma_op:.6f}",
                              "Convexidade — variação do delta", "#10b981", "Γ"), unsafe_allow_html=True)
    with g3:
        st.markdown(kpi_card("Vega (ν)", f"{vega_op:.4f}",
                              "Sensibilidade à volatilidade", "#f59e0b", "ν"), unsafe_allow_html=True)

    st.markdown("""
    <div style="
        background:#0d1420; border:1px solid #1e2d40; border-radius:8px;
        padding:1.2rem 1.5rem; margin:1rem 0; font-family:'IBM Plex Mono',monospace;
        font-size:0.72rem; color:#7e95b0; line-height:2;
    ">
        <span style="color:#0ea5e9;">Δ Delta</span> — variação de ~R$Δ no preço da opção para cada R$1 no ativo&nbsp;&nbsp;·&nbsp;&nbsp;
        <span style="color:#10b981;">Γ Gamma</span> — convexidade: quanto o Delta muda a cada R$1 no ativo&nbsp;&nbsp;·&nbsp;&nbsp;
        <span style="color:#f59e0b;">ν Vega</span> — impacto de +1% de volatilidade sobre o preço da opção
    </div>
    """, unsafe_allow_html=True)

    # Tabela de sensibilidade
    st.markdown(section_title("Tabela de Sensibilidade — Variação ±20% no Preço do Ativo"), unsafe_allow_html=True)

    precos_range = np.linspace(S0 * 0.80, S0 * 1.20, 11)
    tabela_grega = []
    for s in precos_range:
        d = delta_bs(s, strike, vencimento_anos, taxa_livre_risco, vol_anual, tipo_opcao)
        g = gamma_bs(s, strike, vencimento_anos, taxa_livre_risco, vol_anual)
        p = black_scholes(s, strike, vencimento_anos, taxa_livre_risco, vol_anual, tipo_opcao)
        var_s = pct = (s - S0) / S0 * 100
        tabela_grega.append({
            "Preço Ativo (R$)":  f"{s:.2f}",
            "Δ Preço (%)":       f"{var_s:+.1f}%",
            "Preço Opção (R$)":  f"{p:.4f}",
            "Delta (Δ)":         f"{d:.4f}",
            "Gamma (Γ)":         f"{g:.6f}",
            "Valor Posição (R$)": f"{quantidade_opcoes * p:,.2f}"
        })
    st.dataframe(pd.DataFrame(tabela_grega), use_container_width=True, hide_index=True)

# ──────────────────────────────────────────────
# TAB 4 — TEORIA
# ──────────────────────────────────────────────
with tab4:

    st.markdown(section_title("Fundamentos Teóricos"), unsafe_allow_html=True)

    teorias = [
        ("📐  VaR Paramétrico", "#0ea5e9", """
Assume que os retornos da carteira seguem **distribuição normal**.

**Fórmula:**
```
VaR = Z × σ × V × √h
```
onde `Z` = quantil normal, `σ` = volatilidade, `V` = valor, `h` = horizonte.

**Vantagens:** simples, rápido, fácil de comunicar.
**Limitações:** assume normalidade; não captura caudas gordas; inadequado para opções.
        """),
        ("📊  VaR Histórico", "#10b981", """
Usa diretamente os **retornos históricos observados** — sem hipótese distribucional.

**Passos:**
1. Calcular retorno histórico da carteira
2. Ordenar retornos do pior ao melhor
3. Selecionar o percentil correspondente ao nível de confiança

**Vantagens:** não exige normalidade; usa dados reais; intuitivo.
**Limitações:** depende da janela; assume que o passado representa o futuro.
        """),
        ("⚡  VaR Full Valuation", "#f59e0b", """
**Reprecifica toda a carteira** (incluindo opções via Black-Scholes) em cada cenário histórico.

Por que é necessário para opções?
- Call: `max(S − K, 0)` — payoff não linear
- Put: `max(K − S, 0)` — payoff não linear

O Full Valuation captura a convexidade que o VaR Paramétrico ignora.

**Vantagens:** capta não linearidade; mais preciso para derivativos.
**Limitações:** computacionalmente mais custoso; sensível à janela histórica.
        """),
        ("⚖️  Black-Scholes (1973)", "#a78bfa", """
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
        ("⚠️  Limitações do VaR", "#ef4444", """
1. **Não informa a magnitude da perda além do VaR** (cauda cega)
2. **Depende criticamente da janela histórica** — crises ausentes → VaR subestimado
3. **Caudas gordas** — mercados têm assimetria e excesso de curtose
4. **Não substitui stress test** — use junto com Expected Shortfall e análise de cenários
5. **Correlações instáveis** — em crises, correlações se aproximam de 1

Na prática, bancos usam VaR com Expected Shortfall, stress test e limites de perda.
        """),
    ]

    for titulo, cor, texto in teorias:
        with st.expander(titulo):
            st.markdown(f"""
            <div style="border-left:3px solid {cor}; padding-left:1rem; margin-bottom:0.5rem;">
            </div>
            """, unsafe_allow_html=True)
            st.markdown(texto)

# ============================================================
# RODAPÉ
# ============================================================

st.markdown("""
<div style="
    margin-top: 3rem;
    padding-top: 1.2rem;
    border-top: 1px solid #1e2d40;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 0.5rem;
">
    <div style="font-family:'IBM Plex Mono',monospace; font-size:0.62rem; color:#3d5470;">
        VaR Dashboard · Modelagem Aplicada ao Mercado Financeiro
    </div>
    <div style="font-family:'IBM Plex Mono',monospace; font-size:0.62rem; color:#3d5470;">
        Black-Scholes · Paramétrico · Histórico · Full Valuation
    </div>
</div>
""", unsafe_allow_html=True)
