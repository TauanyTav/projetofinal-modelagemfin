"""
Risk Lab — Calculadora de Value at Risk (VaR) v4.0
Modelagem Aplicada ao Mercado Financeiro
Métodos: Paramétrico, Histórico e Monte Carlo
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
from scipy.stats import norm, skew, kurtosis
from datetime import date, timedelta
import warnings

warnings.filterwarnings("ignore")

# ===================== PAGE CONFIG =====================
st.set_page_config(
    page_title="Risk Lab — VaR Calculator",
    page_icon="📉",
    layout="wide"
)

# ===================== CORES =====================
PRIMARY = "#22d3ee"
SUCCESS = "#34d399"
AMBER   = "#fbbf24"
DANGER  = "#f87171"
VIOLET  = "#a78bfa"
BG      = "#080d1a"
CARD    = "#0f172a"
BORDER  = "#1e293b"
TEXT    = "#f8fafc"
MUTED   = "#64748b"

# ===================== CSS =====================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

#MainMenu, footer, header, .stDeployButton,
div[data-testid="stToolbar"] {
    display: none !important;
}

html, body, .stApp {
    background-color: #080d1a !important;
    color: #f8fafc !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}

.main .block-container {
    padding: 1.5rem 2.5rem 3rem;
    max-width: 1600px;
}

section[data-testid="stSidebar"] {
    background: #0f172a !important;
    border-right: 1px solid #1e293b !important;
}

.stTextInput input,
.stNumberInput input,
.stDateInput input,
[data-baseweb="select"] > div {
    background: #080d1a !important;
    border: 1px solid #1e293b !important;
    color: #f8fafc !important;
    border-radius: 6px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.85rem !important;
}

.stButton > button {
    background: linear-gradient(135deg, #22d3ee 0%, #0ea5e9 100%) !important;
    color: #080d1a !important;
    border: none !important;
    border-radius: 6px !important;
    font-weight: 700 !important;
    padding: 0.6rem 1rem !important;
    width: 100% !important;
    font-size: 0.9rem !important;
    box-shadow: 0 4px 14px -4px rgba(34,211,238,0.4) !important;
    transition: all 0.2s ease !important;
}

.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px -4px rgba(34,211,238,0.5) !important;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    border-bottom: 1px solid #1e293b !important;
    background: transparent !important;
}

.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #64748b !important;
    border: none !important;
    padding: 0.5rem 1.2rem !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    border-radius: 4px 4px 0 0 !important;
}

.stTabs [aria-selected="true"] {
    background: #0f172a !important;
    color: #22d3ee !important;
}

.stSlider [data-baseweb="slider"] {
    padding: 0.5rem 0 !important;
}

.stSelectbox label, .stTextInput label, .stNumberInput label,
.stSlider label, .stDateInput label, .stRadio label {
    color: #94a3b8 !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em !important;
    text-transform: uppercase !important;
}

.stRadio [data-baseweb="radio"] {
    background: transparent !important;
}

hr {
    border-color: #1e293b !important;
    margin: 0.75rem 0 !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #080d1a; }
::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #334155; }
</style>
""", unsafe_allow_html=True)

# ===================== HELPERS HTML =====================
def kpi_card(label: str, value: str, color: str = TEXT, sublabel: str = ""):
    sub = f'<div style="color:{MUTED};font-size:0.7rem;margin-top:0.2rem;font-family:\'JetBrains Mono\',monospace">{sublabel}</div>' if sublabel else ""
    return f"""
    <div style="background:{CARD};border:1px solid {BORDER};border-radius:8px;padding:1.1rem 1.25rem;height:100%">
        <div style="color:{MUTED};font-size:0.65rem;text-transform:uppercase;font-weight:700;letter-spacing:0.08em">{label}</div>
        <div style="font-size:1.55rem;font-weight:700;font-family:'JetBrains Mono',monospace;color:{color};margin:0.25rem 0 0">{value}</div>
        {sub}
    </div>"""

def section_header(title: str, subtitle: str = ""):
    sub = f'<p style="color:{MUTED};font-size:0.82rem;margin:0.2rem 0 0">{subtitle}</p>' if subtitle else ""
    return f"""
    <div style="margin-bottom:1.25rem">
        <h3 style="color:{TEXT};font-size:1.1rem;font-weight:700;margin:0;letter-spacing:-0.01em">{title}</h3>
        {sub}
    </div>"""

def badge(text: str, color: str):
    return f'<span style="background:{color}18;color:{color};border:1px solid {color}30;border-radius:4px;padding:0.15rem 0.5rem;font-size:0.7rem;font-weight:700;font-family:\'JetBrains Mono\',monospace">{text}</span>'

# ===================== LÓGICA DE DADOS =====================
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_data(ticker: str, start: str, end: str) -> pd.Series:
    """Baixa dados do Yahoo Finance e retorna retornos diários."""
    try:
        raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        if raw.empty:
            return pd.Series(dtype=float)
        close = raw["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        close = close.dropna()
        returns = close.pct_change().dropna()
        return returns
    except Exception:
        return pd.Series(dtype=float)

def gerar_retornos_sinteticos(n: int = 756) -> pd.Series:
    """Retornos sintéticos baseados em S&P 500 para fallback."""
    np.random.seed(42)
    dates = pd.bdate_range(end=date.today(), periods=n)
    r = np.random.normal(0.0004, 0.012, n)
    # Adiciona fat-tails e clustering realistas
    r[np.random.choice(n, 15, replace=False)] *= np.random.uniform(-4, -2, 15)
    return pd.Series(r, index=dates, name="Sintético (SPX-like)")

# ===================== CÁLCULO DE VaR =====================
def var_parametrico(retornos: pd.Series, confianca: float, horizonte: int, capital: float):
    mu = retornos.mean()
    sigma = retornos.std()
    z = norm.ppf(1 - confianca)
    var_1d = -(mu + z * sigma)
    var_h  = var_1d * np.sqrt(horizonte)
    cvar   = -(mu - sigma * norm.pdf(norm.ppf(1 - confianca)) / (1 - confianca))
    return {
        "var_pct": var_1d,
        "var_h_pct": var_h,
        "var_brl": var_1d * capital,
        "var_h_brl": var_h * capital,
        "cvar_pct": cvar,
        "cvar_brl": cvar * capital,
        "mu": mu,
        "sigma": sigma,
    }

def var_historico(retornos: pd.Series, confianca: float, horizonte: int, capital: float):
    var_1d  = -np.percentile(retornos, (1 - confianca) * 100)
    var_h   = var_1d * np.sqrt(horizonte)
    losses  = retornos[retornos <= -var_1d]
    cvar    = -losses.mean() if len(losses) > 0 else var_1d
    return {
        "var_pct": var_1d,
        "var_h_pct": var_h,
        "var_brl": var_1d * capital,
        "var_h_brl": var_h * capital,
        "cvar_pct": cvar,
        "cvar_brl": cvar * capital,
    }

def var_montecarlo(retornos: pd.Series, confianca: float, horizonte: int, capital: float, n_sim: int = 10_000):
    mu    = retornos.mean()
    sigma = retornos.std()
    np.random.seed(0)
    sims  = np.random.normal(mu, sigma, (n_sim, horizonte))
    paths = np.cumprod(1 + sims, axis=1)
    pnl   = (paths[:, -1] - 1)
    var_h = -np.percentile(pnl, (1 - confianca) * 100)
    var_1d = var_h / np.sqrt(horizonte)
    tail   = pnl[pnl <= -var_h]
    cvar   = -tail.mean() if len(tail) > 0 else var_h
    return {
        "var_pct": var_1d,
        "var_h_pct": var_h,
        "var_brl": var_1d * capital,
        "var_h_brl": var_h * capital,
        "cvar_pct": cvar,
        "cvar_brl": cvar * capital,
        "pnl_sim": pnl,
    }

# ===================== GRÁFICOS =====================
MPL_STYLE = {
    "figure.facecolor":  BG,
    "axes.facecolor":    CARD,
    "axes.edgecolor":    BORDER,
    "axes.labelcolor":   MUTED,
    "xtick.color":       MUTED,
    "ytick.color":       MUTED,
    "grid.color":        BORDER,
    "grid.linestyle":    "--",
    "grid.alpha":        0.6,
    "text.color":        TEXT,
    "font.family":       "monospace",
    "font.size":         9,
}

def fig_distribuicao(retornos, var_par, var_hist, var_mc, confianca):
    with plt.rc_context(MPL_STYLE):
        fig, ax = plt.subplots(figsize=(10, 4.2))
        n, bins, patches = ax.hist(retornos * 100, bins=60, density=True,
                                   color=PRIMARY, alpha=0.25, edgecolor="none")
        # Colorir caudas
        threshold = -var_hist["var_pct"] * 100
        for patch, left in zip(patches, bins[:-1]):
            if left < threshold:
                patch.set_facecolor(DANGER)
                patch.set_alpha(0.55)

        # Curva normal sobreposta
        x = np.linspace(retornos.min() * 100, retornos.max() * 100, 300)
        ax.plot(x, norm.pdf(x, retornos.mean() * 100, retornos.std() * 100),
                color=PRIMARY, lw=1.8, alpha=0.9, label="Normal ajustada")

        # Linhas VaR
        ax.axvline(-var_par["var_pct"]  * 100, color=VIOLET, lw=1.5, ls="--", label=f"VaR Paramétrico")
        ax.axvline(-var_hist["var_pct"] * 100, color=AMBER,  lw=1.5, ls="--", label=f"VaR Histórico")
        ax.axvline(-var_mc["var_pct"]   * 100, color=SUCCESS, lw=1.5, ls="--", label=f"VaR Monte Carlo")

        ax.set_xlabel("Retorno Diário (%)")
        ax.set_ylabel("Densidade")
        ax.set_title(f"Distribuição de Retornos — Cauda de {(1-confianca)*100:.0f}%", pad=10)
        ax.legend(framealpha=0, labelcolor=TEXT, fontsize=8)
        ax.grid(True, axis="y")
        fig.tight_layout()
        return fig

def fig_retornos_historicos(retornos, var_hist):
    with plt.rc_context(MPL_STYLE):
        fig, ax = plt.subplots(figsize=(10, 3.8))
        colors = [DANGER if r < -var_hist["var_pct"] else PRIMARY for r in retornos]
        ax.bar(range(len(retornos)), retornos * 100, color=colors, alpha=0.7, width=1.0)
        ax.axhline(-var_hist["var_pct"] * 100, color=AMBER, lw=1.4, ls="--", label=f"VaR {(var_hist['var_pct']*100):.2f}%")
        ax.axhline(0, color=BORDER, lw=0.8)
        ax.set_xlabel("Dias")
        ax.set_ylabel("Retorno (%)")
        ax.set_title("Série Histórica de Retornos", pad=10)
        ax.legend(framealpha=0, labelcolor=TEXT, fontsize=8)

        # Anotar breaches
        n_breach = sum(1 for r in retornos if r < -var_hist["var_pct"])
        ax.text(0.01, 0.04, f"Breaches: {n_breach} ({n_breach/len(retornos)*100:.1f}%)",
                transform=ax.transAxes, color=DANGER, fontsize=8.5, fontweight="bold")
        fig.tight_layout()
        return fig

def fig_montecarlo(pnl_sim, var_mc, capital):
    with plt.rc_context(MPL_STYLE):
        fig, ax = plt.subplots(figsize=(10, 4.0))
        n, bins, patches = ax.hist(pnl_sim * 100, bins=80, density=True,
                                   color=VIOLET, alpha=0.3, edgecolor="none")
        threshold = -var_mc["var_pct"] * 100 * np.sqrt(1)  # 1d equiv
        for patch, left in zip(patches, bins[:-1]):
            if left < -var_mc["var_h_pct"] * 100:
                patch.set_facecolor(DANGER)
                patch.set_alpha(0.65)

        ax.axvline(-var_mc["var_h_pct"] * 100, color=DANGER, lw=1.8, ls="--",
                   label=f"VaR MC {(-var_mc['var_h_pct']*100):.2f}%")
        ax.axvline(-var_mc["cvar_pct"] * 100, color=AMBER, lw=1.4, ls=":",
                   label=f"CVaR {(-var_mc['cvar_pct']*100):.2f}%")

        ax.set_xlabel("P&L Simulado (%)")
        ax.set_ylabel("Densidade")
        ax.set_title("Distribuição Monte Carlo — 10.000 Simulações", pad=10)
        ax.legend(framealpha=0, labelcolor=TEXT, fontsize=8)
        fig.tight_layout()
        return fig

def fig_comparativo(results: dict, capital: float):
    with plt.rc_context(MPL_STYLE):
        metodos = ["Paramétrico", "Histórico", "Monte Carlo"]
        var_vals  = [results["par"]["var_h_pct"]*100, results["hist"]["var_h_pct"]*100, results["mc"]["var_h_pct"]*100]
        cvar_vals = [results["par"]["cvar_pct"]*100,  results["hist"]["cvar_pct"]*100,  results["mc"]["cvar_pct"]*100]

        x    = np.arange(len(metodos))
        w    = 0.35
        fig, ax = plt.subplots(figsize=(8, 3.8))

        b1 = ax.bar(x - w/2, var_vals,  w, label="VaR",  color=PRIMARY, alpha=0.8)
        b2 = ax.bar(x + w/2, cvar_vals, w, label="CVaR", color=DANGER,  alpha=0.8)

        for bar in list(b1) + list(b2):
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.05,
                    f"{h:.2f}%", ha="center", va="bottom", fontsize=8, color=TEXT)

        ax.set_xticks(x)
        ax.set_xticklabels(metodos, fontsize=9)
        ax.set_ylabel("Perda (%)")
        ax.set_title("Comparativo VaR vs CVaR por Método", pad=10)
        ax.legend(framealpha=0, labelcolor=TEXT, fontsize=8)
        ax.grid(True, axis="y")
        fig.tight_layout()
        return fig

# ===================== SIDEBAR =====================
with st.sidebar:
    st.markdown(f"""
    <div style="padding:0.5rem 0 1.5rem">
        <div style="font-size:1.3rem;font-weight:800;color:{PRIMARY};letter-spacing:-0.02em">
            📉 Risk Lab
        </div>
        <div style="color:{MUTED};font-size:0.72rem;margin-top:0.1rem">Value at Risk Calculator v4.0</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f'<div style="color:{MUTED};font-size:0.68rem;text-transform:uppercase;font-weight:700;letter-spacing:0.06em;margin-bottom:0.5rem">Ativo</div>', unsafe_allow_html=True)
    ticker = st.text_input("Ticker (Yahoo Finance)", value="PETR4.SA", label_visibility="collapsed")

    col1, col2 = st.columns(2)
    with col1:
        data_ini = st.date_input("De", value=date.today() - timedelta(days=3*365))
    with col2:
        data_fim = st.date_input("Até", value=date.today())

    st.markdown("---")

    st.markdown(f'<div style="color:{MUTED};font-size:0.68rem;text-transform:uppercase;font-weight:700;letter-spacing:0.06em;margin-bottom:0.5rem">Parâmetros</div>', unsafe_allow_html=True)

    capital = st.number_input("Capital (R$)", min_value=1_000.0, max_value=1e10,
                               value=1_000_000.0, step=10_000.0, format="%.0f")

    confianca = st.select_slider(
        "Nível de Confiança",
        options=[0.90, 0.95, 0.99, 0.999],
        value=0.95,
        format_func=lambda x: f"{x*100:.1f}%"
    )

    horizonte = st.slider("Horizonte (dias úteis)", 1, 252, 10)

    st.markdown("---")
    calcular = st.button("⚡  Calcular VaR", use_container_width=True)

    st.markdown(f"""
    <div style="margin-top:2rem;padding:0.75rem;background:{BG};border:1px solid {BORDER};border-radius:6px">
        <div style="color:{MUTED};font-size:0.67rem;line-height:1.6">
            <b style="color:{AMBER}">Métodos implementados:</b><br>
            • Paramétrico (Normal)<br>
            • Simulação Histórica<br>
            • Monte Carlo (10k sims)<br><br>
            <b style="color:{AMBER}">Fonte:</b> Yahoo Finance<br>
            <b style="color:{AMBER}">Fallback:</b> Dados sintéticos
        </div>
    </div>
    """, unsafe_allow_html=True)

# ===================== HEADER PRINCIPAL =====================
st.markdown(f"""
<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:1.5rem;padding-bottom:1rem;border-bottom:1px solid {BORDER}">
    <div>
        <h1 style="font-size:1.6rem;font-weight:800;margin:0;letter-spacing:-0.03em;color:{TEXT}">
            Value at Risk <span style="color:{PRIMARY}">Dashboard</span>
        </h1>
        <p style="color:{MUTED};font-size:0.82rem;margin:0.2rem 0 0">
            Análise quantitativa de risco · Três metodologias · Comparativo integrado
        </p>
    </div>
    <div style="text-align:right;color:{MUTED};font-size:0.72rem">
        Modelagem Aplicada ao<br>Mercado Financeiro
    </div>
</div>
""", unsafe_allow_html=True)

# ===================== ESTADO INICIAL =====================
if not calcular:
    st.markdown(f"""
    <div style="background:{CARD};border:1px solid {BORDER};border-radius:10px;padding:3rem 2rem;text-align:center;margin-top:1rem">
        <div style="font-size:3rem;margin-bottom:1rem">📉</div>
        <h3 style="color:{TEXT};font-size:1.2rem;font-weight:700;margin:0 0 0.5rem">Pronto para calcular</h3>
        <p style="color:{MUTED};font-size:0.85rem;max-width:420px;margin:0 auto 1.5rem">
            Configure o ativo, capital e parâmetros na barra lateral e clique em <b style="color:{PRIMARY}">Calcular VaR</b>.
        </p>
        <div style="display:flex;gap:1rem;justify-content:center;flex-wrap:wrap">
            {''.join([badge(t, PRIMARY) for t in ["PETR4.SA","VALE3.SA","ITUB4.SA","AAPL","SPY","BTC-USD"]])}
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ===================== PROCESSAMENTO =====================
with st.spinner(f"Carregando dados de {ticker.upper()}..."):
    retornos = fetch_data(ticker.upper(), str(data_ini), str(data_fim))

usando_fallback = False
if retornos is None or len(retornos) < 60:
    retornos = gerar_retornos_sinteticos()
    usando_fallback = True
    st.warning(f"⚠️  Não foi possível obter dados para **{ticker}** no período. Usando série sintética (SPX-like) para demonstração.")

# Calcular os três VaRs
var_par  = var_parametrico(retornos, confianca, horizonte, capital)
var_hist = var_historico(retornos, confianca, horizonte, capital)
var_mc   = var_montecarlo(retornos, confianca, horizonte, capital)

results  = {"par": var_par, "hist": var_hist, "mc": var_mc}

# ===================== KPIs TOPO =====================
ticker_label = f"{ticker.upper()}" + (" (Sintético)" if usando_fallback else "")

st.markdown(f"""
<div style="display:flex;align-items:center;gap:0.75rem;margin-bottom:1rem">
    <span style="font-size:1rem;font-weight:700;color:{TEXT}">{ticker_label}</span>
    {badge(f"IC {confianca*100:.0f}%", PRIMARY)}
    {badge(f"H = {horizonte}d", VIOLET)}
    {badge(f"{len(retornos)} obs.", AMBER)}
    {badge("Sintético" if usando_fallback else "Yahoo Finance", SUCCESS)}
</div>
""", unsafe_allow_html=True)

c1, c2, c3, c4, c5, c6 = st.columns(6)
with c1: st.markdown(kpi_card("VaR Param. 1d", f"{var_par['var_pct']*100:.2f}%",   VIOLET, f"R$ {var_par['var_brl']:,.0f}"), unsafe_allow_html=True)
with c2: st.markdown(kpi_card("VaR Hist. 1d",  f"{var_hist['var_pct']*100:.2f}%",  AMBER,  f"R$ {var_hist['var_brl']:,.0f}"), unsafe_allow_html=True)
with c3: st.markdown(kpi_card("VaR MC 1d",     f"{var_mc['var_pct']*100:.2f}%",    SUCCESS,f"R$ {var_mc['var_brl']:,.0f}"), unsafe_allow_html=True)
with c4: st.markdown(kpi_card(f"VaR Hist. {horizonte}d", f"{var_hist['var_h_pct']*100:.2f}%", DANGER, f"R$ {var_hist['var_h_brl']:,.0f}"), unsafe_allow_html=True)
with c5: st.markdown(kpi_card("CVaR (ES) Hist.", f"{var_hist['cvar_pct']*100:.2f}%", DANGER, f"R$ {var_hist['cvar_brl']:,.0f}"), unsafe_allow_html=True)
with c6:
    vol_anual = retornos.std() * np.sqrt(252) * 100
    st.markdown(kpi_card("Volatilidade Anual", f"{vol_anual:.1f}%", PRIMARY, f"σ diário {retornos.std()*100:.2f}%"), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ===================== ABAS =====================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊  Distribuição", "📈  Histórico", "🎲  Monte Carlo",
    "⚖️  Comparativo", "📋  Estatísticas"
])

with tab1:
    st.markdown(section_header("Distribuição de Retornos", "Histograma com curva normal ajustada e cortes de VaR"), unsafe_allow_html=True)
    fig = fig_distribuicao(retornos, var_par, var_hist, var_mc, confianca)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown(f"""
        <div style="background:{CARD};border:1px solid {BORDER};border-radius:8px;padding:1rem">
            <div style="color:{VIOLET};font-size:0.68rem;font-weight:700;text-transform:uppercase;letter-spacing:0.06em">Paramétrico</div>
            <div style="margin-top:0.5rem;font-family:'JetBrains Mono',monospace;font-size:0.82rem;line-height:1.8;color:{TEXT}">
                μ = {var_par['mu']*100:.4f}%<br>
                σ = {var_par['sigma']*100:.4f}%<br>
                VaR 1d = {var_par['var_pct']*100:.3f}%<br>
                VaR {horizonte}d = {var_par['var_h_pct']*100:.3f}%<br>
                CVaR = {var_par['cvar_pct']*100:.3f}%
            </div>
        </div>""", unsafe_allow_html=True)
    with col_b:
        st.markdown(f"""
        <div style="background:{CARD};border:1px solid {BORDER};border-radius:8px;padding:1rem">
            <div style="color:{AMBER};font-size:0.68rem;font-weight:700;text-transform:uppercase;letter-spacing:0.06em">Histórico</div>
            <div style="margin-top:0.5rem;font-family:'JetBrains Mono',monospace;font-size:0.82rem;line-height:1.8;color:{TEXT}">
                N = {len(retornos)} obs.<br>
                Mín = {retornos.min()*100:.3f}%<br>
                VaR 1d = {var_hist['var_pct']*100:.3f}%<br>
                VaR {horizonte}d = {var_hist['var_h_pct']*100:.3f}%<br>
                CVaR = {var_hist['cvar_pct']*100:.3f}%
            </div>
        </div>""", unsafe_allow_html=True)
    with col_c:
        st.markdown(f"""
        <div style="background:{CARD};border:1px solid {BORDER};border-radius:8px;padding:1rem">
            <div style="color:{SUCCESS};font-size:0.68rem;font-weight:700;text-transform:uppercase;letter-spacing:0.06em">Monte Carlo</div>
            <div style="margin-top:0.5rem;font-family:'JetBrains Mono',monospace;font-size:0.82rem;line-height:1.8;color:{TEXT}">
                N sims = 10.000<br>
                Horizonte = {horizonte}d<br>
                VaR 1d ≈ {var_mc['var_pct']*100:.3f}%<br>
                VaR {horizonte}d = {var_mc['var_h_pct']*100:.3f}%<br>
                CVaR = {var_mc['cvar_pct']*100:.3f}%
            </div>
        </div>""", unsafe_allow_html=True)

with tab2:
    st.markdown(section_header("Série Histórica de Retornos", "Retornos diários com marcação de breaches de VaR"), unsafe_allow_html=True)
    fig2 = fig_retornos_historicos(retornos, var_hist)
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)

    # Tabela de piores dias
    st.markdown(f'<div style="margin-top:1rem">{section_header("🔴 Piores 10 Dias", "Maiores perdas no período analisado")}</div>', unsafe_allow_html=True)
    piores = retornos.nsmallest(10).reset_index()
    piores.columns = ["Data", "Retorno"]
    piores["Retorno (%)"] = (piores["Retorno"] * 100).map(lambda x: f"{x:.3f}%")
    piores["P&L (R$)"]    = (piores["Retorno"] * capital).map(lambda x: f"R$ {x:,.0f}")
    piores["Breach VaR"]  = piores["Retorno"].apply(
        lambda r: "✅ Sim" if r < -var_hist["var_pct"] else "—"
    )
    piores = piores[["Data", "Retorno (%)", "P&L (R$)", "Breach VaR"]]
    st.dataframe(piores, use_container_width=True, hide_index=True)

with tab3:
    st.markdown(section_header("Monte Carlo — Distribuição de Cenários", "10.000 simulações de trajetórias de retorno"), unsafe_allow_html=True)
    fig3 = fig_montecarlo(var_mc["pnl_sim"], var_mc, capital)
    st.pyplot(fig3, use_container_width=True)
    plt.close(fig3)

    pnl = var_mc["pnl_sim"]
    col_x, col_y, col_z = st.columns(3)
    with col_x:
        st.markdown(kpi_card("Prob. Perda > VaR", f"{(pnl < -var_mc['var_h_pct']).mean()*100:.1f}%", DANGER), unsafe_allow_html=True)
    with col_y:
        st.markdown(kpi_card("Perda Máxima Simulada", f"{pnl.min()*100:.2f}%", DANGER, f"R$ {pnl.min()*capital:,.0f}"), unsafe_allow_html=True)
    with col_z:
        st.markdown(kpi_card("Ganho Máximo Simulado", f"{pnl.max()*100:.2f}%", SUCCESS, f"R$ {pnl.max()*capital:,.0f}"), unsafe_allow_html=True)

with tab4:
    st.markdown(section_header("Comparativo entre Metodologias", "VaR e CVaR para os três métodos no mesmo horizonte"), unsafe_allow_html=True)
    fig4 = fig_comparativo(results, capital)
    st.pyplot(fig4, use_container_width=True)
    plt.close(fig4)

    # Tabela comparativa
    dados_comp = {
        "Método":         ["Paramétrico", "Histórico", "Monte Carlo"],
        "VaR 1d (%)":     [f"{r['var_pct']*100:.3f}%" for r in [var_par, var_hist, var_mc]],
        f"VaR {horizonte}d (%)": [f"{r['var_h_pct']*100:.3f}%" for r in [var_par, var_hist, var_mc]],
        f"VaR {horizonte}d (R$)": [f"R$ {r['var_h_brl']:,.0f}" for r in [var_par, var_hist, var_mc]],
        "CVaR (%)":       [f"{r['cvar_pct']*100:.3f}%" for r in [var_par, var_hist, var_mc]],
        "CVaR (R$)":      [f"R$ {r['cvar_brl']:,.0f}" for r in [var_par, var_hist, var_mc]],
    }
    st.dataframe(pd.DataFrame(dados_comp), use_container_width=True, hide_index=True)

with tab5:
    st.markdown(section_header("Estatísticas Descritivas", "Caracterização completa da distribuição de retornos"), unsafe_allow_html=True)

    sk = skew(retornos)
    ku = kurtosis(retornos)

    stats = {
        "Observações":        f"{len(retornos):,}",
        "Retorno Médio (d)":  f"{retornos.mean()*100:.4f}%",
        "Retorno Médio (a)":  f"{retornos.mean()*252*100:.2f}%",
        "Vol. Diária":        f"{retornos.std()*100:.4f}%",
        "Vol. Anual":         f"{retornos.std()*np.sqrt(252)*100:.2f}%",
        "Mínimo":             f"{retornos.min()*100:.3f}%",
        "Máximo":             f"{retornos.max()*100:.3f}%",
        "Assimetria":         f"{sk:.4f}",
        "Curtose":            f"{ku:.4f}",
        "Sharpe Approx.":     f"{retornos.mean()/retornos.std()*np.sqrt(252):.3f}",
    }

    col_e, col_f = st.columns([1, 1])
    with col_e:
        for k, v in list(stats.items())[:5]:
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;padding:0.5rem 0;border-bottom:1px solid {BORDER}">
                <span style="color:{MUTED};font-size:0.82rem">{k}</span>
                <span style="font-family:'JetBrains Mono',monospace;font-size:0.82rem;color:{TEXT}">{v}</span>
            </div>""", unsafe_allow_html=True)
    with col_f:
        for k, v in list(stats.items())[5:]:
            color_val = DANGER if k == "Mínimo" else (SUCCESS if k == "Máximo" else TEXT)
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;padding:0.5rem 0;border-bottom:1px solid {BORDER}">
                <span style="color:{MUTED};font-size:0.82rem">{k}</span>
                <span style="font-family:'JetBrains Mono',monospace;font-size:0.82rem;color:{color_val}">{v}</span>
            </div>""", unsafe_allow_html=True)

    # Alertas interpretativos
    st.markdown("<br>", unsafe_allow_html=True)
    alertas = []
    if abs(sk) > 0.5:
        alertas.append((AMBER, f"Assimetria significativa ({sk:.2f}): distribuição {'negativa' if sk < 0 else 'positiva'}. VaR paramétrico pode subestimar risco."))
    if ku > 1:
        alertas.append((DANGER, f"Curtose elevada ({ku:.2f}): caudas pesadas (fat tails). Eventos extremos mais frequentes que o normal."))
    if ku <= 1 and abs(sk) <= 0.5:
        alertas.append((SUCCESS, "Distribuição próxima à normal. VaR paramétrico é uma boa aproximação."))

    for cor, msg in alertas:
        st.markdown(f"""
        <div style="background:{cor}12;border-left:3px solid {cor};border-radius:0 6px 6px 0;padding:0.75rem 1rem;margin-bottom:0.5rem;font-size:0.83rem;color:{TEXT}">
            {msg}
        </div>""", unsafe_allow_html=True)

# ===================== FOOTER =====================
st.markdown(f"""
<div style="margin-top:3rem;padding-top:1rem;border-top:1px solid {BORDER};display:flex;justify-content:space-between;align-items:center">
    <span style="color:{MUTED};font-size:0.72rem">Risk Lab v4.0 · Trabalho Final · Modelagem Aplicada ao Mercado Financeiro</span>
    <span style="color:{MUTED};font-size:0.72rem">Dados: Yahoo Finance · Fallback: Série Sintética</span>
</div>
""", unsafe_allow_html=True)
