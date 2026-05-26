"""
Calculadora de Value at Risk (VaR) — v3.1 (Professional Edition)
Trabalho Final | Modelagem Aplicada ao Mercado Financeiro
Melhorias: Persistência de estado via session_state, UI Customizada Avançada.
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
st.set_page_config(page_title="Risk Lab — Premium VaR", page_icon="📉", layout="wide")

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

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

#MainMenu, footer, header, .stDeployButton, div[data-testid="stToolbar"] {{ display: none !important; visibility: hidden !important; }}

html, body, .stApp {{
    background-color: {BG} !important;
    color: {TEXT} !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}}
.main .block-container {{ padding: 2rem 3rem 4rem; max-width: 1600px; }}
section[data-testid="stSidebar"] {{ background: {CARD} !important; border-right: 1px solid {BORDER} !important; }}

/* Inputs e Sidebar */
.stTextInput input, .stNumberInput input, .stDateInput input, [data-baseweb="select"] > div {{
    background: {BG} !important; border: 1px solid {BORDER} !important;
    color: {TEXT} !important; border-radius: 8px !important;
    font-family: 'JetBrains Mono', monospace !important; font-size: 0.85rem !important;
}}
.stTextInput label, .stNumberInput label, .stDateInput label, .stSelectbox label {{
    color: {MUTED} !important; font-size: 0.75rem !important; font-weight: 600 !important; letter-spacing: 0.05em !important;
}}

/* Botão Principal Estilizado */
.stButton > button {{
    background: linear-gradient(135deg, {PRIMARY} 0%, #0ea5e9 100%) !important;
    color: #080d1a !important; border: none !important; border-radius: 8px !important;
    font-weight: 700 !important; padding: 0.6rem 1.5rem !important; width: 100% !important;
    box-shadow: 0 4px 20px -4px rgba(34,211,238,0.4) !important;
    transition: all 0.2s ease !important;
}}
.stButton > button:hover {{ transform: translateY(-1px); box-shadow: 0 6px 24px -2px rgba(34,211,238,0.6) !important; }}

/* Custom Tab Styling para tirar a cara de Streamlit padrão */
.stTabs [data-baseweb="tab-list"] {{ gap: 8px; border-bottom: 1px solid {BORDER} !important; padding-bottom: 4px; }}
.stTabs [data-baseweb="tab"] {{
    background: transparent !important; color: {MUTED} !important;
    border: none !important; padding: 0.6rem 1.2rem !important;
    font-weight: 600 !important; font-size: 0.9rem !important; border-radius: 6px !important;
}}
.stTabs [aria-selected="true"] {{ background: {CARD} !important; color: {PRIMARY} !important; }}

/* Componentes de Cards Profissionais */
.kpi-card {{
    background: {CARD}; border: 1px solid {BORDER}; border-radius: 10px; padding: 1.5rem; height: 100%;
}}
.kpi-label {{ color: {MUTED}; font-size: 0.75rem; text-transform: uppercase; font-weight: 600; letter-spacing: 0.05em; }}
.kpi-value {{ color: {TEXT}; font-size: 1.75rem; font-weight: 700; font-family: 'JetBrains Mono', monospace; margin: 0.25rem 0; }}
.var-card {{
    background: {CARD}; border: 1px solid {BORDER}; border-left: 4px solid var(--accent); border-radius: 10px; padding: 1.5rem;
}}
.section-title {{
    color: {TEXT}; font-size: 1.25rem; font-weight: 700; margin: 2rem 0 1rem; padding-bottom: 0.5rem; border-bottom: 1px solid {BORDER};
}}
</style>
""", unsafe_allow_html=True)

def kpi(label, value, sub="", color=PRIMARY):
    return f'<div class="kpi-card"><div class="kpi-label">{label}</div><div class="kpi-value" style="color:{color}">{value}</div><div style="color:{MUTED}; font-size:0.8rem">{sub}</div></div>'

def var_card(label, value, pct, color, desc):
    return f'<div class="var-card" style="--accent:{color}"><div class="kpi-label">{label}</div><div style="font-family:\'JetBrains Mono\'; font-size:1.8rem; font-weight:700; color:{color}; margin: 0.25rem 0;">{value}</div><div style="color:{MUTED}; font-size:0.8rem; font-weight:500">{pct} do portfólio</div><p style="color:{MUTED}; font-size:0.8rem; margin-top:0.75rem; padding-top:0.75rem; border-top:1px solid {BORDER}; line-height:1.4">{desc}</p></div>'

def section(title, sub=""):
    return f'<div class="section-title">{title}<span style="color:{MUTED}; font-size:0.85rem; font-weight:400; margin-left:0.75rem">{sub}</span></div>'

# ===================== FINANÇAS =====================
def bs(S, K, T, r, sigma, tipo="call"):
    if T <= 0: return max(S - K, 0) if tipo == "call" else max(K - S, 0)
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return (S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)) if tipo == "call" else (K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1))

def greeks(S, K, T, r, sigma, tipo="call"):
    if T <= 0 or sigma <= 0: return 0.0, 0.0, 0.0
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    delta = norm.cdf(d1) if tipo == "call" else norm.cdf(d1) - 1
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T)
    return delta, gamma, vega

@st.cache_data(ttl=600, show_spinner=False)
def baixar_dados(tickers_str, inicio):
    tickers = [t.strip() for t in tickers_str.split(",") if t.strip()]
    try:
        df = yf.download(tickers, start=inicio, auto_adjust=True, progress=False)
        if df.empty: return None
        return df["Close"] if "Close" in df.columns else df
    except:
        return None

def chart_style():
    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": CARD,
        "axes.edgecolor": BORDER, "axes.labelcolor": MUTED,
        "axes.titlecolor": TEXT, "text.color": TEXT,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "grid.color": BORDER, "grid.alpha": 0.4,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.facecolor": CARD, "legend.edgecolor": BORDER, "legend.labelcolor": TEXT,
        "font.family": "sans-serif", "figure.dpi": 120,
    })

# ===================== SIDEBAR =====================
st.sidebar.markdown(f'<h3 style="color:{TEXT}; font-weight:700; margin-bottom: 1rem">⚡ RISK LAB</h3>', unsafe_allow_html=True)

tickers_input = st.sidebar.text_input("Ativos da Carteira", "PETR4.SA, VALE3.SA, ITUB4.SA, ^BVSP")
qty_input = st.sidebar.text_input("Quantidades nominais", "1000, 800, 1200, 10")
data_ini = st.sidebar.date_input("Início do histórico VaR", pd.to_datetime("2022-01-01"))
confianca = st.sidebar.selectbox("Nível de Confiança", [0.90, 0.95, 0.99], 1)
horizonte = st.sidebar.number_input("Horizonte temporal (dias)", 1, 30, 1)

st.sidebar.markdown(f'<hr style="border-color:{BORDER}">', unsafe_allow_html=True)
opt_ativo = st.sidebar.selectbox("Ativo Objeto (Opção)", tickers_input.split(","))
opt_tipo = st.sidebar.selectbox("Direito", ["call", "put"])
strike = st.sidebar.number_input("Preço de Exercício (K)", 1.0, 500.0, 30.0)
calcular = st.sidebar.button("EXECUTAR ANÁLISE")

# ===================== ENGINE (SESSION STATE) =====================
if calcular:
    with st.spinner("Buscando dados de mercado via Yahoo Finance..."):
        precos_full = baixar_dados(tickers_input, "2007-01-01")
        
        if precos_full is None:
            st.error("Erro na requisição dos dados. Certifique-se de que as tags dos ativos estão corretas.")
            st.stop()
            
            convert_tickers = precos_full.columns.tolist()
        try:
            qtds = [float(q.strip()) for q in qty_input.split(",")]
            if len(qtds) < len(precos_full.columns): 
                qtds += [0] * (len(precos_full.columns) - len(qtds))
        except:
            qtds = [1000] * len(precos_full.columns)
        
        quantidades = dict(zip(precos_full.columns, qtds))
        precos_var = precos_full.loc[pd.to_datetime(data_ini):]
        retornos = precos_var.pct_change().dropna()
        ultimos = precos_var.iloc[-1]
        
        v_acoes = sum(quantidades[t] * ultimos[t] for t in precos_full.columns)
        pesos = np.array([(quantidades[t] * ultimos[t]) / v_acoes for t in precos_full.columns])
        ret_port = retornos.dot(pesos)
        
        mu, sig = ret_port.mean(), ret_port.std()
        z = norm.ppf(confianca)
        var_param = (z * sig * np.sqrt(horizonte) - mu * horizon-te) * v_acoes
        var_hist = -np.percentile(ret_port, (1-confianca)*100) * v_acoes
        es_hist = -ret_port[ret_port <= np.percentile(ret_port, (1-confianca)*100)].mean() * v_acoes

        S0 = ultimos[opt_ativo.strip()]
        sig_an = retornos[opt_ativo.strip()].std() * np.sqrt(252) if opt_ativo.strip() in retornos.columns else 0.25
        delta, gamma, vega = greeks(S0, strike, 0.25, 0.11, sig_an, opt_tipo)

        # Salvando os cálculos no estado da aplicação para evitar o reset ao trocar de abas
        st.session_state["analise_pronta"] = True
        st.session_state["v_acoes"] = v_acoes
        st.session_state["var_hist"] = var_hist
        st.session_state["es_hist"] = es_hist
        st.session_state["var_param"] = var_param
        st.session_state["ret_port"] = ret_port
        st.session_state["confianca"] = confianca
        st.session_state["horizonte"] = horizonte
        st.session_state["precos_full"] = precos_full
        st.session_state["tickers"] = precos_full.columns.tolist()
        st.session_state["pesos"] = pesos
        st.session_state["opt_ativo"] = opt_ativo
        st.session_state["delta"] = delta
        st.session_state["gamma"] = gamma
        st.session_state["vega"] = vega

# RENDERIZAÇÃO DA INTERFACE PRINCIPAL
if st.session_state.get("analise_pronta", False):
    chart_style()
    tabs = st.tabs(["📊 Sumário de Risco", "📈 Distribuições", "⚠️ Teste de Stress", "⚙️ Derivativos & Gregas", "📋 Histórico"])

    with tabs[0]:
        st.markdown(section("Valoração Corrente"), unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.markdown(kpi("Exposure MTM", f"R$ {st.session_state['v_acoes']:,.2f}", "Capital total alocado", PRIMARY), unsafe_allow_html=True)
        c2.markdown(kpi(f"Value at Risk ({st.session_state['confianca']*100:.1f}%)", f"R$ {st.session_state['var_hist']:,.2f}", f"Projeção para {st.session_state['horizonte']} dia(s)", SUCCESS), unsafe_allow_html=True)
        c3.markdown(kpi("Expected Shortfall", f"R$ {st.session_state['es_hist']:,.2f}", "Média das perdas severas", DANGER), unsafe_allow_html=True)
        
        st.markdown(section("Métricas Comparativas de Risco"), unsafe_allow_html=True)
        v1, v2 = st.columns(2)
        v1.markdown(var_card("Método Paramétrico (Variança-Covariança)", f"R$ {st.session_state['var_param']:,.2f}", f"{st.session_state['var_param']/st.session_state['v_acoes']*100:.2f}%", PRIMARY, "Calculado sob premissa estatística de normalidade multivariada."), unsafe_allow_html=True)
        v2.markdown(var_card("Método de Simulação Histórica", f"R$ {st.session_state['var_hist']:,.2f}", f"{st.session_state['var_hist']/st.session_state['v_acoes']*100:.2f}%", SUCCESS, "Apuração empírica sem premissas distribucionais sobre os retornos."), unsafe_allow_html=True)

    with tabs[1]:
        st.markdown(section("Análise de Cauda e Frequências"), unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(11, 3.5))
        ax.hist(st.session_state['ret_port'], bins=60, color="#1e1b4b", edgecolor=BORDER, alpha=0.7, label="Retornos do Portfólio")
        limite_var = np.percentile(st.session_state['ret_port'], (1-st.session_state['confianca'])*100)
        ax.hist(st.session_state['ret_port'][st.session_state['ret_port'] <= limite_var], bins=15, color=DANGER, alpha=0.6, label="Região de Cauda (Perda)")
        ax.axvline(limite_var, color=PRIMARY, linestyle='-', linewidth=1.5, label=f"VaR Limiar")
        ax.set_ylabel("Frequência", fontsize=8)
        ax.set_xlabel("Variação Percentual Diária", fontsize=8)
        ax.legend(frameon=True, fontsize=8)
        st.pyplot(fig)

    with tabs[2]:
        st.markdown(section("Cenários de Stress Históricos", "Simulação de cauda baseada em crises macroeconômicas"), unsafe_allow_html=True)
        
        cenarios = {
            "Crise do Subprime (2008)": ("2008-08-01", "2008-10-30"),
            "Joesley Day (2017)": ("2017-05-17", "2017-05-31"),
            "Greve dos Caminhoneiros (2018)": ("2018-05-21", "2018-06-05"),
            "Crash COVID-19 (Março 2020)": ("2020-02-20", "2020-04-01"),
            "Crise Energética/Inflação (2021)": ("2021-08-01", "2021-10-30"),
            "Início da Guerra na Ucrânia (2022)": ("2022-02-24", "2022-03-15"),
        }
        
        stress_results = []
        for nome, (ini, fim) in cenarios.items():
            try:
                dados_periodo = st.session_state['precos_full'].loc[ini:fim]
                if len(dados_periodo) > 1:
                    ret_periodo = (dados_periodo.iloc[-1] / dados_periodo.iloc[0]) - 1
                    perda_pct = sum(ret_periodo[t] * st.session_state['pesos'][i] for i, t in enumerate(st.session_state['tickers']))
                    perda_financeira = perda_pct * st.session_state['v_acoes']
                    stress_results.append({
                        "Evento Histórico": nome,
                        "Retorno no Período": f"{perda_pct*100:.2f}%",
                        "Impacto MTM Estimado": f"R$ {perda_financeira:,.2f}"
                    })
            except:
                continue
        
        if stress_results:
            st.dataframe(pd.DataFrame(stress_results), use_container_width=True, hide_index=True)
        else:
            st.info("Sua carteira contém ativos sem histórico longo o suficiente para os períodos de crise testados.")

    with tabs[3]:
        st.markdown(section(f"Sensibilidades de Primeira e Segunda Ordem — Ativo: {st.session_state['opt_ativo']}"), unsafe_allow_html=True)
        g1, g2, g3 = st.columns(3)
        g1.markdown(kpi("Delta (Δ)", f"{st.session_state['delta']:.4f}", "Exposição direcional primária ao ativo objeto", PRIMARY), unsafe_allow_html=True)
        g2.markdown(kpi("Gamma (Γ)", f"{st.session_state['gamma']:.5f}", "Risco de curvatura (aceleração do Delta)", SUCCESS), unsafe_allow_html=True)
        g3.markdown(kpi("Vega (ν)", f"{st.session_state['vega']:.4f}", "Sensibilidade financeira a variações na volatilidade", AMBER), unsafe_allow_html=True)

    with tabs[4]:
        st.markdown(section("Changelog e Controle de Versão"), unsafe_allow_html=True)
        versões = [
            {"Versão": "v3.1", "Data": "Maio 2026", "Melhoria": "Correção de quebra de escopo via st.session_state e redesenho de UI Dark integrada."},
            {"Versão": "v3.0", "Data": "Maio 2026", "Melhoria": "Implementação do core de testes de estresse macroeconômico retroativo."},
            {"Versão": "v2.0", "Data": "Maio 2026", "Melhoria": "Estruturação matemática do Expected Shortfall e precificação Black-Scholes."},
        ]
        st.dataframe(pd.DataFrame(versões), use_container_width=True, hide_index=True)

else:
    st.markdown(f"""
    <div style="text-align:center; padding: 8rem 2rem 5rem 2rem;">
        <span style="font-size:3rem; padding:1rem; background:{CARD}; border: 1px solid {BORDER}; border-radius:50%">📉</span>
        <h2 style="margin-top:1.5rem; font-weight:700;">Risk Lab — Gestão de Risco Quantitativo</h2>
        <p style="color:{MUTED}; font-size:0.95rem; max-width:500px; margin: 0.5rem auto 1.5rem;">Defina a composição do portfólio e horizontes de probabilidade na barra lateral para inicializar os motores de cálculo.</p>
    </div>
    """, unsafe_allow_html=True)
