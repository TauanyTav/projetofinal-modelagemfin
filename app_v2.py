"""
Calculadora de Value at Risk (VaR) — v3
Trabalho Final | Modelagem Aplicada ao Mercado Financeiro
Novas implementações: Stress Test Histórico e Log de Versões.
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
st.set_page_config(page_title="Risk Lab — VaR & Stress", page_icon="📉", layout="wide")

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
html, body, .stApp {{ background: {BG} !important; color: {TEXT} !important; font-family: 'Inter', sans-serif !important; }}
.stApp {{ background-image: radial-gradient(ellipse 80% 50% at 0% 0%, rgba(34,211,238,0.05), transparent 60%), radial-gradient(ellipse 60% 40% at 100% 100%, rgba(167,139,250,0.05), transparent 60%) !important; }}
.main .block-container {{ padding: 1.5rem 2rem 4rem; max-width: 1600px; }}
section[data-testid="stSidebar"] {{ background: {CARD} !important; border-right: 1px solid {BORDER} !important; }}
.kpi-card {{ background: linear-gradient(180deg, {CARD}, #0d1626); border: 1px solid {BORDER}; border-radius: 12px; padding: 1.25rem; height: 100%; }}
.kpi-label {{ color: {MUTED}; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.1em; font-weight: 600; }}
.kpi-value {{ color: {TEXT}; font-size: 1.8rem; font-weight: 700; font-family: 'JetBrains Mono', monospace; margin: 0.4rem 0 0.2rem; }}
.var-card {{ background: linear-gradient(180deg, {CARD}, #0d1626); border: 1px solid {BORDER}; border-top: 3px solid var(--accent); border-radius: 12px; padding: 1.5rem; }}
.section-title {{ color: {TEXT}; font-size: 1.1rem; font-weight: 700; margin: 2rem 0 0.8rem; padding-bottom: 0.6rem; border-bottom: 1px solid {BORDER}; }}
</style>
""", unsafe_allow_html=True)

def kpi(label, value, sub="", color=PRIMARY):
    return f'<div class="kpi-card"><div class="kpi-label">{label}</div><div class="kpi-value" style="color:{color}">{value}</div><div style="color:{MUTED}; font-size:0.75rem">{sub}</div></div>'

def var_card(label, value, pct, color, desc):
    return f'<div class="var-card" style="--accent:{color}"><div class="kpi-label">{label}</div><div style="font-family:\'JetBrains Mono\'; font-size:2rem; font-weight:700; color:{color}">{value}</div><div style="color:{MUTED}; font-size:0.75rem">{pct} do portfólio</div><p style="color:{MUTED}; font-size:0.78rem; margin-top:0.8rem; padding-top:0.8rem; border-top:1px solid {BORDER}">{desc}</p></div>'

def section(title, sub=""):
    return f'<div class="section-title">{title}<span style="color:{MUTED}; font-size:0.85rem; font-weight:400; margin-left:0.5rem">{sub}</span></div>'

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

@st.cache_data(ttl=600)
def baixar_dados(tickers_str, inicio):
    tickers = [t.strip() for t in tickers_str.split(",") if t.strip()]
    df = yf.download(tickers, start=inicio, auto_adjust=True, progress=False)
    if df.empty: return None
    return df["Close"] if "Close" in df.columns else df

# ===================== SIDEBAR =====================
st.sidebar.markdown(f'<h2 style="color:{PRIMARY}">⚡ Risk Lab v3</h2><hr style="border-color:{BORDER}">', unsafe_allow_html=True)
tickers_input = st.sidebar.text_input("Tickers (separados por vírgula)", "PETR4.SA, VALE3.SA, ITUB4.SA, ^BVSP")
qty_input = st.sidebar.text_input("Quantidades", "1000, 800, 1200, 10")
data_ini = st.sidebar.date_input("Data de início (VaR)", pd.to_datetime("2022-01-01"))
confianca = st.sidebar.selectbox("Nível de Confiança", [0.90, 0.95, 0.99], 1)
horizonte = st.sidebar.number_input("Horizonte (Dias)", 1, 30, 1)

st.sidebar.markdown("---")
opt_ativo = st.sidebar.selectbox("Ativo para Opção (Gregas)", tickers_input.split(","))
opt_tipo = st.sidebar.selectbox("Tipo de Opção", ["call", "put"])
strike = st.sidebar.number_input("Strike (K)", 1.0, 500.0, 30.0)
calcular = st.sidebar.button("▶ CALCULAR TUDO")

# ===================== LOGIC =====================
if calcular:
    with st.spinner("Processando dados históricos..."):
        # Para o Stress Test, baixamos desde 2007 para garantir os marcos
        precos_full = baixar_dados(tickers_input, "2007-01-01")
        
        if precos_full is None:
            st.error("Erro ao baixar dados. Verifique os tickers.")
            st.stop()
            
        tickers = precos_full.columns.tolist()
        try:
            qtds = [float(q.strip()) for q in qty_input.split(",")]
            if len(qtds) < len(tickers): qtds += [0] * (len(tickers) - len(qtds))
        except:
            qtds = [1000] * len(tickers)
        
        quantidades = dict(zip(tickers, qtds))
        
        # Filtro para VaR
        precos_var = precos_full.loc[pd.to_datetime(data_ini):]
        retornos = precos_var.pct_change().dropna()
        ultimos = precos_var.iloc[-1]
        
        v_acoes = sum(quantidades[t] * ultimos[t] for t in tickers)
        pesos = np.array([ (quantidades[t] * ultimos[t]) / v_acoes for t in tickers])
        ret_port = retornos.dot(pesos)
        
        # Estatísticas
        mu, sig = ret_port.mean(), ret_port.std()
        z = norm.ppf(confianca)
        var_param = (z * sig * np.sqrt(horizonte) - mu * horizonte) * v_acoes
        var_hist = -np.percentile(ret_port, (1-confianca)*100) * v_acoes
        es_hist = -ret_port[ret_port <= np.percentile(ret_port, (1-confianca)*100)].mean() * v_acoes

        # Gregas
        S0 = ultimos[opt_ativo.strip()]
        sig_an = retornos[opt_ativo.strip()].std() * np.sqrt(252)
        delta, gamma, vega = greeks(S0, strike, 0.25, 0.11, sig_an, opt_tipo)

    # ===================== TABS =====================
    tabs = st.tabs(["📊 Resumo", "📈 Gráficos", "⚠️ Stress Test", "📋 Histórico de Versões", "⚙️ Gregas"])

    with tabs[0]:
        st.markdown(section("Resumo da Carteira"), unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.markdown(kpi("Valor Total", f"R$ {v_acoes:,.2f}", "Ações e ETFs", PRIMARY), unsafe_allow_html=True)
        c2.markdown(kpi(f"VaR Histórico ({confianca*100}%)", f"R$ {var_hist:,.2f}", f"Horizonte: {horizonte} dia(s)", SUCCESS), unsafe_allow_html=True)
        c3.markdown(kpi("Expected Shortfall", f"R$ {es_hist:,.2f}", "Perda média na cauda", DANGER), unsafe_allow_html=True)
        
        v1, v2 = st.columns(2)
        v1.markdown(var_card("VaR Paramétrico", f"R$ {var_param:,.2f}", f"{var_param/v_acoes*100:.2f}%", PRIMARY, "Baseado em distribuição Normal"), unsafe_allow_html=True)
        v2.markdown(var_card("VaR Histórico", f"R$ {var_hist:,.2f}", f"{var_hist/v_acoes*100:.2f}%", SUCCESS, "Baseado em dados reais do período"), unsafe_allow_html=True)

    with tabs[2]:
        st.markdown(section("Stress Test - Marcos Econômicos", "Impacto histórico real na carteira atual"), unsafe_allow_html=True)
        
        # Definição dos Cenários
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
                # Calculando o retorno acumulado no período
                dados_periodo = precos_full.loc[ini:fim]
                if len(dados_periodo) > 1:
                    ret_periodo = (dados_periodo.iloc[-1] / dados_periodo.iloc[0]) - 1
                    perda_pct = sum(ret_periodo[t] * pesos[i] for i, t in enumerate(tickers))
                    perda_financeira = perda_pct * v_acoes
                    stress_results.append({
                        "Evento Histórico": nome,
                        "Queda no Período": f"{perda_pct*100:.2f}%",
                        "Perda Estimada (R$)": f"R$ {perda_financeira:,.2f}",
                        "Status": "⚠️ Crítico" if perda_pct < -0.10 else "ℹ️ Moderado"
                    })
            except:
                continue
        
        if stress_results:
            st.table(pd.DataFrame(stress_results))
            
            # Gráfico de Stress
            df_st = pd.DataFrame(stress_results)
            df_st["Valor"] = df_st["Perda Estimada (R$)"].str.replace("R$ ","").str.replace(",","").astype(float)
            fig_st, ax_st = plt.subplots(figsize=(10, 4))
            ax_st.barh(df_st["Evento Histórico"], df_st["Valor"], color=DANGER)
            ax_st.set_title("Perda Financeira por Evento (R$)", color=TEXT)
            plt.xticks(color=MUTED); plt.yticks(color=MUTED)
            ax_st.set_facecolor(CARD); fig_st.patch.set_facecolor(BG)
            st.pyplot(fig_st)
        else:
            st.info("Dados insuficientes para os marcos históricos. Verifique o tempo de vida dos ativos.")

    with tabs[3]:
        st.markdown(section("Histórico de Versões"), unsafe_allow_html=True)
        versões = [
            {"Versão": "v3.0", "Data": "26/05/2024", "Alterações": "Adição de Stress Test Histórico, Aba de Versões e melhoria na performance de download."},
            {"Versão": "v2.1", "Data": "15/05/2024", "Alterações": "Correção analítica do ES Paramétrico e ajuste de identação no wrapper YFinance."},
            {"Versão": "v2.0", "Data": "10/05/2024", "Alterações": "Nova UI em Dark Mode, cálculos de Expected Shortfall e Gregas Black-Scholes."},
            {"Versão": "v1.0", "Data": "01/04/2024", "Alterações": "Lançamento inicial: Cálculo de VaR Paramétrico e Histórico simples."},
        ]
        st.table(versões)

    with tabs[1]:
        st.markdown(section("Distribuição de Retornos e P&L"), unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.hist(ret_port, bins=50, color=PRIMARY, alpha=0.6, label="Retornos Reais")
        ax.axvline(np.percentile(ret_port, (1-confianca)*100), color=DANGER, linestyle='--', label=f"VaR {confianca*100}%")
        ax.set_facecolor(CARD); fig.patch.set_facecolor(BG)
        plt.legend()
        st.pyplot(fig)

    with tabs[4]:
        st.markdown(section(f"Sensibilidade (Gregas) - {opt_ativo}"), unsafe_allow_html=True)
        g1, g2, g3 = st.columns(3)
        g1.markdown(kpi("Delta (Δ)", f"{delta:.4f}", "Sensibilidade ao preço", PRIMARY), unsafe_allow_html=True)
        g2.markdown(kpi("Gamma (Γ)", f"{gamma:.4f}", "Aceleração do Delta", SUCCESS), unsafe_allow_html=True)
        g3.markdown(kpi("Vega (ν)", f"{vega:.4f}", "Sensibilidade à Volatilidade", AMBER), unsafe_allow_html=True)

else:
    st.markdown(f"""
    <div style="text-align:center; padding:5rem">
        <h1 style="font-size:4rem">📉</h1>
        <h2>Bem-vindo ao Risk Lab v3</h2>
        <p style="color:{MUTED}">Ajuste os parâmetros na barra lateral e clique em Calcular para ver o VaR e o Stress Test.</p>
    </div>
    """, unsafe_allow_html=True)
