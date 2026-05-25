"""
Calculadora de Value at Risk (VaR)
Trabalho Final — Modelagem Aplicada ao Mercado Financeiro
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.stats import norm

# ============================================================
# CONFIGURAÇÃO DA PÁGINA
# ============================================================

st.set_page_config(
    page_title="Calculadora de VaR",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Calculadora de Value at Risk (VaR)")
st.markdown("**Trabalho Final — Modelagem Aplicada ao Mercado Financeiro**")
st.markdown("---")

# ============================================================
# FUNÇÕES BLACK-SCHOLES E GREGAS
# ============================================================

def black_scholes(S, K, T, r, sigma, tipo="call"):
    if T <= 0:
        return max(S - K, 0) if tipo == "call" else max(K - S, 0)
    if sigma <= 0:
        return max(S - K * np.exp(-r * T), 0) if tipo == "call" else max(K * np.exp(-r * T) - S, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if tipo == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
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
# SIDEBAR — PARÂMETROS DA CARTEIRA
# ============================================================

st.sidebar.header("⚙️ Parâmetros da Carteira")

# Ativos de ações
st.sidebar.subheader("📈 Ações")

tickers_input = st.sidebar.text_input(
    "Tickers (separados por vírgula)",
    value="PETR4.SA, VALE3.SA, ITUB4.SA"
)
tickers = [t.strip() for t in tickers_input.split(",") if t.strip()]

quantidades_input = st.sidebar.text_input(
    "Quantidades (separadas por vírgula, mesma ordem dos tickers)",
    value="1000, 800, 1200"
)
try:
    quantidades_lista = [int(q.strip()) for q in quantidades_input.split(",")]
    if len(quantidades_lista) != len(tickers):
        st.sidebar.error("Número de quantidades deve ser igual ao número de tickers.")
        quantidades_lista = [1000] * len(tickers)
except ValueError:
    st.sidebar.error("Informe apenas números inteiros nas quantidades.")
    quantidades_lista = [1000] * len(tickers)

quantidades_acoes = dict(zip(tickers, quantidades_lista))

st.sidebar.subheader("📅 Período Histórico")
data_inicio = st.sidebar.date_input(
    "Data de início",
    value=pd.to_datetime("2022-01-01")
)

st.sidebar.subheader("📐 Parâmetros de VaR")
nivel_confianca = st.sidebar.selectbox(
    "Nível de confiança",
    options=[0.90, 0.95, 0.975, 0.99],
    index=1,
    format_func=lambda x: f"{x*100:.1f}%"
)
horizonte_dias = st.sidebar.number_input(
    "Horizonte (dias)",
    min_value=1, max_value=30, value=1
)

st.sidebar.subheader("📋 Opção Europeia")
ativo_opcao = st.sidebar.selectbox("Ativo objeto da opção", options=tickers)
tipo_opcao = st.sidebar.selectbox("Tipo da opção", options=["call", "put"])
quantidade_opcoes = st.sidebar.number_input("Quantidade de opções", min_value=0, value=1000, step=100)
strike = st.sidebar.number_input("Strike (K)", min_value=1.0, value=40.0, step=0.5)
taxa_livre_risco = st.sidebar.number_input("Taxa livre de risco (ex: 0.105 = 10,5% a.a.)", min_value=0.0, max_value=1.0, value=0.105, step=0.005, format="%.3f")
vencimento_anos = st.sidebar.number_input("Vencimento (em anos)", min_value=0.01, max_value=5.0, value=0.25, step=0.05, format="%.2f")

# ============================================================
# BOTÃO CALCULAR
# ============================================================

calcular = st.sidebar.button("🚀 Calcular VaR", use_container_width=True)

if not calcular:
    st.info("👈 Configure os parâmetros na barra lateral e clique em **Calcular VaR** para iniciar.")

    with st.expander("📖 Teoria: O que é Value at Risk?", expanded=True):
        st.markdown("""
        **Value at Risk (VaR)** responde à pergunta:

        > *"Qual é a perda máxima esperada de uma carteira, em condições normais de mercado, para determinado nível de confiança e horizonte de tempo?"*

        **Exemplo:** Um VaR diário de R$ 1.000.000 com 95% de confiança significa que em 95% dos dias a perda não ultrapassa esse valor.

        ---
        ### Métodos calculados neste aplicativo:

        | Método | Descrição |
        |--------|-----------|
        | **VaR Paramétrico** | Assume distribuição normal dos retornos. Simples e rápido. |
        | **VaR Histórico** | Usa a distribuição empírica dos retornos históricos. |
        | **VaR Full Valuation** | Reprecifica toda a carteira (incluindo opções) em cada cenário histórico. |
        """)

    st.stop()

# ============================================================
# CARREGAMENTO DE DADOS
# ============================================================

with st.spinner("⏳ Baixando dados do Yahoo Finance..."):
    try:
        precos = yf.download(
            tickers,
            start=str(data_inicio),
            auto_adjust=True,
            progress=False
        )["Close"]

        if isinstance(precos, pd.Series):
            precos = precos.to_frame(tickers[0])

        precos = precos.dropna()

        if precos.empty:
            st.error("Nenhum dado encontrado. Verifique os tickers e a data de início.")
            st.stop()

        retornos = precos.pct_change().dropna()

    except Exception as e:
        st.error(f"Erro ao baixar dados: {e}")
        st.stop()

# ============================================================
# CÁLCULOS
# ============================================================

ultimos_precos = precos.iloc[-1]

# Valor da carteira de ações
valor_acoes = sum(quantidades_acoes[t] * ultimos_precos[t] for t in tickers)

# Opção
S0 = ultimos_precos[ativo_opcao]
vol_anual = retornos[ativo_opcao].std() * np.sqrt(252)

preco_opcao_hoje = black_scholes(S0, strike, vencimento_anos, taxa_livre_risco, vol_anual, tipo_opcao)
valor_opcoes = quantidade_opcoes * preco_opcao_hoje
valor_total_carteira = valor_acoes + valor_opcoes

delta_opcao  = delta_bs(S0, strike, vencimento_anos, taxa_livre_risco, vol_anual, tipo_opcao)
gamma_opcao  = gamma_bs(S0, strike, vencimento_anos, taxa_livre_risco, vol_anual)
vega_opcao   = vega_bs(S0, strike, vencimento_anos, taxa_livre_risco, vol_anual)

# Pesos para VaR paramétrico
pesos = np.array([quantidades_acoes[t] * ultimos_precos[t] / valor_acoes for t in tickers])
retorno_carteira = retornos.dot(pesos)
media_carteira = retorno_carteira.mean()
vol_carteira   = retorno_carteira.std()
percentil      = 1 - nivel_confianca
z              = norm.ppf(1 - nivel_confianca)

# VaR Paramétrico
var_parametrico = -(media_carteira * horizonte_dias + z * vol_carteira * np.sqrt(horizonte_dias)) * valor_acoes

# VaR Histórico
var_historico = -np.percentile(retorno_carteira, percentil * 100) * valor_acoes

# VaR Full Valuation
cenarios_pnl = []
for i in range(len(retornos)):
    choque = retornos.iloc[i]
    novos_precos = ultimos_precos * (1 + choque)
    novo_valor_acoes = sum(quantidades_acoes[t] * novos_precos[t] for t in tickers)
    S_cenario = novos_precos[ativo_opcao]
    T_cenario = max(vencimento_anos - horizonte_dias / 252, 0)
    novo_preco_opcao = black_scholes(S_cenario, strike, T_cenario, taxa_livre_risco, vol_anual, tipo_opcao)
    novo_valor_opcoes = quantidade_opcoes * novo_preco_opcao
    pnl = (novo_valor_acoes + novo_valor_opcoes) - valor_total_carteira
    cenarios_pnl.append(pnl)

cenarios_pnl = np.array(cenarios_pnl)
var_full_valuation = -np.percentile(cenarios_pnl, percentil * 100)

# ============================================================
# EXIBIÇÃO — ABA PRINCIPAL
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs(["📊 Resumo", "📈 Gráficos", "🔢 Gregas da Opção", "📚 Teoria"])

# ===== ABA 1: RESUMO =====
with tab1:
    st.subheader("💼 Composição da Carteira")
    col1, col2, col3 = st.columns(3)
    col1.metric("Valor das Ações", f"R$ {valor_acoes:,.2f}")
    col2.metric("Valor das Opções", f"R$ {valor_opcoes:,.2f}")
    col3.metric("Valor Total", f"R$ {valor_total_carteira:,.2f}")

    st.markdown("---")

    # Tabela de posições de ações
    st.subheader("📋 Posições em Ações")
    dados_posicoes = []
    for t in tickers:
        preco = ultimos_precos[t]
        qtd = quantidades_acoes[t]
        valor = qtd * preco
        peso = valor / valor_acoes
        dados_posicoes.append({
            "Ticker": t,
            "Último Preço (R$)": f"{preco:.2f}",
            "Quantidade": qtd,
            "Valor (R$)": f"{valor:,.2f}",
            "Peso (%)": f"{peso*100:.1f}%"
        })
    st.dataframe(pd.DataFrame(dados_posicoes), use_container_width=True, hide_index=True)

    # Opção
    st.subheader("🎯 Posição em Opção")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Ativo Objeto", ativo_opcao)
    col2.metric("Tipo", tipo_opcao.upper())
    col3.metric("Preço BS (R$)", f"{preco_opcao_hoje:.4f}")
    col4.metric("Valor Total Opções (R$)", f"{valor_opcoes:,.2f}")

    st.markdown("---")

    # Resultados VaR
    st.subheader("⚠️ Resultados de VaR")
    st.markdown(f"**Nível de confiança:** {nivel_confianca*100:.1f}%  |  **Horizonte:** {horizonte_dias} dia(s)")

    col1, col2, col3 = st.columns(3)
    col1.metric(
        "VaR Paramétrico (Ações)",
        f"R$ {var_parametrico:,.2f}",
        help="Calculado assumindo distribuição normal dos retornos da carteira de ações."
    )
    col2.metric(
        "VaR Histórico (Ações)",
        f"R$ {var_historico:,.2f}",
        help="Calculado usando o percentil histórico dos retornos da carteira de ações."
    )
    col3.metric(
        "VaR Full Valuation (Ações + Opções)",
        f"R$ {var_full_valuation:,.2f}",
        help="Reprecifica toda a carteira (incluindo opções via Black-Scholes) em cada cenário histórico."
    )

    # Tabela comparativa
    st.markdown("#### Comparação dos Métodos")
    df_comp = pd.DataFrame({
        "Método": ["VaR Paramétrico — Ações", "VaR Histórico — Ações", "VaR Full Valuation — Ações + Opções"],
        "VaR (R$)": [var_parametrico, var_historico, var_full_valuation],
        "VaR (% do valor total)": [
            f"{var_parametrico/valor_total_carteira*100:.2f}%",
            f"{var_historico/valor_total_carteira*100:.2f}%",
            f"{var_full_valuation/valor_total_carteira*100:.2f}%"
        ]
    })
    df_comp["VaR (R$)"] = df_comp["VaR (R$)"].apply(lambda x: f"R$ {x:,.2f}")
    st.dataframe(df_comp, use_container_width=True, hide_index=True)

    # Volatilidade
    st.markdown("---")
    st.subheader("📉 Estatísticas de Risco")
    col1, col2, col3 = st.columns(3)
    col1.metric("Volatilidade Diária da Carteira", f"{vol_carteira*100:.2f}%")
    col2.metric("Volatilidade Anual da Carteira", f"{vol_carteira*np.sqrt(252)*100:.2f}%")
    col3.metric(f"Volatilidade Anual de {ativo_opcao}", f"{vol_anual*100:.2f}%")


# ===== ABA 2: GRÁFICOS =====
with tab2:
    st.subheader("Distribuição Histórica dos Retornos da Carteira de Ações")
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    ax1.hist(retorno_carteira, bins=50, color="#4C72B0", edgecolor="white", alpha=0.85)
    ax1.axvline(
        np.percentile(retorno_carteira, percentil * 100),
        color="red", linestyle="--", linewidth=1.8,
        label=f"VaR Histórico ({nivel_confianca*100:.0f}%)"
    )
    ax1.set_xlabel("Retorno diário")
    ax1.set_ylabel("Frequência")
    ax1.legend()
    ax1.set_title("Distribuição Histórica dos Retornos — Carteira de Ações")
    st.pyplot(fig1)
    plt.close(fig1)

    st.subheader("Distribuição de P&L — Full Valuation (Ações + Opções)")
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.hist(cenarios_pnl, bins=50, color="#55A868", edgecolor="white", alpha=0.85)
    ax2.axvline(
        np.percentile(cenarios_pnl, percentil * 100),
        color="red", linestyle="--", linewidth=1.8,
        label=f"VaR Full Valuation ({nivel_confianca*100:.0f}%)"
    )
    ax2.set_xlabel("P&L da carteira (R$)")
    ax2.set_ylabel("Frequência")
    ax2.legend()
    ax2.set_title("Distribuição de P&L — Full Valuation")
    st.pyplot(fig2)
    plt.close(fig2)

    st.subheader("Comparação entre Métodos de VaR")
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    metodos = ["Paramétrico\n(Ações)", "Histórico\n(Ações)", "Full Valuation\n(Ações + Opções)"]
    valores = [var_parametrico, var_historico, var_full_valuation]
    cores = ["#4C72B0", "#55A868", "#C44E52"]
    bars = ax3.bar(metodos, valores, color=cores, edgecolor="white")
    ax3.bar_label(bars, labels=[f"R$ {v:,.0f}" for v in valores], padding=4, fontsize=9)
    ax3.set_ylabel("VaR (R$)")
    ax3.set_title("Comparação entre Métodos de VaR")
    st.pyplot(fig3)
    plt.close(fig3)

    st.subheader(f"Sensibilidade do Preço da Opção ({tipo_opcao.upper()}) ao Preço do Ativo")
    precos_sim = np.linspace(S0 * 0.7, S0 * 1.3, 200)
    precos_op_sim = [black_scholes(s, strike, vencimento_anos, taxa_livre_risco, vol_anual, tipo_opcao) for s in precos_sim]
    fig4, ax4 = plt.subplots(figsize=(10, 4))
    ax4.plot(precos_sim, precos_op_sim, color="#4C72B0", linewidth=2)
    ax4.axvline(strike, color="gray", linestyle="--", label=f"Strike = {strike}")
    ax4.axvline(S0, color="orange", linestyle="--", label=f"Preço atual = {S0:.2f}")
    ax4.set_xlabel("Preço do ativo objeto (R$)")
    ax4.set_ylabel("Preço da opção (R$)")
    ax4.set_title(f"Sensibilidade do Preço da {tipo_opcao.upper()} ao Preço do Ativo")
    ax4.legend()
    st.pyplot(fig4)
    plt.close(fig4)

    st.subheader("Preços Históricos dos Ativos")
    fig5, ax5 = plt.subplots(figsize=(10, 4))
    for t in tickers:
        ax5.plot(precos[t] / precos[t].iloc[0], label=t)
    ax5.set_ylabel("Retorno acumulado (base 1)")
    ax5.set_title("Retorno Acumulado dos Ativos")
    ax5.legend()
    st.pyplot(fig5)
    plt.close(fig5)


# ===== ABA 3: GREGAS =====
with tab3:
    st.subheader(f"📐 Gregas da {tipo_opcao.upper()} — {ativo_opcao}")

    col1, col2, col3 = st.columns(3)
    col1.metric("Delta (Δ)", f"{delta_opcao:.4f}", help="Sensibilidade do preço da opção ao preço do ativo objeto.")
    col2.metric("Gamma (Γ)", f"{gamma_opcao:.6f}", help="Sensibilidade do Delta ao preço do ativo objeto.")
    col3.metric("Vega (ν)", f"{vega_opcao:.4f}", help="Sensibilidade do preço da opção à volatilidade (por 1% de vol).")

    st.markdown("---")
    st.markdown("""
    **Interpretação das Gregas:**

    - **Delta (Δ):** Para uma variação de R$ 1 no preço do ativo, o preço da opção varia aproximadamente Δ reais.
    - **Gamma (Γ):** Mede a convexidade. Quanto maior o Gamma, mais a opção se comporta de forma não linear.
    - **Vega (ν):** Para um aumento de 1 ponto percentual na volatilidade, o preço da opção varia Vega reais.
    """)

    # Tabela de cenários de delta
    st.subheader("Análise de Delta em diferentes preços do ativo")
    precos_range = np.linspace(S0 * 0.80, S0 * 1.20, 9)
    tabela_delta = []
    for s in precos_range:
        d = delta_bs(s, strike, vencimento_anos, taxa_livre_risco, vol_anual, tipo_opcao)
        g = gamma_bs(s, strike, vencimento_anos, taxa_livre_risco, vol_anual)
        p = black_scholes(s, strike, vencimento_anos, taxa_livre_risco, vol_anual, tipo_opcao)
        tabela_delta.append({
            "Preço do Ativo (R$)": f"{s:.2f}",
            "Preço da Opção (R$)": f"{p:.4f}",
            "Delta": f"{d:.4f}",
            "Gamma": f"{g:.6f}"
        })
    st.dataframe(pd.DataFrame(tabela_delta), use_container_width=True, hide_index=True)


# ===== ABA 4: TEORIA =====
with tab4:
    st.subheader("📚 Teoria e Interpretação")

    with st.expander("🔵 VaR Paramétrico", expanded=False):
        st.markdown("""
        O VaR Paramétrico assume que os retornos seguem **distribuição normal**.

        **Fórmula:**
        ```
        VaR = Z × σ × V
        ```
        onde:
        - `Z` = quantil da distribuição normal
        - `σ` = volatilidade da carteira
        - `V` = valor da carteira

        **Vantagens:** Simples, rápido, fácil de comunicar.

        **Limitações:** Assume normalidade; pode subestimar caudas gordas; não captura bem opções.
        """)

    with st.expander("🟢 VaR Histórico", expanded=False):
        st.markdown("""
        O VaR Histórico usa diretamente os **retornos históricos observados**, sem supor distribuição.

        **Passos:**
        1. Calcular o retorno histórico da carteira.
        2. Ordenar os retornos.
        3. Escolher o percentil do nível de confiança.

        **Vantagens:** Não exige normalidade; usa dados reais.

        **Limitações:** Depende da janela histórica; assume que o passado representa o futuro.
        """)

    with st.expander("🔴 VaR Full Valuation", expanded=False):
        st.markdown("""
        O VaR Full Valuation **reprecifica toda a carteira** (incluindo opções via Black-Scholes) em cada cenário histórico.

        **Por que usar?**

        Opções têm payoff não linear:
        - Call: `max(S - K, 0)`
        - Put: `max(K - S, 0)`

        O Full Valuation captura essa convexidade, sendo o método mais adequado para carteiras com derivativos.

        **Vantagens:** Captura não linearidade; mais preciso para opções.

        **Limitações:** Mais custoso computacionalmente; ainda limitado pela janela histórica.
        """)

    with st.expander("🟡 Black-Scholes", expanded=False):
        st.markdown(r"""
        **Fórmula para Call europeia:**
        ```
        C = S·N(d1) - K·e^{-rT}·N(d2)
        ```
        **Para Put europeia:**
        ```
        P = K·e^{-rT}·N(-d2) - S·N(-d1)
        ```
        **Onde:**
        ```
        d1 = [ln(S/K) + (r + σ²/2)·T] / (σ·√T)
        d2 = d1 - σ·√T
        ```
        **Hipóteses:** sem arbitragem, volatilidade constante, taxa constante, retornos lognormais.
        """)

    with st.expander("⚠️ Limitações do VaR", expanded=False):
        st.markdown("""
        1. **Não informa a magnitude da perda além do VaR.**
        2. **Depende da janela histórica usada.**
        3. **Pode subestimar eventos extremos** (caudas gordas do mercado).
        4. **Pode falhar em carteiras com opções** se não usar Full Valuation.
        5. **Não substitui stress test.** Na prática, usa-se VaR junto com Expected Shortfall, limites de perda e análise de cenários macroeconômicos.
        """)

# ============================================================
# RODAPÉ
# ============================================================
st.markdown("---")
st.caption("Calculadora de VaR — Trabalho Final de Modelagem Aplicada ao Mercado Financeiro | Desenvolvido com Streamlit")
