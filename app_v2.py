"""
Calculadora de Value at Risk (VaR) — v3.2 (Professional Edition)
Trabalho Final | Modelagem Aplicada ao Mercado Financeiro
Melhorias: Correção estrita de hifens de quebra de linha (sintaxe) e session_state robusto.
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

/* Custom Tab Styling */
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
.kpi-value {{ color: {TEXT}; font-size: 1.75rem; font-weight: 700; font-family: 'JetBrains Mono', monospace; margin: 0.25rem 0
