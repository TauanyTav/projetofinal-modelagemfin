# Injeção de CSS Seguro e Avançado
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

#MainMenu,
footer,
header,
.stDeployButton,
div[data-testid="stToolbar"] {
    display: none !important;
    visibility: hidden !important;
}

html,
body,
.stApp {
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
    padding: 0.5rem 1rem !important;
    width: 100% !important;
    box-shadow: 0 4px 14px -4px rgba(34,211,238,0.3) !important;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    border-bottom: 1px solid #1e293b !important;
}

.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #64748b !important;
    border: none !important;
    padding: 0.5rem 1rem !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
}

.stTabs [aria-selected="true"] {
    background: #0f172a !important;
    color: #22d3ee !important;
    border-radius: 4px 4px 0 0;
}

.kpi-card {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 8px;
    padding: 1.25rem;
    height: 100%;
}

.kpi-label {
    color: #64748b;
    font-size: 0.7rem;
    text-transform: uppercase;
    font-weight: 600;
    letter-spacing: 0.05em;
}

.kpi-value {
    font-size: 1.6rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    margin: 0.2rem 0;
}

.var-card {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 8px;
    padding: 1.25rem;
}

</style>
""", unsafe_allow_html=True)
