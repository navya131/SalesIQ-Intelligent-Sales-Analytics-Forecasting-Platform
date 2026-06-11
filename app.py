# app.py

import streamlit as st
from config import APP_TITLE, APP_ICON, SIDEBAR_PAGES, COLORS
import pathlib

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Inject CSS ────────────────────────────────────────────────────────────────
css_path = pathlib.Path("assets/style.css")
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)

# ── Force light baby blue theme ───────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { background-color: #EEF4FB !important; }
    [data-testid="stHeader"]           { background-color: #C8DFEE !important; }
    [data-testid="stToolbar"]          { background-color: #C8DFEE !important; }
    .stApp                             { background-color: #EEF4FB !important; }
</style>
""", unsafe_allow_html=True)

# ── Top navigation bar ────────────────────────────────────────────────────────
st.markdown("""
<div style='
    background: linear-gradient(90deg, #C8DFEE, #D6EAF8);
    padding: 12px 28px;
    border-radius: 0 0 16px 16px;
    border-bottom: 2px solid #A8C8E8;
    display: flex;
    align-items: center;
    gap: 16px;
    margin-bottom: 10px;
    box-shadow: 0 3px 12px rgba(100,160,220,0.15);
'>
    <span style='font-size:1.6rem;'>📊</span>
    <span style='font-size:1.2rem; font-weight:700; color:#1A3550;'>SalesIQ</span>
    <span style='color:#4A7FA5; font-size:0.9rem;'>
        Intelligent Sales Analytics Platform
    </span>
    <span style='margin-left:auto; background:#A8D5BA; color:#1A3550;
                 padding:4px 14px; border-radius:20px; font-size:0.8rem;
                 font-weight:600;'>v2.0</span>
</div>
""", unsafe_allow_html=True)

# ── Sidebar navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 10px 0 6px 0;'>
        <div style='font-size:2rem;'>📊</div>
        <div style='font-size:1.3rem; font-weight:700; color:#1A3550;'>SalesIQ</div>
        <div style='font-size:0.8rem; color:#4A7FA5;'>Intelligent Sales Analytics</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()
    page = st.radio("Navigate", SIDEBAR_PAGES, label_visibility="collapsed")
    st.divider()
    st.markdown("""
    <div style='text-align:center;'>
        <span style='font-size:0.75rem; color:#4A7FA5;'>
            v2.0 · Powered by ML + Streamlit
        </span>
    </div>
    """, unsafe_allow_html=True)

# ── Route pages ───────────────────────────────────────────────────────────────
if page == "🏠 Home":
    from modules.home import show
    show()
elif page == "📂 Data Upload":
    from modules.data_upload import show
    show()
elif page == "🔧 Preprocessing & EDA":
    from modules.preprocessing import show
    show()
elif page == "📈 Sales Forecasting":
    from modules.forecasting import show
    show()
elif page == "🤖 Model Training":
    from modules.model_training import show
    show()
elif page == "🔄 Churn Prediction":
    from modules.churn_prediction import show
    show()
elif page == "💡 Insights & Filters":
    from modules.insights import show
    show()
elif page == "🚨 Anomaly Detection":
    from modules.anomaly_detection import show
    show()
elif page == "📋 Reports":
    from modules.reports import show
    show()