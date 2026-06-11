# modules/home.py
import streamlit as st
from config import COLORS

def show():
    st.markdown("""
    <div style='text-align:center; padding: 40px 0 10px 0;'>
        <h1 style='font-size:2.8rem; color:#2D3142; font-weight:700;'>📊 SalesIQ</h1>
        <p style='font-size:1.15rem; color:#6B7280;'>
            Intelligent Sales Analytics · Forecasting · Churn Prediction · BI Insights
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    cols = st.columns(4)
    cards = [
        ("📂", "Upload Data",       "CSV/Excel support with auto-validation",           "#A8D5BA"),
        ("📈", "Forecast Sales",    "Prophet, ARIMA & XGBoost models",                  "#B5C8E8"),
        ("🔄", "Churn Prediction",  "ML-powered customer churn scoring",                "#F7C5CC"),
        ("📋", "Generate Reports",  "Export to PDF & Excel in one click",               "#FFE5A0"),
    ]
    for col, (icon, title, desc, color) in zip(cols, cards):
        col.markdown(f"""
        <div style='background:{color}30; border:1px solid {color};
             border-radius:14px; padding:22px 16px; text-align:center;'>
            <div style='font-size:2rem;'>{icon}</div>
            <div style='font-weight:600; font-size:1rem; margin:8px 0 4px;'>{title}</div>
            <div style='font-size:0.82rem; color:#6B7280;'>{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.info("👈 **Start by uploading your sales data** using the sidebar → *Data Upload*")

    with st.expander("📌 How to use this app"):
        st.markdown("""
        1. **Upload** your CSV/Excel sales file  
        2. **Preprocess** the data and explore EDA  
        3. **Train models** and **forecast** future sales  
        4. **Predict churn** for your customers  
        5. **Explore insights** with interactive filters  
        6. **Detect anomalies** in your sales pipeline  
        7. **Download reports** as PDF or Excel  
        """)