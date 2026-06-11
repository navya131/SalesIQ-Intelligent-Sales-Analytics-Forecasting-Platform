# modules/anomaly_detection.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from config import CHART_PALETTE

def show():
    st.markdown("## 🚨 Anomaly Detection")
    st.caption("Automatically detect unusual sales patterns using Isolation Forest.")
    if "df" not in st.session_state:
        st.warning("⚠️ Please upload data first."); return

    df = st.session_state["df"].copy()
    num_cols = df.select_dtypes(include=np.number).columns.tolist()

    col1, col2 = st.columns(2)
    feat_cols    = col1.multiselect("Features", num_cols,
        default=[c for c in ["revenue","quantity","discount"] if c in num_cols])
    contamination = col2.slider("Expected anomaly %", 1, 20, 5) / 100

    if st.button("🔍 Detect Anomalies") and feat_cols:
        X = df[feat_cols].fillna(0)
        scaler = StandardScaler()
        X_sc = scaler.fit_transform(X)

        clf = IsolationForest(contamination=contamination, random_state=42)
        df["anomaly"] = clf.fit_predict(X_sc)
        df["anomaly_score"] = clf.score_samples(X_sc)
        df["is_anomaly"] = df["anomaly"].map({-1:"🔴 Anomaly", 1:"🟢 Normal"})

        n_anom = (df["anomaly"] == -1).sum()
        st.error(f"🚨 **{n_anom} anomalies detected** ({n_anom/len(df):.1%} of records)")

        c1, c2 = st.columns(2)
        c1.metric("Total Records",  f"{len(df):,}")
        c2.metric("Anomaly Count",  f"{n_anom:,}")

        st.markdown("### 📋 Anomalous Records")
        anom_df = df[df["anomaly"] == -1][feat_cols + ["is_anomaly","anomaly_score"]].sort_values("anomaly_score")
        st.dataframe(anom_df.reset_index(drop=True), use_container_width=True)

        st.markdown("### 🔍 Anomaly Visualization")
        if len(feat_cols) >= 2:
            fig = go.Figure()
            normal   = df[df["anomaly"] ==  1]
            anomalies= df[df["anomaly"] == -1]
            fig.add_trace(go.Scatter(x=normal[feat_cols[0]], y=normal[feat_cols[1]],
                mode="markers", name="Normal",
                marker=dict(color="#A8D5BA", size=5, opacity=0.6)))
            fig.add_trace(go.Scatter(x=anomalies[feat_cols[0]], y=anomalies[feat_cols[1]],
                mode="markers", name="Anomaly",
                marker=dict(color="#F7C5CC", size=9, symbol="x", opacity=0.9)))
            fig.update_layout(title=f"Anomalies: {feat_cols[0]} vs {feat_cols[1]}",
                              plot_bgcolor="white", paper_bgcolor="white",
                              xaxis_title=feat_cols[0], yaxis_title=feat_cols[1])
            st.plotly_chart(fig, use_container_width=True)

        csv = anom_df.to_csv(index=False).encode()
        st.download_button("⬇️ Download Anomaly Report", csv,
                           "anomalies.csv", "text/csv")