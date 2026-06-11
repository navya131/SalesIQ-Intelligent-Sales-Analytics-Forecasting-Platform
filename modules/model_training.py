# modules/model_training.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from config import CHART_PALETTE
import joblib, io

def show():
    st.markdown("## 🤖 Model Training & Evaluation")
    if "df" not in st.session_state:
        st.warning("⚠️ Please upload data first."); return

    df = st.session_state["df"].copy()
    num_cols = df.select_dtypes(include=np.number).columns.tolist()

    st.markdown("### ⚙️ Configuration")
    col1, col2, col3 = st.columns(3)
    target = col1.selectbox("Target variable", num_cols,
        index=num_cols.index("revenue") if "revenue" in num_cols else 0)
    model_name = col2.selectbox("Algorithm",
        ["Random Forest", "Gradient Boosting", "Linear Regression", "Ridge"])
    test_size = col3.slider("Test split %", 10, 40, 20)

    features = st.multiselect("Feature columns (numeric only)",
        [c for c in num_cols if c != target],
        default=[c for c in num_cols if c != target][:5])

    if st.button("🚀 Train Model") and features:
        df_enc = _encode(df.copy())
        X = df_enc[features].fillna(0)
        y = df_enc[target].fillna(0)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size/100, random_state=42)

        model = _get_model(model_name)
        with st.spinner("Training…"):
            model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae  = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2   = r2_score(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-9))) * 100

        st.session_state["trained_model"] = model
        st.session_state["model_features"] = features
        st.session_state["model_metrics"]  = {"MAE":mae,"RMSE":rmse,"R²":r2,"MAPE":mape}

        st.success(f"✅ {model_name} trained on {len(X_train):,} samples!")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("MAE",   f"{mae:,.2f}")
        col2.metric("RMSE",  f"{rmse:,.2f}")
        col3.metric("R²",    f"{r2:.4f}")
        col4.metric("MAPE",  f"{mape:.2f}%")

        _plot_actual_vs_pred(y_test.values, y_pred, model_name)

        if hasattr(model, "feature_importances_"):
            _plot_feature_importance(model, features)


def _encode(df):
    for col in df.select_dtypes(include="object").columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    return df


def _get_model(name):
    return {
        "Random Forest":       RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting":   GradientBoostingRegressor(n_estimators=100, random_state=42),
        "Linear Regression":   LinearRegression(),
        "Ridge":               Ridge(alpha=1.0),
    }[name]


def _plot_actual_vs_pred(y_true, y_pred, name):
    st.markdown("### 📉 Actual vs Predicted")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=y_true[:100], name="Actual",
        line=dict(color="#4A90D9", width=2.5)
    ))
    fig.add_trace(go.Scatter(
        y=y_pred[:100], name="Predicted",
        line=dict(color="#A8D5BA", width=2.5, dash="dash")
    ))
    fig.update_layout(
        plot_bgcolor="#F8FBFF",
        paper_bgcolor="#F8FBFF",
        font=dict(color="#1A3550", size=13),
        title=f"{name} — Actual vs Predicted (first 100 test samples)",
        title_font=dict(color="#1A3550", size=15),
        xaxis=dict(
            title="Sample Index",
            title_font=dict(color="#1A3550"),
            tickfont=dict(color="#1A3550"),
            gridcolor="#D6E8F7",
        ),
        yaxis=dict(
            title="Revenue (₹)",
            title_font=dict(color="#1A3550"),
            tickfont=dict(color="#1A3550"),
            gridcolor="#D6E8F7",
        ),
        legend=dict(
            font=dict(color="#1A3550"),
            bgcolor="#EEF4FB",
            bordercolor="#B8D4EE",
            borderwidth=1,
        ),
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)


def _plot_feature_importance(model, features):
    st.markdown("### 🎯 Feature Importance")
    imp = pd.DataFrame({
        "Feature":    features,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=True)

    fig = px.bar(
        imp, x="Importance", y="Feature",
        orientation="h",
        color="Importance",
        color_continuous_scale=[
            "#D6EAF8", "#A8D5BA", "#7EC8E3", "#4A90D9"
        ],
        title="Feature Importances",
        text=imp["Importance"].apply(lambda x: f"{x:.3f}")
    )
    fig.update_traces(
        textposition="outside",
        textfont=dict(color="#1A3550", size=12)
    )
    fig.update_layout(
        plot_bgcolor="#F8FBFF",
        paper_bgcolor="#F8FBFF",
        font=dict(color="#1A3550", size=13),
        title_font=dict(color="#1A3550", size=16),
        showlegend=False,
        xaxis=dict(
            title="Importance Score",
            title_font=dict(color="#1A3550"),
            tickfont=dict(color="#1A3550"),
            gridcolor="#D6E8F7",
            showgrid=True,
        ),
        yaxis=dict(
            title="Feature",
            title_font=dict(color="#1A3550"),
            tickfont=dict(color="#1A3550", size=12),
            gridcolor="#D6E8F7",
        ),
        coloraxis_colorbar=dict(
            title="Score",
            tickfont=dict(color="#1A3550"),
            titlefont=dict(color="#1A3550"),
        ),
        margin=dict(l=20, r=80, t=50, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)