# modules/insights.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from config import CHART_PALETTE

def show():
    st.markdown("## 💡 Insights & Filters")
    if "df" not in st.session_state:
        st.warning("⚠️ Please upload data first."); return

    df = st.session_state["df"].copy()

    # ── Sidebar Filters ────────────────────────────────────────────────────────
    st.sidebar.divider()
    st.sidebar.markdown("### 🔧 Filters")
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    filters = {}
    for col in cat_cols[:4]:
        opts = ["All"] + sorted(df[col].dropna().unique().tolist())
        sel = st.sidebar.selectbox(f"{col.title()}", opts, key=f"filter_{col}")
        if sel != "All":
            filters[col] = sel
            df = df[df[col] == sel]

    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    if "revenue" in num_cols:
        rev_min, rev_max = float(df["revenue"].min()), float(df["revenue"].max())
        lo, hi = st.sidebar.slider("Revenue Range",
            rev_min, rev_max, (rev_min, rev_max), step=(rev_max-rev_min)/100)
        df = df[(df["revenue"] >= lo) & (df["revenue"] <= hi)]

    st.caption(f"📌 Showing **{len(df):,}** records after filters")
    st.divider()

    # ── KPI Row ────────────────────────────────────────────────────────────────
    c1,c2,c3,c4 = st.columns(4)
    if "revenue" in df.columns:
        c1.metric("Total Revenue",    f"₹{df['revenue'].sum():,.0f}")
        c2.metric("Avg Order Value",  f"₹{df['revenue'].mean():,.0f}")
    if "quantity" in df.columns:
        c3.metric("Total Units Sold", f"{df['quantity'].sum():,}")
    if "customer_id" in df.columns:
        c4.metric("Unique Customers", f"{df['customer_id'].nunique():,}")

    st.divider()

    # ── Charts ─────────────────────────────────────────────────────────────────
    tabs = st.tabs(["📊 Revenue Analysis", "📦 Product", "🌍 Region", "📅 Trends"])

    with tabs[0]:
        if "revenue" in df.columns and cat_cols:
            dim = st.selectbox("Group by", cat_cols)
            agg = df.groupby(dim)["revenue"].agg(["sum","mean","count"]).reset_index()
            agg.columns = [dim, "Total Revenue", "Avg Revenue", "Orders"]
            fig = px.bar(agg, x=dim, y="Total Revenue",
                         color=dim, color_discrete_sequence=CHART_PALETTE,
                         title=f"Revenue by {dim.title()}")
            fig.update_layout(plot_bgcolor="white", paper_bgcolor="white", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(agg, use_container_width=True)

    with tabs[1]:
        if "product" in df.columns and "revenue" in df.columns:
            prod_rev = df.groupby("product")["revenue"].sum().reset_index().sort_values("revenue", ascending=False)
            fig = px.bar(prod_rev, x="product", y="revenue",
                         color="product", color_discrete_sequence=CHART_PALETTE,
                         title="Revenue by Product")
            fig.update_layout(plot_bgcolor="white", paper_bgcolor="white", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

            fig2 = px.treemap(prod_rev, path=["product"], values="revenue",
                              color_discrete_sequence=CHART_PALETTE,
                              title="Product Revenue Treemap")
            st.plotly_chart(fig2, use_container_width=True)

    with tabs[2]:
        if "region" in df.columns and "revenue" in df.columns:
            reg = df.groupby("region")["revenue"].sum().reset_index()
            fig = px.pie(reg, names="region", values="revenue",
                         color_discrete_sequence=CHART_PALETTE,
                         title="Revenue Share by Region", hole=0.4)
            st.plotly_chart(fig, use_container_width=True)

    with tabs[3]:
        date_col = next((c for c in df.columns if "date" in c.lower()), None)
        if date_col and "revenue" in df.columns:
            df[date_col] = pd.to_datetime(df[date_col])
            df["month"] = df[date_col].dt.to_period("M").astype(str)
            trend = df.groupby("month")["revenue"].sum().reset_index()
            fig = px.line(trend, x="month", y="revenue",
                          color_discrete_sequence=["#A8D5BA"],
                          title="Monthly Revenue Trend", markers=True)
            fig.update_layout(plot_bgcolor="white", paper_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True)