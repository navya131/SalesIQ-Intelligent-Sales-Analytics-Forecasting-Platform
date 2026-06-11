# modules/preprocessing.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from config import CHART_PALETTE

def show():
    st.markdown("## 🔧 Preprocessing & EDA")
    if "df" not in st.session_state:
        st.warning("⚠️ Please upload data first."); return

    df = st.session_state["df"].copy()
    tabs = st.tabs(["🧹 Preprocessing", "📊 EDA — Overview",
                    "📉 Distributions", "🔗 Correlations", "📅 Time Analysis"])

    # ── TAB 1: Preprocessing ─────────────────────────────────────────────────
    with tabs[0]:
        st.markdown("### 🧹 Data Cleaning")
        col1, col2 = st.columns(2)
        with col1:
            missing = df.isnull().sum()
            missing = missing[missing > 0]
            if missing.empty:
                st.success("✅ No missing values detected.")
            else:
                st.warning(f"⚠️ {len(missing)} columns have missing values")
                strategy = st.selectbox("Fill strategy", ["Mean/Mode", "Median", "Drop rows", "Forward fill"])
                if st.button("Apply Fill Strategy"):
                    for col in missing.index:
                        if df[col].dtype in [np.float64, np.int64]:
                            if strategy == "Mean/Mode":   df[col].fillna(df[col].mean(), inplace=True)
                            elif strategy == "Median":    df[col].fillna(df[col].median(), inplace=True)
                            elif strategy == "Drop rows": df.dropna(subset=[col], inplace=True)
                            else:                         df[col].fillna(method="ffill", inplace=True)
                        else:
                            df[col].fillna(df[col].mode()[0], inplace=True)
                    st.session_state["df"] = df
                    st.success("✅ Missing values handled.")

        with col2:
            dups = df.duplicated().sum()
            st.metric("Duplicate Rows", dups)
            if dups > 0 and st.button("Remove Duplicates"):
                df.drop_duplicates(inplace=True)
                st.session_state["df"] = df
                st.success(f"✅ Removed {dups} duplicates.")

        st.divider()
        st.markdown("### 🔢 Data Types")
        st.dataframe(df.dtypes.reset_index().rename(columns={"index":"Column", 0:"Type"}),
                     use_container_width=True)

        st.markdown("### 📐 Statistical Summary")
        st.dataframe(df.describe(include="all").T, use_container_width=True)

    # ── TAB 2: EDA Overview ───────────────────────────────────────────────────
    with tabs[1]:
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = df.select_dtypes(include="object").columns.tolist()

        st.markdown("### 🗃️ Numeric Columns Overview")
        c1, c2, c3 = st.columns(3)
        c1.metric("Numeric Cols",     len(num_cols))
        c2.metric("Categorical Cols", len(cat_cols))
        c3.metric("Total Records",    f"{len(df):,}")

        if "revenue" in df.columns:
            st.markdown("### 💰 Revenue by Category")
            for cat in cat_cols[:3]:
                if cat in df.columns:
                    fig = px.bar(df.groupby(cat)["revenue"].sum().reset_index(),
                                 x=cat, y="revenue", color=cat,
                                 color_discrete_sequence=CHART_PALETTE,
                                 title=f"Revenue by {cat.title()}")
                    fig.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                                      showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)

    # ── TAB 3: Distributions ──────────────────────────────────────────────────
    with tabs[2]:
        st.markdown("### 📉 Value Distributions")
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        sel = st.selectbox("Select column", num_cols)
        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(df, x=sel, nbins=40,
                               color_discrete_sequence=["#A8D5BA"],
                               title=f"Histogram — {sel}")
            fig.update_layout(plot_bgcolor="white", paper_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.box(df, y=sel, color_discrete_sequence=["#B5C8E8"],
                         title=f"Box Plot — {sel}")
            fig.update_layout(plot_bgcolor="white", paper_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 🍩 Categorical Shares")
        cat_cols = df.select_dtypes(include="object").columns.tolist()
        if cat_cols:
            sel_cat = st.selectbox("Categorical column", cat_cols)
            vc = df[sel_cat].value_counts().reset_index()
            vc.columns = [sel_cat, "count"]
            fig = px.pie(vc, names=sel_cat, values="count",
                         color_discrete_sequence=CHART_PALETTE,
                         title=f"Share by {sel_cat.title()}", hole=0.4)
            st.plotly_chart(fig, use_container_width=True)

    # ── TAB 4: Correlations ───────────────────────────────────────────────────
    with tabs[3]:
        st.markdown("### 🔗 Correlation Heatmap")
        num_df = df.select_dtypes(include=np.number)
        corr = num_df.corr()
        fig = px.imshow(corr, color_continuous_scale="RdBu_r",
                        zmin=-1, zmax=1, text_auto=".2f",
                        title="Feature Correlation Matrix")
        fig.update_layout(paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 🔍 Scatter Explorer")
        c1, c2, c3 = st.columns(3)
        num_cols = num_df.columns.tolist()
        x_col = c1.selectbox("X axis", num_cols, index=0)
        y_col = c2.selectbox("Y axis", num_cols, index=min(1, len(num_cols)-1))
        cat_cols = df.select_dtypes(include="object").columns.tolist()
        color_col = c3.selectbox("Color by", ["None"] + cat_cols)
        fig = px.scatter(df, x=x_col, y=y_col,
                         color=None if color_col == "None" else color_col,
                         color_discrete_sequence=CHART_PALETTE, opacity=0.6,
                         title=f"{x_col} vs {y_col}")
        fig.update_layout(plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

    # ── TAB 5: Time Analysis ──────────────────────────────────────────────────
    with tabs[4]:
        date_cols = df.select_dtypes(include=["datetime64","object"]).columns.tolist()
        date_col = None
        for c in date_cols:
            try:
                df[c] = pd.to_datetime(df[c])
                date_col = c; break
            except: pass

        if date_col is None:
            st.info("No date column detected. Using index."); return

        df = df.sort_values(date_col)
        if "revenue" in df.columns:
            freq = st.selectbox("Aggregation", ["D","W","ME"], format_func=lambda x: {"D":"Daily","W":"Weekly","ME":"Monthly"}[x])
            ts = df.set_index(date_col)["revenue"].resample(freq).sum().reset_index()
            ts.columns = ["date", "revenue"]
            fig = px.line(ts, x="date", y="revenue",
                          color_discrete_sequence=["#A8D5BA"],
                          title="Revenue Over Time")
            fig.update_traces(fill="tozeroy", fillcolor="rgba(168,213,186,0.15)")
            fig.update_layout(plot_bgcolor="white", paper_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True)