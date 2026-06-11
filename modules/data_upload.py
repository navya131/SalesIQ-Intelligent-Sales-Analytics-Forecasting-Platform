# modules/data_upload.py
import streamlit as st
import pandas as pd
import numpy as np
from utils.helpers import validate_dataframe, get_df_summary

def show():
    st.markdown("## 📂 Data Upload")
    st.caption("Upload your sales CSV or Excel file to get started.")
    st.divider()

    upload_tab, sample_tab = st.tabs(["⬆️ Upload Your File", "📦 Use Sample Data"])

    with upload_tab:
        uploaded = st.file_uploader(
            "Drop your file here",
            type=["csv", "xlsx", "xls"],
            help="Supported: CSV, Excel (.xlsx, .xls)"
        )

        if uploaded:
            try:
                if uploaded.name.endswith(".csv"):
                    df = pd.read_csv(uploaded)
                else:
                    df = pd.read_excel(uploaded)

                st.session_state["raw_df"] = df
                st.session_state["df"] = df.copy()
                _preview(df)
            except Exception as e:
                st.error(f"❌ Failed to read file: {e}")

    with sample_tab:
        st.info("No data? Load our built-in sample retail sales dataset.")
        if st.button("📦 Load Sample Dataset"):
            df = _generate_sample()
            st.session_state["raw_df"] = df
            st.session_state["df"] = df.copy()
            st.success("✅ Sample dataset loaded!")
            _preview(df)

    if "df" in st.session_state and st.session_state["df"] is not None:
        df = st.session_state["df"]
        st.divider()
        st.markdown("### 🗂️ Dataset Summary")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Rows", f"{df.shape[0]:,}")
        col2.metric("Columns", df.shape[1])
        col3.metric("Missing Values", int(df.isnull().sum().sum()))
        col4.metric("Duplicate Rows", int(df.duplicated().sum()))

        st.markdown("### 📋 Column Info")
        info_df = pd.DataFrame({
            "Column": df.columns,
            "Type": df.dtypes.values,
            "Non-Null": df.notnull().sum().values,
            "Nulls": df.isnull().sum().values,
            "Unique": df.nunique().values,
        })
        st.dataframe(info_df, use_container_width=True)


def _preview(df):
    st.success(f"✅ Loaded **{df.shape[0]:,} rows × {df.shape[1]} columns**")
    with st.expander("👁️ Preview first 20 rows"):
        st.dataframe(df.head(20), use_container_width=True)


def _generate_sample():
    np.random.seed(42)
    n = 1200
    dates = pd.date_range("2021-01-01", periods=n, freq="D")
    products  = np.random.choice(["Laptop","Phone","Tablet","Watch","Earbuds"], n)
    regions   = np.random.choice(["North","South","East","West"], n)
    channels  = np.random.choice(["Online","Retail","Partner"], n)
    customers = [f"CUST{str(i).zfill(4)}" for i in np.random.randint(1, 301, n)]
    qty       = np.random.randint(1, 15, n)
    price     = np.where(products == "Laptop", 75000,
                np.where(products == "Phone",  35000,
                np.where(products == "Tablet", 25000,
                np.where(products == "Watch",  15000, 5000))))
    discount  = np.random.uniform(0, 0.25, n).round(2)
    revenue   = (qty * price * (1 - discount)).round(2)
    csat      = np.random.choice([1,2,3,4,5], n, p=[0.05,0.10,0.20,0.35,0.30])
    churn     = np.where((csat <= 2) | (discount < 0.05), 1, 0)
    churn     = np.where(np.random.rand(n) < 0.12, 1, churn)  # add noise

    return pd.DataFrame({
        "date": dates, "customer_id": customers, "product": products,
        "region": regions, "channel": channels, "quantity": qty,
        "unit_price": price, "discount": discount,
        "revenue": revenue, "csat_score": csat, "churn": churn,
    })