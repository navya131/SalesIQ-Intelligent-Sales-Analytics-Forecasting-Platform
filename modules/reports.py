# modules/reports.py
import streamlit as st
import pandas as pd
import numpy as np
import io, os
from datetime import datetime
from config import COLORS

def show():
    st.markdown("## 📋 Report Generation")
    if "df" not in st.session_state:
        st.warning("⚠️ Please upload data first."); return

    df  = st.session_state["df"]
    tabs = st.tabs(["📊 Report Preview", "📥 Download Excel", "📄 Download PDF"])

    summary = _build_summary(df)

    with tabs[0]:
        st.markdown("### 📊 Executive Summary")
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Total Revenue",  f"₹{summary['total_revenue']:,.0f}")
        c2.metric("Avg Order",      f"₹{summary['avg_order']:,.0f}")
        c3.metric("Records",        f"{summary['records']:,}")
        c4.metric("Churn Rate",     f"{summary['churn_rate']:.1%}")
        st.divider()
        if "model_metrics" in st.session_state:
            st.markdown("### 🤖 Model Performance")
            mm = st.session_state["model_metrics"]
            cols = st.columns(len(mm))
            for col, (k,v) in zip(cols, mm.items()):
                col.metric(k, f"{v:.4f}" if k=="R²" else f"{v:,.2f}")
        st.markdown("### 🗃️ Raw Data Sample")
        st.dataframe(df.head(50), use_container_width=True)

    with tabs[1]:
        st.markdown("### 📥 Excel Report")
        st.info("Download a multi-sheet Excel workbook with all analysis.")
        if st.button("📦 Generate Excel"):
            buf = _make_excel(df, summary)
            st.download_button("⬇️ Download Excel Report",
                               buf, f"SalesIQ_Report_{_ts()}.xlsx",
                               "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    with tabs[2]:
        st.markdown("### 📄 PDF Report")
        st.info("Download a formatted PDF summary report.")
        if st.button("📄 Generate PDF"):
            buf = _make_pdf(df, summary)
            st.download_button("⬇️ Download PDF Report",
                               buf, f"SalesIQ_Report_{_ts()}.pdf",
                               "application/pdf")


def _build_summary(df):
    return {
        "records":       len(df),
        "total_revenue": df["revenue"].sum() if "revenue" in df.columns else 0,
        "avg_order":     df["revenue"].mean() if "revenue" in df.columns else 0,
        "churn_rate":    df["churn"].mean()   if "churn"   in df.columns else 0,
    }


def _make_excel(df, summary):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Raw Data", index=False)

        summ_df = pd.DataFrame([{
            "Metric":"Total Revenue","Value":f"₹{summary['total_revenue']:,.2f}"},
            {"Metric":"Avg Order","Value":f"₹{summary['avg_order']:,.2f}"},
            {"Metric":"Records","Value":summary["records"]},
            {"Metric":"Churn Rate","Value":f"{summary['churn_rate']:.1%}"},
        ])
        summ_df.to_excel(writer, sheet_name="Summary", index=False)

        if "revenue" in df.columns:
            for cat in df.select_dtypes("object").columns[:3]:
                agg = df.groupby(cat)["revenue"].agg(["sum","mean","count"])
                agg.columns = ["Total Revenue","Avg Revenue","Orders"]
                agg.to_excel(writer, sheet_name=f"By {cat.title()[:12]}")

        if "model_metrics" in st.session_state:
            mm_df = pd.DataFrame([st.session_state["model_metrics"]])
            mm_df.to_excel(writer, sheet_name="Model Metrics", index=False)

    buf.seek(0)
    return buf


def _make_pdf(df, summary):
    from fpdf import FPDF

    pdf = FPDF()
    pdf.add_page()

    # Title
    pdf.set_fill_color(168, 213, 186)
    pdf.rect(0, 0, 210, 35, "F")
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_text_color(45, 49, 66)
    pdf.set_y(10)
    pdf.cell(0, 10, "SalesIQ — Intelligent Sales Report", align="C", ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(107, 114, 128)
    pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", align="C", ln=True)

    pdf.ln(10)
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(45, 49, 66)
    pdf.cell(0, 8, "Executive Summary", ln=True)
    pdf.set_font("Helvetica", "", 11)
    metrics = [
        ("Total Revenue", f"Rs {summary['total_revenue']:,.2f}"),
        ("Avg Order Value", f"Rs {summary['avg_order']:,.2f}"),
        ("Total Records", f"{summary['records']:,}"),
        ("Churn Rate", f"{summary['churn_rate']:.1%}"),
    ]
    for label, value in metrics:
        pdf.set_fill_color(249, 247, 244)
        pdf.cell(80, 9, label, border=1, fill=True)
        pdf.cell(110, 9, value, border=1, ln=True)

    if "model_metrics" in st.session_state:
        pdf.ln(6)
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 8, "Model Performance", ln=True)
        pdf.set_font("Helvetica", "", 11)
        for k, v in st.session_state["model_metrics"].items():
            pdf.set_fill_color(249, 247, 244)
            pdf.cell(80, 9, k, border=1, fill=True)
            pdf.cell(110, 9, f"{v:.4f}" if k == "R2" else f"{v:,.4f}", border=1, ln=True)

    pdf.ln(6)
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 8, "Data Sample (first 10 rows)", ln=True)
    pdf.set_font("Helvetica", "", 7)
    cols = df.columns[:8].tolist()
    col_w = 190 // len(cols)
    for col in cols:
        pdf.cell(col_w, 7, str(col)[:14], border=1, fill=True)
    pdf.ln()
    pdf.set_fill_color(255,255,255)
    for _, row in df.head(10).iterrows():
        for col in cols:
            pdf.cell(col_w, 6, str(row[col])[:14], border=1)
        pdf.ln()

    pdf.set_y(-15)
    pdf.set_font("Helvetica","I", 8)
    pdf.set_text_color(107,114,128)
    pdf.cell(0, 8, "SalesIQ v2.0 — Powered by ML + Streamlit", align="C")

    buf = io.BytesIO(pdf.output())
    return buf


def _ts():
    return datetime.now().strftime("%Y%m%d_%H%M%S")