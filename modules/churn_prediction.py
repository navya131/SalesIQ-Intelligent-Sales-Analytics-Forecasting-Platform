# modules/churn_prediction.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_curve, auc, accuracy_score)
from sklearn.preprocessing import LabelEncoder
from config import CHART_PALETTE

def show():
    st.markdown("## 🔄 Churn Prediction")
    if "df" not in st.session_state:
        st.warning("⚠️ Please upload data first."); return

    df = st.session_state["df"].copy()

    # ── Config ────────────────────────────────────────────────────────────────
    st.markdown("### ⚙️ Configuration")
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    all_cols = df.columns.tolist()

    col1, col2, col3 = st.columns(3)
    churn_col = col1.selectbox("Churn column (0/1)",
        [c for c in all_cols if "churn" in c.lower() or df[c].nunique() == 2]
        or num_cols[:1])
    model_name = col2.selectbox("Classifier",
        ["Random Forest", "Gradient Boosting", "Logistic Regression"])
    test_size  = col3.slider("Test split %", 10, 40, 20)

    feature_cols = st.multiselect("Feature columns",
        [c for c in num_cols if c != churn_col],
        default=[c for c in num_cols if c != churn_col][:6])

    if st.button("🚀 Train Churn Model") and feature_cols:
        df_enc = _encode(df.copy())
        X = df_enc[feature_cols].fillna(0)
        y = df_enc[churn_col].fillna(0).astype(int)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size/100, random_state=42, stratify=y)

        clf = _get_clf(model_name)
        with st.spinner("Training classifier…"):
            clf.fit(X_train, y_train)

        y_pred  = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:,1] if hasattr(clf, "predict_proba") else y_pred

        acc = accuracy_score(y_test, y_pred)
        st.success(f"✅ {model_name} trained — Accuracy: **{acc:.2%}**")
        st.session_state["churn_model"]    = clf
        st.session_state["churn_features"] = feature_cols

        # ── Metrics ───────────────────────────────────────────────────────────
        c1, c2, c3, c4 = st.columns(4)
        from sklearn.metrics import precision_score, recall_score, f1_score
        c1.metric("Accuracy",  f"{acc:.2%}")
        c2.metric("Precision", f"{precision_score(y_test,y_pred,zero_division=0):.2%}")
        c3.metric("Recall",    f"{recall_score(y_test,y_pred,zero_division=0):.2%}")
        c4.metric("F1 Score",  f"{f1_score(y_test,y_pred,zero_division=0):.2%}")

        tabs = st.tabs(["📋 Predictions Table", "🥧 Churn Distribution",
                         "📊 Confusion Matrix", "📈 ROC Curve", "🎯 Feature Importance"])

        # ── Tab 1: Prediction Table ────────────────────────────────────────────
        with tabs[0]:
            pred_df = X_test.copy()
            pred_df["Actual Churn"]      = y_test.values
            pred_df["Predicted Churn"]   = y_pred
            pred_df["Churn Probability"] = (y_proba * 100).round(2)
            pred_df["Risk Level"]        = pd.cut(pred_df["Churn Probability"],
                bins=[0,30,60,100], labels=["🟢 Low","🟡 Medium","🔴 High"])
            st.dataframe(pred_df.reset_index(drop=True), use_container_width=True)

            csv = pred_df.to_csv(index=False).encode()
            st.download_button("⬇️ Download Predictions CSV", csv,
                               "churn_predictions.csv", "text/csv")

        # ── Tab 2: Churn Distribution ──────────────────────────────────────────
        with tabs[1]:
            c1, c2 = st.columns(2)
            with c1:
                cnt = pd.Series(y_pred).value_counts().reset_index()
                cnt.columns = ["Churn","Count"]
                cnt["Churn"] = cnt["Churn"].map({0:"Not Churned",1:"Churned"})
                fig = px.pie(cnt, names="Churn", values="Count",
                             color_discrete_sequence=["#A8D5BA","#F7C5CC"],
                             title="Predicted Churn Distribution", hole=0.45)
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                risk_cnt = pred_df["Risk Level"].value_counts().reset_index()
                risk_cnt.columns = ["Risk","Count"]
                fig2 = px.bar(risk_cnt, x="Risk", y="Count",
                              color="Risk",
                              color_discrete_map={
                                  "🟢 Low":"#A8D5BA",
                                  "🟡 Medium":"#FFE5A0",
                                  "🔴 High":"#F7C5CC"},
                              title="Risk Tier Breakdown")
                fig2.update_layout(plot_bgcolor="white",paper_bgcolor="white",showlegend=False)
                st.plotly_chart(fig2, use_container_width=True)

        # ── Tab 3: Confusion Matrix ────────────────────────────────────────────
        with tabs[2]:
            cm = confusion_matrix(y_test, y_pred)
            fig = px.imshow(cm, text_auto=True,
                            labels=dict(x="Predicted", y="Actual",
                                        color="Count"),
                            x=["Not Churned","Churned"],
                            y=["Not Churned","Churned"],
                            color_continuous_scale="mint",
                            title="Confusion Matrix")
            fig.update_layout(paper_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True)

        # ── Tab 4: ROC Curve ───────────────────────────────────────────────────
        with tabs[3]:
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f"AUC = {roc_auc:.3f}",
                line=dict(color="#B5C8E8", width=2.5)))
            fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                line=dict(dash="dash", color="#D1D5DB"))
            fig.update_layout(title="ROC Curve", xaxis_title="FPR",
                              yaxis_title="TPR", plot_bgcolor="white",
                              paper_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True)

        # ── Tab 5: Feature Importance ──────────────────────────────────────────
        with tabs[4]:
            if hasattr(clf, "feature_importances_"):
                imp = pd.DataFrame({"Feature": feature_cols,
                                    "Importance": clf.feature_importances_}
                                   ).sort_values("Importance", ascending=True)
                fig = px.bar(imp, x="Importance", y="Feature", orientation="h",
                             color="Importance", color_continuous_scale="teal",
                             title="Churn Feature Importances")
                fig.update_layout(plot_bgcolor="white", paper_bgcolor="white")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Feature importance not available for this model.")


def _encode(df):
    for col in df.select_dtypes(include="object").columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    return df

def _get_clf(name):
    return {
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting":   GradientBoostingClassifier(n_estimators=100, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=300, random_state=42),
    }[name]