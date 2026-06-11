# modules/forecasting.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from config import CHART_PALETTE

def show():
    st.markdown("## 📈 Sales Forecasting")
    if "df" not in st.session_state:
        st.warning("⚠️ Please upload data first."); return

    df = st.session_state["df"].copy()
    date_col, value_col = _detect_cols(df)

    if not date_col:
        st.error("❌ No date column found. Make sure your data has a 'date' column.")
        return
    if not value_col:
        st.error("❌ No revenue/sales column found.")
        return

    try:
        df[date_col] = pd.to_datetime(df[date_col])
    except:
        st.error("❌ Could not parse date column."); return

    df = df.sort_values(date_col)

    # ── Config ────────────────────────────────────────────────────────────────
    st.sidebar.divider()
    model_choice = st.sidebar.selectbox(
        "Forecast Model",
        ["XGBoost", "Moving Average", "ARIMA", "Prophet"],
        help="XGBoost is fastest and most reliable"
    )
    forecast_horizon = st.sidebar.slider("Forecast Horizon (days)", 7, 180, 30)
    freq = st.sidebar.selectbox(
        "Aggregation", ["D", "W", "ME"],
        format_func=lambda x: {"D":"Daily","W":"Weekly","ME":"Monthly"}[x]
    )

    # ── Build time series ─────────────────────────────────────────────────────
    try:
        ts = df.set_index(date_col)[value_col].resample(freq).sum().reset_index()
        ts.columns = ["ds", "y"]
        ts = ts.dropna()
        ts = ts[ts["y"] > 0]
    except Exception as e:
        st.error(f"❌ Error building time series: {e}"); return

    if len(ts) < 10:
        st.error("❌ Not enough data points. Need at least 10 rows."); return

    # ── Historical chart ──────────────────────────────────────────────────────
    st.markdown(f"### 📅 Historical Sales")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Revenue",   f"₹{ts['y'].sum():,.0f}")
    col2.metric("Avg per Period",  f"₹{ts['y'].mean():,.0f}")
    col3.metric("Data Points",     len(ts))

    fig = px.area(ts, x="ds", y="y",
                  color_discrete_sequence=["#7EC8E3"],
                  title="Historical Sales Trend")
    fig.update_traces(fillcolor="rgba(126,200,227,0.15)")
    fig.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                      xaxis_title="Date", yaxis_title="Revenue (₹)")
    st.plotly_chart(fig, use_container_width=True)

    # ── Run forecast ──────────────────────────────────────────────────────────
    if st.button(f"🚀 Run {model_choice} Forecast", type="primary"):
        with st.spinner(f"⏳ Running {model_choice}..."):
            try:
                if model_choice == "Prophet":
                    forecast_df = _prophet(ts, forecast_horizon, freq)
                elif model_choice == "ARIMA":
                    forecast_df = _arima(ts, forecast_horizon)
                elif model_choice == "XGBoost":
                    forecast_df = _xgboost(ts, forecast_horizon)
                else:
                    forecast_df = _moving_avg(ts, forecast_horizon)

                st.session_state["forecast_df"] = forecast_df
                st.success(f"✅ {model_choice} forecast completed!")
                _plot_forecast(ts, forecast_df, model_choice)
                _show_metrics(ts, forecast_df)

            except Exception as e:
                st.error(f"❌ Forecast failed: {e}")
                st.info("💡 Try selecting **XGBoost** or **Moving Average** instead.")


def _detect_cols(df):
    date_col = None
    for c in df.columns:
        if any(k in c.lower() for k in ["date","time","day","month","year"]):
            date_col = c
            break

    val_col = None
    for c in df.columns:
        if any(k in c.lower() for k in ["revenue","sales","amount","total","price","value"]):
            val_col = c
            break

    return date_col, val_col


def _prophet(ts, horizon, freq):
    try:
        from prophet import Prophet
        import logging
        logging.getLogger("prophet").setLevel(logging.ERROR)
        logging.getLogger("cmdstanpy").setLevel(logging.ERROR)

        m = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            interval_width=0.90,
        )
        m.fit(ts)
        freq_map = {"D": "D", "W": "W", "ME": "ME"}
        future = m.make_future_dataframe(periods=horizon, freq=freq_map[freq])
        fc = m.predict(future)
        fc["model"] = "Prophet"
        return fc[["ds", "yhat", "yhat_lower", "yhat_upper", "model"]]
    except Exception as e:
        st.warning(f"⚠️ Prophet error: {e}. Switching to XGBoost...")
        return _xgboost(ts, horizon)


def _arima(ts, horizon):
    try:
        from statsmodels.tsa.arima.model import ARIMA
        import warnings
        warnings.filterwarnings("ignore")

        model = ARIMA(ts["y"].values, order=(2, 1, 2))
        res   = model.fit()
        fc_vals = res.forecast(steps=horizon)

        future_dates = pd.date_range(
            ts["ds"].iloc[-1], periods=horizon + 1, freq="D"
        )[1:]

        fc_df = pd.DataFrame({"ds": future_dates, "yhat": fc_vals})
        fc_df["yhat_lower"] = fc_df["yhat"] * 0.85
        fc_df["yhat_upper"] = fc_df["yhat"] * 1.15
        fc_df["model"]      = "ARIMA"
        return fc_df

    except Exception as e:
        st.warning(f"⚠️ ARIMA error: {e}. Switching to Moving Average...")
        return _moving_avg(ts, horizon)


def _xgboost(ts, horizon):
    try:
        from xgboost import XGBRegressor

        df2 = ts.copy()
        df2["t"]       = np.arange(len(df2))
        df2["month"]   = pd.to_datetime(df2["ds"]).dt.month
        df2["dayofwk"] = pd.to_datetime(df2["ds"]).dt.dayofweek
        df2["quarter"] = pd.to_datetime(df2["ds"]).dt.quarter

        for lag in [1, 2, 3, 7]:
            df2[f"lag_{lag}"] = df2["y"].shift(lag)

        df2.dropna(inplace=True)

        feats = ["t", "month", "dayofwk", "quarter",
                 "lag_1", "lag_2", "lag_3", "lag_7"]
        X      = df2[feats]
        y_vals = df2["y"]

        model = XGBRegressor(
            n_estimators=200, learning_rate=0.05,
            random_state=42, verbosity=0
        )
        model.fit(X, y_vals)

        preds, dates = [], []
        last_row = df2[feats].iloc[-1].values.copy().astype(float)

        for i in range(horizon):
            pred = float(model.predict(last_row.reshape(1, -1))[0])
            pred = max(0, pred)
            preds.append(pred)
            next_date = ts["ds"].iloc[-1] + pd.Timedelta(days=i + 1)
            dates.append(next_date)
            # update lag features
            last_row[0] += 1
            last_row[1]  = next_date.month
            last_row[2]  = next_date.dayofweek
            last_row[3]  = next_date.quarter
            last_row[7]  = last_row[6]
            last_row[6]  = last_row[5]
            last_row[5]  = last_row[4]
            last_row[4]  = pred

        fc_df = pd.DataFrame({"ds": dates, "yhat": preds})
        fc_df["yhat_lower"] = fc_df["yhat"] * 0.88
        fc_df["yhat_upper"] = fc_df["yhat"] * 1.12
        fc_df["model"]      = "XGBoost"
        return fc_df

    except Exception as e:
        st.warning(f"⚠️ XGBoost error: {e}. Switching to Moving Average...")
        return _moving_avg(ts, horizon)


def _moving_avg(ts, horizon):
    window  = min(14, len(ts))
    avg     = ts["y"].rolling(window).mean().iloc[-1]
    trend   = (ts["y"].iloc[-1] - ts["y"].iloc[-window]) / window
    dates   = pd.date_range(ts["ds"].iloc[-1], periods=horizon + 1, freq="D")[1:]

    preds = [max(0, avg + trend * i) for i in range(horizon)]

    fc_df = pd.DataFrame({"ds": dates, "yhat": preds})
    fc_df["yhat_lower"] = fc_df["yhat"] * 0.90
    fc_df["yhat_upper"] = fc_df["yhat"] * 1.10
    fc_df["model"]      = "Moving Average"
    return fc_df


def _plot_forecast(ts, fc, model_name):
    st.markdown(f"### 🔮 {model_name} Forecast Results")

    fig = go.Figure()

    # Historical
    fig.add_trace(go.Scatter(
        x=ts["ds"], y=ts["y"],
        name="Historical",
        line=dict(color="#7EC8E3", width=2.5)
    ))

    # Confidence band
    if "yhat_upper" in fc.columns:
        fig.add_trace(go.Scatter(
            x=pd.concat([fc["ds"], fc["ds"][::-1]]),
            y=pd.concat([fc["yhat_upper"], fc["yhat_lower"][::-1]]),
            fill="toself",
            fillcolor="rgba(168,213,186,0.2)",
            line=dict(color="rgba(255,255,255,0)"),
            name="Confidence Band"
        ))

    # Forecast line
    fig.add_trace(go.Scatter(
        x=fc["ds"], y=fc["yhat"],
        name="Forecast",
        line=dict(color="#A8D5BA", width=2.5, dash="dash")
    ))

    fig.update_layout(
        title=f"{model_name} — Sales Forecast",
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis_title="Date",
        yaxis_title="Revenue (₹)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Forecast table
    st.markdown("### 📋 Forecast Data Table")
    display_fc = fc[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    display_fc.columns = ["Date", "Forecasted Revenue", "Lower Bound", "Upper Bound"]
    display_fc["Date"] = display_fc["Date"].dt.strftime("%Y-%m-%d")
    for col in ["Forecasted Revenue", "Lower Bound", "Upper Bound"]:
        display_fc[col] = display_fc[col].apply(lambda x: f"₹{x:,.2f}")
    st.dataframe(display_fc, use_container_width=True)

    # Download forecast
    csv = fc.to_csv(index=False).encode()
    st.download_button(
        "⬇️ Download Forecast CSV",
        csv,
        "forecast_results.csv",
        "text/csv"
    )


def _show_metrics(ts, fc):
    st.markdown("### 📊 Forecast Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avg Daily Forecast",  f"₹{fc['yhat'].mean():,.0f}")
    c2.metric("Peak Forecast",       f"₹{fc['yhat'].max():,.0f}")
    c3.metric("Total Forecast",      f"₹{fc['yhat'].sum():,.0f}")
    c4.metric("Forecast Days",       len(fc))