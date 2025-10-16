import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go
from accrue_tracker.db_utils import get_rates_data, save_forecast, get_forecast_history, init_db
from accrue_tracker.model_selector import select_best_model, generate_forecast
from accrue_tracker.alert_utils import send_model_performance_alert, send_rate_threshold_alert
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Cashramp Market Tracker",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.metric-container {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    padding: 1rem;
    border-radius: 0.5rem;
    color: white;
    text-align: center;
}
.alert-success { background-color: #d4edda; padding: 1rem; border-radius: 0.5rem; border-left: 5px solid #28a745; }
.alert-warning { background-color: #fff3cd; padding: 1rem; border-radius: 0.5rem; border-left: 5px solid #ffc107; color: #856404; }
.alert-danger { background-color: #f8d7da; padding: 1rem; border-radius: 0.5rem; border-left: 5px solid #dc3545; }
</style>
""", unsafe_allow_html=True)

# Sidebar Controls
st.sidebar.header("Model Settings")
models_to_try = st.sidebar.multiselect(
    "Models to Consider",
    ["prophet", "arima"],
    default=["prophet", "arima"],
    help="Select which forecasting models to evaluate"
)
validation_split = st.sidebar.slider(
    "Validation Split",
    0.1, 0.5, 0.2, step=0.05,
    help="Fraction of data to use for model validation"
)
alert_threshold = st.sidebar.number_input(
    "Model Performance Alert (MAPE %)",
    min_value=0.0, value=10.0, step=0.1,
    help="Send alert if model MAPE exceeds this threshold"
)

st.sidebar.header("Market Rate Alerts")
rate_alert_mode = st.sidebar.radio(
    "Trigger Alert When:",
    ["Below Threshold (bad for buying)", "Above Threshold (bad for selling)"],
    index=0,
    help="Choose when to receive rate threshold alerts"
)
rate_alert_threshold = st.sidebar.number_input(
    "Rate Threshold",
    min_value=0.0, value=10.0, step=0.1,
    help="Rate value that triggers threshold alerts"
)

st.sidebar.header("Email Settings")
email_enabled = st.sidebar.checkbox("Enable Email Alerts", value=False)

# SMTP Configuration from secrets
smtp_config = None
alert_recipient = None
alert_sender = None

if email_enabled:
    try:
        smtp_config = {
            "smtp_server": st.secrets["smtp"]["server"],
            "smtp_port": st.secrets["smtp"]["port"],
            "smtp_user": st.secrets["smtp"]["user"],
            "smtp_pass": st.secrets["smtp"]["password"]
        }
        alert_recipient = st.secrets["email"]["recipient"]
        alert_sender = st.secrets["email"]["sender"]
        st.sidebar.success("SMTP configuration loaded from secrets")
    except Exception as e:
        st.sidebar.error(f"Failed to load SMTP secrets: {e}")
        email_enabled = False

# Data Loader
@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_data():
    """Load market rates data from database."""
    return get_rates_data()

@st.cache_resource(ttl=1800)  # Cache models for 30 minutes
def train_and_forecast(df_country, horizon, models_to_try, validation_split):
    """Train models and generate forecast for both deposit and withdrawal rates."""
    try:
        # Train model for deposit rate
        deposit_name, deposit_model, deposit_metrics = select_best_model(
            df_country, models_to_try, validation_split, target_column="depositRate"
        )

        deposit_forecast = generate_forecast(
            deposit_name, deposit_model, horizon, df_country["timestamp"].iloc[-1], column_name="deposit"
        )

        # Train model for withdrawal rate
        withdrawal_name, withdrawal_model, withdrawal_metrics = select_best_model(
            df_country, models_to_try, validation_split, target_column="withdrawalRate"
        )

        withdrawal_forecast = generate_forecast(
            withdrawal_name, withdrawal_model, horizon, df_country["timestamp"].iloc[-1], column_name="withdrawal"
        )

        # Merge forecasts
        forecast_df = deposit_forecast.merge(withdrawal_forecast, on="forecast_time")

        return {
            "deposit": {"name": deposit_name, "model": deposit_model, "metrics": deposit_metrics},
            "withdrawal": {"name": withdrawal_name, "model": withdrawal_model, "metrics": withdrawal_metrics},
            "forecast": forecast_df
        }
    except Exception as e:
        st.error(f"Model training failed: {str(e)}")
        return None

# Main App
# Ensure database tables exist
init_db()

df = load_data()

st.title("Cashramp Market Rates Tracker")
st.markdown("*Self-Tuning Forecasting & Alerting System*")

if df.empty:
    st.warning("""
    **No data available yet.**

    Please run one of the following:
    - `poetry run python seed_data.py` (for testing with sample data)
    - `poetry run python fetch_rates.py` (to fetch real data)
    - Wait for the GitHub Actions workflow to collect data automatically
    """)
    st.stop()

# Country Selection
countries = df["country"].unique()
selected_country = st.selectbox("Select Country", countries, help="Choose country to analyze")
df_country = df[df["country"] == selected_country]

if len(df_country) < 10:
    st.error(f"Insufficient data for {selected_country}. Need at least 10 data points, have {len(df_country)}.")
    st.stop()

# Latest Metrics
latest = df_country.iloc[-1]
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="metric-container">
        <h3>Latest Deposit Rate</h3>
        <h2>{:.4f}</h2>
        <p>{}</p>
    </div>
    """.format(latest['depositRate'], latest['timestamp'].strftime('%Y-%m-%d %H:%M')),
    unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-container">
        <h3>Latest Withdrawal Rate</h3>
        <h2>{:.4f}</h2>
        <p>{}</p>
    </div>
    """.format(latest['withdrawalRate'], latest['timestamp'].strftime('%Y-%m-%d %H:%M')),
    unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-container">
        <h3>Data Points</h3>
        <h2>{}</h2>
        <p>Historical Records</p>
    </div>
    """.format(len(df_country)), unsafe_allow_html=True)

# Historical Rates Chart
st.subheader("Historical Rates")

# Create interactive Plotly chart with initial zoom on last 24 hours
fig = go.Figure()

# Add deposit rate trace
fig.add_trace(go.Scatter(
    x=df_country["timestamp"],
    y=df_country["depositRate"],
    mode='lines',
    name='Deposit Rate',
    line=dict(color='#667eea', width=2)
))

# Add withdrawal rate trace
fig.add_trace(go.Scatter(
    x=df_country["timestamp"],
    y=df_country["withdrawalRate"],
    mode='lines',
    name='Withdrawal Rate',
    line=dict(color='#764ba2', width=2)
))

# Set initial x-axis range to last 24 hours
now = df_country["timestamp"].max()
last_24h = now - timedelta(hours=24)

fig.update_layout(
    xaxis_title="Time",
    yaxis_title="Rate",
    hovermode='x unified',
    xaxis=dict(
        range=[last_24h, now],  # Initial zoom to last 24 hours
        rangeslider=dict(visible=True),  # Add range slider for easy navigation
    ),
    height=500,
    margin=dict(l=0, r=0, t=0, b=0)
)

st.plotly_chart(fig, use_container_width=True)
st.caption(f"Total data points: {len(df_country)} | Chart initially zoomed to last 24 hours - use controls to pan/zoom")

# Forecasting Section
st.subheader("Forecast (Auto Model Selection)")
horizon = st.slider("Forecast Horizon (hours)", 12, 168, 24, help="Number of hours to forecast ahead")

if not models_to_try:
    st.error("Please select at least one model to train.")
    st.stop()

# Train models and generate forecast
with st.spinner("Training models and generating forecast..."):
    result = train_and_forecast(df_country, horizon, models_to_try, validation_split)

if result is None:
    st.stop()

forecast_df = result["forecast"]
deposit_info = result["deposit"]
withdrawal_info = result["withdrawal"]

# Model Performance Display
st.markdown("### Model Performance")
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Deposit Rate Model**")
    subcol1, subcol2, subcol3, subcol4 = st.columns(4)
    with subcol1:
        st.metric("Model", deposit_info["name"].upper())
    with subcol2:
        st.metric("MAE", f"{deposit_info['metrics']['mae']:.4f}")
    with subcol3:
        st.metric("RMSE", f"{deposit_info['metrics']['rmse']:.4f}")
    with subcol4:
        st.metric("MAPE", f"{deposit_info['metrics']['mape']:.2f}%")

with col2:
    st.markdown("**Withdrawal Rate Model**")
    subcol1, subcol2, subcol3, subcol4 = st.columns(4)
    with subcol1:
        st.metric("Model", withdrawal_info["name"].upper())
    with subcol2:
        st.metric("MAE", f"{withdrawal_info['metrics']['mae']:.4f}")
    with subcol3:
        st.metric("RMSE", f"{withdrawal_info['metrics']['rmse']:.4f}")
    with subcol4:
        st.metric("MAPE", f"{withdrawal_info['metrics']['mape']:.2f}%")

# Model Performance Alert
if deposit_info["metrics"]["mape"] > alert_threshold or withdrawal_info["metrics"]["mape"] > alert_threshold:
    st.markdown(f"""
    <div class="alert-danger">
        <strong>Model Performance Alert!</strong><br>
        One or more models are performing poorly (MAPE > {alert_threshold}%).
        Consider retraining with more data or adjusting parameters.
    </div>
    """, unsafe_allow_html=True)

    if email_enabled and smtp_config and alert_recipient and alert_sender:
        # smtp_config already loaded from secrets

        if st.button("Send Performance Alert Email"):
            # Send alert for whichever model is performing poorly
            worst_model = deposit_info if deposit_info["metrics"]["mape"] > withdrawal_info["metrics"]["mape"] else withdrawal_info
            success = send_model_performance_alert(
                worst_model["name"], worst_model["metrics"], alert_threshold,
                alert_recipient, alert_sender, smtp_config
            )
            if success:
                st.success("Performance alert email sent!")
            else:
                st.error("Failed to send performance alert email.")

# Forecast Display
st.subheader("Forecast Results")

# Create interactive Plotly chart for forecast
forecast_fig = go.Figure()

# Add deposit forecast
forecast_fig.add_trace(go.Scatter(
    x=forecast_df["forecast_time"],
    y=forecast_df["predicted_deposit"],
    mode='lines+markers',
    name='Predicted Deposit Rate',
    line=dict(color='#667eea', width=2),
    marker=dict(size=4)
))

# Add withdrawal forecast
forecast_fig.add_trace(go.Scatter(
    x=forecast_df["forecast_time"],
    y=forecast_df["predicted_withdrawal"],
    mode='lines+markers',
    name='Predicted Withdrawal Rate',
    line=dict(color='#764ba2', width=2),
    marker=dict(size=4)
))

forecast_fig.update_layout(
    xaxis_title="Time",
    yaxis_title="Rate",
    hovermode='x unified',
    height=400,
    margin=dict(l=0, r=0, t=0, b=0)
)

st.plotly_chart(forecast_fig, use_container_width=True)

# Save forecast to database
try:
    save_forecast(
        forecast_df,
        selected_country,
        deposit_info["name"],
        withdrawal_info["name"],
        horizon
    )
except Exception as e:
    st.warning(f"Could not save forecast to history: {e}")

# Show Prediction Evolution Over Time
st.subheader("Prediction Evolution")
st.caption("See how predictions have changed over time as new data arrives")

forecast_history = get_forecast_history(selected_country, hours_back=168)

if not forecast_history.empty:
    # Group forecasts by creation time
    creation_times = forecast_history['created_at'].unique()

    if len(creation_times) > 1:
        evolution_fig = go.Figure()

        # Plot each forecast set with decreasing opacity (older = more transparent)
        for i, created_at in enumerate(creation_times[-10:]):  # Show last 10 forecast sets
            forecast_set = forecast_history[forecast_history['created_at'] == created_at]
            opacity = 0.3 + (i / len(creation_times[-10:])) * 0.7  # Gradient from 0.3 to 1.0

            evolution_fig.add_trace(go.Scatter(
                x=forecast_set["forecast_time"],
                y=forecast_set["predicted_deposit"],
                mode='lines',
                name=f'Deposit ({pd.to_datetime(created_at).strftime("%m/%d %H:%M")})',
                line=dict(color='#667eea', width=1),
                opacity=opacity,
                showlegend=i >= len(creation_times[-10:]) - 3  # Only show last 3 in legend
            ))

        # Add actual rates for comparison
        evolution_fig.add_trace(go.Scatter(
            x=df_country["timestamp"],
            y=df_country["depositRate"],
            mode='lines',
            name='Actual Deposit Rate',
            line=dict(color='#28a745', width=2)
        ))

        evolution_fig.update_layout(
            xaxis_title="Time",
            yaxis_title="Deposit Rate",
            hovermode='x unified',
            height=400,
            margin=dict(l=0, r=0, t=0, b=0)
        )

        st.plotly_chart(evolution_fig, use_container_width=True)
        st.caption(f"Showing evolution of predictions from {len(creation_times)} forecast runs")
    else:
        st.info("Not enough forecast history yet. Predictions will accumulate over time.")
else:
    st.info("No forecast history available yet. Check back after a few hours of running.")

# Rate Threshold Alerts
min_rate = forecast_df["predicted_deposit"].min()
max_rate = forecast_df["predicted_deposit"].max()
alert_triggered = False

if rate_alert_mode.startswith("Below") and min_rate < rate_alert_threshold:
    alert_triggered = True
    st.markdown(f"""
    <div class="alert-warning">
        <strong>Rate Threshold Alert!</strong><br>
        Forecast predicts rate dropping below {rate_alert_threshold} in next {horizon}h (Lowest: {min_rate:.4f}).
        This may be unfavorable for buying.
    </div>
    """, unsafe_allow_html=True)

elif rate_alert_mode.startswith("Above") and max_rate > rate_alert_threshold:
    alert_triggered = True
    st.markdown(f"""
    <div class="alert-warning">
        <strong>Rate Threshold Alert!</strong><br>
        Forecast predicts rate rising above {rate_alert_threshold} in next {horizon}h (Highest: {max_rate:.4f}).
        This may be unfavorable for selling.
    </div>
    """, unsafe_allow_html=True)

if alert_triggered and email_enabled and smtp_config and alert_recipient and alert_sender:
    # smtp_config already loaded from secrets

    if st.button("Send Rate Alert Email with CSV"):
        success = send_rate_threshold_alert(
            rate_alert_mode, rate_alert_threshold, min_rate, max_rate, horizon,
            alert_recipient, alert_sender, smtp_config, forecast_df
        )
        if success:
            st.success("Rate threshold alert email sent with CSV attachment!")
        else:
            st.error("Failed to send rate threshold alert email.")

# Forecast Data Download
st.subheader("Download Forecast Data")
csv = forecast_df.to_csv(index=False)
st.download_button(
    label="Download Forecast CSV",
    data=csv,
    file_name=f"cashramp_forecast_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
    mime="text/csv"
)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray;">
    Cashramp Market Rate Tracker v2.0 | Powered by Prophet & ARIMA |
    <a href="https://github.com/your-repo" target="_blank">View Source</a>
</div>
""", unsafe_allow_html=True)