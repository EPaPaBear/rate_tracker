import streamlit as st
import pandas as pd
from datetime import datetime
from db_utils import get_rates_data
from model_selector import select_best_model, generate_forecast
from alert_utils import send_model_performance_alert, send_rate_threshold_alert
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
    """Train models and generate forecast."""
    try:
        chosen_name, model, metrics = select_best_model(
            df_country, models_to_try, validation_split
        )

        forecast_df = generate_forecast(
            chosen_name, model, horizon, df_country["timestamp"].iloc[-1]
        )

        return chosen_name, model, metrics, forecast_df
    except Exception as e:
        st.error(f"Model training failed: {str(e)}")
        return None, None, None, None

# Main App
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
chart_data = df_country.set_index("timestamp")[["depositRate", "withdrawalRate"]]
st.line_chart(chart_data)

# Forecasting Section
st.subheader("Forecast (Auto Model Selection)")
horizon = st.slider("Forecast Horizon (hours)", 12, 168, 24, help="Number of hours to forecast ahead")

if not models_to_try:
    st.error("Please select at least one model to train.")
    st.stop()

# Train models and generate forecast
with st.spinner("Training models and generating forecast..."):
    result = train_and_forecast(df_country, horizon, models_to_try, validation_split)

if result[0] is None:
    st.stop()

chosen_name, model, metrics, forecast_df = result

# Model Performance Display
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Selected Model", chosen_name.upper())
with col2:
    st.metric("MAE", f"{metrics['mae']:.4f}")
with col3:
    st.metric("RMSE", f"{metrics['rmse']:.4f}")
with col4:
    st.metric("MAPE", f"{metrics['mape']:.2f}%")

# Model Performance Alert
if metrics["mape"] > alert_threshold:
    st.markdown(f"""
    <div class="alert-danger">
        <strong>Model Performance Alert!</strong><br>
        The {chosen_name.upper()} model is performing poorly (MAPE: {metrics['mape']:.2f}% > {alert_threshold}%).
        Consider retraining with more data or adjusting parameters.
    </div>
    """, unsafe_allow_html=True)

    if email_enabled and smtp_config and alert_recipient and alert_sender:
        # smtp_config already loaded from secrets

        if st.button("Send Performance Alert Email"):
            success = send_model_performance_alert(
                chosen_name, metrics, alert_threshold,
                alert_recipient, alert_sender, smtp_config
            )
            if success:
                st.success("Performance alert email sent!")
            else:
                st.error("Failed to send performance alert email.")

# Forecast Display
st.subheader("Forecast Results")
st.line_chart(forecast_df.set_index("forecast_time")[["predicted_deposit"]])

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