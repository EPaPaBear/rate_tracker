import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from accrue_tracker.db_utils import get_rates_data
from accrue_tracker.model_selector import select_best_model, generate_forecast
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Model Performance Analysis",
    layout="wide"
)

st.title("Model Performance Analysis")
st.markdown("*Detailed analysis of forecast accuracy and model diagnostics*")

# Load data
@st.cache_data(ttl=300)
def load_data():
    return get_rates_data()

df = load_data()

if df.empty:
    st.warning("No data available. Please ensure the database has been populated.")
    st.stop()

# Sidebar controls
st.sidebar.header("Analysis Settings")
countries = df["country"].unique()
selected_country = st.sidebar.selectbox("Select Country", countries)
df_country = df[df["country"] == selected_country]

models_to_analyze = st.sidebar.multiselect(
    "Models to Analyze",
    ["prophet", "arima"],
    default=["prophet", "arima"]
)

validation_split = st.sidebar.slider("Validation Split", 0.1, 0.5, 0.2, step=0.05)

# Check if we have enough data
if len(df_country) < 20:
    st.error(f"Insufficient data for analysis. Need at least 20 data points, have {len(df_country)}.")
    st.stop()

# Model Comparison Section
st.header("Model Comparison")

if len(models_to_analyze) == 0:
    st.warning("Please select at least one model to analyze.")
    st.stop()

# Train models and collect metrics
@st.cache_resource(ttl=1800)
def analyze_models(df_country, models_to_analyze, validation_split):
    results = {}

    for model_name in models_to_analyze:
        try:
            chosen_name, model, metrics = select_best_model(
                df_country, [model_name], validation_split
            )
            results[model_name] = {
                "model": model,
                "metrics": metrics,
                "name": chosen_name
            }
        except Exception as e:
            st.error(f"Failed to train {model_name}: {str(e)}")

    return results

with st.spinner("Training and analyzing models..."):
    model_results = analyze_models(df_country, models_to_analyze, validation_split)

if not model_results:
    st.error("No models could be trained successfully.")
    st.stop()

# Metrics Comparison Table
st.subheader("Performance Metrics Comparison")

metrics_data = []
for model_name, result in model_results.items():
    metrics = result["metrics"]
    metrics_data.append({
        "Model": model_name.upper(),
        "MAE": f"{metrics['mae']:.4f}",
        "RMSE": f"{metrics['rmse']:.4f}",
        "MAPE (%)": f"{metrics['mape']:.2f}%"
    })

metrics_df = pd.DataFrame(metrics_data)
st.dataframe(metrics_df)

# Best model identification
best_model_name = min(model_results.keys(), key=lambda k: model_results[k]["metrics"]["mae"])
st.success(f"Best performing model: **{best_model_name.upper()}** (lowest MAE)")

# Forecast vs Actual Comparison
st.header("Forecast vs Actual Comparison")

# Generate forecasts for visualization
split_idx = int(len(df_country) * (1 - validation_split))
train_data = df_country.iloc[:split_idx]
actual_data = df_country.iloc[split_idx:]

forecast_horizon = len(actual_data)

if forecast_horizon > 0:
    # Create comparison chart
    fig, axes = plt.subplots(len(model_results), 1, figsize=(12, 6 * len(model_results)))
    if len(model_results) == 1:
        axes = [axes]

    for idx, (model_name, result) in enumerate(model_results.items()):
        model = result["model"]

        try:
            # Generate forecast for validation period
            forecast_df = generate_forecast(
                model_name, model, forecast_horizon,
                train_data["timestamp"].iloc[-1]
            )

            # Plot comparison
            ax = axes[idx]

            # Plot historical data
            ax.plot(train_data["timestamp"], train_data["depositRate"],
                   label="Training Data", color="blue", alpha=0.7)

            # Plot actual validation data
            ax.plot(actual_data["timestamp"], actual_data["depositRate"],
                   label="Actual", color="green", linewidth=2)

            # Plot forecast
            ax.plot(forecast_df["forecast_time"], forecast_df["predicted_deposit"],
                   label="Forecast", color="red", linestyle="--", linewidth=2)

            ax.set_title(f"{model_name.upper()} Model - Forecast vs Actual")
            ax.set_xlabel("Time")
            ax.set_ylabel("Deposit Rate")
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Add vertical line at train/validation split
            ax.axvline(x=train_data["timestamp"].iloc[-1], color="orange",
                      linestyle=":", alpha=0.7, label="Train/Validation Split")

        except Exception as e:
            st.error(f"Failed to generate forecast comparison for {model_name}: {str(e)}")

    plt.tight_layout()
    st.pyplot(fig)

# Residual Analysis
st.header("Residual Analysis")

residual_data = {}
for model_name, result in model_results.items():
    model = result["model"]

    try:
        # Generate predictions for validation period
        forecast_df = generate_forecast(
            model_name, model, forecast_horizon,
            train_data["timestamp"].iloc[-1]
        )

        # Calculate residuals
        if len(forecast_df) == len(actual_data):
            residuals = actual_data["depositRate"].values - forecast_df["predicted_deposit"].values
            residual_data[model_name] = residuals

    except Exception as e:
        st.warning(f"Could not calculate residuals for {model_name}: {str(e)}")

if residual_data:
    # Residual plots
    fig, axes = plt.subplots(2, len(residual_data), figsize=(6 * len(residual_data), 10))
    if len(residual_data) == 1:
        axes = axes.reshape(-1, 1)

    for idx, (model_name, residuals) in enumerate(residual_data.items()):
        # Residuals over time
        axes[0, idx].plot(range(len(residuals)), residuals, 'bo-', alpha=0.7)
        axes[0, idx].axhline(y=0, color='red', linestyle='--')
        axes[0, idx].set_title(f"{model_name.upper()} - Residuals Over Time")
        axes[0, idx].set_xlabel("Time Step")
        axes[0, idx].set_ylabel("Residual")
        axes[0, idx].grid(True, alpha=0.3)

        # Residual distribution
        axes[1, idx].hist(residuals, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
        axes[1, idx].axvline(x=np.mean(residuals), color='red', linestyle='--',
                            label=f'Mean: {np.mean(residuals):.4f}')
        axes[1, idx].set_title(f"{model_name.upper()} - Residual Distribution")
        axes[1, idx].set_xlabel("Residual Value")
        axes[1, idx].set_ylabel("Frequency")
        axes[1, idx].legend()
        axes[1, idx].grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)

    # Residual statistics
    st.subheader("Residual Statistics")
    residual_stats = []
    for model_name, residuals in residual_data.items():
        residual_stats.append({
            "Model": model_name.upper(),
            "Mean": f"{np.mean(residuals):.6f}",
            "Std Dev": f"{np.std(residuals):.6f}",
            "Min": f"{np.min(residuals):.4f}",
            "Max": f"{np.max(residuals):.4f}"
        })

    residual_stats_df = pd.DataFrame(residual_stats)
    st.dataframe(residual_stats_df)

# Feature Importance (for Prophet)
st.header("Model Insights")

prophet_models = [name for name, result in model_results.items() if name == "prophet"]
if prophet_models:
    st.subheader("Prophet Model Components")
    model_name = prophet_models[0]
    model = model_results[model_name]["model"]

    try:
        # Generate forecast with components
        future = model.make_future_dataframe(periods=24, freq="H")
        forecast = model.predict(future)

        # Plot components
        fig = model.plot_components(forecast)
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Failed to plot Prophet components: {str(e)}")

# Model Recommendations
st.header("Recommendations")

if model_results:
    best_metrics = model_results[best_model_name]["metrics"]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Current Best Model")
        st.info(f"""
        **Model**: {best_model_name.upper()}
        **MAE**: {best_metrics['mae']:.4f}
        **MAPE**: {best_metrics['mape']:.2f}%
        """)

    with col2:
        st.subheader("Improvement Suggestions")

        suggestions = []

        if best_metrics['mape'] > 15:
            suggestions.append("High MAPE (>15%) - Consider collecting more data or feature engineering")
        elif best_metrics['mape'] > 10:
            suggestions.append("Moderate MAPE (10-15%) - Model performance is acceptable but can be improved")
        else:
            suggestions.append("Good MAPE (<10%) - Model is performing well")

        if len(df_country) < 100:
            suggestions.append("Increase data collection frequency for better model training")

        if len(model_results) == 1:
            suggestions.append("Try additional models (Prophet, ARIMA) for comparison")

        for suggestion in suggestions:
            st.write(suggestion)

# Export Results
st.header("Export Analysis")

if st.button("Generate Analysis Report"):
    report_data = {
        "analysis_timestamp": pd.Timestamp.now(),
        "country": selected_country,
        "data_points": len(df_country),
        "validation_split": validation_split,
        "best_model": best_model_name,
        "best_metrics": model_results[best_model_name]["metrics"],
        "all_results": {name: result["metrics"] for name, result in model_results.items()}
    }

    st.json(report_data)

    # Prepare CSV for download
    export_data = []
    for model_name, result in model_results.items():
        metrics = result["metrics"]
        export_data.append({
            "Model": model_name,
            "MAE": metrics["mae"],
            "RMSE": metrics["rmse"],
            "MAPE": metrics["mape"]
        })

    export_df = pd.DataFrame(export_data)
    csv = export_df.to_csv(index=False)

    st.download_button(
        label="Download Performance Report CSV",
        data=csv,
        file_name=f"model_performance_{selected_country}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv"
    )