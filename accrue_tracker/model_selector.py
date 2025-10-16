import pandas as pd
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import logging
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(filename="model_selection.log", level=logging.INFO, format='%(asctime)s - %(message)s')

def evaluate_forecast(y_true, y_pred):
    """Calculate forecast evaluation metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)  # Calculate RMSE manually

    # Handle division by zero in MAPE calculation
    mask = y_true != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = 0.0

    return {"mae": mae, "rmse": rmse, "mape": mape}

def select_best_model(df, models_to_try=["prophet", "arima"], validation_split=0.2, target_column="depositRate"):
    """
    Train multiple models and select the best one based on validation metrics.

    Args:
        df: DataFrame with 'timestamp' and rate columns
        models_to_try: List of model names to evaluate
        validation_split: Fraction of data to use for validation
        target_column: Column to forecast ('depositRate' or 'withdrawalRate')

    Returns:
        tuple: (best_model_name, best_model, best_metrics)
    """
    if df.empty or len(df) < 10:
        raise ValueError("Insufficient data for model training (need at least 10 data points)")

    df = df.copy()

    # Ensure we have the right column names
    if 'timestamp' in df.columns and target_column in df.columns:
        df = df.rename(columns={"timestamp": "ds", target_column: "y"})
    elif 'ds' not in df.columns or 'y' not in df.columns:
        raise ValueError(f"DataFrame must have 'timestamp' and '{target_column}' columns or 'ds' and 'y' columns")

    df = df.sort_values("ds")

    # Convert datetime to timezone-naive (Prophet requirement)
    if df['ds'].dtype == 'datetime64[ns, UTC]' or str(df['ds'].dtype).endswith('+00:00]'):
        df['ds'] = df['ds'].dt.tz_localize(None)

    # Ensure we have valid data
    df = df.dropna(subset=['y'])
    if len(df) < 10:
        raise ValueError("Insufficient valid data after cleaning (need at least 10 data points)")

    split_idx = int(len(df) * (1 - validation_split))
    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]

    if len(val_df) == 0:
        raise ValueError("Validation set is empty. Increase dataset size or reduce validation_split.")

    results = {}

    # Train Prophet model
    if "prophet" in models_to_try:
        try:
            m = Prophet(
                daily_seasonality=False,
                weekly_seasonality=True,
                yearly_seasonality=False,
                interval_width=0.95
            )
            # Suppress Prophet's verbose output
            import logging as prophet_logging
            prophet_logging.getLogger('prophet').setLevel(prophet_logging.WARNING)

            m.fit(train_df)

            # Create future dataframe for validation period
            last_date = train_df['ds'].max()
            future_dates = pd.date_range(start=last_date, periods=len(val_df)+1, freq="H")[1:]
            future_df = pd.DataFrame({'ds': future_dates})

            forecast = m.predict(future_df)
            y_pred = forecast["yhat"].values

            metrics = evaluate_forecast(val_df["y"].values, y_pred)

            results["prophet"] = {
                "model": m,
                "metrics": metrics
            }
            logging.info(f"Prophet model trained successfully. MAE: {results['prophet']['metrics']['mae']:.4f}")
        except Exception as e:
            logging.error(f"Prophet training failed: {e}")

    # Train ARIMA model
    if "arima" in models_to_try:
        try:
            # Use simple ARIMA orders to avoid complex fitting
            arima_orders = [(1,1,1), (2,1,2), (1,1,0), (0,1,1), (1,0,1)]
            best_arima = None
            best_aic = float('inf')

            for order in arima_orders:
                try:
                    arima_model = ARIMA(train_df["y"], order=order)
                    arima_fit = arima_model.fit()
                    if arima_fit.aic < best_aic:
                        best_aic = arima_fit.aic
                        best_arima = arima_fit
                except:
                    continue

            if best_arima is not None:
                forecast = best_arima.forecast(steps=len(val_df))

                results["arima"] = {
                    "model": best_arima,
                    "metrics": evaluate_forecast(val_df["y"].values, forecast.values)
                }
                logging.info(f"ARIMA model trained successfully. MAE: {results['arima']['metrics']['mae']:.4f}")
            else:
                logging.error("ARIMA: No suitable order found")

        except Exception as e:
            logging.error(f"ARIMA training failed: {e}")

    if not results:
        raise RuntimeError("No models could be trained successfully. Check your data quality.")

    # Select best model based on MAE
    best_model_name = min(results, key=lambda k: results[k]["metrics"]["mae"])
    best_model = results[best_model_name]["model"]
    best_metrics = results[best_model_name]["metrics"]

    logging.info(f"Best model selected: {best_model_name.upper()}, Metrics: {best_metrics}")
    return best_model_name, best_model, best_metrics

def generate_forecast(model_name, model, horizon_hours=24, last_timestamp=None, column_name="deposit"):
    """
    Generate forecast using the trained model.

    Args:
        model_name: 'prophet' or 'arima'
        model: Trained model object
        horizon_hours: Number of hours to forecast
        last_timestamp: Last timestamp in the training data
        column_name: Name for the prediction column ('deposit' or 'withdrawal')

    Returns:
        DataFrame with forecast_time and predicted_{column_name} columns
    """
    if model_name == "prophet":
        # Create future dates starting from the last timestamp
        if last_timestamp is not None:
            # Remove timezone if present
            if hasattr(last_timestamp, 'tz_localize'):
                last_timestamp = last_timestamp.tz_localize(None)
            future_dates = pd.date_range(start=last_timestamp, periods=horizon_hours+1, freq="H")[1:]
        else:
            future_dates = pd.date_range(start=pd.Timestamp.now(), periods=horizon_hours, freq="H")

        future_df = pd.DataFrame({'ds': future_dates})
        forecast = model.predict(future_df)
        forecast_df = forecast[["ds", "yhat"]].rename(
            columns={"ds": "forecast_time", "yhat": f"predicted_{column_name}"}
        )

    elif model_name == "arima":
        forecast_values = model.forecast(steps=horizon_hours)
        if last_timestamp is not None:
            forecast_times = pd.date_range(
                start=last_timestamp,
                periods=horizon_hours + 1,
                freq="H"
            )[1:]  # Exclude the start time
        else:
            forecast_times = pd.date_range(
                start=pd.Timestamp.now(),
                periods=horizon_hours,
                freq="H"
            )

        forecast_df = pd.DataFrame({
            "forecast_time": forecast_times,
            f"predicted_{column_name}": forecast_values
        })
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    return forecast_df