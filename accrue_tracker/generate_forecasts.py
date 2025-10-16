"""
Automated forecast generation script for scheduled execution.
Generates predictions for all countries and saves them to the database.
"""

import sys
from datetime import datetime, UTC
from .db_utils import get_rates_data, save_forecast, init_db
from .model_selector import select_best_model, generate_forecast
import warnings
warnings.filterwarnings('ignore')

def generate_forecasts_for_country(country="GH", horizon_hours=24, models_to_try=["prophet", "arima"]):
    """Generate and save forecasts for a specific country."""
    print(f"\n{'='*60}")
    print(f"Generating forecasts for {country}")
    print(f"{'='*60}")

    # Load data
    df = get_rates_data(country)

    if df.empty or len(df) < 10:
        print(f"Insufficient data for {country}. Need at least 10 data points, have {len(df)}.")
        return False

    print(f"Loaded {len(df)} data points for {country}")
    print(f"Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    try:
        # Train model for deposit rate
        print("\nTraining deposit rate model...")
        deposit_name, deposit_model, deposit_metrics = select_best_model(
            df, models_to_try, validation_split=0.2, target_column="depositRate"
        )
        print(f"  Selected: {deposit_name.upper()}")
        print(f"  MAPE: {deposit_metrics['mape']:.2f}%")

        deposit_forecast = generate_forecast(
            deposit_name, deposit_model, horizon_hours, df["timestamp"].iloc[-1], column_name="deposit"
        )

        # Train model for withdrawal rate
        print("\nTraining withdrawal rate model...")
        withdrawal_name, withdrawal_model, withdrawal_metrics = select_best_model(
            df, models_to_try, validation_split=0.2, target_column="withdrawalRate"
        )
        print(f"  Selected: {withdrawal_name.upper()}")
        print(f"  MAPE: {withdrawal_metrics['mape']:.2f}%")

        withdrawal_forecast = generate_forecast(
            withdrawal_name, withdrawal_model, horizon_hours, df["timestamp"].iloc[-1], column_name="withdrawal"
        )

        # Merge forecasts
        forecast_df = deposit_forecast.merge(withdrawal_forecast, on="forecast_time")

        # Save to database
        print(f"\nSaving {len(forecast_df)} forecast points to database...")
        save_forecast(
            forecast_df,
            country,
            deposit_name,
            withdrawal_name,
            horizon_hours
        )

        print(f"\nForecast Summary:")
        print(f"  Deposit Rate:    {forecast_df['predicted_deposit'].min():.4f} - {forecast_df['predicted_deposit'].max():.4f}")
        print(f"  Withdrawal Rate: {forecast_df['predicted_withdrawal'].min():.4f} - {forecast_df['predicted_withdrawal'].max():.4f}")
        print(f"  Forecast horizon: {horizon_hours} hours")
        print(f"\n[SUCCESS] Forecast generation completed successfully for {country}")

        return True

    except Exception as e:
        print(f"\n[ERROR] Error generating forecast for {country}: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function to generate forecasts for all countries."""
    print("\n" + "="*60)
    print(f"Automated Forecast Generation")
    print(f"Timestamp: {datetime.now(UTC).isoformat()}")
    print("="*60)

    # Initialize database (ensure forecasts table exists)
    init_db()
    print("Database initialized")

    # Configuration
    countries = ["GH"]  # Add more countries as needed
    horizon_hours = 24
    models_to_try = ["prophet", "arima"]

    # Generate forecasts for each country
    results = {}
    for country in countries:
        success = generate_forecasts_for_country(
            country=country,
            horizon_hours=horizon_hours,
            models_to_try=models_to_try
        )
        results[country] = success

    # Summary
    print("\n" + "="*60)
    print("Forecast Generation Summary")
    print("="*60)
    successful = sum(1 for v in results.values() if v)
    print(f"Successful: {successful}/{len(results)}")
    for country, success in results.items():
        status = "[OK]" if success else "[FAIL]"
        print(f"  {status} {country}")
    print("="*60 + "\n")

    # Exit with appropriate code
    if successful == len(results):
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()
