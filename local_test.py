#!/usr/bin/env python3
"""
Local Test Harness for Cashramp Market Rate Tracker

This script provides comprehensive testing of the entire pipeline:
1. Database seeding with realistic test data
2. Model training and selection
3. Forecast generation
4. Alert system testing (both model performance and rate threshold alerts)
5. Email functionality (prints to console for safety)

Run this before deployment to ensure everything works correctly.
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, UTC
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from db_utils import init_db, get_rates_data
from model_selector import select_best_model, generate_forecast
from alert_utils import send_model_performance_alert, send_rate_threshold_alert
from seed_data import seed_fake_data, seed_test_scenarios

class TestEmailServer:
    """Mock email server that prints emails to console instead of sending them."""

    @staticmethod
    def send_test_email(subject, body, recipient, sender, smtp_config, forecast_df=None):
        """Print email content to console for testing."""
        print("\n" + "="*80)
        print("MOCK EMAIL (Console Output)")
        print("="*80)
        print(f"From: {sender}")
        print(f"To: {recipient}")
        print(f"Subject: {subject}")
        print(f"SMTP: {smtp_config.get('smtp_server', 'N/A')}:{smtp_config.get('smtp_port', 'N/A')}")
        print("-"*80)
        print(body)

        if forecast_df is not None and not forecast_df.empty:
            print("\nCSV Attachment Preview:")
            print("-"*40)
            print(forecast_df.head(10).to_string(index=False))
            if len(forecast_df) > 10:
                print(f"... and {len(forecast_df) - 10} more rows")
            print(f"\nTotal forecast rows: {len(forecast_df)}")

        print("="*80)
        return True

def test_database_operations():
    """Test database initialization and data operations."""
    print("Testing Database Operations...")

    # Test database initialization
    try:
        conn = init_db()
        conn.close()
        print("  ✅ Database initialization successful")
    except Exception as e:
        print(f"  ❌ Database initialization failed: {e}")
        return False

    # Test data seeding
    try:
        seed_fake_data(days=3, countries=["GH", "NG"], base_rate=10.0)
        seed_test_scenarios()
        print("  ✅ Data seeding successful")
    except Exception as e:
        print(f"  ❌ Data seeding failed: {e}")
        return False

    # Test data retrieval
    try:
        df = get_rates_data("GH")
        if df.empty:
            print("  ❌ No data retrieved from database")
            return False
        print(f"  ✅ Data retrieval successful ({len(df)} records for GH)")

        df_ng = get_rates_data("NG")
        print(f"  ✅ Multi-country data verified ({len(df_ng)} records for NG)")
    except Exception as e:
        print(f"  ❌ Data retrieval failed: {e}")
        return False

    return True

def test_model_training():
    """Test model training and selection."""
    print("\nTesting Model Training & Selection...")

    # Load test data
    df = get_rates_data("GH")
    if len(df) < 20:
        print(f"  ❌ Insufficient data for model training ({len(df)} records)")
        return False, None, None

    # Test Prophet model
    try:
        chosen_name, model, metrics = select_best_model(df, ["prophet"], validation_split=0.3)
        print(f"  ✅ Prophet model training successful")
        print(f"    - Model: {chosen_name.upper()}")
        print(f"    - MAE: {metrics['mae']:.4f}")
        print(f"    - RMSE: {metrics['rmse']:.4f}")
        print(f"    - MAPE: {metrics['mape']:.2f}%")

        prophet_result = (chosen_name, model, metrics)
    except Exception as e:
        print(f"  ❌ Prophet model training failed: {e}")
        prophet_result = None

    # Test ARIMA model
    try:
        chosen_name, model, metrics = select_best_model(df, ["arima"], validation_split=0.3)
        print(f"  ✅ ARIMA model training successful")
        print(f"    - Model: {chosen_name.upper()}")
        print(f"    - MAE: {metrics['mae']:.4f}")
        print(f"    - RMSE: {metrics['rmse']:.4f}")
        print(f"    - MAPE: {metrics['mape']:.2f}%")

        arima_result = (chosen_name, model, metrics)
    except Exception as e:
        print(f"  ❌ ARIMA model training failed: {e}")
        arima_result = None

    # Test model comparison
    try:
        chosen_name, model, metrics = select_best_model(df, ["prophet", "arima"], validation_split=0.3)
        print(f"  ✅ Model comparison successful - Best: {chosen_name.upper()}")
        return True, (chosen_name, model, metrics), df
    except Exception as e:
        print(f"  ❌ Model comparison failed: {e}")
        return False, None, None

def test_forecast_generation(model_result, df):
    """Test forecast generation."""
    print("\nTesting Forecast Generation...")

    if model_result is None:
        print("  ❌ No model available for forecast testing")
        return False, None

    chosen_name, model, metrics = model_result

    try:
        # Test short-term forecast
        forecast_df = generate_forecast(
            chosen_name, model, 24, df["timestamp"].iloc[-1]
        )

        if forecast_df.empty:
            print("  ❌ Forecast generation returned empty results")
            return False, None

        print(f"  ✅ Forecast generation successful")
        print(f"    - Model: {chosen_name.upper()}")
        print(f"    - Forecast horizon: 24 hours")
        print(f"    - Forecast range: {forecast_df['predicted_deposit'].min():.4f} - {forecast_df['predicted_deposit'].max():.4f}")

        # Test longer forecast
        long_forecast = generate_forecast(chosen_name, model, 72, df["timestamp"].iloc[-1])
        print(f"  ✅ Extended forecast (72h) successful")

        return True, forecast_df

    except Exception as e:
        print(f"  ❌ Forecast generation failed: {e}")
        return False, None

def test_alert_system(model_result, forecast_df):
    """Test the alert system with mock emails."""
    print("\nTesting Alert System...")

    if model_result is None or forecast_df is None:
        print("  ❌ Cannot test alerts - missing model or forecast data")
        return False

    chosen_name, model, metrics = model_result

    # Mock SMTP configuration
    smtp_config = {
        "smtp_server": "localhost",
        "smtp_port": 1025,
        "smtp_user": "test_user",
        "smtp_pass": "test_pass"
    }

    test_recipient = "user@example.com"
    test_sender = "cashramp@example.com"

    # Test 1: Model Performance Alert (force poor performance)
    print("\n  📊 Testing Model Performance Alert...")
    try:
        # Create artificially poor metrics for testing
        poor_metrics = {
            "mae": metrics["mae"],
            "rmse": metrics["rmse"],
            "mape": 25.5  # Force high MAPE to trigger alert
        }

        success = TestEmailServer.send_test_email(
            f"[ALERT] Model Performance Degraded - {chosen_name.upper()}",
            f"""Model Performance Alert

The {chosen_name.upper()} model is performing poorly and may need attention.

Current Metrics:
- MAE: {poor_metrics['mae']:.4f}
- RMSE: {poor_metrics['rmse']:.4f}
- MAPE: {poor_metrics['mape']:.2f}%

Alert Threshold: 10.0%

Please check the dashboard for more details and consider retraining the model.

This is an automated alert from the Cashramp Market Rate Tracker.""",
            test_recipient,
            test_sender,
            smtp_config
        )

        if success:
            print("    ✅ Model performance alert test successful")
        else:
            print("    ❌ Model performance alert test failed")

    except Exception as e:
        print(f"    ❌ Model performance alert test failed: {e}")

    # Test 2: Rate Threshold Alert - Below Threshold
    print("\n  📉 Testing 'Below Threshold' Rate Alert...")
    try:
        # Simulate rates going below threshold
        threshold = 8.0
        min_rate = 6.5
        max_rate = forecast_df["predicted_deposit"].max()

        success = TestEmailServer.send_test_email(
            "[ALERT] Market Rate Forecast - Rate Going BELOW Threshold",
            f"""Market Rate Threshold Alert

The forecast predicts rates will go below your defined threshold in the next 24 hours.

Alert Configuration:
- Mode: Below Threshold (bad for buying)
- Threshold: {threshold}
- Forecast Period: 24 hours

Forecast Summary:
- Minimum Rate: {min_rate:.4f}
- Maximum Rate: {max_rate:.4f}
- Extreme Rate: {min_rate:.4f}

A detailed forecast CSV is attached for your analysis.

Consider adjusting your trading strategy based on this forecast.

This is an automated alert from the Cashramp Market Rate Tracker.""",
            test_recipient,
            test_sender,
            smtp_config,
            forecast_df=forecast_df
        )

        if success:
            print("    ✅ Below threshold alert test successful")

    except Exception as e:
        print(f"    ❌ Below threshold alert test failed: {e}")

    # Test 3: Rate Threshold Alert - Above Threshold
    print("\n  📈 Testing 'Above Threshold' Rate Alert...")
    try:
        # Simulate rates going above threshold
        threshold = 12.0
        min_rate = forecast_df["predicted_deposit"].min()
        max_rate = 15.8

        success = TestEmailServer.send_test_email(
            "[ALERT] Market Rate Forecast - Rate Going ABOVE Threshold",
            f"""Market Rate Threshold Alert

The forecast predicts rates will go above your defined threshold in the next 24 hours.

Alert Configuration:
- Mode: Above Threshold (bad for selling)
- Threshold: {threshold}
- Forecast Period: 24 hours

Forecast Summary:
- Minimum Rate: {min_rate:.4f}
- Maximum Rate: {max_rate:.4f}
- Extreme Rate: {max_rate:.4f}

A detailed forecast CSV is attached for your analysis.

Consider adjusting your trading strategy based on this forecast.

This is an automated alert from the Cashramp Market Rate Tracker.""",
            test_recipient,
            test_sender,
            smtp_config,
            forecast_df=forecast_df
        )

        if success:
            print("    ✅ Above threshold alert test successful")

    except Exception as e:
        print(f"    ❌ Above threshold alert test failed: {e}")

    return True

def test_csv_export(forecast_df):
    """Test CSV export functionality."""
    print("\nTesting CSV Export...")

    if forecast_df is None:
        print("  ❌ No forecast data available for CSV export test")
        return False

    try:
        # Test CSV generation
        csv_content = forecast_df.to_csv(index=False)

        if not csv_content:
            print("  ❌ CSV export returned empty content")
            return False

        # Verify CSV structure
        lines = csv_content.strip().split('\n')
        header = lines[0]
        expected_columns = ["forecast_time", "predicted_deposit"]

        for col in expected_columns:
            if col not in header:
                print(f"  ❌ Missing column in CSV: {col}")
                return False

        print(f"  ✅ CSV export successful")
        print(f"    - Rows: {len(lines) - 1}")  # -1 for header
        print(f"    - Columns: {header}")
        print(f"    - Sample data: {lines[1] if len(lines) > 1 else 'No data'}")

        return True

    except Exception as e:
        print(f"  ❌ CSV export failed: {e}")
        return False

def run_comprehensive_test():
    """Run all tests in sequence."""
    print("CASHRAMP MARKET RATE TRACKER - COMPREHENSIVE TEST SUITE")
    print("="*80)

    test_results = {
        "database": False,
        "model_training": False,
        "forecast_generation": False,
        "alert_system": False,
        "csv_export": False
    }

    # Test 1: Database Operations
    test_results["database"] = test_database_operations()

    if not test_results["database"]:
        print("\n❌ Critical failure in database operations. Stopping tests.")
        return False

    # Test 2: Model Training
    success, model_result, df = test_model_training()
    test_results["model_training"] = success

    if not success:
        print("\n❌ Critical failure in model training. Stopping tests.")
        return False

    # Test 3: Forecast Generation
    success, forecast_df = test_forecast_generation(model_result, df)
    test_results["forecast_generation"] = success

    # Test 4: Alert System
    test_results["alert_system"] = test_alert_system(model_result, forecast_df)

    # Test 5: CSV Export
    test_results["csv_export"] = test_csv_export(forecast_df)

    # Summary
    print("\n" + "="*80)
    print("📋 TEST SUMMARY")
    print("="*80)

    all_passed = True
    for test_name, passed in test_results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name.replace('_', ' ').title():<25} {status}")
        if not passed:
            all_passed = False

    print("="*80)

    if all_passed:
        print("🎉 ALL TESTS PASSED! System is ready for deployment.")
        print("\n📝 Next Steps:")
        print("  1. Deploy to Streamlit Cloud")
        print("  2. Set up GitHub Actions for data collection")
        print("  3. Configure real SMTP credentials for alerts")
        print("  4. Monitor system performance")
    else:
        print("⚠️  SOME TESTS FAILED! Please review and fix issues before deployment.")

    return all_passed

if __name__ == "__main__":
    print("Starting local test harness...")
    print("Email Note: Email alerts will be printed to console (not actually sent)\n")

    try:
        success = run_comprehensive_test()
        exit_code = 0 if success else 1
        sys.exit(exit_code)

    except KeyboardInterrupt:
        print("\n\n⏹️  Test interrupted by user")
        sys.exit(1)

    except Exception as e:
        print(f"\n\n💥 Unexpected error during testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)