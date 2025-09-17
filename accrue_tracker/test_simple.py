#!/usr/bin/env python3
"""
Simple Test for Cashramp Market Rate Tracker
A simpler version without Unicode characters that can cause encoding issues.
"""

import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from db_utils import get_rates_data
from model_selector import select_best_model, generate_forecast
from seed_data import seed_fake_data

def test_basic_functionality():
    """Test the core functionality of the system."""
    print("Starting basic functionality test...")

    # Test 1: Database setup and seeding
    print("\n1. Testing database setup and data seeding...")
    try:
        # Seed some test data
        seed_fake_data(days=3, countries=["GH"], base_rate=10.0)
        print("   [PASS] Database seeded successfully")
    except Exception as e:
        print(f"   [FAIL] Database seeding failed: {e}")
        return False

    # Test 2: Data retrieval
    print("\n2. Testing data retrieval...")
    try:
        df = get_rates_data("GH")
        if len(df) > 0:
            print(f"   [PASS] Retrieved {len(df)} records")
        else:
            print("   [FAIL] No data retrieved")
            return False
    except Exception as e:
        print(f"   [FAIL] Data retrieval failed: {e}")
        return False

    # Test 3: Model training
    print("\n3. Testing model training...")
    try:
        chosen_name, model, metrics = select_best_model(df, ["prophet"], validation_split=0.3)
        print(f"   [PASS] Model trained: {chosen_name.upper()}")
        print(f"   [INFO] MAPE: {metrics['mape']:.2f}%")
    except Exception as e:
        print(f"   [FAIL] Model training failed: {e}")
        return False

    # Test 4: Forecast generation
    print("\n4. Testing forecast generation...")
    try:
        forecast_df = generate_forecast(chosen_name, model, 24, df["timestamp"].iloc[-1])
        if len(forecast_df) > 0:
            print(f"   [PASS] Generated {len(forecast_df)} forecast points")
            print(f"   [INFO] Forecast range: {forecast_df['predicted_deposit'].min():.4f} - {forecast_df['predicted_deposit'].max():.4f}")
        else:
            print("   [FAIL] No forecast generated")
            return False
    except Exception as e:
        print(f"   [FAIL] Forecast generation failed: {e}")
        return False

    print("\n" + "="*60)
    print("ALL BASIC TESTS PASSED!")
    print("System is working correctly.")
    print("="*60)
    return True

if __name__ == "__main__":
    print("Cashramp Market Rate Tracker - Simple Test")
    print("="*50)

    try:
        success = test_basic_functionality()
        if success:
            print("\nNext steps:")
            print("1. Run: poetry run streamlit run app.py")
            print("2. Open your browser to test the dashboard")
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        sys.exit(1)