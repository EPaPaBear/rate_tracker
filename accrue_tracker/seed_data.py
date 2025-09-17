import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, UTC
from db_utils import init_db

def seed_fake_data(days=7, hours_per_day=24, countries=["GH"], base_rate=10.0):
    """
    Seed the database with realistic fake market rate data for testing.

    Args:
        days: Number of days of historical data to generate
        hours_per_day: Data points per day (24 = hourly)
        countries: List of country codes to generate data for
        base_rate: Base deposit rate to vary around
    """
    print(f"Seeding database with {days} days of fake data...")

    # Initialize database
    conn = init_db()
    c = conn.cursor()

    # Clear existing data for clean testing
    c.execute("DELETE FROM rates")
    conn.commit()

    now = datetime.now(UTC)
    total_hours = days * hours_per_day

    for country in countries:
        print(f"Generating data for country: {country}")
        rows = []

        for i in range(total_hours):
            # Generate timestamp (going backwards from now)
            ts = now - timedelta(hours=total_hours - i)

            # Create realistic rate variation
            # Base trend with some seasonality and noise
            trend = 0.1 * np.sin(i / (24 * 7)) * 2  # Weekly pattern
            daily_cycle = 0.05 * np.sin(i / 24 * 2 * np.pi)  # Daily pattern
            noise = np.random.normal(0, 0.02)  # Random noise

            # Generate deposit rate
            deposit_rate = base_rate + trend + daily_cycle + noise

            # Ensure positive rates
            deposit_rate = max(deposit_rate, 0.1)

            # Withdrawal rate is typically slightly lower
            withdrawal_rate = deposit_rate * (0.98 + np.random.normal(0, 0.005))
            withdrawal_rate = max(withdrawal_rate, 0.1)

            # Add some occasional spikes/drops for testing alerts
            if np.random.random() < 0.02:  # 2% chance of spike
                spike_factor = 1 + np.random.uniform(-0.3, 0.4)  # ±30-40% change
                deposit_rate *= spike_factor
                withdrawal_rate *= spike_factor

            rows.append((
                ts.isoformat(),
                country,
                ts.weekday(),
                ts.hour,
                round(deposit_rate, 6),
                round(withdrawal_rate, 6)
            ))

        # Insert data
        c.executemany("""
            INSERT INTO rates (timestamp, country, weekday, hour, depositRate, withdrawalRate)
            VALUES (?, ?, ?, ?, ?, ?)
        """, rows)

    conn.commit()
    conn.close()

    print(f"Successfully seeded {total_hours * len(countries)} data points")
    print(f"Countries: {', '.join(countries)}")
    print(f"Time range: {(now - timedelta(hours=total_hours)).strftime('%Y-%m-%d %H:%M')} to {now.strftime('%Y-%m-%d %H:%M')}")

def seed_test_scenarios():
    """
    Create specific test scenarios for alert testing.
    """
    print("Creating test scenarios for alert validation...")

    conn = sqlite3.connect("market_rates.db")
    c = conn.cursor()

    now = datetime.now(UTC)

    # Scenario 1: Declining rates (for "below threshold" alerts)
    declining_data = []
    for i in range(12):  # 12 hours of declining rates
        ts = now - timedelta(hours=12-i)
        rate = 15.0 - (i * 0.8)  # Decline from 15.0 to ~5.4
        declining_data.append((
            ts.isoformat(), "GH", ts.weekday(), ts.hour,
            rate, rate * 0.98
        ))

    # Scenario 2: Rising rates (for "above threshold" alerts)
    rising_data = []
    base_time = now - timedelta(hours=24)
    for i in range(12):  # 12 hours of rising rates
        ts = base_time + timedelta(hours=i)
        rate = 8.0 + (i * 0.6)  # Rise from 8.0 to ~14.6
        rising_data.append((
            ts.isoformat(), "NG", ts.weekday(), ts.hour,
            rate, rate * 0.98
        ))

    # Insert test scenarios
    test_data = declining_data + rising_data
    c.executemany("""
        INSERT INTO rates (timestamp, country, weekday, hour, depositRate, withdrawalRate)
        VALUES (?, ?, ?, ?, ?, ?)
    """, test_data)

    conn.commit()
    conn.close()

    print("Test scenarios created:")
    print("  Declining rates (GH): Testing 'below threshold' alerts")
    print("  Rising rates (NG): Testing 'above threshold' alerts")

def show_data_summary():
    """Display summary of seeded data."""
    conn = sqlite3.connect("market_rates.db")

    # Get summary statistics
    summary_query = """
    SELECT
        country,
        COUNT(*) as data_points,
        MIN(timestamp) as earliest,
        MAX(timestamp) as latest,
        ROUND(AVG(depositRate), 4) as avg_deposit_rate,
        ROUND(MIN(depositRate), 4) as min_deposit_rate,
        ROUND(MAX(depositRate), 4) as max_deposit_rate
    FROM rates
    GROUP BY country
    ORDER BY country
    """

    df = pd.read_sql_query(summary_query, conn)
    conn.close()

    print("\nData Summary:")
    print("=" * 80)
    for _, row in df.iterrows():
        print(f"Country: {row['country']}")
        print(f"  Data Points: {row['data_points']}")
        print(f"  Time Range: {row['earliest']} to {row['latest']}")
        print(f"  Deposit Rate Range: {row['min_deposit_rate']} - {row['max_deposit_rate']} (avg: {row['avg_deposit_rate']})")
        print()

if __name__ == "__main__":
    # Default seeding parameters
    import argparse

    parser = argparse.ArgumentParser(description="Seed database with fake market rate data")
    parser.add_argument("--days", type=int, default=7, help="Days of historical data")
    parser.add_argument("--countries", nargs="+", default=["GH"], help="Country codes")
    parser.add_argument("--base-rate", type=float, default=10.0, help="Base deposit rate")
    parser.add_argument("--test-scenarios", action="store_true", help="Add test scenarios for alerts")
    parser.add_argument("--summary", action="store_true", help="Show data summary after seeding")

    args = parser.parse_args()

    try:
        # Seed main data
        seed_fake_data(
            days=args.days,
            countries=args.countries,
            base_rate=args.base_rate
        )

        # Add test scenarios if requested
        if args.test_scenarios:
            seed_test_scenarios()

        # Show summary if requested
        if args.summary:
            show_data_summary()

        print(f"\nDatabase seeding complete!")
        print(f"Run 'poetry run streamlit run app.py' to view the dashboard")

    except Exception as e:
        print(f"Error during seeding: {e}")
        exit(1)