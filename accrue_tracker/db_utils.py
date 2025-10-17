import sqlite3
import pandas as pd
from datetime import datetime, timedelta, UTC

DB_PATH = "market_rates.db"

def init_db():
    """Initialize the SQLite database with the rates and forecasts tables."""
    conn = sqlite3.connect(DB_PATH)

    # Create rates table
    conn.execute("""
    CREATE TABLE IF NOT EXISTS rates (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        country TEXT,
        weekday INTEGER,
        hour INTEGER,
        depositRate REAL,
        withdrawalRate REAL
    )
    """)

    # Create forecasts table to track predictions over time
    conn.execute("""
    CREATE TABLE IF NOT EXISTS forecasts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at TEXT,
        forecast_time TEXT,
        country TEXT,
        predicted_deposit REAL,
        predicted_withdrawal REAL,
        deposit_model TEXT,
        withdrawal_model TEXT,
        horizon_hours INTEGER
    )
    """)

    conn.commit()
    return conn

def get_rates_data(country="GH", limit=None):
    """Load rates data from database."""
    conn = sqlite3.connect(DB_PATH)
    query = "SELECT * FROM rates WHERE country = ? ORDER BY timestamp ASC"
    if limit:
        query += f" LIMIT {limit}"

    df = pd.read_sql_query(query, conn, params=(country,))
    conn.close()

    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    return df

def prune_old_data(days=90):
    """Remove data older than specified days to keep database lean."""
    conn = sqlite3.connect(DB_PATH)
    cutoff = (datetime.now(UTC) - timedelta(days=days)).isoformat()
    deleted_rates = conn.execute("DELETE FROM rates WHERE timestamp < ?", (cutoff,)).rowcount
    deleted_forecasts = conn.execute("DELETE FROM forecasts WHERE created_at < ?", (cutoff,)).rowcount
    conn.commit()
    conn.close()
    return deleted_rates + deleted_forecasts

def save_forecast(forecast_df, country, deposit_model, withdrawal_model, horizon_hours):
    """Save forecast predictions to database for tracking over time."""
    conn = sqlite3.connect(DB_PATH)
    created_at = datetime.now(UTC).isoformat()

    for _, row in forecast_df.iterrows():
        conn.execute("""
            INSERT INTO forecasts (created_at, forecast_time, country, predicted_deposit, predicted_withdrawal, deposit_model, withdrawal_model, horizon_hours)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            created_at,
            row["forecast_time"].isoformat() if hasattr(row["forecast_time"], 'isoformat') else str(row["forecast_time"]),
            country,
            row["predicted_deposit"],
            row["predicted_withdrawal"],
            deposit_model,
            withdrawal_model,
            horizon_hours
        ))

    conn.commit()
    conn.close()

def get_forecast_history(country="GH", hours_back=168):
    """Get historical forecasts to show prediction evolution over time."""
    conn = sqlite3.connect(DB_PATH)
    cutoff = (datetime.now(UTC) - timedelta(hours=hours_back)).isoformat()

    query = """
        SELECT * FROM forecasts
        WHERE country = ? AND created_at >= ?
        ORDER BY created_at ASC, forecast_time ASC
    """

    df = pd.read_sql_query(query, conn, params=(country, cutoff))
    conn.close()

    if not df.empty:
        # Parse timestamps with UTC timezone handling
        df['created_at'] = pd.to_datetime(df['created_at'], format='ISO8601', errors='coerce')
        df['forecast_time'] = pd.to_datetime(df['forecast_time'], format='ISO8601', errors='coerce')

        # # Convert to timezone-naive for consistency
        # df['created_at'] = df['created_at'].dt.tz_localize(None)
        # df['forecast_time'] = df['forecast_time'].dt.tz_localize(None)

    return df