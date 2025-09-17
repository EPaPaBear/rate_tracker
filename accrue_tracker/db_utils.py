import sqlite3
import pandas as pd
from datetime import datetime, timedelta, UTC

DB_PATH = "market_rates.db"

def init_db():
    """Initialize the SQLite database with the rates table."""
    conn = sqlite3.connect(DB_PATH)
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
    deleted_rows = conn.execute("DELETE FROM rates WHERE timestamp < ?", (cutoff,)).rowcount
    conn.commit()
    conn.close()
    return deleted_rows