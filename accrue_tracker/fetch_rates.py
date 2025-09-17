import requests
import sqlite3
from datetime import datetime, UTC

GRAPHQL_ENDPOINT = "https://api.useaccrue.com/graphql/"
COUNTRIES = ["GH"]  # Add more country codes as needed

QUERY_TEMPLATE = """
query {
  cashrampMarketRate(countryCode: "%s", caller: user) {
    depositRate
    withdrawalRate
  }
}
"""

def init_db():
    conn = sqlite3.connect("market_rates.db")
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

def fetch_and_store():
    conn = init_db()
    now = datetime.now(UTC)
    weekday = now.weekday()
    hour = now.hour

    for country in COUNTRIES:
        response = requests.post(GRAPHQL_ENDPOINT, json={"query": QUERY_TEMPLATE % country})
        response.raise_for_status()
        data = response.json()["data"]["cashrampMarketRate"]

        conn.execute(
            "INSERT INTO rates (timestamp, country, weekday, hour, depositRate, withdrawalRate) VALUES (?, ?, ?, ?, ?, ?)",
            (now.isoformat(), country, weekday, hour, data["depositRate"], data["withdrawalRate"])
        )

    conn.commit()
    conn.close()

if __name__ == "__main__":
    try:
        fetch_and_store()
        print("Market rates fetched successfully")
    except Exception as e:
        print(f"Error fetching market rates: {e}")
        # Import here to avoid circular import issues
        from db_utils import prune_old_data
        # Even if fetch fails, we can still prune old data
        try:
            deleted = prune_old_data(90)
            print(f"Pruned {deleted} old records")
        except Exception as prune_error:
            print(f"Could not prune old data: {prune_error}")
        exit(1)