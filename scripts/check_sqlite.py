from pathlib import Path
import sqlite3
import pandas as pd

DB_PATH = Path("data/synthetic/synthetic_echo_chamber_policy_dataset.sqlite")

conn = sqlite3.connect(DB_PATH)

tables = [
    "users",
    "follows",
    "posts",
    "comments",
    "likes",
    "comment_likes",
    "policy_events",
    "user_interaction_edges",
    "daily_summary"
]

for table in tables:
    query = f"SELECT COUNT(*) AS row_count FROM {table}"
    result = pd.read_sql_query(query, conn)
    print(table, ":", result.loc[0, "row_count"])

conn.close()