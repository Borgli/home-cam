import sqlite3
import json
import time
import os
import threading

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'surveillance.db')
_local = threading.local()

SCHEMA = """
CREATE TABLE IF NOT EXISTS detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    channel INTEGER NOT NULL,
    class_name TEXT NOT NULL,
    class_id INTEGER NOT NULL,
    confidence REAL NOT NULL,
    x1 REAL,
    y1 REAL,
    x2 REAL,
    y2 REAL,
    tracker_id INTEGER
);

CREATE TABLE IF NOT EXISTS zone_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    zone_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    class_name TEXT NOT NULL,
    tracker_id INTEGER
);

CREATE TABLE IF NOT EXISTS zones (
    id TEXT PRIMARY KEY,
    camera INTEGER NOT NULL,
    type TEXT NOT NULL,
    label TEXT,
    coords TEXT NOT NULL,
    classes TEXT,
    color TEXT DEFAULT '#00ff88',
    enabled INTEGER DEFAULT 1,
    created_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_detections_timestamp ON detections(timestamp);
CREATE INDEX IF NOT EXISTS idx_detections_channel ON detections(channel);
CREATE INDEX IF NOT EXISTS idx_detections_class ON detections(class_name);
CREATE INDEX IF NOT EXISTS idx_zone_events_timestamp ON zone_events(timestamp);
"""


def get_db():
    """Get thread-local database connection."""
    if not hasattr(_local, 'conn') or _local.conn is None:
        _local.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        _local.conn.row_factory = sqlite3.Row
        _local.conn.execute("PRAGMA journal_mode=WAL")
        _local.conn.execute("PRAGMA synchronous=NORMAL")
    return _local.conn


def init_db():
    """Initialize database schema."""
    conn = get_db()
    conn.executescript(SCHEMA)
    conn.commit()


def insert_detection(timestamp, channel, class_name, class_id, confidence, bbox=None, tracker_id=None):
    """Insert a detection record."""
    conn = get_db()
    x1, y1, x2, y2 = bbox if bbox else (None, None, None, None)
    conn.execute(
        "INSERT INTO detections (timestamp, channel, class_name, class_id, confidence, x1, y1, x2, y2, tracker_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (timestamp, channel, class_name, class_id, confidence, x1, y1, x2, y2, tracker_id)
    )
    conn.commit()


def insert_detections_batch(detections):
    """Insert multiple detections at once."""
    if not detections:
        return
    conn = get_db()
    conn.executemany(
        "INSERT INTO detections (timestamp, channel, class_name, class_id, confidence, x1, y1, x2, y2, tracker_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        detections
    )
    conn.commit()


def query_db(sql, params=None):
    """Execute a read-only query and return results as list of dicts."""
    sql_stripped = sql.strip().upper()
    if not sql_stripped.startswith('SELECT'):
        raise ValueError("Only SELECT queries are allowed")

    conn = get_db()
    cursor = conn.execute(sql, params or ())
    columns = [desc[0] for desc in cursor.description]
    rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
    return rows


def get_schema_info():
    """Get database schema as text for LLM context."""
    conn = get_db()
    cursor = conn.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
    tables = [row[0] for row in cursor.fetchall() if row[0]]

    # Also get sample data
    info = "DATABASE SCHEMA:\n\n"
    for table_sql in tables:
        info += table_sql + ";\n\n"

    # Row counts
    for table_name in ['detections', 'zones', 'zone_events']:
        try:
            count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
            info += f"-- {table_name}: {count} rows\n"
        except Exception:
            pass

    return info
