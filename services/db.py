from __future__ import annotations

import sqlite3
from pathlib import Path
from datetime import datetime
from flask import g

from config import get_settings

# Resolve paths and settings
settings = get_settings()
BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = (BASE_DIR / settings.db_path).resolve()
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def get_db() -> sqlite3.Connection:
    db = getattr(g, '_db', None)
    if db is None:
        # Note: isolation_level=None enables autocommit by default; keep default and commit explicitly
        db = g._db = sqlite3.connect(DB_PATH)
        db.row_factory = sqlite3.Row
        try:
            db.execute("PRAGMA journal_mode=WAL;")
            db.execute("PRAGMA synchronous=NORMAL;")
        except Exception:
            pass
    return db


def close_db(e=None):  # noqa: unused-argument
    db = getattr(g, '_db', None)
    if db is not None:
        db.close()
        g._db = None


def _new_db_conn() -> sqlite3.Connection:
    """Create a new standalone DB connection (not tied to Flask g)."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
    except Exception:
        pass
    return conn


def init_db():
    """Initialize database schema and bootstrap admin if configured."""
    db = get_db()
    # Improve SQLite concurrency
    try:
        db.execute("PRAGMA journal_mode=WAL;")
        db.execute("PRAGMA synchronous=NORMAL;")
    except Exception:
        pass

    # History table
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            user TEXT,
            lang TEXT,
            provider TEXT,
            model TEXT,
            payload_json TEXT NOT NULL,
            prompt_text TEXT NOT NULL,
            response_text TEXT NOT NULL
        )
        """
    )

    # Jobs table (background processing queue)
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            started_at TEXT,
            finished_at TEXT,
            user TEXT,
            lang TEXT,
            status TEXT NOT NULL,
            provider TEXT,
            model TEXT,
            payload_json TEXT NOT NULL,
            prompt_text TEXT NOT NULL,
            response_text TEXT,
            error_text TEXT,
            history_id INTEGER
        )
        """
    )

    # Users table
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            is_admin INTEGER NOT NULL DEFAULT 0,
            is_active INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL
        )
        """
    )

    # Drafts table
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS drafts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            user TEXT,
            lang TEXT,
            payload_json TEXT NOT NULL
        )
        """
    )

    # Bootstrap admin from ENV if provided and users table is empty
    try:
        count_row = db.execute("SELECT COUNT(*) AS c FROM users").fetchone()
        count = (count_row[0] if isinstance(count_row, tuple) else count_row["c"]) or 0
        if count == 0 and (settings.auth_username and settings.auth_password):
            from werkzeug.security import generate_password_hash
            db.execute(
                "INSERT INTO users (username, password_hash, is_admin, is_active, created_at) VALUES (?, ?, 1, 1, ?)",
                (
                    settings.auth_username.strip(),
                    generate_password_hash(settings.auth_password),
                    datetime.utcnow().isoformat(timespec='seconds'),
                ),
            )
            db.commit()
    except Exception:
        # ignore bootstrap errors to not break app startup
        pass

    db.commit()
