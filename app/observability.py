from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


OBS_DIR = Path("observability")
OBS_DB = OBS_DIR / "events.db"
OBS_JSONL = OBS_DIR / "events.jsonl"
MAX_EVENT_ROWS = 100


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_db() -> None:
    OBS_DIR.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(OBS_DB)
    try:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                endpoint TEXT NOT NULL,
                question TEXT,
                csv_path TEXT,
                namespace TEXT,
                route TEXT,
                route_reason TEXT,
                answer TEXT,
                rag_similarity_json TEXT,
                sql_query TEXT,
                sql_error TEXT,
                citations_count INTEGER,
                latency_ms REAL,
                success INTEGER NOT NULL,
                error_message TEXT
            )
            """
        )
        con.commit()
    finally:
        con.close()


def log_event(event: dict[str, Any]) -> None:
    _ensure_db()
    con = sqlite3.connect(OBS_DB)
    try:
        con.execute(
            """
            INSERT INTO events (
                ts, endpoint, question, csv_path, namespace, route, route_reason,
                answer, rag_similarity_json, sql_query, sql_error, citations_count,
                latency_ms, success, error_message
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.get("ts", _utc_now_iso()),
                event.get("endpoint"),
                event.get("question"),
                event.get("csv_path"),
                event.get("namespace"),
                event.get("route"),
                event.get("route_reason"),
                event.get("answer"),
                json.dumps(event.get("rag_similarity")) if event.get("rag_similarity") is not None else None,
                event.get("sql_query"),
                event.get("sql_error"),
                event.get("citations_count"),
                event.get("latency_ms"),
                1 if event.get("success", False) else 0,
                event.get("error_message"),
            ),
        )
        # Keep only the latest N events for lightweight history/observability storage.
        con.execute(
            f"""
            DELETE FROM events
            WHERE id NOT IN (
                SELECT id FROM events ORDER BY id DESC LIMIT {MAX_EVENT_ROWS}
            )
            """
        )
        con.commit()
    finally:
        con.close()

    # Also persist full event payload for richer debugging (including optional embeddings).
    with OBS_JSONL.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")

