"""
Central dashboard storage and query helpers.
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import closing
from datetime import datetime


DEFAULT_CENTRAL_DB = "central_dashboard.db"
CLASS_KEYS = ("Glass", "Metal", "Paper", "Plastic", "Unknown")


class CentralDashboardStore:
    """Owns the central dashboard SQLite database."""

    def __init__(self, db_path: str = DEFAULT_CENTRAL_DB):
        self.db_path = db_path
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with closing(self._connect()) as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS machines (
                    machine_id TEXT PRIMARY KEY,
                    machine_name TEXT,
                    site_name TEXT,
                    line_name TEXT,
                    is_active INTEGER NOT NULL DEFAULT 1,
                    last_seen_at TEXT,
                    last_status TEXT
                );

                CREATE TABLE IF NOT EXISTS minute_stats (
                    machine_id TEXT NOT NULL,
                    minute_bucket TEXT NOT NULL,
                    session_id TEXT,
                    detected_total INTEGER NOT NULL DEFAULT 0,
                    sorted_total INTEGER NOT NULL DEFAULT 0,
                    glass_count INTEGER NOT NULL DEFAULT 0,
                    metal_count INTEGER NOT NULL DEFAULT 0,
                    paper_count INTEGER NOT NULL DEFAULT 0,
                    plastic_count INTEGER NOT NULL DEFAULT 0,
                    unknown_count INTEGER NOT NULL DEFAULT 0,
                    avg_height_cm REAL,
                    last_updated_at TEXT NOT NULL,
                    PRIMARY KEY (machine_id, minute_bucket)
                );

                CREATE TABLE IF NOT EXISTS machine_live_status (
                    machine_id TEXT PRIMARY KEY,
                    session_id TEXT,
                    status TEXT NOT NULL,
                    active_objects INTEGER NOT NULL DEFAULT 0,
                    last_error TEXT,
                    last_sync_at TEXT NOT NULL,
                    payload_updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS sync_audit (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    machine_id TEXT NOT NULL,
                    minute_bucket TEXT NOT NULL,
                    received_at TEXT NOT NULL,
                    result TEXT NOT NULL,
                    message TEXT
                );
                """
            )
            conn.commit()

    @staticmethod
    def validate_payload(payload: dict) -> list[str]:
        required = ("machine_id", "minute_bucket", "stats", "live_status")
        missing = [key for key in required if not payload.get(key)]
        stats = payload.get("stats") or {}
        live_status = payload.get("live_status") or {}
        if "detected_total" not in stats:
            missing.append("stats.detected_total")
        if "sorted_total" not in stats:
            missing.append("stats.sorted_total")
        if "class_counts" not in stats:
            missing.append("stats.class_counts")
        if "status" not in live_status:
            missing.append("live_status.status")
        if "active_objects" not in live_status:
            missing.append("live_status.active_objects")
        return missing

    def upsert_minute_payload(self, payload: dict) -> None:
        now = datetime.now().isoformat(timespec="seconds")
        stats = payload["stats"]
        live_status = payload["live_status"]
        class_counts = stats.get("class_counts") or {}
        with closing(self._connect()) as conn:
            conn.execute(
                """
                INSERT INTO machines (
                    machine_id, machine_name, site_name, line_name,
                    is_active, last_seen_at, last_status
                ) VALUES (?, ?, ?, ?, 1, ?, ?)
                ON CONFLICT(machine_id) DO UPDATE SET
                    machine_name=excluded.machine_name,
                    site_name=excluded.site_name,
                    line_name=excluded.line_name,
                    is_active=1,
                    last_seen_at=excluded.last_seen_at,
                    last_status=excluded.last_status
                """,
                (
                    payload["machine_id"],
                    payload.get("machine_name", payload["machine_id"]),
                    payload.get("site_name", ""),
                    payload.get("line_name", ""),
                    now,
                    live_status.get("status", "unknown"),
                ),
            )
            conn.execute(
                """
                INSERT INTO minute_stats (
                    machine_id, minute_bucket, session_id,
                    detected_total, sorted_total,
                    glass_count, metal_count, paper_count, plastic_count, unknown_count,
                    avg_height_cm, last_updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(machine_id, minute_bucket) DO UPDATE SET
                    session_id=excluded.session_id,
                    detected_total=excluded.detected_total,
                    sorted_total=excluded.sorted_total,
                    glass_count=excluded.glass_count,
                    metal_count=excluded.metal_count,
                    paper_count=excluded.paper_count,
                    plastic_count=excluded.plastic_count,
                    unknown_count=excluded.unknown_count,
                    avg_height_cm=excluded.avg_height_cm,
                    last_updated_at=excluded.last_updated_at
                """,
                (
                    payload["machine_id"],
                    payload["minute_bucket"],
                    payload.get("session_id"),
                    int(stats.get("detected_total", 0)),
                    int(stats.get("sorted_total", 0)),
                    int(class_counts.get("Glass", 0)),
                    int(class_counts.get("Metal", 0)),
                    int(class_counts.get("Paper", 0)),
                    int(class_counts.get("Plastic", 0)),
                    int(class_counts.get("Unknown", 0)),
                    stats.get("avg_height_cm"),
                    now,
                ),
            )
            conn.execute(
                """
                INSERT INTO machine_live_status (
                    machine_id, session_id, status, active_objects, last_error,
                    last_sync_at, payload_updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(machine_id) DO UPDATE SET
                    session_id=excluded.session_id,
                    status=excluded.status,
                    active_objects=excluded.active_objects,
                    last_error=excluded.last_error,
                    last_sync_at=excluded.last_sync_at,
                    payload_updated_at=excluded.payload_updated_at
                """,
                (
                    payload["machine_id"],
                    payload.get("session_id"),
                    live_status.get("status", "unknown"),
                    int(live_status.get("active_objects", 0)),
                    live_status.get("last_error"),
                    now,
                    now,
                ),
            )
            conn.execute(
                """
                INSERT INTO sync_audit (machine_id, minute_bucket, received_at, result, message)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    payload["machine_id"],
                    payload["minute_bucket"],
                    now,
                    "ok",
                    "",
                ),
            )
            conn.commit()

    def build_dashboard_snapshot(self, query_date: str, machine_id: str | None = None) -> dict:
        date_start = f"{query_date} 00:00:00"
        date_end = f"{query_date} 23:59:59"
        if "T" in query_date:
            date_start = query_date
            date_end = query_date
        filters = "WHERE substr(minute_bucket, 1, 10) BETWEEN ? AND ?"
        params: list[object] = [date_start[:10], date_end[:10]]
        if machine_id:
            filters += " AND machine_id = ?"
            params.append(machine_id)

        with closing(self._connect()) as conn:
            stats_row = conn.execute(
                f"""
                SELECT
                    COALESCE(SUM(detected_total), 0) AS total_detected,
                    COALESCE(SUM(sorted_total), 0) AS total_sorted,
                    COALESCE(SUM(glass_count), 0) AS glass_count,
                    COALESCE(SUM(metal_count), 0) AS metal_count,
                    COALESCE(SUM(paper_count), 0) AS paper_count,
                    COALESCE(SUM(plastic_count), 0) AS plastic_count,
                    COALESCE(SUM(unknown_count), 0) AS unknown_count
                FROM minute_stats
                {filters}
                """,
                params,
            ).fetchone()

            trend_rows = conn.execute(
                f"""
                SELECT minute_bucket, detected_total, sorted_total
                FROM minute_stats
                {filters}
                ORDER BY minute_bucket
                """,
                params,
            ).fetchall()

            machine_params = [machine_id] if machine_id else []
            machine_filter = "WHERE mls.machine_id = ?" if machine_id else ""
            machine_rows = conn.execute(
                f"""
                SELECT
                    m.machine_id,
                    COALESCE(m.machine_name, m.machine_id) AS machine_name,
                    m.site_name,
                    m.line_name,
                    mls.session_id,
                    mls.status,
                    mls.active_objects,
                    mls.last_error,
                    mls.last_sync_at
                FROM machine_live_status mls
                JOIN machines m ON m.machine_id = mls.machine_id
                {machine_filter}
                ORDER BY m.machine_id
                """,
                machine_params,
            ).fetchall()

        pie_data = [
            {"name": "Glass", "value": stats_row["glass_count"]},
            {"name": "Metal", "value": stats_row["metal_count"]},
            {"name": "Paper", "value": stats_row["paper_count"]},
            {"name": "Plastic", "value": stats_row["plastic_count"]},
            {"name": "Unknown", "value": stats_row["unknown_count"]},
        ]
        pie_data = [item for item in pie_data if item["value"]]

        return {
            "stats": {
                "total_detected": stats_row["total_detected"],
                "total_sorted": stats_row["total_sorted"],
                "active_machines": sum(1 for row in machine_rows if row["status"] == "running"),
            },
            "pie_chart": {"data": pie_data},
            "trend_chart": {
                "categories": [row["minute_bucket"][-8:-3] for row in trend_rows],
                "series": [
                    {
                        "name": "Detected",
                        "type": "line",
                        "data": [row["detected_total"] for row in trend_rows],
                        "smooth": True,
                    },
                    {
                        "name": "Sorted",
                        "type": "line",
                        "data": [row["sorted_total"] for row in trend_rows],
                        "smooth": True,
                    },
                ] if trend_rows else [],
            },
            "machines": [dict(row) for row in machine_rows],
        }


def default_dashboard_config() -> dict:
    return {
        "CENTRAL_DASHBOARD_DB": os.environ.get("CENTRAL_DASHBOARD_DB", DEFAULT_CENTRAL_DB),
        "DASHBOARD_API_KEY": os.environ.get("DASHBOARD_API_KEY", ""),
    }
