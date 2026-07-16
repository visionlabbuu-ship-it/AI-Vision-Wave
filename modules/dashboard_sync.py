"""
Machine-side dashboard sync worker.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import requests


CLASS_NAMES = ("Glass", "Metal", "Paper", "Plastic", "Unknown")


def parse_iso_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value)


def iso_minute_floor(value: datetime) -> datetime:
    return value.replace(second=0, microsecond=0)


class MachineSyncClient:
    def __init__(self, base_url: str, api_key: str, timeout_s: float = 10.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_s = timeout_s

    def send_payload(self, payload):
        response = requests.post(
            f"{self.base_url}/api/ingest/minute-summary",
            json=payload,
            headers={"X-API-Key": self.api_key},
            timeout=self.timeout_s,
        )
        response.raise_for_status()
        return response.json()


class MachineSyncStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with closing(self._connect()) as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS sync_state (
                    machine_id TEXT PRIMARY KEY,
                    last_synced_minute TEXT,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS sync_outbox (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    machine_id TEXT NOT NULL,
                    minute_bucket TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    retry_count INTEGER NOT NULL DEFAULT 0,
                    last_error TEXT,
                    created_at TEXT NOT NULL,
                    sent_at TEXT,
                    UNIQUE(machine_id, minute_bucket)
                );
                """
            )
            conn.commit()

    def enqueue_payload(self, machine_id: str, minute_bucket: str, payload: dict) -> None:
        now = datetime.now().isoformat(timespec="seconds")
        with closing(self._connect()) as conn:
            conn.execute(
                """
                INSERT INTO sync_outbox (
                    machine_id, minute_bucket, payload_json, status, retry_count,
                    last_error, created_at, sent_at
                ) VALUES (?, ?, ?, 'pending', 0, NULL, ?, NULL)
                ON CONFLICT(machine_id, minute_bucket) DO UPDATE SET
                    payload_json=excluded.payload_json,
                    status='pending',
                    last_error=NULL
                """,
                (machine_id, minute_bucket, json.dumps(payload), now),
            )
            conn.commit()

    def pending_items(self):
        with closing(self._connect()) as conn:
            rows = conn.execute(
                """
                SELECT id, machine_id, minute_bucket, payload_json, retry_count
                FROM sync_outbox
                WHERE status = 'pending'
                ORDER BY minute_bucket, id
                """
            ).fetchall()
        return rows

    def mark_sent(self, row_id: int, machine_id: str, minute_bucket: str):
        now = datetime.now().isoformat(timespec="seconds")
        with closing(self._connect()) as conn:
            conn.execute(
                """
                UPDATE sync_outbox
                SET status='sent', sent_at=?, last_error=NULL
                WHERE id=?
                """,
                (now, row_id),
            )
            conn.execute(
                """
                INSERT INTO sync_state (machine_id, last_synced_minute, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(machine_id) DO UPDATE SET
                    last_synced_minute=excluded.last_synced_minute,
                    updated_at=excluded.updated_at
                """,
                (machine_id, minute_bucket, now),
            )
            conn.commit()

    def mark_failed(self, row_id: int, error: str):
        with closing(self._connect()) as conn:
            conn.execute(
                """
                UPDATE sync_outbox
                SET retry_count = retry_count + 1,
                    last_error = ?,
                    status = 'pending'
                WHERE id = ?
                """,
                (error, row_id),
            )
            conn.commit()

    def get_last_synced_minute(self, machine_id: str):
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT last_synced_minute FROM sync_state WHERE machine_id = ?",
                (machine_id,),
            ).fetchone()
        return row["last_synced_minute"] if row else None


@dataclass
class MachineSummary:
    machine_id: str
    minute_bucket: str
    session_id: str | None
    detected_total: int
    sorted_total: int
    class_counts: dict
    avg_height_cm: float | None
    active_objects: int
    status: str
    last_error: str | None = None

    def to_payload(self, machine_name: str, site_name: str, line_name: str) -> dict:
        return {
            "machine_id": self.machine_id,
            "machine_name": machine_name,
            "site_name": site_name,
            "line_name": line_name,
            "minute_bucket": self.minute_bucket,
            "session_id": self.session_id,
            "stats": {
                "detected_total": self.detected_total,
                "sorted_total": self.sorted_total,
                "class_counts": self.class_counts,
                "avg_height_cm": self.avg_height_cm,
            },
            "live_status": {
                "status": self.status,
                "active_objects": self.active_objects,
                "last_error": self.last_error,
            },
        }


class MachineSyncService:
    def __init__(
        self,
        machine_db_path: str,
        sync_db_path: str,
        machine_id: str,
        machine_name: str,
        site_name: str,
        line_name: str,
        dashboard_client: MachineSyncClient,
    ):
        self.machine_db_path = machine_db_path
        self.store = MachineSyncStore(sync_db_path)
        self.machine_id = machine_id
        self.machine_name = machine_name
        self.site_name = site_name
        self.line_name = line_name
        self.dashboard_client = dashboard_client

    def _connect_machine_db(self):
        conn = sqlite3.connect(self.machine_db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _detect_schema(self, conn):
        rows = conn.execute("PRAGMA table_info(detections)").fetchall()
        columns = {row["name"] for row in rows}
        if "final_class" in columns:
            class_col = "final_class"
        elif "class_name" in columns:
            class_col = "class_name"
        else:
            class_col = None
        has_session = "session_id" in columns
        return class_col, has_session

    def _build_summary_for_minute(self, minute_bucket: str) -> MachineSummary:
        minute_dt = parse_iso_datetime(minute_bucket)
        minute_start = minute_dt.strftime("%Y-%m-%d %H:%M:%S")
        minute_end = (minute_dt + timedelta(minutes=1)).strftime("%Y-%m-%d %H:%M:%S")
        with closing(self._connect_machine_db()) as conn:
            class_col, has_session = self._detect_schema(conn)
            select_cols = "status, height_cm, timestamp"
            if class_col:
                select_cols = f"{class_col} AS class_name, " + select_cols
            if has_session:
                select_cols = "session_id, " + select_cols
            rows = conn.execute(
                f"""
                SELECT {select_cols}
                FROM detections
                WHERE timestamp >= ? AND timestamp < ?
                ORDER BY timestamp
                """,
                (minute_start, minute_end),
            ).fetchall()
            latest = conn.execute(
                f"""
                SELECT {select_cols}
                FROM detections
                ORDER BY timestamp DESC
                LIMIT 20
                """
            ).fetchall()

        class_counts = {name: 0 for name in CLASS_NAMES}
        heights = []
        sorted_total = 0
        session_id = None
        for row in rows:
            session_id = session_id or row["session_id"] if "session_id" in row.keys() else session_id
            class_name = row["class_name"] if "class_name" in row.keys() else None
            if class_name not in class_counts:
                class_name = "Unknown"
            if class_name:
                class_counts[class_name] += 1
            if row["height_cm"] is not None:
                heights.append(float(row["height_cm"]))
            if str(row["status"]).lower() in {"sorted", "completed"}:
                sorted_total += 1

        active_objects = 0
        latest_session = None
        for row in latest:
            if "session_id" in row.keys() and latest_session is None:
                latest_session = row["session_id"]
            if str(row["status"]).lower() not in {"sorted", "completed", "picking"}:
                active_objects += 1

        detected_total = len(rows)
        status = "running" if (detected_total or active_objects) else "idle"
        avg_height = round(sum(heights) / len(heights), 4) if heights else None
        return MachineSummary(
            machine_id=self.machine_id,
            minute_bucket=minute_bucket,
            session_id=session_id or latest_session,
            detected_total=detected_total,
            sorted_total=sorted_total,
            class_counts=class_counts,
            avg_height_cm=avg_height,
            active_objects=active_objects,
            status=status,
        )

    def _collect_due_payloads(self, now_iso: str) -> int:
        now_dt = parse_iso_datetime(now_iso)
        target_minute = iso_minute_floor(now_dt) - timedelta(minutes=1)
        minute_bucket = target_minute.isoformat(timespec="seconds")
        last_synced = self.store.get_last_synced_minute(self.machine_id)
        if last_synced == minute_bucket:
            return 0
        summary = self._build_summary_for_minute(minute_bucket)
        payload = summary.to_payload(self.machine_name, self.site_name, self.line_name)
        self.store.enqueue_payload(self.machine_id, minute_bucket, payload)
        return 1

    def run_once(self, now_iso: str | None = None):
        if now_iso is None:
            now_iso = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
        queued = self._collect_due_payloads(now_iso)
        sent = 0
        failed = 0
        for item in self.store.pending_items():
            payload = json.loads(item["payload_json"])
            try:
                self.dashboard_client.send_payload(payload)
                self.store.mark_sent(item["id"], item["machine_id"], item["minute_bucket"])
                sent += 1
            except Exception as exc:
                self.store.mark_failed(item["id"], str(exc))
                failed += 1
        return {"queued": queued, "sent": sent, "failed": failed}
