import sqlite3


def create_local_detection_db(db_path):
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            object_id INTEGER,
            final_class TEXT,
            height_cm REAL,
            status TEXT,
            timestamp DATETIME
        )
        """
    )
    rows = [
        ("session-1", 1, "Glass", 2.0, "Sorted", "2026-07-13 10:00:05"),
        ("session-1", 2, "Paper", 4.0, "Sorted", "2026-07-13 10:00:20"),
        ("session-1", 3, "Metal", 6.0, "Tracking", "2026-07-13 10:00:35"),
    ]
    conn.executemany(
        "INSERT INTO detections (session_id, object_id, final_class, height_cm, status, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    conn.close()


def test_database_manager_migrates_old_detection_schema(tmp_path):
    from modules.database import DatabaseManager

    db_path = tmp_path / "legacy.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            object_id INTEGER,
            final_class TEXT,
            vision_class TEXT,
            spectrum_class TEXT,
            height_cm REAL,
            status TEXT,
            timestamp DATETIME
        )
        """
    )
    conn.commit()
    conn.close()

    db = DatabaseManager(str(db_path))
    db.close()

    conn = sqlite3.connect(db_path)
    columns = {
        row[1]
        for row in conn.execute("PRAGMA table_info(detections)").fetchall()
    }
    conn.close()
    assert "belt_x_cm" in columns
    assert "belt_y_cm" in columns


def test_machine_sync_service_builds_summary_and_marks_outbox_sent(tmp_path):
    from modules.dashboard_sync import MachineSyncClient, MachineSyncService

    local_db = tmp_path / "machine.db"
    sync_db = tmp_path / "sync.db"
    create_local_detection_db(local_db)

    captured = []

    class DummyClient(MachineSyncClient):
        def send_payload(self, payload):
            captured.append(payload)
            return {"ok": True}

    service = MachineSyncService(
        machine_db_path=str(local_db),
        sync_db_path=str(sync_db),
        machine_id="SORTER-01",
        machine_name="Sorter 01",
        site_name="Factory A",
        line_name="Line 1",
        dashboard_client=DummyClient("http://example.com", "secret"),
    )

    result = service.run_once(now_iso="2026-07-13T10:01:10+07:00")
    assert result["sent"] == 1
    assert captured[0]["stats"]["detected_total"] == 3
    assert captured[0]["stats"]["sorted_total"] == 2
    assert captured[0]["stats"]["class_counts"]["Glass"] == 1
    assert captured[0]["live_status"]["active_objects"] == 1

    conn = sqlite3.connect(sync_db)
    rows = conn.execute(
        "SELECT status, retry_count FROM sync_outbox"
    ).fetchall()
    conn.close()
    assert rows == [("sent", 0)]


def test_machine_sync_service_leaves_failed_payload_pending(tmp_path):
    from modules.dashboard_sync import MachineSyncClient, MachineSyncService

    local_db = tmp_path / "machine.db"
    sync_db = tmp_path / "sync.db"
    create_local_detection_db(local_db)

    class FailingClient(MachineSyncClient):
        def send_payload(self, payload):
            raise RuntimeError("network down")

    service = MachineSyncService(
        machine_db_path=str(local_db),
        sync_db_path=str(sync_db),
        machine_id="SORTER-01",
        machine_name="Sorter 01",
        site_name="Factory A",
        line_name="Line 1",
        dashboard_client=FailingClient("http://example.com", "secret"),
    )

    result = service.run_once(now_iso="2026-07-13T10:01:10+07:00")
    assert result["failed"] == 1

    conn = sqlite3.connect(sync_db)
    rows = conn.execute(
        "SELECT status, retry_count, last_error FROM sync_outbox"
    ).fetchall()
    conn.close()
    assert rows == [("pending", 1, "network down")]
