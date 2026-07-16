import sqlite3


def build_payload(minute_bucket="2026-07-13T10:00:00+07:00", machine_id="SORTER-01"):
    return {
        "machine_id": machine_id,
        "machine_name": "Sorter 01",
        "site_name": "Factory A",
        "line_name": "Line 1",
        "minute_bucket": minute_bucket,
        "session_id": "session-1",
        "stats": {
            "detected_total": 12,
            "sorted_total": 10,
            "class_counts": {
                "Glass": 3,
                "Metal": 2,
                "Paper": 4,
                "Plastic": 1,
                "Unknown": 2,
            },
            "avg_height_cm": 4.5,
        },
        "live_status": {
            "status": "running",
            "active_objects": 2,
            "last_error": None,
        },
    }


def test_ingest_endpoint_upserts_and_dashboard_reads_central_db(tmp_path):
    from dashboard_app import create_app

    db_path = tmp_path / "central.db"
    app = create_app(
        {
            "TESTING": True,
            "CENTRAL_DASHBOARD_DB": str(db_path),
            "DASHBOARD_API_KEY": "secret",
        }
    )
    client = app.test_client()

    response = client.post(
        "/api/ingest/minute-summary",
        json=build_payload(),
        headers={"X-API-Key": "secret"},
    )
    assert response.status_code == 200
    assert response.get_json()["ok"] is True

    response = client.post(
        "/api/ingest/minute-summary",
        json=build_payload(),
        headers={"X-API-Key": "secret"},
    )
    assert response.status_code == 200

    response = client.get("/api/data?date=2026-07-13")
    payload = response.get_json()
    assert payload["stats"]["total_detected"] == 12
    assert payload["stats"]["total_sorted"] == 10
    assert payload["stats"]["active_machines"] == 1
    assert payload["pie_chart"]["data"]
    assert payload["machines"][0]["machine_id"] == "SORTER-01"

    conn = sqlite3.connect(db_path)
    count = conn.execute("SELECT COUNT(*) FROM minute_stats").fetchone()[0]
    conn.close()
    assert count == 1


def test_ingest_requires_api_key(tmp_path):
    from dashboard_app import create_app

    app = create_app(
        {
            "TESTING": True,
            "CENTRAL_DASHBOARD_DB": str(tmp_path / "central.db"),
            "DASHBOARD_API_KEY": "secret",
        }
    )
    client = app.test_client()

    response = client.post("/api/ingest/minute-summary", json=build_payload())
    assert response.status_code == 401


def test_ingest_rejects_missing_required_fields(tmp_path):
    from dashboard_app import create_app

    app = create_app(
        {
            "TESTING": True,
            "CENTRAL_DASHBOARD_DB": str(tmp_path / "central.db"),
            "DASHBOARD_API_KEY": "secret",
        }
    )
    client = app.test_client()

    response = client.post(
        "/api/ingest/minute-summary",
        json={"machine_id": "SORTER-01"},
        headers={"X-API-Key": "secret"},
    )
    assert response.status_code == 400

