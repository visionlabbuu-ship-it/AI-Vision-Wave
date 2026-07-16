from __future__ import annotations

import logging
import os
from datetime import date

from flask import Flask, jsonify, render_template, request

from modules.dashboard_central import CentralDashboardStore, default_dashboard_config


logging.getLogger("werkzeug").setLevel(logging.ERROR)


def create_app(config: dict | None = None) -> Flask:
    app = Flask(__name__)
    app.config.update(default_dashboard_config())
    if config:
        app.config.update(config)

    store = CentralDashboardStore(app.config["CENTRAL_DASHBOARD_DB"])
    app.extensions["central_dashboard_store"] = store

    @app.route("/")
    def index():
        return render_template("Test_web.html")

    @app.route("/api/ingest/minute-summary", methods=["POST"])
    def ingest_minute_summary():
        api_key = app.config.get("DASHBOARD_API_KEY", "")
        if api_key and request.headers.get("X-API-Key") != api_key:
            return jsonify({"ok": False, "error": "unauthorized"}), 401

        payload = request.get_json(silent=True) or {}
        missing = store.validate_payload(payload)
        if missing:
            return jsonify({"ok": False, "error": "missing_fields", "fields": missing}), 400

        store.upsert_minute_payload(payload)
        return jsonify(
            {
                "ok": True,
                "machine_id": payload["machine_id"],
                "minute_bucket": payload["minute_bucket"],
            }
        )

    @app.route("/api/data")
    def get_data():
        query_date = request.args.get("date", default=date.today().strftime("%Y-%m-%d"))
        machine_id = request.args.get("machine_id") or None
        snapshot = store.build_dashboard_snapshot(query_date, machine_id=machine_id)
        return jsonify(snapshot)

    return app


app = create_app()


if __name__ == "__main__":
    host = os.environ.get("DASHBOARD_HOST", "0.0.0.0")
    port = int(os.environ.get("DASHBOARD_PORT", "5000"))
    print(f"Dashboard server started. Open http://127.0.0.1:{port} in your browser.")
    app.run(host=host, port=port)
