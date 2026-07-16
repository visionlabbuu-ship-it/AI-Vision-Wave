#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import time

from modules.dashboard_sync import MachineSyncClient, MachineSyncService


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Push 1-minute local machine summaries to the central dashboard."
    )
    parser.add_argument("--machine-db", default=os.environ.get("MACHINE_DB_PATH", "system_data.db"))
    parser.add_argument("--sync-db", default=os.environ.get("SYNC_DB_PATH", "machine_sync.db"))
    parser.add_argument("--dashboard-url", default=os.environ.get("DASHBOARD_URL", "http://127.0.0.1:5000"))
    parser.add_argument("--api-key", default=os.environ.get("DASHBOARD_API_KEY", ""))
    parser.add_argument("--machine-id", default=os.environ.get("MACHINE_ID", "SORTER-01"))
    parser.add_argument("--machine-name", default=os.environ.get("MACHINE_NAME", "Sorter 01"))
    parser.add_argument("--site-name", default=os.environ.get("SITE_NAME", "Factory"))
    parser.add_argument("--line-name", default=os.environ.get("LINE_NAME", "Line 1"))
    parser.add_argument("--interval-seconds", type=int, default=int(os.environ.get("SYNC_INTERVAL_SECONDS", "60")))
    parser.add_argument("--once", action="store_true", help="Run one sync cycle and exit.")
    return parser


def build_service(args: argparse.Namespace) -> MachineSyncService:
    client = MachineSyncClient(args.dashboard_url, args.api_key)
    return MachineSyncService(
        machine_db_path=args.machine_db,
        sync_db_path=args.sync_db,
        machine_id=args.machine_id,
        machine_name=args.machine_name,
        site_name=args.site_name,
        line_name=args.line_name,
        dashboard_client=client,
    )


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    service = build_service(args)

    if args.once:
        result = service.run_once()
        print(result)
        return 0

    print(
        f"Starting dashboard sync worker for {args.machine_id} -> {args.dashboard_url} "
        f"(interval={args.interval_seconds}s)"
    )
    while True:
        result = service.run_once()
        print(result)
        time.sleep(max(1, args.interval_seconds))


if __name__ == "__main__":
    raise SystemExit(main())
