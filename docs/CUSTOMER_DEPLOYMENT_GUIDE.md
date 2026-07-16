# Customer Deployment Guide

## 1. Files To Prepare Before Uploading To GitHub

- `index.py`
- `dashboard_app.py`
- `run_dashboard_sync.py`
- `modules/`
- `templates/`
- `scripts/`
- `requirements-dashboard.txt`
- `.env.customer.example`
- `docs/CUSTOMER_DEPLOYMENT_GUIDE.md`
- `docs/PROJECT_SUMMARY.md`
- `docs/TECHNICAL_DOCUMENTATION.md`

Keep hardware assets with the machine runtime checkout:

- YOLO model files such as `yolov26s_fixed.pt`
- `Model/` directory for spectrum ML assets
- `calibration_data.json`
- `offsets.json`
- any robot/camera device permissions required on the target machine

## 2. Deployment Layout

Recommended split:

- Each sorting machine runs:
  - `scripts/run_machine.sh`
  - `scripts/run_sync_worker.sh`
- The central web server runs:
  - `scripts/install_customer.sh`
  - `scripts/run_dashboard.sh`

## 3. Central Dashboard Setup

1. Clone the repository.
2. Run:

```bash
chmod +x scripts/*.sh
./scripts/install_customer.sh
```

3. Edit `.env.customer`:

```bash
DASHBOARD_API_KEY=strong-secret
CENTRAL_DASHBOARD_DB=central_dashboard.db
DASHBOARD_HOST=0.0.0.0
DASHBOARD_PORT=5000
```

4. Start the dashboard:

```bash
./scripts/run_dashboard.sh
```

5. Open `http://<server-ip>:5000`.

## 4. Machine Setup

1. Copy the full machine runtime repository to the Jetson.
2. Ensure the machine runtime dependencies, model files, calibration files, and hardware permissions are already provisioned.
3. Edit `.env.customer` with the machine identity and dashboard endpoint:

```bash
DASHBOARD_URL=http://dashboard-server:5000
DASHBOARD_API_KEY=strong-secret
MACHINE_ID=SORTER-01
MACHINE_NAME=Sorter 01
SITE_NAME=Factory A
LINE_NAME=Line 1
MACHINE_DB_PATH=system_data.db
SYNC_DB_PATH=machine_sync.db
SYNC_INTERVAL_SECONDS=60
```

4. Start the machine UI:

```bash
./scripts/run_machine.sh
```

5. In a second terminal or launcher shortcut, start the sync worker:

```bash
./scripts/run_sync_worker.sh
```

## 5. One-Shot Sync Test

Use this to validate API connectivity without waiting for the loop:

```bash
./scripts/run_sync_worker.sh --once
```

## 6. Double-Click Behavior

On Linux desktop environments, mark each script executable and allow launching executable text files. The customer can then double-click:

- `run_machine.sh`
- `run_dashboard.sh`
- `run_sync_worker.sh`

This is the shell-script equivalent of a packaged `.exe`.

## 7. Operational Notes

- `index.py` keeps running even if dashboard sync fails.
- Sync failures remain queued in `machine_sync.db` and retry on the next cycle.
- The dashboard reads only from `central_dashboard.db`.
- Use a reverse proxy or firewall rules before exposing the dashboard beyond the local network.

## 8. GitHub Upload Checklist

- Remove local-only archives, datasets, and temporary files that should not be versioned.
- Confirm model and calibration files are included only if the repository is allowed to contain them.
- Verify `.env.customer` is not committed with real secrets.
- Commit scripts with executable permissions.
- Run the verification commands listed below before pushing.

```bash
./.venv/bin/python -m pytest tests/test_dashboard_app.py tests/test_dashboard_sync.py -q
./.venv/bin/python -m py_compile dashboard_app.py run_dashboard_sync.py modules/*.py
bash -n scripts/install_customer.sh scripts/run_machine.sh scripts/run_dashboard.sh scripts/run_sync_worker.sh
```
