# Central Dashboard Sync Design

**Goal**

Separate the machine-control runtime from the customer-facing web dashboard so each sorting machine can keep running independently while pushing 1-minute summary data to a central dashboard database.

**Context**

- `index.py` is the machine runtime and must remain resilient when the network or dashboard is unavailable.
- `dashboard_app.py` is the web dashboard and should read only from its own central database.
- Multiple machines may report to the same dashboard.
- The customer needs executable shell scripts for install and runtime startup, plus usage documentation suitable for GitHub delivery.

## Requirements

1. Each machine keeps its own local runtime database for control-path logging.
2. The central dashboard uses a separate database owned by `dashboard_app.py`.
3. Each running machine syncs summary data every 1 minute.
4. Sync failures must not interrupt `index.py`.
5. The central dashboard must support data from multiple machines.
6. The repository must include customer-facing shell scripts for install and launch.
7. The repository must include docs covering setup, launch, and operations.

## Architecture

The system is split into three responsibilities:

- `Machine Runtime`: `index.py` plus local SQLite data used by the sorter, robot, and sensor flow.
- `Machine Sync Worker`: a separate worker that reads local machine data, aggregates a 1-minute summary, stores pending uploads in an outbox, and pushes payloads to the central dashboard API.
- `Central Dashboard`: `dashboard_app.py` with a central SQLite database that stores machine registry, minute summaries, and live machine status.

This keeps control-path work independent from reporting-path work. If the central API is down, the machine still sorts and records locally.

## Data Model

### Machine-side

- Existing local detection database remains the source for per-object events.
- Add a separate SQLite outbox database for sync state to avoid risky schema coupling with the production detection DB.

Machine outbox tables:

- `sync_state`
  - `machine_id TEXT PRIMARY KEY`
  - `last_synced_minute TEXT`
  - `updated_at TEXT`
- `sync_outbox`
  - `id INTEGER PRIMARY KEY AUTOINCREMENT`
  - `machine_id TEXT NOT NULL`
  - `minute_bucket TEXT NOT NULL`
  - `payload_json TEXT NOT NULL`
  - `status TEXT NOT NULL`
  - `retry_count INTEGER NOT NULL DEFAULT 0`
  - `last_error TEXT`
  - `created_at TEXT NOT NULL`
  - `sent_at TEXT`
  - unique key on `(machine_id, minute_bucket)`

### Central dashboard

- `machines`
  - `machine_id TEXT PRIMARY KEY`
  - `machine_name TEXT`
  - `site_name TEXT`
  - `line_name TEXT`
  - `is_active INTEGER NOT NULL DEFAULT 1`
  - `last_seen_at TEXT`
  - `last_status TEXT`
- `minute_stats`
  - `machine_id TEXT NOT NULL`
  - `minute_bucket TEXT NOT NULL`
  - `session_id TEXT`
  - `detected_total INTEGER NOT NULL`
  - `sorted_total INTEGER NOT NULL`
  - `glass_count INTEGER NOT NULL`
  - `metal_count INTEGER NOT NULL`
  - `paper_count INTEGER NOT NULL`
  - `plastic_count INTEGER NOT NULL`
  - `unknown_count INTEGER NOT NULL`
  - `avg_height_cm REAL`
  - `last_updated_at TEXT NOT NULL`
  - unique key on `(machine_id, minute_bucket)`
- `machine_live_status`
  - `machine_id TEXT PRIMARY KEY`
  - `session_id TEXT`
  - `status TEXT NOT NULL`
  - `active_objects INTEGER NOT NULL`
  - `last_error TEXT`
  - `last_sync_at TEXT NOT NULL`
  - `payload_updated_at TEXT NOT NULL`
- `sync_audit`
  - `id INTEGER PRIMARY KEY AUTOINCREMENT`
  - `machine_id TEXT NOT NULL`
  - `minute_bucket TEXT NOT NULL`
  - `received_at TEXT NOT NULL`
  - `result TEXT NOT NULL`
  - `message TEXT`

## API

The dashboard exposes a machine-ingest endpoint:

- `POST /api/ingest/minute-summary`

Request body:

- `machine_id`
- `machine_name`
- `site_name`
- `line_name`
- `minute_bucket`
- `session_id`
- `stats`
- `live_status`

Behavior:

- Validate required fields.
- Upsert `machines`, `minute_stats`, and `machine_live_status`.
- Insert an audit row.
- Return success JSON with the effective `machine_id` and `minute_bucket`.

Authentication is API-key based via request header and environment variable.

## Query Model

The web UI reads only from the central database:

- KPI totals from `minute_stats`
- Live machine table from `machine_live_status`
- Pie chart by material from aggregated class columns in `minute_stats`
- Trend chart from grouped `minute_stats` buckets

The UI must support filtering by date and machine.

## Error Handling

- Machine sync errors are logged into `sync_outbox.last_error`.
- Failed minute payloads remain queued and retry on the next worker cycle.
- Duplicate minute uploads are treated as idempotent updates.
- Dashboard API validation errors return `400`.
- Authentication failures return `401`.
- Control-path runtime never blocks on sync.

## Packaging

Customer-facing shell scripts:

- `scripts/install_customer.sh`
- `scripts/run_machine.sh`
- `scripts/run_dashboard.sh`
- `scripts/run_sync_worker.sh`

These scripts should:

- Create or reuse a Python virtual environment when appropriate.
- Install the dashboard-specific dependencies with a clean requirements file.
- Start the correct program with environment variables loaded from `.env` or exported shell variables.

## Documentation

Add customer-facing docs covering:

- Machine runtime launch
- Sync worker launch
- Central dashboard launch
- Required environment variables
- API key setup
- GitHub upload checklist

## Testing

- Unit tests for dashboard DB upsert and query shaping
- API tests for ingest endpoint validation and idempotency
- Unit tests for machine summary aggregation and outbox retry flow
- Schema migration test for `DatabaseManager` on older local DB files

## Non-Goals

- Replacing the machine runtime UI
- Replacing SQLite with PostgreSQL
- Streaming sub-minute live events
- Full installer packaging beyond executable shell scripts
