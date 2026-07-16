# Central Dashboard Sync And Packaging Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a separate central dashboard database and ingest API, a machine-side 1-minute sync worker with outbox retry behavior, customer launch scripts, and usage documentation.

**Architecture:** Keep `index.py` on the machine control path with its local DB, add a separate machine sync module that aggregates and pushes minute summaries, and move `dashboard_app.py` to a central-db-backed app that serves only central data. Use dedicated shell scripts and docs for customer delivery.

**Tech Stack:** Python 3.10, Flask, SQLite, pandas, requests, pytest, shell scripts

## Global Constraints

- Keep machine control-path behavior independent from dashboard/network availability.
- Dashboard reads only from its own central SQLite database.
- Sync cadence is 1 minute while the machine is running.
- Support multiple machines with idempotent upsert by `machine_id + minute_bucket`.
- Use executable shell scripts for install and launch.

---

### Task 1: Add Dashboard Data Layer And API

**Files:**
- Create: `modules/dashboard_central.py`
- Modify: `dashboard_app.py`
- Test: `tests/test_dashboard_app.py`

**Interfaces:**
- Consumes: central DB path string, ingest payload dict
- Produces: `CentralDashboardStore`, `create_app()`, `upsert_minute_payload(payload: dict) -> None`

- [ ] Write failing tests for ingest validation, idempotent upsert, and dashboard query responses.
- [ ] Run the dashboard tests and confirm they fail for the expected missing interfaces.
- [ ] Implement the central dashboard store and refactor `dashboard_app.py` to use an app factory plus ingest endpoint.
- [ ] Re-run dashboard tests until they pass.

### Task 2: Add Machine Sync Worker And Outbox

**Files:**
- Create: `modules/dashboard_sync.py`
- Modify: `modules/database.py`
- Test: `tests/test_dashboard_sync.py`

**Interfaces:**
- Consumes: local machine DB path, sync state DB path, dashboard URL, API key, machine metadata
- Produces: `MachineSummary`, `MachineSyncStore`, `MachineSyncClient`, `MachineSyncService`

- [ ] Write failing tests for minute aggregation, outbox persistence, retry marking, and old DB schema migration.
- [ ] Run the sync tests and confirm they fail for the expected missing interfaces.
- [ ] Implement minimal sync store, aggregator, API client, and local DB migration updates.
- [ ] Re-run sync tests until they pass.

### Task 3: Add Machine/Sync Launch CLIs

**Files:**
- Create: `run_dashboard_sync.py`
- Modify: `dashboard_app.py`
- Test: `tests/test_dashboard_sync.py`

**Interfaces:**
- Consumes: environment variables and CLI args
- Produces: executable CLI entry point for sync worker polling

- [ ] Write failing tests for configuration parsing and one-shot sync execution.
- [ ] Run targeted tests and confirm failure is caused by missing CLI wiring.
- [ ] Implement the CLI wrapper and reuse the sync module interfaces.
- [ ] Re-run targeted tests until they pass.

### Task 4: Add Customer Scripts And Dependency Split

**Files:**
- Create: `requirements-dashboard.txt`
- Create: `scripts/install_customer.sh`
- Create: `scripts/run_machine.sh`
- Create: `scripts/run_dashboard.sh`
- Create: `scripts/run_sync_worker.sh`

**Interfaces:**
- Consumes: repository root, shell environment variables, optional `.env.customer`
- Produces: executable customer entry points

- [ ] Add the minimal dashboard dependency file and shell scripts with safe defaults.
- [ ] Validate shell syntax with `bash -n`.
- [ ] Mark scripts executable and verify their help/usage output where applicable.

### Task 5: Add Delivery Documentation

**Files:**
- Create: `docs/CUSTOMER_DEPLOYMENT_GUIDE.md`
- Modify: `docs/TECHNICAL_DOCUMENTATION.md`
- Modify: `docs/PROJECT_SUMMARY.md`

**Interfaces:**
- Consumes: implemented runtime commands and environment variables
- Produces: customer-facing setup and operations docs

- [ ] Document machine, sync worker, and central dashboard setup.
- [ ] Document GitHub upload checklist and runtime expectations.
- [ ] Review the docs for accuracy against the implemented scripts and commands.

### Task 6: Verify End-To-End

**Files:**
- Test: `tests/test_dashboard_app.py`
- Test: `tests/test_dashboard_sync.py`

**Interfaces:**
- Consumes: all implemented code
- Produces: verified delivery summary

- [ ] Run `pytest tests/test_dashboard_app.py tests/test_dashboard_sync.py -q`.
- [ ] Run `python -m py_compile dashboard_app.py run_dashboard_sync.py modules/*.py`.
- [ ] Run `bash -n scripts/install_customer.sh scripts/run_machine.sh scripts/run_dashboard.sh scripts/run_sync_worker.sh`.
- [ ] Record residual risks and deployment assumptions in the final summary.
