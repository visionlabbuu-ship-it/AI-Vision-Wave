#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${CUSTOMER_VENV:-$ROOT_DIR/.customer-venv}"

if [[ -f "$ROOT_DIR/.env.customer" ]]; then
  # shellcheck disable=SC1090
  source "$ROOT_DIR/.env.customer"
fi

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  echo "[dashboard] Missing virtualenv at $VENV_DIR. Run scripts/install_customer.sh first."
  exit 1
fi

cd "$ROOT_DIR"
exec "$VENV_DIR/bin/python" dashboard_app.py
