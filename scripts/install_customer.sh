#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${CUSTOMER_VENV:-$ROOT_DIR/.customer-venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "[install] Root directory: $ROOT_DIR"
echo "[install] Virtualenv: $VENV_DIR"

if [[ ! -d "$VENV_DIR" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

"$VENV_DIR/bin/python" -m pip install --upgrade pip
"$VENV_DIR/bin/python" -m pip install -r "$ROOT_DIR/requirements-dashboard.txt"

mkdir -p "$ROOT_DIR/runtime"

if [[ ! -f "$ROOT_DIR/.env.customer" && -f "$ROOT_DIR/.env.customer.example" ]]; then
  cp "$ROOT_DIR/.env.customer.example" "$ROOT_DIR/.env.customer"
  echo "[install] Created .env.customer from example. Please edit machine/site/API settings."
fi

echo "[install] Completed."
