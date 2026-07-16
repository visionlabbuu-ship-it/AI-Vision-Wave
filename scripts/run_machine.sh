#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -f "$ROOT_DIR/.env.customer" ]]; then
  # shellcheck disable=SC1090
  source "$ROOT_DIR/.env.customer"
fi

if [[ -n "${MACHINE_PYTHON:-}" ]]; then
  PYTHON_CMD="$MACHINE_PYTHON"
elif [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PYTHON_CMD="$ROOT_DIR/.venv/bin/python"
elif [[ -x "$ROOT_DIR/Orin_venv/bin/python" ]]; then
  PYTHON_CMD="$ROOT_DIR/Orin_venv/bin/python"
else
  PYTHON_CMD="python3"
fi

cd "$ROOT_DIR"
exec "$PYTHON_CMD" index.py
