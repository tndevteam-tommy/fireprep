#!/usr/bin/env bash
# Hardcore - realistic parkour gameplay test-suite.
# Runs headless (no window / OpenGL needed). Requires Python 3.10+, numpy and pytest.
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python3}"

if ! "$PY" -c "import numpy" 2>/dev/null; then
    echo "numpy is required: $PY -m pip install -r requirements.txt" >&2
    exit 1
fi

if "$PY" -c "import pytest" 2>/dev/null; then
    exec "$PY" -m pytest tests -q "$@"
else
    echo "pytest not installed - falling back to the minimal built-in runner (pip install pytest for full output)" >&2
    exec "$PY" tests/run_fallback.py "$@"
fi
