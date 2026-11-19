#!/usr/bin/env bash
# ----------------------------------------------------------------------------
# Cyberduke test runner
#
#   ./run_tests.sh            # lint + format check + unit tests
#   ./run_tests.sh --fast     # unit tests only
#   ./run_tests.sh --fix      # auto-fix lint/format, then run everything
#   ./run_tests.sh --smoke    # additionally boot the real game for a few
#                             # frames under Xvfb (Linux; needs xvfb-run)
#
# Uses the currently active virtualenv if there is one; otherwise creates
# ./.venv and installs the project in editable mode with dev dependencies.
# ----------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

FAST=0; FIX=0; SMOKE=0
for arg in "$@"; do
    case "$arg" in
        --fast)  FAST=1 ;;
        --fix)   FIX=1 ;;
        --smoke) SMOKE=1 ;;
        -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
        *) echo "unknown option: $arg" >&2; exit 2 ;;
    esac
done

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    if [[ ! -d .venv ]]; then
        echo ">> creating .venv"
        python3 -m venv .venv
    fi
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

python - <<'PY' || { echo ">> installing project + dev dependencies"; pip install -q -e ".[dev]"; }
import importlib, sys
for m in ("pytest", "ruff", "pygame", "OpenGL", "numpy", "cyberduke"):
    importlib.import_module(m)
PY

status=0
if [[ $FAST -eq 0 ]]; then
    if [[ $FIX -eq 1 ]]; then
        echo ">> ruff --fix"; ruff check --fix src tests play.py
        echo ">> ruff format"; ruff format src tests play.py
    fi
    echo ">> ruff check";        ruff check src tests play.py        || status=1
    echo ">> ruff format check"; ruff format --check src tests play.py || status=1
fi

echo ">> pytest"
python -m pytest || status=1

if [[ $SMOKE -eq 1 ]]; then
    echo ">> smoke test (boot game under Xvfb for 60 frames)"
    if command -v xvfb-run >/dev/null; then
        SDL_AUDIODRIVER=dummy xvfb-run -a -s "-screen 0 1024x768x24" \
            python -m pytest tests/smoke --smoke -q || status=1
    else
        echo "   xvfb-run not found; skipping"
    fi
fi

if [[ $status -eq 0 ]]; then echo ">> ALL GREEN"; else echo ">> FAILURES"; fi
exit $status
