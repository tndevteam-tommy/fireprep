#!/usr/bin/env bash
# Run the whole test suite. On headless Linux the display-backed tests spawn
# xvfb-run automatically; set UNPREPARED_TEST_DISPLAY=0 to skip them.
#   ./run_tests.sh            -> everything
#   ./run_tests.sh fast       -> headless tests only
set -euo pipefail
cd "$(dirname "$0")"
export SDL_AUDIODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1
if [[ "${1:-}" == "fast" ]]; then
    exec python3 -m pytest -m "not display" "${@:2}"
fi
exec python3 -m pytest "${@}"
