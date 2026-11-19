#!/usr/bin/env bash
# =====================================================================================
#  Firestarter - the all-in-one litmus test.
#
#  Proves the game works *and* looks right:
#    1. unit        physics & chemistry, bushcraft, forestry, economy, audio synthesis,
#                   UI overlay, meshes/particles/lighting, gameplay through real input
#    2. headless    `main.py --headless` simulation smoke test
#    3. system      entry points, live audio mix captured from SDL (disk driver)
#    4. renderer    OpenGL 3.3 renderer: every pass, every preset, resize, menus
#    5. playback    scripted play sessions (real keyboard/mouse events) with gameplay
#                   assertions and screenshots -> test_artifacts/playback/
#    6. visual      objective image checks on every screenshot (exposure, colour, detail,
#                   night/day/season/fire expectations, animation) + HTML report
#
#  Usage:  ./run_tests.sh [options]
#     --quick          fast pass: skip slow unit tests, run only playback scripts 04 & 07
#     --scripts LIST   comma separated playback scripts to run (e.g. 01,05)
#     --no-gl          skip everything that needs an OpenGL display (stages 4-6)
#     --no-playback    skip stage 5 (visual checks then use existing screenshots, if any)
#     --fail-fast      stop at the first failing stage (default: run every stage)
#     --clean          remove the virtualenv and test_artifacts afterwards
#     -h | --help
#
#  Environment: FIRESTARTER_VENV (default ./.venv), FIRESTARTER_ARTIFACTS (default
#  ./test_artifacts), PYTHON (default python3).  An X display is used if $DISPLAY is set,
#  otherwise the GL stages run inside `xvfb-run` (install the `xvfb` package).
#  Exit status is non-zero if any stage fails.
# =====================================================================================
set -u
cd "$(dirname "$0")"
ROOT="$(pwd)"
VENV="${FIRESTARTER_VENV:-$ROOT/.venv}"
ART="${FIRESTARTER_ARTIFACTS:-$ROOT/test_artifacts}"
PY="${PYTHON:-python3}"
QUICK=0; NO_GL=0; NO_PLAYBACK=0; CLEAN=0; FAIL_FAST=0; SCRIPTS="${FIRESTARTER_SCRIPTS:-}"

while [ $# -gt 0 ]; do
  case "$1" in
    --quick) QUICK=1 ;;
    --no-gl) NO_GL=1 ;;
    --no-playback) NO_PLAYBACK=1 ;;
    --fail-fast) FAIL_FAST=1 ;;
    --clean) CLEAN=1 ;;
    --scripts) shift; SCRIPTS="$1" ;;
    -h|--help) awk 'NR>1 {print} NR>2 && /^# =====/ {exit}' "$0"; exit 0 ;;
    *) echo "unknown option: $1"; exit 2 ;;
  esac
  shift
done
[ $QUICK -eq 1 ] && [ -z "$SCRIPTS" ] && SCRIPTS="04,07"

BOLD=$'\e[1m'; GREEN=$'\e[32m'; RED=$'\e[31m'; YEL=$'\e[33m'; DIM=$'\e[2m'; RST=$'\e[0m'
[ -t 1 ] || { BOLD=""; GREEN=""; RED=""; YEL=""; DIM=""; RST=""; }
declare -a NAMES RESULTS TIMES
FAILED=0
T_START=$(date +%s)

banner() { echo; echo "${BOLD}== $1 ${RST}${DIM}$2${RST}"; }
record() {  # name status seconds
  NAMES+=("$1"); RESULTS+=("$2"); TIMES+=("$3")
  [ "$2" = "FAIL" ] && FAILED=1
}
run_stage() {  # name description command...
  local name="$1" desc="$2"; shift 2
  if [ $FAIL_FAST -eq 1 ] && [ $FAILED -eq 1 ]; then record "$name" SKIP 0; return; fi
  banner "$name" "$desc"
  local t0=$(date +%s)
  if "$@"; then record "$name" PASS $(( $(date +%s) - t0 )); else record "$name" FAIL $(( $(date +%s) - t0 )); fi
}

# ------------------------------------------------------------------ environment
banner "setup" "virtualenv $VENV"
if [ ! -x "$VENV/bin/python" ]; then
  "$PY" -m venv "$VENV" || { echo "${RED}could not create virtualenv${RST}"; exit 1; }
fi
# shellcheck disable=SC1091
. "$VENV/bin/activate"
python -m pip install -q --disable-pip-version-check -r requirements-dev.txt || { echo "${RED}pip install failed${RST}"; exit 1; }
python -c "import sys; print('python', sys.version.split()[0])"
mkdir -p "$ART"
export FIRESTARTER_ARTIFACTS="$ART"
export SDL_AUDIODRIVER="${SDL_AUDIODRIVER:-dummy}"
export PYTHONDONTWRITEBYTECODE=1

GLRUN=()
if [ $NO_GL -eq 0 ]; then
  if [ -n "${DISPLAY:-}" ] || [ -n "${WAYLAND_DISPLAY:-}" ]; then
    echo "display: ${DISPLAY:-$WAYLAND_DISPLAY}"
  elif command -v xvfb-run >/dev/null 2>&1; then
    GLRUN=(xvfb-run -a -s "-screen 0 1280x720x24")
    echo "display: virtual framebuffer (xvfb-run)"
  else
    echo "${YEL}no display and no xvfb-run: GL stages will be skipped (install xvfb)${RST}"
    NO_GL=1
  fi
fi

PYTEST=(python -m pytest -p no:cacheprovider -q -rfE)
UNIT_ARGS=(tests/unit)
[ $QUICK -eq 1 ] && UNIT_ARGS+=(-m "not slow")

# ------------------------------------------------------------------ stages
run_stage "1-unit" "physics, bushcraft, forestry, economy, audio, UI, rendering CPU side, gameplay via input" \
  "${PYTEST[@]}" "${UNIT_ARGS[@]}" --junitxml="$ART/junit-1-unit.xml"

run_stage "2-headless" "python main.py --headless" python main.py --headless

run_stage "3-system" "entry points + live audio mix captured from SDL" \
  "${PYTEST[@]}" tests/integration/test_entrypoints.py tests/integration/test_audio_capture.py --junitxml="$ART/junit-3-system.xml"

if [ $NO_GL -eq 0 ]; then
  run_stage "4-renderer" "OpenGL passes, presets, resize, menus" \
    "${GLRUN[@]}" "${PYTEST[@]}" tests/integration/test_render_engine.py --junitxml="$ART/junit-4-renderer.xml"
  if [ $NO_PLAYBACK -eq 0 ]; then
    rm -rf "$ART/playback"
    FIRESTARTER_SCRIPTS="$SCRIPTS" run_stage "5-playback" "scripted play sessions ${SCRIPTS:-(all)} -> $ART/playback" \
      "${GLRUN[@]}" "${PYTEST[@]}" tests/integration/test_playback.py --junitxml="$ART/junit-5-playback.xml"
  else
    record "5-playback" SKIP 0
  fi
  run_stage "6-visual" "image analysis of every screenshot" \
    "${PYTEST[@]}" tests/visual --junitxml="$ART/junit-6-visual.xml"
else
  record "4-renderer" SKIP 0; record "5-playback" SKIP 0; record "6-visual" SKIP 0
fi

python -m tests.visual.report "$ART" >/dev/null 2>&1 && REPORT="$ART/report.html" || REPORT=""

# ------------------------------------------------------------------ summary
echo
echo "${BOLD}==================== Firestarter test summary ====================${RST}"
for i in "${!NAMES[@]}"; do
  case "${RESULTS[$i]}" in
    PASS) c=$GREEN ;; FAIL) c=$RED ;; *) c=$YEL ;;
  esac
  printf "  %-12s %s%-5s%s %5ss\n" "${NAMES[$i]}" "$c" "${RESULTS[$i]}" "$RST" "${TIMES[$i]}"
done
echo "  total time   $(( $(date +%s) - T_START ))s"
[ -n "$REPORT" ] && echo "  report       $REPORT"
[ -d "$ART/playback" ] && echo "  screenshots  $ART/playback/ ($(ls "$ART"/playback/*.png 2>/dev/null | wc -l) frames)"
if [ $FAILED -eq 0 ]; then echo "${GREEN}${BOLD}ALL STAGES PASSED${RST}"; else echo "${RED}${BOLD}SOME STAGES FAILED${RST}"; fi

if [ $CLEAN -eq 1 ]; then
  deactivate 2>/dev/null || true
  rm -rf "$VENV" "$ART"
  find "$ROOT" -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null
fi
exit $FAILED
