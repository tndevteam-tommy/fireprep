"""
Firestarter - Main Game Entry Point (DESIGN Section 8).

    python main.py                       play (window, audio, tutorial)
    python main.py --headless            simulation-only verification tick (no window)
    python main.py --playback automation/scripts/01_tutorial_walkthrough.json --out test_artifacts/playback
    python main.py --record my_session.json

Run `python main.py --help` for every option.
"""

import argparse
import json
import os
import sys
import types

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
if "firestarter" not in sys.modules:
    pkg = types.ModuleType("firestarter")
    pkg.__path__ = [CURR_DIR]
    pkg.__file__ = os.path.join(CURR_DIR, "__init__.py")
    sys.modules["firestarter"] = pkg
if CURR_DIR not in sys.path:
    sys.path.insert(0, CURR_DIR)

from firestarter.config import WINDOW_WIDTH, WINDOW_HEIGHT, START_HOUR, DEFAULT_SEED  # noqa: E402


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Firestarter - thermodynamics & bushcraft simulation")
    ap.add_argument("--headless", action="store_true", help="run a short simulation check without a window")
    ap.add_argument("--width", type=int, default=WINDOW_WIDTH)
    ap.add_argument("--height", type=int, default=WINDOW_HEIGHT)
    ap.add_argument("--fullscreen", action="store_true")
    ap.add_argument("--preset", default="High", choices=["Low", "Medium", "High", "Ultra"])
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--start-hour", type=float, default=START_HOUR)
    ap.add_argument("--no-audio", action="store_true")
    ap.add_argument("--no-tutorial", action="store_true")
    ap.add_argument("--fixed-dt", type=float, default=None, help="fixed simulation step per frame (deterministic)")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--playback", default=None, help="JSON playback script to drive the game")
    ap.add_argument("--out", default=os.path.join(CURR_DIR, "test_artifacts", "playback"), help="playback output dir")
    ap.add_argument("--record", default=None, help="record the input event stream to this JSON file")
    return ap.parse_args(argv)


def headless_check() -> int:
    """Simulation-only smoke test: build a fire lay, light it and burn for a minute."""
    import numpy as np
    from firestarter.game import GameState
    print("[Simulation] Headless verification: building and lighting a teepee fire...")
    st = GameState(tutorial=False)
    w = st.world
    from firestarter.simulation.inventory import make_tinder, make_kindling
    from firestarter.thermodynamics.combustion_body import orientation_to_axis
    import math
    pit = w.fire_pit_pos
    w.thermal_grid.add_body(make_tinder("cedar_shavings").to_body(pit + np.array([0, 0.045, 0]), np.zeros(3)))
    for a in range(0, 360, 72):
        it = make_kindling()
        th = math.radians(a)
        bottom = pit + np.array([0.12 * math.cos(th), it.radius, 0.12 * math.sin(th)])
        o = np.array([62.0, a + 180.0, 0.0])
        w.thermal_grid.add_body(it.to_body(bottom + orientation_to_axis(o) * it.length / 2, o))
    for i in range(60 * 60):
        if i < 120:
            w.thermal_grid.add_pilot(pit + np.array([0, 0.03, 0.03]), 0.1)
        st.step(1.0 / 60.0, move=False)
    g = w.thermal_grid
    ok = g.stats_ok = g.total_energy_produced_joules > 1e5
    print(f"[Simulation] {'Verification passed' if ok else 'FAILED'}: stage={g.fire_stage()}, core {g.core_coal_temp:.0f} C, "
          f"energy {g.total_energy_produced_joules / 1e6:.2f} MJ, credits pending {st.phone.credits.pending:.1f}")
    return 0 if ok else 1


def main(argv=None) -> int:
    args = parse_args(argv)
    print("=" * 70)
    print("  FIRESTARTER: Thermodynamics & Bushcraft Physical Simulation")
    print("  'Life will grow, fire will come, and the cycle will repeat.'")
    print("=" * 70)
    if args.headless:
        return headless_check()

    from firestarter.game import Game
    from firestarter.automation.playback import Playback, Recorder

    playback = None
    settings = {}
    if args.playback:
        playback = Playback.load(args.playback, args.out)
        settings = playback.script.get("settings", {})
    fixed_dt = settings.get("fixed_dt", args.fixed_dt)
    if playback is not None and fixed_dt is None:
        fixed_dt = 1.0 / 30.0
    game = Game(width=int(settings.get("width", args.width)), height=int(settings.get("height", args.height)),
                seed=int(settings.get("seed", args.seed)), preset=settings.get("preset", args.preset),
                fullscreen=args.fullscreen, audio=not (args.no_audio or settings.get("no_audio", False)),
                start_hour=float(settings.get("start_hour", args.start_hour)),
                tutorial=not (args.no_tutorial or settings.get("no_tutorial", False)), fixed_dt=fixed_dt)
    game.playback = playback
    if args.record:
        game.recorder = Recorder(args.record, fixed_dt or 1.0 / 60.0)
    return game.run(max_frames=args.max_frames)


if __name__ == "__main__":
    sys.exit(main())
