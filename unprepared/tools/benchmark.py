"""Measure rendering performance per graphics preset and print a report.

    python tools/benchmark.py [--frames 60] [--presets low medium high] [--json out.json]

Opens the game window, renders a fixed set of camera positions at each preset
and reports average / worst frame times per phase, plus the preset the game
would pick automatically on this machine.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

VIEWS = (  # (x, z, yaw, pitch, hour)
    (-12.0, -6.0, 0.0, 0.0, 11),
    (-2.0, 6.0, 55.0, 4.0, 11),
    (10.0, 6.0, 0.0, 0.0, 22),
    (30.0, 10.0, 45.0, 0.0, 22),
)


def run_benchmark(frames: int, presets) -> dict:
    from unprepared.app import PygameApp
    from unprepared.config import GameSettings
    from unprepared.graphics import GraphicsSettings

    with tempfile.TemporaryDirectory(prefix="unprepared_bench_") as workdir:     # scratch saves, removed afterwards
        app = PygameApp(GameSettings(echo_messages_to_stdout=False, random_seed=1), save_file=os.path.join(workdir, "s.json"),
                        graphics=GraphicsSettings(), skip_menu=True, auto_quality=False)
        try:
            return _measure(app, frames, presets)
        finally:
            app.shutdown()


def _measure(app, frames: int, presets) -> dict:
    """Render every view ``frames`` times at each preset; per-preset frame statistics."""
    import pygame
    from unprepared.core.math3d import Vector3D
    from unprepared.perf import FrameProfiler, recommend_quality

    game = app.game
    results = {"renderer": app.renderer.renderer_name, "presets": {}}
    for preset in presets:
        app.graphics.apply_preset(preset)
        app.rebuild_renderer()
        app.renderer.prebake(game)                   # (frames are measured, not the baking of the town)
        profiler = FrameProfiler(history=frames * len(VIEWS))
        for (x, z, yaw, pitch, hour) in VIEWS:
            game.world.clock.hour = hour
            game.player.position = Vector3D(x, 0.0, z)
            game.player.rotation_y, game.player.rotation_x = yaw, pitch
            game.player.has_lit_torch = hour >= 20
            game.update(0.016)
            for _ in range(frames):
                with profiler.phase("render3d"):
                    app.renderer.render_scene_only(game)
                with profiler.phase("hud"):
                    app.renderer.hud.render(game)
                with profiler.phase("flip"):
                    pygame.display.flip()
                profiler.end_frame()
        results["presets"][preset] = profiler.stats()
    high_ms = results["presets"].get("high", {}).get("frame_ms", 0.0)
    results["recommended"] = recommend_quality(high_ms, app.renderer.renderer_name)
    return results


def format_report(results: dict) -> str:
    lines = [f"Renderer: {results['renderer']}", f"{'preset':<8} {'avg ms':>8} {'worst ms':>9} {'fps':>6}   phases"]
    for preset, stats in results["presets"].items():
        phases = "  ".join(f"{k} {v:.1f}" for k, v in stats["phases"].items())
        lines.append(f"{preset:<8} {stats['frame_ms']:>8.2f} {stats['worst_ms']:>9.2f} {stats['fps']:>6.1f}   {phases}")
    lines.append(f"Recommended preset for this machine: {results['recommended']}")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames", type=int, default=60, help="frames per camera view per preset")
    ap.add_argument("--presets", nargs="*", default=["low", "medium", "high"])
    ap.add_argument("--json", default=None, help="also write the raw results to this file")
    args = ap.parse_args(argv)
    results = run_benchmark(args.frames, args.presets)
    print(format_report(results))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
