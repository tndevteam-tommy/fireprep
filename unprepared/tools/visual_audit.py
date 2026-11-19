"""Search the level for visual glitches - z-fighting, things through walls, the ground through a
floor, things floating, decals off their walls - and print what it finds.

    python tools/visual_audit.py [--checks zfight through_wall ...] [--limit 50] [--near 0.003]

No window is opened: the static scene is "drawn" into a recorder (``unprepared.rendering.audit``)
and the triangles and boxes examined. Exits with status 1 if anything is found (so it can gate a
build), 0 if the level is clean.
"""
from __future__ import annotations

import argparse
import collections
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

CHECKS = ("zfight", "through_wall", "ground_in_floor", "floating", "decal_off_wall")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--level", default="town")
    ap.add_argument("--checks", nargs="*", default=list(CHECKS), choices=CHECKS)
    ap.add_argument("--limit", type=int, default=50, help="findings to list (all are counted)")
    ap.add_argument("--near", type=float, default=None, help="how close two faces count as one plane (metres)")
    args = ap.parse_args(argv)

    from unprepared.config import GameSettings
    from unprepared.rendering import audit
    from unprepared.world import World, get_level

    started = time.perf_counter()
    world = World(get_level(args.level), GameSettings(echo_messages_to_stdout=False))
    triangles = None
    if "zfight" in args.checks:
        triangles = audit.record_static_scene(world)
        print(f"recorded {len(triangles)} triangles of the static scene")
    findings = audit.audit_world(world, args.checks, triangles=triangles, near=args.near if args.near is not None else audit.NEAR)
    elapsed = time.perf_counter() - started
    counts = collections.Counter(f.kind for f in findings)
    print(f"{len(findings)} finding(s) in {elapsed:.1f} s" + (": " + ", ".join(f"{k} {v}" for k, v in counts.items()) if findings else ""))
    for finding in findings[:args.limit]:
        print("  " + str(finding))
    if len(findings) > args.limit:
        print(f"  ... and {len(findings) - args.limit} more")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
