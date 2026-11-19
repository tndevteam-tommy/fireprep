"""Subprocess entry point: run one scenario with one implementation, pickle the result.

    python -m tests.parity.run_scenario --impl legacy|new --scenario NAME --workdir DIR --out FILE
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", choices=("legacy", "new"), required=True)
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mask", action="append", default=[], help="x1,y1,x2,y2 window rect excluded from pixel hashes")
    args = ap.parse_args()

    from tests.parity import harness
    from tests.parity.scenarios import ALL_SCENARIOS
    scenario = ALL_SCENARIOS[args.scenario]()
    masks = [tuple(int(v) for v in m.split(",")) for m in args.mask]
    runner = harness.run_legacy if args.impl == "legacy" else harness.run_new
    result = runner(scenario, args.workdir, mask_rects=masks)
    with open(args.out, "wb") as f:
        pickle.dump(result, f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
