"""Dump the captured frames of one or more scenarios to PNG files for visual review.

    python -m tests.parity.render_gallery [--impl new|legacy] [scenario ...]

Output goes to tests/parity/_artifacts/gallery/<impl>_<scenario>_f<frame>.png
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import shutil

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_DIR)

from tests.parity.display import ARTIFACT_DIR, run_scenario_subprocess   # noqa: E402
from tests.parity.scenarios import SCENARIOS       # noqa: E402

ASSET_DIR = os.path.join(PROJECT_DIR, "assets", "textures")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", default="new", choices=("new", "legacy"))
    ap.add_argument("scenarios", nargs="*")
    args = ap.parse_args()
    names = args.scenarios or sorted(SCENARIOS)
    out_dir = os.path.join(ARTIFACT_DIR, "gallery")
    os.makedirs(out_dir, exist_ok=True)
    for name in names:
        with tempfile.TemporaryDirectory(prefix=f"gallery_{name}_") as workdir:     # removed once the run is read
            for f in os.listdir(ASSET_DIR):
                shutil.copy(os.path.join(ASSET_DIR, f), workdir)
            result = run_scenario_subprocess(args.impl, name, workdir)
        for fr in result.frames:
            if fr.png:
                path = os.path.join(out_dir, f"{args.impl}_{name}_f{fr.index:04d}.png")
                with open(path, "wb") as fh:
                    fh.write(fr.png)
                print(path)
        if result.error:
            print(f"  !! {args.impl} crashed: {result.error}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
