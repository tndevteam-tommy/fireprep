"""Regenerate every procedural texture into assets/textures (overwrites existing files).

    python tools/textures/gen_textures.py [--out DIR] [--seed N] [--only key ...]
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")      # textures need no sound card (as tools/benchmark.py)
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame  # noqa: E402

from tools.textures.generators import ARTWORK, TEXTURE_KEYS, write_textures  # noqa: E402

DEFAULT_OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "assets", "textures")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--only", nargs="*", default=None, choices=TEXTURE_KEYS + tuple(ARTWORK), metavar="KEY",
                    help="texture keys to regenerate (default: all); one of: " + ", ".join(TEXTURE_KEYS + tuple(ARTWORK)))
    args = ap.parse_args(argv)
    pygame.init()
    keys = tuple(args.only) if args.only else TEXTURE_KEYS + tuple(ARTWORK)
    for path in write_textures(args.out, args.seed, keys):
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
