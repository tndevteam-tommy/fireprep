#!/usr/bin/env python3
"""Cyberduke launcher.

Run the game without installing the package:

    python play.py          (or double-click this file on a desktop with Python associated)

The launcher puts ./src on the import path, checks that the runtime dependencies are
present with a readable message if they are not, and keeps the console open on a crash
so a double-click launch never vanishes silently.
"""

from __future__ import annotations

import contextlib
import os
import pathlib
import sys
import traceback

ROOT = pathlib.Path(__file__).resolve().parent
SRC = ROOT / "src"

REQUIRED = {
    "pygame": "pygame",
    "OpenGL": "PyOpenGL",
    "numpy": "numpy",
}


def _pause_if_interactive() -> None:
    """Keep a double-click-launched console window open until the user dismisses it."""
    if sys.stdin is not None and sys.stdin.isatty() and not os.environ.get("CYBERDUKE_NO_PAUSE"):
        with contextlib.suppress(EOFError):
            input("\nPress Enter to close...")


def _check_dependencies() -> list[str]:
    missing = []
    for module, package in REQUIRED.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(package)
    return missing


def main() -> int:
    if not (SRC / "cyberduke").is_dir():
        print(f"Cannot find the game sources at {SRC}", file=sys.stderr)
        _pause_if_interactive()
        return 1
    sys.path.insert(0, str(SRC))

    missing = _check_dependencies()
    if missing:
        print("Cyberduke needs the following Python packages, which are not installed:\n")
        for pkg in missing:
            print(f"    {pkg}")
        requirements = ROOT / "requirements.txt"
        print(f"\nInstall them with:\n\n    {sys.executable} -m pip install -r {requirements}\n")
        _pause_if_interactive()
        return 1

    try:
        from cyberduke.main import main as game_main

        game_main(sys.argv[1:])
    except SystemExit as exc:  # pygame.quit() path raises SystemExit(None)
        return int(exc.code or 0)
    except Exception:  # noqa: BLE001 - top-level crash handler for double-click launches
        print("\nCyberduke crashed:\n", file=sys.stderr)
        traceback.print_exc()
        _pause_if_interactive()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
