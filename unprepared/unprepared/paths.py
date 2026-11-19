"""Where the game keeps its files: the per-user data directory.

Saves (``savegame.json`` and the ``saves/`` slots) and ``graphics.json`` live in
one directory chosen by platform convention, never next to the code or in
whatever directory the game happens to be launched from:

* Windows  ``%APPDATA%\\Unprepared``          (``~\\AppData\\Roaming\\Unprepared``)
* macOS    ``~/Library/Application Support/Unprepared``
* Linux    ``$XDG_DATA_HOME/unprepared``      (``~/.local/share/unprepared``)

``UNPREPARED_DATA_DIR`` overrides the location (portable installs, tests).
Saves an older version left beside ``main.py`` are copied in on first use
(:func:`migrate_legacy_saves`).
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from typing import List, Optional

APP_NAME = "Unprepared"
DATA_DIR_ENV = "UNPREPARED_DATA_DIR"
SAVE_FILE_NAME = "savegame.json"
SAVES_SUBDIR = "saves"
GRAPHICS_FILE_NAME = "graphics.json"


def user_data_dir(app_name: str = APP_NAME, platform: Optional[str] = None, environ: Optional[dict] = None) -> str:
    """The per-user directory for this app (not created). ``platform``/``environ`` are
    injectable for tests; they default to the running system."""
    env = os.environ if environ is None else environ
    override = env.get(DATA_DIR_ENV)
    if override:
        return os.path.abspath(os.path.expanduser(override))
    system = sys.platform if platform is None else platform
    home = os.path.expanduser("~")
    if system.startswith("win"):
        base = env.get("APPDATA") or os.path.join(home, "AppData", "Roaming")
        return os.path.join(base, app_name)
    if system == "darwin":
        return os.path.join(home, "Library", "Application Support", app_name)
    base = env.get("XDG_DATA_HOME") or os.path.join(home, ".local", "share")
    return os.path.join(base, app_name.lower())


def default_save_file() -> str:
    """Full path of the quick-save file in the user data directory."""
    return os.path.join(user_data_dir(), SAVE_FILE_NAME)


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def migrate_legacy_saves(legacy_dir: str, data_dir: str) -> List[str]:
    """Copy a quick save, named saves and graphics settings that an older version kept in
    ``legacy_dir`` (the project directory) into ``data_dir`` - only files that do not
    exist there yet, and only when the data directory holds no saves of its own.
    Returns the paths written. The originals are left in place."""
    copied: List[str] = []
    if os.path.exists(os.path.join(data_dir, SAVE_FILE_NAME)) or os.path.isdir(os.path.join(data_dir, SAVES_SUBDIR)):
        return copied
    candidates = []
    if os.path.isfile(os.path.join(legacy_dir, SAVE_FILE_NAME)):
        candidates.append((os.path.join(legacy_dir, SAVE_FILE_NAME), os.path.join(data_dir, SAVE_FILE_NAME)))
    if os.path.isfile(os.path.join(legacy_dir, GRAPHICS_FILE_NAME)):
        candidates.append((os.path.join(legacy_dir, GRAPHICS_FILE_NAME), os.path.join(data_dir, GRAPHICS_FILE_NAME)))
    legacy_slots = os.path.join(legacy_dir, SAVES_SUBDIR)
    if os.path.isdir(legacy_slots):
        for name in sorted(os.listdir(legacy_slots)):
            if name.lower().endswith(".json"):
                candidates.append((os.path.join(legacy_slots, name), os.path.join(data_dir, SAVES_SUBDIR, name)))
    for src, dst in candidates:
        if os.path.exists(dst):
            continue
        try:
            ensure_dir(os.path.dirname(dst))
            shutil.copy2(src, dst)
            copied.append(dst)
        except OSError:
            continue
    return copied


def write_json(path: str, data: dict, indent: int = 4) -> str:
    """Write ``data`` to ``path`` atomically: to a temporary file beside it first, then renamed
    into place (``os.replace``), so a crash or a full disk mid-write never leaves a truncated
    file where a good save (or ``graphics.json``) used to be."""
    path = os.path.abspath(path)
    ensure_dir(os.path.dirname(path))
    tmp = f"{path}.tmp"
    try:
        with open(tmp, "w") as f:
            json.dump(data, f, indent=indent)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
    return path
