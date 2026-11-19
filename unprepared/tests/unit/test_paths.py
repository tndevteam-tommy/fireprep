"""The per-user data directory: platform conventions, the env override, legacy migration."""
from __future__ import annotations

import json
import os

from unprepared.config import GameSettings
from unprepared.paths import (DATA_DIR_ENV, GRAPHICS_FILE_NAME, SAVE_FILE_NAME, default_save_file, migrate_legacy_saves,
                              user_data_dir)


def test_platform_conventions():
    home = os.path.expanduser("~")
    assert user_data_dir(platform="win32", environ={"APPDATA": "/appdata"}) == os.path.join("/appdata", "Unprepared")
    assert user_data_dir(platform="win32", environ={}) == os.path.join(home, "AppData", "Roaming", "Unprepared")
    assert user_data_dir(platform="darwin", environ={}) == os.path.join(home, "Library", "Application Support", "Unprepared")
    assert user_data_dir(platform="linux", environ={}) == os.path.join(home, ".local", "share", "unprepared")
    assert user_data_dir(platform="linux", environ={"XDG_DATA_HOME": "/xdg"}) == os.path.join("/xdg", "unprepared")


def test_env_override_wins_everywhere(tmp_path):
    env = {DATA_DIR_ENV: str(tmp_path / "portable"), "APPDATA": "/appdata"}
    for platform in ("win32", "darwin", "linux"):
        assert user_data_dir(platform=platform, environ=env) == str(tmp_path / "portable")


def test_default_save_file_and_settings_resolution(tmp_path, monkeypatch):
    monkeypatch.setenv(DATA_DIR_ENV, str(tmp_path))
    assert default_save_file() == str(tmp_path / SAVE_FILE_NAME)
    s = GameSettings()
    assert s.save_file is None
    assert s.resolved_save_file() == str(tmp_path / SAVE_FILE_NAME)
    assert s.resolved_save_file("/explicit/s.json") == "/explicit/s.json"
    assert GameSettings(save_file="/from/settings.json").resolved_save_file() == "/from/settings.json"


def test_game_and_graphics_use_the_data_directory(tmp_path, monkeypatch):
    from unprepared.game import Game
    from unprepared.graphics import graphics_settings_path
    monkeypatch.setenv(DATA_DIR_ENV, str(tmp_path / "data"))
    g = Game(settings=GameSettings(echo_messages_to_stdout=False))
    assert g.saves.filename == str(tmp_path / "data" / SAVE_FILE_NAME)
    assert g.saves.save_dir == str(tmp_path / "data" / "saves")
    assert g.save_game() and os.path.exists(tmp_path / "data" / SAVE_FILE_NAME)
    assert graphics_settings_path(g.saves.filename) == str(tmp_path / "data" / GRAPHICS_FILE_NAME)


def test_legacy_saves_are_copied_once(tmp_path):
    legacy, data = tmp_path / "project", tmp_path / "data"
    legacy.mkdir(); (legacy / "saves").mkdir()
    (legacy / SAVE_FILE_NAME).write_text(json.dumps({"version": 6}))
    (legacy / GRAPHICS_FILE_NAME).write_text(json.dumps({"fov": 70}))
    (legacy / "saves" / "save-1.json").write_text("{}")
    (legacy / "saves" / "notes.txt").write_text("ignored")
    copied = migrate_legacy_saves(str(legacy), str(data))
    assert sorted(os.path.basename(p) for p in copied) == [GRAPHICS_FILE_NAME, "save-1.json", SAVE_FILE_NAME]
    assert (data / SAVE_FILE_NAME).exists() and (data / "saves" / "save-1.json").exists()
    assert (legacy / SAVE_FILE_NAME).exists()                      # originals untouched
    (legacy / "saves" / "save-2.json").write_text("{}")
    assert migrate_legacy_saves(str(legacy), str(data)) == []      # the data directory has saves now: never again


def test_migration_is_a_no_op_without_legacy_files(tmp_path):
    assert migrate_legacy_saves(str(tmp_path / "nothing"), str(tmp_path / "data")) == []
    assert not (tmp_path / "data").exists()
