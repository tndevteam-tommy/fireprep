"""Named save slots: listing (newest first), labels, new slots, quick save, loading a chosen file."""
import json
import os
import time

import pytest

from unprepared.config import GameSettings
from unprepared.game import Game
from unprepared.world.save import SaveInfo, SaveManager


@pytest.fixture
def game(tmp_path):
    return Game(settings=GameSettings(echo_messages_to_stdout=False, random_seed=2), save_file=str(tmp_path / "savegame.json"))


def test_no_saves_yet(game):
    assert game.saves.list_saves() == [] and game.saves.latest() is None and not game.saves.has_saves()


def test_quick_save_and_named_slots_are_listed_newest_first(game, tmp_path):
    game.player.gold = 12.0
    game.save_game()                                          # quick save
    os.utime(game.saves.filename, (time.time() - 100, time.time() - 100))
    game.player.gold = 34.0
    path = game.saves.save_new(game.world, game.player)      # a named slot
    saves = game.saves.list_saves()
    assert [s.quick for s in saves] == [False, True]          # newest first
    assert saves[0].path == path and saves[0].name.startswith("save-")
    assert saves[1].name == "Quick Save" and saves[1].path == os.path.abspath(game.saves.filename)
    assert "34g" in saves[0].label and "12g" in saves[1].label and "Day 1 07:00" in saves[0].label
    assert game.saves.latest() == saves[0]
    assert os.path.dirname(path) == str(tmp_path / "saves")


def test_new_save_paths_never_collide(game):
    a = game.saves.new_save_path()
    open(a, "w").write("{}")
    b = game.saves.new_save_path()
    assert a != b and os.path.dirname(a) == os.path.dirname(b)


def test_load_a_chosen_save(game):
    game.player.gold = 1.0
    first = game.saves.save_new(game.world, game.player)
    game.player.gold = 2.0
    game.save_game()
    assert game.load_game(first) and game.player.gold == 1.0
    assert game.load_game() and game.player.gold == 2.0        # default: the quick save


def test_describe_handles_old_and_broken_files(tmp_path):
    old = tmp_path / "old.json"
    old.write_text(json.dumps({"version": 3, "day_count": 2, "simulated_hour": 14, "player": {"gold": 7.0}}))
    label = SaveManager.describe(str(old))
    assert label.startswith("Day 3 14:00 - 7g - ")
    broken = tmp_path / "broken.json"
    broken.write_text("not json")
    assert SaveManager.describe(str(broken)) == "broken.json (unreadable)"      # listed, flagged, never loaded


@pytest.mark.parametrize("content", [
    {"meta": "yesterday"},                                   # a meta block that is not an object
    {"meta": {"day": 1, "hour": None, "gold": 3}},           # ... with an hour that is not a number
    {"meta": {"day": "two", "hour": 7, "gold": 3}},
    {"meta": {"day": 1, "hour": 7, "gold": 3, "saved_at": "noon"}},
    {"player": [1, 2], "day_count": "x"},                    # no meta, and nothing shaped like a save
])
def test_describe_never_raises_on_valid_json_that_is_not_a_save(tmp_path, content):
    """The labels feed every menu - the start screen's Continue among them - so an odd file is
    flagged, never a crash that keeps the game from reaching its menu."""
    odd = tmp_path / "odd.json"
    odd.write_text(json.dumps(content))
    assert SaveManager.describe(str(odd)) == "odd.json (unreadable)"


def test_one_odd_file_in_the_saves_folder_does_not_hide_the_others(game):
    game.player.gold = 12.0
    good = game.saves.save_new(game.world, game.player)
    with open(os.path.join(game.saves.save_dir, "odd.json"), "w") as f:
        json.dump({"meta": "?"}, f)
    os.makedirs(os.path.join(game.saves.save_dir, "folder.json"))          # not even a file
    labels = {os.path.basename(s.path): s.label for s in game.saves.list_saves()}
    assert set(labels) == {os.path.basename(good), "odd.json"}
    assert "12g" in labels[os.path.basename(good)] and labels["odd.json"] == "odd.json (unreadable)"
    assert game.load_game(os.path.join(game.saves.save_dir, "odd.json")) is False and game.player.gold == 12.0


def test_save_meta_block(game):
    game.player.gold = 99.5
    game.save_game()
    data = json.load(open(game.saves.filename))
    assert data["version"] == 9
    assert data["meta"]["gold"] == 99.5 and data["meta"]["day"] == 0 and data["meta"]["hour"] == 7 and data["meta"]["level"] == 1
    assert data["meta"]["saved_at"] <= time.time()


def test_loading_a_pre_ownership_save_marks_carried_goods_as_yours(game, tmp_path):
    from unprepared.entities import Item
    game.save_game()
    data = json.load(open(game.saves.filename))
    data["version"] = 4
    for item in data["player"]["inventory"]:
        item.pop("owner", None)
    json.dump(data, open(game.saves.filename, "w"))
    assert game.load_game()
    for item in game.player.inventory:
        assert item.owner == (None if item.is_stolen else Item.PLAYER_OWNER)


def test_saveinfo_name():
    assert SaveInfo("/x/saves/save-1.json", "l", 0.0).name == "save-1"
    assert SaveInfo("/x/savegame.json", "l", 0.0, quick=True).name == "Quick Save"
