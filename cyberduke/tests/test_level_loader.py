"""JSON/TOML level files round-trip the typed level exactly and fail loudly when broken."""

import json
from dataclasses import replace

import pytest

from cyberduke.levels import (
    FACILITY,
    LevelFormatError,
    level_from_dict,
    level_to_dict,
    load_level,
    loader,
    save_level,
)
from cyberduke.simulation import InputState, Simulation
from cyberduke.world import Room


def test_dict_round_trip_is_exact():
    assert level_from_dict(level_to_dict(FACILITY)) == FACILITY


def test_json_file_round_trip_is_exact(tmp_path):
    path = tmp_path / "facility.json"
    save_level(FACILITY, path)
    loaded = load_level(path)
    assert loaded == FACILITY
    # and it is plain, human-editable JSON
    data = json.loads(path.read_text())
    assert data["schema"] == loader.SCHEMA_VERSION
    assert [s["name"] for s in data["sectors"]] == [
        "StagingBay",
        "AirlockCorridor",
        "DatacoreVault",
        "FungalGreenhouse",
    ]


def test_toml_round_trip_when_writer_available(tmp_path):
    tomli_w = pytest.importorskip("tomli_w")
    tomllib = pytest.importorskip("tomllib")
    del tomli_w, tomllib
    path = tmp_path / "facility.toml"
    save_level(FACILITY, path)
    assert load_level(path) == FACILITY


def test_toml_reading_works_without_a_writer():
    tomllib = pytest.importorskip("tomllib")
    minimal = """
schema = 1
name = "Tiny"
nav_bounds = [-5.0, 5.0, -5.0, 5.0]
spawn = [0.0, 0.0, 0.0]

[[sectors]]
name = "Box"
bounds = [-5.0, 5.0, -5.0, 5.0]
walk_regions = [{ min_x = -4.0, max_x = 4.0, min_z = -4.0, max_z = 4.0 }]

[[terminals]]
terminal_id = "t"
x = 1.0
y = 0.0
z = 1.0
objective = true
"""
    level = loader.loads(minimal, "toml")
    assert level.name == "Tiny" and level.sectors[0].walk_regions[0].max_x == 4.0
    assert tomllib.loads(minimal)["schema"] == 1


def test_loaded_level_behaves_like_the_built_in_one(tmp_path):
    path = tmp_path / "facility.json"
    save_level(FACILITY, path)
    loaded = load_level(path)
    assert Room(loaded).get_obstacles() == Room(FACILITY).get_obstacles()
    sim = Simulation(loaded)
    for _ in range(120):
        sim.update(1 / 60, InputState(forward=1.0))
    assert sim.player.z < FACILITY.spawn[1] - 1.0


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda d: d.update(schema=99), "schema"),
        (lambda d: d["sectors"][0]["walk_regions"][0].update(requires_door="nope"), "unknown door"),
        (lambda d: d["doors"][0].update(unlocked_by="ghost"), "unknown terminal"),
        (lambda d: d.update(on_caught="explode"), "on_caught"),
        (lambda d: d["terminals"][0].update(objective=True), "at most one objective"),
        (lambda d: d["doors"][2].update(key_intel="nothing"), "keyed to intel"),
        (lambda d: d["sectors"][0]["props"][0].update(kind="sphere"), "kind"),
        (lambda d: d["sectors"][0]["walk_regions"][0].update(inset=["top"]), "inset"),
        (lambda d: d["sectors"][0]["lights"][0].update(ambient=[1, 2]), "three numbers"),
        # a typo used to load fine and crash (KeyError) when the player reached the exit
        (lambda d: d.update(exit_to="night_markte"), "unknown level 'night_markte'"),
    ],
)
def test_broken_files_fail_with_a_clear_message(mutate, message):
    data = level_to_dict(FACILITY)
    mutate(data)
    with pytest.raises(LevelFormatError, match=message):
        level_from_dict(data)


def test_unknown_extension_is_rejected(tmp_path):
    with pytest.raises(LevelFormatError, match="json or .toml"):
        save_level(FACILITY, tmp_path / "level.yaml")


def test_cli_export_and_check(tmp_path, capsys):
    path = tmp_path / "out.json"
    assert loader.main(["export", str(path)]) == 0
    assert loader.main(["check", str(path)]) == 0
    out = capsys.readouterr().out
    assert "4 sectors" in out and "4 guards" in out


def test_modified_level_changes_the_game():
    data = level_to_dict(FACILITY)
    data["spawn"] = [0.0, 30.0, 90.0]
    level = level_from_dict(data)
    assert level == replace(FACILITY, spawn=(0.0, 30.0, 90.0))
    assert Simulation(level).player.z == 30.0


# ----------------------------------------------------------------------------- netrunning fields
def test_factions_and_hack_flags_survive_the_round_trip():
    data = level_to_dict(FACILITY)
    assert data["factions"][0] == {
        "faction_id": "halcyon", "name": "Halcyon Dynamics Security", "color": [1.0, 0.55, 0.1]
    }  # fmt: skip
    assert data["factions"][1]["faction_id"] == "halcyon_bio"
    datacore = next(t for t in data["terminals"] if t["terminal_id"] == "datacore_terminal")
    assert datacore["secured"] and datacore["critical"] and datacore["intel"]
    assert not datacore["hardened"]
    east = next(t for t in data["terminals"] if t["terminal_id"] == "node_east_terminal")
    assert east["hardened"]
    blast, bulkhead, greenhouse, gate, exit_door = data["doors"]
    assert blast["high_security"] and bulkhead["unlocked_by"] is None
    assert greenhouse["key_intel"] == "greenhouse door cipher" and gate["key_intel"] is None
    assert data["guards"][1]["essential"] and data["guards"][1]["guard_id"] == "vault_guard"
    assert data["guards"][2]["wall_pos"] is None and len(data["guards"][2]["patrol_route"]) == 5
    assert data["exit_zone"]["min_z"] == 78.2
    assert data["sectors"][2]["lights"][1]["ceiling_y"] == 5.98
    assert data["cameras"][0]["camera_id"] == "airlock_camera"
    assert data["vending_machines"][0]["machine_id"] == "neon_cold_vendor"
    assert level_from_dict(data) == FACILITY


def test_missing_netrunning_fields_default_sensibly():
    data = level_to_dict(FACILITY)
    for key in ("factions",):
        data.pop(key)
    data["doors"] = data["doors"][:2]  # the greenhouse doors key on / are opened by the rest
    data["terminals"] = data["terminals"][:2]
    data["guards"] = data["guards"][:2]
    data["cameras"] = data["cameras"][:1]
    data["exit_zone"] = None
    data.pop("exit_to")
    for t in data["terminals"]:
        for k in ("faction", "intel", "secured", "critical", "hardened"):
            t.pop(k)
    for g in data["guards"]:
        for k in ("guard_id", "faction", "essential", "patrol_route"):
            g.pop(k)
    for d in data["doors"]:
        d.pop("high_security")
        d.pop("faction")
        d.pop("key_intel")
    for sct in data["sectors"]:
        for light in sct["lights"]:
            light.pop("ceiling_y")
    for c in data["cameras"]:
        c.pop("camera_id")
        c.pop("faction")
    for v in data["vending_machines"]:
        v.pop("machine_id")
        v.pop("faction")
    level = level_from_dict(data)
    assert level.factions == () and level.terminals[0].faction == "security"
    assert not level.terminals[1].secured and level.guards[1].guard_id == ""
    assert not level.doors[0].high_security
    sim = Simulation(level)
    assert [g.hack_id for g in sim.guards] == ["guard_1", "guard_2"]
    assert sim.cameras[0].hack_id == "camera_1" and sim.vending_machines[0].hack_id == "vending_1"
    assert list(sim.hacking.networks) == ["security"]


def test_undeclared_faction_reference_is_rejected_when_a_table_is_declared():
    data = level_to_dict(FACILITY)
    data["guards"][0]["faction"] = "ghosts"
    with pytest.raises(LevelFormatError, match="undeclared factions"):
        level_from_dict(data)


def test_duplicate_hack_ids_and_faction_ids_are_rejected():
    data = level_to_dict(FACILITY)
    data["cameras"][0]["camera_id"] = "blast_door"
    with pytest.raises(LevelFormatError, match="unique"):
        level_from_dict(data)
    data = level_to_dict(FACILITY)
    data["factions"].append({"faction_id": "halcyon", "name": "dup"})
    with pytest.raises(LevelFormatError, match="faction ids must be unique"):
        level_from_dict(data)
