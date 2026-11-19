"""The Night Market Alley: the level past the facility's exit door, and the level
transition that gets the player there.

Structure (the same no-invisible-walls / no-clipping rules the facility is held to),
navigation (the alley is walkable end to end round the stalls and the crowd), the crowd
itself (civilians keep apart, keep moving, steer round the player, never raise an alarm),
the merchants (work their stalls, turn to a customer), the hand-over from the facility
and the win at the alley mouth.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import replace

import pytest

from cyberduke.ai import Civilian, Merchant
from cyberduke.config import NPC, PLAYER, SIMULATION
from cyberduke.core.geometry import AABB
from cyberduke.core.state import GameState
from cyberduke.levels import LEVELS, build_facility, build_night_market
from cyberduke.levels.loader import level_from_dict, level_to_dict
from cyberduke.simulation import InputState, Simulation
from cyberduke.world import Room

DT = 1 / 60
LEVEL = build_night_market()
ROOM = Room(LEVEL)
AGENT_R = max(NPC.collision_radius, PLAYER.collision_radius)


def _colliders():
    return [c for s in LEVEL.sectors for c in s.colliders]


def _overlaps_xz(a: AABB, b: AABB) -> bool:
    return a.min_x < b.max_x and b.min_x < a.max_x and a.min_z < b.max_z and b.min_z < a.max_z


def run(sim: Simulation, seconds: float, inp: InputState | None = None) -> None:
    for _ in range(int(round(seconds / DT))):
        sim.update(DT, inp or InputState())


def street_sim() -> Simulation:
    return Simulation(build_night_market())


COMMUTERS = ("bartender", "regular")  # (they come and go by metro: ``ai.nightlife``)


def wanderers_sim(civilians: bool = True) -> Simulation:
    """The street with nobody playing a role: the alley's people just wander and merchants
    just work (the fallback every role drops back to - see tests/test_alley.py for the
    stories); the commuters stay at home across town."""
    level = replace(
        LEVEL,
        civilians=tuple(replace(c, role="") for c in LEVEL.civilians if c.role not in COMMUTERS)
        if civilians
        else (),
        merchants=tuple(replace(m, role="") for m in LEVEL.merchants),
    )
    return Simulation(level)


# ----------------------------------------------------------------------------- structure
def test_the_street_is_a_registered_level_with_a_brief_and_no_way_on():
    """No winning any more: the alley has no exit zone (the mouth is walled off to the
    player - the city beyond is not built yet), and being caught in it means being
    thrown out of wherever you were caught (number 16), not a new game."""
    assert LEVELS["night_market"]() is LEVEL
    assert LEVEL.name == "Night Market Alley" and LEVEL.brief
    assert LEVEL.exit_zone is None and LEVEL.exit_to is None, "nowhere further to go"
    assert LEVEL.on_caught == "evict"
    assert not LEVEL.guards, "no guards on the street"
    assert [t.terminal_id for t in LEVEL.terminals] == ["dells_computer"]
    assert [c.camera_id for c in LEVEL.cameras] == ["home_camera_16"], "one camera: in No. 16"
    assert len(LEVEL.civilians) >= 5 and len(LEVEL.merchants) == 2


def test_the_street_round_trips_through_the_level_loader():
    data = level_to_dict(LEVEL)
    again = level_from_dict(data)
    assert again.name == LEVEL.name and again.brief == LEVEL.brief
    assert again.civilians == LEVEL.civilians and again.merchants == LEVEL.merchants
    assert again.exit_zone == LEVEL.exit_zone and again.exit_to is None
    assert len(again.props) == len(LEVEL.props)
    assert again == LEVEL, "everything: the flat, its door, panel, camera, stick and routes"


def test_the_facility_points_at_the_street():
    facility = build_facility()
    assert facility.exit_to == "night_market" and "night_market" in LEVELS
    assert facility.brief


def test_no_invisible_walls_every_collider_overlaps_a_visible_prop():
    for c in _colliders():
        assert any(_overlaps_xz(c, p.footprint()) for p in LEVEL.props), f"{c} is invisible"


def test_every_solid_prop_lies_inside_a_collider():
    colliders = _colliders()
    for p in LEVEL.props:
        if not p.solid:
            continue
        fp = p.footprint()
        inside = any(
            c.min_x - AGENT_R <= fp.min_x and fp.max_x <= c.max_x + AGENT_R
            and c.min_z - AGENT_R <= fp.min_z and fp.max_z <= c.max_z + AGENT_R
            for c in colliders
        )  # fmt: skip
        assert inside, f"solid prop {p.label!r} at ({p.x}, {p.z}) is outside every collider"


def test_big_floor_level_props_are_solid_or_known_decoration():
    walkover_y = 0.12
    for p in LEVEL.props:
        fp = p.footprint()
        if p.solid or "surface" in p.tags or "compound" in p.tags:
            continue
        if fp.max_y <= walkover_y or fp.min_y >= PLAYER.body_height_standing:
            continue
        area = (fp.max_x - fp.min_x) * (fp.max_z - fp.min_z)
        assert area < 0.6 or fp.max_y - fp.min_y < 0.3, (
            f"prop {p.label!r} at ({p.x}, {p.y}, {p.z}) is large and floor-level but not solid"
        )


def test_no_clutter_clips_a_collider_or_a_person():
    clutter = [p for p in LEVEL.props if "clutter" in p.tags]
    assert len(clutter) >= 30, "it is an alley"
    assert all(not p.solid for p in clutter)
    for p in clutter:
        fp = p.footprint()
        for c in _colliders():
            # (resting on top of it - a cup on a table - is not clipping it)
            if c.max_y > 0.05 and c.min_y < fp.max_y and c.max_y > fp.min_y + 0.01:
                assert not (
                    c.min_x < fp.max_x - 0.02 and fp.min_x + 0.02 < c.max_x
                    and c.min_z < fp.max_z - 0.02 and fp.min_z + 0.02 < c.max_z
                ), f"{p.label!r} at ({p.x:.2f}, {p.z:.2f}) clips a collider"  # fmt: skip
        for m in LEVEL.merchants:
            assert math.hypot(p.x - m.x, p.z - m.z) > 0.45, f"{p.label!r} is under {m.merchant_id}"


def test_labels_and_lights():
    """Lights for the alley, the street lamps of the main street and one or two for each
    flat: the renderer uses the eight nearest the eye (``render.quality.lights_for``)."""
    assert all(p.label for p in LEVEL.props)
    assert 0 < len(LEVEL.lights) <= 30
    lamps = [light for light in LEVEL.lights if light.night_only]
    assert len(lamps) >= 6, "street lamps (and door lamps): dimmed by day"


# ----------------------------------------------------------------------------- navigation
def _reachable(room: Room, start, radius=PLAYER.collision_radius, step=0.25):
    """Flood fill of the walkable floor (walk regions minus colliders) at ``step`` m."""
    obstacles = room.get_obstacles()
    x0, z0 = start
    seen = {(0, 0)}
    queue = deque([(0, 0)])

    def free(i, j):
        x, z = x0 + i * step, z0 + j * step
        if not room.is_valid_position(x, z, radius):
            return False
        return not any(o.overlaps_cylinder(x, z, radius, 0.0, 1.8) for o in obstacles)

    while queue:
        i, j = queue.popleft()
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            n = (i + di, j + dj)
            if n not in seen and free(*n):
                seen.add(n)
                queue.append(n)
    return {(x0 + i * step, z0 + j * step) for i, j in seen}


def test_spawn_waypoints_and_stalls_are_on_walkable_floor():
    x, z, _ = LEVEL.spawn
    assert ROOM.is_valid_position(x, z)
    for c in LEVEL.civilians:
        if c.y > 0.0:  # (the regular, up on the metro's platform: nowhere the netrunner goes)
            continue
        assert ROOM.is_valid_position(c.x, c.z), c.civilian_id
        for wx, wz in c.waypoints:
            assert ROOM.is_valid_position(wx, wz), f"{c.civilian_id} waypoint ({wx}, {wz})"
    # merchants stand behind their counters, inside the stall's keep-out box, but not
    # inside anything you can see
    for m in LEVEL.merchants:
        for p in LEVEL.props:
            if p.solid or p.footprint().max_y > 0.4:
                assert not p.footprint().overlaps_cylinder(m.x, m.z, 0.22, 0.0, 1.8), (
                    f"{m.merchant_id} stands inside {p.label!r}"
                )


def test_the_alley_and_the_street_are_walkable_from_the_facility_door_to_the_barricades():
    """The player can walk up the alley, out of its mouth and along the pavement of the
    main street both ways as far as the barricades at the ends of the built city - not
    over the guard rail into the road, not past a barricade - and reach every waypoint and
    both stall counters from the spawn without squeezing through anything narrower than
    their own body."""
    from cyberduke.levels.main_street import EAST_END, KERB_Z, WEST_END

    floor = _reachable(ROOM, LEVEL.spawn[:2])
    assert len(floor) > 2500, "a long alley and a long street"
    xs = [x for x, z in floor if z > 43.0]
    assert min(xs) < WEST_END + 1.0 and max(xs) > EAST_END - 1.0, "along the street both ways"
    assert min(xs) > WEST_END and max(xs) < EAST_END, "and not past the barricades"
    assert max(z for _, z in floor) < KERB_Z - 0.1, "not over the rail into the road"
    for wx, wz in LEVEL.civilians[0].waypoints:
        assert any(math.hypot(x - wx, z - wz) < 0.3 for x, z in floor), f"waypoint {wx, wz}"
    for m in LEVEL.merchants:
        # the customer side of the counter: 1.8 m from the merchant toward the alley
        # (NPC yaw: 0 = +Z, 90 = +X)
        cx = m.x + 1.8 * math.sin(math.radians(m.face_yaw))
        cz = m.z + 1.8 * math.cos(math.radians(m.face_yaw))
        assert any(math.hypot(x - cx, z - cz) < 0.3 for x, z in floor), m.merchant_id


def test_walking_straight_up_the_alley_stops_at_the_guard_rail():
    """There is no winning: up the alley and out of its mouth onto the main street's
    pavement, the guard rail along the kerb stops the netrunner (the far side is a
    backdrop); the ends of the street are barricaded."""
    from cyberduke.levels.main_street import FACE_Z, KERB_Z

    sim = street_sim()
    for _ in range(int(60.0 / DT)):
        sim.player.yaw = 90.0
        sim.update(DT, InputState(forward=1.0))
    assert sim.state is GameState.PLAYING
    assert FACE_Z < sim.player.z < KERB_Z, "out on the pavement, stopped at the rail"


# ----------------------------------------------------------------------------- the crowd
def test_civilians_keep_apart_keep_moving_and_stay_in_the_alley():
    sim = wanderers_sim()
    civs = sim.civilians
    assert len(civs) == len([c for c in LEVEL.civilians if c.role not in COMMUTERS])
    starts = [(c.x, c.z) for c in civs]
    min_gap = 99.0
    moved = [0.0] * len(civs)
    sim.player.x, sim.player.z = 0.0, -1.2
    last = starts[:]
    for _ in range(int(30.0 / DT)):
        sim.update(DT, InputState())
        for i, c in enumerate(civs):
            moved[i] += math.hypot(c.x - last[i][0], c.z - last[i][1])
            last[i] = (c.x, c.z)
            assert -4.3 < c.x < 4.3 and -2.2 < c.z < 43.6, f"{c.civilian_id} left the alley"
            for o in civs[i + 1 :]:
                min_gap = min(min_gap, math.hypot(c.x - o.x, c.z - o.z))
            for m in sim.merchants:
                assert math.hypot(c.x - m.x, c.z - m.z) > 0.4, (
                    f"{c.civilian_id} walked into a stall"
                )
    assert min_gap >= 0.45, f"the crowd walked through itself ({min_gap:.2f} m)"
    assert all(d > 4.0 for d in moved), f"someone stood still for 30 s: {moved}"
    assert all(c.state in ("WALKING", "PAUSING") for c in civs)


def test_civilians_steer_round_the_player_and_are_solid_to_them():
    sim = wanderers_sim()
    civ = sim.civilians[0]
    # park the player on the civilian's route, right in its face
    civ.x, civ.z, civ.yaw = 0.0, 5.0, 0.0  # facing up the alley (+Z)
    civ.target = (0.0, 12.0)
    civ.state = "WALKING"
    sim.player.x, sim.player.z = 0.0, 7.0  # dead ahead
    closest = 99.0
    for _ in range(int(8.0 / DT)):
        sim.update(DT, InputState())
        closest = min(closest, math.hypot(civ.x - sim.player.x, civ.z - sim.player.z))
    assert closest >= PLAYER.collision_radius + NPC.collision_radius - 0.05
    assert civ.z > 9.0, "it got past, round the player"
    # and the player cannot walk through a civilian either: their body is an obstacle
    assert civ.body_box().contains_xz(civ.x, civ.z)
    assert civ.body_box() in sim.player_obstacles or any(
        b.min_x <= civ.x <= b.max_x and b.min_z <= civ.z <= b.max_z for b in sim.player_obstacles
    )
    # ... they are pushed aside, out of the way (nobody can pin the netrunner: see
    # tests/test_shove.py) - but never walked through
    civ.x, civ.z = 0.0, 20.0
    civ.state, civ.pause_timer = "PAUSING", 99.0
    p = sim.player
    p.x, p.z = -1.0, 20.0
    p.yaw = 0.0  # camera yaw 0 = +X: straight at them
    into = []
    for _ in range(int(1.5 / DT)):
        sim.update(DT, InputState(forward=1.0))
        into.append(
            civ.body_box().overlaps_cylinder(p.x, p.z, p.radius, p.floor_y, p.floor_y + 1.8)
        )
    assert not any(into), "never walked through"
    assert abs(civ.z - 20.0) > 0.3, "pushed aside"


def test_civilians_never_raise_the_alarm_and_look_at_whoever_is_close():
    sim = wanderers_sim()
    civ = sim.civilians[0]
    civ.x, civ.z, civ.yaw = 0.0, 5.0, 0.0  # facing +Z
    sim.player.x, sim.player.z = 0.8, 7.0  # ahead and a little to the right
    civ.state, civ.pause_timer = "PAUSING", 99.0
    run(sim, 3.0)
    assert civ.watching == (0.8, 7.0) and civ.pose.head_yaw > 5.0, "eye contact"
    assert sim.detection.level == 0.0 and sim.state is GameState.PLAYING
    # stand in front of every one of them for a while: nobody ever reports
    for c in sim.civilians:
        c.x, c.z, c.yaw = 0.0, 20.0, 0.0
        c.state, c.pause_timer = "PAUSING", 99.0
    sim.player.x, sim.player.z = 0.0, 21.5
    run(sim, 5.0)
    assert sim.detection.level == 0.0


def test_merchants_work_the_stall_and_turn_to_a_customer():
    sim = wanderers_sim(civilians=False)  # nobody else about to be greeted
    cook = next(m for m in sim.merchants if m.style == "cook")
    hawker = next(m for m in sim.merchants if m.style == "hawk")
    sim.player.x, sim.player.z = 0.0, -1.2  # far from both
    run(sim, 2.0)
    for m in (cook, hawker):
        assert abs(m.yaw - m.face_yaw) < 1.0 and m.greeting is None
    poses = []
    for _ in range(30):
        sim.update(DT, InputState())
        poses.append((tuple(cook.pose.arm_r), tuple(hawker.pose.arm_r)))
    assert len(set(poses)) > 10, "both keep working: the animation is not static"
    # a customer at the counter (NPC yaw: 0 = +Z, 90 = +X): the cook turns from the wok
    sim.player.x = cook.x + 1.9 * math.sin(math.radians(cook.face_yaw + 40.0))
    sim.player.z = cook.z + 1.9 * math.cos(math.radians(cook.face_yaw + 40.0))
    run(sim, 2.0)
    assert cook.greeting is not None
    assert abs(cook.yaw - (cook.face_yaw + 40.0)) < 3.0, "turned to the customer"
    assert abs(hawker.yaw - hawker.face_yaw) < 1.0, "the other stall is too far to notice"
    sim.player.x, sim.player.z = 0.0, -1.2
    run(sim, 3.0)
    assert cook.greeting is None and abs(cook.yaw - cook.face_yaw) < 1.0, "back to the wok"


def test_pause_freezes_the_street():
    sim = street_sim()
    run(sim, 1.0)
    before = [(c.x, c.z, tuple(c.pose.arm_l)) for c in sim.npcs]
    sim.update(DT, InputState(toggle_pause=True))
    run(sim, 2.0)
    assert [(c.x, c.z, tuple(c.pose.arm_l)) for c in sim.npcs] == before


def test_a_merchant_and_a_civilian_can_be_built_standalone():
    m = Merchant(0.0, 0.0, 0.0, (-5.0, 5.0, -5.0, 5.0), face_yaw=90.0, style="hawk")
    c = Civilian(1.0, 0.0, 1.0, (-5.0, 5.0, -5.0, 5.0), [(2.0, 2.0), (-2.0, -2.0)], seed=3)
    assert m.blocks_player and c.blocks_player
    assert not c.spec.has_armor and not m.spec.has_armor, "no corporate plating"
    assert c.spec.shirt_color != m.spec.shirt_color or c.spec.hair_color != m.spec.hair_color


# ----------------------------------------------------------------------------- hand-over
def test_stepping_into_the_facility_exit_loads_the_street_with_the_deck_intact():
    sim = Simulation()
    sim.player.intel.append("greenhouse door cipher")
    assert sim.level.name == "Corporate Facility" and sim.level_generation == 0
    sim.player.x, sim.player.z = 0.0, 78.6
    run(sim, SIMULATION.level_fade_out + 2 * DT)  # (the picture fades out at the door)
    assert sim.level.name == "Night Market Alley" and sim.level_generation == 1
    assert sim.state is GameState.PLAYING and not sim.paused
    assert (sim.player.x, sim.player.z, sim.player.yaw) == sim.level.spawn
    assert sim.player.intel == ["greenhouse door cipher"]
    assert sim.hacking.statuses() == []
    # the networks on the street: number 16's own, the vending machine's, the
    # tenement's door locks (every other front door's panel) and the metro's ticket
    # gates; a ping by the staff door
    # finds only the panel of Nix's door, a few steps up the alley (Dell's flat is forty
    # metres up it, the machine eighteen)
    assert set(sim.hacking.networks) == {"dells_home", "neon_cold", "tenement", "metro_transit"}
    sim.update(DT, InputState(scan=True))
    assert sim.scanner.revealed_ids == ["panel17"]


def test_the_picture_fades_out_at_the_biodome_door_and_back_in_on_the_alley():
    """Meeting the biodome's outer staff door: the netrunner stops there and the picture
    fades to black (nothing they press does anything, and nobody can catch them on the
    threshold); the alley is loaded behind the black and the picture fades back in on it."""
    sim = Simulation()
    assert sim.screen_black == 0.0
    sim.player.x, sim.player.z = 0.0, 78.6
    sim.update(DT, InputState())
    assert sim.leaving == "night_market" and sim.level.name == "Corporate Facility"
    blacks = [sim.screen_black]
    where = (sim.player.x, sim.player.z, sim.player.yaw)
    sim.detection.level = 99.9  # (about to be caught...)
    while sim.level.name == "Corporate Facility":
        sim.update(DT, InputState(forward=-1.0, look_dx=40.0, scan=True))
        assert sim.state is GameState.PLAYING, "nobody catches the netrunner on the way out"
        if sim.level.name == "Corporate Facility":
            assert (sim.player.x, sim.player.z, sim.player.yaw) == where, "stopped at the door"
            blacks.append(sim.screen_black)
            sim.detection.level = 99.9
        assert len(blacks) < 2 * SIMULATION.level_fade_out / DT, "the fade never ended"
    assert all(b2 >= b1 for b1, b2 in zip(blacks, blacks[1:], strict=False)), "darker and darker"
    assert blacks[0] < 0.1 and blacks[-1] > 0.9
    # on the alley: black at first, then the picture comes back
    assert sim.level.name == "Night Market Alley" and sim.leaving is None
    assert sim.screen_black > 0.9
    assert sim.detection.level == 0.0, "a clean slate on the street"
    fading = []
    run(sim, SIMULATION.level_fade_in * 0.5)
    fading.append(sim.screen_black)
    run(sim, SIMULATION.level_fade_in * 0.5 + 2 * DT)
    fading.append(sim.screen_black)
    assert 0.2 < fading[0] < 0.8 and fading[1] == 0.0


def test_a_new_game_from_the_street_starts_over_in_the_facility():
    sim = Simulation()
    sim.player.intel.append("greenhouse door cipher")
    sim.player.x, sim.player.z = 0.0, 78.6
    run(sim, SIMULATION.level_fade_out + 2 * DT)
    assert sim.level.name == "Night Market Alley"
    sim.quests.start("number16")
    sim.new_game()
    assert sim.level.name == "Corporate Facility" and sim.level_generation == 2
    assert sim.state is GameState.PLAYING and sim.player.intel == []
    assert not sim.quests.quests, "no quests carried over"
    assert len(sim.guards) == 4


def test_reset_on_the_street_restarts_the_street_not_the_facility():
    sim = street_sim()
    run(sim, 2.0)
    sim.player.x, sim.player.z = 0.0, 20.0
    sim.update(DT, InputState(reset=True))
    assert sim.level.name == "Night Market Alley"
    assert (sim.player.x, sim.player.z) == sim.level.spawn[:2]
    assert [(c.x, c.z) for c in sim.civilians] == [(c.x, c.z) for c in LEVEL.civilians]


@pytest.mark.parametrize("name", sorted(LEVELS))
def test_every_registered_level_validates_and_stands_up(name):
    level = LEVELS[name]()
    level_from_dict(level_to_dict(level))  # the loader's validation
    sim = Simulation(level)
    run(sim, 0.5)
    assert sim.state is GameState.PLAYING


def test_the_people_of_the_street_never_raise_the_alarm_and_stay_about_the_alley():
    """With their stories running (ai.alley): nobody reports the player, nobody leaves the
    alley other than round the corner onto the main road, and a merchant at a loose end
    looks up at whoever steps up to the counter."""
    sim = street_sim()
    for _ in range(int(60.0 / DT)):
        sim.update(DT, InputState())
        for c in sim.civilians:
            if c.hidden or c.role in COMMUTERS:  # (the bar and the metro: tests/test_nightlife.py)
                continue
            inside = -4.3 < c.x < 4.3 and -2.2 < c.z < 43.6
            round_the_corner = c.z >= 42.5 and abs(c.x) < 8.0
            at_home = c.role == "resident" and LEVEL.private_areas[0].contains(c.x, c.z)
            at_home = at_home or (c.role == "resident" and c.x >= -4.6 and 29.6 < c.z < 31.4)
            assert inside or round_the_corner or at_home, (
                f"{c.civilian_id} wandered off to {(c.x, c.z)}"
            )
    assert sim.detection.level == 0.0 and sim.state is GameState.PLAYING
    dealer = next(m for m in sim.merchants if m.role == "junk_dealer")
    sim.player.x, sim.player.z = 2.2, 24.6  # at the junk table
    for _ in range(int(4.0 / DT)):
        sim.update(DT, InputState())
    assert dealer.greeting is not None, "the dealer sees somebody at the table"


def test_the_wall_skirting_stops_at_the_doors():
    """The concrete skirting along the walls used to run straight across every front
    door, hiding all but the top of it."""
    skirting = [p.footprint() for p in LEVEL.props if p.label == "Wall base skirting"]
    doors = [p for p in LEVEL.props if p.label.startswith("Front door")]
    doors += [p for d in LEVEL.doors for p in d.leaf if p.label.startswith("Front door")]
    doors = [d for d in doors if abs(d.x) < 5.0]  # (the alley's: 14, 16, 17, 21)
    assert len(doors) == 4
    for door in doors:
        for s in skirting:
            same_side = (s.min_x < 0) == (door.x < 0)
            assert not (same_side and s.min_z < door.z + 0.5 and s.max_z > door.z - 0.5), door.label


# ----------------------------------------------------------------------------- doorways
def _door_frame(d):
    """(middle x, z, the way in (x, z), across the doorway (x, z)) of a door."""
    o = d.obstacle
    inward = {"-x": (-1.0, 0.0), "+x": (1.0, 0.0), "-z": (0.0, -1.0), "+z": (0.0, 1.0)}[d.inside]
    return o.center_x, o.center_z, inward, (inward[1], inward[0])


@pytest.mark.parametrize("off_centre", [-0.4, -0.25, 0.25, 0.4])
def test_every_front_door_takes_somebody_a_little_off_centre(off_centre):
    """Walking straight at a front door (a home's, the bar's) a little to one side of its
    middle, the netrunner goes through it - in and out again - instead of stopping on the
    jamb: the walls either side of a doorway stop a little short of it for anybody walking
    (``doorway.ease_doorways``) and a body that does catch the edge is eased round it
    (``PlayerController._ease_round``). It used to take a dead-centre approach: 20 cm off
    and the netrunner stuck on the frame."""
    from cyberduke.levels.doorway import DOOR_GIVE

    sim = street_sim()
    sim.clock.rate = 0.0
    for n in sim.npcs:
        n.hidden = True
    p = sim.player
    hinged = [d for d in LEVEL.doors if d.hinge is not None]
    assert len(hinged) >= 9
    for data in hinged:
        door = sim.room.doors[data.door_id]
        cx, cz, (ix, iz), (lx, lz) = _door_frame(data)
        for way in (1.0, -1.0):  # in, then out
            p.x = cx - ix * 1.5 * way + lx * off_centre
            p.z = cz - iz * 1.5 * way + lz * off_centre
            p.yaw = math.degrees(math.atan2(iz * way, ix * way))
            for _ in range(int(3.0 / DT)):
                door.unlock()
                sim.update(DT, InputState(forward=1.0))
            depth = ((p.x - cx) * ix + (p.z - cz) * iz) * way
            assert depth > 0.6, (data.door_id, off_centre, "in" if way > 0 else "out")
    # (what is drawn is as it was: only the colliders give)
    for data in hinged:
        assert data.passage is not None
        o = data.obstacle
        wide = (o.max_z - o.min_z) if data.inside in ("-x", "+x") else (o.max_x - o.min_x)
        p_ = data.passage
        strip = (p_.max_z - p_.min_z) if data.inside in ("-x", "+x") else (p_.max_x - p_.min_x)
        assert strip >= wide - 0.1 + 2 * DOOR_GIVE - 1e-6, data.door_id
