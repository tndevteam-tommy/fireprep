"""Consistency checks on the shipped level data (meshes vs. colliders vs. regions vs. entities)."""

import math

import pytest

from cyberduke.config import NPC, PLAYER
from cyberduke.core.geometry import AABB
from cyberduke.levels import build_facility
from cyberduke.world import Prop, Room

LEVEL = build_facility()
ROOM = Room(LEVEL)
AGENT_R = max(NPC.collision_radius, PLAYER.collision_radius)


def _overlaps_xz(a: AABB, b: AABB) -> bool:
    return a.min_x < b.max_x and b.min_x < a.max_x and a.min_z < b.max_z and b.min_z < a.max_z


def _colliders():
    return [c for s in LEVEL.sectors for c in s.colliders] + [d.obstacle for d in LEVEL.doors]


def _visible_props():
    return LEVEL.props + [p for d in LEVEL.doors for p in d.leaf]


# ----------------------------------------------------------------------------- structure
def test_level_has_four_sectors_with_unique_names():
    names = [s.name for s in LEVEL.sectors]
    assert names == ["StagingBay", "AirlockCorridor", "DatacoreVault", "FungalGreenhouse"]
    assert len(set(names)) == 4


def test_exactly_one_objective_terminal_and_door_terminal_exists():
    objectives = [t for t in LEVEL.terminals if t.objective]
    assert len(objectives) == 1
    ids = {t.terminal_id for t in LEVEL.terminals}
    for door in LEVEL.doors:
        assert door.unlocked_by is None or door.unlocked_by in ids, (
            f"door {door.door_id} unlocked by unknown terminal"
        )


def test_walk_region_door_references_resolve():
    door_ids = {d.door_id for d in LEVEL.doors}
    for region in LEVEL.walk_regions:
        assert region.requires_door in (None, *door_ids)


# ----------------------------------------------------------------------------- props vs colliders
def test_no_invisible_walls_every_collider_overlaps_a_visible_prop():
    props = _visible_props()
    for c in _colliders():
        hit = any(_overlaps_xz(c, p.footprint()) for p in props)
        assert hit, f"collider {c} has no visible geometry"


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


def test_big_floor_level_props_are_marked_solid_or_are_known_decoration():
    """Anything a player could plausibly bump into must either be solid (and therefore
    covered by a collider) or be explicitly understood as walk-over decoration."""
    walkover_y = 0.12  # floor plates, grates, puddles, guide lights
    for p in LEVEL.props:
        fp = p.footprint()
        if p.solid or "surface" in p.tags or "compound" in p.tags:
            continue  # compound props are checked by test_compound_props_are_covered_by_colliders
        if p.kind == "disc":
            continue  # light, haze and puddles: nothing there to bump into
        if fp.max_y <= walkover_y or fp.min_y >= PLAYER.body_height_standing:
            continue
        footprint_area = (fp.max_x - fp.min_x) * (fp.max_z - fp.min_z)
        height = fp.max_y - fp.min_y
        # Small clutter (cans, boxes, datachips, wall trims) is allowed to be non-solid
        assert footprint_area < 0.6 or height < 0.3, (
            f"prop {p.label!r} at ({p.x}, {p.y}, {p.z}) is large and floor-level but not solid"
        )


def test_compound_props_are_covered_by_colliders():
    """Rotated props (the dome's curved wall panels) cannot sit inside one axis-aligned
    collider; instead every point along them must lie inside *some* collider, so the wall
    is never a visual with nothing behind it."""
    colliders = _colliders()
    tagged = [p for p in LEVEL.props if "compound" in p.tags]
    assert tagged, "the dome has compound wall panels"
    for p in tagged:
        assert p.kind == "box"
        half = p.size[0] / 2.0
        steps = max(2, int(p.size[0] / 0.5))
        for k in range(steps + 1):
            wx, _, wz = p.local_to_world(-half + k * p.size[0] / steps, 0.0, 0.0)
            inside = any(c.contains_xz(wx, wz, margin=AGENT_R) for c in colliders)
            assert inside, f"{p.label!r}: point ({wx:.1f}, {wz:.1f}) has no collider behind it"


# ----------------------------------------------------------------------------- regions
def test_spawn_and_entity_positions_are_on_walkable_floor():
    x, z, _ = LEVEL.spawn
    assert ROOM.is_valid_position(x, z)
    for g in LEVEL.guards:
        assert ROOM.is_valid_position(g.x, g.z), "guard spawn"
        for px, pz in g.patrol_points:
            assert ROOM.is_valid_position(px, pz), f"{g.guard_id} patrol point ({px}, {pz})"
    # interaction points: 1.5 m in front of each terminal (its chair side, with the
    # renderer's rotation handedness) is reachable, collision-free floor
    from cyberduke.entities import Terminal

    obstacles = ROOM.get_obstacles()
    for t in LEVEL.terminals:
        fx, fz = Terminal(t.x, t.y, t.z, yaw=t.yaw).front
        assert ROOM.is_valid_position(fx, fz), f"{t.terminal_id}: ({fx:.1f}, {fz:.1f})"
        assert not any(
            c.overlaps_cylinder(fx, fz, PLAYER.collision_radius, 0.0, 1.8) for c in obstacles
        ), f"{t.terminal_id}: chair side ({fx:.1f}, {fz:.1f}) is inside a collider"


def test_lights_have_fixtures_and_fit_the_fixed_function_budget():
    assert 0 < len(LEVEL.lights) <= 8
    assert all(light.fixture != "none" for light in LEVEL.lights)


def test_prop_footprints():
    box = Prop.box(1.0, 0.5, 2.0, 2.0, 1.0, 4.0)
    assert box.footprint() == AABB(0.0, 2.0, 0.0, 4.0, 0.0, 1.0)
    cyl = Prop.cylinder(0.0, 0.0, 0.0, 0.5, 0.5, 2.0)
    assert cyl.footprint() == AABB(-0.5, 0.5, -0.5, 0.5, 0.0, 2.0)
    rotated = Prop.box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, rotate=((45.0, 0.0, 1.0, 0.0),))
    assert rotated.footprint().max_x == pytest.approx(0.70711, abs=1e-4)
    assert rotated.footprint().max_y == pytest.approx(0.5)
    ramp = Prop.box(0.0, 1.375, 21.5, 3.2, 0.1, 6.06, rotate=((-27.0, 1.0, 0.0, 0.0),))
    fp = ramp.footprint()
    assert fp.min_y == pytest.approx(0.0, abs=0.05) and fp.max_y == pytest.approx(2.75, abs=0.05)
    assert fp.min_z == pytest.approx(18.8, abs=0.25) and fp.max_z == pytest.approx(24.2, abs=0.25)


def test_stair_handrails_connect_to_the_deck_rails_and_end_square_to_the_floor():
    vault = LEVEL.sector("DatacoreVault")
    cw = LEVEL.catwalk
    for side in (-1, 1):
        x = side * 6.02
        posts = [p for p in vault.props if p.x == x and "post" in p.label]
        assert len(posts) == 2, "a vertical post at each end of the stair rail"
        bottom, top = sorted(posts, key=lambda p: p.z)
        assert not bottom.rotate and not top.rotate, "posts are square to the floor"
        assert bottom.footprint().min_y == pytest.approx(0.0, abs=1e-6)
        deck_rail = next(p for p in vault.props if p.x == x and "catwalk handrail" in p.label)
        # the top post spans the deck rail's vertical extent, at the deck rail's start - and
        # caps it by a centimetre (level with it, the two tops shared a plane and z-fought)
        assert top.footprint().min_y == pytest.approx(deck_rail.footprint().min_y, abs=1e-6)
        assert top.footprint().max_y == pytest.approx(deck_rail.footprint().max_y + 0.01, abs=1e-6)
        assert top.z == pytest.approx(deck_rail.footprint().min_z, abs=1e-6) == cw.stairs_z_end
        sloped = next(p for p in vault.props if p.x == x and "sloped" in p.label)
        fp = sloped.footprint()
        assert fp.max_y == pytest.approx(deck_rail.footprint().max_y, abs=0.06), (
            "rail meets the deck rail"
        )
        assert fp.min_z >= cw.stairs_z_start - 0.06 and fp.max_z <= cw.stairs_z_end + 0.06


def test_stairs_leave_an_approach_lane_in_front_of_them():
    cw = LEVEL.catwalk
    entrance_wall_z = 18.3
    assert cw.stairs_z_start - entrance_wall_z >= 2.0, "room to walk up to the bottom step"


def test_walkable_surfaces_are_tagged():
    surfaces = [p for p in LEVEL.props if "surface" in p.tags]
    assert len(surfaces) == 3 + 2 * LEVEL.catwalk.steps  # 2 decks, skybridge, every step


def test_labels_are_present_for_readability():
    unlabeled = [p for p in LEVEL.props if not p.label]
    assert not unlabeled, f"{len(unlabeled)} props without labels"


# ----------------------------------------------------------------------------- sight
SIGHT_BAND = (0.2, 4.6)  # vertical range eye->chest rays can occupy
CLUTTER_VOLUME = 0.05  # m^3: cans, chips, cables are allowed to be see-through
DECAL_THICKNESS = 0.1  # thin plates mounted on a blocker count as part of it


def _overlaps_3d(a: AABB, b: AABB, pad: float = 0.0) -> bool:
    return (
        a.min_x - pad < b.max_x and b.min_x < a.max_x + pad
        and a.min_z - pad < b.max_z and b.min_z < a.max_z + pad
        and a.min_y - pad < b.max_y and b.min_y < a.max_y + pad
    )  # fmt: skip


def test_every_opaque_prop_blocks_sight_unless_declared_transparent():
    """Every solid object bounces vision unless it is transparent. A prop bigger than clutter
    inside the sight band must overlap a sight blocker (collider or occluder); thin decals
    mounted on a blocker count; anything else must carry the 'transparent' tag."""
    blockers = ROOM.get_sight_blockers()
    offenders = []
    for p in LEVEL.props:
        if "transparent" in p.tags:
            continue
        fp = p.footprint()
        dims = (fp.max_x - fp.min_x, fp.max_y - fp.min_y, fp.max_z - fp.min_z)
        if dims[0] * dims[1] * dims[2] < CLUTTER_VOLUME:
            continue
        if fp.max_y <= SIGHT_BAND[0] or fp.min_y >= SIGHT_BAND[1]:
            continue
        if any(_overlaps_3d(fp, b) for b in blockers):
            continue
        if min(dims) <= DECAL_THICKNESS and any(
            _overlaps_3d(fp, b, DECAL_THICKNESS) for b in blockers
        ):
            continue  # decal on a wall / cabinet face
        offenders.append((p.label, (p.x, p.y, p.z)))
    assert not offenders, f"opaque props that do not block sight: {offenders}"


def test_transparent_colliders_are_explicitly_justified():
    """A see-through collider must correspond to geometry that really has gaps (balusters)."""
    vault = LEVEL.sector("DatacoreVault")
    for c in vault.colliders:
        if c.blocks_sight:
            continue
        balusters = [p for p in vault.props if "baluster" in p.label and c.contains_xz(p.x, p.z)]
        assert balusters, f"see-through collider {c} has no baluster geometry"


def test_walkable_surfaces_are_occluders():
    occluders = [o for s in LEVEL.sectors for o in s.occluders]
    for p in LEVEL.props:
        if "surface" not in p.tags:
            continue
        fp = p.footprint()
        assert any(_overlaps_3d(fp, o) for o in occluders), f"{p.label} has no occluder"


# ----------------------------------------------------------------------------- netrunning
def test_hack_target_ids_are_explicit_and_unique():
    ids = [
        *(t.terminal_id for t in LEVEL.terminals),
        *(d.door_id for d in LEVEL.doors),
        *(c.camera_id for c in LEVEL.cameras),
        *(v.machine_id for v in LEVEL.vending_machines),
        *(g.guard_id for g in LEVEL.guards),
    ]
    assert all(ids), "the shipped level names every device"
    assert len(ids) == len(set(ids))


def test_every_entity_references_a_declared_faction():
    declared = {f.faction_id for f in LEVEL.factions}
    assert declared == {"halcyon", "halcyon_bio"}
    assert set(LEVEL.faction_ids_in_use) == declared


def test_the_doors_demonstrate_every_kind_of_lock():
    blast, bulkhead, greenhouse, gate, exit_door = LEVEL.doors
    assert blast.door_id == "blast_door" and blast.high_security
    assert blast.unlocked_by == "staging_terminal"
    assert bulkhead.door_id == "vault_bulkhead" and not bulkhead.high_security
    assert bulkhead.unlocked_by is None, "a plain networked lock: quickhacks only"
    assert greenhouse.door_id == "greenhouse_door" and greenhouse.high_security
    assert greenhouse.key_intel == "greenhouse door cipher" and greenhouse.unlocked_by is None
    assert gate.door_id == "maze_gate" and gate.unlocked_by == "node_east_terminal"
    assert exit_door.door_id == "street_exit" and exit_door.unlocked_by == "node_west_terminal"
    assert gate.high_security and exit_door.high_security
    assert gate.faction == exit_door.faction == "halcyon_bio"


def test_the_vault_bulkhead_fills_the_partition_opening():
    """The partition wall at z=18 leaves a 3.2 m x 4.2 m opening; the bulkhead's obstacle
    must seal exactly that (no gap to squeeze through, no overlap into the walls)."""
    bulkhead = LEVEL.doors[1]
    o = bulkhead.obstacle
    assert (o.min_x, o.max_x) == (-1.6, 1.6) and o.min_z < 18.0 < o.max_z
    assert o.min_y == 0.0 and o.max_y == pytest.approx(4.2)
    vault = LEVEL.sector("DatacoreVault")
    lintel = [c for c in vault.colliders if c.min_x == -1.6 and c.max_x == 1.6 and c.min_y > 0]
    assert lintel and lintel[0].min_y == pytest.approx(o.max_y), "the leaf meets the lintel"
    # The leaf prop covers the obstacle (see test_no_invisible_walls) and slides into the wall
    leaf = bulkhead.leaf[0].footprint()
    assert leaf.min_x <= o.min_x and leaf.max_x >= o.max_x and leaf.max_y >= o.max_y - 0.01


def test_guards_declare_who_is_essential():
    staging, vault, gh_guard, gh_elite = LEVEL.guards
    assert staging.guard_id == "staging_guard" and not staging.essential
    assert vault.guard_id == "vault_guard" and vault.essential
    assert gh_guard.guard_id == "greenhouse_guard" and not gh_guard.essential
    assert gh_elite.guard_id == "greenhouse_elite" and gh_elite.essential
    assert gh_guard.patrol_route and gh_elite.patrol_route and gh_guard.wall_pos is None


# ----------------------------------------------------------------------------- greenhouse
def _maze_path(walls, start, goal, blocked=()):
    """BFS over the greenhouse cell graph; ``walls``/``blocked`` are cell-pair edges."""
    from collections import deque

    from cyberduke.levels.facility import GH_CELLS

    cells = set(GH_CELLS)
    closed = {tuple(sorted(w)) for w in walls} | {tuple(sorted(b)) for b in blocked}
    prev = {start: None}
    queue = deque([start])
    while queue:
        c = queue.popleft()
        if c == goal:
            break
        for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            n = (c[0] + d[0], c[1] + d[1])
            if n in cells and n not in prev and tuple(sorted((c, n))) not in closed:
                prev[n] = c
                queue.append(n)
    if goal not in prev:
        return None
    path, c = [], goal
    while c is not None:
        path.append(c)
        c = prev[c]
    return path[::-1]


def test_greenhouse_maze_has_a_short_gated_route_and_a_long_way_round():
    from cyberduke.levels.facility import GH_CELLS, GH_GATE_CELLS, GH_WALLS

    entry, exit_cell = (0, -4), (0, 4)
    assert entry in GH_CELLS and exit_cell in GH_CELLS and len(GH_CELLS) == 69
    short = _maze_path(GH_WALLS, entry, exit_cell)
    long = _maze_path(GH_WALLS, entry, exit_cell, blocked=[GH_GATE_CELLS])
    assert short is not None and long is not None, "connected either way"
    assert tuple(sorted(GH_GATE_CELLS)) in {
        tuple(sorted(e)) for e in zip(short, short[1:], strict=False)
    }, "the gate sits on the short route"
    assert len(short) - 1 == 16 and len(long) - 1 == 30, "the detour is nearly twice as long"
    # every cell is reachable from the entrance (no sealed pockets)
    for cell in GH_CELLS:
        assert _maze_path(GH_WALLS, entry, cell) is not None, cell


def test_greenhouse_control_nodes_are_open_chambers_on_their_routes():
    from cyberduke.levels.facility import GH_CHAMBERS, GH_GATE_CELLS, GH_WALLS

    closed = {tuple(sorted(w)) for w in GH_WALLS}
    for ci, cj in GH_CHAMBERS:
        block = [(ci, cj), (ci + 1, cj), (ci, cj + 1), (ci + 1, cj + 1)]
        for a in block:
            for b in block:
                if abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1:
                    assert tuple(sorted((a, b))) not in closed, f"chamber {ci, cj} is split"
    east = [(3, -2), (4, -2), (3, -1), (4, -1)]
    west = [(-4, 0), (-3, 0), (-4, 1), (-3, 1)]
    short = _maze_path(GH_WALLS, (0, -4), (0, 4))
    long = _maze_path(GH_WALLS, (0, -4), (0, 4), blocked=[GH_GATE_CELLS])
    assert any(c in east for c in short), "the east node (gate terminal) is on the short route"
    assert not any(c in west for c in short)
    assert any(c in west for c in long), "the west node (exit terminal) is on the long route"
    east_t = next(t for t in LEVEL.terminals if t.terminal_id == "node_east_terminal")
    west_t = next(t for t in LEVEL.terminals if t.terminal_id == "node_west_terminal")
    from cyberduke.levels.facility import gh_cell_center

    ex, ez = gh_cell_center(4, -1)
    assert abs(east_t.x - ex) < 1.5 and abs(east_t.z - ez) < 3.0
    wx, wz = gh_cell_center(-4, 0)
    assert abs(west_t.x - wx) < 1.5 and abs(west_t.z - wz) < 3.0


def test_greenhouse_cameras_form_a_hop_chain_within_quickhack_range():
    from cyberduke.config import HACKING
    from cyberduke.core.geometry import distance_3d

    cams = {c.camera_id: c for c in LEVEL.cameras}
    chain = [
        "greenhouse_entry_camera",
        "east_node_camera",
        "greenhouse_spine_camera",
        "exit_camera",
    ]
    for a, b in zip(chain, chain[1:], strict=False):
        d = distance_3d((cams[a].x, cams[a].y, cams[a].z), (cams[b].x, cams[b].y, cams[b].z))
        assert d <= HACKING.quickhack_range, f"{a} -> {b} is {d:.1f} m"
        assert d <= HACKING.scan_range
    assert all(cams[c].faction == "halcyon_bio" for c in chain)
    assert all(cams[c].y > 2.6 for c in chain), "above the hedges, so feeds see each other"


def test_vault_lights_are_a_matched_hanging_pair_flanking_the_monolith():
    vault = LEVEL.sector("DatacoreVault")
    hanging = [light for light in vault.lights if light.fixture == "hanging_cage"]
    assert len(hanging) == 2 and not any(light.fixture == "flood_bar" for light in vault.lights)
    left, right = sorted(hanging, key=lambda light: light.x)
    assert left.x == -right.x and left.z == right.z == 32.0 and left.y == right.y
    assert left.ceiling_y == right.ceiling_y == pytest.approx(5.98), "cables reach the 6 m ceiling"
    assert abs(left.x) < 6.0, "inside the catwalk columns, over the floor"


def test_light_budget_is_exactly_full():
    assert len(LEVEL.lights) == 8


def test_exit_zone_lies_past_the_street_exit_door_inside_the_vestibule():
    zone = LEVEL.exit_zone
    street = next(d for d in LEVEL.doors if d.door_id == "street_exit")
    assert zone is not None and zone.min_z > street.obstacle.max_z
    assert zone.min_x >= street.obstacle.min_x and zone.max_x <= street.obstacle.max_x
    assert ROOM.is_valid_position(zone.center_x, zone.center_z)


def test_dome_glass_tiers_lean_inward_and_meet_each_other():
    """Each sloping tier's top edge is the next tier's bottom edge, so the dome closes."""
    panels = [p for p in LEVEL.sector("FungalGreenhouse").props if p.label == "Dome glass panel"]
    assert len(panels) == 32
    east = sorted((p for p in panels if abs(p.z - 61.0) < 0.01 and p.x > 0), key=lambda p: p.y)
    lower, upper = east
    lower_top = lower.local_to_world(0.0, lower.size[1] / 2.0, 0.0)
    lower_bottom = lower.local_to_world(0.0, -lower.size[1] / 2.0, 0.0)
    upper_bottom = upper.local_to_world(0.0, -upper.size[1] / 2.0, 0.0)
    upper_top = upper.local_to_world(0.0, upper.size[1] / 2.0, 0.0)
    assert lower_bottom[0] == pytest.approx(14.7) and lower_bottom[1] == pytest.approx(5.0)
    assert lower_top[0] == pytest.approx(11.0) and lower_top[1] == pytest.approx(8.0)
    assert upper_bottom[0] == pytest.approx(11.0) and upper_top[1] == pytest.approx(9.6)
    assert upper_top[0] < upper_bottom[0] < lower_bottom[0], "leaning inward, tier on tier"


# ----------------------------------------------------------------------------- set dressing
def _mushrooms():
    caps = [p for p in LEVEL.props if p.label == "Mushroom cap"]
    stalks = [p for p in LEVEL.props if p.label == "Mushroom stalk"]
    return caps, stalks


def _overlap_3d_strict(a: AABB, b: AABB, eps: float = 0.005) -> bool:
    return (
        a.min_x + eps < b.max_x and b.min_x + eps < a.max_x
        and a.min_z + eps < b.max_z and b.min_z + eps < a.max_z
        and a.min_y + eps < b.max_y and b.min_y + eps < a.max_y
    )  # fmt: skip


def test_fungus_grows_in_varied_clusters_not_rows():
    caps, stalks = _mushrooms()
    assert len(caps) == len(stalks) >= 80, "plenty of it, all over the greenhouse"
    radii = sorted(c.size[0] for c in caps)
    assert radii[0] < 0.1 and radii[-1] > 0.3, "tiny to large"
    assert len({round(r, 2) for r in radii}) > 20, "no two alike"
    assert not [p for p in LEVEL.props if "trunk" in p.label.lower()], "no more giant fungus"
    node_beds = [p for p in LEVEL.props if p.label == "Hedge post planter"]
    assert len(node_beds) >= 40, "every hedge post is a small planter now"
    on_node_beds = [c for c in caps if any(b.footprint().contains_xz(c.x, c.z) for b in node_beds)]
    assert len(on_node_beds) >= len(node_beds) * 1.5, "each covered in a spread of small ones"
    assert max(c.size[0] for c in on_node_beds) <= 0.26 * 0.5 + 1e-6, "small ones only"
    # and the fungus lives in the biodome only
    for name in ("StagingBay", "AirlockCorridor", "DatacoreVault"):
        assert not [p for p in LEVEL.sector(name).props if p.label == "Mushroom cap"], name


def test_no_mushroom_clips_anything():
    """Every cap and stalk is clear of every solid prop (racks, trunks, desks, walls...),
    of every other mushroom, and sits on a surface - never inside a planter bed, never
    poking through a hedge, never through the dome wall."""
    caps, stalks = _mushrooms()
    solids = [
        p for p in LEVEL.props if p.solid and p.label not in ("Mushroom cap", "Mushroom stalk")
    ]
    solid_boxes = [p.footprint() for p in solids]
    colliders = _colliders()
    for cap in caps:
        fp = cap.footprint()
        for box in solid_boxes:
            assert not _overlap_3d_strict(fp, box), (
                f"cap at ({cap.x:.2f}, {cap.z:.2f}) clips a solid prop"
            )
        if cap.y < 0.3:  # a floor mushroom: never inside a wall, hedge or desk collider
            for c in colliders:
                assert not c.contains_xz(cap.x, cap.z, margin=cap.size[0] - 0.01), (
                    f"cap at ({cap.x:.2f}, {cap.z:.2f}) grows inside a collider"
                )
    for i, a in enumerate(caps):
        fa = a.footprint()
        for b in caps[i + 1 :]:
            fb = b.footprint()
            if _overlap_3d_strict(fa, fb):
                # circles, not boxes: real overlap means the centres are closer than the radii
                d = math.hypot(a.x - b.x, a.z - b.z)
                assert d >= a.size[0] + b.size[0] - 0.005, (
                    f"caps at ({a.x:.2f}, {a.z:.2f}) and ({b.x:.2f}, {b.z:.2f}) intersect"
                )
    bed_tops = {0.0, 0.56} | {
        p.y + p.size[1] / 2.0 for p in LEVEL.props if p.label == "Hedge post planter"
    }
    for stalk in stalks:
        assert any(abs(stalk.y - top) < 1e-6 for top in bed_tops), "on the floor or on a bed top"


def test_no_trash_clips_anything_or_lands_where_the_game_needs_the_floor():
    """Every piece of generated clutter is clear of every collider, desk, door and of every
    other piece; none of it sits in a doorway, on a patrol lane or on a terminal's chair
    side, and all of it is walk-over (non-solid) so no route is ever blocked."""
    from cyberduke.entities import Terminal

    clutter = [p for p in LEVEL.props if "clutter" in p.tags and "Mushroom" not in p.label]
    assert len(clutter) >= 60, "the place is a mess"
    assert all(not p.solid for p in clutter)
    colliders = _colliders()
    for p in clutter:
        fp = p.footprint()
        if p.label == "Spilled soil":
            continue  # spills run up against the beds on purpose
        for c in colliders:
            if c.max_y > 0.05 and c.min_y < fp.max_y:
                assert not (
                    c.min_x < fp.max_x - 0.02 and fp.min_x + 0.02 < c.max_x
                    and c.min_z < fp.max_z - 0.02 and fp.min_z + 0.02 < c.max_z
                ), f"{p.label!r} at ({p.x:.2f}, {p.z:.2f}) clips a collider"  # fmt: skip
    for t in LEVEL.terminals:
        desk = Terminal(t.x, t.y, t.z, yaw=t.yaw)
        bounds = desk.get_bounds()
        fx, fz = desk.front
        for p in clutter:
            assert not bounds.contains_xz(p.x, p.z), f"{p.label!r} is on the {t.terminal_id} desk"
            assert math.hypot(p.x - fx, p.z - fz) > 0.45, (
                f"{p.label!r} is where you stand at {t.terminal_id}"
            )
    for d in LEVEL.doors:
        o = d.obstacle
        for p in clutter:
            assert not (o.min_x <= p.x <= o.max_x and o.min_z - 1.0 <= p.z <= o.max_z + 1.0), (
                f"{p.label!r} is in the {d.door_id} doorway"
            )
    for g in LEVEL.guards:
        for p in clutter:
            if g.patrol_route:
                assert all(math.hypot(p.x - px, p.z - pz) > 0.45 for px, pz in g.patrol_route), (
                    f"{p.label!r} is on a waypoint of {g.guard_id}"
                )
            else:
                assert not (
                    abs(p.x - g.x) < 0.5 and g.patrol_z_min - 0.4 < p.z < g.patrol_z_max + 0.4
                ), f"{p.label!r} is on {g.guard_id}'s lane"


def test_every_desk_looks_different():
    styles = [t.style for t in LEVEL.terminals]
    assert len(set(styles)) >= 4, styles
    assert "damaged" in styles and "single" in styles and "overgrown" in styles


# ----------------------------------------------------------------------------- collider heights
def _collider_height_excess(level, margin=0.3):
    """Colliders that reach more than ``margin`` above everything drawn inside them: an
    invisible wall over a low planter blocks sight lines and thrown noises for nothing."""
    props = [
        p
        for p in (*level.props, *(q for d in level.doors for q in d.leaf))
        if p.kind != "disc" and "fog" not in p.tags and "clutter" not in p.tags
    ]
    bad = []
    for sector in level.sectors:
        for c in sector.colliders:
            tops = []
            for p in props:
                f = p.footprint()
                ox = min(f.max_x, c.max_x) - max(f.min_x, c.min_x)
                oz = min(f.max_z, c.max_z) - max(f.min_z, c.min_z)
                if ox > 0.02 and oz > 0.02 and f.min_y < c.max_y and f.max_y > c.min_y + 0.01:
                    tops.append(f.max_y)
            if not tops or c.max_y - max(tops) > margin:
                bad.append((sector.name, c, max(tops, default=None)))
    return bad


def test_no_collider_is_taller_than_what_is_drawn_inside_it():
    from cyberduke.levels import build_night_market

    for level in (LEVEL, build_night_market()):
        assert _collider_height_excess(level) == [], level.name


def test_greenhouse_planters_are_exactly_as_tall_as_their_tallest_mushroom():
    from cyberduke.levels.facility import GH_NODE_PLANTERS

    colliders = LEVEL.sector("FungalGreenhouse").colliders
    caps = [p.footprint() for p in LEVEL.props if p.label == "Mushroom cap"]
    for planter in GH_NODE_PLANTERS:
        collider = next(
            c
            for c in colliders
            if (c.min_x, c.max_x, c.min_z, c.max_z)
            == (planter.min_x, planter.max_x, planter.min_z, planter.max_z)
        )
        on_it = [
            c.max_y
            for c in caps
            if planter.min_x <= c.center_x <= planter.max_x
            and planter.min_z <= c.center_z <= planter.max_z
        ]
        assert on_it, "every planter grows something"
        assert collider.max_y == pytest.approx(max(planter.max_y, *on_it), abs=1e-3)
        assert collider.max_y < 1.6, "a standing netrunner stays visible over it"


# ----------------------------------------------------------------------------- fungus light
def _boxes_under(x: float, z: float, top: float):
    return [
        f
        for p in LEVEL.props
        if p.kind == "box" and not (p.tags & {"clutter", "fog", "transparent"})
        for f in [p.footprint()]
        if f.min_x <= x <= f.max_x and f.min_z <= z <= f.max_z and abs(f.max_y - top) < 0.06
    ]


def test_every_mushroom_lights_the_surface_it_grows_on_and_no_further():
    """The pool of light under a cap lies on top of whatever the mushroom grows on - above
    the moss of a planter, where it used to be buried - and stops at that surface's edges
    instead of hanging in the air past the rim of the planter."""
    pools = [p for p in LEVEL.props if p.label == "Mushroom glow"]
    caps = [p for p in LEVEL.props if p.label == "Mushroom cap"]
    assert len(pools) == len(caps) > 200
    for pool in pools:
        ground = min((abs(pool.y - t) for t in (0.0, 0.42, 0.46, 0.5, 0.56)), default=1.0)
        assert ground < 0.03, f"pool at {pool.y:.3f} is not on a floor or a planter"
        under = _boxes_under(pool.x, pool.z, round(pool.y - 0.008, 3))
        assert under, f"nothing under the pool at ({pool.x:.2f}, {pool.z:.2f})"
        top = max(f.max_y for f in under)
        assert top <= pool.y <= top + 0.012, "on the surface (above the moss), not in it"
        widest = max(under, key=lambda f: (f.max_x - f.min_x) * (f.max_z - f.min_z))
        fp = pool.footprint()
        assert widest.min_x - 1e-6 <= fp.min_x and fp.max_x <= widest.max_x + 1e-6
        assert widest.min_z - 1e-6 <= fp.min_z and fp.max_z <= widest.max_z + 1e-6


def test_mushroom_light_splashes_up_the_walls_next_to_it():
    """Next to a rack, a planter side or the dome wall the light goes up the face too: a
    soft splash lying on the face (not floating in front of it, not behind it)."""
    from cyberduke.levels.dressing import SPLASH_OFF, _faces

    splashes = [p for p in LEVEL.props if p.label == "Mushroom glow on the wall"]
    assert len(splashes) > 40
    faces = [f for p in LEVEL.props if p.kind == "box" for f in _faces(p)]
    for s in splashes:
        yaw = s.rotate[0][0]
        nx, nz = math.sin(math.radians(yaw)), math.cos(math.radians(yaw))
        on = [
            f
            for f in faces
            if abs(f.nx - nx) < 1e-3
            and abs(f.nz - nz) < 1e-3
            and abs((s.x - f.x) * nx + (s.z - f.z) * nz - SPLASH_OFF) < 0.005
            and abs((s.x - f.x) * -nz + (s.z - f.z) * nx) <= f.half + s.size[0]
        ]
        assert on, f"splash at ({s.x:.2f}, {s.y:.2f}, {s.z:.2f}) is on no wall face"
        fp = s.footprint()  # (clipped: what is left of the disc)
        assert fp.min_y >= min(f.y0 for f in on) - 1e-3 and fp.max_y <= max(f.y1 for f in on) + 1e-3
        f = on[0]
        ends = [
            (c[0] - f.x) * -nz + (c[1] - f.z) * nx
            for c in ((fp.min_x, fp.min_z), (fp.max_x, fp.max_z))
        ]
        assert min(ends) >= -f.half - 1e-3 and max(ends) <= f.half + 1e-3, "cut to the face's width"
        assert 0.0 < s.material.alpha < 0.5, "fainter than the pool on the ground"
