from cyberduke.core.geometry import AABB
from cyberduke.levels import build_facility
from cyberduke.world import Room


def make_room():
    return Room(build_facility())


def blast_door(room):
    return room.doors["blast_door"]


def test_all_obstacles_are_aabbs_with_explicit_heights():
    for obs in make_room().get_obstacles():
        assert isinstance(obs, AABB)
        assert obs.max_y > obs.min_y
        assert obs.max_x > obs.min_x and obs.max_z > obs.min_z


def test_only_the_stair_balustrades_are_see_through():
    """Everything solid blocks sight unless it is genuinely transparent: the stair rails are
    balusters with gaps; the catwalk rails are solid panels and must occlude."""
    room = make_room()
    see_through = [o for o in room.get_obstacles() if not o.blocks_sight]
    assert len(see_through) == 2
    cw = room.catwalk
    assert all((o.min_z, o.max_z) == (cw.stairs_z_start, cw.stairs_z_end) for o in see_through)


def test_sight_blockers_include_walkable_occluders_but_not_transparent_rails():
    room = make_room()
    blockers = room.get_sight_blockers()
    assert all(b.blocks_sight for b in blockers)
    assert any(b.min_y == 2.65 and b.max_y == 2.75 for b in blockers), "catwalk deck occludes"
    assert len(blockers) > len([o for o in room.get_obstacles() if o.blocks_sight])


def test_blast_door_obstacle_is_present_only_while_shut():
    room = make_room()
    door = blast_door(room)
    assert door.obstacle in room.get_obstacles()
    door.open_amount = door.travel
    assert door.obstacle not in room.get_obstacles()


def test_door_opens_only_after_unlock_and_stops_at_full_travel():
    room = make_room()
    door = blast_door(room)
    room.update(1.0)
    assert door.open_amount == 0.0
    door.unlocked = True
    for _ in range(100):
        room.update(0.1)
    assert door.open_amount == door.travel
    assert door.passable


def test_is_valid_position_tracks_door_state():
    room = make_room()
    assert room.is_valid_position(0.0, 6.0)  # spawn
    assert room.is_valid_position(0.0, 14.0)  # airlock interior is always a valid area...
    assert not room.is_valid_position(0.0, 10.0)  # ...but the doorway strip is sealed
    assert not room.is_valid_position(0.0, 10.3)  # (plus the player radius on the far side)
    blast_door(room).open_amount = blast_door(room).travel
    assert room.is_valid_position(0.0, 10.0)
    assert room.is_valid_position(0.0, 10.3)
    assert room.is_valid_position(0.0, 30.0)
    assert not room.is_valid_position(20.0, 30.0)


def test_every_light_has_a_visible_fixture():
    for light in make_room().get_lights():
        assert light.fixture != "none", f"light at {light.position} has no fixture"


def test_fixed_function_light_budget():
    assert len(make_room().get_lights()) <= 8
