import pytest

from unprepared.config import InteractionSettings
from unprepared.core.math3d import Vector3D
from unprepared.entities import AIState, Door, Item, NPC, Player
from unprepared.systems import NPCAction, TargetKind, find_interaction_target

S = InteractionSettings()


@pytest.fixture
def player():
    # facing -z
    return Player(Vector3D(0.0, 0.0, 0.0), inventory=[])


@pytest.fixture
def door():
    # hinge at z=-3 directly ahead; aim point (1.5, 1.5, -3)
    return Door("shop_door", Vector3D(0.0, 0.0, -3.0), is_locked=True)


@pytest.fixture
def npc():
    bob = NPC("Bob", False, False, True, Vector3D(0.0, 0.0, -1.4))
    bob.state = AIState.SHOPKEEPING
    return bob


def test_nothing_in_range(player):
    assert find_interaction_target(player, [], [], S) is None


def test_door_locked_no_tools(player, door):
    t = find_interaction_target(player, [door], [], S)
    assert t.kind is TargetKind.DOOR and t.obj is door and t.prompt == "[Locked]"
    assert t.is_door and not t.is_npc


def test_door_with_lockpick_needs_crouching(player, door):
    player.inventory.append(Item("lp", "Pick", is_lockpick=True))
    assert find_interaction_target(player, [door], [], S).prompt == "[Locked]"
    player.is_crouching = True
    assert find_interaction_target(player, [door], [], S).prompt == "[E] Pick Lock"
    player.inventory.clear()
    assert find_interaction_target(player, [door], [], S).prompt == "[Locked] (no lockpick)"


def test_key_beats_lockpick(player, door):
    player.inventory += [Item("lp", "Pick", is_lockpick=True), Item("k", "Key", is_key=True, key_door_id="shop_door")]
    assert find_interaction_target(player, [door], [], S).prompt == "[E] Unlock with Key"


def test_wrong_key_is_not_enough(player, door):
    player.inventory.append(Item("k", "Key", is_key=True, key_door_id="other"))
    assert find_interaction_target(player, [door], [], S).prompt == "[Locked]"


def test_unlocked_door_prompts(player, door):
    door.is_locked = False
    assert find_interaction_target(player, [door], [], S).prompt == "[E] Open Door"
    door.is_open = True
    assert find_interaction_target(player, [door], [], S).prompt == "[E] Close Door"


def test_door_out_of_range(player, door):
    door.hinge_position.z = -6.0   # eye->hinge distance > 5
    assert find_interaction_target(player, [door], [], S) is None


def test_door_behind_player(player, door):
    door.hinge_position.z = 3.0
    assert find_interaction_target(player, [door], [], S) is None


def test_most_centred_candidate_wins(player, door, npc):
    door.hinge_position.x = 0.3                 # leaf 0.3..3.3: the crosshair ray (x = 0) misses it
    npc.position.z = -2.8                       # NPC straight ahead (just in front of the door), the door's aim point off to the right
    t = find_interaction_target(player, [door], [npc], S)
    assert t.kind is TargetKind.NPC
    npc.position.x = 2.5                        # NPC off to the side: the door is now better centred
    assert find_interaction_target(player, [door], [npc], S).kind is TargetKind.DOOR


def test_walls_hide_npcs_and_containers(player, npc):
    from unprepared.core.math3d import BoundingBox3D
    wall = BoundingBox3D(Vector3D(-2, 0, -1.0), Vector3D(2, 4, -0.8))
    assert find_interaction_target(player, [], [npc], S) is not None
    assert find_interaction_target(player, [], [npc], S, occluders=[wall]) is None


def test_npc_prompts(player, npc):
    t = find_interaction_target(player, [], [npc], S)
    assert t.kind is TargetKind.NPC and t.obj is npc and t.prompt == "[E] Talk / Trade with Bob"
    npc.state = AIState.EATING            # shop closed: talk only
    assert find_interaction_target(player, [], [npc], S).prompt == "[E] Talk to Bob"
    npc.state = AIState.SHOPKEEPING
    npc.can_trade = False
    assert find_interaction_target(player, [], [npc], S).prompt == "[E] Talk to Bob"


def test_npc_out_of_range(player, npc):
    npc.position.z = -8.5   # beyond the range and the crosshair ray's reach (range + 2)
    assert find_interaction_target(player, [], [npc], S) is None


def test_npc_talk_distance_is_natural(player, npc):
    npc.position.z = -5.0   # standing a good five units away
    assert find_interaction_target(player, [], [npc], S).prompt == "[E] Talk / Trade with Bob"


def test_crouching_turns_talk_into_pickpocket(player, npc):
    player.is_crouching = True
    player.current_eye_height = 0.9
    t = find_interaction_target(player, [], [npc], S)
    assert t.prompt == "[E] Pickpocket Bob" and t.npc_action is NPCAction.PICKPOCKET
    npc.position.z = -3.0   # beyond the 2.5 pickpocket reach but within talking range
    t = find_interaction_target(player, [], [npc], S)
    assert t.prompt == "[Sneak closer to pickpocket Bob]" and t.npc_action is NPCAction.TOO_FAR_TO_PICKPOCKET


def test_container_target(player):
    from unprepared.core.math3d import BoundingBox3D
    from unprepared.world import Container
    shelf = Container("shelf", "Shop Shelf", BoundingBox3D(Vector3D(-1, 0, -3.5), Vector3D(1, 1.5, -3)))
    t = find_interaction_target(player, [], [], S, containers=[shelf])
    assert t.is_container and t.prompt == "[E] Search Shop Shelf" and t.obj is shelf
    shelf2 = Container("far", "Chest", BoundingBox3D(Vector3D(-1, 0, -6), Vector3D(1, 1, -5)))
    assert find_interaction_target(player, [], [], S, containers=[shelf2]) is None


def test_npc_needs_aim(player, npc):
    player.rotation_y = 90.0
    assert find_interaction_target(player, [], [npc], S) is None


def test_better_aligned_door_wins_regardless_of_order(player):
    a = Door("a", Vector3D(-1.5, 0.0, -3.0), is_locked=False)      # aim point dead ahead
    b = Door("b", Vector3D(0.0, 0.0, -3.5), is_locked=False)       # aim point 1.5 to the right
    assert find_interaction_target(player, [b, a], [], S).obj is a
    assert find_interaction_target(player, [a, b], [], S).obj is a


def test_custom_settings(player, npc):
    tight = InteractionSettings(npc_range=1.0, crosshair_ray=False)
    assert find_interaction_target(player, [], [npc], tight) is None                                  # eye 2.1 away
    assert find_interaction_target(player, [], [npc], InteractionSettings(npc_range=2.2, crosshair_ray=False)) is not None


def test_matches_baseline_targeting(baseline):
    """Randomised comparison with the original ``get_targeted_interaction`` (same geometry;
    the prototype's 3.0 NPC range is used since the game now doubles it, and the new
    crosshair-ray and "must face the doorway" rules are switched off because the prototype
    had neither - this checks the aim-cone fallback alone)."""
    import random
    from types import SimpleNamespace
    L = baseline.models
    new = SimpleNamespace(Vector3D=Vector3D, Item=Item, Door=Door, NPC=NPC)
    rng = random.Random(7)

    def build(m, pos, yaw, pitch, has_pick, has_key, locked, opened, near_pos):
        if m is L:
            p = m.Player(m.Vector3D(*pos))
        else:
            from unprepared.config import PlayerSettings
            p = Player(m.Vector3D(*pos), inventory=[], settings=PlayerSettings(eye_height=1.6))   # the prototype's camera
        p.inventory = []
        p.is_crouching = True        # the game only offers lock picking while crouched
        if has_pick:
            p.inventory.append(m.Item("lp", "Pick", is_lockpick=True))
        if has_key:
            p.inventory.append(m.Item("k", "Key", is_key=True, key_door_id="shop_door"))
        p.rotation_y, p.rotation_x = yaw, pitch
        doors = [m.Door("hideout_door", m.Vector3D(-14.5, 0.0, 5.0), is_locked=False),
                 m.Door("shop_door", m.Vector3D(8.5, 0.0, 0.0), is_locked=locked, is_open=opened)]
        npcs = [m.NPC("Barnaby", False, False, True, m.Vector3D(10, 0, -5)),
                m.NPC("Vance", True, False, False, m.Vector3D(-5, 0, 10)),
                m.NPC("Near", False, False, True, m.Vector3D(*near_pos))]
        for n in npcs:
            n.state = m.AIState.SHOPKEEPING if m is L else AIState.SHOPKEEPING   # prompts assume open shops
        return p, doors, npcs

    for _ in range(1500):
        pos = (rng.uniform(-20, 20), 1.0, rng.uniform(-20, 20))
        args = (pos, rng.uniform(0, 360), rng.uniform(-60, 60), rng.random() < 0.5, rng.random() < 0.5,
                rng.random() < 0.5, rng.random() < 0.5,
                (pos[0] + rng.uniform(-3, 3), 0, pos[2] + rng.uniform(-3, 3)))
        lt, lo, lprompt = L_main_target(baseline, *build(L, *args))
        nt = find_interaction_target(*build(new, *args),
                                     InteractionSettings(npc_range=3.0, door_facing_min=0.0, crosshair_ray=False))
        if lt is None:
            assert nt is None, args
        else:
            assert nt is not None, args
            # the prototype gave doors absolute priority; the game now picks the best-centred
            # candidate, so only the door prompt wording is compared when both agree on a door
            if lt == "door" and nt.kind.value == "door":
                assert nt.prompt == lprompt, args
            assert getattr(nt.obj, "door_id", None) == getattr(lo, "door_id", None)
            assert getattr(nt.obj, "name", None) == getattr(lo, "name", None)


def L_main_target(baseline, player, doors, npcs):
    return baseline.main.get_targeted_interaction(player, doors, npcs)


def test_door_seen_through_a_wall_is_not_targetable(player, door):
    """A door off to the side, behind the wall the player faces, must not prompt."""
    from unprepared.core.math3d import BoundingBox3D
    door.hinge_position = Vector3D(0.5, 0.0, -2.0)             # doorway x 0.5..3.5 at z=-2, aim point (2.0, 1.5, -2)
    wall = BoundingBox3D(Vector3D(0.5, 0, -2.0), Vector3D(0.8, 4, 3.0))   # wall along x=0.5 between player and doorway
    assert find_interaction_target(player, [door], [], S) is not None
    assert find_interaction_target(player, [door], [], S, occluders=[wall]) is None


def test_door_seen_edge_on_is_not_targetable(player, door):
    """Walking along a wall, a door in that wall sits inside the aim cone but edge-on: no prompt."""
    door.hinge_position = Vector3D(0.5, 0.0, -2.0)             # aim point (2.0, 1.5, -2)
    player.rotation_y = 90.0                                    # facing +x, parallel to the door's wall
    assert find_interaction_target(player, [door], [], S) is None
    assert find_interaction_target(player, [door], [], InteractionSettings(door_facing_min=0.0)) is not None   # prototype rule
    player.rotation_y = 45.0                                    # turn towards it: fine
    assert find_interaction_target(player, [door], [], S) is not None


def test_closed_door_hides_what_is_behind_it(player, door, npc):
    door.hinge_position.x = 0.5                                 # leaf 0.5..3.5 (the crosshair ray at x = 0 misses it)
    npc.position = Vector3D(2.0, 0.0, -4.5)                     # behind the middle of the closed leaf
    assert find_interaction_target(player, [], [npc], S).kind is TargetKind.NPC
    t = find_interaction_target(player, [door], [npc], S)
    assert t.kind is TargetKind.DOOR                            # the door itself is still targetable
    door.is_open = True                                         # leaf folds away: NPC visible through the doorway
    player.rotation_y = 0.0
    t = find_interaction_target(player, [door], [npc], S)
    assert t.kind is TargetKind.NPC


class TestCrosshairRay:
    """Whatever the crosshair is on is the target - any part of it, at any distance."""

    def test_door_under_the_crosshair_from_up_close(self, player, door):
        player.position = Vector3D(2.5, 0.0, -2.6)              # nose against the far end of the leaf
        player.rotation_x = 10.0                                # looking slightly up at it
        assert find_interaction_target(player, [door], [], S).kind is TargetKind.DOOR
        player.rotation_x = -45.0                               # looking down at its foot
        assert find_interaction_target(player, [door], [], S).kind is TargetKind.DOOR

    def test_door_under_the_crosshair_from_a_steep_angle(self, player, door):
        player.position = Vector3D(-1.5, 0.0, -2.0)             # beside the doorway, looking along the leaf
        player.rotation_y, player.rotation_x = player.yaw_towards(Vector3D(2.9, 0.0, -3.0)), 0.0
        assert find_interaction_target(player, [door], [], S).kind is TargetKind.DOOR

    def test_open_leaf_is_targetable_too(self, player, door):
        door.is_locked = False
        door.is_open = True                                     # leaf folds into -z along x = 0
        player.position = Vector3D(0.8, 0.0, -4.5)              # standing in the doorway, looking at the folded leaf
        player.rotation_y, player.rotation_x = player.yaw_towards(Vector3D(0.0, 0.0, -4.5)), 0.0
        t = find_interaction_target(player, [door], [], S)
        assert t.kind is TargetKind.DOOR and t.prompt == "[E] Close Door"

    def test_wall_in_front_blocks_the_ray(self, player, door):
        from unprepared.core.math3d import BoundingBox3D
        wall = BoundingBox3D(Vector3D(-2, 0, -1.5), Vector3D(2, 4, -1.2))
        assert find_interaction_target(player, [door], [], S, occluders=[wall]) is None

    def test_nearest_hit_wins(self, player, door, npc):
        npc.position = Vector3D(0.0, 0.0, -1.2)                 # between the player and the door, under the crosshair
        player.rotation_x = -35.0                               # looking down at him
        assert find_interaction_target(player, [door], [npc], S).kind is TargetKind.NPC

    def test_ray_can_be_disabled(self, player, door):
        player.position = Vector3D(2.5, 0.0, -2.6)
        player.rotation_x = 10.0
        assert find_interaction_target(player, [door], [], InteractionSettings(crosshair_ray=False)) is None
