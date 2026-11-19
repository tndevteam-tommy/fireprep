"""Game-level glue added in the sixth iteration: fades around relocations, yes/no questions,
lockable containers, screens closing when caught, door lintels, facing the guard."""
import pytest

from unprepared.core.math3d import Vector3D

from tests.helpers import spot
from unprepared.config import GameSettings
from unprepared.entities import AIState, Item
from unprepared.game import Confirmation, Game, GameMode, ScreenFade
from unprepared.world.world import Relocation


@pytest.fixture
def game(tmp_path):
    return Game(settings=GameSettings(echo_messages_to_stdout=False, random_seed=4, max_log_messages=100),
                save_file=str(tmp_path / "s.json"))


def send_guards_away(game):
    for g in game.world.guards():
        g.position = spot(game.world, -38.0, 38.0)
        g.path, g.current_task, g.schedule = [], None, []
        g.state = AIState.GUARDING


class TestScreenFade:
    def test_alpha_curve_and_callback_at_black(self):
        hits = []
        fade = ScreenFade(lambda: hits.append("black"))
        assert fade.alpha == 0.0
        fade.update(0.3)
        assert fade.alpha == pytest.approx(0.5) and hits == []
        fade.update(0.3)                                   # fully black: the move happens now
        assert fade.alpha == 1.0 and hits == ["black"] and fade.applied
        fade.update(0.2 + 0.35)
        assert fade.alpha == pytest.approx(0.5) and hits == ["black"]
        fade.update(0.35)
        assert fade.done and fade.alpha == 0.0

    def test_relocation_is_deferred_until_black_and_freezes_input(self, game):
        before = game.player.position.copy()
        game.world.relocate_player(spot(game.world, 5.0, 5.0), yaw=90.0, reason="test")
        assert game.in_transition and game.world.relocating
        assert game.player.position == before and not game.accepts_free_look
        from unprepared.systems import MoveIntent
        game.update(0.3, MoveIntent(forward=True))
        assert game.player.position == before and not game.movement.is_moving    # frozen
        game.update(0.35)
        assert (game.player.position.x, game.player.position.z) == (5.0, 5.0) and game.player.rotation_y == 90.0
        assert not game.world.relocating and game.fade_alpha == 1.0
        for _ in range(12):
            game.update(0.1)
        assert not game.in_transition and game.accepts_free_look

    def test_two_relocations_in_one_fade(self, game):
        game.world.relocate_player(spot(game.world, 5.0, 5.0))
        game.update(0.7)
        assert game.player.position.x == 5.0
        game.world.relocate_player(spot(game.world, -5.0, -5.0))    # already black: applied at once
        assert game.player.position.x == -5.0 and game.in_transition

    def test_relocation_without_a_handler_is_immediate(self, game):
        game.world.relocation_handler = None
        r = game.world.relocate_player(spot(game.world, 1.0, 2.0), yaw=45.0)
        assert isinstance(r, Relocation) and r.applied
        assert (game.player.position.x, game.player.position.z) == (1.0, 2.0) and game.player.rotation_y == 45.0

    def test_quick_load_cancels_a_fade(self, game):
        game.save_game()
        game.world.relocate_player(spot(game.world, 5.0, 5.0))
        assert game.in_transition
        assert game.load_game()
        assert not game.in_transition and not game.pending_relocations


class TestConfirmations:
    def test_ask_yes_no_and_alt(self, game):
        hits = []
        game.ask("Sure?", on_yes=lambda: hits.append("yes"), on_no=lambda: hits.append("no"),
                 alt_label="[T] alt", on_alt=lambda: hits.append("alt"))
        assert game.mode is GameMode.CONFIRM and not game.mouse_captured and not game.accepts_free_look
        assert isinstance(game.confirmation, Confirmation) and game.confirmation.text == "Sure?"
        assert game.interact() is None and game.mode is GameMode.CONFIRM       # [E] does nothing here
        assert game.confirm_yes() and hits == ["yes"] and game.confirmation is None
        game.ask("Again?", on_yes=lambda: hits.append("yes2"))
        assert game.escape() is True and hits == ["yes"]                        # ESC = no (no handler: fine)
        game.ask("Third?", on_yes=lambda: hits.append("yes3"), on_alt=lambda: hits.append("alt3"), alt_label="[T]")
        assert game.wait() is True and hits[-1] == "alt3"                       # [T] picks the third option
        assert game.confirm_no() is False and game.confirm_alt() is False       # nothing open

    def test_enter_in_a_question_means_yes(self, game):
        hits = []
        game.ask("Q", on_yes=lambda: hits.append(1))
        assert game.transfer_confirm() is None and hits == [1]

    def test_snapshot_shows_the_question(self, game):
        game.ask("Q?", on_yes=lambda: None)
        assert game.snapshot()["confirmation"] == "Q?" and game.snapshot()["mode"] == "confirm"


class TestLockableContainers:
    def go_to_lockbox(self, game):
        send_guards_away(game)
        game.player.position = spot(game.world, 51.2, 14.25)           # in the jail office, east of the lockbox
        game.player.rotation_y, game.player.rotation_x = 270.0, -20.0
        game.update(0.0)

    def test_prompts(self, game):
        self.go_to_lockbox(game)
        assert game.interaction_prompt == "[Confiscation Lockbox: Locked]"
        game.toggle_crouch()
        game.update(0.0)
        assert game.interaction_prompt == "[E] Pick the Confiscation Lockbox lock"
        game.player.inventory = [i for i in game.player.inventory if not i.is_lockpick]
        game.update(0.0)
        assert game.interaction_prompt == "[Confiscation Lockbox: Locked] (no lockpick)"
        game.player.add_item(Item("cell_key", "Cell Key", value=1.0, is_key=True, key_door_id="cell_door"))   # the jailer's key opens it
        game.update(0.0)
        assert game.interaction_prompt == "[E] Unlock Confiscation Lockbox with Key"
        game.interact()
        assert not game.world.find_container("jail_storage").is_locked
        assert game.messages.messages[-1] == "Unlocked the confiscation lockbox with your key."
        game.update(0.0)
        assert game.interaction_prompt == "[E] Search Confiscation Lockbox"

    def test_picking_the_lockbox(self, game):
        self.go_to_lockbox(game)
        game.toggle_crouch()
        game.update(0.0)
        game.interact()
        assert game.mode is GameMode.LOCKPICK and game.lockpicking.target is game.world.find_container("jail_storage")
        assert game.lockpicking.target_id == "jail_storage" and game.lockpicking.difficulty == 3
        assert game.messages.messages[-1].startswith("Started lockpicking the confiscation lockbox (a Strong lock, ")
        game.lockpicking.pick_angle = game.lockpicking.sweet_spot
        game.lockpicking.wobble_amplitude = 0.0
        game.lockpicking.set_chance = 1.0                      # (the pins set this time: a Strong lock beats a novice most days)
        game.lockpick_attempt()
        assert not game.world.find_container("jail_storage").is_locked and game.mode is GameMode.EXPLORE
        game.update(0.0)
        game.interact()
        assert game.mode is GameMode.CONTAINER

    def test_standing_player_is_told_to_crouch(self, game):
        self.go_to_lockbox(game)
        game.interact()
        assert game.messages.messages[-1] == "The confiscation lockbox is locked. Crouch to pick it, or find the key."
        assert game.mode is GameMode.EXPLORE

    def test_lock_state_is_saved(self, game):
        chest = game.world.find_container("jail_storage")
        chest.unlock_and_open()
        game.save_game()
        chest.lock_and_close()
        game.load_game()
        assert not chest.is_locked


class TestBeingCaught:
    def test_screens_close_and_the_player_turns_to_face_the_guard(self, game):
        vance = game.world.find_npc("Guard Captain Vance")
        vance.path, vance.current_task, vance.schedule = [], None, []
        vance.position = spot(game.world, 3.0, 0.0)
        game.player.position = spot(game.world, 0.0, 0.0)
        game.player.rotation_y, game.player.rotation_x = 0.0, -30.0
        game.open_pickpocket(game.world.find_npc("Jae the Fence"))
        assert game.mode is GameMode.PICKPOCKET
        game.world.player_caught(vance)
        assert game.mode is GameMode.EXPLORE
        assert game.player.frozen and game.player.turn_target is not None
        assert game.player.rotation_y == 0.0                     # no jerk: the turn is smooth ...
        yaws = []
        for _ in range(12):
            game.update(0.1)
            yaws.append(game.player.rotation_y)
        assert yaws[0] == pytest.approx(15.0) and all(b >= a for a, b in zip(yaws, yaws[1:]))   # 150 deg/s
        assert game.player.rotation_y == pytest.approx(90.0) and game.player.rotation_x == 0.0  # ... and lands on him
        game.world.find_door("shop_door").is_locked = True
        game.lockpicking.begin(game.world.find_door("shop_door"))
        game.ask("Q", on_yes=lambda: None)
        game.world.player_caught(vance)
        assert not game.lockpicking.active and game.confirmation is None and game.mode is GameMode.EXPLORE

    def test_jailing_mid_minigame_clears_the_screen(self, game):
        vance = game.world.find_npc("Guard Captain Vance")
        vance.path, vance.current_task, vance.schedule = [], None, []
        vance.position = spot(game.world, 16.0, -3.0)
        vance.facing_yaw = 180.0
        game.player.position = spot(game.world, 16.0, -1.0)               # within his reach
        game.player.gold = 0.0
        game.lockpicking.begin(game.world.find_door("shop_door"))
        assert game.in_lockpicking
        vance.remember_crime("Pickpocketing", game.world.day_count, True, 25.0)
        game.update(0.05)
        assert game.world.jail.serving and not game.in_lockpicking and game.mode is GameMode.EXPLORE
        assert game.frozen and game.fade_alpha == 0.0              # a couple of seconds to take it in ...
        for _ in range(16):
            game.update(0.1)
        assert game.world.area_of(game.player.position) == "main street" and game.frozen
        for _ in range(20):
            game.update(0.1)                                        # ... then the fade to the cell
        assert game.world.area_of(game.player.position) == "the jail cell"

    def test_being_caught_locks_the_controls_for_a_moment(self, game):
        from unprepared.systems import MoveIntent
        vance = game.world.find_npc("Guard Captain Vance")
        game.world.player_caught(vance)
        assert game.frozen and not game.accepts_free_look
        start = game.player.position.copy()
        game.update(0.5, MoveIntent(forward=True))
        assert game.player.position == start and not game.interact()
        for _ in range(20):
            game.update(0.1)
        assert not game.frozen and game.accepts_free_look


class TestLintels:
    def test_a_beam_tops_every_door_without_blocking_anyone(self, game):
        w = game.world
        beams, counted = {}, []
        for door in w.doors:                                # (along x or along z: a beam over either)
            h = door.hinge_position
            lo, hi = door.span

            def along(b):
                return ((b.min_point.x, b.max_point.x) if door.along_x else (b.min_point.z, b.max_point.z))

            def across(b):
                return ((b.min_point.z, b.max_point.z) if door.along_x else (b.min_point.x, b.max_point.x))
            over = [b for b in w.lintels if along(b) == (lo, hi) and across(b)[0] <= door.line + 0.26
                    and across(b)[1] >= door.line - 0.26]
            if not over:       # a gate: under the gatehouse's arch; a door in a cave: under the rock
                assert door.gate is not None or w.building_at(door.center).underground, door.door_id
                for k in range(1, 10):                  # (rock or an arch right over it, all along)
                    x, z = door.to_xz(lo + (hi - lo) * k / 10, door.line)
                    assert any(wl.min_point.x <= x <= wl.max_point.x and wl.min_point.z <= z <= wl.max_point.z
                               and h.y + door.height - 0.3 <= wl.min_point.y <= h.y + door.height + 1.2
                               for wl in w.walls_near(door.center, 8.0)), door.door_id
                continue
            # (one beam - or, where a string course crosses it, the pieces of one the course leaves)
            beam = beams[door.door_id] = min(over, key=lambda b: b.min_point.y)
            assert beam.min_point.y == pytest.approx(h.y + door.height), door.door_id
            probes = [door.to_xz(door.along_of(door.center.x, door.center.z), door.line + d) for d in (-0.3, 0.3)]
            roofed = [b for b in (w.building_at(Vector3D(x, h.y + 0.5, z)) for x, z in probes) if b is not None]
            if not roofed:     # a gate in a curtain wall (Greyhold's): the beam fills the gap up to the arch over it
                assert door.gate is not None, door.door_id
                for k in range(1, 10):                  # (beam and arch together close it, all along)
                    x, z = door.to_xz(lo + (hi - lo) * k / 10, door.line)
                    spans = sorted((b.min_point.y, b.max_point.y) for b in (*w.lintels, *w.walls_near(door.center, 8.0))
                                   if b.min_point.x <= x <= b.max_point.x and b.min_point.z <= z <= b.max_point.z)
                    reach = h.y + door.height
                    for y1, y2 in spans:
                        if y1 <= reach + 1e-6:
                            reach = max(reach, y2)
                    assert reach >= h.y + door.height + 0.5, door.door_id
                counted.extend(b for b in w.lintels if b not in counted and b.min_point.y < h.y + door.height + 0.6
                               and b.max_point.y > h.y + door.height - 1e-6
                               and lo - 1e-6 <= along(b)[0] and along(b)[1] <= hi + 1e-6
                               and across(b)[0] <= door.line + 0.26 and across(b)[1] >= door.line - 0.26)
                continue
            eaves = min(b.wall_height / max(1, b.floors) for b in roofed)
            # (up to the lower eaves - or to a wall heading the doorway that stops short of them: the
            # steeple's, under its open belfry)
            heads = [wl.min_point.y for wl in w.walls_near(door.center, 8.0)
                     if along(wl)[0] <= lo + 1e-6 and along(wl)[1] >= hi - 1e-6 and across(wl)[0] <= door.line + 1e-6
                     and across(wl)[1] >= door.line - 1e-6 and h.y + door.height - 0.05 <= wl.min_point.y < h.y + eaves
                     and wl.max_point.y < h.y + eaves - 0.05]
            # (or to the underside of an upper floor's boards over it)
            heads += [f.min_point.y for f in w.props if f.kind == "floor"
                      and f.min_point.x - 0.3 <= door.center.x <= f.max_point.x + 0.3
                      and f.min_point.z - 0.3 <= door.center.z <= f.max_point.z + 0.3
                      and h.y + door.height - 0.05 <= f.min_point.y < h.y + eaves]
            assert max(b.max_point.y for b in over) == pytest.approx(min([h.y + eaves, *heads])), door.door_id
            assert all(b not in w.colliders() and b not in w.static_obstacles() for b in over)
            counted.extend(over)
        assert len(w.lintels) == len(counted)
        textures = {door_id: beam.texture for door_id, beam in beams.items()}
        assert textures["shop_door"] == "building" and textures["guardhouse_door"] == "wall" and textures["cell_door"] == "wall"
        # NPCs still path through the doorways
        hesk = w.find_npc("Jailer Hesk")
        assert w.find_path(hesk, spot(game.world, 53.0, 3.0))
