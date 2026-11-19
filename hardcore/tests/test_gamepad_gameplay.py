"""Playing with a controller: the default layout, analog movement and look, climbing with the
stick, rebinding, menus and feedback - driven through the same GamepadInput / InputRouter the game
loop uses, on the headless simulation.
"""
import math
import numpy as np

from harness import *  # noqa
from pad_helpers import *  # noqa
from core.gamepad import ControllerBindings, PadState, radial_deadzone
from core.input_buffer import InputBuffer
from core.gamepad_input import GamepadInput, Haptics, PAUSE
from core.input_router import InputRouter
from render.hud import ToastQueue


class PadSim:
    """A Sim played with a controller (what main.py does each frame, at the physics rate)."""

    def __init__(self, sim, bindings=None):
        self.sim = sim
        self.bindings = bindings or ControllerBindings()
        self.router = InputRouter(sim.input, lambda: sim.player.scheme)
        self.pad = GamepadInput(self.bindings, self.router)
        self.state = PadState()
        self.results = []

    # buttons / triggers go through PadState exactly like device events do
    def _route(self, transitions):
        for control, down in transitions:
            self.results.append(self.pad.on_button(control, down))

    def press(self, *controls):
        for c in controls:
            self._route(self.state.set_button(c, True))

    def release(self, *controls):
        for c in controls:
            self._route(self.state.set_button(c, False))

    def trigger(self, name, value):
        self._route(self.state.set_axis(name, value))

    def stick(self, side, x, y_up):
        """Stick deflection with y up positive (converted to SDL's y-down)."""
        ax, ay = ("leftx", "lefty") if side == "left" else ("rightx", "righty")
        self.state.set_axis(ax, x)
        self.state.set_axis(ay, -y_up)

    def move(self, x, y_up):
        self.stick(self.bindings.move_stick, x, y_up)

    def look(self, x, y_up):
        self.stick(self.bindings.look_stick, x, y_up)

    def step(self, n=1):
        for _ in range(n):
            turn, up = self.pad.update(self.state, DT)
            self.sim.player.apply_look(turn, up)
            self.sim.step()

    def run(self, seconds, until=None):
        for _ in range(int(round(seconds / DT))):
            self.step()
            if until is not None and until(self.sim):
                return True
        return until is None

    def tap(self, *controls):
        self.press(*controls)
        self.step()
        self.release(*controls)

    @property
    def speed(self):
        return float(np.hypot(self.sim.vel[0], self.sim.vel[2]))


def _open_floor(yaw=0.0):
    ls, obs = build_custom([])
    return PadSim(Sim(ls, obs, spawn=(0.0, 0.9, 0.0), yaw=yaw, scheme="flow"))


def _tower(h=5.4):
    ls, obs = build_custom([(-2.0, 0.0, -5.0, 4.0, h, 4.0)])
    return ls, obs, obs[1]


def _pad_hang(ls, obs, grab="x"):
    """Walk to the wall with the stick, grab with X (the default grab button) -> hang."""
    ps = PadSim(Sim(ls, obs, spawn=(0.0, 0.9, 1.6), scheme="flow"))
    ps.move(0.0, 1.0)
    assert ps.run(3.0, until=lambda s: s.player.grab_candidate() is not None)
    ps.press(grab)
    assert ps.run(1.0, until=lambda s: s.state == ParkourState.LEDGE_HANG)
    ps.release(grab)
    ps.move(0.0, 0.0)
    ps.run(0.3)
    return ps


# ---------------------------------------------------------------- default layout
def test_a_jumps_on_open_ground():
    ps = _open_floor()
    ps.tap("a")
    assert ps.sim.had_event("JUMP")
    assert ps.run(2.0, until=lambda s: s.max_height_reached() > 0.9 + 0.3)


def test_a_again_in_the_air_flips_and_the_stick_picks_the_flip():
    for stick, kind in (((0.0, 1.0), "front"), ((0.0, -1.0), "back"), ((1.0, 0.0), "right")):
        ps = _open_floor()
        ps.move(*stick)
        ps.press("a")
        ps.step(10)
        ps.release("a")
        ps.step(2)
        ps.tap("a")
        spin = ps.sim.fsm.spin
        assert ps.sim.had_event("FLIP") and spin is not None and spin.kind == kind, (stick, spin and spin.kind)
        ps.move(0.0, 0.0)
        assert ps.run(2.0, until=lambda s: s.player.on_ground and s.fsm.spin is None)


def test_x_grabs_a_hold_and_b_lets_go():
    ls, obs, tower = _tower()
    ps = _pad_hang(ls, obs, grab="x")
    assert abs(ps.sim.fsm.active_rail.height - 2.0) < 1e-3
    ps.tap("b")
    assert ps.sim.events[-1][1] == "DROP"
    assert ps.run(1.5, until=lambda s: s.player.on_ground)


def test_holding_a_reaches_and_catches_like_space():
    ls, obs, tower = _tower()
    ps = PadSim(Sim(ls, obs, spawn=(0.0, 3.9, -0.64), scheme="flow"))
    ps.sim.player.on_ground = False
    ps.press("a")
    assert ps.run(2.0, until=lambda s: s.state == ParkourState.LEDGE_HANG), "A held in the air catches the hold"


def test_right_trigger_reaches_so_the_thumb_can_keep_aiming():
    ls, obs, tower = _tower()
    ps = PadSim(Sim(ls, obs, spawn=(0.0, 3.9, -0.64), scheme="flow"))
    ps.sim.player.on_ground = False
    ps.trigger("righttrigger", 0.8)
    assert ps.run(2.0, until=lambda s: s.state == ParkourState.LEDGE_HANG)


def test_left_trigger_is_sprint_without_a_lock():
    ps = _open_floor()
    ps.move(0.0, 1.0)
    ps.trigger("lefttrigger", 1.0)
    ps.run(config.RUN_BUILD_TIME + 1.0)
    assert abs(ps.speed - config.SPRINT_SPEED) < 0.05, "LT held = full sprint after the build-up"
    assert not ps.sim.player.scheme.run_mode, "the trigger never locks sprint"
    ps.trigger("lefttrigger", 0.0)
    ps.run(1.5)
    assert abs(ps.speed - config.WALK_SPEED) < 0.05, "releasing LT drops back to a walk"


def test_a_light_trigger_rest_does_not_sprint():
    ps = _open_floor()
    ps.move(0.0, 1.0)
    ps.trigger("lefttrigger", 0.15)
    ps.run(1.5)
    assert abs(ps.speed - config.WALK_SPEED) < 0.05


def test_left_stick_click_is_the_sprint_lock():
    ps = _open_floor()
    ps.tap("leftstick")
    assert ps.sim.player.scheme.run_mode, "L3 toggles the sprint lock on"
    ps.move(0.0, 1.0)
    ps.run(config.RUN_BUILD_TIME + 1.0)
    assert abs(ps.speed - config.SPRINT_SPEED) < 0.05, "locked: sprinting with no trigger held"
    ps.tap("leftstick")
    assert not ps.sim.player.scheme.run_mode
    ps.run(1.5)
    assert abs(ps.speed - config.WALK_SPEED) < 0.05
    assert Action.RUN_TOGGLE in ps.results, "the toggle is reported so the game can show 'Sprint lock ON/OFF'"


# ------------------------------------------------------------------- movement
def test_stick_tilt_sets_the_walking_speed():
    for tilt in (0.35, 0.6, 1.0):
        ps = _open_floor()
        ps.move(0.0, tilt)
        ps.run(1.5)
        _, _, m = radial_deadzone(0.0, tilt, ps.bindings.deadzone)
        assert abs(ps.speed - config.WALK_SPEED * m) < 0.05, (tilt, ps.speed)
    ps = _open_floor()
    ps.move(0.0, 0.1)
    ps.run(1.0)
    assert ps.speed < 1e-3, "inside the deadzone the player stands still (no stick drift)"


def test_sprint_needs_the_stick_pushed_most_of_the_way():
    ps = _open_floor()
    ps.trigger("lefttrigger", 1.0)
    ps.move(0.0, 0.6)
    ps.run(2.0)
    assert ps.speed < config.WALK_SPEED and ps.sim.state == ParkourState.GROUND, "a half-pushed stick walks"
    ps.move(0.0, 1.0)
    ps.run(1.0)
    assert ps.sim.state == ParkourState.SPRINT and ps.speed > config.JOG_SPEED - 0.1


def test_stick_moves_in_any_direction_not_just_eight():
    ps = _open_floor(yaw=0.0)       # facing -Z, right is +X
    a = math.radians(30.0)          # 30 degrees right of straight ahead
    ps.move(math.sin(a), math.cos(a))
    ps.run(1.5)
    vx, vz = float(ps.sim.vel[0]), float(ps.sim.vel[2])
    heading = math.degrees(math.atan2(vx, -vz))
    assert abs(heading - 30.0) < 1.0, heading
    assert abs(ps.speed - config.WALK_SPEED) < 0.05


def test_dpad_is_digital_movement():
    ps = _open_floor()
    ps.press("dpup")
    ps.run(1.5)
    assert abs(ps.speed - config.WALK_SPEED) < 0.05
    ps.release("dpup")
    ps.run(1.5)
    assert ps.speed < 0.05


def test_partial_tilt_jump_pushes_a_shorter_leap_and_air_steering_scales():
    dists = []
    for tilt in (0.4, 1.0):
        ps = _open_floor()
        ps.move(0.0, tilt)
        ps.step()
        ps.sim.vel[:] = 0.0           # standing jump: only the leg push moves the body
        ps.press("a")
        ps.step()
        z0 = float(ps.sim.pos[2])
        ps.run(2.0, until=lambda s: s.player.on_ground and s.time > 0.2)
        dists.append(z0 - float(ps.sim.pos[2]))
    assert 0.3 < dists[0] < dists[1], dists


def test_keyboard_and_stick_hold_the_same_action_together():
    ps = _open_floor()
    ps.router.press(("key", "w"), Action.FORWARD)
    ps.move(0.0, 1.0)
    ps.run(0.5)
    ps.move(0.0, 0.0)          # the stick lets go, the key is still down
    ps.run(0.8)
    assert abs(ps.speed - config.WALK_SPEED) < 0.05, "W still walks after the stick is released"
    assert ps.sim.input.is_down(Action.FORWARD)
    ps.router.release(("key", "w"), Action.FORWARD)
    assert not ps.sim.input.is_down(Action.FORWARD)


def test_keyboard_movement_is_unchanged_by_the_analog_path():
    ls, obs = build_custom([])
    sim = Sim(ls, obs, spawn=(0.0, 0.9, 0.0), scheme="flow")
    sim.hold("W", "D")
    sim.run(1.5)
    assert abs(np.hypot(sim.vel[0], sim.vel[2]) - config.WALK_SPEED) < 0.05
    assert sim.player.intents.move_analog is None and sim.player.intents.move_amount == 1.0


# ----------------------------------------------------------------------- look
def test_right_stick_turns_and_tilts_the_camera():
    ps = _open_floor()
    yaw0 = ps.sim.player.yaw
    ps.look(1.0, 0.0)
    ps.run(0.5)
    turned = yaw0 - ps.sim.player.yaw
    assert 110.0 < turned < 180.0, f"full right for 0.5 s turns right quickly ({turned:.1f} deg)"
    ps.look(0.0, 0.0)
    yaw1 = ps.sim.player.yaw
    ps.run(0.3)
    assert ps.sim.player.yaw == yaw1, "a centred stick leaves the camera alone"
    ps.look(0.0, 1.0)
    ps.run(2.0)
    assert ps.sim.player.pitch == config.PITCH_MAX, "looking up is clamped"
    ps.bindings.invert_y = True
    ps.run(0.5)
    assert ps.sim.player.pitch < config.PITCH_MAX - 30.0, "inverted: stick up looks down"


def test_look_sensitivity_scales_the_turn():
    turns = []
    for sens in (0.5, 1.0, 2.0):
        ps = _open_floor()
        ps.bindings.look_sensitivity = sens
        ps.look(0.5, 0.0)
        ps.run(0.5)
        turns.append(-ps.sim.player.yaw)
    assert abs(turns[0] * 2.0 - turns[1]) < 1e-3 and abs(turns[1] * 2.0 - turns[2]) < 1e-3


def test_swapped_sticks_move_with_the_right_and_look_with_the_left():
    ps = _open_floor()
    ps.bindings.swap_sticks = True
    ps.stick("right", 0.0, 1.0)
    ps.run(1.0)
    assert abs(ps.speed - config.WALK_SPEED) < 0.05 and ps.sim.player.yaw == 0.0, "right stick moves"
    ps.stick("right", 0.0, 0.0)
    ps.stick("left", 1.0, 0.0)
    ps.run(0.5)
    assert ps.sim.player.yaw < -100.0, "left stick looks"


# ------------------------------------------------------------ climbing with a stick
def test_stick_up_climbs_and_tops_out():
    ls, obs, tower = _tower()
    ps = _pad_hang(ls, obs)
    ps.move(0.0, 1.0)
    assert ps.run(3.0, until=lambda s: any(e == "CLIMB_UP_DONE" for _, e in s.events))
    assert abs(ps.sim.fsm.active_rail.height - 3.8) < 1e-3
    assert ps.run(6.0, until=lambda s: s.standing_on(tower)), "holding the stick up climbs all the way over"


def test_a_slightly_off_axis_push_up_climbs_and_does_not_shimmy():
    ls, obs, tower = _tower()
    ps = _pad_hang(ls, obs)
    a = math.radians(25.0)
    ps.move(math.sin(a), math.cos(a))
    ps.run(0.6)
    assert ParkourState.CLIMB_UP in ps.sim.states and ParkourState.SHIMMY not in ps.sim.states


def test_stick_sideways_shimmies_even_a_little_off_axis():
    ls, obs, tower = _tower()
    ps = _pad_hang(ls, obs)
    x0 = float(ps.sim.pos[0])
    a = math.radians(20.0)              # pushing right, 20 degrees up
    ps.move(math.cos(a), math.sin(a))
    ps.run(0.5)
    assert ps.sim.state == ParkourState.SHIMMY and float(ps.sim.pos[0]) - x0 > 0.4
    assert ParkourState.CLIMB_UP not in ps.sim.states


def test_stick_flick_down_climbs_down_or_drops():
    ls, obs, tower = _tower()
    ps = _pad_hang(ls, obs)
    ps.move(0.0, 1.0)
    assert ps.run(3.0, until=lambda s: any(e == "CLIMB_UP_DONE" for _, e in s.events))
    ps.move(0.0, 0.0)
    ps.run(0.4)
    ps.move(0.0, -1.0)
    ps.step()
    ps.move(0.0, 0.0)
    assert ps.sim.events[-1][1] == "CLIMB_DOWN"
    ps.run(2.0)
    assert abs(ps.sim.fsm.active_rail.height - 2.0) < 1e-3


def test_a_with_the_stick_dynos_and_ejects():
    ls, obs, tower = _tower()
    ps = _pad_hang(ls, obs)
    ps.tap("a")
    assert ps.sim.events[-1][1] == "DYNO_UP"

    ps = _pad_hang(ls, obs)
    ps.move(1.0, 0.0)
    ps.tap("a")
    assert ps.sim.events[-1][1] == "DYNO_LATERAL" and float(ps.sim.vel[0]) > 0

    ps = _pad_hang(ls, obs)
    ps.move(-1.0, 0.0)
    ps.tap("a")
    assert ps.sim.events[-1][1] == "DYNO_LATERAL" and float(ps.sim.vel[0]) < 0

    ps = _pad_hang(ls, obs)
    ps.move(0.0, -1.0)               # stick down and A together: the backward cat leap
    ps.tap("a")
    assert ps.sim.events[-1][1] == "WALL_EJECT"


def test_full_route_on_the_controller_kick_catch_climb_over():
    """Sprint lock, run at the wall, jump with A, reach with RT through the kick and the catch,
    then climb straight over the top with the stick held up."""
    ls, obs, tower = _tower()
    ps = PadSim(Sim(ls, obs, spawn=(0.0, 0.9, 14.0), scheme="flow"))
    ps.tap("leftstick")
    ps.move(0.0, 1.0)
    ps.run(5.0, until=lambda s: s.pos[2] < -1.0 + 0.3 + 1.8)
    ps.press("a")                      # held, like Space: full leap, reach, kick, catch
    assert ps.run(8.0, until=lambda s: s.standing_on(tower))
    evs = [e for _, e in ps.sim.events]
    for e in ("JUMP", "WALL_KICK", "GRAB", "CLIMB_UP_FAST", "MANTLE_FAST", "MANTLE_DONE"):
        assert e in evs, evs


def test_tap_a_then_reach_with_the_right_trigger_also_gets_up():
    ls, obs, tower = _tower()
    ps = PadSim(Sim(ls, obs, spawn=(0.0, 0.9, 14.0), scheme="flow"))
    ps.tap("leftstick")
    ps.move(0.0, 1.0)
    ps.run(5.0, until=lambda s: s.pos[2] < -1.0 + 0.3 + 1.8)
    ps.tap("a")
    ps.trigger("righttrigger", 1.0)
    assert ps.run(8.0, until=lambda s: s.standing_on(tower))
    assert "GRAB" in [e for _, e in ps.sim.events]


# ------------------------------------------------------------------ rebinding
def test_rebinding_changes_what_a_button_does():
    ls, obs, tower = _tower()
    ps = _pad_hang(ls, obs)
    ps.bindings.bind("y", "c")              # Y -> C (let go)
    ps.tap("y")
    assert ps.sim.events[-1][1] == "DROP"
    ps.run(1.5)
    ps.bindings.clear("a")
    n = len(ps.sim.events)
    ps.tap("a")
    ps.run(0.3)
    assert len(ps.sim.events) == n, "an unbound button does nothing"
    ps.bindings.bind("rightshoulder", "escape")
    ps.press("rightshoulder")
    assert ps.results[-1] == PAUSE, "a button bound to Esc opens the pause menu"
    ps.bindings.bind("leftshoulder", "f7")
    ps.press("leftshoulder")
    assert ps.results[-1] is None, "a key with no game action is harmless"


def test_releasing_a_rebound_button_releases_what_it_pressed():
    ps = _open_floor()
    ps.press("a")
    assert ps.sim.input.is_down(Action.JUMP)
    ps.bindings.bind("a", "e")
    ps.release("a")
    assert not ps.sim.input.is_down(Action.JUMP), "no stuck action after a rebind"


def test_start_pauses_and_cannot_be_rebound():
    ps = _open_floor()
    ps.press("start")
    assert ps.results[-1] == PAUSE
    ps.release("start")
    assert ps.results[-1] is None


# ---------------------------------------------------------------- menus / pausing
def test_buttons_held_through_the_menu_are_ignored_until_released():
    ps = _open_floor()
    ps.press("a")                       # A confirmed "Resume" and is still down
    # what set_paused does on the way in and out of the menu
    ps.router.release_all()
    ps.pad.release_all()
    ps.sim.input.clear()
    ps.pad.suppress_held(ps.state)
    ps.step(3)
    assert not ps.sim.input.is_down(Action.JUMP) and not ps.sim.had_event("JUMP"), "no jump on resume"
    ps.release("a")
    ps.tap("a")
    assert ps.sim.had_event("JUMP"), "the next press works"


def test_pad_menu_navigation_steps_once_per_press_and_repeats_when_held():
    pi = GamepadInput(ControllerBindings(), InputRouter(InputBuffer()))
    st = PadState()
    st.set_button("dpdown", True)
    assert pi.menu_event("dpdown", True) == "down"
    assert pi.menu_update(st, 0.016) == [], "the press already stepped: no double step"
    steps = []
    for _ in range(60):
        steps += pi.menu_update(st, 0.016)
    assert len(steps) >= 3 and set(steps) == {"down"}, "holding repeats"
    st.set_button("dpdown", False)
    assert pi.menu_update(st, 0.016) == []
    st.set_axis("lefty", -0.9)
    assert pi.menu_update(st, 0.016) == ["up"], "the left stick navigates too"
    st.set_axis("lefty", 0.3)
    assert pi.menu_update(st, 0.016) == []
    assert pi.menu_event("a", True) == "confirm"
    assert pi.menu_event("b", True) == "back"
    assert pi.menu_event("start", True) == "back"
    assert pi.menu_event("a", False) is None and pi.menu_event("x", True) is None


def test_no_controller_releases_the_stick():
    ps = _open_floor()
    ps.move(0.0, 1.0)
    ps.run(0.5)
    assert ps.sim.input.move_axis is not None
    ps.pad.update(None, DT)             # pad unplugged
    ps.sim.step(30)
    assert ps.sim.input.move_axis is None and not ps.sim.input.is_down(Action.FORWARD)
    assert ps.speed < 0.5


# -------------------------------------------------------------------- feedback
def test_haptics_buzz_on_catches_kicks_and_hard_landings_only():
    h = Haptics()
    assert h.on_tick("GRAB", False, False, 0.0) is not None
    assert h.on_tick("WALL_KICK", False, False, 2.0) is not None
    assert h.on_tick("JUMP", True, False, 0.0) is None, "jumping is not buzzed (it would be constant)"
    assert h.on_tick(None, True, True, 0.0) is None
    assert h.on_tick(None, False, True, -3.0) is None, "a hop's landing is not felt"
    soft = h.on_tick(None, False, True, -7.0)
    hard = h.on_tick(None, False, True, -14.0)
    assert soft is not None and hard is not None and hard[0] > soft[0] and hard[2] > soft[2]
    for low, high, ms in list(Haptics.EVENTS.values()) + [hard]:
        assert 0.0 <= low <= 1.0 and 0.0 <= high <= 1.0 and 0 < ms <= 250


def test_toasts_replace_by_key_fade_and_expire():
    q = ToastQueue(duration=1.0)
    q.push("Sprint lock ON", key="run")
    q.push("Controller connected", key="pad")
    q.push("Sprint lock OFF", key="run")
    texts = [t for t, _ in q.visible()]
    assert texts == ["Controller connected", "Sprint lock OFF"], "a toggle replaces its previous notice"
    assert all(a == 0.0 for _, a in q.visible()), "fades in"
    q.update(0.5)
    assert all(a == 1.0 for _, a in q.visible())
    q.update(0.4)
    assert all(0.0 < a < 1.0 for _, a in q.visible()), "fades out"
    q.update(0.2)
    assert q.visible() == []
