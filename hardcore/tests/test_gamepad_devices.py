"""Real SDL device path with virtual controllers: Xbox, PlayStation, Nintendo and generic pads
through SDL's GameController API, unrecognised devices through the raw-joystick fallback,
hot-plugging, several pads at once, and a pad driving the game end to end."""
import numpy as np

from pad_helpers import *  # noqa


def _env():
    pygame = headless_pygame()
    import sdl_virtual as sv
    if not sv.available():
        skip("SDL virtual joysticks unavailable (needs SDL >= 2.24)")
    from core.gamepad_device import GamepadManager, PadEvent, PadDevice
    return pygame, sv, GamepadManager, PadEvent, PadDevice


class _Pads:
    """Context manager: virtual pads + a manager, always cleaned up."""

    def __init__(self, **manager_kw):
        self.pygame, self.sv, GM, self.PadEvent, self.PadDevice = _env()
        self.pygame.event.get()
        self.pads = []
        self.manager = GM(**manager_kw)

    def add(self, spec, **kw):
        pad = self.sv.VirtualPad(spec, **kw)
        self.pads.append(pad)
        return pad

    def pump(self):
        return self.sv.pump(self.pygame, self.manager)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        for p in self.pads:
            p.detach()
        self.pump()
        self.manager.close()
        self.pygame.event.get()
        return False


def _buttons(events, PadEvent):
    return [(e.control, e.down) for e in events if e.kind == PadEvent.BUTTON]


def test_known_pads_open_as_gamepads_with_the_right_family():
    from core.gamepad import Family
    with _Pads() as env:
        sv = env.sv
        expected = {sv.XBOX: Family.XBOX, sv.DUALSENSE: Family.PLAYSTATION, sv.DUALSHOCK4: Family.PLAYSTATION,
                    sv.SWITCH_PRO: Family.NINTENDO, sv.GENERIC: Family.GENERIC}
        for spec in expected:
            env.add(spec)
        added = [e for e in env.pump() if e.kind == env.PadEvent.ADDED]
        assert len(added) == len(expected), "one ADDED per pad (CONTROLLER/JOY duplicates merged)"
        by_name = {d.name: d for d in env.manager.devices.values()}
        for spec, family in expected.items():
            dev = by_name[spec[0]]
            assert dev.kind == env.PadDevice.GAMEPAD and dev.family == family, (spec, dev.family)
        assert env.manager.active is not None


def test_buttons_sticks_and_triggers_decode_from_sdl_events():
    with _Pads() as env:
        sv, PadEvent = env.sv, env.PadEvent
        pad = env.add(sv.XBOX)
        env.pump()
        dev = env.manager.active
        pad.button(sv.BTN["a"], True)
        pad.button(sv.BTN["leftstick"], True)
        assert _buttons(env.pump(), PadEvent) == [("a", True), ("leftstick", True)]
        assert dev.state.is_down("a")
        pad.button(sv.BTN["a"], False)
        assert _buttons(env.pump(), PadEvent) == [("a", False)]
        pad.axis(sv.AX["leftx"], 1.0)
        pad.axis(sv.AX["lefty"], -0.5)
        env.pump()
        assert abs(dev.state.axes["leftx"] - 1.0) < 1e-3 and abs(dev.state.axes["lefty"] + 0.5) < 1e-3
        pad.trigger(sv.AX["lefttrigger"], 0.2)
        assert _buttons(env.pump(), PadEvent) == [], "a light rest on the trigger is not a press"
        assert 0.15 < dev.state.axes["lefttrigger"] < 0.25
        pad.trigger(sv.AX["lefttrigger"], 0.8)
        assert _buttons(env.pump(), PadEvent) == [("lefttrigger", True)]
        pad.trigger(sv.AX["lefttrigger"], 0.0)
        assert _buttons(env.pump(), PadEvent) == [("lefttrigger", False)]
        for name in ("dpup", "dpleft", "start", "back", "rightshoulder", "y"):
            pad.button(sv.BTN[name], True)
            assert _buttons(env.pump(), PadEvent) == [(name, True)], name
            pad.button(sv.BTN[name], False)
            env.pump()


def test_unplugging_releases_held_buttons_and_reports_the_active_pad():
    with _Pads() as env:
        sv, PadEvent = env.sv, env.PadEvent
        pad = env.add(sv.DUALSENSE)
        env.pump()
        pad.button(sv.BTN["x"], True)
        pad.trigger(sv.AX["righttrigger"], 1.0)
        env.pump()
        pad.detach()
        events = env.pump()
        kinds = [e.kind for e in events]
        assert ("x", False) in _buttons(events, PadEvent) and ("righttrigger", False) in _buttons(events, PadEvent)
        removed = [e for e in events if e.kind == PadEvent.REMOVED]
        assert len(removed) == 1 and removed[0].was_active
        assert kinds.index(PadEvent.REMOVED) > kinds.index(PadEvent.BUTTON), "releases come first"
        assert env.manager.active is None and not env.manager.devices


def test_events_follow_instance_ids_after_another_pad_leaves():
    """SDL's device index shifts when an earlier pad is unplugged; events carry instance ids."""
    with _Pads() as env:
        sv, PadEvent = env.sv, env.PadEvent
        first = env.add(sv.XBOX)
        second = env.add(sv.DUALSENSE)
        env.pump()
        first.detach()
        env.pump()
        assert len(env.manager.devices) == 1
        second.button(sv.BTN["b"], True)
        events = env.pump()
        assert _buttons(events, PadEvent) == [("b", True)]
        assert env.manager.active.name == sv.DUALSENSE[0]


def test_the_last_pad_used_becomes_the_active_one():
    with _Pads() as env:
        sv, PadEvent = env.sv, env.PadEvent
        one = env.add(sv.XBOX)
        two = env.add(sv.GENERIC)
        env.pump()
        assert env.manager.active.name == sv.XBOX[0], "the first pad is active until another is used"
        two.axis(sv.AX["lefty"], 0.2)
        env.pump()
        assert env.manager.active.name == sv.XBOX[0], "a drifting stick does not steal focus"
        two.button(sv.BTN["a"], True)
        events = env.pump()
        assert events[0].kind == PadEvent.ACTIVATED and env.manager.active.name == sv.GENERIC[0]
        two.button(sv.BTN["a"], False)
        env.pump()
        one.trigger(sv.AX["righttrigger"], 1.0)
        env.pump()
        assert env.manager.active.name == sv.XBOX[0], "pulling a trigger counts as using the pad"
        two.detach()
        events = env.pump()
        removed = [e for e in events if e.kind == PadEvent.REMOVED]
        assert removed and not removed[0].was_active and env.manager.active.name == sv.XBOX[0]


def test_unrecognised_device_uses_the_raw_joystick_fallback():
    with _Pads() as env:
        sv, PadEvent = env.sv, env.PadEvent
        stick = env.add(("Cheap USB Gamepad", 0x1234, 0x5678), kind="joystick", naxes=4, nbuttons=12, nhats=1)
        env.pump()
        dev = env.manager.active
        assert dev is not None and dev.kind == env.PadDevice.JOYSTICK and "unrecognised" in dev.describe()
        stick.button(0, True)
        stick.button(6, True)                       # separate L2 button on PS-style pads
        assert _buttons(env.pump(), PadEvent) == [("a", True), ("lefttrigger", True)]
        stick.hat(0, sv.HAT_UP | sv.HAT_LEFT)
        assert sorted(_buttons(env.pump(), PadEvent)) == [("dpleft", True), ("dpup", True)]
        stick.hat(0, 0)
        assert sorted(_buttons(env.pump(), PadEvent)) == [("dpleft", False), ("dpup", False)]
        stick.axis(2, 1.0)                          # right stick x on a 4-axis pad
        stick.axis(1, -1.0)
        env.pump()
        assert dev.state.axes["rightx"] > 0.99 and dev.state.axes["lefty"] < -0.99


def test_raw_fallback_when_the_gamecontroller_api_is_missing():
    """Old / stripped pygame builds without pygame._sdl2.controller still get a working pad."""
    with _Pads(use_gamecontroller=False) as env:
        sv, PadEvent = env.sv, env.PadEvent
        pad = env.add(sv.XBOX)
        env.pump()
        dev = env.manager.active
        assert dev.kind == env.PadDevice.JOYSTICK and dev.family == "xbox"
        pad.button(0, True)
        assert ("a", True) in _buttons(env.pump(), PadEvent)
        pad.trigger(4, 1.0)                          # 6-axis raw pad: axis 2 is a trigger, axis 4 right stick y
        env.pump()


def test_rumble_reaches_the_pad_with_the_requested_strength():
    with _Pads() as env:
        pad = env.add(env.sv.DUALSENSE, record_rumble=True)
        env.pump()
        assert env.manager.rumble(0.25, 0.55, 90) is True
        assert pad.rumbles and abs(pad.rumbles[-1][0] - 0.25) < 0.01 and abs(pad.rumbles[-1][1] - 0.55) < 0.01


def test_rumble_never_crashes_even_when_unsupported():
    with _Pads() as env:
        env.add(env.sv.XBOX)
        env.pump()
        assert env.manager.rumble(0.5, 0.5, 100) in (True, False)
    with _Pads() as env:
        assert env.manager.rumble(0.5, 0.5, 100) is False, "no pad: nothing to rumble"


def test_virtual_pad_plays_the_game_end_to_end():
    """SDL events -> GamepadManager -> GamepadInput -> InputBuffer -> PlayerController."""
    from harness import Sim, build_custom, DT, ParkourState
    from core.gamepad import ControllerBindings
    from core.gamepad_input import GamepadInput
    from core.input_router import InputRouter
    import config
    with _Pads() as env:
        sv, PadEvent = env.sv, env.PadEvent
        ls, obs = build_custom([])
        sim = Sim(ls, obs, spawn=(0.0, 0.9, 0.0), scheme="flow")
        router = InputRouter(sim.input, lambda: sim.player.scheme)
        pad_input = GamepadInput(ControllerBindings(), router)

        def frame(n=1):
            for _ in range(n):
                for e in env.pump():
                    if e.kind == PadEvent.BUTTON and e.device is env.manager.active:
                        pad_input.on_button(e.control, e.down)
                turn, up = pad_input.update(env.manager.active_state, DT)
                sim.player.apply_look(turn, up)
                sim.step()

        pad = env.add(sv.DUALSENSE)
        frame()
        pad.axis(sv.AX["lefty"], -1.0)              # stick forward
        frame(90)
        assert abs(np.hypot(sim.vel[0], sim.vel[2]) - config.WALK_SPEED) < 0.05
        pad.button(sv.BTN["a"], True)                # Cross
        frame(2)
        assert sim.had_event("JUMP") and sim.state == ParkourState.AIRBORNE
        pad.button(sv.BTN["a"], False)
        pad.axis(sv.AX["lefty"], 0.0)
        pad.axis(sv.AX["rightx"], 1.0)
        frame(30)
        assert sim.player.yaw < -100.0, "the right stick turned the camera"
        pad.axis(sv.AX["rightx"], 0.0)
        pad.detach()
        frame(60)
        assert sim.input.move_axis is None and not sim.input.held_keys, "unplugged: nothing stays held"


def test_a_pad_plugged_in_before_the_game_starts():
    """Opened by the start-up scan; SDL's DEVICEADDED events for it must not open it twice."""
    pygame, sv, GM, PadEvent, PadDevice = _env()
    pygame.event.get()
    pad = sv.VirtualPad(sv.XBOX)
    manager = GM()
    try:
        assert len(manager.devices) == 1 and manager.active is not None, "found by the scan"
        events = sv.pump(pygame, manager)
        assert not [e for e in events if e.kind == PadEvent.ADDED], "no duplicate ADDED"
        assert len(manager.devices) == 1
        for _ in range(3):                    # still alive after the duplicate handles went away
            pad.button(sv.BTN["a"], True)
            assert _buttons(sv.pump(pygame, manager), PadEvent) == [("a", True)]
            pad.button(sv.BTN["a"], False)
            assert _buttons(sv.pump(pygame, manager), PadEvent) == [("a", False)]
    finally:
        pad.detach()
        sv.pump(pygame, manager)
        manager.close()
        pygame.event.get()
