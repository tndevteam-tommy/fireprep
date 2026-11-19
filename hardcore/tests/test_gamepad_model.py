"""Gamepad model: defaults, family detection, signal processing and persistence (pure Python)."""
import json
import math
import os
import tempfile

from pad_helpers import *  # noqa
import config
from core.gamepad import (ControllerBindings, PadState, StickDirections, LookProcessor, MenuRepeater, Family,
                          detect_family, parse_guid, raw_layout, radial_deadzone, control_label,
                          controls_for_family, BINDABLE_CONTROLS, DEFAULT_BINDINGS, PAUSE_CONTROL,
                          default_settings_path)
from core.keymap import action_for_key, key_display_name, key_action_label, KEYBOARD_ACTIONS
from core.input_buffer import Action


# ---------------------------------------------------------------- defaults
def test_default_bindings_match_the_requested_layout():
    b = ControllerBindings()
    assert action_for_key(b.key_for("a")) == Action.JUMP, "A = jump"
    assert action_for_key(b.key_for("x")) == Action.GRAB, "X = grab"
    assert action_for_key(b.key_for("lefttrigger")) == Action.SPRINT, "LT = sprint while held (no lock)"
    assert action_for_key(b.key_for("leftstick")) == Action.RUN_TOGGLE, "left stick click = sprint lock"
    assert action_for_key(b.key_for("b")) == Action.LET_GO
    assert b.move_stick == "left" and b.look_stick == "right", "standard: left stick moves, right stick looks"
    for control, action in (("dpup", Action.FORWARD), ("dpdown", Action.BACK),
                            ("dpleft", Action.LEFT), ("dpright", Action.RIGHT)):
        assert action_for_key(b.key_for(control)) == action
    assert PAUSE_CONTROL == "start" and PAUSE_CONTROL not in BINDABLE_CONTROLS, "Start always pauses"


def test_every_default_binding_points_at_a_real_game_key():
    for control, key in DEFAULT_BINDINGS.items():
        assert control in BINDABLE_CONTROLS
        assert key in KEYBOARD_ACTIONS, f"{control} -> {key} does nothing"


def test_key_names_and_action_labels_are_human_readable():
    assert key_display_name("left shift") == "Left Shift"
    assert key_display_name("space") == "Space"
    assert key_display_name("e") == "E"
    assert key_display_name("caps lock") == "Caps Lock"
    assert key_display_name(None) == "Unbound"
    assert key_display_name("f5") == "F5"
    assert key_action_label("caps lock") == "Sprint lock (toggle)"
    assert key_action_label("escape") == "Pause menu"
    assert key_action_label("q") == "No action in Hardcore"


# -------------------------------------------------------------- families
def test_family_detection_by_name():
    cases = {
        "Xbox Series X Controller": Family.XBOX,
        "Xbox 360 Controller": Family.XBOX,
        "XInput Controller": Family.XBOX,
        "Microsoft X-Box One pad": Family.XBOX,
        "PS4 Controller": Family.PLAYSTATION,
        "PS5 Controller": Family.PLAYSTATION,
        "DualSense Wireless Controller": Family.PLAYSTATION,
        "Sony Interactive Entertainment Wireless Controller": Family.PLAYSTATION,
        "PS3 Controller": Family.PLAYSTATION,
        "Nintendo Switch Pro Controller": Family.NINTENDO,
        "Joy-Con (L/R)": Family.NINTENDO,
        "8BitDo SN30 Pro": Family.GENERIC,
        "USB Gamepad": Family.GENERIC,
        "Logitech Dual Action": Family.GENERIC,
        "": Family.GENERIC,
        None: Family.GENERIC,
    }
    for name, family in cases.items():
        assert detect_family(name) == family, name


def test_family_detection_falls_back_to_the_usb_vendor():
    # SDL GUIDs: bus(2) crc(2) vendor(2) 0(2) product(2) 0(2) version(2) driver(2), little endian
    sony = "030000004c050000e60c000000016800"
    ms = "030000005e040000120b000000000000"
    nin = "030000007e0500000920000000000000"
    assert parse_guid(sony) == (0x054C, 0x0CE6)
    assert detect_family("Wireless Controller", sony) == Family.PLAYSTATION
    assert detect_family("Controller", ms) == Family.XBOX
    assert detect_family("Pro Controller", nin) == Family.NINTENDO
    assert detect_family("Gamepad", "0300000079000000110000000000") == Family.GENERIC
    # Windows XInput-style GUIDs carry text, not a vendor id
    assert parse_guid("78696e70757401000000000000000000") == (None, None)
    assert parse_guid("nonsense") == (None, None)
    assert parse_guid("") == (None, None)


def test_labels_follow_the_family_and_the_button_position():
    assert control_label("a", Family.XBOX) == "A"
    assert control_label("a", Family.PLAYSTATION) == "Cross"
    assert control_label("y", Family.PLAYSTATION) == "Triangle"
    assert control_label("lefttrigger", Family.PLAYSTATION) == "L2"
    assert control_label("leftstick", Family.PLAYSTATION) == "L3"
    # positional: the bottom button of a Nintendo pad is labelled B, the right one A
    assert control_label("a", Family.NINTENDO) == "B" and control_label("b", Family.NINTENDO) == "A"
    assert control_label("lefttrigger", Family.NINTENDO) == "ZL"
    assert control_label("start", Family.XBOX) == "Menu"
    assert control_label("back", Family.PLAYSTATION, long=True) == "Create / Share"
    assert "touchpad" in controls_for_family(Family.PLAYSTATION)
    assert "touchpad" not in controls_for_family(Family.XBOX)


def test_pause_menu_hint_uses_the_pads_own_button_names():
    from core.gamepad import pad_menu_hint
    assert "A to confirm" in pad_menu_hint(Family.XBOX) and "Menu to resume" in pad_menu_hint(Family.XBOX)
    ps = pad_menu_hint(Family.PLAYSTATION)
    assert "Cross to confirm" in ps and "Circle or Options" in ps
    assert "B to confirm" in pad_menu_hint(Family.NINTENDO), "Nintendo's bottom button is B"


def test_raw_layout_guesses_both_common_orders():
    ps_style = raw_layout(12, 4, 1)
    assert ps_style["buttons"][6] == "lefttrigger" and ps_style["buttons"][9] == "start"
    assert ps_style["axes"] == {0: "leftx", 1: "lefty", 2: "rightx", 3: "righty"}
    assert ps_style["hat"] == 0
    xinput_style = raw_layout(11, 6, 1)
    assert xinput_style["buttons"][7] == "start" and xinput_style["buttons"][8] == "leftstick"
    assert xinput_style["axes"][2] == "lefttrigger" and xinput_style["axes"][5] == "righttrigger"
    tiny = raw_layout(4, 2, 0)
    assert set(tiny["buttons"]) == {0, 1, 2, 3} and tiny["hat"] is None and tiny["axes"] == {0: "leftx", 1: "lefty"}


# ------------------------------------------------------- signal processing
def test_radial_deadzone_rescales_without_a_jump():
    dz = 0.15
    assert radial_deadzone(0.1, 0.1, dz) == (0.0, 0.0, 0.0), "drift inside the deadzone is ignored"
    x, y, m = radial_deadzone(0.16, 0.0, dz)
    assert 0.0 < m < 0.02, "just past the deadzone the output starts from ~0 (no jump)"
    x, y, m = radial_deadzone(1.0, 0.0, dz)
    assert m == 1.0 and x == 1.0
    x, y, m = radial_deadzone(config.PAD_STICK_OUTER, 0.0, dz)
    assert abs(m - 1.0) < 1e-6, "worn sticks that stop short of 1.0 still reach full deflection"
    # square-gated pads report ~1.41 on the diagonal: clamped to the unit circle, direction kept
    x, y, m = radial_deadzone(1.0, 1.0, dz)
    assert m == 1.0 and abs(x - y) < 1e-9 and abs(math.hypot(x, y) - 1.0) < 1e-9
    # monotonic
    last = -1.0
    for i in range(101):
        _, _, m = radial_deadzone(i / 100.0, 0.0, dz)
        assert m >= last
        last = m


def test_stick_directions_cardinal_bands_diagonals_and_hysteresis():
    d = StickDirections()
    assert d.update(0.0, 0.3, 0.3) == set(), "a light push is not a direction"
    assert d.update(0.0, 0.6, 0.6) == {"forward"}
    assert d.update(0.0, 0.4, 0.4) == {"forward"}, "hysteresis: stays held above the release threshold"
    assert d.update(0.0, 0.3, 0.3) == set()
    # pushing up but 25 degrees off to the side is still just "up" (climbing never shimmies)
    a = math.radians(90 - 25)
    assert d.update(math.cos(a), math.sin(a), 1.0) == {"forward"}
    # a true diagonal holds both, like W+D
    a = math.radians(45)
    assert d.update(math.cos(a), math.sin(a), 1.0) == {"forward", "right"}
    d.reset()
    a = math.radians(180 + 10)
    assert d.update(math.cos(a), math.sin(a), 1.0) == {"left"}
    d.reset()
    # angular hysteresis: a held direction survives a few degrees past the press angle
    a = math.radians(90 - 58)
    assert "forward" in d.update(math.cos(a), math.sin(a), 1.0)
    a = math.radians(90 - 63)
    assert "forward" in d.update(math.cos(a), math.sin(a), 1.0)
    a = math.radians(90 - 70)
    assert "forward" not in d.update(math.cos(a), math.sin(a), 1.0)


def test_triggers_have_hysteresis_and_report_transitions():
    st = PadState()
    assert st.set_axis("lefttrigger", 0.1) == []
    assert st.set_axis("lefttrigger", config.PAD_TRIGGER_PRESS) == [("lefttrigger", True)]
    assert st.is_down("lefttrigger")
    assert st.set_axis("lefttrigger", 0.22) == [], "still held between the thresholds"
    assert st.set_axis("lefttrigger", 0.1) == [("lefttrigger", False)]
    assert st.set_axis("lefttrigger", 5.0) == [("lefttrigger", True)] and st.axes["lefttrigger"] == 1.0
    assert st.set_axis("leftx", -3.0) == [] and st.axes["leftx"] == -1.0
    assert st.set_button("a", True) == [("a", True)] and st.set_button("a", True) == []
    assert sorted(st.clear()) == [("a", False), ("lefttrigger", False)]
    assert not st.down and all(v == 0.0 for v in st.axes.values())


def test_look_curve_sensitivity_invert_and_rim_boost():
    lp = LookProcessor()
    dt = 1.0 / 60.0
    assert lp.update(0.1, 0.0, dt) == (0.0, 0.0), "deadzone"
    half, _ = lp.update(0.5, 0.0, dt)
    lp.reset()
    full, _ = lp.update(1.0, 0.0, dt)
    assert 0.0 < half < full * 0.35, "response curve: half tilt turns much slower than full (fine aim)"
    assert abs(full - config.PAD_LOOK_YAW_RATE * dt) < 1e-6, "full tilt = the base yaw rate"
    lp.reset()
    fast, _ = lp.update(1.0, 0.0, dt, sensitivity=2.0)
    assert abs(fast - 2.0 * full) < 1e-9
    lp.reset()
    _, up = lp.update(0.0, -1.0, dt)            # SDL: stick pushed up = negative y
    assert up > 0.0, "stick up looks up"
    lp.reset()
    _, up_inv = lp.update(0.0, -1.0, dt, invert_y=True)
    assert up_inv == -up
    # holding the stick at the rim ramps up the turn speed (quick 180s)
    lp.reset()
    rates = [lp.update(1.0, 0.0, dt)[0] / dt for _ in range(90)]
    assert abs(rates[0] - config.PAD_LOOK_YAW_RATE) < 1e-6
    assert abs(rates[-1] - config.PAD_LOOK_YAW_RATE * (1.0 + config.PAD_LOOK_BOOST)) < 1e-6
    assert all(b >= a - 1e-9 for a, b in zip(rates, rates[1:])), "the boost ramps in smoothly"
    lp.update(0.6, 0.0, dt)
    assert abs(lp.update(1.0, 0.0, dt)[0] / dt - config.PAD_LOOK_YAW_RATE) < 1e-6, "easing off resets the boost"


def test_look_turn_is_frame_rate_independent():
    def turn_in_one_second(fps):
        lp = LookProcessor()
        dt = 1.0 / fps
        return sum(lp.update(0.7, 0.0, dt)[0] for _ in range(fps))
    a, b, c = turn_in_one_second(30), turn_in_one_second(60), turn_in_one_second(144)
    assert abs(a - b) < 1e-6 and abs(b - c) < 1e-6


def test_menu_repeater_steps_once_then_repeats():
    r = MenuRepeater(delay=0.4, rate=0.1)
    assert r.update("down", 0.016) == ["down"]
    steps = []
    for _ in range(int(0.38 / 0.016)):
        steps += r.update("down", 0.016)
    assert steps == [], "no repeat before the delay"
    for _ in range(int(0.32 / 0.016)):
        steps += r.update("down", 0.016)
    assert 2 <= len(steps) <= 4
    assert r.update(None, 0.016) == []
    assert r.update("up", 0.016) == ["up"]


# ------------------------------------------------------------ persistence
def test_bindings_round_trip_and_reset():
    b = ControllerBindings()
    b.bind("y", "C")
    b.clear("x")
    b.look_sensitivity = 1.7
    b.invert_y = True
    b.swap_sticks = True
    b.deadzone = 0.22
    b.vibration = False
    c = ControllerBindings.from_dict(json.loads(json.dumps(b.to_dict())))
    assert c.key_for("y") == "c" and c.key_for("x") is None
    assert (c.look_sensitivity, c.invert_y, c.swap_sticks, c.deadzone, c.vibration) == (1.7, True, True, 0.22, False)
    assert c.move_stick == "right" and c.look_stick == "left"
    c.reset()
    assert c.to_dict() == ControllerBindings().to_dict()
    try:
        c.bind("start", "space")
        raise AssertionError("start must not be rebindable")
    except ValueError:
        pass


def test_bad_or_hostile_settings_files_fall_back_to_defaults():
    d = ControllerBindings().to_dict()
    junk = {"bindings": {"a": 5, "x": "", "nonsense": "space", "b": None, "y": "q"},
            "look_sensitivity": "fast", "deadzone": 99, "invert_y": "yes", "vibration": 0,
            "swap_sticks": True}
    b = ControllerBindings.from_dict(junk)
    assert b.key_for("a") == d["bindings"]["a"], "non-string key ignored"
    assert b.key_for("x") == d["bindings"]["x"], "empty key ignored"
    assert b.key_for("b") is None and b.key_for("y") == "q"
    assert b.look_sensitivity == 1.0
    assert b.deadzone == config.PAD_DEADZONE_RANGE[1], "out-of-range values are clamped"
    assert b.invert_y is False and b.vibration is True and b.swap_sticks is True
    assert ControllerBindings.from_dict([1, 2]).to_dict() == d
    assert ControllerBindings.from_dict({"look_sensitivity": float("nan")}).look_sensitivity == 1.0
    assert ControllerBindings.from_dict({"look_sensitivity": True}).look_sensitivity == 1.0


def test_save_and_load_files():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "sub", "controller.json")
        assert ControllerBindings.load(path).to_dict() == ControllerBindings().to_dict(), "missing file = defaults"
        b = ControllerBindings()
        b.bind("rightshoulder", "e")
        assert b.save(path)
        assert ControllerBindings.load(path).key_for("rightshoulder") == "e"
        with open(path, "w") as f:
            f.write("{ not json")
        assert ControllerBindings.load(path).to_dict() == ControllerBindings().to_dict(), "corrupt file = defaults"
        # an unwritable location does not crash the game
        blocker = os.path.join(tmp, "file")
        open(blocker, "w").close()
        assert b.save(os.path.join(blocker, "controller.json")) is False


def test_settings_path_override_and_sliders_clamp():
    old = os.environ.get("HARDCORE_CONFIG_DIR")
    os.environ["HARDCORE_CONFIG_DIR"] = "/some/where"
    try:
        assert default_settings_path() == os.path.join("/some/where", "controller.json")
    finally:
        if old is None:
            del os.environ["HARDCORE_CONFIG_DIR"]
        else:
            os.environ["HARDCORE_CONFIG_DIR"] = old
    b = ControllerBindings()
    for _ in range(100):
        b.adjust_sensitivity(1)
        b.adjust_deadzone(-1)
    assert b.look_sensitivity == config.PAD_LOOK_SENSITIVITY_RANGE[1]
    assert b.deadzone == config.PAD_DEADZONE_RANGE[0]
