"""The controller screen: selecting a button waits for a keyboard key and binds it; pressing a
controller button finds it on the picture; settings; layout geometry; and the rendered picture
(live highlights, stick caps, trigger fill) on a headless pygame surface."""
import math

from pad_helpers import *  # noqa
from core.gamepad import ControllerBindings, PadState, Family, controls_for_family, PAUSE_CONTROL, control_label
from core.controller_menu import ControllerMenuModel, ControllerLayout, OPTIONS
from core.menu_model import MenuModel


def _screen(family=Family.XBOX, w=1280, h=720):
    saved = []
    m = ControllerMenuModel(ControllerBindings(), w, h, on_change=lambda b: saved.append(b.to_dict()))
    m.set_device("Test pad", family)
    m.open()
    return m, saved


def _centre(rect):
    x, y, w, h = rect
    return (x + w / 2.0, y + h / 2.0)


# ---------------------------------------------------------------- bind flow
def test_select_a_button_then_press_a_key_binds_it():
    m, saved = _screen()
    m.select(("control", "y"))
    assert m.waiting == "y" and m.status_kind == "prompt"
    assert m.handle_key("c") is None
    assert m.waiting is None and m.bindings.key_for("y") == "c"
    assert "Let go" in m.status and m.status_kind == "ok"
    assert saved and saved[-1]["bindings"]["y"] == "c", "every change is handed to the saver"


def test_any_key_can_be_bound_even_navigation_keys_while_waiting():
    m, _ = _screen()
    for key in ("return", "space", "up", "w", "left shift", "tab", "f5"):
        m.select(("control", "rightshoulder"))
        m.handle_key(key)
        assert m.bindings.key_for("rightshoulder") == key, key


def test_escape_cancels_and_backspace_unbinds():
    m, saved = _screen()
    m.select(("control", "a"))
    assert m.handle_key("escape") is None, "Esc cancels the wait, it does not leave the screen"
    assert m.waiting is None and m.bindings.key_for("a") == "space" and not saved
    m.select(("control", "a"))
    m.handle_key("backspace")
    assert m.bindings.key_for("a") is None and "unbound" in m.status
    # without waiting: Backspace on a focused control unbinds it too
    m.focus = ("control", "x")
    m.handle_key("delete")
    assert m.bindings.key_for("x") is None
    assert m.handle_key("escape") == ControllerMenuModel.BACK, "Esc leaves when nothing is waiting"


def test_binding_a_key_already_used_says_where_else_it_is():
    m, _ = _screen()
    m.select(("control", "y"))
    m.handle_key("space")
    assert "also on A" in m.status


def test_start_cannot_be_rebound():
    m, _ = _screen()
    m.select(("control", PAUSE_CONTROL))
    assert m.waiting is None and m.status_kind == "warn"
    assert m.bindings.key_for(PAUSE_CONTROL) is None


def test_pressing_a_controller_button_finds_it_and_start_goes_back():
    m, _ = _screen()
    assert m.handle_pad("lefttrigger", True) is None
    assert m.focus == ("control", "lefttrigger") and "Left Shift" in m.status
    m.handle_pad("lefttrigger", False)
    assert m.focus == ("control", "lefttrigger"), "releasing keeps the selection"
    m.handle_pad("guide", True)
    assert m.focus == ("control", "lefttrigger"), "the guide button is not in the list"
    assert m.handle_pad(PAUSE_CONTROL, True) == ControllerMenuModel.BACK
    # while waiting for a key, controller buttons do not steal the selection or leave
    m.select(("control", "x"))
    assert m.handle_pad("b", True) is None and m.handle_pad(PAUSE_CONTROL, True) is None
    assert m.waiting == "x"


def test_enter_on_a_focused_control_waits_for_a_key():
    m, _ = _screen()
    m.handle_pad("rightshoulder", True)
    m.handle_key("return")
    assert m.waiting == "rightshoulder"
    m.handle_key("e")
    assert m.bindings.key_for("rightshoulder") == "e"


# ------------------------------------------------------------------- mouse
def test_clicking_a_label_or_the_button_on_the_picture_selects_it():
    for family in Family.ALL:
        m, _ = _screen(family)
        for control in controls_for_family(family):
            m.waiting = None
            m.handle_mouse("click", _centre(m.layout.callouts[control]["rect"]))
            assert m.waiting == control, (family, control, "label")
            m.waiting = None
            m.handle_mouse("click", m.layout.anchor(control))
            assert m.waiting == control, (family, control, "picture")
        m.handle_mouse("click", (2, 2))
        assert m.waiting is None, "clicking empty space cancels"


def test_hover_moves_the_focus_but_not_while_waiting():
    m, _ = _screen()
    m.handle_mouse("move", _centre(m.layout.callouts["b"]["rect"]))
    assert m.focus == ("control", "b")
    m.select(("control", "x"))
    m.handle_mouse("move", _centre(m.layout.callouts["b"]["rect"]))
    assert m.focus == ("control", "x") and m.waiting == "x"


def test_settings_toggles_sliders_reset_and_back():
    m, saved = _screen()
    L = m.layout
    m.handle_mouse("click", _centre(L.options["invert_y"]["rect"]))
    assert m.bindings.invert_y
    m.handle_mouse("click", _centre(L.options["swap_sticks"]["rect"]))
    assert m.bindings.swap_sticks and m.bindings.move_stick == "right"
    m.handle_mouse("click", _centre(L.options["vibration"]["rect"]))
    assert not m.bindings.vibration
    s0 = m.bindings.look_sensitivity
    m.handle_mouse("click", _centre(L.options["sensitivity"]["plus"]))
    m.handle_mouse("click", _centre(L.options["sensitivity"]["plus"]))
    assert abs(m.bindings.look_sensitivity - (s0 + 0.2)) < 1e-9
    m.handle_mouse("click", _centre(L.options["deadzone"]["minus"]))
    assert m.bindings.deadzone < ControllerBindings().deadzone
    # keyboard: left/right adjusts the focused setting, the wheel too
    m.focus = ("option", "sensitivity")
    m.handle_key("left")
    m.handle_wheel(-1)
    assert abs(m.bindings.look_sensitivity - s0) < 1e-9
    m.bindings.bind("y", "c")
    m.handle_mouse("click", _centre(L.options["reset"]["rect"]))
    assert m.bindings.to_dict() == ControllerBindings().to_dict()
    assert m.handle_mouse("click", _centre(L.options["back"]["rect"])) == ControllerMenuModel.BACK
    assert len(saved) >= 7


def test_keyboard_navigation_reaches_every_item_and_wraps():
    m, _ = _screen()
    seen = set()
    for _ in range(len(m.items)):
        seen.add(m.focus)
        m.handle_key("down")
    assert seen == set(m.items)
    assert m.focus == ("control", "a"), "wrapped back to the start"
    # left/right hop between the label columns and the settings panel
    m.focus = ("control", "lefttrigger")
    m.handle_key("right")
    assert m.focus[0] == "control" and m.focus[1] in m.layout.columns[1]
    m.handle_key("right")
    assert m.focus[0] == "option"
    m.focus = ("option", "reset")
    m.handle_key("left")
    assert m.focus[0] == "control"


def test_switching_family_keeps_the_state_consistent():
    m, _ = _screen(Family.PLAYSTATION)
    m.select(("control", "touchpad"))
    assert m.waiting == "touchpad"
    m.set_device("Xbox pad", Family.XBOX)
    assert m.waiting is None, "the touchpad does not exist on an Xbox pad"
    assert m.focus in m.items


# ------------------------------------------------------------------ layout
def test_layout_fits_the_screen_and_nothing_overlaps():
    for family in Family.ALL:
        for (w, h) in ((1280, 720), (1920, 1080), (1024, 768)):
            L = ControllerLayout(family, w, h)
            rects = [e["rect"] for e in L.callouts.values()] + [e["rect"] for e in L.options.values()]
            for x, y, rw, rh in rects + [L.panel_rect, L.status_rect]:
                assert x >= 0 and y >= 0 and x + rw <= w + 0.5 and y + rh <= h + 0.5, (family, w, h)
            boxes = [e["rect"] for e in L.callouts.values()] + [e["rect"] for e in L.options.values()]
            for i, a in enumerate(boxes):
                for b in boxes[i + 1:]:
                    overlap = not (a[0] + a[2] <= b[0] or b[0] + b[2] <= a[0] or
                                   a[1] + a[3] <= b[1] or b[1] + b[3] <= a[1])
                    assert not overlap, (family, a, b)
            # every shown control has a label and can be hit on the picture
            for control in controls_for_family(family):
                assert control in L.callouts and L.hit(L.anchor(control)) == ("control", control), (family, control)
            assert PAUSE_CONTROL in L.callouts


def test_playstation_layout_is_symmetric_xbox_is_offset():
    xb = ControllerLayout(Family.XBOX)
    ps = ControllerLayout(Family.PLAYSTATION)
    assert ps.anchor("leftstick")[1] == ps.anchor("rightstick")[1], "DualShock/DualSense: both sticks low"
    assert xb.anchor("leftstick")[1] < xb.anchor("rightstick")[1], "Xbox: left stick high, right stick low"
    assert ps.anchor("dpup")[1] < ps.anchor("leftstick")[1], "PlayStation: D-pad above the left stick"


# ------------------------------------------------------------- pause menu model
def test_pause_menu_has_a_controller_entry():
    from render.menu import PauseMenu       # importable without a GL context
    m = MenuModel(PauseMenu.ITEMS)
    m.open()
    assert [i[1] for i in m.items] == [MenuModel.RESUME, MenuModel.MULTIPLAYER, MenuModel.CONTROLLER,
                                       MenuModel.FULLSCREEN, MenuModel.QUIT]
    m.handle("down")
    assert m.handle("confirm") == MenuModel.MULTIPLAYER
    m.handle("down")
    assert m.handle("confirm") == MenuModel.CONTROLLER
    m.handle("down")
    assert m.handle("confirm") == MenuModel.FULLSCREEN


# ------------------------------------------------------------------ rendering
HOT = (255, 160, 40)


def _render(model, state):
    pygame = headless_pygame()
    from render.controller_view import ControllerView
    view = ControllerView(model)
    surf = view.draw(state, 0.0)
    return pygame, surf


def _near(c, ref, tol=40):
    return all(abs(int(c[i]) - ref[i]) <= tol for i in range(3))


def test_view_lights_up_pressed_buttons_on_the_picture():
    for family in Family.ALL:
        m, _ = _screen(family)
        st = PadState()
        _, idle = _render(m, st)
        idle = idle.copy()
        for control in ("a", "leftshoulder", "dpleft", "back"):
            st = PadState()
            st.set_button(control, True)
            _, surf = _render(m, st)
            L = m.layout
            geo = L.controls[control]
            if geo["shape"] == "circle":
                px = (geo["center"][0] - geo["r"] * 0.6, geo["center"][1] - geo["r"] * 0.6)
            else:
                x, y, w, h = geo["rect"]
                px = (x + w * 0.25 + 1, y + h * 0.25 + 1)
            px = (int(px[0]), int(px[1]))
            assert not _near(idle.get_at(px), HOT), (family, control, "idle is not lit")
            if control == "a":
                # face buttons keep their colour (A stays green on Xbox) but brighten, with an amber halo
                assert sum(surf.get_at(px)[:3]) > sum(idle.get_at(px)[:3]) + 120, (family, surf.get_at(px))
                halo = (int(geo["center"][0]), int(geo["center"][1] + geo["r"] + 3 * L.s))
                assert surf.get_at(halo)[0] > idle.get_at(halo)[0] + 40, (family, "halo")
            else:
                x, y, w, h = [int(v) for v in geo["rect"]]
                pts = [(x + i, y + j) for i in range(1, w - 1) for j in range(1, h - 1)]
                lit = sum(_near(surf.get_at(p), HOT, 60) for p in pts) / len(pts)
                lit_idle = sum(_near(idle.get_at(p), HOT, 60) for p in pts) / len(pts)
                assert lit > 0.5 and lit_idle < 0.05, (family, control, lit, lit_idle)
            # its label box lights up too
            lx, ly, lw, lh = L.callouts[control]["rect"]
            assert _near(surf.get_at((int(lx + 8), int(ly + lh / 2))), HOT, 50), (family, control, "label")


def test_view_moves_stick_caps_and_fills_triggers():
    m, _ = _screen()
    L = m.layout
    st = PadState()
    st.set_axis("rightx", 1.0)
    st.set_axis("lefttrigger", 0.5)
    _, surf = _render(m, st)
    cx, cy = L.anchor("rightstick")
    r = L.controls["rightstick"]["r"]
    cap_x = cx + r * 0.42
    right_of_cap = surf.get_at((int(cap_x + r * 0.5), int(cy)))
    left_of_centre = surf.get_at((int(cx - r * 0.5), int(cy)))
    assert sum(right_of_cap[:3]) > sum(left_of_centre[:3]) + 60, "the cap followed the stick to the right"
    x, y, w, h = L.controls["lefttrigger"]["rect"]
    low = surf.get_at((int(x + w / 2), int(y + h * 0.9)))
    high = surf.get_at((int(x + w / 2), int(y + h * 0.1)))
    assert low[0] > high[0] + 60, "the trigger fills from the bottom as it is pulled"


def test_view_renders_every_family_and_state_opaque():
    pygame = headless_pygame()
    for family in Family.ALL:
        m, _ = _screen(family)
        st = PadState()
        for c in ("a", "b", "x", "y", "leftstick", "rightstick", "dpup", "start", "guide"):
            st.set_button(c, True)
        st.set_axis("leftx", -1.0)
        st.set_axis("righttrigger", 1.0)
        m.select(("control", "x"))
        _, surf = _render(m, st)
        assert surf.get_size() == (1280, 720)
        # the surface is uploaded as RGB: every pixel opaque, no see-through AA edges
        assert not (surf.get_flags() & pygame.SRCALPHA)
        m.device_name = None
        _render(m, None)                   # no controller connected
    m, _ = _screen(Family.PLAYSTATION, 1920, 1080)
    _, surf = _render(m, PadState())
    assert surf.get_size() == (1920, 1080)
