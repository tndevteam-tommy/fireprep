"""The netrunner HUD and the world-to-screen projector, drawn into a recording context
(no OpenGL window). Checks that every overlay path runs and places markers sensibly."""

import math

import pytest
from tests.test_render_backends import RecordingContext

from cyberduke.render.hud import Hud, target_status_text
from cyberduke.render.projection import Projector
from cyberduke.render.scene import ViewportConfig
from cyberduke.simulation import InputState, Simulation
from cyberduke.systems.hacking import HackKind

DT = 1 / 60
VP = ViewportConfig()


@pytest.fixture(autouse=True)
def _fonts():
    import pygame

    pygame.font.init()
    yield


def projector_for(sim):
    v = sim.viewpoint
    return Projector(VP.width, VP.height, VP.fov_y_deg, VP.near, VP.far, v.eye, v.target)


def texts(ctx):
    """Strings the HUD rasterised (the label cache records an upload per new string)."""
    return [c for c in ctx.calls if c[0] == "upload"]


# ----------------------------------------------------------------------------- projector
def test_point_straight_ahead_projects_to_the_screen_centre():
    p = Projector(1024, 768, 60.0, 0.1, 55.0, (0.0, 1.6, 0.0), (0.0, 1.6, 1.0))
    x, y, depth = p.project(0.0, 1.6, 10.0)
    assert (x, y) == pytest.approx((512.0, 384.0))
    assert 0.0 < depth < 1.0


def test_points_behind_the_viewer_are_not_projected():
    p = Projector(1024, 768, 60.0, 0.1, 55.0, (0.0, 1.6, 0.0), (0.0, 1.6, 1.0))
    assert p.project(0.0, 1.6, -5.0) is None


def test_right_and_up_map_to_plus_x_and_minus_y_pixels():
    # Looking along +X, the viewer's right is +Z and up is +Y (y grows downward on screen)
    p = Projector(1024, 768, 60.0, 0.1, 55.0, (0.0, 1.6, 0.0), (1.0, 1.6, 0.0))
    cx, cy, _ = p.project(10.0, 1.6, 0.0)
    rx, _, _ = p.project(10.0, 1.6, 3.0)
    _, uy, _ = p.project(10.0, 4.6, 0.0)
    assert rx > cx and uy < cy
    assert p.on_screen(cx, cy) and not p.on_screen(-10, cy) and p.on_screen(-10, cy, margin=20)


def test_projector_agrees_with_the_fov():
    """A point at the edge of the vertical FOV lands on the top edge of the viewport."""
    p = Projector(1024, 768, 60.0, 0.1, 55.0, (0.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    _, y, _ = p.project(0.0, math.tan(math.radians(30.0)) * 10.0, 10.0)
    assert y == pytest.approx(0.0, abs=0.5)


# ----------------------------------------------------------------------------- hud
def test_hud_draws_markers_for_scanned_targets_and_the_focus_menu():
    sim = Simulation()
    look = sim.terminals[0]
    sim.player.yaw = math.degrees(math.atan2(look.z - sim.player.z, look.x - sim.player.x))
    sim.update(DT, InputState(scan=True))
    sim.update(DT, InputState())
    assert sim.hacking.focus is look
    ctx = RecordingContext()
    hud = Hud(ctx, VP.width, VP.height)
    hud.draw(sim, projector_for(sim), DT)
    names = [c[0] for c in ctx.calls]
    assert names[0] == "begin_overlay" and names[-1] == "end_overlay"
    projector = projector_for(sim)
    in_front = [
        s for s in sim.hacking.statuses() if projector.project(*s.target.hack_position) is not None
    ]
    assert len(in_front) == 2, "the terminal and the guard; the doors/camera are behind"
    assert len(sim.hacking.statuses()) >= 4, "...but those are still highlighted (pinged)"
    # every marker is 4 corner brackets of 2 rects; plus crosshair (4), empty meter (1),
    # panel (2), the level title card just after arriving (3) and the clock (2)
    assert names.count("rect") == len(in_front) * 8 + 4 + 1 + 2 + 3 + 2
    assert len(texts(ctx)) >= 12, "target labels, routes, menu entries, net status, hints"
    assert names.count("image") >= len(in_front) * 2


def test_hud_draws_the_camera_feed_frame_and_upload_bar():
    sim = Simulation()
    sim.hacking.apply(sim.cameras[0], HackKind.HIJACK)
    sim.update(DT, InputState(scan=True))
    sim.update(DT, InputState(hack_slot=1))  # force open the bulkhead, straight ahead
    assert sim.hacking.upload is not None and sim.hijacked_camera is not None
    ctx = RecordingContext()
    Hud(ctx, VP.width, VP.height).draw(sim, projector_for(sim), DT)
    rects = [c for c in ctx.calls if c[0] == "rect"]
    full_screen = [r for r in rects if r[3] == VP.width and r[4] == VP.height]
    assert full_screen, "the feed tint covers the screen"
    assert sum(1 for r in rects if r[3] == VP.width and r[4] == 1) > 100, "scanlines"


def test_hud_without_a_projector_and_in_end_states_still_draws():
    sim = Simulation()
    ctx = RecordingContext()
    hud = Hud(ctx, VP.width, VP.height)
    for state in ("PLAYING", "CAUGHT"):
        sim.state = type(sim.state)(state)
        hud.draw(sim)
    assert [c[0] for c in ctx.calls].count("begin_overlay") == 2


def test_target_status_text_covers_every_kind_and_state():
    sim = Simulation()
    datacore = sim.terminals[1]
    terminal = sim.terminals[2]  # the maintenance desk: the one that can be destroyed
    guard, cam, vm = sim.guards[0], sim.cameras[0], sim.vending_machines[0]
    blast, bulkhead = sim.room.doors["blast_door"], sim.room.doors["vault_bulkhead"]
    assert target_status_text(terminal) == "ONLINE"
    assert target_status_text(datacore) == "PRIVATE / HAS DATA"
    datacore.steal_intel()
    assert target_status_text(datacore) == "PRIVATE / DATA COPIED"
    terminal.overload()
    assert target_status_text(terminal) == "DESTROYED"
    assert target_status_text(guard) == "PACING"
    guard.blind(4.0)
    assert target_status_text(guard) == "BLINDED 4s"
    guard.knock_out()
    assert target_status_text(guard) == "OFFLINE"
    assert target_status_text(cam) == "SWEEPING"
    cam.hijack()
    assert target_status_text(cam) == "HIJACKED"
    cam.shut_down(9.0)
    assert target_status_text(cam) == "DOWN 9s"
    assert target_status_text(vm) == "READY"
    vm.trigger_distraction()
    assert target_status_text(vm) == "SHORTED"
    vm.is_sparking = False
    assert target_status_text(vm) == "OUT OF ORDER", "the sparks are over; still broken"
    vm.repair()
    assert target_status_text(vm).startswith("RECHARGING")
    assert target_status_text(blast) == "HIGH SECURITY"
    assert target_status_text(bulkhead) == "LOCKED"
    bulkhead.unlock()
    assert target_status_text(bulkhead) == "OPENING"
    bulkhead.update(10.0)
    assert target_status_text(bulkhead) == "OPEN"


def test_hud_shows_the_encrypted_door_prompt_and_the_feed_switch_hint():
    from cyberduke.render.hud import Hud

    sim = Simulation()
    sim.player.x, sim.player.z = 0.0, 40.5
    sim.update(DT, InputState())
    ctx = RecordingContext()
    Hud(ctx, VP.width, VP.height).draw(sim, projector_for(sim), DT)
    labels = [c for c in ctx.calls if c[0] == "upload"]
    assert labels, "a prompt was rasterised"
    # the label cache is keyed by text; peek at it for the strings
    hud = Hud(RecordingContext(), VP.width, VP.height)
    hud.draw(sim, projector_for(sim), DT)
    texts = {k[0] for k in hud.labels._labels}
    assert "GREENHOUSE DOOR: ENCRYPTED - NEEDS GREENHOUSE DOOR CIPHER" in texts
    sim.player.intel.append("greenhouse door cipher")
    sim.update(DT, InputState())
    hud.draw(sim, projector_for(sim), DT)
    texts = {k[0] for k in hud.labels._labels}
    assert "GREENHOUSE DOOR: CIPHER ACCEPTED" in texts
    sim.hacking.apply(sim.hacking.target("greenhouse_entry_camera"), HackKind.HIJACK)
    sim.update(DT, InputState())
    hud.draw(sim, projector_for(sim), DT)
    texts = {k[0] for k in hud.labels._labels}
    assert "[Q] DISCONNECT   MOUSE: PAN / TILT   [<] [>] SWITCH FEED (1/4)" in texts


def test_hud_draws_the_terminal_interface_instead_of_the_scanner_while_jacked_in():
    from cyberduke.render.hud import Hud

    sim = Simulation()
    sim.player.x, sim.player.z = 3.5, -2.5
    for _ in range(int(3.0 / DT)):
        sim.update(DT, InputState(interact=True))
        if sim.hacking.session is not None:
            break
    assert sim.hacking.session is sim.terminals[0]
    hud = Hud(RecordingContext(), VP.width, VP.height)
    hud.draw(sim, projector_for(sim), DT)
    texts = {k[0] for k in hud.labels._labels}
    assert "TERMINAL: STAGING TERMINAL" in texts
    assert "[1] UNLOCK DOORS" in texts and "[2] HIJACK CAMERAS" in texts
    assert "[3] DISABLE CAMERAS" in texts and "[4] SHORT CIRCUIT (DISTRACT & DESTROY)" in texts
    assert not any("STEAL DATA" in t for t in texts), "nothing to steal here: no such row"
    assert "LOCKED: CRITICAL SYSTEM" in texts, "and why that one is refused"
    assert "[Q] DISCONNECT" in texts
    assert not any(t.startswith("TARGET:") for t in texts), "no quickhack menu competing"


def test_hud_draws_the_pause_menu_over_a_frozen_frame():
    from cyberduke.core.menu import PauseMenu
    from cyberduke.render.hud import Hud

    sim = Simulation()
    sim.update(DT, InputState(toggle_pause=True))
    menu = PauseMenu(graphics="HIGH")
    menu.move(1)
    ctx = RecordingContext()
    hud = Hud(ctx, VP.width, VP.height)
    before = hud.time
    hud.draw(sim, projector_for(sim), 0.0, menu=menu)
    assert hud.time == before, "a paused frame does not animate the overlay either"
    texts = {k[0] for k in hud.labels._labels}
    assert "PAUSED" in texts
    assert {"  RESUME", "> SAVE GAME", "  LOAD GAME", "  GRAPHICS", "  QUIT TO DESKTOP"} <= texts
    assert "<  HIGH  >" in texts
    full_screen = [c for c in ctx.calls if c[0] == "rect" and c[3] == VP.width]
    assert full_screen, "the world is dimmed behind the menu"
    hud.draw(sim, projector_for(sim), 0.0)  # no menu passed: nothing of it is drawn
    assert [c for c in ctx.calls if c[0] == "image"], "the rest of the HUD still draws"


# ----------------------------------------------------------------------------- title card
def test_the_level_title_card_shows_on_arrival_and_fades_after_a_few_seconds():
    from cyberduke.render.hud import TITLE_SECONDS

    sim = Simulation()
    ctx = RecordingContext()
    hud = Hud(ctx, VP.width, VP.height)
    hud.draw(sim, projector_for(sim), DT)
    labels = hud.labels._labels
    assert any(k[0] == "CORPORATE FACILITY" for k in labels), "the level name, on arrival"
    assert any(k[0] == sim.level.brief for k in labels), "...with its brief"
    assert 0.0 < hud.title_time_left < TITLE_SECONDS
    # it runs down with the HUD clock and then stops being drawn
    for _ in range(int(TITLE_SECONDS / DT) + 5):
        hud.draw(sim, projector_for(sim), DT)
    assert hud.title_time_left <= 0.0
    ctx.calls.clear()
    hud.draw(sim, projector_for(sim), DT)
    rects_without_title = [c for c in ctx.calls if c[0] == "rect"]
    # a paused HUD (dt 0) never runs the card down - it freezes with everything else
    # and the next level's arrival brings a new card
    from cyberduke.config import SIMULATION

    sim.player.x, sim.player.z = 0.0, 78.6
    for _ in range(int((SIMULATION.level_fade_out + SIMULATION.level_fade_in) / DT) + 3):
        sim.update(DT, InputState())  # (out of the door: the fade out, and in again)
    assert sim.level.name == "Night Market Alley" and sim.screen_black == 0.0
    ctx.calls.clear()
    hud.draw(sim, projector_for(sim), 0.0)
    assert hud.title_time_left == TITLE_SECONDS, "paused: the card does not run down"
    assert len([c for c in ctx.calls if c[0] == "rect"]) == len(rects_without_title) + 3
    assert any(k[0] == "NIGHT MARKET ALLEY" for k in labels)
    assert any(k[0].startswith("INTEL ON DECK") for k in labels), "no terminals to count"


# ----------------------------------------------------------------------------- layout
def _record_texts(hud):
    """Wraps ``hud.text`` to record every label as (text, x, y, w, h) in screen pixels."""
    drawn = []
    original = hud.text

    def text(s, x, y, size=14, color=(255, 255, 255)):
        _, w, h = hud.labels.get(s, size, color)
        drawn.append((s, x, y, w, h))
        original(s, x, y, size, color)

    hud.text = text
    return drawn


def _overlaps(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah


def _breached_vault_view():
    """Standing in the vault doorway with the Halcyon net breached: networked targets on
    both sides of the view and one far off to the right (the vending machine)."""
    sim = Simulation()
    for door in sim.room.doors.values():
        door.unlock()
        door.open_amount = door.travel
    sim.player.x, sim.player.z, sim.player.yaw = 0.0, 20.0, 90.0
    sim.hacking.breach("halcyon", "test")
    sim.update(DT, InputState())
    return sim


def test_labels_pinned_to_the_screen_edge_stay_on_the_screen():
    """Regression: an off-screen target's name was drawn 60 px left of its edge pin, so on
    the left edge it was cut off ("G GUARD") and long names ran off the right edge."""
    sim = _breached_vault_view()
    sim.player.yaw = 150.0  # half of the networked devices are now off to one side
    sim.update(DT, InputState())
    hud = Hud(RecordingContext(), VP.width, VP.height)
    drawn = _record_texts(hud)
    hud.draw(sim, projector_for(sim), DT)
    assert drawn
    for s, x, _y, w, _h in drawn:
        assert x >= 0 and x + w <= VP.width, f"{s!r} runs off the screen at x={x:.0f}"


def test_the_focus_panel_is_as_wide_as_what_it_says():
    """Regression: the panel was a fixed 300 px, and a long refusal ("LOCKED: ENCRYPTED
    (NEEDS GREENHOUSE DOOR CIPHER)") ran out of it over the world labels."""
    sim = _breached_vault_view()
    assert sim.hacking.focus is sim.room.doors["greenhouse_door"]
    ctx = RecordingContext()
    hud = Hud(ctx, VP.width, VP.height)
    drawn = _record_texts(hud)
    hud.draw(sim, projector_for(sim), DT)
    px, py, pw, ph = hud._focus_panel_rect(sim)
    refusal = next(d for d in drawn if d[0].startswith("LOCKED: ENCRYPTED"))
    assert refusal[1] + refusal[3] <= px + pw, "the refusal fits inside the panel"
    assert ("rect", px, py, pw, ph) in ctx.calls, "and the panel is drawn at that size"
    names = {t.display_name for t in sim.hacking.targets}  # the panel itself says "TARGET: ..."
    world_labels = [d for d in drawn if d[0] in names]
    assert world_labels
    for s, x, y, w, h in world_labels:  # the stacked world labels stay clear of the panel
        assert not _overlaps((px, py, pw, ph), (x, y, w, h)), f"{s!r} is under the panel"


def test_the_pause_menu_panel_holds_its_key_hint():
    from cyberduke.core.menu import PauseMenu

    sim = Simulation()
    sim.update(DT, InputState(toggle_pause=True))
    ctx = RecordingContext()
    hud = Hud(ctx, VP.width, VP.height)
    drawn = _record_texts(hud)
    hud.draw(sim, projector_for(sim), 0.0, menu=PauseMenu())
    hint = next(d for d in drawn if d[0].startswith("[UP]/[DOWN]"))
    outline = max((c for c in ctx.calls if c[0] == "rect_outline"), key=lambda c: c[3] * c[4])
    _, px, _, pw, _ = outline
    assert px <= hint[1] and hint[1] + hint[3] <= px + pw, "the hint used to spill out"


def test_the_title_card_clears_the_status_column_and_the_upload_bar():
    """Regression: the arrival band ran straight through the network status lines in the
    top right corner (and the upload / link bars under the detection meter)."""
    from cyberduke.render.hud import TITLE_Y

    sim = Simulation()
    hud = Hud(RecordingContext(), VP.width, VP.height)
    drawn = _record_texts(hud)
    hud.draw(sim, projector_for(sim), DT)
    band = (0, TITLE_Y - 14, VP.width, 60)
    status = [d for d in drawn if "NET:" in d[0] or "SCAN" in d[0] or "INTEL" in d[0]]
    assert status
    for s, x, y, w, h in status:
        assert not _overlaps(band, (x, y, w, h)), f"title band over {s!r}"
    assert TITLE_Y - 14 > 130, "below the upload bar (y 104..130)"


def test_the_caught_screen_says_what_happens_next_and_fades_to_black():
    """No win banner any more. Caught, a red YOU WERE CAUGHT screen says what it costs -
    in the facility a new game, in the city being thrown out - then fades to black."""
    from cyberduke.config import SIMULATION
    from cyberduke.core.state import GameState

    sim = Simulation()
    sim.state, sim.state_timer = GameState.CAUGHT, 0.5
    hud = Hud(RecordingContext(), VP.width, VP.height)
    drawn = _record_texts(hud)
    hud.draw(sim, None, DT)
    texts = [d[0] for d in drawn]
    assert "YOU WERE CAUGHT" in texts and "THE RUN IS OVER - A NEW GAME BEGINS" in texts
    fills = []

    class Colours(RecordingContext):
        def rect(self, x, y, w, h, rgba):
            fills.append((w, h, tuple(rgba)))
            super().rect(x, y, w, h, rgba)

    hud = Hud(Colours(), VP.width, VP.height)
    sim.state_timer = SIMULATION.caught_hold + SIMULATION.caught_fade * 0.99
    hud.draw(sim, None, DT)
    black = [f for f in fills if f[:2] == (VP.width, VP.height) and f[2][:3] == (0.0, 0.0, 0.0)]
    assert black and black[-1][2][3] > 0.95, "faded (nearly) to black"
    reds = [f for f in fills if f[:2] == (VP.width, VP.height) and f[2][0] > 0.5]
    assert reds, "over the red screen"


def test_the_label_cache_evicts_the_least_recently_used_label():
    """Regression: eviction was LIFO, so once full of stale countdowns every new label threw
    out the one made just before it - often one still on screen - and labels thrashed."""
    from cyberduke.render import hud as hud_module

    ctx = RecordingContext()
    cache = hud_module.LabelCache(ctx)
    limit = hud_module._MAX_CACHED_LABELS
    for i in range(limit):
        cache.get(f"label {i}", 10, (255, 255, 255))
    cache.get("label 0", 10, (255, 255, 255))  # still in use: refreshed
    uploads = len([c for c in ctx.calls if c[0] == "upload"])
    cache.get("new label", 10, (255, 255, 255))
    keys = {k[0] for k in cache._labels}
    assert "label 0" in keys and "label 1" not in keys and "new label" in keys
    assert "label 255" in keys, "the most recent one survives"
    cache.get("label 0", 10, (255, 255, 255))
    assert len([c for c in ctx.calls if c[0] == "upload"]) == uploads + 1, "no re-rasterising"


def test_the_hud_shows_the_time_of_day_on_a_small_digital_clock():
    sim = Simulation()
    sim.clock.set(21 * 60 + 7)
    hud = Hud(RecordingContext(), VP.width, VP.height)
    drawn = _record_texts(hud)
    sim.time = 0.1  # (the colon blinks: shown in the first half of every second)
    hud.draw(sim, None, DT)
    texts = [d[0] for d in drawn]
    assert "21:07" in texts
    assert not any(t.startswith("DAY") for t in texts), "the time of day only: no day count"
    clock = next(d for d in drawn if d[0] == "21:07")
    assert clock[1] < 60 and clock[2] < 40, "top left, clear of everything else"


def test_the_fade_at_a_levels_last_door_is_drawn_over_everything():
    """Walking out of the biodome the picture goes black - a full-screen black rect over
    the HUD, as opaque as ``screen_black`` says - and comes back on the alley."""

    class Rects(RecordingContext):
        def rect(self, x, y, w, h, rgba):
            self._rec("rect", x, y, w, h, tuple(rgba))

    sim = Simulation()
    sim.player.x, sim.player.z = 0.0, 78.6
    for _ in range(30):  # half way through the fade out
        sim.update(DT, InputState())
    ctx = Rects()
    Hud(ctx, VP.width, VP.height).draw(sim, None, DT)
    black = [c for c in ctx.calls if c[0] == "rect" and c[3:5] == (VP.width, VP.height)]
    assert black and black[-1][5][:3] == (0.0, 0.0, 0.0)
    assert black[-1][5][3] == pytest.approx(sim.screen_black) and 0.3 < sim.screen_black < 0.7
