"""Orthographic 2D overlay: crosshair, detection meter, hack progress, the netrunner layer
(scanner markers, target menu, upload bar, network status, feed log, camera-feed frame)
and the state banners."""

from __future__ import annotations

import math
from collections import OrderedDict
from typing import Any

import pygame

from cyberduke.config import DETECTION
from cyberduke.core.menu import PauseMenu
from cyberduke.render.context import RGBA, DrawContext
from cyberduke.render.icons import IconCache
from cyberduke.render.projection import Projector
from cyberduke.render.quality import GraphicsSettings, Quality, settings_for
from cyberduke.render.screens import (
    draw_captions,
    draw_caught,
    draw_load_menu,
    draw_menu,
    draw_prompt,
    draw_tab_screen,
    draw_title,
    draw_toasts,
)
from cyberduke.render.ui import Ui
from cyberduke.simulation import Simulation
from cyberduke.systems.hacking import TARGET_KIND_LABELS, Access, TargetStatus

RGB = tuple[int, int, int]

_FONT_NAME = "Arial"
_MAX_CACHED_LABELS = 256
TITLE_SECONDS = 5.0  # how long the level title card stays up after arriving
FOCUS_NOTE_X = 150  # where an option's note (upload time / refusal) starts in the focus panel
TITLE_Y = 150  # the title card's text line (its band spans TITLE_Y - 14 .. TITLE_Y + 46)

CYAN: RGBA = (0.0, 1.0, 1.0, 1.0)
FOCUS_RGBA: RGBA = (1.0, 0.95, 0.35, 1.0)
DIM_RGBA: RGBA = (0.35, 0.65, 0.75, 0.8)
PRIVATE_RGBA: RGBA = (1.0, 0.3, 0.9, 1.0)
OFFLINE_RGBA: RGBA = (0.5, 0.5, 0.55, 0.8)


def _rgb255(rgba: RGBA) -> RGB:
    return tuple(int(max(0.0, min(1.0, c)) * 255) for c in rgba[:3])  # type: ignore[return-value]


class LabelCache:
    """Rasterises strings with pygame and keeps them as textures on the GPU (bounded, least
    recently used out first).

    The eviction used to be ``dict.popitem()`` - *last* in, first out: once the cache was
    full of stale countdown labels ("DOWN 179s", "DOWN 178s", ...) every new label evicted
    the one created just before it, often one still on screen, so labels were rasterised
    again every frame."""

    def __init__(self, ctx: DrawContext) -> None:
        self.ctx = ctx
        self._labels: OrderedDict[tuple[str, int, RGB], tuple[int, int, int]] = OrderedDict()
        self._fonts: dict[int, pygame.font.Font] = {}

    def font(self, size: int) -> pygame.font.Font:
        font = self._fonts.get(size)
        if font is None:  # SysFont looks the font up on disk: do it once per size
            font = self._fonts[size] = pygame.font.SysFont(_FONT_NAME, size, bold=True)
        return font

    def get(self, text: str, size: int, color: RGB) -> tuple[int, int, int]:
        key = (text, size, color)
        cached = self._labels.get(key)
        if cached is not None:
            self._labels.move_to_end(key)
            return cached
        if len(self._labels) >= _MAX_CACHED_LABELS:
            _, (old_tex, _, _) = self._labels.popitem(last=False)
            self.ctx.delete_texture(old_tex)
        surf = self.font(size).render(text, True, color)
        w, h = surf.get_size()
        tex = self.ctx.upload_texture_rgba(pygame.image.tostring(surf, "RGBA", True), w, h)
        self._labels[key] = (tex, w, h)
        return self._labels[key]


def target_status_text(t: Any) -> str:
    """What state a hack target is in, for its marker (render-side convenience)."""
    kind = t.target_kind
    if kind == "guard":
        if t.is_down:
            return "OFFLINE"
        if t.is_blinded:
            return f"BLINDED {t.blind_time_left:.0f}s"
        return t.state
    if kind == "camera":
        if not t.is_online:
            return f"DOWN {t.shutdown_timer:.0f}s"
        return "HIJACKED" if t.hijacked else "SWEEPING"
    if kind == "terminal":
        if t.destroyed:
            return "DESTROYED"
        bits = []
        if t.secured:
            bits.append("PRIVATE")
        if t.intel_stolen:
            bits.append("DATA COPIED")
        elif t.intel:
            bits.append("HAS DATA")
        return " / ".join(bits) if bits else "ONLINE"
    if kind == "door":
        if t.fully_open:
            return "OPEN"
        if t.unlocked:
            return "OPENING"
        return "HIGH SECURITY" if t.high_security else "LOCKED"
    if kind == "vending":
        if t.is_sparking:
            return "SHORTED"
        if getattr(t, "broken", False):
            return "OUT OF ORDER"
        return f"RECHARGING {t.cooldown:.0f}s" if t.cooldown > 0 else "READY"
    return ""


class Hud:
    def __init__(self, ctx: DrawContext, width: int, height: int) -> None:
        self.ctx = ctx
        self.width, self.height = width, height
        self.labels = LabelCache(ctx)
        self.time = 0.0
        self.settings: GraphicsSettings = settings_for(Quality.MEDIUM)
        self._title_generation: int | None = None  # the level the title card was shown for
        self.title_time_left = 0.0
        self.ui = Ui(ctx, self.labels, draw_text=lambda *a: self.text(*a))
        self.icons = IconCache(ctx)
        # picture -> texture (or None): quest pictures, rendered from the world by the
        # scene renderer (``SceneRenderer.quest_picture``)
        self.pictures = None

    # ------------------------------------------------------------------ helpers
    def text(
        self, text: str, x: float, y: float, size: int = 14, color: RGB = (255, 255, 255)
    ) -> None:
        tex, w, h = self.labels.get(text, size, color)
        self.ctx.image(tex, x, y, w, h)

    def text_width(self, text: str, size: int, color: RGB = (255, 255, 255)) -> int:
        _, w, _ = self.labels.get(text, size, color)
        return w

    def text_centered(self, text: str, cx: float, y: float, size: int, color: RGB) -> None:
        _, w, _ = self.labels.get(text, size, color)
        self.text(text, cx - w / 2, y, size, color)

    def text_right(self, text: str, right: float, y: float, size: int, color: RGB) -> None:
        _, w, _ = self.labels.get(text, size, color)
        self.text(text, right - w, y, size, color)

    def meter(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        fraction: float,
        bg: RGBA,
        fg: RGBA,
        outline: RGBA,
    ) -> None:
        self.ctx.rect(x, y, w, h, bg)
        if fraction > 0:
            self.ctx.rect(x, y, w * max(0.0, min(1.0, fraction)), h, fg)
        self.ctx.rect_outline(x, y, w, h, outline)

    def brackets(
        self, cx: float, cy: float, half: float, rgba: RGBA, arm: float | None = None
    ) -> None:
        """Four corner brackets around a point (the scanner marker)."""
        arm = arm if arm is not None else max(4.0, half * 0.45)
        t = 2.0
        for sx in (-1, 1):
            for sy in (-1, 1):
                x, y = cx + sx * half, cy + sy * half
                self.ctx.rect(min(x, x - sx * arm), y - t / 2, arm, t, rgba)
                self.ctx.rect(x - t / 2, min(y, y - sy * arm), t, arm, rgba)

    # ------------------------------------------------------------------ frame
    def draw(
        self,
        sim: Simulation,
        projector: Projector | None = None,
        dt: float = 0.0,
        menu: PauseMenu | None = None,
        game: Any = None,
    ) -> None:
        """The overlay. With a ``game`` (``cyberduke.game.Game``) what is on screen follows
        its mode: the start menu, the load menu, the [Tab] screen, or the HUD (with the
        pause menu over it when paused)."""
        self.time += dt
        self.ui.time = self.time
        self.ui.scanlines = self.settings.hud_scanlines
        ctx, width, height = self.ctx, self.width, self.height
        mode = game.mode.value if game is not None else "PLAYING"
        message = game.message if game is not None and game.message_time > 0 else ""
        if mode in ("TITLE", "SCREEN") or (mode == "LOAD" and game.load_return.value == "TITLE"):
            ctx.begin_overlay(width, height)
            try:
                if mode == "TITLE":
                    draw_title(self.ui, width, height, game.title_menu, message)
                elif mode == "LOAD":
                    draw_load_menu(self.ui, width, height, game.load_menu, message)
                else:
                    pictures = self.pictures or (lambda _p: None)
                    draw_tab_screen(self.ui, width, height, sim, game.screen, self.icons, pictures)
            finally:
                ctx.end_overlay()
            return
        if sim.level_generation != self._title_generation:  # just arrived somewhere
            self._title_generation = sim.level_generation
            self.title_time_left = TITLE_SECONDS
        ctx.begin_overlay(width, height)
        try:
            at_desk = sim.hacking.session is not None
            self._draw_crosshair()
            # At a desk the screen belongs to the terminal: no scanner markers or quickhack
            # menu competing with the interface for the same pixels.
            if projector is not None and not at_desk:
                self._draw_targets(sim, projector)
            self._draw_detection(sim.detection.level)
            self._draw_terminal_progress(sim.hud_terminal)
            self._draw_upload(sim)
            self._draw_terminal_session(sim)
            self._draw_door_prompt(sim)
            if not at_desk:
                self._draw_focus_panel(sim)
            self._draw_network_status(sim)
            self._draw_clock(sim)
            self._draw_log(sim)
            if sim.hijacked_camera is not None:
                self._draw_camera_feed(sim)
            if sim.hijacked_camera is None and not at_desk:
                self._draw_level_title(sim, dt)
            else:
                self.title_time_left = max(0.0, self.title_time_left - dt)
            draw_toasts(self.ui, width, height, sim)
            if sim.state.value == "PLAYING":
                draw_prompt(self.ui, width, height, sim.prompt)
            draw_captions(self.ui, width, height, sim)
            draw_caught(self.ui, width, height, sim)
            if mode == "PAUSED":
                draw_menu(self.ui, width, height, game.pause_menu, message=message)
            elif mode == "LOAD":
                draw_load_menu(self.ui, width, height, game.load_menu, message)
            elif menu is not None:
                draw_menu(self.ui, width, height, menu)
        finally:
            ctx.end_overlay()

    # ------------------------------------------------------------------ pieces
    def _draw_crosshair(self) -> None:
        ctx = self.ctx
        cx, cy = self.width / 2, self.height / 2
        ctx.rect(cx - 2, cy - 8, 4, 6, CYAN)
        ctx.rect(cx - 2, cy + 2, 4, 6, CYAN)
        ctx.rect(cx - 8, cy - 2, 6, 4, CYAN)
        ctx.rect(cx + 2, cy - 2, 6, 4, CYAN)

    def _draw_detection(self, detection_level: float) -> None:
        width = self.width
        bar_w, bar_h = 260, 14
        bar_x, bar_y = (width - bar_w) / 2, 45
        self.text("DETECTION LEVEL", bar_x, bar_y - 18, size=13, color=(200, 240, 255))
        if detection_level > DETECTION.red_level:
            fill = (1.0, 0.2, 0.2, 1.0)
        elif detection_level > DETECTION.yellow_level:
            fill = (1.0, 0.8, 0.0, 1.0)
        else:
            fill = (0.0, 0.8, 1.0, 1.0)
        self.meter(
            bar_x, bar_y, bar_w, bar_h, detection_level / 100.0, (0.1, 0.1, 0.15, 1.0), fill, CYAN
        )

    def _draw_terminal_progress(self, terminal: Any) -> None:
        """The physical link: holding [E] at a desk the netrunner has never been into.
        A breached terminal reconnects instantly, so it never shows a link bar again."""
        if terminal is None or terminal.is_hacked or terminal.progress <= 0:
            return
        h_w, h_h = 220, 12
        h_x, h_y = (self.width - h_w) / 2, 85
        self.text("ESTABLISHING LINK", h_x, h_y - 18, size=13, color=(100, 255, 150))
        green = (0.0, 1.0, 0.4, 1.0)
        self.meter(
            h_x, h_y, h_w, h_h, terminal.progress / 100.0, (0.1, 0.15, 0.1, 1.0), green, green
        )

    def _draw_terminal_session(self, sim: Simulation) -> None:
        """The interface of the computer the player is jacked into: what it can do for them,
        and why the options it refuses are refused."""
        terminal = sim.hacking.session
        if terminal is None:
            return
        ctx = self.ctx
        options = sim.hacking.session_options()
        w = 500
        x = (self.width - w) / 2 + 12  # text inset 12 px inside a panel centred on screen
        y = self.height / 2 - 110
        panel_h = 104 + 22 * len(options)
        ctx.rect(x - 12, y - 12, w, panel_h, (0.02, 0.08, 0.06, 0.82))
        ctx.rect_outline(x - 12, y - 12, w, panel_h, (0.2, 1.0, 0.55, 0.9))
        ctx.rect(x - 12, y - 12, w, 3, (0.2, 1.0, 0.55, 0.9))
        net = sim.hacking.network_of(terminal)
        self.text(f"TERMINAL: {terminal.display_name}", x, y, size=15, color=(140, 255, 190))
        self.text(
            f"{net.faction.name.upper()}  |  {'BREACHED' if net.breached else 'LINK ESTABLISHED'}",
            x, y + 18, size=10, color=(150, 220, 190),
        )  # fmt: skip
        doors = ", ".join(terminal.door_ids).replace("_", " ").upper() or "NONE"
        data = terminal.intel.upper() if terminal.intel else "NONE"
        if terminal.intel_stolen:
            data += " (COPIED)"
        self.text(f"LOCKS: {doors}", x, y + 34, size=10, color=(150, 220, 190))
        self.text(f"DATA: {data}", x, y + 46, size=10, color=(150, 220, 190))
        yy = y + 70
        for i, opt in enumerate(options, start=1):
            if opt.available:
                color, note = (120, 255, 200), f"{opt.spec.upload_time:.1f}s"
            else:
                color, note = (130, 130, 140), opt.reason
            self.text(f"[{i}] {opt.spec.label}", x, yy, size=13, color=color)
            self.text_right(note, x + w - 28, yy + 2, 10, color)
            yy += 22
        self.text("[Q] DISCONNECT", x, yy + 6, size=11, color=(150, 220, 190))
        if sim.hacking.notice_timer > 0.0:
            self.text_right(sim.hacking.notice, x + w - 28, yy + 6, 11, (255, 120, 120))
        if getattr(terminal, "notes", ""):
            # what is on its screen: a file, a note to self
            lines = self.ui.wrap(terminal.notes, w - 48, 13)
            ny = y - 12 + panel_h + 10
            nh = 30 + 18 * len(lines)
            ctx.rect(x - 12, ny, w, nh, (0.03, 0.06, 0.02, 0.85))
            ctx.rect_outline(x - 12, ny, w, nh, (0.8, 0.9, 0.3, 0.8))
            self.text("NOTES.TXT", x, ny + 6, size=10, color=(220, 240, 120))
            for k, line in enumerate(lines):
                self.text(line, x, ny + 22 + 18 * k, size=13, color=(230, 240, 200))

    def _marker_color(self, sim: Simulation, st: TargetStatus) -> RGBA:
        t = st.target
        if st.focused:
            pulse = 0.75 + 0.25 * math.sin(self.time * 8.0)
            return (FOCUS_RGBA[0] * pulse, FOCUS_RGBA[1] * pulse, FOCUS_RGBA[2] * pulse, 1.0)
        if st.access is Access.DIRECT:
            return PRIVATE_RGBA if t.secured else CYAN
        if st.access is Access.NETWORK:
            r, g, b = sim.hacking.network_of(t).faction.color
            return (r, g, b, 1.0)
        return DIM_RGBA

    def _draw_targets(self, sim: Simulation, projector: Projector) -> None:
        placed: list[tuple[float, float, float, float]] = []  # label boxes already on screen
        panel = self._focus_panel_rect(sim)
        if panel is not None:
            placed.append(panel)  # labels stack clear of the focus panel drawn over them
        label_h = 40.0
        for st in reversed(sim.hacking.statuses()):  # far first, so near markers draw on top
            t = st.target
            hit = projector.project(*t.hack_position)
            if hit is None:
                continue
            px, py, _ = hit
            color = self._marker_color(sim, st)
            rgb = _rgb255(color)
            if st.access is None:
                # Pinged, but out of reach (no route, no line of sight): a dim pip so the
                # netrunner knows something is there, with none of the framing and labels
                # that would read as an object floating in mid-air.
                if projector.on_screen(px, py, margin=0):
                    self.ctx.rect(px - 3, py - 3, 6, 6, (*color[:3], 0.55))
                continue
            if not projector.on_screen(px, py, margin=0):
                # Off to the side: pin a small marker to the screen edge so the player can
                # tell where a networked device is without turning to find it.
                ex = max(30.0, min(self.width - 30.0, px))
                ey = max(110.0, min(self.height - 110.0, py))
                self.brackets(ex, ey, 8.0, color, arm=5.0)
                # centred under the pin, but never running off the edge of the screen
                tw = self.text_width(t.display_name, 9, rgb)
                tx = max(4.0, min(self.width - tw - 4.0, ex - tw / 2.0))
                self.text(t.display_name, tx, ey + 12, size=9, color=rgb)
                continue
            half = max(12.0, min(46.0, 160.0 / max(st.distance, 1.0)))
            self.brackets(px, py, half, color)
            # Labels stack downward when two targets line up on screen.
            lx, ly = px + half + 6, py - half
            lw = 150.0
            for _ in range(8):
                if not any(
                    lx < ox + ow and ox < lx + lw and ly < oy + oh and oy < ly + label_h
                    for ox, oy, ow, oh in placed
                ):
                    break
                ly += label_h
            placed.append((lx, ly, lw, label_h))
            self.text(t.display_name, lx, ly, size=12, color=rgb)
            kind = TARGET_KIND_LABELS.get(t.target_kind, t.target_kind.upper())
            route = sim.hacking.explain_access(t)
            self.text(f"{kind} | {route}", lx, ly + 14, size=10, color=(200, 220, 230))
            status = target_status_text(t)
            if status:
                self.text(status, lx, ly + 26, size=10, color=(170, 190, 200))
            if st.on_network and not st.scanned:
                self.text("NET", px - half - 26, py - half, size=10, color=rgb)

    def _focus_lines(self, sim: Simulation):
        """The focus panel's content: title, subtitle and (label, note, colour) rows."""
        focus = sim.hacking.focus
        options = sim.hacking.options(focus)
        net = sim.hacking.network_of(focus)
        kind = TARGET_KIND_LABELS.get(focus.target_kind, focus.target_kind.upper())
        route = sim.hacking.explain_access(focus)
        rows = []
        for i, opt in enumerate(options, start=1):
            if opt.available:
                color, note = (120, 255, 200), f"{opt.spec.upload_time:.1f}s"
            else:
                color, note = (140, 140, 150), opt.reason
            rows.append((f"[{i}] {opt.spec.label}", note, color))
        title = f"TARGET: {focus.display_name}"
        sub = f"{kind} | {net.faction.name.upper()} | {route}"
        return title, sub, rows

    def _focus_panel_rect(self, sim: Simulation) -> tuple[float, float, float, float] | None:
        """Screen rectangle (x, y, w, h) of the focus panel, or None when there is none."""
        if sim.hacking.focus is None or sim.hacking.session is not None:
            return None
        title, sub, rows = self._focus_lines(sim)
        x, y = 24, self.height / 2 - 40
        width = max(
            300.0,
            self.text_width(title, 13) + 24,
            self.text_width(sub, 10) + 24,
            *(FOCUS_NOTE_X + self.text_width(note, 10) + 20 for _, note, _ in rows),
        )
        return (x - 8, y - 8, width, 52 + 18 * len(rows))

    def _draw_focus_panel(self, sim: Simulation) -> None:
        focus = sim.hacking.focus
        x, y = 24, self.height / 2 - 40
        if focus is None:
            if sim.hacking.notice_timer > 0.0:
                self.text(sim.hacking.notice, x, y, size=12, color=(255, 120, 120))
            return
        ctx = self.ctx
        title, sub, rows = self._focus_lines(sim)
        px, py, pw, ph = self._focus_panel_rect(sim)
        # Sized to its content: a long refusal ("LOCKED: ENCRYPTED (NEEDS ...)") used to run
        # out of the fixed 300 px panel into the world labels beside it.
        ctx.rect(px, py, pw, ph, (0.02, 0.06, 0.08, 0.7))
        ctx.rect(px, py, 3, ph, FOCUS_RGBA)
        self.text(title, x, y, size=13, color=(255, 245, 120))
        self.text(sub, x, y + 16, size=10, color=(180, 210, 220))
        yy = y + 36
        for label, note, color in rows:
            self.text(label, x, yy, size=12, color=color)
            self.text(note, x + FOCUS_NOTE_X, yy + 1, size=10, color=color)
            yy += 18
        if sim.hacking.notice_timer > 0.0:
            self.text(sim.hacking.notice, x, yy + 4, size=11, color=(255, 120, 120))

    def _draw_door_prompt(self, sim: Simulation) -> None:
        """An encryption-keyed door nearby: whether the deck holds its cipher."""
        found = sim.keyed_door_in_reach
        if found is None:
            return
        door, has_key = found
        if door.unlocked:
            line, color = f"{door.display_name}: CIPHER ACCEPTED", (120, 255, 160)
        elif has_key:
            line, color = f"{door.display_name}: READING CIPHER...", (255, 230, 120)
        else:
            key = (door.key_intel or "").upper()
            line, color = f"{door.display_name}: ENCRYPTED - NEEDS {key}", (255, 120, 120)
        self.text_centered(line, self.width / 2, self.height / 2 + 40, 13, color)

    def _draw_upload(self, sim: Simulation) -> None:
        up = sim.hacking.upload
        if up is None:
            return
        w, h = 260, 10
        x, y = (self.width - w) / 2, 120
        self.text(
            f"UPLOADING {up.spec.label} -> {up.target.display_name}",
            x, y - 16, size=12, color=(255, 230, 120),
        )  # fmt: skip
        self.meter(
            x, y, w, h, up.progress, (0.12, 0.1, 0.02, 1.0), (1.0, 0.8, 0.2, 1.0), FOCUS_RGBA
        )

    def _draw_network_status(self, sim: Simulation) -> None:
        right, y = self.width - 24, 24
        for net in sim.hacking.networks.values():
            if net.breached:
                line = f"{net.faction.name.upper()} NET: BREACHED via {net.breached_via.upper()}"
                color = _rgb255((*net.faction.color, 1.0))
            else:
                line, color = f"{net.faction.name.upper()} NET: LOCKED", (150, 160, 170)
            self.text_right(line, right, y, 11, color)
            y += 15
        total = sum(1 for t in sim.terminals if t.intel)
        have = len(sim.player.intel)
        intel = f"INTEL {have}/{total}" if total else f"INTEL ON DECK: {have}"
        self.text_right(intel, right, y, 11, (200, 240, 255))
        y += 15
        scanner = sim.scanner
        hint = "[F] PING" if scanner.ready else "PING RECHARGING"
        self.text_right(f"{hint}   [1-9] HACK   [TAB] INVENTORY", right, y, 10, (150, 200, 210))

    def _draw_clock(self, sim: Simulation) -> None:
        """The time of day, top left: a small 24-hour digital clock."""
        clock = getattr(sim, "clock", None)
        if clock is None:
            return
        ctx = self.ctx
        x, y, w, h = 22, 16, 104, 38
        ctx.rect(x, y, w, h, (0.0, 0.03, 0.05, 0.62))
        ctx.rect(x, y, 3, h, CYAN)
        ctx.rect_outline(x, y, w, h, (0.0, 0.55, 0.65, 0.6), 1.0)
        blink = int(sim.time * 2.0) % 2 == 0  # the colon blinks, like any cheap clock's
        text = clock.text if blink else clock.text.replace(":", " ")
        self.text(text, x + 12, y + 6, 24, (120, 240, 255))

    def _draw_log(self, sim: Simulation) -> None:
        lines = list(sim.hacking.log)[-4:]
        y = self.height - 24 - 14 * len(lines)
        for line in lines:
            self.text(line, 24, y, size=10, color=(150, 210, 220))
            y += 14

    def _draw_camera_feed(self, sim: Simulation) -> None:
        ctx, w, h = self.ctx, self.width, self.height
        cam = sim.hijacked_camera
        ctx.rect(0, 0, w, h, (0.0, 0.35, 0.25, 0.10))  # green-tinted feed
        if self.settings.hud_scanlines:
            for y in range(0, h, 6):
                ctx.rect(0, y, w, 1, (0.0, 0.0, 0.0, 0.18))
        frame = (0.6, 1.0, 0.7, 0.9)
        self.brackets(w / 2, h / 2, min(w, h) / 2 - 30, frame, arm=60)
        rec = (1.0, 0.2, 0.2, 1.0) if int(self.time * 2) % 2 == 0 else (0.4, 0.1, 0.1, 1.0)
        name = cam.display_name if cam is not None else "CAMERA"
        label = f"LIVE FEED: {name}"
        lw = self.text_width(label, 13)
        ctx.rect(w / 2 - lw / 2 - 18, 156, 10, 10, rec)
        self.text_centered(label, w / 2, 150, 13, (180, 255, 200))
        feeds = sim.hacking.linked_cameras(cam) if cam is not None else []
        hint = "[Q] DISCONNECT   MOUSE: PAN / TILT"
        if len(feeds) > 1:
            idx = feeds.index(cam) + 1 if cam in feeds else 0
            hint += f"   [<] [>] SWITCH FEED ({idx}/{len(feeds)})"
        self.text_centered(hint, w / 2, 168, 11, (180, 255, 200))

    def _draw_level_title(self, sim: Simulation, dt: float) -> None:
        """The level's name and brief across the top for a few seconds after arriving
        (the band fades out over the last second)."""
        if self.title_time_left <= 0.0 or sim.state.value != "PLAYING":
            return
        fade = min(1.0, self.title_time_left)
        width = self.width
        # Below the detection meter, the link/upload bars and the network status column in
        # the top right corner (the band used to run straight through the latter).
        cx, y = width / 2, TITLE_Y
        self.ctx.rect(0, y - 14, width, 60, (0.0, 0.05, 0.08, 0.55 * fade))
        self.ctx.rect(0, y - 14, width, 2, (0.0, 0.9, 1.0, 0.8 * fade))
        self.ctx.rect(0, y + 44, width, 2, (0.0, 0.9, 1.0, 0.8 * fade))
        self.text_centered(sim.level.name.upper(), cx, y, 22, (120, 240, 255))
        if sim.level.brief:
            self.text_centered(sim.level.brief, cx, y + 28, 12, (200, 230, 235))
        self.title_time_left -= dt
