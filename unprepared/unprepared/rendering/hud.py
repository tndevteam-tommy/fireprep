"""2D overlay: status bar, prompts, panels, trade menu and lockpicking mini-game."""
from __future__ import annotations

import math
from typing import Optional

from OpenGL.GL import (GL_BLEND, GL_DEPTH_TEST, GL_EMISSION, GL_FRONT_AND_BACK, GL_LIGHTING, GL_LINES, GL_LINE_LOOP,
                       GL_MODELVIEW, GL_ONE_MINUS_SRC_ALPHA, GL_POINTS, GL_PROJECTION, GL_SRC_ALPHA, GL_TRIANGLES, glBegin,
                       glBlendFunc, glColor3f, glColor4f, glDisable, glEnable, glEnd, glLineWidth, glLoadIdentity,
                       glMaterialfv, glMatrixMode, glOrtho, glPointSize, glPopMatrix, glPushMatrix, glVertex2f)

from ..config import HudColors, HudLayout
from ..core.math3d import view_direction
from ..systems import Side
from .camera import project
from .primitives import draw_textured_rect_2d, fill_rect_2d
from .text import TextRenderer


def wrap_text(text: str, max_width: int, measure) -> list:
    """Greedy word wrap using ``measure(str) -> (w, h)``; over-long words are split."""
    lines, current = [], ""
    for word in text.split(" "):
        candidate = word if not current else f"{current} {word}"
        if measure(candidate)[0] <= max_width:
            current = candidate
            continue
        if current:
            lines.append(current)
        while measure(word)[0] > max_width and len(word) > 1:    # split a word that never fits
            cut = len(word)
            while cut > 1 and measure(word[:cut])[0] > max_width:
                cut -= 1
            lines.append(word[:cut])
            word = word[cut:]
        current = word
    if current or not lines:
        lines.append(current)
    return lines


def clock_text(clock) -> str:
    """'07:35' from the world clock (the hour and the minutes it has accumulated towards the next).
    Display only: the simulation still ticks by the hour."""
    hour = clock.hour
    minutes = int(max(0.0, min(clock.minute_accumulator, clock.minutes_per_hour - 1e-9)) / clock.minutes_per_hour * 60.0)
    return f"{hour:02d}:{minutes:02d}"


def begin_2d(layout: HudLayout) -> None:
    """Switch to a pixel-space orthographic projection with depth/lighting off and blending on
    (the 3D pass's matrices are pushed and restored by :func:`end_2d`)."""
    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    glOrtho(0, layout.width, layout.height, 0, -1, 1)
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()
    glDisable(GL_DEPTH_TEST)
    glDisable(GL_LIGHTING)
    # Reset material emission & colour state left over from the 3D pass.
    glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, [0.0, 0.0, 0.0, 1.0])
    glColor4f(1.0, 1.0, 1.0, 1.0)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)


def end_2d() -> None:
    glDisable(GL_BLEND)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)
    glPopMatrix()


class HudRenderer:
    def __init__(self, text: TextRenderer, layout: HudLayout, colors: HudColors, controls_help: list):
        self.text = text
        self.layout = layout
        self.colors = colors
        self.controls_help = controls_help
        self.fov = 60.0                  # kept in step with the renderer's projection (for speech bubbles)

    # ------------------------------------------------------------------ frame
    def render(self, game) -> None:
        """Layering, back to front: bubbles, status bar, prompt, crosshair, the side panels, the
        log, then any modal screen on top. While a transfer screen or the lock mini-game is up
        the controls and inventory panels are left out (the screen covers them and shows the
        inventory itself) and the modal stops short of the log panel, so neither bleeds through
        the other."""
        begin_2d(self.layout)
        if getattr(game, "in_equipment", False):
            # the equipment view owns the whole window under the status bar: nothing else is drawn
            self._draw_status_bar(game)
            self._draw_equipment_screen(game)
            self._draw_tutorial_hint(game, over_equipment=True)
            self.draw_fade(getattr(game, "fade_alpha", 0.0))
            end_2d()
            return
        modal = game.in_transfer or game.in_lockpicking
        self._draw_speech_bubbles(game)
        self._draw_status_bar(game)
        if game.interaction_prompt and game.in_explore_mode and not getattr(game, "frozen", False):
            # (no "[E] Pickpocket" over the guard who has you by the collar)
            self._draw_prompt(game.interaction_prompt, getattr(game, "interaction_title", None))
        self._draw_crosshair()
        if getattr(game, "reading", None) is not None and game.in_explore_mode:
            self._draw_reading(*game.reading)
        if not modal and getattr(game, "in_tutorial", False):
            self._draw_controls_panel()                   # (only while the tutorial is teaching them)
        if getattr(game, "show_event_log", True):
            self._draw_log_panel(game)
        else:
            self._draw_toasts(game)
        if game.in_transfer:
            self._draw_transfer_menu(game)
        if game.in_lockpicking:
            self._draw_lockpick_minigame(game)
        if getattr(game, "confirmation", None) is not None:
            self._draw_confirmation(game.confirmation)
        self._draw_tutorial_hint(game)
        self.draw_fade(getattr(game, "fade_alpha", 0.0))
        end_2d()

    TUTORIAL_GAP = 10

    TUTORIAL_MIN_WIDTH = 150     # narrower than this beside a modal screen and the panel goes elsewhere

    def _tutorial_panel(self, game, over_equipment: bool = False):
        """(x1, y1, width, height, wrapped lines) of the tutorial panel, or None without a hint.
        Normally directly under the controls panel (a corner nothing else uses, clear of the
        prompt and the speech bubbles), under the status bar in a small window; beside a modal
        screen (a deal, a shelf, the lock) in the strip to its right, level with its top, so
        neither covers the other (or, in a window too small for that, the inventory panel's
        corner, which the modal leaves empty); over the equipment view at the foot of the list
        column, clear of the stand."""
        hint = getattr(game, "guide_hint", None)
        if hint is None:
            hint = getattr(game, "tutorial_hint", "")
        if not hint:
            return None
        t = self.text
        measure = lambda s: t.measure(s, t.font_small)   # noqa: E731
        line_h = measure("Ag")[1] + 3

        def box(x1, width, bottom=None, top=None):
            lines = wrap_text(hint, width - 24, measure)
            h = 14 + line_h * (len(lines) + 1) + 10
            y1 = top if top is not None else bottom - h
            return x1, y1, width, h, lines, line_h

        if over_equipment:
            _, (lx1, ly1, lx2, ly2), _, _ = self._equipment_geometry()
            return box(lx1, lx2 - lx1, bottom=ly2 - 54)
        modal = self.layout.trade_panel if getattr(game, "in_transfer", False) else \
            self.layout.lockpick_panel if getattr(game, "in_lockpicking", False) else None
        if modal is not None:
            x1, width = modal[2] + 12, self.layout.width - 10 - (modal[2] + 12)
            if width >= self.TUTORIAL_MIN_WIDTH:
                panel = box(x1, width, top=modal[1])
                if panel[1] + panel[3] <= self.layout.log_panel[1] - 6:
                    return panel
            ix1, _, ix2, iy2 = self.layout.inventory_panel              # a small window: the inventory panel's
            return box(ix1, ix2 - ix1, bottom=iy2)                      # corner (not drawn behind a modal screen)
        cx1, _, cx2, cy2 = self.layout.controls_panel
        panel = box(cx1, cx2 - cx1, top=cy2 + self.TUTORIAL_GAP)
        if panel[1] + panel[3] > self.layout.log_panel[1] - 6:          # a small window: under the status bar
            panel = box(10, cx1 - 30, top=self.layout.top_bar_height + 10)
        return panel

    def _draw_tutorial_hint(self, game, over_equipment: bool = False) -> None:
        """The tutorial's current instruction in a gold-framed panel (``_tutorial_panel``)."""
        panel = self._tutorial_panel(game, over_equipment)
        if panel is None:
            return
        x1, y1, width, h, lines, line_h = panel
        c, t = self.colors, self.text
        fill_rect_2d(x1 - 2, y1 - 2, x1 + width + 2, y1 + h + 2, (1.0, 0.84, 0.0, 1.0))       # a gold frame
        fill_rect_2d(x1, y1, x1 + width, y1 + h, (0.08, 0.07, 0.04, 0.97))
        t.draw(getattr(game, "guide_title", "") or "TUTORIAL", x1 + 12, y1 + 10, t.font_small, c.gold)
        for idx, line in enumerate(lines):
            t.draw(line, x1 + 12, y1 + 10 + (idx + 1) * line_h, t.font_small, c.white)

    MODAL_LOG_GAP = 8            # pixels a modal screen stops short of the log panel

    def _modal_bottom(self, y2: int) -> int:
        """Where a modal panel ends: its own bottom, or just above the log panel if that is higher."""
        return min(y2, self.layout.log_panel[1] - self.MODAL_LOG_GAP)

    # ------------------------------------------------------------------ speech bubbles
    BUBBLE_MAX_WIDTH = 280
    BUBBLE_HEAD_HEIGHT = 2.05            # world units above an NPC's feet where the bubble points

    def _draw_speech_bubbles(self, game) -> None:
        """Word-wrapped bubbles above the heads of speakers the player can hear (``Game.speech_bubbles``),
        kept inside the window; when two would overlap the later one is nudged upwards."""
        speakers = game.speech_bubbles() if hasattr(game, "speech_bubbles") else []
        if not speakers:
            return
        player = game.player
        eye = player.eye_position
        dx, dy, dz = view_direction(player.rotation_y, player.rotation_x)
        center = (eye.x + dx, eye.y + dy, eye.z + dz)
        w, h = self.layout.width, self.layout.height
        measure = lambda t: self.text.measure(t, self.text.font_small)   # noqa: E731
        placed = []
        items = []
        for npc, speech in speakers:
            head = (npc.position.x, npc.position.y + self.BUBBLE_HEAD_HEIGHT, npc.position.z)
            hit = project(head, (eye.x, eye.y, eye.z), center, self.fov, w / h, w, h)
            if hit is None:
                continue
            items.append((hit[2], hit[0], hit[1], npc, speech))
        items.sort(key=lambda it: -it[0])            # far speakers first, near ones drawn on top
        for _depth, sx, sy, npc, speech in items:
            lines = wrap_text(speech.text, self.BUBBLE_MAX_WIDTH - 24, measure)
            line_h = measure("Ag")[1] + 2
            bw = max(measure(line)[0] for line in [npc.name, *lines]) + 24
            bh = 12 + line_h * (len(lines) + 1)      # the name on its own line, then the text
            x1 = int(max(6, min(w - bw - 6, sx - bw / 2)))
            y1 = int(sy - bh - 14)
            y1 = max(self.layout.top_bar_height + 6, min(h - bh - 6, y1))
            for _ in range(8):                        # slide up out of any bubble already drawn
                if not any(x1 < px2 and x1 + bw > px1 and y1 < py2 and y1 + bh > py1 for px1, py1, px2, py2 in placed):
                    break
                y1 -= bh + 6
            y1 = max(self.layout.top_bar_height + 6, y1)
            placed.append((x1, y1, x1 + bw, y1 + bh))
            fill_rect_2d(x1, y1, x1 + bw, y1 + bh, self.colors.bubble)
            fill_rect_2d(x1 + 2, y1 + 2, x1 + bw - 2, y1 + bh - 2, self.colors.bubble_fill)
            tip_x = int(max(x1 + 12, min(x1 + bw - 12, sx)))
            if y1 + bh + 12 <= sy:                    # a little tail towards the speaker
                glBegin(GL_TRIANGLES)
                glColor4f(*self.colors.bubble_fill)
                glVertex2f(tip_x - 7, y1 + bh - 2); glVertex2f(tip_x + 7, y1 + bh - 2); glVertex2f(tip_x, y1 + bh + 10)
                glEnd()
            self.text.draw(npc.name, x1 + 12, y1 + 6, self.text.font_small, self.colors.bubble_name)
            for idx, line in enumerate(lines):
                self.text.draw(line, x1 + 12, y1 + 6 + line_h * (idx + 1), self.text.font_small, self.colors.bubble_text)

    def draw_fade(self, alpha: float) -> None:
        """Black overlay for scene transitions (0 = clear, 1 = fully black)."""
        if alpha > 0.0:
            fill_rect_2d(0, 0, self.layout.width, self.layout.height, (0.0, 0.0, 0.0, min(1.0, alpha)))

    def _draw_confirmation(self, question) -> None:
        c, t = self.colors, self.text
        w = self.layout.width
        lines = wrap_text(question.text, 460, lambda s: t.measure(s, t.font))
        options = "   ".join(o for o in (question.yes_label, question.alt_label, question.no_label) if o)
        panel_w, panel_h = 520, 70 + 28 * len(lines) + 30
        x1, y1 = (w - panel_w) // 2, self.layout.prompt_y - panel_h // 2 - 40
        fill_rect_2d(x1 - 3, y1 - 3, x1 + panel_w + 3, y1 + panel_h + 3, c.bubble)     # a frame: it sits over whatever is open
        fill_rect_2d(x1, y1, x1 + panel_w, y1 + panel_h, (0.1, 0.09, 0.12, 1.0))
        y = y1 + 20
        for line in lines:
            t.draw(line, x1 + 30, y, t.font, c.gold)
            y += 28
        t.draw(options, x1 + 30, y + 12, t.font_small, c.white)

    # ------------------------------------------------------------------ widgets
    def _draw_status_bar(self, game) -> None:
        c, t, player = self.colors, self.text, game.player
        fill_rect_2d(0, 0, self.layout.width, self.layout.top_bar_height, c.top_bar)
        shown = getattr(game, "show_event_log", True)
        lit, crouched = player.has_lit_torch, player.is_crouching
        clock = clock_text(game.world.clock)
        stance = "Crouching" if crouched else "Standing"
        items = [   # (the full text, the short one for a narrow window, the colour)
            (f"Time: {clock}", clock, c.white),
            (f"Gold: {player.gold:.1f}g", f"{player.gold:.1f}g", c.gold),
            (f"Torch [1]: {'ON' if lit else 'OFF'}", f"Torch: {'ON' if lit else 'OFF'}", c.torch_on if lit else c.white),
            (f"Stance [Ctrl]: {stance}", stance, c.crouch_on if crouched else c.white),
            (f"Events [~]: {'ON' if shown else 'OFF'}", f"Log [~]: {'ON' if shown else 'OFF'}", c.torch_on if shown else c.scroll_hint),
        ]
        for text, x, color in self.status_layout(items):
            t.draw(text, x, 10, t.font_small, color)

    STATUS_X = (20, 160, 280, 460, 700)          # where the status bar's items stand in a window wide enough
    STATUS_GAP = 12

    def status_layout(self, items) -> list:
        """(text, x, colour) of each status item: the usual places; or - when something would
        run off a narrow window or into its neighbour - spread evenly across it, in the short
        forms if the long ones still do not fit."""
        t = self.text
        width = self.layout.width
        full = [t.measure(text, t.font_small)[0] for text, _, _ in items]
        xs = list(self.STATUS_X[:len(items)])
        if (all(x + w + self.STATUS_GAP <= nxt for x, w, nxt in zip(xs, full, xs[1:]))
                and xs[-1] + full[-1] <= width - 10):
            return [(text, x, color) for (text, _, color), x in zip(items, xs)]
        texts = [text for text, _, _ in items]
        widths = full
        if sum(full) + self.STATUS_GAP * (len(items) - 1) > width - 30:
            texts = [short for _, short, _ in items]
            widths = [t.measure(text, t.font_small)[0] for text in texts]
        gap = max(float(self.STATUS_GAP), (width - 30 - sum(widths)) / max(1, len(widths) - 1))
        out, x = [], 20.0
        for text, w, (_, _, color) in zip(texts, widths, items):
            out.append((text, int(x), color))
            x += w + gap
        return out

    def _draw_prompt(self, prompt: str, title: Optional[str] = None) -> None:
        """The [E] prompt under the crosshair - and, over it, what the thing is called (a door's
        name: "Town Gate", "Jae's House") when it has one."""
        c = self.colors
        color = c.danger if "Locked" in prompt else c.prompt
        pw, ph = self.text.measure(prompt, self.text.font)
        px = self.layout.center[0] - (pw // 2)
        py = self.layout.prompt_y
        fill_rect_2d(px - 12, py - 6, px + pw + 12, py + ph + 6, c.prompt_panel)
        self.text.draw(prompt, px, py, self.text.font, color)
        if title:
            tw, th = self.text.measure(title, self.text.font_small)
            tx = self.layout.center[0] - (tw // 2)
            ty = py - 6 - th - 10
            fill_rect_2d(tx - 10, ty - 4, tx + tw + 10, ty + th + 4, c.prompt_panel)
            self.text.draw(title, tx, ty, self.text.font_small, c.prompt_title)

    def _draw_reading(self, readable, lines) -> None:
        """A notice, a signpost, a journal's pages: the words in a panel over the view."""
        c, t = self.colors, self.text
        width = min(self.layout.width - 80, 640)
        wrapped = []
        for line in lines:
            wrapped += self._wrap(line, t.font_small, width - 40) or [""]
        title = readable.label
        th = t.measure(title, t.font)[1]
        lh = t.measure("Ag", t.font_small)[1] + 4
        height = 24 + th + 10 + lh * len(wrapped) + 30
        x1 = (self.layout.width - width) // 2
        y1 = max(self.layout.top_bar_height + 10, self.layout.center[1] - height - 40)
        fill_rect_2d(x1, y1, x1 + width, y1 + height, c.modal_panel)
        t.draw(title, x1 + 20, y1 + 14, t.font, c.gold)
        y = y1 + 24 + th
        for line in wrapped:
            t.draw(line, x1 + 20, y, t.font_small, c.white)
            y += lh
        t.draw("[E] / [Esc] to put it down", x1 + 20, y + 6, t.font_small, c.scroll_hint)

    def _wrap(self, text: str, font, width: int):
        words, lines, line = text.split(), [], ""
        for word in words:
            trial = f"{line} {word}".strip()
            if line and self.text.measure(trial, font)[0] > width:
                lines.append(line)
                line = word
            else:
                line = trial
        if line:
            lines.append(line)
        return lines

    def _draw_crosshair(self) -> None:
        cx, cy = self.layout.center
        glLineWidth(2.0)
        glBegin(GL_LINES)
        glColor3f(1.0, 1.0, 1.0)
        glVertex2f(cx - 6, cy); glVertex2f(cx + 6, cy)
        glVertex2f(cx, cy - 6); glVertex2f(cx, cy + 6)
        glEnd()

    def _draw_controls_panel(self) -> None:
        x1, y1, x2, y2 = self.layout.controls_panel
        y2 = max(y2, y1 + 20 + 22 * len(self.controls_help))     # never clip: grow with the list
        fill_rect_2d(x1, y1, x2, y2, self.colors.panel)
        for idx, line in enumerate(self.controls_help):
            color = self.colors.gold if idx == 0 else self.colors.white
            self.text.draw(line, x1 + 10, y1 + 10 + idx * 22, self.text.font_small, color)

    def _fit_text(self, text: str, max_width: int, font) -> str:
        """``text`` cut (with an ellipsis) to ``max_width`` pixels so it never runs into whatever
        stands to its right."""
        if self.text.measure(text, font)[0] <= max_width:
            return text
        cut = len(text)
        while cut > 1 and self.text.measure(text[:cut] + "...", font)[0] > max_width:
            cut -= 1
        return text[:cut].rstrip() + "..."

    TAG_COLORS = {"worn": "crouch_on", "theft": "stolen_tag", "free": "scroll_hint"}

    def _draw_item_text(self, text: str, x: float, y: float, item, color, font=None, prefix: str = "",
                        max_width: Optional[int] = None, tag=None) -> None:
        """Item row: an optional cursor ``prefix`` first, then - for stolen items - a bold red '(S)'
        tag and red text; ``tag`` = (text, kind) is drawn at the row's right edge (``x + max_width``)
        and the text is cut short rather than run into it."""
        font = font or self.text.font_small
        right = x + max_width if max_width is not None else None
        if prefix:
            w, _ = self.text.draw(prefix, x, y, font, color)
            x += w
        if item.is_stolen:
            w, _ = self.text.draw("(S) ", x, y, self.text.font_tag, self.colors.stolen_tag)
            x += w
            color = self.colors.danger
        if tag is not None and right is not None:
            label, kind = tag
            tag_font = self.text.font_tag if kind == "theft" else font
            tw = self.text.measure(label, tag_font)[0]
            self.text.draw(label, right - tw, y, tag_font, getattr(self.colors, self.TAG_COLORS.get(kind, "white")))
            right -= tw + 10
        if right is not None:
            text = self._fit_text(text, int(right - x), font)
        self.text.draw(text, x, y, font, color)

    def _item_color(self, item, selected: bool = False):
        """Selected rows in the highlight colour; otherwise an item's name shows how rare it is."""
        if selected:
            return self.colors.selected
        return getattr(self.colors, f"rarity_{getattr(item, 'rarity', 'common')}", self.colors.white)

    TOAST_FADE = 1.0             # seconds a pop-up takes to fade away at the end of its time

    def _draw_toasts(self, game) -> None:
        """With the World Events log hidden, the lines that concern the player (not the town's
        routine news) pop up over the bottom of the view for a few seconds, newest at the bottom."""
        toasts = getattr(game, "toasts", ())
        if not toasts:
            return
        t = self.text
        now = getattr(game, "ui_time", 0.0)
        measure = lambda s: t.measure(s, t.font_small)   # noqa: E731
        width = min(self.layout.width - 80, 760)
        line_h = measure("Ag")[1] + 3
        y = self.layout.height - 24
        for text, until in reversed(list(toasts)):
            lines = wrap_text(text, width - 24, measure)
            h = 10 + line_h * len(lines)
            alpha = max(0.0, min(1.0, (until - now) / self.TOAST_FADE))
            w = max(measure(line)[0] for line in lines) + 24
            x1 = (self.layout.width - w) // 2
            y1 = y - h
            panel = self.colors.panel
            fill_rect_2d(x1, y1, x1 + w, y, (panel[0], panel[1], panel[2], panel[3] * alpha))
            for k, line in enumerate(lines):
                t.draw(line, x1 + 12, y1 + 5 + k * line_h, t.font_small, self.colors.white, alpha=alpha)
            y = y1 - 6

    def _draw_log_panel(self, game) -> None:
        x1, y1, x2, y2 = self.layout.log_panel
        c = self.colors
        fill_rect_2d(x1, y1, x2, y2, c.panel)
        self.text.draw("WORLD DIALOGUE & EVENTS:", x1 + 10, y1 + 10, self.text.font_small, c.log_title)
        width = x2 - x1 - 20
        measure = lambda t: self.text.measure(t, self.text.font_small)   # noqa: E731
        lines = []
        for msg in game.messages.messages:
            lines.extend(wrap_text(msg, width, measure))
        for idx, line in enumerate(lines[-self.layout.max_log_rows:]):
            self.text.draw(line, x1 + 10, y1 + 40 + idx * self.layout.log_line_height, self.text.font_small, c.white)

    # ------------------------------------------------------------------ the [Tab] equipment view
    EQUIP_ROW_H = 24
    EQUIP_LIST_W = 440
    EQUIP_SLOTS_W = 190

    def _equipment_geometry(self):
        """Pixel rectangles of the equipment view: (panel, list column, preview frame, slots column)."""
        px1, py1, px2, py2 = self.layout.equipment_panel
        inner_w = px2 - px1
        list_x1 = px1 + 20
        list_w = min(self.EQUIP_LIST_W, int(inner_w * 0.42))
        slots_x1 = px2 - self.EQUIP_SLOTS_W
        top, bottom = py1 + 55, py2 - 60
        preview = (list_x1 + list_w + 20, top, slots_x1 - 20, bottom)
        return (px1, py1, px2, py2), (list_x1, top, list_x1 + list_w, bottom), preview, (slots_x1, top, px2 - 20, bottom)

    def equipment_preview_rect(self):
        """Where the renderer draws the player's figure (pixels, y down) while the view is open."""
        _, _, preview, _ = self._equipment_geometry()
        x1, y1, x2, y2 = preview
        return (x1 + 3, y1 + 3, x2 - 3, y2 - 3)        # inside the frame's border

    def _draw_equipment_screen(self, game) -> None:
        """The inventory with a cursor on the left, the figure's stand in the middle, what is worn
        in each slot on the right. [Enter] puts the selected item on or takes it off."""
        from ..entities.equipment import SLOT_ORDER, describe_clothing, slot_of
        c, t = self.colors, self.text
        (px1, py1, px2, py2), (lx1, ly1, lx2, ly2), (vx1, vy1, vx2, vy2), (sx1, sy1, sx2, sy2) = self._equipment_geometry()
        fill_rect_2d(px1, py1, px2, py2, c.modal_panel)
        t.draw("EQUIPMENT", px1 + 20, py1 + 14, t.font, c.gold)
        t.draw("[UP/DOWN] select   [ENTER] put on / take off   [LEFT/RIGHT] turn   [TAB/ESC] close",
               px1 + 20, py2 - 36, t.font_small, c.scroll_hint)
        # the inventory column (shorter while the tutorial's panel sits at its foot)
        list_bottom = ly2
        tutorial_panel = self._tutorial_panel(game, over_equipment=True)
        if tutorial_panel is not None:
            list_bottom = min(ly2, tutorial_panel[1] - 4)
        rows_shown = max(1, (list_bottom - ly1 - 30) // self.EQUIP_ROW_H)
        first, items = game.equipment_visible(rows_shown)
        total = len(game.player.inventory)
        title = "INVENTORY" if total <= rows_shown else f"INVENTORY ({first + 1}-{first + len(items)} of {total})"
        t.draw(title, lx1, ly1, t.font_small, c.trade_heading)
        if first > 0:
            t.draw(f"^ {first} more", lx2 - 80, ly1, t.font_small, c.scroll_hint)
        if not items:
            t.draw("(nothing in your pack)", lx1 + 10, ly1 + 30, t.font_small, c.scroll_hint)
        for idx, item in enumerate(items):
            y = ly1 + 30 + idx * self.EQUIP_ROW_H
            selected = first + idx == game.equipment_cursor
            slot = slot_of(item)
            color = self._item_color(item, selected)
            tag = ("worn", "worn") if item.equipped else ((slot.label.lower(), "free") if slot is not None else None)
            self._draw_item_text(f"{item.display_name} ({item.value:g}g)", lx1, y, item, color, prefix="> " if selected else "  ",
                                 max_width=lx2 - lx1, tag=tag)
        if first + len(items) < total:
            t.draw(f"v {total - first - len(items)} more", lx2 - 80, list_bottom - 22, t.font_small, c.scroll_hint)
        selected_item = game.equipment_selection
        if selected_item is not None:                                        # what the thing under the cursor is like
            note = describe_clothing(selected_item, game.settings.clothing)
            rarity = getattr(selected_item, "rarity", "common")
            bonus = getattr(selected_item, "bonus_text", "")
            if rarity != "common" or bonus:
                line = f"{rarity.capitalize()}" + (f": {bonus} while worn" if bonus else "")
                t.draw(self._fit_text(line, lx2 - lx1, t.font_small), lx1, ly2 - 68, t.font_small, self._item_color(selected_item))
            if note:
                t.draw(self._fit_text(f"{selected_item.name}: {note}", lx2 - lx1, t.font_small), lx1, ly2 - 46, t.font_small, c.scroll_hint)
        # the stand: a framed, dark backdrop the renderer paints the figure into afterwards
        fill_rect_2d(vx1, vy1, vx2, vy2, c.bubble)
        fill_rect_2d(vx1 + 3, vy1 + 3, vx2 - 3, vy2 - 3, (0.16, 0.17, 0.22, 1.0))
        # what is worn, slot by slot
        t.draw("WORN", sx1, sy1, t.font_small, c.trade_heading)
        worn = game.player.equipment
        selected_slot = slot_of(game.equipment_selection) if game.equipment_selection is not None else None
        for idx, slot in enumerate(SLOT_ORDER):
            y = sy1 + 30 + idx * 46
            label_color = c.selected if slot is selected_slot else c.scroll_hint
            t.draw(slot.label, sx1, y, t.font_small, label_color)
            item = worn.get(slot)
            if item is None:
                t.draw("- nothing -", sx1 + 10, y + 20, t.font_small, c.scroll_hint)
            else:
                self._draw_item_text(item.name, sx1 + 10, y + 20, item, self._item_color(item), max_width=sx2 - sx1 - 10)
        # skills: the level earned, and what the gear worn adds to it
        skills_y = sy1 + 30 + len(SLOT_ORDER) * 46 + 4
        if skills_y + 22 * 7 <= sy2 - 100:
            t.draw("SKILLS (+ worn)", sx1, skills_y, t.font_small, c.trade_heading)
            player = game.player
            for idx, (skill, _) in enumerate(player.skills.skills.items()):
                base = player.skills.level(skill)
                bonus = player.skill_bonus(skill) if hasattr(player, "skill_bonus") else 0
                text = f"{skill.value}: {base}" + (f"  +{bonus} = {base + bonus}" if bonus else "")
                t.draw(self._fit_text(text, sx2 - sx1, t.font_small), sx1 + 10, skills_y + 22 * (idx + 1), t.font_small,
                       c.crouch_on if bonus else c.white)
        seen, noise = game.player.visibility_factor(), game.player.clothing_noise_factor()
        t.draw("In the shadows", sx1, sy2 - 96, t.font_small, c.trade_heading)
        t.draw(f"Seen: {seen:.2f}x", sx1, sy2 - 70, t.font_small, c.crouch_on if seen < 1.0 else c.danger if seen > 1.0 else c.white)
        t.draw(f"Heard: {noise:.2f}x", sx1, sy2 - 46, t.font_small, c.crouch_on if noise < 1.0 else c.danger if noise > 1.0 else c.white)
        t.draw("steel glints; cloth hides", sx1, sy2 - 22, t.font_small, c.scroll_hint)

    def _draw_transfer_menu(self, game) -> None:
        """Two-column item exchange (trade / pickpocket / container); Tab switches the active column."""
        x1, y1, x2, y2 = self.layout.trade_panel
        y2 = self._modal_bottom(y2)
        c, t, session = self.colors, self.text, game.transfer
        policy = session.policy
        fill_rect_2d(x1, y1, x2, y2, c.modal_panel)

        t.draw(policy.title, x1 + 28, y1 + 16, t.font, c.gold)
        t.draw(policy.help_text, x1 + 28, y1 + 51, t.font_small, c.white)

        row_h = 25
        rows_y = y1 + 140                      # headings at +96, scroll hints on their own line at +116
        session.visible_rows = max(1, (y2 - 20 - rows_y) // row_h)
        col_w = 280
        for side, col_x in ((Side.LEFT, x1 + 28), (Side.RIGHT, x1 + 308)):
            active = session.side is side
            label = policy.left_label if side is Side.LEFT else policy.right_label
            t.draw(("> " if active else "  ") + label, col_x, y1 + 96, t.font_small, c.selected if active else c.trade_heading)
            first, rows = session.visible(side)
            total = len(session.rows(side))
            for idx, item in enumerate(rows):
                selected = active and session.selection[side] == first + idx
                self._draw_item_text(session.row_text(item, side), col_x + 10, rows_y + idx * row_h, item,
                                     self._item_color(item, selected), prefix="> " if selected else "  ",
                                     max_width=col_w - 10, tag=session.row_tag(item, side))
            # scroll hints get their own line under the heading, flush with the column's right edge
            if first > 0:
                hint = f"^ {first} more"
                t.draw(hint, col_x + col_w - t.measure(hint, t.font_small)[0], y1 + 116, t.font_small, c.scroll_hint)
            if first + len(rows) < total:
                hint = f"v {total - first - len(rows)} more"
                t.draw(hint, col_x + col_w - t.measure(hint, t.font_small)[0], y2 - 22, t.font_small, c.scroll_hint)

    def _draw_lockpick_minigame(self, game) -> None:
        x1, y1, x2, y2 = self.layout.lockpick_panel
        y2 = self._modal_bottom(y2)
        c, t, lp = self.colors, self.text, game.lockpicking
        fill_rect_2d(x1, y1, x2, y2, c.lockpick_panel)

        t.draw("LOCKPICKING MINI-GAME", x1 + 53, y1 + 16, t.font, c.gold)
        t.draw("Rotate pick with [LEFT / RIGHT] arrows", x1 + 23, y1 + 56, t.font_small, c.white)
        t.draw("Press [SPACE] when the pick is on the spot", x1 + 33, y1 + 86, t.font_small, c.white)
        if lp.wobble_amplitude > 0:
            t.draw(f"Unsteady hands: the pick wanders {lp.wobble_amplitude:.0f} deg", x1 + 43, y1 + 106, t.font_small, c.scroll_hint)
        tier = getattr(lp, "tier_name", "")
        if tier:
            odds = getattr(lp, "set_chance", 1.0)
            color = c.crouch_on if odds >= 0.6 else c.gold if odds >= 0.25 else c.danger
            t.draw(f"{tier} lock - the pins set {odds:.0%} of the time at your skill", x1 + 23, y1 + 126, t.font_small, color)

        center_x, center_y = self.layout.lockpick_center
        radius = self.layout.lockpick_radius
        is_aligned = lp.is_aligned
        glow = (1.0, 0.9, 0.1) if is_aligned else (0.4, 0.4, 0.5)

        # Lock cylinder ring
        glLineWidth(6.0 if is_aligned else 3.0)
        glBegin(GL_LINE_LOOP)
        for i in range(36):
            rad = math.radians(i * 10)
            glColor3f(*glow)
            glVertex2f(center_x + math.cos(rad) * radius, center_y + math.sin(rad) * radius)
        glEnd()

        # Sweet spot marker
        sweet_rad = math.radians(lp.sweet_spot)
        glPointSize(10.0)
        glBegin(GL_POINTS)
        glColor3f(1.0, 0.8, 0.0)
        glVertex2f(center_x + math.cos(sweet_rad) * radius, center_y + math.sin(sweet_rad) * radius)
        glEnd()

        # Pick needle (drawn where the unsteady hand actually holds it)
        glLineWidth(5.0)
        glBegin(GL_LINES)
        rad = math.radians(lp.effective_angle)
        glColor3f(*((0.9, 0.6, 0.1) if is_aligned else (0.8, 0.8, 0.8)))
        glVertex2f(center_x, center_y)
        glVertex2f(center_x + math.cos(rad) * radius, center_y + math.sin(rad) * radius)
        glEnd()

        if is_aligned:
            t.draw("SWEET SPOT FOUND! Press [SPACE]!", x1 + 53, min(y1 + 351, y2 - 26), t.font_small, c.sweet_spot_text)


# ----------------------------------------------------------------------------- menus & overlays
class MenuRenderer:
    """Draws ``ui.Menu`` screens and the performance overlay with the HUD's text renderer."""

    def __init__(self, text: TextRenderer, layout: HudLayout, colors: HudColors, textures=None):
        self.text = text
        self.layout = layout
        self.colors = colors
        self.textures = textures            # (for the start screen's logo)

    PANEL_W = 560

    def subtitle_lines(self, menu) -> list:
        """The subtitle wrapped to the panel (a long file name in an error must not run off it)."""
        if not menu.subtitle:
            return []
        t = self.text
        return wrap_text(menu.subtitle, self.PANEL_W - 60, lambda s: t.measure(s, t.font_small))

    def panel_height(self, menu, heading: bool = True) -> int:
        _, rows = menu.visible()
        sub = len(self.subtitle_lines(menu))
        return (120 if heading else 70) + 34 * max(1, len(rows)) + (30 + 22 * (sub - 1) if sub else 0)

    LOGO_ASPECT = 900.0 / 440.0             # the logo artwork's width over its height
    LOGO_MAX_WIDTH = 700

    def draw_title(self, menu) -> None:
        """The start screen over the live town: the logo at the top (as large as the window leaves
        room for above the menu), the menu under it on a translucent panel."""
        w, h = self.layout.width, self.layout.height
        logo_h = 0
        if self.textures is not None and "logo" in self.textures:
            room = h - self.panel_height(menu, heading=False) - 40
            logo_w = min(self.LOGO_MAX_WIDTH, w - 80, int(room * self.LOGO_ASPECT))
            logo_h = int(logo_w / self.LOGO_ASPECT)
            if logo_h >= 60:
                begin_2d(self.layout)
                fill_rect_2d(0, 0, w, h, (0.0, 0.0, 0.05, 0.18))            # (a little dusk over the picture)
                draw_textured_rect_2d(self.textures["logo"], (w - logo_w) // 2, 12, (w + logo_w) // 2, 12 + logo_h)
                end_2d()
            else:
                logo_h = 0
        self.draw_menu(menu, dim=False, top=(12 + logo_h + 8) if logo_h else None, alpha=0.82, heading=not logo_h)

    def draw_menu(self, menu, dim: bool = True, top=None, alpha: float = 1.0, heading: bool = True) -> None:
        """A menu on its panel (``heading`` False: without its title - the start screen's logo says
        it all; a subtitle there is news, such as a save that would not load)."""
        c, t = self.colors, self.text
        w, h = self.layout.width, self.layout.height
        begin_2d(self.layout)
        if dim:
            fill_rect_2d(0, 0, w, h, (0.0, 0.0, 0.0, 0.55))
        first, rows = menu.visible()
        total = len(menu.items)
        panel_w, panel_h = self.PANEL_W, self.panel_height(menu, heading)
        x1, y1 = (w - panel_w) // 2, (top if top is not None else max(40, (h - panel_h) // 2))
        fill_rect_2d(x1, y1, x1 + panel_w, y1 + panel_h, (*c.modal_panel[:3], c.modal_panel[3] * alpha))
        if heading:
            t.draw(menu.title, x1 + 30, y1 + 22, t.font, c.gold)
        if first > 0:
            t.draw(f"^ {first} more", x1 + panel_w - 110, y1 + 26, t.font_small, c.scroll_hint)
        y = y1 + (62 if heading else 22)
        subtitle = self.subtitle_lines(menu)
        for k, line in enumerate(subtitle):
            t.draw(line, x1 + 30, y, t.font_small, c.scroll_hint if heading else c.white)
            y += 22 if k < len(subtitle) - 1 else 30
        for offset, item in enumerate(rows):
            idx = first + offset
            selected = idx == menu.cursor
            color = c.selected if selected else (c.white if item.enabled() else c.scroll_hint)
            t.draw(("> " if selected else "  ") + item.label, x1 + 30, y, t.font_small, color)
            value = item.display_value()
            if value:
                t.draw(value, x1 + 300, y, t.font_small, color)
            y += 34
        if first + len(rows) < total:
            t.draw(f"v {total - first - len(rows)} more", x1 + panel_w - 110, y1 + panel_h - 30, t.font_small, c.scroll_hint)
        t.draw(menu.footer, x1 + 30, y1 + panel_h - 30, t.font_small, c.scroll_hint)
        end_2d()

    BUSY_W, BUSY_H = 420, 96

    def draw_busy(self, label: str, progress: Optional[float] = None, detail: str = "") -> None:
        """The box shown while something long is under way (days waited out in the cells, a
        night's sleep, the town being built): what it is, a line of detail and - when it can be
        told - a bar filling with its progress (0 .. 1); without one, a bar of light sweeping
        along it to show the game has not hung."""
        import time as _time
        c, t = self.colors, self.text
        w, h = self.layout.width, self.layout.height
        bw, bh = min(self.BUSY_W, w - 40), self.BUSY_H
        x1, y1 = (w - bw) // 2, (h - bh) // 2
        begin_2d(self.layout)
        fill_rect_2d(x1 - 3, y1 - 3, x1 + bw + 3, y1 + bh + 3, c.bubble)
        fill_rect_2d(x1, y1, x1 + bw, y1 + bh, (0.08, 0.07, 0.1, 0.96))
        t.draw(label + "...", x1 + 20, y1 + 14, t.font, c.gold)
        if detail:
            t.draw(detail, x1 + 20, y1 + 44, t.font_small, c.white)
        bx1, by1, bx2, by2 = x1 + 20, y1 + bh - 22, x1 + bw - 20, y1 + bh - 12
        fill_rect_2d(bx1, by1, bx2, by2, (0.25, 0.22, 0.2, 1.0))
        if progress is not None:
            fill_rect_2d(bx1, by1, bx1 + (bx2 - bx1) * max(0.0, min(1.0, progress)), by2, (0.85, 0.7, 0.3, 1.0))
        else:
            span = (bx2 - bx1) * 0.25
            phase = (_time.perf_counter() * 0.8) % 1.0
            a = bx1 + (bx2 - bx1 - span) * phase
            fill_rect_2d(a, by1, a + span, by2, (0.85, 0.7, 0.3, 1.0))
        end_2d()

    def draw_perf_overlay(self, profiler, lines_extra=()) -> None:
        c, t = self.colors, self.text
        begin_2d(self.layout)
        x1, y1, x2 = 360, 55, 710
        stats = profiler.stats()
        rows = [f"FPS {stats['fps']:.0f}   frame {stats['frame_ms']:.1f} ms   worst {stats['worst_ms']:.1f} ms"]
        rows += [f"  {name:<10} {ms:6.2f} ms" for name, ms in stats["phases"].items()]
        rows += list(lines_extra)
        height = 40 + 18 * len(rows) + 50
        fill_rect_2d(x1, y1, x2, y1 + height, (0.0, 0.0, 0.0, 0.7))
        t.draw("PERFORMANCE  [F3]", x1 + 10, y1 + 8, t.font_small, c.gold)
        for idx, row in enumerate(rows):
            t.draw(row, x1 + 10, y1 + 30 + idx * 18, t.font_small, c.white)
        # frame-time graph: one bar per frame, 33 ms full height
        gx, gy, gh = x1 + 10, y1 + height - 10, 40
        history = list(profiler.frame_history)[-(x2 - x1 - 20):]
        glLineWidth(1.0)
        glBegin(GL_LINES)
        for i, ms in enumerate(history):
            bar = min(gh, gh * ms / 33.3)
            glColor3f(0.3, 0.9, 0.3) if ms <= 16.7 else glColor3f(0.95, 0.6, 0.2) if ms <= 33.3 else glColor3f(0.95, 0.2, 0.2)
            glVertex2f(gx + i, gy)
            glVertex2f(gx + i, gy - bar)
        glEnd()
        end_2d()
