"""A small immediate-mode UI kit for the game's screens, in the HUD's style: dark glass
panels with a cyan hairline, bright corner brackets and a magenta accent, scanlines,
chromatic-split titles, buttons that glow under the cursor, scroll bars, and text that
wraps or is cut short with an ellipsis to fit its box.

Everything draws through a ``DrawContext`` overlay and the HUD's ``LabelCache`` (text is
rasterised by pygame and kept as textures).
"""

from __future__ import annotations

import math

from cyberduke.render.context import RGBA, DrawContext

RGB = tuple[int, int, int]

CYAN: RGBA = (0.0, 0.95, 1.0, 1.0)
CYAN_DIM: RGBA = (0.0, 0.55, 0.65, 0.7)
MAGENTA: RGBA = (1.0, 0.2, 0.75, 1.0)
YELLOW: RGBA = (1.0, 0.85, 0.2, 1.0)
RED: RGBA = (1.0, 0.2, 0.25, 1.0)
GLASS: RGBA = (0.015, 0.05, 0.075, 0.9)
GLASS_LIGHT: RGBA = (0.03, 0.1, 0.14, 0.92)

T_CYAN: RGB = (120, 240, 255)
T_WHITE: RGB = (235, 245, 250)
T_DIM: RGB = (130, 175, 190)
T_FAINT: RGB = (85, 120, 135)
T_OFF: RGB = (55, 72, 82)  # greyed out: a menu row that cannot be chosen now
T_MAGENTA: RGB = (255, 90, 200)
T_YELLOW: RGB = (255, 220, 90)
T_GREEN: RGB = (110, 255, 170)
T_RED: RGB = (255, 90, 100)

ELLIPSIS = "..."


class Ui:
    def __init__(self, ctx: DrawContext, labels, scanlines: bool = True, draw_text=None) -> None:
        self.ctx = ctx
        self.labels = labels  # hud.LabelCache
        self.scanlines = scanlines
        self.time = 0.0
        # draws one label (text, x, y, size, colour); default: straight from the cache
        self._draw_text = draw_text

    # ------------------------------------------------------------------ text
    def width(self, text: str, size: int) -> int:
        if not text:
            return 0
        return self.labels.font(size).size(text)[0]

    def text(self, text: str, x: float, y: float, size: int = 14, color: RGB = T_WHITE) -> None:
        if not text:
            return
        if self._draw_text is not None:
            self._draw_text(text, x, y, size, color)
            return
        tex, w, h = self.labels.get(text, size, color)
        self.ctx.image(tex, x, y, w, h)

    def text_right(self, text: str, right: float, y: float, size: int, color: RGB) -> None:
        self.text(text, right - self.width(text, size), y, size, color)

    def text_center(self, text: str, cx: float, y: float, size: int, color: RGB) -> None:
        self.text(text, cx - self.width(text, size) / 2, y, size, color)

    def fit(self, text: str, width: float, size: int) -> str:
        """``text`` cut short with an ellipsis so it fits ``width`` pixels."""
        if self.width(text, size) <= width:
            return text
        lo, hi = 0, len(text)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.width(text[:mid].rstrip() + ELLIPSIS, size) <= width:
                lo = mid
            else:
                hi = mid - 1
        return text[:lo].rstrip() + ELLIPSIS

    def wrap(self, text: str, width: float, size: int, max_lines: int | None = None) -> list[str]:
        """Word-wraps ``text`` to ``width`` pixels (a word too long for a line is cut);
        with ``max_lines`` the last line ends in an ellipsis if the text goes on."""
        lines: list[str] = []
        for para in text.split("\n"):
            words = para.split()
            line = ""
            for word in words:
                trial = f"{line} {word}" if line else word
                if self.width(trial, size) <= width:
                    line = trial
                    continue
                if line:
                    lines.append(line)
                line = word if self.width(word, size) <= width else self.fit(word, width, size)
            lines.append(line)
        if max_lines is not None and len(lines) > max_lines:
            rest = " ".join(lines[max_lines - 1 :])
            lines = lines[: max_lines - 1] + [self.fit(rest, width, size)]
        return lines

    def glitch(self, text: str, cx: float, y: float, size: int, color: RGB = T_CYAN) -> None:
        """A title with a chromatic split that jitters now and then."""
        jitter = 2.0 if int(self.time * 7) % 23 == 0 else 0.0
        off = 2.0 + jitter
        self.text_center(text, cx - off, y, size, (255, 40, 170))
        self.text_center(text, cx + off, y + 1, size, (30, 220, 255))
        self.text_center(text, cx, y, size, color)

    # ------------------------------------------------------------------ shapes
    def brackets(self, x: float, y: float, w: float, h: float, rgba: RGBA, arm: float = 14) -> None:
        t = 2.0
        ctx = self.ctx
        for cx, sx in ((x, 1), (x + w, -1)):
            for cy, sy in ((y, 1), (y + h, -1)):
                ctx.rect(min(cx, cx + sx * arm), cy - (t if sy < 0 else 0), arm, t, rgba)
                ctx.rect(cx - (t if sx < 0 else 0), min(cy, cy + sy * arm), t, arm, rgba)

    def panel(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        accent: RGBA = CYAN,
        fill: RGBA = GLASS,
        title: str = "",
        title_color: RGB = T_CYAN,
    ) -> None:
        """Dark glass with a hairline, corner brackets, an accent bar and (``title``) a
        label tab on its top edge."""
        ctx = self.ctx
        ctx.rect(x, y, w, h, fill)
        if self.scanlines:
            for yy in range(int(y) + 2, int(y + h), 4):
                ctx.rect(x, yy, w, 1, (0.0, 0.0, 0.0, 0.12))
        ctx.rect_outline(x, y, w, h, (accent[0] * 0.5, accent[1] * 0.5, accent[2] * 0.5, 0.6), 1.0)
        self.brackets(x, y, w, h, accent)
        ctx.rect(x + 18, y - 1, min(90.0, w * 0.25), 3, MAGENTA)
        if title:
            tw = self.width(title, 12) + 20
            ctx.rect(x + 16, y - 10, tw, 18, (0.0, 0.0, 0.0, 0.95))
            ctx.rect_outline(x + 16, y - 10, tw, 18, accent, 1.0)
            self.text(title, x + 26, y - 8, 12, title_color)

    def button(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        label: str,
        hot: bool = False,
        enabled: bool = True,
        accent: RGBA = CYAN,
        size: int = 14,
    ) -> None:
        ctx = self.ctx
        if not enabled:
            ctx.rect(x, y, w, h, (0.05, 0.06, 0.07, 0.8))
            ctx.rect_outline(x, y, w, h, (0.3, 0.35, 0.4, 0.7), 1.0)
            self.text_center(label, x + w / 2, y + (h - size) / 2 - 2, size, T_FAINT)
            return
        pulse = 0.5 + 0.5 * math.sin(self.time * 6.0) if hot else 0.0
        ctx.rect(
            x,
            y,
            w,
            h,
            (
                accent[0] * 0.15 + 0.2 * pulse * accent[0],
                accent[1] * 0.15 + 0.2 * pulse * accent[1],
                accent[2] * 0.15 + 0.2 * pulse * accent[2],
                0.92,
            ),
        )
        ctx.rect_outline(x, y, w, h, accent, 2.0 if hot else 1.0)
        ctx.rect(x, y + h - 3, w, 3, accent if hot else (accent[0], accent[1], accent[2], 0.5))
        self.text_center(label, x + w / 2, y + (h - size) / 2 - 2, size, T_WHITE if hot else T_CYAN)

    def scrollbar(
        self, x: float, y: float, h: float, total: float, view: float, offset: float
    ) -> None:
        """A thin track with a thumb: ``view`` of ``total`` shows, from ``offset``."""
        if total <= view + 1e-6:
            return
        ctx = self.ctx
        ctx.rect(x, y, 4, h, (0.0, 0.35, 0.45, 0.5))
        thumb = max(20.0, h * view / total)
        top = y + (h - thumb) * (offset / max(total - view, 1e-6))
        ctx.rect(x - 1, top, 6, thumb, CYAN)

    def scanlines_over(self, x: float, y: float, w: float, h: float, alpha: float = 0.15) -> None:
        for yy in range(int(y), int(y + h), 3):
            self.ctx.rect(x, yy, w, 1, (0.0, 0.0, 0.0, alpha))

    def divider(self, x: float, y: float, w: float, rgba: RGBA = CYAN_DIM) -> None:
        self.ctx.rect(x, y, w, 1, rgba)
        self.ctx.rect(x, y - 1, 24, 3, rgba)
