"""Draws the controller screen into a pygame Surface (software; uploaded to GL by render/menu.py).

The surface is opaque (anti-aliased edges blend correctly only on an opaque
surface); the overlay's translucency over the game is applied uniformly when it
is drawn (`OVERLAY_ALPHA`).

The picture is live: every button lights up while it is held, the triggers fill
as they are pulled and the stick caps follow the sticks (with the deadzone drawn
inside the well).  Each control has a label with the keyboard key it is bound to
and what that key does.
"""
import math

from core.gamepad import Family, FACE_COLORS, PAUSE_CONTROL, control_label, TRIGGERS
from core.controller_menu import OPTIONS, OPTION_LABELS, SLIDERS, TOGGLES
from core.keymap import key_display_name, key_action_label

try:
    import pygame
    import pygame.gfxdraw as gfx
except ImportError:      # pragma: no cover
    pygame = None

OVERLAY_ALPHA = 0.93

# palette
C_BG = (10, 12, 16)
C_BODY = (44, 48, 56)
C_BODY_EDGE = (92, 98, 112)
C_BODY_SHADE = (36, 39, 46)
C_CTRL = (66, 71, 82)
C_CTRL_EDGE = (118, 124, 138)
C_TEXT = (225, 228, 234)
C_DIM = (150, 156, 166)
C_FAINT = (96, 102, 114)
C_HOT = (255, 160, 40)
C_FOCUS = (0, 220, 255)
C_OK = (120, 220, 140)
C_WARN = (255, 196, 90)
C_BOX = (22, 25, 31)
C_BOX_EDGE = (58, 63, 74)
C_LEADER = (78, 84, 96)


def _mix(a, b, t):
    return tuple(int(round(a[i] + (b[i] - a[i]) * t)) for i in range(3))


class ControllerView:
    def __init__(self, model):
        self.model = model
        self.fonts = {}
        self.text_cache = {}
        self.surface = None
        self._static_key = None
        self._static = None
        self.time = 0.0

    # ----------------------------------------------------------------- text
    def font(self, size):
        size = max(8, int(round(size)))
        f = self.fonts.get(size)
        if f is None:
            if not pygame.font.get_init():
                pygame.font.init()
            f = self.fonts[size] = pygame.font.Font(None, size)
        return f

    def text(self, s, size, color):
        key = (s, int(round(size)), tuple(color))
        surf = self.text_cache.get(key)
        if surf is None:
            if len(self.text_cache) > 600:
                self.text_cache.clear()
            surf = self.text_cache[key] = self.font(size).render(s, True, color)
        return surf

    def blit_text(self, dst, s, size, color, pos, anchor="topleft", max_w=None):
        surf = self.text(s, size, color)
        while max_w is not None and surf.get_width() > max_w and size > 9:
            # shrink long strings until they fit their box
            size = max(9, min(size - 1, size * max_w / surf.get_width()))
            surf = self.text(s, size, color)
        rect = surf.get_rect(**{anchor: (int(round(pos[0])), int(round(pos[1])))})
        dst.blit(surf, rect)
        return rect

    # ------------------------------------------------------------ primitives
    @staticmethod
    def aacircle(dst, center, r, color, width=0):
        x, y, r = int(round(center[0])), int(round(center[1])), int(round(r))
        if r <= 0:
            return
        if width == 0:
            gfx.filled_circle(dst, x, y, r, color)
            gfx.aacircle(dst, x, y, r, color)
        else:
            for i in range(int(width)):
                gfx.aacircle(dst, x, y, r - i, color)

    @staticmethod
    def blend_rect(dst, rect, rgba, radius):
        """Translucent rounded rectangle (alpha-blended onto the opaque surface)."""
        x, y, w, h = [int(round(v)) for v in rect]
        if w <= 0 or h <= 0:
            return
        tmp = pygame.Surface((w, h), pygame.SRCALPHA)
        pygame.draw.rect(tmp, rgba, tmp.get_rect(), border_radius=int(round(radius)))
        dst.blit(tmp, (x, y))

    @staticmethod
    def blend_circle(dst, center, r, rgba):
        r = int(round(r))
        if r <= 0:
            return
        tmp = pygame.Surface((2 * r + 2, 2 * r + 2), pygame.SRCALPHA)
        # pygame.draw writes the RGBA as-is (gfxdraw would blend it into the transparent surface)
        pygame.draw.circle(tmp, rgba, (r + 1, r + 1), r)
        dst.blit(tmp, (int(round(center[0])) - r - 1, int(round(center[1])) - r - 1))

    @staticmethod
    def rrect(dst, rect, color, radius, width=0):
        r = pygame.Rect([int(round(v)) for v in rect])
        pygame.draw.rect(dst, color, r, width, border_radius=int(round(radius)))

    @staticmethod
    def ellipse_points(center, radii, angle, n=48):
        ca, sa = math.cos(angle), math.sin(angle)
        pts = []
        for i in range(n):
            t = 2.0 * math.pi * i / n
            x, y = radii[0] * math.cos(t), radii[1] * math.sin(t)
            pts.append((center[0] + x * ca - y * sa, center[1] + x * sa + y * ca))
        return pts

    # ----------------------------------------------------------------- static
    def _draw_static(self, w, h):
        L = self.model.layout
        surf = pygame.Surface((w, h))
        surf.fill(C_BG)
        s = L.s
        # controller body: an outline pass (slightly larger, edge colour) then the fill
        for grow, color in ((3.0 * s, C_BODY_EDGE), (0.0, C_BODY)):
            x, y, bw, bh = L.body_rect
            self.rrect(surf, (x - grow, y - grow, bw + 2 * grow, bh + 2 * grow), color, L.body_radius + grow)
            for center, radii, ang in L.grips:
                pts = self.ellipse_points(center, (radii[0] + grow, radii[1] + grow), ang)
                gfx.filled_polygon(surf, pts, color)
                gfx.aapolygon(surf, pts, color)
        # a soft top highlight on the body
        x, y, bw, bh = L.body_rect
        hl = pygame.Surface((int(bw), int(bh * 0.45)), pygame.SRCALPHA)
        pygame.draw.rect(hl, (255, 255, 255, 10), hl.get_rect(), border_radius=int(L.body_radius))
        surf.blit(hl, (x, y))
        # settings panel frame
        self.rrect(surf, L.panel_rect, (18, 21, 27), 12 * s)
        self.rrect(surf, L.panel_rect, C_BOX_EDGE, 12 * s, 1)
        self.blit_text(surf, "SETTINGS", 30 * s, C_DIM, (L.panel_rect[0] + 16 * s, L.panel_rect[1] + 16 * s))
        self.blit_text(surf, "CONTROLLER", 64 * s, (255, 255, 255), L.title_pos)
        return surf

    # ------------------------------------------------------------------ draw
    def draw(self, pad_state=None, dt=0.0, size=None):
        """Render the whole screen; returns the Surface."""
        m = self.model
        L = m.layout
        self.time += dt
        w, h = size or (int(L.screen_w), int(L.screen_h))
        key = (m.family, w, h)
        if self._static_key != key:
            self._static = self._draw_static(w, h)
            self._static_key = key
        if self.surface is None or self.surface.get_size() != (w, h):
            self.surface = pygame.Surface((w, h))
        surf = self.surface
        surf.blit(self._static, (0, 0))

        self._draw_device_line(surf)
        self._draw_leaders(surf, pad_state)
        self._draw_controls(surf, pad_state)
        self._draw_callouts(surf, pad_state)
        self._draw_options(surf)
        self._draw_status(surf)
        return surf

    # ----------------------------------------------------------- states
    def _pressed(self, control, pad_state):
        return pad_state is not None and pad_state.is_down(control)

    def _pulse(self):
        return 0.5 + 0.5 * math.sin(self.time * 7.0)

    def _ctrl_colors(self, control, pad_state, base=C_CTRL):
        m = self.model
        pressed = self._pressed(control, pad_state)
        fill = C_HOT if pressed else base
        edge = C_CTRL_EDGE
        if m.waiting == control:
            edge = _mix(C_HOT, (255, 255, 255), self._pulse() * 0.6)
        elif m.focus == ("control", control):
            edge = C_FOCUS
        elif pressed:
            edge = (255, 210, 140)
        return fill, edge, pressed

    # ------------------------------------------------------------ sections
    def _draw_device_line(self, surf):
        m, L = self.model, self.model.layout
        s = L.s
        x, y = L.device_pos
        if m.device_name:
            self.aacircle(surf, (x + 7 * s, y + 10 * s), 6 * s, C_OK)
            family = {Family.XBOX: "Xbox", Family.PLAYSTATION: "PlayStation",
                      Family.NINTENDO: "Nintendo", Family.GENERIC: "generic"}[m.family]
            if m.device_recognised:
                self.blit_text(surf, f"{m.device_name}   -   {family} layout", 28 * s, C_TEXT, (x + 22 * s, y),
                               max_w=820 * s)
            else:
                r = self.blit_text(surf, m.device_name, 28 * s, C_TEXT, (x + 22 * s, y), max_w=360 * s)
                self.blit_text(surf, "unrecognised pad: its buttons are guessed - if another one lights up, "
                                     "just bind that one", 20 * s, C_WARN, (r.right + 14 * s, y + 5 * s),
                               max_w=840 * s - r.width)
        else:
            self.aacircle(surf, (x + 7 * s, y + 10 * s), 6 * s, C_FAINT, 2)
            self.blit_text(surf, "No controller connected - plug one in and it appears here (bindings can still be edited)",
                           26 * s, C_DIM, (x + 22 * s, y), max_w=840 * s)

    def _draw_leaders(self, surf, pad_state):
        m, L = self.model, self.model.layout
        for control, entry in L.callouts.items():
            rx, ry, rw, rh = entry["rect"]
            start = (rx + rw, ry + rh * 0.5) if entry["side"] == 0 else (rx, ry + rh * 0.5)
            end = L.anchor(control)
            color = C_LEADER
            if self._pressed(control, pad_state):
                color = C_HOT
            elif m.waiting == control or m.focus == ("control", control):
                color = C_FOCUS
            # elbow: horizontal out of the box, then straight to the control
            elbow = (start[0] + (22 if entry["side"] == 0 else -22) * L.s, start[1])
            pygame.draw.aaline(surf, color, start, elbow)
            pygame.draw.aaline(surf, color, elbow, end)
            self.aacircle(surf, end, 2.5 * L.s, color)

    def _draw_controls(self, surf, pad_state):
        m, L = self.model, self.model.layout
        s = L.s
        fam = m.family
        face_colors = FACE_COLORS.get(fam, FACE_COLORS[Family.GENERIC])
        b = m.bindings
        for control, geo in L.controls.items():
            if control in ("dpup", "dpdown", "dpleft", "dpright"):
                continue
            if control in ("leftstick", "rightstick"):
                self._draw_stick(surf, control, geo, pad_state)
                continue
            if control in TRIGGERS:
                self._draw_trigger(surf, control, geo, pad_state)
                continue
            if control in ("a", "b", "x", "y"):
                base = _mix(face_colors[control], C_CTRL, 0.55)
                fill, edge, pressed = self._ctrl_colors(control, pad_state, base)
                if pressed:
                    fill = _mix(face_colors[control], (255, 255, 255), 0.35)
                    self.blend_circle(surf, geo["center"], geo["r"] + 6 * s, (*C_HOT, 90))
                self.aacircle(surf, geo["center"], geo["r"], fill)
                self.aacircle(surf, geo["center"], geo["r"], edge, 2)
                glyph_color = (20, 22, 26) if pressed else _mix(face_colors[control], (255, 255, 255), 0.5)
                self._face_glyph(surf, control, geo["center"], geo["r"], glyph_color)
                continue
            fill, edge, pressed = self._ctrl_colors(control, pad_state)
            if control == "guide":
                self.aacircle(surf, geo["center"], geo["r"], fill)
                self.aacircle(surf, geo["center"], geo["r"], C_FAINT, 1)
                self.blit_text(surf, control_label("guide", fam)[:4], 15 * s, C_DIM if not pressed else (20, 20, 20),
                               geo["center"], "center")
                continue
            if pressed:
                x, y, w, h = geo["rect"]
                self.blend_rect(surf, (x - 4 * s, y - 4 * s, w + 8 * s, h + 8 * s), (*C_HOT, 80), geo["radius"] + 4 * s)
            self.rrect(surf, geo["rect"], fill, geo["radius"])
            self.rrect(surf, geo["rect"], edge, geo["radius"], 2)
            if control in ("leftshoulder", "rightshoulder", "touchpad"):
                self.blit_text(surf, control_label(control, fam), 20 * s, (20, 20, 20) if pressed else C_TEXT,
                               geo["center"], "center")
        self._draw_dpad(surf, pad_state)
        # stick roles
        for side, control in (("left", "leftstick"), ("right", "rightstick")):
            geo = L.controls[control]
            role = "MOVE" if b.move_stick == side else "LOOK"
            cx, cy = geo["center"]
            self.blit_text(surf, role, 15 * s, C_DIM, (cx, cy + geo["r"] + 10 * s), "center")

    def _face_glyph(self, surf, control, center, r, color):
        fam = self.model.family
        s = self.model.layout.s
        if fam == Family.PLAYSTATION:
            cx, cy = center
            g = r * 0.48
            if control == "a":          # cross
                pygame.draw.line(surf, color, (cx - g, cy - g), (cx + g, cy + g), max(2, int(2.5 * s)))
                pygame.draw.line(surf, color, (cx - g, cy + g), (cx + g, cy - g), max(2, int(2.5 * s)))
            elif control == "b":        # circle
                self.aacircle(surf, center, g * 1.1, color, max(2, int(2 * s)))
            elif control == "x":        # square
                pygame.draw.rect(surf, color, pygame.Rect(int(cx - g), int(cy - g), int(2 * g), int(2 * g)),
                                 max(2, int(2 * s)))
            else:                       # triangle
                pts = [(cx, cy - g * 1.15), (cx + g * 1.1, cy + g * 0.8), (cx - g * 1.1, cy + g * 0.8)]
                pygame.draw.polygon(surf, color, pts, max(2, int(2 * s)))
        else:
            self.blit_text(surf, control_label(control, fam), 22 * s, color, center, "center")

    def _draw_stick(self, surf, control, geo, pad_state):
        m, L = self.model, self.model.layout
        s = L.s
        side = "left" if control == "leftstick" else "right"
        cx, cy = geo["center"]
        r = geo["r"]
        fill, edge, pressed = self._ctrl_colors(control, pad_state, C_BODY_SHADE)
        # the well
        self.aacircle(surf, (cx, cy), r, C_BODY_SHADE if not pressed else _mix(C_BODY_SHADE, C_HOT, 0.3))
        self.aacircle(surf, (cx, cy), r, edge, 2)
        # deadzone ring (scaled to the cap's travel)
        travel = r * 0.42
        dz = m.bindings.deadzone
        self.aacircle(surf, (cx, cy), max(2.0, travel * dz + 1), C_FAINT, 1)
        x, y = (pad_state.stick(side) if pad_state is not None else (0.0, 0.0))
        mag = math.hypot(x, y)
        if mag > 1.0:
            x, y = x / mag, y / mag
        cap = (cx + x * travel, cy + y * travel)
        active = mag > dz
        cap_color = C_HOT if pressed else (_mix(C_CTRL, C_FOCUS, 0.45) if active else C_CTRL)
        self.aacircle(surf, cap, r * 0.62, cap_color)
        self.aacircle(surf, cap, r * 0.62, C_CTRL_EDGE if not pressed else (255, 220, 160), 1)
        self.aacircle(surf, cap, r * 0.42, _mix(cap_color, (0, 0, 0), 0.18), 1)
        self.blit_text(surf, control_label(control, m.family), 15 * s, (25, 25, 25) if pressed else C_TEXT, cap, "center")

    def _draw_trigger(self, surf, control, geo, pad_state):
        m, L = self.model, self.model.layout
        s = L.s
        fill, edge, pressed = self._ctrl_colors(control, pad_state)
        x, y, w, h = geo["rect"]
        self.rrect(surf, geo["rect"], C_CTRL, geo["radius"])
        value = pad_state.axes.get(control, 0.0) if pad_state is not None else 0.0
        if pressed and value <= 0.0:
            value = 1.0                 # digital triggers (raw pads) only report on / off
        if value > 0.01:
            fh = h * value
            fill_surf = pygame.Surface((int(w), int(h)), pygame.SRCALPHA)
            pygame.draw.rect(fill_surf, (*(C_HOT if pressed else _mix(C_HOT, C_CTRL, 0.45)), 255),
                             pygame.Rect(0, int(h - fh), int(w), int(math.ceil(fh))))
            # clip the fill to the rounded trigger shape
            mask = pygame.Surface((int(w), int(h)), pygame.SRCALPHA)
            pygame.draw.rect(mask, (255, 255, 255, 255), mask.get_rect(), border_radius=int(geo["radius"]))
            fill_surf.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)
            surf.blit(fill_surf, (int(x), int(y)))
        self.rrect(surf, geo["rect"], edge, geo["radius"], 2)
        self.blit_text(surf, control_label(control, m.family), 20 * s, (20, 20, 20) if pressed else C_TEXT,
                       geo["center"], "center")

    def _draw_dpad(self, surf, pad_state):
        m, L = self.model, self.model.layout
        s = L.s
        cx, cy = L.dpad_center
        half, arm = L.dpad_half, L.dpad_arm
        base = [pygame.Rect(int(cx - half), int(cy - half - arm), int(2 * half), int(2 * (arm + half))),
                pygame.Rect(int(cx - half - arm), int(cy - half), int(2 * (arm + half)), int(2 * half))]
        for rc in base:
            pygame.draw.rect(surf, C_CTRL, rc, border_radius=int(3 * s))
        for control, (dx, dy) in (("dpup", (0, -1)), ("dpdown", (0, 1)), ("dpleft", (-1, 0)), ("dpright", (1, 0))):
            fill, edge, pressed = self._ctrl_colors(control, pad_state)
            geo = L.controls[control]
            if pressed:
                self.rrect(surf, geo["rect"], C_HOT, 3 * s)
            if edge != C_CTRL_EDGE:
                self.rrect(surf, geo["rect"], edge, 3 * s, 2)
            # arrow
            tip = (cx + dx * (arm + half * 0.55), cy + dy * (arm + half * 0.55))
            back = (cx + dx * (arm - half * 0.35), cy + dy * (arm - half * 0.35))
            px, py = -dy * half * 0.5, dx * half * 0.5
            color = (25, 25, 25) if pressed else C_DIM
            gfx.aapolygon(surf, [tip, (back[0] + px, back[1] + py), (back[0] - px, back[1] - py)], color)
            gfx.filled_polygon(surf, [tip, (back[0] + px, back[1] + py), (back[0] - px, back[1] - py)], color)
        for rc in base:
            pygame.draw.rect(surf, C_CTRL_EDGE, rc, 1, border_radius=int(3 * s))

    def _draw_callouts(self, surf, pad_state):
        m, L = self.model, self.model.layout
        s = L.s
        for control, entry in L.callouts.items():
            x, y, w, h = entry["rect"]
            pressed = self._pressed(control, pad_state)
            focus = m.focus == ("control", control)
            waiting = m.waiting == control
            box = C_BOX
            edge = C_BOX_EDGE
            if pressed:
                box = _mix(C_BOX, C_HOT, 0.28)
                edge = C_HOT
            if focus:
                edge = C_FOCUS
            if waiting:
                edge = _mix(C_HOT, (255, 255, 255), self._pulse() * 0.6)
                box = _mix(C_BOX, C_HOT, 0.15 + 0.1 * self._pulse())
            self.rrect(surf, entry["rect"], box, 8 * s)
            self.rrect(surf, entry["rect"], edge, 8 * s, 2 if (focus or waiting or pressed) else 1)
            # control chip
            chip_w = 54 * s
            chip = (x + 6 * s, y + 6 * s, chip_w, h - 12 * s)
            self.rrect(surf, chip, C_HOT if pressed else (44, 48, 57), 6 * s)
            self.blit_text(surf, control_label(control, m.family), 19 * s, (20, 20, 20) if pressed else C_TEXT,
                           (chip[0] + chip_w / 2, y + h / 2), "center", max_w=chip_w - 6 * s)
            tx = x + chip_w + 14 * s
            max_w = w - chip_w - 20 * s
            if control == PAUSE_CONTROL:
                self.blit_text(surf, "Pause menu", 22 * s, C_TEXT, (tx, y + 5 * s), max_w=max_w)
                self.blit_text(surf, "fixed - cannot be rebound", 17 * s, C_FAINT, (tx, y + 24 * s), max_w=max_w)
                continue
            if waiting:
                self.blit_text(surf, "Press a key...", 22 * s, C_HOT, (tx, y + 5 * s), max_w=max_w)
                self.blit_text(surf, "Esc cancel - Backspace unbind", 17 * s, C_DIM, (tx, y + 24 * s), max_w=max_w)
                continue
            key = m.bindings.key_for(control)
            self.blit_text(surf, key_display_name(key), 22 * s, C_TEXT if key else C_FAINT, (tx, y + 5 * s), max_w=max_w)
            self.blit_text(surf, key_action_label(key) if key else "click to bind a key", 17 * s, C_DIM if key else C_FAINT,
                           (tx, y + 24 * s), max_w=max_w)

    def _draw_options(self, surf):
        m, L = self.model, self.model.layout
        s = L.s
        for name in OPTIONS:
            entry = L.options[name]
            x, y, w, h = entry["rect"]
            focus = m.focus == ("option", name)
            if focus:
                self.blend_rect(surf, entry["rect"], (255, 158, 38, 46), 8 * s)
                self.rrect(surf, entry["rect"], C_HOT, 8 * s, 1)
            color = (255, 176, 70) if focus else C_TEXT
            if name in ("reset", "back"):
                self.blit_text(surf, OPTION_LABELS[name], 25 * s, color, (x + w / 2, y + h / 2), "center")
                continue
            self.blit_text(surf, OPTION_LABELS[name], 23 * s, color, (x + 12 * s, y + h / 2), "midleft")
            value = m.option_value(name)
            if name in SLIDERS:
                for sign in ("minus", "plus"):
                    bx = entry[sign]
                    self.rrect(surf, bx, (44, 48, 57), 6 * s)
                    cx, cy = bx[0] + bx[2] / 2, bx[1] + bx[3] / 2
                    d = 6 * s * (-1 if sign == "minus" else 1)
                    tri = [(cx + d, cy), (cx - d * 0.6, cy - 7 * s), (cx - d * 0.6, cy + 7 * s)]
                    gfx.aapolygon(surf, tri, C_TEXT)
                    gfx.filled_polygon(surf, tri, C_TEXT)
                mid = (entry["minus"][0] + entry["minus"][2] + entry["plus"][0]) / 2
                self.blit_text(surf, value, 23 * s, color, (mid, y + h / 2), "center")
            else:
                on = value == "On"
                pill = (x + w - 62 * s, y + 10 * s, 50 * s, h - 20 * s)
                self.rrect(surf, pill, (40, 150, 90) if on else (60, 64, 74), (h - 20 * s) / 2)
                knob_x = pill[0] + (pill[2] - pill[3] / 2 - 2 * s if on else pill[3] / 2 + 2 * s)
                self.aacircle(surf, (knob_x, pill[1] + pill[3] / 2), pill[3] / 2 - 3 * s, (235, 238, 242))
        # explain the stick layout under the toggles
        b = m.bindings
        px, py, pw, ph = L.panel_rect
        self.blit_text(surf, f"{b.move_stick.capitalize()} stick: move    {b.look_stick.capitalize()} stick: look",
                       18 * s, C_FAINT, (px + pw / 2, py + ph - 14 * s), "center", max_w=pw - 20 * s)

    def _draw_status(self, surf):
        m, L = self.model, self.model.layout
        s = L.s
        color = {"info": (205, 210, 216), "ok": C_OK, "warn": C_WARN,
                 "prompt": _mix(C_HOT, (255, 235, 200), self._pulse() * 0.5)}.get(m.status_kind, C_TEXT)
        x, y, w, h = L.status_rect
        if m.status:
            self.blit_text(surf, m.status, 27 * s, color, (x + w / 2, y + h / 2), "center", max_w=w)
        self.blit_text(surf, "Click a button or its label, then press a keyboard key   -   Arrows + Enter also work   -   "
                             "Esc or Start: back", 21 * s, C_FAINT, L.hint_pos, "center", max_w=1200 * s)
