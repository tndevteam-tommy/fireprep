"""Pause menu: dark overlay with keyboard / mouse / controller selectable entries.

Text is rasterised with pygame's font module and uploaded as RGBA textures
drawn through the textured UI shader - no fixed-function GL involved.  The
Controller entry opens the controller screen (render/controller_view.py), which
is drawn into a pygame Surface and uploaded as one full-screen texture whenever
it changes.
"""
import numpy as np
import config
from math3d.transform import ortho
from core.menu_model import MenuModel
from core.controller_menu import ControllerMenuModel
from core.gamepad import Family, pad_menu_hint
from render.buffer import MeshBuffer
from render.controller_view import ControllerView, OVERLAY_ALPHA
from render.reticle import ui_scale

# The controller screen is drawn in software: past this size it is drawn smaller and stretched by GL
CONTROLLER_SURFACE_MAX = (1920, 1200)


def controller_surface_size(screen_w, screen_h):
    """Size of the software-drawn controller screen for a window: the window size, capped
    (same aspect) so a 4K screen does not re-upload 25 MB whenever a button lights up."""
    k = min(1.0, CONTROLLER_SURFACE_MAX[0] / float(screen_w), CONTROLLER_SURFACE_MAX[1] / float(screen_h))
    return int(round(screen_w * k)), int(round(screen_h * k))
try:
    import pygame
    from OpenGL.GL import *
except ImportError:
    pass


class TextLabel:
    """A rasterised string living in a GL texture."""

    def __init__(self, text, size=48, color=(255, 255, 255)):
        self.text = text
        font = pygame.font.Font(None, size)
        surf = font.render(text, True, color).convert_alpha()
        self.width, self.height = surf.get_size()
        try:
            data = pygame.image.tobytes(surf, "RGBA", True)
        except AttributeError:            # pygame < 2.1.3
            data = pygame.image.tostring(surf, "RGBA", True)
        self.tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.tex)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, self.width, self.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data)
        glBindTexture(GL_TEXTURE_2D, 0)

    def release(self):
        if self.tex is not None:
            glDeleteTextures([self.tex])
            self.tex = None


class SolidTexture:
    """A tiny opaque white texture for tinted rectangles (overlay, highlight bars)."""

    def __init__(self):
        self.width = self.height = 2
        self.tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.tex)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        data = bytes([255] * (2 * 2 * 4))
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 2, 2, 0, GL_RGBA, GL_UNSIGNED_BYTE, data)
        glBindTexture(GL_TEXTURE_2D, 0)


class SurfaceTexture:
    """A full-screen RGBA texture re-uploaded from a pygame Surface (the controller screen)."""

    def release(self):
        if self.tex is not None:
            glDeleteTextures([self.tex])
            self.tex = None

    def __init__(self, width, height):
        self.width, self.height = width, height
        self.tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.tex)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)
        glBindTexture(GL_TEXTURE_2D, 0)

    def upload(self, surf):
        # The surface is opaque: upload RGB (sampled alpha = 1).  Asking pygame for "RGBA" from a
        # surface without an alpha channel copies its padding byte as alpha - garbage translucency.
        try:
            data = pygame.image.tobytes(surf, "RGB", True)
        except AttributeError:            # pygame < 2.1.3
            data = pygame.image.tostring(surf, "RGB", True)
        glBindTexture(GL_TEXTURE_2D, self.tex)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, self.width, self.height, GL_RGB, GL_UNSIGNED_BYTE, data)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 4)
        glBindTexture(GL_TEXTURE_2D, 0)


class PauseMenu:
    RESUME = MenuModel.RESUME
    CONTROLLER = MenuModel.CONTROLLER
    FULLSCREEN = MenuModel.FULLSCREEN
    MULTIPLAYER = MenuModel.MULTIPLAYER
    QUIT = MenuModel.QUIT
    ITEMS = (("Resume", MenuModel.RESUME), ("Multiplayer", MenuModel.MULTIPLAYER), ("Controller", MenuModel.CONTROLLER),
             ("Fullscreen: Off", MenuModel.FULLSCREEN), ("Quit", MenuModel.QUIT))
    RESULT_ITEMS = ("Join new game", "Leave game")

    def __init__(self, screen_w=config.WINDOW_WIDTH, screen_h=config.WINDOW_HEIGHT):
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.scale = ui_scale(screen_w, screen_h)
        self.fullscreen_on = False
        self.model = MenuModel(self.ITEMS)
        self.screen = "main"               # "main" list or the "controller" screen
        self.controller = None             # ControllerMenuModel (attach_controller)
        self.controller_view = None
        self._controller_tex = None
        self._controller_sig = None
        self.pad_state = None
        self.mp = None                     # MultiplayerMenuModel (attach_multiplayer)
        self.results = None                # {"title", "lines", "waiting"} while a round's results are up
        self.result_focus = 0
        self.result_rects = {}
        self._dyn = {}                     # text labels that change (values, scores)

        # unit quad (pos 0..1, uv 0..1) - y is flipped because textures are uploaded bottom-up
        verts = np.array([
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ], dtype=np.float32)
        indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)
        layout = [(0, 2, GL_FLOAT, False), (1, 2, GL_FLOAT, False)]
        self.quad = MeshBuffer(verts, indices, layout)

        if not pygame.font.get_init():
            pygame.font.init()
        self._blank = SolidTexture()       # tinted rectangles (overlay, highlight bar)
        self._pad_hint_family = Family.XBOX
        self.labels = []
        self._build_labels()

    # ------------------------------------------------------------------ labels
    def _font(self, size):
        return max(10, int(round(size * self.scale)))

    def _item_text(self, item_id, text):
        if item_id == self.FULLSCREEN:
            return "Fullscreen: On" if self.fullscreen_on else "Fullscreen: Off"
        return text

    def _build_labels(self):
        """(Re)rasterise every label at the current UI scale (crisp text at any resolution)."""
        for lab in [getattr(self, "title", None), getattr(self, "hint", None), getattr(self, "pad_hint", None)] + \
                [x for pair in self.labels for x in pair]:
            if lab is not None:
                lab.release()
        self.title = TextLabel("PAUSED", self._font(72), (255, 255, 255))
        self.hint = TextLabel("W/S or mouse to choose   -   Enter / click to confirm   -   Esc to resume   -   F11 fullscreen",
                              self._font(26), (200, 205, 210))
        self.pad_hint = TextLabel(pad_menu_hint(self._pad_hint_family), self._font(24), (150, 156, 166))
        self.labels = []
        for text, item_id in self.model.items:
            text = self._item_text(item_id, text)
            self.labels.append((TextLabel(text, self._font(52), (230, 230, 235)),
                                TextLabel(text, self._font(52), (255, 160, 40))))
        self.model.rects = {i: self._item_rect(i) for i in range(len(self.model.items))}

    def dyn_label(self, text, size, color):
        """A cached label for changing text (multiplayer values, scores)."""
        key = (text, int(size), tuple(color))
        lab = self._dyn.get(key)
        if lab is None:
            if len(self._dyn) > 200:
                for old in self._dyn.values():
                    old.release()
                self._dyn.clear()
            lab = self._dyn[key] = TextLabel(text, int(size), color)
        return lab

    def attach_multiplayer(self, model):
        self.mp = model

    def show_results(self, results):
        """Open the results screen (the game pauses the local player while it is up)."""
        self.results = results
        if not self.model.visible:
            self.model.open()
        if self.screen != "results":
            self.result_focus = 0
        self.screen = "results"

    def set_fullscreen_state(self, on):
        """Show the current display mode on the Fullscreen entry."""
        if bool(on) != self.fullscreen_on:
            self.fullscreen_on = bool(on)
            self._build_labels()

    def resize(self, screen_w, screen_h):
        """The window changed size (fullscreen / back): lay everything out for the new size."""
        if (screen_w, screen_h) == (self.screen_w, self.screen_h):
            return
        self.screen_w, self.screen_h = screen_w, screen_h
        self.scale = ui_scale(screen_w, screen_h)
        self._build_labels()
        if self.controller is not None:
            self.controller.resize(*self._controller_surface_size())
        if self._controller_tex is not None:
            self._controller_tex.release()
            self._controller_tex = None
        self._controller_sig = None

    @property
    def visible(self):
        return self.model.visible

    @property
    def selected(self):
        return self.model.selected

    # ------------------------------------------------------------------ layout
    def _item_rect(self, i):
        w, h = self.labels[i][0].width, self.labels[i][0].height
        x = self.screen_w * 0.5 - w * 0.5
        y = self.screen_h * 0.42 + i * (h + 20 * self.scale)
        return x, y, w, h

    def _controller_surface_size(self):
        return controller_surface_size(self.screen_w, self.screen_h)

    def _to_surface(self, pos):
        sw, sh = self._controller_surface_size()
        return (pos[0] * sw / float(self.screen_w), pos[1] * sh / float(self.screen_h))

    # -------------------------------------------------------------- controller
    def attach_controller(self, bindings, on_change=None):
        """Enable the Controller screen, editing `bindings` (on_change is called after every edit)."""
        self.controller = ControllerMenuModel(bindings, *self._controller_surface_size(), on_change=on_change)
        self.controller_view = ControllerView(self.controller)

    def _set_pad_hint(self, family):
        if family == self._pad_hint_family:
            return
        old = getattr(self, "pad_hint", None)
        self.pad_hint = TextLabel(pad_menu_hint(family), self._font(24), (150, 156, 166))
        if old is not None:
            old.release()
        self._pad_hint_family = family

    def set_pad(self, device):
        """The active controller (or None): shown live on the controller screen."""
        self.pad_state = device.state if device is not None else None
        if device is not None:
            self._set_pad_hint(device.family)
        if self.controller is not None:
            if device is not None:
                self.controller.set_device(device.name, device.family, recognised=device.kind == "gamepad")
            else:
                self.controller.device_name = None

    @property
    def in_controller_screen(self):
        return self.model.visible and self.screen == "controller"

    def _activate(self, choice):
        """Main-list choice -> RESUME / QUIT for the caller; CONTROLLER opens the sub-screen."""
        if choice == self.CONTROLLER:
            if self.controller is not None:
                self.screen = "controller"
                self.controller.open()
            return None
        if choice == self.MULTIPLAYER:
            if self.mp is not None:
                self.screen = "multiplayer"
            return None
        return choice            # RESUME / FULLSCREEN / QUIT: handled by the game loop

    def _leave_controller(self):
        self.controller.close()
        self.screen = "main"

    # ------------------------------------------------------------------- input
    def open(self):
        self.model.open()
        self.screen = "main"

    def close(self):
        self.model.close()
        if self.controller is not None:
            self.controller.close()
        self.screen = "main"

    def navigate(self, command):
        """Menu command from a controller ('up' / 'down' / 'confirm' / 'back' ...)."""
        if not self.model.visible or command is None:
            return None
        if self.screen == "multiplayer":
            return self._mp_action(self.mp.handle_nav(command))
        if self.screen == "results":
            return self._results_nav(command)
        if self.screen != "main":
            return None
        if command in ("left", "right"):
            return None
        return self._activate(self.model.handle(command))

    def handle_pad(self, control, down, command=None):
        """A controller button while paused. On the controller screen every button just
        lights up / is selected (Start goes back); in the list it navigates."""
        if not self.model.visible:
            return None
        if self.screen == "controller":
            if self.controller.handle_pad(control, down) == ControllerMenuModel.BACK:
                self._leave_controller()
            return None
        if self.screen in ("multiplayer", "results"):
            return self.navigate(command) if down else None
        return self.navigate(command)

    # ------------------------------------------------------------ multiplayer
    def _mp_action(self, action):
        """Multiplayer screen action -> ("mp", action) for the game loop; Back returns to the list."""
        if action is None:
            return None
        if action[0] == "back":
            self.mp.save()
            self.screen = "main"
            return None
        self.mp.save()
        return ("mp", action)

    def _results_nav(self, command):
        if command in ("up", "down"):
            self.result_focus = (self.result_focus + (1 if command == "down" else -1)) % len(self.RESULT_ITEMS)
        elif command == "confirm":
            return ("mp", ("join_next",) if self.result_focus == 0 else ("leave",))
        return None

    def handle_event(self, event):
        """Returns PauseMenu.RESUME / PauseMenu.QUIT when an entry is activated, else None."""
        if not self.model.visible:
            return None
        if self.screen == "controller":
            return self._handle_controller_event(event)
        if self.screen == "multiplayer":
            if event.type == pygame.KEYDOWN:
                return self._mp_action(self.mp.handle_key(pygame.key.name(event.key), getattr(event, "unicode", "")))
            if event.type == pygame.MOUSEMOTION:
                return self._mp_action(self.mp.handle_mouse("move", event.pos))
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                return self._mp_action(self.mp.handle_mouse("click", event.pos))
            return None
        if self.screen == "results":
            if event.type == pygame.KEYDOWN:
                name = pygame.key.name(event.key)
                cmd = {"up": "up", "w": "up", "down": "down", "s": "down", "return": "confirm",
                       "enter": "confirm", "space": "confirm"}.get(name)
                return self._results_nav(cmd) if cmd else None
            if event.type in (pygame.MOUSEMOTION, pygame.MOUSEBUTTONDOWN):
                for i, (x, y, w, h) in self.result_rects.items():
                    if x <= event.pos[0] <= x + w and y <= event.pos[1] <= y + h:
                        self.result_focus = i
                        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                            return self._results_nav("confirm")
            return None
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                return self.model.handle("back")
            if event.key in (pygame.K_UP, pygame.K_w):
                return self.model.handle("up")
            if event.key in (pygame.K_DOWN, pygame.K_s):
                return self.model.handle("down")
            if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE):
                return self._activate(self.model.handle("confirm"))
        elif event.type == pygame.MOUSEMOTION:
            return self.model.handle("hover", event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            return self._activate(self.model.handle("click", event.pos))
        return None

    def _handle_controller_event(self, event):
        c = self.controller
        result = None
        if event.type == pygame.KEYDOWN:
            result = c.handle_key(pygame.key.name(event.key))
        elif event.type == pygame.MOUSEMOTION:
            c.handle_mouse("move", self._to_surface(event.pos))
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            result = c.handle_mouse("click", self._to_surface(event.pos))
        elif event.type == pygame.MOUSEWHEEL:
            c.handle_wheel(int(event.y))
        if result == ControllerMenuModel.BACK:
            self._leave_controller()
        return None

    # -------------------------------------------------------------------- draw
    def _draw_label(self, shader, label, x, y, color=(1.0, 1.0, 1.0, 1.0), w=None, h=None):
        shader.set_vec2("u_Offset", (x, y))
        shader.set_vec2("u_Size", (w if w is not None else label.width, h if h is not None else label.height))
        shader.set_vec4("u_Color", color)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, label.tex)
        self.quad.draw()

    def _controller_signature(self):
        c, st = self.controller, self.pad_state
        animated = c.waiting is not None or c.status_kind == "prompt"
        return (c.family, c.device_name, c.device_recognised, c.focus, c.waiting, c.status, c.status_kind,
                repr(sorted(c.bindings.to_dict().items())),
                tuple(sorted(st.down)) if st is not None else None,
                tuple(round(v, 2) for v in st.axes.values()) if st is not None else None,
                int(self.controller_view.time * 24) if animated else 0)

    def _draw_controller(self, text_shader, dt):
        view = self.controller_view
        view.time += dt
        sig = self._controller_signature()
        size = self._controller_surface_size()
        if self._controller_tex is None:
            self._controller_tex = SurfaceTexture(*size)
        if sig != self._controller_sig:
            surf = view.draw(self.pad_state, 0.0, size)
            self._controller_tex.upload(surf)
            self._controller_sig = sig
        # (stretched to the window when the surface is capped below the window size)
        self._draw_label(text_shader, self._controller_tex, 0.0, 0.0, (1.0, 1.0, 1.0, OVERLAY_ALPHA),
                         self.screen_w, self.screen_h)

    def _draw_multiplayer(self, sh):
        from core.mp_menu import LABELS, TEXT_ROWS, BUTTON_ROWS, YOU_ROWS, HOST_ROWS
        k = self.scale
        mp = self.mp
        W, H = self.screen_w, self.screen_h
        title = self.dyn_label("MULTIPLAYER", self._font(56), (255, 255, 255))
        self._draw_label(sh, title, W * 0.5 - title.width * 0.5, H * 0.08)
        mp.rects = {}
        rows = mp.rows()
        col_w = min(W * 0.43, 560 * k)
        cols = {"you": W * 0.5 - col_w - 24 * k, "host": W * 0.5 + 24 * k}
        heads = {"you": "YOU", "host": "HOST A GAME"}
        y_top = H * 0.205
        for col, x in cols.items():
            head = self.dyn_label(heads[col], self._font(22), (150, 200, 255))
            self._draw_label(sh, head, x, y_top - head.height - 8 * k)
            self._draw_label(sh, self._blank, x, y_top - 4 * k, (0.55, 0.75, 1.0, 0.35), col_w, max(1.0, 2 * k))

        def draw_row(row, x, y, w):
            focused = row == mp.focused
            color = (255, 176, 70) if focused else (230, 230, 235)
            lab = self.dyn_label(LABELS[row], self._font(27), color)
            h = lab.height + 12 * k
            if row in BUTTON_ROWS:
                bx = x + w * 0.5 - lab.width * 0.5
                rect = (bx - 20 * k, y - 6 * k, lab.width + 40 * k, h)
                self._draw_label(sh, self._blank, *rect[:2], (1.0, 0.62, 0.15, 0.24 if focused else 0.08), rect[2], rect[3])
                self._draw_label(sh, lab, bx, y)
            else:
                rect = (x - 10 * k, y - 6 * k, w + 20 * k, h)
                if focused:
                    self._draw_label(sh, self._blank, *rect[:2], (1.0, 0.62, 0.15, 0.16), rect[2], rect[3])
                self._draw_label(sh, lab, x, y)
                text = mp.value_text(row)
                if row in TEXT_ROWS and focused and (int(pygame.time.get_ticks() / 500) % 2 == 0):
                    text += "_"
                elif row not in TEXT_ROWS:
                    text = f"<  {text}  >"
                vcol = color
                if row == "team":
                    from game import teams as T
                    vcol = T.UI.get(mp.values["team"], color) if not focused else color
                val = self.dyn_label(text or " ", self._font(27), vcol)
                self._draw_label(sh, val, x + w * 0.42, y)
            mp.rects[row] = rect
            return h

        ends = {}
        for col, members in (("you", YOU_ROWS), ("host", HOST_ROWS)):
            x, y = cols[col], y_top + 10 * k
            for row in members:
                if row not in rows:
                    continue
                if row in BUTTON_ROWS:
                    y += 6 * k
                y += draw_row(row, x, y, col_w) + 3 * k
            ends[col] = y
        # a word on teams under the join column
        note = ("In a game: changing your team switches sides now" if mp.in_session
                else "Your team counts in teams games (you can switch in a game)")
        lab = self.dyn_label(note, self._font(19), (150, 156, 166))
        self._draw_label(sh, lab, cols["you"], ends["you"] + 10 * k)
        y = H * 0.755
        for row in ("leave", "back"):
            if row in rows:
                y += draw_row(row, W * 0.5 - 160 * k, y, 320 * k) + 4 * k
        if mp.status:
            colors = {"info": (205, 210, 216), "ok": (120, 220, 140), "warn": (255, 196, 90)}
            st = self.dyn_label(mp.status, self._font(24), colors.get(mp.status_kind, (220, 220, 220)))
            self._draw_label(sh, st, W * 0.5 - st.width * 0.5, H * 0.885)
        hint = self.dyn_label("Up / Down choose  -  Left / Right change  -  type to edit  -  Enter select  -  Esc back",
                              self._font(20), (150, 156, 166))
        self._draw_label(sh, hint, W * 0.5 - hint.width * 0.5, H * 0.94)

    def _draw_results(self, sh):
        k = self.scale
        r = self.results or {}
        title = self.dyn_label(r.get("title", "ROUND OVER"), self._font(68), (255, 255, 255))
        self._draw_label(sh, title, self.screen_w * 0.5 - title.width * 0.5, self.screen_h * 0.14)
        win = self.dyn_label(r.get("winner", ""), self._font(44), (255, 190, 90))
        self._draw_label(sh, win, self.screen_w * 0.5 - win.width * 0.5, self.screen_h * 0.27)
        y0 = self.screen_h * 0.36
        lines = [(ln + (None,))[:3] for ln in r.get("lines", [])][:16]
        size = self._font(28 if len(lines) <= 7 else 22)
        teamed = any(t for _, _, t in lines)
        y_end = y0
        if teamed:
            # red vs blue: one column per team, in the team's colour
            from game import teams as T
            for team, cx in ((T.RED, self.screen_w * 0.32), (T.BLUE, self.screen_w * 0.68)):
                y = y0
                for line, me, t in lines:
                    if t != team:
                        continue
                    text = line.replace(f"[{T.name(t)}]  ", "")
                    color = (255, 214, 150) if me else T.UI[team]
                    lab = self.dyn_label(text, size, color)
                    self._draw_label(sh, lab, cx - lab.width * 0.5, y)
                    y += lab.height + 3 * k
                y_end = max(y_end, y)
        else:
            # best first; a long list goes into two columns
            half = (len(lines) + 1) // 2 if len(lines) > 8 else len(lines)
            cols = [(lines[:half], self.screen_w * (0.32 if len(lines) > 8 else 0.5)), (lines[half:], self.screen_w * 0.68)]
            for col, cx in cols:
                y = y0
                for line, me, _ in col:
                    lab = self.dyn_label(line, size, (255, 200, 120) if me else (225, 228, 234))
                    self._draw_label(sh, lab, cx - lab.width * 0.5, y)
                    y += lab.height + 3 * k
                y_end = max(y_end, y)
        y = max(y_end + 16 * k, self.screen_h * 0.68)
        self.result_rects = {}
        for i, text in enumerate(self.RESULT_ITEMS):
            focused = i == self.result_focus
            lab = self.dyn_label(text, self._font(44), (255, 160, 40) if focused else (230, 230, 235))
            x = self.screen_w * 0.5 - lab.width * 0.5
            rect = (x - 18 * k, y - 6 * k, lab.width + 36 * k, lab.height + 12 * k)
            if focused:
                self._draw_label(sh, self._blank, rect[0], rect[1], (1.0, 0.62, 0.15, 0.18), rect[2], rect[3])
            self._draw_label(sh, lab, x, y)
            self.result_rects[i] = rect
            y += lab.height + 18 * k
        if r.get("waiting"):
            w = self.dyn_label(r["waiting"], self._font(24), (170, 176, 186))
            self._draw_label(sh, w, self.screen_w * 0.5 - w.width * 0.5, self.screen_h * 0.92)

    def draw(self, text_shader, dt=0.0):
        if not self.model.visible:
            return
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_CULL_FACE)
        text_shader.use()
        text_shader.set_mat4("u_Proj", ortho(0.0, self.screen_w, self.screen_h, 0.0, -1.0, 1.0))
        text_shader.set_int("u_Tex", 0)
        if self.screen == "controller" and self.controller is not None:
            self._draw_controller(text_shader, dt)
            glBindTexture(GL_TEXTURE_2D, 0)
            glEnable(GL_CULL_FACE)
            glEnable(GL_DEPTH_TEST)
            return
        if self.screen in ("multiplayer", "results"):
            self._draw_label(text_shader, self._blank, 0.0, 0.0, (0.0, 0.0, 0.0, 0.68), self.screen_w, self.screen_h)
            if self.screen == "multiplayer":
                self._draw_multiplayer(text_shader)
            else:
                self._draw_results(text_shader)
            glBindTexture(GL_TEXTURE_2D, 0)
            glEnable(GL_CULL_FACE)
            glEnable(GL_DEPTH_TEST)
            return
        # Dim the world
        self._draw_label(text_shader, self._blank, 0.0, 0.0, (0.0, 0.0, 0.0, 0.62), self.screen_w, self.screen_h)
        # Title
        self._draw_label(text_shader, self.title, self.screen_w * 0.5 - self.title.width * 0.5, self.screen_h * 0.28)
        # Entries
        for i, (normal, hot) in enumerate(self.labels):
            x, y, w, h = self._item_rect(i)
            label = hot if i == self.model.selected else normal
            if i == self.model.selected:
                pad_x, pad_y = 18 * self.scale, 6 * self.scale
                self._draw_label(text_shader, self._blank, x - pad_x, y - pad_y, (1.0, 0.62, 0.15, 0.18),
                                 w + 2 * pad_x, h + 2 * pad_y)
            self._draw_label(text_shader, label, x, y)
        self._draw_label(text_shader, self.hint, self.screen_w * 0.5 - self.hint.width * 0.5, self.screen_h * 0.82)
        self._draw_label(text_shader, self.pad_hint, self.screen_w * 0.5 - self.pad_hint.width * 0.5,
                         self.screen_h * 0.82 + self.hint.height + 10 * self.scale)
        glBindTexture(GL_TEXTURE_2D, 0)
        glEnable(GL_CULL_FACE)
        glEnable(GL_DEPTH_TEST)
