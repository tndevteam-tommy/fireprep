"""Window initialization and Modern OpenGL context manager (windowed / fullscreen)."""
import json
import os

import config
try:
    import pygame
except ImportError:
    pass


def display_settings_path():
    """display.json next to the controller settings (HARDCORE_CONFIG_DIR overrides the folder)."""
    from core.gamepad import default_settings_path
    return os.path.join(os.path.dirname(default_settings_path()), "display.json")


def load_display_settings(path=None):
    path = path or display_settings_path()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {"fullscreen": bool(data.get("fullscreen", config.FULLSCREEN))} if isinstance(data, dict) else \
            {"fullscreen": config.FULLSCREEN}
    except (OSError, ValueError):
        return {"fullscreen": config.FULLSCREEN}


def save_display_settings(settings, path=None):
    path = path or display_settings_path()
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"fullscreen": bool(settings.get("fullscreen"))}, f, indent=2)
        os.replace(tmp, path)
        return True
    except OSError as exc:
        print(f"[display] could not save {path}: {exc}")
        return False


def fullscreen_from_args(argv, default):
    """--fullscreen / --windowed on the command line override the saved preference."""
    if "--fullscreen" in argv:
        return True
    if "--windowed" in argv:
        return False
    return default


class WindowManager:
    def __init__(self, width=config.WINDOW_WIDTH, height=config.WINDOW_HEIGHT, title=config.WINDOW_TITLE,
                 fullscreen=False):
        self.width = width
        self.height = height
        self.windowed_size = (width, height)
        self.title = title
        self.mouse_grabbed = True
        self.fullscreen = False

        pygame.init()
        pygame.display.set_caption(title)

        # Enforce Modern OpenGL 3.3 Core Profile
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
        pygame.display.gl_set_attribute(pygame.GL_DOUBLEBUFFER, 1)
        pygame.display.gl_set_attribute(pygame.GL_DEPTH_SIZE, 24)

        self.screen = pygame.display.set_mode((width, height), pygame.OPENGL | pygame.DOUBLEBUF)
        self.clock = pygame.time.Clock()

        # Capture mouse cursor
        self.set_mouse_grab(True)
        if fullscreen:
            self.set_fullscreen(True)

    # ------------------------------------------------------------------ size
    @property
    def size(self):
        """The drawable size right now (changes when going fullscreen / back)."""
        try:
            w, h = pygame.display.get_window_size()
        except (AttributeError, pygame.error):
            w, h = self.screen.get_size()
        return max(1, int(w)), max(1, int(h))

    # ------------------------------------------------------------ fullscreen
    def _is_fullscreen(self):
        fn = getattr(pygame.display, "is_fullscreen", None)
        if fn is not None:
            try:
                return bool(fn())
            except pygame.error:
                pass
        return self.size != self.windowed_size

    def set_fullscreen(self, flag):
        """Switch to borderless fullscreen at the desktop resolution (or back to the window).

        pygame's toggle keeps the OpenGL context (every buffer, texture and shader survives);
        re-creating the display with set_mode is only the fallback.  Returns the new state."""
        flag = bool(flag)
        if flag == self.fullscreen:
            return self.fullscreen
        ok = False
        try:
            pygame.display.toggle_fullscreen()
            ok = self._is_fullscreen() == flag
        except pygame.error as exc:
            print(f"[display] fullscreen toggle failed ({exc}); re-creating the display")
        if not ok:
            try:
                if flag:
                    self.screen = pygame.display.set_mode((0, 0), pygame.OPENGL | pygame.DOUBLEBUF | pygame.FULLSCREEN)
                else:
                    self.screen = pygame.display.set_mode(self.windowed_size, pygame.OPENGL | pygame.DOUBLEBUF)
                ok = True
            except pygame.error as exc:
                print(f"[display] could not switch to {'fullscreen' if flag else 'a window'}: {exc}")
        if ok:
            self.fullscreen = flag
            self.width, self.height = self.size
            # the grab is tied to the window state: apply it again, then drop the cursor jump
            self.set_mouse_grab(self.mouse_grabbed)
            pygame.mouse.get_rel()
            print(f"[display] {'fullscreen' if flag else 'windowed'} {self.width}x{self.height}")
        return self.fullscreen

    def toggle_fullscreen(self):
        return self.set_fullscreen(not self.fullscreen)

    # ---------------------------------------------------------------- input
    def set_mouse_grab(self, grab):
        self.mouse_grabbed = grab
        pygame.event.set_grab(grab)
        pygame.mouse.set_visible(not grab)

    def toggle_mouse_grab(self):
        self.set_mouse_grab(not self.mouse_grabbed)

    def swap(self):
        pygame.display.flip()

    def close(self):
        pygame.quit()
