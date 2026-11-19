"""Scripted-input harness that drives either implementation frame by frame.

Both the frozen prototype (``tests/baseline/main.py``) and the refactored
``unprepared.app.PygameApp`` read input through ``pygame.event.get`` /
``pygame.key.get_pressed`` and time through ``pygame.time.Clock.tick``.  We
replace those with a deterministic script so both see *exactly* the same
input, then hook ``pygame.display.flip`` to snapshot the game state and the
rendered frame every frame.
"""
from __future__ import annotations

import hashlib
import os
import sys

import numpy as np
from dataclasses import dataclass, field
from typing import FrozenSet, List, Optional, Sequence

import pygame

K = pygame.constants
SEED = 1234          # gameplay RNG seed used for every scripted run (lock sweet spots)


# --------------------------------------------------------------------------- script
@dataclass(frozen=True)
class FrameScript:
    dt_ms: int = 16
    events: tuple = ()              # (type, kwargs) tuples; turned into pygame events at runtime
    held: FrozenSet[int] = frozenset()
    # (x, z, yaw, pitch): before the frame, stand the player there (on the floor) facing so - a
    # scenario's "walk across town" in one step. The game only: the prototype (whose town was
    # another) never sees such a frame - the prototype-identical scenarios use none.
    place: Optional[tuple] = None
    skills: Optional[tuple] = None  # ((skill name, level), ...): set before the frame (the game only, like ``place``)


@dataclass
class Scenario:
    name: str
    frames: List[FrameScript] = field(default_factory=list)
    capture_frames: set = field(default_factory=set)   # frame indices to keep full PNGs for
    end_with_quit_event: bool = True
    tutorial: bool = False                              # start the game in the tutorial (an urchin in Jae's debt)

    # builders -------------------------------------------------------------
    def idle(self, n: int = 1, dt_ms: int = 16, held: Sequence[int] = ()) -> "Scenario":
        for _ in range(n):
            self.frames.append(FrameScript(dt_ms, (), frozenset(held)))
        return self

    def hold(self, keys: Sequence[int], n: int, dt_ms: int = 16) -> "Scenario":
        return self.idle(n, dt_ms, keys)

    def press(self, key: int, dt_ms: int = 16, held: Sequence[int] = ()) -> "Scenario":
        self.frames.append(FrameScript(dt_ms, ((K.KEYDOWN, {"key": key, "mod": 0, "unicode": "", "scancode": 0}),), frozenset(held)))
        return self

    def mouse(self, dx: int, dy: int, dt_ms: int = 16) -> "Scenario":
        self.frames.append(FrameScript(dt_ms, ((K.MOUSEMOTION, {"rel": (dx, dy), "pos": (512, 384), "buttons": (0, 0, 0)}),), frozenset()))
        return self

    def place(self, x: float, z: float, yaw: float = 0.0, pitch: float = 0.0) -> "Scenario":
        """Stand the player at (x, z) on the floor there, facing ``yaw`` / ``pitch`` (one frame)."""
        self.frames.append(FrameScript(16, (), frozenset(), (x, z, yaw, pitch)))
        return self

    def skill(self, name: str, level: int) -> "Scenario":
        """Set one of the player's skills (one frame): an old hand at something from the start."""
        self.frames.append(FrameScript(16, (), frozenset(), None, ((name, level),)))
        return self

    def wait_hours(self, hours: int) -> "Scenario":
        """Advance game time quickly: dt=1s frames, 24 frames per simulated hour."""
        return self.idle(24 * hours, dt_ms=1000)

    def capture(self) -> "Scenario":
        self.capture_frames.add(len(self.frames))   # capture the *next* frame rendered
        return self

    def capture_last(self) -> "Scenario":
        self.capture_frames.add(len(self.frames) - 1)
        return self

    def __len__(self) -> int:
        return len(self.frames)


# --------------------------------------------------------------------------- results
@dataclass
class FrameRecord:
    index: int
    state: dict
    pixel_hash: str
    png: Optional[bytes] = None


@dataclass
class RunResult:
    implementation: str
    scenario: str
    frames: List[FrameRecord]
    error: Optional[str] = None
    save_file: Optional[bytes] = None


# --------------------------------------------------------------------------- pygame patching
class _FakeClock:
    def __init__(self, script: "ScriptedPygame"):
        self.script = script

    def tick(self, fps=0):
        return self.script.next_frame()

    def get_fps(self):
        return 60.0


class _FakeKeys:
    def __init__(self, held: FrozenSet[int]):
        self.held = held

    def __getitem__(self, key):
        return key in self.held


class ScriptedPygame:
    """Context manager that replaces pygame's clock/input/flip with the script."""

    def __init__(self, scenario: Scenario, snapshot_fn, width: int = 1024, height: int = 768, mask_rects=()):
        self.scenario = scenario
        self.snapshot_fn = snapshot_fn
        self.width, self.height = width, height
        self.mask_rects = list(mask_rects)      # (x1, y1, x2, y2) window rects excluded from the pixel hash
        self.frame = -1
        self.records: List[FrameRecord] = []
        self._orig = {}
        self.on_place = None                    # the game's hook for ``FrameScript.place`` (None: ignored)

    # script playback -------------------------------------------------------
    def current(self) -> FrameScript:
        if 0 <= self.frame < len(self.scenario.frames):
            return self.scenario.frames[self.frame]
        return FrameScript()

    def next_frame(self) -> int:
        self.frame += 1
        return self.current().dt_ms

    def _events(self):
        if self.on_place is not None and (self.current().place is not None or self.current().skills):
            self.on_place(self.current())
        out = [pygame.event.Event(t, **kw) for t, kw in self.current().events]
        if self.frame >= len(self.scenario.frames) and self.scenario.end_with_quit_event:
            out.append(pygame.event.Event(K.QUIT))
        return out

    def _keys(self):
        return _FakeKeys(self.current().held)

    def _flip(self):
        from OpenGL.GL import GL_RGB, GL_UNSIGNED_BYTE, glReadPixels
        if sys._getframe(1).f_code.co_name == "show_busy":
            self._orig["flip"]()                # (the app's "please wait" box before a long job: not a frame of the game)
            return
        data = glReadPixels(0, 0, self.width, self.height, GL_RGB, GL_UNSIGNED_BYTE)
        hashed = bytes(data)
        if self.mask_rects:
            img = np.frombuffer(hashed, dtype=np.uint8).reshape(self.height, self.width, 3).copy()
            for x1, y1, x2, y2 in self.mask_rects:      # GL rows are bottom-up
                img[self.height - y2:self.height - y1, x1:x2, :] = 0
            hashed = img.tobytes()
        png = None
        if self.frame in self.scenario.capture_frames:
            surf = pygame.image.fromstring(bytes(data), (self.width, self.height), "RGB", True)
            import io
            buf = io.BytesIO()
            pygame.image.save(surf, buf, "png")
            png = buf.getvalue()
        state = self.snapshot_fn(sys._getframe(1))
        state["frame"] = self.frame
        state["mouse_visible"] = pygame.mouse.get_visible()
        state["mouse_grab"] = pygame.event.get_grab()
        self.records.append(FrameRecord(self.frame, state, hashlib.sha1(hashed).hexdigest(), png))
        self._orig["flip"]()

    # patching ----------------------------------------------------------------
    def __enter__(self):
        self._orig = {
            "Clock": pygame.time.Clock, "get": pygame.event.get, "get_pressed": pygame.key.get_pressed,
            "flip": pygame.display.flip,
        }
        pygame.time.Clock = lambda: _FakeClock(self)
        pygame.event.get = self._events
        pygame.key.get_pressed = self._keys
        pygame.display.flip = self._flip
        return self

    def __exit__(self, *exc):
        pygame.time.Clock = self._orig["Clock"]
        pygame.event.get = self._orig["get"]
        pygame.key.get_pressed = self._orig["get_pressed"]
        pygame.display.flip = self._orig["flip"]
        return False


# --------------------------------------------------------------------------- snapshots
def _r(v, nd=6):
    return round(v, nd)


def _player_state(player) -> dict:
    return {
        "pos": (_r(player.position.x), _r(player.position.y), _r(player.position.z)),
        "yaw": _r(player.rotation_y), "pitch": _r(player.rotation_x),
        "gold": _r(player.gold), "torch": player.has_lit_torch, "crouch": player.is_crouching,
        "inventory": [i.id for i in player.inventory],
        "eye": _r(getattr(player, "current_eye_height", 1.6)),
        "skills": {k.value: {kk: _r(vv) for kk, vv in v.items()} for k, v in player.skills.skills.items()},
    }


def _world_state(world) -> dict:
    return {
        "hour": world.simulated_hour,
        "minute_accumulator": _r(world.minute_accumulator if hasattr(world, "minute_accumulator") else world.clock.minute_accumulator),
        "npcs": [{"name": n.name, "state": n.state.name, "suspicion": _r(n.suspicion_meter), "torch": n.has_torch,
                  "pos": (_r(n.position.x), _r(n.position.y), _r(n.position.z)), "inventory": [i.id for i in n.inventory]}
                 for n in world.npcs],
        "doors": [{"id": d.door_id, "locked": d.is_locked, "open": d.is_open} for d in world.doors],
    }


def snapshot_legacy(frame) -> dict:
    """Read the prototype's loop-local variables straight out of ``main()``'s frame."""
    assert frame.f_code.co_name == "main", frame.f_code.co_name
    L = frame.f_locals
    G = frame.f_globals
    in_trade = L["in_trade_menu"]
    return {
        "running": L["running"],
        "player": _player_state(L["player"]),
        **_world_state(L["world"]),
        "in_trade": in_trade,
        "merchant": L["active_merchant"].name if (in_trade and L["active_merchant"]) else None,
        "trade_selection": L["trade_selection"] if in_trade else None,
        "in_lockpicking": L["in_lockpicking"],
        "pick_angle": _r(L["lock_pick_angle"]), "sweet_spot": _r(L["lock_sweet_spot"]),
        "prompt": L["interaction_prompt"],
        "messages": list(G["message_log"]),
        "moving": L["is_moving"], "sprinting": L["is_sprinting"],
    }


def snapshot_new_app(app) -> dict:
    g = app.game
    return {
        "running": g.running,
        "screen": app.screen.name,
        "player": _player_state(g.player),
        **_world_state(g.world),
        "in_trade": g.in_trade_menu,
        "merchant": g.active_merchant.name if g.active_merchant else None,
        "trade_selection": (g.transfer.selection[g.transfer.side] if g.in_trade_menu else None),
        "in_lockpicking": g.in_lockpicking,
        "pick_angle": _r(g.lockpicking.pick_angle), "sweet_spot": _r(g.lockpicking.sweet_spot),
        "prompt": g.interaction_prompt,
        "messages": g.messages.messages,
        "moving": g.movement.is_moving, "sprinting": g.movement.is_sprinting,
    }


# --------------------------------------------------------------------------- runners
def _install_glu_fallback(legacy_main) -> None:
    """The prototype needs GLU; if libGLU is missing use the package's GL-only equivalents."""
    from OpenGL.GLU import gluLookAt, gluPerspective
    if bool(gluPerspective) and bool(gluLookAt):
        return
    from unprepared.rendering.camera import apply_look_at, apply_perspective
    legacy_main.gluPerspective = apply_perspective
    legacy_main.gluLookAt = lambda ex, ey, ez, cx, cy, cz, ux, uy, uz: apply_look_at((ex, ey, ez), (cx, cy, cz), (ux, uy, uz))


def run_legacy(scenario: Scenario, workdir: str, mask_rects=()) -> RunResult:
    from tests.conftest import load_baseline
    previous_cwd = os.getcwd()
    os.chdir(workdir)                       # the prototype loads its textures and writes its save in the CWD
    baseline = load_baseline()
    # main.py does ``__import__('models')`` at runtime, so keep the legacy modules importable.
    sys.modules.update({name: mod for name, mod in baseline.modules.items() if mod is not None})
    _install_glu_fallback(baseline.main)
    baseline.main.message_log.clear()
    error = None
    with ScriptedPygame(scenario, snapshot_legacy, mask_rects=mask_rects) as script:
        try:
            baseline.main.main()
        except SystemExit:
            pass
        except Exception as exc:  # the prototype crashed
            error = f"{type(exc).__name__}: {exc}"
        finally:
            pygame.quit()
            os.chdir(previous_cwd)
    return RunResult("legacy", scenario.name, script.records, error, _read_save(workdir))


def run_new(scenario: Scenario, workdir: str, mask_rects=()) -> RunResult:
    from unprepared.app import PygameApp
    from unprepared.config import GameSettings
    previous_cwd = os.getcwd()
    os.chdir(workdir)
    settings = GameSettings(echo_messages_to_stdout=False, random_seed=SEED)
    error = None
    app_holder = {}

    def snap(_frame):
        return snapshot_new_app(app_holder["app"])

    with ScriptedPygame(scenario, snap, mask_rects=mask_rects) as script:
        script.on_place = lambda fs: apply_frame_setup(app_holder["app"].game, fs) if "app" in app_holder else None
        try:
            from unprepared.graphics import GraphicsSettings
            app = PygameApp(settings, save_file=os.path.join(workdir, "savegame.json"), graphics=GraphicsSettings(),
                            skip_menu=True, auto_quality=False)
            if scenario.tutorial:
                from unprepared.game import Game
                app.game = Game(settings=settings, save_file=os.path.join(workdir, "savegame.json"), tutorial=True)
            app_holder["app"] = app
            app.run()
        except SystemExit:
            pass
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        finally:
            pygame.quit()
            os.chdir(previous_cwd)
    return RunResult("new", scenario.name, script.records, error, _read_save(workdir))


def apply_frame_setup(game, fs: FrameScript) -> None:
    """The game-only parts of a frame: a ``place`` and any ``skills``."""
    if fs.place is not None:
        place_player(game, fs.place)
    for name, level in (fs.skills or ()):
        skill = next(k for k in game.player.skills.skills if k.value == name)
        game.player.skills.skills[skill]["level"] = level


def place_player(game, place) -> None:
    """Apply a ``FrameScript.place``: the player on the floor at (x, z), facing (yaw, pitch)."""
    from unprepared.core.math3d import Vector3D
    x, z, yaw, pitch = place
    w = game.world
    game.player.position = Vector3D(x, w.floor_height_at(x, z, w.ground_height(x, z)), z)
    game.player.rotation_y, game.player.rotation_x = yaw, pitch


def _read_save(workdir: str) -> Optional[bytes]:
    path = os.path.join(workdir, "savegame.json")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return f.read()
    return None


def dry_run(scenario: Scenario, record: Optional[list] = None):
    """Run a scenario through the headless ``Game`` + real input controller (no window).
    Returns the Game; if ``record`` is a list, a ``Game.snapshot()`` per frame is appended to it."""
    from unprepared.config import GameSettings
    from unprepared.game import Game
    from unprepared.input import PygameInputController
    import tempfile
    # the scenario's quick saves go to a scratch directory that is removed afterwards (a run of the
    # regression suite replays every scenario, several times over)
    with tempfile.TemporaryDirectory(prefix="unprepared_dry_run_") as scratch:
        game = Game(settings=GameSettings(echo_messages_to_stdout=False, random_seed=SEED),
                    save_file=os.path.join(scratch, "savegame.json"), tutorial=scenario.tutorial)
        controller = PygameInputController()
        orig_get, orig_pressed = pygame.event.get, pygame.key.get_pressed
        try:
            for fs in scenario.frames:
                apply_frame_setup(game, fs)
                pygame.event.get = lambda fs=fs: [pygame.event.Event(t, **kw) for t, kw in fs.events]
                pygame.key.get_pressed = lambda fs=fs: _FakeKeys(fs.held)
                controller.process_events(game)
                game.update(fs.dt_ms / 1000.0, controller.poll_movement())
                if record is not None:
                    record.append(game.snapshot())
                if not game.running:
                    break
        finally:
            pygame.event.get, pygame.key.get_pressed = orig_get, orig_pressed
    return game
