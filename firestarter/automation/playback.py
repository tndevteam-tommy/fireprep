"""
Firestarter - Scripted playback & input recording.

A playback script is JSON:

    {
      "name": "tutorial_walkthrough",
      "settings": {"width": 1280, "height": 720, "preset": "High", "seed": 7, "start_hour": 18.5,
                   "fixed_dt": 0.0333, "render_stride": 3},
      "steps": [ {"wait": 1.0}, {"aim_at": {"item": "Birch bark curls"}}, {"click": "left"},
                 {"assert": "world.inventory.count('tinder') >= 1"}, {"screenshot": "gathered"}, ... ]
    }

Input steps are injected as *real* pygame events (KEYDOWN/KEYUP, MOUSEBUTTONDOWN/UP,
MOUSEMOTION) into the SDL event queue, so they travel through exactly the same code path as
a player's keyboard and mouse.  Aiming and walking are done with mouse-motion and held-key
events; only steps explicitly named ``teleport`` / ``exec`` / ``set_time`` stage state
directly (for composing screenshots).

Step reference
  wait, wait_frames, key {hold}, key_down, key_up, click {hold}, mouse_down, mouse_up,
  mouse_move [dx, dy] {duration}, aim_at <target> {duration}, walk_to <target> {stop, timeout},
  saw {duration, amplitude, rate}, teleport [x, z] {look_at}, set_time {hour, season, day},
  weather {rain, cloud, wind, clear}, fast_forward <seconds>, wait_until <expr> {timeout},
  assert <expr> {message}, screenshot <name>, exec <python>, log <text>, expect_sound <name>

Targets: [x, y, z] | {"item": label} | {"tree": "nearest"} | {"pit": [dx, dy, dz]} |
         {"hearthboard": true} | {"rack": true} | {"woodpile": true}
"""

import json
import math
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np

from firestarter.simulation import terrain

BUTTONS = {"left": 1, "middle": 2, "right": 3, "wheel_up": 4, "wheel_down": 5}


class PlaybackError(Exception):
    pass


class Playback:
    def __init__(self, script: Dict[str, Any], out_dir: str, script_path: str = ""):
        self.script = script
        self.name = script.get("name", os.path.splitext(os.path.basename(script_path))[0] or "playback")
        self.steps: List[Dict[str, Any]] = list(script.get("steps", []))
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        s = script.get("settings", {})
        self.render_stride = int(s.get("render_stride", 1))
        self.index = 0
        self.t = 0.0
        self.frame = 0
        self.wait_left = 0.0
        self.wait_frames = 0
        self.pending: List[Dict[str, Any]] = []       # continuous actions (mouse moves, holds, saw)
        self.capture_requests: List[str] = []
        self.results: List[Dict[str, Any]] = []
        self.screenshots: List[str] = []
        self.log: List[str] = []
        self.failed = False
        self.done = False
        self.wall0 = time.time()
        self.until: Optional[Dict[str, Any]] = None
        self.render_now = True
        self.fatal: Optional[str] = None
        self.scratch: Dict[str, Any] = {}          # persists between exec / assert steps

    @staticmethod
    def load(path: str, out_dir: str) -> "Playback":
        with open(path) as f:
            return Playback(json.load(f), out_dir, path)

    # ------------------------------------------------------------------ environment
    def _env(self, game) -> Dict[str, Any]:
        st = game.state
        return {"world": st.world, "phone": st.phone, "book": st.book, "state": st, "game": game, "np": np,
                "math": math, "tutorial": st.tutorial, "input": st.input, "grid": st.world.thermal_grid,
                "inv": st.world.inventory, "player": st.world.player, "weather": st.world.weather,
                "foley": game.foley, "renderer": game.renderer, "scratch": self.scratch, "terrain": terrain}

    def _ns(self, game) -> Dict[str, Any]:
        """Persistent namespace for eval/exec steps (imports & helpers survive between steps)."""
        if not hasattr(self, "_exec_ns"):
            self._exec_ns: Dict[str, Any] = {"__builtins__": __builtins__}
        self._exec_ns.update(self._env(game))
        return self._exec_ns

    def _eval(self, expr: str, game) -> Any:
        return eval(expr, self._ns(game))

    # ------------------------------------------------------------------ event posting
    def _post(self, game, etype, **kw):
        pg = game.pg
        ev = pg.event.Event(etype, **kw)
        pg.event.post(ev)

    def _key(self, game, name: str, down: bool):
        pg = game.pg
        code = pg.key.key_code(name)
        self._post(game, pg.KEYDOWN if down else pg.KEYUP, key=code, mod=0, unicode="", scancode=0)

    def _button(self, game, name, down: bool):
        pg = game.pg
        b = BUTTONS.get(name, name) if isinstance(name, str) else int(name)
        self._post(game, pg.MOUSEBUTTONDOWN if down else pg.MOUSEBUTTONUP, button=b,
                   pos=(game.width // 2, game.height // 2))

    def _motion(self, game, dx: float, dy: float):
        pg = game.pg
        self._post(game, pg.MOUSEMOTION, pos=(game.width // 2, game.height // 2), rel=(dx, dy),
                   buttons=(0, 0, 0), synthetic=True)

    # ------------------------------------------------------------------ targets
    def _target_point(self, game, tgt) -> np.ndarray:
        w = game.state.world
        if isinstance(tgt, (list, tuple)):
            return np.array(tgt, dtype=float)
        if isinstance(tgt, dict):
            if "item" in tgt:
                label = tgt["item"].lower()
                for wi in w.world_items:
                    if label in wi.label.lower():
                        return wi.position.copy()
                raise PlaybackError(f"no world item matching '{tgt['item']}'")
            if "pit" in tgt:
                return w.fire_pit_pos + np.array(tgt["pit"], dtype=float)
            if "hearthboard" in tgt:
                return w.hearthboard_pos + np.array([0.0, 0.03, 0.0])
            if "rack" in tgt:
                return w.drying_rack.position + np.array([0.0, 0.7, 0.0])
            if "woodpile" in tgt:
                return w.woodpile_pos + np.array([0.0, 0.25, 0.0])
            if "tree" in tgt:
                t = w.ecosystem.nearest_tree(w.player.position, 60.0)
                if tgt.get("species"):
                    cands = [x for x in w.ecosystem.trees if x.is_standing and tgt["species"].lower() in x.species.value.lower()
                             and (not tgt.get("stage") or tgt["stage"].lower() in x.stage.value.lower())]
                    cands.sort(key=lambda x: np.linalg.norm(x.position[[0, 2]] - w.player.position[[0, 2]]))
                    t = cands[0] if cands else t
                if t is None:
                    raise PlaybackError("no tree nearby")
                return t.position + np.array([0.0, tgt.get("height", 1.0), 0.0])
            if "tree_id" in tgt:
                for t in w.ecosystem.trees:
                    if t.id == tgt["tree_id"]:
                        return t.position + np.array([0.0, 1.0, 0.0])
        raise PlaybackError(f"bad target {tgt}")

    def _aim_deltas(self, game, point: np.ndarray):
        p = game.state.world.player
        d = point - p.eye
        yaw = math.degrees(math.atan2(d[2], d[0]))
        pitch = math.degrees(math.atan2(d[1], math.hypot(d[0], d[2])))
        dyaw = (yaw - p.yaw + 180.0) % 360.0 - 180.0
        dpitch = pitch - p.pitch
        sens = game.state.settings.mouse_sensitivity
        inv = 1.0 if game.state.settings.invert_y else -1.0
        return dyaw / sens, (dpitch / sens) * inv

    # ------------------------------------------------------------------ frame hooks
    def before_frame(self, game, dt: float) -> bool:
        self.frame += 1
        self.t += dt
        self._run_pending(game, dt)
        if self.fatal:
            return False
        while not self.done:
            if self.wait_left > 0:
                self.wait_left -= dt
                break
            if self.wait_frames > 0:
                self.wait_frames -= 1
                break
            if self.until is not None:
                u = self.until
                try:
                    ok = bool(self._eval(u["expr"], game))
                except Exception as e:
                    ok = False
                    u["error"] = str(e)
                u["elapsed"] += dt
                if ok:
                    self.results.append({"step": u["step"], "type": "wait_until", "expr": u["expr"], "passed": True,
                                         "elapsed": round(u["elapsed"], 2)})
                    self.until = None
                    continue
                if u["elapsed"] > u["timeout"]:
                    self.results.append({"step": u["step"], "type": "wait_until", "expr": u["expr"], "passed": False,
                                         "elapsed": round(u["elapsed"], 2), "message": u.get("message", "timed out")})
                    self.failed = True
                    self.until = None
                    if u.get("fatal", True):
                        self.fatal = f"wait_until timed out: {u['expr']}"
                        return False
                    continue
                break
            if any(p.get("blocking") for p in self.pending):
                break
            if self.index >= len(self.steps):
                self.done = True
                break
            step = self.steps[self.index]
            self.index += 1
            try:
                self._execute(game, step, dt)
                if "screenshot" in step:
                    break          # capture this frame before executing anything else
            except Exception as e:
                self.failed = True
                self.results.append({"step": self.index, "type": "error", "passed": False, "message": f"{type(e).__name__}: {e}",
                                     "spec": step})
                self.fatal = str(e)
                return False
        # rendering cadence
        need = bool(self.capture_requests) or (self.frame % max(1, self.render_stride) == 0)
        self.render_now = need
        if self.done and not self.capture_requests and not self.pending:
            return False
        return True

    def _execute(self, game, step: Dict[str, Any], dt: float):
        st = game.state
        w = st.world
        if "wait" in step:
            self.wait_left = float(step["wait"])
        elif "wait_frames" in step:
            self.wait_frames = int(step["wait_frames"])
        elif "key" in step:
            self._key(game, step["key"], True)
            self.pending.append({"kind": "release_key", "name": step["key"], "t": float(step.get("hold", 0.05)), "blocking": True})
        elif "key_down" in step:
            self._key(game, step["key_down"], True)
        elif "key_up" in step:
            self._key(game, step["key_up"], False)
        elif "click" in step:
            self._button(game, step["click"], True)
            self.pending.append({"kind": "release_button", "name": step["click"], "t": float(step.get("hold", 0.05)), "blocking": True})
        elif "mouse_down" in step:
            self._button(game, step["mouse_down"], True)
        elif "mouse_up" in step:
            self._button(game, step["mouse_up"], False)
        elif "mouse_move" in step:
            dx, dy = step["mouse_move"]
            dur = float(step.get("duration", 0.25))
            self.pending.append({"kind": "move", "dx": dx, "dy": dy, "t": dur, "total": dur, "blocking": True})
        elif "aim_at" in step:
            pt = self._target_point(game, step["aim_at"])
            if "offset" in step:
                pt = pt + np.array(step["offset"], dtype=float)
            dur = float(step.get("duration", 0.35))
            self.pending.append({"kind": "aim", "point": pt, "t": dur, "total": dur, "blocking": True})
        elif "walk_to" in step:
            pt = self._target_point(game, step["walk_to"])
            self.pending.append({"kind": "walk", "point": pt, "stop": float(step.get("stop", 1.2)),
                                 "t": float(step.get("timeout", 20.0)), "blocking": True, "walking": False})
        elif "saw" in step:
            sp = step["saw"]
            self.pending.append({"kind": "saw", "t": float(sp.get("duration", 10.0)), "amp": float(sp.get("amplitude", 140.0)),
                                 "rate": float(sp.get("rate", 1.4)), "phase": 0.0, "blocking": True})
        elif "teleport" in step:
            x, z = step["teleport"]
            w.player.position[0], w.player.position[2] = float(x), float(z)
            w.player.velocity[:] = 0.0
            w.player.snap_to_ground()
            if "look_at" in step:
                w.player.look_at(self._target_point(game, step["look_at"]))
            if "yaw" in step:
                w.player.yaw = float(step["yaw"])
            if "pitch" in step:
                w.player.pitch = float(step["pitch"])
        elif "set_time" in step:
            from firestarter.simulation.weather import Season
            sp = step["set_time"]
            season = Season(sp["season"]) if "season" in sp else None
            w.weather.set_time(float(sp.get("hour", w.weather.time_of_day_hours)), season, sp.get("day"))
        elif "weather" in step:
            sp = step["weather"]
            if sp.get("clear"):
                w.weather.force_precipitation(0.0)
                w.weather.forced_cloud = 0.05
                w.weather.cloud_cover = 0.05
            if "rain" in sp:
                w.weather.force_precipitation(float(sp["rain"]))
            if "cloud" in sp:
                w.weather.forced_cloud = float(sp["cloud"])
                w.weather.cloud_cover = float(sp["cloud"])
            if "wind" in sp:
                w.weather.forced_wind = float(sp["wind"])
            if "fog" in sp:
                w.weather.forced_fog = float(sp["fog"]) if float(sp["fog"]) > 0 else None
                w.weather.fog_density = float(sp["fog"])
            if "wetness" in sp:
                w.weather.ground_wetness = float(sp["wetness"])
            w.weather.update(0.0)
        elif "fast_forward" in step:
            secs = float(step["fast_forward"])
            h = float(step.get("dt", 0.05))
            n = int(secs / h)
            for _ in range(n):
                st.step(h, move=False)
                ev = w.pop_events()
                game.renderer.push_events([e for e in ev if e[0] not in ("collapse",)])
                game.foley.handle_events(ev, w)
        elif "wait_until" in step:
            self.until = {"expr": step["wait_until"], "timeout": float(step.get("timeout", 15.0)), "elapsed": 0.0,
                          "step": self.index, "message": step.get("message", ""), "fatal": step.get("fatal", True)}
        elif "assert" in step:
            expr = step["assert"]
            try:
                ok = bool(self._eval(expr, game))
                err = ""
            except Exception as e:
                ok, err = False, f"{type(e).__name__}: {e}"
            res = {"step": self.index, "type": "assert", "expr": expr, "passed": ok, "message": step.get("message", "")}
            if err:
                res["error"] = err
            if "value" in step:
                try:
                    res["value"] = repr(self._eval(step["value"], game))
                except Exception as e:
                    res["value"] = f"<{e}>"
            self.results.append(res)
            if not ok:
                self.failed = True
        elif "screenshot" in step:
            self.capture_requests.append(step["screenshot"])
        elif "exec" in step:
            exec(step["exec"], self._ns(game))
        elif "log" in step:
            self.log.append(f"[{self.t:7.2f}s] {step['log']}")
        elif "expect_sound" in step:
            name = step["expect_sound"]
            ok = any(p == name or p.startswith(name) for p in game.foley.played)
            self.results.append({"step": self.index, "type": "expect_sound", "expr": name, "passed": ok})
            if not ok:
                self.failed = True
        else:
            raise PlaybackError(f"unknown step {step}")

    def _run_pending(self, game, dt: float):
        keep = []
        for p in self.pending:
            k = p["kind"]
            if k in ("release_key", "release_button"):
                p["t"] -= dt
                if p["t"] <= 0:
                    if k == "release_key":
                        self._key(game, p["name"], False)
                    else:
                        self._button(game, p["name"], False)
                    continue
            elif k == "move":
                frac = min(1.0, dt / max(p["t"], dt))
                dx, dy = p["dx"] * frac, p["dy"] * frac
                p["dx"] -= dx
                p["dy"] -= dy
                self._motion(game, dx, dy)
                p["t"] -= dt
                if p["t"] <= 1e-6:
                    continue
            elif k == "aim":
                dx, dy = self._aim_deltas(game, p["point"])
                frac = min(1.0, dt / max(p["t"], dt))
                self._motion(game, dx * frac, dy * frac)
                p["t"] -= dt
                if p["t"] <= 1e-6:
                    dx, dy = self._aim_deltas(game, p["point"])
                    if abs(dx) > 0.5 or abs(dy) > 0.5:
                        self._motion(game, dx, dy)
                    continue
            elif k == "walk":
                pl = game.state.world.player
                dist = float(np.hypot(p["point"][0] - pl.position[0], p["point"][2] - pl.position[2]))
                dx, _ = self._aim_deltas(game, np.array([p["point"][0], pl.eye[1], p["point"][2]]))
                if abs(dx) > 1.0:
                    self._motion(game, dx * 0.5, 0.0)
                if dist > p["stop"] and p["t"] > 0:
                    if not p["walking"]:
                        self._key(game, "w", True)
                        p["walking"] = True
                    p["t"] -= dt
                else:
                    if p["walking"]:
                        self._key(game, "w", False)
                    continue
            elif k == "saw":
                p["phase"] += dt * p["rate"] * 2 * math.pi
                v = math.cos(p["phase"]) * p["amp"] * p["rate"] * 2 * math.pi * dt
                self._motion(game, v, 0.0)
                p["t"] -= dt
                if p["t"] <= 0:
                    continue
            keep.append(p)
        self.pending = keep

    def after_render(self, game):
        for name in self.capture_requests:
            path = os.path.join(self.out_dir, f"{self.name}__{name}.png")
            game.save_screenshot(path)
            self.screenshots.append(path)
        self.capture_requests = []

    def finish(self, game) -> int:
        if not self.done and not self.fatal:
            self.failed = True
            self.results.append({"type": "incomplete", "passed": False, "message": f"stopped at step {self.index}/{len(self.steps)}"})
        if self.fatal:
            self.failed = True
        perf = game.perf or [0.0]
        report = {
            "name": self.name,
            "passed": not self.failed,
            "fatal": self.fatal,
            "steps_executed": self.index,
            "steps_total": len(self.steps),
            "sim_time_s": round(self.t, 2),
            "frames": self.frame,
            "wall_time_s": round(time.time() - self.wall0, 2),
            "render_ms_mean": round(float(np.mean(perf)), 2),
            "render_ms_p95": round(float(np.percentile(perf, 95)), 2),
            "gl_renderer": getattr(game.renderer, "gl_renderer", "?"),
            "results": self.results,
            "screenshots": [os.path.basename(s) for s in self.screenshots],
            "log": self.log,
            "final_state": game.state.snapshot(),
            "sounds_played": sorted(set(game.foley.played)),
        }
        with open(os.path.join(self.out_dir, f"{self.name}__report.json"), "w") as f:
            json.dump(report, f, indent=2, default=str)
        status = "PASS" if report["passed"] else "FAIL"
        n_ok = sum(1 for r in self.results if r.get("passed"))
        print(f"[Playback] {self.name}: {status}  ({n_ok}/{len(self.results)} checks, {len(self.screenshots)} screenshots, "
              f"{report['wall_time_s']} s wall, {report['render_ms_mean']} ms/frame)")
        for r in self.results:
            if not r.get("passed"):
                print(f"    FAILED step {r.get('step')}: {r.get('type')} {r.get('expr', '')} {r.get('message', '')} {r.get('error', '')}")
        return 0 if report["passed"] else 1


class Recorder:
    """Records the raw input event stream so a play session can be replayed exactly."""

    def __init__(self, path: str, fixed_dt: float):
        self.path = path
        self.fixed_dt = fixed_dt
        self.events: List[Dict[str, Any]] = []

    def record(self, frame: int, event):
        import pygame as pg
        t = event.type
        if t in (pg.KEYDOWN, pg.KEYUP):
            self.events.append({"frame": frame, "type": "key_down" if t == pg.KEYDOWN else "key_up", "name": pg.key.name(event.key)})
        elif t in (pg.MOUSEBUTTONDOWN, pg.MOUSEBUTTONUP):
            self.events.append({"frame": frame, "type": "mouse_down" if t == pg.MOUSEBUTTONDOWN else "mouse_up", "button": event.button})
        elif t == pg.MOUSEMOTION:
            self.events.append({"frame": frame, "type": "motion", "rel": list(event.rel)})

    def save(self):
        steps: List[Dict[str, Any]] = []
        last = 0
        for e in self.events:
            if e["frame"] > last:
                steps.append({"wait_frames": e["frame"] - last})
                last = e["frame"]
            if e["type"] == "key_down":
                steps.append({"key_down": e["name"]})
            elif e["type"] == "key_up":
                steps.append({"key_up": e["name"]})
            elif e["type"] == "mouse_down":
                steps.append({"mouse_down": e["button"]})
            elif e["type"] == "mouse_up":
                steps.append({"mouse_up": e["button"]})
            else:
                steps.append({"mouse_move": e["rel"], "duration": 0.0})
        data = {"name": os.path.splitext(os.path.basename(self.path))[0],
                "settings": {"fixed_dt": self.fixed_dt}, "steps": steps}
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(data, f, indent=1)
        print(f"[Recorder] {len(self.events)} input events saved to {self.path}")
