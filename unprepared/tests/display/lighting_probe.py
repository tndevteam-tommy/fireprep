"""Subprocess body for the fixed-function lighting test: render frames with the *low* preset (no
shader: GL's own lights) and report where GL holds the sun and the nearest torch, in eye space,
against where the camera says they should be - with and without the view-space extras (the held
torch, the tutorial's arrow) drawn at the end of the frame before."""
from __future__ import annotations

import json
import os
import sys

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_DIR)
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")


def main() -> int:
    workdir = sys.argv[1]
    from OpenGL.GL import GL_LIGHT0, GL_LIGHT2, GL_POSITION, glGetLightfv
    from unprepared.app import PygameApp
    from unprepared.config import GameSettings
    from unprepared.game import Game
    from unprepared.graphics import GraphicsSettings
    from unprepared.rendering.atmosphere import compute_atmosphere

    settings = GameSettings(echo_messages_to_stdout=False, random_seed=1)
    graphics = GraphicsSettings()
    graphics.apply_preset("low")                       # the fixed-function pipeline
    app = PygameApp(settings, save_file=os.path.join(workdir, "s.json"), graphics=graphics, skip_menu=True,
                    auto_quality=False)

    def error(game) -> dict:
        """Largest difference between GL's eye-space sun / first point light and the camera's."""
        cam, _ = app.renderer.scene.camera_for(game.player)
        atm = compute_atmosphere(game.world.clock.fractional_hour, settings.lighting)
        sun = cam.to_eye_dir(atm.sun_dir)
        got_sun = glGetLightfv(GL_LIGHT0, GL_POSITION)
        out = {"sun": max(abs(float(a) - b) for a, b in zip(got_sun[:3], sun))}
        lights = app.renderer.lighting.point_lights
        if lights:
            point = cam.to_eye_point(lights[0].position)
            got = glGetLightfv(GL_LIGHT2, GL_POSITION)
            out["torch"] = max(abs(float(a) - b) for a, b in zip(got[:3], point))
        return out

    report = {"shaders": app.renderer.uses_shaders, "frames": []}
    game = app.game
    game.world.clock.hour = 10
    game.player.position = game.world.spot(0.0, 0.0, 3.0)          # on the ground there (the town is hilly)
    for yaw, torch in ((90.0, False), (90.0, True), (200.0, True), (200.0, False), (30.0, False)):
        game.player.rotation_y, game.player.has_lit_torch = yaw, torch
        app.step()
        report["frames"].append(error(game))
    # night, a torch in hand and the tutorial's arrow on screen: every frame ends in view space
    app.game = Game(settings=settings, save_file=os.path.join(workdir, "t.json"), tutorial=True)
    tgame = app.game
    tgame.player.has_lit_torch = True
    for yaw in (270.0, 0.0, 120.0):
        tgame.player.rotation_y = yaw
        app.step()
        report["frames"].append(error(tgame))
    report["waypoint"] = tgame.tutorial_waypoint is not None
    app.shutdown()
    print("REPORT " + json.dumps(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
