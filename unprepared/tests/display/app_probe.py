"""Subprocess body for the display smoke tests: run the app for N frames and report."""
from __future__ import annotations

import json
import os
import sys

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_DIR)
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")


def main() -> int:
    width, height, frames, asset_dir, workdir = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5]
    import pygame
    from OpenGL.GL import GL_RGB, GL_UNSIGNED_BYTE, glReadPixels
    from unprepared.app import PygameApp
    from unprepared.config import GameSettings, WindowSettings

    settings = GameSettings(window=WindowSettings(width=width, height=height, title="probe"),
                            asset_dir=asset_dir, echo_messages_to_stdout=False)
    app = PygameApp(settings, save_file=os.path.join(workdir, "probe_save.json"), skip_menu=True, auto_quality=False)
    for _ in range(frames):
        app.step()
    # Open every UI mode once so all HUD code paths draw at this resolution.
    from tests.helpers import place, stand_by_door
    g = app.game
    g.world.find_door("shop_door").is_locked = True
    stand_by_door(g, "shop_door", distance=1.2)                                    # in front of the shop door (locked)
    g.player.is_crouching = True
    g.update(0.016); g.interact(); app.renderer.render_frame(g); pygame.display.flip()
    assert g.in_lockpicking
    g.escape()
    # somebody talking within earshot: the HUD draws a speech bubble over their head
    jae = g.world.find_npc("Jae the Fence")
    import math
    yaw = math.radians(g.player.rotation_y)                                        # just in front of the player
    place(jae, g.world, g.player.position.x + math.sin(yaw), g.player.position.z - math.cos(yaw))
    jae.say("A line long enough to wrap onto a second row of the bubble, surely.", g.messages.add)
    g.update(0.016); assert g.speech_bubbles(); app.renderer.render_frame(g); pygame.display.flip()
    g.open_trade(g.world.find_npc("Jae the Fence")); app.renderer.render_frame(g); pygame.display.flip()
    g.close_transfer()
    g.open_pickpocket(g.world.find_npc("Guard Mira")); app.renderer.render_frame(g); pygame.display.flip()
    g.close_transfer()
    g.open_container(g.world.containers[0]); app.renderer.render_frame(g); pygame.display.flip()
    g.close_transfer()
    # night with torches, and the fixed-function fallback renderer
    g.world.clock.hour = 23
    g.player.has_lit_torch = True
    g.update(0.016); app.renderer.render_frame(g); pygame.display.flip()
    from unprepared.rendering.renderer import Renderer
    fallback = Renderer(settings, app.input.bindings.controls_help(), use_shaders=False)
    fallback.render_frame(g); pygame.display.flip()
    fallback.release()
    pixels = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)
    report = {
        "frames": g.frame_count, "hour": g.world.simulated_hour, "pixel_bytes": len(bytes(pixels)),
        "window": list(pygame.display.get_surface().get_size()), "caption": pygame.display.get_caption()[0],
        "shaders": app.renderer.uses_shaders,
        "missing_textures": sorted(app.renderer.textures.missing),
        "nonblack": sum(1 for b in bytes(pixels)[:300000] if b) > 0,
    }
    app.shutdown()
    print("REPORT " + json.dumps(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
