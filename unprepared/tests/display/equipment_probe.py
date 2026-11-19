"""Subprocess body for the equipment-view smoke test: open the [Tab] view in the real app with a
full kit on, render it, and report what the stand shows (a figure that changes when turned)."""
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
    from OpenGL.GL import GL_RGB, GL_UNSIGNED_BYTE, glReadPixels
    from unprepared.app import PygameApp
    from unprepared.config import GameSettings
    from unprepared.entities import Item
    from unprepared.entities.equipment import gear_for
    from unprepared.graphics import GraphicsSettings

    settings = GameSettings(echo_messages_to_stdout=False, random_seed=1)
    app = PygameApp(settings, save_file=os.path.join(workdir, "s.json"), graphics=GraphicsSettings(), skip_menu=True,
                    auto_quality=False)
    game = app.game
    w, h = settings.window.width, settings.window.height
    for item_id, name in (("iron_helm", "Iron Helm"), ("plate_cuirass", "Plate Cuirass"), ("short_sword", "Short Sword"),
                          ("wool_cloak", "Wool Cloak")):
        spec = gear_for(item_id)
        game.player.equip(game.player.add_item(Item(item_id, name, value=10.0, gear=spec.gear, gear_color=spec.color, owner="player")))
    game.update(0.016)

    def frame_pixels():
        app.step()
        data = bytes(glReadPixels(0, 0, w, h, GL_RGB, GL_UNSIGNED_BYTE))
        x1, y1, x2, y2 = app.renderer.hud.equipment_preview_rect()
        rows = []
        for y in range(y1, y2):                       # GL rows are bottom-up
            start = ((h - 1 - y) * w + x1) * 3
            rows.append(data[start:start + (x2 - x1) * 3])
        return b"".join(rows)

    report = {"opened": game.open_equipment(), "mode": game.mode.value, "mouse_captured": game.mouse_captured}
    game.preview_spin = False
    game.preview_yaw = 0.0
    front = frame_pixels()
    game.preview_yaw = 90.0
    side = frame_pixels()
    backdrop = bytes((round(0.16 * 255), round(0.17 * 255), round(0.22 * 255)))   # the stand as the HUD painted it
    report["stand_pixels"] = len(front) // 3
    report["figure_pixels_front"] = sum(1 for i in range(0, len(front), 3) if front[i:i + 3] != backdrop)
    report["figure_pixels_side"] = sum(1 for i in range(0, len(side), 3) if side[i:i + 3] != backdrop)
    report["turns"] = front != side
    report["worn"] = {slot.value: item.id for slot, item in game.player.equipment.items()}
    game.equipment_cursor = next(i for i, it in enumerate(game.player.inventory) if it.id == "iron_helm")
    game.equipment_toggle()
    bare = frame_pixels()
    report["helm_off_changes_the_figure"] = bare != side
    game.tab()
    report["closed"] = not game.in_equipment and game.mouse_captured
    app.step()
    app.shutdown()
    print("REPORT " + json.dumps(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
