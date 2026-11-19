"""
Firestarter - UI Renderer (PIL -> premultiplied RGBA texture)
DESIGN Sections 5, 6, 8: HUD, Wilderness Smartphone OS (Store & Topo Map), Firemaster's Field
Compendium, Time Passage menu, pause / settings menu, help overlay and time-skip report.

The UI is authored at a 1280x720 reference size and scaled to the window.  It is only
re-rendered when its *content signature* changes (rounded values + menu state), which keeps
the per-frame cost negligible.
"""

import math
import time as _time
from typing import Tuple, Dict, Any, Optional, List

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from firestarter.forestry.bucking import SECTION_LEN
from firestarter.rendering.ui_theme import PAL, font, text_w, wrap, panel, key_cap, icon, TOOL_ICONS, temp_color, lerp_col, comp_shape
from firestarter.simulation import terrain

OFFHAND_ICONS = {"tinder": "tinder", "kindling": "kindling", "firewood": "firewood", "log": "log", "fungus": "fungus"}
OFFHAND_LABELS = {"tinder": "Tinder", "kindling": "Kindling", "firewood": "Firewood", "log": "Log", "fungus": "Fungus",
                  "none": "Nothing"}
TOOL_NAMES = {"pocket_lighter": "Lighter", "pocket_knife": "Knife", "felling_axe": "Axe", "bow_drill_spindle": "Bow Drill",
              "accelerant": "Gasoline", "none": "Empty hand"}
STAGE_COLORS = {
    "Cold Hearth": (140, 150, 160, 255), "Cold Ashes": (150, 150, 150, 255), "Ambient": (140, 150, 160, 255),
    "Dehydration": (200, 225, 240, 255), "Pyrolysis": (235, 200, 95, 255), "Flaming Combustion": (255, 140, 45, 255),
    "Char Oxidation (Glowing Coals)": (255, 85, 40, 255), "Inert Ash": (170, 170, 170, 255),
}


class UIRenderer:
    def __init__(self, width: int = 1280, height: int = 720):
        self.width = width
        self.height = height
        self.s = min(width / 1280.0, height / 720.0)
        self._sig = None
        self._last_time = 0.0
        self._paper_cache = {}
        self._map_cache = {}
        self.last_image: Optional[Image.Image] = None
        self.min_interval = 0.05

    def resize(self, w: int, h: int):
        self.width, self.height = w, h
        self.s = min(w / 1280.0, h / 720.0)
        self._sig = None
        self._paper_cache.clear()
        self._map_cache.clear()

    # ------------------------------------------------------------------ helpers
    def S(self, v: float) -> int:
        return int(round(v * self.s))

    def F(self, name: str, size: float):
        return font(name, max(8, int(round(size * self.s))))

    # ------------------------------------------------------------------ signature & caching
    def _signature(self, world, player, smartphone, ref_book, input_mgr, eng, extra) -> Tuple:
        g = world.thermal_grid
        w = world.weather
        tgt = world.target
        tut = input_mgr.tutorial.state() if getattr(input_mgr, "tutorial", None) else {}
        nest = world.crafting.active_tinder_nest
        bd = world.crafting.bow_drill
        t = _time.perf_counter()
        anim = int(t * 8) if (world.toasts or tut.get("flash", 0) > 0 or world.breath_start is not None
                              or smartphone.is_open or (nest is not None)) else 0
        return (
            self.width, self.height, anim,
            round(g.core_coal_temp / 5.0), round(g.current_heat_flux / 200.0), round(g.flame_height, 2),
            round(g.draft_velocity, 1), round(g.oxygen_fraction * 200), g.fire_stage(), len(g.bodies),
            w.clock_string(), round(w.ambient_temp_c), w.describe(), w.current_season.value, int(w.day_of_season),
            tuple((t_["text"], round(t_["time"], 1) if t_["time"] < 1.0 else 1) for t_ in world.toasts),
            tgt["kind"] if tgt else None, id(tgt["obj"]) if tgt else None, tgt["label"] if tgt else None,
            self._target_detail(tgt),
            world.ghost is not None and world.ghost.get("slot"), world.current_offhand(), player.primary_hand,
            tuple(sorted(world.inventory.summary()["fuel"].items())), world.inventory.lighter_starts,
            round(world.inventory.lighter_burn_left), round(player.stamina / 4), round(world.breath_charge, 1),
            round(smartphone.credits.credits), round(smartphone.credits.pending, 1),
            smartphone.is_open, smartphone.active_app.value, smartphone.store.selected_index, smartphone.home_selection,
            smartphone.status_message, round(smartphone.map_zoom, 2),
            ref_book.is_open, ref_book.current_page, ref_book.selected_blueprint_key, ref_book.status,
            world.time_engine.is_menu_open, world.time_engine.selected_option_index, getattr(world, "pending_time_skip", None),
            input_mgr.is_paused, input_mgr.pause_index, tuple(sorted(input_mgr.settings.as_dict().items())),
            input_mgr.show_help, tut.get("index"), round(tut.get("flash", 0), 1),
            round(bd.socket_temp / 4.0) if (world.bow_active or bd.has_ember or bd.socket_temp > 40) else 0,
            bd.has_ember, round(bd.cordage_condition, 2), world.bow_active, getattr(world, "friction_mode", "bow"), round(world.crafting.hand_drill.hand_height_cm), world.secondary_held,
            (round(nest.core_temp / 4.0), nest.is_flaming, round(nest.glow_area_cm2, 1)) if nest else None,
            world.structure_key, round(world.structure_score, 1),
            id(world.last_skip_report) if world.last_skip_report else None,
            (int(eng.last_frame_ms) if input_mgr.settings.show_fps else 0),
            tuple(sorted((extra or {}).items())) if extra else None,
        )

    @staticmethod
    def _target_detail(tgt):
        """Changing state of the aimed-at object that the prompt text depends on."""
        if not tgt:
            return None
        obj = tgt.get("obj")
        if tgt["kind"] == "fallen_trunk":
            return (obj.nearest_boundary(min(tgt.get("s", 0.0), obj.bole_length)), round(float(obj.cut_depth.sum()), 3),
                    obj.rounds_left, obj.limbs_left)
        if tgt["kind"] == "fallen_limb":
            return (tgt["limb"].id, round(tgt["limb"].cut_depth, 3))
        if tgt["kind"] == "fuel":
            return getattr(obj, "split_blows", 0)
        return None

    def render_if_changed(self, world, player, hands, smartphone, ref_book, input_mgr, eng=None, extra=None):
        sig = self._signature(world, player, smartphone, ref_book, input_mgr, eng, extra)
        now = _time.perf_counter()
        if sig == self._sig and self.last_image is not None:
            return None, False
        self._sig = sig
        self._last_time = now
        img = self.render_image(world, player, hands, smartphone, ref_book, input_mgr, eng, extra)
        self.last_image = img
        arr = np.asarray(img, dtype=np.uint16)
        a = arr[:, :, 3:4]
        pre = np.concatenate([(arr[:, :, :3] * a // 255), a], axis=2).astype(np.uint8)
        return pre.tobytes(), True

    def render(self, world, player, hands, smartphone, ref_book, input_mgr) -> Tuple[bytes, int, int]:
        """Legacy API: straight RGBA bytes."""
        img = self.render_image(world, player, hands, smartphone, ref_book, input_mgr, None, None)
        return img.tobytes(), self.width, self.height

    # ------------------------------------------------------------------ composition
    def render_image(self, world, player, hands, smartphone, ref_book, input_mgr, eng, extra) -> Image.Image:
        img = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        menu = smartphone.is_open or ref_book.is_open or world.time_engine.is_menu_open or input_mgr.is_paused
        if not menu:
            self._draw_hud(img, d, world, player, smartphone, input_mgr, eng)
        else:
            self._draw_minimal_hud(d, world, smartphone)
        if smartphone.is_open:
            self._draw_phone(img, d, world, smartphone, player)
        if ref_book.is_open:
            self._draw_book(img, d, world, ref_book)
        if world.time_engine.is_menu_open:
            self._draw_time_menu(d, world)
        if getattr(world, "pending_time_skip", None) is not None:
            self._draw_time_passing(d, world.pending_time_skip)
        if input_mgr.is_paused:
            self._draw_pause(d, input_mgr)
        if input_mgr.show_help and not menu:
            self._draw_help(d)
        if input_mgr.settings.show_fps and eng is not None:
            f = self.F("mono", 12)
            d.text((self.S(8), self.height - self.S(20)), f"{eng.last_frame_ms:5.1f} ms  {getattr(eng, 'gl_renderer', '')[:30]}",
                   font=f, fill=PAL["dim"])
        return img

    # ================================================================== HUD
    def _draw_minimal_hud(self, d, world, phone):
        pass

    def _draw_hud(self, img, d, world, player, phone, input_mgr, eng):
        W, H = self.width, self.height
        S = self.S
        self._draw_crosshair(d, world, player)
        self._draw_time_card(d, world, player)
        self._draw_credits(d, phone)
        self._draw_tutorial(d, input_mgr)
        self._draw_toasts(d, world, input_mgr)
        self._draw_hotbar(d, world, player)
        self._draw_vitals(d, world, player)
        g = world.thermal_grid
        near_fire = np.linalg.norm(player.feet[[0, 2]] - g.pit_center[[0, 2]]) < 7.0
        has_fire = bool(g.bodies) or g.coal_bed.char_kg > 0.003 or g.is_burning
        if has_fire and (near_fire or g.is_burning):
            self._draw_fire_panel(d, world)
        if world.bow_active or world.crafting.bow_drill.has_ember or world.crafting.bow_drill.socket_temp > 45.0:
            self._draw_bow_gauge(d, world)
        nest = world.crafting.active_tinder_nest
        if nest is not None and (nest.has_ember or nest.is_flaming):
            self._draw_nest_gauge(d, world, nest)
        rep = world.last_skip_report
        if rep is not None and world.notification_timer > 0.0 and "passed" in world.notification:
            self._draw_skip_report(d, rep)

    def _draw_crosshair(self, d, world, player):
        cx, cy = self.width // 2, self.height // 2
        S = self.S
        tgt = world.target
        col = (240, 240, 240, 200)
        if tgt is not None and tgt["kind"] not in ("ground",):
            col = PAL["amber"]
        if world.ghost is not None:
            col = PAL["green"] if world.ghost["valid"] else PAL["red"]
        r = S(2.2)
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=col)
        if tgt is not None and tgt["kind"] not in ("ground",):
            rr = S(9)
            d.arc([cx - rr, cy - rr, cx + rr, cy + rr], 20, 160, fill=col, width=max(1, S(1.5)))
            d.arc([cx - rr, cy - rr, cx + rr, cy + rr], 200, 340, fill=col, width=max(1, S(1.5)))
        lines = self._prompts(world, player)
        if not lines:
            return
        y = cy + S(92)
        fk = self.F("bold", 11)
        ft = self.F("medium", 13)
        fs = self.F("regular", 11.5)
        for kind, a, b in lines:
            if kind == "title":
                w = text_w(d, a, ft)
                d.text((cx - w / 2 + 1, y + 1), a, font=ft, fill=(0, 0, 0, 160))
                d.text((cx - w / 2, y), a, font=ft, fill=PAL["text"])
                y += S(18)
                if b:
                    w = text_w(d, b, fs)
                    d.text((cx - w / 2 + 1, y + 1), b, font=fs, fill=(0, 0, 0, 160))
                    d.text((cx - w / 2, y), b, font=fs, fill=PAL["dim"])
                    y += S(17)
            else:
                total = text_w(d, a, fk) + 10 + S(6) + text_w(d, b, fs)
                x = cx - total / 2
                x += key_cap(d, x, y, a, fk) + S(6)
                d.text((x + 1, y + 1), b, font=fs, fill=(0, 0, 0, 170))
                d.text((x, y), b, font=fs, fill=PAL["text"])
                y += S(20)

    def _prompts(self, world, player) -> List[Tuple[str, str, str]]:
        tgt = world.target
        out = []
        tool = player.primary_hand
        off = world.current_offhand()
        inv = world.inventory
        if tgt is None or tgt["kind"] == "ground":
            if tgt is not None and off in ("log", "firewood", "kindling") and inv.count(off):
                out.append(("key", "RMB", f"Set {off} down here"))
            return out
        k = tgt["kind"]
        obj = tgt.get("obj")
        if k == "tinder_source":
            out.append(("title", obj.label, f"{obj.count} bundle(s) left" if obj.count else "picked clean - regrows in a few days"))
            if obj.count:
                out.append(("key", "LMB", "Gather tinder"))
        elif k == "kindling_pile":
            out.append(("title", obj.label, f"{obj.count} twigs"))
            out.append(("key", "LMB", "Gather 3 twigs"))
        elif k == "woodpile":
            stock = getattr(obj, "stock", [])
            logs = sum(1 for s_ in stock if s_.kind == "log")
            out.append(("title", "Woodpile - seasoned fuel", f"{len(stock) - logs} split firewood, {logs} logs"))
            out.append(("key", "LMB", "Take firewood (select Log [9] first to take a log)"))
        elif k == "fuel" and obj.fuel is not None:
            f = obj.fuel
            out.append(("title", f.name.capitalize(), f"Moisture {f.moisture * 100:.0f}%  ({f.moisture_label()})  -  surface {f.m_surface * 100:.0f}% / core {f.m_core * 100:.0f}%"))
            if tool == "felling_axe" and f.kind == "log":
                n = int(max(2, min(12, round((f.radius / 0.055) ** 2))))
                blows = max(1, n // 3)
                done = getattr(obj, "split_blows", 0)
                out.append(("key", "LMB", f"Split into {n} pieces" + (f"  -  blow {done + 1}/{blows}" if blows > 1 else "")))
            elif f.mass_kg > 25:
                out.append(("key", "3", f"{f.mass_kg:.0f} kg - split it with the axe before carrying"))
            else:
                out.append(("key", "LMB", "Pick up"))
        elif k == "rack":
            rack = world.drying_rack
            mm = rack.mean_moisture()
            sub = f"{len(rack.logs)}/{rack.capacity} pieces" + (f", mean moisture {mm * 100:.0f}%" if mm is not None else "")
            sub += "  -  tarp: " + ("yes" if rack.has_tarp else "no")
            out.append(("title", "Raised drying rack", sub))
            if inv.tarps > 0 and not rack.has_tarp:
                out.append(("key", "LMB", "Stretch the tarp over the rack"))
            elif rack.logs:
                out.append(("key", "LMB", "Take the driest piece"))
            if off in ("log", "firewood") and inv.count(off):
                out.append(("key", "RMB", f"Stack {off} to season"))
        elif k == "hearthboard":
            bd = world.crafting.bow_drill
            state = "V-notch carved" if bd.has_v_notch else "no V-notch yet"
            out.append(("title", "Bow-drill kit & hearthboard", f"{state}  -  cord {bd.cordage_condition * 100:.0f}%  (+{inv.cordage} spare)"))
            if tool == "pocket_knife":
                out.append(("key", "LMB", "Carve a 45-degree V-notch"))
            elif tool == "bow_drill_spindle" and getattr(world, "friction_mode", "bow") == "hand":
                out.append(("key", "LMB", "Hold + rub the mouse left/right fast to spin"))
                out.append(("key", "RMB", "Press with the river stone / tip the ember"))
                out.append(("key", "4", "Switch to the bow drill"))
            elif tool == "bow_drill_spindle":
                out.append(("key", "LMB", "Hold + saw the mouse left/right to drill"))
                out.append(("key", "RMB", "Bear down / tip the ember into tinder"))
                out.append(("key", "4", "Switch to the hand drill"))
            else:
                out.append(("key", "4", "Equip the friction-fire kit"))
        elif k == "fallen_trunk":
            ft = obj
            sp = ft.species.value.lower()
            out.append(("title", f"Felled {sp}", f"{ft.rounds_left} rounds & {ft.limbs_left} limbs left  -  "
                        f"{ft.moisture() * 100:.0f}% moisture (green)"))
            i = ft.nearest_boundary(min(tgt.get("s", 0.0), ft.bole_length))
            if tool != "felling_axe":
                out.append(("key", "3", "Take the axe to limb & buck it"))
            elif i is not None:
                s_cut = i * SECTION_LEN
                if ft.blocking_limb(s_cut) is not None or (i == ft.n_sections and any(
                        not l.removed and l.s > s_cut - 0.35 for l in ft.limbs)):
                    out.append(("key", "LMB", "Branches in the way - limb this part first"))
                else:
                    need = 2.0 * ft.radius_at(s_cut) * 0.9
                    what = "Cut off the leader" if i == ft.n_sections else "Buck a 45 cm round"
                    out.append(("key", "LMB", f"{what}  -  kerf {ft.cut_depth[i] * 100:.0f}/{need * 100:.0f} cm"))
        elif k == "fallen_limb":
            ft, limb = obj, tgt["limb"]
            y = ft.limb_yield(limb)
            out.append(("title", f"{limb.radius * 200:.0f} cm limb of the felled {ft.species.value.lower()}",
                        f"{len(y['fire_logs'])} fire logs, {y['kindling']} kindling sticks, {y['tinder']} handfuls of tinder"))
            if tool == "felling_axe":
                need = 2.0 * limb.radius * 0.85
                out.append(("key", "LMB", f"Limb it off at the trunk  -  {min(99, limb.cut_depth / need * 100):.0f}%"))
            else:
                out.append(("key", "3", "Take the axe to limb it"))
        elif k == "tree":
            t = obj
            out.append(("title", t.describe(), f"trunk {t.trunk_radius_m * 200:.0f} cm, ~{t.height_m:.0f} m tall"))
            if tool == "felling_axe":
                stage = "face notch" if t.notch_depth_m < t.trunk_radius_m * 0.7 - 1e-9 else "back-cut"
                out.append(("key", "LMB", f"Chop ({stage}) - {inv.axe} axe"))
            elif tool == "pocket_knife":
                if t.has_fungi and t.fungi_count > 0:
                    out.append(("key", "LMB", "Cut a tinder-fungus bracket"))
                elif t.bark_strips_left > 0:
                    out.append(("key", "LMB", "Strip inner bark for cordage"))
            else:
                out.append(("key", "3", "Axe to fell  -  2 knife for bark / fungus"))
        elif k in ("fire_pit", "pit_fuel"):
            g = world.thermal_grid
            if k == "pit_fuel":
                b = obj
                out.append(("title", f"{b.species_type.value} {b.geometry_type}",
                            f"{b.stage.value}  -  {b.t_surface:.0f} C surface  -  moisture {b.m_surface * 100:.0f}% / {b.m_core * 100:.0f}%"))
            else:
                out.append(("title", "Fire pit", f"{g.fire_stage()}  -  {len(g.bodies)} pieces"))
            gh = world.ghost
            if off == "fungus":
                out.append(("key", "RMB", "Light / use the fungus ember"))
            elif gh is not None:
                item = inv.peek(off)
                if item is not None:
                    how = f"slot {gh['slot']}" if gh.get("slot") else ("lean" if gh["orientation"][0] > 10 else "lay")
                    out.append(("key", "RMB", f"Place {item.name.lower()} ({how}, {item.moisture * 100:.0f}% MC)"))
                    out.append(("key", "Wheel", "Rotate   Shift+Wheel tilt   MMB reset"))
                else:
                    out.append(("key", "RMB", f"No {off} in your pack"))
            if tool == "pocket_lighter":
                out.append(("key", "LMB", f"Hold the lighter flame  ({inv.lighter_starts} starts left)"))
            elif tool == "accelerant" and inv.gasoline_kg > 0:
                out.append(("key", "LMB", f"Pour gasoline ({inv.gasoline_kg:.1f} kg) - never on a lit fire!"))
            elif k == "pit_fuel" and obj.t_surface < 70 and not obj.is_flaming:
                out.append(("key", "LMB", "Take it back out"))
            if g.bodies or g.coal_bed.char_kg > 0.005:
                out.append(("key", "F", "Blow into the base (hold ~0.5 s)"))
        return out

    def _draw_time_card(self, d, world, player):
        S = self.S
        w = world.weather
        x, y = S(18), S(16)
        bw, bh = S(248), S(62)
        panel(d, [x, y, x + bw, y + bh], radius=S(10))
        desc = w.describe()
        ic = "moon" if w.is_night() else "sun"
        if w.snow_intensity > 0.05:
            ic = "snow"
        elif w.rain_intensity > 0.05:
            ic = "rain"
        elif w.cloud_cover > 0.6:
            ic = "cloud"
        icon(d, ic, x + S(30), y + S(31), S(34))
        f1 = self.F("bold", 20)
        f2 = self.F("regular", 12)
        d.text((x + S(56), y + S(8)), w.clock_string(), font=f1, fill=PAL["text"])
        d.text((x + S(56) + text_w(d, w.clock_string(), f1) + S(10), y + S(14)), f"{w.ambient_temp_c:.0f}\u00b0C", font=f2, fill=PAL["text"])
        d.text((x + S(56), y + S(37)), f"{w.current_season.value} \u00b7 Day {int(w.day_of_season)} \u00b7 {desc}", font=f2, fill=PAL["dim"])
        # wind arrow relative to the view
        yaw = math.radians(player.yaw)
        wd = w.wind_direction_rad
        rel = wd - yaw
        ax, ay = x + S(222), y + S(40)
        L = S(9)
        dx, dy = math.sin(rel) * L, -math.cos(rel) * L
        d.line([(ax - dx, ay - dy), (ax + dx, ay + dy)], fill=PAL["cyan"], width=max(1, S(2)))
        hx, hy = ax + dx, ay + dy
        d.polygon([(hx, hy), (hx - dx * 0.5 - dy * 0.4, hy - dy * 0.5 + dx * 0.4), (hx - dx * 0.5 + dy * 0.4, hy - dy * 0.5 - dx * 0.4)], fill=PAL["cyan"])
        d.text((x + S(200), y + S(8)), f"{w.wind_speed_m_s:.1f} m/s", font=self.F("regular", 10.5), fill=PAL["faint"])

    def _draw_credits(self, d, phone):
        S = self.S
        c = phone.credits
        f = self.F("bold", 15)
        txt = f"{c.credits:.0f}"
        pend = f"+{c.pending:.1f} pending" if c.pending >= 0.1 else "credits"
        fs = self.F("regular", 10.5)
        w = max(text_w(d, txt, f), 10) + text_w(d, pend, fs) + S(56)
        x2 = self.width - S(18)
        x1 = x2 - w
        y = S(16)
        panel(d, [x1, y, x2, y + S(34)], radius=S(17))
        icon(d, "credits", x1 + S(18), y + S(17), S(20))
        d.text((x1 + S(32), y + S(6)), txt, font=f, fill=PAL["gold"])
        d.text((x1 + S(40) + text_w(d, txt, f), y + S(11)), pend, font=fs, fill=PAL["green"] if c.pending >= 0.1 else PAL["dim"])
        if c.pending >= 1.0:
            fh = self.F("regular", 10)
            hint = "Tab to claim"
            d.text((x2 - text_w(d, hint, fh) - S(6), y + S(38)), hint, font=fh, fill=PAL["dim"])

    def _draw_tutorial(self, d, input_mgr):
        tut = getattr(input_mgr, "tutorial", None)
        if tut is None:
            return
        st = tut.state()
        if not st["active"]:
            return
        S = self.S
        bw = S(560)
        x = (self.width - bw) // 2
        y = S(14)
        ft = self.F("bold", 12)
        fb = self.F("regular", 13)
        fh = self.F("light", 11.5)
        lines = wrap(d, st["text"], fb, bw - S(36))
        bh = S(44) + len(lines) * S(18) + (S(18) if st["hint"] else 0)
        glow = st["flash"] > 0
        panel(d, [x, y, x + bw, y + bh], radius=S(12), fill=(16, 20, 17, 215), outline=PAL["amber"] if glow else (255, 176, 72, 90), width=max(1, S(1.5)))
        d.text((x + S(18), y + S(10)), f"STEP {st['index'] + 1} OF {st['total']}", font=ft, fill=PAL["amber"])
        tw = text_w(d, f"STEP {st['index'] + 1} OF {st['total']}", ft)
        d.text((x + S(26) + tw, y + S(9)), st["title"], font=self.F("medium", 13), fill=PAL["text"])
        for i in range(st["total"]):
            cx = x + bw - S(18) - (st["total"] - 1 - i) * S(13)
            r = S(3.5)
            col = PAL["amber"] if i < st["index"] else ((255, 255, 255, 200) if i == st["index"] else (255, 255, 255, 60))
            d.ellipse([cx - r, y + S(16) - r, cx + r, y + S(16) + r], fill=col)
        yy = y + S(32)
        for ln in lines:
            d.text((x + S(18), yy), ln, font=fb, fill=PAL["text"])
            yy += S(18)
        if st["hint"]:
            d.text((x + S(18), yy + S(1)), st["hint"], font=fh, fill=PAL["dim"])

    def _draw_toasts(self, d, world, input_mgr):
        S = self.S
        tut = getattr(input_mgr, "tutorial", None)
        y = S(118) if (tut and tut.state()["active"]) else S(30)
        f = self.F("regular", 13)
        cols = {"info": PAL["amber"], "success": PAL["green"], "warn": (255, 200, 90, 255), "danger": PAL["red"]}
        for t in list(world.toasts)[-3:][::-1]:
            a = max(0.0, min(1.0, t["time"] / 0.6))
            lines = wrap(d, t["text"], f, S(600))
            w = max(text_w(d, ln, f) for ln in lines) + S(36)
            h = S(12) + len(lines) * S(18)
            x = (self.width - w) // 2
            bg = (14, 16, 15, int(190 * a))
            d.rounded_rectangle([x, y, x + w, y + h], radius=S(8), fill=bg)
            c = cols.get(t.get("kind", "info"), PAL["amber"])
            d.rounded_rectangle([x, y, x + S(4), y + h], radius=S(2), fill=(c[0], c[1], c[2], int(255 * a)))
            yy = y + S(6)
            for ln in lines:
                d.text((x + S(18), yy), ln, font=f, fill=(236, 232, 222, int(255 * a)))
                yy += S(18)
            y += h + S(6)

    def _draw_hotbar(self, d, world, player):
        S = self.S
        inv = world.inventory
        tools = ["pocket_lighter", "pocket_knife", "felling_axe", "bow_drill_spindle", "accelerant"]
        offs = ["tinder", "kindling", "firewood", "log", "fungus"]
        slot = S(48)
        gap = S(6)
        sep = S(18)
        total = 10 * slot + 8 * gap + sep
        x0 = (self.width - total) // 2
        y0 = self.height - S(90)
        panel(d, [x0 - S(10), y0 - S(8), x0 + total + S(10), y0 + slot + S(30)], radius=S(12), fill=(12, 14, 13, 170))
        fk = self.F("bold", 10)
        fc = self.F("medium", 11)
        x = x0
        for i, t in enumerate(tools):
            sel = player.primary_hand == t
            avail = not (t == "accelerant" and inv.gasoline_kg <= 0.01)
            self._slot(d, x, y0, slot, sel, PAL["amber"])
            icon(d, TOOL_ICONS[t], x + slot / 2, y0 + slot / 2 - S(2), slot * (0.62 if avail else 0.5))
            if not avail:
                d.line([(x + S(10), y0 + slot - S(10)), (x + slot - S(10), y0 + S(10))], fill=(255, 255, 255, 90), width=1)
            d.text((x + S(4), y0 + S(2)), str(i + 1), font=fk, fill=PAL["dim"])
            if t == "pocket_lighter":
                for k in range(max(4, inv.lighter_starts)):
                    px = x + S(8) + k * S(9)
                    c = PAL["amber"] if k < inv.lighter_starts else (255, 255, 255, 45)
                    if k < 6:
                        d.rounded_rectangle([px, y0 + slot - S(7), px + S(6), y0 + slot - S(3)], radius=1, fill=c)
                if inv.lighter_burn_left > 0 and world.crafting.lighter_is_lit:
                    frac = inv.lighter_burn_left / 14.0
                    d.rectangle([x + S(4), y0 + slot + S(2), x + S(4) + (slot - S(8)) * frac, y0 + slot + S(4)], fill=PAL["ember"])
            if t == "accelerant" and inv.gasoline_kg > 0:
                d.text((x + slot - S(30), y0 + slot - S(15)), f"{inv.gasoline_kg:.1f}", font=fk, fill=PAL["text"])
            x += slot + gap
        x += sep - gap
        d.line([(x - sep / 2, y0 + S(6)), (x - sep / 2, y0 + slot - S(6))], fill=(255, 255, 255, 50), width=1)
        cur = world.current_offhand()
        for i, o in enumerate(offs):
            sel = cur == o
            n = len(inv.fungi) if o == "fungus" else inv.count(o)
            self._slot(d, x, y0, slot, sel, PAL["green"])
            icon(d, OFFHAND_ICONS[o], x + slot / 2, y0 + slot / 2 - S(2), slot * (0.6 if n else 0.45))
            key = str((i + 6) % 10)
            d.text((x + S(4), y0 + S(2)), key, font=fk, fill=PAL["dim"])
            txt = f"x{n}"
            d.text((x + slot - text_w(d, txt, fc) - S(4), y0 + slot - S(16)), txt, font=fc, fill=PAL["text"] if n else PAL["faint"])
            if o == "fungus" and world.crafting.smoldering_fungus is not None:
                d.ellipse([x + slot - S(12), y0 + S(5), x + slot - S(5), y0 + S(12)], fill=PAL["ember"])
            x += slot + gap
        # caption
        item = inv.peek(cur) if cur != "fungus" else None
        tname = TOOL_NAMES[player.primary_hand]
        if player.primary_hand == "pocket_lighter":
            tname += f" - {inv.lighter_starts} starts"
        elif player.primary_hand == "pocket_knife":
            tname += f" ({inv.knife})"
        elif player.primary_hand == "felling_axe":
            tname += f" ({inv.axe})"
        elif player.primary_hand == "bow_drill_spindle":
            if getattr(world, "friction_mode", "bow") == "hand":
                tname = "Hand Drill - river stone"
            else:
                tname += f" - cord {world.crafting.bow_drill.cordage_condition * 100:.0f}% +{inv.cordage}"
        if cur == "none":
            oname = "Empty hand"
        elif cur == "fungus":
            oname = f"Fungus x{len(inv.fungi)}" + (" (smouldering)" if world.crafting.smoldering_fungus else "")
        elif item is not None:
            oname = f"{item.name} - {item.moisture * 100:.0f}% MC ({item.moisture_label()})"
        else:
            oname = f"No {OFFHAND_LABELS[cur].lower()}"
        f = self.F("regular", 12)
        cap = f"{tname}      |      {oname}"
        d.text(((self.width - text_w(d, cap, f)) / 2, y0 + slot + S(8)), cap, font=f, fill=PAL["dim"])

    def _slot(self, d, x, y, s, sel, col):
        S = self.S
        if sel:
            d.rounded_rectangle([x - S(1), y - S(1), x + s + S(1), y + s + S(1)], radius=S(8), fill=(col[0], col[1], col[2], 40), outline=col, width=max(1, S(2)))
        else:
            d.rounded_rectangle([x, y, x + s, y + s], radius=S(7), fill=(255, 255, 255, 14), outline=(255, 255, 255, 30), width=1)

    def _draw_vitals(self, d, world, player):
        S = self.S
        x, y = S(22), self.height - S(60)
        w = S(170)
        f = self.F("medium", 10.5)
        d.text((x, y - S(16)), "STAMINA", font=f, fill=PAL["dim"])
        d.rounded_rectangle([x, y, x + w, y + S(7)], radius=S(3), fill=(255, 255, 255, 40))
        frac = max(0.0, min(1.0, player.stamina / 100.0))
        d.rounded_rectangle([x, y, x + max(S(6), w * frac), y + S(7)], radius=S(3), fill=lerp_col((230, 90, 60, 255), (130, 210, 120, 255), frac))
        if world.breath_start is not None:
            held = world.breath_charge
            v = world.breath_velocity_from_hold(held)
            yy = y - S(48)
            d.text((x, yy - S(16)), f"BREATH  {v:.1f} m/s", font=f, fill=PAL["text"])
            d.rounded_rectangle([x, yy, x + w, yy + S(10)], radius=S(4), fill=(255, 255, 255, 40))
            # optimal band 1.5 - 4.0 m/s on a 0-7 m/s scale
            bx0, bx1 = x + w * 1.5 / 7.0, x + w * 4.0 / 7.0
            d.rectangle([bx0, yy, bx1, yy + S(10)], fill=(120, 214, 140, 110))
            d.rectangle([x + w * 6.0 / 7.0, yy, x + w, yy + S(10)], fill=(255, 96, 80, 110))
            px = x + w * min(1.0, v / 7.0)
            d.rectangle([px - S(2), yy - S(3), px + S(2), yy + S(13)], fill=(255, 255, 255, 255))

    def _draw_fire_panel(self, d, world):
        S = self.S
        g = world.thermal_grid
        bw, bh = S(268), S(196)
        x = self.width - bw - S(18)
        y = self.height - bh - S(104)
        panel(d, [x, y, x + bw, y + bh], radius=S(12))
        ft = self.F("bold", 11)
        d.text((x + S(14), y + S(10)), "CAMPFIRE", font=ft, fill=PAL["ember"])
        stage = g.fire_stage()
        sc = STAGE_COLORS.get(stage, PAL["dim"])
        fs = self.F("medium", 10.5)
        label = stage.replace(" (Glowing Coals)", "")
        tw = text_w(d, label, fs) + S(14)
        d.rounded_rectangle([x + bw - S(14) - tw, y + S(8), x + bw - S(14), y + S(26)], radius=S(9), fill=(sc[0], sc[1], sc[2], 60), outline=sc)
        d.text((x + bw - S(7) - tw, y + S(9)), label, font=fs, fill=sc)
        T = g.core_coal_temp
        fbig = self.F("bold", 28)
        d.text((x + S(14), y + S(30)), f"{T:.0f}", font=fbig, fill=temp_color(T))
        d.text((x + S(18) + text_w(d, f"{T:.0f}", fbig), y + S(44)), "C core", font=self.F("regular", 11), fill=PAL["dim"])
        # thermometer bar 0..1100 C with stage markers
        bx, by, bwid = x + S(14), y + S(74), bw - S(28)
        for i in range(int(bwid)):
            t = i / bwid * 1100.0
            c = temp_color(t)
            d.line([(bx + i, by), (bx + i, by + S(6))], fill=(c[0], c[1], c[2], 130))
        px = bx + bwid * min(1.0, T / 1100.0)
        d.polygon([(px, by - S(1)), (px - S(5), by - S(8)), (px + S(5), by - S(8))], fill=PAL["text"])
        fm = self.F("regular", 8.5)
        for mark, lab in ((100, "steam"), (300, "pyro"), (450, "flame"), (900, "glow")):
            mx = bx + bwid * mark / 1100.0
            d.line([(mx, by + S(6)), (mx, by + S(10))], fill=(255, 255, 255, 100))
            d.text((mx - text_w(d, lab, fm) / 2, by + S(10)), lab, font=fm, fill=PAL["faint"])
        f = self.F("regular", 11.5)
        fv = self.F("mono", 11.5)
        rows = [
            ("Heat release", f"{g.current_heat_flux / 1000.0:5.1f} kW"),
            ("Flame height", f"{g.flame_height:5.2f} m"),
            ("Draft", f"{g.draft_velocity:5.2f} m/s"),
            ("Oxygen", f"{g.oxygen_fraction * 100:5.1f} %"),
        ]
        yy = y + S(102)
        for i, (k, v) in enumerate(rows):
            cx = x + S(14) + (i % 2) * (bw // 2 - S(4))
            cy = yy + (i // 2) * S(30)
            d.text((cx, cy), k, font=self.F("regular", 9.5), fill=PAL["faint"])
            col = PAL["red"] if (k == "Oxygen" and g.oxygen_fraction < 0.12) else PAL["text"]
            d.text((cx, cy + S(12)), v, font=fv, fill=col)
        struct = world.structure_key.replace("_", " ").title()
        info = f"{struct}" + (f"  {world.structure_score * 100:.0f}%" if world.structure_key != "freeform" else "") + f"  -  {len(g.bodies)} pieces"
        d.text((x + S(14), y + bh - S(30)), info, font=self.F("regular", 10.5), fill=PAL["dim"])
        # sparkline of core temperature
        hist = g.temp_history[-90:]
        if len(hist) > 2:
            sx0, sy0, sw, sh = x + bw - S(104), y + S(32), S(90), S(30)
            pts = []
            mx = max(600.0, max(hist))
            for i, v in enumerate(hist):
                pts.append((sx0 + sw * i / (len(hist) - 1), sy0 + sh - sh * min(1.0, v / mx)))
            d.line(pts, fill=PAL["amber"], width=max(1, S(1.5)))
        steam_share = g.steam_rate * 2.26e6 / max(1.0, g.current_heat_flux)
        if (g.steam_rate > 4e-4 and steam_share > 0.08) or (g.smoke_rate > 2e-4 and g.flame_hrr < 300):
            msg = "hissing steam - wet fuel" if g.steam_rate > g.smoke_rate else "thick smoke - pyrolysis without flame"
            d.text((x + S(14), y + bh - S(16)), msg, font=self.F("light", 10), fill=(200, 220, 235, 255) if "steam" in msg else (235, 205, 120, 255))

    def _draw_bow_gauge(self, d, world):
        S = self.S
        bd = world.crafting.bow_drill
        bw, bh = S(240), S(120)
        x = S(22)
        y = self.height - S(90) - bh - S(40)
        panel(d, [x, y, x + bw, y + bh], radius=S(10))
        hand = getattr(world, "friction_mode", "bow") == "hand"
        d.text((x + S(12), y + S(8)), "HAND DRILL" if hand else "BOW DRILL", font=self.F("bold", 11), fill=PAL["amber"])
        st = "EMBER!" if bd.has_ember else ("smoking" if bd.is_smoking else ("drilling" if world.bow_active else "idle"))
        d.text((x + S(100), y + S(8)), st, font=self.F("medium", 11), fill=PAL["ember"] if bd.has_ember else PAL["dim"])
        T = bd.ember_temp if bd.has_ember else bd.socket_temp

        def bar(yy, label, frac, col, txt):
            d.text((x + S(12), yy), label, font=self.F("regular", 9.5), fill=PAL["faint"])
            d.rounded_rectangle([x + S(80), yy + S(3), x + bw - S(50), yy + S(10)], radius=S(3), fill=(255, 255, 255, 35))
            d.rounded_rectangle([x + S(80), yy + S(3), x + S(80) + max(S(4), (bw - S(130)) * max(0.0, min(1.0, frac))), yy + S(10)], radius=S(3), fill=col)
            d.text((x + bw - S(45), yy - S(1)), txt, font=self.F("mono", 10), fill=PAL["text"])
        bar(y + S(32), "Socket", T / 450.0, temp_color(T), f"{T:.0f}C")
        sx = x + S(80) + (bw - S(130)) * 380.0 / 450.0
        d.line([(sx, y + S(30)), (sx, y + S(46))], fill=(255, 255, 255, 160))
        bar(y + S(52), "Dust", bd.wood_dust_mass / 0.08, (200, 170, 120, 255), f"{bd.wood_dust_mass:.2f}g")
        bar(y + S(72), "Char", bd.dust_char_fraction, (90, 70, 60, 255), f"{bd.dust_char_fraction * 100:.0f}%")
        if hand:
            hd = world.crafting.hand_drill
            frac = hd.hand_height_cm / max(1.0, hd.spindle_length_cm * 0.9)
            bar(y + S(92), "Hands", frac, (210, 190, 160, 255), "stone" if world.secondary_held else f"{hd.hand_height_cm:.0f}cm")
        else:
            bar(y + S(92), "Cord", bd.cordage_condition, PAL["green"], f"{bd.cordage_condition * 100:.0f}%")
        if not bd.has_v_notch:
            d.text((x + S(12), y + bh - S(4) - S(12)), "", font=self.F("light", 10), fill=PAL["red"])

    def _draw_nest_gauge(self, d, world, nest):
        S = self.S
        bw, bh = S(300), S(86)
        x = (self.width - bw) // 2
        y = self.height // 2 + S(96)
        panel(d, [x, y, x + bw, y + bh], radius=S(10))
        d.text((x + S(12), y + S(8)), f"TINDER NEST - {nest.display_name}", font=self.F("bold", 11), fill=PAL["amber"])
        T = nest.core_temp
        bx, by, w = x + S(12), y + S(34), bw - S(24)
        d.rounded_rectangle([bx, by, bx + w, by + S(10)], radius=S(4), fill=(255, 255, 255, 35))
        d.rounded_rectangle([bx, by, bx + max(S(6), w * min(1.0, T / 650.0)), by + S(10)], radius=S(4), fill=temp_color(T))
        fx = bx + w * 550.0 / 650.0
        d.line([(fx, by - S(3)), (fx, by + S(13))], fill=(255, 255, 255, 220), width=max(1, S(2)))
        d.text((fx - S(14), by + S(14)), "flame", font=self.F("regular", 9), fill=PAL["dim"])
        status = {"optimal": "good breath", "too_weak": "too soft", "too_strong": "too hard", "ignited": "FLAME!"}.get(nest.last_breath_status, "")
        d.text((bx, y + bh - S(22)), f"{T:.0f} C   glow {nest.glow_area_cm2:.1f} cm2   {status}", font=self.F("regular", 11), fill=PAL["text"])

    def _draw_skip_report(self, d, rep):
        S = self.S
        lines = [f"{rep['season']}  -  day {rep['day']}  -  {rep['clock']}"]
        if rep.get("rack_before") is not None and rep.get("rack_after") is not None:
            lines.append(f"Drying rack:  {rep['rack_before'] * 100:.0f}%  ->  {rep['rack_after'] * 100:.0f}% moisture")
        for gl in rep.get("ground_logs", [])[:3]:
            lines.append(f"{gl['name']}:  {gl['before'] * 100:.0f}%  ->  {gl['after'] * 100:.0f}%" + ("  (rotted to punk wood)" if gl["punk"] else ""))
        if rep.get("fire_energy_mj", 0) > 0.05:
            lines.append(f"The fire released {rep['fire_energy_mj']:.1f} MJ while you rested")
            if rep.get("fire_after"):
                lines.append(f"...and it is still going: {rep.get('fire_kw_after', 0):.0f} kW, {rep.get('fire_mass_left_kg', 0):.1f} kg of fuel left")
            elif rep.get("fire_pieces_left"):
                lines.append(f"It burned down - {rep['fire_pieces_left']} charred piece(s) ({rep.get('fire_mass_left_kg', 0):.1f} kg) remain")
            elif rep.get("fire_mass_left_kg", 0) > 0.05:
                lines.append(f"It burned down - {rep['fire_mass_left_kg']:.1f} kg of charcoal lumps lie cold in the ash")
            else:
                lines.append("It burned down to cold ash")
        if rep.get("new_saplings"):
            lines.append(f"{rep['new_saplings']} new saplings sprouted")
        if rep.get("fungi_fruited"):
            lines.append(f"Bracket fungi fruited on {rep['fungi_fruited']} dead trunk(s)")
        if rep.get("humus"):
            lines.append(f"{rep['humus']} rotten log(s) crumbled into humus - richer soil")
        f = self.F("regular", 12)
        bw = S(420)
        bh = S(40) + len(lines) * S(19)
        x = (self.width - bw) // 2
        y = self.height // 2 - S(150)
        panel(d, [x, y, x + bw, y + bh], radius=S(12), fill=(14, 16, 15, 215), outline=(255, 176, 72, 120))
        d.text((x + S(16), y + S(10)), f"TIME PASSED: {rep['hours']:.0f} HOURS", font=self.F("bold", 12), fill=PAL["amber"])
        yy = y + S(32)
        for ln in lines:
            d.text((x + S(16), yy), ln, font=f, fill=PAL["text"])
            yy += S(19)

    def _draw_time_passing(self, d, hours):
        S = self.S
        bw, bh = S(360), S(76)
        x, y = (self.width - bw) // 2, self.height // 2 - S(120)
        panel(d, [x, y, x + bw, y + bh], radius=S(12), fill=(14, 16, 15, 225), outline=(255, 176, 72, 120))
        label = f"{hours:.0f} hours" if hours < 48 else f"{hours / 24.0:.0f} days"
        d.text((x + S(18), y + S(14)), "TIME PASSES...", font=self.F("bold", 13), fill=PAL["amber"])
        d.text((x + S(18), y + S(40)), f"{label}: the fire, the racks and the forest live on", font=self.F("regular", 12),
               fill=PAL["text"])

    def _draw_help(self, d):
        S = self.S
        rows = [
            ("W A S D", "Move      Shift sprint      Ctrl crouch (toggle)"),
            ("Mouse", "Look around"),
            ("LMB", "Pick up / use tool (hold: lighter flame, drilling)"),
            ("RMB", "Place fuel / stack on rack / bear down / tip ember"),
            ("Wheel", "Rotate piece   Shift+Wheel tilt   MMB reset"),
            ("1 - 5", "Lighter, Knife, Axe, Drill (4 again: bow/hand), Gas"),
            ("6 - 0 / Q E", "Tinder, Kindling, Firewood, Log, Fungus"),
            ("X", "Empty both hands (Q / E: empty off hand)"),
            ("F (hold)", "Blow: ~0.3-0.8 s is a deep even breath"),
            ("C", "Twist inner bark into cordage"),
            ("Tab", "Smartphone: claim credits, Store, Topo Map"),
            ("B", "Field Compendium (pin blueprints with Enter)"),
            ("T", "Pass time (season your wood)"),
            ("Esc", "Pause & settings      H  this help"),
        ]
        bw, bh = S(560), S(60) + len(rows) * S(24)
        x, y = (self.width - bw) // 2, (self.height - bh) // 2
        panel(d, [x, y, x + bw, y + bh], radius=S(14), fill=(12, 14, 13, 230))
        d.text((x + S(20), y + S(16)), "CONTROLS", font=self.F("bold", 14), fill=PAL["amber"])
        fk = self.F("bold", 11)
        f = self.F("regular", 13)
        yy = y + S(50)
        for k, v in rows:
            key_cap(d, x + S(20), yy, k, fk)
            d.text((x + S(150), yy), v, font=f, fill=PAL["text"])
            yy += S(24)

    # ================================================================== smartphone
    def _draw_phone(self, img, d, world, phone, player):
        S = self.S
        W, H = self.width, self.height
        # dim the world behind
        d.rectangle([0, 0, W, H], fill=(0, 0, 0, 90))
        pw, ph = S(370), S(690)
        px, py = W - pw - S(60), (H - ph) // 2
        # device body with bezel
        d.rounded_rectangle([px - S(10), py - S(10), px + pw + S(10), py + ph + S(10)], radius=S(44), fill=(28, 30, 32, 255), outline=(90, 96, 100, 255), width=max(1, S(2)))
        d.rounded_rectangle([px, py, px + pw, py + ph], radius=S(36), fill=(8, 12, 14, 255))
        # status bar
        w = world.weather
        fs = self.F("medium", 11)
        d.text((px + S(28), py + S(14)), w.clock_string(), font=fs, fill=PAL["text"])
        d.ellipse([px + pw / 2 - S(6), py + S(14), px + pw / 2 + S(6), py + S(26)], fill=(0, 0, 0, 255))
        bat = phone.battery_percent
        bx = px + pw - S(56)
        d.rounded_rectangle([bx, py + S(16), bx + S(24), py + S(26)], radius=S(2), outline=(220, 220, 220, 255), width=1)
        d.rectangle([bx + S(2), py + S(18), bx + S(2) + S(20) * bat / 100.0, py + S(24)], fill=PAL["green"] if bat > 20 else PAL["red"])
        for i in range(4):
            h_ = S(3 + i * 2.5)
            d.rectangle([px + pw - S(90) + i * S(5), py + S(26) - h_, px + pw - S(87) + i * S(5), py + S(26)], fill=(220, 220, 220, 255 if i < 3 else 90))
        content = [px + S(14), py + S(40), px + pw - S(14), py + ph - S(40)]
        app = phone.active_app.value
        if app == "Home":
            self._phone_home(img, d, world, phone, content)
        elif app == "Store ($)":
            self._phone_store(d, world, phone, content)
        else:
            self._phone_map(img, d, world, phone, player, content)
        # home indicator & hints
        d.rounded_rectangle([px + pw / 2 - S(50), py + ph - S(16), px + pw / 2 + S(50), py + ph - S(12)], radius=S(2), fill=(200, 200, 200, 200))
        fh = self.F("regular", 10)
        hint = {"Home": "Arrows select   Enter open   Tab close",
                "Store ($)": "W/S browse   Enter buy   Backspace home",
                }.get(app, "W/S zoom   Q switch app   Backspace home")
        d.text((px + (pw - text_w(d, hint, fh)) / 2, py + ph - S(34)), hint, font=fh, fill=PAL["faint"])

    def _phone_home(self, img, d, world, phone, box):
        S = self.S
        x0, y0, x1, y1 = box
        # wallpaper: dusk forest gradient with ridge silhouettes
        key = ("wall", x1 - x0, y1 - y0)
        if key not in self._paper_cache:
            ww, hh = int(x1 - x0), int(y1 - y0)
            wall = Image.new("RGBA", (ww, hh))
            wd = ImageDraw.Draw(wall)
            for yy in range(hh):
                t = yy / hh
                c = lerp_col((32, 44, 72, 255), (226, 126, 74, 255), t ** 1.6)
                wd.line([(0, yy), (ww, yy)], fill=c)
            rng = np.random.default_rng(3)
            for layer, (col, base) in enumerate((((40, 40, 52, 255), 0.62), ((22, 26, 30, 255), 0.74), ((10, 12, 14, 255), 0.86))):
                pts = [(0, hh)]
                for i in range(0, ww + 8, 8):
                    h0 = hh * base - 40 * math.sin(i * 0.013 + layer) - rng.uniform(0, 18)
                    if layer == 2 and rng.random() < 0.35:
                        h0 -= rng.uniform(20, 55)
                    pts.append((i, h0))
                pts.append((ww, hh))
                wd.polygon(pts, fill=col)
            mask = Image.new("L", (ww, hh), 0)
            ImageDraw.Draw(mask).rounded_rectangle([0, 0, ww - 1, hh - 1], radius=S(24), fill=255)
            wall.putalpha(mask)
            self._paper_cache[key] = wall
        img.alpha_composite(self._paper_cache[key], (int(x0), int(y0)))
        w = world.weather
        fbig = self.F("light", 54)
        t = w.clock_string()
        d.text((x0 + (x1 - x0 - text_w(d, t, fbig)) / 2, y0 + S(26)), t, font=fbig, fill=(255, 255, 255, 240))
        f = self.F("regular", 13)
        sub = f"{w.current_season.value}, day {int(w.day_of_season)}  -  {w.ambient_temp_c:.0f} C  {w.describe()}"
        d.text((x0 + (x1 - x0 - text_w(d, sub, f)) / 2, y0 + S(96)), sub, font=f, fill=(255, 255, 255, 220))
        # credits widget
        cy = y0 + S(136)
        comp_shape(img, [x0 + S(14), cy, x1 - S(14), cy + S(76)], (255, 255, 255, 40), radius=S(16))
        icon(d, "credits", x0 + S(44), cy + S(38), S(32))
        d.text((x0 + S(70), cy + S(10)), f"{phone.credits.credits:.0f} credits", font=self.F("bold", 20), fill=(255, 255, 255, 255))
        claim = f"+{phone.last_claimed:.0f} fire-mastery credits claimed" if phone.last_claimed >= 0.5 else "Keep a hot, clean fire to earn credits"
        d.text((x0 + S(70), cy + S(44)), claim, font=self.F("regular", 11), fill=(255, 230, 180, 255))
        # forecast widget
        fy = cy + S(90)
        comp_shape(img, [x0 + S(14), fy, x1 - S(14), fy + S(96)], (255, 255, 255, 40), radius=S(16))
        d.text((x0 + S(28), fy + S(8)), "FORECAST", font=self.F("bold", 10), fill=(255, 255, 255, 200))
        fc = w.forecast(12)[:5]
        cw = (x1 - x0 - S(40)) / len(fc)
        for i, e in enumerate(fc):
            cx = x0 + S(20) + cw * i + cw / 2
            hr = int(e["hour"])
            d.text((cx - S(10), fy + S(26)), f"{hr:02d}h", font=self.F("regular", 10), fill=(255, 255, 255, 210))
            icon(d, "snow" if e["snow"] else ("rain" if e["precip"] else ("moon" if hr < 5 or hr > 20 else "sun")), cx, fy + S(56), S(22))
            d.text((cx - S(10), fy + S(72)), f"{e['temp']:.0f}", font=self.F("medium", 11), fill=(255, 255, 255, 230))
        # app icons
        ay = fy + S(122)
        apps = [("store", "Store", (236, 150, 60, 255)), ("map", "Topo Map", (90, 170, 120, 255))]
        for i, (ic, label, col) in enumerate(apps):
            cx = x0 + (x1 - x0) * (0.3 + 0.4 * i)
            sel = phone.home_selection == i
            r = S(34)
            d.rounded_rectangle([cx - r, ay, cx + r, ay + 2 * r], radius=S(18), fill=col, outline=(255, 255, 255, 255) if sel else None, width=max(1, S(3)))
            icon(d, ic, cx, ay + r, S(40), (255, 255, 255, 255))
            fl = self.F("medium", 12)
            d.text((cx - text_w(d, label, fl) / 2, ay + 2 * r + S(6)), label, font=fl, fill=(255, 255, 255, 240))
        if phone.status_message and phone.status_message != "Ready":
            fs = self.F("regular", 11)
            d.text((x0 + S(20), y1 - S(30)), phone.status_message[:52], font=fs, fill=(255, 255, 255, 200))

    def _phone_store(self, d, world, phone, box):
        S = self.S
        x0, y0, x1, y1 = box
        d.rounded_rectangle(box, radius=S(24), fill=(18, 20, 19, 255))
        d.text((x0 + S(18), y0 + S(12)), "Outfitters", font=self.F("bold", 20), fill=PAL["text"])
        bal = f"{phone.credits.credits:.0f}"
        fb = self.F("bold", 14)
        icon(d, "credits", x1 - S(30) - text_w(d, bal, fb), y0 + S(24), S(16))
        d.text((x1 - S(18) - text_w(d, bal, fb), y0 + S(14)), bal, font=fb, fill=PAL["gold"])
        state = phone.get_display_state(world.player.position, world.weather.ambient_temp_c)
        y = y0 + S(50)
        fn = self.F("medium", 12.5)
        fc = self.F("regular", 10)
        for it in state["store_catalog"]:
            sel = it["is_selected"]
            h = S(98) if sel else S(46)
            bg = (40, 44, 40, 255) if sel else (28, 31, 29, 255)
            d.rounded_rectangle([x0 + S(10), y, x1 - S(10), y + h], radius=S(12), fill=bg, outline=PAL["amber"] if sel else None, width=max(1, S(1.5)))
            icon(d, it["icon"], x0 + S(34), y + S(23), S(28))
            name_col = PAL["text"] if not it["sold_out"] else PAL["faint"]
            d.text((x0 + S(58), y + S(6)), it["name"][:30], font=fn, fill=name_col)
            meta = it["cat"] + (f"  -  owned {it['owned']}" if it["owned"] else "")
            d.text((x0 + S(58), y + S(25)), meta, font=fc, fill=PAL["faint"])
            price = "OWNED" if it["sold_out"] else f"{it['cost']:.0f}"
            fp = self.F("bold", 13)
            pc = PAL["gold"] if it["affordable"] and not it["sold_out"] else (PAL["faint"] if it["sold_out"] else PAL["red"])
            d.text((x1 - S(22) - text_w(d, price, fp), y + S(12)), price, font=fp, fill=pc)
            if sel:
                lines = wrap(d, it["desc"], self.F("regular", 11), x1 - x0 - S(50))
                yy = y + S(44)
                for ln in lines[:2]:
                    d.text((x0 + S(22), yy), ln, font=self.F("regular", 11), fill=PAL["dim"])
                    yy += S(16)
                btn = "Buy  [Enter]" if not it["sold_out"] else "Owned"
                fbt = self.F("bold", 11)
                bw_ = text_w(d, btn, fbt) + S(20)
                col = PAL["amber"] if it["affordable"] and not it["sold_out"] else (90, 90, 90, 255)
                d.rounded_rectangle([x1 - S(22) - bw_, y + h - S(28), x1 - S(22), y + h - S(8)], radius=S(10), fill=col)
                d.text((x1 - S(12) - bw_, y + h - S(26)), btn, font=fbt, fill=(20, 20, 20, 255))
            y += h + S(6)
        msg = phone.status_message
        if msg and msg != "Ready" and not msg.startswith("Opened"):
            f = self.F("regular", 11)
            lines = wrap(d, msg, f, x1 - x0 - S(30))
            yy = y1 - S(22) - S(16) * len(lines)
            for ln in lines:
                d.text((x0 + S(16), yy), ln, font=f, fill=PAL["green"] if "Purchased" in msg or "claimed" in msg else PAL["red"])
                yy += S(16)

    def _phone_map(self, img, d, world, phone, player, box):
        S = self.S
        x0, y0, x1, y1 = [int(v) for v in box]
        mw, mh = x1 - x0, y1 - y0 - S(70)
        zoom = phone.map_zoom
        half = 60.0 / zoom
        cx_w, cz_w = float(player.position[0]), float(player.position[2])
        key = ("map", mw, mh, round(cx_w / 4) * 4, round(cz_w / 4) * 4, round(zoom, 2))
        if key not in self._map_cache:
            self._map_cache.clear()
            self._map_cache[key] = self._render_map_base(mw, mh, round(cx_w / 4) * 4, round(cz_w / 4) * 4, half, world)
        base, (mcx, mcz) = self._map_cache[key]
        img.alpha_composite(base, (x0, y0 + S(40)))
        d.rectangle([x0, y0, x1, y0 + S(40)], fill=(18, 20, 19, 255))
        d.text((x0 + S(14), y0 + S(8)), "Topo Map", font=self.F("bold", 18), fill=PAL["text"])
        zone, hum = phone.topo_map.get_microclimate_at(cx_w, cz_w)
        d.text((x1 - S(14) - text_w(d, f"{zone}", self.F("regular", 10)), y0 + S(14)), zone, font=self.F("regular", 10), fill=PAL["dim"])
        my0 = y0 + S(40)
        scale = mw / (2 * half)

        def to_px(x, z):
            return x0 + mw / 2 + (x - mcx) * scale, my0 + mh / 2 + (z - mcz) * scale

        # waypoints
        fw = self.F("medium", 10)
        pts = [("Camp", world.fire_pit_pos, PAL["ember"]), ("rack", world.drying_rack.position, PAL["blue"]),
               ("wood", world.woodpile_pos, (210, 170, 110, 255))]
        for t in world.ecosystem.trees:
            if t.has_fungi and t.fungi_count > 0 and t.is_standing and abs(t.position[0] - mcx) < half and abs(t.position[2] - mcz) < half:
                pts.append(("fungi", t.position, (220, 150, 90, 255)))
        for wi in world.ground_logs:
            if wi.fuel is not None and wi.fuel.kind == "log" and abs(wi.position[0] - mcx) < half and abs(wi.position[2] - mcz) < half:
                pts.append(("log", wi.position, (170, 130, 90, 255)))
        for name, p, col in pts:
            px_, py_ = to_px(p[0], p[2])
            if x0 < px_ < x1 and my0 < py_ < my0 + mh:
                r = S(4.5) if name == "Camp" else S(2.5)
                d.ellipse([px_ - r, py_ - r, px_ + r, py_ + r], fill=col, outline=(0, 0, 0, 180))
                if name == "Camp":
                    d.text((px_ - S(14), py_ - S(20)), "Camp", font=fw, fill=(60, 30, 10, 255))
        # player arrow
        ppx, ppy = to_px(cx_w, cz_w)
        a = math.radians(player.yaw)
        fx, fz = math.cos(a), math.sin(a)
        L = S(11)
        tip = (ppx + fx * L, ppy + fz * L)
        left = (ppx - fx * L * 0.6 - fz * L * 0.55, ppy - fz * L * 0.6 + fx * L * 0.55)
        right = (ppx - fx * L * 0.6 + fz * L * 0.55, ppy - fz * L * 0.6 - fx * L * 0.55)
        d.polygon([tip, left, (ppx - fx * L * 0.3, ppy - fz * L * 0.3), right], fill=(255, 255, 255, 255), outline=(30, 110, 220, 255))
        d.text((ppx + S(10), ppy + S(4)), "You", font=fw, fill=(20, 70, 160, 255))
        # north arrow & scale bar
        nx, ny = x1 - S(26), my0 + S(26)
        d.polygon([(nx, ny - S(12)), (nx - S(6), ny + S(4)), (nx + S(6), ny + S(4))], fill=(40, 40, 40, 255))
        d.text((nx - S(4), ny + S(6)), "N", font=self.F("bold", 10), fill=(40, 40, 40, 255))
        sb = 20.0 * scale
        d.line([(x0 + S(14), my0 + mh - S(16)), (x0 + S(14) + sb, my0 + mh - S(16))], fill=(30, 30, 30, 255), width=max(1, S(2)))
        d.text((x0 + S(14), my0 + mh - S(32)), "20 m", font=self.F("regular", 10), fill=(30, 30, 30, 255))
        # footer
        fy = my0 + mh
        d.rectangle([x0, fy, x1, y1], fill=(18, 20, 19, 255))
        f = self.F("regular", 10.5)
        d.text((x0 + S(14), fy + S(8)), f"GPS {cx_w:+.1f}, {cz_w:+.1f}   elev {terrain.height(cx_w, cz_w):.1f} m   RH {hum * 100:.0f}%", font=f, fill=PAL["dim"])
        d.text((x0 + S(14), fy + S(28)), "5 m contours - creek, ridge & basin micro-climates", font=f, fill=PAL["faint"])

    def _render_map_base(self, mw, mh, cx, cz, half, world):
        res = 96
        xs = np.linspace(cx - half, cx + half, res)
        zs = np.linspace(cz - half, cz + half, int(res * mh / mw))
        X, Z = np.meshgrid(xs, zs, indexing="xy")
        Hh = np.asarray(terrain.height(X, Z))
        # hypsometric tint + hillshade
        gy, gx = np.gradient(Hh)
        shade = np.clip(0.75 + (gx * -0.5 + gy * -0.5) * 0.6, 0.35, 1.25)
        lo, hi = -6.0, 30.0
        t = np.clip((Hh - lo) / (hi - lo), 0, 1)
        c_low = np.array([196, 214, 170])
        c_mid = np.array([222, 210, 168])
        c_hi = np.array([205, 180, 150])
        col = np.where(t[..., None] < 0.5, c_low + (c_mid - c_low) * (t[..., None] * 2), c_mid + (c_hi - c_mid) * ((t[..., None] - 0.5) * 2))
        col = col * shade[..., None]
        # forest stipple
        rgb = np.clip(col, 0, 255).astype(np.uint8)
        im = Image.fromarray(rgb, "RGB").resize((mw, mh), Image.BILINEAR).convert("RGBA")
        d = ImageDraw.Draw(im)
        scale = mw / (2 * half)

        def to_px(x, z):
            return (x - cx) * scale + mw / 2, (z - cz) * scale + mh / 2

        # micro-climate zones
        from firestarter.economy.topomap import TopographicalMap
        tm = TopographicalMap()
        overlay = Image.new("RGBA", (mw, mh), (0, 0, 0, 0))
        od = ImageDraw.Draw(overlay)
        for zi in tm.microclimates:
            px_, py_ = to_px(zi["center"][0], zi["center"][2])
            r = zi["radius"] * scale
            c = zi["color"]
            od.ellipse([px_ - r, py_ - r, px_ + r, py_ + r], fill=(c[0], c[1], c[2], 45), outline=(c[0], c[1], c[2], 140), width=1)
            fz = font("medium", max(9, int(11 * self.s)))
            od.text((px_ - d.textlength(zi["name"], font=fz) / 2, py_ - 6), zi["name"], font=fz, fill=(c[0] // 2, c[1] // 2, c[2] // 2, 230))
        im.alpha_composite(overlay)
        # trees as dots
        for tr in world.ecosystem.trees:
            if abs(tr.position[0] - cx) < half and abs(tr.position[2] - cz) < half and tr.is_standing:
                px_, py_ = to_px(tr.position[0], tr.position[2])
                r = 1.4 if tr.stage.value != "Sapling" else 0.8
                d.ellipse([px_ - r, py_ - r, px_ + r, py_ + r], fill=(70, 110, 70, 170))
        # contours
        segs = tm.contour_segments(cx, cz, half, resolution=64)
        for x1_, z1_, x2_, z2_, lvl in segs:
            p1, p2 = to_px(x1_, z1_), to_px(x2_, z2_)
            index = abs(lvl % 25.0) < 1e-6
            d.line([p1, p2], fill=(120, 85, 55, 230 if index else 140), width=2 if index else 1)
        # creek
        xs_c = np.linspace(cx - half, cx + half, 80)
        zc = terrain.creek_centerline_z(xs_c)
        pts = [to_px(x_, z_) for x_, z_ in zip(xs_c, zc)]
        d.line(pts, fill=(70, 130, 210, 255), width=max(2, int(3 * self.s)))
        return im, (cx, cz)

    # ================================================================== reference book
    def _paper(self, w, h):
        key = ("paper", w, h)
        if key not in self._paper_cache:
            rng = np.random.default_rng(7)
            base = np.array([232, 218, 186], dtype=float)
            noise = rng.normal(0, 1, (h // 4 + 1, w // 4 + 1))
            noise = np.asarray(Image.fromarray(((noise - noise.min()) / (np.ptp(noise) + 1e-9) * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC), dtype=float) / 255.0
            yy, xx = np.mgrid[0:h, 0:w]
            vig = 1.0 - 0.18 * (((xx / w - 0.5) * 2) ** 4 + ((yy / h - 0.5) * 2) ** 4)
            arr = base[None, None, :] * (0.92 + 0.1 * noise[..., None]) * vig[..., None]
            im = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB").convert("RGBA")
            self._paper_cache[key] = im
        return self._paper_cache[key]

    def _draw_book(self, img, d, world, book):
        S = self.S
        W, H = self.width, self.height
        d.rectangle([0, 0, W, H], fill=(0, 0, 0, 120))
        bw, bh = S(1080), S(620)
        bx, by = (W - bw) // 2, (H - bh) // 2
        d.rounded_rectangle([bx - S(14), by - S(14), bx + bw + S(14), by + bh + S(14)], radius=S(18), fill=(74, 40, 24, 255), outline=(120, 78, 46, 255), width=max(1, S(3)))
        pw = bw // 2
        paper = self._paper(pw, bh)
        img.alpha_composite(paper, (bx, by))
        img.alpha_composite(paper.transpose(Image.FLIP_LEFT_RIGHT), (bx + pw, by))
        d = ImageDraw.Draw(img)
        # gutter shadow (composited so the paper stays opaque)
        gw = S(24)
        gut = Image.new("RGBA", (2 * gw, bh), (0, 0, 0, 0))
        gd = ImageDraw.Draw(gut)
        for i in range(gw):
            a = int(90 * (1 - i / gw) ** 2)
            gd.line([(gw - 1 - i, 0), (gw - 1 - i, bh)], fill=(60, 40, 20, a))
            gd.line([(gw + i, 0), (gw + i, bh)], fill=(60, 40, 20, a))
        img.alpha_composite(gut, (bx + pw - gw, by))
        data = book.get_current_page_data()
        ink = (48, 34, 22, 255)
        faded = (110, 88, 62, 255)
        accent = (150, 70, 30, 255)
        lx = bx + S(48)
        d.text((lx, by + S(30)), "FIREMASTER'S FIELD COMPENDIUM", font=self.F("medium", 10.5), fill=faded)
        d.text((lx, by + S(52)), data["title"], font=self.F("serif_bold", 24), fill=ink)
        d.text((lx, by + S(88)), data["subtitle"], font=self.F("serif_italic", 15), fill=accent)
        d.line([(lx, by + S(116)), (bx + pw - S(40), by + S(116))], fill=(140, 110, 80, 160), width=1)
        fbody = self.F("serif", 14)
        y = by + S(132)
        paras = data.get("paragraphs") or []
        if data["key"] == "log":
            paras = self._field_log_lines(world)
        for ptxt in paras:
            is_eq = "=" in ptxt and len(ptxt) < 70 and ("sqrt" in ptxt or "P_" in ptxt or "U_" in ptxt)
            f = self.F("mono", 12.5) if is_eq else fbody
            lines = wrap(d, ptxt, f, pw - S(96))
            for ln in lines:
                if y > by + bh - S(70):
                    break
                d.text((lx + (S(14) if is_eq else 0), y), ln, font=f, fill=accent if is_eq else ink)
                y += S(21)
            y += S(9)
        # right page: diagram / blueprint
        rx0, ry0, rx1, ry1 = bx + pw + S(40), by + S(40), bx + bw - S(40), by + bh - S(80)
        if data.get("blueprint") is not None:
            self._draw_blueprint(d, world, data["blueprint"], (rx0, ry0, rx1, ry1), data["locked"], data["pinned"], img)
        elif data.get("diagram"):
            self._draw_diagram(d, world, data["diagram"], (rx0, ry0, rx1, ry1))
        # footer
        fl = self.F("regular", 11)
        d.text((lx, by + bh - S(36)), f"{data['page_number']} / {data['total_pages']}", font=fl, fill=faded)
        hint = "A / D  turn pages      Enter  pin blueprint      B  close"
        d.text((bx + bw - S(40) - text_w(d, hint, fl), by + bh - S(36)), hint, font=fl, fill=faded)
        if data.get("status"):
            st = data["status"]
            d.text((bx + pw + S(40), by + bh - S(62)), st[:80], font=self.F("serif_italic", 12), fill=accent)

    def _field_log_lines(self, world) -> List[str]:
        s = world.stats
        g = world.thermal_grid
        return [
            f"Fires lit: {s['fires_lit']}      Pieces burned to coals: {s['pieces_burned']}",
            f"Hottest coals: {s['best_temp']:.0f} C      Longest fire: {s['longest_fire_min']:.1f} min",
            f"Heat released: {s['energy_mj']:.1f} MJ from {s['fuel_kg']:.2f} kg of fuel"
            + (f" ({s['energy_mj'] / max(0.01, s['fuel_kg']):.1f} MJ/kg)" if s['fuel_kg'] > 0.01 else ""),
            f"Water boiled out of your fuel: {g.total_water_evaporated_kg:.2f} kg",
            f"Trees felled: {s['trees_felled']}      Friction embers made: {s['embers_made']}",
            f"Days in the wild: {s['days_elapsed']:.1f}",
            "Dry wood yields ~18 MJ/kg. Every kilogram of water you burn off costs 2.26 MJ - season your fuel.",
        ]

    def _draw_blueprint(self, d, world, bp, box, locked, pinned, img=None):
        S = self.S
        x0, y0, x1, y1 = box
        d.rounded_rectangle(box, radius=S(8), fill=(22, 42, 66, 255), outline=(120, 170, 230, 255), width=max(1, S(2)))
        # grid
        for gx in range(int(x0), int(x1), S(20)):
            d.line([(gx, y0), (gx, y1)], fill=(36, 60, 88, 255))
        for gy in range(int(y0), int(y1), S(20)):
            d.line([(x0, gy), (x1, gy)], fill=(36, 60, 88, 255))
        fb = self.F("mono", 12)
        d.text((x0 + S(12), y0 + S(10)), f"BLUEPRINT  {bp.name.upper()}", font=fb, fill=(190, 220, 255, 255))
        pit = world.thermal_grid.pit_center
        pieces = [(b.geometry_type, b.position - pit, b.axis) for b in world.thermal_grid.bodies]
        frac = bp.match_pieces(pieces)
        # top view (left) and side view (right)
        views = [("TOP", 0, 2, (x0 + S(20), y0 + S(40), (x0 + x1) / 2 - S(10), y1 - S(70))),
                 ("SIDE", 0, 1, ((x0 + x1) / 2 + S(10), y0 + S(40), x1 - S(20), y1 - S(70)))]
        for name, ax_a, ax_b, (vx0, vy0, vx1, vy1) in views:
            d.text((vx0, vy0 - S(2)), name, font=self.F("mono", 10), fill=(150, 190, 235, 255))
            cx, cy = (vx0 + vx1) / 2, (vy0 + vy1) / 2 + (S(30) if name == "SIDE" else 0)
            sc = min(vx1 - vx0, vy1 - vy0) / 1.05
            if name == "TOP":
                d.ellipse([cx - 0.45 * sc, cy - 0.45 * sc, cx + 0.45 * sc, cy + 0.45 * sc], outline=(120, 160, 210, 150), width=1)
            else:
                d.line([(vx0, cy), (vx1, cy)], fill=(120, 160, 210, 180), width=1)
            for slot in bp.slots:
                p = slot.position
                ax = slot.axis
                L, r = slot.dimensions[0], slot.dimensions[1]
                col = (140, 240, 170, 255) if slot.is_filled else (200, 225, 255, 230)
                if slot.expected_type == "tinder":
                    col = (255, 170, 80, 255) if not slot.is_filled else (140, 240, 170, 255)
                    a_, b_ = p[ax_a] * sc + cx, (-p[ax_b] if name == "SIDE" else p[ax_b]) * sc + cy
                    rr = max(S(4), L * 0.5 * sc * (1.0 if name == "TOP" else 0.4))
                    d.ellipse([a_ - rr, b_ - rr * (1 if name == "TOP" else 0.6), a_ + rr, b_ + rr * (1 if name == "TOP" else 0.6)], outline=col, width=max(1, S(2)))
                    continue
                e1 = p - ax * L / 2
                e2 = p + ax * L / 2
                sgn = -1.0 if name == "SIDE" else 1.0
                pa = (e1[ax_a] * sc + cx, sgn * e1[ax_b] * sc + cy)
                pb = (e2[ax_a] * sc + cx, sgn * e2[ax_b] * sc + cy)
                wpx = max(2, int(2 * r * sc))
                d.line([pa, pb], fill=col, width=wpx)
        fs = self.F("regular", 11.5)
        d.text((x0 + S(12), y1 - S(56)), bp.ideal_use[:70], font=fs, fill=(200, 220, 245, 255))
        d.text((x0 + S(12), y1 - S(36)), bp.draft_profile[:72], font=self.F("light", 11), fill=(170, 195, 230, 255))
        status = f"Your pit matches {frac * 100:.0f}%" + ("   -   PINNED as holographic guide" if pinned else "   -   Enter to pin")
        d.text((x0 + S(12), y1 - S(18)), status, font=self.F("medium", 11), fill=(140, 240, 170, 255) if pinned else (255, 200, 120, 255))
        if locked:
            if img is not None:
                comp_shape(img, box, (20, 20, 25, 205), radius=S(8))
            else:
                d.rounded_rectangle(box, radius=S(8), fill=(20, 20, 25, 255))
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            r = S(46)
            d.ellipse([cx - r, cy - r - S(20), cx + r, cy + r - S(20)], fill=(140, 30, 30, 255), outline=(90, 15, 15, 255), width=max(1, S(3)))
            fl = self.F("serif_bold", 15)
            d.text((cx - text_w(d, "LOCKED", fl) / 2, cy - S(30)), "LOCKED", font=fl, fill=(240, 210, 200, 255))
            msg = "Buy the Field Reference Book Editions, or discover it by building one yourself."
            for i, ln in enumerate(wrap(d, msg, self.F("serif_italic", 13), x1 - x0 - S(60))):
                d.text((cx - text_w(d, ln, self.F("serif_italic", 13)) / 2, cy + S(40) + i * S(18)), ln, font=self.F("serif_italic", 13), fill=(230, 220, 210, 255))

    def _draw_diagram(self, d, world, kind, box):
        S = self.S
        x0, y0, x1, y1 = box
        ink = (60, 42, 28, 255)
        f = self.F("serif", 12)
        fs = self.F("regular", 10.5)
        w, h = x1 - x0, y1 - y0
        if kind == "stages":
            bands = [(0, 100, "Ambient", (150, 170, 190)), (100, 200, "Dehydration", (190, 215, 235)), (200, 450, "Pyrolysis", (235, 205, 110)),
                     (450, 900, "Flaming & glowing", (250, 140, 60)), (900, 1100, "White-hot", (255, 230, 190))]
            bx0, bx1 = x0 + S(40), x1 - S(20)
            yb = y0 + S(60)
            for lo, hi, lab, c in bands:
                a = bx0 + (bx1 - bx0) * lo / 1100.0
                b = bx0 + (bx1 - bx0) * hi / 1100.0
                d.rectangle([a, yb, b, yb + S(40)], fill=(*c, 255))
                d.text((a + S(4), yb + S(46) + (S(14) if lab in ("Pyrolysis", "White-hot") else 0)), lab, font=fs, fill=ink)
            for t in range(0, 1101, 200):
                xx = bx0 + (bx1 - bx0) * t / 1100.0
                d.line([(xx, yb - S(6)), (xx, yb)], fill=ink)
                d.text((xx - S(10), yb - S(22)), f"{t}", font=fs, fill=ink)
            d.text((bx0, yb - S(44)), "surface temperature (C)", font=f, fill=ink)
            # stacked compound decomposition windows
            yy = yb + S(100)
            for lab, lo, hi, c in (("hemicellulose", 200, 260, (200, 150, 90)), ("cellulose", 240, 350, (180, 120, 70)), ("lignin", 280, 500, (120, 80, 50))):
                a = bx0 + (bx1 - bx0) * lo / 1100.0
                b = bx0 + (bx1 - bx0) * hi / 1100.0
                d.rounded_rectangle([a, yy, b, yy + S(14)], radius=S(5), fill=(*c, 255))
                d.text((b + S(8), yy - S(1)), lab, font=fs, fill=ink)
                yy += S(24)
            d.text((bx0, yy + S(16)), "Draper point ~525 C: solids begin to glow", font=self.F("serif_italic", 12), fill=(150, 70, 30, 255))
            d.text((bx0, yy + S(36)), "C + O2 -> CO2 + 393.5 kJ/mol", font=self.F("mono", 12), fill=(150, 70, 30, 255))
        elif kind == "species":
            from firestarter.thermodynamics.wood_properties import WOOD_MATRIX
            rows = list(WOOD_MATRIX.values())
            cols = [("density kg/m3", lambda p: p.bulk_density / 800.0, (140, 100, 60)),
                    ("resin %", lambda p: p.resin_content / 0.25, (220, 160, 60)),
                    ("coal MJ/kg", lambda p: p.charcoal_caloric_value / 30e6, (200, 80, 40))]
            yy = y0 + S(20)
            for p in rows:
                d.text((x0 + S(10), yy), p.species.value, font=self.F("serif_bold", 13), fill=ink)
                for i, (lab, fn, c) in enumerate(cols):
                    by_ = yy + S(20) + i * S(13)
                    v = fn(p)
                    d.rectangle([x0 + S(120), by_, x0 + S(120) + (w - S(220)) * min(1.0, v), by_ + S(9)], fill=(*c, 255))
                    d.text((x1 - S(92), by_ - S(3)), lab, font=self.F("regular", 9), fill=ink)
                yy += S(72)
        elif kind == "moisture":
            cases = [("wet surface / dry core", (0.65, 0.12)), ("dry surface / wet core", (0.08, 0.85)), ("green log", (0.80, 1.10))]
            for i, (lab, (ms, mc)) in enumerate(cases):
                cx = x0 + w * (0.18 + 0.32 * i)
                cy = y0 + S(120)
                R = S(60)
                wet = lambda m: lerp_col((235, 210, 160, 255), (70, 110, 170, 255), min(1.0, m / 1.1))
                d.ellipse([cx - R, cy - R, cx + R, cy + R], fill=wet(ms), outline=ink, width=max(1, S(2)))
                d.ellipse([cx - R * 0.75, cy - R * 0.75, cx + R * 0.75, cy + R * 0.75], fill=wet(mc))
                for rr in (0.55, 0.35, 0.15):
                    d.ellipse([cx - R * rr, cy - R * rr, cx + R * rr, cy + R * rr], outline=(0, 0, 0, 40))
                lines = wrap(d, lab, fs, S(130))
                for j, ln in enumerate(lines):
                    d.text((cx - text_w(d, ln, fs) / 2, cy + R + S(10) + j * S(14)), ln, font=fs, fill=ink)
                d.text((cx - S(36), cy + R + S(42)), f"{ms * 100:.0f}% / {mc * 100:.0f}%", font=self.F("mono", 10.5), fill=ink)
        elif kind == "draft":
            cx = (x0 + x1) / 2
            base = y1 - S(60)
            hgt = S(230)
            d.polygon([(cx - S(90), base), (cx - S(20), base - hgt), (cx + S(20), base - hgt), (cx + S(90), base)], outline=ink, width=max(1, S(2)))
            for k in range(4):
                yy = base - S(30) - k * S(50)
                d.line([(cx, yy), (cx, yy - S(30))], fill=(200, 80, 40, 255), width=max(1, S(3)))
                d.polygon([(cx, yy - S(38)), (cx - S(7), yy - S(26)), (cx + S(7), yy - S(26))], fill=(200, 80, 40, 255))
            for sgn in (-1, 1):
                d.line([(cx + sgn * S(160), base - S(12)), (cx + sgn * S(96), base - S(12))], fill=(60, 110, 170, 255), width=max(1, S(3)))
                d.text((cx + sgn * S(150) - S(20), base - S(34)), "O2 in", font=fs, fill=(60, 110, 170, 255))
            d.text((cx + S(30), base - hgt + S(10)), "H", font=self.F("serif_bold", 16), fill=ink)
            g = world.thermal_grid
            d.text((x0 + S(10), y0 + S(10)), f"Your fire now: U = {g.draft_velocity:.2f} m/s, O2 = {g.oxygen_fraction * 100:.1f} %, C_d = {g.c_d:.2f}",
                   font=self.F("serif_italic", 12.5), fill=(150, 70, 30, 255))
        elif kind == "bowdrill":
            cx = x0 + w * 0.45
            by_ = y1 - S(90)
            d.rectangle([x0 + S(40), by_, x1 - S(60), by_ + S(22)], fill=(205, 165, 115, 255), outline=ink)
            d.polygon([(cx - S(12), by_ + S(22)), (cx, by_ + S(2)), (cx + S(12), by_ + S(22))], fill=(90, 60, 40, 255))
            d.ellipse([cx - S(5), by_ + S(14), cx + S(5), by_ + S(24)], fill=(255, 110, 40, 255))
            d.rectangle([cx - S(8), by_ - S(200), cx + S(8), by_], fill=(220, 190, 140, 255), outline=ink)
            d.ellipse([cx - S(40), by_ - S(232), cx + S(40), by_ - S(196)], fill=(150, 150, 145, 255), outline=ink)
            d.arc([cx - S(170), by_ - S(170), cx + S(170), by_ - S(60)], 190, 350, fill=(120, 80, 45, 255), width=max(2, S(6)))
            d.line([(cx - S(165), by_ - S(118)), (cx + S(165), by_ - S(118))], fill=(200, 180, 140, 255), width=max(1, S(2)))
            for lab, (lx, ly) in (("bearing stone", (cx + S(50), by_ - S(226))), ("spindle", (cx + S(16), by_ - S(160))),
                                  ("bow & cord", (cx - S(160), by_ - S(180))), ("hearthboard", (x0 + S(44), by_ + S(28))),
                                  ("V-notch + ember", (cx + S(20), by_ + S(28)))):
                d.text((lx, ly), lab, font=fs, fill=ink)
            bd = world.crafting.bow_drill
            d.text((x0 + S(10), y0 + S(8)), f"Your socket: {bd.socket_temp:.0f} C   dust {bd.wood_dust_mass:.2f} g   notch: {'yes' if bd.has_v_notch else 'no'}",
                   font=self.F("serif_italic", 12.5), fill=(150, 70, 30, 255))
        elif kind == "breath":
            gx0, gy0, gx1, gy1 = x0 + S(40), y0 + S(30), x1 - S(20), y1 - S(60)
            d.line([(gx0, gy1), (gx1, gy1)], fill=ink, width=max(1, S(2)))
            d.line([(gx0, gy0), (gx0, gy1)], fill=ink, width=max(1, S(2)))
            for lo, hi, c, lab in ((0, 0.8, (150, 170, 200), "too soft"), (1.5, 4.0, (140, 210, 150), "deep, even"), (6.0, 7.0, (235, 120, 100), "scatters")):
                a = gx0 + (gx1 - gx0) * lo / 7.0
                b = gx0 + (gx1 - gx0) * hi / 7.0
                d.rectangle([a, gy0, b, gy1], fill=(*c, 90))
                d.text((a + S(4), gy0 + S(4)), lab, font=fs, fill=ink)
            from firestarter.thermodynamics.airflow_draft import AirflowDraftSolver
            pts = []
            for i in range(60):
                v = 0.1 + 6.8 * i / 59
                net, _, blown = AirflowDraftSolver.calculate_breath_effect(v, 450.0, 20.0, reactivity=8.0)
                yv = gy1 - (gy1 - gy0) * 0.5 - net * (gy1 - gy0) * 0.02
                pts.append((gx0 + (gx1 - gx0) * v / 7.0, max(gy0, min(gy1, yv))))
            d.line(pts, fill=(180, 70, 30, 255), width=max(1, S(2)))
            d.text((gx0, gy1 + S(8)), "breath velocity 0 - 7 m/s     (curve: net heating of the coal)", font=fs, fill=ink)
        elif kind == "fungi":
            cx = x0 + w * 0.3
            d.rectangle([cx - S(40), y0 + S(20), cx + S(40), y1 - S(40)], fill=(150, 140, 125, 255), outline=ink)
            for k, yy in enumerate((y0 + S(90), y0 + S(170), y0 + S(250))):
                d.pieslice([cx + S(30), yy - S(30), cx + S(150), yy + S(30)], 270, 90, fill=(175, 140, 100, 255), outline=ink)
            tx = x0 + w * 0.55
            for i, ln in enumerate(["Tinder fungus  0.8 cm/h", "Birch polypore  1.3 cm/h", "Chaga  0.6 cm/h", "", "An 8 cm bracket carries", "an ember for ~6-13 hours."]):
                d.text((tx, y0 + S(60) + i * S(24)), ln, font=f, fill=ink)
        elif kind == "seasoning":
            from firestarter.forestry.seasoning import RaisedDryingRack, GroundSeasoningSimulator
            from firestarter.simulation.inventory import make_log
            from firestarter.thermodynamics.wood_properties import WoodSpeciesType as WS
            gx0, gy0, gx1, gy1 = x0 + S(50), y0 + S(30), x1 - S(20), y1 - S(60)
            d.line([(gx0, gy1), (gx1, gy1)], fill=ink, width=max(1, S(2)))
            d.line([(gx0, gy0), (gx0, gy1)], fill=ink, width=max(1, S(2)))
            for mm in (0, 40, 80):
                yy = gy1 - (gy1 - gy0) * mm / 110.0
                d.text((x0 + S(10), yy - S(7)), f"{mm}%", font=fs, fill=ink)
            series = []
            rng = np.random.default_rng(1)
            for lab, col, tarp, ground, season in (("tarped rack, summer", (60, 140, 70), True, False, "Summer"),
                                                   ("open rack, autumn", (200, 150, 60), False, False, "Autumn"),
                                                   ("on the soil, autumn", (70, 100, 170), False, True, "Autumn")):
                log = make_log(WS.WHITE_OAK, 0.8, 1.1, rng=rng)
                log.radius = 0.06
                rack = RaisedDryingRack(np.zeros(3), has_tarp=tarp)
                if not ground:
                    rack.add_log(log)
                pts = []
                for day in range(0, 91, 3):
                    if day:
                        if ground:
                            GroundSeasoningSimulator.step_ground_log(log, 3, season, 0.45)
                        else:
                            rack.update_seasoning(3, season, rain_intensity=0.45 if not tarp else 0.15)
                    m = 0.3 * log.m_surface + 0.7 * log.m_core
                    pts.append((gx0 + (gx1 - gx0) * day / 90.0, gy1 - (gy1 - gy0) * min(1.1, m) / 1.1))
                d.line(pts, fill=(*col, 255), width=max(2, S(3)))
                series.append((lab, col))
            for i, (lab, col) in enumerate(series):
                d.rectangle([gx0 + S(10), gy0 + S(4) + i * S(18), gx0 + S(24), gy0 + S(14) + i * S(18)], fill=(*col, 255))
                d.text((gx0 + S(30), gy0 + i * S(18)), lab, font=fs, fill=ink)
            d.text((gx0, gy1 + S(8)), "days 0 - 90  (green oak, 12 cm)", font=fs, fill=ink)
        elif kind == "fieldlog":
            g = world.thermal_grid
            hist = g.temp_history[-180:]
            gx0, gy0, gx1, gy1 = x0 + S(30), y0 + S(40), x1 - S(20), y0 + S(230)
            d.text((gx0, y0 + S(10)), "core temperature, last 3 minutes", font=self.F("serif_italic", 13), fill=ink)
            d.rectangle([gx0, gy0, gx1, gy1], outline=ink)
            if len(hist) > 1:
                mx = max(600.0, max(hist))
                pts = [(gx0 + (gx1 - gx0) * i / (len(hist) - 1), gy1 - (gy1 - gy0) * v / mx) for i, v in enumerate(hist)]
                d.line(pts, fill=(190, 70, 30, 255), width=max(1, S(2)))
            hh = g.hrr_history[-180:]
            if len(hh) > 1:
                mx = max(1000.0, max(hh))
                pts = [(gx0 + (gx1 - gx0) * i / (len(hh) - 1), gy1 - (gy1 - gy0) * v / mx) for i, v in enumerate(hh)]
                d.line(pts, fill=(60, 110, 170, 200), width=1)
            d.text((gx0, gy1 + S(6)), "orange: core C     blue: heat release", font=fs, fill=ink)

    # ================================================================== time & pause menus
    def _draw_time_menu(self, d, world):
        from firestarter.simulation.time_engine import TIME_SKIP_OPTIONS
        S = self.S
        te = world.time_engine
        bw, bh = S(560), S(520)
        x, y = (self.width - bw) // 2, (self.height - bh) // 2
        d.rectangle([0, 0, self.width, self.height], fill=(0, 0, 0, 100))
        panel(d, [x, y, x + bw, y + bh], radius=S(16), fill=(16, 15, 13, 238), outline=(255, 176, 72, 120))
        d.text((x + S(24), y + S(18)), "PASS TIME", font=self.F("bold", 18), fill=PAL["amber"])
        w = world.weather
        d.text((x + S(24), y + S(46)), f"Now: {w.current_season.value}, day {int(w.day_of_season)}, {w.clock_string()}  -  {w.describe()}",
               font=self.F("regular", 12), fill=PAL["dim"])
        yy = y + S(80)
        for i, opt in enumerate(TIME_SKIP_OPTIONS):
            sel = i == te.selected_option_index
            h = S(50)
            d.rounded_rectangle([x + S(18), yy, x + bw - S(18), yy + h - S(6)], radius=S(10),
                                fill=(60, 44, 26, 255) if sel else (26, 24, 21, 255), outline=PAL["amber"] if sel else None)
            d.text((x + S(32), yy + S(5)), opt.label, font=self.F("medium", 14), fill=PAL["text"] if sel else PAL["dim"])
            d.text((x + S(32), yy + S(25)), opt.description, font=self.F("regular", 10.5), fill=PAL["faint"] if not sel else PAL["dim"])
            yy += h
        f = self.F("regular", 11)
        hint = "W/S select     Enter pass time     T / Esc cancel"
        d.text((x + (bw - text_w(d, hint, f)) / 2, y + bh - S(30)), hint, font=f, fill=PAL["faint"])

    def _draw_pause(self, d, input_mgr):
        from firestarter.input.input_manager import PAUSE_ITEMS
        S = self.S
        s = input_mgr.settings
        d.rectangle([0, 0, self.width, self.height], fill=(0, 0, 0, 140))
        bw, bh = S(460), S(80) + len(PAUSE_ITEMS) * S(40)
        x, y = (self.width - bw) // 2, (self.height - bh) // 2
        panel(d, [x, y, x + bw, y + bh], radius=S(16), fill=(14, 16, 15, 240))
        d.text((x + S(26), y + S(20)), "PAUSED", font=self.F("bold", 20), fill=PAL["text"])
        d.text((x + S(130), y + S(27)), "Firestarter", font=self.F("light", 12), fill=PAL["faint"])
        vals = {
            "Master Volume": s.master_volume, "Ambience Volume": s.ambience_volume, "Effects Volume": s.effects_volume,
            "Graphics Preset": s.graphics_preset, "Mouse Sensitivity": s.mouse_sensitivity, "Invert Mouse Y": "On" if s.invert_y else "Off",
            "Sky Clock Speed": f"{s.clock_speed:.0f}x", "Show FPS": "On" if s.show_fps else "Off",
        }
        yy = y + S(66)
        for i, item in enumerate(PAUSE_ITEMS):
            sel = i == input_mgr.pause_index
            d.rounded_rectangle([x + S(16), yy, x + bw - S(16), yy + S(34)], radius=S(9), fill=(40, 42, 40, 255) if sel else (0, 0, 0, 0),
                                outline=PAL["amber"] if sel else None)
            d.text((x + S(30), yy + S(6)), item, font=self.F("medium", 14), fill=PAL["text"] if sel else PAL["dim"])
            v = vals.get(item)
            if isinstance(v, float) and item.endswith("Volume"):
                bx = x + bw - S(170)
                d.rounded_rectangle([bx, yy + S(14), bx + S(120), yy + S(20)], radius=S(3), fill=(255, 255, 255, 40))
                d.rounded_rectangle([bx, yy + S(14), bx + S(120) * v, yy + S(20)], radius=S(3), fill=PAL["amber"])
                d.text((bx + S(128), yy + S(6)), f"{v * 100:.0f}", font=self.F("mono", 11), fill=PAL["dim"])
            elif v is not None:
                txt = f"{v:.2f}" if isinstance(v, float) else str(v)
                d.text((x + bw - S(40) - text_w(d, txt, self.F("regular", 13)), yy + S(7)), f"< {txt} >" if sel else txt, font=self.F("regular", 13), fill=PAL["text"])
            yy += S(40)
        f = self.F("regular", 11)
        hint = "W/S select     A/D change     Enter confirm     Esc resume"
        d.text((x + (bw - text_w(d, hint, f)) / 2, y + bh - S(26)), hint, font=f, fill=PAL["faint"])
