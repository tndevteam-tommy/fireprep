"""
Firestarter - Bushcraft Crafting Controller
Coordinates two-handed bushcraft operations: knife whittling / carving / bark stripping,
cordage twisting, lighter strikes, bow-drill friction, ember nesting and blowing, and
smouldering fungus ember carriers.  Resources come from the player's ``Inventory``.
"""

from typing import Dict, Any, Optional

from firestarter.bushcraft.bow_drill import BowDrillSystem
from firestarter.bushcraft.hand_drill import HandDrillSystem
from firestarter.bushcraft.tinder_nest import TinderNest
from firestarter.bushcraft.fungi import SmolderingFungus, FungusType

LIGHTER_FLAME_SECONDS = 14.0   # one "fire start" worth of butane held open


class BushcraftCraftingController:
    def __init__(self, inventory=None):
        if inventory is None:
            from firestarter.simulation.inventory import Inventory
            inventory = Inventory()
        self.inventory = inventory
        self.bow_drill = BowDrillSystem(has_v_notch=False)
        self.hand_drill = HandDrillSystem(socket=self.bow_drill)   # drills into the same hearthboard socket
        self.active_tinder_nest: Optional[TinderNest] = None
        self.smoldering_fungus: Optional[SmolderingFungus] = None
        self.lighter_is_lit = False
        self.active_task = "idle"   # idle, whittling, twisting_cordage, bow_drilling, blowing, striking_lighter, chopping
        self.task_timer = 0.0
        self.audio_trigger: Optional[str] = None
        self.audio_queue = []
        self.ambient = 20.0

    # ---------------------------------------------------------------- convenience
    @property
    def knife_sharpness(self) -> float:
        return 1.0 if self.inventory.knife == "steel" else 0.7

    @property
    def axe_type(self) -> str:
        return self.inventory.axe

    @property
    def lighter_fluid_starts(self) -> int:
        return self.inventory.lighter_starts

    def _cue(self, name: str):
        self.audio_trigger = name
        self.audio_queue.append(name)

    def _task(self, name: str, duration: float = 0.6):
        self.active_task = name
        self.task_timer = duration

    # ---------------------------------------------------------------- knife work
    def whittle_shavings(self) -> Dict[str, Any]:
        """Whittles a dry kindling stick into curls of shavings (a feather stick)."""
        from firestarter.simulation.inventory import make_tinder
        inv = self.inventory
        stick = inv.take("kindling")
        if stick is None:
            return {"success": False, "message": "No kindling stick to whittle. Gather dry twigs first."}
        material = "cedar_shavings" if stick.species.name == "RED_CEDAR" else "feather_stick"
        n = 2 if inv.knife == "steel" else 1
        for _ in range(n):
            inv.add_fuel(make_tinder(material, moisture=max(0.05, stick.m_surface), source="whittled"))
        self._cue("whittle")
        self._task("whittling", 1.2)
        return {"success": True, "message": f"Whittled crisp {stick.species.value.lower()} curls into {n} tinder bundle(s).",
                "shavings_count": inv.count("tinder")}

    def strip_tree_bark(self) -> Dict[str, Any]:
        """Harvests inner fibrous cambium bark for cordage."""
        self.inventory.bark_strips += 2
        self._cue("whittle")
        self._task("whittling", 1.0)
        return {"success": True, "message": "Stripped long ribbons of inner bark fibre (2 strips).",
                "bark_strips": self.inventory.bark_strips}

    def twist_two_ply_cordage(self) -> Dict[str, Any]:
        """Two-handed reverse-wrap twisting of bark fibres into two-ply cordage."""
        inv = self.inventory
        if inv.bark_strips < 2:
            return {"success": False, "message": "Need 2 strips of inner bark to twist cordage (knife + tree)."}
        inv.bark_strips -= 2
        inv.cordage += 1
        self._cue("cordage_twist")
        self._task("twisting_cordage", 2.5)
        return {"success": True, "message": "Twisted a strong two-ply reverse-wrap cord.", "cordage_count": inv.cordage}

    def carve_hearthboard_notch(self) -> Dict[str, Any]:
        """Carves a 45-degree V-notch collector into the hearthboard."""
        if self.bow_drill.has_v_notch and self.bow_drill.sockets_used == 0:
            return {"success": False, "message": "The hearthboard already has a clean 45-degree V-notch."}
        self.bow_drill.carve_v_notch()
        self._cue("whittle")
        self._task("whittling", 1.5)
        return {"success": True, "message": "Carved a 45-degree V-notch into the hearthboard socket."}

    # ---------------------------------------------------------------- friction fire
    def stroke_bow_drill(self, downward_force_n: float = 55.0, stroke_speed_m_s: float = 1.2, dt: float = 0.1,
                         stamina: float = 100.0) -> Dict[str, Any]:
        inv = self.inventory
        if self.bow_drill.cordage_condition <= 0.0:
            if inv.cordage > 0:
                inv.cordage -= 1
                self.bow_drill.replace_cordage()
                self._cue("whittle")
                return {"success": False, "restrung": True, "message": "Cord snapped - restrung the bow with fresh cordage.",
                        "socket_temp": self.bow_drill.socket_temp}
            return {"success": False, "message": "Cordage snapped and you have no spare. Strip bark (knife on a tree) and twist cord (C).",
                    "socket_temp": self.bow_drill.socket_temp}
        if stamina <= 1.0:
            return {"success": False, "message": "Exhausted - rest a moment.", "socket_temp": self.bow_drill.socket_temp}
        self.active_task = "bow_drilling"
        self.task_timer = 0.25
        res = self.bow_drill.stroke(0.35, stroke_speed_m_s, downward_force_n, dt)
        if self.bow_drill.cordage_condition <= 0.0:
            self._cue("cord_snap")
        return res

    def strike_lighter(self) -> Dict[str, Any]:
        """Strikes the pocket lighter (consumes one of its limited fire starts)."""
        inv = self.inventory
        if inv.lighter_burn_left > 0.0:
            self.lighter_is_lit = True
            return {"success": True, "message": "Lighter flame held.", "starts_remaining": inv.lighter_starts}
        if inv.lighter_starts <= 0:
            self._cue("lighter_fail")
            return {"success": False, "message": "Click... the pocket lighter is out of fuel. Buy refill fluid or make fire by friction."}
        inv.lighter_starts -= 1
        inv.lighter_burn_left = LIGHTER_FLAME_SECONDS
        self.lighter_is_lit = True
        self._cue("lighter_strike")
        self._task("striking_lighter", 0.4)
        return {"success": True, "message": "Lighter struck - pilot flame active.", "starts_remaining": inv.lighter_starts}

    def release_lighter(self):
        self.lighter_is_lit = False

    # ---------------------------------------------------------------- nest & breath
    def blow_tinder(self, breath_velocity: float = 2.5, dt: float = 0.6) -> Dict[str, Any]:
        self._cue("breath")
        self._task("blowing", 0.8)
        if not self.active_tinder_nest:
            return {"status": "no_tinder", "is_flaming": False}
        return self.active_tinder_nest.blow(breath_velocity, dt)

    def transfer_ember_to_nest(self) -> Dict[str, Any]:
        """Tips an ember (bow drill, hand drill or fungus) into a fresh tinder nest."""
        ember_t = self.bow_drill.extract_ember()
        src = "bow drill"
        if ember_t is None:
            ember_t = self.hand_drill.extract_ember()
            src = "hand drill"
        if ember_t is None and self.smoldering_fungus is not None:
            ember_t = self.smoldering_fungus.transfer_ember_to_tinder()
            src = "smouldering fungus"
        if ember_t is None:
            return {"success": False, "message": "No glowing ember to transfer."}
        if not self.active_tinder_nest or self.active_tinder_nest.is_destroyed:
            item = self.inventory.take("tinder")
            if item is None:
                # put the coal back - you need tinder to receive it
                if src == "bow drill":
                    self.bow_drill.has_ember = True
                    self.bow_drill.ember_temp = ember_t
                return {"success": False, "message": "You need a tinder bundle to receive the ember!"}
            self.active_tinder_nest = TinderNest(material=item.material or "cedar_shavings",
                                                 dry_mass_g=item.dry_mass * 1000.0, moisture=item.m_surface,
                                                 ambient_temp=self.ambient)
        self.active_tinder_nest.insert_ember(ember_t)
        self._cue("ember")
        return {"success": True, "message": f"Rolled the glowing coal from the {src} into the {self.active_tinder_nest.display_name.lower()}. Blow gently (F)!"}

    def light_fungus(self, fungus: SmolderingFungus, contact_temp: float) -> bool:
        if fungus.catch_spark_or_ember(contact_temp):
            self.smoldering_fungus = fungus
            return True
        return False

    # ---------------------------------------------------------------- update
    def update(self, dt: float, ambient_temp: float = 20.0):
        self.ambient = ambient_temp
        self.bow_drill.ambient_temp = ambient_temp
        if self.active_task not in ("bow_drilling", "hand_drilling"):
            self.bow_drill.idle(dt)
        self.task_timer -= dt
        if self.task_timer <= 0.0:
            self.active_task = "idle"
        inv = self.inventory
        if self.lighter_is_lit:
            inv.lighter_burn_left = max(0.0, inv.lighter_burn_left - dt)
            if inv.lighter_burn_left <= 0.0:
                self.lighter_is_lit = False
        if self.active_tinder_nest:
            self.active_tinder_nest.update(dt, ambient_temp)
            if self.active_tinder_nest.is_destroyed or (not self.active_tinder_nest.has_ember and not self.active_tinder_nest.is_flaming):
                if self.active_tinder_nest.is_destroyed or self.active_tinder_nest.breath_count > 0:
                    self.active_tinder_nest = None
        if self.smoldering_fungus:
            self.smoldering_fungus.update(dt, ambient_temp)
            if not self.smoldering_fungus.is_smoldering:
                if self.smoldering_fungus in inv.fungi:
                    inv.fungi.remove(self.smoldering_fungus)
                self.smoldering_fungus = None

    def pop_audio(self):
        q = self.audio_queue
        self.audio_queue = []
        self.audio_trigger = None
        return q
