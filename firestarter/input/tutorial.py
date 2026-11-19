"""
Firestarter - Guided Early-Game Tutorial (README "Early-Game Tutorial Walkthrough").

Each step completes automatically by observing real game state, so the tutorial can never
get out of sync with what the player actually did.
"""

from typing import Dict, Any, List, Callable, Optional


class TutorialStep:
    def __init__(self, title: str, text: str, check: Callable[[Any], bool], hint: str = ""):
        self.title = title
        self.text = text
        self.check = check
        self.hint = hint
        self.done = False


class Tutorial:
    def __init__(self, game_state: Any):
        self.gs = game_state   # object exposing world, phone, book
        self.active = True
        self.index = 0
        self.completed_at: Optional[float] = None
        self.flash = 0.0
        w = lambda: self.gs.world
        self.steps: List[TutorialStep] = [
            TutorialStep("Gather tinder",
                         "Pick up dry tinder: birch bark curls, pine needles or dry grass (aim + LMB).",
                         lambda gs: gs.world.inventory.count("tinder") >= 2 or self._pit_has("tinder"),
                         "Tinder piles glow softly when your crosshair is on them."),
            TutorialStep("Gather kindling",
                         "Collect pencil-thin dead twigs from the kindling bundle (LMB).",
                         lambda gs: gs.world.inventory.count("kindling") >= 4 or self._pit_count("kindling") >= 3,
                         "The bundle of twigs lies on the ground right beside the pit."),
            TutorialStep("Build the fire lay",
                         "Select tinder (6) and place it in the pit (RMB). Then lean kindling (7) around it in a mini-teepee.",
                         lambda gs: self._pit_has("tinder") and self._pit_count("kindling") >= 3,
                         "Tip: press B and pin the Teepee page (Enter) for a holographic guide."),
            TutorialStep("Strike the lighter",
                         "Equip the lighter (1), step close and hold LMB on the tinder. Watch the fluid gauge.",
                         lambda gs: gs.world.thermal_grid.flame_hrr > 800.0,
                         "Crouch (Ctrl) to get your hands right into the fire lay."),
            TutorialStep("Feed the flames",
                         "When the kindling is burning hot (>300 C), add split firewood (8). Blow (F) if it wavers.",
                         lambda gs: self._burning_fuel_count() >= 1,
                         "Pencil-thin first, then thumb, then wrist-thick. Never smother it."),
            TutorialStep("Claim your reward",
                         "Let it settle into glowing coals, then open your phone (Tab) to claim fire-mastery credits.",
                         lambda gs: gs.phone.last_claimed > 0.5 or gs.phone.credits.total_earned > 25.0 and gs.phone.is_open,
                         "Hotter, cleaner, more efficient fires earn more credits."),
        ]

    # ---------------------------------------------------------------- state helpers
    def _pit_has(self, kind: str) -> bool:
        return any(b.geometry_type == kind for b in self.gs.world.thermal_grid.bodies)

    def _pit_count(self, kind: str) -> int:
        return sum(1 for b in self.gs.world.thermal_grid.bodies if b.geometry_type == kind)

    def _burning_fuel_count(self) -> int:
        return sum(1 for b in self.gs.world.thermal_grid.bodies
                   if b.geometry_type in ("firewood", "log") and (b.is_flaming or b.t_surface > 350))

    # ---------------------------------------------------------------- api
    @property
    def step_number(self) -> int:
        return self.index + 1

    @property
    def current(self) -> Optional[TutorialStep]:
        if not self.active or self.index >= len(self.steps):
            return None
        return self.steps[self.index]

    def update(self, dt: float, now: float):
        self.flash = max(0.0, self.flash - dt)
        step = self.current
        if step is None:
            return
        # allow several steps to complete in one frame if the player raced ahead
        guard = 0
        while step is not None and step.check(self.gs) and guard < len(self.steps):
            step.done = True
            self.index += 1
            self.flash = 1.5
            guard += 1
            step = self.current
            if step is None:
                self.completed_at = now
                self.gs.world.notify("Tutorial complete! Explore: fell trees, season wood, master the bow drill.", 8.0, "success")

    def skip(self):
        self.active = False

    def hint_text(self) -> str:
        s = self.current
        if s is None:
            return ""
        return f"Step {self.step_number}/{len(self.steps)} - {s.title}: {s.text}"

    def state(self) -> Dict[str, Any]:
        s = self.current
        return {"active": s is not None, "index": self.index, "total": len(self.steps),
                "title": s.title if s else "", "text": s.text if s else "", "hint": s.hint if s else "",
                "flash": self.flash}
