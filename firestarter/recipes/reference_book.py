"""
Firestarter - Firemaster's Field Compendium ('B' key)
DESIGN Section 6 & README: fire recipe index, wood characteristics, friction fire, seasoning
science and a live thermodynamic field log.  Advanced architectures (Lean-To, Swedish Torch,
Star Fire) are locked until the "Field Reference Book Editions" are bought from the Store, or
until the player discovers them by building one freeform.
"""

from typing import List, Dict, Any, Optional

from firestarter.recipes.blueprints import ALL_BLUEPRINTS, FireBlueprint

PAGES: List[Dict[str, Any]] = [
    {
        "key": "combustion", "title": "I. The Four Stages of Combustion", "subtitle": "Thermodynamics & organic chemistry",
        "diagram": "stages",
        "content": [
            "Dehydration (100-120 C): water boils out of the wood, absorbing 2,260 kJ/kg. The temperature stalls near 100 C until the layer is dry - white steam, hissing.",
            "Pyrolysis (200-450 C): hemicellulose (200-260 C), cellulose (240-350 C) and lignin (280-500 C) break down into tar, CO and methane - thick yellow-grey smoke.",
            "Flaming (gas-phase, O2 > 12 %): once volatiles leave the surface faster than ~2 g/m2s they ignite. Flames radiate heat back into the fuel.",
            "Glowing coals (500-900 C): solid carbon oxidises, C + O2 -> CO2 + 393.5 kJ/mol. Hot coals make CO that burns as faint blue flames.",
            "Ash: the mineral blanket insulates coals and starves them of oxygen unless it is blown or knocked away.",
        ],
    },
    {
        "key": "species", "title": "II. Wood Species Matrix", "subtitle": "Density, resins and energy", "diagram": "species",
        "content": [
            "White Pine - softwood, 380 kg/m3, 18.5 % terpenes. Lights readily, burns hot and fast, pops and spits resin.",
            "Red Cedar - softwood, 360 kg/m3, 22 % resinous oils. Superb tinder and kindling; flares quickly.",
            "Paper Birch - hardwood, 650 kg/m3, betulin-rich bark that burns even when damp. Balanced fuel.",
            "White Oak - hardwood, 780 kg/m3, few volatiles but 28.6 MJ/kg coals. Hard to light, lasts for hours.",
            "Punk Wood - fungal-decayed sponge, 120 kg/m3. Bone dry it catches a faint spark and smoulders; wet it is a heat sink.",
        ],
    },
    {
        "key": "moisture", "title": "III. Moisture: Surface vs. Core", "subtitle": "Why wet wood hisses", "diagram": "moisture",
        "content": [
            "Wet surface, dry core: needs a sustained flame to steam the shell dry - then it catches all at once.",
            "Dry surface, wet core: catches easily, but when the heat wave reaches the core the boiling water pours steam through the flames and quenches them.",
            "Green wood (80-110 % moisture): cannot sustain flame without a big fire around it. Season it first.",
            "Stack damp wood near - not in - the fire. Radiant heat slowly dries it.",
        ],
    },
    {
        "key": "airflow", "title": "IV. Airflow & the Chimney", "subtitle": "Stack effect and base vacuum", "diagram": "draft",
        "content": [
            "U_draft = C_d * sqrt(2 g H (T_int - T_amb) / T_int)",
            "P_base = P_amb - 1/2 rho U_draft^2",
            "Taller structures (H) pull more air into the ember bed. Every kilogram of oxygen consumed releases ~13.1 MJ (Huggett).",
            "Gaps under 2 cm between parallel logs choke the intake: C_d -> 0, oxygen falls below 10 % and the fire smoulders and smokes.",
        ],
    },
    {"key": "teepee", "title": "V. Architecture: Teepee Fire", "subtitle": "Rapid ignition, strong draft", "blueprint": "teepee",
     "content": ["Tinder nest in the centre, pencil-thin kindling leaned into a cone, split fuel on the outside.",
                 "The cone is a chimney: flames climb the sticks (upward flame spread is fastest).",
                 "Leave an air channel facing upwind. Burns fast - feed it."]},
    {"key": "log_cabin", "title": "VI. Architecture: Log Cabin Fire", "subtitle": "Balanced multi-directional intake", "blueprint": "log_cabin",
     "content": ["Alternate pairs of split wood in square tiers around a small teepee of kindling.",
                 "Air flows in on all four sides; the walls trap radiant heat.",
                 "Collapses into an even, long-lived coal bed - ideal for cooking."]},
    {"key": "lean_to", "title": "VII. Architecture: Lean-To Fire", "subtitle": "Wind reflector & deflector", "blueprint": "lean_to",
     "content": ["Lay a thick back-log across the windward side.",
                 "Lean kindling from the lee side onto the log over the tinder.",
                 "The log shields the flame from gusts and reflects infrared heat toward you."]},
    {"key": "swedish_torch", "title": "VIII. Architecture: Swedish Torch", "subtitle": "The Canadian candle", "blueprint": "swedish_torch",
     "content": ["Stand split quarters of one log upright, tight together.",
                 "Pack shavings into the central bore and light them from the top.",
                 "The narrow bore is a fierce chimney: a steady torch flame with a flat cooking top."]},
    {"key": "star", "title": "IX. Architecture: Star Fire", "subtitle": "Radial low-fuel smoulder", "blueprint": "star",
     "content": ["Lay logs like spokes with only their tips in the central coals.",
                 "Push them inward as the tips burn away.",
                 "Very little wood is consumed - an all-night fire for the patient."]},
    {"key": "bow_drill", "title": "X. Friction Fire: the Bow Drill", "subtitle": "P = mu * F_N * omega * r", "diagram": "bowdrill",
     "content": ["Carve a 45-degree V-notch into the hearthboard (knife, LMB on the board).",
                 "Kneel (Ctrl), hold LMB and saw the mouse left-right in long even strokes. Hold RMB to bear down on the stone.",
                 "Dust collects in the notch and chars; above ~380 C it becomes a glowing coal.",
                 "Damp boards only steam. Frayed cord snaps - twist more from inner bark (C)."]},
    {"key": "tinder", "title": "XI. Tinder & Breath", "subtitle": "From coal to flame", "diagram": "breath",
     "content": ["Fold the coal into a nest of the finest, driest tinder you have.",
                 "Blow in slow, deep, even breaths (hold F ~0.3-0.8 s). Too soft cools it; too hard scatters it.",
                 "When the glowing core passes ~550 C the fibres flash into flame.",
                 "Birch bark burns even when damp (betulin). Cattail fluff catches sparks best."]},
    {"key": "fungi", "title": "XII. Fungi & Carrying Fire", "subtitle": "Amadou, chaga and birch polypore", "diagram": "fungi",
     "content": ["Tinder fungus and birch polypore grow on standing dead trees - harvest them with your knife.",
                 "A dried bracket lit from an ember smoulders 0.5-1.5 cm per hour without flame.",
                 "Carry it for hours and tip its coal into a tinder nest when you need fire."]},
    {"key": "seasoning", "title": "XIII. Seasoning, Tarps & Racks", "subtitle": "Equilibrium moisture content", "diagram": "seasoning",
     "content": ["Living sapwood holds 80-110 % water (dry basis).",
                 "A raised rack stops capillary wicking from the soil; a tarp stops rain but keeps the breeze.",
                 "A tarped, raised log reaches ~12 % in about 60 summer days. Split wood seasons ~4x faster.",
                 "Logs left on wet soil in autumn soak to ~95 % and rot into punk wood."]},
    {"key": "log", "title": "XIV. Thermodynamic Field Log", "subtitle": "Your fires, measured", "diagram": "fieldlog",
     "content": []},
]


class ReferenceBook:
    def __init__(self):
        self.is_open = False
        self.current_page = 0
        self.selected_blueprint_key: Optional[str] = None   # pinned holographic guide
        self.unlocked_editions = False
        self.discovered: set = set()
        self.field_log: Dict[str, Any] = {}
        self.status = ""

    @property
    def total_pages(self) -> int:
        return len(PAGES)

    def toggle(self) -> bool:
        self.is_open = not self.is_open
        return self.is_open

    def next_page(self):
        self.current_page = (self.current_page + 1) % self.total_pages
        self.status = ""

    def prev_page(self):
        self.current_page = (self.current_page - 1) % self.total_pages
        self.status = ""

    def goto(self, key: str):
        for i, p in enumerate(PAGES):
            if p["key"] == key:
                self.current_page = i

    def is_blueprint_unlocked(self, key: str) -> bool:
        bp = ALL_BLUEPRINTS.get(key)
        if bp is None:
            return False
        return bp.builtin or self.unlocked_editions or key in self.discovered

    def unlock_editions(self):
        self.unlocked_editions = True

    def discover(self, key: str) -> bool:
        if key in ALL_BLUEPRINTS and not self.is_blueprint_unlocked(key):
            self.discovered.add(key)
            return True
        return False

    def select_blueprint_for_overlay(self, key: Optional[str]) -> bool:
        if key is None:
            self.selected_blueprint_key = None
            return True
        if key in ALL_BLUEPRINTS and self.is_blueprint_unlocked(key):
            self.selected_blueprint_key = key
            return True
        return False

    def activate_current(self) -> str:
        """Enter / E on a blueprint page pins (or un-pins) it as the holographic build guide."""
        page = PAGES[self.current_page]
        key = page.get("blueprint")
        if not key:
            self.status = "Nothing to pin on this page."
            return self.status
        if not self.is_blueprint_unlocked(key):
            self.status = "Locked: buy the Field Reference Book Editions or discover it by building one."
            return self.status
        if self.selected_blueprint_key == key:
            self.selected_blueprint_key = None
            self.status = f"Unpinned the {ALL_BLUEPRINTS[key].name} guide."
        else:
            self.selected_blueprint_key = key
            self.status = f"Pinned {ALL_BLUEPRINTS[key].name} as your holographic guide in the fire pit."
        return self.status

    def get_current_page_data(self) -> Dict[str, Any]:
        page = dict(PAGES[self.current_page])
        key = page.get("blueprint")
        locked = bool(key) and not self.is_blueprint_unlocked(key)
        content = page.get("content", [])
        if isinstance(content, list):
            content = "\n".join(content)
        return {
            "page_number": self.current_page + 1,
            "total_pages": self.total_pages,
            "key": page["key"],
            "title": page["title"],
            "subtitle": page["subtitle"],
            "content": content,
            "paragraphs": PAGES[self.current_page].get("content", []),
            "diagram": page.get("diagram"),
            "blueprint": ALL_BLUEPRINTS.get(key) if key else None,
            "blueprint_key": key,
            "locked": locked,
            "pinned": key is not None and key == self.selected_blueprint_key,
            "field_log": self.field_log,
            "status": self.status,
        }
