"""Bot intelligence levels: what a bot notices, remembers and plans at each level.

The host picks one level for the game's bots (Pause > Multiplayer > Bot skill, or
--bot-level).  Every level moves with the same body and the same controls; what
changes is the thinking (game/tactics.py):

Simple   - sees what is in front of it, keeps one emblem in mind and the last few places
           it looked, and wanders off somewhere it has not been to find more; a living bot
           runs for the spot furthest from the zombies.
Complex  - looks about more, keeps three emblems in mind and a quarter of the city as
           searched, searches methodically (the nearest unsearched ground, a tall roof for a
           view over it); a living bot up on a roof with a lead runs on across the rooftops.
Hard     - sees all round and far, never forgets an emblem it has seen, and shares what every
           Hard bot on its side sees and searches (they split the emblems and the map between
           them, and leave an emblem a rival will reach first); in capture the flag one waits
           in a fleeing thief's base; a living bot on the rooftops runs on across them more
           readily.
Chasing a runner (zombie tag, a flag thief) is a straight run at it at every level, and a
fleeing bot otherwise runs by the game's plain rule: both were measured to beat every
cleverer alternative that was tried (game/tactics.py, DESIGN.md 11.6).
"""

SIMPLE = "simple"
COMPLEX = "complex"
HARD = "hard"
LEVELS = (SIMPLE, COMPLEX, HARD)
LABELS = {SIMPLE: "Simple", COMPLEX: "Complex", HARD: "Hard"}
DEFAULT = COMPLEX


class Skill:
    """The numbers behind a level (see the module docstring)."""

    def __init__(self, name, sight_range, sight_cone_deg, emblem_memory, search_memory, position_error,
                 shared, pursuit, flee, replan, explore, scan_cells, vantage):
        self.name = name
        self.sight_range = float(sight_range)          # how far an emblem can be spotted (m)
        self.sight_cone_deg = float(sight_cone_deg)    # full angle it notices things in (360: all round)
        self.emblem_memory = emblem_memory             # emblems it can keep in mind at once (None: all)
        self.search_memory = search_memory             # searched areas it remembers (None: all)
        self.position_error = float(position_error)    # error of a sighting, per metre of distance
        self.shared = bool(shared)                     # pools sightings and searched areas with its side
        self.pursuit = pursuit                         # "direct" / "pack" (+ a camper in a thief's base)
        self.flee = flee                               # "basic" / "evasive" / "mastermind"
        self.replan = float(replan)                    # seconds between tactical decisions in a chase
        self.explore = explore                         # "wander" / "frontier" / "sweep"
        self.scan_cells = int(scan_cells)              # areas it checks over per look round
        self.vantage = bool(vantage)                   # climbs for a view over unsearched ground

    @property
    def label(self):
        return LABELS[self.name]

    def __repr__(self):
        return f"Skill({self.name})"


SKILLS = {
    SIMPLE: Skill(SIMPLE, sight_range=70.0, sight_cone_deg=110.0, emblem_memory=1, search_memory=6,
                  position_error=0.08, shared=False, pursuit="direct", flee="basic", replan=1.0,
                  explore="wander", scan_cells=6, vantage=False),
    COMPLEX: Skill(COMPLEX, sight_range=100.0, sight_cone_deg=170.0, emblem_memory=3, search_memory=100,
                   position_error=0.04, shared=False, pursuit="direct", flee="evasive", replan=0.6,
                   explore="frontier", scan_cells=14, vantage=True),
    HARD: Skill(HARD, sight_range=140.0, sight_cone_deg=360.0, emblem_memory=None, search_memory=None,
                position_error=0.0, shared=True, pursuit="pack", flee="mastermind", replan=0.3,
                explore="sweep", scan_cells=24, vantage=True),
}


def normalize(level):
    """A valid level name (anything unknown is the default)."""
    level = (level or "").strip().lower() if isinstance(level, str) else ""
    return level if level in SKILLS else DEFAULT


def skill(level):
    return SKILLS[normalize(level)]


def cycle(level, step):
    i = LEVELS.index(normalize(level))
    return LEVELS[(i + (1 if step > 0 else -1)) % len(LEVELS)]
