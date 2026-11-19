from __future__ import annotations

from enum import Enum
from typing import Callable, Dict, Optional


class SkillType(Enum):
    SNEAK = "Sneak"
    LOCKPICKING = "Lockpicking"
    PICKPOCKET = "Pickpocket"
    STEALING = "Stealing"        # lifting goods from shelves, tables and chests
    BARTER = "Barter"            # buying and selling
    SMITHING = "Smithing"


class SkillTree:
    """Per-skill level and experience.

    Levels run from 1 to ``MAX_LEVEL`` (100). The XP needed for the next level grows
    *linearly* (``BASE_XP_NEEDED`` + ``XP_PER_LEVEL`` per level gained) so that elite levels
    stay reachable; the slow-down at high levels comes from the rewards themselves, which
    shrink as the easy deeds fall further below the player (``systems/progression.py``).
    """
    BASE_XP_NEEDED = 80.0
    XP_PER_LEVEL = 12.0
    MAX_LEVEL = 100

    def __init__(self):
        self.skills: Dict[SkillType, Dict[str, float]] = {
            skill: {"level": 1, "xp": 0.0, "xp_needed": self.xp_needed_for(1)} for skill in SkillType
        }

    @classmethod
    def xp_needed_for(cls, level: int) -> float:
        return cls.BASE_XP_NEEDED + cls.XP_PER_LEVEL * (max(1, int(level)) - 1)

    def level(self, skill: SkillType) -> int:
        return int(self.skills[skill]["level"])

    def add_xp(self, skill: SkillType, amount: float, notify_callback: Optional[Callable[[str], None]] = None) -> int:
        """Add XP; returns how many levels were gained (a large grant can level several times)."""
        if amount <= 0:
            return 0
        data = self.skills[skill]
        if data["level"] >= self.MAX_LEVEL:
            return 0                                     # mastered: nothing more to learn
        data["xp"] += amount
        gained = 0
        while data["xp"] >= data["xp_needed"] and data["level"] < self.MAX_LEVEL:
            data["level"] += 1
            data["xp"] -= data["xp_needed"]
            data["xp_needed"] = self.xp_needed_for(data["level"])
            gained += 1
            if notify_callback:
                notify_callback(f"SKILL UP! Your {skill.value} skill increased to Level {data['level']}!")
        if data["level"] >= self.MAX_LEVEL:
            data["xp"] = 0.0
        return gained

    def to_dict(self) -> dict:
        return {skill.value: data for skill, data in self.skills.items()}

    def load_dict(self, data_dict: dict) -> None:
        """Load saved skills; the XP requirement is recomputed from the level so saves made
        under an older curve pick up the current one."""
        for skill in SkillType:
            if skill.value in data_dict:
                saved = dict(data_dict[skill.value])
                level = max(1, min(self.MAX_LEVEL, int(saved.get("level", 1))))
                needed = self.xp_needed_for(level)
                self.skills[skill] = {"level": level, "xp": float(saved.get("xp", 0.0)), "xp_needed": needed}
