"""Skill checks: what a situation's challenge is, and the odds against it.

Every check is ``progression.chance(effective level, challenge)`` - the same curve for every
skill. This module only says how hard each kind of attempt is (all numbers in
``StealthSettings``):

* **Sneaking** past an observer: their vigilance (by role - a villager -5, a townsperson 0, a
  guard 15, a bandit 22, a knight 34, a dungeon's master 50), +10 if they are already on edge,
  + ``sneak_visibility_challenge`` x how visible the player is (light, torch, stance and clothes).
* **Pickpocketing**: the victim's vigilance (-20 asleep) + the rarity of what is lifted.
  Separately, the *bystanders* notice a clumsy dip against ``bystander_challenge``.
* **Stealing** from a watched shelf unnoticed: ``theft_challenge`` + the item's rarity.
* **Securing** an epic or legendary piece (chained, mounted, warded): its ``secure_challenge``;
  one ``secure_gate`` levels beyond the player cannot even be attempted.
* **Lockpicking**: the lock's tier (a villager's latch -25 .. legendary 47). The mini-game's
  window and the hand's wobble follow the same margin, and the final "setting the pins" is
  a roll of exactly this chance - so with good timing the odds of a lock are ``chance``.
"""
from __future__ import annotations

import random
from typing import Optional

from ..config import LOCK_TIER_NAMES, ProgressionSettings, StealthSettings
from ..entities import Item, NPC, Player, SkillType
from .progression import DEFAULT as DEFAULT_PROGRESSION
from .progression import chance, effective_level, margin

ROLE_VIGILANCE = {
    "guard": "guard", "jailer": "guard", "smith": "merchant", "tailor": "merchant", "shopkeeper": "merchant",
    "innkeeper": "merchant", "fence": "fence", "rich": "rich", "villager": "villager", "farmer": "villager",
    "hunter": "hunter", "bandit": "bandit", "knight": "knight", "boss": "boss",
}


def vigilance_of(npc: NPC, s: StealthSettings) -> float:
    """How alert this person is by nature: the level's own figure for them, else their role's."""
    explicit = getattr(npc, "vigilance", None)
    if explicit is not None:
        return float(explicit)
    table = dict(s.vigilance)
    key = ROLE_VIGILANCE.get(getattr(npc, "role_tag", ""), "guard" if npc.is_guard else "townsfolk")
    return table.get(key, 0.0)


def observer_challenge(npc: NPC, s: StealthSettings, p: ProgressionSettings = DEFAULT_PROGRESSION) -> float:
    """An observer's vigilance, lower asleep, higher when already suspicious."""
    c = vigilance_of(npc, s)
    if npc.is_asleep:
        c += s.asleep_challenge
    elif npc.suspicion_meter >= npc.settings.schedule_override_threshold:
        c += s.suspicious_challenge
    return c


def rarity_challenge(item: Optional[Item], s: StealthSettings) -> float:
    if item is None:
        return 0.0
    return dict(s.rarity_challenge).get(getattr(item, "rarity", "common"), 0.0)


def sneak_challenge(npc: NPC, visibility: float, s: StealthSettings) -> float:
    return observer_challenge(npc, s) + s.sneak_visibility_challenge * max(0.0, min(1.5, visibility))


def sneak_chance(player: Player, npc: NPC, s: StealthSettings, visibility: float = 0.0,
                 p: ProgressionSettings = DEFAULT_PROGRESSION) -> float:
    """Odds one sneak roll keeps the player out of ``npc``'s mind."""
    return chance(effective_level(player, SkillType.SNEAK, p), sneak_challenge(npc, visibility, s), p)


def pickpocket_challenge(victim: NPC, item: Optional[Item], s: StealthSettings) -> float:
    return observer_challenge(victim, s) + rarity_challenge(item, s)


def pickpocket_chance(player: Player, victim: NPC, s: StealthSettings, p: ProgressionSettings = DEFAULT_PROGRESSION,
                      item: Optional[Item] = None) -> float:
    """Odds the victim does not feel this lift (of ``item``; None = the coin purse)."""
    return chance(effective_level(player, SkillType.PICKPOCKET, p), pickpocket_challenge(victim, item, s), p)


def pickpocket_unseen_chance(player: Player, s: StealthSettings, p: ProgressionSettings = DEFAULT_PROGRESSION) -> float:
    """Odds a dip is too deft for the *bystanders* to notice (only when this fails may they witness it)."""
    return chance(effective_level(player, SkillType.PICKPOCKET, p), s.bystander_challenge, p)


def theft_challenge(item: Optional[Item], s: StealthSettings) -> float:
    return s.theft_challenge + rarity_challenge(item, s)


def stealing_chance(player: Player, s: StealthSettings, p: ProgressionSettings = DEFAULT_PROGRESSION,
                    item: Optional[Item] = None) -> float:
    """Odds that lifting ``item`` from a shelf in plain view goes unnoticed."""
    return chance(effective_level(player, SkillType.STEALING, p), theft_challenge(item, s), p)


def secure_challenge(item: Item, s: StealthSettings) -> Optional[float]:
    """The challenge of working ``item`` free of its mount (None: it simply comes away)."""
    return dict(s.secure_challenge).get(getattr(item, "rarity", "common"))


def secure_verdict(player: Player, item: Item, s: StealthSettings, p: ProgressionSettings = DEFAULT_PROGRESSION):
    """(challenge, odds, possible) of getting a secured piece loose; (None, 1.0, True) for an
    ordinary one. ``possible`` is False when the challenge is ``secure_gate`` levels beyond the
    player's effective Stealing - they cannot even see how it is done."""
    c = secure_challenge(item, s)
    if c is None:
        return None, 1.0, True
    e = effective_level(player, SkillType.STEALING, p)
    return c, chance(e, c, p), e >= c - s.secure_gate


def lock_tier(lockable) -> int:
    return max(0, min(len(LOCK_TIER_NAMES) - 1, int(getattr(lockable, "lock_difficulty", 1))))


def lock_tier_name(lockable) -> str:
    return LOCK_TIER_NAMES[lock_tier(lockable)]


def lock_challenge(lockable, s: StealthSettings) -> float:
    return s.lock_challenge[lock_tier(lockable)]


def lockpick_chance(player: Player, lockable, s: StealthSettings, p: ProgressionSettings = DEFAULT_PROGRESSION) -> float:
    """Odds the pins set once the pick is on the sweet spot: the lock's chance for this player."""
    return chance(effective_level(player, SkillType.LOCKPICKING, p), lock_challenge(lockable, s), p)


def lockpick_tolerance(player: Player, lockable, s: StealthSettings, p: ProgressionSettings = DEFAULT_PROGRESSION) -> float:
    """Degrees either side of the sweet spot that count: wide for a lock far below the player,
    a sliver for one far above."""
    m = margin(effective_level(player, SkillType.LOCKPICKING, p), lock_challenge(lockable, s), p)
    return max(s.lockpick_min_tolerance, min(s.lockpick_max_tolerance, s.lockpick_tolerance_base + s.lockpick_tolerance_per_margin * m))


def lockpick_wobble(player: Player, lockable, s: StealthSettings, p: ProgressionSettings = DEFAULT_PROGRESSION) -> float:
    """Degrees the pick wanders: still for a lock well within the player, wild for one beyond them."""
    m = margin(effective_level(player, SkillType.LOCKPICKING, p), lock_challenge(lockable, s), p)
    return max(0.0, min(s.lockpick_max_wobble, s.lockpick_wobble_base - s.lockpick_wobble_per_margin * m))


def noise_multiplier(player: Player, s: StealthSettings = None, p: ProgressionSettings = DEFAULT_PROGRESSION) -> float:
    """Sneak (skill and gear) makes the player quieter."""
    from .progression import noise_multiplier as curve
    return curve(effective_level(player, SkillType.SNEAK, p), p)


def roll(rng: random.Random, odds: float) -> bool:
    return rng.random() < odds
