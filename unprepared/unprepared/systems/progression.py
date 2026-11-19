"""The one rule behind every skill: how good you are, how hard the task is, what it teaches.

Every skill check in the game - picking a lock, lifting a purse, sneaking past a watcher,
taking something off a shelf unseen, working a legendary piece free of its mount, forging at
the anvil - is decided the same way:

* **Effective level** ``E`` = the skill's base level (1 .. ``max_level`` = 100, earned with
  XP) + the bonuses of everything the player is *wearing* (``Player.skill_bonus``): a
  necklace of +40 Lockpicking turns a level-10 picker into an effective 50.
* **Challenge** ``C`` - how hard this particular attempt is, on the same scale (a villager's
  latch is -25, a legendary lock 47; see ``systems/stealth.py`` for every table).
* **Chance** = ``floor + (ceiling - floor) * sigmoid((E - C) / scale)`` with ``scale`` = 8:
  every 8 levels of margin multiply the odds of success by e. Equal to the challenge is a
  coin toss; 16 levels under it about one in nine; 16 over it about eight in nine.
  ``floor``/``ceiling`` (0.5% / 99.5%) keep a sliver of doubt in everything.

  With a legendary lock (C = 47): E 10 -> 1.5% (all but impossible), E 50 (level 10 + a +40
  necklace) -> 59%, E 100 (level 60 + the necklace) -> 99.4%.

* **XP** for a deed = ``base * magnitude * 2 ** ((C - base_level) / xp_doubling_levels)``:
  every 12 levels the challenge stands above your *base* level doubles what it teaches, every
  12 below halves it (clamped). Practising what is easy for you teaches next to nothing;
  a powerful item that lets a novice tackle hard things makes them learn fast. ``magnitude``
  carries what the deed was worth (``value_xp``: the price of the thing lifted).
* **Levels**: ``SkillTree.xp_needed_for(level)`` grows linearly; levels stop at 100.
"""
from __future__ import annotations

import math

from ..config import ProgressionSettings
from ..entities import Player, SkillType

DEFAULT = ProgressionSettings()


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def chance(effective: float, challenge: float, s: ProgressionSettings = DEFAULT) -> float:
    """Odds that a check at ``effective`` level against ``challenge`` succeeds."""
    return s.chance_floor + (s.chance_ceiling - s.chance_floor) * sigmoid((effective - challenge) / s.logistic_scale)


def margin(effective: float, challenge: float, s: ProgressionSettings = DEFAULT) -> float:
    """How far ahead of the challenge the player is, in units of ``logistic_scale`` (0 = even)."""
    return (effective - challenge) / s.logistic_scale


def effective_level(player: Player, skill: SkillType, s: ProgressionSettings = DEFAULT) -> float:
    """Base level plus what the player's worn gear adds (never below 1)."""
    return max(1.0, float(player.skills.level(skill)) + float(player.skill_bonus(skill)))


def skill_chance(player: Player, skill: SkillType, challenge: float, s: ProgressionSettings = DEFAULT) -> float:
    return chance(effective_level(player, skill, s), challenge, s)


def challenge_factor(base_level: float, challenge: float, s: ProgressionSettings = DEFAULT) -> float:
    """How much a deed of ``challenge`` teaches somebody of ``base_level`` (1 at an even match)."""
    factor = 2.0 ** ((challenge - base_level) / s.xp_doubling_levels)
    return max(s.xp_factor_min, min(s.xp_factor_max, factor))


def xp_reward(base_level: float, base: float, magnitude: float = 1.0, challenge: float = None,
              s: ProgressionSettings = DEFAULT) -> float:
    """XP for a deed: ``base * magnitude`` scaled by how hard it was for this base level.
    ``challenge`` None means "a task at the player's own level" (factor 1)."""
    factor = 1.0 if challenge is None else challenge_factor(base_level, challenge, s)
    return max(0.0, base * magnitude) * factor


def value_xp(value: float, s: ProgressionSettings = DEFAULT) -> float:
    """Reward magnitude for handling an item worth ``value`` gold (capped, a sliver for junk)."""
    return max(s.value_xp_floor, min(s.value_xp_cap, max(0.0, value) * s.value_xp_rate))


def grant(player: Player, skill: SkillType, base: float, magnitude: float = 1.0, notify=None,
          s: ProgressionSettings = DEFAULT, challenge: float = None) -> float:
    """Give the player the reward for a deed of ``challenge`` in ``skill``; returns the XP granted."""
    amount = xp_reward(player.skills.level(skill), base, magnitude, challenge, s)
    if amount > 0:
        player.skills.add_xp(skill, amount, notify_callback=notify)
    return amount


def scaled_yield(level: float, low: int, high: int, max_level: int) -> int:
    """Linear ramp from ``low`` at level 1 to ``high`` at ``max_level`` (and beyond)."""
    if max_level <= 1:
        return high
    t = max(0.0, min(1.0, (max(1.0, level) - 1) / (max_level - 1)))
    return int(low + (high - low) * t + 1e-9)


def noise_multiplier(effective_sneak: float, s: ProgressionSettings = DEFAULT) -> float:
    """How loud the player's steps are for their effective Sneak level: 1 at level 1, falling
    smoothly towards ``noise_floor`` (a master is never quite silent)."""
    return s.noise_floor + (1.0 - s.noise_floor) * math.exp(-(max(1.0, effective_sneak) - 1.0) / s.noise_falloff)
