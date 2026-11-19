"""Poses and procedural animation clips (pure data/maths, no rendering)."""

from cyberduke.animation.clips import (
    INSPECT_WEAPON_DURATION,
    bang_machine_animation,
    blinded_animation,
    cook_animation,
    electrocuted_animation,
    hawker_animation,
    idle_animation,
    inspect_weapon_animation,
    investigate_animation,
    walk_animation,
)
from cyberduke.animation.pose import HumanoidPose

__all__ = [
    "INSPECT_WEAPON_DURATION",
    "HumanoidPose",
    "bang_machine_animation",
    "blinded_animation",
    "cook_animation",
    "electrocuted_animation",
    "hawker_animation",
    "idle_animation",
    "inspect_weapon_animation",
    "investigate_animation",
    "walk_animation",
]
