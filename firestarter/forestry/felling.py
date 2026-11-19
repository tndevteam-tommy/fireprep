"""
Firestarter - Tree Felling & Dynamic Collapse
Implements DESIGN Section 4.1: a directional face notch (to ~1/3 of the diameter) followed by
a back-cut from the opposite side.  When the remaining hinge is thin enough the tree pivots
about its stump as a rigid rod (theta_ddot = 3 g / (2 L) sin(theta)), and on impact the
canopy branches break off into kindling and the trunk is bucked into green logs.
"""

import math
from typing import Dict, Any, List, Tuple

import numpy as np

from firestarter.forestry.tree import Tree, TreeGrowthStage

GRAVITY = 9.81


class TreeFellingSystem:
    @staticmethod
    def _cut_per_swing(axe_type: str, tree: Tree) -> float:
        base = 0.028 if axe_type == "steel" else 0.011
        # dense oak resists the blade, soft pine yields
        density_factor = {"White Oak": 0.7, "Paper Birch": 0.9, "White Pine": 1.15, "Red Cedar": 1.2}.get(tree.species.value, 1.0)
        dead_factor = 1.25 if tree.stage == TreeGrowthStage.STANDING_DEAD else 1.0
        return base * density_factor * dead_factor

    @staticmethod
    def chop_notch(tree: Tree, axe_type: str = "steel", swing_power: float = 1.0) -> Dict[str, Any]:
        """Directional undercut face notch (~1/3 of the trunk diameter)."""
        if tree.is_fallen or tree.is_falling:
            return {"success": False, "message": "Tree has already fallen."}
        target_notch = tree.trunk_radius_m * 0.7
        tree.notch_depth_m = min(target_notch, tree.notch_depth_m + TreeFellingSystem._cut_per_swing(axe_type, tree) * swing_power)
        tree.chop_shake = 1.0
        percent = (tree.notch_depth_m / target_notch) * 100.0
        return {
            "success": True,
            "message": f"Chopping the face notch... {percent:.0f}%",
            "notch_percent": percent,
            "is_complete": tree.notch_depth_m >= target_notch - 1e-9,
        }

    @staticmethod
    def chop_backcut(tree: Tree, axe_type: str = "steel", swing_power: float = 1.0,
                     fall_heading: float = 0.0) -> Dict[str, Any]:
        """Back-cut on the opposite side; the hinge snaps when it is thin enough."""
        if tree.notch_depth_m < (tree.trunk_radius_m * 0.4):
            return {"success": False, "message": "Carve the face notch before the back-cut!"}
        target_backcut = tree.trunk_radius_m * 1.3
        tree.backcut_depth_m = min(target_backcut, tree.backcut_depth_m + TreeFellingSystem._cut_per_swing(axe_type, tree) * swing_power)
        tree.chop_shake = 1.0
        hinge_remaining = (tree.trunk_radius_m * 2.0) - (tree.notch_depth_m + tree.backcut_depth_m)
        if hinge_remaining <= 0.04 and not tree.is_falling:
            tree.is_falling = True
            tree.fall_direction = fall_heading
            tree.fall_angle = 1.5
            return {"success": True, "message": "TIMBER! The hinge snaps and the tree begins to fall!", "tree_falling": True}
        return {"success": True, "message": f"Back-cut... hinge {max(0.0, hinge_remaining) * 100:.1f} cm", "tree_falling": False}

    @staticmethod
    def update_falling_tree(tree: Tree, dt: float, rng: np.random.Generator = None) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        Rigid-rod pivot about the hinge.  Returns (has_hit_ground, spawned_pieces) where each
        piece is a dict describing a green fuel item lying on the ground.
        """
        if not tree.is_falling or tree.is_fallen:
            return False, []
        rng = rng or np.random.default_rng(tree.id)
        L = max(2.0, tree.height_m)
        theta = math.radians(tree.fall_angle)
        alpha = (3.0 * GRAVITY / (2.0 * L)) * math.sin(theta)     # rad/s^2
        tree.fall_velocity += math.degrees(alpha) * dt
        tree.fall_angle += tree.fall_velocity * dt
        if tree.fall_angle < 88.0:
            return False, []

        tree.fall_angle = 88.5        # resting on its crown / branches
        tree.is_falling = False
        tree.is_fallen = True
        tree.stage = TreeGrowthStage.STUMP
        tree.has_fungi = False
        pieces: List[Dict[str, Any]] = []
        d = np.array([math.cos(tree.fall_direction), 0.0, math.sin(tree.fall_direction)])
        side = np.array([-d[2], 0.0, d[0]])
        # Trunk bucked into green logs along the fall line
        n_logs = 3 if tree.trunk_radius_m > 0.12 else 2
        for i in range(n_logs):
            p = tree.position + d * (1.6 + i * 1.3) + side * float(rng.uniform(-0.25, 0.25))
            pieces.append({"kind": "log", "species": tree.species, "pos": p,
                           "yaw": math.degrees(tree.fall_direction) + float(rng.uniform(-20, 20)),
                           "m_surface": tree.living_moisture_surface, "m_core": tree.living_moisture_core})
        # Canopy branches snap into kindling on impact (thinner -> a bit drier)
        for j in range(5):
            p = tree.position + d * (tree.height_m * float(rng.uniform(0.5, 0.9))) + side * float(rng.uniform(-1.2, 1.2))
            pieces.append({"kind": "kindling", "species": tree.species, "pos": p,
                           "yaw": float(rng.uniform(0, 360)),
                           "m_surface": max(0.25, tree.living_moisture_surface * 0.7),
                           "m_core": max(0.30, tree.living_moisture_core * 0.7)})
        return True, pieces
