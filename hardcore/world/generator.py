"""Procedural Level Generator with 5-zone parkour obstacle gauntlet."""
import numpy as np
import config
from world.obstacle import Obstacle
from world.building import BuildingGenerator, BuildingSpec, build_l_block, SOUTH, EAST, NORTH
from world.city import CityGenerator


class LevelGenerator:
    @staticmethod
    def build_zones():
        """The six training zones (no floor, no perimeter). Returns a list of Obstacles."""
        obstacles = []

        # =====================================================================
        # ZONE 1: Variable Jump Runway (Z = -20 to 0, X = -10 to -2)
        # =====================================================================
        hurdles = [
            # Low hop hurdles
            (-8.0, 0.0, -18.0, 4.0, 0.8, 0.5),
            (-8.0, 0.0, -14.0, 4.0, 1.2, 0.5),
            # Full hold high leap
            (-8.0, 0.0, -9.0,  4.0, 1.7, 0.6),
            # Running mantle ledge
            (-8.0, 0.0, -4.0,  4.0, 2.2, 2.0),
        ]
        for x, y, z, w, h, d in hurdles:
            obstacles.append(Obstacle(x, y, z, w, h, d, color=config.COLOR_OBSTACLE))

        # =====================================================================
        # ZONE 2: Crevasse Traverse Wall (Z = 5 to 23, X = -20)
        # =====================================================================
        obstacles.append(Obstacle(-20.0, 0.0, 5.0, 0.8, 3.2, 18.0, color=config.COLOR_OBSTACLE_ACCENT, add_crevasses=True))

        # =====================================================================
        # ZONE 3: Vertical Double-Crevasse Tower (X = 0, Z = 5)
        # =====================================================================
        # Tower featuring 2.0m head crevasse, 3.8m jump crevasse, 5.4m mantle top.
        # Climb: hang 2.0 -> stand on its lip -> reach 3.8 -> stand on its lip -> hands on the 5.4 top -> W to top out.
        obstacles.append(Obstacle(-2.0, 0.0, 5.0, 4.0, 5.4, 4.0, color=(0.32, 0.35, 0.42), add_crevasses=True))
        # Intermediate booster block next to tower
        obstacles.append(Obstacle(-2.0, 0.0, 1.5, 4.0, 1.6, 2.0, color=config.COLOR_OBSTACLE))

        # =====================================================================
        # ZONE 4: The Dyno Chasm (X = 10 to 21.2, Z = 0 to 5)
        # =====================================================================
        # Two opposing towers separated by a 3.2 m gap for the backward wall ejection (cat leap).
        obstacles.append(Obstacle(10.0, 0.0, 0.0, 4.0, 6.0, 5.0, color=(0.38, 0.33, 0.35), add_crevasses=True))
        obstacles.append(Obstacle(17.2, 0.0, 0.0, 4.0, 6.0, 5.0, color=(0.38, 0.33, 0.35), add_crevasses=True))

        # =====================================================================
        # ZONE 5: Pillar Hop Gauntlet (X = 6 to 12.4, Z = 22 to 34.5)
        # =====================================================================
        # Nearest-corner gaps are ~2.8 m: the upper end of a real lateral wall-to-wall cat leap.
        pillars = [
            ( 6.0, 0.0, 22.0, 2.0, 4.8, 2.0),
            (10.4, 0.0, 25.5, 2.0, 4.8, 2.0),
            ( 6.0, 0.0, 29.0, 2.0, 4.8, 2.0),
            (10.4, 0.0, 32.5, 2.0, 4.8, 2.0),
        ]
        for x, y, z, w, h, d in pillars:
            obstacles.append(Obstacle(x, y, z, w, h, d, color=(0.28, 0.38, 0.36), add_crevasses=True))

        # =====================================================================
        # ZONE 6: Buildings (X = -28 to -4, Z = 28 to 48)
        # =====================================================================
        obstacles.extend(BuildingGenerator.build(BuildingSpec(
            -27.0, 29.0, 10.0, 8.0, floors=3, facade_faces=(SOUTH, EAST), door_face=SOUTH,
            balconies=((NORTH, 1, 1.2), (NORTH, 2, 6.0)), rooftop=True, name="building-a")))
        obstacles.extend(build_l_block(-10.0, 38.0))
        return obstacles

    @staticmethod
    def _batch(obstacles):
        all_verts = []
        all_indices = []
        for obs in obstacles:
            idx_offset = len(all_verts)
            all_verts.extend(obs.vertices)
            all_indices.extend(obs.indices + idx_offset)
        return np.array(all_verts, dtype=np.float32), np.array(all_indices, dtype=np.uint32)

    @staticmethod
    def build_playground(ledge_system):
        """
        The stand-alone training playground: floor, 8 m perimeter walls and the six zones.
        Returns:
            obstacles: list of Obstacle instances
            batched_verts: np.ndarray
            batched_indices: np.ndarray
        """
        obstacles = []
        ledge_system.clear()

        # Floor Arena (60m x 80m, y = -0.5 to 0.0)
        floor = Obstacle(-30.0, -0.5, -30.0, 60.0, 0.5, 80.0, color=config.COLOR_FLOOR, add_crevasses=False, rails=False)
        obstacles.append(floor)

        # Arena Perimeter Boundary Walls (8 m, climbable via a crevasse ladder: 2.0 / 3.8 / 5.6 / 7.4 m)
        obstacles.append(Obstacle(-31.0, 0.0, -31.0, 1.0, 8.0, 82.0, color=(0.18, 0.19, 0.22), add_crevasses=True))
        obstacles.append(Obstacle( 30.0, 0.0, -31.0, 1.0, 8.0, 82.0, color=(0.18, 0.19, 0.22), add_crevasses=True))
        obstacles.append(Obstacle(-30.0, 0.0, -31.0, 60.0, 8.0, 1.0, color=(0.18, 0.19, 0.22), add_crevasses=True))
        obstacles.append(Obstacle(-30.0, 0.0,  50.0, 60.0, 8.0, 1.0, color=(0.18, 0.19, 0.22), add_crevasses=True))

        obstacles.extend(LevelGenerator.build_zones())

        # Populate ledge system: clip rails against neighbouring geometry and join corners
        ledge_system.build(obstacles)
        verts, indices = LevelGenerator._batch(obstacles)
        return obstacles, verts, indices

    @staticmethod
    def build_city(ledge_system, seed=7):
        """
        The full map: the training park in the middle of a procedural city
        (streets, blocks of buildings, vehicles, props, set-pieces, 12 m perimeter).
        Returns:
            obstacles, batched_verts, batched_indices, generator (for stats)
        """
        ledge_system.clear()
        gen = CityGenerator(seed)
        obstacles = gen.build(LevelGenerator.build_zones())
        ledge_system.build(obstacles)
        verts, indices = LevelGenerator._batch(obstacles)
        return obstacles, verts, indices, gen
