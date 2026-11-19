"""The shipped level: a three-sector corporate facility.

Generated once from the original hand-written mesh code (every box/cylinder, with its source
comment as ``label``) and then maintained here as data. Two kinds of geometry live side by side:

* ``props``      - what is drawn. Purely visual.
* ``colliders``  - what blocks movement and sight. Hand-tuned gameplay boxes (walls are
                   thicker than they look, cover is slightly generous, railings don't block
                   sight). ``tests/test_level_data.py`` keeps the two consistent.

Coordinates: X/Z ground plane, Y up. Yaw 0 = +Z, 90 = +X.
"""

from __future__ import annotations

import math
from dataclasses import replace

from cyberduke.core.geometry import AABB
from cyberduke.levels.doorway import (
    STAFF_DOOR,
    STAFF_DOOR_HEIGHT,
    STAFF_DOOR_WIDTH,
    staff_door_alarm,
    staff_door_leaf,
    staff_door_trim,
)
from cyberduke.levels.dressing import (
    CAP_MATERIALS,
    Placer,
    light_fungus,
    mushroom_cluster,
    scatter_trash,
)
from cyberduke.world.level import (
    CameraData,
    DoorData,
    FactionData,
    GuardData,
    LevelData,
    SectorData,
    TerminalData,
    VendingMachineData,
    WalkRegion,
)
from cyberduke.world.lights import Light
from cyberduke.world.props import Material, Prop
from cyberduke.world.surfaces import Catwalk

FLAT = Material()  # untextured, lit by vertex colour only


# ========================================================================================
# Sector 1: Staging Bay (Z -10..10)
# ========================================================================================

STAGING_BAY_PROPS = (
    Prop.box(0.0, -0.05, 0.0, 12.0, 0.1, 20.0, Material("floor"), label="FLOOR & DRAINAGE GRATES"),
    Prop.box(-5.6, 0.005, 0.0, 0.6, 0.01, 19.6, Material("grate", (0.3, 0.32, 0.38)), label="Perimeter Drainage Grates"),
    Prop.box(5.6, 0.005, 0.0, 0.6, 0.01, 19.6, Material("grate", (0.3, 0.32, 0.38)), label="Perimeter Drainage Grates"),
    # Soft-edged like every other puddle (it was a hard-edged glossy slab that the lights
    # turned into a bright lavender rectangle on the floor).
    Prop.disc(-1.5, 0.004, -3.8, 1.15, 0.72, 0.45, Material(None, (0.12, 0.2, 0.3), alpha=0.55), slices=28, label="Glossy Chemical / Water Puddle"),
    Prop.disc(-1.25, 0.005, -3.7, 0.75, 0.38, 0.25, Material(None, (0.45, 0.7, 0.95), alpha=0.24), slices=20, label="Glossy Chemical / Water Puddle sheen (neon caught on the wet floor)"),
    Prop.box(4.1, 0.006, -7.85, 0.12, 0.002, 2.9, Material("hazard", (0.8, 0.7, 0.1)), label="Hazard Safety Perimeter Line around Server Bank"),
    Prop.box(0.0, 5.05, 0.0, 12.0, 0.1, 20.0, Material("wall", (0.2, 0.22, 0.26)), label="CEILING & INDUSTRIAL INFRASTRUCTURE"),
    Prop.box(0.0, 4.82, -7.5, 12.0, 0.32, 0.12, Material("metal", (0.15, 0.16, 0.2)), label="Steel I-Beam Girders"),
    Prop.box(0.0, 4.66, -7.5, 12.0, 0.04, 0.34, Material("metal", (0.15, 0.16, 0.2)), label="Steel I-Beam Girders"),
    Prop.box(0.0, 4.82, -2.5, 12.0, 0.32, 0.12, Material("metal", (0.15, 0.16, 0.2)), label="Steel I-Beam Girders"),
    Prop.box(0.0, 4.66, -2.5, 12.0, 0.04, 0.34, Material("metal", (0.15, 0.16, 0.2)), label="Steel I-Beam Girders"),
    Prop.box(0.0, 4.82, 2.5, 12.0, 0.32, 0.12, Material("metal", (0.15, 0.16, 0.2)), label="Steel I-Beam Girders"),
    Prop.box(0.0, 4.66, 2.5, 12.0, 0.04, 0.34, Material("metal", (0.15, 0.16, 0.2)), label="Steel I-Beam Girders"),
    Prop.box(0.0, 4.82, 7.5, 12.0, 0.32, 0.12, Material("metal", (0.15, 0.16, 0.2)), label="Steel I-Beam Girders"),
    Prop.box(0.0, 4.66, 7.5, 12.0, 0.04, 0.34, Material("metal", (0.15, 0.16, 0.2)), label="Steel I-Beam Girders"),
    Prop.box(-2.0, 4.45, 0.0, 1.2, 0.65, 20.0, Material("metal", (0.25, 0.27, 0.32)), label="Giant HVAC Ventilation Duct"),
    Prop.box(-1.39, 4.45, -5.0, 0.04, 0.45, 1.2, Material("grate", (0.1, 0.12, 0.14)), label="Duct Exhaust Louvers"),
    Prop.box(-1.39, 4.45, 0.0, 0.04, 0.45, 1.2, Material("grate", (0.1, 0.12, 0.14)), label="Duct Exhaust Louvers"),
    Prop.box(-1.39, 4.45, 5.0, 0.04, 0.45, 1.2, Material("grate", (0.1, 0.12, 0.14)), label="Duct Exhaust Louvers"),
    Prop.box(-6.0, 2.5, 0.0, 0.1, 5.0, 20.0, Material("wall", (0.25, 0.28, 0.35)), solid=True, label="Left Wall"),
    Prop.box(6.0, 2.5, 0.0, 0.1, 5.0, 20.0, Material("wall", (0.25, 0.28, 0.35)), solid=True, label="Right Wall"),
    Prop.box(0.0, 2.5, -10.0, 12.0, 5.0, 0.1, Material("wall", (0.25, 0.28, 0.35)), solid=True, label="Back Wall"),
    Prop.box(-3.75, 2.5, 10.0, 4.5, 5.0, 0.2, Material("wall", (0.25, 0.28, 0.35)), solid=True, label="Left Wall section (x in [-6.0, -1.5])"),
    Prop.box(3.75, 2.5, 10.0, 4.5, 5.0, 0.2, Material("wall", (0.25, 0.28, 0.35)), solid=True, label="Right Wall section (x in [1.5, 6.0])"),
    Prop.box(0.0, 4.25, 10.0, 3.0, 1.5, 0.2, Material("wall", (0.25, 0.28, 0.35)), label="Top Lintel section (x in [-1.5, 1.5], y in [3.5, 5.0])"),
    Prop.box(-1.5, 1.75, 10.0, 0.25, 3.5, 0.28, Material("metal", (0.18, 0.2, 0.24)), solid=True, label="Left Jamb"),
    Prop.box(1.5, 1.75, 10.0, 0.25, 3.5, 0.28, Material("metal", (0.18, 0.2, 0.24)), solid=True, label="Right Jamb"),
    Prop.box(0.0, 3.55, 10.0, 3.25, 0.2, 0.29, Material("hazard", (0.85, 0.85, 0.85)), label="Overhead Header Beam with Hazard Trim"),
    Prop.box(0.0, 3.3, -9.92, 4.2, 1.4, 0.06, Material("metal", (0.12, 0.13, 0.16)), label="GIANT BACK WALL KATAKANA NEON SIGN (\"サイバー // CYBER\")"),
    Prop.box(0.0, 3.3, -9.88, 4.0, 1.2, 0.02, Material("neon_sign", emission=(1.6, 2.4, 3.6)), label="GIANT BACK WALL KATAKANA NEON SIGN (\"サイバー // CYBER\")"),
    Prop.box(-5.92, 3.8, 0.0, 0.06, 0.12, 18.0, Material("neon_cyan", (0.0, 0.9, 1.0), emission=(0.0, 3.4, 4.0)), label="WALL NEON STRIPS & RECESSED FLOOR RUNWAY LIGHTS"),
    Prop.box(5.92, 3.8, 0.0, 0.06, 0.12, 18.0, Material("neon_magenta", (1.0, 0.0, 0.8), emission=(4.0, 0.0, 3.0)), label="WALL NEON STRIPS & RECESSED FLOOR RUNWAY LIGHTS"),
    Prop.box(-2.2, 0.008, -8.0, 0.1, 0.005, 0.2, Material(None, (0.0, 0.8, 1.0), emission=(0.0, 3.4, 4.0)), label="Recessed Floor Guide Lights"),
    Prop.box(2.2, 0.008, -8.0, 0.1, 0.005, 0.2, Material(None, (1.0, 0.6, 0.0), emission=(4.0, 2.6, 0.0)), label="Recessed Floor Guide Lights"),
    Prop.box(-2.2, 0.008, -4.0, 0.1, 0.005, 0.2, Material(None, (0.0, 0.8, 1.0), emission=(0.0, 3.4, 4.0)), label="Recessed Floor Guide Lights"),
    Prop.box(2.2, 0.008, -4.0, 0.1, 0.005, 0.2, Material(None, (1.0, 0.6, 0.0), emission=(4.0, 2.6, 0.0)), label="Recessed Floor Guide Lights"),
    Prop.box(-2.2, 0.008, 0.0, 0.1, 0.005, 0.2, Material(None, (0.0, 0.8, 1.0), emission=(0.0, 3.4, 4.0)), label="Recessed Floor Guide Lights"),
    Prop.box(2.2, 0.008, 0.0, 0.1, 0.005, 0.2, Material(None, (1.0, 0.6, 0.0), emission=(4.0, 2.6, 0.0)), label="Recessed Floor Guide Lights"),
    Prop.box(-2.2, 0.008, 4.0, 0.1, 0.005, 0.2, Material(None, (0.0, 0.8, 1.0), emission=(0.0, 3.4, 4.0)), label="Recessed Floor Guide Lights"),
    Prop.box(2.2, 0.008, 4.0, 0.1, 0.005, 0.2, Material(None, (1.0, 0.6, 0.0), emission=(4.0, 2.6, 0.0)), label="Recessed Floor Guide Lights"),
    Prop.box(-2.2, 0.008, 8.0, 0.1, 0.005, 0.2, Material(None, (0.0, 0.8, 1.0), emission=(0.0, 3.4, 4.0)), label="Recessed Floor Guide Lights"),
    Prop.box(2.2, 0.008, 8.0, 0.1, 0.005, 0.2, Material(None, (1.0, 0.6, 0.0), emission=(4.0, 2.6, 0.0)), label="Recessed Floor Guide Lights"),
    Prop.box(-2.8, 0.175, 0.5, 1.06, 0.35, 1.06, Material("hazard", (0.85, 0.85, 0.85)), solid=True, label="TACTICAL COVER PILLARS"),
    Prop.box(-2.8, 2.525, 0.5, 0.88, 4.35, 0.88, Material("wall", (0.26, 0.28, 0.34)), solid=True, label="TACTICAL COVER PILLARS"),
    Prop.box(-2.8, 1.2, 0.5, 0.92, 0.12, 0.92, Material("metal", (0.18, 0.2, 0.24)), solid=True, label="TACTICAL COVER PILLARS"),
    Prop.box(-2.8, 2.8, 0.5, 0.92, 0.12, 0.92, Material("metal", (0.18, 0.2, 0.24)), solid=True, label="TACTICAL COVER PILLARS"),
    Prop.box(2.6, 0.175, 1.2, 1.06, 0.35, 1.06, Material("hazard", (0.85, 0.85, 0.85)), solid=True, label="TACTICAL COVER PILLARS"),
    Prop.box(2.6, 2.525, 1.2, 0.88, 4.35, 0.88, Material("wall", (0.26, 0.28, 0.34)), solid=True, label="TACTICAL COVER PILLARS"),
    Prop.box(2.6, 1.2, 1.2, 0.92, 0.12, 0.92, Material("metal", (0.18, 0.2, 0.24)), solid=True, label="TACTICAL COVER PILLARS"),
    Prop.box(2.6, 2.8, 1.2, 0.92, 0.12, 0.92, Material("metal", (0.18, 0.2, 0.24)), solid=True, label="TACTICAL COVER PILLARS"),
    Prop.box(2.6, 1.56, 0.7, 0.1, 0.08, 0.02, Material(None, (1.0, 0.1, 0.1), emission=(4.0, 0.4, 0.4)), solid=True, label="TACTICAL COVER PILLARS"),
    Prop.box(4.85, 0.06, -7.85, 1.15, 0.12, 2.65, Material("metal", (0.14, 0.15, 0.18)), solid=True, label="SERVER MAINFRAME BANKS"),
    Prop.box(4.85, 1.68, -8.67, 1.05, 3.12, 0.78, Material("metal", (0.18, 0.2, 0.24)), solid=True, label="SERVER MAINFRAME BANKS"),
    Prop.box(4.32, 1.68, -8.67, 0.02, 2.95, 0.72, Material("server_rack"), solid=True, label="SERVER MAINFRAME BANKS"),
    Prop.box(4.85, 1.68, -7.85, 1.05, 3.12, 0.78, Material("metal", (0.18, 0.2, 0.24)), solid=True, label="SERVER MAINFRAME BANKS"),
    Prop.box(4.32, 1.68, -7.85, 0.02, 2.95, 0.72, Material("server_rack"), solid=True, label="SERVER MAINFRAME BANKS"),
    Prop.box(4.85, 1.68, -7.03, 1.05, 3.12, 0.78, Material("metal", (0.18, 0.2, 0.24)), solid=True, label="SERVER MAINFRAME BANKS"),
    Prop.box(4.32, 1.68, -7.03, 0.02, 2.95, 0.72, Material("server_rack"), solid=True, label="SERVER MAINFRAME BANKS"),
    Prop.box(4.33, 3.28, -7.85, 0.03, 0.05, 2.4, Material(None, (0.0, 0.8, 1.0), emission=(0.0, 3.2, 4.0)), solid=True, label="SERVER MAINFRAME BANKS"),
    Prop.box(-4.6, 1.55, -8.8, 0.95, 3.1, 0.95, Material("metal", (0.18, 0.2, 0.24)), solid=True, label="SERVER MAINFRAME BANKS"),
    Prop.box(-4.6, 1.55, -8.32, 0.85, 2.9, 0.02, Material("server_rack"), solid=True, label="SERVER MAINFRAME BANKS"),
    Prop.box(-4.5, 0.425, 3.0, 1.3, 0.85, 1.1, Material("crate", (0.85, 0.85, 0.85)), solid=True, label="MIL-SPEC LOOT CRATES"),
    Prop.box(-4.45, 1.2, 2.95, 0.95, 0.7, 0.9, Material("crate", (0.85, 0.85, 0.85)), rotate=((9.0, 0.0, 1.0, 0.0),), solid=True, label="MIL-SPEC LOOT CRATES"),
    Prop.box(-4.42, 1.64, 3.13, 0.08, 0.04, 0.02, Material(None, (0.0, 0.9, 1.0), emission=(0.0, 3.6, 4.0)), solid=True, label="MIL-SPEC LOOT CRATES"),
    Prop.box(4.5, 0.38, 5.2, 1.1, 0.76, 1.05, Material("crate", (0.85, 0.85, 0.85)), solid=True, label="MIL-SPEC LOOT CRATES"),
    Prop.box(4.45, 0.86, 5.2, 0.55, 0.2, 0.32, Material("metal", (0.75, 0.2, 0.1)), rotate=((-18.0, 0.0, 1.0, 0.0),), solid=True, label="MIL-SPEC LOOT CRATES"),
    Prop.cylinder(-5.0, 0.0, -2.0, 0.38, 0.38, 0.95, Material("hazard", (0.24, 0.25, 0.28)), solid=True, label="Biohazard Drum"),
    Prop.box(1.9, 0.06, -3.2, 0.18, 0.12, 0.18, Material("cardboard"), rotate=((24.0, 0.0, 1.0, 0.0),), label="Crumpled Takeout Food Boxes"),
    Prop.cylinder(2.3, 0.04, -3.4, 0.035, 0.035, 0.13, Material("soda_can"), slices=8, rotate=((68.0, 0.0, 1.0, 0.0), (90.0, 1.0, 0.0, 0.0),), label="Crushed Energy Drink Cans"),
    Prop.box(3.6, 0.015, -7.2, 0.12, 0.015, 0.08, Material(None, (1.0, 0.7, 0.0), emission=(4.0, 2.4, 0.0)), rotate=((18.0, 0.0, 1.0, 0.0),), label="Discarded Datachip"),
    Prop.box(-5.92, 1.8, -4.0, 0.08, 0.85, 0.65, Material("metal", (0.2, 0.22, 0.26)), label="Wall Breaker Box"),
    Prop.box(-5.86, 2.0, -4.0, 0.02, 0.06, 0.16, Material(None, (0.0, 0.9, 0.2), emission=(0.0, 3.6, 0.8)), label="Wall Breaker Box"),
)

STAGING_BAY_COLLIDERS = (
    AABB(-3.3, -2.3, 0.0, 1.0, 0.0, 4.7),  # cover pillars (4.7 m tall)
    AABB(2.1, 3.1, 0.7, 1.7, 0.0, 4.7),
    AABB(4.2, 5.5, -9.2, -6.5, 0.0, 3.3),
    AABB(-5.2, -4.0, -9.4, -8.2, 0.0, 3.1),
    AABB(-5.2, -3.8, 2.3, 3.7, 0.0, 1.6),
    AABB(3.9, 5.1, 4.6, 5.8, 0.0, 1.0),
    AABB(-6.5, -5.95, -10.2, 10.2, 0.0, 5.0),
    AABB(5.95, 6.5, -10.2, 10.2, 0.0, 5.0),
    AABB(-6.2, 6.2, -10.5, -9.95, 0.0, 5.0),
    AABB(-6.2, -1.5, 9.9, 10.2, 0.0, 5.0),
    AABB(1.5, 6.2, 9.9, 10.2, 0.0, 5.0),
    # Lintel above the blast door (opaque even when the door is open)
    AABB(-1.5, 1.5, 9.9, 10.2, 3.5, 5.0),
    # Biohazard drum
    AABB(-5.38, -4.62, -2.38, -1.62, 0.0, 0.95),
)

STAGING_CEILING_Y = 5.0  # underside of the staging bay ceiling: the cage lights' cables reach it

STAGING_BAY_LIGHTS = (
    Light(
        3.5, 3.6, -4.0,
        ambient=(0.0, 0.25, 0.35), diffuse=(0.0, 3.4, 4.0), specular=(0.0, 4.0, 4.0),
        fixture="hanging_cage", fixture_emission=(0.0, 3.4, 4.0),
        fixture_color=(0.0, 0.8, 1.0), fixture_texture="neon_cyan", ceiling_y=STAGING_CEILING_Y,
    ),
    Light(
        -4.5, 3.6, -7.5,
        ambient=(0.3, 0.0, 0.22), diffuse=(4.0, 0.0, 3.0), specular=(4.0, 0.8, 3.2),
        fixture="hanging_cage", fixture_emission=(4.0, 0.0, 3.0),
        fixture_color=(1.0, 0.0, 0.7), fixture_texture="neon_magenta", ceiling_y=STAGING_CEILING_Y,
    ),
    Light(
        0.0, 3.8, 1.0,
        ambient=(0.25, 0.16, 0.04), diffuse=(3.0, 2.0, 0.4), specular=(2.0, 1.4, 0.2),
        linear_attenuation=0.025, quadratic_attenuation=0.006,
        fixture="hanging_cage", fixture_emission=(4.0, 2.6, 0.0),
        fixture_color=(1.0, 0.7, 0.0), fixture_texture="hazard", ceiling_y=STAGING_CEILING_Y,
    ),
)

STAGING_BAY_WALK = (
    WalkRegion(-5.6, 5.6, -9.6, 9.88),  # lets the player walk up to the shut door face (z=9.88)
)

STAGING_BAY = SectorData(
    name="StagingBay",
    bounds=(-6.0, 6.0, -10.0, 10.0),
    props=STAGING_BAY_PROPS,
    colliders=STAGING_BAY_COLLIDERS,
    lights=STAGING_BAY_LIGHTS,
    walk_regions=STAGING_BAY_WALK,
)


# ========================================================================================
# Airlock corridor (Z 10..18)
# ========================================================================================

AIRLOCK_CORRIDOR_PROPS = (
    Prop.box(0.0, -0.05, 14.0, 5.0, 0.1, 8.0, Material("floor"), label="Floor & Grates (z=10 to 18, width x=-2.5 to 2.5)"),
    Prop.box(0.0, 0.005, 14.0, 1.2, 0.01, 8.0, Material("grate", (0.35, 0.36, 0.4)), label="Center Grate Runway"),
    Prop.box(-2.35, 0.008, 14.0, 0.18, 0.005, 8.0, Material("hazard", (0.9, 0.8, 0.1)), solid=True, label="Hazard Safety Trim Lines along walls"),
    Prop.box(2.35, 0.008, 14.0, 0.18, 0.005, 8.0, Material("hazard", (0.9, 0.8, 0.1)), solid=True, label="Hazard Safety Trim Lines along walls"),
    Prop.box(0.0, 4.25, 14.0, 5.0, 0.1, 8.0, Material("wall", (0.18, 0.2, 0.24)), label="Ceiling (Height y=4.2)"),
    Prop.box(0.0, 4.12, 10.2, 4.99, 0.25, 0.25, Material("metal", (0.14, 0.15, 0.18)), label="Top beam"),
    Prop.box(-2.4, 2.1, 10.2, 0.2, 4.2, 0.24, Material("metal", (0.14, 0.15, 0.18)), label="Left pillar"),
    Prop.box(2.4, 2.1, 10.2, 0.2, 4.2, 0.24, Material("metal", (0.14, 0.15, 0.18)), label="Right pillar"),
    Prop.box(0.0, 4.12, 12.5, 4.99, 0.25, 0.25, Material("metal", (0.14, 0.15, 0.18)), label="Top beam"),
    Prop.box(-2.4, 2.1, 12.5, 0.2, 4.2, 0.24, Material("metal", (0.14, 0.15, 0.18)), solid=True, label="Left pillar"),
    Prop.box(2.4, 2.1, 12.5, 0.2, 4.2, 0.24, Material("metal", (0.14, 0.15, 0.18)), solid=True, label="Right pillar"),
    Prop.box(0.0, 4.12, 15.0, 4.99, 0.25, 0.25, Material("metal", (0.14, 0.15, 0.18)), label="Top beam"),
    Prop.box(-2.4, 2.1, 15.0, 0.2, 4.2, 0.24, Material("metal", (0.14, 0.15, 0.18)), solid=True, label="Left pillar"),
    Prop.box(2.4, 2.1, 15.0, 0.2, 4.2, 0.24, Material("metal", (0.14, 0.15, 0.18)), solid=True, label="Right pillar"),
    Prop.box(0.0, 4.12, 17.5, 4.99, 0.25, 0.25, Material("metal", (0.14, 0.15, 0.18)), label="Top beam"),
    Prop.box(-2.4, 2.1, 17.5, 0.2, 4.2, 0.24, Material("metal", (0.14, 0.15, 0.18)), label="Left pillar"),
    Prop.box(2.4, 2.1, 17.5, 0.2, 4.2, 0.24, Material("metal", (0.14, 0.15, 0.18)), label="Right pillar"),
    Prop.box(-2.5, 2.1, 14.0, 0.1, 4.2, 8.0, Material("wall", (0.22, 0.24, 0.28)), solid=True, label="Left Wall (x=-2.5)"),
    Prop.box(2.5, 2.1, 14.0, 0.1, 4.2, 8.0, Material("wall", (0.22, 0.24, 0.28)), solid=True, label="Right Wall (x=2.5)"),
    Prop.box(-2.35, 1.2, 14.0, 0.12, 0.12, 7.8, Material("metal", (0.35, 0.38, 0.45)), solid=True, label="Left wall pipe cluster"),
    Prop.box(-2.35, 1.8, 14.0, 0.12, 0.12, 7.8, Material("metal", (0.35, 0.38, 0.45)), solid=True, label="Left wall pipe cluster"),
    Prop.box(-2.35, 2.4, 14.0, 0.12, 0.12, 7.8, Material("metal", (0.35, 0.38, 0.45)), solid=True, label="Left wall pipe cluster"),
    Prop.box(2.35, 2.0, 14.0, 0.14, 0.35, 7.8, Material("metal", (0.35, 0.38, 0.45)), solid=True, label="Right wall cable conduit"),
    Prop.box(-2.28, 1.8, 13.5, 0.04, 0.16, 0.16, Material(None, (0.0, 0.9, 1.0), emission=(0.0, 3.4, 4.0)), solid=True, label="Glowing Pressure Gauge Dials"),
    Prop.box(-2.28, 1.8, 15.0, 0.04, 0.16, 0.16, Material(None, (1.0, 0.6, 0.0), emission=(4.0, 2.0, 0.0)), solid=True, label="Glowing Pressure Gauge Dials"),
)

AIRLOCK_CORRIDOR_COLLIDERS = (
    AABB(-2.5, -2.0, 12.0, 16.0, 0.0, 2.5),
    AABB(2.0, 2.5, 12.0, 16.0, 0.0, 2.5),
    AABB(-3.5, -2.48, 9.8, 18.2, 0.0, 4.2),
    AABB(2.48, 3.5, 9.8, 18.2, 0.0, 4.2),
)

AIRLOCK_CORRIDOR_LIGHTS = (
    Light(
        0.0, 3.6, 14.0,
        ambient=(0.25, 0.18, 0.02), diffuse=(3.2, 2.2, 0.3), specular=(2.5, 1.8, 0.2),
        linear_attenuation=0.025, quadratic_attenuation=0.006,
        fixture="caution_lamp", fixture_emission=(4.0, 2.6, 0.0),
        fixture_color=(1.0, 0.7, 0.0), fixture_texture="hazard", fixture_offset_y=0.05,
    ),
)

AIRLOCK_CORRIDOR_WALK = (
    WalkRegion(-2.5, 2.5, 10.12, 18.0, inset=frozenset({"min_x", "max_x", "min_z"})),  # door shut: keep clear of its far face
    WalkRegion(-2.5, 2.5, 10.0, 18.0, inset=frozenset({"min_x", "max_x"}), requires_door="blast_door"),
)

AIRLOCK_CORRIDOR = SectorData(
    name="AirlockCorridor",
    bounds=(-2.5, 2.5, 10.0, 18.0),
    props=AIRLOCK_CORRIDOR_PROPS,
    colliders=AIRLOCK_CORRIDOR_COLLIDERS,
    lights=AIRLOCK_CORRIDOR_LIGHTS,
    walk_regions=AIRLOCK_CORRIDOR_WALK,
)


# ========================================================================================
# Sector 2: Datacore Vault (Z 18..42)
# ========================================================================================

# ------------------------------------------------------------------------------------------
# Catwalk staircases (both sides). Geometry is shared with world.surfaces.Catwalk, which
# turns the same numbers into floor elevation; keep them in sync via CATWALK below.
# ------------------------------------------------------------------------------------------
CATWALK = Catwalk()
STAIRS_X0, STAIRS_X1 = CATWALK.inner_x, 9.5  # lane width matches the deck (x 6.0..9.5)
STAIRS_Z0, STAIRS_Z1 = CATWALK.stairs_z_start, CATWALK.stairs_z_end
RAIL_X = 6.02  # inner handrail centre line
RAIL_HEIGHT = 0.9  # handrail top above the walking surface, matching the deck rails

STEP_MATERIAL = Material("metal", (0.18, 0.2, 0.24))
NOSING_MATERIAL = Material("hazard", (0.85, 0.85, 0.85))
RAIL_MATERIAL = Material("hazard", (0.8, 0.7, 0.1))


def _staircase_props(side: int) -> tuple[Prop, ...]:
    """Closed-riser steps plus an inner handrail: sloped top rail, vertical end posts
    (square to the floor) and a baluster per step. ``side`` is -1 (left) or +1 (right)."""
    cx = side * (STAIRS_X0 + STAIRS_X1) / 2.0
    width = STAIRS_X1 - STAIRS_X0
    run, rise = CATWALK.step_run, CATWALK.step_rise
    rail_x = side * RAIL_X
    props: list[Prop] = []
    for i in range(CATWALK.steps):
        z0 = STAIRS_Z0 + i * run
        top = (i + 1) * rise
        props.append(Prop.box(cx, top / 2.0, z0 + run / 2.0, width, top, run, STEP_MATERIAL,
                              label=f"Catwalk stairs step {i + 1}", tags=frozenset({"surface"})))
        props.append(Prop.box(cx, top + 0.005, z0 + 0.04, width, 0.01, 0.08, NOSING_MATERIAL,
                              label="Stair nosing hazard strip"))
        # Baluster from the step surface up to the sloped handrail
        zb = z0 + run / 2.0
        rail_y = RAIL_HEIGHT + CATWALK.height * (zb - STAIRS_Z0) / CATWALK.stairs_length
        props.append(Prop.box(rail_x, (top + rail_y) / 2.0, zb, 0.03, rail_y - top, 0.03, RAIL_MATERIAL,
                              label="Stair baluster"))
    # Sloped handrail from (z0, RAIL_HEIGHT) to (z1, height + RAIL_HEIGHT)
    length = math.hypot(CATWALK.stairs_length, CATWALK.height)
    angle = math.degrees(math.atan2(CATWALK.height, CATWALK.stairs_length))
    mid_y = RAIL_HEIGHT + CATWALK.height / 2.0
    props.append(Prop.box(rail_x, mid_y, (STAIRS_Z0 + STAIRS_Z1) / 2.0, 0.05, 0.08, length, RAIL_MATERIAL,
                          rotate=((-angle, 1.0, 0.0, 0.0),), label="Stair handrail (sloped)"))
    # Vertical end posts: square to the floor at the bottom, joining the deck rail at the top
    props.append(Prop.box(rail_x, RAIL_HEIGHT / 2.0, STAIRS_Z0 + 0.035, 0.06, RAIL_HEIGHT, 0.06, RAIL_MATERIAL,
                          label="Stair handrail bottom post"))
    top_post_y0 = CATWALK.height
    props.append(Prop.box(rail_x, (top_post_y0 + CATWALK.height + RAIL_HEIGHT + 0.01) / 2.0, STAIRS_Z1, 0.06,
                          RAIL_HEIGHT + 0.01, 0.06, RAIL_MATERIAL, label="Stair handrail top post (joins deck rail)"))
    return tuple(props)


DATACORE_VAULT_PROPS = (
    Prop.box(0.0, -0.05, 30.0, 20.0, 0.1, 24.0, Material("floor"), label="MAIN VAULT FLOOR SLAB (Width x in [-10, 10], z in [18, 42])"),
    Prop.box(-3.5, -0.02, 30.0, 1.2, 0.01, 16.0, Material("neon_cyan", (0.0, 0.9, 1.0), emission=(0.0, 3.8, 4.0)), label="Glowing coolant fluid base in trench"),
    Prop.box(-3.5, 0.005, 30.0, 1.25, 0.01, 16.2, Material("grate", (0.28, 0.3, 0.34)), label="Heavy Drainage Grating covering the trench"),
    Prop.box(-4.2, 0.006, 30.0, 0.12, 0.005, 16.2, Material("hazard", (0.85, 0.85, 0.85)), label="Hazard warning borders along trenches"),
    Prop.box(-2.8, 0.006, 30.0, 0.12, 0.005, 16.2, Material("hazard", (0.85, 0.85, 0.85)), label="Hazard warning borders along trenches"),
    Prop.box(3.5, -0.02, 30.0, 1.2, 0.01, 16.0, Material("neon_cyan", (0.0, 0.9, 1.0), emission=(0.0, 3.8, 4.0)), label="Glowing coolant fluid base in trench"),
    Prop.box(3.5, 0.005, 30.0, 1.25, 0.01, 16.2, Material("grate", (0.28, 0.3, 0.34)), label="Heavy Drainage Grating covering the trench"),
    Prop.box(2.8, 0.006, 30.0, 0.12, 0.005, 16.2, Material("hazard", (0.85, 0.85, 0.85)), label="Hazard warning borders along trenches"),
    Prop.box(4.2, 0.006, 30.0, 0.12, 0.005, 16.2, Material("hazard", (0.85, 0.85, 0.85)), label="Hazard warning borders along trenches"),
    # (the walkways stop where the skybridge starts: running on under it, their decks and
    # soffits shared its planes and z-fought)
    Prop.box(-7.75, 2.7, 31.85, 3.5, 0.1, 15.3, Material("metal", (0.18, 0.2, 0.24)), label="Left Catwalk Walkway (x in [-9.5, -6.0], z in [24.2, 39.5])", tags=frozenset({"surface"})),
    Prop.box(7.75, 2.7, 31.85, 3.5, 0.1, 15.3, Material("metal", (0.18, 0.2, 0.24)), label="Right Catwalk Walkway (x in [6.0, 9.5], z in [24.2, 39.5])", tags=frozenset({"surface"})),
    Prop.box(0.0, 2.7, 40.75, 19.0, 0.1, 2.5, Material("metal", (0.18, 0.2, 0.24)), label="Rear Transverse Skybridge (z in [39.5, 42.0], width x in [-9.5, 9.5])", tags=frozenset({"surface"})),
    Prop.box(-6.0, 1.325, 24.2, 0.35, 2.65, 0.35, Material("metal", (0.14, 0.15, 0.18)), solid=True, label="Catwalk Structural Support Columns (y=0.0 to 2.65 underside)"),
    Prop.box(-6.0, 1.325, 32.5, 0.35, 2.65, 0.35, Material("metal", (0.14, 0.15, 0.18)), solid=True, label="Catwalk Structural Support Columns (y=0.0 to 2.65 underside)"),
    Prop.box(-6.0, 1.325, 40.5, 0.35, 2.65, 0.35, Material("metal", (0.14, 0.15, 0.18)), solid=True, label="Catwalk Structural Support Columns (y=0.0 to 2.65 underside)"),
    Prop.box(6.0, 1.325, 24.2, 0.35, 2.65, 0.35, Material("metal", (0.14, 0.15, 0.18)), solid=True, label="Catwalk Structural Support Columns (y=0.0 to 2.65 underside)"),
    Prop.box(6.0, 1.325, 32.5, 0.35, 2.65, 0.35, Material("metal", (0.14, 0.15, 0.18)), solid=True, label="Catwalk Structural Support Columns (y=0.0 to 2.65 underside)"),
    Prop.box(6.0, 1.325, 40.5, 0.35, 2.65, 0.35, Material("metal", (0.14, 0.15, 0.18)), solid=True, label="Catwalk Structural Support Columns (y=0.0 to 2.65 underside)"),
    Prop.box(-6.02, 3.2, 31.85, 0.04, 0.9, 15.3, Material("hazard", (0.8, 0.7, 0.1)), solid=True, label="Left catwalk handrail (z in [24.2, 39.5])"),
    Prop.box(6.02, 3.2, 31.85, 0.04, 0.9, 15.3, Material("hazard", (0.8, 0.7, 0.1)), solid=True, label="Right catwalk handrail (z in [24.2, 39.5])"),
    Prop.box(0.0, 3.2, 39.48, 12.0, 0.9, 0.04, Material("hazard", (0.8, 0.7, 0.1)), solid=True, label="Rear Skybridge Front Handrail (spanning x in [-6.0, 6.0] at z=39.48)"),
    Prop.box(0.0, 6.05, 30.0, 20.0, 0.1, 24.0, Material("wall", (0.2, 0.22, 0.26)), label="HIGH CEILING & INDUSTRIAL STRUCTURAL TRUSSES (Height y=6.0)"),
    Prop.box(0.0, 5.815, 24.0, 19.98, 0.37, 0.2, Material("metal", (0.14, 0.15, 0.18)), label="Massive Cross-Room Steel Trusses"),
    Prop.box(0.0, 5.815, 30.0, 19.98, 0.37, 0.2, Material("metal", (0.14, 0.15, 0.18)), label="Massive Cross-Room Steel Trusses"),
    Prop.box(0.0, 5.815, 36.0, 19.98, 0.37, 0.2, Material("metal", (0.14, 0.15, 0.18)), label="Massive Cross-Room Steel Trusses"),
    Prop.box(-10.0, 3.0, 30.0, 0.1, 6.0, 24.0, Material("wall", (0.24, 0.26, 0.32)), solid=True, label="Left Wall (x=-10.0)"),
    Prop.box(10.0, 3.0, 30.0, 0.1, 6.0, 24.0, Material("wall", (0.24, 0.26, 0.32)), solid=True, label="Right Wall (x=10.0)"),
    Prop.box(-5.8, 3.0, 42.0, 8.4, 6.0, 0.1, Material("wall", (0.24, 0.26, 0.32)), solid=True, label="Far Back Vault Wall (z=42.0, with the greenhouse door opening at x in [-1.6, 1.6] under the skybridge)"),
    Prop.box(5.8, 3.0, 42.0, 8.4, 6.0, 0.1, Material("wall", (0.24, 0.26, 0.32)), solid=True, label="Far Back Vault Wall (z=42.0, with the greenhouse door opening at x in [-1.6, 1.6] under the skybridge)"),
    Prop.box(0.0, 4.25, 42.0, 3.2, 3.5, 0.1, Material("wall", (0.24, 0.26, 0.32)), label="Far Back Vault Wall lintel above the greenhouse door (y 2.5..6.0)"),
    Prop.box(-1.745, 1.25, 42.0, 0.3, 2.5, 0.36, Material("metal", (0.14, 0.15, 0.18)), solid=True, label="Greenhouse door frame post"),
    Prop.box(1.745, 1.25, 42.0, 0.3, 2.5, 0.36, Material("metal", (0.14, 0.15, 0.18)), solid=True, label="Greenhouse door frame post"),
    Prop.box(0.0, 2.55, 42.0, 3.9, 0.12, 0.37, Material("metal", (0.14, 0.15, 0.18)), label="Greenhouse door frame head"),
    Prop.box(0.0, 2.5825, 41.8, 1.6, 0.125, 0.04, Material(None, (0.04, 0.3, 0.05), emission=(0.12, 1.6, 0.3)), label="BIO-LEVEL 3 sign over the greenhouse door"),  # above the door frame head (2.49) - no sliver of it shows through the doorway - and under the skybridge deck (2.65)
    Prop.box(-5.8, 3.0, 18.0, 8.4, 6.0, 0.1, Material("wall", (0.24, 0.26, 0.32)), solid=True, label="Entrance Partition Wall from Airlock (z=18.0, with door opening at x in [-1.6, 1.6])"),
    Prop.box(5.8, 3.0, 18.0, 8.4, 6.0, 0.1, Material("wall", (0.24, 0.26, 0.32)), solid=True, label="Entrance Partition Wall from Airlock (z=18.0, with door opening at x in [-1.6, 1.6])"),
    Prop.box(0.0, 5.1, 18.0, 3.2, 1.8, 0.1, Material("wall", (0.24, 0.26, 0.32)), label="Entrance Partition Wall from Airlock (z=18.0, with door opening at x in [-1.6, 1.6])"),
    Prop.box(0.0, 2.75, 32.0, 2.6, 5.5, 2.6, Material("metal", (0.16, 0.18, 0.22)), solid=True, label="Main Tower Hexagonal / Rectangular Blade Cabinet"),
    Prop.box(-1.31, 2.745, 32.0, 2.2, 4.99, 0.02, Material("server_rack"), rotate=((-90.0, 0.0, 1.0, 0.0),), solid=True, label="Server Blade Display Faces on Monolith Sides"),
    Prop.box(1.31, 2.745, 32.0, 2.2, 4.99, 0.02, Material("server_rack"), rotate=((90.0, 0.0, 1.0, 0.0),), solid=True, label="Server Blade Display Faces on Monolith Sides"),
    Prop.box(0.0, 2.745, 33.31, 2.2, 4.99, 0.02, Material("server_rack"), rotate=((0.0, 0.0, 1.0, 0.0),), label="Server Blade Display Faces on Monolith Sides"),
    Prop.box(0.0, 1.5, 32.0, 2.7, 0.12, 2.7, Material("neon_cyan", (0.0, 0.9, 1.0), emission=(0.0, 3.8, 4.0)), solid=True, label="Glowing Cyan Core Ring Conduits on Monolith"),
    Prop.box(0.0, 3.0, 32.0, 2.7, 0.12, 2.7, Material("neon_cyan", (0.0, 0.9, 1.0), emission=(0.0, 3.8, 4.0)), solid=True, label="Glowing Cyan Core Ring Conduits on Monolith"),
    Prop.box(0.0, 4.5, 32.0, 2.7, 0.12, 2.7, Material("neon_cyan", (0.0, 0.9, 1.0), emission=(0.0, 3.8, 4.0)), solid=True, label="Glowing Cyan Core Ring Conduits on Monolith"),
    Prop.box(4.25, 1.5, 24.65, 1.4, 3.0, 2.2, Material("metal", (0.18, 0.2, 0.24)), solid=True, label="Server Cluster 1 (Right Fore)"),
    Prop.box(3.54, 1.5, 24.65, 0.02, 2.8, 2.0, Material("server_rack"), solid=True, label="Server Cluster 1 (Right Fore)"),
    Prop.box(-4.25, 1.5, 36.65, 1.4, 3.0, 2.2, Material("metal", (0.18, 0.2, 0.24)), solid=True, label="Server Cluster 2 (Left Rear)"),
    Prop.box(-3.54, 1.5, 36.65, 0.02, 2.8, 2.0, Material("server_rack"), solid=True, label="Server Cluster 2 (Left Rear)"),
    Prop.box(-4.5, 0.55, 28.5, 1.4, 1.1, 1.8, Material("crate", (0.85, 0.85, 0.85)), solid=True, label="Cover Crate Stack"),
    Prop.cylinder(4.5, 0.0, 33.6, 0.35, 0.35, 2.4, Material("hazard", (0.35, 0.38, 0.44)), solid=True, label="Cryogenic Coolant Tanks"),
    Prop.cylinder(4.5, 0.0, 34.4, 0.35, 0.35, 2.4, Material("hazard", (0.35, 0.38, 0.44)), solid=True, label="Cryogenic Coolant Tanks"),
)

DATACORE_VAULT_COLLIDERS = (
    AABB(-1.4, 1.4, 30.7, 33.35, 0.0, 5.5),  # monolith (5.5 m tall) body is 2.6 m deep (z 30.7..33.3) plus its display face
    # Catwalk support columns (0.35 m visual columns; tight colliders so they never
    # intrude into the stair lane beside them)
    AABB(-6.18, -5.82, 24.0, 24.4, 0.0, 2.65),
    AABB(-6.18, -5.82, 32.3, 32.7, 0.0, 2.65),
    AABB(-6.18, -5.82, 40.3, 40.7, 0.0, 2.65),
    AABB(5.82, 6.18, 24.0, 24.4, 0.0, 2.65),
    AABB(5.82, 6.18, 32.3, 32.7, 0.0, 2.65),
    AABB(5.82, 6.18, 40.3, 40.7, 0.0, 2.65),
    AABB(3.5, 5.0, 23.5, 25.8, 0.0, 2.5),
    AABB(-5.0, -3.5, 35.5, 37.8, 0.0, 2.5),
    AABB(-5.2, -3.8, 27.5, 29.5, 0.0, 1.1),  # the 1.1 m crate stack: a crouched player hides, a standing one does not
    AABB(3.8, 5.2, 33.2, 34.8, 0.0, 2.4),
    AABB(-10.2, -1.6, 17.95, 18.3, 0.0, 6.0),
    AABB(1.6, 10.2, 17.95, 18.3, 0.0, 6.0),
    # Lintel above the vault doorway
    AABB(-1.6, 1.6, 17.95, 18.3, 4.2, 6.0),
    AABB(-10.5, -9.95, 17.8, 42.2, 0.0, 6.0),
    AABB(9.95, 10.5, 17.8, 42.2, 0.0, 6.0),
    AABB(-10.2, -1.6, 41.95, 42.5, 0.0, 6.0),
    AABB(1.6, 10.2, 41.95, 42.5, 0.0, 6.0),
    # Lintel above the greenhouse door (the skybridge deck sits on it at 2.65)
    AABB(-1.6, 1.6, 41.95, 42.5, 2.5, 6.0),
    # Catwalk rail panels: solid striped plates, so they stop the player falling AND hide a
    # crouching player from the floor below.
    AABB(-6.18, -5.82, 24.2, 39.5, 2.65, 3.7),
    AABB(5.82, 6.18, 24.2, 39.5, 2.65, 3.7),
    AABB(-6.0, 6.0, 39.3, 39.7, 2.65, 3.7),
    # Stair handrails (inner side; the outer side is the vault wall). Balusters with gaps:
    # they stop movement but are transparent to sight by design.
    AABB(-6.18, -5.82, STAIRS_Z0, STAIRS_Z1, 0.0, 3.7, blocks_sight=False),
    AABB(5.82, 6.18, STAIRS_Z0, STAIRS_Z1, 0.0, 3.7, blocks_sight=False),
    # Stair back faces: stop ground-level agents under the deck from walking into the
    # stair footprint from behind (which would lift them straight onto the catwalk).
    # Agents on the upper steps / deck have their feet above 2.0 m and pass freely.
    AABB(-STAIRS_X1, -STAIRS_X0, STAIRS_Z1 - 0.2, STAIRS_Z1, 0.0, 2.0),
    AABB(STAIRS_X0, STAIRS_X1, STAIRS_Z1 - 0.2, STAIRS_Z1, 0.0, 2.0),
)

# NPCs cannot climb (their navigation is ground-level only), so the stair footprints are
# no-go boxes for them; see Room.get_nav_obstacles.
DATACORE_VAULT_NAV_BLOCKERS = (
    AABB(-STAIRS_X1, -STAIRS_X0, STAIRS_Z0, STAIRS_Z1, 0.0, 3.0),
    AABB(STAIRS_X0, STAIRS_X1, STAIRS_Z0, STAIRS_Z1, 0.0, 3.0),
)


def _stair_occluders(side: int) -> tuple[AABB, ...]:
    """One opaque box per solid step block (matches the drawn geometry)."""
    x0, x1 = sorted((side * STAIRS_X0, side * STAIRS_X1))
    run, rise = CATWALK.step_run, CATWALK.step_rise
    return tuple(
        AABB(x0, x1, STAIRS_Z0 + i * run, STAIRS_Z0 + (i + 1) * run, 0.0, (i + 1) * rise)
        for i in range(CATWALK.steps)
    )


# Walkable solids block sight but must not collide (agents stand on them): the catwalk
# decks, the skybridge and the staircases. See Room.get_sight_blockers.
DATACORE_VAULT_OCCLUDERS = (
    AABB(-9.5, -6.0, 24.2, 42.0, 2.65, 2.75),  # left catwalk deck
    AABB(6.0, 9.5, 24.2, 42.0, 2.65, 2.75),  # right catwalk deck
    AABB(-9.5, 9.5, 39.5, 42.0, 2.65, 2.75),  # rear skybridge
    *_stair_occluders(-1),
    *_stair_occluders(+1),
)

DATACORE_VAULT_LIGHTS = (
    Light(
        0.0, 4.5, 32.0,
        ambient=(0.0, 0.28, 0.38), diffuse=(0.0, 3.8, 4.0), specular=(0.0, 4.0, 4.0),
        linear_attenuation=0.018, quadratic_attenuation=0.004,
        fixture="halo", fixture_emission=(0.0, 3.8, 4.0),
        fixture_color=(0.0, 0.85, 1.0), fixture_texture="neon_cyan", fixture_offset_y=0.7,
    ),
    # A matched pair of hanging cage lights (the staging bay's fixtures) flanking the
    # monolith: same height, same z, mirrored in x, cables up to the 6 m ceiling.
    Light(
        -5.0, 4.4, 32.0,
        ambient=(0.32, 0.0, 0.24), diffuse=(4.0, 0.0, 3.4), specular=(4.0, 0.6, 3.4),
        fixture="hanging_cage", fixture_emission=(4.0, 0.0, 3.0),
        fixture_color=(1.0, 0.0, 0.7), fixture_texture="neon_magenta", ceiling_y=5.98,
    ),
    Light(
        5.0, 4.4, 32.0,
        ambient=(0.0, 0.25, 0.3), diffuse=(0.2, 3.2, 3.8), specular=(0.1, 3.5, 4.0),
        fixture="hanging_cage", fixture_emission=(0.0, 3.4, 4.0),
        fixture_color=(0.0, 0.8, 1.0), fixture_texture="neon_cyan", ceiling_y=5.98,
    ),
)

DATACORE_VAULT_WALK = (
    WalkRegion(-1.6, 1.6, 17.5, 18.5, inset=frozenset({"min_x", "max_x"})),  # entrance strip from the airlock
    WalkRegion(-9.65, 9.65, 18.0, 41.95, inset=frozenset({"min_x", "max_x", "max_z"})),
)

DATACORE_VAULT = SectorData(
    name="DatacoreVault",
    bounds=(-10.0, 10.0, 18.0, 42.0),
    props=DATACORE_VAULT_PROPS + _staircase_props(-1) + _staircase_props(+1),
    colliders=DATACORE_VAULT_COLLIDERS,
    nav_blockers=DATACORE_VAULT_NAV_BLOCKERS,
    occluders=DATACORE_VAULT_OCCLUDERS,
    lights=DATACORE_VAULT_LIGHTS,
    walk_regions=DATACORE_VAULT_WALK,
)


# ========================================================================================
# Sector 3: Fungal Server Greenhouse (Z 42..80)
# ========================================================================================
#
# A sixteen-sided domed hall behind the vault. The servers here grow in symbiosis with
# bioluminescent fungus; racks and fungal trunks form a maze of hedges on a 2.8 m cell grid
# (2.0 m corridors). The maze is authored as data: ``GH_WALLS`` lists the cell pairs that a
# hedge separates, everything else (props, colliders, chambers, the rim of planter beds that
# fills the gap between the cell disc and the curved wall) is generated from it.
#
#       z=75.7  north wall, STREET EXIT door, short vestibule to the street (exit zone)
#       z=61    dome centre; cells (i, j) at x = 2.8 i, z = 61 + 2.8 j, |i|,|j| <= 4
#       z=46.3  south wall, opening to the spore lock corridor back to the vault door (z=42)
#
GH_PITCH = 2.8
GH_HEDGE = 0.8  # hedge thickness (rack depth)
GH_CENTER_Z = 61.0
GH_APOTHEM = 14.7  # dome wall distance from the centre
GH_SIDES = 16
GH_WALL_H = 5.0  # vertical tier of the dome
GH_DOOR_H = 2.6
GH_ROW_HALF_WIDTH = {0: 4, 1: 4, -1: 4, 2: 4, -2: 4, 3: 3, -3: 3, 4: 2, -4: 2}  # cells per row
GH_CELLS = tuple(
    (i, j) for j in range(-4, 5) for i in range(-GH_ROW_HALF_WIDTH[j], GH_ROW_HALF_WIDTH[j] + 1)
)
# Hedges between cell pairs (generated once with a seeded maze builder, then frozen).
GH_WALLS = (
    ((-4, -1), (-4, 0)), ((-4, 2), (-3, 2)), ((-3, -3), (-3, -2)), ((-3, -2), (-3, -1)),
    ((-3, -1), (-2, -1)), ((-3, 0), (-2, 0)), ((-3, 1), (-2, 1)), ((-2, -3), (-2, -2)),
    ((-2, -2), (-2, -1)), ((-2, -1), (-1, -1)), ((-2, 1), (-2, 2)), ((-2, 2), (-1, 2)),
    ((-2, 3), (-1, 3)), ((-1, -4), (-1, -3)), ((-1, -4), (0, -4)), ((-1, -3), (0, -3)),
    ((-1, -2), (0, -2)), ((-1, -1), (0, -1)), ((-1, 0), (-1, 1)), ((-1, 0), (0, 0)),
    ((-1, 1), (0, 1)), ((-1, 2), (0, 2)), ((-1, 3), (-1, 4)), ((0, -4), (1, -4)),
    ((0, -3), (0, -2)), ((0, 0), (1, 0)), ((0, 1), (1, 1)), ((0, 3), (0, 4)),
    ((0, 3), (1, 3)), ((1, -4), (1, -3)), ((1, -2), (2, -2)), ((1, -1), (2, -1)),
    ((1, 1), (1, 2)), ((1, 2), (1, 3)), ((2, -2), (2, -1)), ((2, 0), (3, 0)),
    ((2, 1), (2, 2)), ((2, 1), (3, 1)), ((2, 2), (2, 3)), ((2, 2), (3, 2)),
    ((2, 3), (2, 4)), ((3, -3), (3, -2)), ((3, -1), (3, 0)), ((3, 0), (4, 0)),
)
GH_GATE_CELLS = ((4, -1), (4, 0))  # the maze gate sits on this cell boundary
# 2x2 chambers ("control nodes"): lower-left cell of each. West: the exit terminal; east: the
# gate terminal. Both lie on the short and the long route respectively.
GH_CHAMBERS = ((-4, 0), (3, -2))

# The greenhouse is kept gloomy on purpose - just bright enough to find the way, lit mostly
# by the fungus itself (bright blue, pulsing) and a dim skylight.
MOSS = Material("moss", (0.28, 0.38, 0.27))
SOIL = Material("soil", (0.42, 0.36, 0.3))
FUNGUS = Material("fungus", (0.4, 0.38, 0.34))
CAP_GLOW = CAP_MATERIALS["blue"]
BIO_RACK = Material("metal", (0.1, 0.13, 0.13))
BIO_RACK_FACE = Material("server_rack", (0.55, 0.85, 0.75))
DOME_PANEL = Material("wall", (0.1, 0.15, 0.16))
DOME_GLASS = Material(None, (0.03, 0.08, 0.08), emission=(0.02, 0.05, 0.07), specular=(0.4, 0.6, 0.6), shininess=80.0)
DOME_GLASS_UPPER = Material(None, (0.04, 0.09, 0.1), emission=(0.05, 0.11, 0.14), specular=(0.4, 0.6, 0.6), shininess=80.0)
DOME_RIB = Material("metal", (0.1, 0.12, 0.13))
VINE = Material("moss", (0.18, 0.3, 0.17))
FOG = Material(None, (0.52, 0.54, 0.58), alpha=0.14)  # unlit soft discs: thin grey haze


def gh_cell_center(i: int, j: int) -> tuple[float, float]:
    return (i * GH_PITCH, GH_CENTER_Z + j * GH_PITCH)


def _gh_chamber_cells() -> set[tuple[int, int]]:
    return {(ci + di, cj + dj) for ci, cj in GH_CHAMBERS for di in (0, 1) for dj in (0, 1)}


GH_NODE_BEDS: list[AABB] = []  # filled by _gh_hedges: the planter tops at the hedge nodes


GH_RACKS: list[tuple[float, float, bool, float]] = []  # (cx, cz, along_z, top) per hedge
GH_NODE_PLANTERS: list[AABB] = []  # the hedge-post planters themselves (soil boxes)


def _gh_hedges() -> tuple[tuple[Prop, ...], tuple[AABB, ...]]:
    """One hedge per wall: a bio-server rack spanning the cell edge, a small planter bed at
    each end (the grid nodes) and vines over the top. Returns the props and, per hedge, the
    zone it occupies (rack plus both node planters) - where fungus may not grow on the
    floor. The colliders are built later, once the fungus is in (``_gh_colliders``)."""
    props: list[Prop] = []
    colliders: list[AABB] = []
    half = GH_PITCH / 2.0
    t = GH_HEDGE / 2.0
    trunks: dict[tuple[float, float], int] = {}
    for k, (a, b) in enumerate(GH_WALLS):
        (ax, az), (bx, bz) = gh_cell_center(*a), gh_cell_center(*b)
        cx, cz = (ax + bx) / 2.0, (az + bz) / 2.0
        along_z = a[0] != b[0]  # cells differ in i: the hedge runs along z at the shared x
        # No two racks are quite alike: height, a slight skew, and every fifth one is dead
        # (faces dark) with its vines hanging lower.
        rack_h = 2.1 + 0.1 * (k % 4)
        skew = ((k * 5) % 7 - 3) * 1.2  # -3.6..+3.6 degrees
        dead = k % 5 == 3
        face = Material("server_rack", (0.18, 0.2, 0.2)) if dead else BIO_RACK_FACE
        cs, sn = math.cos(math.radians(skew)), math.sin(math.radians(skew))
        # the vines on racks that meet overlap at the corner: no two equally thick (their
        # tops would share a plane and z-fight)
        vine_h = 0.12 - 0.004 * (k % 3)
        vine_y = rack_h + 0.002 * (k % 3) + vine_h / 2.0  # (their undersides too)
        GH_RACKS.append((cx, cz, along_z, rack_h + 0.12))  # up to the top of the vines
        if along_z:
            colliders.append(AABB(cx - t, cx + t, cz - half - 0.4, cz + half + 0.4, 0.0, 2.6))
            props.append(Prop.box(cx, rack_h / 2.0, cz, GH_HEDGE, rack_h, 2.0, BIO_RACK, rotate=((skew, 0.0, 1.0, 0.0),), solid=True, label=f"Bio-server rack {k}"))
            for side in (-1, 1):
                off = side * (t + 0.006)  # the face plate rides on the skewed rack
                props.append(Prop.box(cx + off * cs, rack_h / 2.0, cz - off * sn, 0.01, rack_h - 0.2, 1.9, face, rotate=((skew, 0.0, 1.0, 0.0),), label="Bio-server rack face"))
            props.append(Prop.box(cx, vine_y, cz, GH_HEDGE + 0.5, vine_h, 2.2 if not dead else 2.5, VINE, label="Vines draped over the rack"))
            ends = ((cx, cz - half), (cx, cz + half))
            feet = ()
        else:
            colliders.append(AABB(cx - half - 0.4, cx + half + 0.4, cz - t, cz + t, 0.0, 2.6))
            props.append(Prop.box(cx, rack_h / 2.0, cz, 2.0, rack_h, GH_HEDGE, BIO_RACK, rotate=((skew, 0.0, 1.0, 0.0),), solid=True, label=f"Bio-server rack {k}"))
            for side in (-1, 1):
                off = side * (t + 0.006)
                props.append(Prop.box(cx + off * sn, rack_h / 2.0, cz + off * cs, 1.9, rack_h - 0.2, 0.01, face, rotate=((skew, 0.0, 1.0, 0.0),), label="Bio-server rack face"))
            props.append(Prop.box(cx, vine_y, cz, 2.2 if not dead else 2.5, vine_h, GH_HEDGE + 0.5, VINE, label="Vines draped over the rack"))
            ends = ((cx - half, cz), (cx + half, cz))
            feet = ()
        for n, (ex, ez) in enumerate(ends):
            trunks.setdefault((round(ex, 3), round(ez, 3)), (k * 2 + n) % 7)
        del feet
    # Hedge posts: where racks meet, a small planter bed of soil sits in the gap (inside
    # the hedge collider) - the fungus grows on those in clumps (see _gh_fungus).
    for i, ((tx, tz), phase) in enumerate(trunks.items()):
        # a node on the rim edge is already inside the rim planter: no bed of its own
        j_lo = int(round((tz - GH_CENTER_Z) / GH_PITCH - 0.5))
        edges = [
            (GH_ROW_HALF_WIDTH[j] + 0.5) * GH_PITCH for j in (j_lo, j_lo + 1) if j in GH_ROW_HALF_WIDTH
        ]
        if abs(tx) >= min(edges) - 0.05:
            continue
        height = 0.42 + 0.04 * (i % 3)
        props.append(Prop.box(tx, height / 2.0, tz, 0.78, height, 0.78, SOIL, solid=True, label="Hedge post planter"))
        props.append(Prop.box(tx, height - 0.005, tz, 0.66, 0.03, 0.66, MOSS, label="Moss on the hedge post planter"))
        GH_NODE_BEDS.append(AABB(tx - 0.36, tx + 0.36, tz - 0.36, tz + 0.36, height, height + 1.0))
        GH_NODE_PLANTERS.append(AABB(tx - 0.39, tx + 0.39, tz - 0.39, tz + 0.39, 0.0, height))
        del phase
    return tuple(props), tuple(colliders)


GH_RIM_BEDS: list[AABB] = []  # each planter bed's collider footprint (height set later)
GH_RIM_WALLS: list[AABB] = []  # behind the beds: the dome wall and what lies beyond it


def _gh_rim() -> tuple[tuple[Prop, ...], tuple[AABB, ...], tuple[AABB, ...]]:
    """Everything between the cell disc and the curved wall is planter bed: low beds of
    soil, and a collider per row band that seals the rim (so nothing reachable is ever off
    the maze). The south and north bands leave the door channels open. Also returns the
    bed tops as zones for the fungus to grow on."""
    props: list[Prop] = []
    colliders: list[AABB] = []
    beds: list[AABB] = []
    half = GH_PITCH / 2.0
    far = GH_APOTHEM + 2.5
    k = 0
    for j in range(-4, 5):
        z0, z1 = GH_CENTER_Z + j * GH_PITCH - half, GH_CENTER_Z + j * GH_PITCH + half
        edge = (GH_ROW_HALF_WIDTH[j] + 0.5) * GH_PITCH
        for side in (-1, 1):
            lo, hi = (edge, far) if side > 0 else (-far, -edge)
            colliders.append(AABB(lo, hi, z0, z1, 0.0, GH_WALL_H))
            # the bed stops at the wall: width from the cell edge to the dome at this band
            dz = max(abs(z0 - GH_CENTER_Z), abs(z1 - GH_CENTER_Z))
            wall_x = math.sqrt(max(0.0, GH_APOTHEM**2 - dz**2))
            width = max(0.6, wall_x - edge)
            bx = side * (edge + width / 2.0)
            bed_end = edge + width
            GH_RIM_BEDS.append(AABB(min(side * edge, side * bed_end), max(side * edge, side * bed_end), z0, z1, 0.0, 0.56))
            GH_RIM_WALLS.append(AABB(min(side * bed_end, side * far), max(side * bed_end, side * far), z0, z1, 0.0, GH_WALL_H))
            props.append(Prop.box(bx, 0.28, (z0 + z1) / 2.0, width, 0.56, z1 - z0, SOIL, solid=True, label=f"Planter bed {k}"))
            props.append(Prop.box(bx, 0.55, (z0 + z1) / 2.0, max(0.2, width - 0.3), 0.04, z1 - z0 - 0.3, MOSS, label="Moss on the planter bed"))
            if width > 1.0:  # room for fungus: the bed top, kept well inside the dome wall
                inner, outer = edge + 0.15, edge + width - 0.45
                beds.append(AABB(min(side * inner, side * outer), max(side * inner, side * outer), z0 + 0.2, z1 - 0.2, 0.56, 1.5))
            k += 1
    # South band: between the dome wall and the first cell row, except the entry channel
    z_wall_s = GH_CENTER_Z - GH_APOTHEM
    z_row_s = GH_CENTER_Z - 4 * GH_PITCH - half
    for side in (-1, 1):
        lo, hi = (1.6, far) if side > 0 else (-far, -1.6)
        colliders.append(AABB(lo, hi, z_wall_s - 0.5, z_row_s, 0.0, GH_WALL_H))
        near, bed_end = (1.6, 6.7) if side > 0 else (-6.7, -1.6)
        GH_RIM_BEDS.append(AABB(near, bed_end, z_wall_s - 0.5, z_row_s, 0.0, 0.56))
        GH_RIM_WALLS.append(AABB(6.7 if side > 0 else -far, far if side > 0 else -6.7, z_wall_s - 0.5, z_row_s, 0.0, GH_WALL_H))
        props.append(Prop.box(side * 4.2, 0.28, (z_wall_s + z_row_s) / 2.0, 5.0, 0.56, z_row_s - z_wall_s - 0.3, SOIL, solid=True, label="South planter bed"))
    # North band: from the last row to the dome wall, except the exit channel; beyond the
    # wall the street vestibule is walled by the same boxes.
    z_row_n = GH_CENTER_Z + 4 * GH_PITCH + half
    for side in (-1, 1):
        lo, hi = (1.6, far) if side > 0 else (-far, -1.6)
        colliders.append(AABB(lo, hi, z_row_n, 80.5, 0.0, GH_WALL_H))
        face = GH_CENTER_Z + GH_APOTHEM - 0.15  # the dome wall's inner face
        GH_RIM_BEDS.append(AABB(1.6 if side > 0 else -6.7, 6.7 if side > 0 else -1.6, z_row_n, face, 0.0, 0.56))
        GH_RIM_WALLS.append(AABB(6.7 if side > 0 else -far, far if side > 0 else -6.7, z_row_n, 80.5, 0.0, GH_WALL_H))
        GH_RIM_WALLS.append(AABB(1.6 if side > 0 else -6.7, 6.7 if side > 0 else -1.6, face, 80.5, 0.0, GH_WALL_H))
        props.append(Prop.box(side * 4.2, 0.28, (z_row_n + GH_CENTER_Z + GH_APOTHEM) / 2.0, 5.0, 0.56, GH_CENTER_Z + GH_APOTHEM - z_row_n - 0.3, SOIL, solid=True, label="North planter bed"))
        beds.append(AABB(min(side * 2.0, side * 6.2), max(side * 2.0, side * 6.2), z_row_n + 0.2, GH_CENTER_Z + GH_APOTHEM - 0.7, 0.56, 1.5))
        beds.append(AABB(min(side * 2.0, side * 6.2), max(side * 2.0, side * 6.2), z_wall_s + 0.5, z_row_s - 0.2, 0.56, 1.5))
    return tuple(props), tuple(colliders), tuple(beds)


def _gh_dome() -> tuple[Prop, ...]:
    """The globular hall: a vertical sixteen-sided wall tier with the two door openings,
    two sloping glass tiers closing inward above it, ribs, and a glowing skylight cap."""
    props: list[Prop] = []
    n = GH_SIDES
    circ = GH_APOTHEM / math.cos(math.pi / n)
    edge_len = 2.0 * circ * math.sin(math.pi / n)
    for k in range(n):
        theta = k * 360.0 / n  # radial direction, from +x toward +z
        rad = math.radians(theta)
        cx, cz = GH_APOTHEM * math.cos(rad), GH_CENTER_Z + GH_APOTHEM * math.sin(rad)
        yaw = -(theta + 90.0)  # box length axis (local x) along the tangent
        if k in (4, 12):  # north (k=4, +z) and south (k=12, -z): split around the door
            gap = 1.6
            piece = edge_len / 2.0 - gap
            for side in (-1, 1):
                px = cx + side * (gap + piece / 2.0)
                props.append(Prop.box(px, GH_WALL_H / 2.0, cz, piece, GH_WALL_H, 0.3, DOME_PANEL, rotate=((yaw, 0.0, 1.0, 0.0),), label="Dome wall panel beside the door", tags=frozenset({"compound"})))
            props.append(Prop.box(cx, (GH_WALL_H + GH_DOOR_H) / 2.0, cz, 2 * gap + 0.1, GH_WALL_H - GH_DOOR_H, 0.31, DOME_PANEL, rotate=((yaw, 0.0, 1.0, 0.0),), label="Dome wall lintel over the door"))
            if k == 4:
                # The street exit: the same Halcyon staff door the alley sees from outside -
                # frame head, STAFF ONLY sign and a concrete portal on the inside face.
                props.extend(staff_door_trim(0.0, cz - 0.15, -1, portal=True))
                continue
            for side in (-1, 1):
                props.append(Prop.box(cx + side * (gap + 0.115), GH_DOOR_H / 2.0, cz, 0.24, GH_DOOR_H, 0.4, DOME_RIB, solid=True, label="Dome door frame post"))
        else:
            props.append(Prop.box(cx, GH_WALL_H / 2.0, cz, edge_len, GH_WALL_H, 0.3, DOME_PANEL, rotate=((yaw, 0.0, 1.0, 0.0),), label=f"Dome wall panel {k}", tags=frozenset({"compound"})))
        # fungal growth climbing the wall: a vine strip on the inside face
        vx, vz = (GH_APOTHEM - 0.25) * math.cos(rad), GH_CENTER_Z + (GH_APOTHEM - 0.25) * math.sin(rad)
        if k not in (4, 12):
            props.append(Prop.box(vx, 2.4, vz, edge_len * 0.6, 0.25, 0.1, VINE, rotate=((yaw, 0.0, 1.0, 0.0),), label="Vines on the dome wall"))
    # Sloping tiers: (inner apothem, y0, y1)
    tiers = ((GH_APOTHEM, 11.0, GH_WALL_H, 8.0, DOME_GLASS), (11.0, 6.0, 8.0, 9.6, DOME_GLASS_UPPER))
    for a0, a1, y0, y1, glass in tiers:
        mid = (a0 + a1) / 2.0
        length = a0 * 2.0 * math.tan(math.pi / n) + 0.2  # wide enough at the outer edge
        slope = math.hypot(a0 - a1, y1 - y0)
        tilt = math.degrees(math.atan2(a0 - a1, y1 - y0))  # lean inward
        for k in range(n):
            theta = k * 360.0 / n
            rad = math.radians(theta)
            cx, cz = mid * math.cos(rad), GH_CENTER_Z + mid * math.sin(rad)
            yaw = -(theta + 90.0)
            props.append(Prop.box(cx, (y0 + y1) / 2.0, cz, length, slope, 0.2, glass, rotate=((yaw, 0.0, 1.0, 0.0), (tilt, 1.0, 0.0, 0.0)), label="Dome glass panel"))
            # rib at the panel's leading vertex
            vt = math.radians(theta + 180.0 / n)
            r0 = a0 / math.cos(math.pi / n)
            r1 = a1 / math.cos(math.pi / n)
            rm = (r0 + r1) / 2.0
            props.append(Prop.box(rm * math.cos(vt), (y0 + y1) / 2.0, GH_CENTER_Z + rm * math.sin(vt), 0.18, math.hypot(r0 - r1, y1 - y0), 0.18, DOME_RIB, rotate=((-(math.degrees(vt) + 90.0), 0.0, 1.0, 0.0), (math.degrees(math.atan2(r0 - r1, y1 - y0)), 1.0, 0.0, 0.0)), label="Dome rib"))
    props.append(Prop.cylinder(0.0, 9.55, GH_CENTER_Z, 6.3, 6.3, 0.3, Material("spore", (0.4, 0.6, 0.65), emission=(0.08, 0.22, 0.3)), slices=16, label="Dome skylight"))
    props.append(Prop.cylinder(0.0, 9.85, GH_CENTER_Z, 6.5, 6.5, 0.2, DOME_RIB, slices=16, label="Dome skylight rim"))
    return tuple(props)


VESTIBULE_DOOR_FACE_Z = 79.2  # the vestibule's back wall, with the outer staff door on it
VESTIBULE_DOOR_LABEL = "Outer staff door (to the alley)"
VESTIBULE_STRIP_LIGHT = (0.46, 0.33, 0.2)  # the warm ceiling strip's light on the door, baked in


def _gh_floor_and_corridors() -> tuple[tuple[Prop, ...], tuple[AABB, ...]]:
    """Mossy floor, the spore-lock corridor from the vault door to the dome, and the short
    street vestibule beyond the exit door."""
    props: list[Prop] = [
        Prop.box(0.0, -0.05, 61.0, 34.0, 0.1, 38.0, MOSS, label="GREENHOUSE FLOOR (moss over drainage)"),
        Prop.box(0.0, 0.003, 61.0, 1.6, 0.006, 29.0, Material("grate", (0.3, 0.36, 0.32)), label="Central drainage runway"),
    ]
    colliders: list[AABB] = []
    # Spore lock: z 42.1..46.3, x +-1.6, with a low ceiling and spore-filter fans on the walls
    zc = (42.12 + GH_CENTER_Z - GH_APOTHEM - 0.15) / 2.0
    length = GH_CENTER_Z - GH_APOTHEM - 0.15 - 42.12  # (to the dome wall's face)
    for side in (-1, 1):
        props.append(Prop.box(side * 1.7, 1.75, zc, 0.2, 3.5, length, Material("wall", (0.22, 0.28, 0.26)), solid=True, label="Spore lock wall"))
        props.append(Prop.cylinder(side * 1.58, 1.5, zc, 0.3, 0.3, 0.06, DOME_RIB, slices=10, rotate=((side * 90.0, 0.0, 0.0, 1.0),), label="Spore filter fan"))
        colliders.append(AABB(side * 1.6 if side > 0 else -17.0, 17.0 if side > 0 else -1.6, 42.0, GH_CENTER_Z - GH_APOTHEM, 0.0, 5.0))
    props.append(Prop.box(0.0, 3.55, zc, 3.6, 0.1, length, Material("wall", (0.18, 0.2, 0.2)), label="Spore lock ceiling"))
    props.append(Prop.box(0.0, 3.45, zc, 1.2, 0.08, length - 0.4, Material(None, (0.08, 0.3, 0.12), emission=(0.3, 2.6, 0.55)), label="Spore lock UV light strip"))
    # Street vestibule beyond the exit door: z 75.9..79.2, a bare concrete stub between
    # two doors. Its back wall carries the outer staff door - the yellow door the alley
    # sees you come out of (levels/doorway.py). It never opens here: walking up to it (the
    # exit zone in front of it) hands the player over to the city's own level
    # (levels/street.py), which starts just outside that same door.
    props.append(Prop.box(0.0, -0.04, 77.6, 3.2, 0.1, 3.4, Material("concrete", (0.17, 0.2, 0.25), specular=(0.4, 0.45, 0.5), shininess=40.0), label="Street vestibule floor (wet concrete)"))
    for side in (-1, 1):
        props.append(Prop.box(side * 1.7, 2.0, 77.6, 0.2, 4.0, 3.4, Material("wall", (0.3, 0.3, 0.34)), solid=True, label="Street vestibule wall"))
    props.append(Prop.box(0.0, 2.0, 79.3, 3.56, 4.0, 0.2, Material("wall", (0.2, 0.2, 0.24)), solid=True, label="Street vestibule back wall"))
    props.append(Prop.box(0.0, 4.05, 77.6, 3.6, 0.1, 3.4, Material("wall", (0.14, 0.14, 0.18)), label="Street vestibule ceiling"))
    props.append(Prop.box(0.0, 3.96, 77.6, 0.3, 0.06, 2.6, Material(None, (1.0, 0.85, 0.6), emission=(1.7, 1.3, 0.8)), label="Vestibule ceiling light strip"))
    # The dome's one light is cyan, and under it the yellow hazard stripes read as green.
    # The vestibule is lit by its own warm strip instead (the level's eight lights are
    # all spoken for, so its light is baked into the door as emission), and the door you
    # leave by is the same yellow the alley shows.
    leaf = staff_door_leaf(0.0, VESTIBULE_DOOR_FACE_Z, -1)
    lit = replace(leaf.material, emission=VESTIBULE_STRIP_LIGHT)
    props.append(replace(leaf, material=lit, label=VESTIBULE_DOOR_LABEL))
    props.extend(staff_door_trim(0.0, VESTIBULE_DOOR_FACE_Z, -1))
    props.append(staff_door_alarm(0.0, VESTIBULE_DOOR_FACE_Z, -1))
    # the EXIT sign hangs above the door's STAFF ONLY neon, just under the ceiling
    props.append(Prop.box(0.0, 3.72, VESTIBULE_DOOR_FACE_Z - 0.03, 0.9, 0.3, 0.04, Material(None, (0.2, 0.9, 0.3), emission=(0.15, 0.9, 0.25)), label="EXIT sign (to the street)", tags=frozenset({"transparent"})))
    colliders.append(AABB(-2.0, 2.0, 79.2, 80.5, 0.0, 4.1))  # the back wall, up to the ceiling
    return tuple(props), tuple(colliders)


def gh_dome_glass_y(x: float, z: float) -> float:
    """Height of the dome's glass (its mid-surface) straight above a point of the hall: the
    two sloping tiers of the sixteen-sided dome, or the skylight in the middle."""
    r = math.hypot(x, z - GH_CENTER_Z)
    if r < 1e-6:
        return 9.55
    sector = 2.0 * math.pi / GH_SIDES
    theta = math.atan2(z - GH_CENTER_Z, x)
    d = r * math.cos(theta - round(theta / sector) * sector)  # distance along the panel normal
    if d >= 11.0:
        return GH_WALL_H + (GH_APOTHEM - d) / (GH_APOTHEM - 11.0) * (8.0 - GH_WALL_H)
    if d >= 6.0:
        return 8.0 + (11.0 - d) / 5.0 * 1.6
    return 9.55


def _gh_terminal_dressing() -> tuple[Prop, ...]:
    """Hanging grow-lamps over the two control chambers (decoration; the chambers' terminals
    are entities), each on a pair of cables up to the dome glass."""
    props = []
    for ci, cj in GH_CHAMBERS:
        x0, z0 = gh_cell_center(ci, cj)
        x1, z1 = gh_cell_center(ci + 1, cj + 1)
        cx, cz = (x0 + x1) / 2.0, (z0 + z1) / 2.0
        props.append(Prop.box(cx, 3.2, cz, 2.6, 0.08, 0.4, Material(None, (0.3, 0.15, 0.4), emission=(1.6, 0.45, 2.6)), label="Control node grow lamp", tags=frozenset({"transparent"})))
        props.append(Prop.box(cx, 3.27, cz, 2.8, 0.06, 0.5, DOME_RIB, label="Control node grow lamp housing", tags=frozenset({"transparent"})))
        for side in (-1, 1):  # it used to float in mid-air: hang it from the dome
            kx = cx + side * 1.15
            top = gh_dome_glass_y(kx, cz)
            props.append(Prop.box(kx, (3.3 + top) / 2.0, cz, 0.02, top - 3.3, 0.02, DOME_RIB, label="Grow lamp cable (to the dome)", tags=frozenset({"transparent"})))
    return tuple(props)


def _gh_fog() -> tuple[Prop, ...]:
    """A layer of grey haze hanging above the maze, under the glass: thin soft-edged discs
    (full in the middle, fading out like a cloud) plus a few elliptical puffs. Every disc is
    kept well inside the glass tier at its height, so nothing ever intersects the dome and
    pops as the viewpoint moves. The renderer draws them last, depth writes off, drifting."""
    fog = frozenset({"fog", "transparent"})

    def glass_radius(y: float) -> float:  # the lower glass tier: r 14.7 at y 5 -> 11 at y 8
        return GH_APOTHEM - (y - GH_WALL_H) / 3.0 * (GH_APOTHEM - 11.0)

    props = []
    for y in (6.2, 6.9, 7.6):
        r = glass_radius(y) - 0.8
        props.append(Prop.disc(0.0, y, GH_CENTER_Z, r, r, 0.45, FOG, slices=32, label="Fog layer", tags=fog))
    for k in range(9):
        a = k * 40.0 + 7.0
        y = 6.0 + 0.45 * (k % 4)
        rad = math.radians(a)
        reach = glass_radius(y) - 4.5
        d = 2.0 + (k % 3) * 0.32 * reach
        props.append(Prop.disc(d * math.cos(rad), y, GH_CENTER_Z + d * math.sin(rad), 3.6, 2.0, 0.25, FOG,
                               slices=24, rotate=((a * 1.7, 0.0, 1.0, 0.0),), label="Fog puff", tags=fog))
    return tuple(props)


def _gh_fungus(hedge_colliders, rim_colliders, floor_colliders, beds) -> tuple[Prop, ...]:
    """Where the fungus grows: clumps at the feet of the hedges (small, out of the
    walkways) and bigger spreads across the planter beds and at the dome wall."""
    from cyberduke.entities.terminal import Terminal

    desks = [Terminal(11.3, 0.0, 56.1, yaw=-90.0).get_bounds(), Terminal(-11.3, 0.0, 62.4, yaw=90.0).get_bounds()]
    gx0, gz = gh_cell_center(4, 0)[0] - GH_PITCH / 2.0, GH_CENTER_Z - GH_PITCH / 2.0
    gate = AABB(gx0 - 0.3, (GH_ROW_HALF_WIDTH[0] + 0.5) * GH_PITCH + 0.3, gz - 0.5, gz + 0.5, 0.0, 2.6)
    # On the floor, everything solid is a blocker. On the beds the rim fillers are the
    # beds themselves, so there only the hedges and desks count; both placers share one
    # ``placed`` list so no two clumps ever meet.
    floor_placer = Placer(
        seed=1107, blockers=list(hedge_colliders) + list(rim_colliders) + list(floor_colliders) + desks + [gate]
    )
    bed_placer = Placer(seed=2207, blockers=list(hedge_colliders) + desks, placed=floor_placer.placed)
    props: list[Prop] = []
    for k, box in enumerate(hedge_colliders):
        if (k * 13) % 10 >= 8:
            continue  # not every hedge has a clump at its foot
        along_z = (box.max_z - box.min_z) > (box.max_x - box.min_x)
        side = 1 if (k * 7) % 2 == 0 else -1
        if along_z:
            x0 = box.max_x + 0.07 if side > 0 else box.min_x - 0.41
            zone = AABB(x0, x0 + 0.34, box.min_z + 0.5, box.max_z - 0.5, 0.0, 0.6)
        else:
            z0 = box.max_z + 0.07 if side > 0 else box.min_z - 0.41
            zone = AABB(box.min_x + 0.5, box.max_x - 0.5, z0, z0 + 0.34, 0.0, 0.6)
        props.extend(mushroom_cluster(floor_placer, zone, 2 + (k * 3) % 4, (0.3, 0.6), spread=0.7))
    for k, bed in enumerate(beds):
        props.extend(mushroom_cluster(bed_placer, bed, 3 + (k * 5) % 6, (0.5, 1.5), floor_y=0.56))
        props.extend(mushroom_cluster(bed_placer, bed, 2 + k % 4, (0.3, 0.8), floor_y=0.56))
    # every hedge-post planter carries its own random spread of small mushrooms
    node_placer = Placer(seed=3307, blockers=[], placed=floor_placer.placed)
    for k, bed in enumerate(GH_NODE_BEDS):
        props.extend(mushroom_cluster(node_placer, bed, 2 + (k * 7) % 4, (0.22, 0.5), floor_y=bed.min_y, spread=0.3))
    return tuple(props)


def _gh_colliders(fungus: tuple[Prop, ...]) -> tuple[AABB, ...]:
    """Colliders for the hedges and the rim, each as tall as what is really there: a rack
    to the top of its vines; a hedge-post planter or a planter bed to the top of the
    tallest mushroom growing on it (they used to be one 2.6 m / 5 m block, an invisible
    wall over the mushrooms that hid a standing netrunner from guards and scanners); the
    dome wall behind the beds full height."""
    caps = [p.footprint() for p in fungus if p.label == "Mushroom cap"]

    def grown(box: AABB) -> AABB:
        tops = [c.max_y for c in caps if box.min_x <= c.center_x <= box.max_x and box.min_z <= c.center_z <= box.max_z]
        return replace(box, max_y=round(max([box.max_y, *tops]), 3))

    half_rack = 1.02  # the 2 m rack (and its skew) from the middle of the cell edge
    half = GH_PITCH / 2.0
    planters = {(round(b.center_x, 3), round(b.center_z, 3)) for b in GH_NODE_PLANTERS}
    out: list[AABB] = []
    for cx, cz, along_z, top in GH_RACKS:
        t = GH_HEDGE / 2.0
        # up to the planter at each end; a rack ending on the rim runs on into the bed
        ends = ((cx, cz - half), (cx, cz + half)) if along_z else ((cx - half, cz), (cx + half, cz))
        reach = [half_rack if (round(ex, 3), round(ez, 3)) in planters else half + 0.4 for ex, ez in ends]
        if along_z:
            out.append(AABB(cx - t, cx + t, cz - reach[0], cz + reach[1], 0.0, top))
        else:
            out.append(AABB(cx - reach[0], cx + reach[1], cz - t, cz + t, 0.0, top))
    out.extend(grown(b) for b in GH_NODE_PLANTERS)
    out.extend(grown(b) for b in GH_RIM_BEDS)
    out.extend(GH_RIM_WALLS)
    return tuple(out)


_GH_HEDGE_PROPS, _GH_HEDGE_ZONES = _gh_hedges()
_GH_RIM_PROPS, _GH_RIM_ZONES, _GH_BEDS = _gh_rim()
_GH_FLOOR_PROPS, _GH_FLOOR_COLLIDERS = _gh_floor_and_corridors()
_GH_FUNGUS_PROPS = _gh_fungus(_GH_HEDGE_ZONES, _GH_RIM_ZONES, _GH_FLOOR_COLLIDERS, _GH_BEDS)
_GH_COLLIDERS = _gh_colliders(_GH_FUNGUS_PROPS)

GREENHOUSE_LIGHTS = (
    Light(
        0.0, 8.6, GH_CENTER_Z,
        ambient=(0.02, 0.06, 0.06), diffuse=(0.12, 0.5, 0.55), specular=(0.1, 0.4, 0.5),
        linear_attenuation=0.012, quadratic_attenuation=0.0015,
        fixture="halo", fixture_emission=(0.1, 0.35, 0.4),
        fixture_color=(0.5, 0.8, 0.85), fixture_texture="spore", fixture_offset_y=0.85,
    ),
)

GREENHOUSE_WALK = (
    WalkRegion(-16.5, 16.5, 42.0, 79.4, inset=frozenset({"min_x", "max_x", "max_z"})),
)

GREENHOUSE = SectorData(
    name="FungalGreenhouse",
    bounds=(-17.0, 17.0, 42.0, 80.0),
    props=_GH_FLOOR_PROPS + _gh_dome() + _GH_HEDGE_PROPS + _GH_RIM_PROPS
    + tuple(light_fungus(_GH_FUNGUS_PROPS, _GH_FLOOR_PROPS + _gh_dome() + _GH_HEDGE_PROPS + _GH_RIM_PROPS))
    + _gh_terminal_dressing() + _gh_fog(),
    colliders=_GH_COLLIDERS + _GH_FLOOR_COLLIDERS,
    lights=GREENHOUSE_LIGHTS,
    walk_regions=GREENHOUSE_WALK,
)


# ========================================================================================
# Factions
# ========================================================================================

# Everyone and everything in the facility belongs to the corp's security division and sits
# on its one local network: breach any computer on it and every guard, camera, door and
# vending machine lights up - except the datacore terminal, which is secured (private).
HALCYON = FactionData("halcyon", "Halcyon Dynamics Security", color=(1.0, 0.55, 0.1))
# The greenhouse is run by the biotech division on its own network: breaching the security
# net in the facility does not reach in here. Its computers are air-gapped (jack in only),
# and jacking into one is the way onto the biotech network.
HALCYON_BIO = FactionData("halcyon_bio", "Halcyon Biotech Research", color=(0.35, 1.0, 0.45))


# ========================================================================================
# Doors
# ========================================================================================

BLAST_DOOR = DoorData(
    door_id="blast_door",
    obstacle=AABB(-1.55, 1.55, 9.88, 10.12, 0.0, 3.55),
    unlocked_by="staging_terminal",
    passage=WalkRegion(-1.5, 1.5, 9.2, 10.8, inset=frozenset({"min_x", "max_x"})),
    leaf=(
        Prop.box(0.0, 1.75, 10.0, 3.1, 3.55, 0.22, Material("hazard", (0.85, 0.85, 0.85)), label="Sliding Blast Door (Width 3.1 covers [-1.55, 1.55], height 3.55 sits on floor y=0.0)"),
    ),
    indicators=(
        Prop.box(0.0, 3.3, 9.88, 2.6, 0.08, 0.02, FLAT, label="Sector 1 side strip"),
        Prop.box(0.0, 3.3, 10.12, 2.6, 0.08, 0.02, FLAT, label="Airlock side strip"),
    ),
    faction="halcyon",
    # MIL-SPEC lock: a quickhack on the door itself bounces off; only the staging terminal
    # (jacked into, or told to override its doors) opens it.
    high_security=True,
)

DATACORE_INTEL = "greenhouse door cipher"
"""The secured data on the datacore terminal: the key to the greenhouse door."""

# The bulkhead between the airlock and the vault. No terminal controls it - it is a plain
# networked lock, so it yields to a direct FORCE OPEN quickhack (scan it from the corridor,
# under the camera's sweep) or to one sent over the breached network.
VAULT_BULKHEAD = DoorData(
    door_id="vault_bulkhead",
    obstacle=AABB(-1.6, 1.6, 17.88, 18.12, 0.0, 4.2),
    unlocked_by=None,
    passage=WalkRegion(-1.5, 1.5, 17.3, 18.7, inset=frozenset({"min_x", "max_x"})),
    leaf=(
        Prop.box(0.0, 2.1, 18.0, 3.2, 4.2, 0.22, Material("metal", (0.55, 0.58, 0.66)), label="Vault Bulkhead leaf (covers the partition opening x in [-1.6, 1.6], 4.2 m tall)"),
        Prop.box(0.0, 2.1, 18.0, 0.12, 3.6, 0.25, Material("hazard", (0.85, 0.85, 0.85)), label="Vault Bulkhead centre hazard stripe"),
    ),
    indicators=(
        Prop.box(0.0, 3.9, 17.88, 2.6, 0.08, 0.02, FLAT, label="Airlock side strip"),
        Prop.box(0.0, 3.9, 18.12, 2.6, 0.08, 0.02, FLAT, label="Vault side strip"),
    ),
    faction="halcyon",
)


# The way into the greenhouse: a door in the vault's back wall under the skybridge. It is
# encryption-keyed: the cipher is the "secured data" on the datacore terminal. Walk up to
# it carrying that intel and it opens; no quickhack forces it.
GREENHOUSE_DOOR = DoorData(
    door_id="greenhouse_door",
    obstacle=AABB(-1.6, 1.6, 41.88, 42.12, 0.0, 2.5),
    unlocked_by=None,
    travel=3.2,
    passage=WalkRegion(-1.5, 1.5, 41.3, 42.9, inset=frozenset({"min_x", "max_x"})),
    leaf=(
        Prop.box(0.0, 1.25, 42.0, 3.2, 2.5, 0.22, Material("metal", (0.45, 0.55, 0.5)), label="Greenhouse door leaf (under the skybridge, x in [-1.6, 1.6], 2.5 m tall)"),
        Prop.box(0.0, 1.25, 42.0, 0.12, 2.2, 0.25, Material("hazard", (0.6, 0.9, 0.6)), label="Greenhouse door centre stripe"),
    ),
    indicators=(
        Prop.box(0.0, 2.3, 41.88, 2.6, 0.08, 0.02, FLAT, label="Vault side strip"),
        Prop.box(0.0, 2.3, 42.12, 2.6, 0.08, 0.02, FLAT, label="Greenhouse side strip"),
    ),
    faction="halcyon",
    high_security=True,
    key_intel=DATACORE_INTEL,
)

# The maze gate: shuts the short route north out of the east control node. Opened from the
# node's air-gapped terminal (jack in) - or walk the long way round.
_GATE_X0, _GATE_X1 = gh_cell_center(4, 0)[0] - GH_PITCH / 2.0, (GH_ROW_HALF_WIDTH[0] + 0.5) * GH_PITCH
_GATE_Z = (gh_cell_center(*GH_GATE_CELLS[0])[1] + gh_cell_center(*GH_GATE_CELLS[1])[1]) / 2.0
MAZE_GATE = DoorData(
    door_id="maze_gate",
    obstacle=AABB(_GATE_X0, _GATE_X1, _GATE_Z - 0.12, _GATE_Z + 0.12, 0.0, 2.6),
    unlocked_by="node_east_terminal",
    travel=_GATE_X1 - _GATE_X0,
    speed=2.0,
    passable_at=1.3,
    leaf=(
        Prop.box((_GATE_X0 + _GATE_X1) / 2.0, 1.3, _GATE_Z, _GATE_X1 - _GATE_X0, 2.6, 0.2, Material("grate", (0.5, 0.7, 0.6)), label="Maze gate (mesh leaf across the hedge gap)"),
        Prop.box((_GATE_X0 + _GATE_X1) / 2.0, 2.555, _GATE_Z, _GATE_X1 - _GATE_X0 - 0.01, 0.1, 0.3, DOME_RIB, label="Maze gate head rail"),
    ),
    indicators=(
        Prop.box((_GATE_X0 + _GATE_X1) / 2.0, 2.42, _GATE_Z - 0.12, 1.6, 0.06, 0.02, FLAT, label="Gate strip, chamber side"),
        Prop.box((_GATE_X0 + _GATE_X1) / 2.0, 2.42, _GATE_Z + 0.12, 1.6, 0.06, 0.02, FLAT, label="Gate strip, maze side"),
    ),
    faction="halcyon_bio",
    high_security=True,
)

# The way out: the door in the north wall of the dome to the street. Unlocked from the west
# control node's terminal, it opens as you walk up to it. Stepping through it (the exit
# zone) wins.
_EXIT_Z = GH_CENTER_Z + GH_APOTHEM
STREET_EXIT = DoorData(
    door_id="street_exit",
    obstacle=AABB(-1.6, 1.6, _EXIT_Z - 0.12, _EXIT_Z + 0.12, 0.0, GH_DOOR_H),
    unlocked_by="node_west_terminal",
    travel=3.2,
    # The Halcyon staff door (levels/doorway.py): exactly the door the alley sees. Its
    # status light is the alarm lamp over it - red while locked, green once it opens.
    leaf=(
        Prop.box(0.0, STAFF_DOOR_HEIGHT / 2.0, _EXIT_Z, STAFF_DOOR_WIDTH, STAFF_DOOR_HEIGHT, 0.22, STAFF_DOOR, label="Street exit door leaf (north wall, x in [-1.6, 1.6])"),
    ),
    lamps=(staff_door_alarm(0.0, _EXIT_Z - 0.15, -1),),
    # a staff door: unlocked from the west node it stays shut (its lamp green) until
    # somebody walks up to it, and slides shut again behind them
    approach_range=3.0,
    faction="halcyon_bio",
    high_security=True,
)

# Walking up to the vestibule's outer door (the last metre in front of it) is leaving.
EXIT_ZONE = AABB(-1.6, 1.6, 78.2, VESTIBULE_DOOR_FACE_Z, 0.0, 3.0)


# ========================================================================================
# Entities & spawn
# ========================================================================================

TERMINALS = (
        # Sector 1: controls the blast door (BLAST_DOOR.unlocked_by) and is the way onto
        # the Halcyon network. Critical: it is the only way out of the staging bay, so the
        # short-circuit option is locked out - fry it and the mission would be over.
        TerminalData("staging_terminal", 3.5, 0.0, -4.0, faction="halcyon", critical=True, style="damaged"),
        # Sector 2: the mission objective, rotated to face out from the monolith. Secured
        # (air-gapped from the network: scan it, then hack it with a line of sight, or jack
        # in) and critical (it cannot be destroyed - the data must be taken, not burned).
        # Its data is the cipher that opens the greenhouse door (GREENHOUSE_DOOR.key_intel).
        TerminalData("datacore_terminal", 0.0, 0.0, 30.0, yaw=180.0, objective=True, faction="halcyon", intel=DATACORE_INTEL, secured=True, critical=True),
        # Sector 2: a coolant maintenance workstation under the right catwalk. It controls
        # nothing and holds nothing critical, so it is the one computer in the facility that
        # can be short-circuited for a loud distraction.
        TerminalData("vault_maintenance_terminal", 7.5, 0.0, 35.5, yaw=-90.0, faction="halcyon", style="single"),
        # Sector 3: the two control nodes. Air-gapped (hardened): scanning shows them, but
        # only jacking in works - and jacking in breaches the biotech network.
        # East node (inside the gated chamber on the short route): opens the maze gate.
        TerminalData("node_east_terminal", 11.3, 0.0, 56.1, yaw=-90.0, faction="halcyon_bio", hardened=True, style="double"),
        # West node (on the long route): opens the street exit.
        TerminalData("node_west_terminal", -11.3, 0.0, 62.4, yaw=90.0, faction="halcyon_bio", hardened=True, style="overgrown"),
    )
CAMERAS = (
        CameraData(0.0, 3.6, 11.2, mount_yaw=0.0, sweep_angle=35.0, sweep_speed=1.2, camera_id="airlock_camera", faction="halcyon"),
        # Greenhouse cameras, all on the biotech network: hijack one and you can hop between
        # them with the arrow keys, and each is within quickhack range of the next, so a
        # netrunner in the spore lock can hijack a chain of feeds all the way to the exit
        # before setting foot in the maze. Mounted above the hedges; each watches a corridor.
        # 1. Over the spore-lock entrance, looking north up the first corridor cells.
        CameraData(0.0, 3.4, 47.6, mount_yaw=0.0, sweep_angle=20.0, sweep_speed=0.9, camera_id="greenhouse_entry_camera", faction="halcyon_bio"),
        # 2. The central spine (x=0, z 55..70): mounted at its north end looking south.
        CameraData(0.0, 3.4, 69.4, mount_yaw=180.0, sweep_angle=18.0, sweep_speed=0.8, camera_id="greenhouse_spine_camera", faction="halcyon_bio"),
        # 3. Inside the east control node, watching the gate terminal's desk.
        CameraData(7.9, 3.3, 58.6, mount_yaw=140.0, sweep_angle=30.0, sweep_speed=1.0, camera_id="east_node_camera", faction="halcyon_bio"),
        # 4. Over the street exit, looking south along the top corridor.
        CameraData(0.0, 3.4, 75.2, mount_yaw=180.0, sweep_angle=40.0, sweep_speed=1.1, camera_id="exit_camera", faction="halcyon_bio"),
    )
VENDING_MACHINES = (VendingMachineData(-7.5, 0.0, 25.0, machine_id="neon_cold_vendor", faction="halcyon"),)
GUARDS = (
        # Sector 1 patrol. Starts near the top of its lane walking *away* from the spawn so
        # the player is not in its line of sight in the first seconds of the mission.
        GuardData(0.0, 0.0, -3.5, patrol_z_min=-8.5, patrol_z_max=-3.0, wall_pos=(-5.5, -6.0), initial_direction=-1.0, guard_id="staging_guard", faction="halcyon"),
        # Sector 2 datacore walkway patrol, leaning on the right wall facing -X. An elite
        # with hardened cyberware: it can be blinded, never knocked out.
        GuardData(2.1, 0.0, 26.0, patrol_z_min=24.0, patrol_z_max=36.0, wall_pos=(9.45, 30.0), wall_yaw=-90.0, guard_id="vault_guard", faction="halcyon", essential=True),
        # Greenhouse patrols walk waypoint loops through the maze (A* between points).
        # "Spine" guard: the central north-south corridor and the loop west of it.
        GuardData(gh_cell_center(0, -2)[0], 0.0, gh_cell_center(0, -2)[1], patrol_z_min=0.0, patrol_z_max=0.0, wall_pos=None, guard_id="greenhouse_guard", faction="halcyon_bio",
                  patrol_route=(gh_cell_center(0, -2), gh_cell_center(0, 3), gh_cell_center(-1, 1), gh_cell_center(-2, 0), gh_cell_center(-1, -2))),
        # "Gate" guard: an elite covering the east side between the gate and the exit row.
        GuardData(gh_cell_center(4, 1)[0], 0.0, gh_cell_center(4, 1)[1], patrol_z_min=0.0, patrol_z_max=0.0, wall_pos=None, guard_id="greenhouse_elite", faction="halcyon_bio", essential=True,
                  patrol_route=(gh_cell_center(4, 1), gh_cell_center(3, 3), gh_cell_center(1, 4), gh_cell_center(-2, 4), gh_cell_center(1, 3))),
    )


# ========================================================================================
# Set dressing: years of neglect. Trash, fungus up the drains, dead fixtures, the small
# wreckage of a place nobody cleans. Generated deterministically by levels.dressing and
# placed around every collider, desk, door and patrol lane - it is all walk-over clutter,
# so it never blocks a route, and tests/test_level_data.py proves none of it clips.
# ========================================================================================

DEAD_CAGE = Material("metal", (0.12, 0.12, 0.14))
GRAFFITI = Material(None, (0.35, 0.05, 0.25), emission=(0.7, 0.05, 0.45))
SIGN = Material("hazard", (0.85, 0.85, 0.85))
TRASH_BAG = Material("leather", (0.3, 0.3, 0.34), specular=(0.4, 0.4, 0.45), shininess=30.0)
CLUTTER = frozenset({"clutter", "transparent"})
DECAL = frozenset({"transparent"})


def _entity_blockers() -> list[AABB]:
    from cyberduke.entities.terminal import Terminal

    boxes: list[AABB] = []
    for t in TERMINALS:
        term = Terminal(t.x, t.y, t.z, yaw=t.yaw)
        boxes.append(term.get_bounds())
        fx, fz = term.front  # keep the chair side clear for the player to stand at
        boxes.append(AABB(fx - 0.7, fx + 0.7, fz - 0.7, fz + 0.7, 0.0, 2.0))
    for v in VENDING_MACHINES:
        boxes.append(AABB(v.x - 0.7, v.x + 0.7, v.z - 0.6, v.z + 2.0, 0.0, 2.3))
    for d in (BLAST_DOOR, VAULT_BULKHEAD, GREENHOUSE_DOOR, MAZE_GATE, STREET_EXIT):
        o = d.obstacle
        boxes.append(AABB(o.min_x - 0.3, o.max_x + 0.3, o.min_z - 1.2, o.max_z + 1.2, 0.0, 3.0))
    for g in GUARDS:  # patrol lanes and the spots guards stand on
        if g.patrol_route:
            for px, pz in g.patrol_route:
                boxes.append(AABB(px - 0.5, px + 0.5, pz - 0.5, pz + 0.5, 0.0, 2.0))
        else:
            boxes.append(AABB(g.x - 0.6, g.x + 0.6, g.patrol_z_min - 0.5, g.patrol_z_max + 0.5, 0.0, 2.0))
        if g.wall_pos is not None:
            boxes.append(AABB(g.wall_pos[0] - 0.5, g.wall_pos[0] + 0.5, g.wall_pos[1] - 0.5, g.wall_pos[1] + 0.5, 0.0, 2.0))
    return boxes


def _all_colliders() -> list[AABB]:
    out: list[AABB] = []
    for sector in (STAGING_BAY, AIRLOCK_CORRIDOR, DATACORE_VAULT, GREENHOUSE):
        out.extend(sector.colliders)
        out.extend(sector.nav_blockers)
    return out


def _dead_cage_light(x: float, y: float, z: float, ceiling_y: float, yaw: float) -> tuple[Prop, ...]:
    """A hanging cage fixture that has not worked in years: one cable snapped, the frame
    hanging crooked, the tube dark."""
    return (
        Prop.box(x, (y + ceiling_y) / 2.0, z - 0.85, 0.02, ceiling_y - y, 0.02, DEAD_CAGE, label="Dead cage light cable"),
        Prop.box(x + 0.25, y + 0.75, z + 0.6, 0.02, 0.7, 0.02, DEAD_CAGE, rotate=((35.0, 1.0, 0.0, 0.0),), label="Dead cage light snapped cable"),
        Prop.box(x, y + 0.08, z, 0.28, 0.1, 1.9, DEAD_CAGE, rotate=((yaw, 0.0, 1.0, 0.0), (-14.0, 1.0, 0.0, 0.0)), label="Dead cage light frame, hanging crooked", tags=DECAL),
        Prop.box(x, y, z, 0.16, 0.08, 1.76, Material(None, (0.1, 0.12, 0.14)), rotate=((yaw, 0.0, 1.0, 0.0), (-14.0, 1.0, 0.0, 0.0)), label="Dead cage light tube", tags=DECAL),
    )


def _dress_staging(placer: Placer) -> tuple[Prop, ...]:
    props: list[Prop] = []
    # Trash where people actually drop it: along the walls, under the server bank, by the crates
    props.extend(scatter_trash(placer, AABB(-5.5, -2.0, -9.4, 9.0), 9, ("can", "paper", "cable", "chip", "bottle", "rag")))
    props.extend(scatter_trash(placer, AABB(2.0, 5.5, -9.4, 9.0), 9, ("can", "paper", "box", "chip", "puddle", "monitor")))
    props.extend(scatter_trash(placer, AABB(-2.0, 2.0, 6.5, 9.4), 3, ("can", "paper")))
    # Flair: graffiti on the back wall, a dead cage light, a stack of boxes nobody moved
    for k, (gx, gy, ang, length) in enumerate(((-4.3, 1.7, 70.0, 0.9), (-3.9, 1.75, -60.0, 0.8), (-3.3, 1.6, 15.0, 0.7), (-2.7, 1.8, 80.0, 0.9), (-2.2, 1.55, -20.0, 0.6), (-3.4, 1.1, 4.0, 2.2))):
        # (strokes cross each other: never two at one depth)
        props.append(Prop.box(gx, gy, -9.93 + 0.002 * (k % 3), length, 0.09, 0.02, GRAFFITI if k % 2 == 0 else Material(None, (0.05, 0.3, 0.35), emission=(0.1, 0.5, 0.7)), rotate=((ang, 0.0, 0.0, 1.0),), label="Graffiti stroke on the back wall", tags=DECAL))
    props.extend(_dead_cage_light(-3.0, 3.7, 4.5, STAGING_CEILING_Y, 12.0))
    stack = ((-5.0, 0.2, 14.0, 0.36), (-4.85, 0.35, -22.0, 0.3), (-5.1, 0.0, 40.0, 0.2))
    base = 0.0
    for bx, bz, yaw, h in stack:
        props.append(Prop.box(bx, base + h / 2.0, bz, 0.5, h, 0.4, Material("cardboard", (0.8, 0.72, 0.6)), rotate=((yaw, 0.0, 1.0, 0.0),), label="Stacked cardboard boxes, askew", tags=CLUTTER))
        base += h
    return tuple(props)


def _dress_airlock(placer: Placer) -> tuple[Prop, ...]:
    props: list[Prop] = []
    props.extend(scatter_trash(placer, AABB(-1.9, 1.9, 10.6, 17.6), 6, ("can", "paper", "cable", "rag", "chip")))
    # the left pipe cluster has been dripping for years
    props.append(Prop.disc(-1.5, 0.003, 13.5, 0.42, 0.3, 0.4, Material(None, (0.12, 0.2, 0.3), alpha=0.5), slices=18, label="Puddle under the leaking pipe", tags=CLUTTER))
    # a hazard sign that has come loose and hangs by one corner
    props.append(Prop.box(2.42, 2.2, 16.2, 0.02, 0.4, 0.6, SIGN, rotate=((22.0, 1.0, 0.0, 0.0),), label="Loose hazard sign hanging crooked", tags=DECAL))
    return tuple(props)


def _dress_vault(placer: Placer) -> tuple[Prop, ...]:
    props: list[Prop] = []
    props.extend(scatter_trash(placer, AABB(-9.5, -6.3, 18.8, 41.6), 8, ("can", "paper", "box", "cable", "bottle", "rag")))
    props.extend(scatter_trash(placer, AABB(6.3, 9.5, 18.8, 41.6), 8, ("can", "paper", "cable", "chip", "monitor", "bottle")))
    props.extend(scatter_trash(placer, AABB(-2.4, 2.4, 19.0, 23.0), 3, ("paper", "chip", "can")))
    # coolant stains beside the trenches
    for x, z in ((-4.6, 23.0), (4.7, 38.5)):
        props.append(Prop.disc(x, 0.003, z, 0.6, 0.4, 0.35, Material(None, (0.0, 0.8, 0.9), alpha=0.45), slices=18, rotate=((30.0, 0.0, 1.0, 0.0),), label="Coolant stain", tags=CLUTTER))
    # a DANGER sign that fell off the monolith, server blades pulled and left on the floor
    props.append(Prop.box(-2.0, 0.015, 29.2, 0.7, 0.03, 0.5, SIGN, rotate=((-35.0, 0.0, 1.0, 0.0),), label="Fallen DANGER sign", tags=CLUTTER))
    for k in range(3):
        props.append(Prop.box(-2.3 + 0.05 * k, 0.02 + 0.025 * k, 34.4 - 0.04 * k, 0.5, 0.025, 0.9, Material("server_rack"), rotate=((8.0 * k - 5.0, 0.0, 1.0, 0.0),), label="Pulled server blade left on the floor", tags=CLUTTER))
    return tuple(props)


def _dress_greenhouse(placer: Placer) -> tuple[Prop, ...]:
    props: list[Prop] = []
    # soil spilled off the beds, a few things dropped by whoever tends the place
    for x, z in ((-12.0, 52.2), (12.4, 66.0), (-7.0, 73.5), (8.6, 50.2)):
        props.append(Prop.cylinder(x, 0.002, z, 0.55, 0.2, 0.09, SOIL, slices=10, label="Spilled soil", tags=CLUTTER))
    for zone in (AABB(-12.6, -7.4, 59.8, 64.8), AABB(7.4, 12.6, 54.4, 59.2)):  # the control nodes
        props.extend(scatter_trash(placer, zone, 4, ("paper", "can", "cable", "bottle")))
    props.extend(scatter_trash(placer, AABB(-1.3, 1.3, 42.4, 46.0), 3, ("paper", "rag", "can")))
    # the street vestibule past the exit: bags of rubbish, a bottle, the fungus at the wall
    for x, z, yaw, h in ((-1.25, 77.05, 20.0, 0.5), (-1.2, 77.6, -30.0, 0.4), (1.27, 77.1, 70.0, 0.45)):
        props.append(Prop.box(x, h / 2.0, z, 0.5, h, 0.45, TRASH_BAG, rotate=((yaw, 0.0, 1.0, 0.0),), label="Trash bag", tags=CLUTTER))
    props.extend(scatter_trash(placer, AABB(-1.3, 1.3, 76.0, 78.2), 4, ("paper", "bottle", "can")))
    return tuple(props)


def _dress_all() -> dict[str, tuple[Prop, ...]]:
    blockers = _all_colliders() + _entity_blockers()
    placer = Placer(seed=4242, blockers=blockers)
    return {
        "StagingBay": _dress_staging(placer),
        "AirlockCorridor": _dress_airlock(placer),
        "DatacoreVault": _dress_vault(placer),
        "FungalGreenhouse": _dress_greenhouse(placer),
    }


_DRESSING = _dress_all()


def _dressed(sector: SectorData) -> SectorData:
    return replace(sector, props=sector.props + tuple(_DRESSING.get(sector.name, ())))


FACILITY = LevelData(
    name="Corporate Facility",
    brief="Steal the data. Hack and sneak your way out to the street.",
    sectors=tuple(_dressed(s) for s in (STAGING_BAY, AIRLOCK_CORRIDOR, DATACORE_VAULT, GREENHOUSE)),
    doors=(BLAST_DOOR, VAULT_BULKHEAD, GREENHOUSE_DOOR, MAZE_GATE, STREET_EXIT),
    catwalk=CATWALK,
    nav_bounds=(-17.0, 17.0, -11.0, 81.0),
    spawn=(0.0, 6.0, -90.0),
    factions=(HALCYON, HALCYON_BIO),
    exit_zone=EXIT_ZONE,
    exit_to="night_market",
    terminals=TERMINALS,
    cameras=CAMERAS,
    vending_machines=VENDING_MACHINES,
    guards=GUARDS,
)


def build_facility() -> LevelData:
    return FACILITY
