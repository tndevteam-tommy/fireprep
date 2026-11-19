"""
Firestarter - scene staging helpers for playback scripts and visual tests.

These build physically valid fire lays directly (bypassing the player's hands) so that beauty
shots and long-burn scenarios can be composed quickly.  Gameplay tests use real input instead.
"""

import math
import numpy as np

from firestarter.simulation.inventory import make_tinder, make_kindling, make_firewood, make_log
from firestarter.thermodynamics.combustion_body import orientation_to_axis
from firestarter.thermodynamics.wood_properties import WoodSpeciesType as W


def _lean(world, item, angle_deg, base_r, tilt):
    pit = world.fire_pit_pos
    th = math.radians(angle_deg)
    from firestarter.simulation import terrain
    bx, bz = pit[0] + base_r * math.cos(th), pit[2] + base_r * math.sin(th)
    bottom = np.array([bx, float(terrain.height(bx, bz)) + item.radius, bz])
    o = np.array([tilt, angle_deg + 180.0, 0.0])
    body = item.to_body(bottom + orientation_to_axis(o) * item.length / 2.0, o)
    world.thermal_grid.add_body(body)
    return body


def build_teepee(world, n_kindling=6, n_firewood=3, moisture=0.1, rng=None):
    rng = rng or np.random.default_rng(3)
    pit = world.fire_pit_pos
    world.thermal_grid.add_body(make_tinder("cedar_shavings", 0.07).to_body(pit + np.array([0, 0.045, 0]), np.zeros(3)))
    for i in range(n_kindling):
        _lean(world, make_kindling(W.WHITE_PINE, moisture, rng=rng), i * 360.0 / n_kindling + 10, 0.12, 62.0)
    for i in range(n_firewood):
        _lean(world, make_firewood(W.WHITE_PINE, moisture, moisture + 0.03, rng=rng), i * 360.0 / max(1, n_firewood) + 50, 0.2, 56.0)


def light(world, seconds=2.5, dt=1.0 / 30.0):
    pit = world.fire_pit_pos
    t = 0.0
    while t < seconds:
        world.thermal_grid.add_pilot(pit + np.array([0.0, 0.03, 0.04]), 0.1)
        world.update(dt)
        t += dt


def build_log_fire(world, oak_logs=3, lit=True, rng=None):
    """A mature teepee of split pine with oak logs leaned on it - a proper evening campfire."""
    rng = rng or np.random.default_rng(5)
    build_teepee(world, 6, 4, rng=rng)
    if lit:
        light(world)
    for i in range(oak_logs):
        _lean(world, make_log(W.WHITE_OAK, 0.14, 0.17, rng=rng), i * 360.0 / oak_logs + 90, 0.26, 38.0)


def build_choked(world, n=6, rng=None):
    """Split firewood packed edge-to-edge over a tinder bundle (gaps < 2 cm): a choked layout."""
    rng = rng or np.random.default_rng(9)
    from firestarter.simulation import terrain
    pit = world.fire_pit_pos
    world.thermal_grid.add_body(make_tinder("cedar_shavings", 0.07).to_body(pit + np.array([0, 0.045, 0]), np.zeros(3)))
    for layer in range(2):
        for i in range(n // 2):
            it = make_firewood(W.WHITE_PINE, 0.1, 0.13, rng=rng)
            it.radius = 0.032
            x = (i - (n // 2 - 1) / 2.0) * 2 * it.radius * 1.0
            y = float(terrain.height(pit[0] + x, pit[2])) + 0.09 + it.radius + layer * 2 * it.radius
            body = it.to_body(np.array([pit[0] + x, y, pit[2]]), np.array([0.0, 90.0, 0.0]))
            world.thermal_grid.add_body(body)


def add_green_log(world, rng=None):
    rng = rng or np.random.default_rng(13)
    return _lean(world, make_log(W.WHITE_OAK, 0.80, 1.10, rng=rng), 200.0, 0.24, 38.0)


def stock_for_log_cabin(world, rng=None):
    rng = rng or np.random.default_rng(17)
    inv = world.inventory
    inv.add_fuel(make_tinder("cedar_shavings", 0.07))
    for _ in range(7):          # 4 for the inner teepee + 3 for the roof
        inv.add_fuel(make_kindling(W.WHITE_PINE, 0.1, rng=rng))
    for _ in range(8):
        it = make_firewood(W.PAPER_BIRCH, 0.12, 0.14, rng=rng)
        it.length, it.radius = 0.40, 0.03
        inv.add_fuel(it)
