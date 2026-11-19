"""Small builders shared by the physics tests."""

import math

import numpy as np

from firestarter.thermodynamics.combustion_body import CombustionBody, DEFAULT_DIMENSIONS, orientation_to_axis
from firestarter.thermodynamics.wood_properties import WoodSpeciesType as W
from firestarter.thermodynamics.thermal_grid import ThermodynamicGrid


def tinder(m=0.08, sp=W.RED_CEDAR):
    return CombustionBody(sp, geometry_type="tinder", initial_m_surface=m, initial_m_core=m, position=np.array([0, 0.045, 0]))


def lean(theta_deg, kind="kindling", sp=W.WHITE_PINE, m=(0.10, 0.12), base_r=0.13, tilt=62.0):
    L, r = DEFAULT_DIMENSIONS[kind]
    th = math.radians(theta_deg)
    bottom = np.array([base_r * math.cos(th), r, base_r * math.sin(th)])
    o = np.array([tilt, theta_deg + 180.0, 0.0])
    return CombustionBody(sp, geometry_type=kind, initial_m_surface=m[0], initial_m_core=m[1],
                          position=bottom + orientation_to_axis(o) * L / 2, orientation=o)


def flat(x, z, yaw, kind="log", sp=W.WHITE_OAK, m=(0.15, 0.18), y=None):
    L, r = DEFAULT_DIMENSIONS[kind]
    return CombustionBody(sp, geometry_type=kind, initial_m_surface=m[0], initial_m_core=m[1],
                          position=np.array([x, r if y is None else y, z]), orientation=np.array([0.0, yaw, 0.0]))


def teepee(n_kindling=6, kindling_m=(0.10, 0.12)):
    return [tinder()] + [lean(a, m=kindling_m) for a in range(0, 360, 360 // n_kindling)]


def run(bodies, seconds, lighter=2.0, ambient=20.0, wind=1.0, dt=0.05, record=None, grid=None, burner=None):
    """Simulate a fire.  Returns (grid, samples) with one sample per simulated second."""
    g = grid or ThermodynamicGrid()
    for b in bodies:
        g.add_body(b)
    t = 0.0
    samples = []
    next_s = 1.0
    while t < seconds - 1e-9:
        if t < lighter:
            g.add_pilot(np.array([0.0, 0.03, 0.05]), 0.1)
        if burner is not None and t < burner[2]:
            g.add_burner(burner[0], burner[1], 0.1, diameter=0.25)
        g.update(dt, ambient, wind)
        t += dt
        if t >= next_s - 1e-9:
            samples.append({"t": round(t, 2), "hrr": g.flame_hrr, "char": g.char_hrr, "core": g.core_coal_temp,
                            "o2": g.oxygen_fraction, "Lf": g.flame_height,
                            "flaming": sum(1 for b in g.bodies if b.is_flaming), "n": len(g.bodies)})
            if record is not None:
                record(t, g)
            next_s += 1.0
    return g, samples


def first_time(samples, pred):
    for s in samples:
        if pred(s):
            return s["t"]
    return None
