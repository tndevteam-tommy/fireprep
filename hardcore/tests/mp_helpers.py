"""A small multiplayer world (a few buildings and open ground) shared by the bot / server tests."""
import time

from harness import *  # noqa
from world.building import BuildingGenerator, BuildingSpec, SOUTH, NORTH, EAST, WEST
from game.match import World

_WORLD = {}


def small_world():
    """Four buildings of 2-4 storeys with cat-leap alleys, an awning, a balcony - cached."""
    if "w" not in _WORLD:
        specs = [BuildingSpec(-20.0, -40.0, 12.0, 10.0, 2, facade_faces=(SOUTH,), awning=True, name="bA"),
                 BuildingSpec(-5.0, -40.0, 12.0, 10.0, 3, facade_faces=(SOUTH,), name="bB", rooftop=False),
                 BuildingSpec(10.0, -40.0, 10.0, 10.0, 2, facade_faces=(SOUTH, EAST), name="bC",
                              balconies=((WEST, 1, 2.0),)),
                 BuildingSpec(-5.0, -60.0, 12.0, 14.0, 4, facade_faces=(NORTH,), name="bD", rooftop=False)]
        parts, buildings = [], []
        for s in specs:
            ps = BuildingGenerator.build(s)
            parts += ps
            buildings.append((s, ps))
        floor = Obstacle(-80.0, -0.5, -80.0, 160.0, 0.5, 160.0, add_crevasses=False)
        floor.rails.clear()
        obs = [floor] + parts
        ls = LedgeSystem()
        ls.build(obs)
        _WORLD["w"] = World(obs, ls, buildings, spawn_points=[(0.0, 0.9, -15.0), (5.0, 0.9, -15.0),
                                                               (-5.0, 0.9, -15.0), (10.0, 0.9, -15.0)])
    return _WORLD["w"]


def wait_for(pred, timeout=5.0, step=0.01):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(step)
    return pred()
