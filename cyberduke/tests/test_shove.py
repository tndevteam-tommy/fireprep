"""Nobody can pin the netrunner: walking into somebody pushes them aside - somewhere they
can stand - and they go back to what they were doing (``Simulation._shove_people``,
``StreetPerson.shove``, ``alley.Storyteller``)."""

from __future__ import annotations

import math

from cyberduke.ai.alley import Glide, Play, Walk
from cyberduke.animation import street as clips
from cyberduke.levels import build_night_market
from cyberduke.simulation import InputState, Simulation

DT = 1 / 30


def _street() -> Simulation:
    sim = Simulation(build_night_market())
    sim.clock.rate = 0.0  # (the hours people keep do not come into it)
    return sim


def _directed(sim, role: str, story):
    person = next(p for p in sim.civilians + sim.merchants if p.role == role)
    person.storyteller.story = story(person)
    person.storyteller.act = None
    person.last_result = None
    return person


def _standing_at(x: float, z: float, yaw: float, walk: bool = True):
    def story(me):
        if walk:
            yield Walk(x, z, label="walking", stop=0.05)
        yield Glide(x, z, yaw=yaw, duration=0.4, label="stepping", walk_first=False)
        yield Play(lambda t: clips.idle(t, me.seed), 600.0, label="standing about")

    return story


def _walk(sim, yaw: float, seconds: float, watch=None) -> None:
    p = sim.player
    for _ in range(int(seconds / DT)):
        p.yaw = yaw
        sim.update(DT, InputState(forward=1.0))
        if watch is not None:
            watch()


def _wait(sim, seconds: float, park=(0.0, -1.2), watch=None) -> None:
    for _ in range(int(seconds / DT)):
        sim.player.x, sim.player.z = park
        sim.update(DT, InputState())
        if watch is not None:
            watch()


def _collides(sim, person) -> bool:
    return person._body_collides(person.x, person.z, sim.obstacles)


def test_walking_into_somebody_pushes_them_aside_and_they_go_back_to_it():
    sim = _street()
    spot = (-0.8, 21.0)
    person = _directed(sim, "office_worker", _standing_at(*spot, yaw=0.0))
    _wait(sim, 25.0, park=(0.0, 8.0))
    assert person.activity == "standing about" and math.dist((person.x, person.z), spot) < 0.1
    # straight up the alley through where they stand
    p = sim.player
    p.x, p.z = spot[0], spot[1] - 2.0
    moved, inside = [0.0], []

    def watch():
        moved[0] = max(moved[0], math.dist((person.x, person.z), spot))
        inside.append(_collides(sim, person))

    _walk(sim, 90.0, 3.0, watch)  # (camera yaw 90: +z)
    assert p.z > spot[1] + 1.0, "walked on past them - they were pushed aside"
    assert moved[0] > 0.3, "pushed out of the way"
    assert not any(inside), "never into a wall or a thing"
    # left alone, they go back to where they were and carry on
    _wait(sim, 10.0, park=(0.0, 30.0))
    assert math.dist((person.x, person.z), spot) < 0.15
    assert person.activity == "standing about" and person.shoved_from is None


def _stood_there(sim, x: float, z: float, yaw: float):
    person = _directed(sim, "office_worker", _standing_at(x, z, yaw=yaw, walk=False))
    person.x, person.z, person.yaw = x, z, yaw
    return person


def test_pinned_in_a_corner_the_netrunner_pushes_their_way_out():
    """Somebody steps right up against the netrunner standing in a corner - the facility
    wall behind, the tenement to one side: every way out went through them, and the
    netrunner was stuck for good."""
    sim = _street()
    corner = (-3.86, -1.66)  # (hard against the tenement's wall and the facility's)
    spot = (-3.36, -1.16)
    person = _stood_there(sim, *spot, yaw=-135.0)
    p = sim.player
    for _ in range(int(1.0 / DT)):
        p.x, p.z = corner
        sim.update(DT, InputState())
    assert person.activity == "standing about"
    assert person.body_box().overlaps_cylinder(p.x, p.z, p.radius, p.floor_y, p.floor_y + 1.8)
    inside = []
    _walk(sim, 45.0, 2.5, lambda: inside.append(_collides(sim, person)))
    assert math.dist((p.x, p.z), corner) > 1.0, "out of the corner"
    assert not any(inside)
    _wait(sim, 10.0, park=(2.0, 6.0))
    assert math.dist((person.x, person.z), spot) < 0.15, "back where they stood"
    assert person.activity == "standing about"


def test_somebody_who_cannot_be_pushed_is_still_no_trap():
    """A person who cannot be pushed about (sat down, say) in the netrunner's space is no
    trap: the netrunner cannot walk on through them, but can step away."""
    sim = _street()
    person = _stood_there(sim, 0.5, 20.0, yaw=0.0)
    p = sim.player
    for _ in range(int(1.0 / DT)):
        p.x, p.z = 0.05, 20.1
        sim.update(DT, InputState())
    person.shove = lambda *a, **k: False  # (as if they could not be moved)
    start = (p.x, p.z)
    _walk(sim, 0.0, 0.5)  # (camera yaw 0: +x, into them)
    assert p.x <= start[0] + 1e-6, "not through them"
    _walk(sim, 180.0, 1.0)  # away from them
    assert p.x < start[0] - 0.5, "but free to step away"


def test_a_walk_goes_on_after_a_push():
    sim = _street()
    person = _directed(sim, "office_worker", _standing_at(0.6, 30.0, yaw=0.0, walk=True))
    p = sim.player
    _wait(sim, 3.0, park=(0.0, 8.0))
    # in their way, walking into them as they come
    for _ in range(int(8.0 / DT)):
        if math.dist((p.x, p.z), (person.x, person.z)) < 1.2:
            break
        p.x, p.z = person.x, person.z + 2.0
        sim.update(DT, InputState())
    _walk(sim, -90.0, 1.0)  # (camera yaw -90: -z, towards them)
    _wait(sim, 30.0, park=(3.5, 2.0))
    assert math.dist((person.x, person.z), (0.6, 30.0)) < 0.15
    assert person.activity == "standing about"
