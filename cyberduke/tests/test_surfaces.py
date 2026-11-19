"""Stairs / catwalk elevation is a pure function of position and tracking state."""

import pytest

from cyberduke.world.surfaces import Catwalk, ElevationState, floor_height

CW = Catwalk()
GROUND = ElevationState()
UP = ElevationState(on_catwalk=True)


def test_ground_level_everywhere_outside_the_catwalk_footprint():
    for x, z in [(0.0, 0.0), (0.0, 30.0), (-3.0, 20.0), (7.75, 10.0), (7.75, 20.9)]:
        y, state = floor_height(x, z, GROUND, CW)
        assert y == 0.0 and state == GROUND


def test_stairs_are_a_step_function_rising_to_the_deck():
    heights = [floor_height(7.75, z, GROUND, CW)[0] for z in (21.05, 21.45, 22.25, 23.05, 24.15)]
    assert heights[0] == pytest.approx(CW.step_rise)
    assert heights == sorted(heights)
    assert heights[-1] == pytest.approx(CW.height)
    assert len({round(h, 6) for h in heights}) == len(heights), "each sample on a different step"


def test_first_step_is_one_rise_and_steps_are_uniform():
    tops = sorted(
        {
            floor_height(7.75, CW.stairs_z_start + (i + 0.5) * CW.step_run, GROUND, CW)[0]
            for i in range(CW.steps)
        }
    )
    assert tops[0] == pytest.approx(CW.step_rise)
    diffs = [b - a for a, b in zip(tops, tops[1:], strict=False)]
    assert all(d == pytest.approx(CW.step_rise) for d in diffs)


def test_reaching_the_top_commits_to_the_catwalk_and_deck_is_flat():
    y, state = floor_height(7.75, 24.15, GROUND, CW)
    assert state.on_catwalk and y == pytest.approx(CW.height)
    y, state = floor_height(7.75, 30.0, state, CW)
    assert y == CW.height and state.on_catwalk


def test_skybridge_is_part_of_the_deck():
    y, _ = floor_height(0.0, 40.5, UP, CW)
    assert y == CW.height


def test_ground_under_the_deck_is_still_ground():
    y, state = floor_height(7.75, 30.0, GROUND, CW)
    assert y == 0.0 and not state.on_catwalk


def test_descending_the_stairs_releases_the_catwalk_flag_below_the_commit_point():
    y, state = floor_height(7.75, 24.15, UP, CW)
    assert state.on_catwalk
    y, state = floor_height(7.75, 22.0, state, CW)
    assert not state.on_catwalk and 0.0 < y < CW.height
    y, state = floor_height(7.75, 20.9, state, CW)
    assert y == 0.0 and state == GROUND


def test_both_staircases_mirror():
    for z in (21.2, 22.6, 24.1):
        assert floor_height(7.75, z, GROUND, CW) == floor_height(-7.75, z, GROUND, CW)


def test_function_is_pure():
    state = ElevationState(on_catwalk=True)
    floor_height(7.75, 18.9, state, CW)
    assert state == ElevationState(on_catwalk=True)


# ----------------------------------------------------------------------------- flights, decks
def test_a_flight_of_stairs_leads_up_onto_a_deck_people_also_walk_under():
    """A straight flight (``Flight``) climbs a tread at a time to a raised floor (a deck):
    near its top the climber is up on the deck and walks on along it; stepping off the
    deck's far side is stepping off the world's edge (the level stops that) - and anybody
    who never climbed walks about under the deck on the ground."""
    from cyberduke.core.geometry import AABB
    from cyberduke.world.surfaces import Flight, on_flights_and_decks

    flight = Flight(0.0, 4.0, 0.0, 1.0, 0.0, 2.0, 10, "+x")
    deck = AABB(4.0, 8.0, -2.0, 3.0, 1.8, 2.0)
    heights = [flight.height(x, 0.5) for x in (0.1, 1.1, 2.1, 3.1, 3.95)]
    assert heights[0] == pytest.approx(0.2) and heights[-1] == pytest.approx(2.0)
    assert heights == sorted(heights)
    y, state = on_flights_and_decks(3.95, 0.5, 0.0, GROUND, (flight,), (deck,))
    assert y == pytest.approx(2.0) and state.deck_y == pytest.approx(2.0), "up at the top"
    y, state = on_flights_and_decks(6.0, 0.5, 0.0, state, (flight,), (deck,))
    assert y == pytest.approx(2.0) and state.deck_y == pytest.approx(2.0), "along the deck"
    y, below = on_flights_and_decks(6.0, 0.5, 0.0, GROUND, (flight,), (deck,))
    assert y == 0.0 and below.deck_y is None, "underneath it, on the ground"
    y, state = on_flights_and_decks(2.0, 0.5, 0.0, state, (flight,), (deck,))
    assert 0.0 < y < 2.0 and state.deck_y is None, "back down the stair"
    # a flight that climbs the other way
    down = Flight(0.0, 4.0, 0.0, 1.0, 0.0, 2.0, 10, "-x")
    assert down.height(0.05, 0.5) == pytest.approx(2.0)
    assert down.height(3.95, 0.5) == pytest.approx(0.2)
