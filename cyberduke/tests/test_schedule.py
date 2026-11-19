"""The city's day: a clock (``core.clock.WorldClock``), the hours the people of the street
keep (``ai.schedule``) and what they do with them - stalls that shut and open again,
people who go home, go to bed and get up, go off to work; Nix out by his wall nearly all
the time; Dell seeing to the vending machine when it shorts out (the netrunner's chance
at his flat) - and the light of the day out of doors."""

from __future__ import annotations

import math

import pytest

from cyberduke.ai.schedule import SCHEDULES, schedule, stall_hours_open
from cyberduke.core.clock import WorldClock, hhmm, within
from cyberduke.levels import build_night_market
from cyberduke.levels.main_street import HOMES
from cyberduke.simulation import InputState, Simulation

DT = 1 / 30


def street(at: float, rate: float = 0.0) -> Simulation:
    sim = Simulation(build_night_market())
    sim.clock.set(at * 60.0)
    sim.clock.rate = rate
    sim.player.x, sim.player.z = 0.0, -1.2  # out of everybody's way, by the staff door
    return sim


def run(sim, seconds: float, until=None) -> bool:
    for _ in range(int(seconds / DT)):
        sim.player.x, sim.player.z = 0.0, -1.2
        sim.update(DT, InputState())
        if until is not None and until():
            return True
    return until is None


def inside(sim, npc) -> str | None:
    return next((a.area_id for a in sim.level.private_areas if a.contains(npc.x, npc.z)), None)


# ----------------------------------------------------------------------------- the clock
def test_the_clock_runs_wraps_and_reads_like_a_digital_clock():
    clock = WorldClock(hhmm(23, 58), rate=1.0)
    assert clock.text == "23:58"
    clock.advance(2.5)  # (at a game minute a second) two and a half game minutes
    assert clock.text == "00:00"
    assert not hasattr(clock, "day"), "the time of day only: nobody counts the days"
    assert clock.hour == pytest.approx(0.5 / 60.0)
    assert clock.is_between(22.0, 6.0) and not clock.is_between(6.0, 22.0)
    assert within(23.5, 22.0, 2.0) and within(1.0, 22.0, 2.0) and not within(3.0, 22.0, 2.0)
    assert clock.hours_until(14.0) == pytest.approx(14.0 - clock.hour)


def test_the_day_is_dawn_to_dusk():
    clock = WorldClock()
    for hour, light in ((2.0, 0.0), (12.0, 1.0), (22.0, 0.0)):
        clock.set(hour * 60.0)
        assert clock.daylight == pytest.approx(light)
    clock.set(hhmm(6, 45))
    assert 0.0 < clock.daylight < 1.0, "dawn"
    clock.set(hhmm(18, 45))
    assert 0.0 < clock.daylight < 1.0, "dusk"


def test_the_simulation_keeps_the_time_and_a_new_game_starts_at_dusk():
    from cyberduke.config import CLOCK

    sim = Simulation()
    assert sim.clock.minutes == pytest.approx(CLOCK.new_game_at)
    run(sim, 4.0)
    assert sim.clock.minutes == pytest.approx(
        CLOCK.new_game_at + 4.0 * CLOCK.minutes_per_second, abs=0.1
    )
    sim.update(DT, InputState(toggle_pause=True))
    before = sim.clock.minutes
    sim.update(1.0, InputState())
    assert sim.clock.minutes == before, "paused: time stands still"
    sim.update(DT, InputState(toggle_pause=True))
    sim.new_game()
    assert sim.clock.minutes == pytest.approx(CLOCK.new_game_at)


# ----------------------------------------------------------------------------- the hours
def test_the_hours_they_keep():
    assert stall_hours_open("sushi", 20.0) and stall_hours_open("sushi", 1.5)
    assert not stall_hours_open("sushi", 2.5) and not stall_hours_open("sushi", 13.5)
    assert stall_hours_open("sushi", 14.5), "the chef's long shift: 14:00 - 02:00"
    assert stall_hours_open("junk", 10.5) and stall_hours_open("junk", 23.5)
    assert not stall_hours_open("junk", 0.5) and not stall_hours_open("junk", 9.5)
    office = schedule("office_worker")
    assert office.working(12.0) and office.out(18.0) and office.asleep(3.0)
    assert office.indoors(12.0) and not office.at_home(12.0), "at work - in his flat"
    # the evening rush: everybody with a home is out at six (work over), most of them at nine
    out_at_6pm = [r for r in SCHEDULES if r != "resident" and schedule(r).out(18.0)]
    assert set(out_at_6pm) >= {"office_worker", "tinkerer", "courier", "stroller", "smoker"}
    # at four in the morning only Nix is about
    out_at_4am = [r for r in SCHEDULES if r != "resident" and schedule(r).out(4.0)]
    assert out_at_4am == ["smoker"]


def test_nix_is_almost_always_out_by_his_wall():
    nix = schedule("smoker")
    minutes_home = sum(1 for m in range(24 * 60) if nix.at_home(m / 60.0))
    assert 60 <= minutes_home <= 150, "home for an hour or two a day, no more"


def test_everybody_has_a_home_round_the_corner_or_across_the_alley():
    sim = street(20.0)
    life = sim.life
    for role in (
        "smoker",
        "office_worker",
        "tinkerer",
        "courier",
        "stroller",
        "sushi_chef",
        "junk_dealer",
        "resident",
    ):
        home = life.homes.get(role)
        assert home is not None, role
        for name in ("entry", "kitchen", "sofa", "by_sofa", "bath"):
            assert home.place(name) in life.places, (role, name)
        owner = sim.npc_with_role(role)
        assert owner.watch_area is not None and owner.watch_area.area_id == home.area_id
    assert HOMES["smoker"].number == "17", "Nix lives across from where he smokes"
    nix_door = life.door("door17")
    smoke = life.place("smoke")
    assert abs(nix_door.obstacle.center_z - smoke.z) < 0.6 and nix_door.obstacle.max_x < 0 < smoke.x
    # nobody's front door is itself a hack target: each is opened through the access
    # panel beside it (number 16's on Dell's network, the rest on the tenement's) - the
    # only locks hacked directly are the metro's ticket gates (METRO TRANSIT)
    from cyberduke.entities.access_panel import AccessPanel

    doors = {
        d.door_id
        for d in sim.hack_targets
        if not isinstance(d, AccessPanel) and hasattr(d, "door_id")
    }
    assert doors == {"metro_gate_1", "metro_gate_2"}
    panels = {p.door_id: p for p in sim.hack_targets if isinstance(p, AccessPanel)}
    assert set(panels) >= {f"door{plan.number}" for plan in HOMES.values()}
    assert panels["door16"].faction == "dells_home"
    assert {panels[f"door{plan.number}"].faction for plan in HOMES.values()} == {"tenement"}


def test_every_front_door_has_its_own_access_panel():
    """Every flat's door has a panel beside it (on the tenement's network of door locks):
    bypassing one releases that door's lock and no other - and only number 16's counts
    towards the job."""
    sim = street(21.0)
    for plan in HOMES.values():
        door = sim.room.doors[f"door{plan.number}"]
        panel = next(p for p in sim.panels if p.door_id == door.door_id)
        assert panel.door is door and not door.unlocked, plan.number
        box = door.obstacle
        gap = math.hypot(panel.x - box.center_x, panel.z - box.center_z)
        assert gap < 1.5, f"number {plan.number}'s panel is beside its door"
    junk = next(p for p in sim.panels if p.door_id == "door31")
    junk.bypass()
    assert sim.room.doors["door31"].unlocked
    assert not sim.room.doors["door16"].unlocked and not sim.room.doors["door33"].unlocked
    from cyberduke.systems.quests import NUMBER16

    job = sim.quests.start(NUMBER16.quest_id)
    sim.update(DT, InputState())
    assert "lock_released" not in job.flags, "the junk dealer's lock is not number 16's"
    next(p for p in sim.panels if p.door_id == "door16").bypass()
    sim.update(DT, InputState())
    assert "lock_released" in job.flags


# ----------------------------------------------------------------------------- the stalls
def test_the_chef_shuts_the_sushi_bar_at_two_and_opens_it_again_at_two():
    sim = street(1.95, rate=0.5)
    chef = sim.npc_with_role("sushi_chef")
    life = sim.life
    assert life.open["sushi"]
    assert run(sim, 60.0, until=lambda: not life.open["sushi"]), "closing time"
    assert run(sim, 30.0, until=lambda: life.item("sushi_closed") is not None)
    assert life.item("sushi_closed").name == "closed_sign", "the CLOSED sign on the counter"
    assert run(sim, 120.0, until=lambda: inside(sim, chef) == "flat36"), "home round the corner"
    # the afternoon: back to open up
    sim.clock.set(hhmm(13, 58))
    assert run(sim, 150.0, until=lambda: life.open["sushi"]), "open again"
    spot = life.place("sushi.chef")
    assert math.hypot(chef.x - spot.x, chef.z - spot.z) < 0.3 and life.item("sushi_closed") is None


def test_the_junk_dealer_covers_his_table_at_midnight():
    sim = street(23.98, rate=0.5)
    life = sim.life
    dealer = sim.npc_with_role("junk_dealer")
    assert run(sim, 90.0, until=lambda: life.item("junk_closed") is not None)
    assert life.item("junk_closed").name == "stall_cover" and not life.open["junk"]
    assert run(sim, 120.0, until=lambda: inside(sim, dealer) == "flat31")


def test_with_the_stall_shut_nobody_queues_for_sushi():
    from cyberduke.ai.alley import buy_food

    sim = street(4.0)
    life = sim.life
    life.open["sushi"] = False
    worker = sim.npc_with_role("office_worker")
    story = buy_food(worker, life, "nigiri")
    with pytest.raises(StopIteration):
        next(story)  # (nothing to do: it is over before it starts)
    assert worker not in life.counters["sushi"].queue


# ----------------------------------------------------------------------------- home, bed
def test_at_four_in_the_morning_they_are_asleep_in_their_own_beds():
    sim = street(4.0)
    # (not Nix; nor the bartender and his regular, who live across town: ``ai.nightlife``)
    sleepers = [
        n for n in sim.civilians + sim.merchants if n.role not in ("smoker", "bartender", "regular")
    ]
    assert run(sim, 150.0, until=lambda: all(getattr(n, "asleep", False) for n in sleepers))
    for n in sleepers:
        home = sim.life.homes[n.role]
        assert inside(sim, n) == home.area_id, n.role
        assert n.body is not None and n.body.fall_angle > 80.0, "lying down, in bed"
    nix = sim.npc_with_role("smoker")
    assert not nix.asleep and inside(sim, nix) is None, "Nix: out by his wall"


def test_asleep_they_do_not_notice_a_stranger_in_the_flat():
    sim = street(4.0)
    dealer = sim.npc_with_role("junk_dealer")
    assert run(sim, 150.0, until=lambda: dealer.asleep)
    bed = sim.life.place("h31.bedside")
    for _ in range(int(3.0 / DT)):
        sim.player.x, sim.player.z = bed.x, bed.z - 0.3
        sim.update(DT, InputState())
    assert sim.detection.level == 0.0 and dealer.alarm == 0.0


def test_the_office_worker_works_from_home_in_the_day():
    """He used to go off down the METRO steps and vanish until five; he works from home
    now - at the computer in his front room, where anybody could walk in on him - and is
    out for the evening rush after it."""
    sim = street(7.6)
    worker = sim.npc_with_role("office_worker")
    home = sim.life.homes["office_worker"]
    assert run(sim, 200.0, until=lambda: worker.activity == "working at the computer")
    assert inside(sim, worker) == home.area_id and not worker.hidden
    sim.clock.set(hhmm(16, 59))
    sim.clock.rate = 0.5
    assert run(sim, 200.0, until=lambda: inside(sim, worker) is None), "five o'clock: out"


# ----------------------------------------------------------------------------- Dell and the machine
def dell_home(sim):
    dell = sim.npc_with_role("resident")
    area = sim.life.homes["resident"].area_id
    assert run(
        sim, 90.0, until=lambda: inside(sim, dell) == area and "pottering" not in dell.activity
    )
    return dell, area


def test_at_home_dell_knows_nothing_of_a_short_and_the_machine_stays_broken():
    """Shorted while Dell is in his flat: he would not know (it is down the alley, he is
    indoors) - he carries on with his evening - and the machine stays out of order long
    after the sparks have died, waiting for somebody to see to it."""
    from cyberduke.entities.vending_machine import VendingMachine

    sim = street(20.5)
    dell, area = dell_home(sim)
    machine = sim.vending_machines[0]
    machine.trigger_distraction()
    assert machine.broken and machine.is_sparking
    stayed = 0.0
    for _ in range(int(60.0 / DT)):
        run(sim, DT)
        if inside(sim, dell) != area:
            break
        stayed += DT
        assert "vending machine" not in dell.activity, "he cannot know about it"
    assert stayed > VendingMachine.SPARK_DURATION + 5.0, "no hurrying out to it"
    assert machine.broken and not machine.is_sparking, "the sparks are over: still broken"


def test_out_on_an_errand_dell_sees_the_broken_machine_and_kicks_it_back_to_life():
    """When Dell next goes out, the machine is still broken: he sees it on his way, goes
    over, looks it over and kicks it until it shudders back to life - and only then gets
    on with his errand. Nobody else sees to it. The netrunner's flat is empty longer."""
    sim = street(20.5)
    dell, area = dell_home(sim)
    machine = sim.vending_machines[0]
    machine.trigger_distraction()
    assert run(sim, 150.0, until=lambda: dell.activity == "down the step"), "off on an errand"
    left = sim.time
    assert not sim.room.doors["door16"].unlocked, "locked behind him"
    assert run(sim, 40.0, until=lambda: dell.activity == "looking at the vending machine")
    assert machine.broken, "still broken when he gets there"
    assert run(sim, 30.0, until=lambda: dell.activity == "kicking the vending machine")
    assert run(sim, 10.0, until=lambda: not machine.broken), "kicked back to life"
    assert dell.activity == "kicking the vending machine"
    assert run(sim, 15.0, until=lambda: "vending machine" not in dell.activity)
    assert inside(sim, dell) is None, "...and on with his errand, not straight home"
    assert run(sim, 150.0, until=lambda: inside(sim, dell) == area), "and home in the end"
    assert sim.time - left > 45.0, "a good while out of his flat"
    others = [n for n in sim.civilians if n.role != "resident"]
    assert not any("vending machine" in n.activity for n in others)


def test_already_out_dell_turns_aside_the_moment_the_machine_goes():
    """Out of doors, on his way to the sushi bar, when the machine shorts: he turns aside
    to see to it there and then, and picks his errand up again afterwards."""
    sim = street(20.5)
    dell, _ = dell_home(sim)
    assert run(sim, 150.0, until=lambda: dell.activity == "walking to the sushi bar")
    sim.vending_machines[0].trigger_distraction()
    assert run(sim, 1.0, until=lambda: "vending machine" in dell.activity)
    assert dell.storyteller.in_aside
    assert run(sim, 60.0, until=lambda: not sim.vending_machines[0].broken)
    assert run(sim, 30.0, until=lambda: not dell.storyteller.in_aside)
    assert run(sim, 30.0, until=lambda: "sushi" in dell.activity or "counter" in dell.activity)


def test_nobody_else_fixes_the_machine_they_thump_it_and_go_without():
    """The office worker after a can finds the machine out of order: a press of the button,
    a look, a thump - and off without a can. Kicking it back to life is Dell's business."""
    from cyberduke.ai import alley
    from cyberduke.animation import street as clips

    sim = street(20.5)
    worker = sim.npc_with_role("office_worker")
    machine = sim.vending_machines[0]
    machine.trigger_distraction()

    def a_can(me):
        yield from alley.vending_run(me, sim.life)
        while True:
            yield alley.Wait(lambda: False, lambda t: clips.idle(t, me.seed), label="done")

    worker.storyteller.story, worker.storyteller.act = a_can(worker), None
    worker.last_result = None  # (a fresh story takes no result)
    thumped = False
    for _ in range(int(90.0 / DT)):
        run(sim, DT)
        thumped |= worker.activity == "thumping the dead vending machine"
        if worker.activity == "done":
            break
    assert thumped and worker.activity == "done"
    assert worker.hands["r"] is None, "no can"
    assert machine.broken, "still out of order"


def test_with_the_stalls_shut_dell_gets_his_supper_from_the_machine():
    sim = street(3.0)  # (awake - for this - and the sushi bar is shut)
    sim.life.open["sushi"] = False
    dell = sim.npc_with_role("resident")
    from cyberduke.ai import schedule as sched

    awake = sched.Schedule(sleep=None)
    original = sched.SCHEDULES["resident"]
    sched.SCHEDULES["resident"] = awake
    try:
        assert run(sim, 240.0, until=lambda: dell.activity == "taking the can out of the slot")
        assert run(sim, 5.0, until=lambda: dell.hands["r"] == "soda")
    finally:
        sched.SCHEDULES["resident"] = original


# ----------------------------------------------------------------------------- the light of day
def test_out_of_doors_the_day_is_a_little_brighter_and_brown_indoors_no_different():
    from cyberduke.render.scene import time_of_day_light

    base = (0.25, 0.31, 0.43)
    night, night_sky, lamps_night = time_of_day_light(base, 0.0, True)
    noon, noon_sky, lamps_noon = time_of_day_light(base, 1.0, True)
    assert night == base and lamps_night == 1.0
    assert all(n > b for n, b in zip(noon, base, strict=True)), "brighter by day"
    assert noon[0] - base[0] > noon[2] - base[2], "warm: golden-brown, not blue"
    # (the ambient term is a minor part of a street lit by lamps and neon: a lift of
    # about 60% in it measured about 15-35% in whole frames - a fifth brighter, give or take)
    assert sum(noon) / sum(base) < 1.7, "the smog keeps it dim"
    assert noon_sky[0] > noon_sky[2] and noon_sky != night_sky
    assert 0.3 < lamps_noon < 1.0, "street lamps dimmed by day"
    indoors = time_of_day_light(base, 1.0, False)
    assert indoors[0] == base and indoors[2] == 1.0, "the facility has no windows"


def test_dells_kick_lands_on_the_front_of_the_machine_not_through_it():
    """The boot meets the machine's front on each strike: within a few centimetres of
    it, never more than a little way into the cabinet."""
    import numpy as np

    from cyberduke.animation.rig import BodyKit
    from cyberduke.animation.skeleton import body_boxes, box_points

    kit = BodyKit(armor=False, visor=False, core_glow=False, holster=False, blaster=False)
    sim = street(20.5)
    dell, _ = dell_home(sim)
    machine = sim.vending_machines[0]
    face = machine.get_bounds().min_x  # (it stands against the east wall, facing -x)
    assert run(sim, 150.0, until=lambda: dell.activity == "walking to the sushi bar")
    machine.trigger_distraction()
    assert run(sim, 60.0, until=lambda: dell.activity == "kicking the vending machine")
    deepest = -1.0
    while dell.activity == "kicking the vending machine":
        r = math.radians(dell.yaw)
        c, s_ = math.cos(r), math.sin(r)
        m = np.identity(4)
        m[:3, :3] = np.array(((c, 0.0, s_), (0.0, 1.0, 0.0), (-s_, 0.0, c))) * dell.spec.height
        m[:3, 3] = (dell.x, dell.y, dell.z)
        boot = [(n, b) for n, b in body_boxes(dell.pose, kit, m) if n in ("boot_r", "shin_r")]
        deepest = max(deepest, float(box_points(boot, spacing=0.02)[:, 0].max()) - face)
        run(sim, DT)
    assert -0.06 < deepest < 0.04, f"the boot's furthest reach is {deepest:+.3f} m from the front"
