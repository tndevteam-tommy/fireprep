import pytest

from tests.conftest import said
from unprepared.config import PerceptionSettings
from unprepared.core.dialogue import LINES
from unprepared.core.math3d import BoundingBox3D, Vector3D
from unprepared.entities import AIState, Door, Item, NPC, Player, ScheduleTask


def make_npc(is_guard=False, is_fence=False, can_trade=False, pos=(0, 0, 0), facing=270.0):
    npc = NPC("Tester", is_guard, is_fence, can_trade, Vector3D(*pos))
    npc.facing_yaw = facing           # tests place the player at the origin, west of the NPC
    return npc


def make_player(pos=(0, 1, 0), crouch=False, torch=False):
    p = Player(Vector3D(*pos))
    p.is_crouching = crouch
    p.has_lit_torch = torch
    return p


class TestBasics:
    def test_view_distance_by_role(self):
        assert make_npc(is_guard=True).view_distance == 14.0
        assert make_npc().view_distance == 8.0

    def test_home_position_is_a_copy(self):
        npc = make_npc(pos=(1, 0, 1))
        npc.position.x = 5
        assert npc.home_position == Vector3D(1, 0, 1)


    def test_greetings_depend_on_state(self):
        npc = make_npc(can_trade=True)
        assert npc.greeting() in LINES["greet_asleep"]
        npc.state = AIState.EATING
        assert npc.greeting() in LINES["greet_closed"]
        assert not npc.is_open_for_business
        npc.state = AIState.SHOPKEEPING
        assert npc.is_open_for_business and npc.greeting() in LINES["greet_merchant"]
        npc.role_tag = "smith"
        assert npc.greeting() in LINES["greet_smith"]
        plain = make_npc()
        plain.state = AIState.STROLLING
        assert plain.greeting(hour=12) in LINES["greet_townsfolk"] and plain.greeting(hour=23) in LINES["greet_townsfolk_night"]

    def test_title_is_the_name(self):
        assert NPC("Guard Captain Vance", True, False, False, Vector3D(0, 0, 0)).title == "Guard Captain Vance"
        assert NPC("Jailer Hesk", True, False, False, Vector3D(0, 0, 0)).title == "Jailer Hesk"

    def test_stock_hides_items_not_for_sale(self):
        npc = make_npc(can_trade=True)
        npc.inventory = [Item("k", "Key", for_sale=False), Item("b", "Bread")]
        assert [i.id for i in npc.stock] == ["b"]

    def test_only_keys_open_locked_doors_even_for_owners(self):
        npc = make_npc()
        locked = Door("shop", Vector3D(0, 0, 0), is_locked=True, owners=("Tester",))
        assert npc.owns(locked) and not npc.can_open(locked) and not npc.can_lock(locked)   # owner without the key
        npc.inventory.append(Item("k", "Key", is_key=True, key_door_id="shop"))
        assert npc.can_open(locked) and npc.can_lock(locked)
        assert npc.can_open(Door("x", Vector3D(0, 0, 0), is_locked=False))                 # anyone opens an unlocked door
        guard = make_npc(is_guard=True)
        assert guard.faction == "guards" and guard.owns(Door("g", Vector3D(0, 0, 0), owners=("guards",)))
        assert not npc.owns(Door("g", Vector3D(0, 0, 0), owners=("guards",)))

    def test_persistence_roundtrip(self):
        npc = make_npc()
        npc.state = AIState.REPORTING
        npc.suspicion_meter = 42.0
        npc.position = Vector3D(3, 0, 4)
        npc.has_torch = True
        npc.inventory = [Item("k", "Key", for_sale=False)]
        other = make_npc()
        other.load_dict(npc.to_dict())
        assert other.state is AIState.IDLE and other.suspicion_meter == 42.0 and other.position == Vector3D(3, 0, 4)   # busy states are not resumed
        assert other.has_torch and [i.id for i in other.inventory] == ["k"] and not other.inventory[0].for_sale
        assert other.current_task is None and other.path == []
        data = npc.to_dict()
        assert {k: data[k] for k in ("name", "position", "suspicion_meter", "state", "has_torch", "crimes_witnessed", "facing_yaw")} == \
            {"name": "Tester", "position": {"x": 3, "y": 0, "z": 4}, "suspicion_meter": 42.0, "state": 5, "has_torch": True,
             "crimes_witnessed": 0, "facing_yaw": 270.0}
        assert data["inventory"] == [npc.inventory[0].to_dict()]
        assert data["sightings"] == [] and data["known_thefts"] == [] and data["pending_confrontations"] == []
        assert data["crime_memories"] == []

    def test_load_prototype_format_without_new_fields(self):
        npc = make_npc()
        npc.inventory = [Item("b", "Bread")]
        npc.load_dict({"name": "Tester", "position": {"x": 1, "y": 0, "z": 1}, "suspicion_meter": 0.0, "state": 2})
        assert npc.state is AIState.PATROL and [i.id for i in npc.inventory] == ["b"]


class TestSchedule:
    def test_changes_state_and_announces(self):
        npc = make_npc()
        npc.schedule = [ScheduleTask(0, 8, Vector3D(0, 0, 0), AIState.SLEEPING, "zzz"),
                        ScheduleTask(8, 22, Vector3D(0, 0, 0), AIState.SHOPKEEPING, "open")]
        msgs = []
        npc.update_schedule(8, msgs.append)
        assert npc.state is AIState.SHOPKEEPING and npc.current_task is npc.schedule[1]
        assert msgs == ["open"]                  # narration goes to the log as plain text, never as a quoted line
        npc.update_schedule(9, msgs.append)      # same task: silent
        assert msgs == ["open"]

    def test_without_a_world_the_npc_adopts_the_state_in_place(self):
        npc = make_npc()
        npc.schedule = [ScheduleTask(0, 24, Vector3D(9, 0, 9), AIState.PATROL, "go")]
        npc.update_schedule(3)
        assert npc.state is AIState.PATROL and npc.position == Vector3D(0, 0, 0) and npc.path == []

    def test_suspicious_npc_ignores_schedule(self):
        npc = make_npc()
        npc.schedule = [ScheduleTask(0, 24, Vector3D(0, 0, 0), AIState.PATROL, "go")]
        npc.suspicion_meter = 50.0
        npc.update_schedule(10)
        assert npc.state is AIState.SLEEPING

    def test_calmed_npc_resumes_its_task(self):
        npc = make_npc()
        npc.schedule = [ScheduleTask(0, 24, Vector3D(0, 0, 0), AIState.PATROL, "go")]
        npc.update_schedule(1)
        npc.state = AIState.IDLE
        npc.suspicion_meter = 80.0
        npc.update_schedule(2)
        assert npc.state is AIState.IDLE               # still rattled
        npc.suspicion_meter = 10.0
        npc.update_schedule(3)
        assert npc.state is AIState.PATROL

    def test_busy_npcs_ignore_the_schedule(self):
        npc = make_npc(is_guard=True)
        npc.schedule = [ScheduleTask(0, 24, Vector3D(0, 0, 0), AIState.PATROL, "go")]
        npc.state = AIState.RESPONDING
        npc.update_schedule(5)
        assert npc.state is AIState.RESPONDING

    def test_no_task_for_hour_keeps_state(self):
        npc = make_npc()
        npc.schedule = [ScheduleTask(8, 10, Vector3D(0, 0, 0), AIState.PATROL, "go")]
        npc.update_schedule(3)
        assert npc.state is AIState.SLEEPING and npc.task_for(3) is None

    def test_schedule_no_longer_decides_torches(self):
        """Torches are lit by the world wherever it is dark (``World.is_dark_at``), not by the hour."""
        npc = make_npc()
        npc.update_schedule(22)
        assert npc.has_torch is False


class TestPerception:
    def test_not_suspicious_decays_per_second(self):
        npc = make_npc()
        npc.suspicion_meter = 60.0
        npc.evaluate_perception(make_player(), 1.0, True, True, [], 12, delta_time=1.0)
        assert npc.suspicion_meter == 35.0                     # 25 per second
        npc.evaluate_perception(make_player(), 1.0, True, True, [], 12, delta_time=2.0)
        assert npc.suspicion_meter == 0.0

    def test_walking_in_dark_without_torch_is_suspicious_when_seen(self):
        npc = make_npc(pos=(3, 0, 0))
        npc.state = AIState.PATROL
        npc.has_torch = True                                   # night: the NPC carries a torch and can see 12 units
        npc.evaluate_perception(make_player(torch=False, crouch=False, pos=(-6, 1, 0)), 0.1, True, False, [], 23, delta_time=1.0)
        assert npc.suspicion_meter == 0.0 and npc.heard_noise_at is not None   # too dark to see at 9 units, only heard
        npc.evaluate_perception(make_player(torch=True, crouch=True), 0.1, True, False, [], 23, delta_time=1.0)
        assert npc.suspicion_meter > 0                          # lit and crouching: seen skulking

    def test_walking_in_dark_with_torch_is_fine(self):
        npc = make_npc(pos=(3, 0, 0))
        npc.state = AIState.PATROL
        npc.evaluate_perception(make_player(torch=True), 0.1, True, False, [], 23)
        assert npc.suspicion_meter == 0

    def test_crouching_in_daylight_is_suspicious(self):
        npc = make_npc(pos=(3, 0, 0))
        npc.state = AIState.PATROL
        npc.evaluate_perception(make_player(crouch=True), 1.0, True, False, [], 12, delta_time=1.0)
        # per second: seen (0.4 / 3.16 * 80); hearing alone never raises suspicion
        assert npc.suspicion_meter == pytest.approx((0.4 / (3 ** 2 + 1) ** 0.5) * 80.0)
        assert npc.heard_noise_at is not None

    def test_sleeper_wakes_from_loud_noise(self):
        npc = make_npc(pos=(5, 0, 0))
        msgs = []
        npc.evaluate_perception(make_player(crouch=True), 1.0, True, True, [], 12, msgs.append)
        # crouch sprint noise = 3.0*0.3 = 0.9 > 0.8 ; hearing radius 9 >= 5
        assert npc.state is AIState.INVESTIGATING
        assert len(msgs) == 1 and msgs[0].startswith("(Tester woke up!): ") and said(msgs, "woke_up")

    def test_sleeper_ignores_quiet_noise(self):
        npc = make_npc(pos=(2, 0, 0))
        npc.evaluate_perception(make_player(crouch=True), 1.0, True, False, [], 12)
        assert npc.state is AIState.SLEEPING and npc.suspicion_meter == 0.0

    def test_wall_blocks_sight(self):
        npc = make_npc(pos=(6, 0, 0))
        npc.state = AIState.PATROL
        wall = BoundingBox3D(Vector3D(2.5, 0, -5), Vector3D(3, 5, 5))
        npc.evaluate_perception(make_player(crouch=True), 1.0, False, False, [wall], 12)
        assert npc.suspicion_meter == 0.0
        npc.evaluate_perception(make_player(crouch=True), 1.0, False, False, [], 12)
        assert npc.suspicion_meter > 0.0

    def test_guard_alarm(self):
        npc = make_npc(is_guard=True, pos=(1, 0, 0))
        npc.state = AIState.PATROL
        npc.suspicion_meter = 99.0
        msgs = []
        player = make_player(crouch=True)
        npc.evaluate_perception(player, 1.0, True, False, [], 12, msgs.append, delta_time=1.0)
        assert npc.state is AIState.PATROL                   # first he warns ...
        assert said(msgs[-1:], "skulking")
        npc.clock += npc.settings.warning_grace              # ... the grace passes ...
        npc.evaluate_perception(player, 1.0, True, False, [], 12, msgs.append, delta_time=1.0)
        assert npc.state is AIState.ALERT_GUARDS             # ... then the alarm
        assert said(msgs[-1:], "halt")
        assert npc.path == []

    def test_a_warning_buys_a_few_seconds_of_grace_and_is_not_repeated(self):
        """The first time a guard tells the player off, suspicion cannot tip into an alarm
        for ``warning_grace`` seconds, and the same line is not spammed every frame."""
        npc = make_npc(is_guard=True, pos=(1, 0, 0))
        npc.state = AIState.PATROL
        npc.suspicion_meter = 60.0
        msgs = []
        player = make_player(crouch=True)
        for _ in range(50):                                  # 5 s of skulking right in front of him
            npc.evaluate_perception(player, 1.0, True, False, [], 12, msgs.append, delta_time=0.1)
        assert sum(1 for m in msgs if said([m], "skulking")) == 1
        assert npc.state is AIState.PATROL and npc.suspicion_meter == 99.0   # capped just under the alarm
        for _ in range(30):                                  # grace (6 s) runs out
            npc.evaluate_perception(player, 1.0, True, False, [], 12, msgs.append, delta_time=0.1)
        assert npc.state is AIState.ALERT_GUARDS
        # the scolding repeats only after the cooldown
        npc.clock += npc.settings.scold_cooldown
        assert npc.scold("skulking", "again", msgs.append) is True
        assert npc.scold("skulking", "again", msgs.append) is False

    def test_alarm_needs_sight_not_just_hearing(self):
        npc = make_npc(is_guard=True, pos=(1, 0, 0), facing=90.0)     # looking away
        npc.state = AIState.PATROL
        npc.suspicion_meter = 99.0
        npc.evaluate_perception(make_player(crouch=True), 1.0, True, False, [], 12, delta_time=1.0)
        assert npc.state is AIState.PATROL and npc.heard_noise_at is not None

    def test_citizen_alarm(self):
        npc = make_npc(pos=(1, 0, 0))
        npc.state = AIState.SHOPKEEPING
        npc.suspicion_meter = 99.0
        msgs = []
        npc.evaluate_perception(make_player(crouch=True), 1.0, True, False, [], 12, msgs.append, delta_time=1.0)
        assert npc.state is AIState.REPORTING
        assert said(msgs[-1:], "cry_for_help")
        count = len(msgs)
        npc.evaluate_perception(make_player(crouch=True), 1.0, True, False, [], 12, msgs.append, delta_time=1.0)
        assert len(msgs) == count   # already alarmed: no repeat

    def test_fences_grow_suspicious_of_skulking_but_never_raise_the_alarm(self):
        npc = make_npc(is_fence=True, pos=(1, 0, 0))
        npc.state = AIState.SHOPKEEPING
        npc.evaluate_perception(make_player(crouch=True), 1.0, True, False, [], 12, delta_time=5.0)
        assert npc.suspicion_meter >= 100.0 and npc.state is AIState.SHOPKEEPING      # wary, but no cry for the watch

    def test_trespassing_is_noticed_even_standing_upright(self):
        npc = make_npc(pos=(3, 0, 0))
        npc.state = AIState.SHOPKEEPING
        msgs = []
        npc.evaluate_perception(make_player(), 1.0, False, False, [], 12, msgs.append, delta_time=1.0, trespassing=True)
        assert npc.suspicion_meter > 40.0
        assert len(msgs) == 1 and said(msgs, "trespass")

    def test_torch_law_reminder_from_guards(self):
        npc = make_npc(is_guard=True, pos=(3, 0, 0))
        npc.state = AIState.PATROL
        npc.has_torch = True
        msgs = []
        npc.evaluate_perception(make_player(torch=False), 0.1, True, False, [], 23, msgs.append, delta_time=1.0, trespassing=True)
        assert said(msgs, "torch_law")

    def test_torch_extends_view(self):
        npc = make_npc(pos=(10, 0, 0))
        npc.state = AIState.PATROL
        # stationary + crouching + lit torch: suspicious, silent, visibility (0.1+0.9)*0.4 (crouched) * 0.5 (holding still) = 0.2
        npc.evaluate_perception(make_player(torch=True, crouch=True), 0.1, False, False, [], 23)
        assert npc.suspicion_meter == 0.0    # view 8 * 0.4 = 3.2 < 10
        npc.has_torch = True
        npc.evaluate_perception(make_player(torch=True, crouch=True), 0.1, False, False, [], 23, delta_time=1.0)
        dist = (10 ** 2 + 1 ** 2) ** 0.5
        assert npc.suspicion_meter == pytest.approx(0.2 / dist * 80.0)   # view 8 * 1.5 = 12 >= 10


class TestFieldOfView:
    def test_forward_vector(self):
        npc = make_npc(facing=0.0)
        assert npc.forward == pytest.approx((0.0, -1.0))
        npc.facing_yaw = 90.0
        assert npc.forward == pytest.approx((1.0, 0.0))

    def test_faces_half_space(self):
        npc = make_npc(pos=(0, 0, 0), facing=0.0)          # looking towards -z
        assert npc.faces(Vector3D(0, 0, -5)) and npc.faces(Vector3D(4, 0, -1))
        assert not npc.faces(Vector3D(0, 0, 5)) and not npc.faces(Vector3D(-3, 0, 2))
        assert npc.faces(Vector3D(0, 0, 0))                # coincident: counts as seen

    def test_turn_towards(self):
        npc = make_npc(facing=0.0)
        npc.turn_towards(Vector3D(5, 0, 0))
        assert npc.facing_yaw == pytest.approx(90.0)
        npc.turn_towards(Vector3D(0, 0, 5))
        assert npc.facing_yaw == pytest.approx(180.0)

    def test_cannot_see_or_witness_behind(self):
        npc = make_npc(is_guard=True, pos=(3, 0, 0), facing=90.0)   # looking east, player to the west
        npc.state = AIState.PATROL
        assert not npc.can_see(Vector3D(0, 1, 0), [])
        assert npc.witness_crime("Pickpocketing", Vector3D(0, 0, 0), []) is False
        npc.evaluate_perception(make_player(crouch=True), 1.0, False, False, [], 12)
        assert npc.suspicion_meter == 0.0
        npc.turn_towards(Vector3D(0, 0, 0))
        assert npc.can_see(Vector3D(0, 1, 0), [])
        assert npc.witness_crime("Pickpocketing", Vector3D(0, 0, 0), []) is True

    def test_hearing_works_from_behind(self):
        npc = make_npc(pos=(2, 0, 0), facing=90.0)
        npc.state = AIState.PATROL
        npc.evaluate_perception(make_player(crouch=True), 1.0, True, False, [], 12, delta_time=1.0)
        assert npc.suspicion_meter == 0.0 and npc.heard_noise_at is not None   # heard (turns to look) but not seen

    def test_sneak_hidden_flag_blocks_sight_only(self):
        npc = make_npc(pos=(3, 0, 0))
        npc.state = AIState.PATROL
        npc.sneak_hidden = True
        npc.evaluate_perception(make_player(crouch=True), 1.0, False, False, [], 12, delta_time=1.0)
        assert npc.suspicion_meter == 0.0                   # seen component suppressed, nothing heard (still)


class TestGuardMemory:
    def test_sightings_and_theft_reports_lead_to_confrontations(self):
        from unprepared.entities import TheftReport
        guard = make_npc(is_guard=True)
        guard.remember_sighting("the shop", 1)
        guard.remember_sighting("the shop", 1)               # de-duplicated
        assert len(guard.sightings) == 1
        far = TheftReport("Barnaby", "the shop", 5, {"wine": 1}, {"wine": "Bottle of Wine"})   # too long ago
        guard.learn_theft(far)
        assert guard.known_thefts == [far] and guard.pending_confrontations == []
        recent = TheftReport("Barnaby", "the shop", 2, {"wine": 1}, {"wine": "Bottle of Wine"})
        guard.learn_theft(recent)
        assert guard.pending_confrontations == [recent]
        elsewhere = TheftReport("Barnaby", "Barnaby's house", 2, {"ring": 1}, {"ring": "Ring"})
        guard.learn_theft(elsewhere)
        assert guard.pending_confrontations == [recent]

    def test_memory_round_trips_through_save_data(self):
        from unprepared.entities import TheftReport
        guard = make_npc(is_guard=True)
        guard.remember_sighting("the shop", 1)
        guard.learn_theft(TheftReport("Barnaby", "the shop", 1, {"wine": 2}, {"wine": "Bottle of Wine"}))
        other = make_npc(is_guard=True)
        other.load_dict(guard.to_dict())
        assert other.sightings[0].area == "the shop" and other.known_thefts[0].items == {"wine": 2}
        assert len(other.pending_confrontations) == 1


class TestWitnessCrime:
    def test_sleeping_sees_nothing(self):
        assert make_npc().witness_crime("x", Vector3D(0, 0, 0), []) is False

    def test_guard_witness(self):
        npc = make_npc(is_guard=True, pos=(5, 0, 0))
        npc.state = AIState.PATROL
        msgs = []
        assert npc.witness_crime("Pickpocketing", Vector3D(0, 0, 0), [], msgs.append) is True
        assert npc.state is AIState.ALERT_GUARDS and npc.crimes_witnessed == 1
        assert msgs == ["(Tester witnessed Pickpocketing!): \"Caught you red-handed!\""]

    def test_guard_calls_out_every_crime_with_escalating_lines(self):
        npc = make_npc(is_guard=True, pos=(5, 0, 0))
        msgs = []
        for crime in ("Pickpocketing", "Shoplifting", "Lockpicking a door", "Pickpocketing", "Pickpocketing"):
            npc.state = AIState.PATROL                  # back on his rounds between one crime and the next
            assert npc.witness_crime(crime, Vector3D(0, 0, 0), [], msgs.append) is True
        assert npc.crimes_witnessed == 5 and npc.state is AIState.ALERT_GUARDS
        lines = [m.split(": ")[1] for m in msgs]
        assert lines == ['"Caught you red-handed!"', '"Again?! Stop right there, thief!"',
                         '"That\'s enough! One more and you\'re coming with me!"',
                         '"You\'re under arrest the moment I get my hands on you!"',
                         '"You\'re under arrest the moment I get my hands on you!"']
        assert "Shoplifting" in msgs[1]

    def test_citizen_witness(self):
        npc = make_npc(pos=(5, 0, 0))
        npc.state = AIState.SHOPKEEPING
        msgs = []
        assert npc.witness_crime("Lockpicking a door", Vector3D(0, 0, 0), [], msgs.append) is True
        assert npc.state is AIState.REPORTING
        assert len(msgs) == 1 and msgs[0].startswith("(Tester witnessed Lockpicking a door!): ") and said(msgs, "thief_guards")

    def test_too_far_or_blocked(self):
        npc = make_npc(pos=(9, 0, 0))
        npc.state = AIState.SHOPKEEPING
        assert npc.witness_crime("x", Vector3D(0, 0, 0), []) is False
        npc.position = Vector3D(5, 0, 0)
        wall = BoundingBox3D(Vector3D(2, 0, -1), Vector3D(3, 5, 1))
        assert npc.witness_crime("x", Vector3D(0, 0, 0), [wall]) is False
        assert npc.state is AIState.SHOPKEEPING


def test_custom_perception_settings():
    npc = NPC("T", True, False, False, Vector3D(0, 0, 0), settings=PerceptionSettings(guard_view_distance=3.0))
    assert npc.view_distance == 3.0
