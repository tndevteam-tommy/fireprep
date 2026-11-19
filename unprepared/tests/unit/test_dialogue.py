"""NPC speech: phrasings, scripted conversations between neighbours, speech bubbles."""
import random

import pytest

from tests.helpers import spot
from tests.conftest import bubbled, said
from unprepared.core import dialogue
from unprepared.entities import AIState
from unprepared.systems import MoveIntent


# ----------------------------------------------------------------------------- the module
class TestPhrasings:
    def test_every_kind_has_at_least_one_phrasing_and_all_format_cleanly(self):
        fmt = {"item": "X", "gold": "3", "role": "smith", "room": "room one", "Room": "Room one", "nights": 2}
        for kind, options in dialogue.LINES.items():
            assert options, kind
            for text in options:
                assert text.format(**fmt), kind

    def test_line_varies_with_the_rng_but_stays_within_the_kind(self):
        seen = {dialogue.line("skulking", random.Random(seed)) for seed in range(40)}
        assert seen <= set(dialogue.LINES["skulking"]) and len(seen) > 1

    def test_short_names(self):
        class N:
            def __init__(self, name): self.name = name
        assert dialogue.short_name(N("Shopkeeper Barnaby")) == "Barnaby"
        assert dialogue.short_name(N("Guard Captain Vance")) == "Captain"
        assert dialogue.short_name(N("Guard Orrin")) == "Orrin"
        assert dialogue.short_name(N("Jailer Hesk")) == "Hesk"
        assert dialogue.short_name(N("Jae the Fence")) == "Jae"
        assert dialogue.short_name(N("Tamsin")) == "Tamsin"

    def test_time_of_day_bands(self):
        assert [dialogue.time_of_day(h) for h in (5, 11, 12, 17, 18, 21, 22, 4)] == \
            ["morning", "morning", "day", "day", "evening", "evening", "night", "night"]


class TestBuildConversation:
    @pytest.fixture
    def people(self, game):
        w = game.world
        return w.find_npc("Tamsin"), w.find_npc("Gareth the Smith"), w.find_npc("Guard Orrin")

    def test_shape_greeting_topics_goodbye(self, people):
        tamsin, gareth, _ = people
        conv = dialogue.build_conversation(tamsin, gareth, 9, random.Random(1), exchanges=2)
        assert conv.participants == (tamsin, gareth) and not conv.finished and conv.index == 0
        assert len(conv.lines) == 2 + 2 * 2 + 2
        speakers = [s for s, _ in conv.lines]
        assert speakers[0] is tamsin and speakers[1] is gareth          # greeting and reply
        assert speakers[-2] is tamsin and speakers[-1] is gareth        # goodbye and reply
        assert conv.lines[0][1] in {g.format(other="Gareth") for g in dialogue.GREETINGS["morning"]}
        assert conv.lines[-1][1] in {g.format(other="Tamsin") for g in dialogue.FAREWELL_REPLIES}

    def test_greeting_matches_the_hour(self, people):
        tamsin, gareth, _ = people
        for hour, band in ((7, "morning"), (14, "day"), (19, "evening"), (23, "night")):
            conv = dialogue.build_conversation(tamsin, gareth, hour, random.Random(3), exchanges=1)
            assert conv.lines[0][1] in {g.format(other="Gareth") for g in dialogue.GREETINGS[band]}

    def test_a_topic_about_the_smith_is_answered_by_the_smith(self, people):
        tamsin, gareth, _ = people
        smith_openers = {o for o, _ in dialogue.TOPICS["smith"]}
        smith_replies = {r for _, r in dialogue.TOPICS["smith"]}
        found = False
        for seed in range(60):
            conv = dialogue.build_conversation(tamsin, gareth, 9, random.Random(seed), exchanges=2)
            for (s1, t1), (s2, t2) in zip(conv.lines[2:-2:2], conv.lines[3:-2:2]):
                if t1 in smith_openers:
                    assert s1 is tamsin and s2 is gareth and t2 in smith_replies
                    found = True
        assert found

    def test_guards_bring_guard_talk(self, people):
        tamsin, _, orrin = people
        openers = set()
        for seed in range(40):
            conv = dialogue.build_conversation(tamsin, orrin, 9, random.Random(seed), exchanges=2)
            openers |= {t for _, t in conv.lines[2:-2:2]}
        assert openers & {o for o, _ in dialogue.TOPICS["guard"]}

    def test_recent_openers_are_avoided(self, people):
        tamsin, gareth, _ = people
        recent = [o for topic in ("weather", "town", "gossip", "smith", "trade") for o, _ in dialogue.TOPICS[topic][1:]]
        for seed in range(30):
            conv = dialogue.build_conversation(tamsin, gareth, 9, random.Random(seed), exchanges=2, recent=recent)
            for _, text in conv.lines[2:-2:2]:
                assert text not in recent

    def test_a_conversation_never_goes_over_the_same_topic_twice(self, game):
        """Two guards bring the watch into the pool four times over (their role and their badge,
        each): it still tends to come up first, but a two-exchange chat never repeats it - the same
        opener and reply word for word used to turn up in about one such chat in eleven."""
        orrin, mira = game.world.find_npc("Guard Orrin"), game.world.find_npc("Guard Mira")
        about_the_watch = {o.format(other=name) for o, _ in dialogue.TOPICS["guard"] for name in ("Orrin", "Mira")}
        first_about_the_watch = 0
        for seed in range(300):
            conv = dialogue.build_conversation(orrin, mira, 12, random.Random(seed), exchanges=2)
            openers = [text for _, text in conv.lines[2:-2:2]]
            assert len(openers) == 2 and len(set(openers)) == 2, (seed, openers)
            assert sum(o in about_the_watch for o in openers) <= 1, (seed, openers)      # two different topics
            first_about_the_watch += openers[0] in about_the_watch
        assert first_about_the_watch > 300 * 0.4             # the weighting survives: the watch is their favourite subject

    def test_one_or_two_exchanges_by_default(self, people):
        tamsin, gareth, _ = people
        lengths = {len(dialogue.build_conversation(tamsin, gareth, 9, random.Random(s)).lines) for s in range(30)}
        assert lengths <= {6, 8} and len(lengths) == 2


class TestPlayerGreeting:
    def test_greeting_depends_on_who_you_talk_to(self, game):
        w, rng = game.world, random.Random(5)
        kinds = {
            "Jae the Fence": "greet_fence", "Guard Orrin": "greet_guard", "Gareth the Smith": "greet_smith",
            "Hilde the Innkeeper": "greet_innkeeper", "Tamsin": "greet_townsfolk",
        }
        for name, kind in kinds.items():
            npc = w.find_npc(name)
            if npc.can_trade and not npc.is_fence:
                npc.state = AIState.SHOPKEEPING
            assert dialogue.player_greeting(npc, 9, rng) in dialogue.LINES[kind], name
        mirelle = w.find_npc("Mirelle the Tailor")
        mirelle.state = AIState.SLEEPING
        assert dialogue.player_greeting(mirelle, 9, rng) in dialogue.LINES["greet_asleep"]
        mirelle.state = AIState.EATING
        assert dialogue.player_greeting(mirelle, 9, rng) in dialogue.LINES["greet_closed"]
        assert dialogue.player_greeting(w.find_npc("Pell"), 23, rng) in dialogue.LINES["greet_townsfolk_night"]


# ----------------------------------------------------------------------------- in the world
def quiet(game):
    """Nobody starts a conversation on their own."""
    for npc in game.world.npcs:
        npc.next_chat_time = 1e9


class TestConversations:
    def test_start_pauses_both_and_plays_a_line_at_a_time(self, game):
        quiet(game)
        w = game.world
        lys, dorran = w.find_npc("Lys"), w.find_npc("Dorran")
        conv = w.start_conversation(lys, dorran, exchanges=1)
        assert lys.conversation is conv and dorran.conversation is conv
        assert lys.chat_pause and dorran.chat_pause
        assert lys.faces(dorran.position) and dorran.faces(lys.position)
        n = len(game.messages.messages)
        game.update(0.5, MoveIntent())
        assert lys.speech is None and conv.index == 0                 # the first line takes a moment
        game.update(0.5, MoveIntent())
        assert lys.speech is not None and lys.speech.text == conv.lines[0][1] and conv.index == 1
        spoken = []
        for _ in range(80):                                           # ~40 s: the whole exchange
            game.update(0.5, MoveIntent())
            for npc in (lys, dorran):
                if npc.speech is not None and (npc, npc.speech.text) not in spoken:
                    spoken.append((npc, npc.speech.text))
        assert conv.finished and lys.conversation is None and dorran.conversation is None
        assert not lys.chat_pause and not dorran.chat_pause
        assert spoken[:len(conv.lines)] == list(conv.lines)           # every line went up in a bubble, in order ...
        assert not any(m.split('"')[1] in {t for _, t in conv.lines} for m in game.messages.messages[n:] if '"' in m)   # ... none logged

    def test_after_a_chat_neither_starts_another_for_a_while(self, game):
        quiet(game)
        w = game.world
        lys, dorran = w.find_npc("Lys"), w.find_npc("Dorran")
        w.start_conversation(lys, dorran, exchanges=1)
        for _ in range(80):
            game.update(0.5, MoveIntent())
        assert lys.next_chat_time > w.elapsed + 60 and dorran.next_chat_time > w.elapsed + 60

    def test_a_chat_breaks_off_when_one_of_them_has_better_things_to_do(self, game):
        quiet(game)
        w = game.world
        lys, dorran = w.find_npc("Lys"), w.find_npc("Dorran")
        conv = w.start_conversation(lys, dorran, exchanges=2)
        game.update(1.0, MoveIntent())
        dorran.state = AIState.ALERT_GUARDS                          # not sociable any more
        game.update(0.1, MoveIntent())
        assert conv.finished and lys.conversation is None and not lys.chat_pause

    def test_neighbours_who_stop_near_each_other_get_talking(self, game):
        w = game.world
        for npc in w.npcs:
            npc.next_chat_time = 1e9
        lys, dorran = w.find_npc("Lys"), w.find_npc("Dorran")
        lys.next_chat_time = dorran.next_chat_time = 0.0              # at breakfast, at the same table
        for _ in range(120):
            game.update(0.5, MoveIntent())
            if lys.conversation is not None:
                break
        assert lys.conversation is not None and lys.conversation.participants == (lys, dorran)

    def test_people_in_different_rooms_or_far_apart_do_not_talk(self, game):
        w = game.world
        for npc in w.npcs:
            npc.next_chat_time = 1e9
        lys, barnaby = w.find_npc("Lys"), w.find_npc("Shopkeeper Barnaby")
        lys.next_chat_time = barnaby.next_chat_time = 0.0             # at breakfast in different houses
        for _ in range(60):
            game.update(0.5, MoveIntent())
        assert lys.conversation is None and barnaby.conversation is None

    def test_at_most_two_conversations_and_recent_topics_are_remembered(self, game):
        quiet(game)
        w = game.world
        assert w.MAX_CONVERSATIONS == 2
        names = ("Lys", "Dorran", "Tamsin", "Pell")
        a, b, c, d = (w.find_npc(n) for n in names)
        w.start_conversation(a, b, exchanges=2)
        w.start_conversation(c, d, exchanges=2)
        assert len(w.conversations) == 2 and len(w.recent_topics) == 8
        for _ in range(4):
            game.update(0.5, MoveIntent())
        assert len(w.conversations) == 2

    def test_sleepers_and_the_alarmed_are_not_sociable(self, game):
        npc = game.world.find_npc("Tamsin")
        assert npc.is_sociable
        npc.state = AIState.SLEEPING
        assert not npc.is_sociable
        npc.state = AIState.EATING
        npc.suspicion_meter = 80.0
        assert not npc.is_sociable


class TestAmbientChatter:
    def test_a_shopper_buys_something_now_and_then(self, game):
        quiet(game)
        w = game.world
        barnaby, tamsin = w.find_npc("Shopkeeper Barnaby"), w.find_npc("Tamsin")
        barnaby.state, barnaby.position = AIState.SHOPKEEPING, spot(game.world, -13.0, -12.2)
        barnaby.schedule, barnaby.current_task, barnaby.path = [], None, []
        tamsin.state, tamsin.position = AIState.SHOPPING, spot(game.world, -13.0, -8.4)
        tamsin.schedule, tamsin.current_task, tamsin.path = [], None, []
        tamsin.chatter_timer = 0.5
        before_t, before_b = tamsin.gold, barnaby.gold
        n = len(game.messages.messages)
        for _ in range(10):
            game.update(0.5, MoveIntent())
        assert bubbled(tamsin, "shopping") and len(game.messages.messages) == n    # a bubble, not a log line
        assert tamsin.gold < before_t and barnaby.gold == before_b + (before_t - tamsin.gold)

    def test_the_smith_at_work_hammers(self, game):
        quiet(game)
        gareth = game.world.find_npc("Gareth the Smith")
        gareth.state = AIState.WORKING
        gareth.schedule, gareth.current_task, gareth.path = [], None, []
        gareth.chatter_timer = 0.1
        game.update(0.5, MoveIntent())
        assert bubbled(gareth, "working_smith")


# ----------------------------------------------------------------------------- speech bubbles
class TestSpeechBubbles:
    def test_a_line_is_shown_for_a_while_then_expires(self, game):
        jae = game.world.find_npc("Jae the Fence")
        speech = jae.say("Hello there.", game.messages.add)
        assert game.messages.messages[-1] == '(Jae the Fence): "Hello there."'
        assert speech.until - speech.started == pytest.approx(7.5 + 0.16 * len("Hello there."))
        assert speech.until - speech.started > 9.0                        # long enough to read at leisure
        assert jae.current_speech(speech.started + 1.0) is speech
        assert jae.current_speech(speech.until + 0.01) is None and jae.speech is None

    def test_longer_lines_stay_up_longer_and_a_duration_can_be_forced(self, game):
        jae = game.world.find_npc("Jae the Fence")
        short = jae.say("Hm.")
        long_ = jae.say("A much longer sentence that takes a while to read through.")
        assert long_.until - long_.started > short.until - short.started
        forced = jae.say("Quick.", duration=0.5)
        assert forced.until - forced.started == 0.5

    def test_log_prefix_replaces_the_name_in_the_log_only(self, game):
        orrin = game.world.find_npc("Guard Orrin")
        orrin.say("Stop!", game.messages.add, log_prefix="Guard Orrin witnessed Theft!")
        assert game.messages.messages[-1] == '(Guard Orrin witnessed Theft!): "Stop!"'
        assert orrin.speech.text == "Stop!"

    def test_you_hear_people_you_face_within_earshot(self, game):
        quiet(game)
        jae = game.world.find_npc("Jae the Fence")          # at (-16, 12); the player at (-18, 3) faces -z
        jae.say("Psst.", game.messages.add)
        assert game.speech_bubbles() == []                  # behind us
        game.player.rotation_y = 180.0                      # face +z: Jae is ahead, 9 away
        assert [(n.name, s.text) for n, s in game.speech_bubbles()] == [("Jae the Fence", "Psst.")]
        game.player.position = spot(game.world, -32.0, -20.0)  # 32 away
        assert game.speech_bubbles() == []

    def test_facing_rule_uses_the_hear_radius_and_dot(self, game):
        quiet(game)
        assert game.HEAR_RADIUS == 12.0 and game.FACING_DOT == 0.3
        jae = game.world.find_npc("Jae the Fence")
        jae.position = spot(game.world, game.player.position.x, game.player.position.z + 11.0)
        jae.say("Here.")
        game.player.rotation_y = 180.0
        assert len(game.speech_bubbles()) == 1
        game.player.rotation_y = 180.0 + 80.0               # almost side-on: dot ~0.17 < 0.3
        assert game.speech_bubbles() == []
        game.player.rotation_y = 180.0 + 60.0               # dot 0.5
        assert len(game.speech_bubbles()) == 1

    def test_bubbles_vanish_once_the_line_has_been_up_long_enough(self, game):
        quiet(game)
        jae = game.world.find_npc("Jae the Fence")
        game.player.rotation_y = 180.0
        jae.say("Brief.", duration=1.0)
        assert len(game.speech_bubbles()) == 1
        for _ in range(30):
            game.update(0.05, MoveIntent())
        assert game.speech_bubbles() == []

    def test_hud_draws_a_bubble_per_heard_line(self, game):
        """The HUD layer turns (npc, speech) pairs into bubbles at the speaker's head."""
        from unprepared.rendering.hud import HudRenderer
        assert hasattr(HudRenderer, "_draw_speech_bubbles")


# ----------------------------------------------------------------------------- taking a theft report
class TestTheftReportExchange:
    """A guard answering a shopkeeper's cry runs over, takes the report face to face - the replies
    following what each of them actually saw - and then both go back to their day."""

    def _setup(self, game, guard_saw=False, merchant_saw=False):
        from unprepared.entities import TheftReport
        w = game.world
        barnaby, orrin = w.find_npc("Shopkeeper Barnaby"), w.find_npc("Guard Orrin")
        for n in w.guards():
            if n is not orrin:
                n.schedule, n.current_task, n.path = [], None, []
                n.position, n.state = spot(game.world, 110.0, 10.0), AIState.IDLE        # far out in the country
        barnaby.schedule, barnaby.current_task, barnaby.path = [], None, []
        barnaby.position, barnaby.state, barnaby.facing_yaw = spot(game.world, -13.0, -12.2), AIState.SHOPKEEPING, 180.0
        w.find_door("shop_door").is_locked = False
        orrin.position, orrin.state = spot(game.world, 6.0, 4.0), AIState.GUARDING      # on his post, keeps his routine
        quiet(game)
        if guard_saw:
            orrin.remember_sighting("behind the counter", w.day_count)
        if merchant_saw:
            barnaby.remember_sighting("behind the counter", w.day_count)
        report = TheftReport(barnaby.name, "behind the counter", w.day_count, {"wine": 2}, {"wine": "Bottle of Wine"})
        w.report_theft(report, "opening up")
        return barnaby, orrin, report

    def _play(self, game, orrin, seconds=60.0):
        started = False
        for _ in range(int(seconds / 0.25)):
            game.update(0.25, MoveIntent())
            if orrin.conversation is not None:
                started = True
                assert orrin.conversation.official and orrin.conversation.logged
            if started and orrin.conversation is None:
                break
        return started

    def test_the_guard_runs_over_and_they_face_each_other(self, game):
        barnaby, orrin, report = self._setup(game)
        assert orrin.state is AIState.RESPONDING and orrin.report_to is report
        assert said(game.messages.messages, "cry_response")
        for _ in range(200):
            game.update(0.25, MoveIntent())
            if orrin.conversation is not None:
                break
        assert orrin.conversation is not None and orrin.position.distance_to(barnaby.position) <= game.world.CHAT_RANGE
        assert orrin.faces(barnaby.position) and barnaby.faces(orrin.position)
        assert orrin.chat_pause and barnaby.chat_pause and orrin.report_to is None

    def test_nobody_saw_anything(self, game):
        barnaby, orrin, report = self._setup(game)
        assert self._play(game, orrin)
        log = game.messages.messages
        assert any(t in m for m in log for t in dialogue.REPORT_LINES["ask_seen"])
        assert any(m.startswith("(Shopkeeper Barnaby)") and any(t == m.split('"')[1] for t in dialogue.REPORT_LINES["merchant_saw_nothing"]) for m in log)
        assert any(m.startswith("(Guard Orrin)") and m.split('"')[1] in dialogue.REPORT_LINES["guard_no_lead"] for m in log)
        assert orrin.pending_confrontations == []

    def test_the_merchant_saw_someone_and_gives_the_guard_a_lead(self, game):
        barnaby, orrin, report = self._setup(game, merchant_saw=True)
        assert orrin.pending_confrontations == []                      # nothing of his own yet
        assert self._play(game, orrin)
        log = game.messages.messages
        assert any(m.split('"')[1] in {t.format(area="behind the counter") for t in dialogue.REPORT_LINES["merchant_saw"]}
                   for m in log if m.startswith("(Shopkeeper Barnaby)"))
        assert any(m.split('"')[1] in dialogue.REPORT_LINES["guard_lead"] for m in log if m.startswith("(Guard Orrin)"))
        assert orrin.pending_confrontations == [report]

    def test_a_guard_with_his_own_sighting_needs_no_description(self, game):
        barnaby, orrin, report = self._setup(game, guard_saw=True, merchant_saw=True)
        assert self._play(game, orrin)
        log = game.messages.messages
        assert any(m.split('"')[1] in {t.format(area="behind the counter") for t in dialogue.REPORT_LINES["guard_saw"]}
                   for m in log if m.startswith("(Guard Orrin)"))
        assert not any(t in m for m in log for t in dialogue.REPORT_LINES["ask_seen"])
        assert orrin.pending_confrontations == [report]

    def test_both_go_back_to_their_day_afterwards(self, game):
        barnaby, orrin, report = self._setup(game)
        assert self._play(game, orrin)
        for _ in range(40):
            game.update(0.25, MoveIntent())
        assert orrin.state in (AIState.TRAVELING, AIState.GUARDING) and orrin.report_to is None
        assert orrin.current_task is not None and orrin.current_task.target_state is AIState.GUARDING
        assert not orrin.chat_pause and not barnaby.chat_pause and barnaby.state is AIState.SHOPKEEPING
        assert orrin.goal is None

    def test_the_guard_follows_a_shopkeeper_who_has_moved_on(self, game):
        barnaby, orrin, report = self._setup(game)
        barnaby.position = spot(game.world, -13.0, -8.4)                    # stepped out from behind the counter
        assert self._play(game, orrin)
        assert orrin.position.distance_to(spot(game.world, -13.0, -8.4)) <= game.world.CHAT_RANGE

    def test_a_shopkeeper_who_caught_you_at_it_remembers_the_spot(self, game):
        w = game.world
        barnaby = w.find_npc("Shopkeeper Barnaby")
        assert barnaby.sightings == []
        w.declare_crime("Shoplifting", spot(game.world, -16.8, -12.5), witnessed=False, reporter=barnaby.name)
        assert [s.area for s in barnaby.sightings] == ["behind the counter"]


class TestReportsAreNotInterrupted(TestTheftReportExchange):
    """A guard taking one shopkeeper's report is not pulled off it by another shopkeeper's cry:
    that report goes to another guard (or waits for one). Only a crime in front of him
    interrupts - and then he returns, apologises and finishes."""

    def _second_report(self, game):
        from unprepared.entities import TheftReport
        w = game.world
        mirelle = w.find_npc("Mirelle the Tailor")
        mirelle.schedule, mirelle.current_task, mirelle.path = [], None, []
        mirelle.position, mirelle.state = spot(game.world, 11.5, -11.1), AIState.SHOPKEEPING
        w.find_door("tailor_door").is_locked = False
        report = TheftReport(mirelle.name, "the tailor's back room", w.day_count, {"needles": 1}, {"needles": "Steel Needles"})
        w.report_theft(report, "opening up")
        return mirelle, report

    def test_a_second_report_goes_to_another_guard(self, game):
        w = game.world
        barnaby, orrin, first = self._setup(game)
        vance = w.find_npc("Guard Captain Vance")
        vance.position, vance.state = spot(game.world, -4.0, 2.0), AIState.GUARDING     # a second guard, free and in earshot
        assert orrin.report_to is first
        mirelle, second = self._second_report(game)
        assert orrin.report_to is first and orrin.on_a_report                        # not pulled off Barnaby's
        assert vance.report_to is second and vance.state is AIState.RESPONDING        # Vance takes Mirelle's
        assert w.unanswered_reports == []

    def test_with_nobody_free_the_report_waits_for_the_next_free_guard(self, game):
        w = game.world
        barnaby, orrin, first = self._setup(game)
        mirelle, second = self._second_report(game)
        assert orrin.report_to is first and [r for r, _ in w.unanswered_reports] == [second]
        assert self._play(game, orrin)                                                 # Barnaby's report, start to finish
        for _ in range(120):
            game.update(0.25, MoveIntent())
            if orrin.report_to is second:
                break
        assert orrin.report_to is second and w.unanswered_reports == []               # then Mirelle's
        assert self._play(game, orrin)
        assert orrin.report_to is None and not orrin.on_a_report

    def test_a_stale_report_is_let_go(self, game):
        w = game.world
        barnaby, orrin, first = self._setup(game)
        mirelle, second = self._second_report(game)
        w.elapsed += w.REPORT_WAIT + 1.0
        game.update(0.25, MoveIntent())
        assert w.unanswered_reports == []

    def test_a_crime_in_front_of_him_interrupts_and_he_comes_back_to_apologise(self, game):
        w = game.world
        barnaby, orrin, report = self._setup(game, merchant_saw=True)
        for _ in range(200):
            game.update(0.25, MoveIntent())
            if orrin.conversation is not None:
                break
        assert orrin.conversation is not None and orrin.report_in_progress is report
        game.update(0.25, MoveIntent())                                               # a line or two in ...
        # ... and the player picks the shopkeeper's pocket right in front of him
        game.player.position = spot(game.world, orrin.position.x, orrin.position.z + 1.0)
        orrin.turn_towards(game.player.position)
        witnesses = w.report_crime_to_witnesses("Trespassing", game.player.position)     # (a crime he settles with a fine)
        assert orrin in witnesses and orrin.is_alarmed
        orrin.charge_ready_at = 0.0                                                   # no chase needed: he is right there
        assert orrin.report_interrupted is report and orrin.report_in_progress is None and orrin.report_apologise
        assert orrin.conversation is None or orrin.conversation.finished               # the exchange broke off
        assert orrin.pending_confrontations == []                                     # (the lead was never handed over)
        # he deals with the thief - the fine is paid on the spot - and goes back to Barnaby
        game.player.gold = 1000.0
        game.update(0.25, MoveIntent())
        assert orrin.crime_memories == [] and game.player.gold < 1000.0              # charged
        assert orrin.report_to is report and orrin.state is AIState.RESPONDING and orrin.report_interrupted is None
        game.player.position = spot(game.world, -12.0, 3.0)                              # out of his way
        assert self._play(game, orrin)
        log = game.messages.messages
        apologies = {t.format(merchant="Barnaby") for t in dialogue.REPORT_LINES["apology"]}
        assert any(m.startswith("(Guard Orrin)") and m.split('"')[1] in apologies for m in log)
        assert any(m.split('"')[1] in dialogue.REPORT_LINES["guard_lead"] for m in log if m.startswith("(Guard Orrin)"))
        assert orrin.pending_confrontations == [report] and not orrin.on_a_report and not orrin.report_apologise
