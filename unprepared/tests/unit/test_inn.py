"""The Drowsy Dragon: renting a room from the keeper, what the rent buys, checkout and eviction."""

from tests.conftest import bubbled, said
from tests.helpers import spot
from unprepared.entities import AIState
from unprepared.game import Game, GameMode
from unprepared.systems import MoveIntent, Side, TransferOutcome
from unprepared.world.inn import RENT_ITEMS
from unprepared.world.levels.town import INN_BAR, INN_ROOM_BEDS
from unprepared.world.levels.town import STARTING_GOLD


def at_the_bar(game):
    """The player across the bar from Hilde, facing her (she is behind the bar all day)."""
    hilde = game.world.find_npc("Hilde the Innkeeper")
    hilde.state, hilde.position = AIState.SHOPKEEPING, game.world.spot(*INN_BAR)
    hilde.schedule, hilde.current_task, hilde.path = [], None, []
    hilde.facing_yaw = 180.0
    game.player.position = spot(game.world, INN_BAR[0], INN_BAR[2] + 2.6)
    game.player.rotation_y, game.player.rotation_x = 0.0, -15.0
    game.update(0.016, MoveIntent())
    return hilde


def beside_the_bed(w):
    """Where a guest stands in their rented room."""
    bed = w.inn.rooms[w.inn.rented_room].bed
    return spot(w, bed[0], bed[2] + 1.6)


def rent(game, item_id="rent_night"):
    hilde = at_the_bar(game)
    assert game.interaction_prompt == "[E] Talk / Trade with Hilde the Innkeeper"
    game.interact()
    assert game.mode is GameMode.TRADE
    rows = game.transfer.rows(Side.LEFT)
    index = [i.id for i in rows].index(item_id)
    for _ in range(index):
        game.cursor_down()
    result = game.transfer_confirm()
    return hilde, result


class TestRenting:
    def test_the_keeper_offers_rooms_as_services(self, game):
        hilde = at_the_bar(game)
        game.interact()
        rows = game.transfer.rows(Side.LEFT)
        services = [i for i in rows if i.category == "service"]
        assert [i.id for i in services] == ["rent_night", "rent_week"]
        assert {i.id: i.value for i in services} == {"rent_night": 10.0, "rent_week": 50.0}
        assert RENT_ITEMS == {"rent_night": 1, "rent_week": 7}
        assert hilde.inventory and not any(i.is_key and i.for_sale for i in hilde.inventory)

    def test_a_night_unlocks_room_one_and_nothing_changes_hands(self, game):
        inn = game.world.inn
        assert inn.present and inn.rented_room is None and not inn.is_rented(0, 7)
        hilde, result = rent(game)
        assert result.outcome is TransferOutcome.SERVICE and result.price == 10.0
        assert game.player.gold == STARTING_GOLD - 10.0 and hilde.gold == 260.0
        assert not any(i.id == "rent_night" for i in game.player.inventory)           # a service, not goods
        assert any(i.id == "rent_night" for i in hilde.inventory)                    # still on offer
        assert inn.rented_room == 0 and inn.rent_end_day == 1 and inn.is_rented(0, 7)
        door = game.world.find_door("inn_room_1_door")
        assert not door.is_locked and door.is_open
        assert said(game.messages.messages, "rent_night", room="room one", Room="Room one")
        assert game.mode is GameMode.TRADE                                            # the menu stays open

    def test_a_week_is_seven_nights(self, game):
        _, result = rent(game, "rent_week")
        assert result.outcome is TransferOutcome.SERVICE and game.player.gold == STARTING_GOLD - 50.0
        assert game.world.inn.rent_end_day == 7
        assert said(game.messages.messages, "rent_week", room="room one")

    def test_paying_again_extends_the_stay(self, game):
        hilde, _ = rent(game)
        result = game.transfer_confirm()
        assert result.outcome is TransferOutcome.SERVICE and game.player.gold == STARTING_GOLD - 20.0
        assert game.world.inn.rent_end_day == 2
        assert said(game.messages.messages, "rent_more", room="room one", Room="Room one", nights=2)

    def test_too_poor_to_rent(self, game):
        game.player.gold = 5.0
        _, result = rent(game)
        assert result.outcome is TransferOutcome.NOT_ENOUGH_GOLD and game.world.inn.rented_room is None

    def test_rooms_have_a_posted_price_barter_does_not_touch(self, game):
        """"Rooms are ten a night": a service is neither marked up nor haggled down."""
        from unprepared.entities import SkillType
        game.player.skills.skills[SkillType.BARTER]["level"] = 30
        _, result = rent(game)
        assert result.outcome is TransferOutcome.SERVICE and result.price == 10.0

    def test_renting_teaches_barter(self, game):
        from unprepared.entities import SkillType
        rent(game)
        assert game.player.skills.skills[SkillType.BARTER]["xp"] > 0


class TestTheRoom:
    def test_a_paying_guest_is_welcome_in_the_room_and_nowhere_else_upstairs(self, game):
        w = game.world
        rent(game)
        game.close_screens()
        room_one = next(a for a in w.level.areas if a.name == "room one")
        room_two = next(a for a in w.level.areas if a.name == "room two")
        assert w.area_trusts_player(room_one) and not w.area_trusts_player(room_two)
        where = beside_the_bed(w)
        assert w.area_of(where) == "room one"
        game.player.position = where
        hilde = w.find_npc("Hilde the Innkeeper")
        assert not w.is_trespassing(game.player, hilde)
        game.player.position = spot(w, INN_ROOM_BEDS[1][0], INN_ROOM_BEDS[1][2] + 1.6)
        assert w.is_trespassing(game.player, hilde)

    def test_without_renting_the_rooms_are_locked_and_off_limits(self, game):
        w = game.world
        for door_id in ("inn_room_1_door", "inn_room_2_door", "inn_room_3_door"):
            assert w.find_door(door_id).is_locked
        game.player.position = spot(w, INN_ROOM_BEDS[0][0], INN_ROOM_BEDS[0][2] + 1.6)
        assert w.is_trespassing(game.player, w.find_npc("Hilde the Innkeeper"))

    def test_the_room_is_yours_until_noon_on_the_last_day(self, game):
        inn = game.world.inn
        rent(game)
        assert inn.is_rented(0, 23) and inn.is_rented(1, 11) and not inn.is_rented(1, 12) and not inn.is_rented(2, 0)
        assert inn.nights_left(0) == 1 and inn.nights_left(1) == 0

    def test_at_checkout_the_room_is_locked_again_and_you_are_told(self, game):
        w = game.world
        rent(game)
        game.close_screens()
        game.player.position = spot(game.world, -12.0, 3.0)                    # out on the street
        w.wait_hours(24 + 5)                                                  # 07:00 -> 12:00 the next day
        assert w.inn.rented_room is None and w.find_door("inn_room_1_door").is_locked
        assert any("rent on room one has run out" in m for m in game.messages.messages)

    def test_a_guest_still_in_the_room_at_checkout_is_shown_out(self, game):
        w = game.world
        rent(game)
        game.close_screens()
        game.player.position = beside_the_bed(w)
        game.player.rotation_y = 0.0
        for _ in range(29):
            game.wait()                                                       # [T] hour by hour to noon next day
            game.update(0.5, MoveIntent())
            if game.fade is not None:
                break
        assert said(game.messages.messages, "rent_over")
        for _ in range(200):
            game.update(0.05, MoveIntent())
        assert w.inn.rented_room is None
        assert w.area_of(game.player.position) not in ("room one", "the inn kitchen")
        assert w.find_door("inn_room_1_door").is_locked
        # ... out of the *room*, into the hall: the inn is open to all, only the room was ours
        assert w.area_of(game.player.position) == "the inn" and w.building_at(game.player.position).name == "inn"
        assert any("shown out of room one" in m for m in game.messages.messages)

    def test_everybody_is_welcome_in_the_hall_at_every_hour(self, game):
        """The inn is a public house: somebody coming in at night - with a room or without - is
        nobody's trespasser, not the watch's on the night round, not the keeper's. The kitchen,
        the keeper's own room and the rooms of other guests stay off limits."""
        w = game.world
        w.wait_hours(17)                                                      # 00:00, no room rented
        mira, hilde = w.find_npc("Guard Mira"), w.find_npc("Hilde the Innkeeper")
        hall = spot(game.world, -32.5, -8.0)
        game.player.position = hall
        assert w.player_unwelcome_at(hall) is None
        assert not w.is_trespassing(game.player, mira) and not w.is_trespassing(game.player, hilde)
        game.player.position = spot(game.world, -33.3, -19.0)                    # room one, not rented
        assert w.player_unwelcome_at(game.player.position) is not None
        game.player.position = spot(game.world, -37.5, -12.5)                    # behind the bar, in the kitchen
        assert w.player_unwelcome_at(game.player.position) is not None
        w.inn.rent(w, game.player, 1)
        game.close_screens()
        for where in (hall, spot(game.world, -33.3, -19.0)):                     # a guest: the hall and their room
            game.player.position = where
            assert w.player_unwelcome_at(where) is None
            assert not w.is_trespassing(game.player, mira) and not w.is_trespassing(game.player, hilde)

    def test_checkout_while_asleep_puts_you_in_the_hall_not_the_street(self, game):
        """Sleeping past checkout is the price of not paying the bill - but the bill is for the
        room, so the sleeper wakes in the hall, not in the street."""
        w = game.world
        rent(game)                                                            # one night: checkout noon tomorrow
        game.close_screens()
        w.wait_hours(23)                                                      # 06:00 next day
        game.player.position = beside_the_bed(w)
        assert game.sleep()                                                   # until 20:00: straight through checkout
        for _ in range(400):
            game.update(0.016, MoveIntent())
            if game.fade is None:
                break
        assert w.clock.hour == 20 and w.inn.rented_room is None
        assert w.area_of(game.player.position) == "the inn" and w.building_at(game.player.position).name == "inn"

    def test_checkout_striking_as_the_guest_lies_down_is_not_lost(self, game):
        """Noon strikes in the second or two the screen takes to go dark: the keeper's showing-out
        happens at black like any other, the guest sleeps on in the hall - and nothing is left
        half-done. (The move used to wait for a fade that never came: ``World.relocating`` stayed
        on for good, which refuses every save and switches the jail, curfews and charges off.)"""
        w = game.world
        rent(game)                                                            # one night: checkout noon tomorrow
        game.close_screens()
        w.wait_hours(28)                                                      # 11:00 next day
        w.clock.minute_accumulator = w.clock.minutes_per_hour - 0.5           # noon in a fraction of a second
        game.player.position = beside_the_bed(w)
        assert game.sleep()
        game.update(0.3, MoveIntent())
        assert w.clock.hour == 12 and w.relocating                            # shown out while the screen darkens
        for _ in range(400):
            game.update(0.016, MoveIntent())
            if game.fade is None:
                break
        assert w.clock.hour == 20 and w.inn.rented_room is None
        assert w.area_of(game.player.position) == "the inn"                   # woke in the hall, as ever
        assert not w.relocating and game.pending_relocations == []
        assert game.save_refusal is None and game.save_game()

    def test_the_stay_survives_a_save_and_load(self, game, tmp_path):
        rent(game)
        game.close_screens()
        assert game.save_game()
        g2 = Game(settings=game.settings, save_file=game.saves.filename)
        assert g2.world.inn.rented_room is None
        assert g2.load_game()
        assert g2.world.inn.rented_room == 0 and g2.world.inn.rent_end_day == 1
        assert not g2.world.find_door("inn_room_1_door").is_locked
        room_one = next(a for a in g2.world.level.areas if a.name == "room one")
        assert g2.world.area_trusts_player(room_one)


class TestNoInn:
    def test_a_level_without_an_inn_refuses_the_service(self, game):
        from unprepared.world.inn import InnState
        assert not InnState.from_def(None).present
        game.world.inn = InnState.from_def(None)
        _, result = rent(game)
        assert result.outcome is TransferOutcome.NOT_THEIR_TRADE and game.player.gold == STARTING_GOLD


class TestSleepersHearOnlyTheirOwnRoom:
    """The keeper asleep behind her shut door is not woken by a guest crossing the hall."""

    def _night(self, game):
        w = game.world
        w.inn.rent(w, game.player, 1)                 # a guest (a visitor without a room is welcomed, which wakes her)
        game.close_screens()
        hilde = w.find_npc("Hilde the Innkeeper")
        hilde.schedule, hilde.current_task, hilde.path = [], None, []
        hilde.state, hilde.position = AIState.SLEEPING, spot(game.world, -37.75, -20.0)
        w.find_door("keeper_room_door").lock_and_close()
        w.clock.hour = 2
        return hilde

    def test_walls_and_shut_doors_make_separate_chambers(self, game):
        w = game.world
        hall, keeper_room = spot(game.world, -34.0, -14.0), spot(game.world, -37.75, -20.0)
        assert w.same_room(hall, keeper_room)                                  # one building ...
        assert not w.same_chamber(hall, keeper_room)                           # ... but a wall and a door between
        assert w.noise_attenuation(hall, keeper_room) == game.settings.perception.wall_noise_attenuation
        assert w.same_chamber(hall, spot(game.world, -26.0, -10.0))               # across the open hall
        assert w.same_chamber(spot(game.world, -13.0, -8.4), spot(game.world, -13.0, -12.2))   # the shop floor and behind the counter
        w.find_door("keeper_room_door").unlock_and_open()
        assert w.same_chamber(spot(game.world, -36.4, -15.5), keeper_room)        # through the open doorway

    def test_running_through_the_hall_does_not_wake_the_keeper(self, game):
        hilde = self._night(game)
        game.player.position = spot(game.world, -33.3, -12.0)
        game.player.rotation_y = 0.0                                           # towards the guest rooms
        for _ in range(60):
            game.update(0.05, MoveIntent(forward=True, sprint=True))
        assert hilde.state is AIState.SLEEPING and game.player.position.z < -14.0

    def test_running_about_in_her_room_does(self, game):
        hilde = self._night(game)
        game.player.position = spot(game.world, -37.0, -18.0)
        game.player.rotation_y = 90.0
        for _ in range(20):
            game.update(0.05, MoveIntent(left=True, right=True, sprint=True))
        assert hilde.state is not AIState.SLEEPING


class TestOnTheHouse:
    """Food and ale in the inn come with the room."""

    def _open(self, game, name):
        table = game.world.find_container(name)
        game.player.position = spot(game.world, table.center.x, table.center.z + 1.3)
        game.player.rotation_y, game.player.rotation_x = 0.0, -60.0
        game.update(0.016, MoveIntent())
        game.interact()
        assert game.mode is GameMode.CONTAINER, game.interaction_prompt
        return table

    def test_a_guest_helps_themselves_without_it_being_theft(self, game):
        rent(game)
        game.close_screens()
        table = self._open(game, "inn_table_1")
        assert "on the house" in game.transfer.row_text(table.inventory[0], Side.LEFT)
        result = game.transfer_confirm()
        assert result.outcome is TransferOutcome.MOVED
        ale = game.player.inventory[-1]
        assert ale.id == "ale" and not ale.is_stolen
        assert game.messages.messages[-1] == "Took Mug of Ale (on the house)."
        assert game.player.crimes_witnessed == 0
        keeper = game.world.find_npc("Hilde the Innkeeper")
        assert keeper.state is not AIState.REPORTING

    def test_without_a_room_it_is_theft(self, game):
        table = self._open(game, "inn_table_1")
        assert "on the house" not in game.transfer.row_text(table.inventory[0], Side.LEFT)
        game.transfer_confirm()
        assert game.player.inventory[-1].id == "ale" and game.player.inventory[-1].is_stolen

    def test_the_keepers_stock_count_stays_in_step(self, game):
        from unprepared.systems import commerce
        w = game.world
        keeper = w.find_npc("Hilde the Innkeeper")
        rent(game)
        game.close_screens()
        self._open(game, "inn_kitchen_table")
        game.transfer_confirm()                                       # a pot of stew from the kitchen
        assert commerce.take_stock(keeper, w) is None                 # nothing "missing"
        game.close_screens()
        w.inn.end_stay(w)
        self._open(game, "inn_kitchen_table")
        game.transfer_confirm()                                       # now we are just a thief
        assert commerce.take_stock(keeper, w) is not None

    def test_only_food_and_drink_are_free(self, game):
        rent(game)
        w = game.world
        table = w.find_container("inn_table_1")
        from unprepared.entities import Item
        candle = Item("candle_stub", "Tallow Candle", value=2.0)
        assert not w.is_complimentary(table, candle)
        assert w.is_complimentary(table, table.inventory[0])
        assert not w.is_complimentary(w.find_container("barnaby_table"), table.inventory[0])


class TestSleepUntilMorning:
    def _bed_in_sight(self, game):
        bed = next(p for p in game.world.props if p.name == "inn_room_1_bed")
        game.player.position = spot(game.world, bed.center.x, bed.center.z + 1.8)
        game.player.rotation_y, game.player.rotation_x = 0.0, -35.0
        game.update(0.016, MoveIntent())
        return bed

    def test_a_bed_that_is_not_yours_cannot_be_slept_in(self, game):
        bed = self._bed_in_sight(game)
        assert game.target is not None and game.target.is_bed and game.target.obj is bed
        assert game.interaction_prompt == "[Not your bed]"
        game.interact()
        assert game.fade is None and game.messages.messages[-1] == "That is not your bed."

    def test_the_rented_rooms_bed_is_yours_until_checkout(self, game):
        w = game.world
        rent(game)
        game.close_screens()
        bed = self._bed_in_sight(game)
        assert w.player_may_sleep_in(bed) and game.interaction_prompt == "[E] Sleep Until Night"      # it is 07:00
        assert not w.player_may_sleep_in(next(p for p in w.props if p.name == "inn_room_2_bed"))
        w.wait_hours(24 + 5)                                          # checkout has come and gone
        assert not w.player_may_sleep_in(bed)

    def test_sleeping_fades_out_skips_to_six_and_fades_back_in(self, game):
        w = game.world
        rent(game)
        game.close_screens()
        w.clock.hour, w.clock.minute_accumulator = 21, 30.0
        self._bed_in_sight(game)
        game.interact()
        assert game.fade is not None and game.fade.reason == "sleep" and game.frozen
        assert game.messages.messages[-1] == "You settle down to sleep until morning..."
        assert game.fade.OUT_SECONDS >= 1.5                           # a slow fade
        before = game.player.position.copy()
        for _ in range(12):
            game.update(0.1, MoveIntent(forward=True))                # still going dark: nothing moves, nothing happens
        assert game.player.position == before and 0.0 < game.fade_alpha < 1.0 and w.clock.hour == 21
        for _ in range(60):
            game.update(0.1, MoveIntent())
        assert game.fade is None and not game.frozen
        assert w.clock.hour == 6 and w.day_count == 1 and w.clock.minute_accumulator < 20.0   # (the fade-in took a few minutes)
        assert "You wake at 06:00, rested." in game.messages.messages
        mira = w.find_npc("Guard Mira")
        assert mira.task_for(6) is not None and mira.state is not AIState.SLEEPING    # the town lived on meanwhile


    def test_by_day_a_bed_sleeps_until_nightfall(self, game):
        from unprepared.game import Game
        w = game.world
        rent(game)
        game.close_screens()
        w.clock.hour, w.clock.minute_accumulator = 10, 0.0
        self._bed_in_sight(game)
        assert game.interaction_prompt == "[E] Sleep Until Night"
        game.interact()
        assert game.messages.messages[-1] == "You settle down to sleep until night..."
        for _ in range(60):
            game.update(0.1, MoveIntent())
        assert w.clock.hour == Game.NIGHTFALL_HOUR == 20 and w.day_count == 0
        assert "You wake at 20:00, rested." in game.messages.messages
        self._bed_in_sight(game)
        assert game.interaction_prompt == "[E] Sleep Until Morning"                 # ... and now the night offer
        assert Game.sleep_target(18) == (6, "morning") and Game.sleep_target(5) == (6, "morning")    # dusk and small hours
        assert Game.sleep_target(6) == (20, "night") and Game.sleep_target(17) == (20, "night")

    def test_a_bed_the_player_owns_works_the_same_way(self, game):
        w = game.world
        bed = next(p for p in w.props if p.name == "pell_bed")
        assert not w.player_may_sleep_in(bed)
        bed.owner = "player"
        assert w.player_may_sleep_in(bed)
        assert game.sleep_until_morning(bed) and game.fade is not None


class TestTheWelcome:
    """Somebody who comes in without a room is welcomed by the keeper - woken if need be."""

    def _into_the_hall_at(self, game, hour):
        w = game.world
        w.clock.hour = hour
        w.resync_schedules(announce=False, teleport=True)
        game.player.position = spot(game.world, -32.5, -8.0)                     # just inside the door
        game.update(0.05)
        return w.find_npc("Hilde the Innkeeper")

    def test_at_night_the_keeper_wakes_and_comes_over(self, game):
        w = game.world
        hilde = self._into_the_hall_at(game, 2)                               # 02:00: she was asleep behind her door
        assert hilde.state is AIState.WELCOMING and hilde.awake_for_business and w.inn.welcome == "going"
        for _ in range(300):                                                  # up, through her door, across the hall
            game.update(0.05)
            if w.inn.welcome == "greeting":
                break
        assert w.inn.welcome == "greeting"
        assert hilde.position.distance_to(game.player.position) <= w.CHAT_RANGE + 1.5
        assert bubbled(hilde, "inn_welcome_woken") and said(game.messages.messages, "inn_welcome_woken")
        assert hilde.is_open_for_business                                     # a room can be had right here
        assert not w.is_trespassing(game.player, hilde) and not w.is_trespassing(game.player, w.find_npc("Guard Mira"))
        # rent one: she says her piece, the welcome is over and she goes back to bed
        game.open_trade(hilde)
        rent_row = next(i for i in game.transfer.rows(Side.LEFT) if i.id == "rent_night")
        game.transfer.selection[Side.LEFT] = game.transfer.rows(Side.LEFT).index(rent_row)
        assert game.transfer_confirm().outcome is TransferOutcome.SERVICE
        game.close_transfer()
        game.update(0.05)
        assert w.inn.welcome is None and hilde.state is not AIState.WELCOMING and not hilde.awake_for_business
        for _ in range(400):
            game.update(0.05)
            if hilde.state is AIState.SLEEPING:
                break
        assert hilde.state is AIState.SLEEPING                                # no rest for an innkeeper - but some

    def test_a_visitor_who_takes_no_room_is_left_after_a_while_and_welcomed_again_next_visit(self, game):
        w = game.world
        hilde = self._into_the_hall_at(game, 2)
        for _ in range(300):
            game.update(0.05)
            if w.inn.welcome == "greeting":
                break
        w.inn.welcome_timer = w.inn.WELCOME_PATIENCE + 1.0                    # she has waited long enough
        game.update(0.05)
        assert w.inn.welcome is None and hilde.state is not AIState.WELCOMING
        game.update(0.05)
        assert w.inn.welcome is None and w.inn.welcomed_this_visit             # not again this visit ...
        game.player.position = spot(game.world, -32.5, -3.0)                     # ... out in the street ...
        game.update(0.05)
        assert not w.inn.welcomed_this_visit
        for _ in range(200):                                                  # (let her get back to bed)
            game.update(0.05)
        game.player.position = spot(game.world, -32.5, -8.0)                     # ... and back in: welcomed anew
        game.update(0.05)
        assert w.inn.welcome is not None

    def test_by_day_she_greets_from_behind_the_bar_without_leaving_it(self, game):
        w = game.world
        hilde = self._into_the_hall_at(game, 8)                               # 08:00: behind the bar
        bar = hilde.position.copy()
        assert hilde.state is AIState.WELCOMING
        game.update(0.05)
        assert w.inn.welcome == "greeting" and hilde.position == bar           # greeted from her post
        assert bubbled(hilde, "inn_welcome") and hilde.is_open_for_business
        for _ in range(60):
            game.update(0.05)
        assert hilde.position == bar

    def test_a_guest_with_a_room_is_not_welcomed_twice(self, game):
        w = game.world
        w.inn.rent(w, game.player, 1)
        game.close_screens()
        hilde = self._into_the_hall_at(game, 2)
        assert hilde.state is AIState.SLEEPING and w.inn.welcome is None

    def test_the_welcome_is_a_moment_not_a_state(self, game):
        w = game.world
        self._into_the_hall_at(game, 2)
        assert w.inn.welcome is not None
        w.inn.welcome, w.inn.welcome_timer = "greeting", 30.0
        game.save_game()
        import json
        assert "welcome" not in json.dumps(json.load(open(game.saves.filename))["inn"])
        assert game.load_game()
        # loading starts afresh: the visitor is still in the hall without a room, so a new welcome begins at zero
        assert w.inn.welcome_timer < 1.0 and w.inn.welcome in (None, "going")
