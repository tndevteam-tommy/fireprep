"""Two-sided item transfer sessions: trading, pickpocketing, containers.

A :class:`TransferSession` shows two inventories side by side (left: the other
party, right: the player), lets the player switch sides with Tab, move a cursor
within the active side and "confirm" the selected item, which moves one unit of
it across.  What a move *means* (price, theft, crime report...) is decided by a
:class:`TransferPolicy`; the session and the HUD widget are shared by every kind
of exchange, so new uses (lockboxes, corpses, stashes) only need a policy.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple

import random

from ..config import EconomySettings, ProgressionSettings, SkillRewards, StealthSettings
from ..entities import AIState, Item, NPC, Player, SkillType
from ..entities.inventory import add_item, take_item
from . import commerce, progression
from .stealth import (pickpocket_challenge, pickpocket_chance, pickpocket_unseen_chance, roll, secure_verdict,
                      stealing_chance, theft_challenge)

COIN_ITEM_ID = "gold_coins"
COINS_CATEGORY = "coins"          # loose coin in a chest: taken by the handful, straight into the purse
COIN_HANDFUL = 40
SECURED_HOW = {"epic": "chained to its stand", "legendary": "bound in wards and iron"}

Notify = Optional[Callable[[str], None]]
CrimeReport = Optional[Callable[[str], list]]     # returns the NPCs who witnessed the crime


class Side(Enum):
    LEFT = "left"     # the other party (merchant stock, victim's pockets, container)
    RIGHT = "right"   # the player


class TransferOutcome(Enum):
    MOVED = "moved"
    NOT_ENOUGH_GOLD = "not_enough_gold"
    MERCHANT_BROKE = "merchant_broke"   # the merchant cannot afford what the player is selling
    NOT_THEIR_TRADE = "not_their_trade" # a specialist does not deal in that kind of goods
    SERVICE = "service"                 # a service (a room at the inn) was bought: nothing changes hands
    REFUSED = "refused"                 # not allowed right now (the tutorial keeps the player to the job)
    REFUSED_STOLEN = "refused_stolen"   # merchant calls the guards; session must end
    SECURED = "secured"                 # an epic/legendary piece would not come free this time
    HOPELESS = "hopeless"               # ... and one far beyond the player's Stealing cannot even be tried
    CAUGHT = "caught"                   # failed pickpocket: the victim noticed; session must end
    NOTHING = "nothing"                 # cursor on no item


@dataclass(frozen=True)
class TransferResult:
    outcome: TransferOutcome
    item: Optional[Item] = None
    price: float = 0.0
    side: Optional[Side] = None         # side the item was taken from
    close_session: bool = False


def move_one(src: List[Item], dst: List[Item], item: Item, mutate: Optional[Callable[[Item], None]] = None) -> Item:
    """Move one unit of ``item`` from ``src`` to ``dst`` (stack-aware); ``mutate`` may adjust the moved unit first."""
    moved = take_item(src, item, 1)
    if mutate is not None:
        mutate(moved)
    return add_item(dst, moved)


# ----------------------------------------------------------------------------- policies
class TransferPolicy:
    """Rules for one kind of exchange. Subclasses override the hooks they need."""
    title = "TRANSFER"
    left_label = "THEIRS"
    right_label = "YOURS"
    help_text = "[TAB] switch side, [UP/DOWN] select, [ENTER] move item, [ESC] close"
    can_take_all = False           # whether [X] may sweep the whole left column across (containers only)

    def visible_left(self, items: List[Item]) -> List[Item]:
        return list(items)

    def row_text(self, item: Item, side: Side, session: "TransferSession") -> str:
        return item.display_name

    def row_tag(self, item: Item, side: Side) -> Optional[Tuple[str, str]]:
        """A short tag drawn at the row's right edge: (text, kind) with kind "worn" (something the
        player has on), "theft" (taking it is a crime) or "free"; None for nothing to say."""
        if side is Side.RIGHT and item.equipped:
            return ("worn", "worn")
        return None

    def give_verb(self) -> str:
        """What moving a thing right -> left means, for the "are you sure?" about a worn item."""
        return "Hand over"

    def take(self, session: "TransferSession", item: Item) -> TransferResult:   # left -> right
        move_one(session.left_items, session.right_items, item)
        return TransferResult(TransferOutcome.MOVED, item, side=Side.LEFT)

    def give(self, session: "TransferSession", item: Item) -> TransferResult:   # right -> left
        move_one(session.right_items, session.left_items, item)
        return TransferResult(TransferOutcome.MOVED, item, side=Side.RIGHT)


class TradePolicy(TransferPolicy):
    """Buying from a merchant's stock (for-sale items only) and selling to them.

    Merchants pay out of their own purse (``NPC.gold``) and pocket what the player
    spends. Every deal trains Barter: the pricier the goods the more XP, less and
    less as the skill grows; a good haggler pays less of the markup and is offered
    more for what they sell - but never more than it is worth."""
    left_label = "MERCHANT STOCK (BUY)"
    right_label = "YOUR INVENTORY (SELL)"

    def __init__(self, merchant: NPC, player: Player, economy: EconomySettings, notify: Notify = None,
                 rewards: Optional[SkillRewards] = None, progression_settings: Optional[ProgressionSettings] = None,
                 on_service: Optional[Callable[[Item], bool]] = None,
                 on_refused_stolen: Optional[Callable[[NPC], None]] = None,
                 allowed: Optional[Callable[[Item], Optional[str]]] = None):
        self.merchant, self.player, self.economy, self.notify = merchant, player, economy, notify
        self.allowed = allowed                # returns why a unit may not move right now (None: it may)
        self.on_refused_stolen = on_refused_stolen   # the merchant's cry of "Guards!" reaches the world
        self.rewards = rewards or SkillRewards()
        self.progression = progression_settings or ProgressionSettings()
        self.on_service = on_service          # handles "service" items (renting a room); returns False to refuse

    @property
    def title(self) -> str:
        return f"TRADING WITH: {self.merchant.name}   (purse: {self.merchant.gold:.0f}g)"

    def visible_left(self, items: List[Item]) -> List[Item]:
        return [i for i in items if i.for_sale]

    # Three prices (see ``EconomySettings``): an item's *worth* never changes with who is looking
    # at it; the merchant *asks* worth plus a markup that haggling whittles away (never below the
    # worth); and *offers* a fraction of the worth that grows a little with every Barter level
    # (never above the worth). Buying and selling back always loses money.
    @property
    def barter_level(self) -> float:
        """Effective Barter: the skill plus what the player wears (a merchant's chain helps)."""
        return progression.effective_level(self.player, SkillType.BARTER, self.progression)

    @property
    def markup(self) -> float:
        """The merchant's markup over the worth for this player: haggled down by the Barter level."""
        e = self.economy
        haggled = 1.0 - e.barter_markup_cut_per_level * (self.barter_level - 1)
        return e.merchant_markup * max(0.0, min(1.0, haggled))

    @property
    def sell_fraction(self) -> float:
        """What fraction of an item's worth the merchant offers for it at this Barter level."""
        e = self.economy
        return max(0.0, min(e.sell_fraction_cap, e.sell_fraction_base + e.sell_fraction_per_level * (self.barter_level - 1)))

    def buy_price(self, item: Item) -> float:
        """The asking price: worth plus the (haggled) markup. Services (a room at the inn) have a
        posted price - their worth - that nobody marks up or haggles over."""
        if item.category == "service":
            return round(item.value, 1)
        return round(item.value * (1.0 + self.markup), 1)

    def sell_price(self, item: Item) -> float:
        """The merchant's offer: a fraction of the worth (half as much again for hot goods, which
        only a fence takes), capped at the worth itself."""
        base = item.sell_value(self.economy.stolen_goods_multiplier)
        return round(base * self.sell_fraction, 1)

    def give_verb(self) -> str:
        return "Sell"

    def row_text(self, item: Item, side: Side, session: "TransferSession") -> str:
        """'Name - asking price (worth)' on the merchant's side, 'Name (offer, worth)' on the player's:
        the worth is always on show so a deal can be judged against it."""
        if side is Side.LEFT:
            price = self.buy_price(item)
            if price == round(item.value, 1):
                return f"{item.display_name} - {price:g}g"
            return f"{item.display_name} - {price:g}g (worth {item.value:g}g)"
        return f"{item.display_name} - {self.sell_price(item):g}g (worth {item.value:g}g)"

    def _train(self, price: float) -> None:
        progression.grant(self.player, SkillType.BARTER, self.rewards.barter + progression.value_xp(price, self.progression),
                          notify=self.notify, s=self.progression, challenge=StealthSettings().barter_challenge)

    def _refused(self, item: Item) -> Optional[TransferResult]:
        reason = self.allowed(item) if self.allowed is not None else None
        if reason is None:
            return None
        self._say(reason)
        return TransferResult(TransferOutcome.REFUSED, item)

    def take(self, session, item):
        refused = self._refused(item)
        if refused is not None:
            return refused
        price = self.buy_price(item)
        if self.player.gold < price:
            self._say("Not enough gold to buy this item!")
            return TransferResult(TransferOutcome.NOT_ENOUGH_GOLD, item, price, Side.LEFT)
        if item.category == "service":
            if self.on_service is None or not self.on_service(item):
                return TransferResult(TransferOutcome.NOT_THEIR_TRADE, item, price, Side.LEFT)
            self.player.gold -= price
            self.merchant.gold += price
            self._train(price)
            return TransferResult(TransferOutcome.SERVICE, item, price, Side.LEFT)
        self.player.gold -= price
        self.merchant.gold += price
        move_one(session.left_items, session.right_items, item, lambda unit: unit.transfer_to(Item.PLAYER_OWNER))
        commerce.note_sale(self.merchant, item, -1)
        self._say(f"Bought {item.name} for {price:g}g.")
        self._train(price)
        return TransferResult(TransferOutcome.MOVED, item, price, Side.LEFT)

    def give(self, session, item):
        refused = self._refused(item)
        if refused is not None:
            return refused
        if item.is_stolen and not self.merchant.is_fence:
            self.merchant.say(self.merchant.phrase("stolen_goods", item=item.name), self.notify)
            self.merchant.state = AIState.REPORTING
            if self.on_refused_stolen is not None:
                self.on_refused_stolen(self.merchant)
            return TransferResult(TransferOutcome.REFUSED_STOLEN, item, side=Side.RIGHT, close_session=True)
        if not self.merchant.buys(item):
            self.merchant.say(self.merchant.phrase("wrong_trade", role=self.merchant.role_tag or "specialist"), self.notify)
            return TransferResult(TransferOutcome.NOT_THEIR_TRADE, item, side=Side.RIGHT)
        price = self.sell_price(item)
        if self.merchant.gold < price:
            self.merchant.say(self.merchant.phrase("cant_afford", gold=f"{self.merchant.gold:.0f}"), self.notify)
            return TransferResult(TransferOutcome.MERCHANT_BROKE, item, price, Side.RIGHT)
        self.player.gold += price
        self.merchant.gold -= price
        # sold: the merchant's now - and a fence launders what it buys (no longer 'hot')
        move_one(session.right_items, session.left_items, item,
                 lambda unit: unit.transfer_to(self.merchant.name, launder=self.merchant.is_fence))
        commerce.note_sale(self.merchant, item, +1)
        self._say(f"Sold {item.name} for {price:g}g.")
        self._train(price)
        return TransferResult(TransferOutcome.MOVED, item, price, Side.RIGHT)

    def _say(self, msg: str) -> None:
        if self.notify:
            self.notify(msg)


def coin_purse(gold: float) -> Item:
    """The victim's coins shown as a row in their pockets (a handful is lifted per success)."""
    return Item(COIN_ITEM_ID, "Gold Coins", value=1.0, for_sale=False, stackable=True, quantity=int(gold))


class PickpocketPolicy(TransferPolicy):
    """Lifting items (and handfuls of coin) from, or planting them on, an NPC. Every lift
    rolls twice against the Pickpocket skill: whether the *victim* feels it (a fumble is
    caught on the spot) and whether the *bystanders* notice (``unseen_chance``) - only a
    clumsy dip is reported to whoever is watching. Stolen goods are flagged and reward
    Pickpocket XP - more for valuable things and awake or armed victims, less and less as
    the skill grows."""
    right_label = "YOUR INVENTORY (PLANT)"

    def __init__(self, victim: NPC, player: Player, rewards: SkillRewards, notify: Notify = None,
                 report_crime: CrimeReport = None, rng: Optional[random.Random] = None,
                 stealth: Optional[StealthSettings] = None, on_caught: Optional[Callable[[NPC], None]] = None,
                 economy: Optional[EconomySettings] = None, progression_settings: Optional[ProgressionSettings] = None):
        self.victim, self.player, self.rewards, self.notify, self.report_crime = victim, player, rewards, notify, report_crime
        self.rng = rng or random.Random()
        self.stealth = stealth or StealthSettings()
        self.economy = economy or EconomySettings()
        self.progression = progression_settings or ProgressionSettings()
        self.on_caught = on_caught
        self.title = f"PICKPOCKETING: {victim.name}"
        self.left_label = "THEIR POCKETS (STEAL)"

    @property
    def chance(self) -> float:
        """Odds the victim does not feel a lift of their purse (a rarer thing is harder: ``chance_for``)."""
        return pickpocket_chance(self.player, self.victim, self.stealth, self.progression)

    def chance_for(self, item: Optional[Item]) -> float:
        """Odds the victim does not feel ``item`` go (the coin purse: as common as it gets)."""
        thing = None if item is None or item.id == COIN_ITEM_ID else item
        return pickpocket_chance(self.player, self.victim, self.stealth, self.progression, item=thing)

    @property
    def unseen_chance(self) -> float:
        return pickpocket_unseen_chance(self.player, self.stealth, self.progression)

    def expose_to_bystanders(self) -> list:
        """Roll the skill against the onlookers; a clumsy dip lets them witness the crime.
        Returns the witnesses (empty when the dip was deft or nobody is reporting)."""
        if self.report_crime is None or roll(self.rng, self.unseen_chance):
            return []
        return self.report_crime("Pickpocketing") or []

    def visible_left(self, items: List[Item]) -> List[Item]:
        rows = list(items)
        if self.victim.gold >= 1.0:
            rows.insert(0, coin_purse(self.victim.gold))
        return rows

    @property
    def purse_grab(self) -> int:
        """Coins a lift takes from a purse: a deft hand (Pickpocket level) takes a bigger handful."""
        level = progression.effective_level(self.player, SkillType.PICKPOCKET, self.progression)
        return int(self.economy.purse_grab + self.economy.purse_grab_per_level * max(0.0, level - 1))

    def give_verb(self) -> str:
        return "Plant"

    def row_text(self, item, side, session):
        if item.id == COIN_ITEM_ID and side is Side.LEFT:
            return f"{item.display_name}  (grab up to {self.purse_grab})"
        return f"{item.display_name} ({item.value}g)"

    def _xp_magnitude(self, value: float) -> float:
        return self.rewards.pickpocket + progression.value_xp(value, self.progression)

    def _challenge(self, item: Item) -> float:
        thing = None if item.id == COIN_ITEM_ID else item
        return pickpocket_challenge(self.victim, thing, self.stealth)

    def take(self, session, item):
        self.expose_to_bystanders()
        challenge = self._challenge(item)       # a sleeper teaches less than a wary guard: the challenge says so
        if not roll(self.rng, self.chance_for(item)):
            # Fumbled: the victim feels the hand in their pocket.
            progression.grant(self.player, SkillType.PICKPOCKET,
                              self._xp_magnitude(item.value) * self.stealth.pickpocket_fail_xp_multiplier,
                              notify=self.notify, s=self.progression, challenge=challenge)
            if self.notify:
                self.notify(f"Fumbled! {self.victim.name} catches your hand in their pocket.")
            if self.on_caught:
                self.on_caught(self.victim)
            return TransferResult(TransferOutcome.CAUGHT, item, side=Side.LEFT, close_session=True)

        if item.id == COIN_ITEM_ID:
            grabbed = float(min(self.purse_grab, int(self.victim.gold)))
            self.victim.gold -= grabbed
            self.player.gold += grabbed
            progression.grant(self.player, SkillType.PICKPOCKET, self._xp_magnitude(grabbed), notify=self.notify,
                              s=self.progression, challenge=challenge)
            if self.notify:
                self.notify(f"Lifted {grabbed:.0f} gold from {self.victim.name}'s purse!")
            return TransferResult(TransferOutcome.MOVED, item, grabbed, Side.LEFT)

        move_one(session.left_items, session.right_items, item, Item.steal)      # still the victim's, by rights
        progression.grant(self.player, SkillType.PICKPOCKET, self._xp_magnitude(item.value), notify=self.notify,
                          s=self.progression, challenge=challenge)
        if self.notify:
            self.notify(f"Pickpocketed {item.name} from {self.victim.name}!")
        return TransferResult(TransferOutcome.MOVED, item, side=Side.LEFT)

    def give(self, session, item):
        def plant(unit: Item) -> None:
            if self.victim.is_fence:
                unit.transfer_to(self.victim.name, launder=True)     # slipped to a fence: laundered like a sale
            elif not unit.is_stolen:
                unit.transfer_to(self.victim.name)                   # a gift, as far as anyone can tell
            # stolen goods planted on an honest citizen stay stolen, and stay their owner's
        move_one(session.right_items, session.left_items, item, plant)
        if self.notify:
            self.notify(f"Planted {item.name} on {self.victim.name}.")
        return TransferResult(TransferOutcome.MOVED, item, side=Side.RIGHT)


class ContainerPolicy(TransferPolicy):
    """Taking from / putting into a container. Taking from somebody else's container is
    theft: the unit is flagged stolen and - unless a Stealing roll keeps it unnoticed -
    the crime is reported to whoever is watching. The player's *own* belongings (say, in
    the jail's confiscation lockbox) are simply taken back. Stealing XP scales with the
    value of the goods and shrinks with the skill."""
    right_label = "YOUR INVENTORY (PUT)"
    help_text = "[TAB] side, [UP/DOWN] select, [ENTER] one, [X] all, [ESC] close"
    can_take_all = True            # [X] empties the container (pockets cannot be emptied in one dip)

    def __init__(self, container, player: Player, notify: Notify = None, report_crime: CrimeReport = None,
                 rewards: Optional[SkillRewards] = None, stealth: Optional[StealthSettings] = None,
                 rng: Optional[random.Random] = None, progression_settings: Optional[ProgressionSettings] = None,
                 complimentary: Optional[Callable[[Item], bool]] = None,
                 on_complimentary: Optional[Callable[[Item], None]] = None,
                 allowed: Optional[Callable[[Item], Optional[str]]] = None):
        self.container, self.player, self.notify, self.report_crime = container, player, notify, report_crime
        self.allowed = allowed                    # returns why a unit may not be taken right now (None: it may)
        self.rewards = rewards or SkillRewards()
        self.stealth = stealth or StealthSettings()
        self.rng = rng or random.Random()
        self.progression = progression_settings or ProgressionSettings()
        self.complimentary = complimentary        # items the owner lets the player help themselves to (inn food for a guest)
        self.on_complimentary = on_complimentary  # told when one is taken (the keeper's stock count stays in step)
        self.title = container.label.upper()
        self.left_label = f"{container.label.upper()} (TAKE)"

    @property
    def is_theft(self) -> bool:
        return self.container.owner is not None

    def is_complimentary(self, item: Item) -> bool:
        return self.complimentary is not None and self.complimentary(item)

    def give_verb(self) -> str:
        return "Put away"

    def is_theft_of(self, item: Item) -> bool:
        """Would taking this unit be a crime: somebody else's, in their container, not on the house."""
        return self.is_theft and not item.is_players and not self.is_complimentary(item)

    def row_text(self, item, side, session):
        if side is Side.LEFT and self.is_theft and self.is_complimentary(item):
            return f"{item.display_name} (on the house)"
        return f"{item.display_name} ({item.value}g)"

    def row_tag(self, item, side):
        """Every row that would be a theft says so, loud and red, before it is taken."""
        if side is Side.LEFT:
            if self.is_theft_of(item):
                return ("THEFT", "theft")
            if self.is_theft and self.is_complimentary(item):
                return ("free", "free")
            return None
        return super().row_tag(item, side)

    def take(self, session, item):
        reason = self.allowed(item) if self.allowed is not None else None
        if reason is not None:
            if self.notify:
                self.notify(reason)
            return TransferResult(TransferOutcome.REFUSED, item)
        if item.is_players or not self.is_theft or self.is_complimentary(item):
            on_the_house = self.is_theft and not item.is_players
            if on_the_house and self.on_complimentary is not None:
                self.on_complimentary(item)
            message = (f"Took back your {item.name}." if item.is_players
                       else f"Took {item.name} (on the house)." if on_the_house else f"Took {item.name}.")
            if item.category == COINS_CATEGORY:
                grabbed = self._grab_coins(session, item)
                if self.notify:
                    self.notify(f"Took {grabbed:.0f} gold.")
                return TransferResult(TransferOutcome.MOVED, item, grabbed, Side.LEFT)
            # yours already, nobody's, or given: either way it leaves as the player's own
            move_one(session.left_items, session.right_items, item, lambda unit: unit.transfer_to(Item.PLAYER_OWNER))
            if self.notify:
                self.notify(message)
            return TransferResult(TransferOutcome.MOVED, item, side=Side.LEFT)
        # an epic or legendary piece is mounted, chained or warded: getting it loose comes first
        secure, odds, possible = secure_verdict(self.player, item, self.stealth, self.progression)
        if secure is not None:
            if not possible:
                if self.notify:
                    self.notify(f"The {item.name} is {SECURED_HOW.get(item.rarity, 'secured')} - you cannot even see how it "
                                f"could be freed. (Stealing far too low: years of practice, or better gear, would help.)")
                return TransferResult(TransferOutcome.HOPELESS, item, side=Side.LEFT)
            if not roll(self.rng, odds):
                progression.grant(self.player, SkillType.STEALING, self.rewards.secure_fail, notify=self.notify,
                                  s=self.progression, challenge=secure)
                if self.report_crime and not roll(self.rng, stealing_chance(self.player, self.stealth, self.progression, item)):
                    self.report_crime(self.container.crime_name)       # the rattle of the attempt gives you away
                if self.notify:
                    self.notify(f"You work at the {item.name}, but it will not come free ({odds:.0%} at your skill).")
                return TransferResult(TransferOutcome.SECURED, item, side=Side.LEFT)
            progression.grant(self.player, SkillType.STEALING, self.rewards.secure_success, notify=self.notify,
                              s=self.progression, challenge=secure)
        challenge = theft_challenge(item, self.stealth)
        unnoticed = roll(self.rng, stealing_chance(self.player, self.stealth, self.progression, item))
        witnesses = []
        if not unnoticed and self.report_crime:
            witnesses = self.report_crime(self.container.crime_name) or []
        magnitude = self.rewards.stealing + progression.value_xp(item.value, self.progression)
        if witnesses:
            magnitude *= self.stealth.stealing_fail_xp_multiplier
        if item.category == COINS_CATEGORY:
            grabbed = self._grab_coins(session, item)
            progression.grant(self.player, SkillType.STEALING, self.rewards.stealing + progression.value_xp(grabbed, self.progression),
                              notify=self.notify, s=self.progression, challenge=challenge)
            if self.notify:
                self.notify(f"Stole {grabbed:.0f} gold from the {self.container.label.lower()}.")
            return TransferResult(TransferOutcome.MOVED, item, grabbed, Side.LEFT)
        move_one(session.left_items, session.right_items, item, Item.steal)      # flagged; still the owner's by rights
        progression.grant(self.player, SkillType.STEALING, magnitude, notify=self.notify, s=self.progression, challenge=challenge)
        if self.notify:
            self.notify(f"Stole {item.name} from the {self.container.label.lower()}.")
        return TransferResult(TransferOutcome.MOVED, item, side=Side.LEFT)

    def _grab_coins(self, session, item: Item) -> float:
        """A handful of loose coins goes straight into the purse (coin carries no owner's mark)."""
        grabbed = min(item.quantity, COIN_HANDFUL)
        item.quantity -= grabbed
        if item.quantity <= 0 and item in session.left_items:
            session.left_items.remove(item)
        self.player.gold += grabbed * item.value
        return grabbed * item.value

    def give(self, session, item):
        move_one(session.right_items, session.left_items, item)
        if self.notify:
            self.notify(f"Put {item.name} in the {self.container.label.lower()}.")
        return TransferResult(TransferOutcome.MOVED, item, side=Side.RIGHT)


# ----------------------------------------------------------------------------- session
class TransferSession:
    def __init__(self, left_items: List[Item], right_items: List[Item], policy: TransferPolicy, visible_rows: int = 14):
        self.left_items = left_items
        self.right_items = right_items
        self.policy = policy
        self.side = Side.LEFT
        self.selection: Dict[Side, int] = {Side.LEFT: 0, Side.RIGHT: 0}
        self.visible_rows = visible_rows
        self.scroll: Dict[Side, int] = {Side.LEFT: 0, Side.RIGHT: 0}      # first visible row per column

    def visible(self, side: Side):
        """(first index, rows shown) for a column, keeping the cursor in view."""
        rows = self.rows(side)
        max_scroll = max(0, len(rows) - self.visible_rows)
        first = min(self.scroll[side], max_scroll)
        cursor = self.selection[side]
        if cursor < first:
            first = cursor
        elif cursor >= first + self.visible_rows:
            first = cursor - self.visible_rows + 1
        self.scroll[side] = max(0, min(first, max_scroll))
        return self.scroll[side], rows[self.scroll[side]:self.scroll[side] + self.visible_rows]

    # rows -------------------------------------------------------------------
    def rows(self, side: Side) -> List[Item]:
        return self.policy.visible_left(self.left_items) if side is Side.LEFT else list(self.right_items)

    @property
    def active_rows(self) -> List[Item]:
        return self.rows(self.side)

    def selected_item(self, side: Optional[Side] = None) -> Optional[Item]:
        side = side or self.side
        rows = self.rows(side)
        idx = self.selection[side]
        return rows[idx] if 0 <= idx < len(rows) else None

    def row_text(self, item: Item, side: Side) -> str:
        return self.policy.row_text(item, side, self)

    def row_tag(self, item: Item, side: Side):
        return self.policy.row_tag(item, side)

    def giving_away_worn(self) -> Optional[Item]:
        """The worn item the cursor is on in the player's column (the one a confirm would give away)."""
        if self.side is not Side.RIGHT:
            return None
        item = self.selected_item()
        return item if item is not None and item.equipped else None

    # navigation -------------------------------------------------------------
    def switch_side(self) -> Side:
        self.side = Side.RIGHT if self.side is Side.LEFT else Side.LEFT
        self.clamp()
        return self.side

    def move_selection(self, delta: int) -> int:
        count = len(self.active_rows)
        self.selection[self.side] = (self.selection[self.side] + delta) % max(1, count)
        return self.selection[self.side]

    def clamp(self) -> None:
        for side in Side:
            self.selection[side] = min(self.selection[side], max(0, len(self.rows(side)) - 1))

    # action -----------------------------------------------------------------
    def confirm(self) -> TransferResult:
        item = self.selected_item()
        if item is None:
            return TransferResult(TransferOutcome.NOTHING)
        result = self.policy.take(self, item) if self.side is Side.LEFT else self.policy.give(self, item)
        self.clamp()
        return result

    def take_all(self) -> List[TransferResult]:
        """[X]: move everything on the left across, one unit at a time through the policy (so every
        unit is rolled for, rewarded and reported exactly as if taken by hand). Only for policies
        that allow it (containers); stops at the first unit that will not move."""
        results: List[TransferResult] = []
        if not getattr(self.policy, "can_take_all", False):
            return results
        budget = sum(max(1, item.quantity) for item in self.rows(Side.LEFT))
        while budget > 0:
            rows = self.rows(Side.LEFT)
            if not rows:
                break
            result = self.policy.take(self, rows[0])
            results.append(result)
            budget -= 1
            if result.outcome is not TransferOutcome.MOVED or result.close_session:
                break
        self.clamp()
        return results
