"""Scripted gameplay scenarios covering every feature of the game.

They drive the real input controller, so each scenario is a list of frames
with key presses / mouse motion / held keys, exactly as a player would produce.
Used by: the golden regression tests (headless), the prototype parity tests,
the pixel-golden tests and ``render_gallery.py``.

Geometry reminders (see ``unprepared/world/levels/town.py``; the town is on a hill, so the player's
height follows the ground - the scripts only ever say where on the map):
  Market Street runs east-west (z -6 .. 6) with the well at the origin; player spawn (-4.5, 4) facing -z
  the hideout door hinge (-13.5, 7.5) on the hideout's north wall (leaf x -13.5 .. -10.5, swings out onto the street)
  shop door hinge (-15, -6) (south wall of the general store; leaf x -15 .. -12, opens inwards)
  Barnaby's front door hinge (-71.5, 7) (leaf x -71.5 .. -68.5); his table seat (-71.5, 12), his bed (-67.3, 13.4)
  guard house (up the hill on Upper Street) x 42..58 z -53..-43, door hinge (48.5, -43); the day post at (53.5, -40.5)
  jail x 48..58 z 6..16, door hinge (51.5, 6); the cell x 53.5..57.5 z 12..15.5 behind bars hinged at (54, 12)
  smithy shed x -65..-58 z -14..-6 (open sides): anvil (-62.5..-61.5, -9.4..-8.6), ore deposit (-64..-61, -21..-18)
  07:00 positions: Barnaby at breakfast (-71.5, 12), Jae in his den (-10, 12), Vance starting his patrol at (37, -3),
  Orrin on the day post, Mira asleep in the guard house, Hesk at his post in the jail (50.8, 10.5)
Long walks across town are a ``place`` frame (the game only - the prototype-identical scenarios use none).
Mouse: 0.15 deg per pixel, so 1200 px = 180 degrees.  Walking: 4 u/s -> 0.064 u per 16 ms frame.
Yaw 0 faces -z, yaw 90 faces +x, yaw 180 faces +z, yaw 270 faces -x.
Crouch (LCTRL) is the theft stance: lock picking and pickpocketing only work while crouched.
"""
from __future__ import annotations

import random
from typing import Callable, Dict

from .harness import K, SEED, Scenario

W, A, S, D, SHIFT = K.K_w, K.K_a, K.K_s, K.K_d, K.K_LSHIFT


def walk(sc: Scenario, key: int, units: float, sprint: bool = False, crouched: bool = False) -> Scenario:
    per_frame = 0.064 * (1.8 if sprint else 1.0) * (0.5 if crouched else 1.0)    # sneaking is half speed
    held = [key, SHIFT] if sprint else [key]
    return sc.hold(held, max(1, round(units / per_frame)))


def turn(sc: Scenario, degrees: float) -> Scenario:
    return sc.mouse(round(degrees / 0.15), 0)


def pitch(sc: Scenario, degrees: float) -> Scenario:
    return sc.mouse(0, -round(degrees / 0.15))


def sweet_spot_for_attempt(n: int) -> float:
    """The n-th (0-based) random draw of the lock mini-game's stream for the test seed: each attempt
    draws its sweet spot, and a pick on the spot draws once more (whether the pins set)."""
    rng = random.Random(SEED)
    value = 0.0
    for _ in range(n + 1):
        value = rng.uniform(0.0, 360.0)
    return value


def rotate_to_sweet_spot(sc: Scenario, attempt: int) -> Scenario:
    """Press RIGHT until the pick (starting at 0) is within tolerance of the sweet spot."""
    steps = round(sweet_spot_for_attempt(attempt) / 10.0) % 36
    for _ in range(steps):
        sc.press(K.K_RIGHT)
    return sc


def go_to_shop_door(sc: Scenario, wait_seconds: int = 13) -> Scenario:
    """To (-13.5, -4) facing the shop door (-z), two paces off. Waits first (the morning patrol
    walks the street; Barnaby leaves for the shop at 08:00, 24 s after the start)."""
    sc.idle(wait_seconds, dt_ms=1000)
    return sc.place(-13.5, -4.0, 0.0)


def go_to_house_door(sc: Scenario, wait_seconds: int = 0) -> Scenario:
    """To (-70, 3) facing Barnaby's front door (+z), four units off."""
    if wait_seconds:
        sc.idle(wait_seconds, dt_ms=1000)
    return sc.place(-70.0, 3.0, 180.0)


def go_to_guard_post(sc: Scenario) -> Scenario:
    """To (51.5, -40.5) facing +x: the guard on the day post outside the guard house stands two units east."""
    return sc.place(51.5, -40.5, 90.0)


# --------------------------------------------------------------------------- scenarios
def idle_day() -> Scenario:
    return Scenario("idle_day").capture().idle(30).capture_last()


def look_around() -> Scenario:
    sc = Scenario("look_around")
    sc.idle(2)
    turn(sc, 45).idle(3).capture()
    pitch(sc, 30).idle(3).capture()
    pitch(sc, 200).idle(2).capture()          # clamps at +89
    pitch(sc, -400).idle(2).capture()         # clamps at -89
    pitch(sc, 89)
    turn(sc, -135).idle(3).capture()
    sc.mouse(7, -3).mouse(-2, 5).idle(2).capture_last()
    return sc


def walk_and_sprint() -> Scenario:
    sc = Scenario("walk_and_sprint")
    walk(sc, W, 2.0).capture()
    walk(sc, S, 1.0)
    walk(sc, A, 1.5).capture()
    walk(sc, D, 3.0)
    turn(sc, 90)
    walk(sc, W, 2.0, sprint=True).capture()
    sc.hold([W, S], 5)                         # cancel out
    sc.hold([SHIFT], 3)                        # sprint key alone: not sprinting
    sc.hold([W, D, SHIFT], 20).idle(5).capture_last()
    return sc


def wall_collision() -> Scenario:
    sc = Scenario("wall_collision")
    sc.place(-24.0, -1.0, 0.0)
    walk(sc, W, 12.0, sprint=True).idle(2).capture()      # the inn's south wall (z = -6) stops us at z ~ -5.6
    walk(sc, S, 0.2)                                        # (a hair off it: a shut door's leaf stands proud of the wall)
    turn(sc, 90)                                            # face +x
    walk(sc, W, 24.0, sprint=True).idle(2).capture()      # run along the walls; the shop's shut door is flush with them
    turn(sc, 90)                                            # face +z
    walk(sc, W, 5.0).idle(2).capture()                    # the well: a solid block at the origin
    sc.hold([W, D], 60).idle(2).capture()                   # diagonal into it: slides along its side
    sc.place(-70.0, 0.0, 270.0)                             # the west end of the street, facing -x
    walk(sc, W, 15.0, sprint=True).idle(2).capture_last()   # the west town wall (inner face x = -79) stops us
    return sc


def hideout_door() -> Scenario:
    sc = Scenario("hideout_door")
    sc.place(-12.0, 3.0, 180.0)                # face +z: the hideout door is 4.5 units away
    walk(sc, W, 2.0).idle(2).capture()         # z -> 5, well within reach: "[E] Open Door"
    sc.press(K.K_e).idle(3).capture()          # opened (the leaf swings out, towards us, along x = -13.5)
    sc.press(K.K_e).idle(3).capture()          # closed
    walk(sc, W, 3.0).idle(2).capture()         # walk into the closed leaf: blocked at z ~ 7.1
    walk(sc, S, 1.5)
    sc.press(K.K_e).idle(2)                    # open again
    walk(sc, W, 5.0).idle(2).capture()         # through the doorway (z -> ~10.5)
    turn(sc, 180).idle(2).capture()            # look back at the open door from inside
    sc.press(K.K_e).idle(3).capture_last()     # close it from inside
    return sc


def shop_lockpick() -> Scenario:
    sc = Scenario("shop_lockpick")
    go_to_shop_door(sc).idle(2).capture()      # face -z: "[Locked]" while standing
    sc.press(K.K_LCTRL).idle(20).capture()     # crouch: camera drops, prompt becomes "[E] Pick Lock"
    sc.press(K.K_e).idle(3).capture()          # mini-game opens, pick at 0 (attempt 0: a fresh sweet spot)
    sc.press(K.K_LEFT).press(K.K_LEFT).idle(2).capture()     # angle 340
    sc.mouse(300, 100).hold([W], 5)            # look/move ignored while picking
    sc.press(K.K_1).press(K.K_e).idle(1)       # torch toggle and interact ignored too
    sc.press(K.K_ESCAPE).idle(2).capture()     # cancel
    sc.press(K.K_e).idle(2)                    # attempt 1: pick reset to 0, new sweet spot
    rotate_to_sweet_spot(sc, 1)
    sc.idle(2).capture()                       # aligned: glowing ring
    sc.press(K.K_SPACE).idle(3).capture()      # success: door unlocked & open
    sc.press(K.K_LCTRL).idle(20)               # stand up
    walk(sc, W, 4.0).idle(2).capture_last()    # walk into the shop
    return sc


def mine_and_forge() -> Scenario:
    """Buy a pickaxe, mine ore at the deposit behind the smithy and forge a batch of picks."""
    sc = Scenario("mine_and_forge")
    _to_barnabys_counter(sc)
    sc.press(K.K_e).idle(2)                    # trade: the pickaxe and the mallet sit on the east shelf (composite stock)
    for _ in range(9):
        sc.press(K.K_DOWN)                     # down to the Pickaxe row
    sc.idle(1).capture()
    sc.press(K.K_RETURN).idle(2).capture()     # bought for 43.8g (35g worth + the 25% markup)
    sc.press(K.K_RETURN).idle(2).capture()     # and the mallet (12g -> 15g), now on the same row: the anvil needs one
    sc.press(K.K_ESCAPE).idle(1)
    pitch(sc, 20)
    walk(sc, S, 4.0)                           # back out of the shop (z -> -4)
    sc.place(-59.5, -19.5, 270.0, -10.0)       # over the bridge to the ore deposit behind the smith's shed, facing -x
    sc.idle(2).capture()                       # "[E] Mine Iron Ore from the Ore Deposit"
    sc.press(K.K_e).press(K.K_e).idle(2).capture()    # two pieces of ore
    sc.place(-60.5, -9.0, 270.0, -20.0)        # under the shed roof beside the kiln, facing the anvil (x -62.5..-61.5)
    sc.idle(2).capture()                       # "[E] Forge Iron Lockpicks (1 Iron Ore; up to 1 at your skill)"
    sc.press(K.K_e).idle(2).capture()          # a novice smith: each forging may ruin the ore (and still teach)
    sc.press(K.K_e).idle(2).capture_last()     # the second ore
    return sc


def lockpick_out_of_picks() -> Scenario:
    sc = Scenario("lockpick_out_of_picks")
    go_to_shop_door(sc).press(K.K_LCTRL).idle(20)
    sc.press(K.K_e).idle(2).capture()
    sc.press(K.K_LEFT).press(K.K_LEFT).press(K.K_LEFT).press(K.K_LEFT).press(K.K_LEFT)   # 310: far from most spots
    sc.press(K.K_SPACE).idle(3).capture()      # snap with the only pick -> back to explore, "[Locked] (no lockpick)"
    sc.press(K.K_e).idle(2).capture_last()     # nothing to pick with
    return sc


def _into_the_hideout(sc: Scenario) -> Scenario:
    """Open the hideout door and walk in up to Jae (at (-10, 12), facing the door)."""
    sc.place(-12.0, 4.5, 180.0)                # three paces from the door, facing it
    sc.press(K.K_e).idle(1)                    # open it
    walk(sc, W, 6.0)                           # z -> 10.5 (Jae is 2.5 away, off to our left)
    return sc.place(-12.0, 10.5, 126.9, -15.0).idle(2)    # face him


def trade_fence() -> Scenario:
    sc = Scenario("trade_fence")
    _into_the_hideout(sc).capture()            # "[E] Talk / Trade"
    sc.press(K.K_e).idle(3).capture()          # trade menu, merchant column active
    sc.mouse(500, 0).hold([W, SHIFT], 3)       # ignored while trading
    sc.press(K.K_TAB).press(K.K_UP).idle(2).capture()         # Tab to our column, wrap up to the goblet
    sc.press(K.K_RETURN).idle(3).capture()     # sold: a quarter of its worth, halved for hot goods (15g)
    sc.press(K.K_TAB).idle(1)                  # back to the stock
    sc.press(K.K_RETURN).idle(3).capture()     # buy one Master Lockpick at the asking price (20g + 25%; stack of 3 -> 2)
    sc.press(K.K_DOWN).press(K.K_RETURN).idle(2).capture()    # gold ring: 187.5 > our purse -> "Not enough gold"
    sc.press(K.K_TAB).press(K.K_UP).press(K.K_RETURN).idle(2).capture()     # our column, first row: sell the Iron Lockpick
    sc.press(K.K_LCTRL).idle(1)                # crouch toggle is ignored in the menu
    sc.press(K.K_ESCAPE).idle(3).capture()     # close menu
    sc.press(K.K_e).idle(2)                    # re-open
    sc.press(K.K_e).idle(2).capture_last()     # E closes
    return sc


def _to_barnabys_counter(sc: Scenario, hours: int = 1) -> Scenario:
    """Wait for opening time (or ``hours`` on: at 15:00 the captain stands his post at the west end
    of the street, in earshot of a shout from the store), open the (unlocked but shut) shop door
    and walk up to the counter."""
    sc.wait_hours(hours)                       # 08:00: Barnaby sets off across town
    sc.idle(24, dt_ms=1000)                    # ... and arrives behind the counter, shop unlocked
    go_to_shop_door(sc, wait_seconds=2).idle(2)
    sc.press(K.K_e).idle(2)                    # "[E] Open Door": customers open the door themselves
    walk(sc, W, 4.4)                           # z -> -8.4 (the counter is at z -11.2..-10.6, Barnaby behind it)
    return pitch(sc, -20).idle(2)


def trade_merchant_refuses_stolen() -> Scenario:
    """Selling stolen goods to Barnaby. NOTE: crashes the prototype (NameError)."""
    sc = Scenario("trade_merchant_refuses_stolen")
    _to_barnabys_counter(sc, hours=8).capture()
    sc.press(K.K_e).idle(2).capture()
    sc.press(K.K_TAB).press(K.K_UP).idle(1).capture()         # our column, wrap to the stolen goblet
    sc.press(K.K_RETURN).idle(3).capture()                    # refused: "Guards!" - the menu closes
    sc.idle(420).capture_last()                               # ~7 s: a guard comes running and settles it (a fine)
    return sc


def trade_merchant_buy() -> Scenario:
    sc = Scenario("trade_merchant_buy")
    _to_barnabys_counter(sc, hours=8)
    sc.press(K.K_e).idle(2).capture()                        # stock shows Fresh Bread (4), Crisp Apple (6), torch; no keys
    sc.press(K.K_RETURN).press(K.K_RETURN).idle(2).capture() # two loaves -> "Fresh Bread (2)" in our bag
    sc.press(K.K_DOWN).press(K.K_RETURN).idle(2).capture()   # an apple
    sc.press(K.K_DOWN).press(K.K_RETURN).idle(2)             # the torch (25g worth, 31.2 asked): just affordable
    sc.press(K.K_TAB).press(K.K_RETURN).idle(2).capture()    # our column: sell the Iron Lockpick
    sc.press(K.K_DOWN).press(K.K_RETURN).idle(2).capture()   # the goblet: refused (stolen), menu closes
    sc.idle(420).capture_last()                              # ~7 s: the guard he shouted for arrives; broke -> jail
    return sc


def shop_closed_at_breakfast() -> Scenario:
    sc = Scenario("shop_closed_at_breakfast")
    go_to_house_door(sc).idle(2).capture()     # "[E] Open Door" (unlocked while he's home)
    sc.press(K.K_e).idle(2)
    walk(sc, A, 0.5)                           # x -> -39.5: clear of his table (facing +z, "left" is +x)
    walk(sc, W, 9.0)                           # z -> 12, level with Barnaby at his table seat (-41.5, 12)
    turn(sc, 90)                               # face -x: he is 2 away
    pitch(sc, -20).idle(2).capture()           # "[E] Talk to Shopkeeper Barnaby"
    sc.press(K.K_e).idle(2).capture_last()     # "The shop is closed right now..."
    return sc


def shopkeeper_commute() -> Scenario:
    """Watch Barnaby leave for work at 08:00 and come home at 20:00 from the street outside
    his house: standing at (-66, 2) facing south-west puts his front door in view; frames
    are captured twice a second."""
    sc = Scenario("shopkeeper_commute")
    sc.place(-66.0, 2.0, 219.0)                # his door (-70, 7) straight ahead
    sc.idle(2).capture()
    sc.idle(19, dt_ms=1000)                    # most of the way to 08:00
    for _ in range(44):
        sc.idle(1, dt_ms=500).capture()        # 22 s: hour tick + the walk to work
    sc.wait_hours(11)
    sc.idle(6, dt_ms=1000)
    for _ in range(44):
        sc.idle(1, dt_ms=500).capture()        # 20:00 closing time, walk home, dinner
    sc.idle(2).capture_last()
    return sc


def home_burglary() -> Scenario:
    """09:00: Barnaby is at work and his front door is locked - pick it and walk in."""
    sc = Scenario("home_burglary")
    sc.wait_hours(2)
    go_to_house_door(sc).idle(2).capture()     # "[Locked]"
    sc.press(K.K_LCTRL).idle(20).capture()     # crouch -> "[E] Pick Lock"
    sc.press(K.K_e).idle(2)
    rotate_to_sweet_spot(sc, 0)
    sc.idle(1).capture()
    sc.press(K.K_SPACE).idle(2).capture()      # open
    walk(sc, W, 6.0, crouched=True).idle(2).capture_last()    # inside (sneaking is slow): bed and table
    return sc


def pickpocket_keys_at_night() -> Scenario:
    """22:00: pick the sleeping shopkeeper's locked front door, sneak in and lift his keys."""
    sc = Scenario("pickpocket_keys_at_night")
    sc.wait_hours(15)                          # 22:00
    sc.press(K.K_1)                            # torch lit: moving in the dark without one is what draws attention
    go_to_house_door(sc, wait_seconds=30).idle(2).capture()     # 23:15, the night watch far off east: "[Locked]"
    sc.press(K.K_1).press(K.K_LCTRL).idle(20)  # torch out, crouch: walking upright would wake him (noise radius 14 vs 4.3)
    sc.press(K.K_e).idle(2)                    # pick the lock
    rotate_to_sweet_spot(sc, 0)
    sc.press(K.K_SPACE).idle(2).capture()      # open
    walk(sc, A, 1.0, crouched=True)            # x -> -69 (facing +z, "left" is +x)
    walk(sc, W, 10.25, crouched=True)          # z -> 13.25, level with Barnaby asleep on his bed at (-67.3, 13.4)
    turn(sc, -90)                              # face +x: he is 1.7 away
    pitch(sc, -20).idle(2).capture()           # "[E] Pickpocket Shopkeeper Barnaby"
    sc.press(K.K_e).idle(2).capture()          # his pockets: purse, keys, bread, apples, torch (every dip rolls against
    #                                            the onlookers: with the seeded rolls as they fall, this one goes unfelt)
    sc.press(K.K_DOWN).press(K.K_RETURN).idle(2).capture()   # shop key
    sc.press(K.K_RETURN).idle(2).capture()     # house key
    sc.press(K.K_TAB).press(K.K_DOWN).press(K.K_DOWN).press(K.K_RETURN).idle(2).capture()   # plant the stolen goblet on him
    sc.press(K.K_ESCAPE).idle(2).capture_last()
    return sc


def pickpocket_guard() -> Scenario:
    """Pick the pockets of the guard on post - awake, he catches you; unable to pay, you go to jail."""
    sc = Scenario("pickpocket_guard")
    go_to_guard_post(sc)
    pitch(sc, -20).idle(2).capture()           # "[E] Talk to Guard Orrin"
    sc.press(K.K_LCTRL).idle(20).capture()     # crouch: "[E] Pickpocket Guard Orrin"
    sc.press(K.K_e).idle(2).capture()          # into his pockets (he is no bystander to his own robbery: nobody else about)
    sc.press(K.K_DOWN).press(K.K_RETURN).idle(2).capture()   # guard key: success or a fumble (seeded roll: he feels the hand)
    sc.press(K.K_RETURN).idle(2).capture()     # (a fumble closes the menu; a second press is then ignored)
    sc.idle(420).capture_last()                # ~7 s: the chase's delay passes and he settles it
    return sc


def guards_respond() -> Scenario:
    """Rob Pell at breakfast: he cries for help and the guards nearby come running - then the fine."""
    sc = Scenario("guards_respond")
    sc.idle(6, dt_ms=1000)                     # the morning patrol sets off
    sc.place(25.5, 3.0, 180.0)                 # in line with Pell's door (hinge (24, 6), leaf 24..27), facing it
    sc.idle(2).capture()                       # "[E] Open Door" - unlocked while he is home
    sc.press(K.K_e).idle(2)
    walk(sc, W, 7.0)                           # z -> 10, level with Pell at his table seat (23.5, 10.2)
    turn(sc, 180)
    sc.press(K.K_e).idle(1)                    # shut the door behind us (a passing patrol would tell us off)
    turn(sc, -90)                              # face -x: Pell is 2 away
    pitch(sc, -20).idle(2).capture()           # "[E] Talk to Pell" (trespassing!)
    sc.press(K.K_LCTRL).idle(20)
    sc.press(K.K_e).idle(2).capture()          # crouched: his pockets
    sc.press(K.K_DOWN).press(K.K_RETURN).idle(2)              # the lift: a fumble (seeded) - "Thief! Guards!" and out we go
    sc.press(K.K_ESCAPE).idle(1)
    sc.press(K.K_LCTRL)
    pitch(sc, 20)
    turn(sc, 90)                               # face -z... out of the house
    for _ in range(24):
        sc.idle(1, dt_ms=500).capture()        # the guards arrive, spot us and settle the account
    sc.idle(2).capture_last()
    return sc


def jail_and_release() -> Scenario:
    """Too poor to pay a fine: jailed, items confiscated, released two days later."""
    sc = Scenario("jail_and_release")
    _into_the_hideout(sc)
    sc.press(K.K_e).idle(2)                    # trade with Jae: buy picks (25g asked apiece) to go nearly broke
    sc.press(K.K_RETURN).press(K.K_RETURN).idle(1)              # two Master Lockpicks: 10g left
    sc.press(K.K_ESCAPE).idle(1)
    go_to_guard_post(sc)                       # up the hill: Orrin on his post 2 units east
    pitch(sc, -20).press(K.K_LCTRL).idle(20)
    sc.press(K.K_e).idle(2).capture()          # into his pockets
    sc.press(K.K_DOWN).press(K.K_RETURN).idle(3).capture()     # the lift: a fumble - he has us, and what is left in our purse can't cover it
    sc.idle(540).capture()                     # ~9 s: charged, jailed (controls frozen, the fade to black) and back in: the cell
    sc.press(K.K_t).idle(2).capture()          # [T] in a cell asks: serve it all, just an hour, or cancel
    sc.press(K.K_t).idle(2).capture()          # [T] again: just the one hour
    sc.press(K.K_t).idle(1)
    sc.press(K.K_RETURN).idle(3).capture()     # [ENTER]: wait out the sentence -> released at 06:00, bars open
    # Out through the open bars (the leaf folds north along x = 54) to the confiscation lockbox
    # (unlocked for us) in the office's south-west corner and take our things back.
    sc.idle(4, dt_ms=1000)                     # let Hesk take up his post (people are solid now)
    turn(sc, 180)                              # we were put in facing the back wall; face -z, the gate
    walk(sc, W, 5.6)                           # z -> 7.4, through the bars and past the end of the open leaf
    turn(sc, -90)                              # face -x
    walk(sc, W, 2.7)                           # x -> 52.3, round the leaf into the office
    turn(sc, -90)                              # face +z
    walk(sc, W, 6.8)                           # z -> 14.2, past Hesk at his post (50.8, 10.5)
    turn(sc, 90)                               # face -x: the lockbox (x 48.8..50.3, z 13.5..15) is 2 units away
    pitch(sc, -25).idle(2).capture()           # "[E] Search Confiscation Lockbox"
    sc.press(K.K_e).idle(2).capture()          # top row: the goblet we stole long ago - confiscated for good, theft to take
    sc.press(K.K_DOWN).press(K.K_RETURN).press(K.K_RETURN).idle(1).capture()   # "Took back your ..." - ours, not stolen goods
    sc.press(K.K_ESCAPE).idle(1)
    pitch(sc, 25)
    turn(sc, 90)                               # face -z: the jail door (hinge (51.5, 6), leaf 51.5..54.5) straight ahead
    walk(sc, W, 6.0).idle(2).capture()         # z -> ~8.2: "[E] Open Door" - the jail door is never locked
    sc.press(K.K_e).idle(2)
    walk(sc, W, 5.0, sprint=True)              # out of the jail house onto the street; Hesk goes back to work
    sc.idle(3).capture_last()
    return sc


def curfew_eviction() -> Scenario:
    """A welcome guest overstays: at 22:00 Jae asks the player to leave his den, asks again,
    and finally shows them the door (fade to black, outside)."""
    sc = Scenario("curfew_eviction")
    sc.place(-12.0, 4.5, 180.0)
    sc.press(K.K_e).idle(1)                    # hideout door
    walk(sc, W, 6.0).idle(2)                   # z -> 10.5, inside (Jae trusts us here)
    for _ in range(15):
        sc.press(K.K_t, dt_ms=16)              # 07:00 -> 22:00 (each [T] is an hour)
    sc.idle(2).capture()                       # Jae, at his supper: "It's late. Time for you to go"
    for _ in range(20):
        sc.idle(1, dt_ms=1000)                 # we linger; he asks again, then throws us out
    sc.idle(2).capture()                       # mid-fade
    sc.idle(90).capture_last()                 # outside the hideout door, facing away from it
    return sc


def wait_key() -> Scenario:
    sc = Scenario("wait_key")
    sc.idle(2).capture()
    sc.press(K.K_t).idle(2).capture()          # 08:00: Barnaby on his way to the shop
    sc.press(K.K_t).press(K.K_t).idle(2).capture_last()
    return sc


def shoplifting() -> Scenario:
    """Steal from the shelves while Barnaby watches from the counter."""
    sc = Scenario("shoplifting")
    sc.wait_hours(1)
    sc.idle(24, dt_ms=1000)                    # shop open, Barnaby behind the counter
    go_to_shop_door(sc, wait_seconds=2).idle(2)
    sc.press(K.K_e).idle(2)                    # open the shop door
    walk(sc, W, 5.5)                           # z -> -9.5, on the shop floor in front of the counter (z -11.2..-10.6)
    turn(sc, -90)                              # face -x: slip around the west end of the counter (x -16..-10)
    walk(sc, W, 3.4)                           # x -> -16.9 (past the open door leaf; Dorran the porter works at the east end)
    turn(sc, 90)                               # face -z
    walk(sc, W, 3.0)                           # z -> -12.5: behind the counter (Barnaby objects!)
    sc.idle(2).capture()                       # the west shelf (x -17.5..-14.5, z -15.5..-14.7): "[E] Search Shop Shelf"
    sc.press(K.K_e).idle(2).capture()          # shelf contents
    sc.press(K.K_RETURN).idle(2).capture()     # candlestick: "Stole ... " and Barnaby (facing the shop floor) shouts
    sc.press(K.K_RETURN).press(K.K_RETURN).idle(2).capture()    # two bottles of wine -> "Bottle of Wine (2)" [STOLEN]
    sc.press(K.K_TAB).press(K.K_RETURN).idle(2).capture()       # leave our lockpick on the shelf
    sc.press(K.K_ESCAPE).idle(2).capture_last()
    return sc


def guard_shifts() -> Scenario:
    """Watch the guard house across a day: Orrin in from his post for his meal at 13:00 and out
    on patrol at 14:00, Mira in at 21:00 and out on the night watch at 22:00."""
    sc = Scenario("guard_shifts")
    sc.place(42.0, -37.0, 60.0)                # Upper Street: the guard house door and the post in view
    sc.idle(2).capture()
    sc.idle(24 * 6 - 12, dt_ms=1000)           # just before 13:00
    for _ in range(40):
        sc.idle(1, dt_ms=500).capture()        # Orrin leaves the post for the table
    sc.idle(24 * 8 - 20, dt_ms=1000)           # just before 21:00
    for _ in range(40):
        sc.idle(1, dt_ms=500).capture()        # Mira comes in for supper before the night watch
    sc.idle(2).capture_last()
    return sc


def save_and_load() -> Scenario:
    sc = Scenario("save_and_load")
    sc.press(K.K_F9).idle(2).capture()         # no save file yet
    sc.press(K.K_1).press(K.K_LCTRL)
    walk(sc, W, 3.0, crouched=True)
    turn(sc, 30)
    sc.press(K.K_F5).idle(2).capture()         # saved
    sc.press(K.K_1).press(K.K_LCTRL)
    walk(sc, S, 5.0)
    sc.wait_hours(2).idle(2).capture()         # 09:00, Barnaby at the shop
    sc.press(K.K_F9).idle(2).capture()         # loaded: 07:00, torch on, crouching, position + yaw restored
    sc.idle(5).capture_last()
    return sc


def full_day_cycle() -> Scenario:
    sc = Scenario("full_day_cycle")
    turn(sc, -45)
    for _ in range(25):                        # 07:00 -> 08:00 next day
        sc.idle(23, dt_ms=1000).capture().idle(1, dt_ms=1000)
    sc.idle(2).capture_last()
    return sc


def night_stealth() -> Scenario:
    sc = Scenario("night_stealth")
    sc.press(K.K_1).idle(2).capture()          # torch on (day)
    sc.press(K.K_LCTRL).idle(2).capture()      # crouch ...
    sc.press(K.K_LCTRL).idle(1)                # ... and stand again: crouching in a guard's view is suspicious even by day
    sc.wait_hours(13).idle(2).capture()        # 20:00 - night, NPC torches; Mira stands the post outside the guard house
    go_to_guard_post(sc).idle(2).capture()     # facing her, two units away
    sc.press(K.K_1).idle(1)                    # torch out next to the guard
    sc.hold([A, D], 120).capture()             # "moving" in the dark beside her: heard + seen, suspicion builds
    sc.press(K.K_1).idle(2).capture()          # torch on: no longer suspicious
    sc.idle(30).capture_last()
    return sc


def escape_quits() -> Scenario:
    """ESC in the explore mode. The prototype quits outright; the app opens the pause menu,
    where UP wraps to "Exit to Menu", ENTER asks for confirmation, DOWN + ENTER confirms and
    shows the start menu, and there UP wraps to "Exit" and ENTER quits. Headless (``Game``
    only, as in the goldens) ESC is a no-op in the explore mode, so the menu keys do nothing."""
    sc = Scenario("escape_quits", end_with_quit_event=False)
    sc.idle(3).press(K.K_ESCAPE).capture()                         # frame 3: pause menu
    sc.idle(2).press(K.K_UP).press(K.K_RETURN).capture()           # frame 7: "Exit to Menu?" confirmation
    sc.press(K.K_DOWN).press(K.K_RETURN).capture()                 # frame 9: start menu
    sc.press(K.K_UP).press(K.K_RETURN).capture_last()              # frame 11: Exit
    return sc


def mode_edge_cases() -> Scenario:
    sc = Scenario("mode_edge_cases")
    sc.press(K.K_LEFT).press(K.K_SPACE).press(K.K_UP).press(K.K_RETURN).press(K.K_x).idle(1).capture_last()   # ignored in explore
    return sc


def equipment_view() -> Scenario:
    """Dead of night: empty the smith's rack with [X], then dress up in the [Tab] view - a dagger
    at the hip, a helm, a mail shirt - turning the figure to look, and try to wear a lump of ore."""
    sc = Scenario("equipment_view")
    sc.wait_hours(19)                          # 02:00: the smith asleep indoors, the night watch far off
    sc.press(K.K_1)                            # torch lit: the law after dark
    sc.place(-62.8, -12.2, 0.0)                # under the shed: the rack (x -64.4..-61.2, z -13.9..-13.2) a pace ahead
    sc.idle(2).capture()                       # "[E] Search Weapon Rack"
    sc.press(K.K_e).idle(2).capture()          # the rack: dagger, sword, cap, helm, mail, jerkin, ore, mallet - every row "THEFT"
    sc.press(K.K_x).idle(2).capture()          # [X]: everything, one unit at a time (Stealing XP, the alarm if seen)
    sc.press(K.K_ESCAPE).idle(1)
    sc.press(K.K_TAB).idle(2).capture()        # the equipment view: the bag, the figure in its boots, the empty slots
    sc.press(K.K_DOWN).press(K.K_DOWN).press(K.K_DOWN)        # the Iron Dagger
    sc.press(K.K_RETURN).idle(2).capture()     # taken up: at the hip
    sc.press(K.K_DOWN).press(K.K_DOWN).press(K.K_DOWN)        # the Iron Helm
    sc.press(K.K_RETURN).idle(2).capture()
    sc.press(K.K_DOWN).press(K.K_RETURN).idle(2).capture()    # the Chain Shirt: "Heard" goes up - steel clanks
    for _ in range(6):
        sc.press(K.K_LEFT)                     # turn the figure a quarter round
    sc.idle(2).capture()
    sc.press(K.K_DOWN).press(K.K_DOWN).press(K.K_RETURN).idle(1)   # the ore: "You cannot wear or wield Iron Ore (4)."
    sc.press(K.K_UP).press(K.K_UP).press(K.K_RETURN).idle(1).capture()   # the mail again: off
    sc.press(K.K_ESCAPE).idle(2).capture_last()               # back to the street, the dagger and helm still on
    return sc


def lockpick_escape_and_keys() -> Scenario:
    sc = Scenario("lockpick_escape_and_keys")
    go_to_shop_door(sc).press(K.K_LCTRL).idle(20)
    sc.press(K.K_e).idle(2).capture()          # picking
    sc.press(K.K_RIGHT).press(K.K_F5).press(K.K_TAB).press(K.K_r).idle(2).capture()   # world/menu keys are ignored here
    sc.press(K.K_ESCAPE).idle(2).capture()     # cancel
    sc.press(K.K_e).idle(2).capture()          # re-enter: pick reset to 0, fresh sweet spot
    sc.press(K.K_F9).idle(2).capture_last()    # load is refused inside the mini-game (and the log says why)
    return sc


# --------------------------------------------------------------------------- the new town
def _to_the_inn_bar(sc: Scenario) -> Scenario:
    """Walk into the Drowsy Dragon (door hinge (-34, -6), leaf -34..-31, opens inwards) and up to the
    bar (x -39.5..-35, z -11.8..-11.2); Hilde stands behind it at (-36, -12.8) facing the room."""
    sc.place(-32.5, -3.0, 0.0)                 # in line with the doorway, facing it
    sc.idle(1)
    sc.press(K.K_e).idle(1)                    # "[E] Open Door" (the inn is open all day)
    walk(sc, W, 7.2)                           # z -> -10.2, just in front of the bar
    turn(sc, -90)                              # face -x
    walk(sc, W, 3.5)                           # x -> -36, straight across the bar from the keeper
    turn(sc, 90)                               # face -z: Hilde is 2.6 away behind the bar
    return pitch(sc, -15).idle(2)


def inn_rent_a_room() -> Scenario:
    """Rent a room for the night from the innkeeper: the room's door unlocks for us, the bed is
    ours to sleep in, and at noon the next day the keeper wants the room back."""
    sc = Scenario("inn_rent_a_room")
    _to_the_inn_bar(sc).capture()              # "[E] Talk / Trade with Hilde the Innkeeper"
    sc.press(K.K_e).idle(2).capture()          # her stock: the rooms (services) then food
    sc.press(K.K_RETURN).idle(2).capture()     # "A Room for the Night": paid, room one unlocked and opened for us
    sc.press(K.K_RETURN).idle(2).capture()     # and a second night on top: checkout the day after tomorrow
    sc.press(K.K_ESCAPE).idle(1)
    pitch(sc, 15)
    walk(sc, D, 2.7)                           # x -> -33.3, in line with room one's doorway (x -34.1..-32.5 at z = -16.5)
    walk(sc, W, 4.8).idle(2).capture()         # z -> -15: the open door of our room ahead
    walk(sc, W, 4.0).idle(2).capture()         # inside room one (no trespass warning: it is ours)
    turn(sc, 180).idle(1)
    sc.press(K.K_e).idle(2).capture()          # shut the door behind us
    turn(sc, 180)
    sc.press(K.K_t).idle(2).capture()          # [T]: an hour passes in our own room
    pitch(sc, -35).idle(2).capture()           # the bed (-33.25, -20) is right in front of us: "[E] Sleep Until Night" (it is day)
    sc.press(K.K_e).idle(2).capture()          # the slow fade to black ...
    sc.idle(120, dt_ms=50).capture()           # ... the afternoon passes ...
    sc.idle(60).capture()                      # ... and it is 20:00 in our room: "[E] Sleep Until Morning" now
    sc.press(K.K_e).idle(2).capture()          # sleep again: the night passes ...
    sc.idle(120, dt_ms=50)
    sc.idle(60).capture_last()                 # ... and it is 06:00
    return sc


def smith_trade() -> Scenario:
    """The blacksmith only deals in weapons, armour and ore: he buys our goblet? No - he refuses
    it and every other trinket, but sells a dagger."""
    sc = Scenario("smith_trade")
    sc.wait_hours(1)                           # 08:00: Gareth goes out to the anvil under the shed
    sc.idle(10, dt_ms=1000)
    sc.place(-60.0, -7.0, 292.0, -15.0)        # under the shed: Gareth at the anvil (-62, -7.8), two paces off
    sc.idle(2).capture()                       # "[E] Talk / Trade with Gareth the Smith"
    sc.press(K.K_e).idle(2).capture()          # his rack: daggers, swords, helms, ore
    sc.press(K.K_RETURN).idle(2).capture()     # an Iron Dagger (40g)
    sc.press(K.K_TAB).press(K.K_DOWN).press(K.K_RETURN).idle(2).capture()   # our boots - worn: "Sell the boots you are wearing?"
    sc.press(K.K_RETURN).idle(2).capture()                                   # yes - and the smith: "not my trade" (boots stay on)
    sc.press(K.K_ESCAPE).idle(2).capture_last()
    return sc


def town_life() -> Scenario:
    """Stand by the well through the morning and watch the town go about its business: strollers,
    shoppers, the porter waiting for the store to open, neighbours stopping for a word (speech
    bubbles), the inn filling up."""
    sc = Scenario("town_life")
    sc.place(-6.0, -1.0, 70.0).idle(2).capture()      # by the well, looking along the street
    sc.idle(20, dt_ms=1000)                    # 07:20 .. 08:00: the street fills
    for _ in range(60):
        sc.idle(1, dt_ms=500).capture()        # 30 s around 08:00: everyone sets off for the day
    sc.wait_hours(2)                           # 10:00-ish: Tamsin shopping, Fenwick at the inn
    for _ in range(20):
        sc.idle(1, dt_ms=500).capture()
    sc.idle(2).capture_last()
    return sc


def tower_climb() -> Scenario:
    """Dead of night: pick the guard house door, creep past the sleeping watch, pick the door
    into the tower and climb the winding stair to the look-out platform."""
    sc = Scenario("tower_climb")
    sc.skill("Lockpicking", 45)                # an old hand: a Strong lock (the watch's) is no trouble at 45
    sc.wait_hours(19)                          # 02:00: Mira walks the night watch, the rest sleep
    sc.press(K.K_1)                            # torch lit (the law after dark; Mira is at the far end of her round)
    sc.place(50.0, -40.0, 0.0)                 # up the hill: the guard house door (hinge (48.5, -43), leaf 48.5..51.5) ahead
    sc.press(K.K_1).press(K.K_LCTRL).idle(20)  # torch out, crouch
    sc.idle(2).capture()                       # "[E] Pick Lock"
    sc.press(K.K_e).idle(2)
    rotate_to_sweet_spot(sc, 0)
    sc.press(K.K_SPACE).idle(2).capture()      # the guard house door swings open
    walk(sc, W, 4.2, crouched=True)            # z -> -44.2, through the doorway and in
    walk(sc, D, 2.0, crouched=True)            # x -> 52: east of the table (x 48.5..51.5), out of earshot of the bunks
    walk(sc, W, 8.0, crouched=True)            # z -> -52.2, past the sleepers to the north wall
    walk(sc, A, 1.0, crouched=True)            # x -> 51, in line with the tower door (leaf 50..52)
    sc.idle(2).capture()                       # "[E] Pick Lock" again: the way into the tower
    sc.press(K.K_e).idle(2)
    rotate_to_sweet_spot(sc, 2)                # (the lock drew its spot, then the pins their roll: this is the third draw)
    sc.press(K.K_SPACE).idle(2).capture()      # it swings open towards us, into the guard house
    walk(sc, W, 2.1, crouched=True)            # z -> -54.3, through the doorway (still quiet: they are asleep next door)
    turn(sc, -90)                              # face -x
    walk(sc, W, 1.7, crouched=True)            # x -> 49.3, the foot of the stair
    sc.press(K.K_LCTRL).idle(20)               # stand up for the climb
    turn(sc, 90)                               # face -z (north): up the west wall
    for leg in range(10):                      # ten legs of the winding stair (two and a half turns)
        walk(sc, W, 3.4)                       # 49.3 <-> 52.7 / -54.3 <-> -57.7: the steps lift us as we go
        if leg in (3, 7):
            sc.capture()
        turn(sc, 90)
    walk(sc, W, 1.9).idle(2).capture()         # from the last step onto the platform
    turn(sc, 90)
    pitch(sc, -20).idle(2).capture_last()      # the town from the top of the tower
    return sc


def tutorial_start() -> Scenario:
    """A new game: the urchin listens to Jae, lights a torch (the law), crosses the street, picks
    the store's lock, takes the candlestick - a watchman steps in - is jailed, waits out the
    night, gets their things from the lockbox, puts on their boots in the [Tab] view and waits
    for nightfall with [T]."""
    sc = Scenario("tutorial_start", tutorial=True)
    sc.idle(2).capture()                       # in the hideout: Jae's first line, the tutorial panel
    sc.idle(14, dt_ms=1000).capture().idle(1)  # the debt explained; the torch law: "[1]"
    sc.press(K.K_1).idle(2).capture()          # lit: "cross the street" - and go at once
    turn(sc, 90)                               # we stood facing -x; face -z: the door is 3.5 ahead
    walk(sc, W, 1.5)                           # z -> 9.5
    sc.press(K.K_e).idle(1)                    # open it (folds out onto the street along x = -13.5)
    walk(sc, W, 13.5)                          # z -> -4, across the street to the north side
    sc.place(-13.5, -4.0, 0.0)                 # in line with the store's door, two paces off, facing it
    sc.press(K.K_LCTRL).idle(20).capture()     # crouch: "[E] Pick Lock"
    sc.press(K.K_e).idle(2).capture()          # the mini-game, the lesson in the panel
    rotate_to_sweet_spot(sc, 0)
    sc.press(K.K_SPACE).idle(2).capture()      # open; "inside: the west shelf"
    walk(sc, W, 5.8, crouched=True)            # z -> -9.8, inside and past the open leaf - still crouched: quiet feet
    turn(sc, -90)                              # face -x
    walk(sc, W, 3.1, crouched=True)            # x -> -16.6, clear of the counter's end
    turn(sc, 90)                               # face -z
    walk(sc, W, 3.2, crouched=True)            # z -> -13: the west shelf a pace ahead, the counter between us and the street
    sc.idle(2).capture()                       # "[E] Search Shop Shelf"
    sc.press(K.K_e).idle(2).capture()          # every row THEFT; the candlestick first
    sc.press(K.K_RETURN).idle(2).capture()     # taken - and seen: the shelf screen closes, a watchman shouts
    sc.idle(90).capture()                      # the view has swung round to him: "sneaking about!"
    sc.idle(90).capture()                      # three seconds: he lays hands on us - the bill
    sc.idle(400).capture()                     # the fade: the cell
    sc.press(K.K_t).idle(2).capture()          # "wait out your sentence (1 day left)?"
    sc.press(K.K_RETURN).idle(3).capture()     # serve it: released at six, bars and lockbox open
    sc.idle(4, dt_ms=1000)
    turn(sc, 180)                              # put in facing the back wall; face the gate
    walk(sc, W, 5.6)                           # through the bars and past the open leaf
    turn(sc, -90)
    walk(sc, W, 2.7)
    turn(sc, -90)
    walk(sc, W, 6.8)                           # down the office to the lockbox
    turn(sc, 90)
    pitch(sc, -25).idle(2).capture()           # "[E] Search Confiscation Lockbox"
    sc.press(K.K_e).idle(2)
    sc.press(K.K_x).idle(2).capture()          # [X]: everything back
    sc.press(K.K_ESCAPE).idle(2).capture()     # "press [Tab]: your boots"
    sc.press(K.K_TAB).idle(2).capture()        # the view, the lesson along the bottom
    sc.press(K.K_DOWN).press(K.K_RETURN).idle(2).capture()    # the boots, on
    sc.press(K.K_TAB).idle(2).capture()        # "thieves work at night: [T]"
    for _ in range(14):                        # 06:00 -> 20:00, an hour at a time
        sc.press(K.K_t).idle(1)
    sc.idle(2).capture()                       # night: "back to the store, properly this time"
    sc.press(K.K_1).idle(2).capture_last()     # the torch lit again for the street
    return sc


SCENARIOS: Dict[str, Callable[[], Scenario]] = {
    "idle_day": idle_day,
    "look_around": look_around,
    "walk_and_sprint": walk_and_sprint,
    "wall_collision": wall_collision,
    "hideout_door": hideout_door,
    "shop_lockpick": shop_lockpick,
    "mine_and_forge": mine_and_forge,
    "lockpick_out_of_picks": lockpick_out_of_picks,
    "trade_fence": trade_fence,
    "trade_merchant_buy": trade_merchant_buy,
    "trade_merchant_refuses_stolen": trade_merchant_refuses_stolen,
    "shop_closed_at_breakfast": shop_closed_at_breakfast,
    "shopkeeper_commute": shopkeeper_commute,
    "home_burglary": home_burglary,
    "pickpocket_keys_at_night": pickpocket_keys_at_night,
    "pickpocket_guard": pickpocket_guard,
    "guards_respond": guards_respond,
    "jail_and_release": jail_and_release,
    "curfew_eviction": curfew_eviction,
    "wait_key": wait_key,
    "shoplifting": shoplifting,
    "guard_shifts": guard_shifts,
    "save_and_load": save_and_load,
    "full_day_cycle": full_day_cycle,
    "night_stealth": night_stealth,
    "escape_quits": escape_quits,
    "mode_edge_cases": mode_edge_cases,
    "equipment_view": equipment_view,
    "tutorial_start": tutorial_start,
    "lockpick_escape_and_keys": lockpick_escape_and_keys,
    "inn_rent_a_room": inn_rent_a_room,
    "smith_trade": smith_trade,
    "town_life": town_life,
    "tower_climb": tower_climb,
}


def prototype_sell_stolen_crash() -> Scenario:
    """Legacy-only: reproduce the prototype's NameError. Uses the prototype's NPC
    placement (Barnaby at (10, -5) inside the shop, no collision)."""
    sc = Scenario("prototype_sell_stolen_crash")
    sc.wait_hours(1)                           # 08:00: the prototype's Barnaby is SHOPKEEPING
    turn(sc, 90)
    walk(sc, W, 22.0, sprint=True)             # through the shop wall to x=10 (prototype has no collision)
    walk(sc, D, 2.4)                           # z -> -3.6
    turn(sc, -90)
    pitch(sc, -30).idle(2)
    sc.press(K.K_e).idle(2)
    sc.press(K.K_UP).idle(1)                   # wraps to the stolen goblet
    sc.press(K.K_RETURN).idle(3).capture_last()
    return sc


LEGACY_ONLY: Dict[str, Callable[[], Scenario]] = {"prototype_sell_stolen_crash": prototype_sell_stolen_crash}
ALL_SCENARIOS: Dict[str, Callable[[], Scenario]] = {**SCENARIOS, **LEGACY_ONLY}

# Scenarios whose *rendering and player-visible state* must still match the frozen
# prototype frame for frame (nothing in them touches the redesigned systems).
PROTOTYPE_IDENTICAL = ("idle_day", "look_around", "mode_edge_cases")

# The prototype crashes in this situation (NameError); the refactor's handling of the
# same situation is covered by ``trade_merchant_refuses_stolen`` (goldens) and unit tests.
CRASHING_IN_PROTOTYPE = ("prototype_sell_stolen_crash",)
