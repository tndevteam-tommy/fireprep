# Unprepared

A rogue-like stealth & theft RPG built with **pygame** and **PyOpenGL**.
Sneak through Emberhollow, a walled market town on a hillside under the mountains - shops,
an inn, a smithy, a bakery, a chapel, cottages, two manors of the rich and the watchtower at
the top of the hill - and out through its gate to the village, the woods, the shore, three
caves dug into the hills and the old keep beyond. Pick locks, pickpocket, loot shelves and
strongboxes, search hiding places, fence stolen goods and stay out of sight of the town watch,
who remember what you did - and will chase you down to fine or jail you for it. Thirty-three
townsfolk - twenty-six civilians and seven of the watch - and twenty people out in the
country live their days around you, trading, working, strolling, drinking and gossiping in
speech bubbles; every one of the townsfolk has a story of their own to tell you.

```
python main.py
```

The game opens on a start menu (Continue / New Game / New Game - skip the
tutorial / Load Game / Graphics Settings / Exit - *Continue* appears once there
is a save and picks up the newest one) under the game's logo, while the camera
drifts slowly down the streets of a live Emberhollow on a sunny afternoon - the
townsfolk go about their day behind the menu. On first launch it benchmarks the
renderer for a moment and picks a graphics preset; change it any time from the
menu.

**New Game** starts with the tutorial (see *The tutorial* below): a street
urchin in Jae's debt, led through the torch law, locks, theft, the cells, the
[Tab] view, waiting for night, the fence, the anvil and the inn before the town
opens up - and the town gate with it: until then the gate is barred and its
guards turn you away. Then Jae has a job for you (talk to him). *New Game - skip
the tutorial* starts on the well square with the usual kit and the gate open.

| Key | Action |
| --- | --- |
| `W A S D` / mouse | move / look |
| `Left Shift` / `Left Ctrl` | sprint / toggle crouch (the camera drops, you move at half speed; crouching is the *theft* stance - no sprinting while crouched) |
| `E` | interact (`E` or `Esc` also puts down whatever you are reading, wherever you are looking): open/close doors (whatever part of the door is under the crosshair; its name shows over the prompt), talk or trade, search shelves, tables, lockboxes and the bones of the dead, read notices, signposts, plaques and journals (again, or walk off, to put them down), mine ore (with a pickaxe), forge lockpicks at the anvil (ore + a mallet), sleep in a bed that is yours (until night by day, until morning at night) |
| crouch + `E` | pick a locked door or lockbox, pick an NPC's pockets (within 2.5 units), or search a hiding place - loose stones, a floorboard, a cask nobody drinks from - which only somebody crouched and close (2.2 units) finds at all |
| `1` | toggle torch (carrying a lit one is the law after dark) |
| `T` | wait one hour - with the town's eyes open: a keeper closing up, a host at curfew or a guard with a grudge cuts the wait short (in a jail cell: asks whether to wait out the whole sentence - however many days it is) |
| `F5` / `F9` | quick save / quick load (`savegame.json`); named saves from the pause menu go to `saves/` next to it. Everything lives in the per-user data directory (see *Saves* below). Saving is refused (with a line in the log, and the pause-menu entry greyed out) while you are being marched to the cells or shown the door; loading drops whatever the old session was in the middle of |
| `F3` | performance overlay (fps, frame phases, draw counts) |

The controls panel is only on screen during the tutorial; the carried inventory is
shown in the `Tab` view and the trade / container screens (there is no inventory
panel on the HUD).
| `Esc` | pause menu (Resume / Save Game / Load Game / Graphics / Exit to Menu); inside a screen or question: back out |
| `Enter` / `Esc` (or `Y` / `N`) | answer a yes/no question |
| `Tab ↑ ↓ Enter Esc` | trade / pickpocket / container screens: Tab switches column, Enter moves one item (or grabs a handful of coins); `X` in a container screen takes everything (never from somebody's pockets) |
| `Tab` | (outside any screen) the **equipment view**: your figure on a stand, your pack with a cursor - `↑ ↓` select, `Enter` puts the item on or takes it off, `← →` turn the figure, `Tab`/`Esc` close. The world stands still while it is open |
| `` ` `` (shown as `~`) | show / hide the **World Events** log (hidden by default; while it is hidden, lines that concern you pop up for a few seconds and fade) |
| `← → Space Esc` | lockpicking mini-game |

## Requirements

```
pip install -r requirements.txt          # pygame, PyOpenGL, numpy
pip install -r requirements-dev.txt      # + pytest (for the test suite)
```

GLU is **not** required (the renderer builds its own projection/view matrices).

### Saves

Saves and `graphics.json` are kept in a per-user directory, never in the
working directory the game was launched from:

| | |
| --- | --- |
| Windows | `%APPDATA%\Unprepared` |
| macOS | `~/Library/Application Support/Unprepared` |
| Linux | `$XDG_DATA_HOME/unprepared` (`~/.local/share/unprepared`) |

Set `UNPREPARED_DATA_DIR` to use another directory (a portable install). Saves
an older version left next to `main.py` are copied across on first start.
Files are written atomically (to a temporary file, then renamed into place), so
a crash mid-save never truncates a good save; a file that *is* corrupt shows up
in the menus as `(unreadable)` and loading it reports the problem instead of
crashing.
numpy is used for the NPC navigation grid. Lighting uses a small GLSL 1.20
shader when available and falls back to fixed-function lighting otherwise
(the *low* preset uses the fallback on purpose).

### Graphics settings

*Graphics Settings* (start menu and pause menu) holds a preset - **low**,
**medium** (the default) or **high** - and the individual settings it sets, each of
which can be changed on its own afterwards (`←/→` step a setting, `Enter` steps it up):

| Setting | |
| --- | --- |
| Quality preset | low / medium / high (*auto* while it is the one the benchmark picked) |
| Full screen | on: the desktop's resolution; off: the window size from the config. Switching re-opens the window in place |
| Draw distance | 60 - 1000 units (low 100, medium 160, high 700 - the whole realm) |
| Field of view, object / character fade, contrast | how wide the view is; how far out (as a share of the draw distance) things and people are drawn; the picture's contrast |
| Distance haze | how thick the day's air is towards the horizon (0 - 1.5; less: further to see) |
| Torch & lantern lights | lights per frame in the shader (2 - 16) |
| Anti-aliasing | off / 2x / 4x / 8x multisampling (the window re-opens to apply it) |
| Texture filtering | anisotropic filtering off / 2x - 16x (as far as the driver supports it) |
| Vertical sync | on / off (the window re-opens to apply it) |
| Shader lighting, stars, roofs & floors, performance overlay | the GLSL lighting (off: fixed-function), the night sky's stars, roofs and upper floors, the `F3` overlay |
| Re-detect best settings | run the benchmark again and pick the preset it suggests |

The *high* preset is meant for a proper graphics card; the first-launch benchmark
only picks it when the machine renders it in under 10 ms a frame.

## Commands

| Command | What it does |
| --- | --- |
| `python main.py` | play |
| `python tools/textures/gen_textures.py [--out DIR] [--seed N] [--only key ...]` | regenerate the procedural PNG textures in `assets/textures/` (89 tiles - ground, wall, building, wood, door, roof, stone, ore, cobble, the shop signs, stream, lake, sand, snow, cliff, cave rock, ashlar, wallpaper, paintings, ... - and the start screen's `logo`) |
| `python tools/benchmark.py [--frames N] [--presets low medium high] [--json out.json]` | open the window, render fixed camera views at each preset and print avg/worst frame times per phase plus the preset the game would auto-select |
| `python tools/visual_audit.py [--checks ...] [--limit N] [--near M]` | search the level for visual glitches without opening a window: faces fighting in one plane (z-fighting), things poking through walls, the ground through a floor, things floating, windows and paintings off their walls. Exits 1 if anything is found |
| `./run_tests.sh [fast]` | the whole test suite (`fast` = headless only); display tests run under `xvfb-run` automatically when there is no `$DISPLAY` |
| `pytest -m "not display"` | unit tests + scenario goldens (a few minutes; the scenario replays dominate) |
| `UNPREPARED_UPDATE_GOLDENS=1 pytest tests/regression` | re-record the scenario goldens after an intended behaviour change (review the diff) |
| `python -m tests.parity.render_gallery [scenario ...]` | render the capture frames of scripted scenarios to `tests/parity/_artifacts/gallery/` for eyeballing |

## The realm

* **Emberhollow** (`world/levels/town.py`, `manors.py`, `quarters.py`) stands on a hill: the
  gate at the bottom (south), the well in the middle, Market Street east-west through it,
  Upper Street above with the two manors, four houses across the way and the guard house, the
  watchtower on the knoll at the top. Every floor is level, the streets are not. The millstream
  runs through the west of the town under a bridge and on to the lake by the woods.
* **The edges of the map** are the land itself, never a wall: the Greyspine mountains rise
  behind the town to the north - a broken rampart at their foot, too steep to climb, craggy
  peaks behind it and the snow-capped high peaks beyond; the Thornwood thickens to
  an impassable thicket to the west and the Deepwood to the south; to the east a beach runs
  down to the sea, which is soon too deep to wade.
* **Beyond the gate** (`world/levels/wilds.py`): Thornbury village (cottages with flimsy
  latches, fields, a notice board), the Thornwood with the hunter's hut and Hrolf the hermit
  smith's forge (he buys ore; there are seams in the rocks by his clearing and in the quarry),
  the lake, Saint Wendel's ruined shrine to the south, Greyhold - the old keep on its rise by
  the shore - and three caves, dug down into the hills rather than built (`world/caves.py`):
  the **Howling Deep** under the crag at the mountain's foot (the bandit king's hall, his
  crew's bunks, a spring, a vault), the **Saltcove** in the bluff over the beach (the
  smugglers' den and store, their boat on the sand) and the **old mine** in the spur to the
  north-east, worked out and fallen in. Graded trails - never too steep to walk, with
  signposts at the forks - join the roads to every one of them. Caves are dark but for their
  fires and the glowing mushrooms; there is ore in their walls, bones in their corners
  (searchable: the dead keep a coin or two) and hiding places for those who crouch and look.
  Outlaws and knights keep their own peace: caught in their domain you are thrown out, your
  loot taken back and a quarter of your purse with it - no cells.
* **Locks** come in six tiers - Flimsy, Simple, Sturdy, Strong, Master, Legendary - and the
  mini-game shows the tier and your odds. **Every skill** runs 1 to 100 on one rule: your level
  plus what your gear adds, against the challenge (a lock's tier, an observer's vigilance, an
  item's rarity); a novice has ~1% against a legendary lock, a master with the right amulet
  finds it trivial.
* **Rare things** are kept where they are hardest to reach: dark clothing in manors, the
  smugglers' stash and the keep; the three legendary pieces in the Deep's vault behind the
  bandit king, the keep's treasury and a lady's strongbox. Epic and legendary pieces are
  secured as well as locked away - they need a real thief's hands.
* **Stories.** Talk to the townsfolk more than once and, at a civil hour, they tell you about
  themselves a line at a time - their trade, their habits, their secrets (the old crier took
  an arrow in the knee; the butler keeps a second ledger). Notices, journals and letters in
  the caves tell the rest: who the bandit king was, what happened in the mine, why the
  smugglers' old man stares at a crack in the wall.
* **Jae's job**: steal Lord Vane's ravenfeather cloak; Jae pays, and you keep the cloak.

## Project layout

```
main.py                 thin entry point
unprepared/             the game package (see docs/ARCHITECTURE.md)
  config.py             every tunable number in one place (frozen dataclasses)
  graphics.py           GraphicsSettings: presets high/medium/low (draw distance, MSAA, full screen, ...), persisted next to the save
  perf.py               FrameProfiler, renderer detection, preset recommendation
  ui.py                 Menu / MenuItem (pure data + callbacks)
  core/                 3D math, message log, day/night rules, dialogue (phrasings, conversations)
  entities/             Item (ownership, gear), Door (leaf / bars), Player (+ equipment), NPC (+ memories, purse), SkillTree,
                        equipment (slots, the gear catalogue, outfits as worn items)
  world/                data-driven levels (areas, buildings, roads, lights, jail, inn, readables), World state
                        (indoors, solid people, curfews, conversations, pantries), the inn, NPC navigation grid
                        (A*), terrain (hills, pads, streams, lakes, the sea, graded trails, caves' floors, holes),
                        caves (chambers dug into the hills), placement, spatial index, builders, save slots (v9);
                        levels/: town, manors, quarters (the townsfolk's houses), stories, wilds
  systems/              movement, collision, targeting (crosshair ray), transfer screens, lockpicking,
                        progression (one XP/odds rule for every skill), stealth checks, commerce,
                        crafting, justice (fines, eviction, jail)
  game.py               headless simulation + action API (no pygame / OpenGL imports)
  quests.py             Jae's job after the tutorial (offer, guide panel + arrow, hand-in)
  attract.py            the start screen's live town: the camera's stroll through the streets
  input/                key bindings, pygame event -> action mapping
  rendering/            textures, primitives (doors, bars, tables), atmosphere, shaders, lighting (room-bounded
                        lights that spill through open doors), gear (every piece of equipment as a model on a
                        figure), models, sky (sun halo, stars), buildings (floors, ceilings, roofs, fascia,
                        open sheds, spires), roads (cobbles), terrain (the heightfield with rock, sand and snow,
                        draped paving and trails, running water, the sea), structures and furnishings (from
                        wells and bridges to chandeliers, bones and banners), scene (+ the equipment preview),
                        text, HUD (speech bubbles, door names, reading, the [Tab] view) + menus
  app.py                pygame window, screens (start / playing / pause / save / load / confirm / graphics), main loop
assets/textures/        PNG textures (missing ones render as a magenta placeholder)
tools/                  texture generator CLI, benchmark CLI
tests/                  unit, regression (goldens), parity (vs the frozen prototype), display suites
docs/ARCHITECTURE.md    design notes, extension guide, change log
```

## The tutorial

A new game begins at ten at night in the hideout. Jae found you in the gutter,
fed you, and wants paying back: the silver candlestick from Barnaby's store.
The gold-framed tutorial panel (under the controls panel; beside a shelf, deal
or lock screen while one is open) tells you what to do at each step, in plain
words, and a **yellow waypoint arrow** low in the middle of the view points the
way to the next place you need to be (through the door of whatever building you
are in first). The town's **guards stand still, blind and deaf, for the whole
tutorial** - the lessons are in locks and shadows, not in being chased - and
until the candlestick is sold every other door, shelf, pocket and deal is shut:
you can only do the job you were given.

1. Jae explains the debt (three lines).
2. It is night, so the torch law comes first: `1` lights your torch; a guard
   who sees you without one after dark stops you, and a lit torch makes you
   easy to see.
3. Leave the hideout and cross the street to the store; crouch and pick its
   lock - the panel explains the mini-game. Snap your last pick and Jae slips
   you another (he keeps doing so for the whole tutorial: it never stalls for
   want of a pick).
4. The west shelf: every row is marked THEFT; take the candlestick anyway.
5. The shelf screen closes and a watchman who was not there a moment ago stands
   a couple of paces behind you - *"I thought I saw you sneaking about!"* - you
   turn to face him, and after a moment the town's justice takes over: the
   charge, the bill (unpaid: the cells), the stolen goods back to the shelf,
   your things into the confiscation lockbox but for one pick. A first offence:
   one night, not a week. The watchman leaves the town again once you are in
   the cell.
6. The cells: `T` waits out the sentence - or pick the bars with your last pick
   (a jailbreak, and your things stay locked up).
7. Free: the lockbox in the jail office gives everything back (`X`).
8. `Tab`: put your boots on; the panel explains SEEN and HEARD.
9. Thieves work at night: `T` passes an hour; keep pressing it until the clock
   passes 20:00, and light your torch again for the street.
10. The candlestick again, properly - unseen (the guards are still statues, but
    Barnaby and the townsfolk are not).
11. Sell it to Jae for next to nothing; he pays you sixty gold on top.
12. A pickaxe and a mallet (bought or stolen), ore from the deposit behind the
    smithy, lockpicks at the anvil.
13. Somewhere to sleep: walk into the Drowsy Dragon (or rent a room). Tutorial
    complete: the closing words stay in the panel for half a minute, the
    guards take up their rounds, and the town is yours.

Saves made during the tutorial remember where you were in it.

## Gameplay notes

* **The town.** Emberhollow climbs a hillside inside its curtain wall: the gate
  at the bottom, the well in the middle of Market Street - a round stone well
  with a windlass, a bucket and a little roof, water glinting down the shaft -
  and Upper Street above, with the watchtower on the knoll at the top. Market
  Street, north side: Gareth's smithy with its open-sided shed (anvil, forge,
  weapon rack) and the ore deposit behind it, across the millstream's bridge the
  Drowsy Dragon inn (bar, bread oven, three guest rooms), Barnaby's general
  store, Mirelle the tailor and the old granary. South side: the cottages of
  Barnaby, Tamsin, Fenwick, Jae (the hideout, straight across from the store),
  Lys & Dorran and Pell, Wynn's bakery, and the jail at the east end; Gate Road
  runs down from the well past the chapel (a nave with a steeple, its bell in
  a belfry open to all four winds, and a graveyard; the chapel's door stands
  unlocked from the morning bell to nightfall, 06:00-21:00, whether or not
  Sister Maude is in) and the stable to the gate and the gate watch's lodge; Tanners'
  Row and the cooper's yard lie to the east, two cottages by the town wall.
  Upper Street: the manors of Lord Vane and Lady Ashgrove with a little park
  and its fountain between them - two storeys of dressed stone and render under
  a cornice, a porch on columns, sash windows, chimneys; inside a kitchen with
  its range, a store, a dining room with a fireplace, the servant's room, and
  upstairs a study, a dressing room, the bedroom and a second bedroom or music
  room - and across the way the herbwife's, the locksmith's, the Reeve's and the
  minstrel's houses; the guard house at the east end. Every place of business
  hangs a sign beside its door, and every door has a name. You start on the
  well square (the tutorial: in the hideout).
* **The watchtower.** A hollow square tower behind the guard house at the top of the hill, entered by
  a locked door from inside it: a winding wooden stair climbs two and a half
  turns round the newel column, lanterns on the way up and slit windows in the
  walls, to a look-out platform behind a parapet. You climb it step by step like
  anyone else (you cannot jump a step or walk off the platform), and Guard
  Aldric, the tower watch, climbs it at dawn and comes down for his meals.
* **Dress.** Everybody wears something of their own: tunics and hose in the
  dyes a small town can afford, Hilde and Lys in cloth aprons, Gareth and
  Dorran in leather ones, Barnaby in a cap and merchant's sash, Tamsin in a
  widow's scarf, Jae in a deep hood; the watch wear their steel breastplates
  and kettle helmets.
* **Routines.** All fifty-three NPCs sleep, eat, work, shop, stroll and drink on
  daily schedules - and many of them keep a *round* of several different days
  (the widow visits the chapel one day and the park the next, the cooper fells
  timber in the woods every third day, the servants' errands change, the
  Reeve sits at home or walks his bounds) while the shopkeepers keep the same
  hours every day - walking path-found routes and operating the doors they are
  allowed through. Barnaby trades from behind his counter 08:00-13:00 and
  14:00-20:00 (the shop door is unlocked - not opened - for business), Mirelle
  09:00-18:00, Hilde keeps the inn open from 05:00 to midnight and Gareth sells
  from his anvil while he works. Dorran hauls crates for Barnaby, Lys cooks at
  the inn, Tamsin, Fenwick and Pell shop, stroll the street (stopping at the
  well) and fill the inn in the evening. Everyone locks their own door whenever
  they are out or asleep, eats from their kitchen table (laid again every third
  night) and somebody bound for a place that is still shut waits on its
  doorstep and tries again every so often. Three guards rotate shifts between
  the patrol through every street, the day post outside the guard house and the
  post by the inn; Bram and Wenna keep the town gate round the clock, changing
  sides every few hours; Aldric keeps the tower watch and Jailer Hesk the jail.
  The rich take the air on Upper Street, sneer at the store and the tailor's and
  dine at home while their servants cook, serve, dust upstairs and run errands
  (and meet the stable lad in the park); Sister Maude rings the chapel bell
  morning and evening, Old Bertram cries the news by the well, Wynn bakes before
  dawn, Jeb lights the torches at dusk and walks the streets all night. Out in
  the country the farmers work their fields, the hunter stalks the woods, Hrolf
  hammers at his anvil, the smugglers haul, cook, play cards and keep a lookout
  on the beach, the bandits drill, dice and stand watch at the mouth of the Deep
  while their king sleeps three hours a night, and three knights and a squire
  keep the old keep.
* **Talk.** NPCs speak in bubbles above their heads that stay up long enough to
  read at leisure (you hear people within twelve units whom you are roughly
  facing). What somebody *does* ("Barnaby opened the shop for business.") is
  narrated plainly in the event log; only what they *say* is quoted. Neighbours who stop near each other strike up short conversations -
  greeting, a topic or two coloured by who they are (the smith is asked about
  his iron, guards about the watch), goodbye - and the same gossip is not
  repeated for a while; at most two conversations run at once, one within your
  earshot. Shoppers haggle (and pay the merchant a few coins), drinkers call for
  another, the smith's hammer rings; down in the caves the outlaws talk shop in
  longer exchanges (the boat is late, the dice lean, what is behind the iron
  door). Town chatter stays out of the event log (the World Events log, `~`); only what concerns you -
  warnings, greetings, reports - is written there, and the comings and goings
  of people near you (within forty units, or under the same roof). Talking to
  someone yourself gets a greeting that fits them and the hour - and every
  other time, by day, the next line of their story.
* **Reports.** A shopkeeper who finds goods missing shouts for the watch, and
  a guard comes running to take the report *from him*, the two of them facing
  each other: "what's been taken?", "did you see anyone suspicious?" - and the
  answer depends on what each actually saw. A guard with his own sighting of you
  needs no description; a shopkeeper who caught you poking about hands the guard
  a lead; if nobody saw anything there is nothing to go on. Both go back to their
  day afterwards. A guard taking one shopkeeper's report is never pulled off it
  by another's cry - that report goes to another guard, or waits for one to come
  free. Only a crime committed in front of him interrupts: he deals with the
  criminal, comes back, apologises, and takes the report from the top. **Hearsay convicts**: a guard with a lead - his own sighting or
  the shopkeeper's description - calls you out the next time he sees you, takes
  back the reporter's stolen goods you still carry (only those: a bottle you
  bought and paid for is yours, whatever it is called) and charges you with the
  theft as if he had seen it himself; he forgets it in time like any other
  grudge. Getting away is not the same as getting away with it - be careful.
  An honest merchant offered stolen goods shouts for the guards too, and they come.
* **The inn.** Hilde sells a room for the night (10 g) or the week (50 g) like
  any other item - and she lets rooms whenever she is awake, wiping tables or
  sweeping up included, not only from behind the bar (only sleep, or an alarm,
  stops her). The first room is unlocked and opened for you and counts as
  yours - nobody objects to you being in it - until noon on your last day,
  when it is locked again and a guest still inside is shown out *of the room*,
  into the hall (the inn is a public house; only the room was yours) - asleep
  through checkout, you wake in the hall: the pain of forgetting to pay your
  bill. Paying again extends the stay. A paying guest helps themselves to the inn's food and ale
  ("on the house"), and the keeper asleep behind her shut door is not woken by
  guests crossing the hall - only by a racket in her own room. The inn is a
  public house at every hour: **you are always welcome in the hall**, room or
  no room, and nobody - not the keeper, not the night watch - treats you as a
  trespasser there (the kitchen, the keeper's own room and rooms you have not
  paid for stay off limits). Walk in without a room and the keeper comes to
  welcome you and offer one - from behind the bar if she is there, otherwise
  she comes over, woken if need be ("Mm? A guest, at this hour..."): innkeepers
  get little rest. She stays a while so you can rent, then goes back to what she
  was doing - or to bed.
* **Sleeping.** Look at a bed that is yours - the one in your rented room, or
  any bed you own - and `E` offers **Sleep Until Night** by day (06:00-18:00;
  you wake at 20:00, when the streets empty) or **Sleep Until Morning** at dusk
  and at night (you wake at 06:00): the screen fades slowly to black, the hours
  pass (the town lives on meanwhile) and it comes back. Anyone else's bed is
  just that.
* **Shops.** Shopkeepers are paranoid about anyone *crouching* on their premises.
  Every second you creep about the shop in their sight your Sneak skill is rolled;
  the first failed roll gets you told to stand up straight or get out, the next
  (eight seconds on) a last warning, and the one after that you are thrown out
  of the shop - exactly as a householder shows a lingering guest the door. Stand
  up or leave and the keeper goes back to business. The same at **closing time**:
  when the keeper leaves the premises (Barnaby at 20:00, Mirelle at 18:00, Hilde
  at midnight) anybody still on the shop floor is shown out - unless they are
  crouching and a Sneak roll succeeds, in which case the keeper locks up none
  the wiser and the thief is alone with the shelves (and trespassing). A paying
  inn guest is always left be. Waiting (`T`) never skips any of this: the wait
  is interrupted the moment the keeper confronts you.
* **Fences.** Jae - any fence - welcomes your custom and buys anything hot, but
  he is as wary as any shopkeeper: he turns to watch whoever is in his den,
  tells a creeping customer to stand up straight (and shows them the door on
  the third ask), and his suspicion builds like a merchant's. He never calls
  the watch over skulking - only over being robbed himself. Picking a fence's
  pockets means getting to them unseen, which takes a high Sneak skill and very
  dark, very quiet clothes.
* **Specialists.** The smith deals only in weapons, armour, tools and ore, the
  tailor in clothing and tools (her cloaks and slippers quieten your steps);
  offer them anything else and they refuse without closing the screen. Barnaby
  and Jae buy anything (Jae only stolen goods, no questions asked). Jae is open
  for business at *any* hour: knock him up in the middle of the night and he
  gets out of bed, deals, and goes back to sleep - after seeing you out, since
  the hideout's curfew still stands. No rest for the wicked. Forging
  lockpicks at the anvil takes ore *and a mallet*; Gareth carries one (and a
  pickaxe) for his own work and sells them from his rack, as does Barnaby.
* **Doors and keys.** Owning a door does not open it: every NPC needs the key
  item in its pockets (guards carry a guard-house key; Hesk also the cell key).
  NPCs shut behind them whatever they opened, relock what was locked and always
  lock their own doors - even ones you unlocked. Keys you use only *unlock*;
  press `E` again to open. A beam tops every doorway. Every NPC gets a fresh set
  of keys and supplies at midnight (guards daily, Barnaby and Jae every third day).
* **Indoors and out.** Buildings have floors, ceilings and roofs, and their walls
  muffle sound: footsteps in the street reach a sleeper inside at a quarter of
  their noise, so running past the guard house wakes nobody - running around
  *inside* it does. A shout for the guards carries across town in the open and
  about half as far through walls; only guards within earshot come running.
  Every room has a wall lantern and every table a candle (generated from the
  level, so new houses are lit automatically). The torch rule is simple:
  outdoors, once the sun is low enough (about 17:30 until 06:30) everyone
  carries a lit torch, whatever lights are about; indoors a torch is lit only
  where no lantern or candle reaches. Inside a building, noise carries freely
  through a room but only a quarter of it through a wall or a shut door - so
  the inn's hall and its guest rooms are separate chambers.
* **Solid people.** You cannot walk through NPCs and they cannot walk through you
  or each other: someone in the way is stepped around, a crowd converging on one
  spot stands *around* it, and an NPC hemmed in for long gives up and tries again
  on the hour.
* **Seen vs heard.** Noise only wakes NPCs and makes guards walk over to look;
  it never raises the alarm by itself. Being *seen* does: suspicion builds with
  visibility and distance inside an NPC's field of view, and at 100 the crime
  is declared. Sneaking (crouch-moving) is a skill roll once a second - success
  hides you from that NPC's sight; your Sneak skill also quietens your steps.
  Guards tell you off once - "you there, what are you skulking about for?" - and
  give you a few seconds to comply before the alarm can go up; the same warning
  is not repeated for a while. A guard's dressing-down is delivered to your face:
  your controls lock for a moment and the view swings round to look at them.
  A guard who has walked over to look into a noise keeps watching a suspect he
  can see rather than glancing about. Sight is checked against your chest, so
  crouching behind the well, a counter or a table really does hide you.
* **Eye level.** Your camera stands at the townsfolk's eye level (they are 1.74
  tall, eyes at 1.6; you look out from 1.55, 0.9 when crouched) - nobody towers
  over anybody, and the crosshair meets doors, counters and faces where you
  expect.
* **Trust, trespass and curfews.** Houses, the guard house and the jail cell are
  restricted; lingering there without being trusted (Jae trusts you in the
  hideout) earns a warning and mounting suspicion. A fence's house is always open
  to you by day - the standing rule - and nobody, guards included, objects. Even
  a welcome guest is asked to leave late at night: the host asks, waits, asks
  again and finally shows you the door. After dark the law requires a lit torch. The jail house itself is
  public and its door is never locked - only the barred cell is off limits.
  Light spills through an open doorway: a torch carried over a threshold keeps
  lighting the open leaf beside it, and a lit room shows through its open door.
* **Crimes, fines and jail.** Each guard keeps a memory of your crimes -
  witnessed ones for 3-5 weeks, reported ones for 3-5 days. The next time a
  guard with a grudge sees you he shouts *Halt!* and comes after you - at a run.
  A sprinting thief can outpace him and slip away for now (the grudge stays); if
  he catches up, any screen you had open closes, you turn to face him, stolen
  goods go back to their owners and you are presented with the bill (see
  **Crimes and punishments** below): the fines are paid as far as your purse
  goes - paid in full and with no sentence owed, you are free (and, if you were
  somewhere you had no business being, thrown out); otherwise it is the cells.
  Your controls lock for a couple of seconds while it sinks in, and
  the march to the cell waits a moment longer before the screen fades. Jail means your belongings locked in the
  confiscation lockbox (except one lockpick; what is left of your purse stays yours),
  the screen fading to black and back in the cell. The cell is a barred gate that
  is *sealed* while you are inside - the jailer never opens it to check on you,
  however much noise you make. You are released at 06:00 on your last day: the
  bars and the lockbox are opened, your things are yours to take back (never
  "stolen"), and the jailer waits - even in the middle of the night - until you
  have left before going back to his routine. Picking your way out earlier is a
  jailbreak. Press `T` in the cell to wait out the whole sentence.

  **Crimes and punishments.** Every crime carries a fine, days in the cells
  served *on top of* paying it, and more days served *instead* if you cannot pay.
  Theft is punished by *where* it happened: in the street, or inside any building.

  | Crime | Fine | Jail (fine paid) | Jail (fine unpaid) |
  | --- | --- | --- | --- |
  | Skulking about (ignoring a guard's "stand up straight" more than once) | 20 g | - | 1 day |
  | No torch after dark (ignoring the torch law after repeated warnings) | 5 g | - | 1 day |
  | Trespassing | 15 g | - | 1 day |
  | Fencing stolen goods (offering them to an honest merchant) | 30 g | - | 1 day |
  | Pickpocketing, lockpicking or stealing **outdoors** | 100 g | 2 days | 3 days |
  | Pickpocketing, lockpicking or stealing **inside any building** | 500 g | 5 days | 7 days |
  | Jailbreak | 100 g | 2 days | 3 days |
  | **Every stolen item on you when charged** | + its value | + 1 day each | + 1 more day for each you cannot pay for |

  The bill is settled in order of severity: each crime is paid if the purse
  covers it (its fine and its "paid" days), otherwise its "unpaid" days are
  served; then every stolen item you are carrying costs its value and a day,
  and a day more if its value cannot be covered either - and the goods go back
  to their owners regardless. The same crime known to several guards is charged
  once. (So being caught skulking with a 120 g stolen goblet in your pack and
  50 g in your purse is 20 g paid, the goblet unpaid, and two days in the
  cells: one for the stolen goblet, one more because it could not be paid
  for... fence it first.) While you serve, the bed in the cell is yours to
  sleep in (and only until you are released, or break out); `T` waits out a
  sentence of any length in a few seconds, the town living its days meanwhile,
  under a box that shows how far it has got.
* **Merchants and money.** Everyone carries coin: guards a little, merchants
  a lot, so they can pay for what you sell (a merchant who cannot afford
  something says so). Purses are topped back up every three days with the
  restock, never cut. Pickpocketing shows the victim's purse as the first row -
  each lift grabs up to 20 coins at Pickpocket level 1, two more per level. Every
  dip into somebody's pockets rolls your Pickpocket skill twice: whether the
  victim feels it (a fumble is caught on the spot) and whether the people *around*
  notice - only a clumsy dip gives the onlookers a chance to witness it, and the
  victim is not one of the onlookers (feeling the hand is their own roll). One
  fumble is one crime: the victim shouts once, an onlooker who already saw you
  reach in does not shout again, and the guards hold it against you once -
  though a second theft later in the day is a new crime with new witnesses.
  Merchants never sell their keys; crouch and
  pick their pockets instead (or plant something on them). Shop shelves can be
  looted - every row that would be a theft is marked **THEFT** in red in the
  container screen, so there is never any doubt (what is yours, or on the
  house, is not) - in view of the shopkeeper that is shoplifting, unless your
  Stealing skill keeps it unnoticed; he counts stock when opening and closing and shouts
  for the guards, who come to take the report and note whether they saw you
  lurking there. Everything belongs to somebody: stealing flags an item `(S)` but
  it stays its owner's, which is how a guard knows where to return it. Ownership
  only changes hands honestly - a sale, a gift, a meal on the house - or through
  a fence, who launders what it buys (or what you slip into its pockets).
* **Equipment.** Everything anybody wears is a real, physical item mounted on
  their figure in one of five slots - **head** (helmets, caps, scarves, hoods,
  coifs), **neck** (aprons, sashes, necklaces, cloaks), **body** (tunics, mail,
  breastplates, plate), **feet** (boots, shoes, slippers) and **weapon** (a
  dagger or a sword at the hip, a sling) - and the townsfolk are dressed by the
  same mechanism: the watch's breastplates and kettle helms, Gareth's leather
  apron, Jae's hood, Barnaby's cap and sash. Press `Tab` to see yourself: your
  figure turns on its stand wearing whatever you have put on; `Enter` on an item
  in your pack equips it (one per slot - a helmet replaces a cap) or takes it off.
  What you wear shows `[worn]` in the inventory panel and in every trade screen
  (selling, planting or putting away something you have on asks first), goes
  with you into the lockbox when you are jailed, and comes off when sold,
  planted or confiscated.
* **Clothes and the shadows.** The game knows what your gear is *made of* and
  what *colour* it is, and both decide how easily you are seen and heard -
  only while it is worn. **Metal** glints and clanks; **leather** is dull and
  fairly quiet; **cloth** is dim and silent. Pale shades stand out, dark ones
  hide (white is lighter than brown is lighter than black). The body counts
  most for being seen, the shoes most for being heard. The `Tab` view totals it
  up - *Seen 0.4x, Heard 0.7x* - and describes the item under the cursor
  ("black cloth - hard to see, silent"). Head-to-toe black cloth makes you a
  shadow even at a low Sneak level: crouched and holding still you are all but
  invisible, even at noon, even a pace from a guard, even in his torchlight;
  moving by day you are a visible shadow that is slow to raise suspicion, and
  every sneak roll is easier. Full plate with a white cloak is the opposite:
  the steel glints in the dark, so the watch sees you at night without a
  torch - and sprinting down the street in it is a racket that wakes the
  neighbourhood through the walls and gets you told off by everyone awake
  ("Must you rattle like a tinker's cart?") - not a crime, with a torch lit,
  just a very bad idea. Accordingly the bright and shiny is common and cheap
  (a white linen shirt for 4 g, an iron helm for 20 g, a plate cuirass for 90 g
  on the smith's rack), brown leather is the middle ground, and dark cloth is
  rare and dear, reaching the tailor's rack only as you rise in level - the
  velvet cloak, a black hood (Jae sells one too, at a thief's price), a black
  tunic, a black cloak and, rarest and dearest of all at 520 g, black cloth
  shoes.
* **Skills.** Six skills - Sneak, Lockpicking, Pickpocket, Stealing, Barter,
  Smithing - all follow one rule: rewards pour in early and dwindle as the skill
  grows (an elite earns about a fifth of a novice's XP for the same deed), odds
  climb smoothly towards near-certainty (an elite sneak is all but invisible at
  noon in front of a guard, an elite pickpocket barely ever fumbles), and the
  value of what you handle sets the reward (a 300 g necklace teaches far more
  than a spool of thread; a sleeping victim less than an awake one; a guard more
  than a merchant). Failing still teaches a little. Sneak XP flows whenever you
  crouch-move: a trickle even when alone, more near anyone who has not noticed
  you (sleepers and turned backs alike - pottering about a sleeping shopkeeper's
  house is the classic way to train early on), a stream right under an awake
  nose, more for guards. **Barter** works on three prices: an item's *worth*
  (what is always shown), the merchant's *asking price* (the worth plus a 25%
  markup at level 1, haggled down 2% of the markup per level, never below the
  worth) and the merchant's *offer* for what you sell (a quarter of the worth at
  level 1, one more percent of it per level, never above the worth; a fence pays
  half that for stolen goods). Whatever your skill, buying something and selling
  it straight back loses money; rooms at the inn have a posted price nobody
  haggles over. Smithing decides the
  anvil: a novice often ruins the ore and gets one pick when it works; an elite
  gets up to fifteen picks per ore.
* **Waiting and loading.** Anything that takes more than a moment - a night's
  sleep, days in the cells, the town being built for a new game, the scene
  being rebuilt after a graphics change - puts up a box saying what is going
  on (with a bar where it can tell how far it has got) instead of leaving the
  window frozen. Long waits pass the hours quickly: the first hour is lived in
  full, the rest a stroke of the clock at a time with everybody put where they
  were going (eating, unlocking and locking up on the way).
* **Bridges.** Their rails are solid, and the townsfolk keep to the deck:
  nobody wades the stream beside a bridge or walks through its rails.
* **Day and night.** The sun rises at 06:00 and sets at 18:00 (the moon takes
  the opposite arc), stars come out, fog closes in after dark and torches
  (yours and the guards') light the way. Indoors every room has its lantern and
  every table its candle, and no light leaks through a wall into the room next
  door. On the lower graphics presets a distance fog closes in just before the
  draw distance, so nothing pops in or out at the edge of the world.

## Tests

```
./run_tests.sh              # everything (display tests under xvfb-run when headless)
./run_tests.sh fast         # headless suite: unit tests + scenario goldens
pytest -m display           # OpenGL smoke tests, menu driving, benchmark tool, prototype parity
```

* `tests/unit` - ~1100 unit tests (parametrised) for every module (entities, world, systems, game API, input, save slots, menus,
  textures, perf, dialogue and conversations, the inn, the town's routines, the watchtower, saving and loading mid-action,
  equipment and the [Tab] view, the tutorial (the frozen guards, the waypoint), the HUD's panel placement, the terrain
  (level floors, nothing floating), z-fighting planes, the gate, Jae's job, the rich, the wardens, and - in
  `test_realm.py` - the caves, the trails, the edges of the map, door names, reading, hiding places, the manors'
  upper floors, stories, topics and pantries; every day of every NPC's round) and - in `test_screens_and_settings.py` -
  the graphics presets, the start screen's stroll, doors on north-south walls, the HUD's panels and the cave, mine,
  fountain and window repairs; in `test_town_and_time.py` - the mountains, the chapel, the bridges, the prisoner's
  bed, waits of any length, quick hours and the frame-time work; in `test_visual_audit.py` - the visual audit,
  which keeps the whole level free of z-fighting and clipping). `tests/helpers.py` places things on the hillside by
  map position.
* `tests/regression` - 34 scripted play-throughs (`tests/parity/scenarios.py`)
  replayed headlessly and compared with committed golden state files
  (`UNPREPARED_UPDATE_GOLDENS=1` re-records). Pixel goldens (local,
  renderer-specific) can be recorded the same way for `test_pixel_goldens.py`.
* `tests/parity` - runs the **original prototype** (frozen in `tests/baseline/`)
  side by side with the game for the player-loop scenarios that must still
  match, checks the prototype's crash is fixed and that ESC now pauses instead
  of quitting, plus randomised checks of the perception rules kept verbatim.
* `tests/display` - launches the real app: default and custom resolutions, a
  missing-asset run, `python main.py`, the start/pause/save/load/confirm/graphics
  menus driven by synthetic key events (including Continue and Exit to Menu, the live
  town behind the start screen, full screen on and off, the draw distance, anti-aliasing,
  texture filtering and a corrupt save picked from the start screen), where the
  fixed-function (*low* preset) lights sit relative to the camera, and `tools/benchmark.py`.

See `docs/ARCHITECTURE.md` for the design and the change log.
