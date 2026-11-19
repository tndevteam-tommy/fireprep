# Architecture

## Why the restructure

The prototype was three files: a 700-line `main.py` containing window setup,
texture generation, all OpenGL drawing, the HUD, input handling, trading,
lockpicking, movement, lighting *and* the game loop – with about ten loose
local variables (`in_trade_menu`, `lock_pick_angle`, `active_merchant`, …)
acting as the game state. `models.py` held the entities and `world.py` a
hard-coded town. Nothing could be exercised without opening an OpenGL window.

The refactor keeps the *behaviour* byte-for-byte (see "Verification") but
splits the code along one rule:

> **Simulation never imports pygame or OpenGL. Presentation never mutates
> simulation state.**

```
            ┌──────────────┐   events    ┌──────────────┐   reads    ┌──────────────┐
 pygame ───▶│ input/       │───actions──▶│ game.Game    │◀───────────│ rendering/   │───▶ OpenGL
            │ controller   │             │ (headless)   │            │ Renderer     │
            └──────────────┘             └──────────────┘            └──────────────┘
                     ▲                          ▲                          ▲
                     └──────────── app.PygameApp owns all three ───────────┘
```

## Layers

| Layer | Module(s) | Responsibility | Depends on |
| --- | --- | --- | --- |
| config | `config.py` | Frozen dataclasses with every tunable (speeds, ranges, XP, perception rates, stealth/crafting/justice numbers, atmosphere colours, HUD layout). | – |
| graphics / perf / ui | `graphics.py`, `perf.py`, `ui.py` | `GraphicsSettings` (mutable, persisted as JSON next to the save; presets `high/medium/low`), `FrameProfiler` + `recommend_quality` (start-up benchmark and renderer sniffing), `Menu`/`MenuItem` (data + callbacks, no drawing). | – |
| core | `core/math3d.py`, `core/messages.py`, `core/daynight.py` | `Vector3D`, AABB, segment/box raycast, view-direction helpers; bounded `MessageLog`; time-of-day rules. | config |
| entities | `entities/*.py` | `Item` (+ ``owner``: every item belongs to somebody - an NPC, a faction, the player; theft flags it stolen but leaves the owner, a sale / gift / fence transfers it; + ``gear``/``gear_color``/``equipped``: what it is worn as), `equipment` (`EquipSlot` head/neck/body/feet/weapon, `GEAR_CATALOGUE` by item id, `Figure` - the person under the gear -, `equipment_from_outfit`: an `Outfit` as real worn items), `Door` (+`DoorSchedule`, owners, lock difficulty, `DoorKind.LEAF/BARS`, ``sealed``), `SkillTree` (six skills, linear XP curve), `Player` (level, eye-height lerp, noise multiplier from what is *worn*, `equip/unequip`, `face_towards`), `NPC` (+`AIState` incl. `WAITING`, `ScheduleTask`, `CrimeMemory`, sightings, hunger, purse, body collision, rate-limited scolding, `equipment` derived from its outfit). Each owns its own rules (perception, schedules) and its own `to_dict/load_dict`. | core |
| world | `world/level.py`, `world/levels/town.py`, `world/world.py`, `world/navigation.py`, `world/save.py` | **Data-driven levels**: a `LevelDefinition` (walls and props with texture keys, doors with owners, NPC definitions with inventories & daily schedules, player spawn kit) is turned into live entities by `World`. `AreaDef`s carry ownership / restriction / public hours / trust / curfew hours, `BuildingDef`s give floors, ceilings, roofs *and the indoors/outdoors notion*, `ContainerDef`s may be locked lockboxes or four-legged tables, `StationDef`s are work stations (ore deposit, anvil), `JailDef` describes a level's jail. `WorldClock` handles simulated time, `NavGrid` path-finding, `SaveManager` persistence (format v8, named slots, atomic writes, older files still load, corrupt ones reported). | entities |
| systems | `systems/*.py` | Stateless rules operating on entities: `apply_movement` + `move_with_collision` (people are solid, pre-existing overlaps never pin you), `find_interaction_target` (crosshair ray first, aim cone second), `TransferSession` + policies (trade with purses and Barter, pickpocket with coin purses, container with ownership and Stealing rolls), `LockpickMinigame` (any lockable: doors and lockboxes), `progression` (one XP / odds rule for every skill), `stealth` checks, `commerce`, `crafting` (skill-scaled forging), `justice` (fines, eviction, sealed cells, release, jailbreak). | world |
| game | `game.py` | `Game`: owns World + Player + UI mode (`GameMode.EXPLORE/TRADE/PICKPOCKET/CONTAINER/LOCKPICK/CONFIRM/EQUIPMENT`) and exposes the **action API** (`interact()`, `toggle_torch()`, `wait()`, `ask()`/`confirm_yes/no/alt()`, `cursor_up/down()`, `transfer_confirm()`, `loot_all()`, `tab()` / `equipment_toggle()` / `rotate_preview()`, `lockpick_rotate()`…) plus `update(dt, move_intent)`. It installs the world's presentation hooks: relocations become a `ScreenFade` (move applied at black, input frozen) and "caught" events close every screen. | systems |
| input | `input/bindings.py`, `input/controller.py` | `KeyBindings` (rebindable) and the pygame event → `Game` action translation. ~60 lines, no logic. | game, pygame |
| rendering | `rendering/*.py` | `TextureRegistry` (loader only; a missing file is a magenta placeholder and is listed in `.missing`), `camera` (GLU-free), `atmosphere` (sun/moon/sky/fog/star alpha by fractional hour), `shaders` (GLSL 1.20 per-pixel lighting + fog + contrast, fixed-function fallback), `lighting` (gathers torches up to the preset's light cap, ranked by what the camera can see; room-bounded lights spill through the nearest open door), `gear` (every equipment kind as a model mounted on a `Figure`; `draw_figure` dresses NPCs and the player alike), `models` (NPC body/face/torch, sleeping pose), `sky` (stars, sun with an additive halo, moon), `buildings` (floors, ceilings, gabled roofs), `primitives` (mirrored door leaves, barred gates, four-legged tables), `SceneRenderer` (honours `GraphicsSettings`; draws door lintels; records draw stats; `draw_player_preview` paints the player's figure under studio light into the [Tab] view's frame), `TextRenderer`, `HudRenderer` (+ fade overlay, yes/no questions, the equipment view) + `MenuRenderer` (scrolling lists). Reads from `Game`, never writes. | game, OpenGL |
| app | `app.py` | `PygameApp`: window creation, `Screen` state machine (START / PLAYING / PAUSE / SAVE / LOAD / CONFIRM / GRAPHICS), one `step()` = input → update → render → flip, profiler phases, mouse capture sync, start-up benchmark. | everything |
| tools | `tools/textures/`, `tools/benchmark.py` | Development CLIs outside the game package: the procedural texture generator and the per-preset frame-time report. | rendering |

### The frame

```python
def step(self):                                  # app.PygameApp
    dt = self.clock.tick(fps) / 1000
    if self.screen is Screen.PLAYING:
        self._process_game_events()              # ESC -> pause menu, F3 -> overlay, rest -> input controller
        self.game.update(dt, self.input.poll_movement())
    else:
        self._process_menu_events()              # UP/DOWN/LEFT/RIGHT/ENTER/ESC drive the active Menu
    self._sync_mouse_capture()                   # captured only while PLAYING and game.mouse_captured
    self.renderer.render_scene_only(self.game)   # the world is drawn behind menus too
    self.renderer.hud.render(self.game)          # PLAYING only
    self.menus.draw_menu(self.menu)              # if a menu is open
    pygame.display.flip()
```

Each phase is timed by `FrameProfiler` (shown by the F3 overlay). `Game.update`
does, in order: player movement with collision (if allowed by the mode) → world
clock (door + NPC schedules on the hour, midnight inventory resets, jail
sentences) → NPC movement (doors, eating) → NPC perception (hearing, sight,
trespass, torch law, charging wanted players) → interaction targeting.

## How to extend

* **New level** – write a `LevelDefinition` in `world/levels/` and
  `register_level("name", factory)`. Walls/props take a `texture` key, doors
  `owners` + `key_id` + `lock_difficulty` (or `open_hour/close_hour` for unowned
  doors), areas an `owner`/`restricted`/`public_hours`/`trusted_by`, buildings a
  roof texture, stations a kind, NPCs inventories, `shop_containers`, restock
  rules and schedules (`TaskDef`). No code elsewhere changes.
* **New work station** – add a `StationDef` (kind `"mine"` or `"anvil"`) to the
  level; a new kind needs one branch in `systems/crafting.py`. `requires_item_id`
  names a tool the player must carry (a pickaxe, a mallet).
* **New modelled prop** – add a `draw_<kind>` to `rendering/structures.py` and
  register it in `DRAWERS`; the level places it with `PropDef(kind=...)`.
* **New tower / stair** – `world.tower.build_tower(...)` gives you walls, steps,
  platform, lights and a `StairDef`; add them to the level (`stairs=`) and put an
  `open_top` `BuildingDef` over the footprint. Any `PropDef(kind="step"/"floor",
  solid=False)` is a walkable floor for the player.
* **New outfit** – `NPCDef(outfit=Outfit(...))`, or a role branch in
  `entities.outfit.outfit_for`.
* **New town with a jail** – give the level a `JailDef` (barred cell door, lockbox
  container, cell and jail areas, cell spot, jailer and post); everything in
  `systems/justice.py` works off it. A level without one fines what it can and warns.
* **New skill** – add a `SkillType`, a base reward in `SkillRewards`, and call
  `progression.grant(player, skill, base, magnitude)` where the deed happens;
  odds come from `progression.success_chance`.
* **New question to the player** – `game.ask(text, on_yes, on_no, alt_label, on_alt)`;
  the HUD draws it, Enter/Esc/T answer it.
* **New NPC routine** – add `TaskDef`s (see "NPC daily routines"); put
  `for_sale=False` on personal items such as keys.
* **New item / NPC** – add `ItemDef` / `NPCDef` entries to the level data.
* **New interaction** – add a function in `systems/`, a method on `Game`
  that calls it, a key in `KeyBindings` and one `elif` in
  `PygameInputController._handle_keydown`. Unit-test it through `Game` without
  a window.
* **New texture** – drop `assets/textures/<key>.png` and reference `<key>`
  from level data; `TextureRegistry` loads lazily and shows a magenta
  placeholder for missing files. To generate it procedurally add a function to
  `tools/textures/generators.py` and run `python tools/textures/gen_textures.py --only <key>`.
* **New graphics option** – add a field to `GraphicsSettings` (and to the
  presets if it differs per tier), read it in `SceneRenderer`, add a `MenuItem`
  in `PygameApp.open_graphics_menu` (`rebuild=True` if GL resources change).
* **New menu / screen** – add a `Screen` value, an `open_*` method building a
  `Menu`, and route to it from the start or pause menu.
* **Balancing** – edit `config.py`. `GameSettings` is immutable; tests create
  variants with `GameSettings(player=PlayerSettings(walk_speed=...))`.
* **Resolution** – `WindowSettings(width, height)`; `HudLayout.for_window`
  keeps panels anchored. (The prototype's 1024×768 layout is reproduced exactly.)
* **HUD** – `HudRenderer` draws from `HudLayout` + `HudColors`; text goes
  through `TextRenderer.draw` which caches glyph textures.

## Rendering

`SceneRenderer.render` per frame: `compute_atmosphere(fractional_hour)` gives
sun/moon directions, colours, sky/fog colour and fog density (all smooth
functions of time); `LightingRig` transforms them and every lit torch into eye
space for the shader (`shaders.py`: ambient + sun + moon + up to 8 point lights
per pixel, exp2 fog; emissive materials such as flames and the sun disc fade
the fog so they stay visible from afar). Sun and moon are camera-facing
billboards drawn before the world with depth writes off. NPCs are drawn by
`models.draw_npc`: body + head rotated to `facing_yaw`, white eye spheres with
pupils on the front face (closed/lying when asleep), and a stick-and-flame torch
at night; the player's torch is drawn in view space. If GLSL is unavailable the
same data drives fixed-function lights and `glFog`.

**Baked geometry** (`rendering/batching.py`). Nothing static is issued through
immediate mode every frame any more: `Compiled.record(draw)` runs a draw routine once
with a recording `material` callback and captures it into display lists - one list per
run of identical material settings, which are replayed *outside* the lists so the same
recording works for the GLSL uniforms and the fixed-function colour/material state.
`StaticScene` bakes a world's ground, roads, walls, lintels, props, containers,
stations, the well and signs, lantern and candle bodies, floors, ceilings and roofs
into 10-unit chunks (a solid part and a roofs part, so the "Roofs & floors" option
needs no rebuild) culled by the draw distance; NPC outfits are recorded once each in
local space (`SceneRenderer.outfit_for`) and drawn under their transform, door leaves
and bars likewise (`door_geometry`). Live every frame: door poses, NPC poses, eyes
and torches, lantern/candle flames (`draw_fixture_flame`), the forge and oven, the sky
and the held torch. A street frame went from ~34 000 GL calls to ~4 600. Lists are
keyed to the world and texture registry that built them and freed by
`SceneRenderer.release` (the renderer is rebuilt when shaders/mipmaps change).

Night skies are a dark natural blue with `star_count` twinkling points
(`sky.py`, alpha from `atmosphere.star_alpha`) and the fog colour equals the sky
colour so the horizon disappears instead of glowing. Buildings (`BuildingDef`)
get a textured floor and a gabled roof (`buildings.py`) so interiors are dark by
day; the `roofs` option switches both off.

### Graphics settings, presets and auto-detection

`GraphicsSettings` (`graphics.py`) is the one mutable settings object: quality
preset, FOV, object/character fade distances (fraction of the far plane),
contrast, shaders, mipmaps, stars, sky bodies, roofs, torch light cap, ground
cell count, perf overlay. It is saved as `graphics.json` beside the quick save in the
per-user data directory (`paths.py`: `%APPDATA%\Unprepared`, `~/Library/Application
Support/Unprepared` or `~/.local/share/unprepared`; `UNPREPARED_DATA_DIR` overrides,
and saves an older version left next to `main.py` are copied in once). On first start
(`auto_quality`) `PygameApp.benchmark` renders 40 frames with the *high* preset
and `perf.recommend_quality` picks a preset from the average frame time (≤10 ms
high, ≤22 ms medium, else low); a software renderer string (llvmpipe, softpipe,
SwiftShader, Mesa offscreen) always yields *low*. "Re-detect best settings" in
the graphics menu reruns it. Presets change the whole bundle; touching any
single option turns `auto_quality` off. `tools/benchmark.py` prints the same
measurements for every preset with per-phase breakdowns.

## Interaction model

`E` does whatever the `InteractionTarget` says. `find_interaction_target`
scores every door, container and NPC in range by how centred it is in the
view (ignoring ones hidden behind walls) and the prompt depends on the
player's stance - **crouching is the theft stance**:

| target | standing | crouching |
| --- | --- | --- |
| unlocked door | open / close | open / close |
| locked door, key in inventory | unlock (then open) | unlock |
| locked door, no key | `[Locked]` | `[E] Pick Lock` (needs a lockpick) |
| NPC | talk, or trade when the shop is open | `[E] Pickpocket` within 2.5 u, else "sneak closer" |
| container (shelf, chest, table) | search | search |
| ore deposit | mine (needs a pickaxe) | mine |
| anvil | forge lockpicks from 1 iron ore (yield and odds from Smithing) | forge |
| lockbox (locked container) | `[Locked]` / unlock with key | `[E] Pick the ... lock` |

**Two-stage targeting.** First a ray from the camera along the view direction
is cast against the solid box of every door leaf (in its current pose), container,
station and NPC in range; the nearest hit that no wall or closed leaf gets in
front of is the target - so any visible part of a door works, from any angle and
however close you stand. Only if the crosshair is on nothing does the older
aim-cone rule apply: targets hidden behind walls or closed door leaves are
ignored and a door only counts when the player faces its doorway
(`InteractionSettings.door_facing_min`). `crosshair_ray=False` restores the
prototype's cone-only behaviour (used by the parity test).

### Progression (`systems/progression.py`)

Every skill obeys one rule set (`ProgressionSettings`):

* `xp_reward(level, base, magnitude) = base * magnitude / (1 + level_decay * (level - 1))` -
  a novice is showered with XP, an elite (50+) earns about a fifth as much for
  the same deed. `magnitude` carries what the deed was worth: `value_xp(gold)`
  (half a point per gold, capped at 250, a sliver for junk) for stealing,
  pickpocketing and bartering; lock difficulty for lockpicking; awake (x1) vs
  asleep (x0.5) and guard (x1.5) victims for pickpocketing; failure multipliers
  everywhere so a fumble still teaches a little.
* `success_chance(level, base) = 1 - (1 - base) * chance_decay ** (level - 1)` -
  odds climb smoothly towards (never reaching) `chance_ceiling`: sneaking (base
  0.45, -0.15 vs guards, -0.2 x visibility), pickpocketing (0.45, +0.3 asleep,
  -0.15 guards), stealing unnoticed (0.3), forging (0.4).
* `SkillTree.xp_needed_for(level) = 100 + 30 * (level - 1)` - linear, so elite
  levels stay reachable; the slow-down comes from the rewards shrinking.
* Sneak XP (`World.sneak_xp_rate`) flows while crouch-moving: `sneak_xp_alone`/s
  with nobody about, and per NPC within `sneak_xp_range` who has not noticed you
  `sneak_xp_behind`/s (asleep in the same room, or back turned) or
  `sneak_xp_in_front`/s (awake, facing you, line of sight), x1.5 for guards, plus
  the per-second sneak roll's success/fail rewards. Crouching halves movement
  speed (`crouch_speed_multiplier`).
* Barter (`EconomySettings`): three prices per item - its *worth* (`Item.value`, always
  shown), the *asking price* (`merchant_markup` 25% over the worth at level 1, less
  `barter_markup_cut_per_level` of the markup per level, gone at 51, never below the worth)
  and the *offer* (`sell_fraction_base` 25% of the worth + `sell_fraction_per_level` 1%,
  capped at `sell_fraction_cap` = the worth). Smithing:
  `scaled_yield` 1..15 picks per ore between level 1 and `lockpicks_max_level`.

### Skill checks (`systems/stealth.py`)

* **Sneaking** - while crouch-moving in view of an awake NPC, a roll is made once a
  second (`sneak_chance`: 0.55 + 0.08/level, -0.15 vs guards). Success hides the
  player from that NPC's *sight* for a second (+4 Sneak XP); failure adds 35
  suspicion and a "skulking" remark (+1 XP). Unseen sneaking trickles 0.5 XP/s.
* **Pickpocketing** - every dip (opening the pockets, and each lift) first rolls
  `pickpocket_unseen_chance` (base 0.4, climbing with the Pickpocket level):
  only when it fails is the crime put to the bystanders (`report_crime`), so a
  deft thief is not convicted by whoever happens to be facing them. A lift then
  rolls `pickpocket_chance` (0.45 base, +0.3 asleep, -0.15 guards). A fumble
  (+10 XP) wakes the victim, throws the player out of the victim's building
  (`World.exit_of(area)`), alarms the victim and reports the crime.
* **Lockpicking** - the sweet-spot tolerance is `15 * (1 + 0.2/level) /
  lock_difficulty` (4..25 degrees) and the needle *wobbles* (`lockpick_wobble`,
  14 degrees at level 1, shrinking with skill, growing with lock difficulty;
  `LockpickMinigame.update(dt)` animates it and `effective_angle` is what is
  tested); a miss snaps the pick and ends the attempt (+15 XP), success pays
  `75 * difficulty`.
* **Hearing vs sight** (`NPC.update_perception`): noise (`noise_multiplier` -
  sneaking gets quieter with skill) within `noise x 10` units only *wakes* a
  sleeper or sets `heard_noise_at`; guards walk over to look
  (`INVESTIGATING`). Only being seen raises suspicion: seen +80/s scaled by
  visibility/distance inside the field of view (`view_half_angle_degrees`,
  default 90 = nothing behind them; line of sight from eye height to chest
  height, blocked by walls, props and closed doors). An NPC's own torch lights
  the player up within `torch_light_reach`. Suspicion decays 25/s; at 100 the
  NPC declares the crime. Shopkeepers keep an eye on anyone on their premises
  regardless of facing unless a sneak roll hides them; fences ignore skulking.
* **Indoor lights** (`world/fixtures.py`): `generate_fixtures` gives every room
  (a named area inside a building, or the building itself) a wall lantern on the
  nearest doorless wall and every table a candle; levels may add `LightDef`s or
  switch the rule off (`auto_lights`). They are point lights (`lantern_color`,
  `candle_color`) and drawn by `models.draw_fixture`. `World.is_dark_at` is
  night-time outdoors or indoors out of every fixture's reach, and
  `update_npcs` lights each awake NPC's torch wherever that is true - no longer
  by the hour.
* **Indoors vs outdoors** (`World.building_at / same_room / noise_attenuation`):
  the `BuildingDef` footprints define "inside". Noise made in another room (or
  outside while the listener is inside) reaches them at
  `wall_noise_attenuation` (25%), so sprinting past the guard house does not
  wake the off-duty shift but sprinting *inside* it does. Cries for help use
  `shout_radius` in the open and `shout_wall_attenuation` through walls; only
  guards within earshot respond.
* **Warnings are rate-limited** (`NPC.scold`): the same kind of dressing-down
  ("skulking", "trespass", "torch") is not repeated within `scold_cooldown`,
  and every warning opens a `warning_grace` window during which suspicion is
  capped just under the alarm, so the player has time to comply. A guard's
  warning also turns the player to face them (`Player.hold_attention`).
* **Trespass** - a restricted `AreaDef` (houses, guard house, jail cell) adds
  `trespass_suspicion_gain`/s with a warning unless the area `trusted_by` the
  NPC includes the player's patron or it is within `public_hours`. At night a
  guard who sees you without a lit torch tells you to light one.
* **Curfews** (`AreaDef.curfew_hours`, `World._tick_curfews`): in a trusted area
  after curfew the resident owner goes `WAITING`, asks the player to leave every
  `curfew_ask_interval` seconds and after `curfew_asks_before_eviction` requests
  throws them out (`World.throw_out`). Leaving ends it.
* `Player.level` = 1 + (total skill levels gained) // 2.

### Shopkeeping (`systems/commerce.py`)

`MerchantStock` unions a shopkeeper's pockets with its `shop_containers`, so the
trade screen sells shelf goods too (sales adjust the expected stock). On arrival
at the counter and when leaving the shop the keeper `take_stock`s: anything
missing versus the last count becomes a `TheftReport` (area, day, items) that
every guard learns. Guards remember `Sighting`s (area + day) whenever the player
makes them suspicious; a report matching a sighting within two days becomes a
pending confrontation, delivered the next time the guard sees the player
(`World._confront`). **Hearsay convicts** - by design, to reward care: the guard
takes back the *reporter's stolen* goods the player still carries (matched by
`Item.owner == report.reporter and is_stolen`, and no more units than the report
lists - a bottle bought and paid for is the player's whatever it is called),
records the theft as a witnessed crime (`remember_crime("Theft", witnessed=True)`)
and charges on the spot like any other grudge; it fades like one too. With no
goods to recover the player gets a warning and a watchful eye.
Every `restock_days` the shop is topped up (`restock`) and `StockTier`s whose
`min_player_level` is reached are added to the shelves. At midnight every NPC
whose `inventory_reset_days` is due gets its defined pockets back (fresh keys
and rations), so stolen keys are a temporary advantage.

### Alarms, memories and justice (`systems/justice.py`)

`World.declare_crime(crime, witness)` is the single alarm entry point: the
witness calls it out, every guard *within earshot* (`can_hear_cry`, sleeping
ones included) drops its task and goes `RESPONDING` towards the scene at 1.6x
speed, and each stores a `CrimeMemory` (crime, day, fine from
`JusticeSettings.fines`, expiry: 21-35 days if witnessed, 3-5 days if merely
reported). A shopkeeper's theft report (`World.report_theft`) likewise brings
the guards who can hear it to take the report; one who has a `Sighting` of the
player lurking there says so and queues a confrontation.

Declaring a crime starts the clock on a **chase**: every guard who learns of it gets
`charge_ready_at = now + JusticeSettings.charge_delay` (3 s). The next time a guard
holding an unexpired memory *sees* the player he shouts "Halt!" and `pursue`s them
(`ALERT_GUARDS`, running, re-planning the route every half second while they are in
view; out of sight for `SEARCH_SECONDS` he gives up - "Gone... but I'll know that
face" - and the memory stays). Only once the delay has passed *and* the guard is
within `charge_reach` (2.5 u) does `charge_player` run - so a thief who sprints has
a real chance to get away. (A confrontation over a shopkeeper's report - "Hand it
over!" - is face to face already and settles on the spot.) `charge_player`: `World.player_caught` (the player turns to face the guard,
the game closes the mini-game / pocket screen / question), `justice.judge` works
out the bill and the fines are paid as far as the purse goes, stolen goods are
returned to their owners (`World.owner_inventory_for`), and either that is the end
of it - plus `evict_if_trespassing`: caught somewhere they have no business being,
the player is also shown the door - or the days owed are served.

**Penalties** (`JusticeSettings`, `Penalty(fine, jail_days, jail_days_unpaid)`):
a crime's penalty is fixed by `penalty_for(crime, indoors)` when it is declared
(`CrimeMemory` keeps it): the *theft crimes* (pickpocketing, lockpicking,
shoplifting, theft) cost `outdoors` (100 g / 2 d / 3 d) or `indoors` (500 g / 5 d
/ 7 d) by `World.is_indoors(location)`; the rest have penalties of their own
("Skulking about" 20 g, "No torch after dark" 5 g, "Trespassing" 15 g, "Fencing
stolen goods" 30 g - none with a sentence unless unpaid, then a day; "Jailbreak"
100 g / 2 / 3). `judge` pays the crimes in order of severity (a crime the purse
covers costs its fine and `jail_days`, one it does not costs `jail_days_unpaid`),
then every stolen unit the player carries (`stolen_item_jail_days` plus its value,
`stolen_item_unpaid_jail_days` more when that cannot be covered), the same crime
known to several guards counting once (`CrimeMemory.key`). The torch law becomes
the crime "No torch after dark" either through the ordinary suspicion alarm (an
unlit walker at night) or after `torch_warnings_before_crime` ignored warnings
(`NPC.torch_warnings` / `pending_crime`).

The jail is level data (`JailDef` → `JailState`): `imprison` leaves what is left of
the purse alone, moves the belongings (minus one lockpick) into the confiscation lockbox tagged
as the player's own property, locks the lockbox, locks and *seals* the barred
cell door and requests a relocation into the cell. While sealed, no NPC opens
the bars (`NPC.can_open`), guards never investigate noise from the cell ("Keep
it down in there!"), and the nav grid treats the cell as closed. `update_jail`
releases at `release_hour` of the last day: bars and lockbox open, the jailer
walks to the post and `WAITING`s - through his bedtime if need be - until the
ex-prisoner has left the jail house (or `release_grace` runs out), then the
lockbox is locked again (`end_release`). Leaving the jail area while still
serving is noticed at once (`check_jailbreak`) and is a `Jailbreak` crime. The
jail house's outer door is never locked and owned by nobody, so a released
prisoner can always walk out. `[T]` in the cell asks whether to wait out the
whole sentence (`Game.wait_out_sentence`).

**Relocations and fades.** Every forced move - eviction, imprisonment - goes
through `World.relocate_player`, which hands a `Relocation` to the game's
handler. `Game` wraps it in a `ScreenFade` (0.6 s to black, 0.2 s hold, 0.7 s
back; a `jail_fade_delay` of 2 s first when the reason is "jailed"), applies
the move at black, freezes input meanwhile, and the HUD draws `fade_alpha`.
Being caught (`World.player_caught`) also locks the controls for
`caught_freeze_seconds` (`Game.frozen`). `World.relocating` suppresses judgement
(jailbreak checks, charging, curfews) until the move has landed.

**Trust.** `World.area_trusts_player(area)`: outside curfew hours an area is
trusting if it lists `trusted_by` or - the standing rule - its owner is a fence;
then nobody, guards included, treats the player as a trespasser there and a fine
never comes with an eviction. Loading a save resyncs routines with
`doors=False`, so NPCs resume their tasks without re-applying lock/unlock side
effects and every door stays exactly as saved.


### Transfer sessions (`systems/transfer.py`)

Trading, pickpocketing and containers share one `TransferSession`: two item
lists (left = the other party, right = the player), `Tab` switches the active
column, ↑/↓ move the cursor, Enter moves **one unit** of the selected item
across (stacks split and merge via `entities/inventory.py`). A
`TransferPolicy` decides what a move means:

| policy | left → right | right → left |
| --- | --- | --- |
| `TradePolicy` | buy at `buy_price` - the *asking price*: worth x (1 + `markup`), the markup haggled down by Barter and never below the worth; services (rooms) at their posted worth; stock = `for_sale` items only; the merchant's purse grows | sell at `sell_price` - the *offer*: worth x `sell_fraction` (a quarter at level 1, +1%/level, capped at the worth; half that for hot goods at a fence) if the merchant can afford it (`MERCHANT_BROKE` otherwise); stolen goods make a non-fence merchant call the guards and end the session; Barter XP by price. Rows show the worth beside the price; `sell_price <= value <= buy_price` always, so buy-and-sell-back loses money |
| `PickpocketPolicy` | steal: flagged stolen, value-scaled Pickpocket XP, crime reported to the *bystanders* (the victim excluded: whether they feel it is the lift's own roll); the first row is the victim's coin purse (`purse_grab` coins per lift). A fumble (`Game._pickpocket_caught`) alarms the victim once and declares the crime with them as its witness | plant the item on the victim |
| `ContainerPolicy` | take; your own property (`Item.owner == "player"`) is simply taken back; from an *owned* container: Stealing roll (unnoticed or reported), flagged stolen, value-scaled Stealing XP. `[X]` (`TransferSession.take_all`, `can_take_all`) sweeps the whole container across one unit at a time through the same rules - containers only, never pockets | put |

The HUD draws all three with one widget, so lockboxes, chests or corpses later
only need another policy (and a `ContainerDef` in the level).

## NPC daily routines

Every NPC has a `schedule`: a list of `ScheduleTask`s covering the day. A task
says *where* to be (`target_pos`), *what* to do there (`target_state`:
`SLEEPING`, `EATING`, `SHOPKEEPING`, `PATROL`, ...) and which doors to operate:

| field | effect |
| --- | --- |
| `unlock_doors` / `open_doors` | unlocked (resp. unlocked + opened, e.g. the shop for business) on arrival |
| `close_doors` / `lock_doors` | closed (resp. locked) on arrival - Barnaby locks his house at bedtime |
| `patrol_points` | after arriving, keep walking the route point to point (pausing `NPCSettings.patrol_pause`) |
| `depart_text` / `action_text` | announcements when setting off / arriving |

On every hour tick `NPC.update_schedule` picks the task for that hour; if it
changed, `begin_task` asks the world for a path (`World.find_path`), the NPC
goes `TRAVELING` and `update_movement` (called every frame from `Game.update`)
walks it along the waypoints at `NPCSettings.move_speed`. **Door rules**
(`NPC.update_movement`): a closed door is opened only when the step actually
crosses the doorway *and* the NPC may pass (`NPC.can_open`: unlocked, or the
matching key in its pockets - ownership alone grants nothing); the NPC closes
behind it what it opened and relocks what was locked; a door it *owns*
(`Door.owners`: NPC names or a faction tag such as `"guards"`) is always closed
and locked behind it, even if the player unlocked it; an already-open door is
left alone. Nobody is ever trapped: `World.push_out_of_doors` nudges the player
out of a closing leaf. A `DoorKind.BARS` door (the jail cell) is exempt from all
of this: NPCs open it if they may pass but never shut or lock it as a side
effect, it is see-through, and while *sealed* (a prisoner inside) no NPC opens
it whatever keys it holds - the ordinary leaf-door code is untouched.

**Solid people** (`NPC.update_movement`, `World.bodies_near`): a step that would
overlap another awake NPC or the player is refused; a sidestep is tried first,
being blocked within `CROWD_TOLERANCE` of the goal for `BLOCKED_PATIENCE`
seconds counts as arriving (a crowd stands around a spot instead of inside each
other), and being blocked anywhere for `BLOCKED_GIVE_UP` seconds abandons the
walk (`IDLE`, re-planned on the next hour tick). The player collides with
upright NPCs too (`World.npc_colliders`); anything already overlapping the
player is ignored for that move so nobody is ever pinned. Arrival applies the task's door actions and state
(`unlock_doors` only unlocks - the shop stays shut-but-unlocked for business);
sleeping NPCs are placed *on* their bed and rendered lying flat
(`World.npc_body_box`); eating NPCs take food from the task's `food_container`
first, then their own pockets, and complain when there is none. Investigating /
responding / following NPCs leave the routine and return to it on the next hour
once calm. Game start replays the day so far, so owned doors between areas start
locked.

Game start places everyone at the task for the start hour and replays the
door side effects of the tasks already completed that day; loading a save
keeps the saved positions and has NPCs walk to where they should be.

Barnaby's day in `world/levels/town.py`: bed (00-07) -> breakfast at the table
(07-08, locks his front door when he leaves) -> counter (08-13, unlocks the
shop) -> lunch in the back room (13-14) -> counter (14-20, locks the shop when
he leaves) -> dinner at home (20-21, unlocks his front door) -> bed (21-24,
locks it). The three guards rotate through the same four task kinds with
8-hour offsets (`guard()` in the level file): Vance patrols 06-14, eats, stands
the tower post 15-22 and sleeps 22-06; Orrin guards the door 06-14, patrols
14-22, eats and sleeps 23-06; Mira walks the night watch 22-06, eats, sleeps
07-14 and guards 14-22 - so the town always has one guard on the route, one on
post and one resting in their own bed. Jae stays in the hideout. Adding a
routine to a new NPC is purely data (`TaskDef`s); new activities only need a
new `AIState` value.

### Navigation (`world/navigation.py`)

The level's static geometry (walls, props, containers, anvil) is rasterised
once into a numpy occupancy grid over `LevelDefinition.bounds` (0.5 u cells,
inflated by the agent radius). Per query the
doors the agent cannot pass are overlaid, A* (8-connected, no corner cutting)
finds a cell path and a string-pull pass straightens it. Start/goal cells inside
obstacles snap to the nearest free cell; an unreachable goal yields an empty
path and the NPC simply adopts the task state where it stands.

## Verification strategy

```
tests/
  baseline/        frozen verbatim copy of the original main.py / models.py / world.py
  unit/            ~1000 headless tests for every module (no display)
  regression/
    test_scenario_goldens.py  34 scripted play-throughs replayed headlessly vs committed goldens
    test_pixel_goldens.py     same scenarios in the real renderer vs local framebuffer goldens (opt-in)
    goldens/                  committed state goldens (digest of every frame + capture-frame snapshots)
  parity/
    scenarios.py              the scenario scripts (shared by regression, parity and the gallery)
    harness.py                deterministic pygame clock/event/key/flip replacement, dry_run()
    test_parity_e2e.py        prototype vs game player-loop state for scenarios that must still match,
                              the fixed NameError crash, ESC -> pause menu instead of quitting
    test_parity_logic.py      randomised side-by-side runs of the perception/noise rules kept verbatim
    render_gallery.py         dumps capture frames to PNG for eyeballing
  display/         OpenGL smoke tests of the real app: resolutions, missing assets (placeholders),
                   python main.py, start/pause/graphics menus via synthetic keys, tools/benchmark.py
```

Scenarios are authored against the headless game (`harness.dry_run`) and are
the single source of truth for three suites: the **state goldens** (fast,
portable, committed - behaviour changes show up as golden diffs in review),
the optional **pixel goldens** (renderer regressions on a given machine) and the
**gallery** for visual review. The prototype parity harness still replaces
`pygame.time.Clock`, `pygame.event.get`, `pygame.key.get_pressed` and
`pygame.display.flip` with a scripted version, runs the prototype's `main()` and
`PygameApp` in separate subprocesses and compares the player-visible state of
every frame (rendering is no longer compared: the town and HUD have changed).

## Change log vs the prototype

### Fixed / resolved (all covered by tests)

| Issue in the prototype | Now |
| --- | --- |
| Selling stolen goods to a merchant crashed (`NameError`) | merchant refuses, reports you, menu closes (`test_prototype_crash_is_fixed`) |
| No collision: the player walked through walls and doors | sub-stepped, sliding box collision against walls, props, the anvil and door leaves (`systems/collision.py`) |
| Lock sweet spot always 45 deg, pick angle carried over between locks | fresh random sweet spot and pick reset per attempt (seedable via `GameSettings.random_seed`) |
| A snapped pick was never lost | the pick is consumed; losing the last one ends the mini-game |
| `E` during lockpicking restarted the mini-game and re-reported the crime; the prompt showed through the panel; needle colour never set | `E`/prompt suppressed in the mini-game, needle is grey until aligned |
| Torch, crouch, forge, pickpocket, save, load worked inside menus | only in explore mode |
| `[P]` missing from the controls panel | listed |
| Saves lacked door state, orientation, NPC inventories, clock minutes | save format v2 (v1 files still load) |
| Jae stood inside the hideout's wall (never a witness) | moved to (-12, 13) |
| "(Guard Guard Captain Vance ...)" | `NPC.title` never doubles the role |
| `is_sprinting` while standing still holding Shift | requires movement |
| Night began at 20:00 for torches/sky but 21:00 for ambient light | `DayNightCycle.is_night` is the single rule |
| Emoji in messages rendered as boxes | removed |
| Trade cursor could point past the list after a purchase | clamped |
| Big XP grants levelled only once | loops |
| All NPCs started asleep regardless of the hour; Barnaby sold his shop key; pickpocketing took nothing | routines applied at start; `Item.for_sale`; pickpocketing lifts the victim's top item |
| Line of sight ignored doors and props | closed door leaves and props block sight |

### Gameplay changes in the third iteration

* Crouch is required for lock picking and pickpocketing; `P` is gone, everything is `E` (`interaction model` above). The camera lowers while crouched.
* Talking distance doubled (6 u from the eye); prompts never target things behind walls.
* Trade/pickpocket/container screens share the Tab-switched two-column `TransferSession`; pickpocketing can plant items.
* Items can stack (`Item.stackable/quantity`, shown as `Name (n)`); lockpicks, food, wine... Forged picks merge into the stack.
* Dialogue panel spans the width of the window and word-wraps; long messages are no longer cut off.
* Town grew to 80x80 with a guard house (3 beds, large table) and a watchtower on the east wall; the guard-only door relocks behind guards.
* Three guards on rotating shifts with a patrol route; escalating call-outs per witnessed crime.
* Barnaby's shop has two shelves of loot (`ContainerDef`, owner = Barnaby, crime "Shoplifting"), a counter and a back-room table; he locks his house at night and when leaving (`lock_doors` on arrival in bed; as the owner he locks the door behind him whenever he walks out).
* Save format v3 (containers, stacks, crime counters); v1/v2 files still load.

### Gameplay changes in the fourth iteration

* Stolen items carry no "Stolen" prefix; the HUD draws a bold red `(S)` tag. Fences launder what they buy.
* Unlocking never opens: keys unlock, `E` again opens; NPCs shut every door behind them and relock the ones that were locked (or marked `relock_after_passage`), even if the player had unlocked them. Task `unlock_doors` only unlocks; `open_doors` opens the shop for business.
* Inventory panel scrolls with ↑/↓ (no item limit).
* Eyes and field of view for NPCs; line of sight from eye height; NPCs face their walking direction or a task `facing`.
* Jae has a bed, a table and a routine inside the hideout.
* Composite merchant stock, stock-taking, theft reports, guard sightings and confrontations, 3-day restocks with level-gated tiers (`StockTier`).
* Iron ore (mined with a pickaxe at the corner deposit) is consumed to forge three picks.
* Skill checks for sneaking / pickpocketing / lockpicking; per-second suspicion rates.
* New renderer: GLSL lighting, atmosphere, fog, sun & moon, torches, procedural textures, NPC models.
* Save data grew (facing, expected stock, guard memory, day count); older files still load.

### Gameplay and engine changes in the fifth iteration

* **Menus**: start screen (New / Load / Graphics / Exit), ESC pause menu (Resume
  / Save / Load / Graphics / Exit Without Saving), graphics menu with quality
  preset, FOV, fade distances, contrast, shaders, stars, roofs, perf overlay and
  re-detection. ESC no longer quits from the explore mode (`escape_quits`
  scenario documents the difference to the prototype).
* **Graphics settings & performance**: `GraphicsSettings` presets persisted as
  JSON, start-up benchmark picks one (software renderers -> low), `FrameProfiler`
  phases and draw stats behind F3, `tools/benchmark.py`.
* **Visuals**: natural dark-blue night sky with stars, fog matching the sky,
  textured floors and gabled roofs, improved ore texture, per-pixel contrast.
  Texture generation moved out of the package into `tools/textures/`.
* **Doors**: ownership no longer opens doors - keys do; NPCs close/relock what
  they opened, owners always lock their own doors; no trapping; `relock_after_passage` removed.
* **Perception split**: hearing wakes and draws investigation only; sight
  raises suspicion; sneak roll once a second; skill reduces noise; trespass in
  restricted areas (with trust and public hours); torch law at night; the
  shopkeeper watches anyone on his premises; guards turn towards and follow skulkers.
* **Justice**: alarms send all guards; crime memories with witnessed/reported
  lifetimes; wanted players are charged on sight (goods returned, fines or jail);
  a jail building with cell, jailer, confiscation chest, sentences, release and
  jailbreaks. Save format v4 stores the jail state.
* **Daily life**: NPCs eat (table first, then pockets); midnight inventory
  resets (fresh keys); `T` waits an hour; no sprinting while crouched; lockpick
  wobble that shrinks with skill; the anvil is a station used with `E` (`R` gone).
* **Targeting**: doors hidden behind walls or seen edge-on are not targetable;
  closed doors hide what is behind them.
* **Tests**: justice, guard AI, perf, wobble and texture unit tests; scenarios
  `guards_respond`, `jail_and_release`, `wait_key`, `shoplifting`; display tests
  for menus, missing assets and the benchmark tool.

### Gameplay and engine changes in the sixth iteration

* **Indoors/outdoors**: building footprints muffle noise (25% through walls);
  shouts carry `shout_radius` in the open, half through walls; only guards in
  earshot respond. Running past the guard house no longer wakes the sleepers.
* **Jail rework**: `JailDef` per level; the cell is a `DoorKind.BARS` gate that
  nobody shuts as a side effect and that is *sealed* while a prisoner is inside
  (no jailer "check-ups"); the jail house is public with a never-locked door;
  confiscated belongings are the player's property (taking them back is not
  theft) in a locked `lockbox` opened on release; release at 06:00; the jailer
  waits up until the ex-prisoner leaves; `[T]` in the cell offers to serve the
  whole sentence; fines inside a restricted area also evict.
* **Solid people**: NPC-NPC and NPC-player collision with sidestepping, "near
  enough" arrival for crowds, give-up-and-retry when hemmed in.
* **Targeting**: crosshair-ray first (any part of a door, any distance), aim cone second.
* **Warnings**: per-kind cooldown and a grace window before the alarm; guards'
  scoldings, fines and evictions turn the player to face them and close any screen.
* **Curfews**: `AreaDef.curfew_hours`; Jae asks, asks again, then throws you out.
* **Progression**: six skills (Stealing and Barter added), linear XP curve, one
  reward/odds rule (`systems/progression.py`), value-scaled XP, reworked Sneak XP,
  skill-scaled forging (failure, 1..15 picks), Barter prices.
* **Economy**: NPC purses (`NPCDef.gold`), merchants pay from them, coin rows in
  pickpocketing, purses restored each restock cycle.
* **Presentation**: fade to black around evictions and imprisonment; yes/no
  questions; ceilings under roofs; lintel beams over doors; mirrored door
  textures; barred gates; four-legged tables; lockbox texture; sun halo; slimmer
  controls panel.
* **Menus**: Continue (newest save), Save Game / Load Game slot lists (newest
  first, scrollable, overwrite confirmation, `saves/` directory), Exit to Menu
  with confirmation back to the start screen. Save format v5.
* ESC in the explore mode is now purely the app's business (`Game.escape()`
  returns False); the headless game no longer quits on ESC.

### Seventh iteration

* Ceilings span the roof base (overhang included); textures are no longer
  uploaded upside down (padlocks hang the right way up, shelf goods sit on their
  boards); shelves and lockboxes show their picture on the room-facing side only
  (`draw_fronted_box`, `lockbox_side`); scroll hints in transfer screens have
  their own line.
* Door lock state survives save/load (`resync_schedules(doors=False)`).
* A fence's house welcomes the player by day; guards never evict from it.
* Wall lanterns and table candles in every room; NPC torches lit by darkness.
* Sneak XP: trickle alone, "behind" rate for sleepers, longer range; crouching
  halves speed. Being caught freezes the controls; the jail fade is delayed.

### Eighth iteration - the town

* **A bigger town** (`world/levels/town.py`, rewritten): one cobbled main
  street (`RoadDef`s; `World._build_roads` adds a doorstep strip for every door
  that stands back from the paving) with the well at the origin; north side the
  smithy with an open-sided shed (`BuildingDef.open_sides`: roof on posts, no
  floor, not indoors for sound/trust/lighting but still a roof for the
  renderer), the ore deposit behind it, the inn, the general store, the tailor
  and the guard house + watchtower; south side six cottages and the jail. Named
  spots are module constants (`SHOP_COUNTER`, `INN_SEATS`, `PATROL_ROUTE`,
  `STROLL_WEST/EAST`, ...). Fifteen NPCs: four merchants (Barnaby, Hilde the
  innkeeper, Gareth the smith, Mirelle the tailor), Jae, five townsfolk, four
  guards and the jailer. New `AIState`s `STROLLING` (a round with
  `patrol_points`), `SHOPPING`, `WORKING`, `DRINKING`, `CLEANING`; the schedule
  helpers `resident_tasks`, `stroll`, `visit`, `guard` keep the data short.
  `NPCSettings.move_speed` is 3.6 (a game hour is 24 real seconds).
* **Specialists**: `NPCDef.trade_categories` / `NPC.buys(item)` and
  `Item.category`; `TradePolicy.give` answers `NOT_THEIR_TRADE` without closing.
  The smith sells while `WORKING` (`is_open_for_business`).
* **The inn** (`world/inn.py`, `InnDef` in the level): rooms are `service`
  items in the keeper's stock; `TradePolicy.take` hands them to
  `Game._buy_service` -> `InnState.rent` (unlocks + opens the room, extends on
  repeat); `World.area_trusts_player` trusts the rented room until
  `checkout_hour` on the last day; `InnState.update` relocks and evicts. Saved
  under `"inn"`.
* **Dialogue** (`core/dialogue.py`): every spoken line is one of several
  phrasings (`LINES[kind]`, `NPC.phrase`) so tests check *kinds*
  (`tests.conftest.said`). `NPC.say` logs `(Name): "..."` and sets a `Speech`
  the HUD draws as a word-wrapped bubble (`Game.speech_bubbles`: within
  `HEAR_RADIUS`, roughly in front of the player). `build_conversation` scripts
  greeting -> 1-2 topical exchanges (role-coloured; the person a topic is about
  answers) -> goodbye; `World._tick_talk` starts them between sociable NPCs who
  stop within `CHAT_RANGE` in the same room (guards rarely), at most
  `MAX_CONVERSATIONS`, one within the player's earshot, with recent openers
  avoided; `_ambient_chatter` adds shop/inn/work lines and simulated purchases.
* **Waiting**: somebody whose path is blocked by a locked door
  (`begin_task` finds no route) walks to `World.waiting_spot_for` (the
  doorstep), says so, and retries every `NPCSettings.locked_out_retry` seconds
  (`NPC.retry_blocked_task`); the next task abandons the wait.
* **Pantries**: `World.pantries` remembers the food every kitchen table started
  with and `_daily_upkeep` lays it again every `EconomySettings.pantry_restock_days`.
* **Movement fixes**: a woken sleeper steps off the bed (`_get_out_of_bed`); an
  owner carried through a doorway in one long frame still shuts and locks it
  (`_will_pass_through` looks at the step just taken; owned doors are always
  relocked); near a crowded goal an NPC stops instead of shoving; an
  investigating guard keeps watching a suspect he can see.
* **Attention**: `Player.hold_attention` + `update_turning` lock the controls
  and swing the view smoothly (`PlayerSettings.attention_turn_speed`) towards
  a scolding guard (`PerceptionSettings.scold_freeze_seconds`) or one who has
  caught you (`caught_freeze_seconds`) - yaw and, since the thirteenth iteration,
  pitch too (`Player.pitch_towards`: level with their face when standing, up at it
  from a crouch).
* **Barter** weighed prices by value (`TradePolicy.barter_margin`; replaced in the
  tenth iteration by the three-price model).
* **Lighting**: point lights are confined to their room
  (`PointLight.bounds`, `uLightBounds` in the shader) so nothing bleeds through
  walls; floors are inset by the wall thickness; the kiln is a `LightDef("fire")`
  that also keeps the yard "lit" for torches (`World.is_dark_at`). Fascia boards
  close the gap between roof and walls (`draw_fascia`). The sun rises at 05:30
  and sets at 19:30 (`atmosphere.SUNRISE/SUNSET`), the moon opposite.
* **Jail**: the cell's gate is two units further south so its open leaf never
  blocks the office; the jailer's post faces it.
* The lock mini-game draws sweet spots from its own RNG (`Game.lockpick_rng`)
  so scripted runs can predict them whatever the town is chattering about.
* Test suite: `test_dialogue`, `test_inn`, `test_town_rework`; all scenarios
  re-routed for the new map plus `inn_rent_a_room`, `smith_trade`, `town_life`.

### Ninth iteration - bugs, the look of the town, the tower

* **Fixes**: pantries are laid every third night (`pantry_restock_days`, tables
  stocked for three days); town chatter (conversations, ambient lines, "I'll
  wait here") is bubbles only - `NPC.say` without a callback - while warnings,
  greetings and reports still go to the log; sleepers hear only their own
  *chamber* (`World.same_chamber`: same building and no wall or shut door on
  the segment between - `walls_of(building)`), so a guest crossing the inn's
  hall no longer wakes the keeper; bubbles stay up ~3x longer and longer still
  for long lines (`NPC.SPEECH_BASE_SECONDS/PER_CHAR`).
* **Theft reports as exchanges**: `report_theft` sends guards running with
  `NPC.report_to` set; `_go_and_take_report` follows the shopkeeper and, once
  beside them, `_take_report` builds `dialogue.build_report_exchange` (an
  *official*, *logged* `Conversation` - seen through unless somebody is asleep or
  alarmed, with an `on_finish` hook) whose lines depend on `saw_someone(guard)`
  and `saw_someone(merchant)`: everybody now records sightings when their
  suspicion passes the threshold and when they witness/report a crime. The
  merchant's lead becomes the guard's `pending_confrontations`; `_confront`
  words it accordingly. Both turn to face each other (`_begin_conversation`)
  and `resume_routine` afterwards. A merchant refusing stolen goods now
  `declare_crime("Fencing stolen goods")`, so his "Guards!" brings them.
* **Light and torches**: the sun is back to 06:00-18:00 (`core.daynight.sun_direction`,
  re-exported by `rendering.atmosphere`); `DayNightCycle.is_torch_time(hour)` is
  now the sun's elevation (`TORCH_SUN_ELEVATION`: torches from ~17:30 to 06:30)
  and `World.is_dark_at` is the plain rule - outdoors: torch time; indoors: no
  fixture in reach.
* **Inn**: `World.is_complimentary` - food and drink in the keeper's containers
  are free for a paying guest (`ContainerPolicy.complimentary`/`on_complimentary`
  keeps the stock count straight). `Game.sleep_until_morning` (`TargetKind.BED`,
  beds are crosshair targets that lose to anyone lying on them;
  `World.player_may_sleep_in`: `Prop.owner == "player"` or the rented room's bed)
  fades slowly (`ScreenFade(out=1.8, hold=0.8, in=1.4, reason="sleep")`) and
  `World.wait_until(6)`.
* **Pickpocketing**: `PickpocketPolicy.purse_grab` = `purse_grab + purse_grab_per_level x (level-1)`.
* **The player's height**: `player.position` is at the feet (y = floor, save
  v6 lowers older saves), eye 1.55 / crouched 0.9, collision box from
  `collision_foot_clearance` (0.25) to `collision_head_height` (1.7) - 3D, so
  upper wall pieces and overhead steps count only where they should. NPC sight
  aims at `NPC.chest_height(player)` (eye - 0.2): crouching behind low cover hides.
* **Outfits** (`entities/outfit.py`): `Outfit` data (tunic/legs tints over a
  neutral `cloth` texture, `armor`, `apron` cloth/leather, `hat` helmet/cap/scarf/
  hood/coif, `sash`, `hair`), `outfit_for(name, role)` picks one per role with
  name-stable colours; `NPCDef.outfit` overrides. `models.draw_outfit` dresses the
  model; sleepers keep their tunic colour.
* **Modelled props** (`rendering/structures.py`, `PropDef.kind`/`facing`/`solid`):
  the well (ring, water disc, posts, windlass, bucket, gable roof), the forge
  (waist-high hearth, sunken coal bed with small flames, hood and chimney,
  bellows), the inn's bread oven (mouth, fire set back, warming shelf) and shop
  signs (`sign_*` textures on a bracketed board, `solid=False`). The `fire`
  fixture is now a light only. New textures: cloth, leather, water, coals, four
  signs (generators in `tools/textures`).
* **Smithing**: `StationDef.requires_item_id` applies to crafting stations too -
  the anvil wants a mallet; the smith carries a mallet and a pickaxe and sells
  mallets (as does Barnaby); his trade includes tools.
* **Distance fog**: `LightingRig.begin_frame(fog_far=object draw distance)` ->
  shader `uFogFar` (linear fog closing at the draw distance, combined with the
  weather's exp2 fog) or, fixed-function, `GL_LINEAR` when it is the thicker of
  the two (`distance_fog_wins`).
* **The watchtower** (`world/tower.py`): `build_tower` returns walls (with
  window openings: wall pieces split around them), props (40 wooden `step`s
  winding round a solid newel column, `floor` platform pieces minus the head
  room over the top steps), lanterns and a `StairDef` (bottom, waypoints, top,
  region). Steps and platforms are *floors*: `World.floor_height_at` puts the
  player on the highest one within `STEP_UP` of their feet each frame
  (`Game.update`), `World.colliders` makes the unreachable ones solid, and
  `sight_blockers` includes them. `World.find_path` splices `_stair_path`
  waypoints (with y) before/after the grid path; `NPC.update_movement` rises
  with them; `bodies_near` ignores people on another level; `NavGrid` ignores
  obstacles starting above head height (lintels, window pieces); a look-out up
  high hears cries from the street (`can_hear_cry`). `Door.swing_out` lets the
  tower door fold into the guard house. The guard house grew west (30..42) for a
  fourth bed; Guard Aldric keeps the tower watch (07-14, 15-22); Vance's evening
  post moved to the street by the inn (`POST_WEST`); guards carry a tower key.
  `BuildingDef.open_top` skips the ceiling and roof.
* Tests: `test_tower`, new inn/sleep/on-the-house tests, dialogue report-exchange
  tests, outfit/height/targeting updates; scenario `tower_climb` (pick both doors
  at 02:00 and climb to the platform) and the inn scenario sleeps till morning.

### Intentional differences kept from the first refactor

No GLU dependency, cached glyph textures, NPC torch lights capped at 6,
assets in `assets/textures/`, window title "Unprepared", mouse released on
window close, `Player()` starts empty (the kit is level data).

### Known quirks / tuning candidates (not bugs)

* Hearing radius is `noise x 10` and ignores walls (a sprint is heard 28 units
  away) - but noise now only wakes and attracts, it never convicts.
* The jail release grace period (`release_grace`, 180 s) is *elapsed* game
  time, independent of the time scale.
* The aim cone is wide (`aim_dot_threshold` 0.6, about 53 degrees) for parity
  with the prototype; the door-facing rule trims its worst case.
* A casual conversation is cut short when a participant's routine moves them
  on (the next hour tick), so a greeting occasionally goes unanswered.
* From the tower platform a guard's 14-unit view reaches little more than the
  yard below; the watch is scenery and a route, not a sniper's nest.
* Phrasings (which of several lines an NPC says) come from `World.phrase_rng`, a
  stream of their own seeded like the gameplay RNG, so how something is worded
  never changes the outcome of a roll; the lock mini-game has a third stream for
  the same reason. Conversation *timing* still depends on the whole run (goldens
  capture the whole log).
* The town is large for a walking pace of ~3.5 units a second: a walk from one end
  to the other (the cooper's supper after a drink at the inn, the tanner to the
  inn) can take most of an in-game hour (24 real seconds), so a one-hour task at
  the far end is sometimes reached just as the next begins.
* The draw distance (far plane 100, distance fog before it) hides the far side of
  the mountains and the sea's horizon; the map's edges are the land itself.
* The rock a cave is dressed with is set into its walls on purpose (a boulder half
  in the rock face); only what lands on something furnished is taken up.
* Doors only turn on hinges along x (walls running east-west); every doorway in a
  north-south wall is an open way.

### Sixth iteration: review fixes

* **Saves live in the per-user data directory** (`paths.py`; Windows `%APPDATA%`,
  macOS `Application Support`, Linux XDG) with an `UNPREPARED_DATA_DIR` override and
  a one-time copy of saves left beside `main.py`. `GameSettings.save_file` is `None`
  by default and resolved through `resolved_save_file`; `graphics.json` sits beside
  the quick save. `graphics.json` is no longer committed.
* **Everything has an owner** (`Item.owner`): built for the NPC or container that
  holds it (`ItemDef.build(owner=...)`, restocks and midnight resets included);
  `Item.steal()` flags without changing hands, `Item.transfer_to()` is the honest
  change of hands (buying, selling, a gift planted on somebody, a meal on the
  house) and launders through a fence. `World.owner_inventory_for` returns stolen
  goods by owner (their shelf that stocks it, else their pockets, a faction's
  table or lockbox, else the confiscation lockbox for goods of unknown origin).
  Save format v7 (older saves: whatever an NPC or container held is theirs).
* **A chase before the charge** (`JusticeSettings.charge_delay` / `charge_reach`,
  `NPC.pursue`, `NPC.charge_ready_at`): see "Justice". Guards run when alerted;
  an NPC hemmed in by a crowd now also tries stepping straight away from it.
* **Pickpocketing rolls against the onlookers** (`pickpocket_unseen_chance`)
  instead of reporting every dip to whoever is facing the player.
* **Schedules catch up every frame** (`NPC.catch_up_schedule`): an NPC too
  suspicious or busy at the hour tick takes up the hour's task the moment it is
  free instead of an hour late.
* **Witnesses look at the chest** like all perception (`witness_crime(...,
  target_height)`): a crouching thief behind a counter is hidden from a crime
  report exactly as from ordinary sight.
* `WorldClock` carries the sub-hour remainder over instead of dropping it;
  `World.exit_of` never "throws out" into another building (the watchtower's exit
  is the guard house's); a fumbled pickpocket no longer double-prefixes its line;
  the graphics menu keeps editing the live `GraphicsSettings` after "Re-detect"
  (`update_from_dict` instead of a new object); ESC no longer discards the rest of
  the frame's events; `Player.clothing_noise_factor` is clamped (`clothing_noise_floor`);
  mipmaps via `glGenerateMipmap`; dead tunables, aliases and helpers removed.

### Seventh iteration: shops, sleep, punishments

* **Waits are perceptive and interruptible** (`World.wait_hours(..., player=)`,
  `interrupt_wait`): perception runs every step of a `[T]` wait and the wait ends the
  moment the player is caught or shown the door ("Your wait is interrupted."). You can
  no longer wait out a shopkeeper.
* **Closing time clears the premises** (`World._clear_the_premises`, from
  `on_task_left`): a player on the keeper's indoor premises when the keeper leaves is
  shown out; crouching, a `sneak_chance` roll decides whether the keeper notices
  (success: locked in, trespassing, Sneak XP). Paying inn guests are left alone.
* **Skulking customers** (`World.start_eviction(npc, "skulking", ...)`, generalising
  the curfew wait): a shopkeeper open for business whose sneak check catches a
  crouching customer on their indoor premises asks them to stand up, again after
  `curfew_ask_interval`, and throws them out on the third; standing up or leaving ends
  it; the keeper's warning grace keeps the generic alarm from firing meanwhile. Open-air
  premises (the smith's yard) keep the ordinary scold.
* **Trade availability** (`NPCDef.trade_availability`): `at_work` (default), `awake`
  (the innkeeper lets rooms while cleaning or eating), `always` (the fence:
  `wake_for_business` rouses him, `awake_for_business` keeps his routine and the
  curfew off him until `close_transfer` → `back_to_business_as_usual`).
* **Sleep until night / morning** (`Game.sleep`, `sleep_target`): by day (06-18) a
  bed of the player's sleeps until 20:00, otherwise until 06:00; the bed prompt says
  which.
* **Punishments** reworked (see "Justice"): `Penalty`, `penalty_for`, `judge`,
  `Verdict`; `gold_per_jail_day` / `min_jail_days` / `max_jail_days` are gone; jail no
  longer confiscates the purse.
* **Lighting**: indoor light bounds stop `WALL_FACE_INSET` (0.05) inside the outer
  wall faces instead of the whole wall thickness, so door leaves (which sit in the wall
  plane) are lit from inside.
* **Models**: the well's roof is a slab (`draw_gable_roof(thickness=...)`) on a
  ceiling board, its bucket a round pail; the ore deposit is a cluster of lumpy
  boulders (`draw_rock`, `StationDef.kind="ore"`); shop signs lost their brackets.


### Tenth iteration - review fixes: saves, justice bookkeeping, the economy

* **Nothing leaks across a load.** `SaveManager.apply` calls `World.reset_transient` first
  (conversations, a relocation still fading, an interrupted wait, the jail's transfer lock)
  and every `NPC.load_dict` calls `NPC.reset_transient` (a chat in progress, a wait on the
  player, warnings given, investigations, the fence's "woken for business"). `Game.load_game`
  drops its own fade / pending relocations / locked camera and runs `update(0)` at once, so
  the loaded state is judged immediately (a prisoner saved outside the cell is a jailbreak on
  loading, not an hour later). Loading during an eviction fade used to leave
  `World.relocating` stuck on - which silently switched off charges, curfews and the jail.
* **Saving is refused mid-transition.** `Game.save_refusal` / `can_save`: inside a screen, while
  the controls are locked or a fade runs, and while `World.save_blocked` - the player is being
  moved, or `JailState.transfer_pending` (set on the first line of `justice.imprison`, cleared
  by the relocation's `on_applied` hook once they stand in the cell). The pause menu's *Save
  Game* entry is greyed out with the reason, F5 logs it, and `PygameApp.save_to` goes through
  `Game.save_game` like F5 does. A save taken between sentence and cell used to load as a
  jailbreak.
* **Saves are atomic and corruption is handled** (`paths.write_json`: temp file + `os.replace`,
  used for saves and `graphics.json`). `SaveManager.read` / `load` report a truncated, non-JSON
  or non-save file ("corrupt ... nothing was loaded") instead of raising; `describe` labels it
  `(unreadable)`; a write error is reported, not raised; the load menu shows the reason.
* **One deed, one crime.** `NPC.witness_crime` (via `sees_crime`) lets an NPC already raising
  the alarm see a crime without shouting or counting it again; `NPC.remember_crime` returns the
  existing memory when the same crime (name, day, indoors) was declared within
  `CRIME_REPEAT_WINDOW` (5 s, `CrimeMemory.declared_at`) - upgrading it to *witnessed* if the
  guard now saw it - and `World.declare_crime` sends nobody running twice for it; a second theft
  later in the day is a new crime. The victim of a pickpocket is excluded from the bystanders'
  roll (`report_crime_to_witnesses(exclude=)`), and a fumble (`Game._pickpocket_caught`)
  alarms the victim once and declares the crime with them as its witness instead of putting it
  to the town a third time. A fumbled lift used to produce three witnessings and three lines.
* **Confrontations recover only the reporter's stolen goods**, and only the reported quantity
  (`World._confront`); bought goods of the same kind stay the player's (they used to be taken
  and, being the player's, routed into the confiscation lockbox). The jail's own
  `confiscate()` - everything but one lockpick - is unchanged.
* **The economy** (`EconomySettings`, `TradePolicy`): worth / asking price / offer, see
  "Transfer sessions". Rows show the worth beside the price; `sell_price <= value <= buy_price`
  holds at every Barter level (`test_buying_and_selling_back_always_loses_money`), closing the
  buy-and-sell-back XP farm and gold pump. Services keep their posted price. The player starts
  with `STARTING_GOLD` (60 g) so a pickaxe and a mallet are affordable at asking price; the two
  "Fresh Bread" items (`bread` 3 g on every table, `bread_1` 15 g in Barnaby's pocket) are one
  item now.
* **The inn** shows a guest whose rent has run out out of the *room* into the hall
  (`World.show_out_of_room` / `exit_of_room`), not into the street; sleeping through checkout
  ends in the hall.
* **HUD**: modal screens (transfer, lock mini-game, menus) are opaque, the controls and
  inventory panels are not drawn under a transfer / lockpick screen, and those screens stop
  short of the log panel (`HudRenderer._modal_bottom`) - nothing bleeds through anything. Task
  narration (`ScheduleTask.action_text`) is logged plainly, never as a quoted line. The status
  bar shows the clock to the minute (`clock_text`, display only).
* **Rendering**: the fragment shader soft-clips above a knee (`KNEE` 0.75, asymptotic to
  white) so lanterns, candles and a held torch no longer bleach interiors (the fixed-function
  fallback has no such stage). `LightingRig.gather` ranks lights by whether they can reach
  what the camera sees - unbounded lights and those of the player's own room first, candles
  behind other buildings' walls last - and drops lights that cannot reach anything within the
  draw distance; in the street at night the watch's torches and the forge get slots before the
  town's indoor candles.
* Docs brought back in step (test counts, scenario count, texture keys, timings).

### Eleventh iteration - equipment, the [Tab] view, loot-all, review fixes

* **Equipment** (`entities/equipment.py`, `rendering/gear.py`): every worn thing is an `Item`
  with a ``gear`` kind (-> `EquipSlot`: head, neck, body, feet, weapon), a ``gear_color`` and
  an ``equipped`` flag. `GEAR_CATALOGUE` maps the level's item ids to their gear (`ItemDef.build`
  fills it in; saves from before equipment get it by id on loading and put on what they carried).
  `draw_figure(figure, equipment)` draws a `Figure` (hose, hair, a plain shirt when nothing covers
  it) and then each slot in body order - feet, body, neck, head, weapon - through `GEAR_DRAWERS`:
  tunic, breastplate, chain shirt, plate cuirass (pauldrons and faulds); sash, cloth and leather
  aprons, necklace (chain and jewel), cloak (over the shoulders and down the back, a clasp);
  kettle helm, cap, leather cap, scarf, coif, hood (hides the hair, a cowl); boots, shoes,
  slippers; dagger and jeweled dagger at the right hip, sword at the left, sling. The
  townsfolk's `Outfit` is turned into worn items by `equipment_from_outfit` (`NPC.equipment`,
  kept apart from the pockets: a helmet on a head is not pickpocketed) and drawn by the same
  code - pixel-identical to the hand-placed boxes they replaced (checked by rendering every NPC
  front, side and back before and after). `Player.equip/unequip/toggle_equipped` keep one item
  per slot; `take_item` unworns whatever leaves an inventory (sold, planted, confiscated).
  `Player.clothing_noise_factor` counts worn gear (and slotless things such as gloves).
* **The [Tab] view** (`GameMode.EQUIPMENT`, `Game.tab/open_equipment/equipment_toggle/
  rotate_preview`, `HudRenderer._draw_equipment_screen`, `SceneRenderer.draw_player_preview`,
  `LightingRig.begin_studio`): the pack with a cursor and slot tags on the left, the figure on a
  stand in the middle (a scissored 3D viewport with its own projection and neutral key/fill
  lighting, turning by itself until [Left]/[Right] take over), what is worn per slot on the
  right, the footstep factor below. [Enter] puts on / takes off; [Tab]/[Esc] close; nothing else
  of the HUD is drawn meanwhile. Mouse released, movement and interaction blocked; being caught
  or loading a save closes it. Layout from `HudLayout.equipment_panel` (anchored for other
  window sizes). Save format v8.
* **[X] takes everything** in a container screen (`TransferSession.take_all`, `Game.loot_all`,
  `KeyBindings.loot_all`): each unit goes through `ContainerPolicy.take` exactly as by hand
  (rolled, rewarded, reported - a witness shouts once); never in pockets or trade.
* **Fixes**: a paying inn guest is trusted in the hall (`World.area_trusts_player` includes
  `InnDef.area` while rented) - the night watch no longer calls a late guest a trespasser;
  `LightingRig._doorway_patch` gives every room-bounded light (and a torch carried indoors) a
  second rectangle over the nearest open leaf (`PointLight.patch`, `uLightPatch` in the shader),
  so a door you walk through keeps catching the light; shop signs show their device on the two
  faces only (`primitives.draw_board`), plain wood on the edges; `Item.split` uses
  `dataclasses.replace`.

### Twelfth iteration - clothes in the shadows, the inn at every hour, worn and theft tags

* **The clothing model** (`ClothingSettings`, `entities/equipment.py`: `material_of`, `luminance`,
  `clothing_effect`, `describe_clothing`; `Player.visibility_factor / clothing_noise_factor /
  outfit_effects`): every worn item contributes to two multipliers by its *material* (metal
  +0.35 visibility / +0.25 noise, leather -0.04 / -0.02, cloth -0.08 / -0.08; `GEAR_MATERIALS`
  per gear kind, `Item.material` overrides - black *cloth* shoes use the `shoes` model), its
  *shade* (luminance of `gear_color` against `shade_pivot` 0.45 x `shade_slope` 0.5; untinted
  pieces take their material's shade), the kind's coverage (a necklace is a sliver of metal)
  and the slot's weight (visibility: body 1.0, neck 0.9, head 0.5, feet 0.3, weapon 0.25;
  noise: feet 2.0, body 1.0, neck 0.6, weapon 0.6, head 0.3), clamped to [0.25, 2.5] and
  [0.25, 2.2]. `Player.get_visibility(ambient, is_moving)` multiplies in the factor and, crouched
  *and* still, `still_crouch_multiplier` (0.5); `NPC.evaluate_perception` passes `is_moving` and
  puts an NPC's own torchlight (`torch_lit_visibility`) through the same clothes and stance
  instead of a flat floor. Explicit `noise_modifier`s on gear are gone (material does it);
  slotless things (gloves) keep theirs. A step heard at `PerceptionSettings.racket_noise` (3.6)
  or louder draws a "racket" remark (`scold("racket", ...)`, guard and citizen phrasings, no
  grace window, never a crime) from anyone awake in earshot; sleepers wake through walls as
  before (sprinting in plate is ~4.3, x0.25 through a wall = 1.07 > the 0.8 waking threshold).
  The [Tab] view shows *Seen / Heard* multipliers and describes the selected item.
* **The economy of stealth** (`GEAR_CATALOGUE`, `town.py`): bright and shiny is common and
  cheap (white linen shirt 4 g, white cloak 8 g, red cap 3 g on the tailor's rack; iron helm
  20, chain shirt 40, plate cuirass 90 on the smith's), brown leather (jerkin 45, boots 30,
  cap 25) the middle, dark cloth rare and dear through the tailor's tiers (brown wool cloak
  120 @2, velvet cloak 260 @3, black hood 180 @4, black tunic 300 @5, shadow boots 260 @6,
  black cloak 340 @7, black cloth shoes 520 @8); Jae sells a black hood for 240.
* **The inn at every hour**: "the inn" is no longer restricted (`AreaDef` without public
  hours), so nobody is ever a trespasser in the hall; `_clear_the_premises` skips public
  premises (the keeper going to bed clears nobody out). A visitor in the hall without a room is
  welcomed (`World._tick_inn_welcome`, `InnState.welcome*`, `AIState.WELCOMING`, dialogue
  `inn_welcome` / `inn_welcome_woken`): from behind the bar if the keeper is there (she stays
  put), otherwise she comes over - woken if asleep (`awake_for_business`, so
  `back_to_business_as_usual` returns her to bed afterwards) - says her piece, is open for
  business where she stands (`trade_availability="awake"` allows `WELCOMING`), follows a
  visitor who wanders off, and after `WELCOME_PATIENCE` (40 s), a rental, the visitor leaving or
  an alarm goes back to her routine. Once per visit; never saved.
* **Worn and theft tags** (`TransferPolicy.row_tag`, `HudRenderer._draw_item_text(tag=)`): the
  player's column tags worn items `worn`; a container row that would be a theft is tagged
  **THEFT** in red (`ContainerPolicy.is_theft_of`), `free` for food on the house. Row text is
  cut with an ellipsis rather than run into a tag. Giving away something worn (selling,
  planting, putting away: `TransferSession.giving_away_worn`, `give_verb`) asks first through
  `Game.ask`; [Enter] proceeds, [Esc] keeps it on.

### Thirteenth iteration - the tutorial, wary fences, reports that are not interrupted

* **The tutorial** (`tutorial.py`, `Game(tutorial=True)`, `PygameApp.new_game` / `new_game_free`):
  a `Stage` machine (`INTRO` -> `TORCH` -> `TO_THE_SHOP` -> `IN_THE_SHOP` -> `CAUGHT` -> `JAIL`
  -> `LOCKBOX` -> `DRESS` -> `WAIT_FOR_NIGHT` -> `STEAL_AGAIN` -> `SELL` -> `TOOLS` -> `MINE` ->
  `FORGE` -> `INN` -> `DONE`) ticked from `Game.update`. `begin()` sets the scene: 22:00, the
  hideout, an empty purse, three picks, boots in the pack (the lesson), no goblet,
  `World.suppress_curfews` (Jae has taken you in) and `World.freeze_guards()`. The torch law is
  the first lesson because it is night (`TORCH`: a lit torch, or by day `TORCH_LESSON_SECONDS`
  of explanation, moves on; picking the store's lock regardless skips ahead). The catch is
  scripted: `_the_watch_arrives` closes the shelf screen and *spawns* a watchman (`WATCHMAN`,
  "Guard Tobin", an `NPC` that is not part of the town and is exempt from the freeze) at
  `_spot_for_the_watch` - `WATCH_DISTANCE` from the thief on the open floor of the same room,
  towards the store's door for choice, with a clear line of sight - alarmed with
  `WATCHMAN_LINE`, the crime declared, every scolding pre-spent; the thief's view swings round
  to him (`Player.hold_attention`, which now also pitches to the face: upwards from a crouch)
  and he stands there for `WATCH_DELAY` seconds before `_keep_the_watch_on_them` has him lay
  hands on them (`justice.charge_player` directly, whatever he could or could not see); it also
  keeps him in reach and summons him afresh when a save taken mid-arrest is loaded into a new
  run. `_dismiss_the_watch` removes him from `World.npcs` - conversation ended, crimes forgotten
  - as soon as the player is in the cell; `_trim_sentence` makes it a single night (`PITY_LINE`,
  logged as Captain Vance's). Jae's lines are a timed script (`JAE_INTRO`, `JAE_PAID`); every
  hint (`HINTS`) is plain prose, and `_enter` logs its opening sentences (`LOG_LEAD_CHARS`).
  `refuses(kind, obj)` is the leash - consulted by `Game.interact` (doors, containers,
  stations), `_interact_npc` (trade, pockets), `loot_all` and, through the policies' ``allowed``
  hook, every unit moved in a container or a deal (`TransferOutcome.REFUSED`): until the
  candlestick is sold only the store's door, the hideout's and the jail's, the west shelf (the
  candlestick only), the lockbox and Jae's deal (the candlestick only) are open. A missing pick
  is replaced in every stage but `CAUGHT`/`JAIL` (`_keep_a_pick`: the tutorial never stalls for
  want of one); being jailed early (seen at the lock) jumps to `JAIL`; a jailbreak counts as
  out; `WAIT_FOR_NIGHT` completes once it is night (`DayNightCycle.is_night`) and a [T] has been
  pressed (`note_wait`; or a few seconds have passed); losing the candlestick again means
  stealing it again; `INN` completes inside the inn building or with a room rented, and only
  then does `_finish` unfreeze the guards, lift the curfew suppression and put `DONE_MESSAGE`
  (the town is open) in the panel, where `update` leaves it for `DONE_SECONDS`. The HUD draws
  `Game.tutorial_hint` in a gold-framed panel (`_tutorial_panel` / `_draw_tutorial_hint`: under
  the controls panel; beside a modal screen - a deal, a shelf, the lock - in the strip to its
  right, level with its top, or in the undrawn inventory panel's corner when the window is too
  small for that; at the foot of the list column in the [Tab] view; `tests/unit/test_hud_layout.py`
  pins the arithmetic with a stub text measurer), and hides the interaction prompt while the
  controls are locked (no "[E] Pickpocket" over the guard who has you by the collar). Saved
  under ``"tutorial"`` (`SaveManager.save(extra=)`, `last_loaded`; `from_dict` re-applies the
  freeze - or lifts it - and drops a finished tutorial's closing words; the spawned watchman is
  serialised with the other NPCs but ignored on load, hence the re-summons). `Scenario.tutorial`
  starts scripted runs in it (`tutorial_start`: hideout to the lockbox, the boots and fourteen
  [T]s to nightfall, with the watchman in view for three seconds).
* **Frozen guards** (`World.guards_frozen`, `unfrozen_guards`, `is_frozen(npc)`,
  `freeze_guards(except_for=)`, `unfreeze_guards()`): a held guard is a statue with a torch -
  `update_npcs` skips their schedule catch-up, movement and AI, `update_time` their hourly
  task change, `update_perception` them entirely, and they witness nothing
  (`report_crime_to_witnesses`), remember and answer no cry (`declare_crime`), take no report
  (`_dispatch_report`), run no curfew and start no conversation. Unfreezing puts each guard on
  the current hour's task without announcing it. Not saved: the tutorial owns it.
* **The waypoint arrow** (`Tutorial.waypoint` -> `Game.tutorial_waypoint`, drawn by
  `rendering/waypoint.py` from `SceneRenderer.render`): `_destination` names the place each
  stage wants the player (the store's door and then its west shelf, the lockbox, Jae, the east
  shelf, the ore, the anvil, the inn's door; None for `CAUGHT`, `JAIL`, `DRESS`, `DONE`) and
  `_route` goes through doors - from inside a building towards somewhere else the waypoint is
  that building's street door (`World.door_of`), from the street towards a place indoors the
  door of the building it is in. `draw_waypoint_arrow` draws a solid yellow arrow (shaft and
  head with thickness, emissive so it glows at night) in view space below the crosshair, turned
  by `relative_yaw(player, target)`, tilted towards the camera, depth test off so no wall hides
  it; not drawn in the [Tab] view.
* **Fences are wary** (`NPC.evaluate_perception` no longer skips fences; `World.update_perception`
  likewise; `_shopkeeper_watches` / `player_on_premises_of` cover every `can_trade` NPC; a new
  `_keeper_objects_to_skulking` starts the stand-up-or-out eviction the moment a watching keeper's
  suspicion passes the threshold while the player crouches on the premises, not only on a failed
  sneak roll). A fence never raises the alarm from suspicion (only `witness_crime` - being robbed
  - does). Standard for every `is_fence` NPC.
* **Reports are not interrupted**: `report_theft` dispatches only to guards not `on_a_report`
  (`report_to` / `report_in_progress` / `report_interrupted`); with nobody free the report waits
  in `World.unanswered_reports` for the next free guard (`REPORT_WAIT` 10 min). A guard who
  witnesses a crime mid-report keeps it (`raise_alarm` -> `report_interrupted`), deals with the
  criminal and, when he would otherwise resume his routine, goes back (`World.resume_report`) and
  takes the report again with an apology (`REPORT_LINES["apology"]`, `build_report_exchange(
  apologise=True)`); a broken exchange no longer cancels a chase. A guard arriving while the
  shopkeeper is still shouting waits for them to calm down.
* **[Tab] pauses the world**: `Game.update` advances only the stand while the view is open.

### Fourteenth iteration - code review fixes

Every fix below has a test that fails without it (`tests/unit/...` unless noted); the state
goldens are unchanged but for `lockpick_escape_and_keys`, which now logs the refused quick load.

* **Closing time means leaving the premises** (`World.on_task_left`): a keeper is closing up only
  when the next task takes them out of the building the counter stands in. The old rule
  excused Barnaby's back room by area *name*, so Mirelle - lunching at the side table of her
  own shop at 13:00, inside its public hours - took stock and showed customers into the street
  (`test_shops.py::test_lunch_on_the_premises_is_not_closing_time`).
* **A relocation is never left in flight**: the world asking for a move during the fade-out of
  *lying down* (checkout striking as the guest goes to bed) queued it behind a fade that never
  applied it - `World.relocating` stayed on for good, refusing every save and switching the
  jail, curfews and charges off, with the player locked in the room. The sleep fade applies
  pending moves at black (`Game._sleep_fade`), and a move still pending when any fade ends gets
  a fade of its own (`test_inn.py`, `test_game.py::TestFades`).
* **Loading drops the tutorial's watchman** (`World.reset_transient`): NPCs that are not part of
  the level leave with the session, so an F9 to before the theft no longer leaves Guard Tobin in
  the store with a Shoplifting memory that every later fine (`justice.judge` totals all guards)
  would have charged.
* **Saves**: `SaveManager.describe` - which feeds the start menu's Continue - never raises (a
  `meta` that is not an object, a non-numeric hour: "(unreadable)"); `list_saves` skips files
  that vanish mid-listing; `load` catches `AttributeError`/`IndexError` too (`"jail": null`) and
  a save that fails part-way through `apply` is rolled back to the game as it was, instead of
  leaving a half-loaded world behind (`test_save.py`, `test_save_slots.py`).
* **`graphics.json`**: JSON that is not an object is ignored like a corrupt file (it crashed the
  start-up), an unknown preset or a value of the wrong type falls back to the default (an
  unknown preset crashed the options menu), and `GraphicsSettings.save` reports an unwritable
  file (`False`) instead of raising (`test_perf.py`). A corrupt texture PNG becomes the magenta
  placeholder like a missing one (`test_textures.py`).
* **Rendering**: the view matrix is loaded *before* the frame is lit - the fixed-function
  pipeline (the *low* preset, software renderers) places lights through the current modelview,
  so the sun, moon and torches had been set with the previous frame's camera, or with none
  after a frame ending with the held torch or the tutorial's arrow (a torch placed 40+ units
  off; `tests/display/test_app_smoke.py::test_fixed_function_lights_sit_where_the_camera_sees_them`).
  `draw_roof` gave long-ridged roofs (most of the town) normals with two components swapped,
  not at right angles to the slope: north slopes went near-black at noon
  (`test_buildings.py`). `SOFTWARE_RENDERERS` includes SwiftShader, as documented above.
* **Smaller fixes**: `justice.confiscate` / `return_stolen_goods` take worn things off
  (`equipped` no longer lingers on a rack or in the lockbox); `NPC.load_dict` forgets the old
  session's flags *before* judging the loaded state (a roused fence saved asleep loaded as
  `IDLE`); a conversation never goes over the same topic twice (two guards, whose pool lists the
  watch four times, repeated an exchange word for word in one chat in eleven; the weighting
  stays); `AreaDef.is_open_to_public` wraps midnight like `curfew_active`; an ownerless
  restricted or curfewed area no longer counts every faction-less NPC as its owner; [F9] inside
  a screen says why it is refused, as [F5] does.
* **Navigation** is ~2.4x faster with bit-identical paths: stale heap entries are skipped, the
  occupancy grid is read as nested lists during a query, and `_line_clear` drops clamps that
  could never apply.
* **Tests and tooling**: `test_progress_fraction` asserted nothing; `harness.dry_run`,
  `render_gallery` and `tools/benchmark.py` left a temporary directory behind per run (hundreds
  per suite run) and now clean up; the harness runners restore the working directory;
  `test_every_scenario_plays_to_the_end` reuses the golden runs instead of replaying every
  scenario a second time (and lost a conditional whose branches were identical);
  `gen_textures.py --only` rejects unknown keys instead of writing magenta files.

### Fifteenth iteration - the realm beyond the gate

**Fixes.**

* **Distant streaking** was z-fighting: SDL's default 16-bit depth buffer plus coplanar faces
  (wall corners authored overlapping, fascia flush with the gables, clothing layers sharing
  planes, the tower's column flush with its platform, a lintel over a gap a wall already filled).
  Measured by rendering one frame twice - `GL_LESS` and `GL_LEQUAL` - and counting the pixels
  that differ (depth ties); 16-bit emulated by scaling the near plane. Before: 66-151 tie pixels
  on one NPC at 10-70 units, up to 16k in the frame; after: 0 on the NPC, <= 10 in the frame.
  The window asks for a 24-bit depth buffer (16 as a fallback; shown in the F3 overlay), walls
  and lintels are made disjoint (`world/geometry.py`), gear layers are `LAYER` apart, and
  `tests/unit/test_streaking.py` keeps every one of those planes apart - the static world and
  every pair of gear kinds worn together.
* **Beds**: one `builders.bed` for every bed - `BED_LENGTH` along its head-foot line whichever
  way it faces (the guard house's "double" bed was a single turned on its side), a headboard,
  a blanket whose *top* carries the picture (the sides are plain cloth) and a pillow
  (`rendering/structures.draw_bed`); sleepers lie with their head at the headboard.
* **The tower's top lantern** hangs on the parapet, its cap under the coping, not in mid air.
* Smaller: lintels over a door between two buildings stop at the lower eaves (the tower door's
  ran up the tower's face above the guard house roof); a guard on his way to *take* a theft
  report walks on past an innocent bystander; "stable" is not a table; the torch law is the
  street's (a guard no longer demands a torch indoors); candles only on real tables.

**Changes.**

* **Raz is Jae** everywhere.
* **One rule for every skill** (`systems/progression.py`, `systems/stealth.py`): levels 1..100,
  effective level = level + worn bonuses, chance = a logistic curve of (effective - challenge).
  Locks come in six tiers (Flimsy .. Legendary, `config.LOCK_TIER_NAMES`): a village latch, a
  cottage door, a shop, the watch's doors, the manors, the Deep's vault. The lock mini-game's
  window, the hand's wobble and the final "pins set" roll all follow the same margin.
* **Rarity and legendary gear**: common .. legendary; rare things are harder to lift unseen,
  epic and legendary ones are secured (a challenge of their own, and out of reach altogether
  below a gate). The legendary three - the Amulet of the Open Door, the Nightweave Cloak, the
  Fingers of the Magpie - glow, and add +35..+40 to a skill. Dark clothing is no longer stock:
  it is loot. The HUD colours names by rarity, shows bonuses and effective skill levels in the
  [Tab] view and the lock's tier and odds in the mini-game.

**The new town and the country** (`world/levels/town.py`, `world/levels/wilds.py`,
`world/builders.py`).

* **Terrain** (`world/terrain.py`): hills, seeded noise, streams that carve downhill, lakes in
  basins - and **pads**: every building's floor (and every `FlatDef`) is level, exactly, over its
  footprint and a one-cell skirt, easing back to the slope; pads blend by inverse distance (order
  no longer matters) and `pad_conflicts` lists overlapping pads at different levels (kept empty
  by `tests/unit/test_terrain.py`). A stream's ribbon of water is cut just past the rim of the
  lake it runs into (`rendering/terrain._clipped_to_lakes`): two blended sheets over the same
  ground read as a darker rectangle in the lake. `Placer` lifts the local level data onto the ground: walls in
  the open become stepped segments with a footing, props sit on their lowest ground, bridges on
  their highest bank. The terrain is computed once per level and shared.
* **Emberhollow** climbs from the gate to the watchtower on its knoll; the well is the middle of
  the town. Market Street, Upper Street (the manors and the guard house), North and East Lanes,
  Gate Road and South Lane; the millstream in under the north wall, over a bridge, out under the
  south wall through grates and on to the lake by the Thornwood. Barrels, firewood, orchards,
  hedges, market stalls, cracked paving and walls, water stains, a rotting granary, a chapel, a
  stable, torches on the houses and along the curtain wall (they burn from dusk to dawn).
* **The town gate**: two leaves (`DoorDef.hinge_right`, `gate`), never shut by passing NPCs, kept
  round the clock by two gate guards who change sides every few hours; barred and refused
  (a guard turns you away) until the tutorial is over, open after it (or at once without it).
* **The barracks** sleeps six (Bram and Wenna joined the watch); the patrol walks all the streets.
* **The rich**: Lord Aurelian Vane and Lady Seraphine Ashgrove in two-storey manors
  (`builders.straight_stair`, an upper floor with a railed stairwell, Master locks, coffers and
  strongboxes of coin, rare and legendary pieces). They dress in doublet / gown, mantle / choker,
  feathered hat / circlet (`Outfit.body`, `neck`) and talk down to everybody, the watch included.
* **Jae's job** (`unprepared/quests.py`): after the tutorial, talking to Jae offers "Sticking It
  to the Jerks" - the Ravenfeather Cloak from Vane's wardrobe upstairs. The guide panel and the
  arrow show the way; handing it in pays 150 gold and the cloak is the player's to keep
  (laundered). Saved with the game.
* **Beyond the gate**: Thornbury (three cottages with flimsy latches, a barn, fields), the
  Thornwood (a seeded forest, the hunter's hut, an ore seam), the smugglers' cave, the Howling
  Deep (a sentry, Grimmaw the Bandit King before a legendary vault door) and Greyhold, the old
  keep (two knights, a treasury). Caves are buildings with flat rock roofs and no lanterns.
  Cliffs fence the map in. **Wardens** (`NPCDef.warden`): outlaws and knights answer to no
  watch - one who catches you in their domain warns you, then drags you out, takes back what
  you stole from them and a quarter of your purse.

**Engine.** Navigation over a 660x580 grid: a room graph (regions cut by doorways) answers
"unreachable" at once instead of flooding the map, the A* runs on flat indices, routes are
cached. A spatial index (`world/spatial.py`) serves collisions, lines of sight, the floor under
a walker, props under a sleeper and the indoor lights; calm NPCs out of sight and earshot skip
perception; far-off NPCs' comings and goings stay out of the log. Terrain patches are drawn
before the paving, and paving joins without seams. Saves are version 9 (older ones put the
player back at the spawn: the old coordinates point at nothing now).

Tests: `test_terrain.py`, `test_streaking.py`, `test_beyond_the_gate.py` are new; the rest of
the suite works in map coordinates through `tests/helpers.py` (a spot on the floor, a door's
outside) instead of y = 0. Scenario scripts take `place` frames for the long walks (the game
only) and the goldens were re-recorded and reviewed.

### Sixteenth iteration - the realm filled in

**The land** (`world/terrain.py`, `rendering/terrain.py`).

* `TerrainDef` gained **trails** (`TrailDef`: a dirt way graded along its length - `grade_profile`
  keeps it under `max_grade`, its ends pinned to the ground or the pad they meet), the **sea**
  (`SeaDef`: a coastline, a beach and a shelf; `water_level_at` and `sea_distance` know it),
  **holes** (cells not drawn: the mouth of a cave) and the thresholds for the ground's looks
  (`rock_slope`, `snow_line`). `Terrain.landform` computes the natural ground once per level;
  `height` interpolates over the mesh's own triangles (so what is drawn and what is walked on
  agree exactly) and `heights_at` does it for many points at once.
* **Pads may be underground** (`Pad.underground`, from `BuildingDef.underground`): a cave's floor
  is levelled under the hillside without touching the surface; `surface`/`underground_at`/
  `walk_grid` tell the two apart and `pad_conflicts` ignores caves (their floors meet at steps).
* The ground is drawn as triangles from numpy arrays (`primitives.draw_arrays`), with
  per-vertex-alpha **overlays** blended over the grass (`material(vertex_alpha=True)`): bare rock
  on the steeps - projected side-on on a crag, so it shows strata instead of streaks - sand on
  the beach, snow above the snow line. Paving and trails are draped over the mesh triangle by
  triangle (`draped_pieces`, `drape_polygon`) and batched; a road's edge fades only where no
  other road continues it (`side_intervals`). Water is two drifting, rippling layers
  (`WATER_LAYERS`; `stream` and `lake` textures), the stream clipped at the lake, the sea out to
  `SEA_REACH`.
* **Walking**: `World.can_step` refuses ground steeper than `WALK_GRADE` (rise over run 1.0),
  the open sea, the map's edge and a step down through a hillside into the cave beneath it;
  `Game.update` slides along a refused step. The navigation grid blocks the same cells
  (`World._unwalkable`; steps and bridges are exempt). **The map has no walls at its edges**:
  the Greyspine mountains to the north, the thicket (`_edge_of_the_wood`) to the west and south,
  the sea to the east.

**Caves** (`world/caves.py`). `dig(prefix, chambers, passages, mouth)` turns rectangles of floor
(`Chamber`: an underground building at its own elevation) into rock walls set back by random
amounts (`WALL_JITTER`), lintels over every passage (as tall as the higher floor needs; a lower
chamber is raised to fit), a mouth whose rock reaches up to the hillside (`WallDef.reach_surface`
-> `Placer._up_to_the_surface`) with crags and a brow heaped round it, a hole in the drawn ground
over the way in, and dressing (boulders, stalagmites, stalactites) kept off every lane through a
chamber. `clear_the_floor` (run by `Part.clear_cave_floors` after the furnishing) takes up any
dressing that landed on a chest, a bedroll, a seam or a spot somebody stands on. `mushrooms`
adds glowing caps (light kind `glow`). Underground, the shader's shade zones are strict and the
fog dark; cave roofs block sight. `World.building_at` and `area_def_at` are height-aware, so a
cave under a wood is the cave below and the wood above.

**Reading, names and hiding places.** `ReadableDef`/`Readable` (notice boards, signposts,
plaques, journals): `[E] Read the notice` opens the text (`Game.reading`, HUD `_draw_reading`)
until `[E]` again or the player walks off (`READING_REACH`); the first reading goes into the log
(`Game.read_already`, saved). Every door has a `label`, shown over the prompt
(`InteractionTarget.title`, `Game.interaction_title`, `HudColors.prompt_title`).
`ContainerDef.hidden` makes a hiding place the crosshair only finds for a crouching player within
`HIDDEN_REACH`.

**The town filled in** (`levels/manors.py`, `levels/quarters.py`, `levels/stories.py`). Manors
rebuilt room by room (kitchen, store, dining room, servant's room; study, dressing room through
it, bedroom, second bedroom) with their servants; `StairDef.rooms` routes a walk upstairs through
the doorways (`World._stair_path`). Quarters: the chapel (nave, steeple with a spire roof and a
bell, vestry, graveyard), cottages, bakery, cooper, stable, four houses on Upper Street, Tanners'
Row, the gate lodge. Every townsperson has `NPCDef.lines`, told by `NPC.greeting` every other
time at a civil hour (`personal_index`, `times_greeted`: saved). Dialogue has topics for every
trade; the one a topic is *about* replies, the squire and the bandit king speak first, outlaw
exchanges run to several lines. Tower crenellations (`tower.crenellations`; `in_the_open` towers
carry theirs on `WallDef.rest_on`).

**Fixes found on the way.** `Prop.is_bed` is by kind (a bedroom's rug was a bed you could sleep
in); the gate lodge no longer fights the stable's pad; candles are only set on tables under a
roof; Tobin carries the keep key he needs at dawn; every table holds three days of meals for
whoever eats at it; every schedule covers the whole day; NPC spots and patrol points are off
the furniture; the dressing room (whose door opened onto the stairwell) is reached through the
study; "The captain's got us doubling the rounds" is no longer said by the porter to the captain.
Every door in a building's outer wall gets a threshold (`World._thresholds`: the floor inside stops
at the wall's inner face, and the ground showed between it and the leaf), as do the chapel's open
ways; the vestry's eaves no longer reach into the nave; lanterns hang on walls (a porch lantern
and the auto-hung ones doubled up and floated in the chapel's open side - its buildings now carry
their own lights, `lit=False`); Greyhold's merlons sit on their towers' tops on sloping ground
(`WallDef.rest_on`) instead of sinking into them; the cliff texture is side-projected strata.
**News is local**: with fifty-three people about, the log carries the comings and goings of
those within `NEWS_RANGE` (40) of the player or under the same roof - the whole town's hourly news
pushed the warnings that matter out of a six-line log.

**Performance.** Baking the scene takes about as long as before the map grew (~6 s); the
simulation runs at ~5 ms a frame with every NPC; the level imports in ~0.7 s.

**Tests**: `tests/unit/test_realm.py` is new (caves, land, names and notices, buildings,
townsfolk); the rest of the suite follows the new map (positions through `tests/helpers.py` and
level constants), and the scenario goldens were re-recorded and reviewed.

### Seventeenth iteration - the screen, the start, the week, the doors

**Graphics** (`graphics.py`, `app.py`, `rendering/renderer.py`, `shaders.py`, `textures.py`,
`atmosphere.py`).

* New settings, all in the presets and the graphics menu: `fullscreen` (the desktop's resolution;
  `app.desktop_size`), `vsync`, `antialiasing` (multisampling: `AA_STEPS` 0/2/4/8),
  `anisotropy` (`ANISOTROPY_STEPS`, clamped to the driver's `max_anisotropy`), `draw_distance`
  (`DRAW_DISTANCE_RANGE` 60-1000; `GraphicsSettings.far_plane` replaces the fixed
  `RenderSettings.far_plane` - the projection is rebuilt when it or the FOV changes) and `haze`
  (`compute_atmosphere(hour, settings, haze)` scales the day's fog density by it, so a long draw
  distance is not lost in fog; `LightingRig.haze` carries it to the scene).
* The presets were shifted: **high** is for a proper GPU (draw distance 700 - the whole realm -
  thin haze, 16 shader lights, 8x MSAA, 16x anisotropy, 1500 stars, 40 ground cells); **medium**
  is the old high and the default for an unmeasured machine; **low** unchanged in spirit.
* `open_gl_window(w, h, fullscreen, samples, vsync)` asks for the multisampled buffer and falls
  back step by step (without vsync, without multisampling, then a 16-bit depth buffer) so an
  unsupported request never stops the game. `PygameApp.apply_display()` re-opens the window when a setting it
  depends on (`_wanted_display`: fullscreen, samples, vsync) has changed, rebuilds the renderer
  and menu renderer (their GL objects died with the context) and re-lays the HUD for the new
  size (`settings.window` replaced). The display is shut and opened afresh for it: SDL keeps an
  OpenGL window's old size and pixel format when the mode is merely set again (found by the menu
  probe - full screen came up at the window's size). A fullscreen request that fails drops back
  to the window.
* The lighting shader is generated for a given light count (`fragment_shader(max_lights)`,
  `LightingShader.max_lights`, `MAX_SHADER_LIGHTS` 16); raising "Torch & lantern lights" past
  what the shader was built for rebuilds it.

**The start screen** (`attract.py`, `hud.MenuRenderer.draw_title`, `tools/textures/generators.gen_logo`).
`TitleView` keeps a live `Game` at 14:00 behind the start menu: the town's time and NPCs run
(`World.update_time`, `update_npcs` - no player logic), the camera (`TitleCamera`) glides a
Catmull-Rom loop through `TITLE_ROUTE` - waypoints on the streets, checked clear of buildings by
a test - looking ahead along the way. The menu is drawn in a translucent panel under the logo
(`assets/textures/logo.png`, 900x440, generated like every other texture: `gen_textures.py
--only logo`); the menu has no subtitle of its own (the logo carries the tagline), so one there
is news - a save that would not load, wrapped to the panel (`MenuRenderer.subtitle_lines`).
Leaving a session for the menu starts a fresh title view; a save picked from the start screen
loads into a new `Game` that replaces the title's only once it has loaded. The roving eye is
no obstacle to the townsfolk (`World.player_is_solid`).

**Days that differ** (`world/npc.py`, `world/world.py`, `levels/*`). `NPCDef.rota` is a list of
alternative day plans; an NPC runs `day_plans[day % len]` (`NPC.use_day`), switched at midnight
(`World.update_time`) and on load (`resync_schedules`). Nineteen townsfolk have rotas (the widow's
chapel and park days, the cooper's felling day in the woods, the servants' changing errands, the
Reeve walking his bounds, ...); shopkeepers keep one plan. The builders grew the knobs for it
(`resident_tasks(supper=...)`, `gate_guard(supper=...)`, `rich_resident(other_days=...)`,
`servant(other_errands=...)`, `villager(other_days=...)`). Every plan of every NPC is checked
to leave at least a quarter of each task's hours after the walk to it (from where the previous
task left them), and every spot to be free of furniture.

**Doors in either direction** (`entities/door.py` and everything that measured a door along x).
`Door.axis` is "x" (the leaf spans x, the wall runs east-west - as before) or "z" (the leaf
spans z in a north-south wall); `along_of`/`across_of`/`to_xz`/`span`/`line`/`side_of`/`point`/
`strip` hide the difference from the collider, the swing (`closed_angle`/`open_angle`),
thresholds, lintels, doorsteps, `push_out_of_doors`, the NPCs' door crossing, the aim cone and the
doorway light patch. New doors: the chapel vestry's (Maude's key), the manors' servants' doors
between kitchen and hall, Greyhold's north and south gates (open, `gate="greyhold_gate"`).
`builders.house_walls` takes a west/east `door_side`.

**HUD** (`rendering/hud.py`, `core/messages.py`, `game.py`, `input/`). The inventory panel and
its scrolling are gone (the `Tab` view and the screens show the pack); the controls panel shows
only during the tutorial. The World Events log is hidden by default and toggled with `` ` ``
(`KeyBindings.toggle_log`, shown as `~`, `Game.toggle_event_log`); while it is hidden, lines that
are not town news show as fading toasts (`Game.toasts`, `TOAST_SECONDS`). `MessageLog.add(msg,
kind)` tags lines; the world's comings and goings are `news`. Reading: `E` or `Esc` puts the text
down whatever the crosshair is on. The status bar gained "Events [~]" and lays its items out by
their measured widths (`HudRenderer.status_layout`: the usual places, spread evenly - in short
forms if need be - in a narrow window).

**Fixes.** The old mine's sump steps start flush with the gallery floor and fill the passage's
width (`rock_steps`), and `World.can_step` lets the player step off a prop onto ground within a
step's height - the climb out of the sump no longer sticks; the gallery's timbers stand clear of
the way. Cave mouths: the jambs reach the hillside (`JAMB`), the hole in the ground spans the way
in and its jambs, thin slivers of surface rock over the mouth are cut (`SURFACE_PIECE`) and the
crags and brow stand wholly outside the chamber - Rusk's cove no longer shows ground and rock
clipping across its opening. The park is levelled under the fountain (`FlatDef` "park"). Windows
are placed clear of doors and signs by `builders.wall_windows`; the bakery's sign and a manor
painting moved off a window.

**Tests**: `tests/unit/test_screens_and_settings.py` is new (presets, the title stroll, z-axis
doors, the HUD and reading, the repairs); `test_realm.py` checks every plan of every rota; the
scenario goldens were re-recorded and reviewed.

### Eighteenth iteration - mountains, the chapel, bridges, long waits, the audit, frame time

**The Greyspine** (`levels/wilds._mountains`, `world/terrain.py`). Three ranks of gaussian hills
instead of two overlapping walls of them (which summed to a 270-unit cliff): a broken rampart at
the foot (steep enough that nobody climbs past z -85), craggy front peaks - each a broad shoulder
with a sharp summit stacked on it - and snow-capped high peaks behind (the highest ~110). Ridged
noise roughens them: `HillDef.crags` is the share of a hill's height that `Terrain._ridges` (three
octaves of `1 - 2|n|` on a lattice of its own, `TerrainDef.crag_scale`) adds or cuts - ribs,
gullies, broken summits. `HillDef.sharpness` / `profile` allow non-gaussian profiles (unused by
the level: their long tails lifted the town). The snow line is 50 and snow clings to steep rock
(`(1 - 0.3 rock)`). `_raw_scalar` and the grid agree (a test). Every cave chamber is still more
than a metre under the hillside (a test); the Deep's north cubby no longer overlaps the king's
hall (their floors fought).

**The chapel** (`levels/quarters.py`). The steeple is 15.5 tall, its belfry (12-15.5) over the
nave's ridge and open on all four sides: the lintel over the steeple door had filled the street
side's arch - `World._build_lintels` now stops at the first wall heading a doorway that ends short
of the eaves (and under an upper floor's boards). The door has a schedule (`DoorDef.open_hour` /
`close_hour`, `CHAPEL_HOURS` 6-21): `Door.open_hours_cover` exempts it from owners locking it
behind them, `apply_schedule` unlocks it again on any hour in its open hours, `resync_schedules`
sets it on load. Sister Maude's home door is the vestry's. Schedule messages are town news.

**Bridges** (`World._bridge_rails`, `NavGrid(fences=...)`). A bridge's rails are `Prop`s (kind
"rail") - colliders for the player, not sight blockers - and are rasterised onto the navigation
grid *after* the deck is cleared over the water, so ways over the stream run onto the deck at its
ends and down its middle (NPCs used to walk beside the rails, in the water, and through them).

**Graves** (`furnishings.draw_headstone`): the stone stands at the head of the plot (its far end,
looking the way the prop faces, over the mound) instead of in the middle with the mound running
past the plot's end.

**The prisoner's bed** (`World.is_prisoners_bed`, `player_may_sleep_in`): a bed in the cell area
is the player's while `JailState.sealed`, and no longer.

**Long waits** (`World.pass_hour`, `settle_npcs`, `waiting`, `sleeping_until`; `game.LongTask`).
A quick hour jumps the clock to the stroke, runs the tick (doors, upkeep, jail, inn, schedules)
with `World.quick_travel` set - `find_path` then only asks `NavGrid.route_exists` (rooms and
doors, no search) - and puts everybody who set off where they were going: they arrive (eat,
unlock), an owner leaving a building locks up behind (as `_replay_day_until` does). Sleep lives
the first hour in full and the rest quick (the last stroke unsettled: the town is seen on the
move); the sentence has no cap (it was 15 days, ~3 minutes of frozen window) and takes ~0.1 s a
day. `Game.start_long_task` runs a generator of progress: at once headless (`long_task_budget`
None: tests and goldens), otherwise `Game.update` advances it by the app's per-frame budget and
nothing else moves; `frozen`, `save_refusal` and `load_game` respect it. The [T] hour is one too
(`World.waiting` yields every `WAIT_SLICE` steps). The app draws `MenuRenderer.draw_busy` over
it (after `LongTask.SHOW_AFTER`), and `PygameApp.show_busy` before anything that blocks: a new
game, a load, a renderer rebuild, reopening the window, the benchmark, baking the scene.

**The visual audit** (`rendering/audit.py`, `tools/visual_audit.py`, `test_visual_audit.py`).
`GeometryRecorder` stands in for OpenGL (every `gl*` name the rendering modules imported, and
`draw_arrays`) while `StaticScene._build`'s draw routines, the animated structures and every door
(shut and open) run, keeping world-space triangles tagged with their owner, texture, polygon
offset and blending. `find_zfighting` buckets them by plane (`NEAR` 3 mm: as good as coplanar
at a distance) and clips same-facing pairs of different owners against each other, ignoring
faces that look into something solid or the ground (`Solids`); `find_through_walls`,
`find_ground_in_floors`, `find_floating` and `find_decals_off_walls` work on the boxes. It found
and this iteration fixed: the manors' downstairs walls topped flush with the upper floor (now
`UNDER_BOARDS`) and their doors' lintels likewise; the porches' column colliders drawn as square
blocks (kind "collider": not drawn); cave ceiling slabs and floors in the planes of the walls'
outer faces (slab inset `CAVE_SLAB_INSET`, cave floors drawn as their top alone); the cubby
overlap; open door leaves in the plane of the wall (set back `OPEN_LEAF_SET`); thresholds ending
in the plane of the shut leaf's edges (`THRESHOLD_INTO_JAMB`); neighbouring pines' facets (each
tree's cones turned its own way).

**Frame time and load time.** Measured headless / on llvmpipe, before -> after:

| | before | after |
| --- | --- | --- |
| simulation, mean frame | 8.4 ms | 3.4 ms |
| worst frame on the hour | 200-400 ms | 40-55 ms |
| first frame (baking) | 6.5 s | 1.4 s (medium) / 2.0 s (high) |
| new game / load from the menu, first frame | 5.3 s | 0.2 s |
| renderer rebuild (graphics change) | 8 s | 1.6 s |
| `Game()` (the first; then ~0.5 s) | 1.5-1.9 s | ~1.0 s |
| days in the cells | 11 s a day, 15 days max | ~0.1 s a day, no limit |

How: PyOpenGL's per-call error checking off (`unprepared/__init__._quiet_opengl`,
`UNPREPARED_GL_DEBUG=1` keeps it); material uniforms sent only when they change
(`LightingShader.set_material`); view-frustum culling of chunks, doors, flames, animated props
and NPCs (`camera.Frustum`, `Chunk.in_view` with a padded box and y range), pixel-identical on
the test views; chunks baked as first wanted (the first frame whole, then nearest first within
`SceneRenderer.bake_budget`, the rest ahead with what is left: `StaticScene._bake_ahead`;
`Renderer.prebake` for benchmarks); a bake reused by another world of the same level
(`StaticScene.matches`); the next hour's walks planned in its last minutes
(`World.plan_ahead`, `PLAN_FROM_MINUTE`, the route cache) and people set off a few a frame
(`World.switch_budget`) - both only in the app, so headless timing (and the goldens) is as it was;
A* with flat score lists reused between searches and closed doorways written into the grid; the
string-pull's line of sight as an exact grid traversal (never clearer than the old sampling: a
test); door geometry cached per pose (`Door._extent`, `collider`, `center`); `World.doors_in`,
buildings and areas looked up by grid cell (`_by_cell`), the crosshair's candidates limited to
what is near (`interaction_candidates`: the same target as searching everything, a test); terrain
pads and hills worked out only where they reach (`Terrain._window`, `_raw_grid`).

**Tests**: `tests/unit/test_town_and_time.py` and `test_visual_audit.py` are new; the lintel test
knows about heads and boards; the scenario goldens were re-recorded and reviewed.

**The window's real size** (found on a desktop with a window manager and a panel: the display
tests had only been run on a bare X server). A full-screen window is first placed in the work
area - the screen less its panels (1400x813 on a 1400x850 screen) - and made full screen a
moment later; the game had read the size straight after opening it and laid the view and the
HUD out for the smaller one. `PygameApp._settle_window` now waits (up to `SETTLE_SECONDS`,
handling the window's events) for a new full-screen window to take the size asked for, and
`_follow_window_size` checks every frame that the window is the size the game is laid out for -
if the window manager had its way after all, `Renderer.resize` sets the viewport, the
projection's aspect and the HUD's layout (and `MenuRenderer.layout`) for the size it is, without
touching the GL context. The menu probe checks both, the second with a resize behind the game's
back.
