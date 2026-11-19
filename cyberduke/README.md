# Cyberduke: Tactical Infiltration

A retro-futuristic, low-poly 3D stealth and infiltration game built with **Python**, **Pygame**, and **OpenGL** (a modern VBO/GLSL renderer, with the original fixed-function pipeline kept as a backend).

Inspired by the sneaking, quick-hacking, and corporate espionage elements of *Cyberpunk 2077* combined with the grit and attitude of classic retro shooters, **Cyberduke** features a scanner-and-quickhack netrunning layer with faction networks, an extensible hierarchical skeletal animation system, multi-sector level architecture, 2D A* pathfinding with raycast string-pulling smoothing, sensory guard AI with acoustic distraction states, security camera laser sweeps, environmental occlusion raycasting, and procedurally generated cyberpunk textures - and, around all that, the start of an RPG: an inventory with cyber mods to install, people to talk to, a first quest (steal a data stick from a flat across the alley), and save games. The street keeps time: a 24-hour clock, a smoggy golden-brown day and a neon night, and people who live in their own flats and keep their own hours. The people are rounded, cartoon-ish humans of believable heights, with hands whose fingers close round what they hold.

---

## Table of Contents

- [Features](#features)
- [Mission Objectives & Gameplay Flow](#mission-objectives--gameplay-flow)
- [Facility Layout & Sector Architecture](#facility-layout--sector-architecture)
- [Level 2: Night Market Alley](#level-2-night-market-alley)
  - [Number 16](#number-16)
  - [The Main Street, the Homes & the Hours](#the-main-street-the-homes--the-hours)
  - [The Neon Moth & the Metro](#the-neon-moth--the-metro)
  - [The Weather](#the-weather)
- [Inventory, Cyber Mods, Quests & Dialogue](#inventory-cyber-mods-quests--dialogue)
- [Being Caught](#being-caught)
- [Start Menu, Saves & Settings](#start-menu-saves--settings)
- [Project Architecture](#project-architecture)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Controls](#controls)
  - [Graphics quality](#graphics-quality)
- [Netrunning: Scanner, Quickhacks & Faction Networks](#netrunning-scanner-quickhacks--faction-networks)
- [Interactive Entities & Hazards](#interactive-entities--hazards)
  - [Security Camera](#security-camera)
  - [Vending Machine ("NEON COLD")](#vending-machine-neon-cold)
  - [Terminals & Blast Doors](#terminals--blast-doors)
- [Implementing Custom NPCs](#implementing-custom-npcs)
  - [1. Pose Generation & Keyframing](#1-pose-generation--keyframing)
  - [2. Appearance & Props](#2-appearance--props)
  - [3. Behaviour, Pathfinding & Perception](#3-behaviour-pathfinding--perception)
  - [4. Step-by-Step Example: Datapad Courier](#4-step-by-step-example-datapad-courier)
- [Roadmap](#roadmap)

---

## Features

- **Intelligent A* Pathfinding & Collision Avoidance**:
  - 2D A* pathfinder (`ai/pathfinding.py`) on an agent-inflated grid ($r = 0.35\text{m}$, $0.4\text{m}$ resolution): room-local routes in ~1 ms, a full cross-sector route in ~50 ms.
  - Greedy raycast string-pulling (path smoothing) using `core.geometry.segment_clear_2d` to turn jagged grid paths into natural, direct trajectories around columns, server banks, and barrels.
  - Step-level physical collision detection with obstacle surface sliding, preventing NPCs from ever clipping through walls, columns, or cover props.
  - Fast-path direct line-of-sight optimization: skips grid searches entirely when the line of sight to the goal is clear.
- **Sensory Guard AI & Acoustic Distraction**:
  - State machine supporting patrol routines, configurable wall leaning (`wall_pos`, `wall_yaw`), weapon inspection, acoustic investigations, and perimeter scanning.
  - Human-scale vision: an 11.5 m, 110° cone while alert (3 m while engrossed in gear). Detection gain falls off with distance and is much lower for a crouching player.
  - Suspicion: a sighting makes the guard stop and look (`ALERTED`); once the detection meter passes the yellow threshold it creeps toward the last-seen position (`SUSPICIOUS`), looks around on arrival, and resumes its routine when the player hides.
  - Ray-box line-of-sight (LOS) occlusion against everything opaque: walls, cover, rail panels, catwalk decks and stairs (`Room.get_sight_blockers()`). Only geometry that really has gaps (the stair balustrades) is transparent, and a level test enforces that every opaque prop is covered.
  - Arrival-gated acoustic distraction routine: the vending machine emits a `NoiseEmitted` event; guards within its radius navigate around obstacles to the investigation point, face the source, bang on it and keep watching it (45 seconds for the vending machine, 40 for a fried computer, 15 by default) before returning.
  - A single `DetectionSystem` sums all observers' sightings per frame and applies one decay, so detection is order-independent and never flutters.
- **Surveillance Hazards**:
  - Wall-mounted security cameras with sweeping motor heads ($\pm 35^\circ$ sweep).
  - Laser frustum fans clipped in 3D against the geometry you actually see (`render/beams.py`: walls, doorframes, racks, crates, door leaves and desks): a ray passes over a low crate and lands behind it, or stops on a wall face and drops a vertical seam line to the floor.
  - Progressive detection escalation (+65/s when upright, dampened to +18/s when crouching) with ray-box LOS checks through open doorways.
- **Multi-Sector Facility Architecture**:
  - Modular sector design separating Staging Bay, Airlock Corridor, and the multi-tiered Datacore Vault.
  - Coolant drainage trenches with emissive liquid beneath grates, elevated catwalks ($Y = 2.75\text{m}$) with access stairs, hazard handrails, steam pipes with pressure gauges, and central datacore monoliths.
  - Dynamic floor height interpolation (`room.floor_height(x, z, state)`) allowing the player to climb ramps and traverse catwalks.
  - Unified obstacle collision and spatial bounding across sectors.
- **Hierarchical Humanoid Skeleton, rounded bodies and hands** (`animation/rig.py`, `animation/hands.py`, `render/shapes.py`, `render/batch.py`):
  - Independent rotation channels for neck, head (pitch/yaw/roll), shoulders, elbows, hips, and knees.
  - **One description of the body** (`animation.rig`) drives both the renderer and the collision skeleton (the rag doll, the tests): each part a box - and what is drawn in it a rounded shape: a skull with a narrower jaw and a face over its front, a cap of hair with a clean hairline, tapering round-ended limbs with round shoulders, elbows and knees, a torso with a waist and a collar, boots with a sole, a heel and a toe cap, a guard's wraparound visor.
  - **Hands**: a palm, four two-bone fingers and a thumb. What a hand holds decides how (`GRIP_FOR_ITEM`): fingers curled round a can (which turns with the hand - tipped up at the lips), a phone held up in both hands - one at either side of it, the palms turned in to each other, the fingers curled in under it and the phone resting up on them (an inch above the hands' middle, so no finger goes through it), the thumbs up by its edges, tipped up to the face (`TWO_HANDED`: the `text` grip; `street.PHONE_FINGERS`, `PHONE_STAGGER` - one hand a little further up it than the other, so their fingers lie side by side; `street.phone` gives anybody on the phone one of their own) - a cigarette between two fingers, a knife in a fist, a box on the flat of the hand, a parcel between two flat palms. The wrist bends so the thing held sits exactly where the arm IK put it. Clips give empty hands their own shapes (a fist to knock, a pointing finger, open hands talking, fingers tapping keys, the thumb scrolling a phone, a beckoning curl) - `HumanoidPose.grip_l/r`, `fingers_l/r`.
  - **Drawn in one batch per person** (`render.batch.BatchContext`): the ~60 parts are transformed on the CPU in one NumPy pass and handed to the GPU as one buffer with a draw per texture; further off the hands are mittens and the parts coarser.
  - Normalized socket system for rendering dynamic, holsterable props (weapons, datapads, tools) that transition smoothly between hips, chest, and hands.
- **Stealth & Infiltration Mechanics**:
  - First-person crouching system that lowers eye-line, dampens movement profile, and minimizes camera/guard detection build-up.
  - Multi-stage terminal hacking requiring physical proximity and continuous decryption channels.
- **Netrunning: Scanner, Quickhacks & Faction Networks** (`systems/scanner.py`, `systems/hacking.py`):
  - `[F]` fires a scanner ping that reveals every electronic target within 16 m *through walls*: computers, cameras, doors, vending machines and people with cyberware. Targets you can act on get screen-space markers (a pinged target out of reach is just a dim pip - no phantom doors floating behind hedges); the one under the crosshair becomes the focus and lists its hacks on `[1]`-`[9]`.
  - Per-target menus with design restrictions baked into the data: a `critical` computer cannot be overloaded, an `essential` (elite) guard cannot be knocked out, a `high_security` door cannot be forced, a `secured` computer is never reachable over the network.
  - Computers: **OVERRIDE DOORS**, **STEAL DATA**, **OVERLOAD** (fries it for a loud distraction - and it controls nothing afterwards). People: **BLIND** (six seconds of no perception, hands over the visor), **ELECTROCUTE** (permanently down). Doors: **FORCE OPEN** (the metro's ticket gates too: METRO TRANSIT). Cameras: **SHUT DOWN** (dark for three minutes, the head hanging from its swivel), **HIJACK FEED** (look through the lens, pan it, and launch direct hacks from there). Vending machines: **DISTRACT**, repeatable.
  - Every hack uploads for a moment, then takes effect through the target's own primitives (`guard.blind`, `camera.shut_down`, `terminal.overload` -> `NoiseEmitted`, ...), so effects run on the same perception, detection, door and noise code as the rest of the game.
  - **Factions** each share one local network. Hacking any computer on it - a direct quickhack or jacking in with `[E]` - *breaches* it: every member is highlighted across the map and can be hacked from anywhere, no line of sight needed. `secured` computers show up on the scanner but must be hacked with a line of sight; `hardened` (air-gapped) ones cannot be hacked wirelessly at all - jack in or nothing.
  - **Terminal interface**: holding `[E]` at a computer establishes the link (once - a breached computer reconnects instantly, no link bar) and opens *its* screen: UNLOCK DOORS, HIJACK CAMERAS, DISABLE CAMERAS (every camera on its network at once; a direct quickhack only ever takes down the one it is aimed at), SHORT CIRCUIT (DISTRACT & DESTROY), and STEAL DATA only where there is data that matters (the datacore's cipher). Each refusal says why. Being jacked in is the breach; what you do with the machine is a separate decision.
  - **Camera feeds**: a hijacked camera is free-look (pan within its sweep, tilt from slightly up to straight down), scans and direct hacks launch from the lens, and `[<]`/`[>]` switch between every camera on the same network. Shutting one down cuts it out of the security system for three minutes - its head goes limp and hangs from the swivel - and that countdown runs regardless of what the netrunner does: a dark camera can still be climbed into and looked through.
  - **Distractions with teeth**: a shorted vending machine sparks and pops for 18 s and the guard who comes hits it for 12 s before staring at it for 45 s in all; an overloaded computer crackles for 30 s and holds a guard's attention for 40 s. Noises carry their own inspection time and gesture (`NoiseEmitted.inspect_duration`, `gesture`).
  - **Encryption-keyed doors** open for a player who walks up carrying the right stolen data (`DoorData.key_intel`); the **exit zone** (`LevelData.exit_zone`) is the way out: it loads the next level (`LevelData.exit_to`, a name in `levels.LEVELS`). There is no winning: where the world is not built yet, it ends in a barricade (the main street is closed at both ends, and a guard rail keeps you off the road - its far side is scenery).
  - Everything is unlocked for the player for now; `Player.known_hacks` / `learn()` is the hook for the levelling system.
- **Two levels and a hand-over** (`levels/__init__.py: LEVELS`, `Simulation.load_level`):
  - Stepping into the facility's exit zone loads the **Night Market Alley**: the same `Simulation` rebuilds the room, entities and networks in place (`level_generation` ticks), the player keeps the stolen intel on the deck, and the renderer notices by itself - cached geometry is dropped and the new room's lights taken (`SceneRenderer.follow_level`). A title card with the level's name and `brief` comes up for five seconds on arrival.
  - Caught, what it costs depends on where (`LevelData.on_caught`): in the facility a red **YOU WERE CAUGHT** screen, a fade to black and a **new game**; in somebody's home (`PrivateArea`), being **thrown out** of it - see [Being Caught](#being-caught).
- **Inventory & cyber mods** (`core/inventory.py`, `[Tab]`): everybody carries things - the player, the guards, the people of the street. The player starts with a deck, 140 eddies and two cyber mods that do nothing until they are installed with EQUIP: the **Sonaris Ping Amplifier** (the scanner reaches 60 % further) and the **Zetatech Quickjack Link** (65 % faster hacking at arm's length). See [Inventory, Cyber Mods, Quests & Dialogue](#inventory-cyber-mods-quests--dialogue).
- **People to talk to and a first quest** (`core/dialogue.py`, `systems/quests.py`): `[E]` at somebody who has something to say opens a captioned conversation; Nix the smoker wants the data stick from **number 16**, a shabby flat behind the alley's west wall with a hackable access panel by its door, a home camera and an owner who does not like strangers (`levels/apartment.py`).
- **Save games & a start menu** (`game.py`, `systems/savegame.py`): the game opens on a start menu over the living alley; saves go to `~/Unprepared/Saves`, one JSON file per save named by its date and time, never overwritten, holding the whole state of the run. Fullscreen and the graphics preset are kept in `~/Unprepared/settings.json` and applied at startup.
- **Civilians & merchants** (`ai/civilian.py`, `ai/alley.py`): a street that is not a threat, and has a life. With a `role`, each person lives out a **story** - a generator of *acts* (walk there, step up, play a gesture, wait for something) run against the street's shared state (`AlleyLife`: the orders at each counter, the things lying about, who asked whom for a chat): the sushi chef makes what is ordered from the cooler to the counter, customers queue, order, pay, take, eat and bin the box, a courier's parcel reaches the resident of number 16. Gestures put the hands where the job needs them (two-bone arm IK, `animation/reach.py`; clips in `animation/street.py`) and people hold what they use (`render/items.py`). Everybody looks at whoever comes close and **steers round each other and the player** (`BaseNPC.navigate_towards(avoid=...)`); without a role a civilian just wanders and a merchant just works. All are solid to the player (`BaseNPC.blocks_player` -> `Simulation.player_obstacles`) - who can push them aside, somewhere they can stand, when they are in the way (`Simulation._shove_people`, `StreetPerson.shove`; they stagger, then go back to what they were doing: nobody can pin the netrunner in a corner), never report a sighting - except somebody in their own home (`StreetPerson.watch_area`: Dell, inside number 16) -, and get a seeded look (`civilian_spec`: shirt, trousers, skin, hair, a height of 1.75 - 2 m, no plating).
- **A clock, a day and a night** (`core/clock.py`, `render/scene.py: time_of_day_light`): the world keeps a 24-hour clock (`WorldClock`: half a game minute per real second - a day is 48 minutes; a new game starts at 19:30), shown top left on the HUD as a small military-time digital clock (the time of day only: nothing counts the days that go by). The outdoor level's light follows it: by day a smoggy golden-brown sky and ambient (about a fifth brighter than the night - measured 1.1-1.35x the mean screen brightness at noon), the street lamps off; dawn 05:30-08:00 and dusk 17:30-20:00 blend between the two.
- **Homes and hours** (`levels/flats.py`, `levels/main_street.py`, `ai/schedule.py`): everybody of the street lives somewhere - Nix at number 17 across the alley from his wall, the rest in flats round the corner on the main street - and keeps hours: the junk dealer is home from midnight to ten, the sushi chef from two to two, the office worker working from home in the day and busy at the stalls after five, Nix almost never home. Nobody appears or disappears: whoever leaves the street walks home and lets themselves in at their own front door, and comes back out of it. The stalls shut (a CLOSED sign, a cover over the stock) when their merchants go home, and the customers go elsewhere. At home people potter about the flat, watch TV (flicking through the channels), work at the computer and sleep in their beds. See [The Main Street, the Homes & the Hours](#the-main-street-the-homes--the-hours).
- **Weather** (`systems/weather.py`, `render/weather.py`): rain in spells - a drizzle fading in slowly, steady rain, downpours, thunderstorms with lightning - puddles that fill and dry and that cars throw spray out of, and fog that comes and goes over hours and hides the end of the street. See [The Weather](#the-weather).
- **A night out** (`levels/bar.py`, `levels/metro.py`, `ai/nightlife.py`, `entities/metro.py`): the **Neon Moth** round the corner - a bar with a bartender who comes in on the metro, opens at five, pulls pints for whoever drops in (the people of the street, a regular off the train) and goes home on the metro after three - and the **metro station** at the east end: a glass hall with ticket gates (no pass, no entry - unless you force one), a stair anybody can climb, a footbridge and a platform over the road where a two-car monorail train stops every couple of minutes. Riders walk into the train's doors and the doors close on them: nobody appears or disappears.
- **Traffic** (`entities/traffic.py`, `world/vehicles.py`, `render/traffic.py`): coupes, taxis and vans drive up and down the widened main street between tunnels at either end, wheels turning, following the car in front at a safe distance, brake lights and headlight pools on the wet road.
- **Faces and looks** (`animation/look.py`, `animation/anatomy.py`, `assets/faces.py`): everybody their own hair, cybernetic eyes, chrome and clothes; faces painted per skin tone, mark and expression, moving as people talk, eat, smoke, drink and shout.
- **Pause & Graphics Settings**:
  - `[Esc]` opens a pause menu (resume / save / load / graphics / display / new game / main menu / quit). A paused game consumes no time at all - no timer, no animation, no upload, no door - and resumes exactly where it stopped (`Simulation.paused`, enforced by a snapshot test over every moving part).
  - Quality presets (LOW / MEDIUM / HIGH) set view distance, the hardware light budget, the ambient light level, the animated-detail distance, spark and scanner effects, camera-beam density and the HUD overlay, and can be changed live from the menu.
  - On startup the GL renderer/vendor strings pick the preset: software rasteriser -> LOW, recognised discrete GPU -> HIGH, otherwise MEDIUM (`--graphics low|medium|high` overrides).
- **Atmospheric 8-Point Neon Lighting**:
  - Up to eight `Light` records per level drive both the hardware lights (`GL_LIGHT0`-`GL_LIGHT7`) and their visible fixtures, so they cannot drift apart; `lights_for` keeps the nearest within the preset's budget.
  - Sector-specific color grading: Staging Bay cyan/magenta/amber neon, Airlock industrial yellow caution lamps, Datacore halo/catwalk floodlights, the greenhouse's blue fungus and grow lamps, the alley's wok flame, stall neon and door lamps, and the main street's lamps along the kerb (`Light.night_only`: off by day).
  - **Soft-edged light** is a first-class primitive: `DrawContext.soft_disc` (a fan with per-vertex alpha; a `glow` shader on the modern backend) and `Prop.disc` draw the pools under the mushrooms and lanterns, the neon on wet pavement, puddles and the greenhouse haze without a hard rim anywhere.
  - **Light pools lie on what they light**: a disc can be cut to a rectangle (`Prop.clip`), so a mushroom's pool stops at the edge of the planter it grows on instead of hanging in the air past it, and sits on top of the moss rather than buried under it; the same light splashes up every rack face, planter side and dome wall next to the cap, fainter the further away the face is (`levels/dressing.light_fungus`). All the pools of a colour in a sector are baked into one mesh (`render/glow.py`, `DrawContext.glow_mesh`) - one draw call instead of hundreds.
- **Zero-Dependency Procedural Textures**:
  - NumPy generation (periodic FFT noise, blur, relief shading, strokes and cracks) of 128x128 textures: metal grates, floor tiles, wall panels, brick, concrete, asphalt, wood, server racks, hazard stripes, terminal UIs, beverage cans, fabric, leather, armor plating, skin and the faces (cartoon eyes - big white ovals, round pupils with a catch light, block brows - to suit the blocky characters; a neutral mouth, a grin and gritted teeth: `skin`, `skin_smile`, `skin_grimace`).
  - Surface materials are **seamless** and tile at a real-world size (`@texture("brick", tile=1.0)`: one tile per metre): both backends repeat them across a prop's actual size instead of stretching one copy over a 20 m wall. Uploaded with a full mip chain (trilinear filtering, anisotropic where the driver has it), so distant surfaces do not shimmer.

---

## Mission Objectives & Gameplay Flow

The objective is the data on the datacore terminal. Every stage has a sneaking route and a
netrunning route, and they mix freely:

```text
 ┌────────────────────────────────────────────────────────────────────────┐
 │ Stage 1: Infiltrate Sector 1 (Staging Bay)                             │
 │ - Evade (or BLIND / ELECTROCUTE) the Sector 1 CyberGuard patrol.       │
 │ - Reach & jack into the Netrunner Terminal (X=3.5, Z=-4.0) with [E],   │
 │   or scan it and quickhack OVERRIDE DOORS from across the bay.         │
 │ - Either way the MIL-SPEC blast door at Z=10.0 opens (it cannot be     │
 │   forced directly) and the Halcyon security network is breached:       │
 │   every camera, door, guard and vending machine lights up on the HUD.  │
 └──────────────────────────────────┬─────────────────────────────────────┘
                                    │
                                    ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │ Intermission: Sector Airlock Corridor                                  │
 │ - Navigate corridor between Z=10.0 and Z=18.0.                         │
 │ - Crouch beneath the sweeping Security Camera laser fan (Z=11.2,       │
 │   Y=3.6), SHUT it DOWN, or HIJACK its feed and work from the ceiling.  │
 │ - The vault bulkhead at Z=18.0 is a plain networked lock: FORCE OPEN   │
 │   it in sight, or over the network before you even enter the corridor.│
 └──────────────────────────────────┬─────────────────────────────────────┘
                                    │
                                    ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │ Stage 2: Sector 2 Datacore Vault Espionage                             │
 │ - Enter Datacore Vault (Z=18.0 -> 42.0).                               │
 │ - DISTRACT with the "NEON COLD" Vending Machine (X=-7.5, Z=25.0) - by  │
 │   hand or remotely, as often as it recharges - or BLIND the elite      │
 │   vault guard (hardened cyberware: it cannot be knocked out).          │
 │ - The Datacore Terminal (X=0.0, Z=30.0) is SECURED: it is not on the   │
 │   network. Scan it and STEAL DATA with a line of sight (the vault     │
 │   doorway, the catwalk rail), or jack in at the desk.                  │
 │ - It is CRITICAL: OVERLOAD is refused - the data must be taken, not    │
 │   burned. The data is the GREENHOUSE DOOR CIPHER: walk up to the door  │
 │   at the back of the vault (Z=42, under the skybridge) and it opens.   │
 └──────────────────────────────────┬─────────────────────────────────────┘
                                    │
                                    ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │ Stage 3: Sector 3 Fungal Server Greenhouse (the way out)               │
 │ - A domed hall (Z=46 -> 76) of bio-server hedges and glowing fungus,   │
 │   laid out as a maze. Two patrols (one elite) walk waypoint loops;     │
 │   four cameras watch the corridors. It is the biotech division's own   │
 │   network: nothing you breached so far reaches in here.               │
 │ - The short route north runs through the EAST CONTROL NODE and its    │
 │   MAZE GATE. The node's terminal is AIR-GAPPED: jack in with [E] to    │
 │   open the gate (and breach the biotech network) - or take the long   │
 │   way round, twice the distance, through the west of the maze.        │
 │ - The STREET EXIT in the north wall answers only to the WEST CONTROL   │
 │   NODE's terminal (air-gapped too). Jack in, get back to the exit, and │
 │   step out onto the street.                                            │
 │ - Netrunner route: from the spore lock, hijack the entry camera, then  │
 │   hop feed to feed (entry -> east node -> spine -> exit) scanning and  │
 │   shutting cameras down before your body ever enters the maze.        │
 │ - Past the exit door a short vestibule ends at the outer staff door:   │
 │   walking up to it the picture fades to black, and back in on the next │
 │   level (out of that same yellow door into the alley) with the data    │
 │   still on your deck.                                                  │
 └──────────────────────────────────┬─────────────────────────────────────┘
                                    │
                                    ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │ Level 2: Night Market Alley (the free world)                           │
 │ - Out of the yellow door into a tenement alley: a wok stall, a         │
 │   cyberware junk stall, a crowd that parts around you, dumpsters,      │
 │   lanterns on cables, the main road glowing at the far end.            │
 │ - Nobody out here reports you to Halcyon. Nix, the smoker by the wall, │
 │   has a job for a netrunner: the data stick in NUMBER 16 (a flat with  │
 │   an access panel, a home camera and an owner who throws you out).     │
 │ - Out of the alley's mouth: the main street's pavement, the homes of   │
 │   the people you met, a car at the kerb, the METRO and the UNDERPASS.  │
 │   Day and night come round; everybody keeps their own hours.           │
 │ - There is no winning: the game goes on. Barricades close the street   │
 │   at both ends - the city beyond is not built yet.                     │
 └────────────────────────────────────────────────────────────────────────┘
```

---

## Facility Layout & Sector Architecture

The facility is divided into four sectors, defined as data in `levels/facility.py` and brought to life by `Room`:

```text
    Z=79.2              [outer staff door: EXIT ZONE (Z 78.2-79.2) -> Night Market Alley]
    Z=75.9..79.2        [street vestibule]
    Z=75.7 ╭─────────────[Street Exit]───────────────╮
           │    Sector 3: Fungal Server Greenhouse    │
           │   16-sided dome, 69-cell hedge maze      │
           │  [West Node]  ~  ~  ~  ~  ~  [East Node] │
           │       ~  ~  ~  ~  ~  ~  ~  ~  [gate]     │
    Z=46.3 ╰─────────────[spore lock]────────────────╯
    Z=42.0 ┌───────────[Greenhouse Door]─────────────┐
           │        Sector 2: Datacore Vault         │
           │  Catwalk (Y=2.75)[===========] Catwalk  │
           │       [Trench]  [DATACORE]  [Trench]    │
           │                 Terminal 2              │
           │  [Vending]                              │
    Z=18.0 ├───────────────[Bulkhead]────────────────┤
           │      Sector Airlock Corridor            │
           │     [Camera] Steam Pipes & Grates       │
    Z=10.0 ├───────────────[Blast Door]──────────────┤
           │        Sector 1: Staging Bay            │
           │  Server Racks, Crates, Pillars          │
           │                 Terminal 1              │
    Z=-10.0└─────────────────────────────────────────┘
           X=-10.0            X=0.0            X=10.0
```

### 1. Sector 1: Staging Bay ($Z \in [-10.0, 10.0], X \in [-6.0, 6.0]$)
- Features heavy server banks, structural concrete pillars, tactical cargo crates, and a Japanese neon sign.
- Guard 1 patrols between $Z = -8.5$ and $Z = -3.0$, occasionally leaning against the left wall at $(-5.5, -6.0)$ facing $+X$ (`wall_yaw = 90.0`).
- Contains the Stage 1 Netrunner Terminal ($X=3.5, Z=-4.0$) controlling the primary blast door.

### 2. Sector Airlock: Corridor ($Z \in [10.0, 18.0], X \in [-2.5, 2.5]$)
- Pneumatic transition airlock with reinforced bulkheads, wall-mounted steam pipes, illuminated pressure gauges, and floor drainage grates.
- Overhead industrial amber caution lighting (`GL_LIGHT3` at $X=0.0, Y=3.6, Z=14.0$).
- Guarded by an active `SecurityCamera` ($Z=11.2, Y=3.6$) whose laser rays truncate cleanly against the airlock walls.

### 3. Sector 2: Datacore Vault ($Z \in [18.0, 42.0], X \in [-10.0, 10.0], Y \in [0.0, 6.0]$)
- **Central Datacore Monolith**: Rises 5.5m high at $X=0.0, Z=32.0$ with glowing data bus lines, housing the primary extraction terminal ($Z=30.0$) facing outward into the room ($180^\circ$ yaw).
- **Dual Coolant Trenches**: Recessed drainage canals at $X = \pm 3.5$ with glowing cyan/green emissive coolant fluid under heavy iron grates.
- **Elevated Perimeter Catwalks**: Raised walkways at $Y=2.75\text{m}$ (clearing guards' heads) reached by eight-step staircases ($Z \in [21.0, 24.2]$) on both sides, with handrails that tie into the deck rails. NPCs cannot climb them (their navigation is ground-level), which the level declares as a no-go zone.
- **Tactical Cover & Distractions**: Auxiliary server racks, cryogenic storage cylinders, crate clusters, and the "NEON COLD" vending machine ($X=-7.5, Z=25.0$).
- Guard 2 patrols the clear right walkway lane ($X = 2.1, Z \in [24.0, 36.0]$) and leans against the right wall at $(9.45, 30.0)$ facing $-X$ (`wall_yaw = -90.0`). It is an `essential` elite (gold visor): BLIND works on it, ELECTROCUTE does not.
- The vault's entrance at $Z=18.0$ is sealed by the **vault bulkhead**, a standard-security networked door with no controlling terminal.
- A matched pair of hanging cage lights (magenta left, cyan right, cables to the 6 m ceiling) flanks the monolith at $X = \pm 5.0, Z = 32.0$.
- The **greenhouse door** at $Z=42.0$ (under the rear skybridge) is encryption-keyed to the datacore's data.

### Set dressing (every sector)
Years of neglect: cans, printouts, flattened boxes, loose cable, puddles, a broken monitor face-up on the floor, graffiti, a dead cage light hanging crooked, a loose hazard sign, coolant stains, pulled server blades, fungus coming up the drains in every room, and an alley with lit windows past the exit. All of it is generated deterministically (`levels/dressing.py`) and placed with rejection sampling around every collider, desk, doorway and patrol lane, so it never clips and never blocks a route; no two desks share a layout (`TerminalData.style`: triple, double, single, damaged, overgrown). Fungus grows only in the greenhouse (a test says so); the other rooms get the trash, stains and puddles.

### 4. Sector 3: Fungal Server Greenhouse ($Z \in [42.0, 80.0], X \in [-17.0, 17.0]$)
- A **sixteen-sided dome** (apothem 14.7 m, centre $Z=61$): a vertical wall tier with the two door openings, two sloping glass tiers, ribs, a dim skylight 9.6 m up, and a layer of grey haze drifting above the maze (translucent `fog`-tagged props, drawn in a late pass). Planter beds of soil fill the rim between the cell grid and the curved wall. The hall is kept gloomy on purpose - just bright enough to navigate, lit mostly by the fungus.
- The **maze**: 69 cells on a 2.8 m grid (2.0 m corridors). Hedges are 2 m bio-server racks (no two the same height, some skewed, every fifth one dead) meeting at small soil planters on the grid nodes (`GH_NODE_BEDS`), draped in vines. The fungus itself - bright blue, pulsing, each cap pooling soft light on the ground - grows in seeded clumps of varying size at the hedges' feet, over the node planters and across the rim beds, never in rows (`levels/dressing.py`); a test proves no cap or stalk clips anything. `GH_WALLS` in `levels/facility.py` is the authored wall list; everything else is generated from it and kept honest by the traversability and level-data tests (every cell reachable, no pockets, the short route needs the gate, the long route is 30 cells to its 16).
- Two 2x2 **control node** chambers hold the air-gapped terminals: the east node (on the short route, behind the maze gate) and the west node (on the long route). Each is lit by a grow lamp; the east node has its own camera on the desk.
- Two route patrols (`patrol_route` waypoints, A* between them): the spine guard on the central corridor and the west loop, and an elite on the east side and the top corridor.
- Four cameras on the biotech network, each within quickhack range and line of sight of the next.
- The **spore lock** corridor joins the vault door to the dome; the **street vestibule** beyond the exit door is a bare concrete stub between two doors: the dome's exit, and on its back wall the **outer staff door** - the same yellow Halcyon door (leaf, frame head, STAFF ONLY neon, alarm lamp, `levels/doorway.py`) the alley shows you come out of, with the EXIT sign over it. It never opens here: the last metre of floor in front of it is the exit zone, so walking up to it is walking out of it (it used to be a bare wall you walked into). The dome's one light is cyan, under which the yellow stripes read as green, so the vestibule's warm ceiling strip is baked into the door as emission (`VESTIBULE_STRIP_LIGHT`; the level's eight lights are spoken for).

---

## Level 2: Night Market Alley

`levels/street.py`, one sector: the alley, $X \in [-4.5, 4.5]$, $Z \in [-2.5, 42.5]$, opening onto the main street's pavement ($X \in [-34, 32]$, $Z \in [42.5, 45]$, `levels/main_street.py`), plus the flats behind the walls (number 16, `levels/apartment.py`; the others built from a `FlatPlan`, `levels/flats.py`), the Neon Moth bar round the corner (`levels/bar.py`) and the metro station at the east end (`levels/metro.py`). Out here nobody reports you to Halcyon - the only things to hack and the only people to hide from are in the homes.

```text
    Z=54   ══ far pavement: towers, neon, shopfronts (backdrop) ══════════════════════
    Z=50.5..53   eastbound lane  ──>  cars, taxis, vans (out of one tunnel, into the other)
    Z=50..50.5   median: the monorail's pillars; the beam 7 m up, the trains along it
    Z=47.45..50  <──  westbound lane        the metro platform over it (X 26..43.6, 7.6 m up)
    Z=45..47.45  parking strip: the coupe at the kerb
    Z=42.5..45 pavement: [UNDERPASS] 35 33 31 NEON MOTH ║alley║ 32 34 36 [METRO hall]
               X=-34 barricade                  X=0            X=32 barricade  X=±44 tunnels
    Z=42.5 ═══════════╗  alley mouth  ╔═══════════════
           ║          │               │ [21]         ║   [n] = a tenement door with a stoop
           ║ dumpster │               │ bike         ║
           ║ [16]     │               │              ║   number 16: Dell's door (it opens) and his flat
           ║          │               │              ║   behind the wall - see "Number 16" below
           ║          │               │ CYBERWARE    ║   Z 23.8-27.2: narrow trestle table, tarp,
           ║          │               │ junk stall   ║   chips / optics / decks, the dealer, stock
           ║          │               │ VENDING      ║   Z 17.6: NEON COLD vending machine
           ║ SUSHI &  │               │              ║   Z 12-16: counter with the board, knife and
           ║ WOK bar  │               │              ║   boxes; the chef's lane; wok, rice cooker
           ║ [14]     │               │              ║   and fish cooler on the bench behind
           ║ [17]     │               │ dumpster     ║   number 17: Nix's door, across from his wall
    Z=-2.5 ╚═════[ yellow door: HALCYON - STAFF ONLY ]═╝  <- spawn; Nix the smoker by the wall
           X=-4.5                                X=+4.5
```

- **Tenement facades** on both sides: brick, skirting (it stops at the doors now - it used to run straight across them), downpipes, rows of windows (warm, cool, a TV flicker, dark, some with curtains or an air conditioner), three home doors with stoops (14, 16, 21) and number 17's, lamps over 14, 16 and 17, overhead cables with paper lanterns (each pooling light on the asphalt), two dumpsters, a bike, trash bags and scattered litter (`Placer` seed 808).
- **The people of the street** (`ai/alley.py`) each have a story - a loop of jobs that follow on from one another and bring them to the stalls, the doors and each other:
  - **the sushi chef** (SUSHI & WOK) takes an order, crouches to the fish cooler for a fish (a whole salmon, lying on its side: a tapering scaly body - dark speckled back, silver flanks, pale belly - a rounded head with a gold-ringed eye, gill covers, a forked tail, dorsal, adipose, anal, pectoral and pelvic fins: `render/items.py`, `shapes.fish_body`, `shapes.fin`, the `fish_skin` texture), lays it on the board, picks up the butcher knife lying flat there and cuts the fish into sushi (the blade under the grip, edge down) - four nigiri in a row on the board, the same four that go into the box (`SUSHI_PIECES`; salmon with its stripes of fat over a block of rice) - puts the knife back, takes a box off the stack, packs the sushi, closes the box and sets it on the counter in front of the customer - or tosses a chow mein in the wok (a bowl on its burner, heaped with noodles, its handle towards the chef) and fills a noodle box - and between customers wipes the counter down, stirs the wok, tidies and calls out to passers-by - the hand up beside the head, the palm to the front, waving them over (`street.beckon`: arm IK beside the head, the `wave` grip with the palm turned forward, the fingers curling "come here" now and then);
  - **the junk dealer** (CYBERWARE CHEAP) haggles, picks the thing up, wraps it and holds it out over the table; fetches the courier's parcel from the stock at the back; between customers puts stock out, arranges the table and waves people over (`clips.hawker_animation`: the palm to the front);
  - **the office worker** comes down from the main road, queues, points at the menu, orders, pays, waits (checking the phone, held up in both hands), takes the box, sits on a stool and eats, gets down (well clear of the stool), bins the box and goes home - and comes back later (noodles, eaten standing by the awning pole, now and then);
  - **the tinkerer** browses the junk table, picks a deck up and turns it in the light, puts it back, thinks it over (hand on chin), haggles, pays, takes the wrapped purchase and strolls off with it (nobody stands about admiring what they bought) - and stops for noodles on the way out;
  - **the courier** collects a parcel for number 16, carries it there in both hands, tucks it under an arm at the foot of the step, steps up and knocks - on the door's face: the rapping forearm comes forward exactly to it (`clips.knock(t, door)`, measured against the drawn forearm box) instead of a hand's length through it. If the **resident** is in, he steps back down, they open the door and it changes hands on the doorstep; if not, he turns along the step, takes the parcel in both hands, bends down (knees and back: `HumanoidPose.lean`, `clips.stoop_reach`) and sets it in the corner of the step clear of the door and its frame, letting go of it exactly where it then lies - and the resident finds it the next time they come out, bends down for it the same way, reads the label, tucks it under an arm and takes it in. The resident also comes home with the shopping, and goes out for sushi to take home;
  - **the smoker** hangs about by the wall near the Halcyon door with a cigarette and a phone, chats with the **stroller** (who wanders the alley, looks up at the neon and browses without buying), and gets a box of noodles now and then - or a can from the vending machine, drunk with the elbow up and out and the can tipping up in his fingers at every sip (`animation.street.drink`). The cigarette sticks out from between the fingers away from the face (across the forearm, forwards and out: `render.items` "outward"), filter at the lips on a drag, turning with the hand as it comes down; whoever he was looking at, his head comes back round to the cigarette as it comes up for a drag and turns back to them after (`HumanoidPose.head_held`: the glance at a passer-by gives way to it) - and whatever else his look gives him, never a respirator over the mouth he smokes with (`civilian.SMOKING_ROLES`); after every second or third puff the hand goes out to the side and the thumb flicks the ash off - three taps, the cigarette rolling in the fingers (`HumanoidPose.tilt_r`). And it smokes (`render/smoke.py`): a thin trail curls up off the lit end while it smoulders and stops while he draws on it; after the drag a lungful is blown out of his mouth in a puff that billows and thins; the cherry glows a dull red, flaring bright orange on the drag (`HumanoidPose.ember`, `render.items.cherry_glow`).
  - **Eating** is a bite at a time: the chopsticks' *tips* go down into the box to the next piece of sushi (or to the chow mein), pick it up (`chopsticks_sushi`), carry it up to the mouth - the sticks turning from pointing down into the box to pointing back at the lips (`HumanoidPose.aim_r`) - and the box holds one piece fewer each time (`sushi_open4..1`, `noodles4..0`) until it goes, empty, in the bin.
  - **Faces and looks** (`animation/look.py`, `animation/anatomy.py`, `assets/faces.py`): this is a city a century into organic cybernetics, and nobody looks like anybody else. Each person of the street is drawn from their seed (`street_look`): a hairstyle (`animation/hairdo.py`: every style grown on the skull's own surface, all of it in a hair texture whose strands run the way the hair grows, gathered into locks with dark partings between them - a short crop; a mohawk of fat gelled liberty spikes fanning back along a strip of hair, the sides shaved (nine spikes standing evenly from the forehead to the nape by the length over the skull - `hairdo.midline_along`: spaced by the skull's height, four used to bunch over the forehead and three down the back with a bare strip over the crown - the tallest over the crown, every one round and as slender for its length as the rest, each coming to the same sharp point: `hairdo.CREST`, `CREST_SLENDER`); gelled spikes all over with a few hanging over the forehead; a side shave - one side shaved to stubble, the rest falling over to the jaw on the other with a fringe swept across; a bob cut level at the jaw with a blunt fringe; long straight hair - a hime cut: the fringe, locks framing the face to the chin, the rest to the shoulders; a top knot, the sides shaved and the top slicked back into it; twin tails tied high with neon ties; a buzz cut; a bare scalp) in loud colours - the tips of the spikes and tails dyed, the underside of a bob or long hair a neon colour that shows as they move, eyes that have been replaced (a glowing iris, a chrome eye, a big optic lens - or organic ones), chrome (a jaw plate, ports at the temples, a cyber-ear with its LED, a neck port, a whole chrome arm), extras (a respirator, goggles, shades, earrings, a collar, shoulder pads), a tee, a vinyl jacket or a vest with neon trim, stripes down the legs, glowing soles - and a mark on the face (circuit lines, a tattoo, neon eye paint, a scar). The guards all wear the Halcyon uniform - armour, visor, sidearm - but each has their own haircut (a regulation cut - buzzed, cropped or shaved: nothing that would go through the floor when one goes down), rank stripes and visor red (`guard_look`; an elite's visor is gold). The decoration is drawn, not collided with (`rig.DECOR` parts are left out of the collision body). Heads are taller (3 cm more forehead) with a flatter back to the skull than the front (`anatomy`: one description the mesh, the rig and the face painter all use). **Faces** are painted per skin tone, mark and **expression** (`face_image`, made on demand and uploaded lazily: `render.textures.TextureMap`): neutral, smile, grimace, frown, talk, talk_o, mumble, shout, eat, pursed - shaded like faces (colour in the cheeks, shadow in the eye sockets, pores), the whites of the eyes and the teeth always white. The face matches the rest of the head seamlessly: its painting fades back, towards its edges, to the very flesh the head, ears, nose, neck and hands are drawn with (`faces.plain_skin`, `_feature_weight`), and that skin is drawn in each person's own tone the way the face is - the tone in the picture, lit white (`flesh_name`: a toned flesh texture made on demand) - so a lamp lights both alike (the tone used to be multiplied into the lighting on the head but not on the face, and the face showed as a different colour, with a darker rim at its sides). The face moves with what they are doing (`HumanoidPose.expression`): talking cycles syllable by syllable (`street.speech`), reading the menu is muttered, a bite opens the mouth wide and chewing works it, a drag on a cigarette purses the lips and the smoke comes out of a round mouth, a sip purses them, a kick is a grimace, an owner who finds the netrunner at home shouts, Dell frowns at a dead vending machine, a guard frowns when something is up and bares its teeth closing in. The chef and the stroller rest on a smile; everybody else on a neutral face.
  - **Getting about**: a queue forms behind whoever is already in it (`queue_spot`: one place further back along the counter per person ahead, taken on arrival) and moves up when it does; a customer with their box steps aside away from the queue and eats out of the way. Walking, people steer round each other, slide along a corner instead of pressing into it, walk round somebody who is standing in their way, stop short of a spot somebody is standing on (`BaseNPC.arrival == "taken"`), and - when two of them are in each other's way and getting nowhere - one steps back and lets the other by; somebody standing about (queueing, eating, smoking) shuffles a step aside for a person held up by them (`alley.make_way`). A walk only ends where it was going (`Walk` checks the distance: the router's "pinned, give up" counts as arrival, which had the office worker throwing his box at a bin ten metres off and vanishing a few steps up the alley), a body left inside a collider by a scripted step walks out of it, and nobody ever goes out of sight: leaving the alley, people walk home and let themselves in at their own front door (`alley.leave_and_return` - they used to pop out of existence at the alley's mouth, or at the METRO steps, and back into it a minute later).
  - Things are real objects: the knife lies on the board until the chef picks it up, the fish becomes sushi on the board, the box sits on the counter until the customer takes it, the parcel lies on the step; people hold what they are using (`HumanoidPose.hand_l` / `hand_r`, drawn by `render/items.py`), and the door of number 16 swings open and shut.
  - Places are data (`LevelData.places`: "sushi.chef", "sushi.board", "door16.stoop"...), as are the roles (`CivilianData.role`, `MerchantData.role`) and the loose things (`LevelData.items`); a role whose places are missing falls back to wandering.
- **The main street** past the alley mouth: a pavement to walk along the fronts of the blocks, the far side of the road scenery (a backdrop of towers with windows and big neon, shopfronts with signs and awnings, a monorail beam). See [The Main Street, the Homes & the Hours](#the-main-street-the-homes--the-hours). There is no winning and no walking off into a city that is not built yet: barricades close the street at both ends, a guard rail runs along the kerb.
- Lights in the alley (the lamps over doors 14, 16 and 17 hang out from the wall so they light the doors, not just the steps), six street lamps along the kerb (night only), one in every flat and two in number 16 - 24 in all; the renderer keeps the nearest within the preset's budget. The alley is kept gloomy but navigable.
- `tests/test_street.py` holds the level to the facility's rules (no invisible walls, every solid prop inside a collider, clutter that never clips), flood-fills the alley and the pavement from the spawn to the barricades, both counters and every waypoint with a player-sized body, and drives the crowd, the merchants and the hand-over; it checks there is no exit zone, no way past the barricades or onto the road, and that every flat's places are reachable for its owner. `tests/test_alley.py` runs five minutes of street life and checks every story happens in order (the chef's from the cooler to the counter, a customer's from the queue to the bin, the sushi eaten piece by piece, the parcel's from the dealer to the resident), that the knife is always exactly one place, that nobody walks through anybody, walks on the spot, vanishes or jumps from one place to another (and that those who leave the alley go home to their own flats), that boxes are binned from beside the bin, that no arm or parcel goes into the shut door of number 16, and that everything held can be drawn.

### Number 16

Dell's flat (`levels/apartment.py`, folded into the street level as data): a ground-floor one-bedroom nobody has maintained in thirty years, entered from the alley through the door of number 16.

```text
      z=35.6 ┌───────────────────────────────────────────┐
             │ fridge   KITCHEN (hotplate, sink)  counter │
      z=32.7 ├──────────────┬─────┐    opening    ┌─────┤
             │   BEDROOM    │ TV  │               │desk │
             │  mattress    │     LIVING ROOM     │     │  front door (z 29.9..31.1)
      z=29.1 ├────nightstand┤bed door                   ║  ── the alley, the step,
             │   BATHROOM   │bath door    coffee table  │     the access panel
             │ shower toilet│ camera   SOFA             │
      z=26.8 └──────────────┴──────────────────────────┘
           x=-10.8        x=-8.2                      x=-4.8
```

- **The door** is a real, hinged door of the level (`DoorData.hinge` / `swing`): locked from outside, it swings in as somebody walks up to it once unlocked, and from the inside it always opens (`DoorData.inside`, a latch). Dell locks it behind him. Like every front door in the street (the homes', the bar's) it is a little easier to walk through than it looks: the walls either side stop 12 cm short of the opening for anybody walking (`doorway.ease_doorways`: the drawn wall and frame are as they were) and the doorway's walkable strip is widened to match and reaches on into the rooms either side - and a body that still catches the jamb is eased round it towards the opening (`PlayerController._ease_round`) instead of stopping dead. It used to take a dead-centre approach: 20 cm off and the netrunner stuck on the frame (`tests/test_street.py` walks at every front door 25 and 40 cm off centre, in and out).
- **The access panel** beside it (`PanelData` -> `entities/access_panel.py`) is on the flat's own little network (`dells_home`): ping it `[F]`, and with it under the crosshair run **BYPASS LOCK** `[1]` - the door unlocks. The netrunner can see it from the alley. Every other front door in the street has an access panel beside it too (`flats.panel`, on the latch side under the door's number), on the tenement's one cheap building-wide lock system (`tenement`: TENEMENT DOORLOCKS): bypass one and that door is open to the netrunner - only number 16's counts for the job.
- **His kitchen** has a real fridge-freezer and microwave (`furnish.fridge`, `furnish.microwave`): a yellowed, dented fridge on a plinth with a freezer door over the fridge door, long chrome bar handles on stand-offs, a badge, magnets, his NUKA-KOLA sticker and rust coming through at its foot; a microwave on the counter with a door and its window, a handle, a control panel and a clock blinking 00:00 (never set).
- **The home camera** in the living room corner (`CameraData.watch_area = "flat16"`) sweeps the living room, but only reports somebody *inside* the flat - walking past the open door is not a crime. It can be SHUT DOWN or HIJACKED like any camera.
- **Dell** (the `resident` role in `ai/alley.py`) lives in it: cooking on the hotplate, slumped on the sofa flicking through the channels with the remote, typing at his computer, washing up, at the bathroom mirror, looking for a clean shirt (`HOME_CHORES`); he goes out now and then (the shopping, sushi to take home while the stall is open; a can from the vending machine in the alley when it is shut), answers the door to the courier and brings parcels in, and sleeps in his bed from 02:00 to 09:30 (asleep, he minds nothing). He only minds a stranger **inside** his flat - and only once he has *noticed* them: while he can see you his eyes fill the detection meter, but he goes on with what he was doing, only half aware of something, until the meter turns **red** (`DETECTION.red_level`, 50); then he drops everything, turns on you and shouts (captions: `Bark`). Out of his sight before it goes red, and he never noticed at all.
- **The vending machine** (NEON COLD, on its own network, across the alley from the junk stall): use it `[E]` or DISTRACT it remotely and it shorts out, banging and sparking - and then stays **broken** (`VendingMachine.broken`: the marquee dead but for a stutter, the display dim, the slot's lamp blinking red, the odd spark) until somebody kicks it back to life, or five minutes go by. That somebody is Dell - but only if he is **outside** and sees it: from inside his flat he knows nothing about it. Out on an errand, on any walk out of doors, it catches his eye (`alley.ASIDES`: the walk is put on hold) and over he goes: looks it over, tries the buttons, steps in and kicks it until it shudders back to life (`animation.street.kick`; the third kick fixes it), stares it down - and only then gets on with his errand. So the way to use it is to wait for him to go out, then short it: it keeps him out of his flat that much longer. Anybody else who wants a can finds it out of order, thumps it and goes without.
- **His computer** by the front door (a `home`-style terminal) has a note on it: the stick is not plugged in any more - "Moved it to the bedroom. Nightstand." The **data stick** lies on the crate he uses as a nightstand (`PickupData` -> `entities/pickup.py`; `[E]` takes it).
- **Caught inside** - by the camera or by Dell - you are thrown out (see [Being Caught](#being-caught)): onto the alley by the step, without the stick (it goes back on the nightstand), and the door is locked behind you. The quest goes on.


### The Main Street, the Homes & the Hours

**The main street** (`levels/main_street.py`, part of the street level): out of the alley's mouth a pavement runs left and right along the faces of the two tenement blocks - brick with rows of lit and dark windows (and each flat's own lit window, barred, by its front door), the Neon Moth just round the corner, front doors with their numbers and lamps, six street lamps along the kerb (lit at night), a guard rail along the kerb (the traffic lanes are not for walking), and the far side a backdrop of shops and towers. The road is two lanes and a parking strip wide (`PARKING_Z`, `MEDIAN`, `LANE_WEST`, `LANE_EAST`: traffic keeps to the right), lane markings along it, a kerbed median down the middle with the monorail's pillars on it (hazard-banded concrete, a steel cap under the beam) and the beam 7 m up, a strip of cyan light along each side. At either end the built city stops at a concrete **portal wall** (`TUNNEL_X`): the blocks run into it, and the road and the monorail dive into tunnels through mouths in it (hazard trim, an EXPRESSWAY sign, amber lamps down each lane and light strips along the walls, running on out of sight). At the west end the **UNDERPASS** leads down to the rest of the city and at the east end the **METRO** station's glass front goes up the block; barricades close the pavement before the portals (striped boards on trestles, a glowing red sign panel, a warning lamp).

**The car** parked at the near kerb (`main_street.car`) is a low cyberpunk coupe: a gunmetal-blue body with a glass canopy and windscreen, a chrome waistline, a neon sill strip, a headlight bar and a red tail strip, a glowing number plate, a rear wing and mirrors, and four wheels (tyre, rim, spokes, a glowing hub ring, under arches cut in the body) with a magenta underglow on the road. It stands on the road, inside one collider, and nothing in it z-fights (`render.audit`). It is the same coupe as the ones driving past (`world/vehicles.py`).

**The traffic** (`entities/traffic.py`, drawn by `render/traffic.py`; the lanes are `LevelData.lanes`): now and then a vehicle comes out of the distance in one of the tunnels and drives down its lane and off into the other tunnel - a **coupe** (eight paint jobs, neon sills, glowing hubs, an underglow), a yellow **taxi** (a chequer band down each side, a lit TAXI box on the roof) or a delivery **van** (a raked windscreen, an advertisement down its side under a neon stripe, an amber light bar), each model a list of parts in its own frame (`VehicleModel`: the body baked into one batch per paint job, each wheel drawn apart and turning with the distance driven - `wheel_turn`). Each has a speed it likes to drive at; nobody overtakes: a car that comes up behind a slower one eases off and follows it at a safe distance (the intelligent driver model: it brakes harder the faster it is closing, the brake lights come on, and it gets going again when the gap opens), and a vehicle only comes out of a tunnel when the lane is clear there. The headlights throw a pool of light on the wet road ahead, the tail lights a red glow behind. The road is busiest in the evening rush and quietest in the small hours (`Traffic.interval`), and busy already when the netrunner first looks at it (the traffic runs for a minute before the street is first seen). `tests/test_traffic.py`.

**The homes** (`levels/flats.py`, furnished with `levels/furnish.py`): every person of the street has a home - a ground-floor flat behind their own hinged front door, built from a `FlatPlan`. Their owner unlocks the door, goes in, locks it behind them and lives there: cooking, watching TV, working at the computer, washing up, getting dressed, and sleeping in the bed (`alley.Sleep`: lying on it, the body laid flat on the cover by the rag doll's `lying_on`, at the sleeper's own size, the top of the head on the pillow - the bed's place is where the crown goes - so a tall sleeper's feet reach further down a 2.2 m bed and nobody's head goes through the wall behind it). `tests/test_homes.py` plays every chore of every home through its whole loop and checks no part of the body goes into a wall or into furniture it is not using (the wardrobe chore stood so close to the bathroom wall that the reaching arm went through it), and that everybody fits their bed.

**Furnished like homes** (`levels/furnish.py`): a front room with a kitchenette - a counter of cupboards with doors, drawers and handles, a sink let into the worktop (a stainless basin you can see down into, with a drain and a tap over it), the hob with a pot on it and an extractor hood over it, wall cupboards, a tiled splashback, a pantry shelf, the fridge with everybody's notes on its door and the cereal on top of it, and food about (a loaf, a kettle, noodle cups, a fruit bowl, jars, bottles, takeaway boxes); a sofa with its cushions and a throw facing the TV (a flat screen on a stand, or a deep old tube set with its aerial), a coffee table, a rug, a standard lamp, a lampshade over the light; a desk with its chair and computer (two monitors, or a laptop) under **a window** - frame, cross bars, sill, venetian blinds, a roller blind or curtains - that looks out on the city or the alley (a screen the renderer keeps up with the time of day: lit towers and rain on the glass by night, grey towers under a smog-brown sky by day), and the same window lit and barred from the street. A bedroom with its bed, the nightstand with a lamp and an alarm clock, a wardrobe with its doors, a shelf of books, a picture over the bed, a rug. A bathroom tiled to shoulder height: **a toilet** with its cistern and flush button, a flared pedestal, the bowl with its seat and the water in it and the lid up; a basin on a pedestal with its tap, a mirror (on a mirror cabinet in some) and a glass shelf of toiletries; a shower with its curtain on a rail, a towel on a rail, the loo roll, a bath mat. Everything obeys the levels' rules: no two faces in one plane (`render.audit` - zero hits), solid furniture inside colliders, decorations `transparent`.

**No two alike**: each flat is done up in a style of its own (`flats.STYLES`, one per flat): the walls (faded regency stripes, painted breeze block, wood panelling, lilac plaster, off-white plaster, a seventies geometric paper, an old damask), the floor (parquet, concrete, carpet, chequered lino, blue lino, dark boards), the bathroom tiles (subway, hexagon, old green, beige, blue), the sofa (mustard, black leather, olive, tan leather, grey, red, oxblood), the window's blinds or curtains, the TV, the rug, the lampshade, the towels - all new procedural textures (`assets/homes.py`: wallpapers, floors, tiles, posters, prints, photos, a calendar, a certificate, a tool board, rugs, book spines, a cereal box, fridge notes, window views). And each has the things of whoever lives there (`flats._LIVES`): Nix's band poster, dartboard, records, cartons of cigarettes and ashtrays; the junk dealer's steel shelving crammed with salvage (dead decks with their lights still on, coils of cable, an old monitor); the tinkerer's tool board, parts bins, soldering station and the little robot he made; the stroller's bookcase, plants, prints and fairy lights; the office worker's certificate, binders, calendar and the family photo on his desk; the courier's map of the district with the routes marked, parcels by the door, bike helmet and energy drinks; the sushi chef's fish chart, cookbooks, knives on a magnetic strip and a lucky cat. Number 16 is still Dell's grim little flat - now with a sink full of dishes, cupboards, a real toilet under a cracked mirror, shelves of old games, last year's calendar, pizza boxes, a photo by the bed, and a grimy window with bent blinds onto the alley. `tests/test_furnishings.py` holds every home to it.

**Television** (`ai/alley.py`: `TvSet`, `WatchTv`; `render/television.py`; the channels in `assets/homes.py`): every home has a set. Sitting down on the sofa, the owner picks up the remote and points it at the set: on it comes, and they flick through the channels (`CHANNEL_LINEUP`) - past the dead ones, whose snow crawls and rolls like real snow (a fresh frame at 18 a second, a dark bar rolling down it, a tear sliding across), with a moment of snow as the set retunes after every click - until something holds them: **the news** (NCN: an anchor whose mouth moves, a live picture of the arcology burning, a ticker running along the bottom), **cartoons** (a yellow duck in a cape chased over the hills by a grumpy robot) or **the music channel**; the adverts do not. A while later they may flick on again. The set stays on while they potter about and goes off when they go to bed or go out. The screens are props tagged `screen` / `tv:<home>` that the renderer leaves out of the cached geometry and draws live each frame with the frame of the moment (frozen while paused); the sets are saved with the game. `tests/test_television.py`.

| Number | Where | Who |
| :--- | :--- | :--- |
| 16 | the alley's west wall, by the dumpster | Dell (the resident) - see [Number 16](#number-16) |
| 17 | the alley's west wall, across from the Halcyon door | Nix (the smoker) - across from where he smokes |
| 31, 33, 35 | the main street, left of the mouth | the junk dealer, the tinkerer, the stroller |
| 32, 34, 36 | the main street, right of the mouth | the office worker, the courier, the sushi chef |

**The hours** (`ai/schedule.py`: `Schedule(home, sleep, work)` per role, by the `WorldClock`):

| Who | At home | Asleep | Otherwise |
| :--- | :--- | :--- | :--- |
| sushi chef | 02:00 - 14:00 | 03:00 - 12:30 | at the stall (open 14:00 - 02:00) |
| junk dealer | 00:00 - 10:00 | 01:00 - 09:00 | at the stall (open 10:00 - 00:00) |
| office worker | 23:00 - 07:30 | 23:30 - 07:00 | working from home 07:30 - 17:00 (at the computer in his front room); after five at the stalls - the evening rush |
| tinkerer | 23:30 - 11:00 | 00:30 - 10:00 | browsing the junk, eating |
| courier | 22:00 - 09:00 | 23:00 - 08:00 | on the parcel round |
| stroller | 02:30 - 12:00 | 03:30 - 11:00 | wandering, chatting to Nix |
| Nix | 05:00 - 06:30, 13:30 - 14:15 | 05:06 - 06:24 | by his wall, ready to talk business |
| Dell | home, but for his errands | 02:00 - 09:30 | out for sushi, or to the vending machine |
| the bartender | across town (he comes and goes by metro) | - | at the Neon Moth: in after four, open 17:00 - 03:00 |
| a regular | across town (by metro) | - | drinking at the Neon Moth 20:30 - 00:30 |

A stall shuts when its merchant goes home (a little A-frame CLOSED sign goes up on the counter, facing the alley; a cover goes over the junk table; `AlleyLife.open`) - a customer already served is served first - and opens when they come back; nobody queues at a shut stall. **The vending machine** in the alley (NEON COLD) is where people get a can when the food is shut - and, shorted out by the netrunner, it stays broken until Dell, out on an errand, sees it and kicks it back to life (see [Number 16](#number-16)). Between errands everybody goes home, to their own flat, never out of sight (`alley.leave_and_return`). The clock, the weather, the stalls, who is asleep, the televisions and a broken vending machine are saved with the game.

`tests/test_schedule.py` runs the street's day with the clock set or fast-forwarded: the clock and the daylight, the hours each keeps (Nix almost always out), a home for everybody, the chef shutting the sushi bar at two and opening it at two, the junk dealer covering his table at midnight, nobody queueing at a shut stall, everybody asleep in their own bed at four in the morning (and asleep, not noticing a stranger), the office worker working from home, Dell at home knowing nothing of a shorted vending machine (which stays broken), out on an errand seeing it, kicking it back to life (his boot on its front, not through it) and only then getting on with his errand, nobody else fixing it, Dell getting his supper from it when the stalls are shut, and the day a little brighter and browner out of doors only.

### The Neon Moth & the Metro

**The Neon Moth** (`levels/bar.py`) is a bar on the main street just round the corner from the alley, behind its own brick front: big windows of real glass (you can see in) with NEON MOTH BAR hanging in one and an OPEN sign in the other (lit only while the bar is open: a screen prop tagged `open:bar`), the name over the door in neon (`assets/signs.py`: neon lettering laid out for the board it goes on), and a hinged door with a porthole. Inside: a chequered floor, plum walls over wood panelling with a brass dado rail, neon along the tops of the walls, a long panelled bar down the right with a steel top, a pink neon strip under its lip, a brass foot rail, beer taps, a drip tray, a till, beer mats and a bowl of snacks; four red stools along it; behind it the back bar - cupboards, a bottle fridge, a glass-washer sink let into its top, glass shelves of bottles lit from behind; two round tables with two chairs each and a candle; a jukebox in the corner, a dartboard, posters, a neon moth on the wall, lamps over the bar and a television up on the back wall for the customers (the street's `TvSet` machinery: `tv:bar`).

**The metro** (`levels/metro.py`, `entities/metro.py`, drawn by `render/metro.py`): the station is the glass front the height of the east block behind the METRO doorway. Inside it: ticket gates across the hall just inside the doorway (NO PASS NO ENTRY - the netrunner has no pass: the gate line is as far as you get, unless you hack your way through - see below), a long straight stair up the back wall to a landing, and a covered footbridge from the landing out over the pavement to the **platform** over the road: a deck on columns in the parking strip, a yellow edge, a glass balustrade, a canopy with lamps, benches, the station's name (NEON DISTRICT, line 3), a departures board. Every couple of minutes (less often in the small hours) a two-car monorail train comes out of a tunnel, runs along the beam, slows into the station and stops with all four doors on the platform; its door leaves slide open, it waits (longer for anybody still on their way to a door), closes up and pulls away into the tunnel at the other end - and the next one comes the other way. The gates' flaps fold back for whoever has a pass.

**Hacking the gates** (`levels/metro.py`: `GATE_DOORS`, `WALK_REGIONS`, `FLIGHT`, `DECKS`): each gate lane's flaps are a lock on the station's own network (**METRO TRANSIT**, `GATE_FACTION`): ping it `[F]` from the doorway and **FORCE OPEN** it `[1]`. A forced gate's flaps fold back for whoever walks up to it (`DoorData.approach_range`) - and a minute later, once nobody is in the way, the station's system sets it again (`DoorData.relock_after`). Through it the netrunner can walk the whole station: the paid side of the hall (and under the landing), **up the stair a tread at a time** (`world.surfaces.Flight`: the feet on the tread underfoot all the way up; `LevelData.flights`), along the landing, over the footbridge and out onto the **platform** to wait for the train with the regulars - the raised floors are `LevelData.decks` (`ElevationState.deck_y`: up on them once near the top of the stair, on the ground under them otherwise). Nobody steps off the edge: the stair's open side is closed by its stringer and balustrade all the way down (no stepping off it, no walking in under it), the landing and the footbridge by their balustrades and sides, the platform by its edge and its ends (the walk regions' insets); benches, the bin, the canopy posts, the departures board and the advert screen are solid up there. The crowd still never wanders in (`metro.NAV_BLOCKERS`): only the commuters go up, by script.

**Nobody appears or disappears** (`ai/nightlife.py`): the people who come and go by metro walk the whole way - along the street to the station, a pass to a gate's reader and through, up the stair, over the bridge, a wait on the platform (a phone, a look down the line) - and step in through a door, through its vestibule and on into the dark of the car (the back of each vestibule is the dark inside of the car: `vehicles.TRAIN_DARKNESS`), gone from sight - nobody stands in the doorway facing out waiting for the doors to shut. The stair is climbed (and come down) a step at a time (`alley.Stairs`, `animation.street.stairs`): a foot on every tread in turn, the other lifted over the edge and put down two treads on, the knee coming up with it, the body leaning into the climb (upright and watching the steps on the way down), the arms swinging; the feet are worked out by leg IK to stand on the treads - no boot or shin ever goes into a step; the train takes them away down the tunnel (`StreetPerson.riding`: aboard, out of sight). Coming back, they come out of the dark of the car as its doors open, through the vestibule onto the platform, and walk all the way down again. A rider is still aboard after a save and a load.

* **The bartender** lives across town: he comes in on a train after four, walks down through the station and along the street, lets himself in and sets up; at five he opens - the door unlocked (it opens for whoever walks up to it), the television on. Until three in the morning he works the bar: pulls a pint at the taps for whoever orders one and sets it down in front of them (walking it out to the tables), collects the empties and washes them at the back bar's sink, and in between wipes down the bar, polishes glasses, has a word with whoever is sat at it, rings up the till, looks up at the television (changing the channel if nothing is on). At three it is last orders: what was ordered is served, nobody new comes in, and once the last customer has gone he clears up, switches the television off, goes out, locks up and walks back to the station for a train home.
* **A regular** comes in on the metro after half past eight for a few pints at the bar - and goes home on the metro after midnight.
* **The people of the street** drop in now and then (the office worker after his sushi, the tinkerer after his noodles, the stroller on his rounds): in through the door, a stool at the bar (a chair at a table if the bar is full), a pint ordered, waited for (looking up at the television), taken up - "cheers" - and drunk a sip at a time (`street.drink`, sat on the stool), a word or a look at the television between sips; the empty glass left on the counter for the bartender, and out again.

`tests/test_nightlife.py`: the bar furnished like a bar, its door open to the netrunner only while the bar is open, the OPEN sign lit only then; the station's gates stopping the netrunner - until one is forced open (the scanner finds it; through it, up the stair a tread at a time, over the bridge onto the platform and back down; never off the edge of the landing, the bridge or the platform; under the landing on the hall floor; the gate setting itself again a minute later; the crowd still routed round the paid side), the stair rising to the platform, the platform level with the train and all its doors on the platform; the stair climbed a step at a time (no foot ever in a step, always a foot on a tread, the legs taking turns); trains stopping, opening and shutting their doors, holding them for a late passenger, one way then the other, fewer in the small hours, and drawn on the beam with their doors sliding open; an evening of the bar - the bartender serving whoever comes in, in order, the regular coming in by metro; riders walking into the dark of the car and only going out of sight in there - never waiting in a doorway facing out (and coming out of it); the bar closing at three and the bartender going home by metro, coming back the next afternoon and opening at five; a rider still aboard after a save.

### The Weather

The city has weather (`systems/weather.py`, drawn by `render/weather.py` and the scene), and like the clock it goes on whichever level the netrunner is in - it only shows out of doors (`LevelData.outdoor`). It moves in game time, slowly:

* **Rain** comes in spells - a dry spell, a **drizzle**, steady **rain**, a **downpour**, a **thunderstorm** - each lasting a while and followed by another (`SPELLS`, `FOLLOWS`: a Markov chain - a drizzle often thickens into rain, a storm blows over into rain or a downpour, a dry spell usually ends in a drizzle). The rain never jumps: a drizzle fades in over a quarter of an hour, a storm front comes in faster, and the rain dies away slowly when a spell ends. It falls as streaks round the eye (as many as the quality preset allows: `GraphicsSettings.rain_drops`), anchored in the world, slanting a little in the wind, with little crowns of splash where it lands - only where there is sky overhead (`LevelData.open_sky`: the alley, the pavement, the road), never under a roof (`LevelData.shelters`: the stalls' awnings, the shopfronts' awnings, the metro's platform canopy and footbridge, the parked car). From a flat you see it out of the window and the door.
* **Puddles** fill while it rains and dry out slowly after (faster by day). They gather on the open ground (`puddle_sites`: the alley, the pavement, the road either side of the median), clear of everything standing there, each with a depth of its own - the deepest dips fill first and are the last to dry: dark, glossy water that catches the street lamps, with the neon in it and a wet stain round it; rings spread across them while it rains. The street starts the night damp.
* **Thunderstorms**: lightning every few seconds to half a minute - only in a storm, and only while the rain is heavy - a sharp flicker of two or three strokes that lights up the whole street and the sky.
* **Fog** has its own spells (clear, a haze, a mist, a real fog) - likelier at night and towards dawn, and after rain - and fades in and out over hours. It fades the street into a murk the colour of the night (or the smog of the day), neon and screens glowing through it further than anything else, and limits how far can be seen (`visibility`): the far plane comes in to match (`weather_light`, never nearer than `MIN_SIGHT`), so the end of the street is simply lost in it. A downpour thickens the air too; heavy rain clouds dim the day. In a room off the street there is less of it. (The fixed-function backend has no fog: its fog was worked out per vertex, and a long wall went grey right in front of the eye.)
* **Spray**: a car driving through a puddle on the road throws water up off its wheels - the faster, the more.

The weather is saved with the game. `tests/test_weather.py`: the same weather for the same seed, a drizzle fading in slowly, every kind of spell coming round with the rain never jumping, puddles filling in the rain and drying after (faster by day), puddles only on open ground clear of everything, deepest first, lightning only in a storm, fog coming in over hours and limiting the view (thinner indoors, none in the facility), a flash lighting up the street and the sky, cars throwing spray only through water and only at speed, the rain only under the open sky and never under a roof, and the weather saved and loaded.

---

## Inventory, Cyber Mods, Quests & Dialogue

### The [Tab] screen

`[Tab]` opens the netrunner's deck over a frozen world: two tabs, **INVENTORY** and **QUESTS** (`[` and `]` switch; the tab last looked at is the one that opens next time). Everything on it works with the keys and with the mouse (click, wheel). It is laid out from the window size every frame, so it fills a 1024x768 window and a fullscreen desktop alike.

- **INVENTORY**: a grid of slots (five across) with a neon icon of each item (`assets/icons.py`, generated with NumPy like the textures) and its name underneath - wrapped to two lines and cut short with an ellipsis, never wider than its slot. A stack shows its count (`x140`), an installed mod an `E` badge. A long inventory scrolls (wheel, `[PgUp]`/`[PgDn]`, or moving the selection past the edge) with a scroll bar; only whole rows are clickable. On the right the selected item's details: a large icon, its kind, the body slot a mod goes in, what it is worth, what it does and a description - and for a cyber mod an **EQUIP / UNEQUIP** button (`[Enter]`, `[E]` or a click).
- **QUESTS**: the quest log as a scrolling list on the left (ACTIVE / COMPLETE, who gave it); on the right the selected quest's current objective, **pictures** of what it is about - rendered from the game world itself, from a camera placed for each picture (`QuestDef.pictures`: the door of number 16 with its access panel, the data stick on the nightstand; a picture of another level than the one you are in is taken from a stand-in simulation of that level) - and its details: the brief and a timestamped log, in a panel that scrolls (wheel, `[PgUp]`/`[PgDn]`, `[A]`/`[D]`).

### Items and who carries them

`core/inventory.py`: `ItemDef` (the catalogue, `ITEMS`) and `Inventory` (stacks in the order they were picked up, a capacity, `add` / `remove` / `count` / `has`, `equip` / `unequip`, and `stat(name)` - the product of every installed mod's multiplier for that stat). **Everybody has one**: the player (`Player.inventory`), every guard (a Halcyon badge, eddies, a stim) and every person of the street by role (`NPC_START`: Nix has his cigarettes, phone - and the Wraith damper he pays with; Dell has the keys to number 16). Stolen data goes in as `DATA SHARD`s too (`Player.add_intel`), and keyed doors still read it off the deck.

| Cyber mod | Slot | Installed |
| :--- | :--- | :--- |
| **Sonaris Ping Amplifier Mk.II** (start) | CORTEX | `PING_RANGE` x1.6: the scanner ping reaches 60 % further (`Scanner.pulse_from(range_scale=...)`) |
| **Zetatech Quickjack Link** (start) | HANDS | `CLOSEUP_HACK_SPEED` x1.65: everything hacked from arm's length goes 65 % faster - establishing the link at a desk, the terminal interface's actions, and a quickhack on a target within 3 m (`hacking.CLOSE_RANGE`) |
| **Militech Wraith Signal Damper** (Nix's reward) | NERVOUS SYSTEM | `DETECTION_GAIN` x0.65: cameras and eyes fill the detection meter 35 % slower (`DetectionSystem.gain_scale`) |

The two starting mods do nothing until they are installed: open the inventory, pick one, EQUIP.

### Talking to people

`[E]` at somebody in front of you (within 2.2 m) who has something to say starts a conversation (`core/dialogue.py`: `Conversation` of `Line`s): captions along the bottom of the screen with the speaker's name, typed out a letter at a time. `[E]` shows the rest of the line at once, `[E]` again moves on - and left alone, each line moves on by itself after long enough to read it. The person you talk to stops what they are doing and faces you (`BaseNPC.talking_to`) until it is over, or until you walk away. Remarks shouted in passing (Dell finding a stranger in his flat) are `Bark`s: the same captions, no reply.

### The first quest: "Something Special"

`systems/quests.py` holds the quests as data (`QuestDef`: title, giver, location, description, pictures), the log (`QuestLog` of `QuestState`s: flags, a timestamped journal, done) and the `QuestDirector` that drives them from what happens in the world (who says what, and when a step is done). A notice comes up on every step (`NEW QUEST: SOMETHING SPECIAL`, `QUEST COMPLETE`).

1. **Nix**, the smoker by the wall near the Halcyon door, has a job for somebody with a deck: Dell in number 16 has been skimming off the wrong people and keeps it on a data stick. Bring it to Nix and he will give you *something special*.
2. Ping the access panel beside number 16's door `[F]` and **BYPASS LOCK** `[1]`.
3. Get inside - mind the camera, and Dell (he goes out now and then).
4. Dell's computer by the door has a note: the stick is on the bedroom nightstand. Take it `[E]`.
5. Back to Nix: he takes the stick and hands over the **Militech Wraith** damper (it moves from his inventory to yours). Talk to him again and he tells you to get lost.

Caught inside, you are thrown out and the stick goes back on the nightstand; the quest goes on, its objective back to finding the stick.

---

## Being Caught

There is no winning - and being caught does not always end the run. When the detection meter fills, a red **YOU WERE CAUGHT** screen comes up over the frozen moment, saying who spotted you and what happens now; then it fades to black, and the picture fades back in on the outcome (`config.SIMULATION`: `caught_hold` 3 s, `caught_fade` 1.5 s, `evict_hold` 2.2 s, `fade_in` 1.2 s). The same black takes you out of the biodome: at its outer staff door the netrunner stops, the picture fades to black (`level_fade_out` 1 s - nothing you press does anything, and nobody catches you on the threshold), the alley is loaded behind it, and the picture fades back in on the alley (`level_fade_in` 1.4 s; `Simulation.screen_black`). What it costs is the level's call (`LevelData.on_caught`):

- **In the facility** (`"restart"`): *the run is over - a new game begins*. A new game is a new game: back at the facility's spawn with the starting inventory (the mods uninstalled), no quests, nothing breached (`Simulation.new_game`).
- **In somebody's home** (`"evict"`, a `PrivateArea`: number 16, or any of the other flats - each owner minds who is in theirs while they are home and awake): *you're being thrown out of number 16*. Out on the alley by the step, without anything of the owner's you took from inside (`PrivateArea.confiscate`: the data stick goes back on the nightstand), the door locked behind you, the detection meter empty. Everything else - the level, your inventory, the quest - carries on (`Simulation.evict`).

Out in the alley and on the street nobody reports you: the owners and Dell's camera only mind a stranger *inside* their homes.

---

## Start Menu, Saves & Settings

`game.py` (`Game`) is the game around the simulation - no pygame, no OpenGL, so it is tested headlessly like the rest (`tests/test_game.py`). `main.py` turns keys and the mouse into `InputState`s and draws whatever `Game.mode` says is on screen (`TITLE`, `PLAYING`, `PAUSED`, `SCREEN`, `LOAD`); the app-level things the game cannot do itself (go fullscreen, apply a graphics preset, quit) come back to it as requests.

- **The start menu** opens over the night market going about its business, seen from a slowly drifting camera: **CONTINUE** (the newest save; greyed out while there is none), **NEW GAME**, **LOAD GAME**, **GRAPHICS**, **DISPLAY**, **QUIT TO DESKTOP**.
- **The pause menu** (`[Esc]`): **RESUME**, **SAVE GAME**, **LOAD GAME**, **GRAPHICS**, **DISPLAY**, **NEW GAME**, **MAIN MENU**, **QUIT TO DESKTOP**. A paused game consumes no time at all.
- **Saving** writes a new file every time, named by the date and time - `~/Unprepared/Saves/save_2031-04-02_23-15-07.json` (a second save in the same second gets `_2`): **a save never overwrites another**. There is no saving on the red screen - no saving your way out of being caught. **LOAD GAME** lists them newest first with the level, the time played, the time of day in the city and where you were (`INSIDE NUMBER 16`, `ON THE STREET`...).
- **What is saved** (`systems/savegame.py`, `SAVE_VERSION` 1): the level; the netrunner (position, heading, crouch, inventory with what is installed, stolen intel); every door (open, unlocked), terminal (link progress, breached, data stolen, doors overridden, fried), camera (shut-down countdown), vending machine, access panel and thing to pick up; every guard (where, its routine, knocked out or blinded); the people of the street (where they are, what they carry) and the things lying about (the knife on the board); the networks breached, the scanner's reveals and the hack log; the quests; the detection meter; the time of day (`WorldClock`) and which stalls are open. Deliberately not: a conversation or a terminal session in progress, a half-done upload, a guard mid-investigation (it goes back to its routine) and the street's errands in progress (each person starts their story again from where they stand - somebody asleep gets up and, if it is still night, goes back to bed). *Anything new a run can change belongs in `savegame.py`, with a round-trip test in `tests/test_savegame.py`.*
- **Settings**: fullscreen (at the desktop's resolution) or a 1024x768 window, and the graphics preset, are kept in `~/Unprepared/settings.json` and applied at start-up; both can be changed from either menu and take effect at once. Switching the display mode closes the window and opens it again with a fresh renderer (`render.scene.close_window`: SDL cannot reliably switch an OpenGL window in place, and the old context's display lists must be forgotten with it). `--fullscreen` / `--no-fullscreen` and `--graphics` override the file for that run. `CYBERDUKE_USER_DIR` moves the whole `Unprepared` folder (the tests point it at a temporary one).

---

## Project Architecture

The code base is an installable package (`src/cyberduke`) with a hard layering rule:
**everything that touches OpenGL lives in `render/`; nothing else imports OpenGL or pygame** (bar `main.py`, the one place that reads the keyboard and owns the window).
That rule is enforced by `tests/test_architecture.py`, and it is what lets the entire
simulation run headlessly under `pytest`.

```text
src/cyberduke/
├── main.py              # Window (windowed / fullscreen), input mapping, frame loop (pygame)
├── game.py              # Game: start menu, pause, [Tab] screen, save/load, settings - no pygame/GL
├── simulation.py        # Simulation + InputState: the whole game step, no pygame/GL
├── config.py            # Every tunable constant (player, NPC, detection, timing)
├── core/                # geometry, interp, clock, state, events (EventBus), menus, inventory,
│                        #   dialogue, the [Tab] screen's state, user files (~/Unprepared)
├── world/               # Level data model (props, colliders, walk regions, lights, doors) and the live Room
├── levels/              # Shipped levels as typed data (facility.py: ~1,600 props, ~180 colliders, 8 lights, entities;
│                        #   street.py + main_street.py: ~1,700 props, ~350 colliders, 24 lights; apartment.py; flats.py; doorway.py)
├── entities/            # Player + PlayerController, Terminal, SecurityCamera, VendingMachine, AccessPanel, Pickup
├── ai/                  # Pathfinder (A* + string-pulling), BaseNPC, fsm (State/StateMachine), CyberGuard,
│                        #   civilians, street life (alley.py) and the hours they keep (schedule.py)
├── animation/           # HumanoidPose, procedural clips, the body (rig), hands and grips, the knockout rag doll
├── systems/             # DetectionSystem, Scanner, HackingSystem (quickhacks + networks), quests, savegame
├── assets/              # Procedural texture and icon generators (NumPy only)
└── render/              # ALL OpenGL: primitives, gl_state, matrices, textures, lighting, humanoid,
                         #   sectors, entity_renderers, effects, beams, projection, hud, ui, screens, icons, scene,
                         #   smoke (cigarette smoke), audit (the z-fighting finder), shapes (the
                         #   rounded body parts), batch (a person in one draw)
tests/                   # pytest suite (unit + opt-in Xvfb boot smoke test)
play.py                  # no-install launcher (double-clickable)
run_tests.sh             # lint + format check + tests (+ --smoke)
```

### Module Breakdown

| Package / Module | Primary Responsibility |
| :--- | :--- |
| `simulation.py` | `Simulation` owns the room, player, entities (terminals, cameras, vending machines, access panels, pickups), NPCs (guards, civilians, merchants), detection system, quests (`quests`, `director`), the conversation and captions, and the `PLAYING / CAUGHT` state machine (there is no winning). `update(dt, InputState)` is the one entry point; `dt` is clamped to `SIMULATION.max_dt`, and a `paused` simulation consumes no time at all. Caught: the red screen, the fade, then `new_game()` or `evict()` by `LevelData.on_caught`, and a fade back in (`fade_in`). `[E]` is `talk_target` / `pickup_in_reach` / `prompt`. `load_level` swaps the world in place (the exit zone of a level with `exit_to` does this, behind a fade to black and back - `leaving`, `screen_black`; `level_generation` tells renderers), `reset` restarts the current level, `new_game` the whole run. |
| `game.py` | `Game` and its `Mode`s (`TITLE`, `PLAYING`, `PAUSED`, `SCREEN`, `LOAD`): the start menu over the drifting street (`title_view`), the pause and load menus (keys and mouse hotspots), the [Tab] screen (`toggle_equipped`), `save()` / `load()` / continue, and `Settings`; returns requests (`"display"`, `"graphics"`, `"quit"`) for `main.py`. |
| `core/inventory.py` | `ItemDef` / `ITEMS` (the item catalogue: cyber mods with their `slot` and stat `effects`, quest items, money, gear, junk), `ItemStack`, `Inventory` (`add` / `remove` / `count` / `has` / `find`, `equip` / `unequip` / `toggle_equipped`, `stat()`, `to_dict` / `from_dict`), the stats (`PING_RANGE`, `CLOSEUP_HACK_SPEED`, `DETECTION_GAIN`) and what everybody starts with (`PLAYER_START`, `NPC_START`). |
| `core/dialogue.py` | `Line`, `Conversation` (typed-out captions, `advance()` on `[E]`, lines that move on by themselves, `on_done`, `partner`) and `Bark` (a remark, no reply). |
| `core/screens.py` | `TabScreen`: the [Tab] screen's state (tab - kept between openings -, selected slot and grid scroll, selected quest and list/detail scroll) and `Hotspot`s the renderer leaves for the next frame's clicks. |
| `core/userdata.py` | `user_dir()` (`~/Unprepared` or `$CYBERDUKE_USER_DIR`), `saves_dir()`, `Settings` (fullscreen, graphics) with `load_settings` / `save_settings`. |
| `systems/quests.py` | `QuestDef` (with `Picture`s: a camera eye and target in a level), `QuestState`, `QuestLog`, and the `QuestDirector` that runs "Something Special": who says what (`conversation_for`), the steps (`update`: the lock released, inside, the note read, the stick found), the objective text, and what an eviction does to it. |
| `systems/savegame.py` | `snapshot` / `restore` of a whole run, `save_game` (a new date-and-time-named file every time), `list_saves` (`SaveInfo`, newest first), `load_game`, `SAVE_VERSION`, `SaveError`. |
| `systems/weather.py` | The city's weather (`Weather`, kept from level to level like the clock): spells of rain (`SPELLS`, a Markov chain `FOLLOWS`: dry, drizzle, rain, downpour, thunderstorm) the rain eases into, the wet the puddles fill with (`PuddleSite`, `puddle_sites`), lightning in a storm (`flash`), fog spells fading over hours (`FOG_SPELLS`; `fog_density`, `visibility`), the spray off cars' wheels in a puddle (`spray`, `Drop`); saved with the game. |
| `entities/access_panel.py` | `AccessPanel`: a door's keypad as a hack target (`target_kind = "panel"`, BYPASS LOCK -> `bypass()` unlocks its door). |
| `entities/pickup.py` | `Pickup`: an item lying in the world that `[E]` takes (reach 1.7 m, within 50 degrees of where you look); `restore()` puts it back. |
| `entities/traffic.py` | `Traffic`, `Car`: vehicles out of one tunnel and into the other along `LevelData.lanes`, following the car in front (the intelligent driver model), busier in the rush hours. |
| `entities/metro.py` | `Metro`: the train's timetable and run (away in a tunnel, arriving, stopped with its doors opening and shutting, departing), who is boarding and aboard, the ticket gates. |
| `world/vehicles.py` | The vehicles as parts in their own frame: the coupe, the taxi and the van (`MODELS`, `VehicleModel`, paint jobs, wheels apart from the body), and the metro's car (`train_car`, its doors). |
| `levels/apartment.py` | Number 16 as data: props and colliders, walk regions, the hinged door, the access panel, the home camera, Dell's computer with its note, the data stick, the `PrivateArea`, the resident's places and routes, two lights. |
| `animation/rig.py` | The humanoid body, part by part, written once: `build_group(builder, pose, group, kit)` emits every part as `part(name, size, shape, material)` - the renderer draws a rounded shape in the box, the collision skeleton (`animation/skeleton.py`) takes the box. `BodyKit` (armour, visor, holster, fingers), `HEAD_SIZE`, `VISOR_BOX`, `FACE_PROUD`. |
| `animation/hands.py` | Hands: `Grip` (finger and thumb curls, fan, thumb, which way the palm faces, where a held thing sits in the hand, an up-axis for upright things), `GRIPS`, `GRIP_FOR_ITEM`, `grip_name` (what is held decides; an empty hand takes the pose's `grip_l/r`), `hand_frame` (the wrist bent so the contact point lands on the arm IK's hand point, the hand turned about the forearm), `held_axis` (a can turns with the hand), `outward` (a cigarette's direction). |
| `render/shapes.py` | The rounded unit shapes (NumPy meshes filling the unit cube): lofts of superellipse sections with tapering profiles and rounded ends - head, face, hair cap, visor band, nose, ears, neck, torso, pelvis, limbs, spheres, palm, finger bones, boot - each also in a coarser version for people further off. |
| `render/batch.py` | `BatchContext`: a `DrawContext` that collects a person's parts as instances and, on `flush`, transforms every vertex by its part's matrix in one NumPy pass (the meshes' layout cached per set of parts) and hands the lot to the backend's `draw_batch` - one upload, a draw per texture. |
| `core/clock.py` | `clamp_dt` (frame-hitch cap) and `WorldClock`: the time of day in minutes (`day`, `rate` = game minutes per real second from `config.CLOCK`, `advance`, `set`, `hour`, `text` "HH:MM", `is_between`, `daylight()` 0..1 with dawn 05:30-08:00 and dusk 17:30-20:00); `within(hour, from, to)` for spans past midnight. Saved with the game. |
| `ai/schedule.py` | `Schedule(home, sleep, work)` per role (`SCHEDULES`): `at_home`, `asleep`, `working` (from home), `indoors`, `out`; `stall_hours_open(stall, hour)`. The stories in `ai/alley.py` ask it when to go home (`keep_hours`, `go_home`, `stay_home`), when to sleep (`sleep_in_bed`) and when a stall shuts. |
| `levels/flats.py` | `FlatPlan` (number, owner, facing, where on the face, door, style) and what it builds: the shell (brick front, papered or plastered walls, party walls, floor, ceiling - no two surfaces in one plane), the window (inside and out), the hinged front door, the kitchen, front room, desk, bedroom and bathroom, each flat's `Style` (`STYLES`) and its owner's things (`_LIVES`), the owner's places (the desk and its chair too) and routes, walk regions, the `PrivateArea`, the ceiling light. |
| `levels/furnish.py` | Furnishings in any frame (`Builder`: a `FlatPlan`'s, or the world's - `WorldFrame` - for number 16; `Wall`: things on a wall by how far along, how high, how far out): windows with views and blinds or curtains, framed pictures and posters, shelves, books, jars, bottles, plants, rugs, lamps, fairy lights, cupboard fronts and wall cupboards, a worktop with a sink let into it, taps, an extractor hood, fridge details, a toilet, a pedestal basin and mirror, towel rails, loo rolls, food. |
| `levels/main_street.py` | The main street: the blocks' faces with windows and doors, pavement, kerb, the road (a parking strip, two lanes, the median and its markings), guard rail, street lamps, barricades, the monorail's beam and pillars, the portal walls and tunnels at either end, the UNDERPASS doorway, the car (`car()`), the traffic lanes (`LANES`), the street flats (`STREET_FLATS`, `FLAT17`, `HOMES`), walk region and places. |
| `levels/bar.py` | The Neon Moth as data: its brick front with glass windows and signs, the bar, the back bar, stools, tables, jukebox, television, lights, the hinged door, the walk region and the places people use in it (the stools and chairs, where a glass goes for each, the bartender's spots behind the bar). |
| `levels/metro.py` | The metro station as data: the hall behind its glass front, the ticket gates, the stair, the landing, the footbridge, the platform; the way up and down as places; `DATA` (`MetroData`: where the trains run and stop). |
| `render/smoke.py` | `SmokeRenderer`: cigarette smoke as particles from the pose (`HumanoidPose.ember`, `inhaling`, `exhale`): a trail curling up off the tip (none while drawing on it), the lungful blown out after a drag, the cherry's halo; billboarded soft discs, frame-`dt` driven (still when paused). |
| `render/weather.py` | `WeatherRenderer`: the rain (streaks round the eye anchored in the world, only under the open sky, never under a roof; splash crowns where it lands), the puddles (one baked mesh each of wet stain, lit glossy water and neon sheen per step of wetness; rings spreading on them in the rain), the spray; `fog_colour`. The scene's `weather_light` adds rain-cloud gloom, lightning and the fog (the far plane pulled in to what can be seen). |
| `render/audit.py` | The z-fighting finder: `GeometryRecorder` (a `DrawContext` that records world-space faces instead of drawing), `record_level`, `find_zfights` (coplanar, same-facing, overlapping, visible faces from different draws), `audit_level`; CLI `python -m cyberduke.render.audit`. |
| `core/geometry.py` | `AABB` (with explicit Y range and a `blocks_sight` flag), `overlaps_cylinder`, 2D/3D slab segment tests, `line_of_sight_clear`. The **only** collision/LOS maths in the project. |
| `world/level.py` | The level data model: `LevelData` → `SectorData` (props, colliders, lights, walk regions), `DoorData` (sliding, or hinged with `hinge` / `swing`, `inside` latch), entity placements (`TerminalData` with `notes`, `CameraData` with `watch_area`, `VendingMachineData`, `PanelData`, `PickupData`, `GuardData`, `CivilianData`, `MerchantData`), `Place`s, routes and `ItemData` for the street life, `PrivateArea`s, `platforms` (steps to stand on), spawn, nav bounds, `exit_zone` / `exit_to`, `on_caught`, `brief`. |
| `world/props.py` | `Prop` (box / cylinder / soft-edged `disc` + `Material`, rotation, `solid`, `label`, `tags`) with exact `footprint()` bounds. |
| `levels/dressing.py` | `Placer` (seeded rejection sampling against colliders, entity bounds, doors and previously placed clutter), `mushroom_cluster`, `scatter_trash`. Everything it makes is walk-over clutter tagged `clutter`/`transparent`. |
| `levels/loader.py` | `level_to_dict` / `level_from_dict`, `save_level` / `load_level` (JSON, TOML), validation, and the `python -m cyberduke.levels` CLI. |
| `levels/street.py` | The Night Market Alley as data: facades (with the doors of 14, 16, 17 and 21), stalls, street furniture, the NEON COLD vending machine, the main street (`levels/main_street.py`) and the backdrop beyond it, lights, waypoints, `CivilianData` / `MerchantData` placements with their roles, the places and loose items the street life uses (`ALLEY_PLACES`, `ALLEY_ITEMS`), number 16 folded in from `levels/apartment.py`, the wall at the alley mouth (`ALLEY_WALL_Z`) and `on_caught="evict"`. No exit zone: nowhere further to go yet. |
| `levels/__init__.py` | `LEVELS`: the registry of shipped levels by name (`"facility"`, `"night_market"`) that `LevelData.exit_to` refers to. |
| `ai/civilian.py` | `Civilian` and `Merchant` (`StreetPerson`s): with a `role` they run its story from `ai/alley.py`, otherwise waypoint wandering with pauses / working a stall and turning to customers; eye contact, body avoidance, never report to the corporation - an owner minds a stranger in their home (`_mind_home`: their eyes fill the meter, but they only notice - turn, shout - once it is red); `civilian_spec(seed, role)` for a seeded look (and a smile for the `SMILING_ROLES`). |
| `ai/nightlife.py` | The bar and the metro: the bartender's night (in by metro, open at five, serving, clearing, chores, closing at three, home by metro), a regular's evening, a drink at the bar for anybody (`drink_at_the_bar`), and getting on and off the trains (`take_the_metro`, `ride_in`: through the gates, up and down the stair, into and out of a door's vestibule) - registered with `ai/alley.py`'s `STORIES`. |
| `ai/alley.py` | Street life: `AlleyLife` (counters and their orders, loose items, chats, swinging doors, the live vending machines, every home's television - `TvSet`, the bar - `BarState`, `BarOrder` -, the live metro), the acts (`Walk` - which checks it really got there -, `Glide`, `Turn`, `Play`, `Wait`, `Sleep`, `WatchTv`; `Away` only for tests: nobody goes out of sight), the stories (`STORIES`: sushi chef, junk dealer, office worker, tinkerer, courier, resident, smoker, stroller), what can catch somebody's eye on a walk (`ASIDES`: Dell and a broken vending machine - the `Storyteller` puts the walk on hold, plays the aside and takes the walk up again), the places each needs (`NEEDS`), queues (`queue_spot`) and `make_way`. |
| `levels/facility.py` | The shipped four-sector facility as data. Meshes (`props`) and gameplay colliders are authored side by side; `tests/test_level_data.py` keeps them consistent (no invisible walls, every solid prop inside a collider - or, for `compound`-tagged rotated panels, covered by several - spawns on walkable floor). The greenhouse (dome, maze hedges, rim planters, control nodes) is generated from a frozen wall list, `GH_WALLS`. Every collider is as tall as what is drawn inside it (a test holds it to 0.3 m): a rack to the top of its vines, a planter to its tallest mushroom - no invisible wall over a low bed hiding a standing netrunner. |
| `levels/doorway.py` | The Halcyon staff door (leaf, frame head, STAFF ONLY sign, alarm lamp, concrete portal) shared by both levels, so the greenhouse exit and the door you see from the alley are the same door. `DoorData.lamps` are status lamps beside a doorway that glow with the door's state. |
| `world/room.py` | `Room(level)`: live door state, `get_obstacles()` (colliders + shut doors), `is_valid_position()` from walk regions, elevation, lights. Nothing level-specific. |
| `world/surfaces.py` | Pure `floor_height(x, z, state, catwalk) -> (y, state)` for stairs (a step function) and catwalk decks; `Flight` (a straight flight of stairs anybody climbs: `height`, `fraction`) and `on_flights_and_decks` (its treads, and the raised floors it leads to - `ElevationState.deck_y` - that people also walk under). |
| `core/menu.py` | `Menu` / `PauseMenu` / `TitleMenu` / `LoadMenu`: rows (greyed-out ones are skipped), cursor wrapping, values cycled with left/right (graphics, display), what a key press asks for (`MenuAction`). No pygame, so the menus' behaviour is unit-tested. |
| `render/quality.py` | `Quality` / `GraphicsSettings` presets, `detect_quality(renderer, vendor)` from the GL strings, `lights_for` (the nearest N lights, in level order so the set does not flicker). |
| `core/events.py` | `EventBus` (typed `subscribe`, queued `emit` + `dispatch`, immediate `publish`) and the event types `NoiseEmitted`, `TerminalHacked`, `DoorUnlocked`, `NetworkBreached`, `IntelStolen`, `HackApplied`. Interactables emit; doors, NPCs and the hacking system react. |
| `world/doors.py` | `SlidingDoor`: opens on `DoorUnlocked(door_id)` (sent by its controlling terminal when someone chooses to override the locks, or by a quickhack on the door) or `unlock()` - merely jacking into the controlling computer does *not* open it; `high_security` doors ignore quickhacks aimed at them; `key_intel` doors open for a player who walks up with that data (`Simulation._unlock_keyed_doors`). Exposes its obstacle box while shut. |
| `ai/fsm.py` | `State` (`enter` / `update -> next` / `exit`) and `StateMachine` with `time_in_state`. |
| `entities/` | Behaviour-only entity classes. `PlayerController` does movement with per-axis sliding; observers (`SecurityCamera`, guards) *report* sightings to a `SightingSink`. |
| `ai/pathfinding.py` | Grid A* on an agent-inflated grid (no corner cutting) with raycast string-pulling. Search budget scales with the level's `nav_bounds`; blocked cells are memoised per search and tested against a `SpatialHash` of the obstacles (a maze route through 100+ colliders stays well under a tenth of a second). Returns a `Path` (`waypoints`, `found`, `adjusted`, `destination`). |
| `ai/npc.py` | `BaseNPC`: heading control, `navigate_towards` (A* + sliding + `avoid` bodies: walking round somebody in the way, stopping short of a spot somebody stands on, giving way when two get nowhere, cutting a corner once it is out of the way, walking out of a collider; `arrival` says why it stopped: "there", "taken" or "stalled"), `can_see_target` (range, FOV incl. head yaw, 3D occlusion), `body_box` / `blocks_player`. `HumanoidSpec` describes appearance as data (colours, height, plating, the `look`, the face's tone, mark and resting mood - `face_for(expression)`). |
| `ai/guard.py` | `CyberGuard` = `StateMachine` of small state classes: `Pacing` (or `PatrolRoute` for waypoint loops) `-> WalkingToWall -> Leaning -> WalkingToPatrol` (the wall trip only with a `wall_pos`), `Investigating -> Inspecting` (noise: the noise's own duration and gesture - stare, or hit the machine), the reaction chain `Alerted -> Suspicious -> LookingAround` (sightings), and the quickhack states `Blinded` (no perception, then resume) and `KnockedOut` (terminal). `blind()` / `knock_out()`; `essential` guards get the gold-visor `ELITE_GUARD_SPEC`. Timings, vision and suspicion thresholds in `config.GuardConfig`. |
| `systems/detection.py` | `DetectionSystem.begin_frame / report_sighting / end_frame`: additive gain while seen, fixed decay while unseen, order-independent. |
| `systems/scanner.py` | `Scanner.pulse_from(origin, targets)`: reveals everything within `scan_range` (no sight test) for `reveal_duration`; cooldown; `ScanPulse` for the renderer's expanding ring. |
| `systems/hacking.py` | `HackKind` / `HackSpec` / `HACKS_BY_TARGET_KIND`; the `Hackable` protocol (`target_kind`, `hack_id`, `faction`, `secured`, `hack_position`); `HackingSystem`: access routes (`DIRECT` = scanned + range + line of sight from the current `Viewpoint`, `NETWORK` = breached faction net, never for `secured`), crosshair focus, `options(target)` with per-target availability rules, timed `Upload`s, `apply()` through entity primitives, `Network` per `FactionData`, `breach()`. |
| `entities/terminal.py` | `Terminal`: holding `[E]` establishes the link and opens its interface (`HackingSystem.connect` / `session_options` / `session_request`); quickhack primitives `unlock_doors()`, `steal_intel()`, `overload()`; flags `secured`, `critical`, `intel`, `door_ids`. |
| `entities/camera.py` | `SecurityCamera`: sweep + sightings (a 3D ray from the lens to the chest: it sees over a coffee table, not through a wall or the crate you crouch behind; a home camera's `watch_area` - it only reports somebody inside it); `shut_down(seconds)` (three minutes out of the security system, head drooping; the countdown is never shortened), `hijack()` (works while dark) / `steer(yaw, tilt)` / `release()`, `viewpoint` (lens, at `current_tilt`). `HackingSystem.linked_cameras` / `cycle_camera` hop between a network's live feeds. |
| `animation/` | `HumanoidPose` dataclass (joint angles, hip roll, shoulder shrug, the upper body's `lean` at the hips, things held in each hand, where a held stick points (`aim_l`/`aim_r`) and how it is rolled in the fingers (`tilt_l`/`tilt_r`)), `mix_poses`; `walk_animation`, `inspect_weapon_animation(t, duration)` (time-scaled to the caller's duration), `investigate_animation`, `bang_machine_animation`, `blinded_animation`, `electrocuted_animation`, `idle_animation(t, seed)`, `cook_animation`, `hawker_animation`. `street.py`: everyday gestures (talk, point, reach, offer, hold up, eat - a bite at a time, smoke - with the ash flicked off, phone, knock - on the door's face, browse, think, wipe, chop, wrap, crouch, stoop to something low, sit, toss, carry). `reach.py`: two-bone arm IK (`arm_to`, bent forward with the `lean` too) and where a posed hand is. `skeleton.py`: the humanoid as boxes - the boxes the renderer draws its rounded parts in (both built by `rig.py`; a test compares them), for collision. `ragdoll.py`: the knocked-out collapse (`Collapse.begin` / `step`) and `lying_on` (a sleeper on a bed, at their own size). `street.drink`: a can to the lips with the elbow out. `anatomy.py`: the head as numbers (its size, its shape - a flatter back than front - and where the eyes, nose, mouth and hairline are), shared by the mesh, the rig and the face painter. `look.py`: how somebody looks (`Look`: hair, eyes, top, trim, implants, extras) and their colours (`Palette`), drawn from a seed (`street_look`, `guard_look`). `HumanoidPose.expression`: the face of the moment (`street.speech`, `street.mutter`). `hairdo.py`: every hairstyle as meshes grown on the skull (crop, crest, spikes, side shave, bob, long hair, top knot, twin tails: `STYLES`, `pieces`, `raw`, `box`), textured along the way the hair grows. `street.stairs`: up and down a stair a tread at a time (leg IK puts the feet on the treads). |
| `assets/textures.py` | One small, seeded generator per texture style (`@texture("wall", tile=2.0)`), returning `128x128x3` uint8 arrays (row 0 = top). `tile` marks a seamless surface material and its world size; `tiled_textures()` / `tile_size()` expose them. `assets/homes.py` adds the homes' wallpapers, floors and tiles, pictures, window views and the television channels (`TV_CHANNELS`, `tv_frame`: each a loop of frames), with a 3x5 pixel font for their lettering. `assets/faces.py` paints faces per skin tone, mark and expression, on demand (`face_name` -> `face_image`); `assets/signs.py` the neon signs with words on them (`neon_sign`: laid out for the board they go on). |
| `render/context.py` | `DrawContext`: the drawing API all mesh code targets (matrix stack, colour/emission/specular, unit cube/cylinder, `soft_disc` with a per-vertex alpha falloff, unlit lines, `cached()` static geometry, 2D overlay). `set_texture_tiles` + `cube_repeat` / `cylinder_repeat`: tiled textures repeat by the primitive's transformed size. |
| `render/backends/legacy.py` | Fixed-function OpenGL 1.x; `cached()` = display lists; light positions re-specified after every frame's view transform (GL stores them in eye space). The original look. |
| `render/backends/modern.py` | OpenGL 2.1 / GLSL 1.20: `cached()` bakes geometry into one VBO per key drawn as one index range per texture (tiled uvs baked in, `tiled_uvs`); dynamic draws use unit-primitive VBOs + uniforms (`u_uv_repeat`); the fixed-function light model evaluated **per pixel**; logarithmic depth (`log_depth_coefficient`), so layers a centimetre apart never z-fight at a distance. |
| `render/scene.py` | `SceneRenderer(viewport, room, backend)`: picks the backend, follows level swaps (`follow_level`: drops cached geometry, takes the new lights) (`auto` = modern on a GPU, legacy on a software renderer; modern falls back to legacy if shaders fail) and draws a `Simulation` from `sim.viewpoint` (the player's eyes or a hijacked camera's lens); on an outdoor level (`LevelData.outdoor`) the sky colour, the ambient light and the night-only lamps follow the clock (`time_of_day_light`). `smoke.py` draws cigarette smoke. `props.py`, `sectors.py` (static geometry cached per sector, culled only by the preset's view distance - never by doors, because a sector owns the walls it shares with its neighbours; `pulse`-tagged props drawn live with breathing emission; translucent props - fog, light pools - in a late, depth-write-off, back-to-front pass via `Material.alpha`), `entity_renderers.py`, `humanoid.py` (the body through `animation.rig`, each person one `render.batch` on a real backend, with whatever the hands hold), `items.py` (the things people handle and leave lying about), `hud.py` hold the meshes; `glow.py` bakes the light pools; `lighting.py` uploads `Light` records and draws their fixtures (`ceiling_y` sets where hanging cables attach); `effects.py` draws the scan ring and electrical sparks; `beams.py` clips the camera beams against the visible geometry in 3D (exact oriented boxes) and finds what each camera hangs from; `television.py` draws the screens live - every home's TV (and the bar's) showing what is on it this moment, the windows' views by day or night, the bar's OPEN sign lit while it is open; `traffic.py` draws the cars where they are (each body and wheel baked once per model and paint into a batch in its own frame, the wheels turning, brake lights, the light they throw on the road); `metro.py` the train on its beam with its door leaves sliding, and the ticket gates' flaps. |
| `render/scene.py` (quest pictures, windows) | `capture(sim, eye, target)` renders the world from a quest picture's camera into a 16:9 texture (taken once, when the QUESTS tab first needs it; another level's pictures from a stand-in simulation). `open_window(viewport, fullscreen)` / `close_window()` (which also forgets the primitives' display lists, that die with the context). |
| `render/projection.py` | `Projector`: pure-NumPy world-to-screen for the HUD's scanner markers (same matrices as the modern backend, so markers line up on both). |
| `render/hud.py` | Crosshair, the clock (top left: `HH:MM` with a blinking colon), detection meter (cyan, yellow past `DETECTION.yellow_level`, red past `red_level`), terminal progress (and a home computer's notes), the level title card on arrival, and the netrunner layer: target markers (through walls, pinned to the screen edge when off-screen), the focus menu with locked reasons, upload bar, network status, hack log, camera-feed frame. With a `game` it draws what the game's mode says: the start menu, the load menu, the [Tab] screen, or the HUD with captions, notices, the `[E]` prompt, the caught screen and the pause menu over it. |
| `render/ui.py` | `Ui`: a small immediate-mode UI kit in the HUD's style - glass panels with corner brackets, buttons, scroll bars, dividers, chromatic-split titles, scanlines, and text that wraps or is cut short with an ellipsis to fit its box (`wrap`, `fit`). Clipping via `DrawContext.clip` / `unclip` (scissor). |
| `render/screens.py` | The full screens and overlays drawn with `Ui`: the [Tab] screen (inventory grid, item details and the EQUIP button; quest list, objective, pictures, scrolling details), the start / pause / load menus, captions, notices, the prompt, the caught screen and its fades. Each leaves its clickable rectangles for the next frame. |
| `render/icons.py` / `assets/icons.py` | The item icons: NumPy-generated 96 px neon line art (`@icon("mod_ping")`), uploaded once and cached (`IconCache`). |

## Getting Started

### Prerequisites

- Python 3.10+
- An OpenGL 1.x-capable driver (no GLU required)

### Installation

```bash
git clone https://github.com/your-username/cyberduke.git
cd cyberduke
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"            # runtime deps + pytest/ruff
```

Run the game with any of:

```bash
python play.py       # launcher: no install needed, checks dependencies, keeps the console open on a crash
cyberduke            # console script (after pip install -e .)
python -m cyberduke  # module form
```

`play.py` can also be double-clicked on desktops where `.py` files are associated with Python.

### Level files

The shipped levels are typed Python (`levels/facility.py`, `levels/street.py`, registered by
name in `levels.LEVELS`), but levels can be exported to and loaded from JSON or TOML
(`pip install -e ".[toml]"` for TOML writing):

```bash
python -m cyberduke.levels export my_level.json   # or .toml
python -m cyberduke.levels check  my_level.json   # validate + summary
cyberduke --level my_level.json                   # play it
```

Files are schema-versioned and validated on load (door/terminal references, one objective
terminal, prop kinds, walk-region edges, `exit_to` needs an `exit_zone`), so a broken file fails
at startup with a clear message. A file level's `exit_to` names a registered level to load next.

### Renderer

```bash
cyberduke --renderer modern   # VBOs + GLSL, per-pixel lighting
cyberduke --renderer legacy   # fixed-function OpenGL 1.x + display lists (the original look)
cyberduke                      # auto: modern on a GPU, legacy on a software renderer
```

Both backends draw the same mesh code through `render.context.DrawContext` and light it the
same way (per vertex on legacy, per pixel on modern): the lights stay where the level puts
them and the quality preset's ambient level applies on both. Light colours are scaled by
`render.scene.LIGHT_EXPOSURE` (0.75) on upload, one knob for the overall brightness of the
night.

The window asks for a 24-bit depth buffer (`render.scene.open_window`; it falls back to 16
bits with a warning). With SDL's default 16 bits, signs on their panels, screens in their
bezels and rack faces z-fought at a distance - thin black lines flickering through the glow.

### Graphics quality

`[Esc]` pauses: the simulation stops consuming time entirely (nothing moves, no timer ticks,
no animation or upload advances) and the menu takes over the screen. The graphics preset and
the display mode (fullscreen / window) are rows of both the start menu and the pause menu;
see [Start Menu, Saves & Settings](#start-menu-saves--settings).

| Preset | View distance | Lights | Detail |
| :--- | :--- | :--- | :--- |
| **LOW** | 30 m | 2 | flat ambient, no sparks or scanner ring, 4-ray camera beams, no feed scanlines, 350 streaks of rain |
| **MEDIUM** | 45 m | 5 | sparks, scanner ring, 8-ray beams, 900 streaks of rain |
| **HIGH** | 75 m | 8 | the full eight-light neon grading, 48 m of animated bioluminescence, 12-ray beams, 1600 streaks of rain |

(In a fog the view distance comes in to as far as can be seen: see [The Weather](#the-weather).)

```bash
cyberduke --graphics high     # or low / medium / auto; default: as saved in ~/Unprepared/settings.json, else auto
cyberduke --fullscreen        # or --no-fullscreen; default: as saved
```

Auto-detection reads `GL_RENDERER`/`GL_VENDOR`: a software rasteriser (llvmpipe, swrast, ...)
gets LOW, a recognised discrete GPU (GeForce/RTX, Radeon RX, Apple M-series, ...) gets HIGH,
anything else MEDIUM. The preset also decides whether the modern per-pixel backend is
preferred. Changing it in a menu re-applies everything live and remembers it for next time.

### Debugging

`CYBERDUKE_GL_DEBUG=1` turns PyOpenGL's per-call error checking back on (off by default for speed).

### Tests & Lint

```bash
./run_tests.sh            # ruff check + ruff format --check + pytest
./run_tests.sh --fast     # pytest only
./run_tests.sh --fix      # auto-fix lint/format, then run everything
./run_tests.sh --smoke    # also boot the real game under Xvfb (Linux): 60 frames on each backend, then a
                          # scripted session - NEW GAME, the deck screen, the pause menu, SAVE GAME
```

The simulation is fully testable without a window: see `tests/test_simulation.py` for
driving a `Simulation` with synthetic `InputState` frames - `test_full_playthrough_from_the_spawn_to_the_street`
walks and hacks the whole game, facility door to the barricade at the end of the main street, in one run.

The game around it is too: `tests/test_game.py` drives `Game` through the start menu, the
pause menu, the deck screen (keys and mouse), saving, loading and the settings file (with
`CYBERDUKE_USER_DIR` pointed at a temporary folder); `tests/test_savegame.py` round-trips
runs in both levels (mid-quest, guards down, cameras dark) and checks saves are never
overwritten; `tests/test_number16.py` plays "Something Special" - the conversation, the panel,
the flat, the camera and Dell, the stick, the reward - and both ways of being caught;
`tests/test_inventory.py` the items and mods and what they do to the scanner, the hacking and
the detection meter; `tests/test_screens.py` draws the deck screen and the menus into a
recording context and checks every slot label fits its slot and every hotspot works.

**Z-fighting** is tested for, not eyeballed: `tests/test_zfighting.py` runs the audit (`render/audit.py`) over every shipped level - every prop, door, computer, camera, vending machine, light fixture, pool of light and fog bank, drawn into a recorder instead of the screen - and fails on any two surfaces in one plane; it also reproduces the glitches once seen in number 16 (the ceiling through the outside wall, the street's asphalt through the floor, the mirror in the tiles) to prove the audit catches them. Run it by hand with

```bash
python -m cyberduke.render.audit                   # every level
python -m cyberduke.render.audit --level night_market --gap 0.002
```

`tests/test_fog.py` walks a camera through the greenhouse haze and past the wok's steam and checks no fog bank jumps between frames (each bank's drift is keyed to its own index in its sector, not to its place in the back-to-front sort); `tests/test_smoke.py` smokes a cigarette (a trail while smouldering, none on a drag, a puff on the exhale, the cherry bright on a drag and dull otherwise); `tests/test_schedule.py` runs the street's day.

**People**: `tests/test_hands.py` holds the hands to holding things (every item has a grip; the fingers move to hold it; a can sits against the palm with the fingers round it, not in it; a fist to knock, a pointing index, typing fingers and a scrolling thumb that move; the drinker's elbow out and the can tipped up, rim at the lips; a blinded guard's palms over the visor - in front of the face wherever the head jerks, never inside it; a wave with the palm to the front beside the head; a phone held up in both hands, one either side of it - no palm, finger or thumb in the phone at any moment, the two hands not in each other), the rounded shapes to being closed and facing outwards, and a person to one batch holding every part; `tests/test_homes.py` plays every chore of every home and puts everybody to bed (nothing into a wall or furniture, heads on pillows); `tests/test_shove.py` walks into people - pushed aside and back to what they were doing, pinned in a corner and out again, an unpushable person still no trap, a walk that goes on after a push. `tests/test_looks.py` holds the hair to its strands (every piece in the hair texture, the strands along the way it grows, no rolls or slabs of hair, long hair and bobs clear of the eyes and the shoulders) and everybody to a look of their own (no two people of the street alike, all sorts of hair and eyes, everybody with some chrome), heads to having a forehead and a flatter back, and faces to moving with what people do (talking syllable by syllable, a bite opening the mouth and chewing, the lips pursed on a drag, muttering over the menu, an owner shouting at an intruder); `tests/test_guard.py` the guards to one uniform each their own and a face that shows what they are doing; `tests/test_skeleton.py` the hair, chrome and neon to being drawn but not collided with; `tests/test_textures.py` every expression to its own mouth in the same place, white teeth and eyes whatever the skin, and the face's edges to the colour of the rest of the head in every skin tone (the skin drawn in the person's tone, lit like the face). `tests/test_looks.py` also holds every spike of a mohawk to a sharp point and the crest to spikes evenly from the forehead to the nape (no gap over the crown); `tests/test_alley.py` the smoker to turning back to his cigarette for every drag. `tests/test_items.py` the fish to being a fish (a shaped, scaly body, eight fins, two eyes, no boxes) and the sushi cut on the board to being what goes into the box (the same number of pieces). **The street**: `tests/test_weather.py` (see [The Weather](#the-weather)), `tests/test_traffic.py` (the lanes, the tunnels, cars keeping their distance, slowing behind a slower one, coming and going only in the tunnels, busier in the rush hour, the models free of z-fighting and drawn where they are with their wheels rolling) and `tests/test_nightlife.py` (the bar, the station, the trains and the people who use them - see [The Neon Moth & the Metro](#the-neon-moth--the-metro)).

`tests/test_traversability.py` flood-fills the whole level with the real movement code: every
cell a player-sized body fits in must be reachable from the spawn (or be declared a `NoGoZone`
in the level), no reachable position may be a trap, sweeping along every handrail must never
stall, and NPC-sized agents must be able to path to every cell they fit in.

`tests/test_ragdoll.py` knocks a guard out - falling back, forward, left and right - in
front of walls at several distances, a crate, a low planter, a column, a corner, a pipe at
waist height, a wall ahead, a wall to one side and in a narrow corridor, and checks every
drawn box of the body at every frame of the collapse: nothing goes into the obstacles (or
more than 2 cm into the floor), and the body ends on the floor - or draped over something
low - never propped up against a wall, and never lying in the air on a folded arm (a forward
fall lands flat, arms out or tucked alongside, whichever fits). `tests/test_visual_regressions.py` holds the drawn geometry to what the
screenshots showed was wrong (faces on every side of the head, boots in the floor, door leaves
through walls, beams through crates, floating cameras...).

### Controls

| Input | Action |
| :--- | :--- |
| **W, A, S, D** | Move forward, strafe left, move backward, strafe right |
| **Mouse** | Look around (Yaw / Pitch) |
| **C** | Toggle Crouch (lowers eye-line, dampens movement profile, reduces detection build-up) |
| **E** | Interact: talk to the person in front of you / skip or advance a caption; take the thing in front of you; hold within 2.5 m to jack into a terminal and open its interface; press within 2.5 m to trigger a vending machine distraction |
| **F** | Scanner ping: reveal every electronic target within 16 m (through walls) for 12 s (further with the Ping Amplifier installed) |
| **Tab** | The deck screen: INVENTORY / QUESTS (again, or `Esc`, to close) |
| **[ / ]** | Previous / next tab on the deck screen |
| **Arrows, W A S D, Enter, E** (deck screen) | Select an item / quest; EQUIP / UNEQUIP the selected mod |
| **Mouse, wheel, PgUp / PgDn** (deck screen and menus) | Point and click; scroll the inventory grid, the quest details, the save list |
| **1 - 9** | Run the numbered option: the open terminal's interface, or a quickhack on the target under the crosshair |
| **Q** | Disconnect from a terminal interface or a hijacked camera feed |
| **Left / Right** | While hijacking: switch to the previous / next live camera on the same network |
| **Mouse** (hijacking) | Pan and tilt the camera head |
| **Esc** | Pause menu (resume / save / load / graphics / display / new game / main menu / quit); back out of the deck screen or the load menu |
| **Up / Down, Left / Right, Enter** (or the mouse) | Navigate the start, pause and load menus, change the selected setting, confirm |

---

## Netrunning: Scanner, Quickhacks & Faction Networks

```text
 [F] ping ────▶ targets revealed (markers through walls, 12 s)
                      │
                      ▼
   crosshair ──▶ FOCUS: menu of hacks, each AVAILABLE or LOCKED with a reason
                      │
        [1-9] ────────▼───────── upload (0.6 - 2.5 s) ──▶ effect via the target's primitive
                      │
   access route:  DIRECT  = scanned + within 14 m + line of sight from the viewpoint
                  NETWORK = the target's faction network is breached (anywhere, no LOS)
                            - never for SECURED computers
```

| Target | Hacks | Data flags that restrict it |
| :--- | :--- | :--- |
| Computer (`Terminal`) | Remotely: OVERRIDE DOORS, OVERLOAD, and STEAL DATA where there is data. At the desk (hold `[E]`): UNLOCK DOORS, HIJACK CAMERAS, DISABLE CAMERAS (the whole network's), SHORT CIRCUIT (DISTRACT & DESTROY - `NoiseEmitted` pulls guards over for 40 s; the machine controls nothing afterwards), STEAL DATA | `critical`: no OVERLOAD / SHORT CIRCUIT. `secured`: scanner-visible, never on the network. `hardened`: air-gapped, jack-in only (which still breaches its network). No `intel`: no STEAL DATA row at all. |
| Person with cyberware (`CyberGuard`) | BLIND (6 s: `can_see_player` is false, both palms clamped over the visor - just in front of the face, the fingers flat over the brow, the elbows out, turning with the head as it jerks about: never through it (`clips.blinded_animation`, arm IK to points on the face; the `face` grip) - then resumes its routine having lost the player), ELECTROCUTE (`KnockedOut`: convulses on its feet for a couple of seconds, then goes down like a dead weight (`animation/ragdoll.py`) - back, forward or to either side, at random (`knock_out(direction)`): the knees buckle and it tips over, ending on its back, face down (head turned) or on its side (knees drawn up). If something stops it - a wall, a rack, a tall crate - it does not stay propped up: it slides down until it is sitting, then topples off to whichever side has room, the head sliding off what it leaned on, and ends on the floor; something low (a planter) it can end draped over. Every drawn box of the body is tested in 3D against the obstacles each frame - limbs in the way fold rather than hold the body up - and it always rests on its lowest point. Permanently down, ignores noises) | `essential`: no ELECTROCUTE (elite, gold visor). |
| Door (`SlidingDoor`) | FORCE OPEN | `high_security`: refused - open it from its terminal. `key_intel`: encryption-keyed - refused; walk up carrying the data. `relock_after`: sets itself again that long after it was forced (the metro's ticket gates: a minute). |
| Camera (`SecurityCamera`) | SHUT DOWN (180 s out of the security system, the head droops and hangs, no sightings; ejects a hijacker, and the countdown cannot be cut short), HIJACK FEED (works on dark cameras too; the view moves to the lens; mouse pans and tilts; `[<]`/`[>]` hop between the network's cameras; the body stays put; scans and direct hacks launch from the feed) | - |
| Vending machine | DISTRACT (the same short-circuit as `[E]`, from anywhere on the network; sparks for 18 s, the guard bangs on it and watches it for 45 s; repeats after the 22 s cooldown) | - |

**Hacking a computer breaches its faction's network.** Jacking in with `[E]` (which opens the
computer's own interface) or any successful quickhack on a non-secured terminal opens it (`NetworkBreached`): every member of that faction -
guards, cameras, doors, vending machines, other computers - is highlighted across the map in the
faction colour and reachable from anywhere. The facility has two factions (`levels/facility.py`):
**Halcyon Dynamics Security** runs the first three sectors - the staging terminal is the way in,
the datacore terminal is `secured` (its own firewall) and `critical`, so it must be hacked with a
line of sight or at the desk, and can never be destroyed - and **Halcyon Biotech Research** runs
the greenhouse on a separate network whose only computers are `hardened`: jacking into one of
them is the only way onto it.

Levels declare factions and flags as data:

```python
FactionData("halcyon", "Halcyon Dynamics Security", color=(1.0, 0.55, 0.1))
TerminalData("datacore_terminal", 0.0, 0.0, 30.0, yaw=180.0, objective=True,
             faction="halcyon", intel="datacore image: project HALCYON", secured=True, critical=True)
GuardData(2.1, 0.0, 26.0, ..., guard_id="vault_guard", faction="halcyon", essential=True)
DoorData("blast_door", ..., unlocked_by="staging_terminal", high_security=True)
DoorData("vault_bulkhead", ..., unlocked_by=None)   # a plain networked lock: quickhacks only
DoorData("greenhouse_door", ..., high_security=True, key_intel=DATACORE_INTEL)  # opens for the cipher
TerminalData("node_east_terminal", ..., faction="halcyon_bio", hardened=True)   # jack in or nothing
GuardData(..., wall_pos=None, patrol_route=(gh_cell_center(0, -2), gh_cell_center(0, 3), ...))
LevelData(..., exit_zone=AABB(-1.6, 1.6, 78.2, 79.2), exit_to="night_market")  # out into the alley
```

Progression hook: `Player.known_hacks` holds every `HackKind` for now; `forget_all_hacks()` /
`learn(kind)` are what a levelling system will call. Unknown hacks still appear in the menu,
locked with `NOT LEARNED`.

---

## Interactive Entities & Hazards

### Security Camera
- **Mount Point**: Airlock ceiling at $X=0.0, Y=3.6, Z=11.2$.
- **Sweep Behavior**: Sine-wave oscillation across $\pm 35^\circ$ sweep angle.
- **Beam Truncation**: Visual laser frustum rays are clipped in 3D against the drawn geometry (walls, pipes, racks, door leaves, desks - exact oriented boxes, not the generous gameplay colliders), ending at the exact point of contact with vertical seam drop lines down upright faces instead of penetrating outside the facility or stopping in mid-air. The camera hangs from the ceiling (or the greenhouse dome) on a drop rod.
- **Detection Rate**: $+65.0$/s standing; crouching dampens gain to $+18.0$/s. Fully blocked by physical walls and closed doors - a 3D ray from the lens to the chest, so it looks down over low furniture but not through the crate you crouch behind.
- **Quickhacks**: SHUT DOWN (lens dark, no beam, head drooping for 180 s) and HIJACK FEED (the scene is drawn from the lens; the beam turns green and shows what the feed covers).

### Vending Machine ("NEON COLD")
- **Placement**: Datacore Vault at $X=-7.5, Z=25.0$ beneath the left catwalk; and in the Night Market Alley at $X=3.8, Z=17.6$ against the east wall, facing the alley (`VendingMachine` is yaw-aware: its bounds and front turn with it), on its own little network (`neon_cold`). People of the street buy cans from it (`alley.vending_run`); shorted, it stays broken (`broken`, `repair()`, `BROKEN_FOR` 300 s) until Dell - out and about, never from inside his flat - sees it and kicks it back to life; anybody else just thumps it and goes without.
- **Visuals**: Backlit neon marquee, glowing beverage product display, illuminated coin/cred slot, and delivery hopper.
- **Acoustic Distraction**: Pressing `[E]` within 2.5m - or the DISTRACT quickhack from anywhere on the breached network - triggers an electrical short-circuit sound and dispenses a soda can. Emits an acoustic pulse ($r = 22.0\text{m}$) that pulls nearby guards into an A*-routed investigation: they bang on the machine, then watch it, for 45 seconds in all. Repeatable once the 22 s cooldown lapses.

### Terminals & Blast Doors
- **Stage 1 Terminal ($X=3.5, Z=-4.0$)**: Holding `[E]` for ~1.7 s establishes the link (that is the breach, and it exposes the Halcyon network) and opens the terminal's interface: UNLOCK DOORS (the blast door at $Z=10.0$), HIJACK CAMERAS, SHORT CIRCUIT, STEAL DATA. It is `critical` - the only way out of the staging bay runs through it - so SHORT CIRCUIT is refused. The same actions exist as remote quickhacks.
- **Vault Maintenance Terminal ($X=7.5, Z=35.5$)**: a coolant workstation under the right catwalk that controls nothing and is not critical: the computer to fry when what you want is a loud distraction at the far end of the vault.
- **Blast Door ($Z=10.0$)**: MIL-SPEC, `high_security` - a FORCE OPEN quickhack on the door bounces off; it only answers to the staging terminal.
- **Vault Bulkhead ($Z=18.0$)**: a standard networked lock with no controlling terminal. FORCE OPEN it with a line of sight from the corridor (or from the hijacked camera), or over the network.
- **Datacore Central Terminal ($X=0.0, Z=30.0$)**: Primary extraction terminal rotated $180^\circ$ outward at the base of the central monolith. `secured` (never on the network) and `critical` (cannot be overloaded). Its data - by jacking in or a direct STEAL DATA - is the **greenhouse door cipher**.
- **Greenhouse Door ($Z=42.0$)**: encryption-keyed. Walk up to it carrying the cipher and it reads it off your deck and opens (the HUD says so); without it, or with a quickhack, it stays shut.
- **Control Node Terminals (greenhouse)**: `hardened` - air-gapped. Scans show them, no quickhack reaches them; jacking in opens their door (east: the maze gate; west: the street exit) and breaches the biotech network.
- **Street Exit ($Z=75.7$)**: the north wall of the dome - the same yellow Halcyon staff door you see from the alley. Unlocked from the west control node its lamp turns green, but it stays shut until you walk up to it (`DoorData.approach_range`): it slides open in front of you and shut again behind. Walking through it onto the street is the end of the facility - and the start of the street.

---

## Implementing Custom NPCs

The NPC architecture is built around four isolated layers:

```text
 animation/   HumanoidPose + clips          (what the body is doing)
 ai/npc.py    BaseNPC + HumanoidSpec        (where it is, how it moves/sees, how it looks - as data)
 ai/guard.py  CyberGuard                    (behaviour state machine)
 render/      humanoid.py + prop registry   (how it is drawn - the only layer that knows OpenGL)
```

### 1. Pose Generation & Keyframing

A clip is any function returning a `HumanoidPose`. Joint vectors:
- `arm_l` / `arm_r`: `[pitch, yaw, roll, elbow_bend]` (degrees)
- `leg_l` / `leg_r`: `[hip_pitch, hip_yaw, knee_bend]` (degrees)
- `prop_pos` / `prop_rot`: prop translation / rotation relative to the character origin.

```python
import math
from cyberduke.animation import HumanoidPose

def typing_on_datapad_animation(t: float) -> HumanoidPose:
    tap = math.sin(t * 4.0) * 5.0
    return HumanoidPose(
        bob=math.sin(t * 1.5) * 0.003,
        head_pitch=28.0,
        arm_l=[-15.0, -25.0, 10.0, -90.0],
        arm_r=[-10.0, 15.0, -10.0, -95.0 + tap],
        has_prop=True,
        prop_pos=[0.0, 1.25, 0.32],
        prop_rot=[35.0, 0.0, 0.0],
    )
```

### 2. Appearance & Props

Appearance is plain data (`HumanoidSpec`: colours, a height, armour, the face). Prop meshes
are registered by name with the renderer and drawn through the `DrawContext` API, so the
NPC module never imports OpenGL:

```python
# render side (e.g. render/props.py)
from cyberduke.render.context import DrawContext
from cyberduke.render.humanoid import register_prop

def draw_datapad(ctx: DrawContext) -> None:
    ctx.color(0.12, 0.12, 0.14)
    ctx.box(0.0, 0.0, 0.0, 0.22, 0.02, 0.15)
    with ctx.emissive((0.0, 0.8, 1.0), color=(0.0, 0.8, 1.0)):
        ctx.box(0.0, 0.015, 0.0, 0.19, 0.01, 0.12)
    ctx.color(1.0, 1.0, 1.0)

register_prop("datapad", draw_datapad)
```

Things held in the hands are items (`render/items.py`: `ITEMS` - a drawer and how it sits in
a hand) set on the pose (`pose.hand_r = "phone"`); the hand takes the item's grip
(`animation.hands.GRIP_FOR_ITEM`), or give an empty hand a shape of its own in the clip:
`pose.grip_r = "point"` (`"fist"`, `"open"`, `"flat"`, `"type"`, `"beckon"`...), and move
the fingers with `pose.fingers_r = (thumb, index, middle, ring, little)` (0 straight, 1
curled).

```python
# simulation side
from cyberduke.ai.npc import HumanoidSpec

COURIER_SPEC = HumanoidSpec(hair_color=(0.8, 0.5, 0.1), prop="datapad")
```

### 3. Behaviour, Pathfinding & Perception

Subclass `BaseNPC`:
- `self.navigate_towards(x, z, dt, obstacles=..., stop_dist=0.15, avoid=[(x, z, r), ...]) -> bool` - A* route on a 0.4 m grid, string-pulled, with per-axis obstacle sliding. `True` on arrival; a goal inside an obstacle counts as reached at the nearest free point. `avoid` lists other bodies to steer round (the crowd code in `ai/civilian.py` passes every other NPC and the player). `self.arrival` says why it stopped: `"there"`, `"taken"` (somebody is standing on the goal) or `"stalled"` (pinned short of it - check the distance if it matters, as `ai.alley.Walk` does).
- `self.blocks_player = True` makes the body solid to the player (`body_box()` joins `Simulation.player_obstacles`).
- `self.rotate_towards(yaw, dt) -> bool` - smooth heading change.
- `self.can_see_target(x, z, max_range, fov_angle, obstacles, target_y) -> bool` - range, FOV (including head yaw) and 3D occlusion; boxes flagged `blocks_sight=False` (railings) are see-through. Pass `player.sight_target_y` for the player.
- `self.advance_walk_cycle(dt)` - applies the walk pose for the distance covered.
- `self.bind(bus)` - subscribe to events (`NoiseEmitted`, `TerminalHacked`, ...) the NPC should react to.
- For anything beyond two or three states, build the behaviour from `ai.fsm.State` classes as `CyberGuard` does: each state is a ~20-line class with `enter` / `update` / `exit`, and the machine tracks `time_in_state` for you.

### 4. Step-by-Step Example: Datapad Courier

```python
# ai/courier.py
from cyberduke.ai.npc import BaseNPC, HumanoidSpec
from cyberduke.ai.pathfinding import Bounds
from cyberduke.animation import HumanoidPose
from cyberduke.core.geometry import AABB
from cyberduke.entities.player import Player
from cyberduke.systems.detection import SightingSink

COURIER_SPEC = HumanoidSpec(hair_color=(0.8, 0.5, 0.1), prop="datapad")


class CorporateCourier(BaseNPC):
    def __init__(self, x: float, y: float, z: float, nav_bounds: Bounds) -> None:
        super().__init__(x, y, z, spec=COURIER_SPEC, nav_bounds=nav_bounds, walk_speed=0.9, turn_speed=180.0)
        self.state = "PATROLLING_A"
        self.read_timer = 0.0

    def update(self, player: Player, dt: float, obstacles: list[AABB], detection: SightingSink | None = None,
               sight_blockers: list[AABB] | None = None, others=()) -> bool:
        if self.state == "PATROLLING_A":
            if self.navigate_towards(3.0, 4.0, dt, obstacles=obstacles):
                self.state, self.read_timer = "READING", 0.0
        elif self.state == "READING":
            self.read_timer += dt
            self.pose = HumanoidPose(head_pitch=26.0, arm_l=[-10, -15, 5, -85], arm_r=[-10, 15, -5, -85],
                                     has_prop=True, prop_pos=[0.0, 1.22, 0.30], prop_rot=[30.0, 0.0, 0.0])
            if self.read_timer > 6.0:
                self.state = "PATROLLING_B"
        elif self.state == "PATROLLING_B":
            if self.navigate_towards(-3.0, 5.5, dt, obstacles=obstacles):
                self.state, self.read_timer = "READING", 0.0

        seen = self.can_see_target(player.x, player.z, max_range=2.5, fov_angle=30.0,
                                   obstacles=obstacles, target_y=player.sight_target_y)
        if seen and detection is not None and not player.crouching:
            detection.report_sighting(10.0)   # a jumpy civilian adds a little suspicion
        return seen
```

To add it to the game, give it a placement record in `world/level.py` (as `CivilianData` /
`MerchantData` do), construct it in `Simulation._build` and include it in `Simulation.npcs`; the
update loop, the player's collision with solid bodies and the rendering (`render.humanoid.draw_npc`
draws every NPC in `sim.npcs`) are automatic. `ai/civilian.py` is the worked example.

To give a street person a *story* instead, write a generator in `ai/alley.py` that yields acts
(`Walk`, `Glide`, `Turn`, `Play(clip, seconds, events={t: fn})`, `Wait(until, clip)`),
register it in `STORIES` with the places it needs in `NEEDS`, add those `Place`s to the level
and set `CivilianData.role`. For example, a busker:

```python
def busker(me, life):
    spot = life.place("busk")
    yield Walk(spot.x, spot.z, label="finding a good spot")
    yield Turn(spot.yaw)
    while True:
        yield Play(lambda t: clips.beckon(t), 8.0, label="singing for coins")
        yield Play(lambda t: clips.talk(t, me.seed), 3.0, label="chatting up the crowd")
```

## Roadmap

- [x] **Modular Sector Architecture**: Multi-sector expansion (Staging Bay $\to$ Airlock Corridor $\to$ Datacore Vault with coolant trenches and catwalks).
- [x] **Surveillance Hazards**: Wall-mounted security cameras with dynamic laser sweep frustums and ray-AABB wall truncation.
- [x] **Environmental Distraction AI**: Interactive vending machine triggering arrival-gated 15-second guard investigation routines.
- [x] **Intelligent A* Pathfinding**: Built-in $0.4\text{m}$ grid A* navigation with raycast string-pulling smoothing and step-level obstacle sliding for NPCs.
- [x] **Master 7-Point Neon Lighting**: 4x boosted ambient, diffuse, and specular illumination across all sectors.
- [x] **Restructure Phases 0-3**: package layout, single geometry core, render/simulation split, DetectionSystem + headless test suite.
- [x] **Restructure Phases 4-5**: event bus (`NoiseEmitted`, `TerminalHacked`), reactive doors, guard behaviour as FSM state classes, duration-correct animation clips.
- [x] **Restructure Phase 6**: the level is data (`levels/facility.py`): props, colliders, walk regions, lights, doors and entity placements, with consistency tests.
- [x] **Restructure Phase 7**: display lists for all static geometry and PyOpenGL error checking off by default (`CYBERDUKE_GL_DEBUG=1` re-enables): ~27,900 → ~280 GL wrapper calls per frame.
- [x] **Restructure Phase 8**: pathfinder budget scales with the level, blocked-cell memoisation, no corner cutting, reachable-point arrival for blocked goals.
- [x] **Remote Quickhacking Suite**: scanner pulse, crosshair focus, per-target menus with data-driven restrictions, timed uploads; BLIND / ELECTROCUTE, OVERRIDE DOORS / STEAL DATA / OVERLOAD, FORCE OPEN, SHUT DOWN / HIJACK FEED, DISTRACT - every effect wired through the existing perception, detection, door and noise code.
- [x] **Factions & Networks**: one network per faction; breaching any computer exposes every member across the map; `secured` computers stay private. The vault bulkhead joins the blast door so both door security levels exist in the level.
- [x] **Camera feeds**: free pan/tilt when hijacked, linked feeds (`[<]`/`[>]`), scans and hacks from the lens, dead cameras hang from their swivels; long, gestured distractions (guards bang on shorted machines, stare at fried computers).
- [x] **Sector 3: Fungal Server Greenhouse**: a domed maze of bio-server hedges and bioluminescent fungus on a second faction network, with waypoint patrols, a camera hop chain, air-gapped control nodes, a cipher-keyed door in from the vault, a gated short cut, and the street exit.
- [x] **Pause & graphics presets**: `[Esc]` pause menu that freezes the world completely, LOW/MEDIUM/HIGH quality presets applied live, and GPU auto-detection on startup.
- [x] **Terminal interface**: computers answer with their own screen (unlock doors / hijack cameras / short circuit / steal data) instead of a single hold-to-win progress bar.
- [x] **Dressed set**: seeded, collision-checked trash in every room and fungus only in the greenhouse, per-desk layouts, a gloomy greenhouse lit by blue bioluminescence with soft-edged light pools, puddles and haze (`soft_disc`), electrocutions that convulse standing up and fall without passing through walls.
- [x] **Visual pass**: seamless, world-scaled 128 px textures with mipmaps (brick, concrete, asphalt, wood for the alley), cartoon faces with the eyes where the visor is, a darker night, 24-bit / logarithmic depth (no flickering lines through distant signs), colliders as tall as what they hold, one staff door on both sides of the exit (that waits, shut, for you to walk up to it), fungus light that lies on the surfaces it lights, and rag-doll knockouts that go down any way and end on the floor.
- [ ] **Levelling**: start with `Player.known_hacks` empty and `learn()` hacks from stolen intel / experience; RAM cost and cooldowns per hack.
- [ ] **Network reactions**: guards notice a dead camera or a downed colleague and raise the alert; a breached network can be traced back to the netrunner.
- [ ] **Spatial Partitioning**: Replace linear obstacle checks with a 2D Quadtree / BVH for high NPC and obstacle density.
- [ ] **Expanded Audio Feedback**: Procedural or synthesized footstep volume thresholds, camera hums, and alarm klaxons.
- [x] **Inventory & Cyberware**: an inventory for everybody, a [Tab] deck screen with an icon grid and item details, cyber mods installed with EQUIP (a longer ping, faster close-up hacking, a signal damper), stolen data carried as data shards.
- [ ] **More cyberware**: spend eddies and stolen intel at vendors; locomotion upgrades (double jumps, optical camo); mods with drawbacks.
- [x] **Level 2: Night Market Alley**: the exit zone loads a second level (the player keeps the deck, the renderer follows), a tenement alley with two stalls and their merchants, a crowd that parts around the player, and the main road beyond.
- [x] **Street life**: every person in the alley has a story (sushi chef, junk dealer, office worker, tinkerer, courier, resident, smoker, stroller) played out as jobs in a sensible order, with real objects changing hands and the people meeting at counters, doors and the wall.
- [x] **Street life, polished**: queues that form and move up, people who walk round each other (and give way) instead of walking on the spot, walks that end where they were going; knocking on a door rather than through it, parcels set down and picked up with a bend of the knees and back; sushi eaten a piece at a time with the chopsticks' tips, a cigarette held out from the face with the ash flicked off; faces with expressions; and a vestibule that ends at the yellow door you leave by.
- [x] **No winning**: the run no longer ends - where the world is not built yet it ends in a barricade (the ends of the main street).
- [x] **Being caught, with consequences**: a red YOU WERE CAUGHT screen and a fade to black; in the facility a new game, in somebody's home being thrown out of it (and losing what you took).
- [x] **Talking, and a first quest**: captioned conversations on `[E]`; Nix's "Something Special" - hack the access panel of number 16, get past the home camera and Dell, take the data stick from the nightstand, get paid in cyberware; a QUESTS tab with pictures taken in the game world.
- [x] **Number 16**: a poor flat behind the alley wall - living room, kitchen, bedroom, bathroom - with a hinged door, an access panel, a home camera and its owner living his life in it.
- [x] **Start menu, saves & settings**: a start menu over the living street, save / load from the menus into never-overwritten, date-named JSON files in `~/Unprepared/Saves`, fullscreen, and settings remembered in `~/Unprepared/settings.json`.
- [x] **Clean surfaces**: a z-fighting audit over every level (`render/audit.py`) and every hit fixed - number 16's ceiling, floor and mirror, catwalks, door frames, trusses, decals; fog banks that never jump as you walk; a forward-falling rag doll that lands flat instead of floating on an arm.
- [x] **Smoke**: a cigarette trail that stops on the drag, a puff blown out after it, and a cherry that glows dull and flares bright.
- [x] **The main street, homes and hours**: the alley opens onto a pavement with a car at the kerb, the METRO and the UNDERPASS; a flat for everybody of the street (Nix at number 17, six round the corner); a 24-hour clock on the HUD, a smoggy day and a neon night; schedules - the stalls open and shut, people go to work, come home and sleep in their beds, Dell gets a can from the new vending machine in the alley (and comes out to it when it shorts - a distraction just for him).
- [x] **People, rounded**: heads, limbs, boots and torsos rounded off; hands with fingers that close round what they hold (a can that tips up at the lips, a phone under the thumb, a cigarette between two fingers) and make shapes of their own (a fist, a pointing finger, typing); people of believable heights who fit their beds and keep their arms out of the walls; the netrunner pushes people out of the way instead of getting pinned; the wok the right way up, the knife's edge down, an A-frame CLOSED sign facing the alley.
- [x] **Lives, not props**: nobody on the street appears or disappears - leaving the alley they walk home and let themselves in (the office worker works from home); a fade to black and back at the biodome's last door; no day counter on the clock; an owner who only notices a stranger once the meter is red; a vending machine that stays broken until Dell, out on an errand, kicks it back to life (a distraction that keeps him out longer).
- [x] **Homes, furnished**: every flat done up its own way and full of its owner's things - windows with views, blinds and curtains, pictures, shelves and books, kitchens with cupboard doors, sinks and taps, food about, tiled bathrooms with a real toilet and a basin on a pedestal; televisions with channels (the news, cartoons, music, adverts, the crawling snow of the dead ones) that people flick through with the remote.
- [ ] **The streets, properly**: more than one alley; vendors who sell (spend the stolen intel), fixers who hand out the next job, and a way back into corporate property.
