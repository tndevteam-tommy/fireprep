# FirePrep

> *A castle-age stealth-and-theft RPG where the guards are smarter, the forests are alive, and every rooftop is a way in.*

**FirePrep** is a Python / pygame / OpenGL game built by rebuilding **Unprepared** - a rogue-like stealth and theft RPG set in the walled hill town of Emberhollow - around the strongest systems from three sibling projects:

| Donor | What FirePrep takes from it |
| --- | --- |
| **CyberDuke** | Guard AI, stealth and detection, the humanoid character rig, character animation, hands and models |
| **Firestarter** | Natural environments, flora and plant life, weather and seasons, lighting, atmosphere, texture realism |
| **Hardcore** | Collision-accurate parkour, ledge and edge climbing, climbing animation, multiplayer networking |
| **Unprepared** | **The base game** - world, NPC routines, justice system, economy, quests, saves, tests, tooling |

**Status:** planning / pre-alpha. This document is the project charter and sprint plan. Nothing described under *Planned* exists yet.

**License:** MIT, Copyright (c) 2026 Thomas R. Nifong (see `LICENSE.md`).

---

## Table of Contents

1. [The end goal](#1-the-end-goal)
2. [Design pillars](#2-design-pillars)
3. [The four source projects](#3-the-four-source-projects)
4. [What FirePrep adds to Unprepared](#4-what-fireprep-adds-to-unprepared)
5. [Target architecture](#5-target-architecture)
6. [Key technical decisions](#6-key-technical-decisions)
7. [Roadmap and milestones](#7-roadmap-and-milestones)
8. [Sprint breakdown](#8-sprint-breakdown)
9. [Testing and quality strategy](#9-testing-and-quality-strategy)
10. [Risks and mitigations](#10-risks-and-mitigations)
11. [Non-goals](#11-non-goals)
12. [Open questions](#12-open-questions)
13. [Getting started (provisional)](#13-getting-started-provisional)
14. [Credits](#14-credits)

---

## 1. The end goal

Sneak through Emberhollow and the realm around it as a thief who is no longer limited to the street.

When FirePrep is finished:

- **Everyone looks and moves like they belong in a castle-age world.** All 53 NPCs and the player share one animated humanoid rig, dressed in rebuilt wool, linen, leather, mail and plate, with faces, hair, hands and gestures that fit the work they do (a smith swinging a hammer, a baker kneading, a guard leaning on a wall).
- **Guards perceive the world properly.** Vision cones with real occlusion, order-independent detection, acoustic investigation, patrol and wall-lean behaviours - layered under Unprepared's existing memory, report, hearsay and justice systems, which stay intact.
- **Every building is a climb.** Timber-frame houses, the chapel's belfry, the manors' cornices, the watchtower, the curtain wall and the old keep expose generated holds, ledges, eaves and parapets. You can hang, shimmy, mantle, and drop onto a roof - and guards have to start looking up.
- **The world is alive.** Forests of procedurally grown trees, seasons, weather, wildlife, physically-motivated firelight and synthesised ambient audio replace Unprepared's current natural environments and raise texture realism across the realm.
- **You are not alone.** A persistent, server-authoritative realm lets small groups (target: tens of players per realm shard, not thousands) share Emberhollow as rival thieves, crews or thief-takers - with Hardcore's proven networking, bots and match modes adapted to heist play.

The tone is **Skyrim-inspired castle-age**: stone, thatch, timber, torchlight, wool and iron. FirePrep is inspired by that aesthetic; all assets are original and procedurally generated, consistent with all four donor projects, none of which ship art files.

---

## 2. Design pillars

1. **Simulation never imports pygame or OpenGL; presentation never mutates simulation state.** Unprepared and CyberDuke both enforce this and it is what makes a 1,100-test headless suite possible. FirePrep keeps it as a tested rule (it is also what makes a dedicated multiplayer server cheap).
2. **Data-driven worlds.** Levels, buildings, outfits, climb profiles and quests are data. New houses get lights, holds and doors automatically.
3. **Everything is a skill roll against a challenge.** Unprepared's single progression rule (level + gear vs. challenge) extends to climbing, rather than bolting on a second system.
4. **Stealth and traversal are one game.** Noise, light, vision, weather and hold-grip all feed the same perception model.
5. **Procedural, zero-art-asset.** Textures, meshes, animation and audio are generated; the first launch synthesises what it needs.
6. **Tested, not eyeballed.** Z-fighting audits, traversability flood-fills, golden replays, loopback network tests and frame-time budgets gate every sprint.

---

## 3. The four source projects

### 3.1 Unprepared - the base game

A rogue-like stealth and theft RPG (pygame + PyOpenGL, GLSL 1.20 with a fixed-function fallback). Emberhollow is a hillside market town inside a curtain wall: shops, an inn, a smithy, a bakery, a chapel, cottages, two manors and a watchtower, with a surrounding realm of a village, the Thornwood, a lake, the old keep Greyhold, and three caves. 53 NPCs (33 in town, 20 in the country) live on daily and multi-day schedules; they talk, trade, sleep, lock their doors and remember what you did.

Strengths FirePrep inherits: pick locks (six tiers), pickpocket, loot, fence goods; a justice system with witnessed vs. reported memories, hearsay convictions, fines, cells and jailbreaks; a tutorial; a single XP/odds rule for all skills; data-driven levels; baked static geometry in 10-unit chunks; graphics presets with a first-launch benchmark; atomic, versioned saves; a layered test suite (~1,100 unit tests, 34 scripted golden replays, parity tests, display tests, a visual audit).

Gaps FirePrep fills: boxy NPC and gear models, simple per-NPC suspicion, no jumping or climbing (even the watchtower is stairs only), a flat procedural natural environment, and strictly single-player.

### 3.2 CyberDuke - guards, stealth, characters

A retro-futuristic low-poly stealth game (Python, pygame, OpenGL with a modern VBO/GLSL backend and a legacy fixed-function backend).

**Taken:**
- **Humanoid body rig** (`animation/rig.py`, `shapes.py`, `batch.py`): one description of the body drives both the renderer and the collision skeleton; rounded parts; roughly 60 parts per person drawn in one batch with distance LOD.
- **Hands and grips** (`animation/hands.py`): palm, two-bone fingers and thumb; what a hand holds decides how it grips; a socket system moves props between hips, chest and hands.
- **Procedural pose and clip system**, faces and looks (per-skin-tone painted faces with expressions), hair and clothing kits (`BodyKit`).
- **Guard AI and detection:** a state-machine guard (patrol, wall-lean, investigate, inspect, scan), a human-scale vision cone with ray-box occlusion, an order-independent `DetectionSystem`, `NoiseEmitted` events with inspect duration and gesture, A* with raycast string-pulling and agent-inflated grids, a spatial hash.
- Engineering patterns: `DrawContext` render-backend abstraction, event bus, z-fighting audit, traversability and rag-doll tests.

**Left behind:** netrunning, scanner and quickhacks, terminals, security cameras, cyber mods, vending machines, traffic, the metro, neon lighting and the sci-fi facility levels.

### 3.3 Firestarter - the living world

A high-fidelity wilderness thermodynamics and bushcraft simulation (Python, ModernGL / OpenGL 3.3 core, pygame-ce, numpy, scipy, Pillow). Fire is modelled from published combustion science; around it sits forestry, seasoning, a seasonal ecosystem and a physically based, painterly renderer.

**Taken:**
- **Procedural forest:** pines, cedars, oaks and birches built from leaf and needle cards painted into a Pillow-generated atlas, wind sway, seasonal colour, snow cover, and a tree lifecycle (seed, sapling, canopy, death, rot, fungus).
- **Sky and weather:** real solar geometry, golden hours, twilight, stars, mist, rain, snowfall, wet ground; four seasons.
- **Renderer features:** HDR, ACES tone mapping, bloom, FXAA, soft shadows, eye adaptation, particles, flame and smoke rendering.
- **Procedural audio:** a synthesised sound bank with a live spatial mixer.
- Its procedural-texture and tree-card pipelines for "overall texture realism".

**Left behind (core scope):** the full combustion/moisture thermodynamic solver, bow-drill and friction-fire crafting, the smartphone and store, and the credit economy. Fire *visuals* are in scope; fire *chemistry* is a stretch item (see [Backlog](#8-sprint-breakdown)).

### 3.4 Hardcore - parkour and multiplayer

A first-person, momentum-driven parkour engine (Python, pygame, PyOpenGL 3.3 core). A kinematic character controller (swept-AABB, variable jump, sprint build-up, wall kick), a parametric ledge-rail graph extracted from geometry, a soft-lock reticle, a climbing state machine (hang, shimmy, corner turns, climb-up, mantle, dyno, wall eject, climb-down, landing roll, flips), planted-hand IK for arms, fingers and legs, a building generator with facades, courses, balconies, scaffolds and fire escapes, and a **multiplayer stack**: a JSON-lines TCP protocol, a server-authoritative listen/dedicated server, interpolated remote players, bots at three skill levels, and three match modes (emblem hunt, capture the flag, zombie tag) for up to 15 players.

**Taken:** the controller, ledge graph and reticle, climbing FSM, arm/hand/leg IK and hand planner, the building-hold generation logic, the gamepad stack, and the network stack, bots and match-mode patterns.

**Left behind:** the abstract grid-line arena look, the playground training zones and the neon "emblem crystal" presentation.

---

## 4. What FirePrep adds to Unprepared

### 4.1 Characters, guards and stealth (from CyberDuke)

| Area | Unprepared today | FirePrep |
| --- | --- | --- |
| Bodies | Box body, head and eye spheres | One rounded, articulated humanoid rig for every NPC and the player |
| Clothing | Gear catalogue rendered as models on a box figure | Castle-age wardrobe rebuilt on rig parts: tunics, hose, cloaks, hoods, aprons, sashes, scarves, kettle helmets, breastplates, boots |
| Textures and styles | Flat procedural tiles | Wool, linen, leather, dyed cloth, mail, steel; period hair and beards; per-skin-tone faces with expressions |
| Animation | Basic walk and pose | Procedural clips for every routine (hammering, kneading, sweeping, bell-ringing, drinking, dicing), talking gestures, guard gestures, grips on every held item |
| Detection | Per-NPC suspicion with sneak rolls | Order-independent `DetectionSystem`, vision cones, ray-box occlusion, vertical awareness; sneak skill becomes a modifier |
| Guards | Schedule-driven patrol, posts, grudges | Guard FSM (pace, patrol route, wall-lean, investigate, inspect, scan) driving the same rounds and posts |
| Noise | Noise multiplier, wall attenuation | `NoiseEmitted` events with inspect duration and gesture; castle-age distractions (thrown coin, rattled chain, kicked barrel) |

**Unchanged on purpose:** the justice layer - memories (witnessed vs. reported), hearsay convictions, shopkeeper confrontations, fines, jail, curfews, the torch law and trust/trespass areas.

### 4.2 Parkour and buildings (from Hardcore)

- First-person movement upgraded to a kinematic controller with variable jump, sprint build-up and wall kick, working on Unprepared's **hills, stairs, trails and shallows**.
- **Ledge and crevasse graph** generated from `BuildingDef`s, walls and props: sills, lintels, timber beams, jetties, corbels, quoins, eaves, ridges, parapets, battlements, balconies, chimneys, buttresses, trellises.
- **Climbing**: hang, shimmy, corner turns, climb-up, mantle, dyno, wall eject, climb-down, landing roll; first-person IK arms and legs; third-person climbing clips on the CyberDuke rig for other players and NPC use.
- **Dynamic buildings:** a seeded facade generator gives every building a `climb_profile`; roofs, upper windows and shutters become entry routes with lock tiers; the watchtower, chapel belfry, manors and Greyhold become set-piece climbs.
- **Climbing aids (proposed):** craftable rope-and-grapple, hooked ladder and plank, made at the smithy anvil. A rope left hanging is evidence; guards report and cut it.
- **Climbing skill** on Unprepared's single XP/odds rule: speed, grip, noise and weather penalties all scale with it. Wet stone is harder to hold.
- **The hard design tension, kept on purpose:** the torch law (carry a lit torch after dark) vs. needing both hands on the wall. A stowed lit torch still lights you; climbing dark breaks the law if you are seen.

### 4.3 The living world (from Firestarter)

- Forest, undergrowth, moss, rocks, fallen logs and fungi replace today's natural environments; the Thornwood and Deepwood become genuinely dense woods (and trees become climbable holds).
- HDR lighting pipeline, real sun and moon paths, golden hour, mist, stars; inverse-square firelight for torches, lanterns, forge, oven and hearths; chimney smoke that tells you who is home.
- Seasons, rain, snow, fog and wetness with **gameplay hooks**: rain masks noise, fog shortens sight, snow leaves tracks, wet stone is slippery, wind covers rustle.
- Wildlife and ambient audio (birds by day, crickets and owls at night, gulls on the shore, startled crows that make a guard look up).
- A texture-realism pass over stone, timber, thatch, plaster, cobble and terrain.

### 4.4 Shared realm (from Hardcore networking)

- Server-authoritative persistent realm running the headless Unprepared `World`; clients send intent and pose, receive chunk-based interest-managed snapshots.
- Interpolated remote avatars (CyberDuke rig, Hardcore poses), names and crews.
- Authoritative locks, containers, pickpocketing, trade and doors; per-player crime memory.
- Hardcore's match modes re-themed as heist play: **Heist Hunt** (emblem hunt: relics hidden on rooftops and in hiding places), **Banner Run** (capture the flag: steal a rival crew's strongbox), **Thief-Taker** (zombie tag: one hunted thief against everyone else). Hardcore's Simple / Complex / Hard bots become AI rival crews.
- Accounts, persistence, validation against the human-performance envelope, and dedicated-server tooling.

---

## 5. Target architecture

FirePrep keeps Unprepared's layering and adds donor subsystems at the seams.

```
            +--------------+  intents   +----------------+   reads   +----------------+
 pygame --->| input/       |----------->| game.Game      |<----------| rendering/     |---> OpenGL
 gamepad    | controller   |            | (headless)     |           | SceneRenderer  |
            +--------------+            +-------+--------+           +-------+--------+
                                                |                            |
                         +----------------------+---------------+            |
                         |            simulation (no GL)         |            |
                         |  world/ ledge graph (HC)  entities/   |            |
                         |  perception & guards (CD)  justice    |            |
                         |  parkour FSM + KCC (HC)  ecology (FS) |            |
                         +--------------------+------------------+            |
                                              | state / intents               |
                                     +--------v---------+            +--------v--------+
                                     | net/ (HC)        |            | DrawContext     |
                                     | server <-> client|            | backends:       |
                                     +------------------+            |  compat (GL2.1) |
                                                                     |  core33 (GL3.3) |
                                                                     +-----------------+
```

Proposed package layout (additions to Unprepared's existing layout in **bold**):

```
main.py
fireprep/
  config.py  graphics.py  perf.py  ui.py
  core/            math3d, messages, daynight, dialogue, **events (EventBus)**
  entities/        Item, Door, Player, NPC, SkillTree (+Climbing), equipment
  world/           levels, World, terrain, navigation (A* + **string-pulling, spatial hash**)
    **ledges/**    rails, ledge graph, hold kinds, rail extraction from BuildingDef/props/trees
    **buildgen/**  facade/course/roof generator, climb profiles, climbing aids
    **ecology/**   flora lifecycle, seasons, weather state, fauna
  systems/         movement, collision, targeting, stealth, justice, commerce, crafting, progression
    **perception/** DetectionSystem, vision, noise events
    **parkour/**   KCC, input buffer, climbing FSM, body/reach model, acrobatics (landing roll)
  **ai/**          guard FSM states, patrol routes, bots (from Hardcore, re-themed)
  **animation/**   rig, shapes, hands/grips, pose + clips, IK (arms, legs, fingers), hand planner
  **net/**         protocol, server, hosted, client, interpolation
  game.py  quests.py  attract.py  app.py
  input/           key bindings, gamepad stack (from Hardcore)
  rendering/
    **backends/**  compat (GLSL 1.20 / fixed function), core33 (GL 3.3 core)
    **flora/** **sky/** **fire/** **post/** (HDR, bloom, shadows)   gear, models (rig batch), scene, hud
  **audio/**       procedural sound bank + spatial mixer
assets/textures/   generated PNGs and foliage atlas
tools/             textures, benchmark, visual_audit, **ledge_audit**, **bot_levels**
tests/             unit, regression (goldens), parity, display, **net**, **traversability**
docs/              ARCHITECTURE.md, **ART_BIBLE.md**, **PORTING_LOG.md**, **ADR/**
```

Enforced by `tests/test_architecture.py`: nothing outside `rendering/`, `input/`, `audio/` (playback only) and `app.py` imports pygame or OpenGL.

---

## 6. Key technical decisions

These come from reading the four projects' documentation side by side. Each is an **Architecture Decision Record (ADR)** to be confirmed in Sprint 0 against the actual source, then written to `docs/ADR/`.

| # | Decision | Why it needs deciding | Recommendation |
| --- | --- | --- | --- |
| ADR-1 | **Renderer baseline and GL binding.** Unprepared: GLSL 1.20 + fixed-function fallback. CyberDuke: legacy + GLSL 1.20 backends behind a `DrawContext`. Hardcore: PyOpenGL 3.3 core. Firestarter: **ModernGL** 3.3 core (HDR, bloom, shadows). | Three of four use PyOpenGL; Firestarter's pipeline needs 3.3 core features; low-end machines still need to run. | Adopt a `DrawContext`-style backend seam (CyberDuke's pattern). Keep the existing GLSL 1.20 path as `compat` for *low/medium*; add `core33` for *high/ultra*. Standardise on **PyOpenGL** and port Firestarter's shaders from ModernGL (GLSL is portable; the wrapper code is thin). |
| ADR-2 | **Controls.** Unprepared: `E` interact, `Tab` equipment, `T` wait. Hardcore: `E` / `Space` grab. Firestarter and CyberDuke also use `Tab` for a phone / deck. | Direct key conflicts. | `Space` = the one parkour button; `E` stays interact; context priority decides when both could apply (the ledge reticle wins when locked on). Unprepared's `Tab` and `T` unchanged; donors' phone and deck screens are not ported. Everything rebindable. |
| ADR-3 | **Scale and units.** Unprepared: NPCs 1.74 m, eyes 1.6 m, player eye 1.55 m (0.9 m crouched). Hardcore: 1.80 m athlete, 2.25 m standing reach, hold gaps <= 2.15 m, 3 m storeys. | Hold spacing and reach are derived from a 1.80 m body; Unprepared's buildings were laid out for its own scale. | Keep **Unprepared's heights canonical** and rescale Hardcore's reach envelope proportionally in one shared `body` config; verify manors' and watchtower storey heights in Sprint 0. |
| ADR-4 | **Stealth migration strategy.** Unprepared's goldens encode current behaviour. | Replacing perception will change replays. | Feature flag `stealth_backend = legacy / cyberduke`; run both until Sprint 7, then re-record goldens after a reviewed diff. |
| ADR-5 | **Package, data directory and save namespace.** CyberDuke's docs show its saves and settings living in `~/Unprepared` (or `$CYBERDUKE_USER_DIR`). | Ported code could write into FirePrep's - or the original Unprepared's - user directory. | Rename to `fireprep`, new per-user directory and `FIREPREP_DATA_DIR`, keep `UNPREPARED_DATA_DIR` as a deprecated alias, save schema v10 with migration from v9. |
| ADR-6 | **Network authority model.** Hardcore's server is authoritative for emblems, flags, tags and scores, but clients report their own positions; it has no accounts, NAT traversal or anti-cheat. | A persistent shared realm needs far more. | Server runs the headless `World`; clients send intents/poses; server validates against the human-performance envelope. Honest target: **dozens of players per shard**, not MMO-scale populations. |
| ADR-7 | **Fire fidelity.** Firestarter's thermodynamics solver is its heart, but the user-facing goal is environment and texture realism. | The solver is expensive and not needed for torches. | Port fire **visuals** and light; defer the combustion solver to the backlog. |
| ADR-8 | **Hold highlighting.** Hardcore glows rails. | Glowing rails clash with a Skyrim-style look. | Setting: off / subtle (chalk, scuffs, worn stone) / bright (Hardcore's glow). Default subtle. |

---

## 7. Roadmap and milestones

Planned as **21 two-week sprints (about 42 weeks)** for a small team (2-4 contributors). Tracks B, C and D can run in parallel after Sprint 2 if there are enough hands; the single-team order below is the safe sequence.

| Milestone | Sprints | Weeks | Playable outcome |
| --- | --- | --- | --- |
| **M0 - Ground Truth** | 0-2 | 1-6 | FirePrep boots as renamed Unprepared, benchmarked, two renderer backends |
| **M1 - New Faces** | 3-7 | 7-16 | Emberhollow populated by rigged, castle-age NPCs with CyberDuke guards |
| **M2 - Up the Wall** | 8-12 | 17-26 | Climb any building; steal Lord Vane's cloak over the rooftops |
| **M3 - Living Realm** | 13-15 | 27-32 | Forests, seasons, weather, wildlife, firelight, new audio |
| **M4 - Shared Realm** | 16-18 | 33-38 | Persistent multiplayer shard with heist modes |
| **M5 - Release Candidate** | 19-20 | 39-42 | Tutorial, quests, balance, performance, packaging |

**Dependencies:** Sprint 12 (stealth x parkour) needs Tracks B and C. Sprint 16 (network) needs the rig (Sprint 3-5) and the climbing poses (Sprint 10). Sprint 13-15 (environment) is largely independent and can slip left.

---

## 8. Sprint breakdown

**Conventions**

- Tags show provenance: **[UP]** Unprepared, **[CD]** CyberDuke, **[FS]** Firestarter, **[HC]** Hardcore, **[NEW]** original work.
- Every sprint ends with a runnable build, a short demo and a review of risks.
- **Global Definition of Done** (applies to every sprint, not repeated below): headless suite green; architecture (layering) test green; visual audit clean; any golden change reviewed; benchmark within budget (target: medium preset no worse than **+15%** frame time vs. the Sprint 0 baseline, unless the sprint states otherwise); `docs/PORTING_LOG.md` and docs updated.
- Dates are relative (Weeks 1-2, ...). Map them to calendar dates when the team and start date are fixed.

### Phase A - Foundations (M0)

#### Sprint 0 - Bootstrap and decisions (Weeks 1-2)
**Goal:** FirePrep exists, is green, and every major decision is written down.

- [UP] Fork Unprepared into `fireprep/`; rename the package, entry point, data directory and environment variable (ADR-5).
- [NEW] Bring donors in as read-only reference (`donors/cyberduke`, `donors/firestarter`, `donors/hardcore`) via subtree or submodule; start `docs/PORTING_LOG.md` (module, origin commit, what changed).
- [NEW] **Source audit:** confirm what the donor documentation claims against the code (module sizes, true dependencies, GL versions, `pygame` vs. `pygame-ce`, `scipy`/`Pillow` needs, test counts).
- [NEW] **License audit:** `LICENSE.md` covers MIT / Thomas R. Nifong; confirm every donor carries the same terms and record third-party notices.
- [UP] **Baseline:** full test suite green, goldens committed, `tools/benchmark.py --json` stored as `benchmarks/baseline_s0.json` for each preset, visual audit clean.
- [NEW] Write and review ADR-1 to ADR-8; confirm player height and storey measurements (ADR-3).
- [NEW] CI pipeline (lint, format, `./run_tests.sh fast`, nightly full suite under `xvfb-run`).

**Done when:** the renamed game plays exactly like Unprepared and the baseline numbers are on file.

#### Sprint 1 - Renderer seam (Weeks 3-4)
**Goal:** scene code draws through an interface instead of a specific GL version.

- [CD] Introduce a `DrawContext` abstraction and backend registry, modelled on CyberDuke's `render/context.py` and `render/backends/`.
- [UP] Wrap the current GLSL 1.20 / fixed-function path as the `compat` backend with no visual change.
- [HC][NEW] Spike a `core33` backend (VAO/VBO/EBO) drawing baked static chunks; port the lighting shader (ambient, sun, moon, up to 16 point lights, exp2 fog, contrast).
- [NEW] Backend selection in `GraphicsSettings` with automatic fallback if shader compilation fails.
- [UP] Extend display tests for pixel parity (tolerance) between backends on fixed camera views.

**Done when:** Emberhollow renders on both backends within tolerance; frame-time comparison recorded.

#### Sprint 2 - Assets, units, saves (Weeks 5-6)
**Goal:** one asset pipeline, one scale, one save format.

- [UP][CD][FS] Merge texture tooling: Unprepared's `gen_textures.py` (89 tiles), CyberDuke's seamless world-scaled tiles (`tile=` metres), Firestarter's Pillow foliage atlas painter. One CLI, deterministic seeds, mip chains and anisotropic filtering.
- [NEW] `config/body.py`: shared body model and reach envelope (ADR-3), referenced by rig, controller and ledge graph.
- [UP] Save schema v10 with a migration from v9 and corruption handling retained.
- [NEW] Graphics presets reconciled: low / medium / high plus *ultra* (Firestarter's tier).
- [UP] Extend `tests/helpers.py` to place things by map position *and* elevation / ledge.
- [NEW] `docs/ART_BIBLE.md` first draft: palette, materials, silhouette rules for the castle-age look.

**Done when (M0):** one command regenerates all textures; v9 saves load; player and building measurements agree.

### Phase B - Characters, guards and stealth (M1)

#### Sprint 3 - Rig and body port (Weeks 7-8)
**Goal:** one rigged, rounded human renders in Emberhollow.

- [CD] Port `animation/rig.py`, `shapes.py`, `batch.py`, the `HumanoidPose` dataclass and `mix_poses`.
- [UP] Replace the box-and-sphere `models.draw_npc` behind a flag; reconcile Unprepared's baked per-outfit recordings with per-frame posed batches (cache static part meshes, pose per frame).
- [UP][NEW] Fit the rig to Unprepared's `Figure` (the person under the gear), eye heights, `facing_yaw`, and sleeping in beds.
- [CD] Distance LOD (mittens and coarser parts further out).
- [NEW] **Performance probe:** all 53 NPCs plus the player in a street view; set per-preset character budgets.
- [CD] Port visual-regression checks for bodies (boots in the floor, limbs through walls).

**Done when:** one hero NPC (Barnaby) animates in place and the crowd probe is within budget.

#### Sprint 4 - Castle-age wardrobe and faces (Weeks 9-10)
**Goal:** every character wears something believable.

- [UP][CD] Map `GEAR_CATALOGUE` slots (head, neck, body, feet, weapon) and outfits onto rig `BodyKit` parts.
- [NEW] Garment set: tunics, hose, cloaks, hoods, aprons (cloth and leather), caps, merchant's sash, widow's scarf, kettle helmet, breastplate, boots, belts, pouches; rare dark clothing keeps its stealth role.
- [NEW] Procedural material styles: wool, linen, leather, dyed cloth in the small-town dye palette, mail, steel, fur.
- [CD] Port the face painter; replace cyber eyes and chrome with period faces, hair and beards, skin tones and expressions (talk, scold, sleep).
- [UP] Rebuild the `[Tab]` equipment stand and gear previews on the new figure.

**Done when:** all 53 NPCs and the player are visually distinct and consistent with `ART_BIBLE.md`.

#### Sprint 5 - Animation, hands and sockets (Weeks 11-12)
**Goal:** routines look like work.

- [CD][NEW] Clip library: idle, walk, run, crouch-walk, turn; sit, eat, drink, sleep; work (hammer, sweep, knead, stir, haul, ring bell, fell timber); talking gestures; guard gestures (scold, "Halt!", point, salute); arrest and cell poses.
- [CD] Port `GRIP_FOR_ITEM`; author grips for mug, loaf, keys, tools, torch, lockpick, coin purse, sword, pike, scroll.
- [CD] Socket system: torch, picks and weapons move smoothly between hand, hip and back.
- [UP] Map Unprepared's NPC schedule acts to clips without touching the schedules themselves.
- [NEW] Third-person player render for mirrors, the `[Tab]` view and (later) multiplayer.

**Done when:** a full day of town life runs with every act animated and no limb clipping in the visual tests.

#### Sprint 6 - Detection core (Weeks 13-14)
**Goal:** a new perception backend behind a feature flag.

- [CD] Port `DetectionSystem` (additive gain, fixed decay, order-independent), vision cones, ray-box line-of-sight against walls, props and closed doors.
- [UP] Re-express Unprepared's rules as policies on the new system: seen vs. heard, torch lights you up, shopkeepers watch their premises, fences ignore skulking, warning grace windows, hearsay, indoors vs. outdoors attenuation.
- [UP] Sneak skill roll becomes a gain modifier (keeps the 55% + 8%/level feel as a tuning reference).
- [NEW] Vertical perception groundwork: pitch limits and target height (standing, crouched, hanging) so roofs can matter in Sprint 12.
- [NEW] `stealth_backend = legacy | cyberduke` flag (ADR-4); scenario goldens run on both and diffs are triaged.

**Done when:** the new backend passes every perception-rule unit test and diffs against goldens are explained.

#### Sprint 7 - Guard behaviours, noise and justice integration (Weeks 15-16)
**Goal:** the guards you meet are CyberDuke's, the law they enforce is Unprepared's.

- [CD] Port the state machine and guard states (pace, patrol route, wall-lean, investigate, inspect, scan); map to the gate watch (Bram, Wenna), tower watch (Aldric), jailer (Hesk), rotating patrol and the lamp-lighter.
- [CD] `NoiseEmitted` events with inspect duration and gesture; keep 25% wall attenuation.
- [CD] A* string-pulling and a spatial hash in `world/navigation.py`, adapted to hills.
- [UP] Justice layer untouched: memories, reports, hearsay, fines, cells, curfews. "Halt!" chases and arrests run through the new FSM with the new animations.
- [NEW] Castle-age distractions: thrown coin or pebble, rattled chain, kicked barrel.
- [UP] Cut the default to `cyberduke`; re-record goldens after a reviewed diff.

**Done when (M1):** a full tutorial and Jae's job play through on the new guards; legacy backend retired or marked deprecated.

### Phase C - Parkour and buildings (M2)

#### Sprint 8 - Kinematic controller and controls (Weeks 17-18)
**Goal:** the player can jump and sprint properly on Unprepared's terrain.

- [HC] Port the KCC: swept-AABB collision, variable jump reservoir, sprint build-up, wall kick, 150 ms input buffer, 120 ms coyote time, fixed 60 Hz step with render interpolation.
- [UP][HC] Reconcile with `apply_movement` / `move_with_collision`: heightfield terrain, graded trails, shallows, winding stairs (the watchtower), "people are solid".
- [UP] Remove the no-jump assumption; keep crouch as the theft stance and "no sprint while crouched".
- [NEW] ADR-2 controls implemented and rebindable; conflicts documented.
- [HC] Jump-distance and speed scripted tests, tuned to town scale.

**Done when:** jumping and sprinting feel right on hills and stairs; no regression in the tutorial or goldens' movement-only scenarios.

#### Sprint 9 - Ledge graph and building rails (Weeks 19-20)
**Goal:** every building face reports its holds.

- [HC] Port `ledge_rail`, `ledge_system`, the body reach model and the soft-lock reticle with climb-down prompt.
- [NEW] Castle-age hold kinds: sill, lintel, beam, jetty, corbel, quoin, eave, ridge, parapet, battlement, balcony, chimney, trellis.
- [NEW] Rail extraction from `BuildingDef`, walls, props and terrain; gabled and hipped roofs (Hardcore's flat tops do not cover this).
- [UP] Reuse the spatial index; store rails in the existing 10-unit chunks.
- [HC][NEW] Traversability audit: flood-fill every reachable hold chain with the real movement code; `tools/ledge_audit.py`.
- [NEW] Hold-highlight setting (ADR-8).

**Done when:** a player can be placed at Barnaby's store and *a test proves* every wall reaches the roof or declares itself unclimbable.

#### Sprint 10 - Climbing FSM, arms and legs (Weeks 21-22)
**Goal:** climbing looks and feels right in first person and third.

- [HC] Port the state machine: hang, shimmy, corner turns, climb-up, mantle vault, dyno, wall eject, climb-down, landing roll; flips as an optional "acrobat" ability.
- [HC][CD] First-person arms and legs (two-bone IK, finger solver, hand planner); castle-age forearms (sleeves, gloves, gauntlets) matching the player's skin and gear; one shared finger/grip model with CyberDuke's hands.
- [NEW] **Climbing skill** on Unprepared's XP/odds rule: speed, grip, noise.
- [CD][HC] Third-person climbing clips on the rig for other players and NPC use.
- [HC] Climb flow and sprint carry-over tuned (Hardcore is deliberately "a step above elite" for pace).

**Done when:** a scripted climb from street to the store roof and back runs in both views with correct hand placement.

#### Sprint 11 - Dynamic buildings and climbing aids (Weeks 23-24)
**Goal:** the town is built to be climbed, and the player can change that.

- [HC] Port the building generator logic (floors, facades, courses at sill/head heights so no hold gap exceeds the reach limit, balconies, rooftop structures, scaffolds).
- [NEW] Re-theme: timber-frame, stone, thatched cottages, jetties, shutters, quoins, chimneys, buttresses, wall-walks; `climb_profile` per `BuildingDef`; seeded variation.
- [NEW] Entry points: shutters, upper windows, roof hatches with lock tiers; occupants and guards react.
- [NEW] Rebuild the watchtower, chapel belfry, manors and Greyhold with climbable exteriors.
- [NEW] **Climbing aids** (proposed): rope-and-grapple, hooked ladder, plank; forged at the anvil; ownership and evidence rules; guards cut ropes.
- [UP] Re-run the z-fighting and clipping audit over every changed building.

**Done when:** at least five building styles generate valid climb routes and the audit is clean.

#### Sprint 12 - Stealth x parkour integration (Weeks 25-26)
**Goal:** climbing is a stealth decision, not a separate game.

- [NEW] Noise model for climbing, landing, rolls and boots (soft vs. shod), with wall muffling.
- [CD][NEW] Guards look up: vertical field of view, roof sightlines, hanging-body sight target; lighting query (`World.is_dark_at`) for shadows and moonlight.
- [UP][NEW] The torch-law tension: stowed lit torch still lights you; climbing dark breaks the law if seen.
- [NEW] Chase rules: guards pursue on the ground and ladders; the grudge decays; fall and landing rules.
- [NEW] Set-piece routes: watchtower climb, belfry, manors' upper floors, curtain wall.
- [NEW] Bot-driven reachability runs (adapt Hardcore's `bot_levels` harness) to find unreachable or trivial routes.

**Done when (M2):** you can steal Lord Vane's ravenfeather cloak entirely by rooftop, and be caught by a guard who looked up.

### Phase D - The living realm (M3)

#### Sprint 13 - Atmosphere, lighting and fire visuals (Weeks 27-28)
**Goal:** the same town, lit like a place.

- [FS] HDR pipeline on the `core33` backend: ACES tone mapping, bloom, FXAA, soft shadows, eye adaptation; presets decide which stages run.
- [FS] Real sun and moon paths (choose a latitude for the realm), golden hour, twilight, stars, mist; keep Unprepared's fractional-hour API.
- [FS] Fire visuals (flame sheets, sparks, smoke by fuel) with inverse-square firelight for torches, lanterns, forge, oven and hearths; chimney smoke marks occupied homes.
- [UP] Preserve per-preset light caps and room-bounded lights that spill through open doors.
- [NEW] Torch radius feeds detection (existing `torch_light_reach`).

**Done when:** a dusk-to-night cycle looks right on *high* and degrades gracefully on *low*; nothing regresses detection.

#### Sprint 14 - Flora, terrain and materials (Weeks 29-30)
**Goal:** the woods are woods.

- [FS] Procedural trees (pine, cedar, oak, birch) from leaf-card atlases with wind sway; undergrowth, grass, ferns, moss, rocks, fallen logs, fungi.
- [FS][NEW] Tree lifecycle visuals (sapling to canopy to rot) at a sparse, world-scale cadence.
- [UP][FS] Terrain texturing: rock, sand, snow, mud and leaf litter blended on the heightfield; shorelines, stream and lake.
- [NEW] Dense Thornwood and Deepwood replace today's "impassable thicket"; **tree branches as ledge rails** (reuse the Sprint 9 extractor).
- [FS] Instanced, chunk-baked forests with distance LOD; draw-distance budget re-measured (Unprepared *high* draws 700 units).
- [UP][NEW] Castle-age material realism pass: normal and roughness maps generated procedurally for stone, timber, thatch, plaster, cobble; consistent texel density.

**Done when:** the full realm renders within budget on *medium* and *high*; trees are climbable.

#### Sprint 15 - Seasons, weather, fauna and audio (Weeks 31-32)
**Goal:** the world changes over time and affects play.

- [FS] Calendar and seasons (configurable length); seasonal palettes, snow cover, falling leaves.
- [FS][CD] Weather state machine: drizzle, rain, storms, snow, fog; wet ground. **Hooks:** rain masks noise, fog cuts sight, snow leaves tracks, wet stone lowers grip.
- [NEW] Fauna: birds by day, crickets and owls at night, deer in the Thornwood, fish in the lake, gulls on the shore; startled crows alert guards.
- [FS] Procedural audio: port the sound-bank synthesiser and spatial mixer; footsteps by surface (cobble, dirt, leaf, snow, wood), fire, wind, rain; add bells, hammering and crowd murmur. Sounds emit `NoiseEmitted`.
- [UP] The `T` wait works with seasons and weather.

**Done when (M3):** a four-season playthrough is stable; weather measurably changes stealth outcomes in scripted scenarios.

### Phase E - Shared realm (M4)

#### Sprint 16 - Network foundation (Weeks 33-34)
**Goal:** two players see each other in Emberhollow.

- [HC] Port the JSON-lines protocol (with a version handshake), the selector-thread server, the hosted child process and dedicated-server CLI, and the client with clock-offset estimation and 0.1 s interpolation delay.
- [NEW] Server runs the headless `World`; clients send intent and pose. NPCs, doors, items and the clock are server-owned.
- [NEW] **Interest management** by the existing 10-unit chunks; delta snapshots; per-client bandwidth budget.
- [NEW] Level-hash handshake so clients and server agree on content.
- [CD][HC] Remote avatars on the CyberDuke rig driven by Hardcore poses.
- [HC] Port the network smoothing tests; add a loopback harness for scripted multi-client scenarios.

**Done when:** two clients walk, climb and see the same town clock and NPCs, smoothly, over loopback and LAN.

#### Sprint 17 - Shared-realm gameplay (Weeks 35-36)
**Goal:** theft, law and play are multiplayer-correct.

- [NEW] Authoritative locks, containers, pickpocketing, trade, doors and sleeping with contention rules.
- [NEW] Per-player crime memories and identity (guards track *who*); disguise by outfit as a stretch.
- [NEW] Social: speech-bubble chat, emotes, crews, shared hideout stash.
- [HC] Re-themed match modes: **Heist Hunt**, **Banner Run**, **Thief-Taker**.
- [HC] Bots become AI rival crews with Simple / Complex / Hard skill levels.
- [NEW] PvP rules (opt-in pickpocketing decided with the owner; see [Open questions](#12-open-questions)).

**Done when:** a crew of four and a rival crew complete a Banner Run with bots filling empty slots.

#### Sprint 18 - Persistence and security (Weeks 37-38)
**Goal:** the realm survives restarts and is hard to abuse.

- [NEW] Persistent realm save (world state, clock, shop stock, rented rooms) and per-character saves (inventory, skills, purse, grudges); simple accounts with tokens.
- [NEW] Server-side validation against the human-performance envelope (speed, jump, climb rates), rate limits, item-duplication guards, kick and ban tools.
- [NEW] Realm shards (target 16-32 players); admin config, logging and metrics.
- [NEW] Anti-griefing rules: capped sentences, respawn and eviction safeguards.
- [HC][NEW] Hosting guide covering port forwarding and optional TLS (Hardcore explicitly lacks NAT traversal).

**Done when (M4):** a 24-hour soak test with bots and a rolling set of clients ends with no desync, no duplication and a clean restart.

### Phase F - Release (M5)

#### Sprint 19 - Content, tutorial and accessibility (Weeks 39-40)
**Goal:** a new player can learn the whole game.

- [UP][NEW] Tutorial rewrite: keep the frozen-guard lessons, add a climbing and rooftop lesson and multiplayer onboarding.
- [UP][NEW] Quests: Jae's job gains a roof route; new jobs for the watchtower, Greyhold and the Saltcove; guild contracts.
- [NEW] Climbable caves, ruins and the keep; new hiding places and rooftop loot.
- [HC] Port the gamepad stack (families, labels, deadzones, rumble) and a controller screen.
- [NEW] Accessibility: hold-highlight modes, colour-blind palettes, subtitles for speech bubbles, toggles for flashing effects.
- [UP] `[Tab]` view gains climbing gear and a skills page.

**Done when:** a playtester who has never seen the game finishes the tutorial and the first job unaided.

#### Sprint 20 - Hardening and release candidate (Weeks 41-42)
**Goal:** ship a release candidate.

- [NEW] Benchmark all presets and both backends; fix frame-time, memory and load-time outliers; finalise character and forest budgets.
- [UP] Full regression, golden and display suites; decide whether the frozen-prototype parity suite is retired.
- [NEW] Save-migration tests for every released schema; soak tests.
- [NEW] Packaging, per-platform data directories, `requirements.txt`, reproducible first-launch asset generation.
- [UP] Rewrite `docs/ARCHITECTURE.md`; write `CHANGELOG.md`, credits and third-party notices.

**Done when (M5):** `v1.0-rc1` tagged with known-issues list.

### Backlog / stretch

- Firestarter's combustion solver for craftable campfires, bracket-fungus embers and forge heat.
- Flips and parkour tricks as unlockable perks.
- Disguises by outfit swap; identity-aware guard memory in multiplayer.
- Dungeon interiors, sewers, rope bridges and grapple traverses.
- Mod support for facade styles and quest packs.
- Larger shards and cross-shard travel.

---

## 9. Testing and quality strategy

FirePrep keeps Unprepared's testing philosophy and adds donor-proven checks.

| Layer | What it proves | Origin |
| --- | --- | --- |
| Unit (headless) | Entities, systems, perception rules, justice, routines, saves, menus | [UP] |
| Scenario goldens | Scripted play-throughs replayed headlessly and compared with committed state | [UP] |
| Architecture test | No GL or pygame imports outside presentation layers | [UP][CD] |
| Visual audit | Z-fighting, clipping, floating props, windows off walls, plus new geometry (holds, rigs, forests) | [UP][CD] |
| Traversability | Flood-fill reachability with the real movement code; no traps; NPC agents can path everywhere | [CD][HC] |
| Ledge audit | Every wall either reaches its roof or declares itself unclimbable; no hold gaps above the reach limit | [HC][NEW] |
| Body regression | Boots in floors, limbs through walls, hand placement on holds | [CD][HC] |
| Parity | New stealth backend vs. legacy until cut-over; renderer backends within pixel tolerance | [UP][NEW] |
| Net | Loopback multi-client scripts, interpolation smoothness, protocol fuzzing, soak tests | [HC][NEW] |
| Bots | Headless bot runs to find unreachable or trivial routes and balance problems | [HC] |
| Performance | `tools/benchmark.py` per preset and backend with a regression budget | [UP][NEW] |

**Policies**

- A golden is only re-recorded with a written reason and a reviewed diff (`UNPREPARED_UPDATE_GOLDENS` becomes `FIREPREP_UPDATE_GOLDENS`).
- Every ported module gets an entry in `PORTING_LOG.md` and keeps (or gains) the donor's tests.
- No sprint closes with a failing audit.

---

## 10. Risks and mitigations

| Risk | Impact | Mitigation |
| --- | --- | --- |
| **Python frame time** with 53 posed characters, forests and HDR on top of a 700-unit draw distance | Unplayable on mid-range machines | Per-preset character and foliage budgets set in Sprints 3 and 14; NumPy batching and LOD (CyberDuke's approach); chunk baking; early probes rather than late surprises |
| **Renderer fragmentation** (GL 2.1, 3.3 core, PyOpenGL vs. ModernGL) | Duplicate work, subtle visual differences | ADR-1; backend seam in Sprint 1; pixel-parity tests; keep `compat` for low-end |
| **Scale mismatch** between Hardcore's athlete and Unprepared's town | Unclimbable or trivially climbable buildings | ADR-3; shared body config; ledge audit; generator guarantees hold spacing |
| **Behaviour drift** from replacing perception | Lost balance, broken goldens | Feature flag, dual-run goldens, reviewed re-record (ADR-4) |
| **Control conflicts** | Confusing input | ADR-2; context priority; rebinding; early playtests |
| **Hand and grip systems duplicated** (CyberDuke hands vs. Hardcore finger solver) | Inconsistent first- and third-person hands | One shared grip model, decided in Sprint 10 |
| **Netcode ambition** ("MMO-like") exceeds a Python server | Missed targets, cheating | Honest scale target (dozens per shard), interest management, validation, early soak tests |
| **Scope creep** from four codebases | Never ships | Milestone slices, stretch backlog, explicit [non-goals](#11-non-goals) |
| **Aesthetic IP** (Skyrim comparison) | Legal and creative confusion | Inspired-by only; original procedural assets; `ART_BIBLE.md` rules |
| **Documentation vs. code gaps** (this plan was drafted from the READMEs and design documents) | Wrong assumptions | Sprint 0 source audit before committing to any porting estimate |
| **Licensing** of donors | Redistribution problems | Sprint 0 license audit; third-party notices |

---

## 11. Non-goals

- Netrunning, scanner, quickhacks, terminals, cameras, cyber mods, traffic, the metro and neon visuals (CyberDuke).
- The smartphone, store, credit economy and thermodynamic combustion solver in the core release (Firestarter).
- The abstract training arena look and emblem-crystal visuals (Hardcore).
- Thousands of concurrent players, NAT traversal services and a commercial anti-cheat.
- Shipping pre-made art or audio files.

---

## 12. Open questions

1. **Team and calendar:** how many contributors, and what is the start date? (Sprint durations assume a small team.)
2. **GL binding:** confirm PyOpenGL as the standard and porting Firestarter off ModernGL (ADR-1).
3. **Canonical player height:** keep Unprepared's 1.74 m scale or adopt Hardcore's 1.80 m? (ADR-3)
4. **Player vs. player:** is pickpocketing and robbing other players allowed, opt-in, or never?
5. **Climbing aids:** keep the rope / ladder / plank crafting proposal in scope?
6. **Seasons:** how long should a season last in real time, and should `T` waits advance them?
7. **Flips:** core feature, optional perk, or dropped for tone?
8. **Fauna:** which animals matter most for the first pass?
9. **Donor licenses:** do all three donors carry the same MIT terms as this repository?
10. **Multiplayer target:** is 16-32 per shard the right ceiling, and is LAN-plus-hosted enough for 1.0?

---

## 13. Getting started (provisional)

FirePrep does not yet run. Until Sprint 0 completes, the commands below are Unprepared's and will be renamed:

```
pip install -r requirements.txt          # pygame, PyOpenGL, numpy (+ scipy, Pillow once Firestarter's audio and atlas tools land)
pip install -r requirements-dev.txt      # + pytest
python main.py                           # play
python tools/textures/gen_textures.py    # regenerate procedural textures
python tools/benchmark.py --json out.json
python tools/visual_audit.py
./run_tests.sh [fast]
```

Saves and settings will live in a per-user directory (`%APPDATA%\FirePrep`, `~/Library/Application Support/FirePrep`, `$XDG_DATA_HOME/fireprep`), overridable with `FIREPREP_DATA_DIR`.

---

## 14. Credits

FirePrep builds on four projects by the same author:

- **Unprepared** - base game, world, NPCs, justice, economy, tests and tooling
- **CyberDuke** - guard AI, detection, rig, animation, hands, faces
- **Firestarter** - forests, ecosystem, sky, weather, rendering and audio
- **Hardcore** - parkour, ledge graph, climbing, IK and multiplayer

Licensed under the MIT License. See `LICENSE.md`.
