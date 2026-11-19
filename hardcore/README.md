# HARDCORE — 3D Parkour Engine (Pygame + Modern OpenGL 3.3 Core)

**Hardcore** is a first-person, momentum-driven parkour engine written in Python utilizing **Pygame**, **PyOpenGL (Modern 3.3 Core Profile)**, and **NumPy**. Inspired by the tactile, kinetic traversal of *Dying Light* and *Mirror's Edge*, the engine treats every surface, edge, and crevasse as an interactive climbing entity.

Hardcore renders a full first-person dynamic skeletal arm, hand, and finger rig driven by analytical Two-Bone Inverse Kinematics (IK) and procedural finger-curling solvers. Players witness their hands realistically reach, grip, crimp, shimmy, and push off procedural surfaces, guided by a contextual, soft-lock reticle.

---

## 1. Game Design & Traversal Mechanics

Traversal in *Hardcore* is continuous, tactile, and momentum-conserving. Every maneuver chains directly into the next.

### 1.0 Human Performance Envelope
Every number in the engine is calibrated against what a talented parkour professional can physically do (see `DESIGN.md §6.0`), tuned a step above it where the game needs pace (sprint build-up, climbing, the wall run - §1.11). The player is a 1.80 m athlete: shoulders at 1.45 m, arm reach 0.70 m, flat-footed overhead reach **2.25 m**, elite standing vertical **~0.95 m**, wall-kick climb-up reach **~3.9 m** from a walking leap and **~5 m** from a full-sprint wall run. Arms are rigid two-bone chains with human elbow limits (35°–178°) — they never stretch, never hyper-extend and never grab anything the shoulders can't reach.

### 1.1 Kinetic Movement & Sprinting
* **Locomotion**: WASD movement (or a controller's left stick - analog, in any direction, with the walking speed following the tilt; see §1.9) with human acceleration limits (20 m/s² up to walking pace, 6 m/s² beyond it), friction dampening, and preserved air inertia. Air steering redirects momentum but cannot add speed.
* **Run (`Caps Lock` toggle, `Left Shift` momentary)**: running is a **jog at once** (5.0 → 6.5 m/s in about a quarter of a second) and a **sprint only after a build-up**: the top speed climbs from 6.5 to 8.5 m/s over ~1.6 s of continuous hard running in a steady direction. Easing off to a walk drains the build-up (twice as fast as it fills), a sharp change of direction throws it away and hanging on a wall resets it - but a speed vault keeps it, and topping out of a flowing climb gives it back (§1.11). The field of view widens with *actual* speed (75° at a walk → 90° at full sprint), and momentum carries the wide view through a leap.
* **Arm pump**: the arms swing in opposition to the legs, one cycle per stride — a loose 7 cm swing at a walk, a hard 24 cm pump with the fists held higher and closer at a sprint, fading out in the air — with the wrists staying in line with the forearms throughout.
* **Camera Kinematics**: First-person head bobbing tied to physical distance traveled, Dutch roll/camera tilting (±4°) during lateral leaps and wall push-offs, and mouse pitch clamping [-89°, +89°].

### 1.2 Variable Jump System & Leg Work (`Spacebar`)
* **Dynamic Impulse**: Tapping `Spacebar` applies an initial vertical velocity impulse (v_y = 4.2 m/s → ~0.45 m hop).
* **Variable Hang Time**: Holding `Spacebar` taps into an active vertical lift reservoir for up to 0.25 seconds, enabling precision low hops on tap and a full ~0.95 m leap on sustained hold.
* **Take-off push**: the legs also drive the body along the held direction. From a standstill that push *is* the jump — a **standing long jump of ~2.8 m** forward, ~2.4 m backward (`S` + `Space`, e.g. away from a wall you are facing) or sideways — while at a sprint the legs have nothing more to give (+0.4 m/s) and the leap is carried by momentum. A forward leap leaves the ground with **15 % more forward momentum** (`config.JUMP_FORWARD_BOOST`), so distances sit a little above the elite envelope: standing ~2.8 m, walking ~4.6 m, jogging ~5.5 m, **full sprint ~6.6 m**; a tapped hop from a sprint covers ~4.2 m. Jump height is unchanged. Stepping off a roof with `W` + `Space` clears a 1.5–2 m gap from standing; with no direction held the jump is straight up.
* **Wall Kick / Wall Run**: Running and jumping *into* a wall (with `Space` held) plants a foot on it once per jump and converts the run-up into height — the faster you hit the wall, the higher it takes you: from a walking leap the hands reach ≈ 3.9 m (a world-class wall climb-up), from a full sprint ≈ 5 m, so a sprint at a building catches its first-storey holds (the 3.8 m crevasse, the 4.0 m window course) and the climb carries on in flow (§1.11).
* **Steps & Stairs**: Curbs, stair treads and bumpers up to 0.4 m are simply walked over (no animation); knee-high obstacles (≤ 0.6 m) are stepped onto while walking and hip-high ones (≤ 1.0 m, car hoods) while running; taller obstacles up to standing reach (2.25 m) are vaulted with `Space`/`E`.

### 1.3 Contextual Targeting Reticle & Visual Reach
* **Soft-Lock Reticle**: A non-pinpoint, magnetic HUD reticle tracks the screen-space projection of the nearest valid ledge rail within a 35° forward cone. 
* **Dynamic Sizing**: The reticle expands when out of range, tightens and highlights (amber) only when the body reach model confirms the hands can actually close on the hold right now, indicating exactly where the character's hands will go if an interaction key is engaged.

### 1.4 Procedural Crevasses & Grip Anchors
Every vertical face of every obstacle procedurally defines and visually highlights its holds. Each crevasse is both a **hand-hold** and a **foot-hold** (you can stand on its lip):
* **Head-Height Crevasses (2.0 m)**: Grabbable from standing or running ground height.
* **Jump-Height Crevasses (3.8 m)**: Grabbable at the apex of a running jump **with a wall kick**, or by standing on the 2.0 m lip.
* **Ladder Crevasses (5.6 m, 7.4 m, …)**: Generated automatically on very tall walls so that no gap in the hold ladder (ground → crevasses → top) exceeds what a climber standing on a lip can reach (2.15 m). Every object in the playground, including the 8 m arena walls, is climbable.
* **Mantle Tops**: The flat, upper boundary of any wall or rooftop obstacle. A top edge with no room to stand behind it (a window sill, a window head, a door lintel) is still a hold: you hang from it and stand on its lip to reach the next one.
* **Buildings (Zone 6)**: Composite structures with recessed windows and a door (sills and heads are real ledges), balconies you can mantle onto from their side edges, a cornice crowning the roof and a rooftop structure; plus an L-shaped block whose inside corner exists on every hold row. Every edge is clipped against neighbouring geometry and **joined to the edges it meets at corners**, so ledges form a continuous graph around and between the blocks. The plain (windowless) walls carry crevasse courses at the facades' own sill and window-head heights (1.0 / 2.2 / 4.0 / 5.2 … m), running corner to corner, so **every row of holds wraps the whole building** — a traverse turns every corner, wall or facade — and **every wall climbs to the roof on its own** (the last course is 0.8 m under the roof edge; a scaffold's top deck always has a hold in reach above it).
* **The City**: The training park sits in the middle of a 300 × 320 m procedural city (20× the original arena): a street grid with 0.15 m curbs and 3 m sidewalks, 28 blocks of 2–6-storey buildings built from the building template (windowed facades toward the streets, doors with awnings, balconies, fire escapes, scaffolding, skybridges across cat-leap alleys, flush neighbours whose ledges join straight on), set-pieces (parking decks with stepped ramps, container yards, a construction-site concrete frame), ~400 parked cars / vans / buses / semi trucks — all standing on their tyres (cars and vans 4, buses 6, the semis 18-wheelers carrying containers, cab clear ahead of the box), in many colours (mostly metro-livery buses with the odd yellow school bus), with procedurally drawn windows, doors and handles, head and tail lights, grilles and plates, and the occasional dent or cracked window; corrugated, rusting shipping containers with lock-rod doors at one end (the same steel on the ground and on the trucks); a worn city of cracked asphalt and paving, plaster / brick / panel buildings in many colours with water damage, graffiti at street level, murals on blind walls, boarded-up and broken windows; ten different billboard ads, lit lamp posts, litter against the walls, and textured trees with rounded limbs and bushy crowns — bus shelters, billboards, benches, planters, dumpsters, and park nature (climbable trees, boulders, a terraced hill, a fountain). A 12 m perimeter wall with a crevasse ladder bounds the map. `config.CITY_SEED` changes the layout; every building roof is reachable by climbing (verified by the tests).

### 1.5 Multi-Key Parkour Combos & Dynamic Traversal
When targeted near a ledge rail:
* **Grab & Anchor (hold `Space` in the air, or `E`)**: Closes the hands on a rail that is within arm's reach of the shoulders, blends the body into a bent-arm hang (eyes 0.30 m below the hands) and locks gravity. Holds lower than the shoulders can be caught mid-air down to chest height; while rising fast only holds above the eyes are caught (apex catch). Top edges that are too low to hang from become a vault instead.
* **Shimmy Left / Right (`A` / `D`)**: Hand-over-hand traverse along the ledge at 1.2 m/s (1.5 m/s in run mode). The hands are *planted*: each stays where it is until the body has moved on, then hops ahead in a small arc - one hand in the air at a time, the leading hand reaching first and the trailing hand catching up. The leading hand stops at the very end of the edge while the body is clamped behind it.
* **Obstructions on an edge**: a traverse that runs into something in the middle of an edge - an awning, a balcony, a fire-escape landing in the way of your legs - switches onto that thing's edge (lower down if need be, turning to face it), and `W` climbs you up on top of it. If it has no edge to take you just stop there; a traverse only ever turns a corner at the end of its edge. Whenever a move leaves you facing a different wall - a corner, a switch, climbing up or down onto the side edge of an awning or balcony - the view turns with your body, so you never end up hanging with the camera looking along the wall.
* **Corner Turns (keep holding `A` / `D`)**: When the edge you are on meets another edge at the same height the traverse continues around it - an **outside corner** (the edge wraps away around the block: the body swings round the corner) or an **inside corner** (the edge continues on the wall beside / behind you: the body rotates into the corner). The lead hand reaches onto the new edge first, the other hand follows, and the camera turns with you. Holding the same key keeps you traversing along the new wall.
* **Climb Down an Edge (`E` while standing at a drop)**: Walk to the edge of whatever you are standing on, facing out over it, and a green chevron appears: `E` reverses the mantle - palms on the surface, turn to face the wall, lower yourself over the edge - ending in a hang from that edge, from where you can climb down hold by hold (`S`), shimmy, or climb straight back up (`W`).
* **Climb Up (`W` held while hanging)**: **Leg-driven climb-up** — pull-up (hands to chest), foot onto the lip, push up only as far as the reach needs, grab the next hold (≈0.85 s; ≈0.55 s in run mode, down to ≈0.35 s in full flow - §1.11). Holds straight up the wall or off to the side (balcony edges, the next bay's window head) qualify; a hold in your look direction is preferred. Blocked if nothing is within 2.15 m of the lip.
* **Top Out (a fresh `W` press while hanging from a top edge)**: Reaching a roof edge — by climbing or by catching it mid-air — always ends in a hang; nothing carries you over the top by itself. Pressing `W` there performs the two-stage **Mantle Vault** (hoist + knee/foot hook, then a step-through): ~0.75 s normally, ~0.45 s in run mode (less in flow, and you leave the edge at a run). Low obstacles you cannot hang from (≤ 2.25 m, hip-high walls, car roofs, dumpsters) are still vaulted directly with `Space`/`E`.
* **Flips (`Space` again in the air)**: after a jump (or a wall kick), press the parkour button a second time before you land and you tuck and flip: a **front flip** (the default, or running), a **back flip** (`S` held, or from a standing jump) or a **side flip** (`A` / `D` held on their own). The whole body turns once round its centre of mass - the view goes with it, you see your knees and the hands on your shins during the tuck - timed to the airtime left so you come round just before the feet touch down (a flip needs ~0.3 s of air: off a flat jump press it as soon as you let go of the jump; off something high it is slower and more relaxed). One flip per jump. If a hold is within reach (or about to be), or you are looking at a hold ahead that you are flying toward (the reticle is on it), the press is a reach as always (`E` is always a reach); while flipping you cannot catch anything.
* **Landing roll (automatic)**: land from higher than **7 ft (2.13 m)** - measured from the top of your flight to where your feet land - and you roll: the body absorbs the landing into a crouch, the hands go to the ground, you roll over along the way you were travelling and come up on your feet in about 0.6 s, keeping your momentum so you run straight on (still sprinting if you were). Falling backwards (letting go of a high hold) you roll backwards, away from the wall. A roll never carries you off the edge of what you landed on, and a jump in the second half of a roll springs straight out of it (a press just before that is buffered into it). Other players see your flips and rolls too.
* **Climb Down / Drop (`S`)**: Lowers the feet onto the hold below and drops the hands to it (reverse of the climb-up, including ledges beside you); if there is no hold below, cleanly releases grip. Uses a fresh `S` press, so a held key never drops you the moment you catch.
* **Dyno Jump Up (`Space` while hanging)**: Feet push off the wall (v_y = 5.9 m/s → ~0.9 m body rise); the hands can catch a hold up to ~1.2 m above the current one.
* **Wall Eject / Back Leap (`Space + S` while hanging)**: Backward cat-leap off the wall (v_horiz = 4.5 m/s, v_y = 3.0 m/s) with an automatic 180° camera turn toward the landing wall. Crosses the 3.2 m Dyno Chasm while dropping one crevasse.
* **Lateral Dyno Leaps (`Space + A` / `Space + D` while hanging)**: Launches sideways along the ledge tangent (5.0 m/s, +2.0 m/s outward, 4.0 m/s up) — a ~2.8 m lateral cat leap between the staggered pillars.

### 1.6 Pause Menu
`Escape` (or `Start` / `Menu` / `Options` on a controller) pauses the simulation, releases the mouse and shows a menu (**Resume** / **Multiplayer** / **Controller** / **Fullscreen** / **Quit**) navigated with `W`/`S` or the arrow keys and `Enter`, with the mouse, or with the controller (D-pad or left stick, `A` to confirm, `B` or `Start` to resume). `Escape` again resumes. **Multiplayer** hosts or joins a game (§1.10); **Controller** opens the controller screen (§1.9); **Fullscreen: Off / On** switches the display mode.

**Fullscreen**: `F11` or `Alt+Enter` at any time, or the pause-menu entry (also with a controller). The game goes borderless fullscreen at the desktop resolution and back to its 1280 × 720 window; the world keeps its proportions at any aspect ratio and the HUD, menus and controller screen are re-laid-out and re-rasterised for the new size (crisp text, scaled uniformly). The choice is remembered in `display.json` beside the controller settings; `python main.py --fullscreen` / `--windowed` override it for one run.

### 1.7 Fully Visual Procedural Arms, Hands, and Fingers
* **Anatomical arm mesh**: tapered eight-sided upper arms and forearms with rounded shoulder, elbow and wrist joints, a palm that widens toward the knuckles and tapered fingers. Every face is wound outward, so with back-face culling the limbs are solid — you never see the joint caps through them. Bone frames are anatomically consistent — the upper arm's roll follows the elbow bend plane, the forearm pronates with the palm, and the **wrist is limited to a 78° cone** from the forearm, so the hand always reads as attached to the arm.
* **Seen from your own eyes**: the camera sits 14 cm ahead of the body centre, in front of the shoulder line, and the shoulders are a real shoulder-width apart (42 cm) and belong to the torso (turning your head swings your eyes, not your shoulders). The arms therefore come into the picture wide and low, from the view's lower corners, as your own do.
* **Arms are heavy**: standing still they hang straight down beside the thighs with a slight bend at the elbow, palms in, and do not move at all. Walking they swing from the shoulder like pendulums, in opposition to the legs. Only when you actually run do they come up into a pump — elbows at 85–130°, the fists rising to the chest at the front of the swing and dropping to the hip at the back, harder and faster toward a sprint — and in the air they come up and out for balance. Reaching for a hold brings them forward as before.
* **Natural hand motion**: free hands ease toward their targets (no snapping), the shoulders trail quick head turns, and the hands anticipate — lifting toward a hold you are looking at on the ground, reaching for it in the air — all without ever exceeding arm's length.
* **Grip model**: the engine knows what kind of edge it is holding. Thin edges (crevasses, lips with < 25 cm behind them) are half-crimped with the fingertips in the slot and the wrist nearly straight (15°); deep ledges (wall tops, sills, balconies) are hooked open-handed over the top (26°). The wrist bends a lot only when the body is above the hands: easing to 70° as the chest rises in a pull-up, palms flat at 80° while propping on a ledge during a mantle or the push-up on a deep lip, and the reverse while lowering over an edge. Hopping hands open on lift-off and close as they land. The arms are depth-tested against the world, so fingers hooked over a ledge are hidden by it.
* **Analytical 2-Bone IK with joint limits**: Solves shoulder, elbow, and wrist joints in real time to touch the *actual* grip point (not the rail midpoint). Bones are rigid (0.33 m / 0.27 m); out-of-reach targets pull the wrist back along the shoulder line, too-close targets respect the 35° elbow flexion limit, and the elbow always bends toward an anatomical pole direction defined in the torso frame (so it looks the same anywhere in the world). While on a rail the torso squares up to the wall no matter where the head looks.
* **Hand Planner**: While on a rail each hand is planted at a definite spot; shimmies, corner turns and hand-overs to the next hold are staggered hops (one hand at a time, lifting in an arc, never past the end of an edge). A hand only lets go for a hold the arm can reach by the time it lands, hops exactly once per hand-over (no ping-pong between edges), and the lead hand alternates from one hand-over to the next. Hops are blended in body space so a reach made while the legs are still pushing the body up travels with the body, swinging slightly wide of the shoulder rather than through it; an open hand travelling along a wall flexes back just enough to keep its fingertips off the surface, and a hand leaving a deep palm press unbends its wrist at a human rate. Around a corner the lead hand reaches only as far round as the arm allows, walks out along the new edge as the body comes round, and the trailing hand follows once it has landed.
* **Procedural Finger Rig**: Hands accurately conform to geometry:
  * *Crimp Grip*: Fingers curl tightly over thin crevasse lips; thumbs brace against the wall face.
  * *Mantle Palm-Press*: Palms flatten against top horizontal surfaces; fingers splay forward.
  * *In-Flight Reach*: Hands anticipate oncoming edges as the player nears a valid target.

### 1.8 Legs, Boots and the Body Below You
Look down and the rest of you is there: a shirt-coloured torso, black pants and brown boots (rounded heel and domed toe on a black sole, a shaft up the shin) on a pair of legs built like the arms — rigid thigh and shin bones, a two-bone IK with **knee limits** (40°–179°: deep flexion to straight, never hyper-extended) and a knee that can only bend forward (and up when the leg is raised), an ankle limited to its real range. They work with the arms at all times (`DESIGN.md §5.6`):
* **Standing**: feet under the hips, a hair of knee, perfectly still. **Walking / running**: a gait locked a quarter-cycle behind the arm swing (left leg forward with the right arm) — the stance foot stays flat on the ground and travels back under you, the swing foot lifts higher and strides further the faster you go, the hips sinking a little so the legs stay in reach.
* **Jumping**: knees drive up on the way up, legs extend down for the landing.
* **Hanging / shimmying**: legs dangle with the toes at the wall (never in it); if the hold you climbed from is still under your feet they rest on it and **step along it** as you traverse.
* **Climbing**: the push leg (alternating each climb) smears the wall below the hold during the pull-up, steps onto the hold once your hips are high enough to allow the knee to close, and drives the body up as the leg extends; climbing down mirrors it with a foot kept on the lower hold until you are below it.
* **Mantling**: a knee comes up and the foot goes *over* the edge onto the top, the other follows once you are over, and you stand up on both; lowering yourself over an edge is the reverse.

### 1.9 Controllers (Xbox, PlayStation, Nintendo, generic)
Plug a controller in at any time (hot-plug, no restart) and play. Pads SDL knows - Xbox 360 / One / Series, DualShock 4, DualSense, Switch Pro and the hundreds of others in SDL's built-in database - are read through SDL's GameController API, which reports every pad in one *positional* layout (the bottom face button is always "A": Cross on PlayStation, B on Nintendo). Anything else (cheap USB pads, arcade sticks) falls back to a raw joystick with a best-guess layout. The pad you touched last is the one the game listens to.

**Default layout**

| Control | Key it acts as | Does |
| :--- | :--- | :--- |
| Left stick | `W A S D` (analog) | Move in any direction; the walking speed follows the tilt. Push up / down / sideways while hanging to climb / climb down / shimmy |
| Right stick | mouse | Look (response curve for fine aim, a turn boost when held at the rim) |
| **A** (Cross) | `Space` | Jump / hold to reach & catch / dyno while hanging / press again in the air to flip |
| **X** (Square) | `E` | Grab (hold to reach) / lower yourself over an edge |
| **LT** (L2) | `Left Shift` | Sprint while held - no lock |
| **Left stick click** (L3) | `Caps Lock` | Sprint lock (toggle; a "Sprint lock ON/OFF" notice confirms it) |
| **B** (Circle) | `Left Ctrl` | Let go |
| **RT** (R2) | `E` | Reach with the trigger, so the right thumb can keep aiming during a leap |
| D-pad | `W A S D` | Digital movement / precise climbing |
| **Start** (Menu / Options) | `Esc` | Pause menu - always (cannot be rebound) |
| Y, LB, RB, View / Share, R3 | - | Free: bind them to whatever you like |

Sprinting on a controller needs the stick pushed most of the way (≥ 80 %); below that you walk at the tilted pace. A tilted stick also pushes a proportionally shorter standing leap and steers less in the air. Keys and pad inputs can be mixed freely (W and the stick both holding "forward" keep you walking until both let go).

**The controller screen** (pause menu → *Controller*) shows a picture of your pad - drawn for its family (Xbox, PlayStation with the ✕ ○ □ △ symbols, touchpad and symmetric sticks, Nintendo, generic). It is live: every button lights up while you press it, the triggers fill as you pull them and the stick caps follow the sticks (with the deadzone drawn in the well). Each control has a label with the key it is bound to and what that key does.
* **To rebind**: click a button on the picture or its label (or press the button on the pad to find it, then `Enter`); it starts pulsing "Press a key..." - press the keyboard key it should act as. `Esc` cancels, `Backspace` / `Delete` unbinds. Any key works, including keys bound elsewhere (you are told where else it is used).
* **Settings**: look sensitivity (0.25-3x), invert look Y, swap sticks (southpaw), stick deadzone (5-35 %, shown live in the stick wells), vibration (a light rumble on catching a hold, a wall kick, launches and hard landings), and *Reset to defaults*.
* `Esc` or the pad's `Start` goes back.

Bindings and settings are saved as you change them to `controller.json` in your user config folder (`%APPDATA%\Hardcore` on Windows, `~/.config/hardcore` elsewhere; override with `HARDCORE_CONFIG_DIR`). A missing or corrupt file just means defaults. If the controller you are playing with is unplugged the game pauses, and the pad's buttons are released so nothing stays held.

### 1.10 Multiplayer — emblem hunt, capture the flag and zombie tag (up to 15 players, bots included)
Three games - emblem hunt and capture the flag are playable **free for all** or in **teams (red vs blue)**; zombie tag is every player for themself:
* **Emblem hunt** - glowing **emblems** (spinning blue crystals in a pulsing aura, easy to spot from a long way off) are hidden around the map - on building roofs, on top of other climbable things and down in the streets. Touch one to pick it up. When the last emblem is taken (or the time runs out) the round ends: the most emblems wins - the player, or in a teams game the team whose players picked up the most.
* **Capture the flag** - *teams*: each team has a base (a glowing ring on open ground, a flag on a pole and a column of light in the team's colour you can see across the city). Take the other team's flag from its stand and carry it into your own base **while your own flag is at home** to score a capture. Touch an enemy who is carrying your flag to **tag** them: the flag goes straight home and so do they. *Free for all*: one gold flag and a gold drop zone; run the flag into the zone to score (then both move somewhere new), and touch whoever has it to steal it (a fresh carrier cannot be robbed for 2.5 s). Falling off the map, leaving or switching sides with a flag sends it home. The most captures when the time is up wins - or the first to the **capture limit**.
* **Zombie tag** - at the start of a round one player is picked at random to be **alive**; everyone else is a green, shambling **zombie** (green skin, glowing eyes, rags, arms reaching out in front, a lurching limp - and your own first-person arms go green and reach forward too). The living player scores **a point for every second they stay alive** and starts some way off with a 5 s head start. A zombie that touches the living player **catches** them: the caught player becomes a zombie, and the zombie that caught them **comes alive** - and starts scoring, with a 3 s head start (and no tag-backs: the one just caught cannot catch them back for 10 s). The living player is a little **faster than the zombies, running (+12 %) and climbing (+15 %)** - roofs are a good place to be. The first to the host's **goal points** wins, or the most points when time runs out. The living player has a gold column of light over them; zombies get a marker pointing to them, the living player markers for the zombies closing in, and a banner says what you are (*YOU'RE ALIVE - RUN! 87 / 120 points* / *YOU'RE A ZOMBIE - catch Ann!*).

Other players are drawn as simple figures in their team's colour with their name above them; they follow your pose - running, hanging, climbing, flipping and rolling - and carry a flag on their back when they have one. Players pass through each other - there is no player collision. The HUD shows the game and the score (RED 2 - 1 BLUE in a teams game), the time left, the scoreboard, a banner while you carry a flag, and markers for the flags and your base (pinned to the edge of the screen when they are off it, with the distance). A notice pops up for every pickup, flag taken, tag, capture, join, leave and team change.

**Hosting** - *Pause → Multiplayer*. The left column is you: your name, **your team** (red or blue, used in teams games) and the address to join. The right column is the game you host: the port (default 24950), **Max players** (2-15, bots included), **Bots** (0 up to Max players - 1: your own slot stays free; 0 means no bots), **Bot skill** (*Simple* / *Complex* / *Hard* - how clever the bots are, see *Bot skill* below; the row is there when there are bots), the **Game** (emblem hunt / capture the flag / zombie tag), **Teams** (free for all / red vs blue; not in zombie tag), the **Match time** (2-60 minutes, or **Unlimited** - for every game: an emblem hunt then ends when the last emblem is found, zombie tag on its goal, capture the flag on its capture limit; with no capture limit either, the round only ends when you leave, and you are told so), and the game's own option - how many emblems to hide (4-40), the capture limit (1-20, or off) or the goal points (30-600). Then *Host a game*. The server starts in a process of its own (it builds its copy of the city, which takes a few seconds - the menu says so and you can keep playing meanwhile); you join it like everyone else. Others on your network join with your machine's IP address; over the internet, forward the TCP port on your router. **Joining** - type the host's address (`192.168.1.20` or `192.168.1.20:24950`) in *Join address* and choose *Join game*. In a teams game you get the side you picked; **changing *Your team* during a game switches sides** (you reappear at your new side's start). *Leave game* ends your session (if you are the host it ends the game for everyone, and they are told so). Everything is saved in `multiplayer.json` beside the controller settings. The whole screen works with the mouse, the keyboard (type into the text rows, `Left` / `Right` change values) or a controller.

From the command line:
```bash
python main.py --host --mode ctf --teams --bots 5 --max-players 8 --time 10 --captures 3 --team red --name Alice
python main.py --host --mode emblems --ffa --bots 2 --emblems 16          # (also --port N, --bot / --no-bot)
python main.py --host --mode zombie --bots 6 --goal 180 --time 0          # zombie tag, no time limit
python main.py --host --mode emblems --bots 6 --bot-level hard            # ... against masterminds
python main.py --join 192.168.1.20:24950 --team blue --name Bob
python -m net.hosted --bots 6 --mode ctf --teams --time 15               # a dedicated server, no game window
```

**Bots and slots**: *Max players* counts everyone. Bots fill the free slots; when a person joins a full game a bot gives its slot up (one from the side the person joins), and it comes back when someone leaves. In a teams game the bots switch sides to keep the teams within one player of each other - people always get the side they ask for.

**Rounds**: when a round ends everyone sees the results (the winner or winning team and the score, every player's score - by team in a teams game). *Join new game* puts you in the next round, which starts as soon as every connected player has joined it, or after 30 s. Each round places everything afresh (new emblem spots, new bases) and starts everyone at their side's spawn.

**The bots** are real players driven by AI through the same controls you have: they go for what the game asks - an emblem they have spotted (they have to find them, like you), the enemy flag, home with it, after the player carrying their flag; on a side of three or more one bot stays back to guard the flag; in zombie tag the zombies hunt the living player and a living bot runs for somewhere far from the zombies, away from the one on its heels. They run the streets with the sprint lock on, meet the walls they climb at pace with a wall run (and climb on in flow), climb walls with the parkour button, shimmy round obstructions, leap between roofs and drop off edges when that is quicker (rolling when they land from high up), ease off before sharp turns into narrow alleys, hop up onto low roofs (bus shelters, awnings) from their edge, give up on holds that lead nowhere and ask to respawn if they ever get stuck. In tests a lone bot, searching the city for the 12 emblems of a round, finds about 4 of them in five minutes at Simple, 8 at Complex and 10 at Hard, and a Hard bot wins almost every race against the other two, and bots capture flags from each other in two-a-side and free-for-all capture the flag (bigger bot teams mostly tag each other's thieves - `DESIGN.md` §11.6) - fair opponents, not perfect ones.

**Bot skill** - the host picks how clever the bots are (*Pause → Multiplayer → Bot skill*, or `--bot-level simple|complex|hard`). Every level has the same body and the same controls; what changes is what they notice, remember and plan (`game/bot_skill.py`, `game/tactics.py`):
* **Emblems have to be found.** No bot knows where the emblems are. A bot *sees* one when it is in range and in its field of view with nothing in the way - an emblem on a roof is hidden from the street below, just as it is from you - and heads for it; a sighting from far away is only a rough idea of where it is, sharper the closer the bot gets. With several in mind it goes for the nearest. With nothing in mind it searches: it remembers which parts of the map it has looked over and heads somewhere it has not, climbing up onto roofs to see over them. An emblem it remembers that somebody else picked up is only forgotten when it sees the spot empty.
* **Simple** bots look ahead of them (70 m, a 110° field of view), keep *one* emblem in mind and the last few places they looked, and wander off somewhere new to find more.
* **Complex** bots look about (100 m, 170°, all round from a roof), keep *three* emblems in mind and a quarter of the city as searched, and search methodically - the nearest unsearched ground, up a tall roof (to its edge) for a view over the streets and roofs round it.
* **Hard** bots see all round and far (140 m), never forget an emblem they have seen and **share everything with the other Hard bots on their side** - one sees it, all know it - and split the emblems and the map between them; they do not race you for an emblem you will get to first, and once the whole map has been looked over without finding everything they walk the roofs. In capture the flag a Hard side posts a hunter in a fleeing thief's own base.
* **Zombie tag** plays the same at every level - on purpose, and measured: zombies run the living player down flat out, and a living bot runs for the spot furthest from the zombies (Complex and Hard bots that are up on a roof with a lead run on across the rooftops). The living player is faster than a zombie, so it is caught when it slips, and only zombies right behind it can take the chance: spreading the pack out - flanking, cutting it off, a net, a ring, waiting where it was heading - was built in many forms and always let the runner get away more often, and a runner that spent its lead climbing onto roofs or changing direction was caught sooner (see `DESIGN.md` §11.6, and `tools/bot_levels.py` to measure it yourself).

Pausing in multiplayer only pauses *you* (the others keep playing). The protocol is plain TCP with one JSON message per line; the server is authoritative for emblems, flags, tags and scores. It is meant for LAN / friends over a forwarded port: there are no accounts, passwords, NAT traversal or anti-cheat (each game reports its own player's position), so play with people you know.

**Smooth remote players**: each game sends its player's pose 30 times a second and the server stamps every pose with the time on its clock it is from. Clients draw the other players 0.1 s in the past on that timeline - not on when a snapshot happened to arrive - so network jitter, and updates and snapshots not being in step, never make someone stutter; a late update is bridged along the last motion for up to 0.1 s instead of freezing. (Protocol version 4: games of different versions cannot join each other - they also move differently.)

### 1.11 Momentum & Flow
The moves that used to stop a run dead now carry it, so a chase across the city keeps its pace:
* **Wall run**: hit a wall at a sprint with `Space` held and the plant takes you up to the first-storey holds (hands at ~5 m). From a walking leap it is the classic ~3.9 m wall climb-up.
* **Climb flow**: the speed you meet a wall with - a wall run, a running leap onto a hold, running off a roof into a catch - is carried into the climb: keep `W` held and the climbs and the top-out are up to 40 % quicker. It fades a little with every hold and drains away if you just hang there; chaining holds with `W` held in run mode builds a rhythm of its own. A 12 m wall takes ~2.7 s from a full sprint (~3 s from a jog, ~5 s from a walk), where it used to take ~6 s whatever the run-up. The moves stay continuous (the body flows through each phase of a climb instead of stopping between them) and the hands keep time: a hand-over always lands on the new hold as the body gets there.
* **Top out at a run**: topping out of a flowing climb sends you off the edge at a run with the sprint build-up restored - straight into the next leap. A slow, dead-hang top-out still steps off at a walk.
* **Speed vaults**: a vault or step-up taken at a run (a car hood, a low wall) is quicker and keeps ~92 % of your speed and the sprint build-up: a car in the way of a sprint costs a fraction of a second, not the sprint.
* **Rolls** are a little shorter (~0.6 s) and a jump pressed in the second half springs out of them.
* **Smooth frames**: the camera is drawn between physics ticks (the simulation is a fixed 60 Hz), so motion is even whether a frame ran two ticks or none.

---

## 2. Technical Stack & Dependencies

* **Python 3.10+**
* **Pygame** (or pygame-ce): Window creation, input event dispatch (keyboard, mouse and controllers through SDL's GameController API), mouse locking, and high-resolution frame clocks.
* **PyOpenGL**: Modern OpenGL 3.3 Core Profile pipeline (VAOs, VBOs, EBOs, and GLSL shaders).
* **NumPy**: High-performance vectorized matrix algebra, Quaternion rotations, analytical IK equations, and mesh buffer staging.
* **pytest** (development only, `requirements-dev.txt`): headless gameplay test-suite.

### 2.1 Running
```bash
pip install -r requirements.txt
python main.py
```

### 2.2 Testing
The whole simulation (input buffer, state machine, collision, body model, arm IK, level generator) runs headless, so realistic-gameplay rules are enforced by an executable test-suite:
```bash
pip install -r requirements-dev.txt   # game requirements + pytest
./run_tests.sh                        # falls back to a minimal runner if pytest is missing
```
The controller tests drive the game with the same code path as the game loop. `tests/test_gamepad_devices.py` goes further and plugs **virtual controllers** into SDL itself (Xbox, DualSense, DualShock 4, Switch Pro, a generic pad and an unrecognised raw joystick, through SDL's virtual-joystick API), so hot-plugging, family detection, event decoding and the raw fallback are tested on the real SDL event path; those tests skip if the SDL that pygame bundles is older than 2.24. `tests/test_building_holds.py` searches every wall of generated buildings of every height (with scaffolds, balconies, fire escapes, awnings) for a climb to the roof that never turns a corner, checks that every course of holds on every city building turns all four corners, that nothing in the city interpenetrates, and the collision buffers; `tests/test_display.py` covers the fullscreen preference and the resolution-independent UI layout. `tests/test_multiplayer.py` runs real servers and clients over loopback sockets (handshake, version / full-server rejects, snapshots, authoritative pickups, rounds and *join new game*, the bots, disconnects, malformed and oversized input, a full game of 15 where bots give their slots to people) and `tests/test_mp_game.py` covers the emblem rules, emblem placement, the navigation grid, the multiplayer screen and the bot climbing buildings to collect every emblem of a round on its own (a full-city run is in the DESIGN notes), and the bot getting out of the spots where it used to be stuck for good on a city round (a sprint into a narrow alley, a bus-shelter roof only a hop reaches, an unreachable last target); `tests/test_mp_modes.py` covers teams and capture the flag - the rules (taking, capturing only with your flag at home, tags, respawns / leaving / switching sides with a flag, capture and time limits, draws, the free-for-all steal and the moving drop zone), bot counts limited by the player limit, picking and switching teams with the bots evening the sides, a capture over real sockets, the host's match time, hosting in a separate server process, results and notices, team-coloured / flipping / flag-carrying avatars, the screen-edge markers, and bots playing both kinds of capture the flag against each other; `tests/test_zombie.py` covers zombie tag - a random living player starting away from the zombies, a point a second, catches swapping the living player and the zombie (either one moving), the head starts and no tag-backs, the goal and the clock, the living player leaving, bot chasing and fleeing, zombies running and climbing slower, a catch over real sockets, the session slowing a zombie and marking who is who, bots playing it, the menu, the zombie look (and a zombie's own arms reaching ahead) - and games of every kind with no time limit; `tests/test_acrobatics.py` covers flips (once per jump, only after a jump, timed to finish before touch-down, chosen by the held keys, never instead of a reach, no catching mid-flip) and rolls (the 7 ft threshold, momentum kept into the run, backward rolls off a wall, never off an edge, a flip that lands high then rolls, springing out of a roll) and a rigid, knee-limited, tucked body through them; `tests/test_edge_switch.py` includes the camera-follows-body regression for climbing onto side-facing holds.
`tests/harness.py` drives the real `PlayerController` with scripted keys on a fixed 60 Hz clock. The suite asserts, among other things: human arm lengths / reach / jump heights / sprint acceleration, rigid bones and elbow limits in every frame of a climb, hands on the actual grip point, no hanging with the feet in the floor, leg-driven crevasse climbs, mantle durations (slow vs. at speed), plausible dyno / eject / cat-leap ranges, crevasses on all four faces with climbable ladder gaps, hand-over-hand shimmies (one hand in the air at a time, lead hand stops at the edge end), outside and inside corner turns (body never clips the block, camera turns, lead hand first), climbing down over an edge, that a top edge is climbed straight over with `W` held (after the same settle every hold gets) and hung from when `W` is released, the wrist cone and the attached forearm/hand meshes, smooth camera-following hands, the grip model (edge kinds, wrist angles per phase, continuous transitions, fingers hidden inside the edge, open hands on hops, depth-tested arms), the arm animation (a running pump that hardens and quickens with speed, arms in opposition and quiet in the air; hand-overs that go once per hand, alternate the lead hand, never slide back down, swing clear of the shoulder and land as the body arrives; corner reaches within arm's length with the trailing hand waiting), run momentum (a jog at once, a sprint only after the build-up, lost on a sharp turn or a hang, a speed-driven view), momentum through the moves (`tests/test_momentum.py`: the sprint wall run to the first-storey holds, climb flow making a sprint up a building faster than a jog or a walk, chaining rhythm, flow draining while hanging, topping out at a run, speed vaults keeping the run, root motion that flows through its keys without overshooting, hands that keep time with a flowing climb, frames drawn evenly between physics ticks), the bot levels (`tests/test_bot_tactics.py`: sight lines, what each level keeps in mind, emblems only gone for once spotted and forgotten only when seen gone, rough sightings sharpening, searching somewhere unsearched, Hard bots sharing and splitting, the chase on the live track and the Hard camper, a runner that commits; `tests/test_bot_levels.py`: smarter bots finding more emblems in the city), remote players drawn on the server's timeline (`tests/test_net_smoothing.py`: steady speed despite jitter and mismatched update / snapshot rates, de-duplicated poses, bridged late updates, the clock offset estimate), the take-off push (standing long jump forward / backward / sideways, leap distance growing with run-up, never beyond sprint momentum, stepping off a roof), the resting arms (hanging still at the sides, swinging at a walk, up only when running, entering the view from its corners), the legs (rigid bones and knee limits in every frame of every state, feet on the ground when standing and in stance, strides that grow with speed, tuck-and-extend jumps, toes clear of walls, the smear-plant-extend of a climb, alternating push legs, feet stepping along a hold, mantles over the edge) and a closed, outward-wound, correctly dressed body mesh, the building generator (sills / heads / door / clipped hidden edges / roof ring), the vehicles (tyre counts per type, tyres on the ground under lifted bodies and standing proud of the sides, the cab clear of the container, material roles and metric face coordinates, container doors at the far end from the cab, containers on the ground, rounded boot toes, colour and livery variety) and the city's character (varied billboard ads, lit lamp posts over the road, plaster / brick / panel walls with graffiti, murals and grime, glass behind facades, bark and leaf trees with clumps, litter, asphalt and pavement), the pause-menu model, the city (deterministic seed, no overlapping buildings, every one of the ~80 building roofs reachable through the hold graph, a scripted climb of a city building, vehicles run over / vaulted / climbed, props and ramps), and that **every object in the tutorial playground - hurdles, walls, towers, pillars, buildings, balconies, rooftop - can be climbed** with either the parkour button or `E`.

---

## 3. High-Level Engine Architecture

The engine decouples physical simulation, input buffering, and skeletal IK from rendering:

```
┌────────────────────────────────────────────────────────┐
│                      Input Manager                     │
│  (Window events, mouse deltas, 150ms Action Queue)     │
└───────────────────────────┬────────────────────────────┘
                            │
┌───────────────────────────▼────────────────────────────┐
│          Kinematic Character Controller (KCC)          │
│   • Swept AABB Collision      • Variable Jump Reservoir│
│   • Ground / Sprint Friction  • Air Momentum Physics   │
└───────────────────────────┬────────────────────────────┘
                            │
┌───────────────────────────▼────────────────────────────┐
│            Parkour State Machine (FSM)                 │
│   • GROUND / SPRINT           • AIRBORNE (Jump/Fall)   │
│   • LEDGE_HANG / SHIMMY       • MANTLE_VAULT (Spline)  │
│   • DYNO_LAUNCH / WALL_EJECT  • Action Lockouts        │
└─────────────────────┬───────────────────┬──────────────┘
                      │                   │
┌─────────────────────▼───────┐   ┌───────▼──────────────┐
│  Parametric Ledge Graph     │   │ Soft-Lock Reticle    │
│  • Procedural Rail Extraction│   │ • Screen-space query │
│  • Segment Projections      │   │ • Magnetic smoothing │
└─────────────────────┬───────┘   └───────┬──────────────┘
                      │                   │
┌─────────────────────▼───────────────────▼──────────────┐
│      Procedural Skeletal Arm / Hand / Finger Rig       │
│   • Clavicle / Shoulder Camera Anchor with Inertia     │
│   • Analytical 2-Bone Arm IK (Shoulder-Elbow-Wrist)    │
│   • Procedural Hand Posing & Finger Crimp/Palm Solvers │
└───────────────────────────┬────────────────────────────┘
                            │
┌───────────────────────────▼────────────────────────────┐
│           Modern OpenGL 3.3 Core Renderer              │
│   • Shader Programs (World, Ledge Glow, Skinned Arms)  │
│   • Dynamic Batched Geometry (VAO/VBO/EBO)             │
│   • Dynamic Camera (FOV Lerp, Bob, Roll, View Matrix)  │
└────────────────────────────────────────────────────────┘
```

---

## 4. Directory Structure

```
hardcore/
├── README.md                   # Project overview and controls
├── DESIGN.md                   # Full mathematical & architectural specification
├── run_tests.sh                # Headless realistic-gameplay test-suite entry point
├── tools/bot_levels.py         # Benchmark: how the bot levels compare in whole games (many seeds)
├── tests/                      # pytest suite + scripted simulation harness
├── main.py                     # Game loop, delta time, system orchestration
├── config.py                   # Global physics, body model, IK, and rendering constants
├── core/
│   ├── window.py               # Pygame initialization & OpenGL 3.3 Core context, fullscreen (display.json)
│   ├── input_buffer.py         # Timestamped input queue (fresh-press vs held semantics)
│   ├── controls.py             # Control scheme -> parkour intents
│   ├── keymap.py               # Keyboard layout: key names -> actions (+ readable names)
│   ├── input_router.py         # Keyboard / pad buttons / stick -> one input buffer (multi-source holds)
│   ├── gamepad.py              # Controls, pad families & labels, bindings + settings file, deadzones / curves
│   ├── gamepad_device.py       # SDL GameController + raw joystick devices, hot-plug, rumble
│   ├── gamepad_input.py        # Active pad -> keys, analog move & look, menu navigation, haptics
│   ├── controller_menu.py      # Controller screen: layout geometry + select-and-press-a-key model
│   ├── mp_menu.py              # Multiplayer screen model (you: name, team, address; host: players, bots, game, teams, time)
│   └── menu_model.py           # Pause-menu selection model (render-free)
├── game/
│   ├── match.py                # Game rules: emblem hunt, capture the flag, zombie tag; free for all / teams; rounds
│   ├── teams.py                # Red and blue
│   ├── nav.py                  # Navigation grid + weighted A* for the bots
│   ├── bot.py                  # Bot AI: drives a real PlayerController through player inputs
│   ├── bot_skill.py            # Bot intelligence levels: Simple / Complex / Hard
│   ├── sight.py                # Line of sight over the level (what a bot can see)
│   ├── tactics.py              # Bot tactics: emblem search & memory, chasing, fleeing, shared hive minds
│   └── session.py              # The game's side of multiplayer: host / join, sync, team switch, results; CLI options
├── net/
│   ├── protocol.py             # JSON-lines wire protocol, validation
│   ├── server.py               # Game server (selector thread): players, slots & bots, teams, match, snapshots
│   ├── hosted.py               # The server in its own process (what hosting uses) + a dedicated-server CLI
│   └── client.py               # Client: handshake, receive thread, interpolated remote players
├── math3d/
│   ├── transform.py            # NumPy 4x4 matrix helpers (Perspective, LookAt, Model)
│   ├── quaternion.py           # Rotations, SLERP, and direction-to-quaternion math
│   └── geometry.py             # Raycasts, segment projections, and distance tests
├── physics/
│   ├── aabb.py                 # Axis-Aligned Bounding Box primitives
│   └── collision.py            # Swept AABB vs static obstacle collision resolution
├── parkour/
│   ├── ledge_rail.py           # Parametric ledge segments + corner links
│   ├── ledge_system.py         # Ledge graph: rail clipping, corner linking, reticle queries
│   ├── body.py                 # Human body model: reach envelope, hang / lip-stand / mantle poses, grab solver
│   ├── controller.py           # Headless PlayerController (camera yaw/pitch, FSM tick, collision step)
│   ├── acrobatics.py           # Flips and the landing roll: the whole-body spin, its camera / body transform
│   └── state_machine.py        # KCC FSM (Ground, Air, Hang, Shimmy, Climb-Up, Mantle, Dyno, Eject, Roll)
├── animation/
│   ├── skeleton.py             # Joint hierarchies, local/world transforms, bone matrices
│   ├── two_bone_ik.py          # Analytical 2-bone arm solver with pole vectors
│   ├── finger_solver.py        # Procedural finger curling and surface contact logic
│   ├── hand_planner.py         # Planted hands: staggered hops along / around edges
│   ├── leg_rig.py              # Legs: gait, jump, hang, foot-on-hold climbing, mantle (knee-limited IK)
│   └── arm_rig.py              # First-person body coordinator: arms + hands, torso, legs; the body mesh
├── world/
│   ├── obstacle.py             # Procedural parkour blocks, pillars, and walls
│   ├── building.py             # Buildings: facades with windows/doors, awnings, balconies, fire escapes, scaffolds
│   ├── props.py                # Vehicles, street furniture, stairs/ramps/decks, trees, boulders, hills
│   ├── city.py                 # Street grid, blocks, set-pieces, vehicle lanes, park nature (seeded)
│   ├── spatial.py              # Uniform-grid broadphase for thousands of obstacles
│   └── generator.py            # Training zones, stand-alone playground, and the full city
└── render/
    ├── shader.py               # GLSL compile, link, and uniform caching utility
    ├── buffer.py               # Modern GL VAO/VBO/EBO abstractions
    ├── reticle.py              # Soft-lock contextual cursor renderer (+ climb-down prompt)
    ├── menu.py                 # Pause menu (font -> GL texture labels), controller / multiplayer / results screens
    ├── controller_view.py      # Controller screen picture, drawn into a pygame Surface
    ├── hud.py                  # Short notices (controller connected, sprint lock on/off)
    ├── mp_render.py            # Other players (team colours, flips, carried flags), emblems, flags & bases, HUD, markers
    ├── renderer.py             # Master render pass (world, ledges, arms, HUD, menu)
    └── shaders/
        ├── world.vert / .frag  # Scene geometry with directional lighting & grid lines
        ├── ledge.vert / .frag  # Glowing emissive rail highlight shader
        ├── arm.vert / .frag    # Skinned matrix palette shader for arms & fingers
        ├── ui.vert / .frag     # Screen-space cursor shader
        ├── text.vert / .frag   # Screen-space textured quads (menu text)
        ├── avatar.vert / .frag # Other players' figures (lit, tinted per player)
        └── aura.vert / .frag   # Additive glow billboards (emblems) and light columns (flags, zones)
```

---

## 5. Controls Reference — "one parkour button"

Design goals: few buttons, intent-driven, forgiving on input timing (150 ms buffer, 120 ms coyote
time) but **fallible** — the hands only close on a hold while you are holding the parkour button,
so releasing it mid-air is a real miss. Pushing a direction does the obvious thing while hanging.

| Input | Context | Action |
| :--- | :--- | :--- |
| **`W, A, S, D`** | Ground / Air | Movement & air steering (air steering redirects, never adds speed) |
| **`Caps Lock`** | Any | **Run toggle**: caps on = run (a 6.5 m/s jog at once, building to an 8.5 m/s sprint over ~1.6 s of straight running), caps off = walk (5.0 m/s). Running also climbs at speed (chaining holds builds a rhythm), traverses quicker and speed-vaults hip-high obstacles without losing the run |
| **`Left Shift` (Hold)** | Any | Momentary run override (same as caps on while held) |
| **`Space` (Tap)** | Ground | Low hop (~0.45 m) in the held direction; if a hold/edge is within reach in front of you: climb / vault it instead |
| **`Space` (Hold)** | Ground → Air | High leap (~0.95 m) driven along the held direction — ~2.8 m standing long jump (~2.4 m backward with `S`, or sideways), ~6.6 m at full sprint — and **reach**: hands catch any hold they pass within arm's reach (a caught top edge is hung from - hold or press `W` to top out); plant a foot on a wall you hit (wall kick) |
| **`Space` (Release)** | Air | Stop reaching — you will fall past ledges |
| **`Space` (Press again)** | Air, after a jump | **Flip**: front flip (default / `W`), back flip (`S`, or from a standing jump), side flip (`A` / `D`) - unless a hold is within reach (then it is a reach) |
| *(automatic)* | Landing from > 7 ft | **Roll** along your travel, keeping the momentum |
| **`W` (Hold)** | Ledge Hang | Climb up: pull-up + foot on the lip to the next hold |
| **`W` (Hold / Press)** | Hanging from a top edge | Top out (mantle over the edge) after the same short settle every hold gets; release `W` to stay hanging |
| **`A` / `D` (Hold)** | Ledge Hang | Shimmy along the ledge |
| **`S` (Tap)** | Ledge Hang | Climb down to the hold below, or drop if there is none |
| **`A` / `D` (Hold) at a corner** | Ledge Hang | Turn the corner onto the joined edge (outside or inside corner) and keep traversing |
| **`Space` (Tap)** | Ledge Hang | **Dyno up** (feet push off the wall, ~0.9 m rise) — hold it afterwards to catch the next hold, or let go to fall |
| **`Space` + `A` / `D`** | Ledge Hang | Lateral cat leap along the wall (turn toward the target mid-air) |
| **`Space` + `S`** | Ledge Hang | Wall eject: backward cat leap with an automatic 180° turn |
| **`Left Ctrl` / `C`** | Ledge Hang | Let go |
| **`E`** | Any | Explicit grab (same as holding `Space` in the air; useful for a static grab without jumping) |
| **`E` (Tap)** | Standing at an edge, facing out | **Climb down**: lower yourself over the edge into a hang (green chevron prompt) |
| **`W` into an obstacle** | Ground | Automatic step-up: knee-high (≤ 0.6 m) while walking, hip-high (≤ 1.0 m) while running |
| **`Mouse`** | Any | Look (yaw & pitch); look direction picks between reachable holds |
| **`Escape`** | Any | Pause menu (Resume / Multiplayer / Controller / Fullscreen / Quit); `Escape` again resumes |
| **`F11`** / **`Alt+Enter`** | Any | Fullscreen on / off (remembered) |
| **Controller** | Any | Everything above, from a pad: left stick = `WASD` (analog), right stick = look, **A** = `Space`, **X** = `E`, **LT** = `Shift` (hold), **L3** = `Caps Lock` (sprint lock), **B** = `Ctrl`, **RT** = `E`, D-pad = `WASD`, **Start** = pause. Rebind in *Pause → Controller* (§1.9) |

Typical lines: *walk up, `Space` (grab), `W` `W` `W` (climb), done* — or — *Caps on, run, hold
`Space` before the wall, keep holding through the kick and the catch, `W` at speed up the holds and straight over the roof edge.*
