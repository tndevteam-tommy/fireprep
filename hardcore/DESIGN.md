# HARDCORE — Technical Design Document & Architecture Specification

This document provides the complete mathematical, architectural, and algorithmic specification for the **Hardcore** parkour engine.

---

# 1. Architectural Pipeline & Tick Lifecycle

The engine operates on a fixed-timestep physics update with variable-timestep visual interpolation (semi-fixed timestep loop):
the physics runs at a fixed 60 Hz and every frame is drawn `alpha = accumulator / FIXED_DT` of the way from the previous tick's
body position to the latest (`PlayerController.prev_pos` / `render_pos(alpha)`; a move of more than 2 m in one tick - a respawn -
is never blended).  The camera, the arms and the body are posed from that position, so motion is even whether a frame ran two
ticks or none (`tests/test_momentum.py` drives the loop's accumulator with jittered 60 Hz and 144 Hz frame timing).

```
 ┌─────────────────────────────────────────────────────────────┐
 │                      Main Loop Tick                         │
 └──────────────────────────────┬──────────────────────────────┘
                                │
 ┌──────────────────────────────▼──────────────────────────────┐
 │ 1. Event Polling & Input Buffer Update                      │
 │    • Gather Pygame keyboard, mouse & controller events.     │
 │    • Push inputs with timestamps into Action Buffer.        │
 │    • Expire actions older than 150ms.                       │
 └──────────────────────────────┬──────────────────────────────┘
                                │
 ┌──────────────────────────────▼──────────────────────────────┐
 │ 2. Physics & Kinematic Character Controller (Fixed dt)      │
 │    • Accumulate frame delta time.                           │
 │    • While accumulator >= FIXED_DT (e.g. 1/60s):            │
 │      a. Update FSM state (Ground, Air, Hang, Mantle, Dyno). │
 │      b. Resolve buffered input combos against current state. │
 │      c. Apply gravity, momentum, and variable jump lift.    │
 │      d. Perform Swept-AABB collision vs static blocks.      │
 └──────────────────────────────┬──────────────────────────────┘
                                │
 ┌──────────────────────────────▼──────────────────────────────┐
 │ 3. Spatial Rail Query & Soft-Lock Reticle Solver            │
 │    • Query Ledge Graph within frustum and distance cone.    │
 │    • Compute screen-space projections of rail segments.     │
 │    • Update reticle spring-damper position & target lock.   │
 └──────────────────────────────┬──────────────────────────────┘
                                │
 ┌──────────────────────────────▼──────────────────────────────┐
 │ 4. Camera Kinematics & View Matrix                          │
 │    • Integrate yaw and pitch; apply Dutch roll & head-bob.  │
 │    • Lerp dynamic FOV based on sprint/mantle state.         │
 │    • Construct View and Projection matrices via NumPy.      │
 └──────────────────────────────┬──────────────────────────────┘
                                │
 ┌──────────────────────────────▼──────────────────────────────┐
 │ 5. Procedural First-Person Arm, Hand, and Finger Solver     │
 │    • Evaluate base procedural locomotion sway / pump.       │
 │    • If targeted/hanging: Anchor wrists via 2-Bone IK.      │
 │    • Run finger-curling solvers (crimp vs palm-press).      │
 │    • Build matrix palette (Uniform array mat4 u_Bones[]).   │
 └──────────────────────────────┬──────────────────────────────┘
                                │
 ┌──────────────────────────────▼──────────────────────────────┐
 │ 6. Modern OpenGL 3.3 Core Render Pass                       │
 │    • Bind framebuffers, clear color & depth buffers.        │
 │    • Render static procedural world (Batched VAO).          │
 │    • Render emissive ledge highlight strips.                │
 │    • Render skinned arm/hand/finger meshes (Bone Palette).  │
 │    • Render screen-space HUD & Contextual Reticle.          │
 │    • Swap window buffers (pygame.display.flip()).           │
 └─────────────────────────────────────────────────────────────┘
```

---

# 2. Input Buffering & Action-Combo System

Human reaction times require an input buffering window so fast parkour combos register reliably without requiring frame-perfect button presses.

### 2.1 Action Buffer Data Structure
* The input manager maintains a queue of `BufferedAction` structs:
  Action = { action_id, timestamp, consumed_flag }
* Buffer duration window: tau_buffer = 0.15 seconds (150 ms).
* At timestamp t_now, any action with (t_now - timestamp) > tau_buffer is purged.

### 2.2 Combo Resolution Matrix
The controller tests for combo chords hierarchically before single-key actions.
Two kinds of key semantics are used:
* **Trigger** keys must be a *fresh* press (still unconsumed in the 150 ms buffer). One-shot
  actions (dynos, ejects, drops) use a trigger so that a held chord never re-fires every tick
  or drops the player the instant they catch a ledge.
* **Modifier** keys may be held *or* tapped recently. Continuous actions (climb, shimmy) accept held keys.

| Action Priority | Buffered Key Combo | Current State | Target State | Condition |
| :--- | :--- | :--- | :--- | :--- |
| **1 (Highest)**| `Space`(trigger), `W`/nothing held | `LEDGE_HANG` | `DYNO_LAUNCH` | Hang pose settled |
| **2** | `Space`(trigger) + `S` | `LEDGE_HANG` | `WALL_EJECT` | Hang pose settled |
| **3** | `Space`(trigger) + `A` | `LEDGE_HANG` | `DYNO_LAUNCH` (left) | Hang pose settled |
| **4** | `Space`(trigger) + `D` | `LEDGE_HANG` | `DYNO_LAUNCH` (right)| Hang pose settled |
| **4b** | `Ctrl`(trigger) | `LEDGE_HANG` | `AIRBORNE` (let go) | — |
| **5** | fresh `W` press | `LEDGE_HANG` on a `MANTLE_TOP` | `MANTLE_VAULT` | Top surface has room to stand. Never automatic: climbing to or catching a top edge always ends in a hang |
| **6** | `W` | `LEDGE_HANG` | `CLIMB_UP` | A hold within reach of a (partial) stand on this lip, up to 2.15 m above it |
| **7** | `S`(trigger) | `LEDGE_HANG` | `CLIMB_UP` (down) | A hold within 2.15 m below whose lip the current grip can be held from |
| **7b** | `S`(trigger) | `LEDGE_HANG` | `AIRBORNE` (drop) | No lower hold |
| **8** | `A` / `D` | `LEDGE_HANG` | `SHIMMY` | Target within rail bounds and body collision-free |
| **8b** | `A` / `D` at the edge end | `SHIMMY` | `CORNER_TURN` | Rail has a link that way and the entry hang pose is free |
| **8c** | fresh `E` | `GROUND` | `MANTLE_VAULT` (reverse) | Standing at a top edge, facing out; hang pose below is free |
| **9** | `Space` / `E` (held or buffered) | `AIRBORNE` / `GROUND` | `LEDGE_HANG` | Hold within arm's reach of the shoulders (§6.0) |
| **10** | `Space` / `E` (held or buffered) | `GROUND` / `AIRBORNE` | `MANTLE_VAULT` | `MANTLE_TOP` too low to hang from (vault of a low obstacle) |
| **11** | `W` into a wall | `GROUND` | `MANTLE_VAULT` | Wall top <= 1.0 m above the feet (auto step-up) |
| **12** | `W` into a wall while reaching | `AIRBORNE` | (wall kick) | Approach >= 2.5 m/s, not falling fast, once per jump |
| **13** | `Space` (Hold) | `GROUND` | `AIRBORNE` | t_hold <= 0.25s |
| **14 (Lowest)**| `Space` (Tap) | `GROUND` | `AIRBORNE` | Instant initial impulse |

A buffered grab press is **not** consumed by a failed grab attempt; it stays valid for the full
150 ms window so the hands close on the ledge the moment it comes within reach.

### 2.3 The Intent Layer & Control Scheme
The state machine never reads keys. Each tick the `FlowScheme` (`core/controls.py`) resolves the
`InputBuffer` into an `Intents` object and the FSM acts on named intents (`grab`, `jump`, `dyno_up`,
`dyno_left/right`, `eject`, `climb_down`, `let_go`, `climb_down_edge`) plus continuous flags
(`climb_up`, `shimmy`, `run`, `jump_held`, `wall_kick`, `auto_step_max`). Buffered
presses are consumed only when the FSM actually acts on an intent, so input buffering is uniform.

| Intent | Keys |
| :--- | :--- |
| `grab` (hands close on a hold) | `Space` **or** `E` held/buffered — release to miss |
| `jump` | fresh `Space` (a grab/vault at a wall takes priority) |
| `wall_kick` | only while reaching (`Space`/`E` held) |
| `climb_up` / `mantle` (top out) | `W` held (or pressed afresh) while hanging from a top edge, once settled |
| `auto_step_max` | 0.6 m walking, 1.0 m running (`W` into the obstacle) |
| `climb_up` | `W` while hanging |
| `shimmy` (and corner turns) | `A`/`D` while hanging |
| `climb_down` (or drop) | fresh `S` while hanging |
| `let_go` | fresh `Ctrl`/`C` |
| `dyno_up` / `dyno_left/right` / `eject` | fresh `Space` while hanging with none / `A`,`D` / `S` held |
| `climb_down_edge` | fresh `E` standing at an edge, facing out |
| `run` | **Caps Lock toggle** (synced from the OS state at start-up) or `Shift` held |
| pause | `Escape` (handled by the window layer, not an intent) |

Controller buttons are bound to keyboard keys and go through the same tables (§2.4), so every intent
above is reachable from a pad with identical buffering semantics.

Design notes: one "parkour button" carries the player's vertical intent (jump → reach → catch → flow
over the edge → dyno), movement keys carry direction (push into the wall to climb, pull back to
descend, sideways to traverse and turn corners), and `Ctrl` is the only way to bail out. Mistakes are
possible and cheap to understand: letting go of `Space` in the air means no catch, tapping instead of
holding at a wall means no kick, a dyno with nothing above lands you back on the same hold only if you
are still reaching. Coyote time (0.12 s) makes running off edges forgiving without removing the fall.

### 2.4 Controllers (`core/gamepad*.py`, `core/input_router.py`)

```
 SDL GameController / raw joystick events
        │  GamepadManager (gamepad_device.py): hot-plug, instance-id -> PadDevice, PadState per pad,
        │  trigger hysteresis, "last pad used is active", rumble
        ▼
 PadEvent(button, control, down)  +  PadState (sticks, triggers)
        │  GamepadInput (gamepad_input.py)
        ├── buttons / triggers ── ControllerBindings: control -> key name -> KEYBOARD_ACTIONS -> Action
        ├── move stick ────────── radial deadzone -> analog (right, forward) + StickDirections (digital)
        └── look stick ────────── LookProcessor -> PlayerController.apply_look(turn, up) per frame
        ▼
 InputRouter: source-counted holds -> InputBuffer (fresh presses buffered as for keys) + move_axis
        ▼
 FlowScheme -> Intents (forward/back/left/right flags + move_analog, move_amount) -> FSM
```

* **Bindings**: a control (SDL positional name: `a` is the bottom face button on every pad) is bound to a
  keyboard key *name*; pressing it acts exactly as that key (so the controller can never do something the
  keyboard cannot, and the 150 ms buffer, trigger/modifier semantics and toggles are shared). `start` is
  hard-wired to pause. Defaults: `a`→Space, `x`→E, `b`→Left Ctrl, `lefttrigger`→Left Shift,
  `leftstick`→Caps Lock, `righttrigger`→E, D-pad→WASD. Stored with the settings in the per-user
  `controller.json`; loading is lenient (unknown / invalid entries fall back to defaults).
* **Sources**: `InputRouter` counts which sources hold an action (`("key", code)`, `("pad", control)`,
  `("stick", direction)`). Each press registers a fresh buffered press; the action is released only when
  the last source lets go.
* **Move stick**: radial deadzone `dz` (default 0.15) with rescaling `m = (|v| - dz) / (outer - dz)`,
  `outer = 0.92`, clamped to the unit circle (square-gated pads). The FSM moves along the exact stick
  direction; ground target speed `WALK_SPEED · m`; sprinting needs `run ∧ forward ∧ m ≥ 0.8`; the
  take-off push and air steering scale with `m`. Keys have `m = 1`, so keyboard play is unchanged.
  Digital directions (climb, shimmy, climb-down / mantle fresh presses, dyno chords) come from
  `StickDirections`: a direction is held while `m ≥ 0.5` (released below 0.35) within 60° of its axis
  (kept until 66°) - diagonals hold both neighbours like W+D, while a push 25° off "up" still only climbs.
* **Look stick**: same deadzone, response `r = m²` applied radially, `230°/s` yaw / `160°/s` pitch at
  sensitivity 1, and a rim boost (+55 % ramped in over 0.35 s after 0.18 s at full deflection) for quick
  turns. Integrated with the real frame time, so the turn rate is frame-rate independent.
* **Triggers**: analog value shown live; digital press at 0.30, release at 0.18.
* **Menus**: D-pad / left stick (0.6) navigate with auto-repeat (0.40 s delay, 0.11 s rate), A confirms,
  B / Start resume. Buttons held when the menu closes are ignored until released (the A that confirmed
  "Resume" never jumps).
* **Devices**: pads with an SDL mapping open as GameControllers (events carry instance ids - device indices
  shift on hot-plug); others open as raw joysticks with `raw_layout()` (XInput-like or "PS-style" DirectInput
  order by button count, 4- or 6-axis, hat 0 → D-pad, raw triggers rescaled from [-1, 1]). Family (Xbox /
  PlayStation / Nintendo / generic) from the name, then the USB vendor id in the GUID. SDL hints set before
  start-up: positional Nintendo buttons, PS4/PS5 rumble over Bluetooth.
* **Haptics** (`Haptics`, optional): short rumbles on GRAB, WALL_KICK, dynos / eject, VAULT and landings
  faster than 6.5 m/s, only while the player is using the pad.
* **Controller screen**: `ControllerLayout` (pure geometry, shared by drawing and hit-testing),
  `ControllerMenuModel` (focus, waiting-for-a-key, settings), `ControllerView` (draws into an opaque pygame
  Surface, uploaded as an RGB texture only when its state signature changes, blended at 93 %).

---

# 3. Parametric Ledge & Crevasse Graph System

Rather than using computationally expensive, jitter-prone real-time mesh raycasting, the environment builds an explicit **Parametric Ledge Graph** during level generation.

### 3.1 Parametric Rail Definition
Every interactable edge is defined by a directed 3D line segment:
L(s) = P1 + s * (P2 - P1),  s in [0, 1]

```
          P1 ─────────────────────── P2   <-- Rail Segment
             │                     │
             │     Wall Face       │
             │                     │
             ▼ Normal (N)          ▼
```

Each rail contains:
* `p1`: R^3 starting vertex.
* `p2`: R^3 ending vertex.
* `normal` (N): Normalized horizontal vector pointing outward, perpendicular to the obstacle face.
* `tangent` (T): Normalized unit vector along the edge: T = (P2 - P1) / ||P2 - P1||.
* `rail_type`:
  * `HEAD_CREVASSE`: Wall indentation at y ≈ 2.0 m.
  * `JUMP_CREVASSE`: Wall indentation at y ≈ 3.8 m.
  * `LADDER_CREVASSE`: Additional indentations every 1.8 m on very tall walls (5.6 m, 7.4 m, …).
  * `MANTLE_TOP`: Uppermost edge of an obstacle with a traversable top surface.
* `thickness`: Rail lip depth (used by procedural fingers to decide crimp curl radius).
* `owner`: The obstacle that generated the rail (used to find holds on the same wall face).

Every crevasse is simultaneously a **hand-hold** and a **foot-hold**: the climber can stand on its lip.

### 3.2 Projection & Spatial Clamping
When the player hangs or shimmies, their position along the rail is parameterized by scalar s:
s = ((P_player - P1) · (P2 - P1)) / ||P2 - P1||^2
s_clamped = clamp(s, s_min, s_max)
where s_min and s_max account for the player's physical shoulder width (0.35 m) preventing hands from hanging off the void past obstacle corners.

### 3.3 Anchor Standoff Transform
The player's body and eye position are locked to the rail anchor:
P_eye = L(s_clamped) + N * d_standoff - y_hat * h_hang
where d_standoff = 0.36 m (chest-to-wall; the 0.6 m body box must not intersect the wall) and
h_hang = 0.30 m (eyes below the hands). With shoulders 0.20 m below the eyes the shoulder→hand
distance is ≈0.54 m for a 0.60 m arm, i.e. an *active* hang with ~115° elbows. A straight-arm dead
hang would be h_hang ≈ 0.37 m; anything larger is anatomically impossible and is rejected by the tests.

If the free-hang pose intersects the ground (hold lower than ~2.07 m) the body is raised until the
feet rest on the ground (*supported grip*) provided the hands are still at/above the shoulders;
otherwise a `MANTLE_TOP` becomes a direct vault and a crevasse is simply not grabbable.

---

### 3.4 Rail Clipping & Corner Links (Ledge Graph)
`LedgeSystem.build(obstacles)` turns the raw per-box rails into a graph:
1. **Clipping**: a rail is *covered* wherever a probe point `RAIL_COVER_PROBE` (0.2 m) outside the edge
   lies inside another box whose vertical span contains the edge (a box whose bottom is flush with the
   edge counts, a box whose top is flush does not). Covered intervals are removed and the surviving
   pieces become new rails; pieces shorter than `MIN_RAIL_LENGTH` (0.6 m, two hands) are dropped. This
   removes edges hidden behind facades, cornices and attached blocks and splits edges exactly at
   inside corners.
2. **Linking**: rails at the same height whose endpoints coincide (`CORNER_LINK_TOL`) are joined.
   Travelling "right" along R (toward `p2`) continues onto Q when `Q.p1 == R.p2`:
   * `dot(Q.N, R.T) > 0`  → **convex** corner: the edge wraps away around the block.
   * `dot(Q.N, R.T) < 0`  → **concave** corner: the edge continues on the wall beside / behind the player.
   * same normal           → **straight**: collinear edge of a neighbouring block.
   Because every rail's tangent is the hanging player's right, the same shimmy key keeps the traverse
   going through any corner type.

---

# 4. Contextual Soft-Lock Reticle

The reticle provides continuous visual feedback of interactable edges without requiring pixel-perfect crosshair aiming.

```
       [ Screen Viewport ]
       ┌────────────────────────┐
       │                        │
       │       (  +  )          │   <-- Default Center Crosshair
       │         \              │
       │          \ (Magnetic)  │
       │           ▼            │
       │         [ < > ]        │   <-- Soft-Locked Target on Nearest Edge
       │    ════════════════    │   <-- Highlighted Ledge Rail
       │                        │
       └────────────────────────┘
```

### 4.1 Targeting Mathematics
1. **Candidate Query**: For all rails L_i within reach distance (d <= 4.5 m):
   * Sample the closest point on the segment Q_i in [P1, P2] to the player's camera position.
   * Compute vector D_i = Q_i - P_cam.
   * Compute forward alignment: cos(theta) = (D_i · V_forward) / ||D_i||.
   * Discard candidate if theta > 35° or if player is facing the wrong side of the wall (D_i · N_i < 0).
2. **Screen-Space Projection**:
   Project Q_i using the Model-View-Projection (MVP) matrix into Normalized Device Coordinates (NDC):
   v_clip = M_proj * M_view * [Q_i, 1.0]^T
   v_ndc = (1 / w_clip) * [x_clip, y_clip]^T
   v_screen = [((x_ndc + 1) / 2) * W, ((1 - y_ndc) / 2) * H]^T
3. **Magnetic Spring Smoothing**:
   The reticle position R does not jump instantaneously. It uses a critically damped spring-damper lerp:
   R_(t+dt) = R_t + (v_screen - R_t) * (1 - exp(-lambda * dt))
   When no ledge is targeted, R springs back to the screen center (W/2, H/2).

### 4.2 Visual States
* **Neutral**: Small, translucent circular dot at screen center.
* **Climb Down (green chevron)**: Standing at an edge of the surface underfoot, facing out over it and within
  0.9 m: `E` lowers the player over that edge. The chevron is pinned to the bottom of the screen when the
  edge itself is below the view.
* **Tracking (Out of Reach)**: Expanded bracket [ ] tracking the edge point, faint cyan color (d in [1.8 m, 4.5 m]).
* **Locked (In Reach)**: Tight diamond snap, high-intensity emissive amber glow (d <= 1.8 m). Signals that pressing `E` will instantly anchor.

---

# 5. First-Person Procedural Arm, Hand, and Finger Skeletal System

The player’s hands, fingers, and arms are drawn in the first-person view, visually demonstrating realistic grips on rails, crevasses, and mantle tops.

### 5.1 Skeletal Hierarchy & Bone Layout
The rig consists of two symmetrical arm hierarchies anchored to an inertially stabilized camera rig:

```
Camera Root (Floating Eye Base)
 └── Chest / Clavicle Base
      ├── Left Clavicle  ──> Left Shoulder  ──> Left Elbow  ──> Left Wrist
      │                                                           └── Left Hand (Metacarpals)
      │                                                                ├── Thumb  [Proximal, Distal]
      │                                                                ├── Index  [Proximal, Inter, Distal]
      │                                                                ├── Middle [Proximal, Inter, Distal]
      │                                                                ├── Ring   [Proximal, Inter, Distal]
      │                                                                └── Pinky  [Proximal, Inter, Distal]
      │
      └── Right Clavicle ──> Right Shoulder ──> Right Elbow ──> Right Wrist
                                                                  └── (Identical Finger Hierarchy)
```
* **Arm joint count**: 2 * (1 Clavicle + 1 Shoulder + 1 Elbow + 1 Wrist + 1 Palm + 14 Finger Phalanges) = 38 joints
  (bones 0..37), followed by the body and legs (bones 38..45, §5.6): `TORSO`, `PELVIS`, `L_THIGH`,
  `L_SHIN`, `L_FOOT`, `R_THIGH`, `R_SHIN`, `R_FOOT` - **46 bones** in one palette, one skinned draw.

### 5.2 Camera, Eyes and Shoulders
The camera is not at the body centre: the eyes sit `CAMERA_FORWARD` (0.14 m) ahead of it along the
*head's* facing (`BodyModel.camera_pos`), in front of the shoulder line, exactly as your own eyes
sit in front of your shoulder joints. The shoulders belong to the torso:
```
body_eye = P_cam - head_forward_h * CAMERA_FORWARD
S_left   = body_eye + torso_right * (-SHOULDER_HALF_WIDTH) + up * (SHOULDER_HEIGHT - EYE_HEIGHT) + torso_forward * SHOULDER_FORWARD
```
with a real shoulder width (`SHOULDER_HALF_WIDTH` = 0.21) and the torso yaw trailing quick head turns
(`TORSO_YAW_FOLLOW`). Turning the head therefore swings the eyes, not the shoulders, and the arms
enter the picture from its lower corners: the shoulders are 4 cm behind and 20 cm below the camera,
a full shoulder-width apart, so an arm reaching forward comes in at > 30 deg from the view axis
(`tests/test_ik_arms.py::test_arms_enter_the_view_from_its_lower_corners`). Physics (grab reach,
hang geometry) still uses the body's own eye point, so nothing about climbing changed.

### 5.3 Analytical Two-Bone Inverse Kinematics (IK)
For each arm, target wrist position W is known (computed from the *actual* grip point `fsm.grip_point`
± half the hand spacing along the rail, or from the procedural locomotion swing). Shoulders hang from a
yaw-only torso frame (they do not pitch with the head); while on a rail the torso faces the wall.

```
   Shoulder (S)
       o
      / \
     /   \  Upper Arm (length L1)
    /     \
   o       o Elbow (E) [Constrained by Pole Vector]
           /
          / Forearm (length L2)
         /
        o Wrist (W)
```

1. **Distance Clamping (rigid bones, human elbow)**:
   D = W - S,  d = ||D||
   d_min = sqrt(L1² + L2² - 2 L1 L2 cos 35°),  d_max = sqrt(L1² + L2² - 2 L1 L2 cos 178°)
   d_solve = clamp(d, d_min, d_max);  the wrist is placed at S + D̂ * d_solve (never at an unreachable W).
   Targets more than ~75° behind the chest plane are folded back in front of the torso.
2. **Elbow Angle via Law of Cosines**:
   cos(alpha) = (L1^2 + d_solve^2 - L2^2) / (2 * L1 * d_solve)
   cos(beta)  = (L1^2 + L2^2 - d_solve^2) / (2 * L1 * L2)
   where alpha is the angle at the shoulder relative to D, and beta is the interior elbow flex angle.
3. **Pole Vector Plane Projection**:
   To prevent the elbow from flipping erratically, an anatomical **Pole Direction** P_pole is defined
   *relative to the shoulder in the torso frame* (down, outward and slightly back). Because it is a
   direction and not a world position, the solution is independent of where the player stands.
   * Construct normal to the limb plane: N_arm = (D x P_pole) / ||D x P_pole||
   * Construct planar perpendicular: U = N_arm x (D / d_solve)
   * Elbow position E:
     E = S + L1 * (cos(alpha) * (D / d_solve) + sin(alpha) * U)
4. **Bone Transforms**:
   * Upper arm rotation aligns along vector (E - S).
   * Forearm rotation aligns along vector (W - E).

### 5.3b Bone Frames & the Wrist Cone
Bones are rigid and their *roll* is anatomical: the upper arm's local +Z lies in the elbow bend plane,
the forearm's local +Z follows the palm normal (pronation / supination), and the hand's finger direction
may deviate at most `WRIST_MAX_ANGLE_DEG` (78°) from the forearm axis - flexion, extension and
deviation combined; pronation is free. When a requested hand pose exceeds the cone the finger
direction is rotated back to the cone boundary in the bend plane (`ArmSkeleton.clamp_wrist`), keeping
the palm normal orthogonal. The mesh is a chain of tapered eight-sided prisms with ellipsoid joint caps
(shoulder 5.4 cm, elbow 4.7 cm, wrist 3.4 × 2.7 × 2.4 cm), a palm widening 6 → 8.4 cm toward the
knuckles and tapered six-sided fingers; the forearm tip and the wrist cap coincide exactly so the
hand always reads as attached.

Free hands (not planted on a rail) are smoothed in an eye-relative frame (`HAND_FOLLOW_RATE`
16 /s) so body motion never drags them and quick head turns are followed with a short lag; the
shoulders themselves trail the camera yaw (`TORSO_YAW_FOLLOW` 14 /s). On the ground a hold being looked
at within ~3 m lifts the hands toward it by up to `GROUND_AIM_BLEND` (35 %); airborne the reach goes all
the way, always clamped to arm's length.

### 5.4 Grip Model: Wrists and Fingers on Every Kind of Edge (`animation/grip.py`)
Real climbing is done with the fingers; the wrist stays nearly straight except when the body is
propped *above* the hands.  The rig therefore never derives the hand orientation from the edge
alone - it builds the hand frame relative to the forearm the IK produced, with a *designed* wrist
extension toward the wall:

| Situation | Kind | Fingers | Wrist extension |
| :--- | :--- | :--- | :--- |
| Hanging from a crevasse / thin lip (< 25 cm of surface behind it) | CRIMP | half crimp: MCP 48°, PIP 88°, DIP -12°, tips in the slot | 15° |
| Hanging from a wall top, sill, balcony or platform edge | LEDGE | open-hand hook: MCP 62°, PIP 42°, DIP 18°, draped over the top | 26° |
| Pull-up (chest rising to the hold) | any | unchanged | eases 15/26° → 70° as the wrist drops from 0.45 m above the shoulder to shoulder level (`hang_extension`) |
| Push-up on a deep lip (CLIMB_UP stand phase) | LEDGE | palm press (5°, 5°, 0°) on the lip | 80° (clamped to the 78° wrist cone) |
| Push-up on a thin edge | CRIMP | stays hooked | up to 70° from body height |
| Mantling over an edge | LEDGE | hook → palm press as the chest passes the edge (blend over 20–50 % of the hoist); palms leave the surface from 60 % and the arms swing into the rest pose with the body | 26° → 80° → 25° |
| Lowering over an edge (climb-down) | LEDGE | palm press first, rolling into the hook as the body drops below the edge (30–65 %) | 80° → 26° |
| Shimmy / corner / hand-over hop | any | the hopping hand opens (8°, 22°, 8°) on lift-off and closes as it lands (45 % of the hop each way), rising 6 cm and coming 11 cm out from the wall | hang → 6° → hang |
| Reaching / pre-reach | - | open, turning toward the hold | relaxed |
| Pushing off (dyno, eject) | - | flat palm on the wall | ~70° |

Wrist positions follow the grip: a hanging hand's wrist sits 4 cm out and 9 cm below a thin edge
(5 cm / 7 cm for a ledge) so the knuckles meet the edge and the curled fingers end *inside* it;
a pressing hand's wrist is 14 cm behind the edge, 2.5 cm above the surface.  Because the arms are
depth-tested against the world, fingers over a ledge or in a crevasse are hidden by it exactly as a
real climber's are.  Every transition is continuous: wrist angle changes are bounded per frame
(tests assert <= 16°/frame through a full climb and a mantle), the release from a press eases the
extension from the press angle to a relaxed 25° as a function of the blend alone so the swinging
forearm cannot cause a jump, and the elbow is kept at least 5 cm clear of the wall by rotating it
about the shoulder-wrist axis toward the pole direction.

#### Hand Planner (traverses, corners, hand-overs)
While on a rail each hand is *planted* at (rail, distance-along-rail). The body slides continuously; a
hand only moves when its resting spot (`grip ± HAND_SPACING/2`, plus one replant-time of look-ahead and a
`SHIMMY_HAND_STRIDE` for the leading hand while traversing) has drifted more than 8 cm away, and only one
hand may be in the air at a time (`HAND_REPLANT_TIME` = 0.18 s for a step along the same edge,
`HANDOVER_TIME` = 0.25 s for a reach to another edge - both divided by the move's `move_rate` (below); lifting `HOP_LIFT` = 6 cm and coming `HOP_OUT` = 11 cm
off the wall mid-hop). Hands never plant closer than `HAND_RAIL_END_MARGIN` to the end of their rail, so the
leading hand stops at the end of the edge. Corner turns feed the planner per-hand targets on the new
rail and climb-ups hand over to the next hold the same way. Each hand's orientation follows the wall
it is currently on.

**The hands keep time with the body**: `ParkourStateMachine.move_rate` is how many times faster than its
nominal pace the current move runs (`climb_scale / motion_scale` for root motion - run mode, climb flow and
a zombie's slower climbing all show in it - or the traverse speed factor while hanging).  The arm rig
divides hop durations and the wrist-opening time by it and multiplies the reach-slide speed by it (clamped
to 0.5-2.5), and a hand-over to the next hold during `CLIMB_UP` is capped to `move_time_left` (never under
`HAND_MIN_HOP_TIME` = 0.06 s), so the lead hand is on the new hold when the body gets there at any pace.

Rules that keep a hand-over human (`arm_rig.py::plant_and_reach`, `hand_planner.py`):
* **Reach gating** - a hop is only launched toward a target within `HANDOVER_REACH_FRACTION` (98.5 %)
  of arm's length of the shoulder *now or where the shoulder will be when the hop lands* (the
  shoulder velocity extrapolated over the hop duration): a climber reaches for where the body is
  going, so the lead hand catches the next hold as the body arrives, and the trailing hand follows.
* **Slipping, not gluing** - a planted hand whose edge has moved out of reach (the legs pushed the
  body up, or the body went round a corner) opens and slides, shoulder-relative and speed-limited
  (`REACH_SLIDE_SPEED` = 1.6 m/s), toward its next hold; the next hop starts from wherever the hand
  is shown, so nothing ever snaps back to an old edge.
* **One hop per hand per hand-over** - a hand keeps its current edge until the next hold is reachable
  and then goes once; the planner retargets a hand in flight along the same rail as the body moves,
  rather than sending it again.
* **Alternating lead** - on a near-tie the hand that did not lead the previous hand-over leads
  (`HandPlanner.last_lead`), across climbs and down-climbs alike.
* **Body-space hops** - in flight a hand is blended between its start and its target *relative to the
  shoulder*, so a hand-over made while the body is still rising travels with the body. If the straight
  path would bring the hand closer than `HOP_MIN_SHOULDER_DIST` (0.30 m) to its own shoulder it swings
  wide laterally (eased in and out so the hop still starts and ends on its holds); the sliding hand
  does the same.
* **Wrist rates** - a hand leaving a deep palm press unbends at `WRIST_OPEN_RATE` (300°/s): its hop is
  stretched to allow it. An open hand travelling along a wall (`HandSpec.wall_clear`) flexes back only
  as far as needed to keep its fingertips off the surface given the wrist's distance from it.
* **Corners** - the lead hand reaches around only as far along the new edge as the arm allows (sampled
  from the corner outward), walks out to its resting spot as the body comes round, and the trailing
  hand keeps its spot on the old edge (slipping to arm's length if need be) until the lead hand has
  landed.

#### Pose D: Hanging Arms, Walking Swing & Running Pump (Ground Movement)
Arms are heavy. When not interacting with ledges the resting arm is posed *forward-kinematically*
from two joint angles - the shoulder swing `theta` (from straight down, + forward), the elbow
interior angle `beta`, plus a small abduction - and the fist that produces is what the IK is given,
so the elbow always lands on a chosen human angle (`FirstPersonArmRig.update`, `fk()`):

| Situation | shoulder `theta` | elbow `beta` | hand |
| :--- | :--- | :--- | :--- |
| standing still | 0 | 168 (12 deg bend) | hangs by the thigh, palm in, fingers down, **no motion at all** |
| walking (≤ 5 m/s) | ±22 deg pendulum, one cycle per stride | 168 → 158 on the forward swing | relaxed, palm toward the thigh |
| running (`SPRINT` state / ≥ jog speed) | 8 ± 30..42 deg | 85 (front) .. 130 (back) | loose fist, thumbs up; the fist comes up to the chest at the front and drops to the hip at the back |
| airborne | 35 | 100 | forearms up and out for balance |

`arms_up` (0 → 1, eased at 6/s) blends the hanging pose into the pump: it is 1 whenever the FSM is
in `SPRINT` (so a hard cut that costs speed does not drop the arms) or the speed is above
`ARMS_UP_SPEED_HIGH`, and 0 at walking pace; the walk/run gait phase advances at
`GAIT_BASE_CADENCE + GAIT_CADENCE_PER_MS * speed` steps/s (1.4–1.75 arm cycles/s) and settles to a
neutral phase when the player stops, so the arms come to rest rather than freezing mid-swing. The
same gait phase drives the legs (§5.6). Hand specs are forearm-relative `OPEN` specs with a fixed
12 deg flexion; the roll flips sign between the hanging (palm in) and pumping (thumbs up) poses
because the bend axis does.

### 5.6 Legs (`animation/leg_rig.py`)
The body below the shoulders is drawn too: a shirt-coloured torso block from the hips to the
shoulders, a pelvis, and two legs - thigh / shin in black pants, a brown boot whose shaft covers the
lower shin: a rounded heel, a ten-sided body tapering toward a domed toe cap (no square tips) and a
thin black sole `ANKLE_HEIGHT` under the ankle (`FOOT_LEN` 0.27 m). Anatomy
(`config.py`): hips `HIP_HEIGHT` = 0.935 m above the feet (`HIP_HALF_WIDTH` 0.10), thigh 0.44, shin
0.42, so a straight leg is a hair longer than the hip height and a standing knee is never locked.

Each leg is the same analytical two-bone chain as an arm, solved with **knee limits**
(`KNEE_MIN_ANGLE_DEG` 40 .. `KNEE_MAX_ANGLE_DEG` 179: heel-to-buttock flexion to straight, never
hyper-extended) and a pole of `forward + 0.35 up (+ a little outward)`, so a knee can only bend the
way a knee bends - forward, and upward when the leg is raised. The foot is a frame whose +Y is the
toe direction, clamped to the ankle's range (20 deg dorsiflexion .. 50 deg plantar flexion from the
right angle to the shin). Ankle targets are eased (free legs *relative to the hip*, so body motion
never drags a foot behind; planted feet in world space) and hard-limited to 12 cm above the hip.

Per FSM state the rig supplies ankle targets (`LegState` telemetry in `legs.last`):
* **standing** - feet under the hips, hip-width apart, 11 deg of knee, perfectly still;
* **walk / run** - a gait a quarter cycle behind the arms (left leg forward with the right arm):
  the stance foot is ground-locked and travels back under the body, the swing foot lifts
  (`0.06 + 0.035 * speed` m, heel up first) and comes through; half-stride `0.08 + 0.05 * speed`
  (max 0.5 m) and the hips sink `0.03 + 0.13 u²` so the spread legs stay within reach;
* **jump / fall** - knees drive up while rising (blend by vertical speed), legs extend down and
  slightly forward for the landing, the push leg a little ahead;
* **hang / shimmy / corner** - legs dangle, knees a touch bent (one more than the other), toes at
  the wall but never inside it (`FOOT_LEN * 0.75 + 2 cm` of standoff); a supported hang stands;
  if the feet last stood on a hold that is still below the hips and within reach (`foothold`), they
  rest on it and **step along it** one at a time (0.16 s, 5 cm lift) as the body traverses;
* **climb-up** - the push leg (alternating per climb) *smears* the wall 0.42 m below the hips during
  the pull-up, steps onto the hold once the hips are `FOOT_ON_HOLD_MIN_HIP_ABOVE` (0.22 m) above it
  (the knee cannot close further with the hips this close to the wall), stays planted while the leg
  extends and the body rises, and the feet end resting on that hold; climbing down mirrors it;
* **mantle** - the lead knee comes up and the foot goes *over* the edge onto the top (up first,
  then in - never through the corner), the trailing foot follows once the body is over the edge,
  and the stand-up ends on both feet; climbing down an edge reverses it (lead foot over first, the
  other stays on the top until the body is below it);
* **dyno / eject** - legs tuck after the push.

`tests/test_legs.py` checks all of it: rigid bones, knee range and bend direction in every frame of
every state, feet on the ground when standing and in stance, stride / lift / cadence growing with
speed, tuck-then-extend in a jump, toes clear of walls and blocks, the smear → plant → extend of a
climb, alternating push legs, feet stepping along a hold while shimmying, a foot kept on the lower
hold while climbing down, the knee-up-and-over of a mantle, and a closed, outward-wound body mesh
dressed in the right materials.

### 5.5 GPU Vertex Skinning Pipeline
Rather than updating thousands of hand mesh vertices on the CPU in Python, bone transformations are solved as a flat palette array:
```glsl
// GLSL 3.3 Core uniform block
uniform mat4 u_Bones[46];
```
Each vertex in the body mesh stores:
* `in vec3 a_Position;`
* `in vec3 a_Normal;`      // every face is wound counter-clockwise seen from outside and its normal
                           // points outward, so back-face culling hides the insides of the limbs and
                           // the joint caps tucked inside them (no seeing through the joints)
* `in vec2 a_UV;`          // uv.x = material: 0 by bone (sleeve / skin), 1 skin, 2 black cloth, 3 brown boot, 4 shirt
* `in vec4 a_BoneIDs;`    // Up to 4 bone influences
* `in vec4 a_Weights;`    // Normalized blend weights (sum w = 1.0)

Linear Blend Skinning (LBS) runs entirely on the GPU in `arm.vert`:
P_skinned = sum_{k=0}^3 a_Weights[k] * (u_Bones[a_BoneIDs[k]] * P_local)

---

# 6. Kinematic Character Controller & Momentum Physics

### 6.0 Human Performance Envelope (Body Model)
Gravity is 2x Earth gravity (-19.62 m/s²) for a snappy feel, so all impulses are specified by the
*outcome* they produce and compared against real parkour performance. The body is a 1.80 m athlete
(`config.py`, enforced by `tests/test_body_model.py`, `tests/test_locomotion.py`):

| Quantity | Value | Real-world reference |
| :--- | :--- | :--- |
| Eye / shoulder height | 1.65 m / 1.45 m | anthropometric 1.80 m adult |
| Upper arm / forearm / hand grip | 0.33 / 0.27 / 0.10 m | 0.30–0.36 / 0.25–0.30 m |
| Max grab reach from shoulder pivot | 0.80 m (incl. 0.10 m shrug/tip-toe) | |
| Flat-footed overhead reach | **2.25 m** | ~1.25–1.30 × height |
| Largest hold gap climbable from a lip | 2.15 m | standing reach minus margin |
| Tap / held jump (CoM rise) | 0.45 m / 0.95 m | elite standing vertical 0.8–1.0 m |
| Wall kick (tic-tac) extra rise | <= 1.1 m (hand reach ≈ 3.9 m) | world-class wall climb-up ≈ 4 m |
| Walk / jog / sprint | 5.0 / 6.5 / 8.5 m/s | trained athlete; 30 km/h needs a run-up |
| Acceleration | 20 m/s² to walk, 6 m/s² beyond; sprint build-up over 1.6 s | a jog in 0.3 s, top speed after ~2 s (game pace) |
| Take-off push | 3.8 m/s from rest → 0.4 m/s at a sprint (√-fade) | legs drive a standing leap, momentum a running one |
| Standing long jump | ~2.8 m forward (+15 % boost), ~2.4 m backward / sideways | elite 2.5–3.2 m |
| Running long jump | ~4.6 m walk, ~5.5 m jog, ~6.6 m sprint (+15 % forward boost) | elite parkour 4–6 m; a little above for game feel |
| Dyno up (feet on wall) | 0.9 m rise, hands +1.2 m | elite campus/dyno |
| Wall eject (backward cat leap) | 4.5 m/s out, 3.0 m/s up | 2.5–3.5 m with a drop |
| Lateral dyno | 5.0 m/s along, 2.0 m/s out, 4.0 m/s up | ~2.8 m lateral cat leap |
| Shimmy | 1.2 m/s (1.5 m/s in run mode) | hand-over-hand traverse |
| Wall kick hand reach | ~3.9 m from a walking leap, ~5 m from a full-sprint wall run | world-class wall climb-up ≈ 4 m (the sprint is game pace) |
| Elbow interior angle | 35°–178° | human flexion limit |

Grab rules (`parkour/body.py::BodyModel.evaluate_grab`):
1. Torso must face the wall (`facing · -N >= 0.35`) and the body must be on the outside of the face.
2. The grip point is the rail point closest to a "reach centre" 0.25 m in front of the shoulders; its
   distance from the shoulder pivot must be <= 0.80 m and it must be in front of the chest.
3. On the ground the hold must be 0.30–2.25 m above the feet. Airborne it may be as low as 0.45 m
   below the shoulders (chest catch); while rising faster than 2 m/s only holds above the eyes are
   caught (apex catch — you fly past face-height holds).
4. The resulting hang body must be collision-free (§3.3); if only the ground is in the way the grip
   is *supported*; a `MANTLE_TOP` too low to hang from becomes a vault.
5. Among reachable holds the one closest to the shoulders and nearest the look direction wins.

```
               ┌───────────────────────────────┐
               │    Base Locomotion Forces     │
               └───────────────┬───────────────┘
                               │
            ┌──────────────────┴──────────────────┐
            ▼                                     ▼
   [ Ground / Sprint ]                     [ Airborne ]
   • Acceleration: 20.0 m/s² (walk)        • Air Steering: 7.0 m/s², never adds speed
                    6.0 m/s² (run)         • Horizontal Drag: 0.25 m/s²
   • Dynamic Friction: 14.0 m/s²           • Gravity: -19.62 m/s²
   • V_max: 5.0 m/s (Walk)                 • Sustained Jump Lift: +12.0 m/s²
            6.5 → 8.5 m/s (Run, built up      (Window: t <= 0.25s)
            over 1.6 s - see 6.1b)         • Wall Kick: v_y = min(0.6 v_y + 4.5 + e, 6.6 + e),
                                             e = 2.6 * wall-run factor (6.4)
```

### 6.1b Sprint Build-Up (`run_time`)
Run mode does not grant 8.5 m/s outright. The FSM keeps a `run_time` accumulator:
* it **fills** (1 s per s, capped at `RUN_BUILD_TIME` = 1.6 s) while on the ground in run mode with
  forward held, the speed at or above walking pace and the heading within 60° of the held direction;
* it is **zeroed** by a sharper change of direction (`RUN_BUILD_TURN_DOT` = 0.5) and by any rail or
  root-motion state (climbing is not running) - except a ground vault / step-up, which is part of the
  run (a speed vault, 7.1) and keeps it;
* it is **restored** to `climb_flow × RUN_BUILD_TIME` when a flowing climb tops out in run mode (7.2b);
* it **drains** at `RUN_BUILD_DECAY` = 2× rate when not running hard on the ground, and at 1× rate in
  the air (a leap keeps most of its build-up).

The run-mode target speed is `JOG_SPEED + (SPRINT_SPEED - JOG_SPEED) * smoothstep(run_time / RUN_BUILD_TIME)`,
so engaging run mode is a jog at once (5.0 → 6.5 m/s at 6 m/s²) and a full sprint after about a second
and a half of committed running (it was three seconds: a multiplayer chase turns too often for that). The camera's target FOV is `BASE + (SPRINT - BASE) * clip((|v_h| - 5.0) / 3.5)`:
it follows the actual speed, not the key, and stays wide through a running jump.

### 6.1 Velocity Integration
Player velocity V = [v_x, v_y, v_z]^T is integrated via Symplectic Euler:
V_(t+dt) = V_t + a_net * dt
P_(t+dt) = P_t + V_(t+dt) * dt

### 6.2 Variable Jump System
When `Spacebar` is pressed on the ground (or within coyote time after running off an edge):
0. **Take-off push** (legs, horizontal): with a movement direction `d` held, the horizontal velocity
   gains `push(|v_h|) * d` where
   `push(s) = max(0.4, P_stand + (P_sprint - P_stand) * sqrt(s / 8.5), P_stand - 0.75 s)`,
   `P_stand` = 3.8 m/s, `P_sprint` = 0.4 m/s. From rest the push *is* the jump (standing long jump
   ≈ 2.4 m, in any held direction - forward off a roof, backward away from a wall with `S`); at a
   sprint the legs add almost nothing and the leap is carried by momentum. With no direction
   held the jump is purely vertical.
   **Forward boost**: when the leap goes forward (within 60° of the facing), the horizontal velocity
   component along it is then multiplied by `JUMP_FORWARD_BOOST` = 1.15 (× the stick's tilt share on a
   controller): standing ≈ 2.8 m, walking ≈ 4.6 m, jogging ≈ 5.5 m, sprint ≈ 6.6 m. Backward / sideways
   leaps and the jump height are unchanged. The total can never exceed
   `(SPRINT_SPEED + P_sprint) · JUMP_FORWARD_BOOST`.
1. Ground flag cleared; initial instantaneous vertical impulse applied:
   v_y = 4.2 m/s  (tap: ~0.45 m rise)
2. Jump timer initialized: t_jump = 0.25 s.
3. During successive frames:
   * If `Spacebar` is held and t_jump > 0:
     v_y = v_y + (g_lift * dt),  g_lift = +12.0 m/s²  (held: ~0.95 m rise)
     t_jump = t_jump - dt
   * If `Spacebar` is released:
     t_jump = 0  (Instantly cuts upward lift)
4. Full downward gravity (g = -19.62 m/s²) applies at all times, producing a snappy, non-floaty parabolic trajectory.

### 6.3 Swept-AABB Collision Resolution
The player collision body is an Axis-Aligned Bounding Box:
* Dimensions: Width = 0.6 m, Depth = 0.6 m, Height = 1.8 m.
* For each movement step delta_P = V * dt:
  1. Broadphase: Expand player AABB by delta_P and query overlapping obstacle boxes.
  2. Narrowphase: Perform 1D Swept Minkowski tests along each axis sequentially (X -> Z -> Y).
  3. **Step offset**: if the horizontal move is blocked by a box whose top is at most `STEP_HEIGHT`
     (0.4 m) above the feet and the body fits on top, the body is lifted onto it and keeps moving
     (curbs, stair treads, bumpers). Only while grounded or falling, never while rising.
  4. On contact with surface normal n_hat:
     * Slide resolution: Zero out velocity component along normal:
       V = V - (V · n_hat) * n_hat
     * If n_hat == [0, 1, 0]^T: Grounded state flagged true.
     * Horizontal contacts report the wall normal, the obstacle and the approach speed
       (`ContactInfo`) to drive wall kicks and automatic step-ups.
  4. A kill plane at y = -30 m respawns the player.

### 6.3b Collision Buffers (holds next to other geometry)
A hold is only refused when something is really in the way:
* **Catch**: if the hang pose at the aimed grip clips something, `BodyModel.slide_to_free_grip` slides the
  grip up to `EDGE_COLLISION_BUFFER` (0.10 m) along the edge to the nearest spot whose hang pose is free and
  that is still a valid reach (arm's length, in front of the chest, body outside the wall). Typical case:
  a wall course right beside a balcony or fire-escape landing, where the two crosses of a tree meet.
* **Climb**: if a climb (or top-out) is blocked at the hands - an awning over a window head - and the
  player is not steering sideways, the FSM looks for the nearest spot within `CLIMB_SHIFT_MAX` (0.6 m)
  along the same hold from which it is clear, with a hangable pose all the way, and shimmies there
  (regular shimmy, no overshoot, never a corner turn) before climbing. Cached per spot; releasing W or
  steering cancels it; with no clear spot in range the climb is reported blocked as before.

* **Traverse blocked mid-edge** (`ParkourStateMachine._switch_target / _begin_edge_switch`): a shimmy step
  whose hang pose would enter something (an awning, a balcony, a fire-escape landing below a course)
  is *not* the end of the edge. If the obstruction has an edge of its own facing the player - from
  `EDGE_SWITCH_UP_MAX` (0.3 m) above down to `EDGE_SWITCH_DOWN_MAX` (2.15 m) below the current hold, with a
  free hang pose reachable within `EDGE_SWITCH_REACH` (1 m) along a collision-free straight path - the
  player switches onto it with an inside-corner move (turn to face it, lower onto it: `EDGE_SWITCH`,
  `EDGE_SWITCH_DONE`), and `W` then tops out onto it. Otherwise the traverse stops (`SHIMMY_BLOCKED`).
* **Corner turns** only start within `CORNER_TURN_END_TOLERANCE` (0.4 m) of the end of the edge, never
  carry the body more than `CORNER_TURN_MAX_SWEEP` (1.5 m), and their swing path must be collision-free
  (`BodyModel.path_is_free`). Root cause of the "thrown round the building" defect: any failed shimmy
  step used to be taken for the end of the edge, so a player who shimmied (or caught a hold with a
  direction held) into an awning was swept along the wall - through the awning - and round the corner
  the edge is joined to, up to ~9 m away. City-wide scan: 620 such throws in 1239 traverse / catch
  scenarios beside awnings, balconies, fire escapes and scaffolds before, none after.
* **Catch settle**: the 0.12 s blend from the catch position into the hang goes round, not through,
  what the player caught it from (across then down, or down then across).

### 6.4 Wall Kick (Tic-Tac)
When airborne, not yet kicked this jump, moving into a wall at >= 2.5 m/s and not falling faster
than 3 m/s, the foot plant converts run-up into height:
v_y = min(0.6 * max(v_y, 0) + 4.5 + e, 6.6 + e),  V += N_wall * 0.4
**Wall run**: `e = WALL_RUN_EXTRA_VY (2.6 m/s) × u`, `u = clip((approach - 7) / (10 - 7), 0, 1)` on the
horizontal speed into the wall (`WALL_RUN_SPEED_RANGE`; a walking leap meets the wall at ~7 m/s, a jog's
at ~8.4, a sprint's at ~10 with the forward boost).  From a walking leap this is the plain kick: a
centre-of-mass rise of 1.4–1.7 m, hand reach ≈ 3.9 m.  From a full sprint the rise is ~2.5 m and the
hands reach ~5 m, so the first-storey holds (the 3.8 m crevasse, the city's 4.0 m course) are caught
whatever the take-off distance, instead of falling back to the 2 m hold (which is what the old kick did
from a sprint: the hands peaked right at 3.8 m).  The kick also charges the climb flow (7.2b) from the
approach speed.

---

# 7. Parkour Action Trajectories & Splines

All root-motion states are keyframed body-centre paths, evaluated as a **monotone cubic Hermite spline**
through the keys (`_eval_root_motion`): the body starts and ends each move at rest and passes every key at
its time, but flows through the interior keys instead of stopping dead at each (the pull-up runs on into
the stand-up, a corner swing does not pause at the corner).  Tangents are per-axis weighted harmonic means
of the neighbouring slopes (Fritsch-Butland) and zero where the path turns back on an axis, so it never
overshoots a key; a single-segment move is exactly the old smoothstep.  Peak speeds stay under ~10 m/s
(0.17 m per tick) even in full climb flow, and a grab blends from the catch position into
the hang pose over 0.12 s (`GRAB_SETTLE_TIME`) instead of teleporting. Hang actions become
available only once the body has settled, so the arms are never asked to reach a hold from a pose
that cannot reach it.

### 7.1 Mantle Vault Trajectory
When a player initiates a mantle (a fresh `W` press while hanging from a `MANTLE_TOP` rail, or a
ground vault) the state machine takes direct kinematic control:
* **Stage 1: Hoist / knee hook**:
  * P0 = P_hang (or the catch position / standing position)
  * P1 = L(s) + y_hat * (h_half + 0.05 m) - N_wall * 0.1 m
  * Arms transition from crimp to palm-press on the top surface.
* **Stage 2: Step-Through**:
  * P_final = L(s) - N_wall * step_in + y_hat * (h_half + 0.02 m),  step_in = clamp(depth/2, 0.35, 0.75)
  * Transitions into `GROUND` locomotion with v_exit = 4.0 m/s (2.0 m/s from a dead hang), rising to
    `MANTLE_FLOW_EXIT_SPEED` (6.5 m/s) with the climb flow, which also restores the sprint build-up.
* **Durations**: 0.50 s + 0.25 s from a hang (the hoist keyed through the pull-up pose at 20 % of it);
  x0.6 in run mode; x(1 - 0.2 · flow) in climb flow (half the climbs' 0.4); ground vaults take 0.40 s per
  metre of wall (min 0.25 s). Climbing to a top edge or catching one mid-air never mantles by itself.
* **Speed vault**: a ground vault / step-up (or a mid-air catch of a top edge too low to hang from)
  records the approach speed - the horizontal speed plus what the wall took (`ContactInfo.approach_speed`)
  - and its direction.  Its duration is scaled by `1 + (VAULT_FAST_SCALE - 1) · u` (0.7 at a sprint, u
  from walk to sprint speed), the sprint build-up is kept through it, and it exits at
  `max(4.0, min(8.5, 0.92 · approach))` along the run's direction (when within 60° of straight over).
* **Validity**: the top surface must have room for the body box (nothing stacked on the edge).

### 7.2 Crevasse Climb-Up (`CLIMB_UP`) — leg work
From a hang on a crevasse at height y_c with a reachable hold above (<= 2.15 m over the lip):
1. **Pull-up** (0.38 s): body rises until the eyes are 0.05 m above the hold (hands at the chest).
2. **Foot on the lip, push up** (up to 0.47 s): the body rises with the foot on the lip only as far as the
   reach requires (`stand_fraction` in `BodyModel.find_hold_above`): a partial push-up for a close hold,
   a full stand (feet at y_c, chest to the wall) for a hold 2.15 m up. An overhang above the lip (a cornice,
   a balcony) limits how far the body can rise and therefore which holds are reachable. Holds off to the
   side (a balcony edge, the next bay's window head) qualify as long as the grip is within arm's reach of
   the shoulders at that height; a hold in the look direction (`AIM_CONE_DEG`) is preferred, otherwise the
   highest.
3. **Settle** (0.12 s): weight committed to the new hold, body eases into its hang pose.
   If the next hold is a `MANTLE_TOP` the climb ends hanging from it; with `W` still held the mantle follows at once (after the usual settle), otherwise the climber hangs until `W` is pressed again.
With `Shift` held (or run mode) all phases are scaled by 0.65 (climb at speed), and by the climb flow below.
**Climb-down** (`S` with a hold below) is the exact reverse: feet lowered onto the lip below,
body lowered to the pull-up pose of the lower hold, hands dropped to it, settle into the hang.

### 7.2b Climb Flow (`climb_flow`)
Momentum carried up a wall, 0..1.  **Charged** (never lowered) by the speed a wall is met with:
`flow_from_speed(s) = clip((s - 6) / (10 - 6), 0, 1)` of the wall kick's approach speed, and of the
body's `flight_speed` on any catch (the horizontal speed it left the ground with, or jumped with; 0 after
a launch off a hold).  **Chained**: every climb-up or top-out started keeps 92 % of it, and in run mode the
rhythm adds 0.15 (up to 0.5 from chaining alone).  **Drained** at 1.5/s while hanging settled without
climbing (or traversing), and zeroed on the ground.  **Used**: climbs and corner turns take
`1 - 0.4 · flow` of their time, top-outs `1 - 0.2 · flow`; a top-out in run mode leaves the edge at
`4.0 + (6.5 - 4.0) · flow` m/s with `run_time = max(run_time, flow · RUN_BUILD_TIME)`.
Measured (12 m wall, jump at 1.6 m, W + Space held): sprint 2.7 s, jog 3.0 s, walk 5.1 s (it was ~6 s
from a sprint - slower than from a jog, since the sprint kick fell back to the 2 m hold).

### 7.3 Corner Turns (`CORNER_TURN`)
Triggered while shimmying when the body can go no further along the rail and the rail has a link in
that direction whose entry hang pose is free (`BodyModel.corner_entry` slides inward from the corner end
until the body fits).
* **Convex**: keys `[mid, end]` over 0.58 s (x climb flow), mid = corner + (R.N + Q.N)̂ · (0.3·√2 + 0.08 m) so the body box
  clears the block corner while swinging round; the camera yaw blends by the signed angle between the
  two wall facings over the same time.
* **Concave**: key `[end]` over 0.42 s - the body barely moves, it rotates into the corner.
The FSM keeps `active_rail = R` until the turn completes, then hangs from Q at the entry parameter and
resumes `SHIMMY` if the key is still held.

**The view turns with the body** (`_turn_with_body(from_rail, to_rail, duration)`): every root-motion move
that ends on a hold facing another way - corner turns, edge switches, and `CLIMB_UP` / climb-down onto a
hold whose wall faces sideways (the side edge of an awning or balcony) - blends the camera yaw by the
signed angle between `-from.N` and `-to.N` over the move, so the player never ends up hanging with the
view looking along the wall ("hovering sideways"). Moves between parallel holds (the ordinary climb) have
a zero turn and leave the view alone.

### 7.4 Climb Down an Edge (reverse mantle)
`E` (fresh press) while standing within 0.9 m of a top edge of the surface underfoot, facing out over it,
with a free hang pose below it: keys `[mid (0.35 s), hang (0.65 s)]` = the mantle path in reverse, with a
180° camera turn over 80 % of the motion. The arms press the top surface for the first 40 % and then take
the edge. Ends in `LEDGE_HANG` (settled) on that edge.

### 7.5 Dyno Jump & Wall Ejection Trajectories
Velocities are set instantaneously from a *settled* hang and a **grab lockout timer** (tau_lockout = 0.22 s)
prevents re-grabbing the same rail immediately:
* **Dyno Up (`Space` while hanging)**:
  V = y_hat * 5.9 m/s + N_wall * 0.6 m/s   (~0.9 m body rise; hands reach ~1.2 m above the old hold)
* **Wall Eject (`Space + S` while hanging)**:
  V = N_wall * 4.5 m/s + y_hat * 3.0 m/s
  *The camera yaw blends 180° over 0.35 s to face the landing wall. Same-height range is only
  ~1.4 m; crossing the 3.2 m chasm means landing one crevasse lower (3.8 m -> 2.0 m).*
* **Lateral Dyno Leap (`Space + A/D` while hanging)**:
  V = ±T_rail * 5.0 m/s + y_hat * 4.0 m/s + N_wall * 2.0 m/s
  *Camera rolls ±4.5° toward the leap direction; the player turns toward the target pillar mid-air.*

### 7.6 Acrobatics: Flips and the Landing Roll (`parkour/acrobatics.py`)
A flip or a roll is a whole-body rotation - a `Spin(kind, duration, heading)` advanced by the FSM every
physics tick.  The collision box stays upright (a tucked body fits in it), so a spin changes the body's
*orientation* only: `body_transform(spin, center)` gives the rigid transform
`p' = pivot + R(axis, ±angle) · (p - crouch·ŷ - pivot)` that the game applies to the eyes (plus a head
tuck: -38° into a front flip / roll, +20° into a back flip) and, in the renderer, to the whole skinned
palette.  The rig itself is posed upright and tucked (`tuck()`: knees to the chest within the knee limits,
hands on the shins; a roll first puts the hands to the ground, `reach()`), so every joint stays anatomical.
* **Axis**: front flip / roll - `right(heading)` with the head going forward and down; back flip - the
  same axis the other way; side flips - about the heading.  The heading is fixed at the start (angular
  momentum), so turning the mouse mid-flip does not twist the spin.  A roll's heading is the travel
  direction: a backward roll (travel against the facing) is the same rotation seen facing the other way.
* **Angle**: `360 · (½u + ½(u - sin 2πu / 2π))` - a steady turn with soft ends; a roll spends its first
  10 % absorbing the landing (`crouch()` sinks the body 0.85 m) and is upright at 86 %, then stands.
* **Flip trigger** (AIRBORNE only): a fresh `jump` after a take-off by jump or wall kick, no flip yet this
  flight, and the airtime left `t_land` (ballistic fall to the surface under the body) with
  `0.92 · t_land ≥ FLIP_MIN_TIME` (0.30 s); duration `clip(0.92 · t_land, 0.30, 0.62)`.  The press is a
  *reach* instead when `find_best_grab` succeeds now or at the predicted pose 0.05-0.25 s ahead (stopping
  at the first predicted pose that would be inside something), or when the look ray has a hold within
  3.5 m that the body is flying toward (the reticle target).  While flipping there is no grab and no
  wall kick.  Kind from the held keys: `S` → back, `A`/`D` alone → side, no keys and < 1 m/s → back, else
  front.  Landing mid-flip `rush()`es the remaining rotation into 0.1 s.
* **Roll trigger**: on touch-down (`state ∈ AIR_PHYSICS` and `on_ground`) the drop `air_peak_y - feet`
  (highest feet of the flight, tracked every airborne tick) is compared with `ROLL_MIN_DROP` = 2.1336 m
  (7 ft).  State `ROLL` for 0.62 s: horizontal speed `max(3.0, 0.95 · v_h)` (≤ sprint) along the travel
  (speed < 1 m/s: the facing, or backward if a wall is in front), decaying 0.5 m/s²; the yaw turns toward
  a forward roll's direction over the first 45 %; if the ground 0.32 m + one step ahead drops more than
  0.6 m the roll stops moving (never off an edge).  The run build-up is kept, so a sprinter is still
  sprinting after the roll.  A jump in the second half springs out of it (a press up to the 150 ms input
  buffer earlier is held for it).  A flip that lands high queues its roll behind the rushed flip.
* **Flights end**: `took_off` / `flipped` are cleared whenever the body is not in an air-physics state
  (standing, hanging, climbing, rolling).  (They used to survive a catch and a top-out, so running off
  the far edge of a roof after jump → catch → mantle turned a coyote-time Space into a flip.)
* **Network**: the state message carries `ac = [kind, degrees, crouch, heading x, heading z]`
  (validated by `clean_acro`); remote avatars interpolate the angle and apply the same transform.

---

# 8. Modern OpenGL 3.3 Core Rendering Pipeline

The rendering system strictly enforces Modern Core Profile architecture. No legacy fixed-function calls (`glBegin`, `glVertex`, `glMatrixMode`) are used.

```
       ┌────────────────────────────────────────────────────────┐
       │                   Render Pipeline                      │
       └───────────────────────────┬────────────────────────────┘
                                   │
      ┌────────────────────────────┼────────────────────────────┐
      ▼                            ▼                            ▼
[ World Pass ]             [ Ledge Emissive Pass ]      [ First-Person Arms ]
• Default Shader           • Emissive Edge Shader       • Skinned Bone Shader
• Batched Static Geometry  • Glowing Line Strips        • Uniform mat4 u_Bones[46]
• Ambient + Directional    • Depth-offset to prevent    • Linear Blend Skinning
• Procedural Grid Lines      Z-fighting                 • Phong specular highlight
      │                            │                            │
      └────────────────────────────┼────────────────────────────┘
                                   │
                                   ▼
                        [ Screen-Space HUD Pass ]
                        • Orthographic 2D Projection
                        • Soft-lock magnetic reticle
                        • Emissive targeting brackets
```

### 8.1 Shader Program Specifications

#### 1. World Geometry Shader (`world.vert` / `world.frag`)
* **Inputs** (`world/obstacle.py::VERTEX_FLOATS` = 14): `vec3 a_Position`, `vec3 a_Normal`, `vec2 a_UV` (**metres across the face**: u along it, v up - on the sides of a facing object u runs back-to-front on both sides), `vec3 a_Color`, `vec3 a_Extra` = (`variant * 1000 + material * 10 + role`, face width, face height). Roles are `SIDE / FRONT / BACK / TOP / BOTTOM`, decided per face from the obstacle's `facing` vector; `variant` (0..9) picks a look within a material (an advertisement, a wall style, a bus livery, a dent). An obstacle can override the material of single faces (`face_materials`) and carry decorative geometry of any orientation (`append_mesh`, `box_mesh`) that has no collision and no holds.
* **Uniforms**: `mat4 u_Model`, `mat4 u_View`, `mat4 u_Proj`, `vec3 u_LightDir`, `vec3 u_CamPos`.
* **Fragment Logic**: directional diffuse, ambient fill, a world-space anti-aliased grid on plain horizontal surfaces, and **procedural materials** drawn in metric face coordinates so every detail is the right size on every vehicle:
  * `CAR_BODY` - door seams and handles, a sill, headlights / grille / plate on the front, tail and indicator lights / plate on the back, hood and boot seams on top; variant bit 1 = a scuffed dent in the rear door (with a perturbed normal); `CAR_GLASS` - side panes with pillars, windshield and rear glass with frames; variant bit 2 = a cracked pane. There are no painted wheel arches: the tyres themselves stand proud of the body sides.
  * `VAN` (cab window, cab and sliding-door seams with handles, stripe), `BUS` (a band of windows with pillars every 1.5 m, a front and a middle door with glass, roof hatches; variant 0 = yellow school bus with black rub rails and bumpers, variants 1..9 = metro livery: white body, a coloured belt and swoosh in a transit colour, roof line, route box and an amber destination sign), `TRUCK_CAB` (tall door with window and handle, chrome step, grille band) - all with windshields, lights, grilles and plates on the front and lights, plates and door glass on the back; a dent low on the side with variant bit 1.
  * `CONTAINER` - corrugated steel (25 cm vertical grooves, shaded with a perturbed normal; across the top), side rails, dark corner castings, rust where the paint has gone, and on the `BACK` end a pair of doors: central seam, two lock rods and a handle bar per door, hinge blocks. The same material dresses the container a truck carries.
  * `WHEEL_SIDE` (tyre ring, sidewall, chrome rim with spokes, hub and five lug nuts in a -1..1 radial uv) and `WHEEL_TREAD` (tread strips); `CHASSIS` (plain dark).
  * `WALL` - building walls in three styles (`variant % 3`): plaster (mottled paint with patches of the old coat), running-bond brick (0.24 x 0.075, each brick its own shade, the odd dark or chipped one, mortar) and cladding panels (1.2 x 1.0 with seams, replaced panels and rust streaks at the seams). Every wall is weathered (`weather()`: water runs from the top, grime low down, hairline cracks); `variant // 3` adds street-level graffiti (overlapping tags in the tag palette, only within 0.35-2.3 m of the ground in *world* height, so they never climb the floors), a mural across a blind wall (sun, hills, stripes, a painted frame, faded and chipped) or heavy water damage.
  * `WINDOW_WALL` - the glass seen through a facade's window recesses: 3 m floors, sill-to-lintel panes with a mullion, a few lit rooms, the odd boarded-up window and the odd broken one (fracture lines and a dark hole).
  * `BILLBOARD` - ten procedural advertisements by variant (logo disc, diagonal stripes, sunset skyline, checkerboard, target rings, dark poster, arrow, polka dots, colour bars, a torn old poster), sun-faded and grimy, on both faces; `LIGHT` is emissive (lamp heads, billboard lamps).
  * `BARK` (vertical fissures and ridges with a perturbed normal, knots) and `LEAVES` (noise-driven leaf clusters in light and dark greens with gaps and a little yellow, normal-perturbed for a bushy look).
  * `ASPHALT` (grain, lighter repair patches, sparse cracks, potholes) and `PAVEMENT` (1 m slabs with joints, settled cracks, chewing-gum spots); `TRASH` (glossy creased plastic).
  Glossy materials (glass, lights, paint, rims, bags) add a specular term.

#### 2. Ledge Rail Emissive Shader (`ledge.vert` / `ledge.frag`)
* **Inputs**: `vec3 in_Position`.
* **Uniforms**: `mat4 u_MVP`, `vec3 u_GlowColor`, `float u_Alpha`.
* **Rendering**: Interactive rails are drawn as extruded thin volume strips slightly biased along surface normals (epsilon = +0.005 m) using `glPolygonOffset(-1.0, -1.0)` to eliminate Z-fighting with wall surfaces.
* **Colors**: Cyan (`#00E5FF`) for active/reachable rails; Amber (`#FF9100`) for locked/grasped rails.

#### 3. First-Person Body Skinning Shader (`arm.vert` / `arm.frag`)
* **Inputs**: `vec3 in_Position`, `vec3 in_Normal`, `vec2 in_UV` (x = material id), `vec4 in_BoneIDs`, `vec4 in_Weights`.
* **Uniforms**: `mat4 u_Bones[46]`, `mat4 u_View`, `mat4 u_Proj`, `mat4 u_Model`.
* **Materials**: skin (forearms, hands), shirt (torso, upper arms), black pants (pelvis, thighs, shins), brown leather boots.

#### 4. UI / Reticle Shader (`ui.vert` / `ui.frag`)
* Rendered in Normalized Device Coordinates (NDC) with depth testing disabled (`glDisable(GL_DEPTH_TEST)`).
* Draws soft-lock brackets, range indicator rings, and center target dot.

---

# 9. Procedural World Generation Specification

The environment is generated procedurally via modular geometric blocks that automatically build the parametric ledge graph.

### 9.1 Obstacle Primitives
Levels are constructed from structured volumetric primitives:
```python
class ObstaclePrimitive:
    def __init__(self, x, y, z, width, height, depth, has_crevasses=True):
        self.bounds = AABB(min=(x, y, z), max=(x + width, y + height, z + depth))
        self.rails = []
        # 1. Top Mantle Edges (All 4 perimeter edges of top surface)
        # 2. Head Crevasses (y = 2.0m on exposed vertical faces)
        # 3. Jump Crevasses (y = 3.8m on exposed vertical faces)
```

### 9.2 Automatic Ledge Rail Extraction Algorithm
For every vertical face of an obstacle:
1. Determine face normal N (e.g. [0, 0, 1] for South face).
2. Compute horizontal tangent T = N x [0, 1, 0].
3. **Mantle Rail**: Instantiate rail along the top edge:
   P1 = C_top_left,  P2 = C_top_right,  type = MANTLE_TOP
4. **Crevasse Rails** (on *every* vertical face): If wall height H > standing reach (2.25 m):
   * Instantiate horizontal rail at y = 2.0 m (`HEAD_CREVASSE`), then y = 3.8 m (`JUMP_CREVASSE`)
     while at least 0.3 m below the top.
5. **Ladder Rails**: While the top is more than 2.15 m above the last crevasse, add another crevasse
   1.8 m higher (`LADDER_CREVASSE`). `Obstacle.crevasse_heights(H)` is the single source of truth and
   guarantees every object — including the 8 m arena walls — has a climbable hold ladder.

### 9.2b Buildings (`world/building.py`)
A `BuildingSpec` (footprint, floors of 3 m, facade faces, door face, balconies, rooftop) expands to boxes:
* **Core**: the solid block, inset 0.5 m on facade faces. Its plain faces carry **crevasse courses at the
  facade line heights** (`BuildingGenerator.course_heights`: sill `3n+1.0` and window head `3n+2.2` of every
  storey, topped up so no gap - the last one to the roof edge included - exceeds `CLIMB_HOLD_GAP_MAX`), and
  they run from *building* corner to building corner: a plain face is coplanar with the end faces of the
  facades beside it, so each course meets the facade's sill / head line (or the next plain face's course)
  at the corner and `link_corners` joins them. Every course therefore wraps the whole building, and every
  wall climbs to the roof by itself (the generic `crevasse_heights` ladder, computed for the core's own
  top hidden under the cornice, left 2- and 5-storey plain walls 2.2 m short of the roof edge).
* **Scaffold**: its decks cover the wall and block the courses behind them; if no hold lies 1.65-2.2 m
  above the top deck (reachable from standing on it, the legs out of the way), one is added at +1.93 m.
* **Facade** per face: horizontal *bands* (ground band 0→1.0 m, spandrels lintel→next sill, top band
  last lintel→roof) and vertical *piers* between 1.2 m windows (1.0 m wide); the ground band is split
  around a 1.2 m door in the middle bay. Band tops are **sills** (1.0, 4.0, 7.0 m), pier tops are
  **window heads** (2.2, 5.2, 8.2 m): holds the climber hangs from and stands on, never surfaces
  (mantle invalid in a 1.2 m recess). Gaps 1.2–1.8 m are realistic window-to-window climbing.
* **Cornice**: 0.3 m proud all round at the roof line; its top edges are the roof edges (mantle-able,
  linked round all four corners) and they overhang the sill beneath - so the last push-up is partial.
* **Balconies**: 2.6 × 1.5 m slabs on plain faces; reached from a crevasse lip beside them (aim at the
  balcony side edge), mantled from their side edge. The slab blocks standing on the crevasse below it.
* **Rooftop structure**: a 2.2 m box on the roof for a second level.
* **City placement** (`world/city.py`): street furniture is kept out of building protrusions in 3D
  (awnings, balconies, fire escapes, scaffolds - a bench under an awning is fine, a billboard post
  through it is not), and a skybridge slides along the shared alley wall to a spot clear of balconies /
  landings (or is skipped): no two structures interpenetrate (tested).

### 9.2c City (`world/city.py`, `world/props.py`)
`LevelGenerator.build_city(seed)` wraps the training zones in a 300 × 320 m city:
* **Grid**: block edges at x ∈ {-141,-101 | -87,-47 | -33,33 | 47,87 | 101,141} and
  z ∈ {-141,-101 | -87,-47 | -33,7 | 21,53 | 67,107 | 121,161}; 14 m streets (3 m sidewalks with
  0.15 m curbs, 8 m roadway); the park cells (x ∈ [-33, 33], z ∈ [-33, 53]) are skipped.
* **Blocks**: two columns (× two rows for deep blocks) of buildings whose widths are drawn so the
  alley between neighbours is 0 (flush, ledges join straight on) or 2.8–4.5 m (cat-leap range).
  Facades face the streets; plain faces get balconies or a fire escape (a landing per floor) and
  occasionally scaffolding (1.2 m platforms every 2 m); doors get awnings; 60 % of qualifying alleys
  get a skybridge slab one floor below the lower roof. Designated blocks become set-pieces: parking
  decks (slab on pillars, parapet, stepped 0.3 m ramp arriving flush), container yards (single and
  2 + 1 stacks), a construction site (concrete slabs on pillars every 3 m with scaffold towers).
* **Vehicles**: lanes 1.3 m off every curb are filled nose-to-tail with cars (hood 1.0 m - run over it,
  roof 1.5 m), vans (2.3 m), buses (3.0 m) and trucks (cab 3.1 m, container 4.0 m). Every vehicle
  stands on tyres (`props._wheel_mesh`: 14-sided cylinders built once per size and translated, attached
  to the body as decorative geometry - `Obstacle.append_mesh` - with no collision and no holds): the body
  box is lifted off the ground (cars / vans / buses 0.3 m, the truck's container 0.9 m) and the tyres'
  outer faces stand `WHEEL_JUT` = 4 cm proud of the body sides. Cars and vans have 4, buses 6 (dual
  rear), the semi is an 18-wheeler: 2 steer + 8 on two drive axles under the cab-over tractor, 8 on
  two axles under the container; the cab sits 0.4 m clear in front of the container (never inside
  it), a tractor frame runs back under the container, and chassis rails and landing-gear legs fill
  the gap. Vehicles have a `facing` (cars, vans, buses along their axis; the truck against it, so
  the container doors are at the far end from the cab) that the material shader uses for fronts,
  backs and doors. Colours vary: cars from `COL_CAR`, vans `COL_VANS`, truck cabs `COL_TRUCKS`, the
  containers they carry from the container palette; buses are mostly metro liveries in nine transit
  colours with the odd yellow school bus; about a third of the vehicles carry a dent or a cracked
  window.
* **Containers** sit on the ground (`y0 = 0`, corner castings and a bottom rail make that read) and use
  the same corrugated material with doors at their negative end; stacks put a third on top at 2.6 m.
* **Furniture**: benches, bins, planters, lamp posts (5 m pole, an arm reaching out over the road and a
  lit lamp head), bus shelters (roof at 2.3 m), billboards (panel top 6.5 m, from a roof or a truck; a
  different advertisement on each, lit by three lamps along the top edge) and piles of litter (bin bags
  and scraps) against the building side of the sidewalks.
* **Worn city**: streets are cracked asphalt with patches and potholes, sidewalks are paving slabs with
  joints and gum; buildings (`BuildingSpec.style` / `decor`, §8.1) are plaster, brick or panel in six
  palettes each, all weathered, many with graffiti at street level, some with a mural on a blind wall
  or heavy water damage; the glass behind the facades has lit, boarded-up and broken windows.
* **Nature** (park): trees (branch cross at 2.0 m, second cross at 3.7 m, canopy top 5.5 m; bark and leaf
  materials, the trunk and branches rounded to octagons by 45-degree twin boxes, a root flare, twigs into the
  crown and nine tilted leaf clumps plus a crown clump around the canopy box - all decorative, the climbable
  boxes are unchanged - climbed
  branch by branch), stepped boulders, a four-terrace hill, a fountain.
* **Perimeter**: 12 m walls with the crevasse ladder.
The layout is deterministic per seed; `ObstacleIndex` (8 m grid) keeps collision, hang-pose and
ground queries O(local) with ~7000 boxes and ~11000 edges.

### 9.3 Standard Playground Level Layout
The default procedural testing level instantiates a complete parkour gauntlet:
* **Zone 1: Locomotion & Variable Jump Runway**:
  * Low hurdles (0.8 m to 1.2 m) testing tap-hops vs full-hold jumps.
* **Zone 2: Crevasse Traverse Wall**:
  * Continuous 20 m long wall with a head-height crevasse (2.0 m) requiring grab and shimmying (`A`/`D`), with corner turns onto the end faces.
* **Zone 3: Vertical Double-Crevasse Tower**:
  * Pillar featuring a 2.0 m crevasse, a 3.8 m crevasse, and a 5.4 m mantle top. Slow route: hang 2.0 -> stand on its lip -> 3.8 -> stand -> mantle. Fast route: sprint, jump, wall kick, catch 3.8 directly, climb at speed.
* **Zone 4: The Dyno Chasm**:
  * Two opposing 6 m towers (crevasses at 2.0 / 3.8 / 5.6 m) separated by a 3.2 m gap, testing the backward Wall Eject (`Space + S`) across the void — landing one crevasse lower.
* **Zone 5: Pillar Hop Array**:
  * Four staggered square columns (2.0 m x 2.0 m x 4.8 m) with ~2.8 m nearest-corner gaps, testing lateral dyno leaps (`Space + A/D`) that land on the neighbouring pillar's lower crevasse.
* **Zone 6: Buildings**:
  * A three-storey building (10 × 8 × 9 m) with a windowed south facade and door, a windowed east facade, two balconies on the plain north face, a cornice and a rooftop structure - sills, window heads, balcony edges and the roof ring exercise holds-that-are-not-surfaces, overhangs, side reaches and corner links on a realistic structure.
  * An L-shaped block (12 m main wing, 6 m annex) providing outside corners and an inside corner on every crevasse row for corner-turning traverses.

---

# 10. Class Contracts & Module Breakdown

### 10.1 `math3d/`
* `Transform`:
  * `perspective(fov_deg, aspect, near, far) -> np.ndarray` (Returns 4x4 projection matrix).
  * `look_at(eye, target, up) -> np.ndarray` (Returns 4x4 view matrix).
  * `translate(v), rotate_x(a), rotate_y(a), rotate_z(a), scale(s) -> np.ndarray`.
* `Quaternion`:
  * `from_axis_angle(axis, angle_rad) -> np.ndarray` (q = [w, x, y, z]).
  * `slerp(q1, q2, t) -> np.ndarray`.
  * `to_rotation_matrix(q) -> np.ndarray` (4x4 rotation matrix).
* `Geometry`:
  * `point_segment_distance(pt, seg_a, seg_b) -> (float, float)` (Returns distance and projection factor s).
  * `ray_aabb_intersection(ray_origin, ray_dir, box_min, box_max) -> float | None`.

### 10.2 `parkour/`
* `LedgeRail`:
  * `p1: np.ndarray`, `p2: np.ndarray`, `normal: np.ndarray`, `tangent: np.ndarray`.
  * `rail_type: int`, `length: float`.
  * `project(point) -> (np.ndarray, float)`: Projects point onto segment, returns 3D anchor and clamped s.
* `LedgeSystem`:
  * `find_best_rail(camera_pos, look_dir, reach_radius) -> (LedgeRail, np.ndarray, float)`.
  * `get_rails_in_frustum(view_proj_matrix) -> list[LedgeRail]`.
* `BodyModel` (`parkour/body.py`):
  * `evaluate_grab(rail, center, facing_h, look_dir, on_ground, obstacles, velocity) -> GrabResult | None`.
  * `find_best_grab(...)`, `find_hold_above(rails, rail, s, obstacles) -> (rail, s) | None`.
  * `hang_center / pull_up_center / lip_stand_center / mantle_mid_center / mantle_target_center(rail, s)`.
* `ParkourStateMachine`:
  * States: `GROUND`, `SPRINT`, `AIRBORNE`, `LEDGE_HANG`, `SHIMMY`, `CLIMB_UP`, `MANTLE_VAULT`, `CORNER_TURN`, `DYNO_LAUNCH`, `WALL_EJECT`.
  * `update(dt, pos, vel, on_ground, contact, forward_h, right_h, look_dir, intents) -> (pos, vel)`.
  * Exposes `grip_point`, `climb_phase`, `climb_next`, `shimmy_dir`, `corner_link`, `corner_entry_s`, `mantle_reverse`, `last_event` for the arm rig, HUD and tests.
* `ControlScheme` / `Intents` (`core/controls.py`):
  * `resolve(input_buffer) -> Intents`; `Intents.has(name)`, `Intents.consume(name)`; `on_key_down(action)` for toggles.
* `LedgeSystem.build(obstacles)`: clip rails, link corners; `LedgeRail.links["left"/"right"] -> RailLink(rail, side, kind)`.
* `BodyModel.find_hold_above(rails, rail, s, obstacles, look_dir) -> (rail, s, stand_fraction)`, `find_hold_below(...)`,
  `corner_entry(link, obstacles)`, `corner_path(...)`, `find_climb_down_edge(rails, center, facing, obstacles)`.
* `HandPlanner` (`animation/hand_planner.py`): `snap`, `update(dt, desired)`, `positions()`, `resting_targets(rail, grip_d, shimmy_dir)`.
* `FirstPersonLegRig` (`animation/leg_rig.py`): `update(dt, player_pos, fwd_h, right, fsm_state, active_rail, grip_point, climb_phase, climb_next, velocity, on_ground, mantle_reverse, climb_direction, supported_hang, gait_phase, gait_weight)`; telemetry `last` (hips, knees, ankles, knee angles, toe directions, `LegState` per leg); owned and driven by `FirstPersonArmRig`, which also concatenates its mesh into the body mesh.
* `BuildingGenerator.build(BuildingSpec) -> [Obstacle]`, `build_l_block(x, z)`.
* `MenuModel` (`core/menu_model.py`): `open/close`, `handle(action, pos) -> RESUME | CONTROLLER | QUIT | None`.
* `InputRouter` (`core/input_router.py`): `press(source, action)`, `release(source, action)`, `set_held(...)`,
  `release_source(predicate)`, `release_all()`.
* `ControllerBindings` (`core/gamepad.py`): `key_for / bind / clear / reset`, `move_stick`, `look_stick`,
  `to_dict / from_dict`, `save(path) / load(path)`; `PadState.set_button / set_axis -> [(control, down)]`;
  `radial_deadzone`, `StickDirections`, `LookProcessor`, `MenuRepeater`, `detect_family(name, guid)`,
  `raw_layout(buttons, axes, hats)`.
* `GamepadManager` (`core/gamepad_device.py`): `process(event) -> [PadEvent] | None`, `active`,
  `active_state`, `rumble(low, high, ms)`.
* `GamepadInput` (`core/gamepad_input.py`): `on_button(control, down) -> PAUSE | Action | None`,
  `update(pad_state, dt) -> (turn_right_deg, look_up_deg)`, `release_all()`, `suppress_held(state)`,
  `menu_event(control, down)`, `menu_update(state, dt)`; `Haptics.on_tick(event, was_on_ground, on_ground, prev_vy)`.
* `ControllerMenuModel` (`core/controller_menu.py`): `open/close`, `select(item)`, `handle_key(name)`,
  `handle_mouse(kind, pos)`, `handle_pad(control, down)`, `handle_wheel(steps)` -> `BACK | None`.
  * `make_scheme(run_mode=False)`.
* `WindowManager` (`core/window.py`): `set_fullscreen(flag)` / `toggle_fullscreen()` (pygame's toggle keeps the
  GL context; `set_mode` is the fallback), `size` (the live drawable size); `load_display_settings()` /
  `save_display_settings()` (`display.json`), `fullscreen_from_args(argv, default)`.
* `MasterRenderer.resize(w, h)`: viewport + `ContextualReticle.resize`, `PauseMenu.resize`, `ToastHud.resize`
  (UI scale `min(w/1280, h/720)`, labels re-rasterised; the controller screen surface is capped at
  1920 × 1200 and stretched).
* `PlayerController` (`parkour/controller.py`):
  * Owns position, velocity, yaw/pitch, FSM, the active control scheme and the collision step; `step(dt, input_buffer)`; `set_scheme(name)`.
  * `apply_mouse(dx, dy)` (mouse counts), `apply_look(turn_right_deg, look_up_deg)` (controller).
  * `grab_candidate()` drives the reticle lock and the pre-reach pose.

### 10.3 `animation/`
* `TwoBoneIK`:
  * `solve_two_bone_ik(shoulder, target_wrist, l1, l2, pole_dir, forward_dir) -> IKResult(elbow, wrist, elbow_angle_deg, clamped)`.
    Bones are rigid; the wrist is clamped to the reach range given by the elbow limits (35°–178°); the pole
    direction is relative to the shoulder in the torso frame.
* `FingerSolver`:
  * `solve_hand(wrist_transform, surface_normal, grip_mode) -> list[np.ndarray]` (Returns 15 relative finger joint rotation matrices).
* `FirstPersonArmRig`:
  * `update(dt, player_state, camera_transform, target_rail, target_anchor) -> None`.
  * `get_bone_palette() -> np.ndarray` (Returns 1D flattened array of 38 mat4 transforms for shader uniform upload).

### 10.4 `render/`
* `GLShader`:
  * `compile(vert_src, frag_src) -> None`.
  * `set_mat4(name, np_matrix)`, `set_vec3(name, np_vec)`, `set_float(name, val)`.
* `MeshBuffer`:
  * Manages `glGenVertexArrays`, `glGenBuffers`, `glBufferData`.
  * Configures interleaved vertex attributes (Position, Normal, UV, BoneIDs, Weights).
* `Renderer`:
  * `draw_world(world, view_matrix, proj_matrix)`.
  * `draw_ledges(ledge_system, view_matrix, proj_matrix)`.
  * `draw_arms(arm_rig, view_matrix, proj_matrix)`.
  * `draw_reticle(reticle_solver, screen_width, screen_height)`.

---

# 11. Multiplayer (`game/`, `net/`)

```
 host's game process                          server process (net/hosted.py)          other players' games
 ┌───────────────────────────────┐   TCP     ┌───────────────────────────────────┐   TCP  ┌────────────────────┐
 │ main loop - MultiplayerSession ├──────────►│ GameServer (selector, 60 Hz tick) │◄───────┤ MultiplayerSession │
 │   └ GameClient                 │ loopback  │   Match: emblems / CTF, teams     │        │   └ GameClient     │
 │ HostLauncher ── stdin/stdout ──┼──────────►│   BotPlayer × n (PlayerController │        └────────────────────┘
 └───────────────────────────────┘  JSON      │     + BotBrain), its own city     │
                                               └───────────────────────────────────┘
```

### 11.1 Topology & protocol
A **listen server**: hosting from the game starts `python -m net.hosted <spec>` - a `GameServer` in a child
process (`ServerProcess`), so the bots (a dozen of them keep a core busy) never take time from the host's
frames and a server crash cannot take the game down.  The child builds its own copy of the level
(`game.match.city_world`: the city is a pure function of its seed - verified identical across processes
and hash seeds) and reports `{"t": "ready", "port"}` / `{"t": "error", "reason"}` and a status four times a
second on stdout; `stop` on stdin (or stdin closing, if the game dies) shuts it down.  Building the city
takes a few seconds, so `HostLauncher.poll()` is called every frame and the game keeps running meanwhile;
then the host joins over the loopback like anyone else.  `python -m net.hosted --bots 6 --mode ctf ...` is
a dedicated server.  (`GameServer` also runs in a thread - the tests do that.)
Messages are JSON objects, one per line (`net/protocol.py`, `PROTOCOL_VERSION` 4: `hello/state/team/ready/
bye` up, `welcome/reject/snap/pickup/flag/zombie/join/leave/team/respawn/over/ready_status/start` down).
`LineReader` caps lines at 256 KiB and drops anything that is not a JSON object with a string `"t"`;
positions go through `finite_vec`, flip / roll poses through `clean_acro`, names through `clean_name`
(printable, ≤ 20 chars, made unique by the server).  A client that stops reading is dropped once its
output buffer passes 4 MiB.  The server loop catches everything.
* **Rates**: clients send `state` 30×/s (keeping the cadence across frames); the server ticks at 60 Hz and
  broadcasts one `snap` 20×/s (encoded once for all clients) with the players and the match's changing state
  (`Match.snapshot()`: team scores, emblems left, the flags and zones).
* **Pose timeline**: the server stamps every player's pose with the time on its own clock it is from
  (`GameServer.now()`, real time since start: `pt` = when a human's `state` arrived, when a bot stepped) and
  every snapshot with `now`.  A client estimates `clock_offset` = local - server from the quickest snapshot
  arrival seen (creeping up by `MP_CLOCK_CREEP` = 2 % of any later one, so an early outlier or clock drift
  cannot pin it), and draws remote players on a clock that *slews* toward that estimate by at most 5 %
  (`MP_CLOCK_SLEW`; jumping only when it is 0.25 s off), so a better estimate never makes everyone lurch.
* **Remote players** are drawn `MP_INTERP_DELAY` = 0.1 s in the past on that timeline, interpolated between
  the two poses around that time (position, shortest-way yaw, a spin's angle; jumps > 5 m are not
  interpolated).  A pose carried by two snapshots (no new update in between: 30 Hz updates, 20 Hz
  snapshots) is recorded once; one past the newest pose (its update is late) is extrapolated along the last
  motion for at most `MP_EXTRAPOLATE_MAX` = 0.1 s, then held.  (It used to be interpolated on *arrival*
  time: jitter and the update / snapshot beat made a steady 8 m/s runner swing between 0 and 16 m/s from
  frame to frame - `tests/test_net_smoothing.py`.)
* There is no player-vs-player collision; players pass through each other (tags are proximity).

### 11.2 Slots, bots and teams (`net/server.py`)
`max_players` (≤ 15) counts everyone.  The host asks for `bots` - clamped to `max_players - 1` (the host's
own slot).  `_fill_bots()` keeps `min(bots, max_players - humans)` bots; a person joining a full game makes
a bot leave (`_make_room`, preferring a bot on the side the person asked for); a person leaving brings one
back.  Only people count toward "server full".
Teams (`game/teams.py`): people get the side they ask for in `hello` (or, asking for nothing valid, the
smaller side); `_balance_bots()` then moves bots (highest id first) from the bigger side until the sides
differ by at most one.  A `team` message switches sides mid-round: the match releases anything the player
carried, everyone is told, the player gets a `respawn` at the new side's spawn, bots re-balance.

### 11.3 Game rules (`game/match.py`)
`GameRules(mode, teams, time_limit, emblems, capture_limit, goal)` is the host's choice; `create_match()`
makes an `EmblemMatch`, a `CtfMatch` or a `ZombieMatch` (both `BaseMatch`: players → team, per-player scores, team scores,
positions, the clock, `results()`, `spawn_point()`, `bot_targets()`).  Every reported move (and every bot
physics step) is a segment `prev → pos`; `apply_move()` returns the events it caused.  A move longer than
6 m is a teleport / respawn and is not swept.
* **Emblems**: `EmblemPlacer` - ~50 % roofs, ~25 % tops of other climbable things, the rest in the
  streets, ≥ 14 m apart, on a free standing spot, hovering 1 m.  Pickup within 1.15 m of the segment; in
  teams each pickup also scores for the team.  The round ends when the last one is taken or on time.
  Teams spawn on their half of the start line.
* **Capture the flag, teams**: `CtfPlacer.bases()` picks two flat, free spots on open ground (a 3 m ring
  clear all round) as close as possible to `0.55 × map size` apart, clamped to 40-110 m (110 m in the
  city); each team spawns on free spots 3.5-9 m from its base.  Rules per move: touching the enemy
  carrier of your flag (`MP_TAG_RADIUS` 1.3 m, from either side) returns it (`returned/tagged`) - and the
  server sends the tagged carrier home; passing within 1.3 m of the enemy flag's touch point (1 m up its
  pole) takes it; being inside your base zone (3 m, feet within -0.6/+2.5 m of it) with the enemy flag
  while your own is at home captures (+1 player, +1 team).  A teleport, leaving or switching sides
  returns a carried flag.
* **Capture the flag, free for all**: one neutral flag placed like an emblem (roof / top / street) and a
  drop zone on open ground 30-80 m away.  Touching the carrier steals the flag unless it changed hands
  less than 2.5 s ago; a capture scores for the carrier and re-places both.
* **End**: the capture limit (if set) ends a round at once; otherwise the time limit.  Winners: the team
  with more points (equal and > 0: a draw; nothing scored: nobody wins) or the top-scoring players.
  After `over` each client shows the results; *Join new game* sends `ready`; the next round (everything
  re-placed, everyone back at their spawn) starts when every connected person is ready or after 30 s.

* **Zombie tag** (`ZombieMatch`): at the round start one player is picked at random (the round's seeded
  RNG) to be *living*; everyone else is a zombie.  The living player spawns on a free street spot
  35-60 m from the start line, the zombies on it.  Every tick the living player gains
  `ZOMBIE_POINTS_PER_SECOND` (1) - `points` keeps the fraction, `scores` the whole points.  A catch is
  the tag test of capture the flag (1.3 m between the mover's segment and the other's last position,
  either of them moving): the catcher becomes the living player, the caught one a zombie.  Nobody can be
  caught for `ZOMBIE_START_GRACE` (5 s) at the start or `ZOMBIE_CATCH_GRACE` (3 s) after a catch, and
  there are no tag-backs: the player just caught cannot catch their catcher for `ZOMBIE_TAGBACK_GRACE`
  (10 s) - others can.  The living player leaving hands life to a random other player (with the catch
  grace); someone joining an empty game is alive.  The round ends when the living player reaches the
  goal (30-600 points) or on time (most points).  Events the match raises by itself (who is alive) go
  through `drain_events()`.  **Speed**: movement is simulated by each client, so each applies its own
  role: `PlayerController.set_speed_scales(run, climb)` - zombies `ZOMBIE_RUN_SCALE` 0.88 on the ground
  speeds (walk / jog / sprint targets) and `ZOMBIE_CLIMB_SCALE` 0.85 on climbs, mantles, corner turns
  (root-motion time) and shimmies; the session sets it from the client's view of who is alive, the
  server sets it on its bots every tick.
* **No time limit**: `GameRules.time_limit` 0 means none (`unlimited`), offered for every game as the
  last *Match time* choice; `time_left` is then `None` on the wire and the HUD says "no time limit".
  An emblem hunt still ends with its last emblem and zombie tag on its goal; capture the flag with no
  capture limit as well never ends (`never_ends()`) - the host is warned.

### 11.4 The bots (`game/bot.py`, `game/nav.py`)
`BotPlayer` is a headless `PlayerController` stepped at 60 Hz inside the server tick; `BotBrain` drives it
only through an `InputBuffer` (the keys a player has) and the camera, so it obeys every rule of the body
model (it rolls when it lands from high up; it never flips - it reaches with `E` in the air, so no fresh
`Space` press).
* **What to do**: `match.bot_targets(pid, role, claimed)` - `Target(id, pos, radius, kind)`: untaken
  emblems (in teams minus those a teammate bot is already after); CTF teams: the enemy flag → home with it
  → after the enemy carrying ours (when nearer than the flag, or once a teammate has theirs) → home to
  meet a teammate carrying theirs; keeps its first bot on `defend` (guard the flag from `GUARD_DISTANCE` = 7 m out toward the enemy base - standing on the stand it tagged every thief the instant it touched the flag - and chase a thief)  FFA CTF: the flag → the carrier → the drop zone.  Zombie tag: a zombie chases
  the living player; a living bot runs for one of the round's street / roof / top spots 12-70 m away,
  past the nearest zombie (on the far side from it), as far from the zombies as possible - kept for up
  to 8 s, or until it is reached or a zombie gets within 8 m of it.  The brain takes the cheapest
  (distance + 2.5 × climb), follows a moving target (goal re-centred, route re-planned when it moved 4 m
  or every 2 s), and stays put when it is where it should be (guarding).  The server resets goals when a
  flag event changes the objectives.
* **Navigation**: `NavGrid` - 1.5 m cells, blocked by anything ≥ 1.05 m tall (+0.35 m margin); a flat
  bytearray for the free test; a straight line of sight skips the search; otherwise weighted A*
  (heuristic × 1.4 on the distance to the goals' centroid less their spread: routes within ~1 % of the
  shortest, 3× faster), string-pulled by line of sight.
* **Ground / hanging / roofs**: as before - sprint lock on, climb only the goal's obstacle, low things or
  the route's end (else detour along the wall), climb with `W` looking at the goal, shimmy when blocked,
  let go of dead-end holds, leap across gaps with something to land on, drop off otherwise;
  `wants_respawn` after 30 s without progress.
* **Wall runs**: running at ≥ 6.2 m/s toward a tall wall (≥ 2.6 m over the feet) it means to climb, with the
  face 1.2-2.1 m ahead and a hold on it 1.8-4.9 m over the feet within 45° of square, the bot squares up,
  taps `Space` with `W` + `E` held and lets the wall run carry it to the high hold (then climbs on in flow).
  On the last leg of a route to a building it runs straight at the bit of wall beside the last waypoint
  (when the way there is clear and the angle square enough) instead of turning at the waypoint, so it
  meets the wall at pace.
* **Cornering**: within `1.2 + 0.15 v` m of a waypoint where the route bends more than ~50°, going faster
  than 5 m/s, it lets go of `W` for a moment (ground friction brakes hard) - at a sprint it used to carry
  on past the mouth of a narrow alley into the building beside it.  A waypoint only counts as reached
  (within 1.2 m) once the next one is in a straight line from where the bot is (or within 0.5 m).
* **Reaching up**: a goal on top of something a hop can reach (≤ `STAND_REACH + 1` m over the feet: a
  bus-shelter roof, an awning, a container) with no wall in the way - nothing under a slab blocks you -
  is approached at its nearest edge (out from under it first) and hopped for with `E` reaching; "step onto
  it" (hold `W`) is only for surfaces a step or vault up.
* **Stuck**: picking the same goal again after giving up on it no longer counts as progress, so a bot that
  cannot reach its last target asks to be respawned after 30 s instead of trying for the rest of the round.
* **Cost**: one bot ≈ 0.3-0.8 ms per tick on the city (controller step + brain) plus ≈ 0.1-0.2 ms of tactics
  (11.6) in an emblem hunt; 14 bots ≈ 50-72 % of a core (emblem hunt: 8.5 / 12.0 / 11.1 ms a tick for Simple /
  Complex / Hard, from 9.7 ms before - the pickup test now checks every emblem at once) - hence the server process.  `BodyModel.find_best_grab`, `_holds_near` and the reticle's
  `LedgeSystem.find_best_rail` use a 6 m XZ grid over the rails (`_rail_grid`), so a query looks at the
  edges near the body only (reach 0.25 ms instead of 3.3 ms, reticle 1.0 instead of 5.9 ms on the city's
  10 000 edges, identical results) - the local player's frames gain from it too.
Measured (the brain alone, handed every emblem - its motor skills; in a server game the bots have to find the
emblems first, 11.6): a lone bot takes all 6 emblems of the test level for each of 8 seeds in 27-39 s (mean 34 s; it
was 34-42 s, mean 39 s, before the momentum changes) and all 12 of a full city round for each of emblem seeds 1-5
in 213-329 s (before: 342 s for seed 1, and stuck on its last emblem for good on seeds 2-5 - the narrow
alley, the bus-shelter roof and the like above); in CTF on the test level 2 v 2 bots capture within minutes,
3 bots in a free for all too (tests).

### 11.5 Client side (`game/session.py`, `core/mp_menu.py`, `render/mp_render.py`)
`MultiplayerSession.update(dt, player)` sends the local pose (with the spin) at 30 Hz, applies `welcome` /
`start` / `respawn` (teleport to the given spawn, not counted as a fall), turns events into HUD notices
(pickups, flags taken / stolen / captured / returned / tagged, catches, who is alive, joins, leaves,
team changes), applies the zombie-tag speed scales to the local player (and the game turns a zombie's
own arms green, reaching forward: `FirstPersonArmRig.update(zombie=True)`, `u_SkinColor`) and keeps
the results; `results_view()` / `hud_state()` are what the screens draw (in zombie tag `hud_state`
also lists the chase: the living player for a zombie, the nearest zombies within 40 m for the living).  `MultiplayerMenuModel` has two
columns - you (name, team, join address) and the game you host (port, max players, bots, game, teams,
match time or unlimited, emblems / capture limit / goal points) - validates them and persists `multiplayer.json`; changing your
team in a session asks the server to switch sides.  `MultiplayerRenderer` draws the other players
(`avatar_parts`: posed from the FSM state, team-coloured shirts, a flag on the back of a carrier, flips and
rolls through `acrobatics.transform_for`; a zombie has green skin, rags, glowing eyes, arms held out in
front and a slow, limping, side-to-side lurch; the living player a gold column of light), culled beyond 150 m or behind the camera; emblems; flags on
their stands, base / drop-zone rings and light columns (the aura shader stretched into a beam); and the
HUD - the game and score, the clock, the scoreboard in team colours, the carrying banner, and
the zombie-tag banner (alive and your points / a zombie and who to catch, the head start or tag-back
wait), and `markers()`: each flag (and, while carrying, your base / the drop zone) or chase target
projected to the screen, or pinned
to the screen edge in its direction when off screen (along the bottom when behind), with its distance.


### 11.6 Bot intelligence levels (`game/bot_skill.py`, `game/sight.py`, `game/tactics.py`)
The host picks the level (`HostSettings.bot_level` → `ServerProcess` spec → `GameServer(bot_level=…)`; the
multiplayer screen's *Bot skill* row, shown when there are bots; `--bot-level`; `GameServer.set_bot_level(level,
pid=None)` changes it live, one bot or all - the benchmark mixes levels that way; `status()` reports it).
**Layers**: `BotBrain` (game/bot.py) is the body and motor skills - it goes where it is told; a `Tactician` per
bot decides *where*, from what the bot can know, and hands the brain `Target`s (the same objects the rules'
`bot_targets` produce).  Bots of one level and one side share a `Hive` (`GameServer.hives[(level, team)]`; free for
all: one per level); a `Tracker` keeps everyone's recent positions.  Tactics run on the server's simulated clock
(`GameServer.time`); a new round resets hives and tacticians.

| | Simple | Complex | Hard |
| :--- | :--- | :--- | :--- |
| Sight (range, field of view) | 70 m, 110° | 100 m, 170° (all round on a roof) | 140 m, 360° |
| Emblems kept in mind | 1 (the nearest) | 3 (the nearest) | all, shared with its side |
| Areas remembered as searched | 6 | 100 (~¼ of the city) | all, shared with its side |
| Sighting error | 8 % of the distance | 4 % | exact |
| Search | wander somewhere unsearched | frontier + vantage roofs | sweep: split the map, skip lost races, walk the roofs when a sweep misses |
| Fleeing (zombie tag) | the game's rule | the same; on a roof with a lead: on across the rooftops | the same, more readily |
| Chasing | flat out | flat out | flat out; a camper in a flag thief's base |

**Sight** (`HeightField`): a 1 m grid of the tallest top in each cell (36 ms for the city); a sight line is
sampled every 0.5 m and blocked where a cell's top is above it (overhangs count as solid to the ground -
pessimistic, never optimistic).  An emblem is seen when its centre or the top of its glow (+0.5 m) is in line of
sight, in range and in the field of view (always within 7 m).  40 sight lines cost 0.5 ms.

**Emblems** (`Tactician._emblem_hunt`): bots look 5× a second.  A sighting's estimate is offset by
`position_error × distance` in a fixed direction per (bot, emblem) - kept on the same surface (halved until the
ground there matches) - and replaced when seen again from closer, so it sharpens as the bot closes in.
`EmblemMemory.remember` evicts the furthest when over capacity.  The bot goes for the known emblem of least
`distance + 2.5 × climb` (the current one kept unless another is 25 % cheaper); teammates' claims (`claimed`,
from the brains' goals) are avoided when there is a choice; Hard bots use the hive's claims and skip an emblem a
rival will reach first (rival's cost < 0.75 × theirs: +60).  A remembered emblem someone took is forgotten when
its spot is seen (or reached); one reached but not there (a rough estimate) is forgotten after 1.5 s unseen.  The
server no longer tells the bots about pickups (only the one that picked it up).

**Search** (`SearchGrid`, `SearchMemory`): the map in 16 m areas, each with up to four spots an emblem could
float over - the street nearest the middle, the main roof level and its far side, the highest top.  Every 0.4 s
a bot checks the unseen spots of the nearest `scan_cells` areas in view (one batched sight test) and marks what it
sees (and anything within 7 m).  An area is searched when all its spots are; `SearchMemory` keeps at most
`search_memory` searched areas (least recently looked at forgotten first).  Exploration (`_explore_target`):
*wander* - a random unsearched area 15-70 m away; *frontier* - the nearest unsearched area weighted toward ones
with unsearched neighbours and the way it is facing; *sweep* - the same, away from the areas hive-mates are
exploring.  Complex and Hard also weigh **vantage roofs**: a roof ≥ 5 m within 45 m, the standing spot 1.5 m in
from its edge facing the unsearched ground (from the middle of a roof its own edge hides everything lower), scored
by the unseen spots it really has a line of sight to.  The street spot of an area is tried first, its roof
(climbed) next; once the whole map has been looked over without finding everything the memory is cleared and the
next sweep walks the roofs.  Places the bot could not get to are avoided for 90 s (`failed`).

**Fleeing** (Complex and Hard; Simple keeps `ZombieMatch._flee_target`): the plain rule, with one addition.  Up
on a building's roof, with no zombie within `SAFE_LEAD` = 25 m and a roof a leap away (`World.roof_links`: gaps
≤ 4 m, the far roof at most 1 m higher), the bot may run on across: a linked roof past the nearest zombie, scored
like the plain rule plus 3 (Hard 6), taken when it beats the plain rule's pick; then kept until reached, 8 s, or a
zombie within 8 m.  Everything else that was built and measured made the runner *easier* to catch
(`tools/bot_levels.py flee`, 32-64 seeds each): avoiding the map's edges and corners, keeping its line while
pressed, path safety, runs of at least 25 m, preferring straight clear lines, reading the zombies' lines,
dead-end avoidance; and - what was asked for, to be hard to follow - spending a lead on climbing onto roofs at a
sprint and on sudden changes of direction: the living bot survived 56-58 s on average against 70 s for the plain
rule, because a bot zombie is not fooled by a jink and climbing gives the lead back.

**Chasing** (`PackPlanner`): every hunter runs the quarry down, straight at its live position; a Hard side with
a flag thief running home posts the hunter that can beat the thief there in the thief's base.  That is the result
of measuring, not the starting design (`tools/bot_levels.py chase`, many seeds each, the living player 13 % faster
than its zombies): a runner that is faster than its hunters is caught when it slips - a turn, a climb, something
in the way - and only hunters close behind it can take the chance.  Every way of spreading the pack out that was
built and measured caught the runner *less* often than the flat-out chase: two flankers a street over, tight
flankers 6 m out, cut-offs where a hunter could be first, a tight net round it on its open sides, far hunters sent
ahead along its escape line, a hunter waiting where it was guessed to be heading (with the guess checked against
where it went and trusted less when wrong), a ring of hunters 30 m round it, and hunters meeting it on its way -
against the game's runner, a runner that flees like a Hard bot, and one that runs in long straight lines the way
people do.  (A spread-out pack kept 0.5 hunters within 15 m of the runner where a flat-out one keeps 1.7.)  With
zombies as fast as the living player, positioning would be worth more; that is a balance choice
(`ZOMBIE_RUN_SCALE`) for another day.

**Measured** (`python tools/bot_levels.py …`, the city, final code):
| Experiment | Simple | Complex | Hard |
| :--- | :--- | :--- | :--- |
| `hunt`: one bot alone, emblems found of 12 in 300 s (6 seeds) | 3.8 (1-7) | 8.0 (4-10) | **10.3** (8-12) |
| `race`: one bot of each level in the same hunt, emblems / outright wins (8 seeds) | 8 / 0 | 26 / 1 | **55 / 7** |
| `chase`: 3 zombies vs the game's runner, seconds to a catch (32 seeds) | 70.2 | 70.2 | 70.2 |
| `flee`: the living bot vs 3 Simple zombies, seconds alive (32 seeds) | 70.2 | 70.2 | 70.2 |

The emblem hunt is where thinking pays, and every level beats the one below it on every seed.  In zombie tag the
levels play the same (see *Fleeing* and *Chasing* above for why).  Capture the flag with bots (3 v 3, small world
and city, 300 s) is a stalemate at every level - a few captures at most and one thief sent home every 4-6 s -
because every opponent near a flag tags a thief that slows to take it; moving the defender off the stand
(`CtfMatch.GUARD_DISTANCE`) cut the tags by a quarter but did not break it.  That is a rules / balance question
(flag and tag radii, base distance) for another day.


---

# 12. Implementation Roadmap & Milestones

1. **Milestone 1 — Core GL 3.3 Context & Math Core**:
   * Modern OpenGL windowing via Pygame.
   * Complete NumPy 3D math suite (transformations, quaternions, geometry projections).
   * Shader compilation pipeline and basic textured/grid floor.
2. **Milestone 2 — Kinematic Controller & Action Buffer**:
   * Timestamped input queue (150 ms buffer window).
   * Swept-AABB collision physics vs static procedural blocks.
   * Momentum integration, sprint FOV lerping, and 0.25s variable jump reservoir.
3. **Milestone 3 — Ledge Graph & Soft-Lock Targeting Reticle**:
   * Procedural block rail extractor (`HEAD_CREVASSE`, `JUMP_CREVASSE`, `MANTLE_TOP`).
   * Emissive ledge highlight shader pass.
   * Screen-space soft-lock magnetic reticle with distance-based bracket tightening.
4. **Milestone 4 — Parkour FSM & Traversal Mechanics**:
   * Ledge anchor snapping and clamped shimmying along rail tangents.
   * Chained multi-crevasse climbing.
   * Hermite spline root motion for the two-stage Mantle Vault.
   * Dyno jumps, wall ejections and lateral dynos (`Space` + direction while hanging).
5. **Milestone 5 — Procedural 2-Bone Arm, Hand, and Finger IK**:
   * Camera-anchored clavicle/shoulder system with rotational inertia.
   * Analytical 2-bone arm solver with pole vector constraints.
   * Procedural finger curling (crimp grip vs flat mantle palm-press vs airborne pre-reach).
   * GPU Linear Blend Skinning (LBS) via 46-bone uniform array palette.
6. **Milestone 6 — Playground Course & Polish**:
   * Build complete 5-zone parkour obstacle course.
   * Integrate camera kinematics (head bobbing, Dutch roll tilts during dynos).
   * Frame-pacing verification, friction tuning, and release polish.
