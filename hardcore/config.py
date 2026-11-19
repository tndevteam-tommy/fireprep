"""Configuration parameters for the Hardcore parkour engine.

All physical tuning values are calibrated against what a talented parkour
professional can actually do (see DESIGN.md §6 "Human Performance Envelope").
Gravity is deliberately 2x Earth gravity for a snappy game feel, so every
impulse below is expressed in terms of the *outcome* it produces (rise height,
jump distance) rather than real-world launch velocity.
"""
import math
import numpy as np

# Display & Window
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
WINDOW_TITLE = "HARDCORE — 3D Parkour Engine (Modern OpenGL 3.3 Core)"
FULLSCREEN = False                # default display mode (F11 / Alt+Enter / the pause menu switch it; remembered
                                  # in display.json; --fullscreen / --windowed on the command line override)
UI_DESIGN_SIZE = (1280, 720)      # the HUD and menus are laid out for this and scaled uniformly to the screen
FPS = 60
FIXED_DT = 1.0 / 60.0

# Camera
BASE_FOV = 75.0
SPRINT_FOV = 90.0
FOV_LERP_SPEED = 9.0
MOUSE_SENSITIVITY = 0.12
PITCH_MIN = -89.0
PITCH_MAX = 89.0
HEAD_BOB_FREQ = 10.0
HEAD_BOB_AMP = 0.035
DUTCH_TILT_SPEED = 8.0
MAX_TILT_DEG = 4.5

# ---------------------------------------------------------------------------
# Player Body Model (meters) — 1.80 m athlete
# ---------------------------------------------------------------------------
PLAYER_WIDTH = 0.6
PLAYER_DEPTH = 0.6
PLAYER_HEIGHT = 1.8
PLAYER_HALF_HEIGHT = PLAYER_HEIGHT * 0.5
PLAYER_EYE_HEIGHT = 1.65
SHOULDER_HEIGHT = 1.45            # above the feet
SHOULDER_HALF_WIDTH = 0.21
SHOULDER_FORWARD = 0.10           # shoulders sit slightly ahead of the AABB centre
CAMERA_FORWARD = 0.14             # the eyes sit ahead of the shoulder line: the camera is this far in front of
                                  # the body centre, so the arms enter the view from its lower corners
# Legs (1.80 m athlete): hip joints just below the body centre, thigh + shin + ankle = hip height
HIP_HEIGHT = 0.935                # above the feet (a hair under the straight-leg length: knees never lock)
HIP_HALF_WIDTH = 0.10
THIGH_LEN = 0.44
SHIN_LEN = 0.42
ANKLE_HEIGHT = 0.08               # ankle joint above the sole
FOOT_LEN = 0.27
KNEE_MIN_ANGLE_DEG = 40.0         # deepest flexion (heel to the buttock)
KNEE_MAX_ANGLE_DEG = 179.0        # straight, never hyper-extended
FOOT_ON_HOLD_MIN_HIP_ABOVE = 0.22 # a foot steps onto a hold only once the hips are this far above it (knee flexion limit)

# Arm segments (shoulder pivot -> elbow -> wrist) plus the hooking part of the hand
UPPER_ARM_LEN = 0.33
FOREARM_LEN = 0.27
HAND_GRIP_LEN = 0.10              # wrist -> crimped finger pads on the hold
ARM_REACH = UPPER_ARM_LEN + FOREARM_LEN + HAND_GRIP_LEN          # 0.70 m
REACH_STRETCH = 0.10              # shoulder shrug / tip-toe allowance
MAX_GRAB_REACH = ARM_REACH + REACH_STRETCH                       # 0.80 m from shoulder pivot
STAND_REACH = SHOULDER_HEIGHT + MAX_GRAB_REACH                   # 2.25 m: highest hold from flat ground
CLIMB_HOLD_GAP_MAX = 2.15         # largest hand-hold gap climbable from a foot-lip (margin under STAND_REACH)
GRAB_BELOW_SHOULDER_MAX = 0.45    # a hold can be caught as low as the chest while airborne
RISING_GRAB_VY = 2.0              # while rising faster than this only holds above the eyes are caught
GRAB_LOOK_WEIGHT = 1.2            # how strongly the look direction picks between reachable holds
AIM_CONE_DEG = 45.0               # a hold this close to the look direction is preferred when climbing
STEP_HEIGHT = 0.40                # curbs, stairs and bumpers this high are simply walked over
GROUND_VAULT_MIN_HEIGHT = 0.45    # lowest top edge worth a vault / step-up (anything lower is a step)
AUTO_STEP_MAX_HEIGHT = 1.00       # running into anything this low steps up automatically
AUTO_STEP_WALK_MAX = 0.60         # flow scheme: walking only steps up knee-high obstacles

# Wrist: the hand may bend at most this far from the forearm axis (flexion/extension/deviation combined)
WRIST_MAX_ANGLE_DEG = 78.0
# Grip model (see animation/grip.py): how far the wrist bends on each kind of edge and phase
GRIP_LEDGE_MIN_DEPTH = 0.25       # an edge with less surface behind it than this is a thin (crimp) edge
WRIST_EXT_HANG_CRIMP = 15.0       # hanging on a thin edge: hand nearly in line with the forearm
WRIST_EXT_HANG_LEDGE = 26.0       # hanging hooked over a deep ledge
WRIST_EXT_PROP = 70.0             # hands at chest level / body above the hands (pull-up top, propping)
WRIST_EXT_PRESS = 80.0            # palm flat on a surface (mantle, lowering over an edge)
WRIST_EXT_OPEN = 6.0              # open hand in the air (hops, reaching)
WRIST_EXT_RELAXED = 25.0          # relaxed hand after letting go of a surface
CLEAR_BLEND_TIME = 0.35           # a freed hand eases into that wall-clearing flex over this long
HAND_OPEN_LEN = 0.21              # wrist -> fingertips of an open hand
WRIST_FLEX_CLEAR_DEG = 55.0       # an open hand near a wall flexes back up to this much so the fingers run along it, not into it
WRIST_EXT_REST = 12.0             # rest / running: fist nearly in line with the forearm, slight flexion
WRIST_REST_ROLL = 70.0            # rest / running: palms rolled to face inward (thumbs up)
WRIST_EXT_HIGH_RISE = 0.45        # wrist this far above the shoulder -> full hang extension
WRIST_EXT_LOW_RISE = -0.05        # wrist at / below the shoulder -> prop extension
WRIST_OUT_CRIMP = 0.04            # wrist position relative to the edge while hanging (out of the face, below)
WRIST_DROP_CRIMP = 0.09
WRIST_OUT_LEDGE = 0.05
WRIST_DROP_LEDGE = 0.07
WRIST_PRESS_IN = 0.14             # palm press: wrist behind the edge, just above the surface
WRIST_PRESS_UP = 0.025
MANTLE_RELEASE_PHASE = 0.60       # mantle progress at which the palms start leaving the surface (the step-through begins at ~0.67)
# Elbow joint limits (interior angle between upper arm and forearm)
ELBOW_MIN_ANGLE_DEG = 35.0        # ~145 deg of flexion is the human maximum
ELBOW_MAX_ANGLE_DEG = 178.0       # never lock straight (prevents IK singularity)

# ---------------------------------------------------------------------------
# Locomotion Dynamics
# ---------------------------------------------------------------------------
WALK_SPEED = 5.0
JOG_SPEED = 6.5                   # run mode: the pace you are at almost at once ...
SPRINT_SPEED = 8.5                # ... and the top speed (~30 km/h) you build up to over RUN_BUILD_TIME
RUN_BUILD_TIME = 1.6              # seconds of continuous running from a jog to full sprint (was 3.0: a
                                  # multiplayer chase turns a lot - top speed has to come back quickly)
RUN_BUILD_DECAY = 2.0             # the build-up drains this many times faster than it fills when you ease off
RUN_BUILD_TURN_DOT = 0.5          # a change of direction sharper than this (cosine) costs the whole build-up
WALK_ACCEL = 20.0                 # m/s^2 up to walk speed
SPRINT_ACCEL = 6.0                # m/s^2 beyond walk speed (5.0 -> 6.5 jog takes ~0.25 s)
GROUND_FRICTION = 14.0
AIR_ACCEL = 7.0                   # steering authority in the air
AIR_CONTROL_MAX_SPEED = 3.0       # air steering never accelerates past this unless already faster
AIR_DRAG = 0.25

# ---------------------------------------------------------------------------
# Variable Jump (legs)
# ---------------------------------------------------------------------------
GRAVITY = -19.62
JUMP_IMPULSE = 4.2                # tap: ~0.45 m rise
JUMP_HOLD_MAX = 0.25
JUMP_LIFT_ACCEL = 12.0            # held: ~0.95 m rise (elite standing vertical)
# Take-off push: the legs also drive the body along the held direction.  Standing still that is
# the whole horizontal speed of the jump (a standing long jump); at a sprint the legs have no
# more to give and the jump is carried by momentum.
JUMP_PUSH_STANDING = 3.8          # m/s added along the input direction from a standstill (~2.6 m standing leap)
JUMP_PUSH_SPRINT = 0.4            # m/s added at full sprint (~6 m running leap); fades as sqrt(speed fraction)
JUMP_FORWARD_BOOST = 1.15         # a forward leap carries 15 % more horizontal momentum out of the take-off

# Wall kick / tic-tac: planting a foot on a wall while airborne converts run-up into height
WALL_KICK_KEEP_VY = 0.6           # fraction of the existing upward velocity carried through the plant
WALL_KICK_BOOST_VY = 4.5          # added by the leg drive
WALL_KICK_MAX_VY = 6.6            # hard cap: ~1.1 m of additional rise after the kick
WALL_KICK_MIN_APPROACH_SPEED = 2.5
WALL_KICK_MIN_VY = -3.0           # cannot kick off a wall while already falling fast
WALL_KICK_PUSH = 0.4              # slight outward push so the body doesn't scrape the wall
# Wall run: the faster you hit the wall, the more of the run-up goes into height.  The approach speed
# (horizontal, into the wall) is mapped over this range to 0..1 and scales an extra kick on top of
# the plain one: a walking leap kicks as before (hands ~3.9 m), a full-sprint leap runs up to hands
# at ~5 m, so it reliably catches the first-storey holds (3.8 m crevasse, 4.0 m window course).
WALL_RUN_SPEED_RANGE = (7.0, 10.0)
WALL_RUN_EXTRA_VY = 2.6           # m/s added to the kick (and to its cap) at full sprint

# Acrobatics (parkour/acrobatics.py): flips in the air, the landing roll
FLIP_MIN_TIME = 0.30              # a flip needs at least this much airtime left (a quick flip off a flat jump)
FLIP_MAX_TIME = 0.62              # ... and never takes longer than this (a relaxed flip off something high)
FLIP_AIRTIME_USE = 0.92           # the flip is timed to end this fraction of the way to the landing
FLIP_PIVOT_UP = 0.25              # the tucked body turns about a point this far above the box centre
FLIP_RUSH_TIME = 0.10             # landing mid-flip: whatever rotation is left is finished within this
FLIP_HOLD_LOOKAHEAD = (0.0, 0.05, 0.1, 0.15, 0.2, 0.25)  # a press with a hold in reach now or this soon is a reach, not a flip
FLIP_AIM_CLEAR_DIST = 3.5         # ... and so is a press while looking at a hold this close that you are flying toward
ROLL_MIN_DROP = 2.1336            # 7 ft: a landing from higher than this (feet, from the top of the flight) rolls
ROLL_TIME = 0.62                  # crouch, roll over, stand up
ROLL_MIN_SPEED = 3.0              # a roll always carries the body this fast along its direction
ROLL_SPEED_KEEP = 0.95            # fraction of the landing speed the roll keeps (it turns the fall into travel)
ROLL_DECEL = 0.5                  # m/s^2 lost while rolling
ROLL_SPIN_START = 0.10            # fraction of the roll spent absorbing the landing before rolling over
ROLL_SPIN_END = 0.86              # the body is upright again at this fraction (then stands up)
ROLL_CROUCH = 0.85                # how far the body sinks into the roll
ROLL_PIVOT_HEIGHT = 0.50          # the rolling body turns about a point this high above the ground
ROLL_JUMP_FROM = 0.50             # a jump pressed after this fraction of a roll springs straight out of it
FLIP_HEAD_TUCK_DEG = 38.0         # chin to the chest in a front flip / roll (you see your knees)
FLIP_HEAD_BACK_DEG = 20.0         # head back into a back flip

# ---------------------------------------------------------------------------
# Parkour & Ledge Traversal
# ---------------------------------------------------------------------------
RETICLE_MAX_DIST = 4.5
DETECTION_CONE_DEG = 35.0
STANDOFF_DIST = 0.36              # chest-to-wall distance while hanging (body AABB half-width is 0.30)
HANG_EYE_OFFSET = 0.34            # eyes below the hands in an active hang (hands hook ~8 cm below the edge; elbows ~120 deg)
HAND_SPACING = 0.40               # distance between the two hands on a rail
RAIL_END_MARGIN = HAND_SPACING * 0.5 + 0.05
SHIMMY_SPEED = 1.2                # hand-over-hand traverse speed
SHIMMY_FAST_SCALE = 1.25          # with run on the traverse is quicker (1.5 m/s)
SHIMMY_HAND_STRIDE = 0.06         # extra reach of the leading hand beyond its resting spot
HAND_REPLANT_TIME = 0.18          # duration of one hand hop along the rail
HANDOVER_TIME = 0.25              # duration of a hand-over to a different edge (a reach, not a step)
PRESS_BLEND_TIME = 0.15           # hook -> palm press easing when the push-up on a deep lip begins
HAND_MIN_HOP_TIME = 0.06          # a hand-over squeezed to land as the body arrives never takes less than this
HOP_LIFT = 0.06                   # a hopping hand rises this much off the edge ...
HOP_OUT = 0.11                    # ... and comes this far out from the wall
REACH_SLIDE_SPEED = 1.9           # m/s (relative to the shoulder) at which a freed hand slides toward the next hold
HANDOVER_REACH_FRACTION = 0.985   # a hand hops to / holds a spot only while it is within this fraction of arm's length
HOP_MIN_SHOULDER_DIST = 0.30      # a hopping hand keeps at least this far from its shoulder (comes out from the wall as it passes it)
WRIST_OPEN_RATE = 360.0           # deg/s: how fast a wrist unbends when a hand lets go of a surface
                                  # (hand timings keep pace with the climbs; a quicker move scales them further: move_rate)
HOP_OPEN_FRACTION = 0.45          # fraction of a hop spent opening / closing the hand
HAND_RAIL_END_MARGIN = 0.06       # a hand can plant no closer than this to the end of an edge
MIN_RAIL_LENGTH = 0.60            # edges shorter than two hands are not holds
RAIL_COVER_PROBE = 0.20           # an edge is covered where a point this far outside it is inside other geometry
CORNER_LINK_TOL = 0.03            # endpoint distance for two edges to be considered joined at a corner
CORNER_TURN_TIME_CONVEX = 0.58    # swing around an outside corner
CORNER_TURN_TIME_CONCAVE = 0.42   # rotate into an inside corner
CORNER_TURN_END_TOLERANCE = 0.40  # a traverse only turns a corner from within this of the end of its edge
CORNER_TURN_MAX_SWEEP = 1.50      # ... and a corner turn never carries the body further than this
EDGE_SWITCH_UP_MAX = 0.30         # a traverse blocked mid-edge by something with an edge of its own (an
EDGE_SWITCH_DOWN_MAX = 2.15       # awning, a balcony) switches onto that edge, this far up / down at most
EDGE_SWITCH_REACH = 1.00          # (the body moves at most this far sideways to hang from it)
EDGE_SWITCH_TIME_PER_M = 0.30     # extra time per metre of drop on top of CORNER_TURN_TIME_CONCAVE
CLIMB_DOWN_EDGE_TIME = 1.00       # lower yourself over an edge into a hang (reverse mantle)
CLIMB_DOWN_EDGE_MAX_DIST = 0.90   # how close to the edge (from the top) you must be
# Collision buffers: holds that are blocked by only a little nearby geometry still work
EDGE_COLLISION_BUFFER = 0.10      # a catch whose hang pose clips something (a wall beside the end of an
                                  # awning / balcony edge, a post) slides the hands this far along the edge
CLIMB_SHIFT_MAX = 0.60            # a climb blocked by an overhang (an awning over the hands) shifts the climber
                                  # up to this far along the hold to where the climb is clear

# Climb-up (crevasse -> next hold using the feet on the lip)
CLIMB_PULL_TIME = 0.38            # pull-up: hands to chest
CLIMB_STAND_TIME = 0.47           # foot onto the lip, stand up, reach the next hold
CLIMB_SETTLE_TIME = 0.12          # commit weight to the new hold
CLIMB_TOTAL_TIME = CLIMB_PULL_TIME + CLIMB_STAND_TIME + CLIMB_SETTLE_TIME
GRAB_SETTLE_TIME = 0.12           # blend from the catch position into the hang pose
CLIMB_FAST_SCALE = 0.65           # with sprint held the climb is done at speed

# Climb flow: momentum carried up a wall.  Hitting a wall at speed (a wall run, a running leap onto a
# hold) charges the flow from the speed; keeping W held with run on chains hold to hold and builds a
# rhythm of its own.  Climbs, top-outs and corner turns take up to CLIMB_FLOW_SPEEDUP less time at
# full flow, and topping out of a flowing climb sends you off at a run with the sprint build-up kept.
CLIMB_FLOW_SPEEDUP = 0.40         # fraction of a climb's time saved at full flow
CLIMB_FLOW_SPEED_RANGE = (6.0, 10.0)   # take-off / wall approach speed mapped to 0..1 flow
CLIMB_FLOW_KEEP = 0.92            # each chained climb keeps this much of the flow ...
CLIMB_FLOW_CHAIN_GAIN = 0.15      # ... and (with run on) the rhythm adds this, up to ...
CLIMB_FLOW_CHAIN_MAX = 0.50       # ... this much flow from chaining alone
CLIMB_FLOW_IDLE_DRAIN = 1.5       # flow lost per second while hanging without climbing

# Mantle (hands on the top edge -> standing on the top surface)
MANTLE_HOIST_TIME = 0.50          # from a dead hang
MANTLE_STEP_TIME = 0.25
MANTLE_FAST_SCALE = 0.60          # momentum mantle straight out of a jump
MANTLE_STEP_IN_MIN = 0.35
MANTLE_STEP_IN_MAX = 0.75
MANTLE_EXIT_SPEED = 4.0
MANTLE_FLOW_EXIT_SPEED = JOG_SPEED    # topping out of a full-flow climb you leave the edge at a run
VAULT_TIME_PER_METER = 0.40       # ground vault/step-up duration scales with wall height
VAULT_TIME_MIN = 0.25
# Speed vault: a vault / step-up taken at a run is quicker and keeps the run going - you come off it
# with VAULT_SPEED_KEEP of the approach speed (never less than MANTLE_EXIT_SPEED) and the sprint
# build-up intact, instead of stumbling out at a walk and having to build the sprint again.
VAULT_SPEED_KEEP = 0.92
VAULT_FAST_SCALE = 0.70           # duration scale at full sprint (interpolated from 1.0 at a walk)

# Dyno & Wall Launch Mechanics (legs pushing off the wall from a hang)
DYNO_UP_VY = 5.9                  # ~0.9 m body rise -> hands reach ~1.15 m above the current hold
DYNO_UP_PUSH = 0.6
WALL_EJECT_VHORIZ = 4.5           # backward cat-leap: ~2.5-3.5 m with a drop
WALL_EJECT_VY = 3.0
WALL_EJECT_TURN_TIME = 0.35       # camera yaw blend to face the landing wall
LATERAL_DYNO_SPEED = 5.0
LATERAL_DYNO_VY = 4.0
LATERAL_DYNO_PUSH = 2.0
GRAB_LOCKOUT_TIME = 0.22
COYOTE_TIME = 0.12                # a jump is still accepted this long after running off an edge
DROP_LOCKOUT_TIME = 0.35
INPUT_BUFFER_TIME = 0.15

# Controls (see core/controls.py)
CONTROL_SCHEME = "flow"

# Gamepad (see core/gamepad.py, core/gamepad_input.py).  Bindings and the user-facing settings
# (sensitivity, invert, deadzone, ...) live in the per-user controller.json; these are the tuning
# constants and the defaults for a fresh install.
PAD_STICK_DEADZONE = 0.15         # radial inner deadzone (fraction of full deflection), user-adjustable
PAD_STICK_OUTER = 0.92            # deflection treated as "fully pushed" (worn sticks rarely reach 1.0)
PAD_DIR_PRESS = 0.50              # stick deflection that counts as pressing a direction (climb, shimmy, ...)
PAD_DIR_RELEASE = 0.35            # ... and releases it again (hysteresis: no flicker at the threshold)
PAD_DIR_HALF_ANGLE = 60.0         # a direction is held within this angle of its axis (45 deg = both)
PAD_DIR_HALF_ANGLE_HOLD = 66.0    # ... and stays held until the stick leaves this angle
PAD_TRIGGER_PRESS = 0.30          # analog trigger pull that counts as a press ...
PAD_TRIGGER_RELEASE = 0.18        # ... and as a release
PAD_SPRINT_MIN_TILT = 0.80        # stick deflection needed to sprint (below it you walk, scaled by the tilt)
PAD_LOOK_YAW_RATE = 230.0         # deg/s at full right-stick deflection, sensitivity 1.0
PAD_LOOK_PITCH_RATE = 160.0       # deg/s vertically (slower: pitch is for aiming, yaw for turning)
PAD_LOOK_EXPONENT = 2.0           # response curve: fine aim near the centre, full speed at the rim
PAD_LOOK_BOOST = 0.55             # extra turn speed after holding the stick at the rim ...
PAD_LOOK_BOOST_DELAY = 0.18       # ... for this long ...
PAD_LOOK_BOOST_RAMP = 0.35        # ... ramping in over this long (quick 180s without twitchy aim)
PAD_LOOK_SENSITIVITY_RANGE = (0.25, 3.0)
PAD_DEADZONE_RANGE = (0.05, 0.35)
PAD_MENU_REPEAT_DELAY = 0.40      # holding a direction in a menu repeats after this ...
PAD_MENU_REPEAT_RATE = 0.11       # ... every this long

# Multiplayer (see net/ and game/)
MP_DEFAULT_PORT = 24950
MP_MAX_PLAYERS = 15               # players per hosted game, bots included (the host takes one slot)
MP_DEFAULT_BOTS = 1
MP_DEFAULT_EMBLEMS = 12
MP_EMBLEM_RANGE = (4, 40)
MP_PICKUP_RADIUS = 1.15           # body centre within this of an emblem picks it up
MP_MATCH_TIME_LIMIT = 600.0       # a match also ends (leader wins) after this long
MP_MATCH_MINUTES = (2, 3, 5, 8, 10, 15, 20, 30, 45, 60, 0)    # match lengths the host can pick (0 = unlimited)
MP_DEFAULT_CAPTURES = 5           # capture the flag: first to this many captures wins early (0 = time only)
MP_CAPTURE_LIMIT_MAX = 20
MP_FLAG_RADIUS = 1.3              # body centre within this of a flag picks it up
MP_TAG_RADIUS = 1.3               # two players this close have touched (tag the flag carrier)
MP_BASE_RADIUS = 3.0              # capture zone around a base / the drop zone
MP_STEAL_COOLDOWN = 2.5           # free-for-all flag: a new carrier cannot be robbed again for this long
MP_CTF_BASE_DISTANCE = (40.0, 110.0)   # the two bases are this far apart (as far as the map allows)
# Zombie tag: one player is alive and scores for every second they stay that way; everyone else is a zombie
ZOMBIE_POINTS_PER_SECOND = 1.0
ZOMBIE_DEFAULT_GOAL = 120         # points to win (the host picks)
ZOMBIE_GOAL_RANGE = (30, 600)
ZOMBIE_GOAL_STEP = 30
ZOMBIE_RUN_SCALE = 0.88           # zombies run at 88 % of a living player's pace ...
ZOMBIE_CLIMB_SCALE = 0.85         # ... and climb / traverse at 85 % of the speed
ZOMBIE_START_GRACE = 5.0          # nobody can be caught for this long at the start of a round ...
ZOMBIE_CATCH_GRACE = 3.0          # ... or right after a catch (the new living player gets a head start)
ZOMBIE_TAGBACK_GRACE = 10.0       # no tag-backs: the player just caught cannot catch their catcher for this long
ZOMBIE_LIVING_SPAWN_DISTANCE = (35.0, 60.0)   # the living player starts this far from the zombies
MP_NEXT_MATCH_TIMEOUT = 30.0      # after a match the next starts when everyone has joined it, or after this
MP_SNAPSHOT_RATE = 20.0           # server -> clients world snapshots per second
MP_STATE_RATE = 30.0              # client -> server player updates per second (was 20: fast movers looked choppy)
MP_INTERP_DELAY = 0.10            # remote players are drawn this far in the past (smooth between snapshots)
MP_EXTRAPOLATE_MAX = 0.10         # a late pose is carried on along the last motion at most this long
MP_CLOCK_CREEP = 0.02             # how fast the server-clock offset estimate rises toward later arrivals
MP_CLOCK_SLEW = 0.05              # the clock remote players are drawn on follows the estimate at most 5 % fast / slow
MP_CLOCK_SNAP = 0.25              # ... unless it is this far off (first snapshot, a stall): then it jumps
MP_SPAWN_POINTS = ((-6.0, 0.9, -24.0), (-2.0, 0.9, -24.0), (2.0, 0.9, -24.0), (6.0, 0.9, -24.0),
                   (-6.0, 0.9, -20.0), (-2.0, 0.9, -20.0), (2.0, 0.9, -20.0), (6.0, 0.9, -20.0),
                   (-6.0, 0.9, -16.0), (-2.0, 0.9, -16.0), (2.0, 0.9, -16.0), (6.0, 0.9, -16.0),
                   (-4.0, 0.9, -28.0), (0.0, 0.9, -28.0), (4.0, 0.9, -28.0), (0.0, 0.9, -12.0))

# World
CITY_SEED = 7                     # procedural city layout seed (the training park is always in the middle)
KILL_PLANE_Y = -30.0

# Skeletal Arm Rig
# (relative to the body's eye point, i.e. the camera minus CAMERA_FORWARD along the head's facing)
SHOULDER_OFFSET_L = np.array([-SHOULDER_HALF_WIDTH, -(PLAYER_EYE_HEIGHT - SHOULDER_HEIGHT), SHOULDER_FORWARD], dtype=np.float32)
SHOULDER_OFFSET_R = np.array([ SHOULDER_HALF_WIDTH, -(PLAYER_EYE_HEIGHT - SHOULDER_HEIGHT), SHOULDER_FORWARD], dtype=np.float32)
# Pole directions are expressed in the camera frame (right, up, forward) relative to the shoulder
# Elbows point down, slightly outward and back: forearms stay near vertical in a hang
POLE_DIR_L = np.array([-0.2, -1.0, -0.3], dtype=np.float32)
POLE_DIR_R = np.array([ 0.2, -1.0, -0.3], dtype=np.float32)
ARM_INERTIA_FACTOR = 0.12
TORSO_YAW_FOLLOW = 14.0           # shoulders trail quick camera turns (1/s)
# Locomotion arm swing (one arm cycle per stride = two steps).  Below running pace the arms hang
# and swing from the shoulder; they only come up into a pump once the player is actually running.
ARMS_UP_SPEED_LOW = WALK_SPEED + 0.1      # hanging swing up to here ...
ARMS_UP_SPEED_HIGH = JOG_SPEED            # ... fully up in the running pump from here
ARM_HANG_ELBOW_BEND_DEG = 12.0            # relaxed hanging arm: slight elbow bend
ARM_HANG_SWING_DEG = 22.0                 # shoulder swing amplitude at walking pace (each way)
GAIT_BASE_CADENCE = 1.9           # steps per second at very low speed ...
GAIT_CADENCE_PER_MS = 0.19        # ... plus this per m/s (5 m/s -> 2.85, 8.5 m/s -> 3.5 steps/s)
GAIT_SWING_JOG = 0.14             # fore-aft fist swing amplitude (m) of the running pump at a jog ...
GAIT_SWING_SPRINT = 0.24          # ... and at full sprint (a hard pump)
GAIT_LIFT_JOG = 0.06              # vertical lift of the forward fist at a jog ...
GAIT_LIFT_SPRINT = 0.11           # ... and at full sprint
GAIT_BLEND_RATE = 8.0             # how fast the swing fades in / out (landing, taking off)
HAND_FOLLOW_RATE = 16.0           # free hands ease toward their targets (1/s)
GROUND_AIM_BLEND = 0.35           # how far the hands lift toward a hold being looked at on the ground

# Colors (RGBA/RGB)
COLOR_BACKGROUND = (0.08, 0.09, 0.11, 1.0)
COLOR_FLOOR = (0.22, 0.23, 0.25)
COLOR_OBSTACLE = (0.35, 0.37, 0.40)
COLOR_OBSTACLE_ACCENT = (0.45, 0.48, 0.52)
COLOR_LEDGE_REACHABLE = (0.0, 0.9, 1.0)       # Cyan
COLOR_LEDGE_LOCKED = (1.0, 0.55, 0.0)         # Amber
COLOR_ARM_SKIN = (0.78, 0.58, 0.46)
COLOR_ARM_SLEEVE = (0.15, 0.16, 0.18)


def rise_height(vy, g=GRAVITY):
    """Peak rise of a body launched upward at vy under gravity g."""
    return (vy * vy) / (2.0 * abs(g))


def velocity_for_rise(h, g=GRAVITY):
    """Launch velocity needed to rise h meters under gravity g."""
    return math.sqrt(2.0 * abs(g) * max(0.0, h))
