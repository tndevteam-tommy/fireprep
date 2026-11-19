"""Tunable gameplay constants. Everything that used to be a magic number lives here."""

from __future__ import annotations

from dataclasses import dataclass

INTERACT_RANGE = 2.5
"""Metres within which [E] operates terminals and the vending machine."""


@dataclass(frozen=True)
class PlayerConfig:
    # (spawn position/yaw come from the level; camera yaw convention: 0 = +X, 90 = +Z)
    eye_height_standing: float = 1.6
    eye_height_crouched: float = 1.0
    body_height_standing: float = 1.8
    body_height_crouched: float = 1.2
    walk_speed: float = 4.5
    crouch_speed: float = 2.0
    collision_radius: float = 0.32
    mouse_sensitivity: float = 0.15
    max_pitch: float = 89.0
    height_smoothing: float = 10.0  # exponential ease rate toward the target eye height


PLAYER = PlayerConfig()


@dataclass(frozen=True)
class NPCConfig:
    body_height: float = 1.9
    eye_height: float = 1.5
    collision_radius: float = 0.35
    walk_speed: float = 1.2
    turn_speed_deg: float = 220.0
    walk_cycle_rate: float = 5.5  # walk-cycle radians advanced per metre walked


NPC = NPCConfig()


@dataclass(frozen=True)
class GuardConfig:
    """CyberGuard routine timings and perception."""

    pacing_duration: float = 6.0  # seconds of patrol before walking to the wall
    lean_duration: float = 15.0  # matches the authored weapon-inspection clip
    investigate_duration: float = 15.0  # default seconds spent inspecting a noise after arrival
    bang_duration: float = 12.0  # seconds spent hitting a sparking machine before just staring
    route_pause: float = 1.5  # seconds a route-patrolling guard lingers at each waypoint
    wall_stop_distance: float = 0.15
    noise_stop_distance: float = 0.25
    # Vision while actively scanning the room (half-angle FOV; 55 deg = 110 deg cone)
    alert_range: float = 11.5
    alert_fov: float = 55.0
    # Vision while head is tilted down at the weapon (head_pitch above this threshold)
    engrossed_head_pitch: float = 20.0
    engrossed_range: float = 3.0
    engrossed_fov: float = 30.0
    # Detection gain falls off with distance: full rate inside near_full_rate_distance,
    # down to far_rate_fraction of it at alert_range.
    near_full_rate_distance: float = 3.0
    far_rate_fraction: float = 0.3
    # Suspicion: on a sighting the guard stops and looks (ALERTED); once the detection
    # meter passes suspicion_threshold it walks slowly to where it last saw the player
    # (SUSPICIOUS). Losing sight for the grace period returns it to its routine.
    suspicion_threshold: float = 35.0  # the middle of the HUD's yellow band (20-50)
    alert_grace: float = 1.5
    suspicion_grace: float = 2.5
    suspicious_speed_factor: float = 0.6
    look_around_duration: float = 2.0
    # Body sway while inspecting a noise
    scan_amplitude_deg: float = 35.0
    scan_rate: float = 1.5


GUARD = GuardConfig()


@dataclass(frozen=True)
class DetectionConfig:
    """Line-of-sight targeting. Observers aim at a point on the player's body; a ray that
    reaches that point unobstructed counts as a sighting."""

    target_height_standing: float = 1.5  # chest, measured from the feet
    target_height_crouched: float = 0.9
    # Stealth-meter gain per second while an observer has line of sight
    guard_rate_standing: float = 45.0
    guard_rate_crouched: float = 12.0
    camera_rate_standing: float = 65.0
    camera_rate_crouched: float = 18.0
    # Meter behaviour
    decay_rate: float = 30.0  # per second while unseen
    max_level: float = 100.0  # reaching this means "caught"
    # The meter's colours on the HUD: cyan, yellow past ``yellow_level``, red past
    # ``red_level``. Somebody at home only *notices* the stranger in their flat (turns,
    # shouts) once it is red - until then they are only half aware of something.
    yellow_level: float = 20.0
    red_level: float = 50.0


DETECTION = DetectionConfig()


@dataclass(frozen=True)
class SimulationConfig:
    target_fps: int = 60
    max_dt: float = 0.1  # hard cap on a single simulation step (seconds)
    # Caught: the red "you were caught" screen stays up this long, then fades to black
    # over ``caught_fade`` seconds; then the price is paid (a new game, or thrown out) and
    # the picture fades back in from black over ``fade_in``.
    caught_hold: float = 3.0
    caught_fade: float = 1.5
    evict_hold: float = 2.2  # (thrown out of somebody's home: a shorter red screen)
    fade_in: float = 1.2
    # Out of a level's last door (the biodome's outer staff door): the picture fades to
    # black over ``level_fade_out`` (the netrunner stops at the door, nobody can catch them
    # on the threshold), the next level is loaded, and it fades in over ``level_fade_in``.
    level_fade_out: float = 1.0
    level_fade_in: float = 1.4

    @property
    def caught_respawn_delay(self) -> float:
        """How long from being caught until the game goes on (the new game)."""
        return self.caught_hold + self.caught_fade


SIMULATION = SimulationConfig()


@dataclass(frozen=True)
class ClockConfig:
    """The city's day (``core.clock.WorldClock``)."""

    # half a game minute every real second: a day goes by in 48 minutes - an evening's
    # rush (17:00 - 23:00) is twelve, the dead of night (02:00 - 07:00) ten
    minutes_per_second: float = 0.5
    new_game_at: float = 19 * 60 + 30  # 19:30: the netrunner goes in at dusk (and out after dark)
    title_at: float = 19 * 60 + 40  # the start menu's street: early evening, busy


CLOCK = ClockConfig()


@dataclass(frozen=True)
class HackingConfig:
    """Scanner and quickhack tuning (the netrunner side of the game)."""

    # Scanner pulse: reveals every electronic target within this radius of the viewpoint
    # (through walls - it is a network ping, not a sight check) for ``reveal_duration``.
    scan_range: float = 16.0
    reveal_duration: float = 12.0
    scan_cooldown: float = 0.8
    pulse_visual_duration: float = 0.9  # expanding ring drawn by the renderer
    # Direct quickhacks need the target revealed, inside this range and in line of sight
    # from the current viewpoint (the player's eyes, or a hijacked camera's lens).
    quickhack_range: float = 14.0
    focus_cone_deg: float = 12.0  # aim assist: targets within this of the crosshair can be focused
    focus_distance_weight: float = 0.25  # degrees of "penalty" per metre: nearer targets win ties
    # Effects
    blind_duration: float = 6.0
    knockout_shock_duration: float = 2.4  # convulsing on its feet before it drops
    knockout_fall_duration: float = 1.05  # from the knees buckling to hitting the floor / a wall
    camera_shutdown_duration: float = 180.0  # a shut-down camera stays dark for three minutes
    terminal_overload_noise_radius: float = 18.0
    terminal_overload_spark_duration: float = 30.0  # keeps crackling while the guard stares
    terminal_overload_inspect_duration: float = 40.0  # a fried computer holds a guard's attention
    vending_inspect_duration: float = 45.0  # a shorted machine gets hit for a while, then watched


HACKING = HackingConfig()
